// Sidecar authentication (Phase 77/78) — the ONLY module that talks to the
// Rust `sidecar_token` command and the ONLY wrapper for HTTP requests to the
// torch/LLM sidecars. The static check scripts/verify-sidecar-fetch.ts fails
// when any other file under src/ calls bare `fetch(`.
//
// Contract (see docs/engineering/SIDECAR_AUTH.md):
//   * Tauri command `sidecar_token({endpoint})` → `string | null`; a rejected
//     promise means "no token available".
//   * Sidecars answer 401 (missing/invalid token) or 403 (bad Origin/Host).
//   * `GET /health` needs no token and reports the auth mode.

import { invoke } from '@tauri-apps/api/core'
import { isTauri } from '../workspace/tauri-fs'

export type SidecarEndpoint = 'torch-local' | 'torch-remote' | 'llm'

export type SidecarAuthKind = 'unauthorized' | 'forbidden'

function unauthorizedMessage(endpoint: SidecarEndpoint): string {
  return `Sidecar-Authentifizierung fehlgeschlagen (${endpoint}): Token fehlt oder ist ungültig — App neu starten oder Sidecar von der App starten lassen.`
}

const FORBIDDEN_MESSAGE = 'Sidecar hat die Anfrage abgelehnt (Origin/Host nicht erlaubt).'

/** Raised by `sidecarFetch` when the sidecar rejects a request as unauthenticated
 *  (401) or as a disallowed Origin/Host (403). The token value is never part of
 *  the message; when the Rust command itself failed, that error text is attached
 *  as the standard `cause` so the user can see why there was no token. */
export class SidecarAuthError extends Error {
  readonly status: number
  readonly endpoint: SidecarEndpoint
  readonly kind: SidecarAuthKind

  constructor(status: number, endpoint: SidecarEndpoint, cause?: unknown) {
    super(status === 403 ? FORBIDDEN_MESSAGE : unauthorizedMessage(endpoint), { cause })
    this.name = 'SidecarAuthError'
    this.status = status
    this.endpoint = endpoint
    this.kind = status === 403 ? 'forbidden' : 'unauthorized'
  }
}

// Per-endpoint token cache. `null` = "we asked, there is no token" (browser-dev
// or tokenless sidecar). A failed `invoke` also stores the error text so a later
// SidecarAuthError can carry it as `cause`.
const tokens = new Map<SidecarEndpoint, string | null>()
const tokenErrors = new Map<SidecarEndpoint, string>()

/** Read (and cache) the token for one endpoint. Only meaningful in Tauri mode;
 *  in browser-dev this always returns null. `refresh: true` bypasses the cache
 *  and asks Rust again (used on the 401 retry path). */
export async function getSidecarToken(
  endpoint: SidecarEndpoint,
  opts: { refresh?: boolean } = {},
): Promise<string | null> {
  if (!isTauri()) return null
  if (!opts.refresh && tokens.has(endpoint)) return tokens.get(endpoint) ?? null
  try {
    const token = await invoke<string | null>('sidecar_token', { endpoint })
    tokens.set(endpoint, typeof token === 'string' && token.length > 0 ? token : null)
    tokenErrors.delete(endpoint)
    return tokens.get(endpoint) ?? null
  } catch (e) {
    // The Rust command rejected: we have no token. Remember WHY so the next
    // SidecarAuthError can attach it as `cause` — the user must see the reason,
    // but this must never block (a missing token is a valid dev state).
    tokens.delete(endpoint)
    tokenErrors.set(endpoint, e instanceof Error ? e.message : String(e))
    return null
  }
}

/** Drop the cached token(s). Called on the 401 retry path; exported so a future
 *  connection switch can also invalidate the cache. */
export function clearSidecarTokens(endpoint?: SidecarEndpoint): void {
  if (endpoint) {
    tokens.delete(endpoint)
  } else {
    tokens.clear()
  }
}

function withToken(init: RequestInit | undefined, token: string | null): RequestInit | undefined {
  if (!token) return init
  const headers = new Headers(init?.headers)
  headers.set('X-SpinoML-Token', token)
  return { ...init, headers }
}

/** The one wrapper every sidecar request goes through. Adds the token header
 *  when one exists, passes `signal`/`body`/`method`/headers through untouched
 *  (streaming responses are never buffered — the Response is returned as is),
 *  and on 401 drops the cached token, re-reads it once and retries EXACTLY once.
 *  A remaining 401/403 throws a SidecarAuthError; network errors propagate. */
export async function sidecarFetch(
  endpoint: SidecarEndpoint,
  url: string,
  init?: RequestInit,
): Promise<Response> {
  const token = await getSidecarToken(endpoint)
  let res = await fetch(url, withToken(init, token))
  if (res.status === 401) {
    // Token missing or stale. Drop it, re-read once, retry exactly once. The
    // 401 body is intentionally not consumed before the retry.
    clearSidecarTokens(endpoint)
    const refreshed = await getSidecarToken(endpoint, { refresh: true })
    res = await fetch(url, withToken(init, refreshed))
  }
  if (res.status === 401 || res.status === 403) {
    throw new SidecarAuthError(res.status, endpoint, tokenErrors.get(endpoint))
  }
  return res
}

export type SidecarHealth = {
  state: 'online' | 'offline' | 'auth-failed'
  auth?: 'token' | 'unauthenticated-dev'
  body?: unknown
  message?: string
}

type RawProbe =
  | { thrown: true; health: SidecarHealth }
  | { thrown: false; status: number; body: unknown }

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null
}

/** One `GET /health` through the wrapper, without interpreting the body. */
async function rawHealthProbe(endpoint: SidecarEndpoint, url: string): Promise<RawProbe> {
  let res: Response
  try {
    res = await sidecarFetch(endpoint, `${url}/health`)
  } catch (e) {
    if (e instanceof SidecarAuthError) {
      return { thrown: true, health: { state: 'auth-failed', message: e.message } }
    }
    // A thrown fetch rejection (connection refused, DNS, abort) is a genuine
    // offline state — distinct from an authenticated-but-rejected sidecar.
    return { thrown: true, health: { state: 'offline', message: e instanceof Error ? e.message : String(e) } }
  }
  if (!res.ok) {
    return { thrown: false, status: res.status, body: undefined }
  }
  let body: unknown
  try {
    body = await res.json()
  } catch (e) {
    return { thrown: true, health: { state: 'offline', message: `Health-Antwort nicht lesbar: ${e instanceof Error ? e.message : String(e)}` } }
  }
  return { thrown: false, status: res.status, body }
}

function classifyHealth(status: number, body: unknown, endpoint: SidecarEndpoint): SidecarHealth {
  if (status < 200 || status >= 300) {
    return { state: 'offline', message: `sidecar HTTP ${status}` }
  }
  const b = isRecord(body) ? body : {}
  const auth = b.auth === 'token' || b.auth === 'unauthenticated-dev' ? b.auth : undefined
  if (b.tokenOk === false) {
    return { state: 'auth-failed', auth, body, message: unauthorizedMessage(endpoint) }
  }
  if (b.ok === true) {
    return { state: 'online', auth, body }
  }
  return { state: 'offline', body, message: 'Sidecar-Health-Antwort ohne ok-Feld' }
}

/** Probe `GET <url>/health`. Network failure → `offline`; a rejected request or
 *  a body with `tokenOk: false` → `auth-failed` with the German message;
 *  `ok: true` → `online` (+ `auth` when the sidecar reports it). Older sidecars
 *  omit the auth keys → plain online.
 *
 *  A `tokenOk: false` body means the cached token is stale (e.g. a remote
 *  tunnel restart issued a new token): unlike a 401 it never triggers the
 *  `sidecarFetch` refresh-retry, so we mirror it here — drop the cache, re-read
 *  the token once (`refresh: true`) and probe EXACTLY once more. No loops. */
export async function probeSidecarHealth(
  endpoint: SidecarEndpoint,
  url: string,
): Promise<SidecarHealth> {
  const first = await rawHealthProbe(endpoint, url)
  if (first.thrown) return first.health

  const firstHealth = classifyHealth(first.status, first.body, endpoint)
  const firstTokenOkFalse = isRecord(first.body) && first.body.tokenOk === false
  if (!firstTokenOkFalse) return firstHealth

  // Stale/absent token: refresh once and re-probe exactly once.
  clearSidecarTokens(endpoint)
  await getSidecarToken(endpoint, { refresh: true })
  const second = await rawHealthProbe(endpoint, url)
  if (second.thrown) return second.health
  return classifyHealth(second.status, second.body, endpoint)
}
