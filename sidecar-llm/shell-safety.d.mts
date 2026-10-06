// Hand-written type declarations for sidecar-llm/shell-safety.mjs.
// Mirrors the module's runtime exports exactly (no `any`): the argv tokenizer
// result is a discriminated union, the URL/ssh policy checks return
// `{ ok: true, ... } | { ok: false, error }`, and `safeFetch` exposes the
// injected `lookup` / `fetchImpl` shapes its callers must satisfy.

export type SplitArgsResult =
  | { ok: true; argv: string[] }
  | { ok: false; error: string }

export function posixQuote(s: unknown): string
export function splitArgs(str: unknown): SplitArgsResult
export function quoteArgv(argv: readonly string[]): string

export function isBlockedAddress(ip: unknown): boolean

export type CheckDownloadUrlResult =
  | { ok: true; url: string }
  | { ok: false; error: string }

export function checkDownloadUrl(url: unknown): CheckDownloadUrlResult

export type CheckSshTargetResult = { ok: true } | { ok: false; error: string }

export function checkSshTarget(s: unknown): CheckSshTargetResult

// The minimal response shape safeFetch needs: both the global `Response` and
// the test's fake response satisfy it structurally.
export interface SafeFetchResponse {
  readonly status: number
  readonly ok: boolean
  readonly headers: { get(name: string): string | null }
}

export interface LookupAddress {
  address: string
  family?: number
}

export interface SafeFetchInit {
  redirect: 'manual' | 'follow' | 'error'
  signal?: AbortSignal
}

export interface SafeFetchOptions {
  lookup?: (
    host: string,
    options: { all: true; verbatim: true },
  ) => Promise<readonly LookupAddress[]>
  fetchImpl?: (url: string, init: SafeFetchInit) => Promise<SafeFetchResponse>
  maxRedirects?: number
  timeoutMs?: number
}

export function safeFetch(url: unknown, opts?: SafeFetchOptions): Promise<SafeFetchResponse>
