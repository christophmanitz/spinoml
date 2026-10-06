// SpinoML LLM sidecar authentication / Host / Origin enforcement (Phase 77/78).
//
// Pure ESM, no I/O except reading the env object passed in. Mirrors the Python
// reference in `sidecar-torch/auth.py` (same decision order, same error codes,
// same /health shapes) and the wire protocol in
// `docs/engineering/SIDECAR_AUTH.md` ("Configuration", "Default origin
// allow-list", "Per-request enforcement order", "LLM sidecar specifics").
//
// Owns:
//   * `loadConfig` / `scrubEnv` — read SPINOML_SIDECAR_TOKEN /
//     SPINOML_REQUIRE_TOKEN / SPINOML_ALLOWED_ORIGINS once at startup,
//     validate, then delete the token from the env so subprocess children
//     (opencode, the Claude CLI, run_script helpers, npm) never inherit it.
//   * `checkHost` / `checkOrigin` / `tokenMatches` — the three primitives
//     whose combination is the per-request gate.
//   * `decide` — the single decision function the HTTP handler calls BEFORE
//     reading any body. Encodes the normative enforcement order in ONE place:
//     Host → Origin → OPTIONS → GET /health → token → otherwise.
//   * `sessionSecretMatches` — constant-time comparison for the per-turn MCP
//     session secret (`/internal/mcp/<requestId>/…`).
//
// The token value is never logged, printed or echoed in any error message.

import { createHash, randomBytes, timingSafeEqual } from 'node:crypto'

const TOKEN_RE = /^[A-Za-z0-9_-]+$/

export const DEFAULT_ORIGINS = Object.freeze([
  'tauri://localhost',
  'http://tauri.localhost',
  'https://tauri.localhost',
  'http://localhost:5173',
  'http://127.0.0.1:5173',
])

export const CODE_UNAUTHORIZED = 'unauthorized'
export const CODE_BAD_ORIGIN = 'bad_origin'
export const CODE_BAD_HOST = 'bad_host'

export const DUPLICATE_HEADER = Symbol.for('spinoml.auth.duplicate_header')

export class AuthConfigError extends Error {
  constructor(message) {
    super(message)
    this.name = 'AuthConfigError'
  }
}

function asOrigins(extra) {
  const out = new Set(DEFAULT_ORIGINS)
  if (typeof extra === 'string' && extra.length > 0) {
    for (const entry of extra.split(',')) {
      const trimmed = entry.trim()
      if (trimmed) out.add(trimmed)
    }
  }
  return Object.freeze([...out])
}

export function loadConfig(env) {
  const rawToken = env?.SPINOML_SIDECAR_TOKEN
  let token = null
  if (typeof rawToken === 'string' && rawToken.length > 0) {
    if (rawToken.length < 32) {
      throw new AuthConfigError(
        `SPINOML_SIDECAR_TOKEN must be at least 32 characters (got ${rawToken.length})`,
      )
    }
    if (!TOKEN_RE.test(rawToken)) {
      throw new AuthConfigError(
        'SPINOML_SIDECAR_TOKEN may only contain characters [A-Za-z0-9_-]',
      )
    }
    token = rawToken
  }
  const requireRaw = String(env?.SPINOML_REQUIRE_TOKEN ?? '') === '1'
  if (requireRaw && token === null) {
    throw new AuthConfigError(
      'SPINOML_REQUIRE_TOKEN=1 requires SPINOML_SIDECAR_TOKEN to be set',
    )
  }
  const requireToken = requireRaw || token !== null
  return {
    token,
    requireToken,
    origins: asOrigins(env?.SPINOML_ALLOWED_ORIGINS),
  }
}

export function scrubEnv(env) {
  if (!env || typeof env !== 'object') return
  // Map-like envs expose `delete`; plain objects / process.env use a property
  // delete. `Reflect.deleteProperty` never throws (it returns false for a
  // non-configurable property), so no catch is needed here.
  if (typeof env.delete === 'function') env.delete('SPINOML_SIDECAR_TOKEN')
  if (Object.prototype.hasOwnProperty.call(env, 'SPINOML_SIDECAR_TOKEN')) {
    Reflect.deleteProperty(env, 'SPINOML_SIDECAR_TOKEN')
  }
}

function hostPart(raw) {
  if (typeof raw !== 'string') return null
  const h = raw.trim()
  if (h === '') return null
  if (h.startsWith('[')) {
    const end = h.indexOf(']')
    if (end === -1) return null
    const host = h.slice(1, end).trim().toLowerCase()
    const rest = h.slice(end + 1)
    if (rest !== '') {
      if (!rest.startsWith(':')) return null
      const port = rest.slice(1)
      if (!/^[0-9]+$/.test(port)) return null
    }
    return host === '::1' ? '::1' : null
  }
  const colon = h.indexOf(':')
  if (colon !== -1) {
    const tail = h.slice(colon + 1)
    if (!/^[0-9]+$/.test(tail)) return null
    return h.slice(0, colon).toLowerCase()
  }
  return h.toLowerCase()
}

export function checkHost(hostHeader) {
  const host = hostPart(hostHeader)
  if (host === null) return false
  return host === '127.0.0.1' || host === 'localhost' || host === '::1'
}

export function checkOrigin(originHeader, cfg) {
  if (originHeader === null || originHeader === undefined) return true
  if (typeof originHeader !== 'string') return false
  if (originHeader === '') return false
  return cfg.origins.includes(originHeader)
}

function safeEqualBytes(a, b) {
  const ah = createHash('sha256').update(a).digest()
  const bh = createHash('sha256').update(b).digest()
  return timingSafeEqual(ah, bh)
}

function utf8(s) {
  // Callers type-guard `s` to a string before calling; TextEncoder.encode never
  // throws for a string (unpaired surrogates become U+FFFD), so no catch here.
  return new TextEncoder().encode(s)
}

export function tokenMatches(supplied, cfg) {
  if (cfg.token === null || cfg.token === undefined) return false
  if (typeof supplied !== 'string' || supplied === '') return false
  return safeEqualBytes(utf8(supplied), utf8(cfg.token))
}

export function sessionSecretMatches(supplied, secret) {
  if (typeof secret !== 'string' || secret === '') return false
  if (typeof supplied !== 'string' || supplied === '') return false
  return safeEqualBytes(utf8(supplied), utf8(secret))
}

export function generateMcpSecret() {
  return randomBytes(32).toString('hex')
}

const HEADER_GET_DUPLICATE = Symbol.for('spinoml.auth.header_get.duplicate')

function makeHeaderGet(headers, rawHeaders) {
  // Node joins duplicate header values with ", " into `headers[name]`, so a
  // raw `req.rawHeaders` walk is the only way to detect a duplicate.
  // Return DUPLICATE_HEADER when the same name appears more than once.
  const seen = new Set()
  const dup = new Set()
  if (Array.isArray(rawHeaders)) {
    for (let i = 0; i + 1 < rawHeaders.length; i += 2) {
      const name = String(rawHeaders[i] ?? '').toLowerCase()
      if (name === '') continue
      if (seen.has(name)) dup.add(name)
      seen.add(name)
    }
  }
  return function headerGet(name) {
    const key = String(name).toLowerCase()
    if (dup.has(key)) return HEADER_GET_DUPLICATE
    const v = headers?.[key]
    if (v === undefined || v === null) return null
    if (Array.isArray(v)) {
      if (v.length === 0) return null
      return v.join(', ')
    }
    return String(v)
  }
}

export function decideFromHeaders(method, url, req, cfg) {
  return decide(method, url, makeHeaderGet(req.headers ?? {}, req.rawHeaders), cfg)
}

function allow(origin, tokenOk) {
  return { kind: 'ok', status: 0, code: '', reason: '', origin, tokenOk: tokenOk === true }
}

function healthLimited(origin) {
  return { kind: 'ok_health_limited', status: 0, code: '', reason: '', origin, tokenOk: false }
}

function reject(status, code, reason, origin) {
  return { kind: 'reject', status, code, reason: reason ?? '', origin: origin ?? null, tokenOk: false }
}

// `/internal/mcp/<requestId>/list|call` — see step 4b in `decide`.
export const MCP_ROUTE_RE = /^\/internal\/mcp\/[^/]+\/(?:list|call)$/

export function decide(method, url, headersGet, cfg) {
  // 1. Host (both modes).
  if (!checkHost(headersGet('Host'))) {
    return reject(403, CODE_BAD_HOST)
  }

  // 2. Origin (both modes).
  const rawOrigin = headersGet('Origin')
  const okOrigin = checkOrigin(rawOrigin, cfg)
  if (!okOrigin) return reject(403, CODE_BAD_ORIGIN)
  const echoOrigin = typeof rawOrigin === 'string' ? rawOrigin : null

  // 3. OPTIONS preflight.
  if (method === 'OPTIONS') {
    return allow(echoOrigin, !cfg.requireToken)
  }

  // 4. GET /health.
  if (method === 'GET' && url === '/health') {
    if (!cfg.requireToken) return allow(echoOrigin, true)
    if (tokenMatches(headersGet('X-SpinoML-Token'), cfg)) return allow(echoOrigin, true)
    return healthLimited(echoOrigin)
  }

  // 4b. The opencode MCP bridge route. The bridge is a child of opencode and must
  // NEVER hold the master token (it would leak into a process the model drives);
  // it authenticates with a per-turn session secret that handleMcpRoute checks in
  // BOTH modes (and the master token is NOT accepted there). So the master-token
  // requirement below must not apply — Host/Origin (steps 1-2) still did.
  if (method === 'POST' && MCP_ROUTE_RE.test(url)) return allow(echoOrigin, false)

  // 5. Everything else.
  if (!cfg.requireToken) return allow(echoOrigin, true)
  const supplied = headersGet('X-SpinoML-Token')
  if (supplied === HEADER_GET_DUPLICATE) {
    return reject(401, CODE_UNAUTHORIZED, 'invalid', echoOrigin)
  }
  if (typeof supplied !== 'string' || supplied === '') {
    return reject(401, CODE_UNAUTHORIZED, 'missing', echoOrigin)
  }
  if (!tokenMatches(supplied, cfg)) {
    return reject(401, CODE_UNAUTHORIZED, 'invalid', echoOrigin)
  }
  return allow(echoOrigin, true)
}

export function healthBody(cfg, fullBody, tokenOk) {
  const base = (fullBody && typeof fullBody === 'object') ? { ...fullBody } : {}
  if (cfg.requireToken) {
    base.auth = 'token'
    base.requiresAuth = true
    base.tokenOk = tokenOk === true
  } else {
    base.auth = 'unauthenticated-dev'
    base.requiresAuth = false
    base.tokenOk = true
  }
  return base
}

export function limitedHealthBody() {
  return { ok: true, auth: 'token', requiresAuth: true, tokenOk: false }
}

export function applyCorsHeaders(res, origin) {
  if (typeof origin === 'string' && origin.length > 0) {
    res.setHeader('Access-Control-Allow-Origin', origin)
    res.setHeader('Vary', 'Origin')
  }
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-SpinoML-Token')
  res.setHeader('Access-Control-Max-Age', '600')
}

export const _internal = { HEADER_GET_DUPLICATE }
