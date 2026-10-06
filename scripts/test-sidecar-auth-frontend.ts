#!/usr/bin/env tsx
// Phase 77/78 — frontend sidecar-auth wrapper, end-to-end against a real fake
// sidecar. Spins up a node:http server on a random loopback port that enforces
// the wire protocol (X-SpinoML-Token required, 401/403/health shapes), and
// stubs the Tauri bridge in globalThis.window before the auth module is loaded.

import http from 'node:http'
import type { AddressInfo } from 'node:net'

let passed = 0
let failed = 0
function check(name: string, cond: boolean, detail = ''): void {
  if (cond) { passed++; process.stdout.write(`  ✓ ${name}\n`) }
  else { failed++; process.stdout.write(`  ✗ ${name}${detail ? '  ' + detail : ''}\n`) }
}

// ── fake Tauri bridge (installed BEFORE the dynamic import) ────────────────
type InvokeFn = (cmd: string, args: Record<string, unknown>) => Promise<unknown>
let invokeCalls = 0
let invokeImpl: InvokeFn = async () => null
const bridge = {
  __TAURI_INTERNALS__: {
    invoke: (cmd: string, args: Record<string, unknown>) => {
      invokeCalls++
      return invokeImpl(cmd, args)
    },
  },
}
function setBridge(on: boolean): void {
  if (on) (globalThis as unknown as { window: typeof bridge }).window = bridge
  else (globalThis as unknown as { window?: unknown }).window = undefined
}
setBridge(false) // start non-Tauri for the import; individual tests opt-in

// ── fake sidecar ────────────────────────────────────────────────────────────
type LogEntry = { path: string; token: string | null; aborted: boolean }
const requestLog: LogEntry[] = []
const cfg = {
  requiredToken: null as string | null,
  // When set, /health is header-aware like the real protocol: tokenOk is true
  // iff the X-SpinoML-Token header equals this value.
  expectedToken: null as string | null,
  forceStatus: 0,                 // 401/403 short-circuit
  health: { ok: true } as Record<string, unknown>,
  streamEnded: false,
}
const server = http.createServer((req, res) => {
  const token = (req.headers['x-spinoml-token'] as string | undefined) ?? null
  const u = new URL(req.url ?? '/', 'http://127.0.0.1')
  const aborted = req.aborted
  requestLog.push({ path: u.pathname, token, aborted })

  req.on('close', () => {
    const last = requestLog[requestLog.length - 1]
    if (last) last.aborted = true
  })

  if (u.pathname === '/health') {
    // Header-aware when cfg.expectedToken is set (mirrors the real protocol);
    // otherwise the canned cfg.health body is returned verbatim.
    const body = cfg.expectedToken !== null
      ? { ok: true, auth: 'token', requiresAuth: true, tokenOk: token === cfg.expectedToken }
      : cfg.health
    res.writeHead(200, { 'content-type': 'application/json' })
    res.end(JSON.stringify(body))
    return
  }
  if (cfg.forceStatus) {
    res.writeHead(cfg.forceStatus, { 'content-type': 'application/json' })
    res.end(JSON.stringify({ code: cfg.forceStatus === 403 ? 'bad_origin' : 'unauthorized' }))
    return
  }
  if (cfg.requiredToken !== null && token !== cfg.requiredToken) {
    res.writeHead(401, { 'content-type': 'application/json' })
    res.end(JSON.stringify({ code: 'unauthorized', reason: 'invalid' }))
    return
  }
  if (u.pathname === '/echo') {
    const headers: Record<string, string | string[] | undefined> = { ...req.headers }
    res.writeHead(200, { 'content-type': 'application/json' })
    res.end(JSON.stringify({ ok: true, got: headers }))
    return
  }
  if (u.pathname === '/stream') {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' })
    res.write('data: first\n\n')
    setTimeout(() => {
      cfg.streamEnded = true
      try { res.write('data: second\n\n'); res.end() } catch { /* connection torn down by abort */ }
    }, 200)
    return
  }
  if (u.pathname === '/slow') {
    const t = setTimeout(() => { try { res.writeHead(200); res.end('ok') } catch { /* aborted */ } }, 3000)
    t.unref?.()
    return
  }
  res.writeHead(200, { 'content-type': 'application/json' })
  res.end(JSON.stringify({ ok: true, path: u.pathname }))
})

const baseUrl = await new Promise<string>((resolve) => {
  server.listen(0, '127.0.0.1', () => {
    const { port } = server.address() as AddressInfo
    resolve(`http://127.0.0.1:${port}`)
  })
})

function reset(): void {
  requestLog.length = 0
  invokeCalls = 0
  cfg.requiredToken = null
  cfg.expectedToken = null
  cfg.forceStatus = 0
  cfg.health = { ok: true }
  cfg.streamEnded = false
}

// ── import the wrapper AFTER the bridge is in place ─────────────────────────
const auth = await import('../src/sidecars/auth')
const { sidecarFetch, probeSidecarHealth, getSidecarToken, clearSidecarTokens, SidecarAuthError } = auth

async function run(): Promise<void> {
  const url = baseUrl
  const SEC = 'SECRET-TOKEN-12345'
  const includesToken = (s: unknown): boolean => typeof s === 'string' && s.includes(SEC)

  // 1. token attached from invoke (Tauri); caller headers preserved
  {
    setBridge(true)
    clearSidecarTokens()
    invokeImpl = async (cmd) => cmd === 'sidecar_token' ? SEC : null
    const r = await sidecarFetch('llm', `${url}/echo`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Caller': 'yes' },
      body: '{}',
    })
    const j = await r.json() as { got: Record<string, string | string[] | undefined> }
    check('token attached from invoke', j.got['x-spinoml-token'] === SEC)
    const ct = j.got['content-type']
    check('caller content-type preserved', typeof ct === 'string' && ct.includes('application/json'))
    check('caller X-Caller header preserved', j.got['x-caller'] === 'yes')
    check('invoke called exactly once (cached)', invokeCalls === 1)
    reset()
  }

  // 2. 401 then refreshed token → retry once → success
  {
    setBridge(true)
    clearSidecarTokens()
    cfg.requiredToken = 'good'
    let n = 0
    invokeImpl = async (cmd) => {
      if (cmd !== 'sidecar_token') return null
      n++
      return n === 1 ? 'bad' : 'good'
    }
    const r = await sidecarFetch('llm', `${url}/echo`, { method: 'GET' })
    check('refreshed token on 401 → 200', r.ok)
    check('exactly 2 requests (original + retry)', requestLog.length === 2)
    check('invoke called twice (cached then refresh)', invokeCalls === 2)
    check('first request used bad token', requestLog[0]?.token === 'bad')
    check('retry used good token', requestLog[1]?.token === 'good')
    reset()
  }

  // 3. permanent 401 → SidecarAuthError, exactly 2 requests, no token in message
  {
    setBridge(true)
    clearSidecarTokens()
    cfg.requiredToken = 'good'
    invokeImpl = async (cmd) => cmd === 'sidecar_token' ? 'bad' : null
    let caught: unknown = null
    try { await sidecarFetch('llm', `${url}/echo`) }
    catch (e) { caught = e }
    check('permanent 401 throws', caught instanceof SidecarAuthError)
    check('permanent 401 status 401', caught instanceof SidecarAuthError && caught.status === 401)
    check('permanent 401 kind unauthorized', caught instanceof SidecarAuthError && caught.kind === 'unauthorized')
    check('permanent 401 made exactly 2 requests', requestLog.length === 2)
    check('token value not in thrown message', caught instanceof Error && !includesToken(caught.message))
    check('token value not in thrown cause', caught instanceof Error && !includesToken(JSON.stringify(caught.cause ?? '')))
    reset()
  }

  // 4. 403 → SidecarAuthError kind forbidden, no retry (1 request)
  {
    setBridge(true)
    clearSidecarTokens()
    invokeImpl = async (cmd) => cmd === 'sidecar_token' ? 'whatever' : null
    cfg.forceStatus = 403
    let caught: unknown = null
    try { await sidecarFetch('llm', `${url}/echo`) }
    catch (e) { caught = e }
    check('403 throws', caught instanceof SidecarAuthError)
    check('403 kind forbidden', caught instanceof SidecarAuthError && caught.kind === 'forbidden')
    check('403 does not retry', requestLog.length === 1)
    check('403 message mentions Origin/Host', caught instanceof Error && caught.message.includes('Origin/Host'))
    reset()
  }

  // 5. non-Tauri (no bridge) → no header, works against a tokenless fake
  {
    setBridge(false)
    clearSidecarTokens()
    cfg.requiredToken = null
    const r = await sidecarFetch('llm', `${url}/echo`, { method: 'GET' })
    const j = await r.json() as { got: Record<string, string | string[] | undefined> }
    check('non-Tauri: no token header sent', j.got['x-spinoml-token'] === undefined)
    check('non-Tauri: invoke never called', invokeCalls === 0)
    reset()
  }

  // 6. invoke rejects → no header + error text visible on SidecarAuthError
  {
    setBridge(true)
    clearSidecarTokens()
    cfg.requiredToken = 'good'
    invokeImpl = async () => { throw new Error('sidecar_token rejected: no Rust command') }
    let caught: unknown = null
    try { await sidecarFetch('llm', `${url}/echo`) }
    catch (e) { caught = e }
    check('invoke-rejects: throws SidecarAuthError', caught instanceof SidecarAuthError)
    check('invoke-rejects: no token header ever sent', requestLog.every((r) => r.token === null))
    check('invoke-rejects: cause carries invoke error text',
      caught instanceof SidecarAuthError && String(caught.cause ?? '').includes('sidecar_token rejected'))
    check('invoke-rejects: token value not in message', caught instanceof Error && !includesToken(caught.message))
    reset()
  }

  // 7. network refused → rejects with the original error; probe says offline
  {
    setBridge(true)
    clearSidecarTokens()
    invokeImpl = async (cmd) => cmd === 'sidecar_token' ? SEC : null
    const dead = baseUrl.replace(/:\d+$/, ':1') // unassigned port → ECONNREFUSED
    let rejected: unknown = null
    try { await sidecarFetch('llm', `${dead}/echo`) }
    catch (e) { rejected = e }
    check('network refused: original error propagates', rejected instanceof Error && !(rejected instanceof SidecarAuthError))
    const probe = await probeSidecarHealth('llm', dead)
    check('probe on refused port → offline', probe.state === 'offline')
    check('probe offline has a message', typeof probe.message === 'string' && probe.message.length > 0)
    reset()
  }

  // 8. probe: online (full body with auth)
  {
    setBridge(true)
    clearSidecarTokens()
    invokeImpl = async (cmd) => cmd === 'sidecar_token' ? SEC : null
    cfg.health = { ok: true, auth: 'token', requiresAuth: true, tokenOk: true, version: 'x' }
    const p = await probeSidecarHealth('llm', url)
    check('probe online', p.state === 'online')
    check('probe reports auth=token', p.auth === 'token')
    reset()
  }

  // 9. probe: auth-failed (tokenOk:false body)
  {
    setBridge(true)
    clearSidecarTokens()
    invokeImpl = async (cmd) => cmd === 'sidecar_token' ? 'wrong' : null
    cfg.health = { ok: true, auth: 'token', requiresAuth: true, tokenOk: false }
    const p = await probeSidecarHealth('llm', url)
    check('probe auth-failed on tokenOk:false', p.state === 'auth-failed')
    check('probe auth-failed carries German message',
      typeof p.message === 'string' && p.message.includes('Authentifizierung'))
    check('probe auth-failed keeps auth=token', p.auth === 'token')
    reset()
  }

  // 10. probe: unauthenticated-dev (no token required)
  {
    setBridge(false)
    clearSidecarTokens()
    cfg.health = { ok: true, auth: 'unauthenticated-dev', requiresAuth: false, tokenOk: true }
    const p = await probeSidecarHealth('llm', url)
    check('probe unauth-dev → online', p.state === 'online')
    check('probe unauth-dev reports auth mode', p.auth === 'unauthenticated-dev')
    reset()
  }

  // 10b. probe retry on tokenOk:false (stale token → refreshed token → online)
  {
    setBridge(true)
    clearSidecarTokens()
    cfg.expectedToken = 'new'
    let n = 0
    invokeImpl = async (cmd) => {
      if (cmd !== 'sidecar_token') return null
      n++
      return n === 1 ? 'stale' : 'new'
    }
    const p = await probeSidecarHealth('llm', url)
    const healthCount = requestLog.filter((r) => r.path === '/health').length
    check('(a) stale token: probe online after refresh', p.state === 'online')
    check('(a) stale token: exactly 2 /health requests', healthCount === 2, `got ${healthCount}`)
    check('(a) stale token: invoke refreshed once', invokeCalls === 2)
    reset()
  }

  // 10c. probe retry: permanently wrong token → auth-failed after exactly 2
  {
    setBridge(true)
    clearSidecarTokens()
    cfg.expectedToken = 'good'
    invokeImpl = async (cmd) => cmd === 'sidecar_token' ? 'wrong' : null
    const p = await probeSidecarHealth('llm', url)
    const healthCount = requestLog.filter((r) => r.path === '/health').length
    check('(b) permanently wrong: auth-failed', p.state === 'auth-failed')
    check('(b) permanently wrong: exactly 2 /health requests (no loop)', healthCount === 2, `got ${healthCount}`)
    reset()
  }

  // 10d. tokenless dev sidecar (no tokenOk key / unauthenticated-dev) → 1 request
  {
    setBridge(false)
    clearSidecarTokens()
    cfg.health = { ok: true, auth: 'unauthenticated-dev', requiresAuth: false }
    const p = await probeSidecarHealth('llm', url)
    const healthCount = requestLog.filter((r) => r.path === '/health').length
    check('(c) tokenless dev: online', p.state === 'online')
    check('(c) tokenless dev: exactly 1 /health request', healthCount === 1, `got ${healthCount}`)
    reset()
  }

  // 10e. no token available (browser mode) against a token-requiring sidecar
  {
    setBridge(false)
    clearSidecarTokens()
    cfg.expectedToken = 'good'
    const p = await probeSidecarHealth('llm', url)
    const healthCount = requestLog.filter((r) => r.path === '/health').length
    check('(d) browser vs token-sidecar: auth-failed', p.state === 'auth-failed')
    check('(d) browser vs token-sidecar: no retry spam (<=2 /health)', healthCount <= 2 && healthCount >= 1, `got ${healthCount}`)
    reset()
  }

  // 11. streaming: first chunk readable before server ends (not buffered)
  {
    setBridge(true)
    clearSidecarTokens()
    invokeImpl = async (cmd) => cmd === 'sidecar_token' ? SEC : null
    cfg.requiredToken = SEC
    const r = await sidecarFetch('llm', `${url}/stream`, { method: 'GET' })
    check('streaming 200', r.ok)
    const reader = r.body?.getReader()
    check('streaming response has a body reader', !!reader)
    if (reader) {
      const first = await reader.read()
      check('first streaming chunk readable before server ends',
        !first.done && new TextDecoder().decode(first.value).includes('first') && !cfg.streamEnded)
      try { await reader.cancel() } catch { /* server may already have ended */ }
    }
    reset()
  }

  // 12. AbortController on a slow request → AbortError
  {
    setBridge(true)
    clearSidecarTokens()
    invokeImpl = async (cmd) => cmd === 'sidecar_token' ? SEC : null
    cfg.requiredToken = SEC
    const ctrl = new AbortController()
    const promise = sidecarFetch('llm', `${url}/slow`, { method: 'GET', signal: ctrl.signal })
    // abort almost immediately so fetch rejects before headers arrive
    setTimeout(() => ctrl.abort(), 20)
    let caught: unknown = null
    try { await promise } catch (e) { caught = e }
    check('abort: rejects with AbortError',
      caught instanceof DOMException && caught.name === 'AbortError')
    reset()
  }

  // 13. clearSidecarTokens / getSidecarToken cache behaviour
  {
    setBridge(true)
    clearSidecarTokens()
    invokeImpl = async (cmd) => cmd === 'sidecar_token' ? 'A' : null
    const a1 = await getSidecarToken('llm')
    const a2 = await getSidecarToken('llm')
    check('getSidecarToken caches: 1 invoke for 2 calls', a1 === 'A' && a2 === 'A' && invokeCalls === 1)
    clearSidecarTokens('llm')
    invokeImpl = async (cmd) => cmd === 'sidecar_token' ? 'B' : null
    const b1 = await getSidecarToken('llm')
    check('clearSidecarTokens forces re-read', b1 === 'B' && invokeCalls === 2)
    const b2 = await getSidecarToken('llm', { refresh: true })
    check('refresh:true forces re-read', b2 === 'B' && invokeCalls === 3)
    reset()
  }

  setBridge(false)
}

try {
  await run()
} finally {
  server.close()
}

console.log(`\n${failed === 0 ? '✓ all sidecar-auth frontend tests passed' : `✗ ${failed} check(s) failed`} (${passed} passed)`)
process.exit(failed === 0 ? 0 : 1)
