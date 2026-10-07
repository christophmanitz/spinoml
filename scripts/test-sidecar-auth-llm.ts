// Phase 77/78 — real-process test for the LLM sidecar's auth / Host / Origin
// layer (sidecar-llm/auth.mjs + main.mjs). See docs/engineering/SIDECAR_AUTH.md.
//
// Starts the REAL sidecar (never a mock) in three configurations and fires
// hostile raw HTTP at it (duplicate headers, missing Host, arbitrary Origin)
// — inputs fetch() cannot produce. Every rejection is checked for its status,
// machine code and the secrecy of the token; the ask round trip is driven by
// the FakeOpenAI server; the `/internal/mcp` route is probed with both the
// master token and a random string to prove the master token is NOT accepted
// there; a mutation proof temporarily removes the gates and re-runs.
//
// Existing tests (test-llm-safety, …) run without a token and must stay
// green: this file only adds new helpers to scripts/lib/llm-harness.ts.
//
// Run: npm run test:sidecar-auth-llm

import { spawnSync } from 'node:child_process'
import { randomBytes, randomUUID } from 'node:crypto'
import { chmodSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import net from 'node:net'
import { networkInterfaces, tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  expectExit,
  freePort,
  rawRequest,
  runMatrix,
  startSidecar,
  type MatrixRow,
  type RawResponse,
} from './lib/auth-probe'
import { LlmHarness } from './lib/llm-harness.ts'
import type { FakeStep, ToolCallSpec } from './lib/fake-openai.ts'
import { createServer as createHttpServer } from 'node:http'

// The /chat probes below must reach the handler and START a turn, but a turn must never reach a
// real provider. With no `llm` config the sidecar defaults to the `subscription` kind — the real
// Claude CLI / OAuth: slow (~10 s per probe), network- and quota-dependent, and a hidden side effect
// of a security test. So every probe names an `openai-compat` provider on a loopback server that
// refuses at once with HTTP 401 (the SDK does not retry 401).
const refusingProvider = createHttpServer((_req, res) => {
  res.writeHead(401, { 'content-type': 'application/json' })
  res.end('{"error":{"message":"refused by the test provider"}}')
})
await new Promise<void>((done) => refusingProvider.listen(0, '127.0.0.1', () => done()))
const providerPort = (refusingProvider.address() as net.AddressInfo).port
const CHAT_PROBE = JSON.stringify({
  user: 'x',
  llm: { kind: 'openai-compat', apiKey: 'sk-test-refused', baseUrl: `http://127.0.0.1:${providerPort}/v1`, model: 'm' },
})

const HERE = dirname(fileURLToPath(import.meta.url))
const REPO = resolve(HERE, '..')
const SIDECAR = join(REPO, 'sidecar-llm', 'main.mjs')
const NODE = process.execPath
const TOKEN = randomBytes(32).toString('hex')
const BAD_TOKEN = 'b'.repeat(64)

const DEFAULT_ORIGINS = [
  'tauri://localhost',
  'http://tauri.localhost',
  'https://tauri.localhost',
  'http://localhost:5173',
  'http://127.0.0.1:5173',
]

const rows: MatrixRow[] = []
let sawWildcard = false

function record(name: string, expected: string, got: string, pass?: boolean): void {
  rows.push({ case: name, expected, got, pass })
}

type ReqOpts = {
  method?: string
  token?: string | string[]
  origin?: string
  host?: string | null
  extraHeaders?: Record<string, string | string[]>
  body?: string
  version?: '1.0' | '1.1'
}

async function req(port: number, path: string, o: ReqOpts = {}): Promise<RawResponse> {
  const headers: Record<string, string | string[]> = {}
  if (o.host !== null) headers.Host = o.host ?? '127.0.0.1'
  if (o.origin !== undefined) headers.Origin = o.origin
  if (o.token !== undefined) headers['X-SpinoML-Token'] = o.token
  Object.assign(headers, o.extraHeaders ?? {})
  const res = await rawRequest({
    port,
    method: o.method ?? 'POST',
    path,
    headers,
    body: o.body ?? '',
    version: o.version,
  })
  const acao = res.headers['access-control-allow-origin']
  if (acao?.some((v) => v === '*')) sawWildcard = true
  return res
}

function json(res: RawResponse): Record<string, unknown> {
  try {
    return JSON.parse(res.body) as Record<string, unknown>
  } catch {
    return {}
  }
}

function corsHeaderNames(headers: Record<string, string[]>): string[] {
  return Object.keys(headers).filter((h) => h.startsWith('access-control-') || h === 'vary')
}

function canonical(obj: Record<string, unknown>): string {
  const keys = Object.keys(obj).sort()
  const ordered: Record<string, unknown> = {}
  for (const k of keys) ordered[k] = obj[k]
  return JSON.stringify(ordered)
}

function leaksToken(haystack: string, token: string): boolean {
  if (haystack.includes(token)) return true
  for (let i = 0; i + 16 <= token.length; i++) {
    if (haystack.includes(token.slice(i, i + 16))) return true
  }
  return false
}

/** First non-loopback IPv4 address of this machine (null on a host without one). */
function firstNonLoopbackIPv4(): string | null {
  for (const addrs of Object.values(networkInterfaces())) {
    for (const a of addrs ?? []) {
      if (a.family === 'IPv4' && !a.internal) return a.address
    }
  }
  return null
}

function portOpen(port: number, host = '127.0.0.1'): Promise<boolean> {
  return new Promise<boolean>((resolveP) => {
    const socket = net.connect({ host, port })
    let settled = false
    const done = (v: boolean): void => {
      if (settled) return
      settled = true
      socket.destroy()
      resolveP(v)
    }
    socket.once('connect', () => done(true))
    socket.once('error', () => done(false))
    setTimeout(() => done(false), 1500)
  })
}

const tc = (name: string, args: unknown): ToolCallSpec => ({
  name,
  arguments: typeof args === 'string' ? args : JSON.stringify(args),
})

// ── unit tests for auth.mjs helpers ─────────────────────────────────────────
// Phase 78: loadConfig must reject too-short / too-weird tokens, accept a 32+
// chars [A-Za-z0-9_-] one; checkHost must accept 127.0.0.1/localhost/[::1]
// (any port) and reject everything else; checkOrigin must be exact-match; the
// duplicate-header sentinel must be returned for a duplicated name.
function unitTest(source: string): { ok: boolean; detail: string; status: number | null } {
  const tmp = mkdtempSync(join(tmpdir(), 'spinoml-auth-llm-unit-'))
  const script = join(tmp, 'run.mjs')
  writeFileSync(script, source)
  try {
    const r = spawnSync(NODE, ['--check', script], { encoding: 'utf8' })
    if (r.status !== 0) return { ok: false, detail: `syntax: ${r.stderr || r.stdout}`, status: r.status }
    const r2 = spawnSync(NODE, [script], {
      encoding: 'utf8',
      env: { ...process.env, NODE_PATH: join(REPO, 'sidecar-llm', 'node_modules') },
    })
    if (r2.status !== 0) return { ok: false, detail: r2.stderr || r2.stdout, status: r2.status }
    return { ok: true, detail: r2.stdout.trim(), status: r2.status }
  } finally {
    rmSync(tmp, { recursive: true, force: true })
  }
}

function runUnitTests(): void {
  const ok = unitTest(`
    import {
      loadConfig, scrubEnv, checkHost, checkOrigin, tokenMatches,
      sessionSecretMatches, decide, decideFromHeaders, generateMcpSecret,
      CODE_UNAUTHORIZED, CODE_BAD_ORIGIN, CODE_BAD_HOST,
    } from ${JSON.stringify(join(REPO, 'sidecar-llm', 'auth.mjs'))}

    let fail = 0
    function t(name, cond, detail = '') {
      if (!cond) { console.error('FAIL', name, detail); fail++ } else { console.log('ok', name) }
    }

    // loadConfig: rejects too-short, weird chars; accepts a 64-hex one.
    t('loadConfig rejects 31 chars',
      (() => { try { loadConfig({ SPINOML_SIDECAR_TOKEN: 'a'.repeat(31) }); return false } catch { return true } })())
    t('loadConfig rejects space',
      (() => { try { loadConfig({ SPINOML_SIDECAR_TOKEN: 'a'.repeat(31) + ' ' }); return false } catch { return true } })())
    t('loadConfig rejects newline',
      (() => { try { loadConfig({ SPINOML_SIDECAR_TOKEN: 'a'.repeat(31) + String.fromCharCode(10) }); return false } catch { return true } })())
    t('loadConfig rejects unicode',
      (() => { try { loadConfig({ SPINOML_SIDECAR_TOKEN: 'a'.repeat(31) + '\\u00e9' }); return false } catch { return true } })())
    t('loadConfig accept 64 hex',
      (() => { const c = loadConfig({ SPINOML_SIDECAR_TOKEN: 'a'.repeat(64) }); return c.token === 'a'.repeat(64) })())
    t('loadConfig REQUIRE_TOKEN=1 without token throws',
      (() => { try { loadConfig({ SPINOML_REQUIRE_TOKEN: '1' }); return false } catch { return true } })())
    t('loadConfig unset token = unauthenticated-dev',
      (() => { const c = loadConfig({}); return c.token === null && !c.requireToken })())

    // scrubEnv: token disappears.
    {
      const env = { SPINOML_SIDECAR_TOKEN: 'abcdefghijklmnopqrstuvwxyz012345' }
      scrubEnv(env)
      t('scrubEnv deletes SPINOML_SIDECAR_TOKEN', !('SPINOML_SIDECAR_TOKEN' in env))
    }

    // checkHost: loopback only.
    t('checkHost 127.0.0.1', checkHost('127.0.0.1'))
    t('checkHost localhost', checkHost('localhost'))
    t('checkHost [::1]', checkHost('[::1]'))
    t('checkHost [::1]:7422', checkHost('[::1]:7422'))
    t('checkHost 127.0.0.1:7422', checkHost('127.0.0.1:7422'))
    t('checkHost evil.example', !checkHost('evil.example'))
    t('checkHost 127.0.0.1.evil.com', !checkHost('127.0.0.1.evil.com'))
    t('checkHost empty', !checkHost(''))
    t('checkHost null', !checkHost(null))
    t('checkHost [::1]:badport', !checkHost('[::1]:badport'))
    t('checkHost [::1 (no close', !checkHost('[::1'))

    // checkOrigin: exact match, no null, no slash, no case change.
    const cfg = { token: null, requireToken: false, origins: ['http://localhost:5173'] }
    t('checkOrigin allowed', checkOrigin('http://localhost:5173', cfg))
    t('checkOrigin none allowed', checkOrigin(undefined, cfg))
    t('checkOrigin null literal rejected', !checkOrigin('null', cfg))
    t('checkOrigin slash variant rejected', !checkOrigin('http://localhost:5173/', cfg))
    t('checkOrigin case rejected', !checkOrigin('HTTP://LOCALHOST:5173', cfg))
    t('checkOrigin evil rejected', !checkOrigin('https://evil.example', cfg))

    // tokenMatches: constant-time-ish, non-string rejected.
    const cfgT = { token: 'a'.repeat(64), requireToken: true, origins: [] }
    t('tokenMatches exact', tokenMatches('a'.repeat(64), cfgT))
    t('tokenMatches one-off', !tokenMatches('a'.repeat(63) + 'b', cfgT))
    t('tokenMatches shorter', !tokenMatches('a'.repeat(63), cfgT))
    t('tokenMatches longer', !tokenMatches('a'.repeat(65), cfgT))
    t('tokenMatches empty', !tokenMatches('', cfgT))
    t('tokenMatches null', !tokenMatches(null, cfgT))

    // sessionSecretMatches: same guarantees for the MCP secret.
    t('secret matches', sessionSecretMatches('xyz', 'xyz'))
    t('secret mismatch', !sessionSecretMatches('xyz', 'xyy'))
    t('secret empty', !sessionSecretMatches('', 'xyz'))

    // decideFromHeaders: duplicate-header detection.
    {
      const cfg2 = loadConfig({ SPINOML_SIDECAR_TOKEN: 'a'.repeat(64) })
      // Valid request, single header.
      const req = { headers: { host: '127.0.0.1', 'x-spinoml-token': 'a'.repeat(64) }, rawHeaders: ['host', '127.0.0.1', 'x-spinoml-token', 'a'.repeat(64)] }
      const d = decideFromHeaders('POST', '/chat', req, cfg2)
      t('decideFromHeaders ok single', d.kind === 'ok')
      // Duplicate header: Node would join with ", " but rawHeaders shows two.
      const dup = { headers: { host: '127.0.0.1', 'x-spinoml-token': 'a'.repeat(64) }, rawHeaders: ['host', '127.0.0.1', 'x-spinoml-token', 'a'.repeat(32), 'x-spinoml-token', 'a'.repeat(32)] }
      const d2 = decideFromHeaders('POST', '/chat', dup, cfg2)
      t('decideFromHeaders dup rejected', d2.kind === 'reject' && d2.code === 'unauthorized')
      // Missing Host → bad_host.
      const noHost = { headers: {}, rawHeaders: [] }
      const d3 = decideFromHeaders('POST', '/chat', noHost, cfg2)
      t('decideFromHeaders no host', d3.kind === 'reject' && d3.code === 'bad_host')
      // Evil Origin → bad_origin.
      const evil = { headers: { host: '127.0.0.1', 'x-spinoml-token': 'a'.repeat(64), origin: 'https://evil.example' }, rawHeaders: ['host', '127.0.0.1', 'x-spinoml-token', 'a'.repeat(64), 'origin', 'https://evil.example'] }
      const d4 = decideFromHeaders('POST', '/chat', evil, cfg2)
      t('decideFromHeaders evil origin', d4.kind === 'reject' && d4.code === 'bad_origin')
      // OPTIONS preflight → ok even without token.
      const opts = { headers: { host: '127.0.0.1', origin: 'http://localhost:5173' }, rawHeaders: ['host', '127.0.0.1', 'origin', 'http://localhost:5173'] }
      const d5 = decideFromHeaders('OPTIONS', '/chat', opts, cfg2)
      t('decideFromHeaders options allowed origin', d5.kind === 'ok' && d5.origin === 'http://localhost:5173')
      // OPTIONS evil origin → reject.
      const optsEvil = { headers: { host: '127.0.0.1', origin: 'https://evil.example' }, rawHeaders: ['host', '127.0.0.1', 'origin', 'https://evil.example'] }
      const d6 = decideFromHeaders('OPTIONS', '/chat', optsEvil, cfg2)
      t('decideFromHeaders options evil origin', d6.kind === 'reject' && d6.code === 'bad_origin')
    }

    // generateMcpSecret is hex 64 chars.
    {
      const s = generateMcpSecret()
      t('mcp secret length 64', s.length === 64)
      t('mcp secret hex', /^[0-9a-f]+$/.test(s))
    }

    process.exit(fail === 0 ? 0 : 1)
  `)
  record('auth.mjs unit tests', 'exit 0', `exit ${ok.status ?? '?'}`, ok.ok)
  if (!ok.ok) console.log(ok.detail)
}

// ── integration: token mode ─────────────────────────────────────────────────
const ENDPOINTS: Array<{ method: string; path: string; body: string }> = [
  { method: 'POST', path: '/chat', body: CHAT_PROBE },
  { method: 'POST', path: '/respond', body: '{"askId":"nope"}' },
  { method: 'GET', path: '/opencode/models', body: '' },
  { method: 'POST', path: '/unknown-endpoint', body: '' },
  { method: 'POST', path: '/internal/mcp/x/list', body: '{}' },
]

const BAD_VARIANTS: Array<{ name: string; make: (path: string) => { path: string; opts: ReqOpts; method?: string } }> = [
  { name: 'missing', make: (p) => ({ path: p, opts: {} }) },
  { name: 'wrong-same-length', make: (p) => ({ path: p, opts: { token: BAD_TOKEN } }) },
  { name: 'shorter', make: (p) => ({ path: p, opts: { token: TOKEN.slice(0, 31) } }) },
  { name: 'longer', make: (p) => ({ path: p, opts: { token: TOKEN + 'a' } }) },
  { name: 'strict-prefix', make: (p) => ({ path: p, opts: { token: TOKEN.slice(0, 63) } }) },
  { name: 'empty-header', make: (p) => ({ path: p, opts: { token: '' } }) },
  { name: 'query-string', make: (p) => ({ path: `${p}?token=${TOKEN}`, opts: {} }) },
  {
    name: 'authorization-bearer',
    make: (p) => ({ path: p, opts: { extraHeaders: { Authorization: `Bearer ${TOKEN}` } } }),
  },
  { name: 'duplicate-header', make: (p) => ({ path: p, opts: { token: [TOKEN, TOKEN] } }) },
]

async function testTokenMode(): Promise<void> {
  const sidecar = await startSidecar({
    cmd: NODE,
    args: [SIDECAR],
    cwd: REPO,
    readyTimeoutMs: 60000,
    stripEnvPrefix: 'SPINOML_',
    portEnv: 'SPINOML_LLM_PORT',
    env: {
      SPINOML_SIDECAR_TOKEN: TOKEN,
      SPINOML_REQUIRE_TOKEN: '1',
      SPINOML_ALLOWED_ORIGINS: 'http://my.tool:9',
    },
  })
  try {
    const port = sidecar.port
    console.log(`\ntoken-mode sidecar on ${sidecar.url}`)

    // Every endpoint rejects every malformed credential with 401 unauthorized
    // (for /internal/mcp the master token is NEVER accepted, see the dedicated
    // section below; here we assert the same 401 + unauthorised code shape).
    for (const ep of ENDPOINTS) {
      for (const v of BAD_VARIANTS) {
        const { path, opts } = v.make(ep.path)
        const res = await req(port, path, { method: ep.method, body: ep.body, ...opts })
        const body = json(res)
        const got = `${res.status} ${String(body.code ?? body.error_code ?? '')}`
        const pass = res.status === 401 && body.code === 'unauthorized'
        record(`${ep.method} ${ep.path} [${v.name}]`, '401 unauthorized', got, pass)
      }
      // With the right token, /internal/mcp still rejects (master token
      // invalid there), everything else reaches the handler (status != 401/403).
      const res = await req(port, ep.path, { method: ep.method, body: ep.body, token: TOKEN })
      if (ep.path.startsWith('/internal/mcp/')) {
        const body = json(res)
        record(`${ep.method} ${ep.path} [master token → 401]`, '401 unauthorized', `${res.status} ${String(body.code ?? '')}`, res.status === 401 && body.code === 'unauthorized')
      } else {
        record(`${ep.method} ${ep.path} [valid token]`, 'not 401/403', String(res.status), res.status !== 401 && res.status !== 403)
      }
    }

    // reason = "missing" for an absent token.
    {
      const res = await req(port, '/chat', { body: CHAT_PROBE })
      const body = json(res)
      record('401 reason for missing token', 'missing', String(body.reason ?? ''), body.reason === 'missing')
      record('401 no-Origin response has no CORS headers', 'none', corsHeaderNames(res.headers).join(',') || 'none', corsHeaderNames(res.headers).length === 0)
    }

    // Host checks.
    {
      const cases: Array<{ name: string; opts: ReqOpts; expect: string }> = [
        { name: 'Host evil.example', opts: { token: TOKEN, host: 'evil.example' }, expect: 'bad_host' },
        { name: 'Host 127.0.0.1.evil.com', opts: { token: TOKEN, host: '127.0.0.1.evil.com' }, expect: 'bad_host' },
        { name: 'missing Host (HTTP/1.0)', opts: { token: TOKEN, host: null, version: '1.0' }, expect: 'bad_host' },
      ]
      for (const c of cases) {
        const res = await req(port, '/chat', { ...c.opts, body: CHAT_PROBE })
        const body = json(res)
        record(c.name, `403 ${c.expect}`, `${res.status} ${String(body.code ?? '')}`, res.status === 403 && body.code === c.expect)
      }
      const okRes = await req(port, '/chat', { token: TOKEN, host: 'localhost:12345', body: CHAT_PROBE })
      record('Host localhost:12345 OK', 'not 401/403', String(okRes.status), okRes.status !== 401 && okRes.status !== 403)
    }

    // Origin checks.
    {
      const evil = await req(port, '/chat', { token: TOKEN, origin: 'https://evil.example', body: CHAT_PROBE })
      const evilBody = json(evil)
      record('Origin https://evil.example', '403 bad_origin', `${evil.status} ${String(evilBody.code ?? '')}`, evil.status === 403 && evilBody.code === 'bad_origin')
      record('evil Origin has no ACAO', 'absent', evil.headers['access-control-allow-origin'] ? 'present' : 'absent', evil.headers['access-control-allow-origin'] === undefined)
      record('evil Origin has no CORS headers at all', 'none', corsHeaderNames(evil.headers).join(',') || 'none', corsHeaderNames(evil.headers).length === 0)

      const nul = await req(port, '/chat', { token: TOKEN, origin: 'null', body: CHAT_PROBE })
      record('Origin null', '403 bad_origin', `${nul.status} ${String(json(nul).code ?? '')}`, nul.status === 403 && json(nul).code === 'bad_origin')

      for (const origin of DEFAULT_ORIGINS) {
        const res = await req(port, '/chat', { token: TOKEN, origin, body: CHAT_PROBE })
        const acao = res.headers['access-control-allow-origin']?.[0]
        const vary = res.headers['vary']?.join(',') ?? ''
        const pass = res.status === 200 && acao === origin && vary.includes('Origin') && acao !== '*'
        record(`Origin ${origin} echoed`, `ACAO=${origin}`, `status=${res.status} acao=${acao ?? '-'}`, pass)
      }

      const extra = await req(port, '/chat', { token: TOKEN, origin: 'http://my.tool:9', body: CHAT_PROBE })
      record('extra origin my.tool:9 OK', 'not 403', String(extra.status), extra.status !== 403)
      const extraBad = await req(port, '/chat', { token: TOKEN, origin: 'http://my.tool:99', body: CHAT_PROBE })
      record('near-miss origin my.tool:99', '403 bad_origin', `${extraBad.status} ${String(json(extraBad).code ?? '')}`, extraBad.status === 403 && json(extraBad).code === 'bad_origin')
    }

    // OPTIONS preflight.
    {
      const allowed = await req(port, '/chat', { method: 'OPTIONS', origin: 'http://localhost:5173' })
      const allowHeaders = allowed.headers['access-control-allow-headers']?.[0] ?? ''
      record('OPTIONS allowed Origin', '204', String(allowed.status), allowed.status === 204)
      record('OPTIONS Allow-Headers has token', 'contains X-SpinoML-Token', allowHeaders, allowHeaders.includes('X-SpinoML-Token'))
      const evil = await req(port, '/chat', { method: 'OPTIONS', origin: 'https://evil.example' })
      record('OPTIONS evil Origin', '403', String(evil.status), evil.status === 403)
      record('OPTIONS evil Origin no ACAO', 'absent', evil.headers['access-control-allow-origin'] ? 'present' : 'absent', evil.headers['access-control-allow-origin'] === undefined)
      record('OPTIONS evil Origin no CORS at all', 'none', corsHeaderNames(evil.headers).join(',') || 'none', corsHeaderNames(evil.headers).length === 0)
      const noToken = await req(port, '/chat', { method: 'OPTIONS' })
      record('OPTIONS without token', '204', String(noToken.status), noToken.status === 204)
      record('OPTIONS no Origin no ACAO', 'absent', noToken.headers['access-control-allow-origin'] ? 'present' : 'absent', noToken.headers['access-control-allow-origin'] === undefined)
      record('OPTIONS no Origin no CORS at all', 'none', corsHeaderNames(noToken.headers).join(',') || 'none', corsHeaderNames(noToken.headers).length === 0)
    }

    // /health shapes.
    {
      const limited = json(await req(port, '/health', { method: 'GET' }))
      const expectedLimited = { ok: true, auth: 'token', requiresAuth: true, tokenOk: false }
      record(
        '/health no token exact body',
        'limited triple',
        canonical(limited),
        canonical(limited) === canonical(expectedLimited) && !('scope' in limited),
      )
      const wrong = json(await req(port, '/health', { method: 'GET', token: BAD_TOKEN }))
      record('/health wrong token same limited', 'limited triple', canonical(wrong), canonical(wrong) === canonical(expectedLimited))
      const full = json(await req(port, '/health', { method: 'GET', token: TOKEN }))
      const fullOk = full.tokenOk === true && full.auth === 'token' && full.requiresAuth === true && full.ok === true
      record('/health right token full body', 'ok+auth+requiresAuth+tokenOk', `tokenOk=${String(full.tokenOk)}`, fullOk)
      const limitedHasDiag = 'diag' in limited
      const fullHasDiag = 'diag' in full
      record(
        '/health diag absent without token, present with',
        'absent/present',
        `${limitedHasDiag ? 'present' : 'absent'}/${fullHasDiag ? 'present' : 'absent'}`,
        !limitedHasDiag && fullHasDiag,
      )
    }

    // Bind address (Phase 78 "do not expose on 0.0.0.0"): the port must not accept
    // connections on a non-loopback interface of this machine.
    {
      const lan = firstNonLoopbackIPv4()
      if (lan === null) {
        console.log('  (bind-address check skipped: this host has no non-loopback IPv4 interface)')
      } else {
        const open = await portOpen(port, lan)
        record(`port not reachable on ${lan}`, 'refused', open ? 'OPEN' : 'refused', !open)
      }
    }

    // Secrecy: a child process must not inherit the token. The sidecar doesn't
    // expose a generic run_script; we drive a real /chat turn whose ask SSE
    // event would be the most likely leak vector if scrubEnv were skipped.
    const harness = await LlmHarness.connect({ baseUrl: sidecar.url })
    try {
      const sseText = await new Promise<string>((resolveP, rejectP) => {
        let buf = ''
        void fetch(`${harness.baseUrl}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-SpinoML-Token': TOKEN, Origin: 'http://localhost:5173' },
          body: JSON.stringify({ user: 'secrecy', messages: [], graph: { input_shape: [1, 4], nodes: [], edges: [] }, training_graph: { nodes: [], edges: [] }, data_graph: { nodes: [], edges: [] }, project: null, autoMode: false, llm: { kind: 'openai-compat', apiKey: 'sk-X', baseUrl: harness.fake.baseUrl, model: 'fake' } }),
        }).then(async (res) => {
          const reader = res.body?.getReader()
          const dec = new TextDecoder()
          if (!reader) return rejectP(new Error('no body'))
          // Read the stream to completion (or up to 5s) — we only need a
          // representative slice that exercises the SSE pipeline.
          const start = Date.now()
          while (Date.now() - start < 5000) {
            const { value, done } = await reader.read()
            if (done) break
            buf += dec.decode(value, { stream: true })
            if (buf.length > 8192) break
          }
          try { await reader.cancel() } catch { /* already gone */ }
          resolveP(buf)
        }).catch(rejectP)
      })
      record('SSE text hides token', 'no token', leaksToken(sseText, TOKEN) ? 'LEAK' : 'clean', !leaksToken(sseText, TOKEN))
      record('harness stdout hides token', 'no token', leaksToken(harness.stdout, TOKEN) ? 'LEAK' : 'clean', !leaksToken(harness.stdout, TOKEN))
      record('harness stderr hides token', 'no token', leaksToken(harness.stderr, TOKEN) ? 'LEAK' : 'clean', !leaksToken(harness.stderr, TOKEN))
    } finally {
      await harness.stop()
    }

    // Ask round trip with the fake OpenAI server.
    await testAskRoundTrip(sidecar.port)

    // /internal/mcp session secret: master token rejected, random string
    // rejected for both existing and unknown session ids with the SAME body.
    await testMcpSecrets()

    // Evil origin from the harness: the fake OpenAI server must see 0
    // requests because the sidecar short-circuits the turn at the auth gate.
    await testEvilOriginNoTurn(sidecar.port)

    const leaked = leaksToken(sidecar.output(), TOKEN)
    record('sidecar stdout/stderr hides token', 'no token/16-char substring', leaked ? 'LEAK' : 'clean', !leaked)
  } finally {
    await sidecar.stop()
  }
}

// Drive a turn that calls `ask_user`, capture the SSE ask id, and assert it
// is a UUID; verify /respond with wrong/missing token → 401 and the ask is
// STILL pending; /respond with the right token → 200 and the turn completes.
async function testAskRoundTrip(sidecarPort: number): Promise<void> {
  const harness = await LlmHarness.connect({ baseUrl: `http://127.0.0.1:${sidecarPort}` })
  try {
    // Pass onAsk that captures but does NOT return an answer (returning
    // undefined leaves the ask pending in the harness — the harness only
    // calls /respond when the callback returns a defined value).
    let observedAskId = ''
    const script: FakeStep[] = [
      { toolCalls: [tc('ask_user', { kind: 'select', prompt: 'pick', options: ['a', 'b'] })] },
      { text: 'done' },
    ]
    const chatPromise = harness.chat({
      script,
      token: TOKEN,
      onAsk: (ask) => {
        observedAskId = ask.id
        return undefined as unknown as string
      },
    })

    // Wait until the ask event is observed (the harness applies events
    // asynchronously; poll for ~10s).
    const deadline = Date.now() + 10_000
    while (!observedAskId && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 50))
    }
    record('ask event id is UUID', 'uuid format', observedAskId, /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(observedAskId))

    // /respond with wrong token → 401.
    {
      const r = await fetch(`${harness.baseUrl}/respond`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-SpinoML-Token': BAD_TOKEN },
        body: JSON.stringify({ askId: observedAskId, answer: 'a' }),
      })
      record('/respond wrong token', '401', String(r.status), r.status === 401)
    }
    // /respond with missing token → 401.
    {
      const r = await fetch(`${harness.baseUrl}/respond`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ askId: observedAskId, answer: 'a' }),
      })
      record('/respond missing token', '401', String(r.status), r.status === 401)
    }
    // Turn must STILL be pending after the rejected /responds (otherwise
    // the next right-token /respond would 404). Right token → 200.
    {
      const r = await fetch(`${harness.baseUrl}/respond`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-SpinoML-Token': TOKEN },
        body: JSON.stringify({ askId: observedAskId, answer: 'a' }),
      })
      record('/respond right token', '200', String(r.status), r.status === 200)
    }

    // The turn must complete now that the ask has been answered.
    const result = await Promise.race([
      chatPromise,
      new Promise<{ ended: boolean; aborted: boolean }>((r) => setTimeout(() => r({ ended: false, aborted: true }), 10_000)),
    ])
    record('turn completes after right /respond', 'ended', `ended=${String(result.ended)} aborted=${String(result.aborted)}`, result.ended === true)

    // /respond with an unknown askId → 404 (still authorised).
    {
      const r = await fetch(`${harness.baseUrl}/respond`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-SpinoML-Token': TOKEN },
        body: JSON.stringify({ askId: randomUUID(), answer: 'x' }),
      })
      record('/respond unknown askId', '404', String(r.status), r.status === 404)
    }
  } finally {
    await harness.stop()
  }
}

// A throw-away "opencode" the sidecar spawns for an opencode turn. It reads
// the inline MCP config (which carries the per-turn requestId in the command
// argv), writes `{requestId, tokenPresent, token}` to a marker file, then
// sleeps so the MCP session stays registered while the test probes it. This
// lets the test (a) know a real session id and (b) observe whether the child
// inherited SPINOML_SIDECAR_TOKEN (the scrubEnv invariant).
interface FakeOpencodeMarker {
  requestId: string
  tokenPresent: boolean
  token: string | null
}

function makeFakeOpencode(dir: string): { bin: string; marker: string } {
  const mjs = join(dir, 'fake-opencode.mjs')
  writeFileSync(
    mjs,
    [
      "import { writeFileSync } from 'node:fs'",
      "const cfg = process.env.OPENCODE_CONFIG_CONTENT || '{}'",
      "let requestId = ''",
      "try { requestId = JSON.parse(cfg)?.mcp?.graph?.command?.[2] ?? '' } catch { /* malformed config → empty */ }",
      'const marker = process.env.FAKE_OPENCODE_MARKER',
      'if (marker) {',
      '  writeFileSync(marker, JSON.stringify({',
      '    requestId,',
      "    tokenPresent: Object.prototype.hasOwnProperty.call(process.env, 'SPINOML_SIDECAR_TOKEN'),",
      "    token: process.env.SPINOML_SIDECAR_TOKEN ?? null,",
      '  }))',
      '}',
      'setInterval(() => {}, 1000)',
    ].join('\n'),
  )
  const bin = join(dir, 'fake-opencode.sh')
  writeFileSync(bin, `#!/bin/bash\nexec '${process.execPath}' '${mjs}'\n`)
  chmodSync(bin, 0o755)
  return { bin, marker: join(dir, 'marker.json') }
}

function readMarker(path: string): FakeOpencodeMarker | null {
  if (!existsSync(path)) return null
  try {
    return JSON.parse(readFileSync(path, 'utf8')) as FakeOpencodeMarker
  } catch {
    return null
  }
}

async function waitMarker(path: string, timeoutMs: number): Promise<FakeOpencodeMarker | null> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    const m = readMarker(path)
    if (m) return m
    await new Promise((r) => setTimeout(r, 100))
  }
  return null
}

// Fire an opencode turn in the background and return an abort handle. The
// fake opencode sleeps, so the turn (and its registered MCP session) stays
// alive until abort().
function driveOpencodeTurn(baseUrl: string, token: string): { abort: () => void } {
  const ctrl = new AbortController()
  void fetch(`${baseUrl}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'X-SpinoML-Token': token },
    body: JSON.stringify({
      user: 'register a session',
      messages: [],
      graph: { input_shape: [1, 4], nodes: [], edges: [] },
      training_graph: { nodes: [], edges: [] },
      data_graph: { nodes: [], edges: [] },
      project: null,
      autoMode: false,
      llm: { kind: 'opencode', model: 'fake' },
    }),
    signal: ctrl.signal,
  }).catch(() => undefined)
  return { abort: () => ctrl.abort() }
}

function mcpProbe(port: number, id: string, secret: string): Promise<RawResponse> {
  return rawRequest({
    port,
    method: 'POST',
    path: `/internal/mcp/${id}/list`,
    headers: { Host: '127.0.0.1', 'Content-Type': 'application/json', 'X-SpinoML-Token': secret },
    body: '{}',
  })
}

// /internal/mcp requires the per-turn session secret, in BOTH modes. The
// master token is never accepted there; a wrong secret and an unknown session
// id yield the SAME 401 body, so a caller cannot enumerate live sessions.
async function testMcpSecrets(): Promise<void> {
  const dir = mkdtempSync(join(tmpdir(), 'spinoml-auth-llm-mcp-'))
  const { bin, marker } = makeFakeOpencode(dir)
  let sidecar: Awaited<ReturnType<typeof startSidecar>> | null = null
  let turn: { abort: () => void } | null = null
  try {
    sidecar = await startSidecar({
      cmd: NODE,
      args: [SIDECAR],
      cwd: REPO,
      readyTimeoutMs: 30000,
      stripEnvPrefix: 'SPINOML_',
      portEnv: 'SPINOML_LLM_PORT',
      env: {
        SPINOML_SIDECAR_TOKEN: TOKEN,
        SPINOML_REQUIRE_TOKEN: '1',
        SPINOML_OPENCODE_BIN: bin,
        FAKE_OPENCODE_MARKER: marker,
      },
    })
    turn = driveOpencodeTurn(sidecar.url, TOKEN)
    const info = await waitMarker(marker, 20_000)
    record('fake opencode captured requestId', 'uuid', info?.requestId ?? 'none', !!info && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(info.requestId))
    record('opencode child env has no master token', 'tokenPresent=false', String(info?.tokenPresent ?? '?'), info?.tokenPresent === false)

    if (!sidecar || !info) return
    const existing = info.requestId
    const unknown = randomUUID()
    const masterOnExisting = await mcpProbe(sidecar.port, existing, TOKEN)
    record('/internal/mcp master token on existing session', '401 unauthorized', `${masterOnExisting.status} ${String(json(masterOnExisting).code ?? '')}`, masterOnExisting.status === 401 && json(masterOnExisting).code === 'unauthorized')
    const masterOnUnknown = await mcpProbe(sidecar.port, unknown, TOKEN)
    record('/internal/mcp master token on unknown session', '401 unauthorized', `${masterOnUnknown.status} ${String(json(masterOnUnknown).code ?? '')}`, masterOnUnknown.status === 401 && json(masterOnUnknown).code === 'unauthorized')
    const randomOnExisting = await mcpProbe(sidecar.port, existing, 'x'.repeat(64))
    const randomOnUnknown = await mcpProbe(sidecar.port, unknown, 'x'.repeat(64))
    record('/internal/mcp random on existing session', '401 unauthorized', `${randomOnExisting.status} ${String(json(randomOnExisting).code ?? '')}`, randomOnExisting.status === 401 && json(randomOnExisting).code === 'unauthorized')
    const bodyExisting = canonical(json(randomOnExisting))
    const bodyUnknown = canonical(json(randomOnUnknown))
    record('existing/unknown session 401 bodies identical', 'identical', bodyExisting === bodyUnknown ? 'identical' : 'DIFFER', bodyExisting === bodyUnknown)
  } finally {
    turn?.abort()
    if (sidecar) await sidecar.stop()
    rmSync(dir, { recursive: true, force: true })
  }
}

// Evil origin POST with a valid token must NOT start a turn.
async function testEvilOriginNoTurn(sidecarPort: number): Promise<void> {
  const harness = await LlmHarness.connect({ baseUrl: `http://127.0.0.1:${sidecarPort}` })
  try {
    const before = harness.fake.requests.length
    const r = await fetch(`${harness.baseUrl}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-SpinoML-Token': TOKEN, Origin: 'https://evil.example' },
      body: JSON.stringify({ user: 'evil', messages: [], graph: { input_shape: [1, 4], nodes: [], edges: [] }, training_graph: { nodes: [], edges: [] }, data_graph: { nodes: [], edges: [] }, project: null, autoMode: false, llm: { kind: 'openai-compat', apiKey: 'sk-X', baseUrl: harness.fake.baseUrl, model: 'fake' } }),
    })
    const after = harness.fake.requests.length
    record('evil Origin /chat status', '403', String(r.status), r.status === 403)
    record('evil Origin did not start turn', `fake.requests delta=0 (got ${String(after - before)})`, `delta=${String(after - before)}`, after - before === 0)
  } finally {
    await harness.stop()
  }
}

// ── integration: unauthenticated-dev mode ───────────────────────────────────
async function testDevMode(): Promise<void> {
  const sidecar = await startSidecar({
    cmd: NODE,
    args: [SIDECAR],
    cwd: REPO,
    readyTimeoutMs: 30000,
    stripEnvPrefix: 'SPINOML_',
    portEnv: 'SPINOML_LLM_PORT',
  })
  try {
    const port = sidecar.port
    console.log(`\nunauthenticated-dev sidecar on ${sidecar.url}`)

    const chat = await req(port, '/chat', { body: CHAT_PROBE })
    record('dev: /chat without token works', 'not 401/403', String(chat.status), chat.status !== 401 && chat.status !== 403)

    const health = json(await req(port, '/health', { method: 'GET' }))
    const healthOk =
      health.auth === 'unauthenticated-dev' &&
      health.requiresAuth === false &&
      health.tokenOk === true &&
      health.ok === true
    record('dev: /health full body', 'auth=unauthenticated-dev', `auth=${String(health.auth)} requiresAuth=${String(health.requiresAuth)}`, healthOk)

    const evilOrigin = await req(port, '/chat', { origin: 'https://evil.example', body: CHAT_PROBE })
    record('dev: evil Origin still 403', '403 bad_origin', `${evilOrigin.status} ${String(json(evilOrigin).code ?? '')}`, evilOrigin.status === 403 && json(evilOrigin).code === 'bad_origin')

    const evilHost = await req(port, '/chat', { host: 'evil.example', body: CHAT_PROBE })
    record('dev: evil Host still 403', '403 bad_host', `${evilHost.status} ${String(json(evilHost).code ?? '')}`, evilHost.status === 403 && json(evilHost).code === 'bad_host')
  } finally {
    await sidecar.stop()
  }
}

// ── config error startup cases ──────────────────────────────────────────────
async function testConfigErrors(): Promise<void> {
  console.log('\nconfig error startup')
  const cases: Array<{ name: string; env: Record<string, string>; tokenValue?: string }> = [
    { name: 'token 31 chars', env: { SPINOML_SIDECAR_TOKEN: 'a'.repeat(31) }, tokenValue: 'a'.repeat(31) },
    { name: 'token with space', env: { SPINOML_SIDECAR_TOKEN: 'a'.repeat(31) + ' ' }, tokenValue: 'a'.repeat(31) + ' ' },
    { name: 'token with newline', env: { SPINOML_SIDECAR_TOKEN: 'a'.repeat(31) + '\n' }, tokenValue: 'a'.repeat(31) + '\n' },
    { name: 'token with unicode', env: { SPINOML_SIDECAR_TOKEN: 'a'.repeat(31) + '\u00e9' }, tokenValue: 'a'.repeat(31) + '\u00e9' },
    { name: 'require token without token', env: { SPINOML_REQUIRE_TOKEN: '1' } },
  ]
  for (const c of cases) {
    const port = await freePort()
    const result = await expectExit(
      {
        cmd: NODE,
        args: [SIDECAR],
        cwd: REPO,
        timeoutMs: 30000,
        stripEnvPrefix: 'SPINOML_',
        env: { ...c.env, SPINOML_LLM_PORT: String(port) },
      },
      2,
    )
    record(`${c.name} exits 2`, 'exit 2', `exit ${String(result.code)}`, result.ok)
    const namesProblem = /SPINOML/.test(result.stderr)
    record(`${c.name} stderr names problem`, 'mentions SPINOML', namesProblem ? 'yes' : result.stderr.trim().slice(0, 60), namesProblem)
    if (c.tokenValue !== undefined) {
      record(`${c.name} stderr hides token`, 'no token value', result.stderr.includes(c.tokenValue) ? 'LEAK' : 'clean', !result.stderr.includes(c.tokenValue))
    }
    const open = await portOpen(port)
    record(`${c.name} never binds port`, 'closed', open ? 'OPEN' : 'closed', !open)
  }
}

// ── mutation proof ──────────────────────────────────────────────────────────
// For each mutation: back up the sources, apply the patch, spawn the sidecar,
// assert the security invariant BREAKS, then restore. The patches are string
// replacements in the sidecar sources. Shown red on purpose: each row asserts
// that the corresponding test WOULD fail without the gate.
async function mutationProof(): Promise<void> {
  console.log('\nmutation proof')
  const tmp = mkdtempSync(join(tmpdir(), 'spinoml-auth-llm-mut-'))
  const authPath = join(REPO, 'sidecar-llm', 'auth.mjs')
  const mainPath = join(REPO, 'sidecar-llm', 'main.mjs')
  const bridgePath = join(REPO, 'sidecar-llm', 'mcp-bridge.mjs')
  const orig = {
    auth: readFileSync(authPath, 'utf8'),
    main: readFileSync(mainPath, 'utf8'),
    bridge: readFileSync(bridgePath, 'utf8'),
  }
  const restore = (): void => {
    writeFileSync(authPath, orig.auth)
    writeFileSync(mainPath, orig.main)
    writeFileSync(bridgePath, orig.bridge)
  }
  const fakeDir = mkdtempSync(join(tmpdir(), 'spinoml-auth-llm-mutfake-'))
  const fake = makeFakeOpencode(fakeDir)

  // invariantHeld = the security property we are proving. The mutation must
  // make it false; a red row is a PASS.
  async function withMutatedSources(opts: {
    name: string
    auth?: string
    main?: string
    env?: Record<string, string>
    probe: (sidecar: Awaited<ReturnType<typeof startSidecar>>, token: string) => Promise<{ observed: string; invariantHeld: boolean }>
  }): Promise<void> {
    restore()
    if (opts.auth !== undefined) writeFileSync(authPath, opts.auth)
    if (opts.main !== undefined) writeFileSync(mainPath, opts.main)
    const token = randomBytes(32).toString('hex')
    let sidecar: Awaited<ReturnType<typeof startSidecar>>
    try {
      sidecar = await startSidecar({
        cmd: NODE,
        args: [SIDECAR],
        cwd: REPO,
        readyTimeoutMs: 30000,
        stripEnvPrefix: 'SPINOML_',
        portEnv: 'SPINOML_LLM_PORT',
        env: { SPINOML_SIDECAR_TOKEN: token, SPINOML_REQUIRE_TOKEN: '1', ...(opts.env ?? {}) },
      })
    } catch (e) {
      record(`mutation ${opts.name}`, 'breaks (red)', `start failed: ${(e as Error).message}`, true)
      restore()
      return
    }
    try {
      const { observed, invariantHeld } = await opts.probe(sidecar, token)
      record(`mutation ${opts.name}`, 'breaks (red)', observed, !invariantHeld)
    } finally {
      await sidecar.stop()
      restore()
    }
  }

  // (a) Remove the token check in decide() — every token is accepted.
  await withMutatedSources({
    name: 'a) token check removed',
    auth: orig.auth.replace(
      / {2}if \(!cfg\.requireToken\) return allow\(echoOrigin, true\)\n {2}const supplied = headersGet\('X-SpinoML-Token'\)/,
      "  if (!cfg.requireToken) return allow(echoOrigin, true)\n  // MUTATION: token check removed\n  return allow(echoOrigin, true)\n  const supplied = headersGet('X-SpinoML-Token')",
    ),
    probe: async (sidecar) => {
      const r = await req(sidecar.port, '/chat', { token: 'wrongwrongwrongwrongwrongwrongwrongwrongwrongwrongwrong' })
      return { observed: `status=${r.status}`, invariantHeld: r.status === 401 }
    },
  })

  // (b) Remove the Origin check — any origin is allowed.
  await withMutatedSources({
    name: 'b) origin check removed',
    auth: orig.auth.replace(
      / {2}if \(!okOrigin\) return reject\(403, CODE_BAD_ORIGIN\)/,
      '  if (!okOrigin) { /* MUTATION: origin check removed */ }',
    ),
    probe: async (sidecar) => {
      const r = await req(sidecar.port, '/chat', { origin: 'https://evil.example' })
      return { observed: `status=${r.status}`, invariantHeld: r.status === 403 }
    },
  })

  // (c) Skip scrubEnv — the opencode child inherits SPINOML_SIDECAR_TOKEN.
  await withMutatedSources({
    name: 'c) scrubEnv skipped',
    main: orig.main.replace(/scrubAuthEnv\(process\.env\)/, '/* MUTATION: scrubEnv skipped */'),
    env: { SPINOML_OPENCODE_BIN: fake.bin, FAKE_OPENCODE_MARKER: fake.marker },
    probe: async (sidecar, token) => {
      const turn = driveOpencodeTurn(sidecar.url, token)
      try {
        const info = await waitMarker(fake.marker, 20_000)
        return { observed: `childTokenPresent=${String(info?.tokenPresent)}`, invariantHeld: info?.tokenPresent === false }
      } finally {
        turn.abort()
        rmSync(fake.marker, { force: true })
      }
    },
  })

  // (d) Accept the master token on /internal/mcp — a probe with the master
  // token then reaches the tool list instead of a 401.
  await withMutatedSources({
    name: 'd) master token accepted on /internal/mcp',
    main: orig.main.replace(
      / {2}const ok = !!session && sessionSecretMatches\(suppliedStr, session\.secret\)/,
      '  const ok = !!session && (sessionSecretMatches(suppliedStr, session.secret) || suppliedStr === AUTH_CONFIG.token)',
    ),
    env: { SPINOML_OPENCODE_BIN: fake.bin, FAKE_OPENCODE_MARKER: fake.marker },
    probe: async (sidecar, token) => {
      const turn = driveOpencodeTurn(sidecar.url, token)
      try {
        const info = await waitMarker(fake.marker, 20_000)
        if (!info) return { observed: 'no session registered', invariantHeld: true }
        const r = await mcpProbe(sidecar.port, info.requestId, token)
        return { observed: `status=${r.status}`, invariantHeld: r.status === 401 }
      } finally {
        turn.abort()
        rmSync(fake.marker, { force: true })
      }
    },
  })

  rmSync(tmp, { recursive: true, force: true })
  rmSync(fakeDir, { recursive: true, force: true })
}

async function main(): Promise<void> {
  runUnitTests()
  await testTokenMode()
  await testDevMode()
  await testConfigErrors()
  await mutationProof()
  record('never emits Access-Control-Allow-Origin: *', 'no wildcard', sawWildcard ? 'WILDCARD SEEN' : 'clean', !sawWildcard)

  const failures = runMatrix(rows)
  console.log(`\n${rows.length - failures.length}/${rows.length} rows passed`)
  if (failures.length) {
    console.log('\nFAILURES:')
    for (const f of failures) console.log(`  ✗ ${f.case}: expected ${f.expected}, got ${f.got}`)
    process.exit(1)
  }
  // The refusing-provider server would keep the event loop (and so the suite) alive forever.
  refusingProvider.closeAllConnections()
  refusingProvider.close()
  process.exit(0)
}

void main()
