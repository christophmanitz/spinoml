// Phase 77/78 — real-process test for the torch sidecar's auth / Host / Origin
// layer (sidecar-torch/auth.py + main.py). See docs/engineering/SIDECAR_AUTH.md.
//
// It starts the REAL sidecar (never a mock) in three configurations and fires
// hostile raw HTTP at it (duplicate headers, missing Host, arbitrary Origin) —
// inputs fetch() cannot produce. Every rejection is checked for its status,
// machine code and the secrecy of the token; a `--mutate` self-check is left to
// the report (see package.json / CLAUDE.md) because the mutation must be in the
// sidecar source, not in the test.
//
// Run: npm run test:sidecar-auth-torch

import { spawnSync } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import { mkdtempSync, rmSync, existsSync } from 'node:fs'
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

const HERE = dirname(fileURLToPath(import.meta.url))
const REPO = resolve(HERE, '..')
const PYTHON = process.env.PYTHON ?? 'python'
const SIDECAR = join(REPO, 'sidecar-torch', 'main.py')
const TOKEN = randomBytes(32).toString('hex') // 64 hex chars
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

// ── unit tests for auth.py helpers ───────────────────────────────────────
function runUnitTests(): void {
  const r = spawnSync(PYTHON, [join(REPO, 'sidecar-torch', 'test_auth.py')], { encoding: 'utf8' })
  record('auth.py unit tests', 'exit 0', `exit ${r.status}`, r.status === 0)
  if (r.status !== 0) {
    console.log(r.stdout)
    console.log(r.stderr)
  }
}

const ENDPOINTS: Array<{ method: string; path: string }> = [
  { method: 'POST', path: '/infer' },
  { method: 'POST', path: '/dataset/inspect' },
  { method: 'POST', path: '/dataset/stats' },
  { method: 'POST', path: '/dataset/smoke' },
  { method: 'POST', path: '/activations' },
  { method: 'POST', path: '/deps/check' },
  { method: 'POST', path: '/deps/install' },
  { method: 'POST', path: '/run_script' },
  { method: 'POST', path: '/unknown-endpoint' },
]

const BAD_VARIANTS: Array<{ name: string; make: (path: string) => { path: string; opts: ReqOpts } }> = [
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

async function testTokenMode(ws: string): Promise<void> {
  const marker = join(ws, 'AUTH_MARKER_PWNED')
  const sidecar = await startSidecar({
    cmd: PYTHON,
    args: [SIDECAR],
    cwd: REPO,
    readyTimeoutMs: 120000,
    stripEnvPrefix: 'SPINOML_',
    portEnv: 'SPINOML_TORCH_PORT',
    env: {
      SPINOML_SIDECAR_TOKEN: TOKEN,
      SPINOML_REQUIRE_TOKEN: '1',
      SPINOML_ALLOWED_ROOTS: ws,
      SPINOML_ALLOWED_ORIGINS: 'http://my.tool:9',
    },
  })
  try {
    const port = sidecar.port
    console.log(`\ntoken-mode sidecar on ${sidecar.url}`)

    // Every endpoint rejects every malformed credential with 401 UNAUTHORIZED.
    for (const ep of ENDPOINTS) {
      for (const v of BAD_VARIANTS) {
        const { path, opts } = v.make(ep.path)
        const res = await req(port, path, { method: ep.method, ...opts })
        const body = json(res)
        const got = `${res.status} ${String(body.error_code ?? '')}`
        const pass = res.status === 401 && body.error_code === 'UNAUTHORIZED'
        record(`${ep.method} ${ep.path} [${v.name}]`, '401 UNAUTHORIZED', got, pass)
      }
      const res = await req(port, ep.path, { method: ep.method, token: TOKEN })
      record(
        `${ep.method} ${ep.path} [valid token]`,
        'not 401/403',
        String(res.status),
        res.status !== 401 && res.status !== 403,
      )
    }

    // reason = "missing" for an absent token.
    {
      const body = json(await req(port, '/infer', {}))
      record('401 reason for missing token', 'missing', String(body.reason ?? ''), body.reason === 'missing')
    }

    // A rejected /run_script must not have executed anything.
    {
      const code = `open(${JSON.stringify(marker)}, 'w').write('pwn')`
      const res = await req(port, '/run_script', {
        token: BAD_TOKEN,
        body: JSON.stringify({ root: ws, relpath: 'auth_marker.py', code, mode: 'shell' }),
      })
      const notRun = !existsSync(marker) && !existsSync(join(ws, 'auth_marker.py'))
      record('rejected /run_script did not execute', 'no marker file', notRun ? 'no marker' : 'MARKER WRITTEN', res.status === 401 && notRun)
    }

    // Host checks.
    {
      const cases: Array<{ name: string; opts: ReqOpts; expect: string }> = [
        { name: 'Host evil.example', opts: { token: TOKEN, host: 'evil.example' }, expect: 'BAD_HOST' },
        { name: 'Host 127.0.0.1.evil.com', opts: { token: TOKEN, host: '127.0.0.1.evil.com' }, expect: 'BAD_HOST' },
        { name: 'missing Host (HTTP/1.0)', opts: { token: TOKEN, host: null, version: '1.0' }, expect: 'BAD_HOST' },
      ]
      for (const c of cases) {
        const res = await req(port, '/infer', c.opts)
        const body = json(res)
        record(c.name, `403 ${c.expect}`, `${res.status} ${String(body.error_code ?? '')}`, res.status === 403 && body.error_code === c.expect)
      }
      const okRes = await req(port, '/infer', { token: TOKEN, host: 'localhost:12345' })
      record('Host localhost:12345 OK', 'not 401/403', String(okRes.status), okRes.status !== 401 && okRes.status !== 403)
    }

    // Origin checks.
    {
      const evil = await req(port, '/infer', { token: TOKEN, origin: 'https://evil.example' })
      const evilBody = json(evil)
      record('Origin https://evil.example', '403 BAD_ORIGIN', `${evil.status} ${String(evilBody.error_code ?? '')}`, evil.status === 403 && evilBody.error_code === 'BAD_ORIGIN')
      record('evil Origin has no ACAO', 'absent', evil.headers['access-control-allow-origin'] ? 'present' : 'absent', evil.headers['access-control-allow-origin'] === undefined)

      const nul = await req(port, '/infer', { token: TOKEN, origin: 'null' })
      record('Origin null', '403 BAD_ORIGIN', `${nul.status} ${String(json(nul).error_code ?? '')}`, nul.status === 403 && json(nul).error_code === 'BAD_ORIGIN')

      for (const origin of DEFAULT_ORIGINS) {
        const res = await req(port, '/infer', { token: TOKEN, origin })
        const acao = res.headers['access-control-allow-origin']?.[0]
        const vary = res.headers['vary']?.join(',') ?? ''
        const pass = res.status === 400 && acao === origin && vary.includes('Origin') && acao !== '*'
        record(`Origin ${origin} echoed`, `ACAO=${origin}`, `status=${res.status} acao=${acao ?? '-'}`, pass)
      }

      const extra = await req(port, '/infer', { token: TOKEN, origin: 'http://my.tool:9' })
      record('extra origin my.tool:9 OK', 'not 403', String(extra.status), extra.status !== 403)
      const extraBad = await req(port, '/infer', { token: TOKEN, origin: 'http://my.tool:99' })
      record('near-miss origin my.tool:99', '403 BAD_ORIGIN', `${extraBad.status} ${String(json(extraBad).error_code ?? '')}`, extraBad.status === 403 && json(extraBad).error_code === 'BAD_ORIGIN')
    }

    // OPTIONS preflight.
    {
      const allowed = await req(port, '/infer', { method: 'OPTIONS', origin: 'http://localhost:5173' })
      const allowHeaders = allowed.headers['access-control-allow-headers']?.[0] ?? ''
      record('OPTIONS allowed Origin', '204', String(allowed.status), allowed.status === 204)
      record('OPTIONS Allow-Headers has token', 'contains X-SpinoML-Token', allowHeaders, allowHeaders.includes('X-SpinoML-Token'))
      const evil = await req(port, '/infer', { method: 'OPTIONS', origin: 'https://evil.example' })
      record('OPTIONS evil Origin', '403', String(evil.status), evil.status === 403)
      record('OPTIONS evil Origin no ACAO', 'absent', evil.headers['access-control-allow-origin'] ? 'present' : 'absent', evil.headers['access-control-allow-origin'] === undefined)
      const noToken = await req(port, '/infer', { method: 'OPTIONS' })
      record('OPTIONS without token', '204', String(noToken.status), noToken.status === 204)
      record('OPTIONS no Origin no ACAO', 'absent', noToken.headers['access-control-allow-origin'] ? 'present' : 'absent', noToken.headers['access-control-allow-origin'] === undefined)
    }

    // /health.
    {
      const limited = json(await req(port, '/health', { method: 'GET' }))
      const expectedLimited = { ok: true, auth: 'token', requiresAuth: true, tokenOk: false }
      record(
        '/health no token exact body',
        'limited triple',
        canonical(limited),
        canonical(limited) === canonical(expectedLimited) && !('torch' in limited) && !('scope' in limited),
      )
      const wrong = json(await req(port, '/health', { method: 'GET', token: BAD_TOKEN }))
      record('/health wrong token same limited', 'limited triple', canonical(wrong), canonical(wrong) === canonical(expectedLimited))
      const full = json(await req(port, '/health', { method: 'GET', token: TOKEN }))
      const fullOk = full.tokenOk === true && full.auth === 'token' && full.requiresAuth === true && typeof full.torch === 'string' && typeof full.scope === 'object'
      record('/health right token full body', 'torch+scope+tokenOk', `tokenOk=${String(full.tokenOk)}`, fullOk)
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

    // Secrecy: a child process must not inherit the token.
    {
      const code =
        "import os\nprint('TOKEN_PRESENT=' + ('1' if 'SPINOML_SIDECAR_TOKEN' in os.environ else '0'))\n"
      const res = await req(port, '/run_script', {
        token: TOKEN,
        body: JSON.stringify({ root: ws, relpath: 'env_probe.py', code, mode: 'shell' }),
      })
      const body = json(res)
      const stdout = String(body.stdout ?? '')
      record('child env has no token', 'TOKEN_PRESENT=0', stdout.includes('TOKEN_PRESENT=0') ? 'TOKEN_PRESENT=0' : stdout.trim().slice(0, 40), res.status === 200 && stdout.includes('TOKEN_PRESENT=0'))
      record('run_script response hides token', 'no token', leaksToken(JSON.stringify(body), TOKEN) ? 'LEAK' : 'clean', !leaksToken(JSON.stringify(body), TOKEN))
    }

    const leaked = leaksToken(sidecar.output(), TOKEN)
    record('sidecar stdout/stderr hides token', 'no token/16-char substring', leaked ? 'LEAK' : 'clean', !leaked)
  } finally {
    await sidecar.stop()
  }
}

async function testDevMode(ws: string): Promise<void> {
  const sidecar = await startSidecar({
    cmd: PYTHON,
    args: [SIDECAR],
    cwd: REPO,
    readyTimeoutMs: 120000,
    stripEnvPrefix: 'SPINOML_',
    portEnv: 'SPINOML_TORCH_PORT',
    env: { SPINOML_ALLOWED_ROOTS: ws },
  })
  try {
    const port = sidecar.port
    console.log(`\nunauthenticated-dev sidecar on ${sidecar.url}`)

    const infer = await req(port, '/infer', {})
    record('dev: request without token works', 'not 401/403', String(infer.status), infer.status !== 401 && infer.status !== 403)

    const health = json(await req(port, '/health', { method: 'GET' }))
    const healthOk =
      health.auth === 'unauthenticated-dev' &&
      health.requiresAuth === false &&
      health.tokenOk === true &&
      typeof health.torch === 'string' &&
      typeof health.scope === 'object'
    record('dev: /health full body', 'auth=unauthenticated-dev', `auth=${String(health.auth)} requiresAuth=${String(health.requiresAuth)}`, healthOk)

    const evilOrigin = await req(port, '/infer', { origin: 'https://evil.example' })
    record('dev: evil Origin still 403', '403 BAD_ORIGIN', `${evilOrigin.status} ${String(json(evilOrigin).error_code ?? '')}`, evilOrigin.status === 403 && json(evilOrigin).error_code === 'BAD_ORIGIN')

    const evilHost = await req(port, '/infer', { host: 'evil.example' })
    record('dev: evil Host still 403', '403 BAD_HOST', `${evilHost.status} ${String(json(evilHost).error_code ?? '')}`, evilHost.status === 403 && json(evilHost).error_code === 'BAD_HOST')
  } finally {
    await sidecar.stop()
  }
}

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
    const port = await startSidecarPort()
    const result = await expectExit(
      {
        cmd: PYTHON,
        args: [SIDECAR],
        cwd: REPO,
        timeoutMs: 120000,
        stripEnvPrefix: 'SPINOML_',
        env: { ...c.env, SPINOML_TORCH_PORT: String(port) },
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

async function startSidecarPort(): Promise<number> {
  return await freePort()
}

async function main(): Promise<void> {
  const ws = mkdtempSync(join(tmpdir(), 'spinoml-auth-'))
  try {
    runUnitTests()
    await testTokenMode(ws)
    await testDevMode(ws)
    await testConfigErrors()
    record('never emits Access-Control-Allow-Origin: *', 'no wildcard', sawWildcard ? 'WILDCARD SEEN' : 'clean', !sawWildcard)
  } finally {
    rmSync(ws, { recursive: true, force: true })
  }

  const failures = runMatrix(rows)
  console.log(`\n${rows.length - failures.length}/${rows.length} rows passed`)
  if (failures.length) {
    console.log('\nFAILURES:')
    for (const f of failures) console.log(`  ✗ ${f.case}: expected ${f.expected}, got ${f.got}`)
    process.exit(1)
  }
}

void main()
