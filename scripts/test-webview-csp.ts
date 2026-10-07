// R052 / LIMITATIONS §2 — real-browser proof that:
//   (1) Monaco is BUNDLED (never fetched from cdn.jsdelivr.net at runtime),
//   (2) the production CSP from src-tauri/tauri.conf.json is ACTIVE and blocks
//       inline scripts + off-policy connect()s.
//
// The real Tauri webview cannot run here (no GTK/WebKit + built app), so this
// drives a real Chromium over the DevTools Protocol, serving the real `dist/`
// build with the real CSP header. No new dependencies: Node 22 global WebSocket.
//
// It builds `dist/` first, then:
//   * static-serves dist/ (SPA fallback) with the CSP header BUILT FROM the
//     tauri.conf.json object form,
//   * optionally fakes the torch/LLM /health sidecars so the app's own loopback
//     requests can be observed to be ALLOWED by `connect-src`,
//   * navigates, asserts render + Monaco mount + tokenized content,
//   * asserts no CSP violation / uncaught exception during load,
//   * asserts every request host is the static server or 127.0.0.1:7421/7422/7424
//     and that cdn.jsdelivr.net never appears,
//   * asserts the Monaco worker came from 'self'/'blob:',
//   * then runs negative probes (inline <script>; fetch example.com) and asserts
//     the CSP blocks both and logs a violation.
//
// Exit 1 on any failed check. If no Chromium can start, prints a SKIPPED line
// and exits 0 (never a silent pass).

import { spawn, spawnSync, type ChildProcess } from 'node:child_process'
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http'
import { createServer as createNetServer } from 'node:net'
import { existsSync, readFileSync, rmSync } from 'node:fs'
import { mkdtemp } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'
import { extname, join, resolve } from 'node:path'
import { tmpdir } from 'node:os'

const ROOT = resolve(fileURLToPath(new URL('.', import.meta.url)), '..')
const DIST = join(ROOT, 'dist')
const CONFIG = join(ROOT, 'src-tauri', 'tauri.conf.json')

const RENDER_TIMEOUT = 20_000
const EDITOR_TIMEOUT = 20_000
const POLL_MS = 250

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms))

function asRecord(v: unknown): Record<string, unknown> {
  return typeof v === 'object' && v !== null ? (v as Record<string, unknown>) : {}
}

function str(v: unknown): string {
  return typeof v === 'string' ? v : ''
}

/* ------------------------------------------------------------------ table -- */

type Row = { check: string; expected: string; got: string; pass: boolean }
const rows: Row[] = []
function record(check: string, expected: string, got: string, pass: boolean): void {
  rows.push({ check, expected, got, pass })
  const status = pass ? 'PASS' : 'FAIL'
  console.log(`  ${status}  ${check.padEnd(26)} expected=${expected}  got=${got}`)
}
function printTable(): void {
  console.log('')
  console.log('check | expected | got | PASS/FAIL')
  console.log('------+----------+-----+----------')
  for (const r of rows) {
    console.log(`${r.check} | ${r.expected} | ${r.got} | ${r.pass ? 'PASS' : 'FAIL'}`)
  }
}

/* ------------------------------------------------------------- CSP config -- */

function readCspHeader(): string | null {
  const cfg = JSON.parse(readFileSync(CONFIG, 'utf8')) as Record<string, unknown>
  const app = asRecord(cfg.app)
  const security = asRecord(app.security)
  const csp = security.csp
  if (csp === null || csp === undefined) return null
  if (typeof csp === 'string') return csp
  if (typeof csp === 'object') {
    return Object.entries(csp as Record<string, unknown>)
      .map(([directive, sources]) => `${directive} ${Array.isArray(sources) ? sources.join(' ') : String(sources)}`)
      .join('; ')
  }
  return null
}

/* ------------------------------------------------------------- ports/fs ---- */

async function freePort(): Promise<number> {
  return new Promise((resolvePort, reject) => {
    const srv = createNetServer()
    srv.on('error', reject)
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address()
      const port = typeof addr === 'object' && addr ? addr.port : 0
      srv.close(() => resolvePort(port))
    })
  })
}

async function portFree(port: number): Promise<boolean> {
  return new Promise((resolveFree) => {
    const srv = createNetServer()
    srv.on('error', () => resolveFree(false))
    srv.listen(port, '127.0.0.1', () => srv.close(() => resolveFree(true)))
  })
}

const CONTENT_TYPES: Record<string, string> = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.ttf': 'font/ttf',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.wasm': 'application/wasm',
  '.map': 'application/json; charset=utf-8',
}

function startStaticServer(dir: string, port: number, cspHeader: string | null): Promise<Server> {
  const server = createServer((req: IncomingMessage, res: ServerResponse) => {
    if (cspHeader) res.setHeader('Content-Security-Policy', cspHeader)
    const rawPath = decodeURIComponent((req.url ?? '/').split('?')[0])
    const wanted = rawPath === '/' ? 'index.html' : rawPath.replace(/^\/+/, '')
    const candidate = resolve(dir, wanted)
    let servePath: string | null = null
    if (candidate.startsWith(resolve(dir)) && existsSync(candidate)) servePath = candidate
    else if (extname(wanted) === '') servePath = join(dir, 'index.html')
    if (!servePath || !existsSync(servePath)) {
      res.statusCode = 404
      res.end('not found')
      return
    }
    res.statusCode = 200
    res.setHeader('Content-Type', CONTENT_TYPES[extname(servePath).toLowerCase()] ?? 'application/octet-stream')
    res.end(readFileSync(servePath))
  })
  return new Promise((resolveServer) => server.listen(port, '127.0.0.1', () => resolveServer(server)))
}

function startFakeSidecar(
  port: number,
  pageOrigin: string,
  hits: string[],
): Promise<Server> {
  const server = createServer((req: IncomingMessage, res: ServerResponse) => {
    hits.push(`${port} ${req.method ?? 'GET'} ${req.url ?? '/'}`)
    res.setHeader('Access-Control-Allow-Origin', req.headers.origin ?? pageOrigin)
    res.setHeader('Access-Control-Allow-Headers', 'content-type, x-spinoml-token')
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    res.setHeader('Vary', 'Origin')
    if (req.method === 'OPTIONS') {
      res.statusCode = 204
      res.end()
      return
    }
    res.statusCode = 200
    res.setHeader('Content-Type', 'application/json')
    res.end(JSON.stringify(req.url?.startsWith('/health') ? { ok: true, auth: 'unauthenticated-dev' } : { ok: true }))
  })
  return new Promise((resolveServer) => server.listen(port, '127.0.0.1', () => resolveServer(server)))
}

/* ---------------------------------------------------------------- CDP ------ */

type CdpEvent = (method: string, params: Record<string, unknown>) => void

class Cdp {
  private nextId = 1
  private readonly pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>()
  onEvent: CdpEvent = () => {}

  constructor(private readonly ws: WebSocket) {
    ws.addEventListener('message', (ev: MessageEvent) => this.handle(ev))
  }

  private handle(ev: MessageEvent): void {
    if (typeof ev.data !== 'string') return
    let msg: Record<string, unknown>
    try {
      msg = JSON.parse(ev.data) as Record<string, unknown>
    } catch {
      return
    }
    const id = msg.id
    if (typeof id === 'number') {
      const p = this.pending.get(id)
      if (!p) return
      this.pending.delete(id)
      if (msg.error) p.reject(new Error(JSON.stringify(msg.error)))
      else p.resolve(msg.result)
      return
    }
    const method = msg.method
    if (typeof method === 'string') this.onEvent(method, asRecord(msg.params ?? {}))
  }

  send(method: string, params: Record<string, unknown> = {}): Promise<unknown> {
    const id = this.nextId++
    return new Promise((resolveSend, rejectSend) => {
      this.pending.set(id, { resolve: resolveSend, reject: rejectSend })
      this.ws.send(JSON.stringify({ id, method, params }))
    })
  }

  async evaluate(expression: string): Promise<unknown> {
    const res = asRecord(await this.send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true }))
    if (res.exceptionDetails) throw new Error(`evaluate threw: ${JSON.stringify(res.exceptionDetails)}`)
    return asRecord(res.result ?? {}).value
  }

  async evaluateBool(expression: string): Promise<boolean> {
    return (await this.evaluate(expression)) === true
  }

  close(): void {
    try {
      this.ws.close()
    } catch {
      /* already closed */
    }
  }
}

async function connectWs(url: string): Promise<WebSocket> {
  return new Promise((resolveWs, reject) => {
    const ws = new WebSocket(url)
    const timer = setTimeout(() => reject(new Error('CDP websocket timeout')), 10_000)
    ws.addEventListener('open', () => {
      clearTimeout(timer)
      resolveWs(ws)
    })
    ws.addEventListener('error', () => {
      clearTimeout(timer)
      reject(new Error('CDP websocket error'))
    })
  })
}

async function waitFor(fn: () => Promise<boolean>, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    try {
      if (await fn()) return true
    } catch {
      /* keep polling */
    }
    await sleep(POLL_MS)
  }
  return false
}

/* --------------------------------------------------------------- browser --- */

async function waitDebugPort(port: number, timeoutMs: number): Promise<boolean> {
  return waitFor(async () => {
    const res = await fetch(`http://127.0.0.1:${port}/json/version`)
    return res.ok
  }, timeoutMs)
}

type Launched = { proc: ChildProcess; stderr: string; usedNoSandbox: boolean }

function spawnBrowser(bin: string, args: string[]): Launched {
  const proc = spawn(bin, args, { detached: true, stdio: ['ignore', 'ignore', 'pipe'] })
  let stderr = ''
  proc.stderr?.on('data', (d: Buffer) => {
    stderr += d.toString()
  })
  return { proc, stderr, usedNoSandbox: args.includes('--no-sandbox') }
}

function killBrowser(proc: ChildProcess | null): void {
  if (!proc || proc.pid === undefined) return
  try {
    process.kill(-proc.pid, 'SIGKILL')
  } catch {
    try {
      proc.kill('SIGKILL')
    } catch {
      /* gone */
    }
  }
}

function buildArgs(debugPort: number, userDataDir: string, noSandbox: boolean): string[] {
  const args = [
    '--headless=new',
    '--disable-gpu',
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-background-networking',
    '--disable-component-update',
    '--disable-sync',
    '--disable-extensions',
    `--user-data-dir=${userDataDir}`,
    `--remote-debugging-port=${debugPort}`,
  ]
  if (noSandbox) args.push('--no-sandbox')
  args.push('about:blank')
  return args
}

/* ----------------------------------------------------------------- main ---- */

async function main(): Promise<number> {
  // (a) build the real dist/ that a packaged app would ship.
  console.log('[test-webview-csp] building dist/ …')
  const build = spawnSync(process.platform === 'win32' ? 'npm.cmd' : 'npm', ['run', 'build'], {
    cwd: ROOT,
    stdio: 'pipe',
    encoding: 'utf8',
  })
  if (build.status !== 0) {
    console.error(build.stdout ?? '')
    console.error(build.stderr ?? '')
    console.error('[test-webview-csp] `npm run build` failed')
    return 1
  }
  console.log('[test-webview-csp] build ok')

  const cspHeader = readCspHeader()
  record('config-csp', 'present in tauri.conf.json', cspHeader ? 'present' : 'missing', cspHeader !== null)
  if (!cspHeader) {
    console.log('  note: app.security.csp is null — served without a CSP header; negative probes must fail')
  }

  const bins = [process.env.SPINOML_CHROMIUM, '/snap/bin/chromium', 'chromium-browser', 'chromium', 'google-chrome', 'google-chrome-stable']
    .filter((b): b is string => typeof b === 'string' && b.length > 0)

  const staticPort = await freePort()
  const debugPort = await freePort()
  const pageOrigin = `http://127.0.0.1:${staticPort}`

  const servers: Server[] = []
  let browser: ChildProcess | null = null
  let cdp: Cdp | null = null
  let userDataDir = ''
  let launched: Launched | null = null

  // collected CDP observations
  const consoleMsgs: { type: string; text: string }[] = []
  const logEntries: { source: string; level: string; text: string }[] = []
  const exceptions: string[] = []
  const requests: { url: string; method: string }[] = []
  const workerTargets: { url: string; type: string }[] = []
  const sidecarHits: string[] = []

  try {
    servers.push(await startStaticServer(DIST, staticPort, cspHeader))

    // (c) optional fake sidecars — only when their ports are free. Busy ports
    //     skip the sidecar assertions with a note; they never fail the run.
    const fakePorts: number[] = []
    for (const port of [7421, 7422]) {
      if (await portFree(port)) {
        servers.push(await startFakeSidecar(port, pageOrigin, sidecarHits))
        fakePorts.push(port)
      } else {
        console.log(`  note: port ${port} busy — fake sidecar skipped`)
      }
    }

    // (d) launch a real browser and connect over CDP.
    userDataDir = await mkdtemp(join(tmpdir(), 'spinoml-csp-'))
    let started = false
    for (const bin of bins) {
      for (const noSandbox of [false, true]) {
        const attempt = spawnBrowser(bin, buildArgs(debugPort, userDataDir, noSandbox))
        const ok = await waitDebugPort(debugPort, noSandbox ? 8_000 : 8_000)
        if (ok) {
          launched = attempt
          browser = attempt.proc
          started = true
          if (noSandbox) console.log(`  note: ${bin} needed --no-sandbox`)
          console.log(`[test-webview-csp] browser: ${bin}${noSandbox ? ' (--no-sandbox)' : ''}`)
          break
        }
        killBrowser(attempt.proc)
        await sleep(500)
      }
      if (started) break
    }
    if (!started || !launched || !browser) {
      console.log(`SKIPPED: no Chromium could be started (tried: ${bins.join(', ')}).`)
      const why = launched?.stderr?.split('\n').slice(-8).filter(Boolean).join(' | ') ?? 'no stderr'
      console.log(`SKIPPED-reason: ${why}`)
      return 0
    }

    const listRes = await fetch(`http://127.0.0.1:${debugPort}/json/list`)
    const targets = (await listRes.json()) as { type?: string; webSocketDebuggerUrl?: string }[]
    const page = targets.find((t) => t.type === 'page' && typeof t.webSocketDebuggerUrl === 'string')
    if (!page?.webSocketDebuggerUrl) throw new Error('no page target')
    cdp = new Cdp(await connectWs(page.webSocketDebuggerUrl))

    cdp.onEvent = (method, params): void => {
      if (method === 'Runtime.consoleAPICalled') {
        const args = Array.isArray(params.args) ? params.args : []
        const text = args.map((a) => str(asRecord(a).value) || str(asRecord(a).description)).join(' ')
        consoleMsgs.push({ type: str(params.type), text })
      } else if (method === 'Runtime.exceptionThrown') {
        const details = asRecord(params.exceptionDetails)
        exceptions.push(str(details.text) || str(asRecord(details.exception).description))
      } else if (method === 'Log.entryAdded') {
        const entry = asRecord(params.entry)
        logEntries.push({ source: str(entry.source), level: str(entry.level), text: str(entry.text) })
      } else if (method === 'Network.requestWillBeSent') {
        const req = asRecord(params.request)
        requests.push({ url: str(req.url), method: str(req.method) })
      } else if (method === 'Target.attachedToTarget') {
        const info = asRecord(params.targetInfo)
        workerTargets.push({ url: str(info.url), type: str(info.type) })
      }
    }

    await cdp.send('Page.enable')
    await cdp.send('Runtime.enable')
    await cdp.send('Log.enable')
    await cdp.send('Network.enable')
    await cdp.send('Target.setAutoAttach', { autoAttach: true, waitForDebuggerOnStart: false, flatten: true })

    await cdp.send('Page.navigate', { url: `${pageOrigin}/` })

    // (e) — app renders.
    const rendered = await waitFor(
      () => cdp!.evaluateBool(`(document.getElementById('root')?.childElementCount ?? 0) > 0`),
      RENDER_TIMEOUT,
    )
    record('app-renders', '#root has element children', rendered ? 'yes' : 'no', rendered)

    // — Monaco mounts.
    const mounted = await waitFor(
      () => cdp!.evaluateBool(`document.querySelector('.monaco-editor') !== null`),
      EDITOR_TIMEOUT,
    )
    record('monaco-mounts', '.monaco-editor in DOM', mounted ? 'yes' : 'no', mounted)

    // — tokenized content.
    const content = asRecord(
      await cdp.evaluate(
        `(() => {
           const lines = document.querySelector('.monaco-editor .view-lines')
           const token = document.querySelector('.monaco-editor .view-lines [class*="mtk"]')
           return { text: (lines?.textContent ?? '').slice(0, 60), hasText: !!lines && (lines.textContent ?? '').trim().length > 0, hasToken: !!token, tokenClass: token?.className ?? '' }
         })()`,
      ),
    )
    const hasText = content.hasText === true
    const hasToken = content.hasToken === true
    record('editor-content', 'view-lines text + .mtk token', `${hasText ? 'text' : 'no-text'}/${hasToken ? str(content.tokenClass) : 'no-token'}`, hasText && hasToken)

    // — CSP violations / uncaught exceptions during load.
    const violationsDuringLoad = [...consoleMsgs, ...logEntries.map((l) => ({ type: l.level, text: l.text }))]
      .filter((m) => /Content Security Policy|Refused to/i.test(m.text))
    record('no-csp-violation', '0 during load', String(violationsDuringLoad.length), violationsDuringLoad.length === 0)
    if (violationsDuringLoad.length > 0) for (const v of violationsDuringLoad.slice(0, 4)) console.log(`      · ${v.text.slice(0, 200)}`)
    record('no-exception', '0 uncaught', String(exceptions.length), exceptions.length === 0)
    if (exceptions.length > 0) for (const e of exceptions.slice(0, 4)) console.log(`      · ${e.slice(0, 200)}`)

    // — network hosts + CDN.
    const parsed = requests
      .map((r) => ({ ...r, url: r.url }))
      .filter((r) => /^https?:/i.test(r.url))
    const allowedOrigins = new Set([
      pageOrigin,
      'http://127.0.0.1:7421',
      'http://127.0.0.1:7422',
      'http://127.0.0.1:7424',
      'http://localhost:7421',
      'http://localhost:7422',
      'http://localhost:7424',
    ])
    const offenders = parsed.filter((r) => {
      try {
        const u = new URL(r.url)
        if (u.pathname.startsWith('/.well-known/')) return false
        return !allowedOrigins.has(u.origin)
      } catch {
        return false
      }
    }).map((r) => r.url)
    record('request-hosts', 'self + loopback only', offenders.length ? `${offenders.length} offender(s)` : 'clean', offenders.length === 0)
    for (const o of offenders.slice(0, 6)) console.log(`      · offender: ${o}`)
    const cdn = requests.filter((r) => /cdn\.jsdelivr\.net|jsdelivr/i.test(r.url)).map((r) => r.url)
    record('no-cdn', '0 cdn.jsdelivr.net requests', String(cdn.length), cdn.length === 0)
    for (const c of cdn.slice(0, 4)) console.log(`      · cdn: ${c}`)

    // — sidecar requests allowed by connect-src (only meaningful with fakes).
    //   Specifically the TORCH sidecar on 7421: its /health proves connect-src
    //   let the app reach it (mutation: drop 7421 → this row goes red).
    if (fakePorts.length === 2) {
      const torchHit = sidecarHits.some((h) => h.startsWith('7421 ') && h.includes('/health'))
      record('sidecar-connect', 'torch 7421 /health reached fake', torchHit ? 'reached' : 'not reached', torchHit)
    } else {
      console.log('  note: sidecar-connect row skipped (ports busy)')
    }

    // — Monaco worker from 'self'/'blob:'. Monaco creates the editor worker
    //   lazily (the default link provider schedules its computation ~1s after
    //   the model is set), so poll for it rather than sampling once.
    let workerOk = false
    let workerDetail = 'none'
    let perfWorkers: string[] = []
    const workerDeadline = Date.now() + 10_000
    while (Date.now() < workerDeadline && !workerOk) {
      const fromTargets = workerTargets.filter((w) => /worker/i.test(w.type) && /(editor|json)\.worker/i.test(w.url))
      const fromReq = requests.filter((r) => /(editor|json)\.worker[^/]*\.js/i.test(r.url))
      try {
        const names = await cdp.evaluate(`performance.getEntriesByType('resource').map((e) => e.name).filter((n) => /worker/i.test(n))`)
        perfWorkers = Array.isArray(names) ? names.map((n) => String(n)) : []
      } catch {
        perfWorkers = []
      }
      const detail = fromTargets.length
        ? `${fromTargets[0].type}:${fromTargets[0].url.split('/').slice(-1)[0]}`
        : fromReq.length
          ? fromReq[0].url.split('/').slice(-1)[0]
          : perfWorkers.length
            ? perfWorkers[0].split('/').slice(-1)[0]
            : ''
      if (detail) {
        workerOk = true
        workerDetail = detail
        break
      }
      await sleep(POLL_MS)
    }
    record('monaco-worker', "worker chunk from 'self'", workerOk ? workerDetail : 'none', workerOk)
    if (!workerOk) {
      console.log(`      · requests: ${requests.map((r) => r.url).join(' , ').slice(0, 600)}`)
      console.log(`      · workerTargets: ${workerTargets.map((w) => `${w.type}:${w.url}`).join(' , ') || 'none'}`)
      console.log(`      · perfWorkers: ${perfWorkers.join(' , ') || 'none'}`)
    }

    // (f) negative probes — the CSP must be ACTIVE in this very run.
    const violationsBefore = violationsDuringLoad.length
    const inlineRan = (await cdp.evaluate(
      `(() => {
         const s = document.createElement('script')
         s.textContent = 'window.__spinomlCspProbe = true'
         document.body.appendChild(s)
         return window.__spinomlCspProbe === true
       })()`,
    )) === true
    record('probe-inline', 'inline <script> blocked', inlineRan ? 'executed' : 'blocked', inlineRan === false)

    const fetchResult = str(await cdp.evaluate(`fetch('https://example.com/').then(() => 'resolved').catch((e) => 'rejected:' + e.name)`))
    record('probe-fetch', 'example.com rejected by CSP', fetchResult || 'empty', fetchResult.startsWith('rejected'))

    const violationAfter = await waitFor(async () => {
      const all = [...consoleMsgs.map((m) => m.text), ...logEntries.map((l) => l.text)]
      const active = all.filter((t) => /Content Security Policy|Refused to/i.test(t))
      return active.length > violationsBefore
    }, 5_000)
    record('probe-csp-active', 'new CSP violation logged', violationAfter ? 'violation seen' : 'none', violationAfter)

    const exampleAttempts = requests.filter((r) => /example\.com/i.test(r.url))
    record('probe-offline', 'no network attempt for example.com', String(exampleAttempts.length), exampleAttempts.length === 0)

    if (consoleMsgs.length > 0 || logEntries.length > 0) {
      console.log('')
      console.log('[test-webview-csp] captured console/log (first 12):')
      for (const m of [...consoleMsgs.map((m) => `${m.type}: ${m.text}`), ...logEntries.map((l) => `${l.source}/${l.level}: ${l.text}`)].slice(0, 12)) {
        console.log(`  · ${m.slice(0, 220)}`)
      }
    }
    return 0
  } finally {
    cdp?.close()
    killBrowser(browser)
    for (const s of servers) {
      try {
        s.close()
      } catch {
        /* ignore */
      }
    }
    if (userDataDir) {
      try {
        rmSync(userDataDir, { recursive: true, force: true })
      } catch {
        /* ignore */
      }
    }
  }
}

main()
  .then((code) => {
    const failed = rows.filter((r) => !r.pass)
    printTable()
    const exitCode = failed.length > 0 ? 1 : code
    console.log('')
    console.log(`[test-webview-csp] ${failed.length} failed / ${rows.length} checks`)
    process.exit(exitCode)
  })
  .catch((e: unknown) => {
    printTable()
    console.error(`[test-webview-csp] harness error: ${e instanceof Error ? (e.stack ?? e.message) : String(e)}`)
    process.exit(1)
  })
