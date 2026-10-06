// Shared test helpers for sidecar HTTP auth tests (Phase 77/78).
//
// Deliberately sidecar-agnostic: the torch test uses it now, a later LLM-sidecar
// test reuses it unchanged. `rawRequest` speaks HTTP over a raw `net` socket so
// hostile inputs fetch() refuses to send — a missing `Host` (HTTP/1.0), an
// arbitrary `Host`/`Origin`, or a duplicated `X-SpinoML-Token` — can still be
// exercised. `startSidecar` owns a real process and kills its whole process
// group on stop, so no straggler python survives a failed run.

import { spawn, type ChildProcess } from 'node:child_process'
import { readFileSync, readdirSync } from 'node:fs'
import net from 'node:net'
import { setTimeout as wait } from 'node:timers/promises'

export type StartedSidecar = {
  url: string
  port: number
  proc: ChildProcess
  output: () => string
  stop: () => Promise<void>
}

export type StartOptions = {
  cmd: string
  args: string[]
  env?: Record<string, string>
  cwd?: string
  port?: number
  // Env var to set to the chosen port (e.g. 'SPINOML_TORCH_PORT'). Optional so
  // the library stays sidecar-agnostic.
  portEnv?: string
  readyPath?: string
  readyTimeoutMs?: number
  // Remove every inherited env var with this prefix before applying `env`, so a
  // developer shell cannot leak a real token into a "no token" test.
  stripEnvPrefix?: string
}

function buildEnv(env: Record<string, string> | undefined, stripPrefix: string | undefined): NodeJS.ProcessEnv {
  const base: NodeJS.ProcessEnv = { ...process.env }
  if (stripPrefix) {
    for (const key of Object.keys(base)) {
      if (key.startsWith(stripPrefix)) delete base[key]
    }
  }
  return { ...base, ...(env ?? {}) }
}

export async function freePort(): Promise<number> {
  return await new Promise<number>((resolve, reject) => {
    const srv = net.createServer()
    srv.on('error', reject)
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address()
      if (addr === null || typeof addr === 'string') {
        reject(new Error('could not determine a free port'))
        return
      }
      const port = addr.port
      srv.close(() => resolve(port))
    })
  })
}

function killGroup(proc: ChildProcess): void {
  if (proc.pid === undefined) return
  try {
    process.kill(-proc.pid, 'SIGKILL')
  } catch {
    try {
      proc.kill('SIGKILL')
    } catch {
      // already gone
    }
  }
}

async function waitExit(proc: ChildProcess, ms: number): Promise<void> {
  if (proc.exitCode !== null || proc.signalCode !== null) return
  await new Promise<void>((resolve) => {
    const t = setTimeout(resolve, ms)
    proc.once('exit', () => {
      clearTimeout(t)
      resolve()
    })
  })
}

export async function startSidecar(opts: StartOptions): Promise<StartedSidecar> {
  const port = opts.port ?? (await freePort())
  const chunks: string[] = []
  const env: Record<string, string> = { ...(opts.env ?? {}) }
  if (opts.portEnv) env[opts.portEnv] = String(port)
  const proc = spawn(opts.cmd, opts.args, {
    cwd: opts.cwd,
    env: buildEnv(env, opts.stripEnvPrefix),
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: true,
  })
  proc.stdout?.on('data', (d: Buffer) => chunks.push(d.toString()))
  proc.stderr?.on('data', (d: Buffer) => chunks.push(d.toString()))
  const output = (): string => chunks.join('')
  const url = `http://127.0.0.1:${port}`
  const readyPath = opts.readyPath ?? '/health'

  const stop = async (): Promise<void> => {
    killGroup(proc)
    await waitExit(proc, 3000)
  }

  const deadline = Date.now() + (opts.readyTimeoutMs ?? 60000)
  let up = false
  while (Date.now() < deadline) {
    if (proc.exitCode !== null) break
    try {
      const r = await fetch(`${url}${readyPath}`)
      if (r.ok) {
        up = true
        break
      }
    } catch {
      // not listening yet
    }
    await wait(150)
  }
  if (!up) {
    await stop()
    throw new Error(`sidecar did not become ready on ${url}\n--- output ---\n${output()}`)
  }
  return { url, port, proc, output, stop }
}

export type RawRequest = {
  port: number
  method: string
  path: string
  headers?: Record<string, string | string[]>
  body?: string
  version?: '1.0' | '1.1'
}

export type RawResponse = {
  status: number
  headers: Record<string, string[]>
  body: string
}

export function parseResponse(raw: string): RawResponse {
  const split = raw.indexOf('\r\n\r\n')
  const head = split === -1 ? raw : raw.slice(0, split)
  const body = split === -1 ? '' : raw.slice(split + 4)
  const lines = head.split('\r\n')
  const statusLine = lines.shift() ?? ''
  const m = /^HTTP\/\d\.\d\s+(\d{3})/.exec(statusLine)
  const status = m ? Number(m[1]) : 0
  const headers: Record<string, string[]> = {}
  for (const line of lines) {
    const colon = line.indexOf(':')
    if (colon === -1) continue
    const name = line.slice(0, colon).trim().toLowerCase()
    const value = line.slice(colon + 1).trim()
    const existing = headers[name] ?? []
    existing.push(value)
    headers[name] = existing
  }
  return { status, headers, body }
}

export function rawRequest(req: RawRequest): Promise<RawResponse> {
  return new Promise<RawResponse>((resolve, reject) => {
    const version = req.version ?? '1.1'
    const headers = { ...(req.headers ?? {}) }
    const hasHeader = (name: string): boolean =>
      Object.keys(headers).some((k) => k.toLowerCase() === name.toLowerCase())

    const lines = [`${req.method} ${req.path} HTTP/${version}`]
    for (const [name, value] of Object.entries(headers)) {
      if (Array.isArray(value)) {
        for (const v of value) lines.push(`${name}: ${v}`)
      } else {
        lines.push(`${name}: ${value}`)
      }
    }
    if (req.body !== undefined && !hasHeader('content-length')) {
      lines.push(`Content-Length: ${Buffer.byteLength(req.body)}`)
    }
    if (!hasHeader('connection')) lines.push('Connection: close')
    const payload = lines.join('\r\n') + '\r\n\r\n' + (req.body ?? '')

    const socket = net.connect({ host: '127.0.0.1', port: req.port })
    const chunks: Buffer[] = []
    let settled = false
    const fail = (err: Error): void => {
      if (settled) return
      settled = true
      socket.destroy()
      reject(err)
    }
    socket.setTimeout(20000, () => fail(new Error('rawRequest timed out')))
    socket.on('error', fail)
    socket.on('data', (d: Buffer) => chunks.push(d))
    socket.on('end', () => {
      if (settled) return
      settled = true
      resolve(parseResponse(Buffer.concat(chunks).toString('utf8')))
    })
    socket.on('connect', () => socket.write(payload))
  })
}

export type MatrixRow = {
  case: string
  expected: string
  got: string
  pass?: boolean
}

export function runMatrix(rows: MatrixRow[]): MatrixRow[] {
  const failures: MatrixRow[] = []
  const widths = rows.map((r) => r.case.length)
  const caseWidth = Math.max(4, ...(widths.length ? widths : [4]))
  console.log('')
  for (const row of rows) {
    const pass = row.pass ?? row.expected === row.got
    if (!pass) failures.push(row)
    const verdict = pass ? 'PASS' : 'FAIL'
    console.log(`  ${row.case.padEnd(caseWidth)} | ${row.expected.padEnd(10)} | ${row.got.padEnd(14)} | ${verdict}`)
  }
  return failures
}

export type ExpectExitOptions = {
  cmd: string
  args: string[]
  env?: Record<string, string>
  cwd?: string
  timeoutMs?: number
  stripEnvPrefix?: string
}

export type ExitResult = {
  code: number | null
  stdout: string
  stderr: string
  ok: boolean
}

// ── Additive helpers for process-lifecycle tests (Phase 13) ──────────────
// SIGTERM → graceful shutdown; SIGKILL bypasses every in-process handler
// (Node can't catch SIGKILL by design). For sidecars that spawn children
// (opencode, run_script), only SIGTERM triggers the child's kill path —
// using SIGKILL leaves orphans. These helpers prefer SIGTERM.

export function pidAlive(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0) return false
  try {
    process.kill(pid, 0)
    return true
  } catch (e) {
    return (e as NodeJS.ErrnoException).code !== 'ESRCH'
  }
}

export async function awaitProcessExit(proc: ChildProcess, ms: number): Promise<boolean> {
  if (proc.exitCode !== null || proc.signalCode !== null) return true
  return await new Promise<boolean>((resolve) => {
    const t = setTimeout(() => resolve(false), ms)
    proc.once('exit', () => {
      clearTimeout(t)
      resolve(true)
    })
  })
}

export async function gracefulStop(proc: ChildProcess, ms = 5000): Promise<number | null> {
  if (proc.exitCode !== null) return proc.exitCode
  try { proc.kill('SIGTERM') } catch { /* already gone */ }
  const exited = await awaitProcessExit(proc, ms)
  if (!exited && proc.exitCode === null && proc.signalCode === null) {
    if (typeof proc.pid === 'number') {
      try { process.kill(-proc.pid, 'SIGKILL') } catch { /* already gone */ }
    }
    await awaitProcessExit(proc, 1500)
  }
  return proc.exitCode
}

// Recursively walk /proc/<pid>/task/*/children — Linux-only. Returns the
// flat set of every descendant PID (excludes `pid` itself). Used to prove
// the SIGTERM handler killed every grandchild of a sidecar process, not
// just the immediate children.
export function descendantsOf(pid: number): Set<number> {
  const out = new Set<number>()
  if (!Number.isInteger(pid) || pid <= 0) return out
  const queue: number[] = [pid]
  while (queue.length > 0) {
    const p = queue.shift()
    if (p === undefined) continue
    if (out.has(p)) continue
    out.add(p)
    try {
      const tids = readdirSync(`/proc/${p}/task`)
      for (const tid of tids) {
        try {
          const text = readFileSync(`/proc/${p}/task/${tid}/children`, 'utf8')
          for (const m of text.matchAll(/(\d+)/g)) {
            const cpid = Number(m[1])
            if (!Number.isNaN(cpid)) queue.push(cpid)
          }
        } catch {
          // thread is gone — ignore
        }
      }
    } catch {
      // /proc/<p> is gone — ignore
    }
  }
  out.delete(pid)
  return out
}

export async function waitDescendantsGone(pid: number, ms: number): Promise<boolean> {
  const deadline = Date.now() + ms
  while (Date.now() < deadline) {
    if (descendantsOf(pid).size === 0) return true
    await new Promise((r) => setTimeout(r, 100))
  }
  return descendantsOf(pid).size === 0
}

export async function waitGone(pid: number, ms: number): Promise<boolean> {
  const deadline = Date.now() + ms
  while (Date.now() < deadline) {
    if (!pidAlive(pid)) return true
    await new Promise((r) => setTimeout(r, 100))
  }
  return !pidAlive(pid)
}

// Briefly try to bind the port; resolves true if the port is free.
// Uses SO_REUSEADDR so it can test a port that another listener JUST
// released (without TIME_WAIT races tripping the probe).
export async function isPortFree(port: number, host = '127.0.0.1'): Promise<boolean> {
  return await new Promise<boolean>((resolve) => {
    const srv = net.createServer()
    srv.on('error', () => resolve(false))
    srv.listen({ port, host }, () => {
      srv.close(() => resolve(true))
    })
  })
}

// Probes /health with a hard timeout — returns true on any complete
// response, false on connection error or hang. Used to verify a request
// arriving during shutdown never hangs > 3 s.
export async function probeHealth(url: string, ms: number): Promise<boolean> {
  const ctl = new AbortController()
  const t = setTimeout(() => ctl.abort(), ms)
  try {
    const r = await fetch(url, { signal: ctl.signal })
    clearTimeout(t)
    return r.status > 0
  } catch {
    clearTimeout(t)
    return false
  }
}

export async function expectExit(opts: ExpectExitOptions, expectedCode: number): Promise<ExitResult> {
  const proc = spawn(opts.cmd, opts.args, {
    cwd: opts.cwd,
    env: buildEnv(opts.env, opts.stripEnvPrefix),
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  let stdout = ''
  let stderr = ''
  proc.stdout?.on('data', (d: Buffer) => {
    stdout += d.toString()
  })
  proc.stderr?.on('data', (d: Buffer) => {
    stderr += d.toString()
  })
  const code = await new Promise<number | null>((resolve) => {
    const t = setTimeout(() => {
      try {
        proc.kill('SIGKILL')
      } catch {
        // already gone
      }
    }, opts.timeoutMs ?? 60000)
    proc.once('exit', (c) => {
      clearTimeout(t)
      resolve(c)
    })
  })
  return { code, stdout, stderr, ok: code === expectedCode }
}
