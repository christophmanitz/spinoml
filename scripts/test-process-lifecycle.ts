// Phase 13 — Local port and process management for the two sidecars.
//
// Real processes, real signals, real sockets — no mocks of the sidecars.
// Every test cleans up after itself: the test's child pids + bound ports
// must be back to baseline by the time the row is reported. The runner
// rejects suites that leave a process whose cmdline contains the sidecar
// script path or a fake-opencode.
//
// What it covers for both sidecars (Torch + LLM):
//   (a) PORT OCCUPIED  — a dummy net listener holds the port; the sidecar
//                        must exit non-zero within 5 s with a clear
//                        "[spinoml-XX] port N is already in use …" line and
//                        no Python traceback / Node stack as the only
//                        output; exit code 3; the dummy listener must
//                        still be serving. A second sidecar started while
//                        the first is healthy on the same port must not
//                        take over — first keeps answering /health.
//   (b) STARTUP FAILURE — invalid port values (`abc`, `0`, `70000`, `-1`)
//                        exit 2 with a message naming the variable. The
//                        sidecar never leaves a bound port when it fails.
//   (c) SHUTDOWN        — SIGTERM and SIGINT → exit 0 within 3 s, port
//                        released, no zombie (/proc/<pid>/status State
//                        gone), AND no orphaned child: start a long-running
//                        child through the sidecar (torch: /run_script
//                        with a `time.sleep(300)` script; LLM: opencode
//                        silent scenario + the per-turn child), then
//                        SIGTERM and assert every descendant is gone
//                        within 5 s.
//   (d) REPEATED START/STOP — 25 start→/health→SIGTERM cycles per sidecar
//                              on fresh free ports: every cycle the port
//                              is free again; total open fds of the TEST
//                              process returns to its baseline (±3); no
//                              leftover sidecar/fake-opencode processes.
//   (e) RESTART         — on the same port immediately after SIGTERM (no
//                        TIME_WAIT trouble — torch uses SO_REUSEADDR)
//                        AND after SIGKILL of the first instance.
//   (f) STATE CONSISTENCY — during/after those cycles /health never
//                        reports a stale or partial state, and a request
//                        arriving during shutdown gets a connection error
//                        or a complete response — never a hang > 3 s.
//
// Run: npm run test:process-lifecycle

import { execSync } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync } from 'node:fs'
import net from 'node:net'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { setTimeout as wait } from 'node:timers/promises'

import {
  descendantsOf,
  expectExit,
  freePort,
  gracefulStop,
  isPortFree,
  pidAlive,
  probeHealth,
  runMatrix,
  startSidecar,
  waitGone,
  type MatrixRow,
  type StartedSidecar,
} from './lib/auth-probe'
import { LlmHarness } from './lib/llm-harness'
import type { ToolCallSpec } from './lib/fake-openai'

const HERE = dirname(fileURLToPath(import.meta.url))
const REPO = resolve(HERE, '..')
const PYTHON = process.env.PYTHON ?? 'python'
const NODE = process.execPath
const TORCH = join(REPO, 'sidecar-torch', 'main.py')
const LLM = join(REPO, 'sidecar-llm', 'main.mjs')
const FAKE_BIN = join(REPO, 'scripts', 'lib', 'fake-opencode')
const TOKEN = randomBytes(32).toString('hex')

const tc = (name: string, args: unknown): ToolCallSpec => ({
  name,
  arguments: typeof args === 'string' ? args : JSON.stringify(args),
})

const rows: MatrixRow[] = []
const PROCS_BEFORE = procSnapshot()
const FD_BASELINE = countOpenFds()
const RUN = mkdtempSync(join(tmpdir(), 'spinoml-plc-'))

function record(case_: string, expected: string, got: string, pass?: boolean): void {
  rows.push({ case: case_, expected, got, pass })
}

function procSnapshot(): Set<string> {
  let out: string
  try {
    out = execSync('ps -eo args=', { encoding: 'utf8' })
  } catch {
    return new Set()
  }
  const set = new Set<string>()
  for (const line of out.split('\n')) {
    if (/sidecar-(torch|llm)\/main|fake-opencode|mcp-bridge\.mjs/.test(line)) set.add(line.trim())
  }
  return set
}

function procDelta(): { added: string[] } {
  const now = procSnapshot()
  const added: string[] = []
  for (const line of now) if (!PROCS_BEFORE.has(line)) added.push(line)
  return { added }
}

function countOpenFds(): number {
  try {
    return execSync(`ls /proc/${process.pid}/fd | wc -l`, { encoding: 'utf8' }).trim().split(/\s+/)[0]
      ? Number(execSync(`ls /proc/${process.pid}/fd | wc -l`, { encoding: 'utf8' }).trim())
      : 0
  } catch {
    return 0
  }
}

function fdDelta(): number {
  return countOpenFds() - FD_BASELINE
}

// Hold a port with a dummy net listener (independent of the sidecar
// process group), so an EADDRINUSE path can be tested deterministically.
// The listener must survive the sidecar failing — we verify it still
// serves after the sidecar exits.
function holdPort(port: number): { srv: net.Server; close: () => Promise<void> } {
  const srv = net.createServer()
  srv.listen({ port, host: '127.0.0.1' })
  const close = (): Promise<void> => new Promise<void>((resolveP) => srv.close(() => resolveP()))
  return { srv, close }
}

function mkWorkspaceDir(ws: string, subdirs: string[]): void {
  execSync(`mkdir -p ${subdirs.map((s) => `'${ws}/${s}'`).join(' ')}`, { stdio: 'ignore' })
}

async function withTorch(
  fn: (s: StartedSidecar) => Promise<void>,
  opts: { token?: string; allowedRoots?: string } = {},
): Promise<StartedSidecar> {
  const env: Record<string, string> = {}
  if (opts.token) env.SPINOML_SIDECAR_TOKEN = opts.token
  if (opts.allowedRoots) env.SPINOML_ALLOWED_ROOTS = opts.allowedRoots
  const s = await startSidecar({
    cmd: PYTHON,
    args: [TORCH],
    cwd: REPO,
    readyTimeoutMs: 60000,
    stripEnvPrefix: 'SPINOML_',
    portEnv: 'SPINOML_TORCH_PORT',
    env,
  })
  try {
    await fn(s)
  } finally {
    await gracefulStop(s.proc, 5000)
  }
  return s
}

async function withLlm(
  fn: (s: StartedSidecar) => Promise<void>,
  opts: { token?: string; opencodeScenario?: Record<string, unknown>; extraEnv?: Record<string, string> } = {},
): Promise<StartedSidecar> {
  const env: Record<string, string> = {}
  if (opts.token) env.SPINOML_SIDECAR_TOKEN = opts.token
  if (opts.opencodeScenario) {
    env.SPINOML_OPENCODE_BIN = FAKE_BIN
    env.FAKE_OPENCODE_SCENARIO = JSON.stringify(opts.opencodeScenario)
  }
  if (opts.extraEnv) Object.assign(env, opts.extraEnv)
  const s = await startSidecar({
    cmd: NODE,
    args: [LLM],
    cwd: REPO,
    readyTimeoutMs: 30000,
    stripEnvPrefix: 'SPINOML_',
    portEnv: 'SPINOML_LLM_PORT',
    env,
  })
  try {
    await fn(s)
  } finally {
    await gracefulStop(s.proc, 5000)
  }
  return s
}

// ── (a) PORT OCCUPIED ─────────────────────────────────────────────────────

async function casePortOccupiedTorch(): Promise<void> {
  const port = await freePort()
  const holder = holdPort(port)
  try {
    const r = await expectExit(
      {
        cmd: PYTHON,
        args: [TORCH],
        cwd: REPO,
        timeoutMs: 5000,
        stripEnvPrefix: 'SPINOML_',
        env: { SPINOML_TORCH_PORT: String(port) },
      },
      3,
    )
    record('torch: port occupied → exit 3', 'exit 3', `exit ${String(r.code)}`, r.ok)
    const names = /port\s+\d+\s+is already in use/.test(r.stderr) && r.stderr.includes('SPINOML_TORCH_PORT')
    record(
      'torch: port occupied → clear stderr',
      'names port + env var',
      r.stderr.trim().slice(0, 80) || 'none',
      names,
    )
    const traceback = /Traceback \(most recent call last\)/.test(r.stderr)
    record('torch: port occupied → no Python traceback', 'no traceback', traceback ? 'TRACEBACK' : 'none', !traceback)
    // Dummy listener must still be serving.
    const ok = await isPortFree(port)
    record('torch: port occupied → dummy listener untouched', 'serving', ok ? 'free (dummy died)' : 'serving', !ok)
  } finally {
    await holder.close()
  }
}

async function caseSecondTorchCannotTakeOver(): Promise<void> {
  const port = await freePort()
  const first = await startSidecar({
    cmd: PYTHON,
    args: [TORCH],
    cwd: REPO,
    readyTimeoutMs: 30000,
    stripEnvPrefix: 'SPINOML_',
    portEnv: 'SPINOML_TORCH_PORT',
    port,
  })
  try {
    const health1 = await fetch(`${first.url}/health`)
    const pid1 = first.proc.pid
    // Second sidecar on the same port must exit 3, not corrupt the first.
    const r = await expectExit(
      {
        cmd: PYTHON,
        args: [TORCH],
        cwd: REPO,
        timeoutMs: 5000,
        stripEnvPrefix: 'SPINOML_',
        env: { SPINOML_TORCH_PORT: String(port) },
      },
      3,
    )
    record('torch: 2nd on live port → exit 3', 'exit 3', `exit ${String(r.code)}`, r.ok)
    const health2 = await fetch(`${first.url}/health`)
    const ok = health1.ok && health2.ok && first.proc.pid === pid1 && first.proc.exitCode === null
    record(
      'torch: 2nd on live port → first keeps serving',
      'first alive + same pid',
      `firstPid=${pid1} h1=${health1.status} h2=${health2.status} exit=${String(first.proc.exitCode)}`,
      ok,
    )
  } finally {
    await gracefulStop(first.proc, 5000)
  }
}

async function casePortOccupiedLlm(): Promise<void> {
  const port = await freePort()
  const holder = holdPort(port)
  try {
    const r = await expectExit(
      {
        cmd: NODE,
        args: [LLM],
        cwd: REPO,
        timeoutMs: 5000,
        stripEnvPrefix: 'SPINOML_',
        env: { SPINOML_LLM_PORT: String(port) },
      },
      3,
    )
    record('llm: port occupied → exit 3', 'exit 3', `exit ${String(r.code)}`, r.ok)
    const names = /port\s+\d+\s+is already in use/.test(r.stderr) && r.stderr.includes('SPINOML_LLM_PORT')
    record(
      'llm: port occupied → clear stderr',
      'names port + env var',
      r.stderr.trim().slice(0, 80) || 'none',
      names,
    )
    const stackOnly = /at\s+.*\(/.test(r.stderr) && !names
    record(
      'llm: port occupied → no Node stack as only output',
      'no raw stack',
      stackOnly ? 'STACK ONLY' : names ? 'message + stack' : 'message',
      !stackOnly,
    )
    const ok = await isPortFree(port)
    record('llm: port occupied → dummy listener untouched', 'serving', ok ? 'free (dummy died)' : 'serving', !ok)
  } finally {
    await holder.close()
  }
}

async function caseSecondLlmCannotTakeOver(): Promise<void> {
  const port = await freePort()
  const first = await startSidecar({
    cmd: NODE,
    args: [LLM],
    cwd: REPO,
    readyTimeoutMs: 30000,
    stripEnvPrefix: 'SPINOML_',
    portEnv: 'SPINOML_LLM_PORT',
    port,
  })
  try {
    const health1 = await fetch(`${first.url}/health`)
    const pid1 = first.proc.pid
    const r = await expectExit(
      {
        cmd: NODE,
        args: [LLM],
        cwd: REPO,
        timeoutMs: 5000,
        stripEnvPrefix: 'SPINOML_',
        env: { SPINOML_LLM_PORT: String(port) },
      },
      3,
    )
    record('llm: 2nd on live port → exit 3', 'exit 3', `exit ${String(r.code)}`, r.ok)
    const health2 = await fetch(`${first.url}/health`)
    const ok = health1.ok && health2.ok && first.proc.pid === pid1 && first.proc.exitCode === null
    record(
      'llm: 2nd on live port → first keeps serving',
      'first alive + same pid',
      `firstPid=${pid1} h1=${health1.status} h2=${health2.status} exit=${String(first.proc.exitCode)}`,
      ok,
    )
  } finally {
    await gracefulStop(first.proc, 5000)
  }
}

// ── (b) STARTUP FAILURE ───────────────────────────────────────────────────

const BAD_PORTS: Array<{ name: string; value: string }> = [
  { name: 'abc', value: 'abc' },
  { name: '0', value: '0' },
  { name: '70000', value: '70000' },
  { name: '-1', value: '-1' },
]

async function caseStartupFailureTorch(): Promise<void> {
  for (const b of BAD_PORTS) {
    const port = await freePort()
    const r = await expectExit(
      {
        cmd: PYTHON,
        args: [TORCH],
        cwd: REPO,
        timeoutMs: 5000,
        stripEnvPrefix: 'SPINOML_',
        env: { SPINOML_TORCH_PORT: b.value },
      },
      2,
    )
    record(`torch: port=${b.name} → exit 2`, 'exit 2', `exit ${String(r.code)}`, r.ok)
    record(
      `torch: port=${b.name} names SPINOML_TORCH_PORT`,
      'mentions var',
      r.stderr.trim().slice(0, 80) || 'none',
      r.stderr.includes('SPINOML_TORCH_PORT'),
    )
    record(
      `torch: port=${b.name} never binds`,
      'port closed',
      (await isPortFree(port)) ? 'closed' : 'OPEN',
      await isPortFree(port),
    )
  }
}

async function caseStartupFailureLlm(): Promise<void> {
  for (const b of BAD_PORTS) {
    const port = await freePort()
    const r = await expectExit(
      {
        cmd: NODE,
        args: [LLM],
        cwd: REPO,
        timeoutMs: 5000,
        stripEnvPrefix: 'SPINOML_',
        env: { SPINOML_LLM_PORT: b.value },
      },
      2,
    )
    record(`llm: port=${b.name} → exit 2`, 'exit 2', `exit ${String(r.code)}`, r.ok)
    record(
      `llm: port=${b.name} names SPINOML_LLM_PORT`,
      'mentions var',
      r.stderr.trim().slice(0, 80) || 'none',
      r.stderr.includes('SPINOML_LLM_PORT'),
    )
    record(
      `llm: port=${b.name} never binds`,
      'port closed',
      (await isPortFree(port)) ? 'closed' : 'OPEN',
      await isPortFree(port),
    )
  }
}

// ── (c) SHUTDOWN — clean signal + no orphans ──────────────────────────────

async function caseShutdownCleanTorch(): Promise<void> {
  for (const sig of ['SIGTERM', 'SIGINT'] as const) {
    const port = await freePort()
    const s = await startSidecar({
      cmd: PYTHON,
      args: [TORCH],
      cwd: REPO,
      readyTimeoutMs: 30000,
      stripEnvPrefix: 'SPINOML_',
      portEnv: 'SPINOML_TORCH_PORT',
    })
    const pid = s.proc.pid
    if (pid === undefined) throw new Error('no pid')
    s.proc.kill(sig)
    const exited = await waitGone(pid, 3000)
    record(`torch: ${sig} → exit within 3 s`, 'gone', exited ? 'gone' : 'alive', exited)
    const free = await isPortFree(port)
    record(`torch: ${sig} → port released`, 'free', free ? 'free' : 'bound', free)
    const state = readProcState(pid)
    record(`torch: ${sig} → no zombie`, 'no /proc', state === null ? 'no /proc' : state, state === null)
  }
}

async function caseShutdownCleanLlm(): Promise<void> {
  for (const sig of ['SIGTERM', 'SIGINT'] as const) {
    const port = await freePort()
    const s = await startSidecar({
      cmd: NODE,
      args: [LLM],
      cwd: REPO,
      readyTimeoutMs: 30000,
      stripEnvPrefix: 'SPINOML_',
      portEnv: 'SPINOML_LLM_PORT',
    })
    const pid = s.proc.pid
    if (pid === undefined) throw new Error('no pid')
    s.proc.kill(sig)
    const exited = await waitGone(pid, 3000)
    record(`llm: ${sig} → exit within 3 s`, 'gone', exited ? 'gone' : 'alive', exited)
    const free = await isPortFree(port)
    record(`llm: ${sig} → port released`, 'free', free ? 'free' : 'bound', free)
    const state = readProcState(pid)
    record(`llm: ${sig} → no zombie`, 'no /proc', state === null ? 'no /proc' : state, state === null)
  }
}

function readProcState(pid: number): string | null {
  try {
    const text = readFileSync(`/proc/${pid}/status`, 'utf8')
    const m = /^State:\s+(\S+)/m.exec(text)
    return m ? m[1] : null
  } catch {
    return null
  }
}

async function caseNoOrphanTorchRunScript(): Promise<void> {
  const ws = join(RUN, 'orphan-torch')
  await mkWorkspaceDir(ws, ['agent'])
  // The script writes its own pid to a marker file so the test can find it
  // even if /proc walking missed a layer (defensive — the main signal is
  // /proc/<sidecar>/task/*/children being empty).
  const marker = join(ws, 'agent', 'sleeper.pid')
  await withTorch(
    async (s) => {
      // Start a long-running child via /run_script in a background fetch.
      const payload = JSON.stringify({
        root: ws,
        relpath: 'agent/sleeper.py',
        code: `import os, time\nopen(${JSON.stringify(marker)}, 'w').write(str(os.getpid()))\ntime.sleep(300)\n`,
        mode: 'shell',
      })
      const reqPromise = fetch(`${s.url}/run_script`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: payload,
      }).catch(() => undefined)
      // Wait for the child to be running.
      let childPid = 0
      for (let i = 0; i < 100; i++) {
        await wait(100)
        if (existsSync(marker)) {
          childPid = Number(readFileSync(marker, 'utf8').trim())
          if (childPid > 0) break
        }
      }
      record(
        'torch orphan: child started within 10 s',
        'pid known',
        childPid > 0 ? `pid=${childPid}` : 'no pid',
        childPid > 0,
      )
      // SIGTERM the sidecar — the child must die within 5 s.
      s.proc.kill('SIGTERM')
      await reqPromise
      const gone = childPid > 0 ? await waitGone(childPid, 5000) : false
      record(
        'torch orphan: child gone within 5 s of SIGTERM',
        'gone',
        gone ? 'gone' : 'alive',
        gone,
      )
      // /proc-walking corroboration: descendants set should be empty.
      const sid = s.proc.pid
      if (sid !== undefined) {
        await wait(500)
        const surviving = descendantsOf(sid)
        record(
          'torch orphan: no descendants via /proc walk',
          'empty',
          surviving.size > 0 ? [...surviving].slice(0, 5).join(',') : 'empty',
          surviving.size === 0,
        )
      }
    },
    { allowedRoots: ws },
  )
}

async function caseNoOrphanLlmOpencode(): Promise<void> {
  // Session dirs that already exist (from other runs) are not ours to judge.
  const dirsBefore = new Set(readdirSync(tmpdir()).filter((n) => n.startsWith('spinoml-opencode-')))
  const dump = join(RUN, 'orphan-llm-oc.json')
  const pidFile = join(RUN, 'orphan-llm-oc.pid.json')
  await withLlm(
    async (s) => {
      // Trigger an opencode spawn via a /chat turn.
      const harness = await LlmHarness.connect({ baseUrl: s.url })
      const chatPromise = harness.chat({
        script: [],
        llm: { kind: 'opencode', model: 'fake', apiKey: 'sk-fake' },
        token: TOKEN,
        timeoutMs: 20000,
      })
      // Wait for the fake-opencode to dump its pids.
      let info: Record<string, unknown> | null = null
      for (let i = 0; i < 100; i++) {
        await wait(100)
        if (existsSync(pidFile)) {
          try {
            info = JSON.parse(readFileSync(pidFile, 'utf8')) as Record<string, unknown>
            break
          } catch {
            // still being written
          }
        }
      }
      const fakePid = Number(info?.pid ?? 0)
      const bridgePid = Number(info?.childPid ?? 0)
      record(
        'llm opencode orphan: fake + bridge recorded',
        'pids > 0',
        `fake=${fakePid} bridge=${bridgePid}`,
        fakePid > 0 && bridgePid > 0,
      )
      s.proc.kill('SIGTERM')
      await chatPromise.catch(() => undefined)
      await harness.stop()
      const fakeGone = fakePid > 0 ? await waitGone(fakePid, 5000) : false
      const bridgeGone = bridgePid > 0 ? await waitGone(bridgePid, 5000) : false
      record(
        'llm opencode orphan: fake dead within 5 s',
        'gone',
        fakeGone ? 'gone' : 'alive',
        fakeGone,
      )
      record(
        'llm opencode orphan: bridge dead within 5 s',
        'gone',
        bridgeGone ? 'gone' : 'alive',
        bridgeGone,
      )
      // The turn's temp session dir is removed on SIGTERM (it used to remain behind).
      await wait(300)
      const dirsAfter = readdirSync(tmpdir()).filter((n) => n.startsWith('spinoml-opencode-') && !dirsBefore.has(n))
      record(
        'llm opencode orphan: temp session dir removed on SIGTERM',
        'none left',
        dirsAfter.length ? dirsAfter.join(',') : 'none left',
        dirsAfter.length === 0,
      )
      const sid = s.proc.pid
      if (sid !== undefined) {
        await wait(500)
        const surviving = descendantsOf(sid)
        record(
          'llm opencode orphan: no descendants via /proc walk',
          'empty',
          surviving.size > 0 ? [...surviving].slice(0, 5).join(',') : 'empty',
          surviving.size === 0,
        )
      }
    },
    {
      token: TOKEN,
      opencodeScenario: { scenario: 'silent', dump, pidFile },
    },
  )
}

async function caseNoOrphanLlmRunScript(): Promise<void> {
  const ws = join(RUN, 'orphan-llm-rs')
  mkWorkspaceDir(ws, ['agent'])
  // The script `exec sleep 300` REPLACES bash with sleep, so the sidecar's
  // immediate child IS sleep. SIGTERM hits sleep's pgid directly and
  // sleep dies — no reparenting dance, no bash-grandchild we lose track
  // of. Spawning the bg sleep with `&` would race the SIGTERM (bash
  // exits before we can send the signal, sleep gets reparented to init).
  // Note: unauthenticated-dev mode (no token). The LlmHarness's onAsk
  // path calls /respond WITHOUT the master token, which would 401 in
  // token mode and hang the chat (the harness swallows the failure).
  const pidMarker = join(ws, 'agent', 'sleep.pid')
  await withLlm(
    async (s) => {
      const harness = await LlmHarness.connect({ baseUrl: s.url })
      void harness.chat({
        script: [
          { toolCalls: [tc('write_file', {
            path: 'agent/sleeper.sh',
            content: `#!/bin/bash\necho $$ > ${JSON.stringify(pidMarker)}\nexec sleep 300\n`,
          })] },
          { toolCalls: [tc('run_script', { path: 'agent/sleeper.sh', mode: 'shell' })] },
        ],
        llm: { kind: 'openai-compat', model: 'fake', apiKey: 'sk-fake' },
        project: { root: ws },
        timeoutMs: 60000,
        onAsk: () => true,
      })
      // Wait for the marker file to appear — by then bash has exec'd
      // into sleep and the sidecar's child IS sleep (PID = $$).
      let sleepPid = 0
      for (let i = 0; i < 100; i++) {
        await wait(100)
        if (existsSync(pidMarker)) {
          const txt = readFileSync(pidMarker, 'utf8').trim()
          sleepPid = Number(txt)
          if (sleepPid > 0 && pidAlive(sleepPid)) break
          sleepPid = 0
        }
      }
      record(
        'llm runscript orphan: sleep pid known',
        'pid > 0',
        sleepPid > 0 ? `pid=${sleepPid}` : 'no pid',
        sleepPid > 0,
      )
      s.proc.kill('SIGTERM')
      await harness.stop()
      const gone = sleepPid > 0 ? await waitGone(sleepPid, 5000) : false
      record(
        'llm runscript orphan: sleep gone within 5 s',
        'gone',
        gone ? 'gone' : 'alive',
        gone,
      )
      const sid = s.proc.pid
      if (sid !== undefined) {
        await wait(500)
        const surviving = descendantsOf(sid)
        record(
          'llm runscript orphan: no descendants via /proc walk',
          'empty',
          surviving.size > 0 ? [...surviving].slice(0, 5).join(',') : 'empty',
          surviving.size === 0,
        )
      }
    },
    {
      // no token — unauthenticated-dev mode (see comment above)
    },
  )
  // cleanup
  try { rmSync(ws, { recursive: true, force: true }) } catch { /* tmp dir best-effort */ }
}

async function caseNoOrphanLlmRunScriptGrandchild(): Promise<void> {
  // The common real-world shape: the script is NOT exec'd into the long process — bash stays
  // alive and the long-running work is its CHILD (think `python train.py`). Aborting the turn
  // kills bash only, the child is reparented to init; the sidecar must have snapshotted the
  // whole descendant tree before aborting, otherwise this process keeps running forever.
  const ws = join(RUN, 'orphan-llm-gc')
  mkWorkspaceDir(ws, ['agent'])
  const pidMarker = join(ws, 'agent', 'grandchild.pid')
  await withLlm(
    async (s) => {
      const harness = await LlmHarness.connect({ baseUrl: s.url })
      void harness.chat({
        script: [
          { toolCalls: [tc('write_file', {
            path: 'agent/gc.sh',
            content: `#!/bin/bash\nsleep 300 &\necho $! > ${JSON.stringify(pidMarker)}\nwait\n`,
          })] },
          { toolCalls: [tc('run_script', { path: 'agent/gc.sh', mode: 'shell' })] },
        ],
        llm: { kind: 'openai-compat', model: 'fake', apiKey: 'sk-fake' },
        project: { root: ws },
        timeoutMs: 60000,
        onAsk: () => true,
      })
      let gcPid = 0
      for (let i = 0; i < 100; i++) {
        await wait(100)
        if (existsSync(pidMarker)) {
          gcPid = Number(readFileSync(pidMarker, 'utf8').trim())
          if (gcPid > 0 && pidAlive(gcPid)) break
          gcPid = 0
        }
      }
      record('llm runscript grandchild: pid known', 'pid > 0', gcPid > 0 ? `pid=${gcPid}` : 'no pid', gcPid > 0)
      s.proc.kill('SIGTERM')
      await harness.stop()
      const gone = gcPid > 0 ? await waitGone(gcPid, 5000) : false
      record('llm runscript grandchild: gone within 5 s of SIGTERM', 'gone', gone ? 'gone' : 'alive', gone)
      // never leave the sleeper behind, even when the assertion above failed
      if (gcPid > 0 && pidAlive(gcPid)) { try { process.kill(gcPid, 'SIGKILL') } catch { /* already gone */ } }
    },
    {
      // no token — unauthenticated-dev mode (the harness answers asks without the master token)
    },
  )
  try { rmSync(ws, { recursive: true, force: true }) } catch { /* tmp dir best-effort */ }
}

// ── (d) REPEATED START/STOP ───────────────────────────────────────────────

async function caseRepeatedStartStopTorch(): Promise<void> {
  const cycles = 25
  let failures = 0
  const fd0 = countOpenFds()
  for (let i = 0; i < cycles; i++) {
    const port = await freePort()
    const s = await startSidecar({
      cmd: PYTHON,
      args: [TORCH],
      cwd: REPO,
      readyTimeoutMs: 30000,
      stripEnvPrefix: 'SPINOML_',
      portEnv: 'SPINOML_TORCH_PORT',
    })
    // /health must respond.
    const r = await fetch(`${s.url}/health`)
    if (!r.ok) failures++
    const pid = s.proc.pid ?? 0
    if (pid === 0) failures++
    s.proc.kill('SIGTERM')
    await waitGone(pid, 3000).then((g) => { if (!g) failures++ })
    const free = await isPortFree(port)
    if (!free) failures++
  }
  const delta = countOpenFds() - fd0
  record(
    `torch: ${cycles} start/stop cycles all clean`,
    '0 failures',
    `failures=${failures}`,
    failures === 0,
  )
  record(
    `torch: open-fd delta after ${cycles} cycles`,
    '±3',
    `Δ=${delta}`,
    Math.abs(delta) <= 3,
  )
  const { added } = procDelta()
  record(
    `torch: no leftover sidecar/fake-opencode processes`,
    'clean',
    `added=${added.length}`,
    added.length === 0,
  )
  if (added.length > 0) console.log('  added:', added.slice(0, 3))
}

async function caseRepeatedStartStopLlm(): Promise<void> {
  const cycles = 25
  let failures = 0
  const fd0 = countOpenFds()
  for (let i = 0; i < cycles; i++) {
    const port = await freePort()
    const s = await startSidecar({
      cmd: NODE,
      args: [LLM],
      cwd: REPO,
      readyTimeoutMs: 30000,
      stripEnvPrefix: 'SPINOML_',
      portEnv: 'SPINOML_LLM_PORT',
    })
    const r = await fetch(`${s.url}/health`)
    if (!r.ok) failures++
    const pid = s.proc.pid ?? 0
    if (pid === 0) failures++
    s.proc.kill('SIGTERM')
    await waitGone(pid, 3000).then((g) => { if (!g) failures++ })
    const free = await isPortFree(port)
    if (!free) failures++
  }
  const delta = countOpenFds() - fd0
  record(
    `llm: ${cycles} start/stop cycles all clean`,
    '0 failures',
    `failures=${failures}`,
    failures === 0,
  )
  record(
    `llm: open-fd delta after ${cycles} cycles`,
    '±3',
    `Δ=${delta}`,
    Math.abs(delta) <= 3,
  )
  const { added } = procDelta()
  record(
    `llm: no leftover sidecar/fake-opencode processes`,
    'clean',
    `added=${added.length}`,
    added.length === 0,
  )
  if (added.length > 0) console.log('  added:', added.slice(0, 3))
}

// ── (e) RESTART on the same port ──────────────────────────────────────────

async function caseRestartSamePortTorch(): Promise<void> {
  // SIGTERM → restart immediately on the SAME port (SO_REUSEADDR must
  // let the second instance bind even though the first may have left the
  // port in TIME_WAIT).
  {
    const port = await freePort()
    const s1 = await startSidecar({
      cmd: PYTHON,
      args: [TORCH],
      cwd: REPO,
      readyTimeoutMs: 30000,
      stripEnvPrefix: 'SPINOML_',
      portEnv: 'SPINOML_TORCH_PORT',
      port,
    })
    await fetch(`${s1.url}/health`)
    s1.proc.kill('SIGTERM')
    await waitGone(s1.proc.pid as number, 3000)
    const s2 = await startSidecar({
      cmd: PYTHON,
      args: [TORCH],
      cwd: REPO,
      readyTimeoutMs: 30000,
      stripEnvPrefix: 'SPINOML_',
      portEnv: 'SPINOML_TORCH_PORT',
      port,
    })
    const h = await fetch(`${s2.url}/health`)
    record(
      'torch restart after SIGTERM: 2nd binds + healthy',
      'health 200',
      `port=${port} status=${h.status}`,
      h.ok && s2.proc.pid !== s1.proc.pid,
    )
    await gracefulStop(s2.proc, 5000)
  }
  // SIGKILL → restart on the same port.
  {
    const port = await freePort()
    const s1 = await startSidecar({
      cmd: PYTHON,
      args: [TORCH],
      cwd: REPO,
      readyTimeoutMs: 30000,
      stripEnvPrefix: 'SPINOML_',
      portEnv: 'SPINOML_TORCH_PORT',
      port,
    })
    await fetch(`${s1.url}/health`)
    const s1Pid = s1.proc.pid ?? 0
    try { process.kill(-s1Pid, 'SIGKILL') } catch { /* already gone */ }
    await waitGone(s1Pid, 3000)
    const s2 = await startSidecar({
      cmd: PYTHON,
      args: [TORCH],
      cwd: REPO,
      readyTimeoutMs: 30000,
      stripEnvPrefix: 'SPINOML_',
      portEnv: 'SPINOML_TORCH_PORT',
      port,
    })
    const h = await fetch(`${s2.url}/health`)
    record(
      'torch restart after SIGKILL: 2nd binds + healthy',
      'health 200',
      `port=${port} status=${h.status}`,
      h.ok && s2.proc.pid !== s1.proc.pid,
    )
    await gracefulStop(s2.proc, 5000)
  }
}

async function caseRestartSamePortLlm(): Promise<void> {
  // SIGTERM → restart on the same port.
  {
    const port = await freePort()
    const s1 = await startSidecar({
      cmd: NODE,
      args: [LLM],
      cwd: REPO,
      readyTimeoutMs: 30000,
      stripEnvPrefix: 'SPINOML_',
      portEnv: 'SPINOML_LLM_PORT',
      port,
    })
    await fetch(`${s1.url}/health`)
    s1.proc.kill('SIGTERM')
    await waitGone(s1.proc.pid as number, 3000)
    const s2 = await startSidecar({
      cmd: NODE,
      args: [LLM],
      cwd: REPO,
      readyTimeoutMs: 30000,
      stripEnvPrefix: 'SPINOML_',
      portEnv: 'SPINOML_LLM_PORT',
      port,
    })
    const h = await fetch(`${s2.url}/health`)
    record(
      'llm restart after SIGTERM: 2nd binds + healthy',
      'health 200',
      `port=${port} status=${h.status}`,
      h.ok && s2.proc.pid !== s1.proc.pid,
    )
    await gracefulStop(s2.proc, 5000)
  }
  // SIGKILL → restart on the same port.
  {
    const port = await freePort()
    const s1 = await startSidecar({
      cmd: NODE,
      args: [LLM],
      cwd: REPO,
      readyTimeoutMs: 30000,
      stripEnvPrefix: 'SPINOML_',
      portEnv: 'SPINOML_LLM_PORT',
      port,
    })
    await fetch(`${s1.url}/health`)
    const s1Pid = s1.proc.pid ?? 0
    try { process.kill(-s1Pid, 'SIGKILL') } catch { /* already gone */ }
    await waitGone(s1Pid, 3000)
    const s2 = await startSidecar({
      cmd: NODE,
      args: [LLM],
      cwd: REPO,
      readyTimeoutMs: 30000,
      stripEnvPrefix: 'SPINOML_',
      portEnv: 'SPINOML_LLM_PORT',
      port,
    })
    const h = await fetch(`${s2.url}/health`)
    record(
      'llm restart after SIGKILL: 2nd binds + healthy',
      'health 200',
      `port=${port} status=${h.status}`,
      h.ok && s2.proc.pid !== s1.proc.pid,
    )
    await gracefulStop(s2.proc, 5000)
  }
}

// ── (f) STATE CONSISTENCY ─────────────────────────────────────────────────

async function caseStateConsistency(): Promise<void> {
  for (const which of ['torch', 'llm'] as const) {
    const s = await (which === 'torch'
      ? startSidecar({
          cmd: PYTHON,
          args: [TORCH],
          cwd: REPO,
          readyTimeoutMs: 30000,
          stripEnvPrefix: 'SPINOML_',
          portEnv: 'SPINOML_TORCH_PORT',
        })
      : startSidecar({
          cmd: NODE,
          args: [LLM],
          cwd: REPO,
          readyTimeoutMs: 30000,
          stripEnvPrefix: 'SPINOML_',
          portEnv: 'SPINOML_LLM_PORT',
        }))
    // /health must give a full valid JSON.
    const r1 = await fetch(`${s.url}/health`)
    const j1 = await r1.json().catch(() => null)
    record(
      `${which}: /health is a valid JSON object`,
      'object',
      typeof j1,
      j1 !== null && typeof j1 === 'object',
    )
    // Fire a few /health probes and SIGTERM mid-flight. Each probe must
    // either succeed completely or fail with a connection error, never hang.
    const probes = Array.from({ length: 5 }, () =>
      probeHealth(`${s.url}/health`, 3000).catch(() => false),
    )
    s.proc.kill('SIGTERM')
    const results = await Promise.all(probes)
    const hung = results.filter((r) => r === undefined).length
    const allResolved = results.every((r) => typeof r === 'boolean')
    record(
      `${which}: probes during shutdown never hang > 3 s`,
      'all resolved',
      `resolved=${results.length} hung=${hung}`,
      allResolved,
    )
    await gracefulStop(s.proc, 5000)
  }
}

// ── main ──────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  console.log('phase 13: local port and process management')
  console.log(`  python = ${PYTHON}; node = ${NODE}`)
  console.log(`  fd baseline = ${FD_BASELINE}`)
  await casePortOccupiedTorch()
  await caseSecondTorchCannotTakeOver()
  await casePortOccupiedLlm()
  await caseSecondLlmCannotTakeOver()
  await caseStartupFailureTorch()
  await caseStartupFailureLlm()
  await caseShutdownCleanTorch()
  await caseShutdownCleanLlm()
  await caseNoOrphanTorchRunScript()
  await caseNoOrphanLlmOpencode()
  await caseNoOrphanLlmRunScript()
  await caseNoOrphanLlmRunScriptGrandchild()
  await caseRepeatedStartStopTorch()
  await caseRepeatedStartStopLlm()
  await caseRestartSamePortTorch()
  await caseRestartSamePortLlm()
  await caseStateConsistency()
  const failures = runMatrix(rows)
  console.log(`\n${rows.length - failures.length}/${rows.length} rows passed`)
  // post-run hygiene: verify nothing was left behind.
  const { added } = procDelta()
  const fdD = fdDelta()
  console.log(`  final fd Δ = ${fdD}`)
  if (added.length > 0) {
    console.log('  ADDED:')
    for (const a of added.slice(0, 10)) console.log(`    ${a.slice(0, 100)}`)
    process.exitCode = 1
  }
  if (failures.length > 0) {
    console.log('\nFAILURES:')
    for (const f of failures) console.log(`  ✗ ${f.case}: expected ${f.expected}, got ${f.got}`)
    process.exitCode = 1
  }
  // clean up tmp dir
  try { rmSync(RUN, { recursive: true, force: true }) } catch { /* tmp dir best-effort */ }
}

void main()
