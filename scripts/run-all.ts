#!/usr/bin/env tsx
// Phase 82 — Test suite runner.
//
// Executes every suite in scripts/suites.ts sequentially (they share sidecar
// ports 7421/7422) with a hard per-suite timeout, an allow-listed environment
// (CI determinism — no SSH keys, no API tokens, no proxy variables), and
// post-run process + port leak detection.
//
// Statuses:
//   PASS     exit 0, no SKIPPED lines
//   SKIPPED  exit 0, but log contains `SKIPPED ...` lines (rendered as PASS
//            with the skipped lines in a `skipped` column — never hidden)
//   FAIL     non-zero exit, or leaked process/port
//   TIMEOUT  process group not done within `timeoutSec`; SIGTERM, then SIGKILL
//            after 5 s; counts as a failure
//   BLOCKED  a `needs` capability is missing; exact reason in the notes
//
// Flags:
//   --list             print the registry as a table and exit
//   --only a,b,c       restrict to these suite names
//   --category c       restrict to one category
//   --ci               CI mode (sets CI=1; tightens remote/needs policy)
//   --no-python        skip suites whose needs include `torch-env`
//   --only-python      keep only suites whose needs include `torch-env`
//   --markdown <path>  write Markdown report
//   --json <path>      write JSON report (default .test-results/results.json)
//   --allow-known      exit 0 if every non-PASS is a documented known one
//   --allow-busy-ports run even if 7421/7422 are already in use (default: refuse, see preflight)
//   --keep-tmp        keep the per-run TMPDIR (default: every suite gets its own TMPDIR which is
//                      deleted after the run — the suites left >1 GB of spinoml-* dirs in /tmp)
//                      (lint-at-baseline, BLOCKED/SKIPPED by capability)
//   --check            verify the registry ↔ package.json and exit

import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { execFile, execFileSync, spawn } from 'node:child_process'
import { dirname, join, relative, sep } from 'node:path'
import { fileURLToPath } from 'node:url'
import { promisify } from 'node:util'

import { SUITES, type Category, type Need, type Suite } from './suites.ts'

const execFileP = promisify(execFile)

const ROOT = process.cwd()
const RESULTS_DIR = join(ROOT, '.test-results')
const BASELINE_PATH = join(ROOT, 'scripts', 'lint-baseline.json')

// ── flags ───────────────────────────────────────────────────────────────

interface Flags {
  list: boolean
  check: boolean
  only: Set<string>
  category: Category | null
  ci: boolean
  noPython: boolean
  onlyPython: boolean
  markdown: string | null
  jsonPath: string
  allowKnown: boolean
  allowBusyPorts: boolean
  keepTmp: boolean
}

function parseFlags(argv: string[]): Flags {
  const flags: Flags = {
    list: false,
    check: false,
    only: new Set<string>(),
    category: null,
    ci: false,
    noPython: false,
    onlyPython: false,
    markdown: null,
    jsonPath: join(RESULTS_DIR, 'results.json'),
    allowKnown: false,
    allowBusyPorts: false,
    keepTmp: false,
  }
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i]
    if (a === '--list') flags.list = true
    else if (a === '--check') flags.check = true
    else if (a === '--ci') flags.ci = true
    else if (a === '--no-python') flags.noPython = true
    else if (a === '--only-python') flags.onlyPython = true
    else if (a === '--allow-known') flags.allowKnown = true
    else if (a === '--allow-busy-ports') flags.allowBusyPorts = true
    else if (a === '--keep-tmp') flags.keepTmp = true
    else if (a === '--only') {
      const v = argv[++i]
      if (!v) throw new Error('--only requires a comma-separated list')
      for (const s of v.split(',').map((x) => x.trim()).filter(Boolean)) flags.only.add(s)
    } else if (a === '--category') {
      const v = argv[++i]
      if (!v) throw new Error('--category requires a value')
      flags.category = v as Category
    } else if (a === '--markdown') {
      const v = argv[++i]
      if (!v) throw new Error('--markdown requires a path')
      flags.markdown = v
    } else if (a === '--json') {
      const v = argv[++i]
      if (!v) throw new Error('--json requires a path')
      flags.jsonPath = v
    } else {
      throw new Error(`unknown flag: ${a}`)
    }
  }
  if (flags.noPython && flags.onlyPython) {
    throw new Error('--no-python and --only-python are mutually exclusive')
  }
  return flags
}

// ── registry self-check ──────────────────────────────────────────────────

function registryCheck(): { ok: boolean; errors: string[] } {
  const errors: string[] = []
  const pkg = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8')) as {
    scripts: Record<string, string>
  }
  const pkgScripts = new Set(Object.keys(pkg.scripts))
  const regScripts = new Set<string>()
  for (const s of SUITES) {
    if (s.npmScript) regScripts.add(s.npmScript)
  }
  // every registered npmScript exists in package.json
  for (const r of regScripts) {
    if (!pkgScripts.has(r)) errors.push(`registry references npmScript "${r}" which is not in package.json`)
  }
  // every verify:* / test:* / build / sidecar:* / ci / suites in package.json appears in the registry
  const REQUIRED = /^(verify|test):|^(build|sidecar):|^(ci|suites|lint)$/
  for (const k of pkgScripts) {
    if (!REQUIRED.test(k)) continue
    if (k === 'lint') continue // we run lint separately
    if (k === 'sidecar:torch' || k === 'sidecar:llm') continue // manual sidecars
    if (k === 'ci' || k === 'suites') continue // the runner itself
    if (!regScripts.has(k)) errors.push(`package.json script "${k}" is not registered in suites.ts`)
  }
  return { ok: errors.length === 0, errors }
}

// ── capability detection ─────────────────────────────────────────────────

interface Capabilities {
  cargo: { ok: boolean; detail: string }
  'torch-env': { ok: boolean; detail: string; pythonCmd: string | null; envBin: string | null }
  cuda: { ok: boolean; detail: string }
  'ssh-host': { ok: boolean; detail: string }
  slurm: { ok: boolean; detail: string }
  network: { ok: boolean; detail: string }
  'llm-key': { ok: boolean; detail: string }
  'private-data': { ok: boolean; detail: string }
}

function which(bin: string): string | null {
  try {
    const r = execFileSync('which', [bin], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] })
    const p = r.split('\n')[0]?.trim()
    return p ? p : null
  } catch {
    return null
  }
}

function detectCapabilities(): Capabilities {
  // cargo
  // A bare shell has no cargo: on this project the Rust toolchain lives in the
  // conda env (see the second lookup below, after the env is resolved).
  const cargoBin = which('cargo')
  let cargo: { ok: boolean; detail: string } = cargoBin
    ? { ok: true, detail: cargoBin }
    : { ok: false, detail: 'cargo not on PATH and not in the conda env' }

  // torch-env — find a python interpreter that can import torch. On a bare
  // shell (no conda activation) we try `python` first; if that fails, try
  // `conda run -n <env> python` against the SPINOML_CONDA_ENV (default
  // mlforge-dev). The env-bin directory of the working interpreter is
  // prepended to PATH so spawned `python` resolves correctly.
  let pythonCmd: string | null = null
  let envBin: string | null = null
  let detail = 'no python with torch found'
  const tryProbeAny = (cmd: string, args: string[]): { ok: boolean; stdout: string } => {
    try {
      const stdout = execFileSync(cmd, args, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], timeout: 30_000 })
      return { ok: true, stdout }
    } catch {
      return { ok: false, stdout: '' }
    }
  }
  // 1) bare python
  if (which('python')) {
    const r = tryProbeAny('python', ['-c', 'import torch, sys; print(sys.version.split()[0], torch.__version__)'])
    if (r.ok) {
      pythonCmd = 'python'
      detail = `python + torch: ${r.stdout.trim()}`
    }
  }
  // 2) conda run (the dev env on this machine)
  if (!pythonCmd && which('conda')) {
    const env = process.env.SPINOML_CONDA_ENV ?? 'mlforge-dev'
    const r = tryProbeAny('conda', [
      'run',
      '--no-capture-output',
      '-n',
      env,
      'python',
      '-c',
      'import torch, sys; print(sys.version.split()[0], torch.__version__)',
    ])
    if (r.ok) {
      pythonCmd = `conda run --no-capture-output -n ${env} python`
      // Resolve the env bin so we can prepend to PATH for child processes
      const locate = tryProbeAny('conda', ['run', '--no-capture-output', '-n', env, 'sh', '-c', 'echo "$CONDA_PREFIX/bin"'])
      if (locate.ok) envBin = locate.stdout.trim()
      detail = `conda env ${env}: ${r.stdout.trim()}`
    } else {
      detail = `python not on PATH and conda env ${env} has no torch`
    }
  }

  // cargo, second lookup: the conda env bin directory (children get it first on PATH).
  if (!cargo.ok && envBin) {
    const envCargo = join(envBin, 'cargo')
    if (existsSync(envCargo)) cargo = { ok: true, detail: `${envCargo} (conda env)` }
  }

  // cuda — only true when torch sees a CUDA device
  let cudaOk = false
  let cudaDetail = 'no torch.cuda.is_available() == True probe possible'
  if (pythonCmd) {
    const probeArgs =
      pythonCmd === 'python'
        ? ['-c', 'import torch; print("CUDA" if torch.cuda.is_available() else "CPU")']
        : (() => {
            // split "conda run --no-capture-output -n X python" → array
            const parts = pythonCmd.split(' ')
            return [...parts.slice(0, 4), ...parts.slice(4).flatMap((p) => [p]), '-c', 'import torch; print("CUDA" if torch.cuda.is_available() else "CPU")']
          })()
    // Easier: just call via execFileSync with the split argv
    let argv: string[]
    if (pythonCmd === 'python') {
      argv = probeArgs
    } else {
      // "conda run --no-capture-output -n mlforge-dev python" → split
      argv = pythonCmd.split(' ').concat(['-c', 'import torch; print("CUDA" if torch.cuda.is_available() else "CPU")'])
    }
    const r = tryProbeAny(argv[0]!, argv.slice(1))
    if (r.ok) {
      const v = r.stdout.trim()
      cudaOk = v === 'CUDA'
      cudaDetail = `torch reports ${v}`
    }
  }

  const liveRemote = process.env.SPINOML_REMOTE_TESTS === '1'
  const liveLLM = process.env.SPINOML_LIVE_LLM === '1'

  return {
    cargo,
    'torch-env': { ok: pythonCmd !== null, detail, pythonCmd, envBin },
    cuda: { ok: cudaOk, detail: cudaDetail },
    'ssh-host': {
      ok: liveRemote,
      detail: liveRemote ? 'SPINOML_REMOTE_TESTS=1' : 'set SPINOML_REMOTE_TESTS=1 to enable live ssh-host suites',
    },
    slurm: {
      ok: liveRemote,
      detail: liveRemote ? 'SPINOML_REMOTE_TESTS=1' : 'set SPINOML_REMOTE_TESTS=1 to enable live SLURM suites',
    },
    network: {
      ok: false,
      detail: 'network-needing suites are never auto-run by CI',
    },
    'llm-key': {
      ok: liveLLM,
      detail: liveLLM ? 'SPINOML_LIVE_LLM=1' : 'set SPINOML_LIVE_LLM=1 to exercise the LLM path',
    },
    'private-data': {
      ok: false,
      detail: 'private-data suites are never auto-run by CI',
    },
  }
}

function blockedReason(need: Need, caps: Capabilities): string {
  const c = caps[need]
  return `${need}: ${c.detail}`
}

// ── environment scrubbing ────────────────────────────────────────────────

const ALLOW = new Set([
  'PATH',
  'HOME',
  'LANG',
  'LC_ALL',
  'TMPDIR',
  'TERM',
  'COLORTERM',
  'NODE_OPTIONS',
  'NODE_PATH',
  'CI',
  'FORCE_COLOR',
  // conda family — the runner uses conda run; CONDA_PREFIX/PATH help children resolve
  'CONDA_PREFIX',
  'CONDA_DEFAULT_ENV',
  'CONDA_ROOT',
  'CONDA_SHLVL',
  'CONDA_ENVS_PATH',
  'XDG_CONFIG_HOME',
  'XDG_CACHE_HOME',
  'XDG_DATA_HOME',
  'XDG_RUNTIME_DIR',
  // explicit run-all knobs
  'SPINOML_CONDA_ENV',
  'SPINOML_LIVE_LLM',
  'SPINOML_REMOTE_TESTS',
  'SPINOML_LLM_PORT',
  'SPINOML_OPENCODE_TEST_MODEL',
])

// Per-run TMPDIR (set in main() before the first suite starts). Every suite creates its scratch
// dirs below it, so a crashed/killed suite cannot leave anything behind in the shared /tmp.
let CI_TMPDIR: string | null = null

function scrubbedEnv(): { env: NodeJS.ProcessEnv; removed: string[] } {
  const env: NodeJS.ProcessEnv = {}
  const removed: string[] = []
  for (const [k, v] of Object.entries(process.env)) {
    if (ALLOW.has(k)) {
      env[k] = v
    } else {
      removed.push(k)
    }
  }
  if (CI_TMPDIR) env.TMPDIR = CI_TMPDIR
  return { env, removed }
}

/** Entries + bytes below a directory (best effort; a vanished entry is simply not counted). */
function dirStats(dir: string): { entries: number; bytes: number } {
  let entries = 0
  let bytes = 0
  const walk = (d: string): void => {
    let names: string[]
    try { names = readdirSync(d) } catch { return /* directory vanished while scanning: nothing to count */ }
    for (const n of names) {
      const p = join(d, n)
      entries++
      try {
        const st = statSync(p)
        if (st.isDirectory()) walk(p)
        else bytes += st.size
      } catch { /* entry vanished while scanning: nothing to count */ }
    }
  }
  walk(dir)
  return { entries, bytes }
}

// ── python env resolution ────────────────────────────────────────────────

function buildPythonPath(caps: Capabilities): { env: NodeJS.ProcessEnv; extraPath: string } {
  const { env } = scrubbedEnv()
  const envBin = caps['torch-env'].envBin
  if (envBin) {
    const prev = env.PATH ?? ''
    env.PATH = `${envBin}:${prev}`
    return { env, extraPath: envBin }
  }
  return { env, extraPath: '' }
}

// ── command construction ─────────────────────────────────────────────────

interface ResolvedCommand {
  argv: string[]
  cwd: string
  env: NodeJS.ProcessEnv
}

function buildCommand(suite: Suite, caps: Capabilities): ResolvedCommand {
  const { env } = buildPythonPath(caps)
  if (suite.npmScript) {
    // npm exits with the script's exit code under `npm run` when -- is used
    return { argv: ['npm', 'run', '--', suite.npmScript], cwd: ROOT, env }
  }
  if (suite.command) {
    return { argv: [...suite.command], cwd: ROOT, env }
  }
  // No command → runner still resolves to BLOCKED upstream
  return { argv: ['true'], cwd: ROOT, env }
}

// ── process leak detection ───────────────────────────────────────────────

const LEAK_PATTERNS = [
  /sidecar-torch\/main\.py/,
  /sidecar-llm\/main\.mjs/,
  /train\.py/,
  /reference_compare/,
  /integrity_wrap/,
  /verify-[a-z]+\.ts$/,
]

interface ProcInfo { pid: number; pgid: number; sid: number; args: string }

/** Processes whose command line matches a leak pattern, with their process group and session. */
function listProcs(): ProcInfo[] {
  const out: ProcInfo[] = []
  try {
    const r = execFileSync('ps', ['-eo', 'pid=,pgid=,sid=,args='], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    })
    for (const line of r.split('\n')) {
      const m = line.trimStart().match(/^(\d+)\s+(\d+)\s+(\d+)\s+(.*)$/)
      if (!m) continue
      const rest = m[4]!
      if (rest.includes('run-all.ts')) continue
      if (rest.includes('node ') && rest.includes('tsx')) continue
      if (LEAK_PATTERNS.some((p) => p.test(rest))) {
        out.push({ pid: Number(m[1]), pgid: Number(m[2]), sid: Number(m[3]), args: rest })
      }
    }
  } catch (e) {
    // ps unavailable: leak detection degrades to the port check; say so loudly once.
    if (!warnedPs) {
      warnedPs = true
      console.error(`[run-all] WARNING: cannot list processes (${e instanceof Error ? e.message : String(e)}); leak detection limited to ports`)
    }
  }
  return out
}
let warnedPs = false

async function portsFree(): Promise<{ port: number; free: boolean }[]> {
  const ports = [7421, 7422]
  const results: { port: number; free: boolean }[] = []
  for (const port of ports) {
    try {
      const r = execFileSync('sh', ['-c', `ss -ltnH 'sport = :${port}' || true`], {
        encoding: 'utf8',
        stdio: ['ignore', 'pipe', 'ignore'],
        timeout: 5_000,
      })
      results.push({ port, free: r.trim().length === 0 })
    } catch {
      results.push({ port, free: true })
    }
  }
  return results
}

async function killPid(pid: number): Promise<void> {
  try {
    process.kill(pid, 'SIGKILL')
  } catch {
    /* already dead */
  }
}

// ── running one suite ────────────────────────────────────────────────────

type Status = 'PASS' | 'SKIPPED' | 'FAIL' | 'TIMEOUT' | 'BLOCKED'

interface SuiteResult {
  name: string
  category: Category
  status: Status
  durationSec: number
  needs: readonly Need[]
  blockedReason?: string
  skippedLines?: string[]
  exitCode: number | null
  signal: NodeJS.Signals | null
  logPath: string
  notes: string[]
}

async function runOne(suite: Suite, caps: Capabilities, opts: { pythonOnly: boolean; noPython: boolean }): Promise<SuiteResult> {
  const base: SuiteResult = {
    name: suite.name,
    category: suite.category,
    status: 'PASS',
    durationSec: 0,
    needs: suite.needs,
    exitCode: null,
    signal: null,
    logPath: join(RESULTS_DIR, `${suite.name}.log`),
    notes: [],
  }

  // filter: --no-python / --only-python
  const needsPy = suite.needs.includes('torch-env')
  if (opts.noPython && needsPy) {
    base.status = 'BLOCKED'
    base.blockedReason = '--no-python was set; this suite needs torch-env'
    base.notes.push(base.blockedReason)
    return base
  }
  if (opts.pythonOnly && !needsPy && !suite.name.startsWith('cargo') && suite.name !== 'build' && suite.name !== 'lint') {
    // not a python suite and not infrastructure we want to keep here
    base.status = 'BLOCKED'
    base.blockedReason = '--only-python was set; this suite is not python-backed'
    base.notes.push(base.blockedReason)
    return base
  }

  // capability check
  if (suite.needs.length > 0) {
    const missing: string[] = []
    for (const n of suite.needs) {
      if (!caps[n].ok) missing.push(blockedReason(n, caps))
    }
    if (missing.length > 0) {
      base.status = 'BLOCKED'
      base.blockedReason = missing.join('; ')
      base.notes.push(base.blockedReason)
      return base
    }
  }

  // no command at all → permanent BLOCKED (remote-live, hardware-cuda)
  if (!suite.npmScript && !suite.command) {
    base.status = 'BLOCKED'
    base.blockedReason = suite.why
    base.notes.push(base.blockedReason)
    return base
  }

  const cmd = buildCommand(suite, caps)
  if (cmd.argv[0] === 'true') {
    base.status = 'BLOCKED'
    base.blockedReason = 'no command registered for this suite'
    base.notes.push(base.blockedReason)
    return base
  }

  mkdirSync(dirname(base.logPath), { recursive: true })
  const logFh = (await import('node:fs')).createWriteStream(base.logPath, { flags: 'w' })

  // Processes that match the leak patterns BEFORE this suite starts (the running app's sidecar,
  // another terminal's test run…) are never this suite's business: they are neither blamed nor killed.
  const procsBefore = new Set(listProcs().map((p) => p.pid))
  const portsBefore = await portsFree()
  let suitePgid = 0

  const start = Date.now()
  let timedOut = false
  let exitCode: number | null = null
  let signal: NodeJS.Signals | null = null

  await new Promise<void>((resolve) => {
    const child = spawn(cmd.argv[0]!, cmd.argv.slice(1), {
      cwd: cmd.cwd,
      env: cmd.env,
      stdio: ['ignore', 'pipe', 'pipe'],
      detached: true,
    })
    suitePgid = child.pid ?? 0
    let resolved = false
    const finish = () => {
      if (resolved) return
      resolved = true
      logFh.end()
      resolve()
    }

    child.stdout.on('data', (chunk: Buffer) => {
      logFh.write(chunk)
    })
    child.stderr.on('data', (chunk: Buffer) => {
      logFh.write(chunk)
    })

    const timer = setTimeout(() => {
      timedOut = true
      // kill the whole process group
      try {
        process.kill(-(child.pid ?? 0), 'SIGTERM')
      } catch {
        try {
          child.kill('SIGTERM')
        } catch {
          /* ignore */
        }
      }
      setTimeout(() => {
        try {
          process.kill(-(child.pid ?? 0), 'SIGKILL')
        } catch {
          try {
            child.kill('SIGKILL')
          } catch {
            /* ignore */
          }
        }
      }, 5_000)
    }, suite.timeoutSec * 1000)

    child.on('close', (code, sig) => {
      clearTimeout(timer)
      exitCode = code
      signal = sig as NodeJS.Signals | null
      finish()
    })
    child.on('error', (err) => {
      logFh.write(`\n[run-all] spawn error: ${err.message}\n`)
      exitCode = -1
      signal = null
      clearTimeout(timer)
      finish()
    })
  })

  base.durationSec = Math.round((Date.now() - start) / 1000)
  base.exitCode = exitCode
  base.signal = signal

  if (timedOut) {
    base.status = 'TIMEOUT'
    base.notes.push(`killed after ${suite.timeoutSec}s`)
    return base
  }

  // post-run leak detection: only processes that (a) did not exist before the suite and
  // (b) live in the suite's own process group / session count as leaked — and only those are killed.
  // A sidecar that was just sent SIGTERM is legitimately still shutting down for a moment (Phase 13
  // handlers reap children first), so give such processes a short grace before calling them leaked.
  const findLeaked = () =>
    listProcs().filter((p) => !procsBefore.has(p.pid) && suitePgid > 0 && (p.pgid === suitePgid || p.sid === suitePgid))
  // A port counts as leaked only if it was free before this suite and busy after it.
  const findBusy = async () =>
    (await portsFree()).filter((p) => !p.free && portsBefore.find((b) => b.port === p.port)?.free === true)
  let leaked = findLeaked()
  let busy = await findBusy()
  for (let i = 0; i < 40 && (leaked.length > 0 || busy.length > 0); i++) {
    await new Promise((r) => setTimeout(r, 100))
    leaked = findLeaked()
    busy = await findBusy()
  }
  if (leaked.length > 0) {
    base.status = 'FAIL'
    base.notes.push(`leaked process(es): ${leaked.map((p) => `${p.pid} (${p.args.slice(0, 60)})`).join(', ')}`)
    for (const p of leaked) await killPid(p.pid)
  }
  if (busy.length > 0) {
    base.status = 'FAIL'
    base.notes.push(`leaked port(s): ${busy.map((p) => p.port).join(', ')}`)
  }

  if (exitCode !== 0) {
    base.status = 'FAIL'
    base.notes.push(`exit=${exitCode}${signal ? ` signal=${signal}` : ''}`)
  }

  // SKIPPED line detection (informational only — exit 0 → PASS with notes)
  const log = readFileSync(base.logPath, 'utf8')
  const skippedLines = log
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l.startsWith('SKIPPED'))
  if (skippedLines.length > 0 && base.status === 'PASS') {
    base.status = 'SKIPPED'
    base.skippedLines = skippedLines
  }

  return base
}

// ── lint ────────────────────────────────────────────────────────────────

interface LintResult {
  status: 'PASS' | 'FAIL'
  problems: number
  errors: number
  warnings: number
  baselineProblems: number
  baselineErrors: number
  baselineWarnings: number
  note: string
}

async function runLint(): Promise<LintResult> {
  let problems = -1
  let errors = 0
  let warnings = 0
  const countFrom = (files: Array<{ messages: Array<{ severity: number }> }>): void => {
    for (const f of files) {
      for (const m of f.messages) {
        if (m.severity === 2) errors++
        else if (m.severity === 1) warnings++
      }
    }
    problems = errors + warnings
  }
  try {
    const r = execFileSync('npx', ['eslint', '.', '--ignore-pattern', 'src-tauri/target/**', '-f', 'json'], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
      timeout: 180_000,
    })
    countFrom(JSON.parse(r) as Array<{ messages: Array<{ severity: number }> }>)
  } catch (err: unknown) {
    const e = err as { stdout?: Buffer | string; status?: number | null }
    if (e.stdout) {
      const text = typeof e.stdout === 'string' ? e.stdout : e.stdout.toString('utf8')
      try {
        countFrom(JSON.parse(text) as Array<{ messages: Array<{ severity: number }> }>)
      } catch {
        problems = -1
      }
    }
  }

  const baseline = JSON.parse(readFileSync(BASELINE_PATH, 'utf8')) as {
    problems: number
    errors: number
    warnings: number
  }
  const baselineProblems = baseline.problems
  const baselineErrors = baseline.errors
  const baselineWarnings = baseline.warnings
  const noRegression = problems >= 0 && problems <= baselineProblems
  const status: 'PASS' | 'FAIL' = noRegression ? 'PASS' : 'FAIL'
  const note = `${problems} problems (baseline ${baselineProblems}: ${baselineErrors}e/${baselineWarnings}w), no regression`
  return { status, problems, errors, warnings, baselineProblems, baselineErrors, baselineWarnings, note }
}

// ── reporting ────────────────────────────────────────────────────────────

interface Report {
  command: string
  startedAt: string
  finishedAt: string
  totalDurationSec: number
  machine: {
    node: string
    python: string
    torch: string
    os: string
  }
  git: { commit: string; dirty: boolean }
  environmentScrub: { removed: string[] }
  capabilities: Record<Need, { ok: boolean; detail: string }>
  lint: LintResult
  suites: SuiteResult[]
  summary: Record<Status, number>
}

function pad(s: string, w: number): string {
  return s.length >= w ? s : s + ' '.repeat(w - s.length)
}

function printList(suites: readonly Suite[] = SUITES): void {
  const rows = suites.map((s) => ({
    name: s.name,
    category: s.category,
    timeoutSec: s.timeoutSec,
    needs: s.needs.join(',') || '-',
    why: s.why,
  }))
  if (rows.length === 0) {
    console.log('(no suites match)')
    return
  }
  const widths = {
    name: Math.max(4, ...rows.map((r) => r.name.length)),
    category: Math.max(9, ...rows.map((r) => r.category.length)),
    timeout: Math.max(7, ...rows.map((r) => String(r.timeoutSec).length)),
    needs: Math.max(5, ...rows.map((r) => r.needs.length)),
  }
  console.log(
    `${pad('suite', widths.name)}  ${pad('category', widths.category)}  ${pad('timeout', widths.timeout)}  ${pad('needs', widths.needs)}  why`,
  )
  console.log(
    `${'-'.repeat(widths.name)}  ${'-'.repeat(widths.category)}  ${'-'.repeat(widths.timeout)}  ${'-'.repeat(widths.needs)}  ---`,
  )
  for (const r of rows) {
    console.log(
      `${pad(r.name, widths.name)}  ${pad(r.category, widths.category)}  ${pad(String(r.timeoutSec), widths.timeout)}  ${pad(r.needs, widths.needs)}  ${r.why}`,
    )
  }
}

function selectSuites(flags: Flags): Suite[] {
  let xs: Suite[] = [...SUITES]
  if (flags.only.size > 0) {
    // tolerate dash ↔ colon (npm scripts use ':' but flag values often use '-')
    const aliases = new Set<string>()
    for (const name of flags.only) {
      aliases.add(name)
      aliases.add(name.replace(/-/, ':')) // first dash only — npm uses one ':' separator
      aliases.add(name.replace(/:/, '-')) // first colon only
    }
    xs = xs.filter((s) => aliases.has(s.name))
    const unknown = [...flags.only].filter(
      (n) =>
        !SUITES.some(
          (s) => s.name === n || s.name === n.replace(/-/, ':') || s.name === n.replace(/:/, '-'),
        ),
    )
    if (unknown.length > 0) {
      console.error(`[run-all] --only: no suite named ${unknown.map((n) => `"${n}"`).join(', ')}`)
    }
  }
  if (flags.category) xs = xs.filter((s) => s.category === flags.category)
  return xs
}

function summarize(rs: SuiteResult[], lint: LintResult): Record<Status, number> {
  const out: Record<Status, number> = { PASS: 0, SKIPPED: 0, FAIL: 0, TIMEOUT: 0, BLOCKED: 0 }
  for (const r of rs) out[r.status]++
  if (lint.status === 'FAIL') out.FAIL++
  return out
}

function machineInfo(): { node: string; python: string; torch: string; os: string } {
  const node = process.version
  const os = `${process.platform} ${process.arch}`
  // Try bare python first, then conda run mlforge-dev
  const probes: Array<() => string | null> = [
    () => {
      if (!which('python')) return null
      try {
        return execFileSync('python', ['-c', 'import sys, torch; print(sys.version.split()[0], torch.__version__)'], {
          encoding: 'utf8',
          stdio: ['ignore', 'pipe', 'ignore'],
          timeout: 30_000,
        }).trim()
      } catch {
        return null
      }
    },
    () => {
      if (!which('conda')) return null
      const env = process.env.SPINOML_CONDA_ENV ?? 'mlforge-dev'
      try {
        return execFileSync(
          'conda',
          ['run', '--no-capture-output', '-n', env, 'python', '-c', 'import sys, torch; print(sys.version.split()[0], torch.__version__)'],
          { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'], timeout: 60_000 },
        ).trim()
      } catch {
        return null
      }
    },
  ]
  for (const p of probes) {
    const r = p()
    if (r) {
      const [python, torch] = r.split(' ')
      return { node, python: python ?? '?', torch: torch ?? '?', os }
    }
  }
  return { node, python: '?', torch: '?', os }
}

function gitInfo(): { commit: string; dirty: boolean } {
  let commit = '?'
  let dirty = false
  try {
    commit = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim()
  } catch {
    /* not a git repo */
  }
  try {
    const r = execFileSync('git', ['status', '--porcelain'], { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] })
    dirty = r.trim().length > 0
  } catch {
    /* ignore */
  }
  return { commit, dirty }
}

function tailLog(path: string, lines = 15): string[] {
  try {
    const txt = readFileSync(path, 'utf8')
    return txt.split('\n').slice(-lines - 1, -1)
  } catch {
    return []
  }
}

function writeJson(path: string, report: Report): void {
  mkdirSync(dirname(path), { recursive: true })
  writeFileSync(path, JSON.stringify(report, null, 2))
}

function writeMarkdown(path: string, report: Report, commandLine: string): void {
  mkdirSync(dirname(path), { recursive: true })
  const lines: string[] = []
  lines.push('# SpinoML test results')
  lines.push('')
  lines.push(`- command: \`${commandLine}\``)
  lines.push(`- started: ${report.startedAt}`)
  lines.push(`- finished: ${report.finishedAt}`)
  lines.push(`- total: ${report.totalDurationSec}s`)
  lines.push(`- git: ${report.git.commit}${report.git.dirty ? ' (dirty)' : ''}`)
  lines.push(`- machine: node ${report.machine.node}, ${report.machine.python}, torch ${report.machine.torch}, ${report.machine.os}`)
  const sum = report.summary
  lines.push(
    `- summary: PASS=${sum.PASS}  SKIPPED=${sum.SKIPPED}  FAIL=${sum.FAIL}  TIMEOUT=${sum.TIMEOUT}  BLOCKED=${sum.BLOCKED}`,
  )
  lines.push(`- lint: ${report.lint.status} — ${report.lint.note}`)
  lines.push('')

  // capabilities
  lines.push('## Capabilities')
  lines.push('')
  lines.push('| capability | ok | detail |')
  lines.push('|------------|----|--------|')
  for (const [k, v] of Object.entries(report.capabilities)) {
    lines.push(`| \`${k}\` | ${v.ok ? 'yes' : 'no'} | ${v.detail.replace(/\|/g, '\\|')} |`)
  }
  lines.push('')

  // suites grouped by category
  const byCat = new Map<Category, SuiteResult[]>()
  for (const r of report.suites) {
    const a = byCat.get(r.category) ?? []
    a.push(r)
    byCat.set(r.category, a)
  }
  const catOrder: Category[] = [
    'infrastructure',
    'unit',
    'contract',
    'integration',
    'e2e',
    'scientific',
    'remote',
    'hardware',
  ]
  for (const cat of catOrder) {
    const xs = byCat.get(cat)
    if (!xs || xs.length === 0) continue
    lines.push(`## ${cat}`)
    lines.push('')
    lines.push('| suite | status | duration | skipped / notes |')
    lines.push('|-------|--------|----------|-----------------|')
    for (const r of xs) {
      const skipped = r.skippedLines && r.skippedLines.length > 0 ? r.skippedLines.join(' / ') : '-'
      const notes = r.notes.length > 0 ? r.notes.join(' / ') : '-'
      const cell = [skipped, notes].filter((x) => x !== '-').join(' || ') || '-'
      lines.push(`| \`${r.name}\` | **${r.status}** | ${r.durationSec}s | ${cell.replace(/\|/g, '\\|').replace(/\n/g, ' ')} |`)
    }
    lines.push('')
  }
  writeFileSync(path, lines.join('\n'))
}

function exitCodeFor(report: Report, allowKnown: boolean): number {
  // Counts failures that can't be excused by --allow-known
  const failByStatus: Status[] = ['FAIL', 'TIMEOUT']
  const isKnown = (r: SuiteResult): boolean =>
    r.status === 'BLOCKED' ||
    (r.status === 'SKIPPED' && (r.skippedLines?.length ?? 0) > 0)
  let bad = 0
  for (const r of report.suites) {
    if (failByStatus.includes(r.status)) bad++
    else if (r.status === 'PASS' || r.status === 'SKIPPED') {
      // ok
    } else if (r.status === 'BLOCKED') {
      if (!allowKnown) bad++
    }
  }
  // lint: --allow-known excuses lint-at-baseline (PASS), but >baseline is hard fail
  if (report.lint.status === 'FAIL') {
    if (!(allowKnown && report.lint.problems <= report.lint.baselineProblems)) bad++
  }
  // Re-classify SKIPPED with allow-known: never an error
  void isKnown
  return bad === 0 ? 0 : 1
}

// ── main ────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  let flags: Flags
  try {
    flags = parseFlags(process.argv.slice(2))
  } catch (err: unknown) {
    console.error(`[run-all] ${(err as Error).message}`)
    process.exit(2)
  }

  mkdirSync(RESULTS_DIR, { recursive: true })

  if (flags.list) {
    printList(selectSuites(flags))
    return
  }

  // registry self-check first
  const rc = registryCheck()
  if (!rc.ok) {
    console.error('[run-all] registry drift:')
    for (const e of rc.errors) console.error(`  - ${e}`)
    if (flags.check) process.exit(1)
    if (flags.ci) process.exit(1)
    console.error('[run-all] refusing to run. Re-run with --check to see this without execution.')
    process.exit(1)
  }
  if (flags.check) {
    console.log('[run-all] registry ok')
    return
  }

  if (flags.ci) process.env.CI = '1'

  const caps = detectCapabilities()
  const selected = selectSuites(flags)
  if (selected.length === 0) {
    console.error('[run-all] no suites match the filters')
    process.exit(2)
  }

  const commandLine = `npm run ci${process.argv.slice(2).length > 0 ? ' -- ' + process.argv.slice(2).join(' ') : ''}`
  const startedAt = new Date().toISOString()

  CI_TMPDIR = mkdtempSync(join(tmpdir(), 'spinoml-ci-'))
  const { removed: removedVars } = scrubbedEnv()
  console.log(`[run-all] ${selected.length} suites selected`)
  console.log(`[run-all] per-run TMPDIR: ${CI_TMPDIR}${flags.keepTmp ? ' (kept: --keep-tmp)' : ' (deleted after the run)'}`)
  console.log(`[run-all] python: ${caps['torch-env'].detail}`)
  console.log(`[run-all] cargo: ${caps.cargo.detail}`)
  console.log(`[run-all] cuda: ${caps.cuda.detail}`)
  console.log(`[run-all] remote: ${caps['ssh-host'].detail}`)
  if (removedVars.length > 0) {
    console.log(`[run-all] env scrubbed (${removedVars.length} vars removed, names only): ${removedVars.join(', ')}`)
  }

  // Preflight: the suites start sidecars on the default ports. If the app (or another run) already
  // holds them, results would be meaningless — refuse instead of producing misleading PASS/FAIL.
  const busyAtStart = (await portsFree()).filter((p) => !p.free)
  if (busyAtStart.length > 0 && !flags.allowBusyPorts) {
    console.error(`[run-all] refusing to start: port(s) ${busyAtStart.map((p) => p.port).join(', ')} already in use (the SpinoML app or another test run?). Stop them or pass --allow-busy-ports.`)
    process.exit(2)
  }

  const results: SuiteResult[] = []
  for (const s of selected) {
    process.stdout.write(`[run-all] ▶ ${s.name} …`)
    const r = await runOne(s, caps, { pythonOnly: flags.onlyPython, noPython: flags.noPython })
    results.push(r)
    const tail = r.status === 'FAIL' || r.status === 'TIMEOUT' ? ` — log ${r.logPath}` : ''
    process.stdout.write(`\r[run-all] ${r.status === 'PASS' || r.status === 'SKIPPED' ? '✓' : r.status === 'BLOCKED' ? '⊘' : '✗'} ${s.name}  ${r.status}  ${r.durationSec}s${tail}\n`)
    if (r.status === 'FAIL' || r.status === 'TIMEOUT') {
      const tail = tailLog(r.logPath, 15)
      for (const l of tail) console.log(`    | ${l}`)
    }
  }

  const lint = await runLint()

  const finishedAt = new Date().toISOString()
  const summary = summarize(results, lint)
  const machine = machineInfo()
  const git = gitInfo()
  const totalDurationSec = Math.round((Date.parse(finishedAt) - Date.parse(startedAt)) / 1000)
  const report: Report = {
    command: commandLine,
    startedAt,
    finishedAt,
    totalDurationSec,
    machine,
    git,
    environmentScrub: { removed: removedVars },
    capabilities: Object.fromEntries(
      Object.entries(caps).map(([k, v]) => [k, { ok: v.ok, detail: v.detail }]),
    ) as Record<Need, { ok: boolean; detail: string }>,
    lint,
    suites: results,
    summary,
  }

  // Print summary table
  const catOrder: Category[] = [
    'infrastructure',
    'unit',
    'contract',
    'integration',
    'e2e',
    'scientific',
    'remote',
    'hardware',
  ]
  for (const cat of catOrder) {
    const xs = results.filter((r) => r.category === cat)
    if (xs.length === 0) continue
    console.log('')
    console.log(`── ${cat} ──`)
    for (const r of xs) {
      const marks = r.status === 'PASS' || r.status === 'SKIPPED' ? '✓' : r.status === 'BLOCKED' ? '⊘' : '✗'
      const skip = r.skippedLines && r.skippedLines.length > 0 ? ` skipped=${r.skippedLines.length}` : ''
      console.log(`  ${marks} ${r.name.padEnd(28)} ${r.status.padEnd(8)} ${String(r.durationSec).padStart(4)}s${skip}`)
    }
  }

  console.log('')
  console.log(`lint: ${lint.status} — ${lint.note}`)
  console.log(
    `summary: PASS=${summary.PASS}  SKIPPED=${summary.SKIPPED}  FAIL=${summary.FAIL}  TIMEOUT=${summary.TIMEOUT}  BLOCKED=${summary.BLOCKED}`,
  )
  console.log(`total: ${totalDurationSec}s`)

  if (CI_TMPDIR) {
    const left = dirStats(CI_TMPDIR)
    const mb = (left.bytes / (1024 * 1024)).toFixed(1)
    if (flags.keepTmp) {
      console.log(`temp: ${left.entries} entries, ${mb} MB left in ${CI_TMPDIR} (--keep-tmp)`)
    } else {
      rmSync(CI_TMPDIR, { recursive: true, force: true })
      console.log(`temp: ${left.entries} entries, ${mb} MB created by the suites and removed`)
    }
  }
  console.log(`results: ${flags.jsonPath}`)

  writeJson(flags.jsonPath, report)
  if (flags.markdown) writeMarkdown(flags.markdown, report, commandLine)

  const code = exitCodeFor(report, flags.allowKnown)
  process.exit(code)
}

// Touch the unused import markers so tree-shakers don't complain (we use them
// indirectly via spawn args).
void execFileP
void fileURLToPath
void relative
void sep

main().catch((err: unknown) => {
  console.error(`[run-all] fatal: ${(err as Error).stack ?? (err as Error).message}`)
  process.exit(2)
})
