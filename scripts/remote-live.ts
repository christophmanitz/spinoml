#!/usr/bin/env tsx
// scripts/remote-live.ts — thin wrapper around the env-gated Rust live tests.
//
// The Rust suite is `cargo test --manifest-path src-tauri/Cargo.toml live_ --
// --ignored --test-threads=1 --nocapture`. This wrapper:
//   - refuses to even spawn cargo when SPINOML_REMOTE_TESTS != 1 or
//     SPINOML_REMOTE_ALIAS is unset (exits 2 with a one-line explanation, so
//     the suites runner can mark it BLOCKED before invoking anything),
//   - preflights `ssh -o BatchMode=yes -o ConnectTimeout=15 <alias> true` so a
//     wrong key/known_hosts shows up HERE (not inside cargo),
//   - parses per-test outcomes: a test that printed `SKIPPED: <name>: ...`
//     counts as SKIPPED even though cargo's harness reports it `ok`; a test is
//     only PASSED if it ran, did not print a SKIPPED marker, and cargo did not
//     report it failed,
//   - fails if `live_connection` did not truly pass (reachability canary),
//   - prints the leftover-check command and fails if THIS run's root dir
//     (<base>/<stamp>, the stamp shared with the tests via
//     SPINOML_REMOTE_RUN_STAMP) still exists on the cluster.
//
// Env: SPINOML_REMOTE_PYTHON defaults to `python3` (the login node has no bare
// `python`); SPINOML_REMOTE_ROOT_BASE defaults to `~/spinoml-live-test`.

import { spawn, spawnSync, type ChildProcess } from 'node:child_process'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const REPO = join(dirname(fileURLToPath(import.meta.url)), '..')

function fail(msg: string, code = 2): never {
  process.stderr.write(msg + '\n')
  process.exit(code)
}

/** Single-quote a string for `bash -c`. */
function sq(s: string): string {
  return "'" + s.replace(/'/g, "'\\''") + "'"
}

const envTests = process.env.SPINOML_REMOTE_TESTS
const envAlias = process.env.SPINOML_REMOTE_ALIAS
if (envTests !== '1' || !envAlias) {
  fail(
    `BLOCKED: set SPINOML_REMOTE_TESTS=1 and SPINOML_REMOTE_ALIAS=<alias> (a Host in ~/.ssh/config that BatchMode=yes can reach without a password prompt) to enable the live remote suite.`,
  )
}
const ALIAS = envAlias

const BASE = process.env.SPINOML_REMOTE_ROOT_BASE ?? '~/spinoml-live-test'
// The tests embed this same stamp (via SPINOML_REMOTE_RUN_STAMP) in their root
// path, so the leftover check below probes the exact dir this run created.
const RUN_STAMP = Math.floor(Date.now() / 1000).toString()

// ── preflight: ssh BatchMode=yes true on the alias ────────────────────────
{
  const r = spawnSync(
    'ssh',
    ['-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '--', ALIAS, 'true'],
    { encoding: 'utf8' },
  )
  if (r.status !== 0) {
    fail(
      `preflight: ssh -o BatchMode=yes ${ALIAS} true failed (exit=${r.status ?? '?'}): ${(r.stderr ?? '').trim() || '(no stderr)'}`,
    )
  }
}

// ── run the cargo test ────────────────────────────────────────────────────
const cargoArgs = [
  'test',
  '--manifest-path', 'src-tauri/Cargo.toml',
  'live_',
  '--',
  '--ignored',
  '--test-threads=1',
  '--nocapture',
]
const child: ChildProcess = spawn('cargo', cargoArgs, {
  cwd: REPO,
  env: { ...process.env, SPINOML_REMOTE_RUN_STAMP: RUN_STAMP },
  stdio: ['ignore', 'pipe', 'pipe'],
})

let stdout = ''
let stderr = ''
child.stdout!.on('data', (b: Buffer) => {
  const s = b.toString('utf8')
  stdout += s
  process.stdout.write(s)
})
child.stderr!.on('data', (b: Buffer) => {
  const s = b.toString('utf8')
  stderr += s
  process.stderr.write(s)
})
await new Promise<void>((resolve) => child.on('exit', () => resolve()))

// ── parse outcomes ────────────────────────────────────────────────────────
// With --nocapture the test's own output (stderr `SKIPPED: ...`) is interleaved
// and the result token may land on its own line, so we do NOT rely on the
// inline `... ok` text. Instead:
//   - collect the `test <name> ...` prefixes to know which tests RAN,
//   - collect `SKIPPED: <name>: ...` markers (the Rust TestGuard threads the
//     test name in) to know which tests self-skipped,
//   - take passed/failed counts from cargo's `test result:` summary, then
//     subtract the self-skipped from `passed` (cargo counts them as ok).
const combined = stdout + '\n' + stderr
const lines = combined.split('\n')

const ranNames = new Set<string>()
for (const line of lines) {
  const m = /^test\s+(?:[\w:]+::)?(live_\w+)\s+\.\.\./.exec(line)
  if (m) ranNames.add(m[1])
}

const skippedByName = new Set<string>()
for (const line of lines) {
  const m = /^\s*SKIPPED:\s*(live_\w+)\s*:/.exec(line)
  if (m) skippedByName.add(m[1])
}

let cargoPassed = 0
let cargoFailed = 0
const sumRe = /test result:\s+\w+\.\s+(\d+) passed;\s+(\d+) failed;\s+(\d+) ignored/
for (const line of lines) {
  const m = sumRe.exec(line)
  if (m) {
    cargoPassed += Number(m[1])
    cargoFailed += Number(m[2])
  }
}

const skipped = skippedByName.size
const passed = Math.max(0, cargoPassed - skipped)
const failed = cargoFailed

console.log('')
console.log(`remote-live: PASS=${passed} FAIL=${failed} SKIPPED=${skipped}`)

if (ranNames.size === 0) {
  console.error('remote-live: zero live_ tests actually ran — refusing to claim success for a misconfigured suite')
  process.exit(2)
}
if (passed === 0 && failed === 0) {
  console.error('remote-live: every test SKIPPED — env not set or every preflight failed')
  process.exit(2)
}
if (failed > 0) {
  console.error(`remote-live: ${failed} test(s) FAILED`)
  process.exit(1)
}

// live_connection is the canary: if it did not TRULY pass, the cluster is not
// actually reachable for training, even though no test reported FAILED.
const liveConnectionOk =
  ranNames.has('live_connection') &&
  !skippedByName.has('live_connection') &&
  passed > 0
if (!liveConnectionOk) {
  console.error('remote-live: live_connection did not truly pass (reachability canary) — treating as failure')
  process.exit(1)
}

// ── leftover check: THIS run's timestamped root must not exist ─────────────
const humanCmd = `ssh ${ALIAS} ls -d ${BASE}/${RUN_STAMP}`
console.log('')
console.log(`leftover-check: ${humanCmd}`)
const remoteCmd = `ls -d ${BASE}/${RUN_STAMP}`
const r = spawnSync('bash', ['-c', `ssh ${sq(ALIAS)} ${sq(remoteCmd)}`], { encoding: 'utf8' })
const leftover = (r.stdout ?? '').trim()
if (leftover.length > 0) {
  console.error(`remote-live: FAILED leftover check — ${leftover} still exists on ${ALIAS}`)
  console.error(`  manual cleanup: ssh ${ALIAS} 'rm -rf ${leftover}'`)
  process.exit(1)
}
console.log(`leftover-check: ok (no <base>/${RUN_STAMP} from this run on ${ALIAS})`)

console.log('remote-live: all OK')
process.exit(0)
