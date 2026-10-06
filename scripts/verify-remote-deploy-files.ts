#!/usr/bin/env tsx
// Phase 77 regression — `remote_sidecar.rs::deploy()` ships every file the
// remote torch sidecar needs (Python sources + the ESPF BPE codebook). Before
// Phase 77 the deploy only uploaded `main.py` + `dataset_handlers.py`, but
// since Phase 45–47 main.py also imports `scope`, `safe_load`, `deps_policy`
// (and Phase 78 adds `auth`), and `dataset_handlers.py` reads
// `sidecar-torch/espf/*` via `Path(__file__).resolve().parent / "espf"`, a
// remote sidecar would die with `ModuleNotFoundError` on start.
//
// This verifier:
//   1. Parses `SIDECAR_FILES` out of `src-tauri/src/remote_sidecar.rs` (the
//      single source of truth, also enforced by a Rust unit test).
//   2. Computes the LOCAL-import closure of `sidecar-torch/main.py` +
//      `dataset_handlers.py` via `scripts/lib/py_local_imports.py` (also
//      picks up everything under `espf/`).
//   3. Asserts that every file in the closure appears in SIDECAR_FILES, and
//      that every file in SIDECAR_FILES exists locally.
//   4. Prints a side-by-side table; exits 1 on any gap.
//
// MUTATION-PROOF: remove `scope.py` from the constant → this script turns
// red with a "missing from SIDECAR_FILES" line for `scope.py`.

import { execFileSync } from 'node:child_process'
import { readFileSync, statSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const REPO = resolve(HERE, '..')

const REMOTE_SIDECAR_RS = join(REPO, 'src-tauri/src/remote_sidecar.rs')
const SIDECAR_DIR = join(REPO, 'sidecar-torch')
const PY_HELPER = join(REPO, 'scripts/lib/py_local_imports.py')

let failures = 0
function check(name: string, ok: boolean, detail = '') {
  if (ok) console.log(`  \u2713 ${name}`)
  else { failures++; console.log(`  \u2717 ${name}${detail ? '  ' + detail : ''}`) }
}

console.log('phase 77: remote-sidecar deploy file closure')

// ── 1. Parse SIDECAR_FILES out of remote_sidecar.rs ─────────────────────────

const rsSrc = readFileSync(REMOTE_SIDECAR_RS, 'utf8')
// We want the constant in its canonical form: `pub const SIDECAR_FILES: &[&str]
// = &[ ... ];`. We match the array body literal, which is small and bounded.
const m = rsSrc.match(/pub\s+const\s+SIDECAR_FILES:\s*&\[&str\]\s*=\s*&\[([\s\S]*?)\];/)
if (!m) {
  console.log('  \u2717 SIDECAR_FILES constant not found in remote_sidecar.rs')
  process.exit(1)
}
// Drop `//` comments first: the constant's comments quote paths such as "espf"
// and must not be read as declared entries.
const body = m[1].replace(/\/\/[^\n]*/g, '')
// Extract every "..." literal — handles escapes and trims.
const re = /"((?:\\.|[^"\\])*)"/g
const declared: string[] = []
let mm: RegExpExecArray | null
while ((mm = re.exec(body)) !== null) declared.push(mm[1])
check(`SIDECAR_FILES parsed (${declared.length} entries)`, declared.length > 0)

// ── 2. Compute the local-import closure ────────────────────────────────────

let closure: string[] = []
try {
  // The helper is pure-stdlib `ast`, so any python3 works (no torch env needed).
  const raw = execFileSync(
    process.env.PYTHON ?? 'python3',
    [PY_HELPER],
    {
      cwd: REPO,
      input: JSON.stringify({ entries: ['main.py', 'dataset_handlers.py'], sidecar_dir: SIDECAR_DIR }),
      encoding: 'utf8',
      stdio: ['pipe', 'pipe', 'inherit'],
    }
  )
  const parsed = JSON.parse(raw)
  closure = parsed.files ?? []
  check(`closure computed (${closure.length} files)`, Array.isArray(closure) && closure.length > 0)
} catch (e) {
  check('closure computed', false, `\n     ${(e as Error).message}`)
}

// ── 3. Every file in the closure must be in SIDECAR_FILES ──────────────────

const declaredSet = new Set(declared)
const missingFromDeploy = closure.filter((f) => !declaredSet.has(f))
check('closure \u2286 SIDECAR_FILES', missingFromDeploy.length === 0,
  missingFromDeploy.length ? `\n      missing from SIDECAR_FILES: ${missingFromDeploy.join(', ')}` : '')

// ── 4. Every file in SIDECAR_FILES must exist locally ──────────────────────

const missingLocally = declared.filter((f) => {
  try { statSync(join(SIDECAR_DIR, f)) } catch { return true }
  return false
})
check('every SIDECAR_FILES entry exists locally', missingLocally.length === 0,
  missingLocally.length ? `\n      missing on disk: ${missingLocally.join(', ')}` : '')

// ── 5. Print a side-by-side table for humans ───────────────────────────────

const maxLen = Math.max(
  ...declared.map((s) => s.length),
  ...closure.map((s) => s.length),
  'CLOSURE'.length
)
console.log()
console.log('  ' + 'DECLARED'.padEnd(maxLen + 2) + 'CLOSURE'.padEnd(maxLen + 2) + 'STATUS')
console.log('  ' + '-'.repeat(maxLen * 2 + 16))
const union = new Set([...declared, ...closure])
for (const f of Array.from(union).sort()) {
  const inDec = declaredSet.has(f) ? '\u2713' : '\u2717'
  const inClo = closure.includes(f) ? '\u2713' : '\u2717'
  const status = (inDec === inClo) ? 'ok' : (inDec ? 'extra' : 'MISSING')
  console.log(`  ${f.padEnd(maxLen + 2)}${inDec}        ${inClo}        ${status}`)
}

// ── 6. Mutation-proof signal ────────────────────────────────────────────────

if (missingFromDeploy.length > 0) {
  console.log()
  console.log(`  >> ${missingFromDeploy.length} file(s) reachable from main.py / dataset_handlers.py are NOT in the deploy list.`)
  console.log('  >> A remote sidecar would die with ModuleNotFoundError on start. Add them to SIDECAR_FILES.')
}

console.log()
if (failures > 0) {
  console.log(`FAIL: ${failures} check(s) failed`)
  process.exit(1)
}
console.log('OK: deploy file list matches local import closure')
