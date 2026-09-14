#!/usr/bin/env tsx
// Phase 24 — Training failure tests.
// Deliberately causes each failure mode the auditor listed and asserts the
// trainer FAILS (exit ≠ 0, status ≠ done, run.failed event) — NEVER SUCCESS,
// NEVER stuck indefinitely in RUNNING.

import { execSync, spawn } from 'node:child_process'
import { writeFileSync, mkdtempSync, readFileSync, existsSync, copyFileSync, chmodSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const repoRoot = join(import.meta.dirname, '..')
const template = join(repoRoot, 'sidecar-torch', 'training_template.py')

let failures = 0
function check(name: string, cond: boolean, detail = '') {
  if (cond) console.log(`    ✓ ${name}`)
  else { failures++; console.log(`    ✗ ${name} ${detail}`) }
}

// ── synthetic data ──
function csv(rows: number, features: number, seed: number): string {
  let s = seed >>> 0
  function rand() { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return (s >>> 0) / 0x100000000 }
  function randn() { const u1 = rand() + 1e-12; return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * rand()) }
  const hdr = Array.from({ length: features }, (_, i) => `f${i}`).concat('label').join(',')
  const lines = [hdr]
  for (let i = 0; i < rows; i++) {
    const label = i % 2 === 0 ? 0 : 1
    const vals = Array.from({ length: features }, (_, j) => {
      const mean = label === 0 ? -0.5 + j * 0.1 : 0.5 + j * 0.1
      return (mean + randn() * 0.8).toFixed(6)
    })
    lines.push([...vals, String(label)].join(','))
  }
  return lines.join('\n') + '\n'
}

// ── standard model.py: Linear(10,64) -> ReLU -> Linear(64,2) ──
const MODEL_PY =
  'import torch\nimport torch.nn as nn\n\n' +
  'class Model(nn.Module):\n' +
  '    def __init__(self):\n' +
  '        super().__init__()\n' +
  '        self.fc1 = nn.Linear(10, 64)\n' +
  '        self.act = nn.ReLU()\n' +
  '        self.fc2 = nn.Linear(64, 2)\n' +
  '    def forward(self, x):\n' +
  '        return self.fc2(self.act(self.fc1(x)))\n'

const SEED = 42
const FEATURES = Array.from({ length: 10 }, (_, i) => `f${i}`)

function makeRunJson(overrides: Record<string, unknown> = {}) {
  return {
    run_id: 'test',
    run_label: 'test',
    created_at: new Date().toISOString(),
    status: 'queued',
    model_path: 'm',
    backend: { kind: 'local' },
    dataset: {
      path: '__CSV__',
      relpath: 'data.csv',
      kind: 'tabular',
      feature_columns: FEATURES,
      target_column: 'label',
    },
    training: {
      epochs: 2,
      batch_size: 32,
      val_split: 0.2,
      split_strategy: 'random',
      seed: SEED,
      log_every_n_steps: 1,
      optimizer: { kind: 'Adam', lr: 0.01, weight_decay: 0 },
      loss: { kind: 'CrossEntropyLoss' },
      scheduler: { kind: 'none' },
      metrics: ['accuracy'],
      callbacks: [],
    },
    ...overrides,
  }
}

// Create a standard run dir; return the dir path. Caller is responsible for any mutations.
function makeDir(name: string, opts?: { csv?: string; modelPy?: string; runJsonOverride?: Record<string, unknown>; deleteCsv?: boolean }) {
  const dir = mkdtempSync(join(tmpdir(), `spinoml-fail-${name}-`))
  writeFileSync(join(dir, 'data.csv'), opts?.csv ?? csv(60, 10, SEED))
  writeFileSync(join(dir, 'model.py'), opts?.modelPy ?? MODEL_PY)
  const runJson = makeRunJson(opts?.runJsonOverride ?? {})
  if (!opts?.deleteCsv) runJson.dataset.path = join(dir, 'data.csv')
  writeFileSync(join(dir, 'run.json'), JSON.stringify(runJson, null, 2))
  copyFileSync(template, join(dir, 'train.py'))
  return dir
}

function tryRun(dir: string, timeoutMs = 120_000): { exitCode: number; stderr: string } {
  try {
    execSync('python -u train.py', { cwd: dir, stdio: 'pipe', timeout: timeoutMs })
    return { exitCode: 0, stderr: '' }
  } catch (e) {
    const err = e as { status?: number; stderr?: Buffer; stdout?: Buffer }
    return { exitCode: err.status ?? 1, stderr: (err.stderr?.toString() ?? '') + (err.stdout?.toString() ?? '') }
  }
}

function readEvents(dir: string): Array<Record<string, unknown>> {
  const p = join(dir, 'events.jsonl')
  if (!existsSync(p)) return []
  return readFileSync(p, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l))
}

function readStatus(dir: string): string {
  const p = join(dir, 'status')
  if (!existsSync(p)) return '<missing>'
  return readFileSync(p, 'utf8').trim()
}

function runCase(name: string, dir: string, opts: { expectStage?: string; expectFailedEvent?: boolean } = {}) {
  const { expectStage, expectFailedEvent = true } = opts
  console.log(`  [${name}]`)
  const { exitCode, stderr } = tryRun(dir)
  const status = readStatus(dir)
  const events = readEvents(dir)
  const failedEv = events.find((e) => e.kind === 'run.failed') as Record<string, unknown> | undefined

  check('exit code ≠ 0', exitCode !== 0, `exit=${exitCode}`)
  check('status ≠ done', status !== 'done', `status=${status}`)
  if (expectFailedEvent) check('run.failed event emitted', !!failedEv, failedEv ? '' : 'no run.failed event')
  if (expectStage && failedEv) check(`stage = ${expectStage}`, failedEv.stage === expectStage, JSON.stringify(failedEv.stage))
  if (!expectFailedEvent) check('process crashed (non-clean exit)', exitCode !== 0 && stderr !== '')
}

// ────────────────────────────────────────────────────────────────────────────
console.log('phase 24: training failure tests')
if (!existsSync(template)) {
  check('training template exists', false, template)
  process.exit(1)
}

// ── 1. Invalid dataset — CSV deleted so load_tabular raises ──
runCase('invalid dataset (missing CSV)', makeDir('ds-missing', { deleteCsv: true }), { expectStage: 'dataset' })

// ── 2. Invalid model — model.py has a syntax error ──
runCase('invalid model (syntax error)',
  makeDir('model-bad', { modelPy: 'class Model:\n  def __init__(self): raise "broken"\n' }), { expectStage: 'model' })

// ── 3. Invalid optimizer — unknown kind ──
runCase('invalid optimizer (unknown kind)',
  makeDir('opt-bad', { runJsonOverride: { training: { ...makeRunJson().training, optimizer: { kind: 'FancyMomentum', lr: 0.01 } } } }),
  { expectStage: 'model' })

// ── 4. Invalid learning rate — negative lr → torch raises ValueError ──
runCase('invalid learning rate (negative lr)',
  makeDir('lr-bad', { runJsonOverride: { training: { ...makeRunJson().training, optimizer: { kind: 'Adam', lr: -0.1 } } } }),
  { expectStage: 'model' })

// ── 5. Missing output directory — replace checkpoints/ dir with a file ──
const missingOutDir = makeDir('out-missing')
writeFileSync(join(missingOutDir, 'checkpoints'), 'not-a-dir')
runCase('missing output dir (checkpoints is a file)', missingOutDir, { expectFailedEvent: false })

// ── 6. Unwritable output directory — chmod 555 after writing all files ──
console.log('  [unwritable output dir]')
const unwritableDir = makeDir('unwritable')
chmodSync(unwritableDir, 0o555)
// dir is now read-only; set_status("running") will fail → unhandled → exit ≠ 0
try {
  const { exitCode, stderr } = tryRun(unwritableDir)
  chmodSync(unwritableDir, 0o755)  // restore for cleanup
  check('exit code ≠ 0', exitCode !== 0, `exit=${exitCode}`)
  // status file stays whatever was last written (or doesn't exist) — must NOT be done
  check('status ≠ done', readStatus(unwritableDir) !== 'done')
  // no run.failed event (couldn't write events) — just confirm the process crashed
  check('process crashed (not clean exit)', stderr.includes('PermissionError') || exitCode !== 0)
} catch { chmodSync(unwritableDir, 0o755) }

// ── 7. NaN input — CSV with NaN in a feature → load_tabular now FAILS LOUDLY ──
const nanCsv = (() => {
  const lines = csv(60, 10, SEED).split('\n')
  // inject NaN into row 2, column f3
  const row = lines[2].split(',')
  row[3] = 'NaN'
  lines[2] = row.join(',')
  return lines.join('\n')
})()
runCase('NaN input', makeDir('nan-input', { csv: nanCsv }), { expectStage: 'dataset' })

// ── 8. Numerical failure during training — model turns NaN after N forwards ──
// Phase 25: a NaN loss produced mid-training must fail with stage 'numeric',
// never be reported as success.
runCase('numerical failure (NaN loss mid-training)',
  makeDir('nan-loss', {
    modelPy:
      'import torch\nimport torch.nn as nn\n\n' +
      'class Model(nn.Module):\n' +
      '    def __init__(self):\n' +
      '        super().__init__()\n' +
      '        self.fc1 = nn.Linear(10, 64)\n' +
      '        self.act = nn.ReLU()\n' +
      '        self.fc2 = nn.Linear(64, 2)\n' +
      '        self.calls = 0\n' +
      '    def forward(self, x):\n' +
      '        self.calls += 1\n' +
      '        out = self.fc2(self.act(self.fc1(x)))\n' +
      '        if self.calls > 8:\n' +
      '            out = out * float("nan")\n' +
      '        return out\n',
    runJsonOverride: { training: { ...makeRunJson().training, epochs: 4 } },
  }),
  { expectStage: 'numeric' })

// ── 9. Training process termination — kill -9 mid-training ──
console.log('  [process termination]')
{
  // 20k rows so ≥3 epochs take long enough on CPU to observe a mid-training kill.
  const dir = makeDir('kill-run', { csv: csv(20000, 10, SEED) })
  const child = spawn('python', ['-u', 'train.py'], { cwd: dir, stdio: 'ignore' })
  // Wait for the FIRST epoch.start event (training genuinely in progress), then kill.
  const evPath = join(dir, 'events.jsonl')
  let started = false
  for (let i = 0; i < 2000; i++) {                   // up to 20s
    await new Promise((r) => setTimeout(r, 10))
    if (existsSync(evPath) && readFileSync(evPath, 'utf8').includes('"kind": "epoch.start"')) { started = true; break }
  }
  check('observed an epoch in progress', started)
  child.kill('SIGKILL')

  // Wait for the process to exit
  const exitCode = await new Promise<number>((resolve) => child.on('close', (code) => resolve(code ?? -1)))
  const status = readStatus(dir)
  check('process killed (exit ≠ 0 or signalled)', exitCode !== 0 || started === false, `exit=${exitCode}`)
  check('status ≠ done', status !== 'done', `status=${status}`)
  check('no run.done event after kill', !readEvents(dir).some((e) => e.kind === 'run.done'))
}

// ── summary ──
console.log(failures === 0 ? '\n✓ all failure tests passed' : `\n✗ ${failures} check(s) failed`)
process.exit(failures === 0 ? 0 : 1)
