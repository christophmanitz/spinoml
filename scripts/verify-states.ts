#!/usr/bin/env tsx
// Phase 30 — Local job state machine.
// Tests the Phase-30 state machine (queued → running → done/failed/cancelled)
// with terminal states protected from stale asynchronous updates.

import { execSync, spawn, spawnSync } from 'node:child_process'
import { writeFileSync, mkdtempSync, readFileSync, existsSync, copyFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const repoRoot = join(import.meta.dirname, '..')
const template = join(repoRoot, 'sidecar-torch', 'training_template.py')

let failures = 0
function check(name: string, cond: boolean, detail = '') {
  if (cond) console.log(`  ✓ ${name}`)
  else { failures++; console.log(`  ✗ ${name} ${detail}`) }
}

// Helper: run python with the trainer module imported in a temp dir
function runTrainerModule(dir: string, script: string): string {
  const r = spawnSync('python', ['-c', script, dir], { cwd: dir, stdio: 'pipe' })
  if (r.status !== 0) throw new Error(r.stderr.toString())
  return r.stdout.toString().trim()
}

function writeRunDir(name: string, opts: { status?: string } = {}) {
  const dir = mkdtempSync(join(tmpdir(), `spinoml-state-${name}-`))
  writeFileSync(join(dir, 'data.csv'),
    'f0,f1,f2,f3,label\n' +
    '1,2,3,4,0\n5,6,7,8,1\n2,3,4,5,0\n6,7,8,9,1\n' +
    '1,3,2,4,0\n5,7,6,8,1\n2,4,3,5,0\n6,7,8,9,1\n' +
    '1,2,3,4,0\n5,6,7,8,1\n2,3,4,5,0\n6,7,8,9,1\n')
  writeFileSync(join(dir, 'model.py'),
    'import torch\nimport torch.nn as nn\n\n' +
    'class Model(nn.Module):\n' +
    '    def __init__(self):\n' +
    '        super().__init__()\n' +
    '        self.fc1 = nn.Linear(4, 16)\n' +
    '        self.act = nn.ReLU()\n' +
    '        self.fc2 = nn.Linear(16, 2)\n' +
    '    def forward(self, x):\n' +
    '        return self.fc2(self.act(self.fc1(x)))\n')
  writeFileSync(join(dir, 'run.json'), JSON.stringify({
    run_id: 'test', run_label: 'test', created_at: new Date().toISOString(), status: 'queued',
    model_path: 'm', backend: { kind: 'local' },
    dataset: { path: join(dir, 'data.csv'), relpath: 'data.csv', kind: 'tabular',
      feature_columns: ['f0','f1','f2','f3'], target_column: 'label' },
    training: { epochs: 3, batch_size: 32, val_split: 0.25, split_strategy: 'random',
      seed: 42, log_every_n_steps: 1,
      optimizer: { kind: 'Adam', lr: 0, weight_decay: 0 },
      loss: { kind: 'CrossEntropyLoss' }, scheduler: { kind: 'none' },
      metrics: ['accuracy'], callbacks: [],
    }, ...(opts.status ? { status: opts.status } : {})
  }, null, 2))
  if (opts.status) writeFileSync(join(dir, 'status'), opts.status + '\n')
  copyFileSync(template, join(dir, 'train.py'))
  return dir
}

// Helper: run trainer module test script
function runStateTest(dir: string, testCode: string): string[] {
  const scriptPath = join(dir, 'test_state.py')
  writeFileSync(join(dir, 'test_state.py'), testCode)
  const r = spawnSync('python', [scriptPath, dir], { cwd: dir, stdio: 'pipe' })
  if (r.status !== 0) throw new Error(r.stderr.toString())
  return r.stdout.toString().trim().split('\n').map((l) => l.trim())
}

// ── 1. Unit tests of the state machine helper (transition_status) ──
console.log('  [state machine: transition_status logic]')
{
  const dir = mkdtempSync(join(tmpdir(), 'spinoml-state-unit-'))
  writeFileSync(join(dir, 'model.py'), 'import torch')
  writeFileSync(join(dir, 'run.json'), JSON.stringify({
    run_id: 'unit', run_label: 'unit', created_at: new Date().toISOString(), status: 'queued',
    model_path: 'm', backend: { kind: 'local' },
    dataset: { path: join(dir, 'data.csv'), relpath: 'data.csv', kind: 'tabular',
      feature_columns: ['f0'], target_column: 'label' },
    training: { epochs: 1, batch_size: 1, val_split: 0, split_strategy: 'random',
      seed: 1, log_every_n_steps: 1, optimizer: { kind: 'Adam', lr: 0 },
      loss: { kind: 'CrossEntropyLoss' }, scheduler: { kind: 'none' },
      metrics: [], callbacks: [],
    }
  }, null, 2))
  copyFileSync(template, join(dir, 'train.py'))

  const testCode = `
import sys, importlib.util, os
spec = importlib.util.spec_from_file_location("trainer", os.path.join(sys.argv[1], "train.py"))
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
status = os.path.join(sys.argv[1], "status")
def u(label, start, nxt):
    with open(status, "w") as f: f.write(start + "\\n")
    ok = trainer.transition_status(nxt)
    print(label, ok, trainer._read_status())
# empty (no status file yet) behaves like pre-launch
u("none->running", "", "running")
u("queued->running", "queued", "running")
# queued -> terminal directly (cancel-before-start)
u("queued->cancelled", "queued", "cancelled")
u("queued->done", "queued", "done")
# terminal states are FINAL: every non-idempotent write is rejected
u("cancelled->queued", "cancelled", "queued")
u("cancelled->running", "cancelled", "running")
u("cancelled->done", "cancelled", "done")
u("cancelled->failed", "cancelled", "failed")
u("done->running", "done", "running")
u("done->cancelled", "done", "cancelled")
u("done->failed", "done", "failed")
u("failed->running", "failed", "running")
u("failed->done", "failed", "done")
u("failed->cancelled", "failed", "cancelled")
# running -> terminal
u("running->done", "running", "done")
u("running->cancelled", "running", "cancelled")
u("running->failed", "running", "failed")
# idempotent re-write of the SAME value is accepted in every state
u("running->running", "running", "running")
u("done->done", "done", "done")
u("cancelled->cancelled", "cancelled", "cancelled")
u("failed->failed", "failed", "failed")
`
  const out = runStateTest(dir, testCode)
  const cases = out.map((l) => l.split(/\s+/))
  const vs = (i: number, k: number) => cases[i][k]
  check('none → running allowed', vs(0, 1) === 'True' && vs(0, 2) === 'running')
  check('queued → running allowed', vs(1, 1) === 'True' && vs(1, 2) === 'running')
  check('queued → cancelled allowed', vs(2, 1) === 'True' && vs(2, 2) === 'cancelled')
  check('queued → done allowed', vs(3, 1) === 'True' && vs(3, 2) === 'done')
  check('cancelled → queued REJECTED', vs(4, 1) === 'False' && vs(4, 2) === 'cancelled')
  check('cancelled → running REJECTED', vs(5, 1) === 'False' && vs(5, 2) === 'cancelled')
  check('cancelled → done REJECTED', vs(6, 1) === 'False' && vs(6, 2) === 'cancelled')
  check('cancelled → failed REJECTED', vs(7, 1) === 'False' && vs(7, 2) === 'cancelled')
  check('done → running REJECTED', vs(8, 1) === 'False' && vs(8, 2) === 'done')
  check('done → cancelled REJECTED', vs(9, 1) === 'False' && vs(9, 2) === 'done')
  check('done → failed REJECTED', vs(10, 1) === 'False' && vs(10, 2) === 'done')
  check('failed → running REJECTED', vs(11, 1) === 'False' && vs(11, 2) === 'failed')
  check('failed → done REJECTED', vs(12, 1) === 'False' && vs(12, 2) === 'failed')
  check('failed → cancelled REJECTED', vs(13, 1) === 'False' && vs(13, 2) === 'failed')
  check('running → done allowed', vs(14, 1) === 'True' && vs(14, 2) === 'done')
  check('running → cancelled allowed', vs(15, 1) === 'True' && vs(15, 2) === 'cancelled')
  check('running → failed allowed', vs(16, 1) === 'True' && vs(16, 2) === 'failed')
  check('same-value rewrite running→running', vs(17, 1) === 'True' && vs(17, 2) === 'running')
  check('same-value rewrite done→done', vs(18, 1) === 'True' && vs(18, 2) === 'done')
  check('same-value rewrite cancelled→cancelled', vs(19, 1) === 'True' && vs(19, 2) === 'cancelled')
  check('same-value rewrite failed→failed', vs(20, 1) === 'True' && vs(20, 2) === 'failed')
}

// ── 2. Integration: cancel before start ──
console.log('  [cancel before start]')
{
  const dir = writeRunDir('cancel-pre', { status: 'cancelled' })
  const r = spawnSync('python', ['-u', 'train.py'], { cwd: dir, stdio: 'pipe' })
  check('exits quickly (no epochs)', r.status === 0)
  check('status stays cancelled', readFileSync(join(dir, 'status'), 'utf8').trim() === 'cancelled')
  const events = (existsSync(join(dir, 'events.jsonl'))
    ? readFileSync(join(dir, 'events.jsonl'), 'utf8').trim().split('\n')
    : []).map((l) => JSON.parse(l))
  check('run.cancelled emitted', events.some((e) => e.kind === 'run.cancelled'))
  check('no epoch.start', !events.some((e) => e.kind === 'epoch.start'))
  check('no run.done', !events.some((e) => e.kind === 'run.done'))
  check('metrics.json status = cancelled', JSON.parse(readFileSync(join(dir, 'metrics.json'), 'utf8')).status === 'cancelled')
}

// ── 3. Integration: cancel during training ──
console.log('  [cancel mid-training via state machine]')
{
  const dir = writeRunDir('cancel-mid')
  const child = spawn('python', ['-u', 'train.py'], { cwd: dir, stdio: 'ignore' })
  let cancelled = false
  for (let i = 0; i < 3000; i++) {
    await new Promise((r) => setTimeout(r, 5))
    if (existsSync(join(dir, 'events.jsonl')) && readFileSync(join(dir, 'events.jsonl'), 'utf8').includes('"kind": "epoch.end", "epoch": 1')) {
      writeFileSync(join(dir, 'status'), 'cancelled\n')
      cancelled = true
      break
    }
  }
  check('cancel injected after epoch 1', cancelled)
  await new Promise<void>((resolve) => child.on('close', () => resolve()))
  check('status = cancelled', readFileSync(join(dir, 'status'), 'utf8').trim() === 'cancelled')
  check('no run.done event', !readFileSync(join(dir, 'events.jsonl'), 'utf8').includes('"kind": "run.done"'))
  check('metrics.json status = cancelled', JSON.parse(readFileSync(join(dir, 'metrics.json'), 'utf8')).status === 'cancelled')
}

// ── 4. CANCELLED → SUCCEEDED race: cancel right after last epoch check ──
console.log('  [CANCELLED → SUCCEEDED race protection]')
{
  const dir = writeRunDir('race-cancel-done')
  const child = spawn('python', ['-u', 'train.py'], { cwd: dir, stdio: 'ignore' })
  let race = false
  for (let i = 0; i < 2000; i++) {
    await new Promise((r) => setTimeout(r, 5))
    if (!existsSync(join(dir, 'events.jsonl'))) continue
    const ev = readFileSync(join(dir, 'events.jsonl'), 'utf8')
    if (ev.includes('"kind": "epoch.start", "epoch": 2')) {
      writeFileSync(join(dir, 'status'), 'cancelled\n')
      race = true
      break
    }
  }
  check('race injected at last epoch start', race)
  await new Promise<void>((resolve) => child.on('close', () => resolve()))
  check('status = cancelled (not done)', readFileSync(join(dir, 'status'), 'utf8').trim() === 'cancelled')
  check('no run.done event', !readFileSync(join(dir, 'events.jsonl'), 'utf8').includes('"kind": "run.done"'))
  check('metrics.json status = cancelled', JSON.parse(readFileSync(join(dir, 'metrics.json'), 'utf8')).status === 'cancelled')
  check('no run.done event after race', !readFileSync(join(dir, 'events.jsonl'), 'utf8').includes('"kind": "run.done"'))
}

// ── 5. Double cancellation (idempotent) ──
console.log('  [double cancellation]')
{
  const dir = writeRunDir('double-cancel')
  const child = spawn('python', ['-u', 'train.py'], { cwd: dir, stdio: 'ignore' })
  for (let i = 0; i < 2000; i++) {
    await new Promise((r) => setTimeout(r, 5))
    if (existsSync(join(dir, 'events.jsonl')) && readFileSync(join(dir, 'events.jsonl'), 'utf8').includes('"kind": "epoch.end", "epoch": 0')) {
      writeFileSync(join(dir, 'status'), 'cancelled\n')
      // second cancel immediately
      writeFileSync(join(dir, 'status'), 'cancelled\n')
      break
    }
  }
  await new Promise<void>((resolve) => child.on('close', () => resolve()))
  check('status = cancelled', readFileSync(join(dir, 'status'), 'utf8').trim() === 'cancelled')
  check('only one run.cancelled event', readFileSync(join(dir, 'events.jsonl'), 'utf8').split('\n').filter((l) => l.includes('"run.cancelled"')).length === 1)
  check('metrics.json status = cancelled', JSON.parse(readFileSync(join(dir, 'metrics.json'), 'utf8')).status === 'cancelled')
}

// ── 6. Trainer guard: SUCCEEDED → CANCELLED rejected (Rust guard can't run without toolchain) ──
console.log('  [trainer guard: SUCCEEDED → CANCELLED rejected]')
{
  const dir = writeRunDir('post-done-cancel')
  execSync('python -u train.py', { cwd: dir, stdio: 'pipe', timeout: 120_000 })
  check('completed = done', readFileSync(join(dir, 'status'), 'utf8').trim() === 'done')
  // A late cancel write races toward the trainer. WITHOUT the Rust guard the
  // file would already say cancelled; the trainer state machine must still
  // refuse to resurrect a SUCCEEDED run into CANCELLED. We test the trainer
  // directly against the 'done' state (no pre-overwrite — that would make the
  // write idempotent and validate nothing).
  const script = [
    'import sys, importlib.util, os',
    'spec = importlib.util.spec_from_file_location("trainer", os.path.join(sys.argv[1], "train.py"))',
    'trainer = importlib.util.module_from_spec(spec)',
    'spec.loader.exec_module(trainer)',
    'print("cur", trainer._read_status())',
    'print("d->c", trainer.transition_status("cancelled"))',
    'print("cur", trainer._read_status())',
  ].join('\n')
  const out = runTrainerModule(dir, script).split('\n').map((l) => l.trim())
  const lastWord = () => out.shift()!.split(/\s+/).pop()
  check('status was done', lastWord() === 'done')
  check('done → cancelled REJECTED by trainer state machine', lastWord() === 'False')
  check('status still done', lastWord() === 'done')
}

console.log(failures === 0 ? '\n✓ all state machine checks passed' : `\n✗ ${failures} check(s) failed`)
process.exit(failures === 0 ? 0 : 1)