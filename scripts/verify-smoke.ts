#!/usr/bin/env tsx
// Phase 23 — Scientific smoke test.
// Creates a synthetic tabular dataset (100 samples, 10 features, 2 classes),
// trains a tiny MLP (Linear→ReLU→Linear) for 5 epochs, and asserts that
// training started, loss is finite and decreasing, metrics are finite,
// checkpoint/logs/metadata exist, and the run exits successfully.

import { execSync } from 'node:child_process'
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

// ── synthetic data generation ──
function generateCsv(rows: number, features: number, seed: number): string {
  // Simple seeded PRNG (xorshift32) so the dataset is reproducible.
  let s = seed >>> 0
  function rand() {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5
    return ((s >>> 0) / 0x100000000)
  }
  function randn() {
    // Box-Muller transform
    const u1 = rand() + 1e-12
    const u2 = rand()
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
  }
  const header = Array.from({ length: features }, (_, i) => `f${i}`).concat('label').join(',')
  const lines = [header]
  for (let i = 0; i < rows; i++) {
    const label = i % 2 === 0 ? 0 : 1
    const vals = Array.from({ length: features }, (_, j) => {
      // class-conditional mean shifts per feature so the classes are separable
      const mean = label === 0 ? -0.5 + j * 0.1 : 0.5 + j * 0.1
      return (mean + randn() * 0.8).toFixed(6)
    })
    lines.push([...vals, String(label)].join(','))
  }
  return lines.join('\n') + '\n'
}

const SMOKE_SEED = 42

console.log('phase 23: scientific smoke test')
if (!existsSync(template)) {
  check('training template exists', false, template)
  process.exit(1)
}

const dir = mkdtempSync(join(tmpdir(), 'spinoml-smoke-'))
writeFileSync(join(dir, 'synthetic.csv'), generateCsv(100, 10, SMOKE_SEED))

writeFileSync(join(dir, 'model.py'),
  'import torch\nimport torch.nn as nn\n\n' +
  'class Model(nn.Module):\n' +
  '    def __init__(self):\n' +
  '        super().__init__()\n' +
  '        self.fc1 = nn.Linear(10, 64)\n' +
  '        self.act = nn.ReLU()\n' +
  '        self.fc2 = nn.Linear(64, 2)\n' +
  '    def forward(self, x):\n' +
  '        return self.fc2(self.act(self.fc1(x)))\n')

const runJson = {
  run_id: 'smoke',
  run_label: 'smoke',
  created_at: new Date().toISOString(),
  status: 'queued',
  model_path: 'm',
  backend: { kind: 'local' },
  dataset: {
    path: join(dir, 'synthetic.csv'),
    relpath: 'synthetic.csv',
    kind: 'tabular',
    feature_columns: Array.from({ length: 10 }, (_, i) => `f${i}`),
    target_column: 'label',
  },
  training: {
    epochs: 5,
    batch_size: 16,
    val_split: 0.2,
    split_strategy: 'random',
    seed: SMOKE_SEED,
    log_every_n_steps: 1,
    optimizer: { kind: 'Adam', lr: 0.01, weight_decay: 0 },
    loss: { kind: 'CrossEntropyLoss' },
    scheduler: { kind: 'none' },
    metrics: ['accuracy'],
    callbacks: [],
  },
}
writeFileSync(join(dir, 'run.json'), JSON.stringify(runJson, null, 2))
copyFileSync(template, join(dir, 'train.py'))

try {
  execSync('python -u train.py', { cwd: dir, stdio: 'pipe', timeout: 120_000 })
} catch (e) {
  const err = e as { stderr?: Buffer; stdout?: Buffer }
  check('trainer exited 0', false, (err.stderr?.toString() ?? '') + (err.stdout?.toString() ?? ''))
  process.exit(1)
}

// ── assertions ──
const status = readFileSync(join(dir, 'status'), 'utf8').trim()
const events = readFileSync(join(dir, 'events.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l))
const epochs = events.filter((e) => e.kind === 'epoch.end')

check('training exits done', status === 'done', `status=${status}`)
check('5 epochs emitted', epochs.length === 5, `got ${epochs.length}`)

// loss must be finite and strictly decreasing (or at least end < start)
const losses = epochs.map((e) => e.train_loss)
check('all losses are finite', losses.every((l) => Number.isFinite(l)), JSON.stringify(losses))
check('loss decreases', losses[losses.length - 1] < losses[0],
  `first=${losses[0]} last=${losses[losses.length - 1]}`)

// val_loss must be finite when present
const valLosses = epochs.map((e) => e.val_loss).filter((v) => v != null)
if (valLosses.length > 0) {
  check('all val_losses are finite', valLosses.every((l) => Number.isFinite(l)), JSON.stringify(valLosses))
}

// metrics (accuracy) must be finite and within [0, 1]
const accs = epochs.map((e) => e.metrics?.accuracy).filter((a) => a != null)
check('accuracy metrics present', accs.length === 5, `got ${accs.length}`)
check('accuracy finite and in [0,1]',
  accs.every((a) => Number.isFinite(a) && a >= 0 && a <= 1),
  JSON.stringify(accs))

// checkpoint exists
check('checkpoint exists', existsSync(join(dir, 'checkpoints', 'best.pt')))

// metrics.json exists and contains expected fields
const metricsPath = join(dir, 'metrics.json')
check('metrics.json exists', existsSync(metricsPath))
if (existsSync(metricsPath)) {
  const mj = JSON.parse(readFileSync(metricsPath, 'utf8'))
  check('metrics.json status=done', mj.status === 'done', JSON.stringify(mj.status))
  check('metrics.json has best_val_loss', typeof mj.best_val_loss === 'number' && Number.isFinite(mj.best_val_loss))
  check('metrics.json has epochs', mj.epochs === 5)
  check('metrics.json has n_params', typeof mj.n_params === 'number' && mj.n_params > 0)
}

// run.provenance + run.determinism events present
check('run.provenance event', events.some((e) => e.kind === 'run.provenance'))
check('run.determinism event', events.some((e) => e.kind === 'run.determinism'))

console.log(failures === 0 ? '\n✓ smoke test passed' : `\n✗ ${failures} check(s) failed`)
process.exit(failures === 0 ? 0 : 1)
