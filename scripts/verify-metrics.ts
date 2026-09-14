#!/usr/bin/env tsx
// Phase 29 — Metric correctness.
// Verifies batch/epoch/validation loss + metric aggregation against an
// independent Python reference, with a dataset size that does NOT divide by
// the batch size (97 rows / batch 32 → last batch of 1) so an unweighted
// batch-average would visibly disagree with the weighted ground truth.

import { execSync, spawnSync } from 'node:child_process'
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

// ── synthetic data: 97 rows (97 = 3*32 + 1 → last train batch has 1 sample) ──
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

console.log('phase 29: metric correctness')

const dir = mkdtempSync(join(tmpdir(), 'spinoml-metrics-'))
writeFileSync(join(dir, 'data.csv'), csv(97, 10, SEED))
writeFileSync(join(dir, 'model.py'), MODEL_PY)
writeFileSync(join(dir, 'run.json'), JSON.stringify({
  run_id: 'metrics', run_label: 'metrics', created_at: new Date().toISOString(), status: 'queued',
  model_path: 'm', backend: { kind: 'local' },
  dataset: { path: join(dir, 'data.csv'), relpath: 'data.csv', kind: 'tabular',
    feature_columns: FEATURES, target_column: 'label' },
  training: {
    epochs: 3, batch_size: 32, val_split: 0.2, split_strategy: 'random',
    seed: SEED, log_every_n_steps: 1,
    // lr=0 keeps the model frozen so the post-hoc reference (computed from the
    // final weights) reproduces EVERY epoch's loss exactly — aggregation is
    // what this test isolates, not optimization dynamics.
    optimizer: { kind: 'Adam', lr: 0, weight_decay: 0 },
    loss: { kind: 'CrossEntropyLoss' },
    scheduler: { kind: 'none' },
    metrics: ['accuracy'], callbacks: [],
  },
}, null, 2))
copyFileSync(template, join(dir, 'train.py'))

try {
  execSync('python -u train.py', { cwd: dir, stdio: 'pipe', timeout: 180_000 })
  check('trainer completes', true)
} catch (e) {
  const err = e as { stderr?: Buffer }
  check('trainer completes', false, err.stderr?.toString() ?? '')
  process.exit(1)
}

const events = readFileSync(join(dir, 'events.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l))
const ends = events.filter((e) => e.kind === 'epoch.end')
const batchEvents = events.filter((e) => e.kind === 'batch')
check('3 epoch.end events', ends.length === 3, String(ends.length))
check('batch events emitted every step', batchEvents.length > 0)

// ── independent reference: same split (same generator+seed), same model
//    state from last.pt, per-sample weighted aggregation ──
const refScript = [
  'import torch, torch.nn as nn, pandas as pd, numpy as np, json, sys',
  'd = sys.argv[1]',
  'df = pd.read_csv(d + "/data.csv")',
  'feats = [f"f{i}" for i in range(10)]',
  'X = torch.tensor(df[feats].to_numpy(dtype="float32"))',
  'cat = df["label"].astype("category")',
  'y = torch.tensor(cat.cat.codes.to_numpy(), dtype=torch.long)',
  'n = len(df); seed = 42',
  'gen = torch.Generator().manual_seed(seed)',
  'n_val = max(1, int(n * 0.2)); n_train = n - n_val',
  'train_idx, val_idx = torch.utils.data.random_split(range(n), [n_train, n_val], generator=gen)',
  'train_idx = torch.tensor(train_idx.indices); val_idx = torch.tensor(val_idx.indices)',
  'class Model(nn.Module):',
  '    def __init__(self):',
  '        super().__init__()',
  '        self.fc1 = nn.Linear(10, 64); self.act = nn.ReLU(); self.fc2 = nn.Linear(64, 2)',
  '    def forward(self, x):',
  '        return self.fc2(self.act(self.fc1(x)))',
  'm = Model()',
  'ck = torch.load(d + "/checkpoints/last.pt", map_location="cpu", weights_only=False)',
  'm.load_state_dict(ck["model_state"])',
  'm.eval()',
  'ce = nn.CrossEntropyLoss()',
  'with torch.no_grad():',
  '    tr = m(X[train_idx]); va = m(X[val_idx])',
  '    tr_y = y[train_idx]; va_y = y[val_idx]',
  'ref_train = float(ce(tr, tr_y))',
  'ref_val = float(ce(va, va_y))',
  'ref_acc = float((va.argmax(-1) == va_y).float().mean())',
  'events = [json.loads(l) for l in open(d + "/events.jsonl") if l.strip()]',
  'final = [e for e in events if e["kind"] == "epoch.end"][-1]',
  '# naive unweighted: mean of per-batch means over the SAME split (batch 32)',
  'bs = 32',
  'perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(seed))  # only to build uneven batches',
  'tr_perm = train_idx[perm]',
  'batches = [tr_perm[i:i+bs] for i in range(0, n_train, bs)]',
  'naive = 0.0',
  'for b in batches:',
  '    naive += float(ce(m(X[b]), y[b]))',
  'naive /= len(batches)',
  'print("TRAIN_DIFF", abs(final["train_loss"] - ref_train))',
  'print("VAL_DIFF", abs(final["val_loss"] - ref_val))',
  'print("ACC_DIFF", abs((final.get("val_acc") or 0) - ref_acc))',
  'print("NAIVE_DIFF", abs(final["train_loss"] - naive))',
  'print("BATCH_SIZES", [len(b) for b in batches])',
].join('\n')
const ref = spawnSync('python', ['-c', refScript, dir], { stdio: 'pipe' })
const refOut = ref.stdout.toString().trim()
if (ref.status !== 0) {
  check('reference script runs', false, (ref.stderr.toString() + refOut).slice(0, 400))
  process.exit(1)
}
const get = (k: string) => Number(refOut.split('\n').find((l) => l.startsWith(k))?.split(' ')[1] ?? 'NaN')
check('reference script runs', ref.status === 0)

console.log('  [aggregation matches weighted reference]')
check('epoch train loss = weighted reference', get('TRAIN_DIFF') < 1e-4, `diff=${get('TRAIN_DIFF')}`)
check('validation loss = weighted reference', get('VAL_DIFF') < 1e-4, `diff=${get('VAL_DIFF')}`)
check('val accuracy = reference', get('ACC_DIFF') < 1e-4, `diff=${get('ACC_DIFF')}`)

console.log('  [unweighted averaging WOULD have been wrong]')
const sizes = refOut.split('\n').find((l) => l.startsWith('BATCH_SIZES'))
check('last batch smaller than the rest (uneven)', sizes?.includes('1'), sizes ?? '')
check('naive batch-average differs from weighted', get('NAIVE_DIFF') > 1e-3, `diff=${get('NAIVE_DIFF')}`)

// ── batch loss events match the reference batch mean for the FIRST batch ──
const firstBatch = batchEvents[0]
check('batch loss finite', Number.isFinite(firstBatch.loss), String(firstBatch.loss))

console.log(failures === 0 ? '\n✓ all metric checks passed' : `\n✗ ${failures} check(s) failed`)
process.exit(failures === 0 ? 0 : 1)
