// Verifies the training-graph compiler (codegen/trainingGenerator.ts) and, for
// the happy path, runs the Phase-13 trainer against the compiled config on a
// tiny CSV — proving graph → run.json → real training end to end (incl. the
// Phase-14 metrics + callback extras).

import { execSync } from 'node:child_process'
import { writeFileSync, mkdtempSync, readFileSync, existsSync, copyFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { compileTrainingGraph } from '../src/codegen/trainingGenerator'
import type { TrainingGraphSnapshot } from '../src/training/graph/store'
import { defaultTrainingParams } from '../src/training/graph/registry'

function n(id: string, trainingType: string, params: Record<string, unknown> = {}) {
  return { id, trainingType, params: { ...defaultTrainingParams(trainingType), ...params } }
}

let failures = 0
function check(name: string, cond: boolean, detail = '') {
  if (cond) {
    console.log(`  ✓ ${name}`)
  } else {
    failures++
    console.log(`  ✗ ${name} ${detail}`)
  }
}

// ── 1. full graph compiles to a plan ──
const full: TrainingGraphSnapshot = {
  nodes: [
    n('t1', 'DatasetSource', { dataset: 'datasets/iris.csv', target: 'species', features: [] }),
    n('t2', 'Split', { val_ratio: 0.25, seed: 7 }),
    n('t3', 'DataLoader', { batch_size: 8, shuffle: true }),
    n('t4', 'ModelSource', { model: 'models/iris-mlp.mlforge' }),
    n('t5', 'Loss', { kind: 'CrossEntropyLoss' }),
    n('t6', 'Optimizer', { kind: 'AdamW', lr: 0.005, weight_decay: 0.01 }),
    n('t7', 'Scheduler', { kind: 'CosineAnnealingLR' }),
    n('t8', 'Metric', { kind: 'accuracy' }),
    n('t9', 'Metric', { kind: 'f1' }),
    n('t10', 'EarlyStopping', { monitor: 'val_loss', patience: 3, mode: 'min' }),
    n('t11', 'GradientClipping', { max_norm: 0.5 }),
    n('t12', 'TrainLoop', { epochs: 12, seed: 7, log_every_n_steps: 1 }),
  ],
  edges: [],
}

console.log('compile: full graph')
const r = compileTrainingGraph(full)
check('ok', r.ok, JSON.stringify(r.issues))
check('plan present', !!r.plan)
check('model relpath', r.plan?.modelRelpath === 'models/iris-mlp.mlforge')
check('dataset relpath', r.plan?.datasetRelpath === 'datasets/iris.csv')
check('target', r.plan?.target === 'species')
check('epochs from TrainLoop', r.plan?.training.epochs === 12)
check('batch from DataLoader', r.plan?.training.batch_size === 8)
check('val_split from Split', r.plan?.training.val_split === 0.25)
check('optimizer kind+lr', r.plan?.training.optimizer.kind === 'AdamW' && r.plan?.training.optimizer.lr === 0.005)
check('scheduler', r.plan?.training.scheduler.kind === 'CosineAnnealingLR')
check('metrics', JSON.stringify(r.plan?.training.metrics) === JSON.stringify(['accuracy', 'f1']))
check('callbacks count', (r.plan?.training.callbacks?.length ?? 0) === 2)

// ── 2. missing TrainLoop → not ok, with issues ──
console.log('compile: missing TrainLoop')
const partial: TrainingGraphSnapshot = { nodes: full.nodes.filter((x) => x.trainingType !== 'TrainLoop'), edges: [] }
const r2 = compileTrainingGraph(partial)
check('not ok', !r2.ok)
check('reports missing TrainLoop', r2.issues.some((m) => m.includes('TrainLoop')))
check('plan null', r2.plan === null)

// ── 3. end-to-end: compiled config trains on a real CSV ──
console.log('end-to-end: compiled config trains')
const repoRoot = join(import.meta.dirname, '..')
const template = join(repoRoot, 'sidecar-torch', 'training_template.py')
if (!existsSync(template)) {
  check('training template exists', false, template)
} else {
  const dir = mkdtempSync(join(tmpdir(), 'mlforge-traingen-'))
  writeFileSync(join(dir, 'iris.csv'),
    'sl,sw,pl,pw,species\n' +
    '5.1,3.5,1.4,0.2,setosa\n4.9,3.0,1.4,0.2,setosa\n4.7,3.2,1.3,0.2,setosa\n5.0,3.4,1.5,0.2,setosa\n' +
    '6.4,3.2,4.5,1.5,versicolor\n6.9,3.1,4.9,1.5,versicolor\n5.5,2.3,4.0,1.3,versicolor\n6.0,2.2,4.0,1.0,versicolor\n' +
    '6.3,3.3,6.0,2.5,virginica\n5.8,2.7,5.1,1.9,virginica\n7.1,3.0,5.9,2.1,virginica\n6.5,3.0,5.8,2.2,virginica\n')
  writeFileSync(join(dir, 'model.py'),
    'import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n' +
    '    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(4, 16)\n        self.act = nn.ReLU()\n        self.fc2 = nn.Linear(16, 3)\n' +
    '    def forward(self, x):\n        return self.fc2(self.act(self.fc1(x)))\n')
  // assemble run.json the way the store would, using the compiled plan
  const plan = r.plan!
  const runJson = {
    run_id: 'verify', run_label: 'verify', created_at: '2026-01-01T00:00:00Z', status: 'queued',
    model_path: plan.modelRelpath, backend: { kind: 'local' },
    dataset: { path: join(dir, 'iris.csv'), relpath: plan.datasetRelpath, kind: 'tabular',
      feature_columns: plan.features, target_column: plan.target },
    training: plan.training,
  }
  writeFileSync(join(dir, 'run.json'), JSON.stringify(runJson, null, 2))
  copyFileSync(template, join(dir, 'train.py'))
  try {
    execSync('python -u train.py', { cwd: dir, stdio: 'pipe' })
    const status = readFileSync(join(dir, 'status'), 'utf8').trim()
    const events = readFileSync(join(dir, 'events.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l))
    const ends = events.filter((e) => e.kind === 'epoch.end')
    const hasMetrics = ends.some((e) => e.metrics && 'accuracy' in e.metrics && 'f1' in e.metrics)
    const stoppedEarlyOrDone = status === 'done'
    check('status done', stoppedEarlyOrDone, status)
    check('epoch.end carries accuracy+f1 metrics', hasMetrics)
    check('extras config event emitted', events.some((e) => e.kind === 'config.extras'))
  } catch (e) {
    const err = e as { stderr?: Buffer; stdout?: Buffer }
    check('trainer ran', false, (err.stderr?.toString() ?? '') + (err.stdout?.toString() ?? ''))
  }
}

console.log(failures === 0 ? '\n✓ all training-gen checks passed' : `\n✗ ${failures} check(s) failed`)
process.exit(failures === 0 ? 0 : 1)
