// Verifies the training-graph compiler (codegen/trainingGenerator.ts) and, for
// the happy path, runs the Phase-13 trainer against the compiled config on a
// tiny CSV — proving graph → run.json → real training end to end (incl. the
// Phase-14 metrics + callback extras).

import { execSync } from 'node:child_process'
import { writeFileSync, mkdtempSync, mkdirSync, readFileSync, existsSync, copyFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { compileTrainingGraph } from '../src/codegen/trainingGenerator'
import type { TrainingGraphSnapshot } from '../src/training/graph/store'
import { defaultTrainingParams } from '../src/training/graph/registry'
import { buildRunSnapshot } from '../src/training/snapshot'

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
    n('t4', 'ModelSource', { model: 'models/iris-mlp.spinoml' }),
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
check('model relpath', r.plan?.modelRelpath === 'models/iris-mlp.spinoml')
check('dataset relpath', r.plan?.datasetRelpath === 'datasets/iris.csv')
check('target', r.plan?.target === 'species')
check('epochs from TrainLoop', r.plan?.training.epochs === 12)
check('batch from DataLoader', r.plan?.training.batch_size === 8)
check('val_split from Split', r.plan?.training.val_split === 0.25)
check('optimizer kind+lr', r.plan?.training.optimizer.kind === 'AdamW' && r.plan?.training.optimizer.lr === 0.005)
check('scheduler', r.plan?.training.scheduler.kind === 'CosineAnnealingLR')
check('metrics', JSON.stringify(r.plan?.training.metrics) === JSON.stringify(['accuracy', 'f1']))
check('callbacks count', (r.plan?.training.callbacks?.length ?? 0) === 2)

// ── 1b. multitask graph: two Head nodes → heads array, no single target ──
const multi: TrainingGraphSnapshot = {
  nodes: [
    n('m1', 'DatasetSource', { dataset: 'datasets/multi.csv', features: [] }),
    n('m2', 'Split', { val_ratio: 0.25, seed: 7 }),
    n('m3', 'DataLoader', { batch_size: 4, shuffle: true }),
    n('m4', 'ModelSource', { model: 'models/multi.spinoml' }),
    n('m5', 'Head', { output: 'cls', target: 'species', loss: 'CrossEntropyLoss', weight: 1 }),
    n('m6', 'Head', { output: 'reg', target: 'score', loss: 'MSELoss', weight: 0.5 }),
    n('m7', 'Optimizer', { kind: 'Adam', lr: 0.01 }),
    n('m8', 'Metric', { kind: 'accuracy' }),
    n('m9', 'TrainLoop', { epochs: 8, seed: 7, log_every_n_steps: 1 }),
  ],
  edges: [],
}
console.log('compile: multitask graph (2 heads)')
const rm = compileTrainingGraph(multi)
check('ok', rm.ok, JSON.stringify(rm.issues))
check('no single target in multitask', rm.plan?.target === '')
check('heads length 2', (rm.plan?.training.heads?.length ?? 0) === 2)
check('head[0] cls→species CE', rm.plan?.training.heads?.[0].output === 'cls'
  && rm.plan?.training.heads?.[0].target === 'species' && rm.plan?.training.heads?.[0].loss === 'CrossEntropyLoss')
check('head[1] reg→score MSE w=0.5', rm.plan?.training.heads?.[1].output === 'reg'
  && rm.plan?.training.heads?.[1].target === 'score' && rm.plan?.training.heads?.[1].weight === 0.5)

// a Head missing its target → not ok
const multiBad: TrainingGraphSnapshot = {
  nodes: multi.nodes.map((x) => x.id === 'm6' ? n('m6', 'Head', { output: 'reg', target: '', loss: 'MSELoss' }) : x),
  edges: [],
}
check('Head without target → not ok', !compileTrainingGraph(multiBad).ok)

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
  const dir = mkdtempSync(join(tmpdir(), 'spinoml-traingen-'))
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
    // Phase 19: the split MUST be provably leak-free — disjoint subsets, strategy recorded.
    const si = events.find((e) => e.kind === 'split.integrity')
    check('split.integrity emitted (Phase 19)',
      !!si && (si as any).overlap === 0 && (si as any).strategy === 'random'
      && (si as any).train_size > 0 && typeof (si as any).val_size === 'number'
      && (si as any).train_size + (si as any).val_size > 0,
      JSON.stringify(si ?? 'no split.integrity event'))
    // Phase 21: the runtime environment is recorded once at launch — software
    // versions + device/dtype must be present for a real (non-mock) stack.
    const envEv = events.find((e) => e.kind === 'config.env')
    check('config.env records software + device (Phase 21)',
      !!envEv && typeof (envEv as any).python === 'string'
      && typeof (envEv as any).torch === 'string'
      && typeof (envEv as any).device === 'string'
      && typeof (envEv as any).dtype === 'string',
      JSON.stringify(envEv))
    // Phase 22: every random source is seeded and documented in run.determinism.
    const detEv = events.find((e) => e.kind === 'run.determinism')
    check('run.determinism documents all seed states (Phase 22)',
      !!detEv && (detEv as any).seed === 7
      && (detEv as any).cudnn_deterministic === true
      && (detEv as any).cudnn_benchmark === false
      && (detEv as any).python_random === true
      && (detEv as any).numpy_random === true,
      JSON.stringify(detEv))
    // The provenance split.strategy must match what was frozen into run.json
    const provEv = events.find((e) => e.kind === 'run.provenance')
    check('split.strategy frozen into provenance (Phase 19)',
      provEv !== undefined && provEv.split?.strategy === 'random',
      JSON.stringify(provEv?.split ?? 'no split in run.provenance'))
    // eval.summary drives the Run-Detail diagrams: iris is classification → a
    // square confusion matrix over the full val set, with one label per class.
    const evalEv = [...events].reverse().find((e) => e.kind === 'eval.summary')
    const cm = evalEv?.confusion
    const squareCm = !!cm && Array.isArray(cm.matrix) && cm.matrix.length > 0
      && cm.matrix.every((r: number[]) => r.length === cm.matrix.length)
      && Array.isArray(cm.labels) && cm.labels.length === cm.matrix.length
    check('eval.summary emits a confusion matrix (classification)', squareCm,
      cm ? `${cm.matrix.length}x${cm.matrix.length}, labels=${JSON.stringify(cm.labels)}` : 'no eval.summary/confusion')
  } catch (e) {
    const err = e as { stderr?: Buffer; stdout?: Buffer }
    check('trainer ran', false, (err.stderr?.toString() ?? '') + (err.stdout?.toString() ?? ''))
  }
}

// ── 4. end-to-end multitask: a 2-head model trains on 2 target columns ──
console.log('end-to-end: multitask (classification + regression heads)')
if (existsSync(template) && rm.plan) {
  const dir = mkdtempSync(join(tmpdir(), 'spinoml-multitask-'))
  // species = classification target, score = regression target (here ~ petal length).
  writeFileSync(join(dir, 'multi.csv'),
    'sl,sw,pl,pw,species,score\n' +
    '5.1,3.5,1.4,0.2,setosa,1.4\n4.9,3.0,1.4,0.2,setosa,1.4\n4.7,3.2,1.3,0.2,setosa,1.3\n5.0,3.4,1.5,0.2,setosa,1.5\n' +
    '6.4,3.2,4.5,1.5,versicolor,4.5\n6.9,3.1,4.9,1.5,versicolor,4.9\n5.5,2.3,4.0,1.3,versicolor,4.0\n6.0,2.2,4.0,1.0,versicolor,4.0\n' +
    '6.3,3.3,6.0,2.5,virginica,6.0\n5.8,2.7,5.1,1.9,virginica,5.1\n7.1,3.0,5.9,2.1,virginica,5.9\n6.5,3.0,5.8,2.2,virginica,5.8\n')
  // model returns a dict keyed by the head output names (cls + reg).
  writeFileSync(join(dir, 'model.py'),
    'import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n' +
    '    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(4, 16)\n        self.act = nn.ReLU()\n        self.cls = nn.Linear(16, 3)\n        self.reg = nn.Linear(16, 1)\n' +
    '    def forward(self, x):\n        h = self.act(self.fc1(x))\n        return {"cls": self.cls(h), "reg": self.reg(h)}\n')
  const plan = rm.plan
  const runJson = {
    run_id: 'verify-mt', run_label: 'verify-mt', created_at: '2026-01-01T00:00:00Z', status: 'queued',
    model_path: plan.modelRelpath, backend: { kind: 'local' },
    dataset: { path: join(dir, 'multi.csv'), relpath: plan.datasetRelpath, kind: 'tabular',
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
    check('status done', status === 'done', status)
    // per-head loss/metric keys namespaced "<output>/…" so the metrics chart picks them up.
    const perHead = ends.some((e) => e.metrics && 'cls/loss' in e.metrics && 'reg/loss' in e.metrics && 'cls/acc' in e.metrics)
    check('epoch.end carries per-head metrics (cls/loss, reg/loss, cls/acc)', perHead,
      JSON.stringify(ends[ends.length - 1]?.metrics))
    // eval.summary carries a per-head array: cls → confusion matrix, reg → scatter.
    const evalEv = [...events].reverse().find((e) => e.kind === 'eval.summary')
    const heads = evalEv?.heads as Array<Record<string, unknown>> | undefined
    const cls = heads?.find((h) => h.output === 'cls')
    const reg = heads?.find((h) => h.output === 'reg')
    check('eval.summary has per-head cls confusion + reg scatter',
      !!cls && cls.task === 'classification' && !!(cls.confusion)
      && !!reg && reg.task === 'regression' && !!(reg.scatter),
      JSON.stringify(heads?.map((h) => ({ output: h.output, task: h.task }))))
    // sample.preds likewise carries per-head rows.
    const sp = [...events].reverse().find((e) => e.kind === 'sample.preds')
    check('sample.preds carries per-head rows', Array.isArray(sp?.heads) && (sp!.heads as unknown[]).length === 2)
  } catch (e) {
    const err = e as { stderr?: Buffer; stdout?: Buffer }
    check('multitask trainer ran', false, (err.stderr?.toString() ?? '') + (err.stdout?.toString() ?? ''))
  }
}

// ── 5. end-to-end external validation: train once, then eval a trained checkpoint
//    on a RENAMED external dataset (no training) → eval.summary + metrics. ──
console.log('end-to-end: external validation (eval-only on a renamed dataset)')
if (existsSync(template) && r.plan) {
  const dir = mkdtempSync(join(tmpdir(), 'spinoml-evalonly-'))
  const irisCsv =
    'sl,sw,pl,pw,species\n' +
    '5.1,3.5,1.4,0.2,setosa\n4.9,3.0,1.4,0.2,setosa\n4.7,3.2,1.3,0.2,setosa\n5.0,3.4,1.5,0.2,setosa\n' +
    '6.4,3.2,4.5,1.5,versicolor\n6.9,3.1,4.9,1.5,versicolor\n5.5,2.3,4.0,1.3,versicolor\n6.0,2.2,4.0,1.0,versicolor\n' +
    '6.3,3.3,6.0,2.5,virginica\n5.8,2.7,5.1,1.9,virginica\n7.1,3.0,5.9,2.1,virginica\n6.5,3.0,5.8,2.2,virginica\n'
  const modelPy = 'import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n' +
    '    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(4, 16)\n        self.act = nn.ReLU()\n        self.fc2 = nn.Linear(16, 3)\n' +
    '    def forward(self, x):\n        return self.fc2(self.act(self.fc1(x)))\n'
  // (a) train a source run → produces checkpoints/best.pt with the trained classes.
  const srcDir = join(dir, 'src'); mkdirSync(srcDir, { recursive: true })
  writeFileSync(join(srcDir, 'iris.csv'), irisCsv)
  writeFileSync(join(srcDir, 'model.py'), modelPy)
  writeFileSync(join(srcDir, 'run.json'), JSON.stringify({
    run_id: 'src', run_label: 'src', created_at: '2026-01-01T00:00:00Z', status: 'queued',
    model_path: 'm', backend: { kind: 'local' },
    dataset: { path: join(srcDir, 'iris.csv'), relpath: 'iris.csv', kind: 'tabular',
      feature_columns: ['sl', 'sw', 'pl', 'pw'], target_column: 'species' },
    training: r.plan.training,
  }, null, 2))
  copyFileSync(template, join(srcDir, 'train.py'))
  // (b) eval-only on a RENAMED external set (a,b,c,d,label) — same data, new names.
  const evalDir = join(dir, 'eval'); mkdirSync(evalDir, { recursive: true })
  writeFileSync(join(evalDir, 'ext.csv'),
    irisCsv.replace('sl,sw,pl,pw,species', 'a,b,c,d,label'))
  writeFileSync(join(evalDir, 'model.py'), modelPy)
  writeFileSync(join(evalDir, 'run.json'), JSON.stringify({
    run_id: 'eval', run_label: 'ext-val', created_at: '2026-01-01T00:00:00Z', status: 'queued',
    model_path: 'm', backend: { kind: 'local' }, eval_only: true,
    validate: { checkpoint_from: join(srcDir, 'checkpoints', 'best.pt'), source_run: 'src' },
    dataset: { path: join(evalDir, 'ext.csv'), relpath: 'ext.csv', kind: 'tabular',
      feature_columns: ['a', 'b', 'c', 'd'], target_column: 'label' },
    training: { ...r.plan.training, val_split: 0 },
  }, null, 2))
  copyFileSync(template, join(evalDir, 'train.py'))
  try {
    execSync('python -u train.py', { cwd: srcDir, stdio: 'pipe' })
    check('source run produced best.pt', existsSync(join(srcDir, 'checkpoints', 'best.pt')))
    execSync('python -u train.py', { cwd: evalDir, stdio: 'pipe' })
    const events = readFileSync(join(evalDir, 'events.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l))
    const kinds = events.map((e) => e.kind)
    check('eval-only ran NO training loop', !kinds.includes('epoch.start'), kinds.join(','))
    check('checkpoint.loaded emitted', kinds.includes('checkpoint.loaded'))
    const ev = [...events].reverse().find((e) => e.kind === 'eval.summary')
    const cm = ev?.confusion
    // confusion labels must be the TRAINED class order (renamed columns notwithstanding).
    check('eval.summary confusion over trained classes', !!cm
      && JSON.stringify(cm.labels) === JSON.stringify(['setosa', 'versicolor', 'virginica']),
      cm ? JSON.stringify(cm.labels) : 'no eval.summary')
    const metrics = JSON.parse(readFileSync(join(evalDir, 'metrics.json'), 'utf8'))
    check('metrics.json marks eval_only + done', metrics.eval_only === true && metrics.status === 'done',
      JSON.stringify(metrics))
  } catch (e) {
    const err = e as { stderr?: Buffer; stdout?: Buffer }
    check('eval-only trainer ran', false, (err.stderr?.toString() ?? '') + (err.stdout?.toString() ?? ''))
  }
}

// ── 6. split_strategy guard (Phase 19) — an unimplemented strategy MUST fail
//    loudly (exit ≠ 0) rather than silently falling back to random. ──
console.log('split-strategy: unimplemented strategy must fail loudly')
if (existsSync(template) && r.plan) {
  const dir = mkdtempSync(join(tmpdir(), 'spinoml-strategy-'))
  writeFileSync(join(dir, 'iris.csv'),
    'sl,sw,pl,pw,species\n' +
    '5.1,3.5,1.4,0.2,setosa\n4.9,3.0,1.4,0.2,setosa\n4.7,3.2,1.3,0.2,setosa\n5.0,3.4,1.5,0.2,setosa\n' +
    '6.4,3.2,4.5,1.5,versicolor\n6.9,3.1,4.9,1.5,versicolor\n5.5,2.3,4.0,1.3,versicolor\n6.0,2.2,4.0,1.0,versicolor\n' +
    '6.3,3.3,6.0,2.5,virginica\n5.8,2.7,5.1,1.9,virginica\n7.1,3.0,5.9,2.1,virginica\n6.5,3.0,5.8,2.2,virginica\n')
  writeFileSync(join(dir, 'model.py'),
    'import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n' +
    '    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(4, 16)\n        self.act = nn.ReLU()\n        self.fc2 = nn.Linear(16, 3)\n' +
    '    def forward(self, x):\n        return self.fc2(self.act(self.fc1(x)))\n')
  writeFileSync(join(dir, 'run.json'), JSON.stringify({
    run_id: 'strategy', run_label: 'strategy', created_at: '2026-01-01T00:00:00Z', status: 'queued',
    model_path: 'm', backend: { kind: 'local' },
    dataset: { path: join(dir, 'iris.csv'), relpath: 'iris.csv', kind: 'tabular',
      feature_columns: ['sl', 'sw', 'pl', 'pw'], target_column: 'species' },
    training: { ...r.plan.training, split_strategy: 'grouped', val_split: 0.25 },
  }, null, 2))
  copyFileSync(template, join(dir, 'train.py'))
  try {
    execSync('python -u train.py', { cwd: dir, stdio: 'pipe' })
    check('grouped strategy failed loudly', false, 'exit code 0 — silent fallback detected')
  } catch (e) {
    const stderr = (e as { stderr?: Buffer }).stderr?.toString() ?? ''
    check('grouped strategy failed loudly', stderr.includes('split_strategy')
      && stderr.includes('grouped'),
      stderr.slice(0, 200))
  }
}

// ── 7. snapshot integrity (Phase 20) — run dir artifacts must be byte-identical
//    to the launch freeze; mutation after launch is a loud, halting failure. ──
console.log('snapshot: frozen run dir verified at launch, mutation fails loudly')
if (existsSync(template) && r.plan) {
  const dir = mkdtempSync(join(tmpdir(), 'spinoml-snapshot-'))
  writeFileSync(join(dir, 'iris.csv'),
    'sl,sw,pl,pw,species\n' +
    '5.1,3.5,1.4,0.2,setosa\n4.9,3.0,1.4,0.2,setosa\n4.7,3.2,1.3,0.2,setosa\n5.0,3.4,1.5,0.2,setosa\n' +
    '6.4,3.2,4.5,1.5,versicolor\n6.9,3.1,4.9,1.5,versicolor\n5.5,2.3,4.0,1.3,versicolor\n6.0,2.2,4.0,1.0,versicolor\n' +
    '6.3,3.3,6.0,2.5,virginica\n5.8,2.7,5.1,1.9,virginica\n7.1,3.0,5.9,2.1,virginica\n6.5,3.0,5.8,2.2,virginica\n')
  const modelPyDef = 'import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n' +
    '    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(4, 16)\n        self.act = nn.ReLU()\n        self.fc2 = nn.Linear(16, 3)\n' +
    '    def forward(self, x):\n        return self.fc2(self.act(self.fc1(x)))\n'
  const modelSpinoml = JSON.stringify({
    format: 'spinoml', version: 1, savedAt: '2026-01-01T00:00:00Z',
    graph: { nodes: [], edges: [] },
  }, null, 2)
  const snapshot = await buildRunSnapshot(modelSpinoml, modelPyDef)
  writeFileSync(join(dir, 'model.py'), modelPyDef)
  writeFileSync(join(dir, 'model.spinoml'), modelSpinoml)
  writeFileSync(join(dir, 'run.json'), JSON.stringify({
    run_id: 'snap', run_label: 'snap', created_at: '2026-01-01T00:00:00Z', status: 'queued',
    model_path: 'm', backend: { kind: 'local' },
    dataset: { path: join(dir, 'iris.csv'), relpath: 'iris.csv', kind: 'tabular',
      feature_columns: ['sl', 'sw', 'pl', 'pw'], target_column: 'species' },
    training: { ...r.plan.training, split_strategy: 'random', val_split: 0.25 },
    snapshot,
  }, null, 2))
  copyFileSync(template, join(dir, 'train.py'))
  // (a) unmutated → snapshot verifies (ok:true) and training completes.
  try {
    execSync('python -u train.py', { cwd: dir, stdio: 'pipe' })
    const events = readFileSync(join(dir, 'events.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l))
    const snapEv = events.find((e) => e.kind === 'run.snapshot')
    check('snapshot verifies on unmutated run dir (ok:true)', snapEv?.ok === true
      && snapEv?.graph?.matches === true && snapEv?.model_py?.matches === true,
      JSON.stringify(snapEv))
  } catch (e) {
    check('snapshot verifies on unmutated run dir (ok:true)', false,
      (e as { stderr?: Buffer }).stderr?.toString() ?? '')
  }
  // (b) mutate model.py after launch → halting failure, no training happens.
  writeFileSync(join(dir, 'model.py'), modelPyDef.replace('self.fc2', 'self.broken_fc2'))
  writeFileSync(join(dir, 'events.jsonl'), '')
  try {
    execSync('python -u train.py', { cwd: dir, stdio: 'pipe' })
    check('mutated model.py fails loudly', false, 'exit code 0 — drifted model was executed')
  } catch (e) {
    const stderr = (e as { stderr?: Buffer }).stderr?.toString() ?? ''
    const events = readFileSync(join(dir, 'events.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l))
    const snapEv = events.find((e) => e.kind === 'run.snapshot')
    const leakedAcc = events.some((e) => e.kind === 'epoch.end')
    check('mutated model.py fails loudly',
      snapEv?.ok === false && snapEv?.model_py?.matches === false
      && stderr.includes('mutated after launch') && !leakedAcc,
      `snap=${JSON.stringify(snapEv)} leakedEpoch=${leakedAcc}`)
  }
}

console.log(failures === 0 ? '\n✓ all training-gen checks passed' : `\n✗ ${failures} check(s) failed`)
process.exit(failures === 0 ? 0 : 1)
