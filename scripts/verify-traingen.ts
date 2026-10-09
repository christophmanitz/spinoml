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
import { coerceTrainingParams, defaultTrainingParams } from '../src/training/graph/registry'
import { buildRunSnapshot } from '../src/training/snapshot'

function n(id: string, trainingType: string, params: Record<string, unknown> = {}) {
  return { id, trainingType, params: { ...defaultTrainingParams(trainingType), ...params } }
}

interface SplitIntegrityEvent {
  overlap: number
  strategy: string
  train_size: number
  val_size: number
  group_column?: string | null
  n_groups_train?: number | null
  n_groups_val?: number | null
  group_overlap?: number
}
interface ConfigEnvEvent {
  python: string
  torch: string
  device: string
  dtype: string
}
interface DeterminismEvent {
  seed: number
  cudnn_deterministic: boolean
  cudnn_benchmark: boolean
  python_random: boolean
  numpy_random: boolean
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
// Column heads must stay byte-identical: the diffusion fields live on the
// score head / training.diffusion only, never as noise on a column head.
check('column heads carry no diffusion keys',
  JSON.stringify(rm.plan?.training.heads) === JSON.stringify([
    { output: 'cls', target: 'species', loss: 'CrossEntropyLoss', weight: 1, label_smoothing: 0 },
    { output: 'reg', target: 'score', loss: 'MSELoss', weight: 0.5, label_smoothing: 0 },
  ]),
  JSON.stringify(rm.plan?.training.heads))
check('no diffusion object without a score head', rm.plan?.training.diffusion === undefined)

// a Head missing its target → not ok
const multiBad: TrainingGraphSnapshot = {
  nodes: multi.nodes.map((x) => x.id === 'm6' ? n('m6', 'Head', { output: 'reg', target: '', loss: 'MSELoss' }) : x),
  edges: [],
}
check('Head without target → not ok', !compileTrainingGraph(multiBad).ok)

// ── 1d. score head only (diffusion) ──
const scoreOnly: TrainingGraphSnapshot = {
  nodes: [
    n('s1', 'DatasetSource', { dataset: 'datasets/iris.csv', target: '', features: [] }),
    n('s2', 'Split', { val_ratio: 0.2, seed: 42 }),
    n('s3', 'DataLoader', { batch_size: 8, shuffle: true }),
    n('s4', 'ModelSource', { model: 'models/iris-mlp.spinoml' }),
    n('s5', 'Head', { output: 'eps_hat', target_kind: 'score', diff_branch: 'wat', sigma_min: 0.05, sigma_max: 6.0, n_rep: 4, weight: 1, loss: 'MSELoss' }),
    n('s6', 'Optimizer', { kind: 'Adam', lr: 0.001 }),
    n('s7', 'TrainLoop', { epochs: 5, seed: 1, log_every_n_steps: 1 }),
  ],
  edges: [],
}
const rs = compileTrainingGraph(scoreOnly)
check('score head compiles ok', rs.ok, JSON.stringify(rs.issues))
check('score head needs no target column', rs.plan?.target === '')
// The score head entry is the contract's, verbatim.
check('score head entry === contract',
  JSON.stringify(rs.plan?.training.heads) === JSON.stringify([
    { output: 'eps_hat', target: '', loss: 'MSELoss', weight: 1, target_kind: 'score', task: 'regression' },
  ]),
  JSON.stringify(rs.plan?.training.heads))
check('training.diffusion === contract',
  JSON.stringify(rs.plan?.training.diffusion)
  === JSON.stringify({ branch: 'wat', sigma_min: 0.05, sigma_max: 6, n_rep: 4 }),
  JSON.stringify(rs.plan?.training.diffusion))
// The branch name is free text — it must survive the registry coercion that
// runs on EVERY param write (otherwise the trainer could never see it).
check('diff_branch survives coerceTrainingParams',
  coerceTrainingParams('Head', { diff_branch: 'wat' }).diff_branch === 'wat')
check('target_kind defaults to column',
  defaultTrainingParams('Head').target_kind === 'column')

// score head together with another head → the objective cannot be shared
const scorePlusCol: TrainingGraphSnapshot = {
  nodes: [...scoreOnly.nodes, n('s8', 'Head', { output: 'cls', target: 'species', loss: 'CrossEntropyLoss' })],
  edges: [],
}
const rspc = compileTrainingGraph(scorePlusCol)
check('score + column head not ok', !rspc.ok && rspc.plan === null)
check('score + column head names the conflict', rspc.issues.some((m) => m.includes('einzige Head')), JSON.stringify(rspc.issues))

// missing diff_branch → error
const missingBranch: TrainingGraphSnapshot = {
  nodes: scoreOnly.nodes.map((x) => x.id === 's5' ? n('s5', 'Head', { output: 'eps_hat', target_kind: 'score', diff_branch: '' }) : x),
  edges: [],
}
const rmb = compileTrainingGraph(missingBranch)
check('missing diff_branch not ok', !rmb.ok && rmb.issues.some((m) => m.includes('diff_branch')), JSON.stringify(rmb.issues))

// sigma_min >= sigma_max → error
const badSigma: TrainingGraphSnapshot = {
  nodes: scoreOnly.nodes.map((x) => x.id === 's5'
    ? n('s5', 'Head', { output: 'eps_hat', target_kind: 'score', diff_branch: 'wat', sigma_min: 6, sigma_max: 0.05 })
    : x),
  edges: [],
}
const rbs = compileTrainingGraph(badSigma)
check('sigma_min>=sigma_max not ok', !rbs.ok && rbs.issues.some((m) => m.includes('sigma_min')), JSON.stringify(rbs.issues))

// ── 1c. grouped split: the Split node's group_column reaches the run config ──
const grouped: TrainingGraphSnapshot = {
  nodes: full.nodes.map((x) => x.id === 't2'
    ? n('t2', 'Split', { strategy: 'grouped', group_column: 'patient_id', val_ratio: 0.25, seed: 7 })
    : x),
  edges: [],
}
const rg = compileTrainingGraph(grouped)
check('grouped split compiles', rg.ok, JSON.stringify(rg.issues))
check('group_column → training.split_group_column',
  rg.plan?.training.split_strategy === 'grouped' && rg.plan?.training.split_group_column === 'patient_id',
  JSON.stringify(rg.plan?.training.split_group_column))
// grouped WITHOUT a group column cannot be leakage-safe → blocked before launch.
const groupedNoCol: TrainingGraphSnapshot = {
  nodes: full.nodes.map((x) => x.id === 't2' ? n('t2', 'Split', { strategy: 'grouped', val_ratio: 0.25, seed: 7 }) : x),
  edges: [],
}
check('grouped split without group_column → not ok',
  !compileTrainingGraph(groupedNoCol).ok
  && compileTrainingGraph(groupedNoCol).issues.some((m) => m.includes('group_column')))

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
      !!si && (si as SplitIntegrityEvent).overlap === 0 && (si as SplitIntegrityEvent).strategy === 'random'
      && (si as SplitIntegrityEvent).train_size > 0 && typeof (si as SplitIntegrityEvent).val_size === 'number'
      && (si as SplitIntegrityEvent).train_size + (si as SplitIntegrityEvent).val_size > 0,
      JSON.stringify(si ?? 'no split.integrity event'))
    // Phase 21: the runtime environment is recorded once at launch — software
    // versions + device/dtype must be present for a real (non-mock) stack.
    const envEv = events.find((e) => e.kind === 'config.env')
    check('config.env records software + device (Phase 21)',
      !!envEv && typeof (envEv as ConfigEnvEvent).python === 'string'
      && typeof (envEv as ConfigEnvEvent).torch === 'string'
      && typeof (envEv as ConfigEnvEvent).device === 'string'
      && typeof (envEv as ConfigEnvEvent).dtype === 'string',
      JSON.stringify(envEv))
    // Phase 22: every random source is seeded and documented in run.determinism.
    const detEv = events.find((e) => e.kind === 'run.determinism')
    check('run.determinism documents all seed states (Phase 22)',
      !!detEv && (detEv as DeterminismEvent).seed === 7
      && (detEv as DeterminismEvent).cudnn_deterministic === true
      && (detEv as DeterminismEvent).cudnn_benchmark === false
      && (detEv as DeterminismEvent).python_random === true
      && (detEv as DeterminismEvent).numpy_random === true,
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
    training: { ...r.plan.training, split_strategy: 'stratified', val_split: 0.25 },
  }, null, 2))
  copyFileSync(template, join(dir, 'train.py'))
  try {
    execSync('python -u train.py', { cwd: dir, stdio: 'pipe' })
    check('stratified strategy failed loudly', false, 'exit code 0 — silent fallback detected')
  } catch (e) {
    const stderr = (e as { stderr?: Buffer }).stderr?.toString() ?? ''
    check('stratified strategy failed loudly', stderr.includes('split_strategy')
      && stderr.includes('stratified'),
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

// ── 8. grouped split + ranking metrics: end-to-end on a tabular run ──
// 6 groups × 4 rows, each group holding BOTH classes (so the val partition is
// never single-class): val_split 0.25 → 6 val rows → two whole groups.
console.log('end-to-end: grouped split (leakage-safe) + auroc/auprc/ef')
if (existsSync(template) && r.plan) {
  const dir = mkdtempSync(join(tmpdir(), 'spinoml-grouped-'))
  const rows = ['sl,sw,patient,active']
  for (let g = 0; g < 6; g++) {
    for (let j = 0; j < 4; j++) {
      rows.push(`${(g * 4 + j) / 10},${(g + j) / 5},P${g},${j >= 2 ? 'yes' : 'no'}`)
    }
  }
  writeFileSync(join(dir, 'groups.csv'), rows.join('\n') + '\n')
  writeFileSync(join(dir, 'model.py'),
    'import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n' +
    '    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(2, 8)\n        self.fc2 = nn.Linear(8, 1)\n' +
    '    def forward(self, x):\n        return self.fc2(torch.relu(self.fc1(x)))\n')
  const plan = r.plan!
  writeFileSync(join(dir, 'run.json'), JSON.stringify({
    run_id: 'grouped', run_label: 'grouped', created_at: '2026-01-01T00:00:00Z', status: 'queued',
    model_path: plan.modelRelpath, backend: { kind: 'local' },
    dataset: { path: join(dir, 'groups.csv'), relpath: plan.datasetRelpath, kind: 'tabular',
      feature_columns: ['sl', 'sw'], target_column: 'active' },
    training: { ...plan.training, epochs: 2, batch_size: 4, split_strategy: 'grouped',
      split_group_column: 'patient', val_split: 0.25, seed: 7,
      loss: { kind: 'BCEWithLogitsLoss' }, metrics: ['auroc', 'auprc', 'ef'] },
  }, null, 2))
  copyFileSync(template, join(dir, 'train.py'))
  try {
    execSync('python -u train.py', { cwd: dir, stdio: 'pipe' })
    const events = readFileSync(join(dir, 'events.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l))
    const si = events.find((e) => e.kind === 'split.integrity') as SplitIntegrityEvent | undefined
    // Whole groups only: sample overlap AND group overlap must both be 0.
    check('grouped split.integrity: strategy + zero sample/group overlap',
      !!si && si.strategy === 'grouped' && si.overlap === 0 && si.group_overlap === 0
      && si.group_column === 'patient' && si.train_size > 0 && si.val_size > 0,
      JSON.stringify(si ?? 'no split.integrity event'))
    check('grouped split.integrity counts whole groups (train+val = 6)',
      !!si && (si.n_groups_train ?? 0) + (si.n_groups_val ?? 0) === 6
      && (si.n_groups_val ?? 0) > 0 && (si.n_groups_train ?? 0) > 0,
      JSON.stringify(si))
    // The val partition holds WHOLE groups → its size is a multiple of the group size (4).
    check('val partition is whole groups (val_size % 4 === 0)', !!si && si.val_size % 4 === 0,
      JSON.stringify({ val_size: si?.val_size, n_groups_val: si?.n_groups_val }))
    const ends = events.filter((e) => e.kind === 'epoch.end')
    const m = (ends[ends.length - 1]?.metrics ?? {}) as Record<string, number>
    const finite01 = (v: unknown) => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1
    check('epoch.end reports auroc/auprc/ef in [0,1]',
      finite01(m.auroc) && finite01(m.auprc) && finite01(m.ef),
      JSON.stringify(m))
    const metricsJson = JSON.parse(readFileSync(join(dir, 'metrics.json'), 'utf8')) as Record<string, unknown>
    check('metrics.json: status done', metricsJson.status === 'done', JSON.stringify(metricsJson.status))
    const mj = (metricsJson.metrics ?? {}) as Record<string, number>
    check('metrics.json carries a finite auroc/auprc/ef in [0,1]',
      finite01(mj.auroc) && finite01(mj.auprc) && finite01(mj.ef),
      JSON.stringify(mj))
    check('metrics.json carries no NaN metric', !JSON.stringify(metricsJson).includes('NaN'))
  } catch (e) {
    const err = e as { stderr?: Buffer; stdout?: Buffer }
    check('grouped trainer ran', false, (err.stderr?.toString() ?? '') + (err.stdout?.toString() ?? ''))
  }

  // (b) grouped split on a column that isn't in the table → loud failure.
  const badDir = mkdtempSync(join(tmpdir(), 'spinoml-grouped-bad-'))
  writeFileSync(join(badDir, 'groups.csv'), rows.join('\n') + '\n')
  copyFileSync(join(dir, 'model.py'), join(badDir, 'model.py'))
  const badCfg = JSON.parse(readFileSync(join(dir, 'run.json'), 'utf8'))
  badCfg.dataset.path = join(badDir, 'groups.csv')
  badCfg.training.split_group_column = 'no_such_column'
  badCfg.training.epochs = 1
  writeFileSync(join(badDir, 'run.json'), JSON.stringify(badCfg, null, 2))
  copyFileSync(template, join(badDir, 'train.py'))
  try {
    execSync('python -u train.py', { cwd: badDir, stdio: 'pipe' })
    check('grouped split with a missing column fails loudly', false, 'exit code 0')
  } catch (e) {
    const stderr = (e as { stderr?: Buffer }).stderr?.toString() ?? ''
    const events = readFileSync(join(badDir, 'events.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l))
    const failed = events.find((ev) => ev.kind === 'run.failed')
    const badMetrics = JSON.parse(readFileSync(join(badDir, 'metrics.json'), 'utf8'))
    check('grouped split with a missing column fails loudly',
      failed?.stage === 'split' && String(failed?.error).includes('no_such_column')
      && badMetrics.status === 'failed' && !events.some((ev) => ev.kind === 'epoch.end'),
      JSON.stringify({ failed, stderr: stderr.slice(0, 160) }))
  }

  // (c) pure AUROC check against a hand-computed example (import the trainer).
  const pureDir = mkdtempSync(join(tmpdir(), 'spinoml-auc-'))
  copyFileSync(template, join(pureDir, 'train.py'))
  writeFileSync(join(pureDir, 'auroc_check.py'),
    'import importlib.util, json\nimport torch\n' +
    'spec = importlib.util.spec_from_file_location("trainer", "train.py")\n' +
    'trainer = importlib.util.module_from_spec(spec)\n' +
    'spec.loader.exec_module(trainer)\n' +
    'scores = torch.tensor([0.1, 0.4, 0.35, 0.8])\n' +
    'labels = torch.tensor([0, 0, 1, 1])\n' +
    'logits = torch.log(scores / (1 - scores))\n' +
    'print(json.dumps(trainer.compute_metrics("binary", logits, labels, ["auroc", "auprc", "ef"])))\n' +
    'single = torch.tensor([0.2, 0.9])\n' +
    'print(json.dumps(trainer.compute_metrics("binary", single, torch.tensor([1, 1]), ["auroc"])))\n')
  try {
    const out = execSync('python -u auroc_check.py', { cwd: pureDir, encoding: 'utf8' })
    const [line1, line2] = out.trim().split('\n').map((l) => JSON.parse(l)) as Array<Record<string, number>>
    check('AUROC on the hand-computed example = 0.75', Math.abs((line1.auroc ?? NaN) - 0.75) < 1e-9,
      JSON.stringify(line1))
    check('AUPRC + EF computed on the same example',
      typeof line1.auprc === 'number' && Number.isFinite(line1.auprc)
      && typeof line1.ef === 'number' && Number.isFinite(line1.ef),
      JSON.stringify(line1))
    check('single-class val set omits auroc (no NaN)', Object.keys(line2).length === 0,
      JSON.stringify(line2))
  } catch (e) {
    check('auroc pure check ran', false, (e as { stderr?: Buffer }).stderr?.toString() ?? '')
  }
}

console.log(failures === 0 ? '\n✓ all training-gen checks passed' : `\n✗ ${failures} check(s) failed`)
process.exit(failures === 0 ? 0 : 1)
