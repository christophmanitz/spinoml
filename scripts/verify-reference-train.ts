#!/usr/bin/env tsx
// Phase 53 — End-to-end reference-experiment training.
//
// Runs the THREE committed reference graphs (mlp, cnn, multi-input) through the
// REAL trainer (sidecar-torch/training_template.py) on synthetic seeded data
// built here, and verifies the complete artifact set, loss behaviour, CPU
// reproducibility and checkpoint <-> generated-model consistency.
//
// Model-level equivalence (graph -> generated Model vs hand-written reference)
// is covered by `npm run verify:reference`; this harness reuses those committed
// fixtures verbatim (examples/reference-experiments/<name>/model.spinoml) and
// never changes them.

import { execSync, spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import {
  copyFileSync,
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { generateFromSnapshot } from '../src/codegen/generator'
import { parseFile } from '../src/persistence/file'
import { buildRunSnapshot } from '../src/training/snapshot'

// ─── paths ───────────────────────────────────────────────────────────────────

const HERE = dirname(fileURLToPath(import.meta.url))
const REPO_ROOT = join(HERE, '..')
const FIXTURES_DIR = join(REPO_ROOT, 'examples', 'reference-experiments')
const TEMPLATE = join(REPO_ROOT, 'sidecar-torch', 'training_template.py')
const CHECK_PY = join(HERE, 'lib', 'reference_train_check.py')
const FP_HEADER = Buffer.from('spinoml-dataset-fp-v1\x00', 'utf8')

type ExpName = 'mlp' | 'cnn' | 'multi-input'
type Kind = 'tabular' | 'manifest'
const EXP_NAMES: ExpName[] = ['mlp', 'cnn', 'multi-input']

const EXPECTED_PARAMS: Record<ExpName, number> = { mlp: 210, cnn: 170, 'multi-input': 130 }
const EPOCHS: Record<ExpName, number> = { mlp: 4, cnn: 6, 'multi-input': 6 }

// ─── tiny seeded PRNG (same xorshift32/Box-Muller as verify-smoke) ────────────

function makeRng(seed: number) {
  let s = seed >>> 0
  function rand(): number {
    s ^= s << 13
    s ^= s >>> 17
    s ^= s << 5
    return (s >>> 0) / 0x100000000
  }
  function randn(): number {
    const u1 = rand() + 1e-12
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * rand())
  }
  return { rand, randn }
}

// ─── synthetic data ──────────────────────────────────────────────────────────

type Dataset = { csv: string; columns: string[]; target: string }

/** A. MLP: f0..f4 carry mean +mu for class 1, f5..f9 carry -mu (and vice versa). */
function mlpData(dataSeed: number): Dataset {
  const { randn } = makeRng(dataSeed)
  const columns = Array.from({ length: 10 }, (_, i) => `f${i}`)
  const lines = [[...columns, 'y'].join(',')]
  for (let i = 0; i < 200; i++) {
    const label = i % 2
    const mu = label === 1 ? 0.6 : -0.6
    const vals = columns.map((_, j) => {
      const mean = j < 5 ? mu : -mu
      return (mean + randn() * 0.9).toFixed(6)
    })
    lines.push([...vals, String(label)].join(','))
  }
  return { csv: lines.join('\n') + '\n', columns, target: 'y' }
}

/** B. CNN: an 8x8 image, class 1 has a bright 4x4 block in the top-left quadrant,
 *  class 0 in the bottom-right, background N(0,0.3). */
function cnnData(dataSeed: number): Dataset {
  const { randn } = makeRng(dataSeed)
  const columns = Array.from({ length: 64 }, (_, i) => `p${i}`)
  const lines = [[...columns, 'y'].join(',')]
  for (let i = 0; i < 200; i++) {
    const label = i % 2
    const vals: number[] = []
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        let v = randn() * 0.3
        const inTopLeft = label === 1 && r < 4 && c < 4
        const inBottomRight = label === 0 && r >= 4 && c >= 4
        if (inTopLeft || inBottomRight) v += 1.5
        vals.push(v)
      }
    }
    lines.push([...vals.map((v) => v.toFixed(6)), String(label)].join(','))
  }
  return { csv: lines.join('\n') + '\n', columns, target: 'y' }
}

/** C. Multi-input: a [6] carries the signal (class 1 → mean +0.5), b [4] is noise. */
function multiInputData(dataSeed: number): Dataset {
  const { randn } = makeRng(dataSeed)
  const columns = [
    ...Array.from({ length: 6 }, (_, i) => `a${i}`),
    ...Array.from({ length: 4 }, (_, i) => `b${i}`),
  ]
  const header = ['id', 'apath', 'bpath', ...columns, 'y'].join(',')
  const lines = [header]
  for (let i = 0; i < 200; i++) {
    const label = i % 2
    const mu = label === 1 ? 0.5 : -0.5
    const a = Array.from({ length: 6 }, () => mu + randn() * 0.5)
    const b = Array.from({ length: 4 }, () => randn() * 1.0)
    lines.push([
      String(i),
      `a/${i}.pt`,
      `b/${i}.pt`,
      ...a.map((v) => v.toFixed(6)),
      ...b.map((v) => v.toFixed(6)),
      String(label),
    ].join(','))
  }
  return { csv: lines.join('\n') + '\n', columns, target: 'y' }
}

function datasetFor(name: ExpName, dataSeed: number): Dataset {
  if (name === 'mlp') return mlpData(dataSeed)
  if (name === 'cnn') return cnnData(dataSeed)
  return multiInputData(dataSeed)
}

// ─── hashing / fingerprints ───────────────────────────────────────────────────

function plainSha256(text: string): string {
  return createHash('sha256').update(text, 'utf8').digest('hex')
}

function fingerprintedFile(path: string): { hash: string; size: number } {
  const buf = readFileSync(path)
  const h = createHash('sha256')
  h.update(FP_HEADER)
  h.update(buf)
  return { hash: h.digest('hex'), size: buf.length }
}

type Fingerprint = { alg: 'sha256'; mode: string; hash: string; size_bytes: number; n_files?: number }

function tabularFingerprint(path: string): Fingerprint {
  const { hash, size } = fingerprintedFile(path)
  return { alg: 'sha256', mode: 'content', hash, size_bytes: size }
}

function manifestFingerprint(manifestPath: string, tablePath: string): Fingerprint {
  const main = fingerprintedFile(manifestPath)
  const table = fingerprintedFile(tablePath)
  const h = createHash('sha256')
  h.update(FP_HEADER)
  for (const [nm, part] of [['manifest', main], ['table', table]] as const) {
    h.update(`${nm}:${part.size}:${part.hash}\n`, 'utf8')
  }
  return {
    alg: 'sha256',
    mode: 'config+content',
    hash: h.digest('hex'),
    size_bytes: main.size + table.size,
    n_files: 2,
  }
}

// ─── result tracking ─────────────────────────────────────────────────────────

type Cat = 'trained' | 'artifacts' | 'ckpt' | 'rerun' | 'otherseed' | 'devdtype'
const CATS: Cat[] = ['trained', 'artifacts', 'ckpt', 'rerun', 'otherseed', 'devdtype']
const CAT_LABEL: Record<Cat, string> = {
  trained: 'trained',
  artifacts: 'artifacts',
  ckpt: 'checkpoint<->model',
  rerun: 'rerun identical',
  otherseed: 'other-seed differs',
  devdtype: 'device/dtype',
}

let failures = 0

class ExpResult {
  readonly name: ExpName
  readonly status: Record<Cat, 'PASS' | 'FAIL' | 'SKIPPED'> = {
    trained: 'PASS',
    artifacts: 'PASS',
    ckpt: 'PASS',
    rerun: 'PASS',
    otherseed: 'PASS',
    devdtype: 'PASS',
  }
  readonly errors: string[] = []
  constructor(name: ExpName) {
    this.name = name
  }
  check(cat: Cat, label: string, cond: boolean, detail = ''): boolean {
    if (cond) {
      console.log(`  ✓ ${label}`)
      return true
    }
    this.status[cat] = 'FAIL'
    failures++
    const msg = `${this.name}/${cat}: ${label}${detail ? ` — ${detail}` : ''}`
    this.errors.push(msg)
    console.log(`  ✗ ${label} ${detail}`)
    return false
  }
}

// ─── JSON helpers (no `any`) ─────────────────────────────────────────────────

type Json = Record<string, unknown>
function asObj(v: unknown): Json | null {
  return typeof v === 'object' && v !== null && !Array.isArray(v) ? (v as Json) : null
}
function num(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}
function str(v: unknown): string | null {
  return typeof v === 'string' ? v : null
}

// ─── run-dir assembly + training ──────────────────────────────────────────────

type RunSpec = {
  name: ExpName
  kind: Kind
  dir: string
  columns: string[] | null
  target: string
}

function assembleRunDir(
  name: ExpName,
  kind: Kind,
  parent: string,
  dataset: Dataset,
): RunSpec {
  const dir = mkdtempSync(join(parent, `spinoml-ref-${name}-`))
  const fixtureText = readFileSync(join(FIXTURES_DIR, name, 'model.spinoml'), 'utf8')
  const modelPy = generateFromSnapshot(parseFile(fixtureText)).code

  // dataset file(s)
  let datasetAbs: string
  if (kind === 'tabular') {
    datasetAbs = join(dir, 'data.csv')
    writeFileSync(datasetAbs, dataset.csv)
  } else {
    datasetAbs = join(dir, 'data.manifest')
    const tableAbs = join(dir, 'data.csv')
    writeFileSync(tableAbs, dataset.csv)
    const manifest = {
      table: 'data.csv',
      pairs: {
        a: { column: 'apath' },
        b: { column: 'bpath' },
      },
      target: { column: 'y', type: 'classification' },
      cache: false,
    }
    writeFileSync(datasetAbs, JSON.stringify(manifest, null, 2))
    writeBranchTensors(tableAbs, dir, dataset.columns)
  }

  writeFileSync(join(dir, 'model.spinoml'), fixtureText)
  writeFileSync(join(dir, 'model.py'), modelPy)
  copyFileSync(TEMPLATE, join(dir, 'train.py'))

  return { name, kind, dir, columns: dataset.columns, target: dataset.target }
}

/** Write a/<row>.pt and b/<row>.pt from the multi-input CSV (needs torch, so we
 *  shell out to the same interpreter the trainer uses). */
function writeBranchTensors(csvPath: string, baseDir: string, columns: string[]): void {
  const nA = columns.filter((c) => c.startsWith('a')).length
  const nB = columns.filter((c) => c.startsWith('b')).length
  const script = [
    'import sys, os, pandas as pd, torch',
    'csv, base, na, nb = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])',
    'df = pd.read_csv(csv)',
    'for name, n in (("a", na), ("b", nb)):',
    '    d = os.path.join(base, name); os.makedirs(d, exist_ok=True)',
    '    cols = [f"{name}{j}" for j in range(n)]',
    '    for _, r in df.iterrows():',
    '        t = torch.tensor([float(r[c]) for c in cols], dtype=torch.float32)',
    '        torch.save(t, os.path.join(d, f"{int(r[\'id\'])}.pt"))',
  ].join('\n')
  const r = spawnSync('python', ['-c', script, csvPath, baseDir, String(nA), String(nB)], { encoding: 'utf8' })
  if (r.status !== 0) {
    throw new Error(`failed to write manifest branch tensors: ${(r.stderr ?? '').slice(0, 400)}`)
  }
}

type Training = {
  epochs: number
  batch_size: number
  val_split: number
  split_strategy: string
  seed: number
  log_every_n_steps: number
  optimizer: { kind: string; lr: number; weight_decay: number }
  loss: { kind: string }
  scheduler: { kind: string }
  metrics: string[]
  callbacks: unknown[]
}

function trainingConfig(name: ExpName, seed: number): Training {
  return {
    epochs: EPOCHS[name],
    batch_size: 32,
    val_split: 0.2,
    split_strategy: 'random',
    seed,
    log_every_n_steps: 1,
    optimizer: { kind: 'Adam', lr: 0.01, weight_decay: 0 },
    loss: { kind: 'CrossEntropyLoss' },
    scheduler: { kind: 'none' },
    metrics: ['accuracy'],
    callbacks: [],
  }
}

async function writeRunJson(spec: RunSpec, fingerprint: Fingerprint, training: Training): Promise<{ modelPy: string; modelSpinoml: string }> {
  const modelSpinoml = readFileSync(join(spec.dir, 'model.spinoml'), 'utf8')
  const modelPy = readFileSync(join(spec.dir, 'model.py'), 'utf8')
  const snapshot = await buildRunSnapshot(modelSpinoml, modelPy)
  const runJson = {
    run_id: `ref-${spec.name}`,
    run_label: `reference ${spec.name}`,
    created_at: '2026-01-01T00:00:00Z',
    status: 'queued',
    model_path: `experiments/runs/ref-${spec.name}/model.spinoml`,
    backend: { kind: 'local' },
    dataset: {
      path: join(spec.dir, spec.kind === 'manifest' ? 'data.manifest' : 'data.csv'),
      relpath: spec.kind === 'manifest' ? 'data.manifest' : 'data.csv',
      kind: spec.kind,
      feature_columns: spec.kind === 'manifest' ? null : spec.columns,
      target_column: spec.target,
      fingerprint,
    },
    training,
    snapshot,
  }
  writeFileSync(join(spec.dir, 'run.json'), JSON.stringify(runJson, null, 2))
  return { modelPy, modelSpinoml }
}

function fingerprintOf(spec: RunSpec): Fingerprint {
  if (spec.kind === 'tabular') return tabularFingerprint(join(spec.dir, 'data.csv'))
  return manifestFingerprint(join(spec.dir, 'data.manifest'), join(spec.dir, 'data.csv'))
}

type TrainOutcome = { ok: boolean; detail: string }

function runTrain(dir: string): TrainOutcome {
  try {
    execSync('python -u train.py > stdout.log 2> stderr.log', {
      cwd: dir,
      shell: '/bin/bash',
      timeout: 180_000,
      stdio: 'ignore',
    })
    return { ok: true, detail: '' }
  } catch (e: unknown) {
    const base = e instanceof Error ? e.message : String(e)
    let detail = base
    try {
      const errLog = existsSync(join(dir, 'stderr.log')) ? readFileSync(join(dir, 'stderr.log'), 'utf8') : ''
      const outLog = existsSync(join(dir, 'stdout.log')) ? readFileSync(join(dir, 'stdout.log'), 'utf8') : ''
      const combined = `${errLog}${outLog}`.trim()
      if (combined) detail = combined.slice(0, 600)
    } catch (readErr: unknown) {
      detail = `${base} (log read failed: ${readErr instanceof Error ? readErr.message : String(readErr)})`
    }
    return { ok: false, detail }
  }
}

function parseEvents(dir: string): Json[] {
  const raw = readFileSync(join(dir, 'events.jsonl'), 'utf8').trim()
  if (!raw) return []
  return raw.split('\n').map((l) => JSON.parse(l) as Json)
}

function epochSeries(dir: string): { train: number[]; val: number[] } {
  const events = parseEvents(dir)
  const ends = events.filter((e) => e.kind === 'epoch.end')
  return {
    train: ends.map((e) => num(e.train_loss) ?? Number.NaN),
    val: ends.map((e) => num(e.val_loss) ?? Number.NaN),
  }
}

// ─── python helper ────────────────────────────────────────────────────────────

type HelperResult = {
  ok?: boolean
  error?: string
  keys_best_ok?: boolean
  keys_last_ok?: boolean
  strict_load_ok?: boolean
  strict_load_error?: string
  n_params?: number
  best_val_loss?: number
  re_val_loss?: number
  val_loss_match?: boolean
  ckpt_vs_metrics_match?: boolean
  env_device?: string
  env_dtype?: string
  param_dtypes?: string[]
  dtype_match?: boolean
  skipped?: number
  outputs_finite?: boolean
}

function runCheck(dir: string, kind: Kind): HelperResult {
  const r = spawnSync('python', [CHECK_PY, dir, kind], { encoding: 'utf8', maxBuffer: 32 * 1024 * 1024 })
  if (r.status !== 0) {
    return { ok: false, error: `helper exited ${r.status}: ${(r.stderr ?? '').slice(0, 400)}` }
  }
  try {
    const parsed = JSON.parse((r.stdout ?? '').trim()) as Record<string, unknown>
    return parsed as HelperResult
  } catch (e: unknown) {
    return { ok: false, error: `helper did not print JSON: ${e instanceof Error ? e.message : String(e)}` }
  }
}

// ─── per-experiment checks ────────────────────────────────────────────────────

function maxAbsDiff(a: number[], b: number[]): number {
  if (a.length !== b.length) return Number.POSITIVE_INFINITY
  let m = 0
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]))
  return m
}

type BaseInfo = { bestValLoss: number | null; device: string | null }

async function checkExperiment(name: ExpName, parent: string, exp: ExpResult, minAcc: boolean): Promise<BaseInfo> {
  const kind: Kind = name === 'multi-input' ? 'manifest' : 'tabular'
  const data = datasetFor(name, 7000 + EXP_NAMES.indexOf(name) * 100)
  const baseSeed = 11
  const epochs = EPOCHS[name]

  console.log(`\n[${name}] building run dir + training (${epochs} epochs)`)
  const spec = assembleRunDir(name, kind, parent, data)
  const training = trainingConfig(name, baseSeed)
  const fingerprint = fingerprintOf(spec)
  const { modelPy, modelSpinoml } = await writeRunJson(spec, fingerprint, training)

  const outcome = runTrain(spec.dir)
  if (!exp.check('trained', 'trainer exited 0', outcome.ok, outcome.detail)) {
    return { bestValLoss: null, device: null }
  }

  // ── 1. loss behaviour + metrics ──
  const metrics = JSON.parse(readFileSync(join(spec.dir, 'metrics.json'), 'utf8')) as Json
  const bestValLoss = num(metrics.best_val_loss)
  const device = str(asObj(metrics.env)?.device) ?? null
  const status = readFileSync(join(spec.dir, 'status'), 'utf8').trim()
  exp.check('trained', `metrics.json status == "done" (${String(metrics.status)})`, metrics.status === 'done')
  exp.check('trained', `status file == "done" (${status})`, status === 'done')

  const events = parseEvents(spec.dir)
  const ends = events.filter((e) => e.kind === 'epoch.end')
  exp.check('trained', `exactly ${epochs} epoch.end events (got ${ends.length})`, ends.length === epochs)
  const trainLosses = ends.map((e) => num(e.train_loss))
  const valLosses = ends.map((e) => num(e.val_loss))
  const valAccs = ends.map((e) => num(e.val_acc))
  exp.check('trained', 'all train_loss finite', trainLosses.every((v) => v !== null))
  exp.check('trained', 'all val_loss finite', valLosses.every((v) => v !== null))
  exp.check('trained', 'all val_acc finite and in [0,1]', valAccs.every((v) => v !== null && v >= 0 && v <= 1))
  const firstTrain = trainLosses[0]
  const lastTrain = trainLosses[trainLosses.length - 1]
  exp.check(
    'trained',
    `final train_loss < first (${String(lastTrain)} < ${String(firstTrain)})`,
    firstTrain !== null && lastTrain !== null && lastTrain < firstTrain,
  )
  const lastAcc = valAccs[valAccs.length - 1]
  if (minAcc) {
    exp.check('trained', `final val_acc >= 0.75 (${String(lastAcc)})`, lastAcc !== null && lastAcc >= 0.75)
  }

  // ── 2. artifact set ──
  const fixtureText = readFileSync(join(FIXTURES_DIR, name, 'model.spinoml'), 'utf8')
  const runDirModelSpinoml = readFileSync(join(spec.dir, 'model.spinoml'), 'utf8')
  const runDirModelPy = readFileSync(join(spec.dir, 'model.py'), 'utf8')
  exp.check('artifacts', 'model.spinoml byte-equal to committed fixture', runDirModelSpinoml === fixtureText)
  exp.check('artifacts', 'model.py byte-equal to freshly generated code', runDirModelPy === modelPy)
  exp.check('artifacts', 'train.py present', existsSync(join(spec.dir, 'train.py')))
  exp.check('artifacts', 'stdout.log present', existsSync(join(spec.dir, 'stdout.log')))
  exp.check('artifacts', 'stderr.log present', existsSync(join(spec.dir, 'stderr.log')))

  const runConfig = JSON.parse(readFileSync(join(spec.dir, 'run.json'), 'utf8')) as Json
  const dsCfg = asObj(runConfig.dataset)
  const trainCfg = asObj(runConfig.training)
  const snap = asObj(runConfig.snapshot)
  const fpObj = asObj(dsCfg?.fingerprint)
  exp.check('artifacts', 'run.json dataset has a fingerprint', !!fpObj && typeof fpObj.hash === 'string')
  exp.check('artifacts', 'run.json training carries the seed', num(trainCfg?.seed) === baseSeed)
  const snapGraph = str(snap?.graph_sha256)
  const snapModel = str(snap?.model_py_sha256)
  exp.check(
    'artifacts',
    'snapshot.graph_sha256 matches independent sha256(model.spinoml)',
    snapGraph === plainSha256(modelSpinoml),
    `${String(snapGraph)}`,
  )
  exp.check(
    'artifacts',
    'snapshot.model_py_sha256 matches independent sha256(model.py)',
    snapModel === plainSha256(modelPy),
    `${String(snapModel)}`,
  )
  exp.check('artifacts', `metrics.json n_params == ${EXPECTED_PARAMS[name]}`, num(metrics.n_params) === EXPECTED_PARAMS[name], String(metrics.n_params))

  const envEv = events.find((e) => e.kind === 'config.env')
  exp.check(
    'artifacts',
    'events config.env has python/torch/device/dtype strings',
    !!envEv && typeof envEv.python === 'string' && typeof envEv.torch === 'string' && typeof envEv.device === 'string' && typeof envEv.dtype === 'string',
  )
  exp.check('artifacts', 'events run.provenance present', events.some((e) => e.kind === 'run.provenance'))
  exp.check('artifacts', 'events run.determinism present', events.some((e) => e.kind === 'run.determinism'))
  exp.check('artifacts', 'events run.snapshot present', events.some((e) => e.kind === 'run.snapshot'))

  // ── 3. checkpoint <-> model ──
  const helper = runCheck(spec.dir, kind)
  if (!exp.check('ckpt', 'python helper succeeded', helper.ok === true, helper.error ?? '')) {
    // continue: report the remaining helper checks as failed
  }
  exp.check('ckpt', 'best.pt has required keys', helper.keys_best_ok === true)
  exp.check('ckpt', 'last.pt has required keys', helper.keys_last_ok === true)
  exp.check('ckpt', 'model_state loads with strict=True', helper.strict_load_ok === true, helper.strict_load_error ?? '')
  exp.check('ckpt', 're-evaluated model outputs are finite', helper.outputs_finite === true)
  exp.check('ckpt', `helper n_params == ${EXPECTED_PARAMS[name]}`, helper.n_params === EXPECTED_PARAMS[name], String(helper.n_params))
  exp.check(
    'ckpt',
    're-evaluated val loss == trainer best val loss (1e-5)',
    helper.val_loss_match === true,
    `re=${String(helper.re_val_loss)} best=${String(helper.best_val_loss)}`,
  )
  exp.check('ckpt', 'checkpoint val_loss == metrics best_val_loss', helper.ckpt_vs_metrics_match === true, `${String(helper.re_val_loss)}`)

  // ── 5. device / dtype ──
  exp.check('devdtype', 'config.env.device == "cpu"', helper.env_device === 'cpu', String(helper.env_device))
  exp.check(
    'devdtype',
    'config.env.dtype matches checkpointed float params',
    helper.dtype_match === true,
    `env=${String(helper.env_dtype)} params=${JSON.stringify(helper.param_dtypes ?? [])}`,
  )

  // ── 4. reproducibility ──
  console.log(`[${name}] reproducibility: rerun (same seed) + other seed`)
  const spec2 = assembleRunDir(name, kind, parent, data)
  await writeRunJson(spec2, fingerprintOf(spec2), trainingConfig(name, baseSeed))
  const out2 = runTrain(spec2.dir)
  if (exp.check('rerun', 'second run (same seed) exited 0', out2.ok, out2.detail)) {
    const s1 = epochSeries(spec.dir)
    const s2 = epochSeries(spec2.dir)
    const dTrain = maxAbsDiff(s1.train, s2.train)
    const dVal = maxAbsDiff(s1.val, s2.val)
    exp.check('rerun', `per-epoch train_loss equal (<=1e-6, max ${dTrain.toExponential(2)})`, dTrain <= 1e-6)
    exp.check('rerun', `per-epoch val_loss equal (<=1e-6, max ${dVal.toExponential(2)})`, dVal <= 1e-6)
  }

  const otherSeed = baseSeed + 101
  const spec3 = assembleRunDir(name, kind, parent, data)
  await writeRunJson(spec3, fingerprintOf(spec3), trainingConfig(name, otherSeed))
  const out3 = runTrain(spec3.dir)
  if (exp.check('otherseed', 'third run (different seed) exited 0', out3.ok, out3.detail)) {
    const s1 = epochSeries(spec.dir)
    const s3 = epochSeries(spec3.dir)
    const dTrain = maxAbsDiff(s1.train, s3.train)
    const dVal = maxAbsDiff(s1.val, s3.val)
    exp.check('otherseed', `train_loss sequence differs (max ${dTrain.toExponential(2)})`, dTrain > 1e-6)
    exp.check('otherseed', `val_loss sequence differs (max ${dVal.toExponential(2)})`, dVal > 1e-6)
  }
  return { bestValLoss, device }
}

// ─── CUDA (only if available) ─────────────────────────────────────────────────

function cudaAvailable(): boolean {
  const r = spawnSync('python', ['-c', 'import torch; print(torch.cuda.is_available())'], { encoding: 'utf8' })
  return (r.stdout ?? '').trim() === 'True'
}

// ─── summary table ────────────────────────────────────────────────────────────

function printSummary(results: ExpResult[], seconds: number): void {
  const headers = ['experiment', ...CATS.map((c) => CAT_LABEL[c])]
  const cells = results.map((r) => [r.name, ...CATS.map((c) => r.status[c])])
  const widths = headers.map((h, i) => Math.max(h.length, ...cells.map((row) => row[i].length)))
  const line = (vals: string[]) => vals.map((v, i) => v.padEnd(widths[i])).join(' | ')
  console.log('')
  console.log(line(headers))
  console.log(widths.map((w) => '-'.repeat(w)).join('-+-'))
  for (const row of cells) console.log(line(row))
  console.log('')
  console.log(`total runtime: ${seconds.toFixed(1)} s (target < 120 s) — ${seconds < 120 ? 'PASS' : 'FAIL'}`)
}

// ─── main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<number> {
  const t0 = Date.now()
  if (!existsSync(TEMPLATE)) {
    console.log(`✗ training template missing: ${TEMPLATE}`)
    return 1
  }
  const parent = mkdtempSync(join(tmpdir(), 'spinoml-ref-train-'))
  const results: ExpResult[] = []
  const baseInfo: Partial<Record<ExpName, BaseInfo>> = {}
  const cuda = cudaAvailable()
  try {
    for (const name of EXP_NAMES) {
      const exp = new ExpResult(name)
      results.push(exp)
      const minAcc = name === 'mlp' || name === 'cnn'
      try {
        baseInfo[name] = await checkExperiment(name, parent, exp, minAcc)
      } catch (e: unknown) {
        exp.check('trained', 'harness completed without throwing', false, e instanceof Error ? e.message : String(e))
      }
    }

    if (cuda) {
      console.log('\n[CUDA] re-running mlp once more and comparing best val loss (<=1e-3)')
      const mlpData2 = datasetFor('mlp', 7000)
      const spec = assembleRunDir('mlp', 'tabular', parent, mlpData2)
      await writeRunJson(spec, fingerprintOf(spec), trainingConfig('mlp', 11))
      const out = runTrain(spec.dir)
      let cudaBest: number | null = null
      let cudaDevice: string | null = null
      if (out.ok) {
        const m = JSON.parse(readFileSync(join(spec.dir, 'metrics.json'), 'utf8')) as Json
        cudaBest = num(m.best_val_loss)
        cudaDevice = str(asObj(m.env)?.device) ?? null
      }
      const cpuBest = baseInfo.mlp?.bestValLoss ?? null
      const within = out.ok && cudaBest !== null && cpuBest !== null && Math.abs(cudaBest - cpuBest) <= 1e-3
      if (within) {
        console.log(`  ✓ cuda best val loss ${cudaBest} vs cpu ${cpuBest} within 1e-3 (device=${cudaDevice})`)
      } else {
        failures++
        console.log(`  ✗ cuda best val loss comparison failed (cuda=${cudaBest} cpu=${cpuBest}) ${out.detail}`)
      }
    } else {
      console.log('SKIPPED  CUDA — torch.cuda.is_available() is False')
    }
  } finally {
    rmSync(parent, { recursive: true, force: true })
  }

  printSummary(results, (Date.now() - t0) / 1000)
  for (const r of results) {
    for (const err of r.errors) console.log(`  ✗ ${err}`)
  }
  if (failures === 0) {
    console.log('\n✓ all reference-experiment training checks passed')
    return 0
  }
  console.log(`\n✗ ${failures} check(s) failed`)
  return 1
}

const code = await main()
process.exit(code)
