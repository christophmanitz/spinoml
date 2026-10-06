#!/usr/bin/env tsx
// Phase 54 — Run manifest.
//
// Runs the REAL trainer (sidecar-torch/training_template.py) on tiny synthetic
// CSVs inside throwaway workspaces (`<tmp>/ws/experiments/runs/<id>/`) and
// asserts the machine-readable `manifest.json`:
//   * git provenance is HONEST (dirty tree / non-repo / no git / detached HEAD)
//     and `reproducible_from_git` is only true for a clean known commit;
//   * the canonical `config_identity_sha256` is stable across run id/label/time/
//     absolute path and changes with any real input (lr, seed, data, model);
//   * content hashes match independent sha256 of the run-dir files;
//   * the lifecycle writes running → done/failed/cancelled/eval-only;
//   * no absolute path / env / hostname leaks, atomic (no `.tmp` leftovers),
//     and a manifest write failure is recorded (`manifest.error`) but never
//     changes the run's outcome.
//
// Regressions (verify:smoke/traingen/checkpoint/failures/cancel/metrics/states/
// reference-train, test:safe-load) are run separately by the acceptance command.

import { execSync, spawn, spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import {
  appendFileSync,
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs'
import { hostname, tmpdir } from 'node:os'
import { basename, join } from 'node:path'

const repoRoot = join(import.meta.dirname, '..')
const template = join(repoRoot, 'sidecar-torch', 'training_template.py')

let failures = 0
function check(name: string, cond: boolean, detail = '') {
  if (cond) console.log(`  ✓ ${name}`)
  else { failures++; console.log(`  ✗ ${name} ${detail}`) }
}

// ── JSON accessors (no `any`) ─────────────────────────────────────────────────
function obj(v: unknown): Record<string, unknown> {
  return v && typeof v === 'object' && !Array.isArray(v) ? (v as Record<string, unknown>) : {}
}
function arr(v: unknown): unknown[] { return Array.isArray(v) ? v : [] }
function str(v: unknown): string | null { return typeof v === 'string' ? v : null }
function notes(m: Record<string, unknown>): string[] {
  return arr(m.notes).filter((x): x is string => typeof x === 'string')
}
function collectStrings(v: unknown, out: string[] = []): string[] {
  if (typeof v === 'string') out.push(v)
  else if (Array.isArray(v)) for (const x of v) collectStrings(x, out)
  else if (v && typeof v === 'object') for (const x of Object.values(v)) collectStrings(x, out)
  return out
}

// ── python / hashing ─────────────────────────────────────────────────────────
const pythonCmd = process.env.PYTHON ?? 'python'
const pythonAbs = ((): string => {
  const r = spawnSync(pythonCmd, ['-c', 'import sys;print(sys.executable)'], { encoding: 'utf8' })
  return (r.stdout ?? '').trim() || pythonCmd
})()

const FP_HEADER = Buffer.from('spinoml-dataset-fp-v1\x00', 'utf8')
function plainSha(text: string): string {
  return createHash('sha256').update(text, 'utf8').digest('hex')
}
function fpFile(path: string): { hash: string; size: number } {
  const buf = readFileSync(path)
  const h = createHash('sha256')
  h.update(FP_HEADER)
  h.update(buf)
  return { hash: h.digest('hex'), size: buf.length }
}

// ── synthetic data + tiny model (same seeded xorshift as verify-smoke) ────────
function csv(rows: number, features: number, seed: number): string {
  let s = seed >>> 0
  function rand() { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; return (s >>> 0) / 0x100000000 }
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

const FEATURES = Array.from({ length: 6 }, (_, i) => `f${i}`)
const MODEL_SPINOML = JSON.stringify({ version: 1, nodes: [], edges: [] })
const MODEL_PY =
  'import torch\nimport torch.nn as nn\n\n' +
  'class Model(nn.Module):\n' +
  '    def __init__(self):\n' +
  '        super().__init__()\n' +
  '        self.fc1 = nn.Linear(6, 16)\n' +
  '        self.act = nn.ReLU()\n' +
  '        self.fc2 = nn.Linear(16, 2)\n' +
  '    def forward(self, x):\n' +
  '        return self.fc2(self.act(self.fc1(x)))\n'

// ── workspace assembly ────────────────────────────────────────────────────────
type Ws = {
  base: string; ws: string; dir: string; id: string; dataPath: string
  fingerprint: { alg: string; mode: string; hash: string; size_bytes: number }
  runJson: Record<string, unknown>
}

function initGit(ws: string, detach: boolean): void {
  const opts = { cwd: ws, stdio: 'pipe' as const }
  execSync('git init -q', opts)
  execSync('git -c user.email=t@t -c user.name=t add tracked.txt', opts)
  execSync('git -c user.email=t@t -c user.name=t commit -q -m init', opts)
  if (detach) execSync('git checkout -q --detach', opts)
}

function makeWorkspace(opts: {
  id: string; label?: string; seed?: number; epochs?: number; lr?: number
  rows?: number; csvSeed?: number; git?: boolean; detach?: boolean
  manifestDir?: boolean; evalOnly?: { checkpointFrom: string } | null
  fpHashOverride?: string; fpMode?: string
}): Ws {
  const base = mkdtempSync(join(tmpdir(), 'spinoml-manifest-'))
  const ws = join(base, 'ws')
  const dir = join(ws, 'experiments', 'runs', opts.id)
  mkdirSync(dir, { recursive: true })
  const dataPath = join(dir, 'data.csv')
  writeFileSync(dataPath, csv(opts.rows ?? 60, FEATURES.length, opts.csvSeed ?? 42))
  const fp = fpFile(dataPath)
  const fingerprint = {
    alg: 'sha256', mode: opts.fpMode ?? 'content',
    hash: opts.fpHashOverride ?? fp.hash, size_bytes: fp.size,
  }
  writeFileSync(join(dir, 'model.py'), MODEL_PY)
  writeFileSync(join(dir, 'model.spinoml'), MODEL_SPINOML)
  copyFileSync(template, join(dir, 'train.py'))
  const runJson: Record<string, unknown> = {
    run_id: opts.id,
    run_label: opts.label ?? opts.id,
    created_at: new Date().toISOString(),
    status: 'queued',
    model_path: `experiments/runs/${opts.id}/model.spinoml`,
    backend: { kind: 'local' },
    dataset: {
      path: dataPath, relpath: 'data.csv', kind: 'tabular',
      feature_columns: FEATURES, target_column: 'label', fingerprint,
    },
    training: {
      epochs: opts.epochs ?? 2, batch_size: 16, val_split: 0.25,
      split_strategy: 'random', seed: opts.seed ?? 42, log_every_n_steps: 1,
      optimizer: { kind: 'Adam', lr: opts.lr ?? 0.01, weight_decay: 0 },
      loss: { kind: 'CrossEntropyLoss' }, scheduler: { kind: 'none' },
      metrics: ['accuracy'], callbacks: [],
    },
    snapshot: {
      version: 1, graph_sha256: plainSha(MODEL_SPINOML), model_py_sha256: plainSha(MODEL_PY),
      preprocessing: [], code_trust: [],
    },
  }
  if (opts.evalOnly) {
    runJson.eval_only = true
    runJson.validate = { checkpoint_from: opts.evalOnly.checkpointFrom }
  }
  writeFileSync(join(dir, 'run.json'), JSON.stringify(runJson, null, 2))
  if (opts.git) {
    writeFileSync(join(ws, 'tracked.txt'), 'tracked v1\n')
    initGit(ws, !!opts.detach)
  }
  if (opts.manifestDir) mkdirSync(join(dir, 'manifest.json'))
  return { base, ws, dir, id: opts.id, dataPath, fingerprint, runJson }
}

// ── running / reading ─────────────────────────────────────────────────────────
function runTrain(dir: string, env?: NodeJS.ProcessEnv): { ok: boolean; stderr: string } {
  const r = spawnSync(pythonCmd, ['-u', 'train.py'], {
    cwd: dir, encoding: 'utf8', env: env ?? process.env, timeout: 180_000, maxBuffer: 32 * 1024 * 1024,
  })
  return { ok: r.status === 0, stderr: (r.stderr ?? '') + (r.stdout ?? '') }
}

function spawnTrainer(dir: string): ReturnType<typeof spawn> {
  return spawn(pythonCmd, ['-u', 'train.py'], { cwd: dir, stdio: 'ignore' })
}

async function waitForEvent(dir: string, needle: string, timeoutMs = 30_000): Promise<boolean> {
  const p = join(dir, 'events.jsonl')
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (existsSync(p) && readFileSync(p, 'utf8').includes(needle)) return true
    await new Promise((r) => setTimeout(r, 10))
  }
  return false
}

function readManifest(dir: string): Record<string, unknown> | null {
  const p = join(dir, 'manifest.json')
  if (!existsSync(p) || !statSync(p).isFile()) return null
  try { return JSON.parse(readFileSync(p, 'utf8')) as Record<string, unknown> } catch { return null }
}
function readEvents(dir: string): Record<string, unknown>[] {
  const p = join(dir, 'events.jsonl')
  if (!existsSync(p)) return []
  return readFileSync(p, 'utf8').trim().split('\n').filter(Boolean)
    .map((l) => { try { return JSON.parse(l) as Record<string, unknown> } catch { return {} } })
}
function readStatus(dir: string): string {
  const p = join(dir, 'status')
  return existsSync(p) ? readFileSync(p, 'utf8').trim() : '<missing>'
}

/** Canonical config-identity hash, computed by importing the real train.py. */
function identityOf(trainPy: string, cfg: Record<string, unknown>): string {
  const cfgPath = join(mkdtempSync(join(tmpdir(), 'spinoml-ident-')), 'cfg.json')
  writeFileSync(cfgPath, JSON.stringify(cfg))
  const script = [
    'import importlib.util, json, sys',
    'spec = importlib.util.spec_from_file_location("trainer", sys.argv[1])',
    'm = importlib.util.module_from_spec(spec)',
    'spec.loader.exec_module(m)',
    'print(m._config_identity(json.load(open(sys.argv[2])))[0])',
  ].join('\n')
  const r = spawnSync(pythonCmd, ['-c', script, trainPy, cfgPath], { encoding: 'utf8' })
  if (r.status !== 0) throw new Error(r.stderr)
  return (r.stdout ?? '').trim()
}

const allDirs: string[] = []
const cleanId = 'clean'

console.log('phase 54: run manifest')

// ── 1. git repo, clean tree ───────────────────────────────────────────────────
console.log('\n[1] git repo, clean tree + hashes + done summary + safety')
const sentinel = 'manifest-secret-9f3a2b'
const clean = makeWorkspace({ id: cleanId, git: true })
allDirs.push(clean.dir)
const r1 = runTrain(clean.dir, { ...process.env, SPINOML_TEST_SENTINEL: sentinel })
check('trainer exits 0', r1.ok, r1.stderr.slice(0, 400))
const m1 = readManifest(clean.dir) ?? {}
const g1 = obj(m1.git)
const head = execSync('git rev-parse HEAD', { cwd: clean.ws }).toString().trim()
check('git.commit == rev-parse HEAD', g1.commit === head, String(g1.commit))
check('git.commit is 40 hex', typeof g1.commit === 'string' && /^[0-9a-f]{40}$/.test(g1.commit as string))
check('git.dirty_tracked == false', g1.dirty_tracked === false)
check('git.untracked_count == 0 (run files excluded)', g1.untracked_count === 0, String(g1.untracked_count))
check('reproducible_from_git == true', m1.reproducible_from_git === true)
check('no uncommitted-changes note when clean', !notes(m1).some((n) => n.includes('uncommitted')))

// ── 6a. software/hardware environment (Phase 62) ─────────────────────────────
console.log('\n[6a] software environment')
const sw1 = obj(m1.software)
check('software.python is a non-empty string', typeof sw1.python === 'string' && (sw1.python as string).length > 0)
check('software.torch is a non-empty string', typeof sw1.torch === 'string' && (sw1.torch as string).length > 0)
check('software.numpy is a non-empty string', typeof sw1.numpy === 'string' && (sw1.numpy as string).length > 0)
check('software.torch_geometric present (version string or explicit null)',
  'torch_geometric' in sw1 && (sw1.torch_geometric === null || typeof sw1.torch_geometric === 'string'))
check('software.os names the platform', typeof sw1.os === 'string' && /linux|darwin|windows/i.test(sw1.os as string), String(sw1.os))
check('software.os contains no hostname', typeof sw1.os === 'string' && !(sw1.os as string).includes(execSync('hostname').toString().trim()))
check('hardware has cpus', typeof obj(m1.hardware).cpus === 'number')

// ── 6. hashes ─────────────────────────────────────────────────────────────────
console.log('\n[6] content hashes')
const h1 = obj(m1.hashes)
check('graph_sha256 == sha256(run-dir model.spinoml)',
  h1.graph_sha256 === plainSha(readFileSync(join(clean.dir, 'model.spinoml'), 'utf8')))
check('model_py_sha256 == sha256(run-dir model.py)',
  h1.model_py_sha256 === plainSha(readFileSync(join(clean.dir, 'model.py'), 'utf8')))
check('train_py_sha256 == sha256(run-dir train.py)',
  h1.train_py_sha256 === plainSha(readFileSync(join(clean.dir, 'train.py'), 'utf8')))
check('dataset_fingerprint == run.json fingerprint',
  h1.dataset_fingerprint === `${clean.fingerprint.alg}:${clean.fingerprint.hash}`,
  String(h1.dataset_fingerprint))

// ── 8a. done lifecycle + summary ──────────────────────────────────────────────
console.log('\n[8a] done lifecycle + summary vs metrics.json')
const met1 = JSON.parse(readFileSync(join(clean.dir, 'metrics.json'), 'utf8')) as Record<string, unknown>
const s1 = obj(m1.summary)
check('status == done', m1.status === 'done', String(m1.status))
check('finished_at set', typeof m1.finished_at === 'string' && (m1.finished_at as string).length > 0)
check('failure is null', m1.failure === null)
check('summary.best_val_loss == metrics.best_val_loss', s1.best_val_loss === met1.best_val_loss,
  `${String(s1.best_val_loss)} vs ${String(met1.best_val_loss)}`)
check('summary.epochs_done == metrics.epochs', s1.epochs_done === met1.epochs,
  `${String(s1.epochs_done)} vs ${String(met1.epochs)}`)
check('summary.n_params == metrics.n_params', s1.n_params === met1.n_params,
  `${String(s1.n_params)} vs ${String(met1.n_params)}`)
check('dtype/device recorded after config.env', str(m1.device) === 'cpu' && str(m1.dtype) === 'fp32',
  `${String(m1.device)}/${String(m1.dtype)}`)
check('identity_fields is a non-empty string list',
  arr(m1.identity_fields).length > 0 && arr(m1.identity_fields).every((x) => typeof x === 'string'))

// ── 9. safety (no paths/env/hostname leaks) ───────────────────────────────────
console.log('\n[9] safety: no absolute paths, env, or hostname')
const text1 = JSON.stringify(m1)
const absExisting = collectStrings(m1).filter((v) => v.startsWith('/') && existsSync(v))
check('no string that is an existing absolute path', absExisting.length === 0, absExisting.join(', '))
check('no sentinel env value', !text1.includes(sentinel))
check('no hostname value', !text1.includes(hostname()))
check('no temp workspace dir name', !text1.includes(basename(clean.base)))
check('dirty_files entries are relative', arr(g1.dirty_files).every((x) => typeof x === 'string' && !x.startsWith('/')))

// ── 2. dirty tracked (modified) + staged-only ─────────────────────────────────
console.log('\n[2] dirty tracked file (modified) + staged-only')
{
  const dirty = makeWorkspace({ id: 'dirty', git: true })
  allDirs.push(dirty.dir)
  appendFileSync(join(dirty.ws, 'tracked.txt'), 'modified\n')
  runTrain(dirty.dir)
  const m = readManifest(dirty.dir) ?? {}
  const g = obj(m.git)
  check('dirty_tracked == true', g.dirty_tracked === true)
  check('tracked.txt listed in dirty_files', arr(g.dirty_files).includes('tracked.txt'), JSON.stringify(g.dirty_files))
  check('reproducible_from_git == false', m.reproducible_from_git === false)
  check('uncommitted-changes note present', notes(m).includes('workspace has uncommitted changes to tracked files'))
}
{
  const staged = makeWorkspace({ id: 'staged', git: true })
  allDirs.push(staged.dir)
  appendFileSync(join(staged.ws, 'tracked.txt'), 'staged-only\n')
  execSync('git add tracked.txt', { cwd: staged.ws, stdio: 'pipe' })
  runTrain(staged.dir)
  const m = readManifest(staged.dir) ?? {}
  const g = obj(m.git)
  check('staged-only change -> dirty_tracked == true', g.dirty_tracked === true)
  check('staged file listed in dirty_files', arr(g.dirty_files).includes('tracked.txt'), JSON.stringify(g.dirty_files))
  check('staged-only -> reproducible_from_git == false', m.reproducible_from_git === false)
}

// ── 3. only an extra untracked file ───────────────────────────────────────────
console.log('\n[3] only an extra untracked file')
{
  const untracked = makeWorkspace({ id: 'untracked', git: true })
  allDirs.push(untracked.dir)
  writeFileSync(join(untracked.ws, 'extra_untracked.txt'), 'x\n')
  runTrain(untracked.dir)
  const m = readManifest(untracked.dir) ?? {}
  const g = obj(m.git)
  check('dirty_tracked == false', g.dirty_tracked === false)
  check('untracked_count == 1', g.untracked_count === 1, String(g.untracked_count))
  check('still reproducible_from_git == true', m.reproducible_from_git === true)
  check('untracked note present',
    notes(m).includes('1 untracked files in the workspace are not part of the commit'), JSON.stringify(notes(m)))
}

// ── 4. not a git repo + PATH without git ──────────────────────────────────────
console.log('\n[4] not a git repository + PATH without git')
{
  const noRepo = makeWorkspace({ id: 'norepo', git: false })
  allDirs.push(noRepo.dir)
  runTrain(noRepo.dir)
  const m = readManifest(noRepo.dir) ?? {}
  const g = obj(m.git)
  check('commit is null', g.commit === null)
  check('git.reason == "not a git repository"', g.reason === 'not a git repository', String(g.reason))
  check('available == false', g.available === false)
  check('reproducible_from_git == false', m.reproducible_from_git === false)
  check('non-repo note present',
    notes(m).includes('workspace is not a git repository: reproducibility from git is not claimed'))
}
{
  const noGit = makeWorkspace({ id: 'nogit', git: false })
  allDirs.push(noGit.dir)
  const bin = mkdtempSync(join(tmpdir(), 'spinoml-nogit-bin-'))
  symlinkSync(pythonAbs, join(bin, 'python'))
  symlinkSync(pythonAbs, join(bin, 'python3'))
  const r = spawnSync(pythonAbs, ['-u', 'train.py'], {
    cwd: noGit.dir, encoding: 'utf8', env: { ...process.env, PATH: bin }, timeout: 180_000,
  })
  check('trainer still exits 0 without git on PATH', r.status === 0, (r.stderr ?? '').slice(0, 300))
  const m = readManifest(noGit.dir) ?? {}
  const g = obj(m.git)
  check('available == false', g.available === false)
  check('reason == "git executable not found"', g.reason === 'git executable not found', String(g.reason))
  check('commit is null', g.commit === null)
}

// ── 5. detached HEAD ──────────────────────────────────────────────────────────
console.log('\n[5] detached HEAD')
{
  const det = makeWorkspace({ id: 'detached', git: true, detach: true })
  allDirs.push(det.dir)
  runTrain(det.dir)
  const m = readManifest(det.dir) ?? {}
  const g = obj(m.git)
  check('branch is null', g.branch === null, String(g.branch))
  check('commit still set (40 hex)', typeof g.commit === 'string' && /^[0-9a-f]{40}$/.test(g.commit as string))
}

// ── 7. config identity: stable + sensitive ────────────────────────────────────
console.log('\n[7] config identity hash')
{
  const X = makeWorkspace({ id: 'ident-a', git: false, label: 'Run A' })
  const Y = makeWorkspace({ id: 'ident-b', git: false, label: 'Run B' })
  allDirs.push(X.dir, Y.dir)
  runTrain(X.dir)
  runTrain(Y.dir)
  const iX = str(obj((readManifest(X.dir) ?? {}).hashes).config_identity_sha256)
  const iY = str(obj((readManifest(Y.dir) ?? {}).hashes).config_identity_sha256)
  check('two runs differing in id/label/time/abs-path -> identical identity',
    !!iX && iX === iY, `${String(iX)} vs ${String(iY)}`)
  const baseCfg = JSON.parse(readFileSync(join(X.dir, 'run.json'), 'utf8')) as Record<string, unknown>
  const baseId = identityOf(join(X.dir, 'train.py'), baseCfg)
  check('module identity == manifest identity', baseId === iX, `${baseId} vs ${String(iX)}`)
  const clone = () => JSON.parse(JSON.stringify(baseCfg)) as Record<string, unknown>
  const lrCfg = clone(); obj(obj(lrCfg.training).optimizer).lr = 0.02
  const seedCfg = clone(); obj(seedCfg.training).seed = 43
  const fpCfg = clone(); obj(obj(fpCfg.dataset).fingerprint).hash = 'deadbeef'.repeat(8)
  const graphCfg = clone(); obj(graphCfg.snapshot).graph_sha256 = 'a'.repeat(64)
  check('changing lr -> different identity', identityOf(join(X.dir, 'train.py'), lrCfg) !== baseId)
  check('changing seed -> different identity', identityOf(join(X.dir, 'train.py'), seedCfg) !== baseId)
  check('changing dataset fingerprint -> different identity', identityOf(join(X.dir, 'train.py'), fpCfg) !== baseId)
  check('changing model graph hash -> different identity', identityOf(join(X.dir, 'train.py'), graphCfg) !== baseId)
}

// ── 8b. failed run ────────────────────────────────────────────────────────────
console.log('\n[8b] failed run writes a failed manifest')
{
  const failed = makeWorkspace({ id: 'failed', git: false })
  allDirs.push(failed.dir)
  rmSync(failed.dataPath, { force: true })
  const rf = runTrain(failed.dir)
  check('trainer exits non-zero', !rf.ok)
  const m = readManifest(failed.dir) ?? {}
  const f = obj(m.failure)
  check('status == failed', m.status === 'failed', String(m.status))
  check('failure.stage == dataset', f.stage === 'dataset', String(f.stage))
  check('failure.message non-empty', typeof f.message === 'string' && (f.message as string).length > 0)
  check('manifest still has git/hashes/seed',
    typeof obj(m.git).commit !== 'undefined' &&
    typeof obj(m.hashes).config_identity_sha256 === 'string' && m.seed === 42)
  const badPaths = collectStrings(m).filter((v) => v.startsWith('/') && existsSync(v))
  check('failed manifest leaks no existing absolute path', badPaths.length === 0, badPaths.join(', '))
}

// ── 8c. cancelled run ─────────────────────────────────────────────────────────
console.log('\n[8c] cancelled run writes a cancelled manifest')
{
  const cancelled = makeWorkspace({ id: 'cancelled', git: false, rows: 2000, epochs: 200 })
  allDirs.push(cancelled.dir)
  const child = spawnTrainer(cancelled.dir)
  const seen = await waitForEvent(cancelled.dir, '"kind": "epoch.start", "epoch": 0')
  check('observed epoch 0 mid-flight', seen)
  child.kill('SIGTERM')
  await new Promise<void>((resolve) => child.on('close', () => resolve()))
  check('status == cancelled', readStatus(cancelled.dir) === 'cancelled', readStatus(cancelled.dir))
  const m = readManifest(cancelled.dir) ?? {}
  check('manifest.status == cancelled', m.status === 'cancelled', String(m.status))
  check('manifest.finished_at set on cancel', typeof m.finished_at === 'string' && (m.finished_at as string).length > 0)
}

// ── 8d. eval-only run ─────────────────────────────────────────────────────────
console.log('\n[8d] eval-only run produces a manifest')
{
  const evalOnly = makeWorkspace({
    id: 'eval-only', git: false, evalOnly: { checkpointFrom: join(clean.dir, 'checkpoints', 'best.pt') },
  })
  allDirs.push(evalOnly.dir)
  const re = runTrain(evalOnly.dir)
  check('eval-only trainer exits 0', re.ok, re.stderr.slice(0, 400))
  const m = readManifest(evalOnly.dir) ?? {}
  check('eval-only manifest status == done', m.status === 'done', String(m.status))
  check('eval-only manifest has a summary', obj(m.summary).n_params !== undefined)
}

// ── 10. manifest write failure is recorded, never fatal ───────────────────────
console.log('\n[10] manifest.json replaced by a directory -> manifest.error, run still done')
{
  const bad = makeWorkspace({ id: 'badmanifest', git: false, manifestDir: true })
  allDirs.push(bad.dir)
  const rb = runTrain(bad.dir)
  check('training still finishes done', rb.ok, rb.stderr.slice(0, 300))
  check('status == done', readStatus(bad.dir) === 'done', readStatus(bad.dir))
  check('manifest.error event present', readEvents(bad.dir).some((e) => e.kind === 'manifest.error'))
}

// ── 11. atomicity: no temp leftovers ──────────────────────────────────────────
console.log('\n[11] atomicity: no manifest.json.tmp leftovers')
{
  const leftovers: string[] = []
  for (const dir of allDirs) {
    if (!existsSync(dir)) continue
    for (const name of readdirSync(dir)) {
      if (name.startsWith('manifest.json.tmp')) leftovers.push(join(dir, name))
    }
  }
  check('no manifest.json.tmp* leftovers', leftovers.length === 0, leftovers.join(', '))
}

console.log(failures === 0 ? '\n✓ all manifest checks passed' : `\n✗ ${failures} check(s) failed`)
process.exit(failures === 0 ? 0 : 1)
