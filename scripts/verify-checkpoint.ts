#!/usr/bin/env tsx
// Phase 26 — Checkpoint correctness.
// Verifies the train → save → stop → load → resume loop preserves the full
// state the auditor listed: model/optimizer/scheduler state, epoch, global
// step, RNG streams (where supported), and the experiment configuration.

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

// ── synthetic data (same generator as verify-smoke) ──
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

function makeRunDir(name: string, opts: { resumeFrom?: string; epochs?: number } = {}) {
  const dir = mkdtempSync(join(tmpdir(), `spinoml-ckpt-${name}-`))
  writeFileSync(join(dir, 'data.csv'), csv(120, 10, SEED))
  writeFileSync(join(dir, 'model.py'), MODEL_PY)
  const runJson: Record<string, unknown> = {
    run_id: 'ckpt', run_label: 'ckpt', created_at: new Date().toISOString(), status: 'queued',
    model_path: 'm', backend: { kind: 'local' },
    dataset: { path: join(dir, 'data.csv'), relpath: 'data.csv', kind: 'tabular',
      feature_columns: FEATURES, target_column: 'label' },
    training: {
      epochs: opts.epochs ?? 6, batch_size: 32, val_split: 0.25, split_strategy: 'random',
      seed: SEED, log_every_n_steps: 1,
      optimizer: { kind: 'Adam', lr: 0.01, weight_decay: 0 },
      loss: { kind: 'CrossEntropyLoss' },
      scheduler: { kind: 'StepLR', step_size: 2, gamma: 0.5 },
      metrics: ['accuracy'], callbacks: [],
    },
  }
  if (opts.resumeFrom) runJson.resume_from = opts.resumeFrom
  writeFileSync(join(dir, 'run.json'), JSON.stringify(runJson, null, 2))
  copyFileSync(template, join(dir, 'train.py'))
  return dir
}

function run(dir: string) {
  try {
    execSync('python -u train.py', { cwd: dir, stdio: 'pipe', timeout: 180_000 })
    return { ok: true }
  } catch (e) {
    const err = e as { stderr?: Buffer }
    return { ok: false, stderr: (err.stderr?.toString() ?? '').slice(0, 400) }
  }
}

function pyEval(script: string, args: string[]): string {
  const r = spawnSync('python', ['-c', script, ...args], { stdio: 'pipe' })
  if (r.status !== 0) throw new Error(r.stderr.toString())
  return r.stdout.toString().trim()
}

function ckptKeys(path: string): string[] {
  return pyEval('import torch, sys\nck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)\nprint(" ".join(sorted(ck.keys())))', [path]).split(' ')
}

function ckptField<T>(path: string, expr: string): T {
  const script = `import torch, sys, json\nck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)\nprint(json.dumps(${expr}))`
  return JSON.parse(pyEval(script, [path])) as T
}

console.log('phase 26: checkpoint correctness')

// ── 1. train → save ──
const dirA = mkdtempSync(join(tmpdir(), 'spinoml-ckpt-a-'))
{
  writeFileSync(join(dirA, 'data.csv'), csv(120, 10, SEED))
  writeFileSync(join(dirA, 'model.py'), MODEL_PY)
  writeFileSync(join(dirA, 'run.json'), JSON.stringify({
    run_id: 'ckpt', run_label: 'ckpt', created_at: new Date().toISOString(), status: 'queued',
    model_path: 'm', backend: { kind: 'local' },
    dataset: { path: join(dirA, 'data.csv'), relpath: 'data.csv', kind: 'tabular',
      feature_columns: FEATURES, target_column: 'label' },
    training: {
      epochs: 6, batch_size: 32, val_split: 0.25, split_strategy: 'random',
      seed: SEED, log_every_n_steps: 1,
      optimizer: { kind: 'Adam', lr: 0.01, weight_decay: 0 },
      loss: { kind: 'CrossEntropyLoss' },
      scheduler: { kind: 'StepLR', step_size: 2, gamma: 0.5 },
      metrics: ['accuracy'], callbacks: [],
    },
  }, null, 2))
  copyFileSync(template, join(dirA, 'train.py'))
  const r = run(dirA)
  check('train run A completes', r.ok, r.stderr ?? '')
  check('best.pt exists', existsSync(join(dirA, 'checkpoints', 'best.pt')))
  check('last.pt exists', existsSync(join(dirA, 'checkpoints', 'last.pt')))
}

const keys = ckptKeys(join(dirA, 'checkpoints', 'last.pt'))
console.log('  [checkpoint preserves full state]')
for (const k of ['model_state', 'optim_state', 'sched_state', 'epoch', 'global_step', 'rng', 'config', 'best_val', 'classes', 'head_classes']) {
  check(`last.pt has ${k}`, keys.includes(k), keys.join(','))
}
{
  const rng = ckptField<Record<string, string>>(join(dirA, 'checkpoints', 'last.pt'),
    '{k: type(v).__name__ for k, v in ck["rng"].items()}')
  check('rng has torch stream', rng.torch === 'Tensor', JSON.stringify(rng ?? {}))
  check('rng has python stream', 'python' in (rng ?? {}))
  check('rng has numpy stream', 'numpy' in (rng ?? {}))
  const cfg = ckptField<Record<string, unknown>>(join(dirA, 'checkpoints', 'last.pt'), 'ck["config"]')
  check('config preserves experiment configuration', (cfg?.training as Record<string, unknown>)?.epochs === 6, JSON.stringify(cfg?.training ?? {}))
  const gs = ckptField<number>(join(dirA, 'checkpoints', 'last.pt'), 'ck["global_step"]')
  check('global_step > 0', gs > 0, String(gs))
  const ep = ckptField<number>(join(dirA, 'checkpoints', 'last.pt'), 'ck["epoch"]')
  check('epoch = 5 (last epoch)', ep === 5, String(ep))
}

// ── 2. stop → resume ──
console.log('  [resume from checkpoint]')
{
  const dirB = makeRunDir('b', { resumeFrom: join(dirA, 'checkpoints', 'last.pt'), epochs: 3 })
  const r = run(dirB)
  check('resume run B completes', r.ok, r.stderr ?? '')
  const events = readFileSync(join(dirB, 'events.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l))
  const resumed = events.find((e) => e.kind === 'run.resumed') as Record<string, unknown> | undefined
  check('run.resumed emitted', !!resumed, JSON.stringify(resumed))
  check('resume start_epoch = 6', resumed?.start_epoch === 6, JSON.stringify(resumed?.start_epoch))
  check('resume global_step carried over', typeof resumed?.global_step === 'number' && (resumed!.global_step as number) > 0)
  const epochStarts = events.filter((e) => e.kind === 'epoch.start').map((e) => e.epoch)
  check('continues at epoch 6,7,8', JSON.stringify(epochStarts) === JSON.stringify([6, 7, 8]), JSON.stringify(epochStarts))
  check('run B done', readFileSync(join(dirB, 'status'), 'utf8').trim() === 'done')
  const ep = ckptField<number>(join(dirB, 'checkpoints', 'last.pt'), 'ck["epoch"]')
  check('run B last.pt epoch = 8', ep === 8, String(ep))
  // scheduler state carried: after 3 more epochs StepLR(step=2) should have stepped once more
  const sched = ckptField<Record<string, unknown>>(join(dirB, 'checkpoints', 'last.pt'), 'ck["sched_state"]')
  check('scheduler state present in resumed run', !!sched && typeof sched === 'object', JSON.stringify(sched ?? 'none'))
}

// ── 3. stop (cancel) leaves a resumable checkpoint ──
console.log('  [cancelled run leaves resumable checkpoint]')
{
  const dirC = makeRunDir('c', { epochs: 30 })
  const child = spawn('python', ['-u', 'train.py'], { cwd: dirC, stdio: 'ignore' })
  // wait for the second epoch to finish, then cancel
  let cancelled = false
  for (let i = 0; i < 3000; i++) {
    await new Promise((r) => setTimeout(r, 10))
    const evPath = join(dirC, 'events.jsonl')
    if (existsSync(evPath) && readFileSync(evPath, 'utf8').includes('"kind": "epoch.end", "epoch": 1')) {
      writeFileSync(join(dirC, 'status'), 'cancelled\n')
      cancelled = true
      break
    }
  }
  check('cancelled mid-run', cancelled)
  await new Promise<void>((resolve) => child.on('close', () => resolve()))
  check('status = cancelled', readFileSync(join(dirC, 'status'), 'utf8').trim() === 'cancelled')
  check('last.pt written on cancel', existsSync(join(dirC, 'checkpoints', 'last.pt')))
  const ep = ckptField<number>(join(dirC, 'checkpoints', 'last.pt'), 'ck["epoch"]')
  check('cancel checkpoint epoch = 1 (last completed)', ep === 1, String(ep))
}

// ── 4. atomic write crash simulation (Phases 27+28) ──
console.log('  [atomic write: crash mid-save leaves the previous valid checkpoint]')
{
  const script = [
    'import sys, importlib.util, os, tempfile',
    'from pathlib import Path',
    'spec = importlib.util.spec_from_file_location("trainer", sys.argv[1])',
    'trainer = importlib.util.module_from_spec(spec)',
    'spec.loader.exec_module(trainer)',
    'import torch',
    'd = Path(tempfile.mkdtemp())',
    'p = d / "best.pt"',
    'obj = {"a": torch.tensor([1.0])}',
    'trainer._atomic_save(obj, p)',
    'print("FIRST_OK", p.exists())',
    'orig = torch.save',
    'def boom(o, f):',
    '    f.write(b"\\x00garbage")  # partial bytes, then die',
    '    raise RuntimeError("simulated crash")',
    'torch.save = boom',
    'try:',
    '    trainer._atomic_save({"b": 2}, p)',
    '    print("NO_EXCEPTION")',
    'except RuntimeError as e:',
    '    print("CRASHED", str(e))',
    'finally:',
    '    torch.save = orig',
    'ck = torch.load(p, map_location="cpu", weights_only=False)',
    'print("SURVIVED", "a" in ck and float(ck["a"][0]) == 1.0)',
    'print("NO_TMP", not (d / "best.pt.tmp").exists())',
  ].join('\n')
  const r = spawnSync('python', ['-c', script, join(dirA, 'train.py')], { stdio: 'pipe' })
  const out = (r.stdout.toString() + r.stderr.toString()).trim()
  check('first atomic save ok', out.includes('FIRST_OK True'), out.slice(0, 300))
  check('crash raised (partial write)', out.includes('CRASHED'), out.slice(0, 300))
  check('previous valid checkpoint survived', out.includes('SURVIVED True'), out.slice(0, 300))
  check('no .tmp leftover', out.includes('NO_TMP True'), out.slice(0, 300))
}

// ── 5. process crash mid-training: the checkpoint on disk is ALWAYS loadable
//    (previous valid or new valid), restart resumes cleanly ──
console.log('  [SIGKILL mid-training: checkpoint loadable, resume works]')
{
  const dirF = makeRunDir('f', { epochs: 30 })
  const child = spawn('python', ['-u', 'train.py'], { cwd: dirF, stdio: 'ignore' })
  // kill at a random-ish point shortly after the second checkpoint save
  let killed = false
  for (let i = 0; i < 4000; i++) {
    await new Promise((r) => setTimeout(r, 5))
    const evPath = join(dirF, 'events.jsonl')
    if (existsSync(evPath) && readFileSync(evPath, 'utf8').includes('"kind": "checkpoint"')) {
      child.kill('SIGKILL')
      killed = true
      break
    }
  }
  check('killed after a checkpoint save', killed)
  await new Promise<void>((resolve) => child.on('close', () => resolve()))
  // whatever .pt files exist must be loadable (atomic write guarantee)
  const ckptDir = join(dirF, 'checkpoints')
  const ptFiles = ['best.pt', 'last.pt'].filter((n) => existsSync(join(ckptDir, n)))
  check('at least one checkpoint exists', ptFiles.length > 0)
  let allLoadable = true
  for (const n of ptFiles) {
    const script = `import torch, sys\ntry:\n    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)\n    print("OK", "model_state" in ck)\nexcept Exception as e:\n    print("CORRUPT", type(e).__name__)`
    const r = spawnSync('python', ['-c', script, join(ckptDir, n)], { stdio: 'pipe' })
    const out = r.stdout.toString().trim()
    if (!out.startsWith('OK True')) { allLoadable = false; check(`checkpoint ${n} loadable`, false, out) }
    else check(`checkpoint ${n} loadable`, true)
  }
  check('all checkpoints loadable after crash', allLoadable)
  check('no .tmp leftover after crash', !existsSync(join(ckptDir, 'best.pt.tmp')) && !existsSync(join(ckptDir, 'last.pt.tmp')))
  // restart: resume from whatever checkpoint exists → clean continuation
  const resumePath = existsSync(join(ckptDir, 'last.pt'))
    ? join(ckptDir, 'last.pt') : join(ckptDir, 'best.pt')
  const dirG = makeRunDir('g', { resumeFrom: resumePath, epochs: 2 })
  const r = run(dirG)
  check('restart after crash resumes cleanly', r.ok, r.stderr ?? '')
  if (r.ok) {
    const events = readFileSync(join(dirG, 'events.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l))
    check('run.resumed emitted after crash restart', events.some((e) => e.kind === 'run.resumed'))
  }
}

// ── 6. corrupted checkpoint is rejected loudly, never silently accepted ──
console.log('  [corrupted checkpoint rejected]')
{
  const dirE = makeRunDir('e', { resumeFrom: join(dirA, 'checkpoints', 'garbage.pt') })
  writeFileSync(join(dirA, 'checkpoints', 'garbage.pt'), '\x00\x01garbage-not-a-torch-file')
  const r = run(dirE)
  check('resume from garbage fails', !r.ok, r.ok ? 'exit 0 on corrupt checkpoint!' : '')
  const events = readFileSync(join(dirE, 'events.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l))
  const failedEv = events.find((e) => e.kind === 'run.failed') as Record<string, unknown> | undefined
  check('run.failed emitted (stage resume)', failedEv?.stage === 'resume', JSON.stringify(failedEv?.stage))
  check('status = failed', readFileSync(join(dirE, 'status'), 'utf8').trim() === 'failed')
}

console.log(failures === 0 ? '\n✓ all checkpoint checks passed' : `\n✗ ${failures} check(s) failed`)
process.exit(failures === 0 ? 0 : 1)
