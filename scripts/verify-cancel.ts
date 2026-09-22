#!/usr/bin/env tsx
// Phase 32 — Cancellation.
// "Cancel before start, during startup, during training, after completion,
// double cancellation — the resulting state must be consistent."
//
// Phase 30 (verify:states) already proved the state MACHINE. Phase 32 proves
// the cancellation PATH: a real SIGTERM/SIGINT delivered to the REAL trainer
// must unwind to the same single, terminal-shielded cancellation point
// (`_finish_cancel` in training_template.py) — graceful exit 0, run.cancelled
// emitted exactly once, a resumable last.pt at the last completed epoch, and a
// late signal after SUCCEEDED must NOT resurrect a done run into cancelled.
// Every cancelled run's events are also re-checked through the Phase-31
// read-boundary vocabulary (first-terminal-wins, no stale trailing events).

import { spawn, spawnSync } from 'node:child_process'
import { writeFileSync, mkdtempSync, readFileSync, existsSync, copyFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { finalTerminal, hasStaleTrailing, parseFinalEvents } from '../src/training/events'
import type { TrainingEvent } from '../src/training/types'

const repoRoot = join(import.meta.dirname, '..')
const template = join(repoRoot, 'sidecar-torch', 'training_template.py')
// Callers can select a specific interpreter; an activated project environment
// supplies the usual Python command.
const python = process.env.PYTHON ?? 'python'

let failures = 0
function check(name: string, cond: boolean, detail = '') {
  if (cond) console.log(`  ✓ ${name}`)
  else { failures++; console.log(`  ✗ ${name} ${detail}`) }
}

// ── synthetic data (same seeded xorshift as verify-failures) ──
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

const FEATURES = Array.from({ length: 10 }, (_, i) => `f${i}`)

function makeRunJson(epochs: number) {
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
      epochs,
      batch_size: 32,
      val_split: 0.2,
      split_strategy: 'random',
      seed: 42,
      log_every_n_steps: 1,
      optimizer: { kind: 'Adam', lr: 0.01, weight_decay: 0 },
      loss: { kind: 'CrossEntropyLoss' },
      scheduler: { kind: 'none' },
      metrics: ['accuracy'],
      callbacks: [],
    },
  }
}

function makeDir(name: string, opts: { rows?: number; epochs?: number; status?: string } = {}) {
  const dir = mkdtempSync(join(tmpdir(), `spinoml-cancel-${name}-`))
  writeFileSync(join(dir, 'data.csv'), csv(opts.rows ?? 1000, 10, 42))
  writeFileSync(join(dir, 'model.py'), MODEL_PY)
  const runJson = makeRunJson(opts.epochs ?? 20)
  runJson.dataset.path = join(dir, 'data.csv')
  writeFileSync(join(dir, 'run.json'), JSON.stringify(runJson, null, 2))
  copyFileSync(template, join(dir, 'train.py'))
  if (opts.status) writeFileSync(join(dir, 'status'), opts.status + '\n')
  return dir
}

function readEvents(dir: string): TrainingEvent[] {
  const p = join(dir, 'events.jsonl')
  if (!existsSync(p)) return []
  return parseFinalEvents(readFileSync(p, 'utf8'))
}

function allEvents(dir: string): TrainingEvent[] {
  const p = join(dir, 'events.jsonl')
  if (!existsSync(p)) return []
  return readFileSync(p, 'utf8').trim().split('\n').filter(Boolean)
    .map((l) => { try { return JSON.parse(l) } catch { return null } }).filter(Boolean)
}

function countKind(events: TrainingEvent[], kind: string): number {
  return events.filter((e) => e.kind === kind).length
}

function readStatus(dir: string): string {
  const p = join(dir, 'status')
  if (!existsSync(p)) return '<missing>'
  return readFileSync(p, 'utf8').trim()
}

function metricsStatus(dir: string): string | null {
  const p = join(dir, 'metrics.json')
  if (!existsSync(p)) return null
  try { return JSON.parse(readFileSync(p, 'utf8')).status as string } catch { return null }
}

/** Wait (≤timeoutMs) for `needle` in events.jsonl. Returns true when seen. */
async function waitForEvent(dir: string, needle: string, timeoutMs = 30_000): Promise<boolean> {
  const p = join(dir, 'events.jsonl')
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (existsSync(p) && readFileSync(p, 'utf8').includes(needle)) return true
    await new Promise((r) => setTimeout(r, 25))
  }
  return false
}

/** Spawn the real trainer; the caller calls `signal` (or relies on a status write). */
function spawnTrainer(dir: string) {
  return spawn(python, ['-u', 'train.py'], { cwd: dir, stdio: 'ignore' })
}

async function waitClose(child: ReturnType<typeof spawnTrainer>, timeoutMs = 90_000): Promise<number> {
  const killer = setTimeout(() => { try { child.kill('SIGKILL') } catch { /* gone */ } }, timeoutMs)
  const code = await new Promise<number>((resolve) => child.on('close', (c) => resolve(c ?? -1)))
  clearTimeout(killer)
  return code
}

/** Last completed epoch before a run.cancelled — the epoch the resumable
 *  last.pt must point at (Phase 32 consistency: stop → load → resume). */
function lastCompletedEpochBeforeCancel(events: TrainingEvent[]): number {
  let last = -1
  for (const e of events) {
    if (e.kind === 'run.cancelled') break
    if (e.kind === 'epoch.end' && typeof e.epoch === 'number') last = e.epoch
  }
  return last
}

/** Load last.pt.epoch via the run dir's own train.py (imports torch lazily). */
function ckptEpoch(dir: string): number {
  const script = [
    'import sys, torch',
    `ckpt = torch.load(r'${join(dir, 'checkpoints', 'last.pt')}', map_location='cpu', weights_only=False)`,
    'print(int(ckpt["epoch"]))',
  ].join('\n')
  const r = spawnSync(python, ['-c', script], { stdio: 'pipe' })
  if (r.status !== 0) throw new Error(r.stderr.toString())
  return parseInt(r.stdout.toString().trim(), 10)
}

console.log('phase 32: cancellation')

// ── 1. Cancel before start ── (status=cancelled pre-launch; no signal needed)
console.log('  [cancel before start (status file) ]')
{
  const dir = makeDir('pre-start', { status: 'cancelled' })
  const r = spawnSync(python, ['train.py'], { cwd: dir, stdio: 'pipe', timeout: 60_000 })
  check('clean exit (0)', r.status === 0, `exit=${r.status}`)
  check('status = cancelled', readStatus(dir) === 'cancelled')
  check('metrics.json status = cancelled', metricsStatus(dir) === 'cancelled')
  const ev = readEvents(dir)
  check('run.cancelled emitted exactly once', countKind(ev, 'run.cancelled') === 1)
  check('no run.done', countKind(ev, 'run.done') === 0)
  check('no epoch.start (exited during startup)', countKind(ev, 'epoch.start') === 0)
  check('first terminal event is run.cancelled', finalTerminal(ev)?.kind === 'run.cancelled')
  check('no events after the terminal (Phase 31)', !hasStaleTrailing(ev))
}

// ── 2. Cancel during startup ── real SIGTERM before the first epoch
console.log('  [cancel during startup (SIGTERM before epoch 0)]')
{
  const dir = makeDir('startup')
  const child = spawnTrainer(dir)
  const seen = await waitForEvent(dir, '"kind": "dataset.loaded"')
  check('training reached dataset.loaded', seen)
  child.kill('SIGTERM')
  const code = await waitClose(child)
  check('graceful exit (0)', code === 0, `exit=${code}`)
  check('status = cancelled', readStatus(dir) === 'cancelled')
  check('metrics.json status = cancelled', metricsStatus(dir) === 'cancelled')
  const ev = readEvents(dir)
  check('run.cancelled emitted exactly once', countKind(ev, 'run.cancelled') === 1)
  check('no run.done', countKind(ev, 'run.done') === 0)
  check('first terminal event is run.cancelled', finalTerminal(ev)?.kind === 'run.cancelled')
  check('no events after the terminal (Phase 31)', !hasStaleTrailing(ev))
}

// ── 3. Cancel during training ── real SIGTERM mid-epoch
console.log('  [cancel during training (SIGTERM mid-epoch)]')
{
  const dir = makeDir('mid-signal')
  const child = spawnTrainer(dir)
  const seen = await waitForEvent(dir, '"kind": "epoch.start", "epoch": 0')
  check('observed epoch 0 mid-flight', seen)
  child.kill('SIGTERM')
  const code = await waitClose(child)
  check('graceful exit (0)', code === 0, `exit=${code}`)
  check('status = cancelled', readStatus(dir) === 'cancelled')
  check('metrics.json status = cancelled', metricsStatus(dir) === 'cancelled')
  const ev = readEvents(dir)
  check('run.cancelled emitted exactly once', countKind(ev, 'run.cancelled') === 1)
  check('no run.done', countKind(ev, 'run.done') === 0)
  check('first terminal event is run.cancelled', finalTerminal(ev)?.kind === 'run.cancelled')
  check('no events after the terminal (Phase 31)', !hasStaleTrailing(ev))
  const expectEpoch = lastCompletedEpochBeforeCancel(ev)
  check(`resumable last.pt at last completed epoch (${expectEpoch})`,
    existsSync(join(dir, 'checkpoints', 'last.pt')) && ckptEpoch(dir) === expectEpoch,
    `last.pt=${existsSync(join(dir, 'checkpoints', 'last.pt')) ? ckptEpoch(dir) : 'missing'}`)
}

// ── 4. Cancel during training ── cooperative status-file write, no signal
console.log('  [cancel during training (status file, cooperative)]')
{
  const dir = makeDir('mid-coop')
  const child = spawnTrainer(dir)
  const seen = await waitForEvent(dir, '"kind": "epoch.start", "epoch": 1')
  check('observed epoch 1 mid-flight', seen)
  writeFileSync(join(dir, 'status'), 'cancelled\n')
  const code = await waitClose(child)
  check('graceful exit (0)', code === 0, `exit=${code}`)
  check('status = cancelled', readStatus(dir) === 'cancelled')
  check('metrics.json status = cancelled', metricsStatus(dir) === 'cancelled')
  const ev = readEvents(dir)
  check('run.cancelled emitted exactly once', countKind(ev, 'run.cancelled') === 1)
  check('no run.done', countKind(ev, 'run.done') === 0)
  check('first terminal event is run.cancelled', finalTerminal(ev)?.kind === 'run.cancelled')
  check('no events after the terminal (Phase 31)', !hasStaleTrailing(ev))
  const expectEpoch = lastCompletedEpochBeforeCancel(ev)
  check(`resumable last.pt at last completed epoch (${expectEpoch})`,
    existsSync(join(dir, 'checkpoints', 'last.pt')) && ckptEpoch(dir) === expectEpoch,
    `last.pt=${existsSync(join(dir, 'checkpoints', 'last.pt')) ? ckptEpoch(dir) : 'missing'}`)
}

// ── 5. Double cancellation ── two SIGTERMs; exactly one run.cancelled
console.log('  [double cancellation (two SIGTERMs)]')
{
  const dir = makeDir('double-signal')
  const child = spawnTrainer(dir)
  const seen = await waitForEvent(dir, '"kind": "epoch.start", "epoch": 1')
  check('observed epoch 1 mid-flight', seen)
  child.kill('SIGTERM')
  try { child.kill('SIGTERM') } catch { /* done */ }
  const code = await waitClose(child)
  check('graceful exit (0)', code === 0, `exit=${code}`)
  check('status = cancelled', readStatus(dir) === 'cancelled')
  const ev = readEvents(dir)
  check('run.cancelled emitted exactly once', countKind(ev, 'run.cancelled') === 1,
    `count=${countKind(ev, 'run.cancelled')}`)
  check('no run.done', countKind(ev, 'run.done') === 0)
  check('first terminal event is run.cancelled', finalTerminal(ev)?.kind === 'run.cancelled')
}

// ── 6. Cancel after completion ── a late cancel must NOT resurrect done
console.log('  [cancel after completion (terminal shield)]')
{
  const dir = makeDir('post-done', { rows: 60, epochs: 1 })
  const r = spawnSync(python, ['-u', 'train.py'], { cwd: dir, stdio: 'pipe', timeout: 60_000 })
  check('run completed (exit 0)', r.status === 0, `exit=${r.status}`)
  check('status = done', readStatus(dir) === 'done')
  const doneEvents = readEvents(dir)
  check('run.done emitted', countKind(doneEvents, 'run.done') === 1)

  // Emulate a SIGTERM that races in AFTER SUCCEEDED: the trainer's own
  // terminal-shielded cancel point must reject it — done stays FINAL, no
  // run.cancelled, metrics stay done (the Rust/ssh stop guard also blocks
  // post-done stops, but the trainer must be immune on the signal path too).
  const script = [
    'import sys, importlib.util, os',
    'spec = importlib.util.spec_from_file_location("trainer", os.path.join(sys.argv[1], "train.py"))',
    'trainer = importlib.util.module_from_spec(spec)',
    'spec.loader.exec_module(trainer)',
    'trainer._finish_cancel(99, "late signal after done")',
    'print("status", trainer._read_status())',
  ].join('\n')
  const preCount = countKind(allEvents(dir), 'run.cancelled')
  const r2 = spawnSync(python, ['-c', script, dir], { cwd: dir, stdio: 'pipe', timeout: 30_000 })
  check('shield call ran', r2.status === 0, r2.status !== 0 ? r2.stderr.toString() : '')
  const out = r2.stdout.toString().trim()
  check('status still done after late cancel', out.endsWith('done'), out)
  check('no run.cancelled appended', countKind(allEvents(dir), 'run.cancelled') === preCount)
  check('metrics.json still done', JSON.parse(readFileSync(join(dir, 'metrics.json'), 'utf8')).status === 'done')
}

console.log(failures === 0 ? '\n✓ all cancellation checks passed' : `\n✗ ${failures} check(s) failed`)
process.exit(failures === 0 ? 0 : 1)
