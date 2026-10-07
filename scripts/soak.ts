// Phase 52 — LONG-RUNNING TEST (soak).
//
// A real-time loop against a torch sidecar (dev mode) and an LLM sidecar
// (token mode, fake provider) mixing repeated /infer, /dataset/inspect,
// /chat turns and — every ~15 s — a tiny REAL training run of the standalone
// trainer (sidecar-torch/training_template.py copied as train.py into a temp
// run dir, exactly the verify-checkpoint/verify-integrity way). Every ~5 s it
// samples RSS / fds / threads / children of BOTH sidecars and its own RSS.
//
// At the end it fits a least-squares slope to the post-warm-up RSS samples
// (first 20% discarded) and FAILS if the projected 1-hour growth > 200 MB, an
// fd/thread slope > 0.05/min, any request returned 5xx, any training run did
// not end `done`, or any process died.
//
// Honest limits: GPU memory is SKIPPED (no CUDA here) and the UI/Rust side
// (saves, metric polling) is not exercised.
//
// Run: npm run test:soak -- --seconds 90 [--out .test-results]

import { spawn } from 'node:child_process'
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { setTimeout as wait } from 'node:timers/promises'
import { fileURLToPath } from 'node:url'

import { pidAlive, startSidecar, gracefulStop } from './lib/auth-probe'
import { LlmHarness } from './lib/llm-harness'
import { median, readProcStats, slopePerMin } from './lib/resource-probe'

const HERE = dirname(fileURLToPath(import.meta.url))
const REPO = resolve(HERE, '..')
const PYTHON = process.env.PYTHON ?? 'python'
const TORCH = join(REPO, 'sidecar-torch', 'main.py')
const TEMPLATE = join(REPO, 'sidecar-torch', 'training_template.py')
const TOKEN = 'a'.repeat(64)
let llmUrl = ''

function argValue(name: string, fallback: string): string {
  const i = process.argv.indexOf(name)
  return i >= 0 && process.argv[i + 1] !== undefined ? process.argv[i + 1] : fallback
}

const SECONDS = Number(argValue('--seconds', '90'))
const OUT = argValue('--out', '.test-results')

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

function csv(rows: number, features: number, seed: number): string {
  let s = seed >>> 0
  const rand = (): number => {
    s ^= s << 13
    s ^= s >>> 17
    s ^= s << 5
    return (s >>> 0) / 0x100000000
  }
  const randn = (): number => {
    const u1 = rand() + 1e-12
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * rand())
  }
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

function linearModel(v: number): string {
  return `import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))  # v${v}
`
}

interface TrainResult {
  ok: boolean
  detail: string
}

function makeRunDir(parent: string, name: string): string {
  const dir = mkdtempSync(join(parent, `run-${name}-`))
  writeFileSync(join(dir, 'data.csv'), csv(60, 10, SEED))
  writeFileSync(join(dir, 'model.py'), MODEL_PY)
  const runJson = {
    run_id: 'soak',
    run_label: 'soak',
    created_at: new Date().toISOString(),
    status: 'queued',
    model_path: 'm',
    backend: { kind: 'local' },
    dataset: {
      path: join(dir, 'data.csv'),
      relpath: 'data.csv',
      kind: 'tabular',
      feature_columns: FEATURES,
      target_column: 'label',
    },
    training: {
      epochs: 2,
      batch_size: 32,
      val_split: 0.25,
      split_strategy: 'random',
      seed: SEED,
      log_every_n_steps: 1,
      optimizer: { kind: 'Adam', lr: 0.01, weight_decay: 0 },
      loss: { kind: 'CrossEntropyLoss' },
      scheduler: { kind: 'StepLR', step_size: 2, gamma: 0.5 },
      metrics: ['accuracy'],
      callbacks: [],
    },
  }
  writeFileSync(join(dir, 'run.json'), JSON.stringify(runJson, null, 2))
  copyFileSync(TEMPLATE, join(dir, 'train.py'))
  return dir
}

async function runTraining(parent: string, name: string): Promise<TrainResult> {
  const dir = makeRunDir(parent, name)
  const okRun = await new Promise<boolean>((resolveP) => {
    const child = spawn(PYTHON, ['-u', 'train.py'], { cwd: dir, stdio: 'ignore' })
    const t = setTimeout(() => {
      try {
        child.kill('SIGKILL')
      } catch {
        // already gone
      }
      resolveP(false)
    }, 120000)
    child.once('exit', (code) => {
      clearTimeout(t)
      resolveP(code === 0)
    })
    child.once('error', () => {
      clearTimeout(t)
      resolveP(false)
    })
  })
  try {
    const status = existsSync(join(dir, 'status')) ? readFileSync(join(dir, 'status'), 'utf8').trim() : ''
    const manifestRaw = existsSync(join(dir, 'manifest.json')) ? readFileSync(join(dir, 'manifest.json'), 'utf8') : ''
    const metricsRaw = existsSync(join(dir, 'metrics.json')) ? readFileSync(join(dir, 'metrics.json'), 'utf8') : ''
    const eventsRaw = existsSync(join(dir, 'events.jsonl')) ? readFileSync(join(dir, 'events.jsonl'), 'utf8') : ''
    let manifestStatus = ''
    try {
      manifestStatus = String((JSON.parse(manifestRaw) as Record<string, unknown>).status ?? '')
    } catch {
      // invalid manifest json → empty
    }
    const hasRunDone = /"kind":\s*"run\.done"/.test(eventsRaw)
    const hasIntegrityOk = /"kind":\s*"run\.integrity"[^\n]*"ok":\s*true/.test(eventsRaw)
    let metricsOk = false
    try {
      const m = JSON.parse(metricsRaw) as Record<string, unknown>
      metricsOk = typeof m === 'object' && m !== null && ('val_loss' in m || 'epochs' in m || 'final' in m || 'status' in m)
    } catch {
      metricsOk = false
    }
    const ok = okRun && status === 'done' && manifestStatus === 'done' && hasRunDone && hasIntegrityOk && metricsOk
    const detail = `exit=${okRun} status=${status || '-'} manifest=${manifestStatus || '-'} run.done=${hasRunDone} integrity=${hasIntegrityOk} metrics=${metricsOk}`
    return { ok, detail }
  } finally {
    try {
      rmSync(dir, { recursive: true, force: true })
    } catch {
      // temp dir best-effort
    }
  }
}

interface Snap {
  t: number
  torchRss: number | null
  torchFd: number | null
  torchThreads: number | null
  torchChildren: number | null
  llmRss: number | null
  llmFd: number | null
  llmThreads: number | null
  llmChildren: number | null
  selfRss: number | null
}

async function minBurst(pid: number, preRead?: () => Promise<void>): Promise<{ rssKb: number | null; fds: number | null; threads: number | null; children: number | null }> {
  // V8/Python allocators grow RSS and only release it on a GC cycle, so a
  // single /proc read is a sawtooth. The minimum over a short burst is the
  // post-GC floor — the memory the process actually retains — which is the
  // honest signal for a leak. Still one "sample" every 5 s.
  let rssKb: number | null = null
  let fds: number | null = null
  let threads: number | null = null
  let children: number | null = null
  for (let i = 0; i < 5; i++) {
    if (preRead) await preRead()
    const s = readProcStats(pid)
    if (s.rssKb !== null) rssKb = rssKb === null ? s.rssKb : Math.min(rssKb, s.rssKb)
    if (s.fds !== null) fds = fds === null ? s.fds : Math.min(fds, s.fds)
    if (s.threads !== null) threads = threads === null ? s.threads : Math.min(threads, s.threads)
    if (s.children !== null) children = children === null ? s.children : Math.min(children, s.children)
    if (i < 4) await wait(200)
  }
  return { rssKb, fds, threads, children }
}

async function snapshot(t0: number, torchPid: number, llmPid: number): Promise<Snap> {
  const t = await minBurst(torchPid)
  // The LLM sidecar runs with --expose-gc; pinging its diag collapses the V8
  // heap before every read so the sample is the retained floor, not sawtooth.
  const l = await minBurst(llmPid, async () => {
    await fetch(`${llmUrl}/health`, { headers: { 'X-SpinoML-Token': TOKEN } })
      .then((r) => r.text())
      .catch(() => undefined)
  })
  const self = readProcStats(process.pid)
  return {
    t: (Date.now() - t0) / 1000,
    torchRss: t.rssKb,
    torchFd: t.fds,
    torchThreads: t.threads,
    torchChildren: t.children,
    llmRss: l.rssKb,
    llmFd: l.fds,
    llmThreads: l.threads,
    llmChildren: l.children,
    selfRss: self.rssKb,
  }
}

function series(samples: Snap[], key: keyof Snap): number[] {
  return samples.map((s) => s[key]).filter((v): v is number => typeof v === 'number')
}

function slopeFor(samples: Snap[], key: keyof Snap): number {
  const xs: number[] = []
  const ys: number[] = []
  for (const s of samples) {
    const v = s[key]
    if (typeof v === 'number') {
      xs.push(s.t)
      ys.push(v)
    }
  }
  return slopePerMin(xs, ys)
}

interface Check {
  name: string
  expected: string
  got: string
  pass: boolean
}

async function main(): Promise<void> {
  if (!Number.isFinite(SECONDS) || SECONDS <= 0) {
    console.error(`invalid --seconds ${argValue('--seconds', '')}`)
    process.exit(2)
  }
  console.log(`phase 52: long-running soak (${SECONDS}s)`)
  console.log('SKIPPED  CUDA — no GPU memory sampling on this host (torch.cuda is never probed)')
  console.log('SKIPPED  UI/Rust — saves and metric polling are not exercised (node-only sidecar soak)')
  console.log(`  out = ${OUT}`)

  const root = mkdtempSync(join(tmpdir(), 'spinoml-soak-'))
  const ws = join(root, 'ws')
  mkdirSync(ws, { recursive: true })
  writeFileSync(join(ws, 'data.csv'), csv(60, 4, 1))

  const torch = await startSidecar({
    cmd: PYTHON,
    args: [TORCH],
    cwd: REPO,
    readyTimeoutMs: 120000,
    stripEnvPrefix: 'SPINOML_',
    portEnv: 'SPINOML_TORCH_PORT',
    env: { SPINOML_ALLOWED_ROOTS: ws },
  })
  const llm = await startSidecar({
    cmd: process.execPath,
    args: ['--expose-gc', join(REPO, 'sidecar-llm', 'main.mjs')],
    cwd: REPO,
    readyTimeoutMs: 60000,
    stripEnvPrefix: 'SPINOML_',
    portEnv: 'SPINOML_LLM_PORT',
    env: { SPINOML_SIDECAR_TOKEN: TOKEN, SPINOML_REQUIRE_TOKEN: '1' },
  })
  const torchPid = torch.proc.pid
  const llmPid = llm.proc.pid
  llmUrl = llm.url
  if (torchPid === undefined || llmPid === undefined) throw new Error('sidecar without pid')

  let harness: LlmHarness | null = null
  const samples: Snap[] = []
  const checks: Check[] = []
  const errorsByKind: Record<string, number> = {}
  const counts = { infer: 0, inspect: 0, chat: 0, train: 0, trainFail: 0 }
  let fivexx = 0
  let died = false

  const bump = (kind: string): void => {
    errorsByKind[kind] = (errorsByKind[kind] ?? 0) + 1
  }

  try {
    harness = await LlmHarness.connect({ baseUrl: llm.url })
    let v = 0
    // Warm both sidecars to steady state BEFORE the timed window: Node's V8
    // heap (and Python/torch's arenas) ramp for ~15 s under load and would
    // otherwise dominate the slope of a short run. The timed window starts
    // only once both are hot.
    const warmStart = Date.now()
    while (Date.now() - warmStart < 12000) {
      for (let k = 0; k < 5; k++) {
        const r = await fetch(`${torch.url}/infer`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code: linearModel(v++), input_shapes: [[1, 8]], input_dtypes: ['float32'] }),
        })
        if (r.status >= 500) fivexx++
        await r.text()
      }
      await harness.chat({ script: [{ text: 'warm' }], token: TOKEN, timeoutMs: 20000 })
    }

    const t0 = Date.now()
    const endAt = t0 + SECONDS * 1000
    let lastSample = 0
    let lastTrain = Date.now() - 15000 // train on the first pass
    while (Date.now() < endAt) {
      if (torch.proc.exitCode !== null || llm.proc.exitCode !== null) {
        died = true
        bump('process-died')
        break
      }
      // a small mixed batch
      for (let k = 0; k < 5; k++) {
        const r = await fetch(`${torch.url}/infer`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code: linearModel(v++), input_shapes: [[1, 8]], input_dtypes: ['float32'] }),
        })
        if (r.status >= 500) fivexx++
        else counts.infer++
        await r.text()
      }
      {
        const r = await fetch(`${torch.url}/dataset/inspect`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ abspath: join(ws, 'data.csv') }),
        })
        if (r.status >= 500) fivexx++
        else counts.inspect++
        await r.text()
      }
      {
        const res = await harness.chat({ script: [{ text: 'soak turn' }], token: TOKEN, timeoutMs: 20000 })
        if (res.fetchError !== undefined) {
          fivexx++
          bump('chat-fetch')
        } else {
          counts.chat++
        }
      }

      if (Date.now() - lastSample >= 5000) {
        samples.push(await snapshot(t0, torchPid, llmPid))
        lastSample = Date.now()
      }
      if (Date.now() - lastTrain >= 15000 && Date.now() < endAt - 3000) {
        counts.train++
        const tr = await runTraining(root, `t${counts.train}`)
        if (!tr.ok) {
          counts.trainFail++
          bump('training')
          console.log(`  training run ${counts.train} NOT done: ${tr.detail}`)
        }
        lastTrain = Date.now()
      }
      await wait(50)
    }
  } catch (e) {
    console.error(`soak loop error: ${(e as Error).message}`)
    process.exitCode = 1
  } finally {
    if (harness) await harness.stop()
    await gracefulStop(torch.proc, 5000)
    await gracefulStop(llm.proc, 5000)
  }

  // final idle then one more sample
  await wait(1000)
  // processes are gone now — one last pre-stop snapshot was taken in the loop;
  // record what we have and evaluate.

  const post = samples.slice(Math.floor(samples.length * 0.2))
  const torchSlope = slopeFor(post, 'torchRss')
  const llmSlope = slopeFor(post, 'llmRss')
  const torchFdSlope = slopeFor(post, 'torchFd')
  const llmFdSlope = slopeFor(post, 'llmFd')
  const torchThreadSlope = slopeFor(post, 'torchThreads')
  const llmThreadSlope = slopeFor(post, 'llmThreads')
  const projectedTorchKb = torchSlope * 60 // KB over 1 h
  const projectedLlmKb = llmSlope * 60
  const MB = 1024

  const add = (name: string, expected: string, got: string, pass: boolean): void => {
    checks.push({ name, expected, got, pass })
  }
  add('RSS growth over 1 h (torch) ≤ 200 MB', '≤ 200 MB', `${(projectedTorchKb / MB).toFixed(2)} MB @${torchSlope.toFixed(2)} KB/min`, projectedTorchKb <= 200 * MB)
  add('RSS growth over 1 h (llm) ≤ 200 MB', '≤ 200 MB', `${(projectedLlmKb / MB).toFixed(2)} MB @${llmSlope.toFixed(2)} KB/min`, projectedLlmKb <= 200 * MB)
  add('fd slope (torch) ≤ 0.05/min', '≤ 0.05', torchFdSlope.toFixed(3), torchFdSlope <= 0.05)
  add('fd slope (llm) ≤ 0.05/min', '≤ 0.05', llmFdSlope.toFixed(3), llmFdSlope <= 0.05)
  add('thread slope (torch) ≤ 0.05/min', '≤ 0.05', torchThreadSlope.toFixed(3), torchThreadSlope <= 0.05)
  add('thread slope (llm) ≤ 0.05/min', '≤ 0.05', llmThreadSlope.toFixed(3), llmThreadSlope <= 0.05)
  add('no unexpected 5xx', '0', String(fivexx), fivexx === 0)
  add('all training runs ended done', '0 failures', String(counts.trainFail), counts.trainFail === 0)
  add('no process died', 'alive', died ? 'DIED' : 'alive', !died)
  const lastSnap = samples.length > 0 ? samples[samples.length - 1] : null
  add('torch no orphan children at end', '0', String(lastSnap?.torchChildren), lastSnap?.torchChildren === 0)
  add('llm no orphan children at end', '0', String(lastSnap?.llmChildren), lastSnap?.llmChildren === 0)
  add('sidecar pids gone after stop', 'gone', `${String(pidAlive(torchPid))}/${String(pidAlive(llmPid))}`, !pidAlive(torchPid) && !pidAlive(llmPid))

  // summary table
  const statRow = (label: string, key: keyof Snap): string[] => {
    const vals = series(samples, key)
    return [label, String(vals.length ? Math.min(...vals) : 'n/a'), String(vals.length ? Math.round(median(vals)) : 'n/a'), String(vals.length ? Math.max(...vals) : 'n/a')]
  }
  console.log('\nRSS / fd / thread samples (min | median | max)')
  const header = ['series', 'min', 'median', 'max']
  const tableRows = [
    statRow('torch RSS KB', 'torchRss'),
    statRow('llm RSS KB', 'llmRss'),
    statRow('self RSS KB', 'selfRss'),
    statRow('torch fd', 'torchFd'),
    statRow('llm fd', 'llmFd'),
    statRow('torch threads', 'torchThreads'),
    statRow('llm threads', 'llmThreads'),
    statRow('torch children', 'torchChildren'),
    statRow('llm children', 'llmChildren'),
  ]
  printSimpleTable(header, tableRows)

  console.log('\nrequest counts / errors')
  printSimpleTable(['kind', 'count'], [
    ['infer', String(counts.infer)],
    ['inspect', String(counts.inspect)],
    ['chat', String(counts.chat)],
    ['training', String(counts.train)],
    ['training failed', String(counts.trainFail)],
    ['5xx', String(fivexx)],
    ...Object.entries(errorsByKind).map(([k, n]) => [`error:${k}`, String(n)]),
  ])

  printSimpleTable(['check', 'expected', 'got', 'verdict'], checks.map((c) => [c.name, c.expected, c.got, c.pass ? 'PASS' : 'FAIL']))
  const failed = checks.filter((c) => !c.pass)
  console.log(`\n${checks.length - failed.length}/${checks.length} checks passed`)

  // write artifacts
  mkdirSync(OUT, { recursive: true })
  const stamp = new Date().toISOString().replace(/[:.]/g, '-')
  const jsonPath = join(OUT, `soak-${stamp}.json`)
  const mdPath = join(OUT, `soak-${stamp}.md`)
  const payload = {
    seconds: SECONDS,
    samples,
    slopes: { torchRss: torchSlope, llmRss: llmSlope, torchFd: torchFdSlope, llmFd: llmFdSlope, torchThreads: torchThreadSlope, llmThreads: llmThreadSlope },
    counts,
    fivexx,
    errorsByKind,
    checks,
    skipped: ['CUDA (no GPU sampling)', 'UI/Rust saves and metric polling'],
  }
  writeFileSync(jsonPath, JSON.stringify(payload, null, 2))
  const md = [
    `# soak ${stamp}`,
    '',
    `duration ${SECONDS}s; ${samples.length} samples; ${checks.length - failed.length}/${checks.length} checks passed`,
    '',
    `| series | min | median | max |`,
    `| --- | --- | --- | --- |`,
    ...tableRows.map((r) => `| ${r.join(' | ')} |`),
    '',
    `| check | expected | got | verdict |`,
    `| --- | --- | --- | --- |`,
    ...checks.map((c) => `| ${c.name} | ${c.expected} | ${c.got} | ${c.pass ? 'PASS' : 'FAIL'} |`),
    '',
    `requests: infer=${counts.infer} inspect=${counts.inspect} chat=${counts.chat} train=${counts.train} trainFail=${counts.trainFail} 5xx=${fivexx}`,
    '',
    `SKIPPED: CUDA (no GPU sampling); UI/Rust saves and metric polling`,
    '',
  ].join('\n')
  writeFileSync(mdPath, md)
  console.log(`\nwrote ${jsonPath}`)
  console.log(`wrote ${mdPath}`)

  if (failed.length > 0) {
    console.log('\nFAILURES:')
    for (const f of failed) console.log(`  ✗ ${f.name}: expected ${f.expected}, got ${f.got}`)
    process.exitCode = 1
  }
  try {
    rmSync(root, { recursive: true, force: true })
  } catch {
    // temp dir best-effort
  }
}

function printSimpleTable(headers: string[], rows: string[][]): void {
  const widths = headers.map((h, i) => Math.max(h.length, ...rows.map((r) => (r[i] ?? '').length)))
  const line = (cells: string[]): string => '  ' + cells.map((c, i) => c.padEnd(widths[i])).join('  ')
  console.log(line(headers))
  console.log('  ' + widths.map((w) => '-'.repeat(w)).join('  '))
  for (const row of rows) console.log(line(row))
}

void main()
