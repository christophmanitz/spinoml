// Phase 51 — RESOURCE LEAK TESTING (torch + LLM sidecars).
//
// Real processes, real sockets, real /proc sampling — no mocked counters.
// Repeated use must not grow processes, memory, file descriptors, threads,
// temp files or in-process state. Phase 13 (test-process-lifecycle) already
// covers start/stop/restart cycles and orphan children; this file does NOT
// duplicate that — it hammers a single long-lived sidecar with requests and
// watches /proc/<pid> + /health.diag for growth.
//
// The RSS bound is MEASURED, not guessed: run it three times, take the worst
// post-warm-up growth, add a 100% + 20 MB margin (see RSS_BOUND_KB below).
//
// Run: npm run test:resource-leaks

import { spawnSync } from 'node:child_process'
import { mkdtempSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { setTimeout as wait } from 'node:timers/promises'
import { fileURLToPath } from 'node:url'

import {
  descendantsOf,
  gracefulStop,
  startSidecar,
  type StartedSidecar,
} from './lib/auth-probe'
import { LlmHarness } from './lib/llm-harness'
import type { ChatResult } from './lib/llm-harness'
import type { ToolCallSpec } from './lib/fake-openai'
import { readProcStats, delta, printTable, type ProcStats } from './lib/resource-probe'

const HERE = dirname(fileURLToPath(import.meta.url))
const REPO = resolve(HERE, '..')
const PYTHON = process.env.PYTHON ?? 'python'
const NODE = process.execPath
const TORCH = join(REPO, 'sidecar-torch', 'main.py')
const LLM = join(REPO, 'sidecar-llm', 'main.mjs')
const FAKE_OPENCODE = join(REPO, 'scripts', 'lib', 'fake-opencode')

const TOKEN = 'a'.repeat(64)

// ── Measured bound (see NOTES in the phase report) ────────────────────────
// Run 3× on this host, worst post-warm-up RSS growth (baseline after 20
// warm-up requests → after 300 infer + 80 inspect + 40 invalid + 20
// activations, 1 s idle) was 124,344 KB (torch) and 75,072 KB (LLM). This is
// a one-time CPython/torch high-water mark, NOT a per-request leak: a 1200-
// request probe showed the RSS flat after the first 300 (231,020 → 231,140
// KB), which is exactly what the soak slope test independently checks.
// Bound = worst-measured × 2 + 20 MB margin (the brief's formula), rounded up.
const TORCH_RSS_BOUND_KB = 270 * 1024
const LLM_RSS_BOUND_KB = 172 * 1024

interface TestRow {
  case: string
  expected: string
  got: string
  pass: boolean
}

const rows: TestRow[] = []

function record(name: string, expected: string, got: string, pass: boolean): void {
  rows.push({ case: name, expected, got, pass })
}

// ── HTTP helpers ──────────────────────────────────────────────────────────

interface HttpResult {
  status: number
  json: Record<string, unknown>
  raw: string
}

async function httpPost(url: string, path: string, body: unknown, token?: string): Promise<HttpResult> {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (token) headers['X-SpinoML-Token'] = token
  const res = await fetch(`${url}${path}`, {
    method: 'POST',
    headers,
    body: typeof body === 'string' ? body : JSON.stringify(body),
  })
  const raw = await res.text()
  let json: Record<string, unknown> = {}
  try {
    json = JSON.parse(raw) as Record<string, unknown>
  } catch {
    // non-JSON body — leave as {}
  }
  return { status: res.status, json, raw }
}

async function diagOf(url: string, token?: string): Promise<Record<string, unknown> | null> {
  const headers: Record<string, string> = {}
  if (token) headers['X-SpinoML-Token'] = token
  try {
    const res = await fetch(`${url}/health`, { headers })
    const body = (await res.json()) as Record<string, unknown>
    const d = body.diag
    return d && typeof d === 'object' ? (d as Record<string, unknown>) : null
  } catch {
    return null
  }
}

// A tiny valid model; `v` is embedded as a comment so each request is a
// distinct source (no cache anywhere hides a leak).
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

const UNKNOWN_LAYER = `import torch
import torch.nn as nn
BOGUS = nn.BogusLayer(8, 4)
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = BOGUS
    def forward(self, x):
        return self.layer(x)
`

function tmpEntries(): Set<string> {
  try {
    return new Set(readdirSync(tmpdir()))
  } catch {
    return new Set()
  }
}

// One-off cache directories that third-party tooling creates the FIRST time it runs under a fresh
// TMPDIR (the runner gives every suite a private one): torch's inductor cache and tsx's loader
// cache. They are created once and do not grow per request, so they are not sidecar leaks.
const BENIGN_TMP_CACHES = /^(torchinductor_[A-Za-z0-9_.-]+|tsx-\d+)$/

function newTmpEntries(before: Set<string>): string[] {
  const after = tmpEntries()
  const added: string[] = []
  for (const name of after) if (!before.has(name) && !BENIGN_TMP_CACHES.test(name)) added.push(name)
  return added
}

function makeCsv(rows: number, features: number): string {
  const hdr = Array.from({ length: features }, (_, i) => `f${i}`).concat('label').join(',')
  const lines = [hdr]
  for (let i = 0; i < rows; i++) {
    const vals = Array.from({ length: features }, (_, j) => ((i % 2 === 0 ? -0.4 : 0.4) + j * 0.1).toFixed(4))
    lines.push([...vals, String(i % 2)].join(','))
  }
  return lines.join('\n') + '\n'
}

function makePt(path: string): boolean {
  const code = 'import sys, torch\ntorch.save(torch.randn(8, 4), sys.argv[1])\n'
  const r = spawnSync(PYTHON, ['-c', code, path], { encoding: 'utf8' })
  return r.status === 0
}

function fdThreadDelta(before: ProcStats, after: ProcStats): { fd: number | null; threads: number | null; children: number | null; rssKb: number | null } {
  return {
    fd: delta(after.fds, before.fds),
    threads: delta(after.threads, before.threads),
    children: delta(after.children, before.children),
    rssKb: delta(after.rssKb, before.rssKb),
  }
}

// ── Torch family ──────────────────────────────────────────────────────────

async function torchFamily(): Promise<void> {
  console.log('\n=== torch sidecar ===')
  const ws = mkdtempSync(join(tmpdir(), 'spinoml-leak-torch-'))
  writeFileSync(join(ws, 'data.csv'), makeCsv(60, 4))
  const ptOk = makePt(join(ws, 'small.pt'))
  if (!ptOk) console.log('  (warning: could not create small.pt — tensor inspect skipped)')

  const s = await startSidecar({
    cmd: PYTHON,
    args: [TORCH],
    cwd: REPO,
    readyTimeoutMs: 120000,
    stripEnvPrefix: 'SPINOML_',
    portEnv: 'SPINOML_TORCH_PORT',
    env: { SPINOML_ALLOWED_ROOTS: ws },
  })
  const pid = s.proc.pid
  try {
    if (pid === undefined) throw new Error('torch sidecar has no pid')
    const tmpBefore = tmpEntries()
    const fivexx: number[] = []

    // Warm up 20 requests so lazy imports / thread-pool init land before the
    // baseline sample.
    for (let i = 0; i < 20; i++) {
      const r = await httpPost(s.url, '/infer', { code: linearModel(i), input_shapes: [[1, 8]], input_dtypes: ['float32'] })
      if (r.status >= 500) fivexx.push(r.status)
    }
    const base = readProcStats(pid)
    const baseDiag = await diagOf(s.url)
    const cpu0 = base.cpuSeconds

    // 300 /infer on a small valid model.
    let inferOk = 0
    for (let i = 0; i < 300; i++) {
      const r = await httpPost(s.url, '/infer', { code: linearModel(20 + i), input_shapes: [[1, 8]], input_dtypes: ['float32'] })
      if (r.status >= 500) fivexx.push(r.status)
      if (r.json.ok === true) inferOk++
    }

    // 80 /dataset/inspect on a small CSV and a small .pt.
    let inspectOk = 0
    for (let i = 0; i < 80; i++) {
      const p = i % 2 === 0 ? join(ws, 'data.csv') : join(ws, 'small.pt')
      const r = await httpPost(s.url, '/dataset/inspect', { abspath: p })
      if (r.status >= 500) fivexx.push(r.status)
      if (r.json.ok === true) inspectOk++
    }

    // 40 deliberately INVALID requests (bad json, unknown kind, scope-denied, oversize).
    let invalidAnswered = 0
    for (let i = 0; i < 40; i++) {
      const mode = i % 4
      let r: HttpResult
      if (mode === 0) r = await httpPost(s.url, '/infer', '{this is not json')
      else if (mode === 1) r = await httpPost(s.url, '/infer', { code: UNKNOWN_LAYER, input_shapes: [[1, 8]], input_dtypes: ['float32'] })
      else if (mode === 2) r = await httpPost(s.url, '/dataset/inspect', { abspath: '/etc/passwd' })
      else r = await httpPost(s.url, '/infer', { code: linearModel(i), input_shapes: [[1000000000]], input_dtypes: ['float32'] })
      if (r.status >= 500) fivexx.push(r.status)
      else invalidAnswered++
    }

    // 20 /activations.
    let actOk = 0
    for (let i = 0; i < 20; i++) {
      const r = await httpPost(s.url, '/activations', { code: linearModel(i), input_shapes: [[1, 8]], input_dtypes: ['float32'] })
      if (r.status >= 500) fivexx.push(r.status)
      if (r.json.ok === true) actOk++
    }

    await wait(1000)
    const after = readProcStats(pid)
    const afterDiag = await diagOf(s.url)
    const d = fdThreadDelta(base, after)
    const cpu1 = after.cpuSeconds
    const cpuPerReq = cpu0 !== null && cpu1 !== null ? (cpu1 - cpu0) / (300 + 80 + 40 + 20) : null
    const diagReqDelta = baseDiag && afterDiag
      ? Number(afterDiag.requests_total) - Number(baseDiag.requests_total)
      : null

    const measured: string[][] = [
      ['fd delta', String(d.fd), '≤ 2', d.fd !== null && d.fd <= 2 ? 'PASS' : 'FAIL'],
      ['thread delta', String(d.threads), '≤ 2', d.threads !== null && d.threads <= 2 ? 'PASS' : 'FAIL'],
      ['child-process delta', String(d.children), '= 0', d.children === 0 ? 'PASS' : 'FAIL'],
      ['RSS growth (KB)', String(d.rssKb), `≤ ${TORCH_RSS_BOUND_KB}`, d.rssKb !== null && d.rssKb <= TORCH_RSS_BOUND_KB ? 'PASS' : 'FAIL'],
      ['CPU s/request', cpuPerReq === null ? 'null' : cpuPerReq.toFixed(5), 'info', 'INFO'],
      ['requests_total delta', String(diagReqDelta), '> 0', diagReqDelta !== null && diagReqDelta > 0 ? 'PASS' : 'FAIL'],
      ['5xx responses', String(fivexx.length), '0', fivexx.length === 0 ? 'PASS' : 'FAIL'],
    ]
    printTable(['torch metric', 'measured', 'bound', 'verdict'], measured)

    record('torch: 300 /infer ok', '300', String(inferOk), inferOk === 300)
    record('torch: 80 /dataset/inspect ok', '80', String(inspectOk), inspectOk === 80)
    record('torch: 20 /activations ok', '20', String(actOk), actOk === 20)
    record('torch: 40 invalid answered (no 5xx)', '40', String(invalidAnswered), invalidAnswered === 40)
    record('torch: no 5xx', '0', String(fivexx.length), fivexx.length === 0)
    record('torch: fd delta ≤ 2', '≤ 2', String(d.fd), d.fd !== null && d.fd <= 2)
    record('torch: thread delta ≤ 2', '≤ 2', String(d.threads), d.threads !== null && d.threads <= 2)
    record('torch: child-process delta = 0', '0', String(d.children), d.children === 0)
    record(`torch: RSS growth ≤ ${TORCH_RSS_BOUND_KB} KB`, 'bounded', `${String(d.rssKb)} KB`, d.rssKb !== null && d.rssKb <= TORCH_RSS_BOUND_KB)
    record('torch: requests_total grew', '> 0', String(diagReqDelta), diagReqDelta !== null && diagReqDelta > 0)
    const added = newTmpEntries(tmpBefore).filter((n) => !n.startsWith('spinoml-leak-torch-'))
    record('torch: no sidecar tmp entries', '0', String(added.length), added.length === 0)
    if (added.length > 0) console.log('  new tmp entries:', added.slice(0, 5).join(', '))
  } finally {
    await gracefulStop(s.proc, 5000)
    rmSync(ws, { recursive: true, force: true })
  }
}

// ── LLM family ────────────────────────────────────────────────────────────

const tc = (name: string, args: unknown): ToolCallSpec => ({
  name,
  arguments: typeof args === 'string' ? args : JSON.stringify(args),
})

async function llmSidecar(scenario: Record<string, unknown>): Promise<StartedSidecar> {
  return await startSidecar({
    cmd: NODE,
    args: [LLM],
    cwd: REPO,
    readyTimeoutMs: 60000,
    stripEnvPrefix: 'SPINOML_',
    portEnv: 'SPINOML_LLM_PORT',
    env: {
      SPINOML_SIDECAR_TOKEN: TOKEN,
      SPINOML_REQUIRE_TOKEN: '1',
      SPINOML_OPENCODE_BIN: FAKE_OPENCODE,
      FAKE_OPENCODE_SCENARIO: JSON.stringify(scenario),
    },
  })
}

/** 150 fake-provider turns in four shapes + `opencodeTurns` after. */
async function openaiCompatTurns(harness: LlmHarness, n: number): Promise<{ fivexx: number; turned: number }> {
  const fivexx: number[] = []
  let turned = 0
  for (let i = 0; i < n; i++) {
    const shape = i % 4
    let result: ChatResult
    if (shape === 0) {
      result = await harness.chat({ script: [{ text: 'plain text reply' }], token: TOKEN, timeoutMs: 20000 })
    } else if (shape === 1) {
      result = await harness.chat({
        script: [
          { toolCalls: [tc('add_layer', { layer_type: 'Linear', params: { in_features: 4, out_features: 2 }, after: 'fc' })] },
          { text: 'mutated' },
        ],
        token: TOKEN,
        timeoutMs: 20000,
      })
    } else if (shape === 2) {
      result = await harness.chat({
        script: [{ toolCalls: [tc('ask_user', { kind: 'select', prompt: 'pick', options: ['a', 'b'] })] }, { text: 'answered' }],
        token: TOKEN,
        timeoutMs: 20000,
        onAsk: (ask) => {
          void harness.respond(ask.id, 'a', TOKEN)
          return undefined as unknown as string
        },
      })
    } else {
      const ctrl = new AbortController()
      setTimeout(() => ctrl.abort(), 150)
      result = await harness.chat({ script: [{ hang: true }], token: TOKEN, signal: ctrl.signal, timeoutMs: 20000 })
    }
    if (result.fetchError !== undefined) fivexx.push(1)
    if (result.ended || result.aborted || result.texts.length > 0) turned++
  }
  return { fivexx: fivexx.length, turned }
}

async function runOpencodeTurns(s: StartedSidecar, n: number, abort: boolean): Promise<{ ok: number; fivexx: number }> {
  let ok = 0
  let fivexx = 0
  for (let i = 0; i < n; i++) {
    const ctrl = new AbortController()
    const res = await fetch(`${s.url}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-SpinoML-Token': TOKEN },
      body: JSON.stringify({
        user: 'opencode turn',
        messages: [],
        graph: { input_shape: [1, 4], nodes: [], edges: [] },
        training_graph: { nodes: [], edges: [] },
        data_graph: { nodes: [], edges: [] },
        project: null,
        autoMode: false,
        llm: { kind: 'opencode', model: 'fake' },
      }),
      signal: ctrl.signal,
    }).catch(() => undefined)
    if (abort) {
      await wait(300)
      ctrl.abort()
      await res?.text().catch(() => undefined)
      ok++
    } else if (res) {
      if (res.status >= 500) fivexx++
      // Drain the SSE so the turn finishes and cleanup runs.
      await res.text().catch(() => undefined)
      if (res.status === 200) ok++
    }
  }
  return { ok, fivexx }
}

async function llmFamily(): Promise<void> {
  console.log('\n=== llm sidecar ===')
  const tmpBefore = tmpEntries()

  // Phase A: openai-compat turns (150) + opencode tool-call turns (15).
  const sA = await llmSidecar({ scenario: 'tool-call' })
  const pidA = sA.proc.pid
  let harness: LlmHarness | null = null
  try {
    if (pidA === undefined) throw new Error('llm sidecar A has no pid')
    harness = await LlmHarness.connect({ baseUrl: sA.url })
    // Warm up a couple of turns so lazy module init settles.
    await harness.chat({ script: [{ text: 'warm' }], token: TOKEN, timeoutMs: 20000 })
    // ALSO warm the opencode path: its first turns do fs work that makes libuv create its
    // lazily-started threadpool (up to UV_THREADPOOL_SIZE=4 threads, once, never again).
    // Taking the baseline before that made the "thread delta <= 2" bound depend on how many
    // pool threads the machine had already started (CI measured 4). A thread LEAK still
    // shows up as growth across the 165 measured turns.
    await runOpencodeTurns(sA, 3, false)
    const baseA = readProcStats(pidA)
    const baseDiagA = await diagOf(sA.url, TOKEN)
    const cpu0 = baseA.cpuSeconds

    const compat = await openaiCompatTurns(harness, 150)
    const oc = await runOpencodeTurns(sA, 15, false)

    await wait(1000)
    const afterA = readProcStats(pidA)
    const afterDiagA = await diagOf(sA.url, TOKEN)
    const dA = fdThreadDelta(baseA, afterA)
    const cpu1 = afterA.cpuSeconds
    const cpuPerReq = cpu0 !== null && cpu1 !== null ? (cpu1 - cpu0) / (150 + 15) : null
    const desc = descendantsOf(pidA)

    const measuredA: string[][] = [
      ['fd delta', String(dA.fd), '≤ 2', dA.fd !== null && dA.fd <= 2 ? 'PASS' : 'FAIL'],
      ['thread delta', String(dA.threads), '≤ 2', dA.threads !== null && dA.threads <= 2 ? 'PASS' : 'FAIL'],
      ['child-process delta', String(dA.children), '= 0', dA.children === 0 ? 'PASS' : 'FAIL'],
      ['RSS growth (KB)', String(dA.rssKb), `≤ ${LLM_RSS_BOUND_KB}`, dA.rssKb !== null && dA.rssKb <= LLM_RSS_BOUND_KB ? 'PASS' : 'FAIL'],
      ['CPU s/request', cpuPerReq === null ? 'null' : cpuPerReq.toFixed(5), 'info', 'INFO'],
      ['requests_total delta', String(baseDiagA && afterDiagA ? Number(afterDiagA.requests_total) - Number(baseDiagA.requests_total) : null), '> 0', 'INFO'],
    ]
    printTable(['llm metric (phase A)', 'measured', 'bound', 'verdict'], measuredA)

    record('llm: 150 fake-provider turns completed', '150', String(compat.turned), compat.turned === 150)
    record('llm: 15 opencode tool-call turns completed', '15', String(oc.ok), oc.ok === 15)
    record('llm: no fake-provider fetch errors', '0', String(compat.fivexx), compat.fivexx === 0)
    record('llm: no opencode 5xx', '0', String(oc.fivexx), oc.fivexx === 0)
    record('llm A: diag.pending_asks = 0', '0', String(afterDiagA?.pending_asks), afterDiagA?.pending_asks === 0)
    record('llm A: diag.mcp_sessions = 0', '0', String(afterDiagA?.mcp_sessions), afterDiagA?.mcp_sessions === 0)
    record('llm A: diag.active_turns = 0', '0', String(afterDiagA?.active_turns), afterDiagA?.active_turns === 0)
    record('llm A: diag.tracked_children = 0', '0', String(afterDiagA?.tracked_children), afterDiagA?.tracked_children === 0)
    record('llm A: diag.open_session_dirs = 0', '0', String(afterDiagA?.open_session_dirs), afterDiagA?.open_session_dirs === 0)
    record('llm A: fd delta ≤ 2', '≤ 2', String(dA.fd), dA.fd !== null && dA.fd <= 2)
    record('llm A: thread delta ≤ 2', '≤ 2', String(dA.threads), dA.threads !== null && dA.threads <= 2)
    record('llm A: child-process delta = 0', '0', String(dA.children), dA.children === 0)
    record(`llm A: RSS growth ≤ ${LLM_RSS_BOUND_KB} KB`, 'bounded', `${String(dA.rssKb)} KB`, dA.rssKb !== null && dA.rssKb <= LLM_RSS_BOUND_KB)
    record('llm A: no descendant processes', '0', String(desc.size), desc.size === 0)
  } finally {
    if (harness) await harness.stop()
    await gracefulStop(sA.proc, 5000)
  }

  // Phase B: opencode silent + client abort (15 turns).
  const sB = await llmSidecar({ scenario: 'silent' })
  const pidB = sB.proc.pid
  try {
    if (pidB === undefined) throw new Error('llm sidecar B has no pid')
    await wait(300)
    const baseB = readProcStats(pidB)
    const oc = await runOpencodeTurns(sB, 15, true)
    await wait(1500)
    const afterB = readProcStats(pidB)
    const afterDiagB = await diagOf(sB.url, TOKEN)
    const dB = fdThreadDelta(baseB, afterB)
    const desc = descendantsOf(pidB)
    record('llm B: 15 opencode aborts handled', '15', String(oc.ok), oc.ok === 15)
    record('llm B: diag.pending_asks = 0', '0', String(afterDiagB?.pending_asks), afterDiagB?.pending_asks === 0)
    record('llm B: diag.mcp_sessions = 0', '0', String(afterDiagB?.mcp_sessions), afterDiagB?.mcp_sessions === 0)
    record('llm B: diag.active_turns = 0', '0', String(afterDiagB?.active_turns), afterDiagB?.active_turns === 0)
    record('llm B: diag.tracked_children = 0', '0', String(afterDiagB?.tracked_children), afterDiagB?.tracked_children === 0)
    record('llm B: diag.open_session_dirs = 0', '0', String(afterDiagB?.open_session_dirs), afterDiagB?.open_session_dirs === 0)
    record('llm B: child-process delta = 0', '0', String(dB.children), dB.children === 0)
    record('llm B: no descendant processes', '0', String(desc.size), desc.size === 0)
  } finally {
    await gracefulStop(sB.proc, 5000)
  }

  // No leftover spinoml-opencode-* temp dirs from either phase.
  const leftover = readdirSync(tmpdir()).filter((n) => n.startsWith('spinoml-opencode-') && !tmpBefore.has(n))
  record('llm: no leftover spinoml-opencode-* dirs', '0', String(leftover.length), leftover.length === 0)
  if (leftover.length > 0) console.log('  leftover:', leftover.slice(0, 5).join(', '))
}

// ── main ──────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  console.log('phase 51: resource leak testing')
  const started = Date.now()
  await torchFamily()
  await llmFamily()

  printTable(['check', 'expected', 'got', 'verdict'], rows.map((r) => [r.case, r.expected, r.got, r.pass ? 'PASS' : 'FAIL']))
  const failed = rows.filter((r) => !r.pass)
  console.log(`\n${rows.length - failed.length}/${rows.length} rows passed in ${((Date.now() - started) / 1000).toFixed(1)}s`)
  if (failed.length > 0) {
    console.log('\nFAILURES:')
    for (const f of failed) console.log(`  ✗ ${f.case}: expected ${f.expected}, got ${f.got}`)
    process.exitCode = 1
  }
}

void main()
