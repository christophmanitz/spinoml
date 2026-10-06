// Phase 11/12 (hardened) — Torch sidecar robustness + crash recovery.
//
// Phase 11 (robustness): startup, shutdown, restart, malformed request,
//   missing fields, invalid dtype, invalid shape, unknown layer, invalid
//   graph, timeout, internal exception, structured errors on every path.
// Phase 12 (crash recovery): kill the sidecar mid-flight → app-side caller
//   must see a clear "unavailable" signal (not stale data), and a restart
//   must bring the service back to a fully working state.
//
// The harness is ISOLATED: it always spawns its own sidecar instance on a
// dedicated test port with SPINOML_TORCH_TIMEOUT=2 (so the slow-client cap
// is testable in seconds, not 30). It never reuses an instance on the
// default 7421 and kills its child on exit. There is deliberately no mock:
// these are the real HTTP/JSON wire semantics a browser/trainer sees.
//
// Run: npm run test:robustness

import { spawn, type ChildProcess } from 'node:child_process'
import { setTimeout as wait } from 'node:timers/promises'
import net from 'node:net'

const PORT = Number(process.env.SPINOML_TEST_PORT ?? '7441')
const SIDECAR = `http://127.0.0.1:${PORT}`
const T = 2.5 // seconds idle-tolerant timeout for a stalled connection (sidecar runs with 2s cap)

type Body = Record<string, unknown>

let failed = 0
let passed = 0

function ok(name: string, cond: boolean, detail = '') {
  if (cond) {
    passed++
    console.log(`  ✓ ${name}`)
  } else {
    failed++
    console.log(`  ✗ ${name}${detail ? ` — ${detail}` : ''}`)
  }
}

function isErrBody(b: Body): b is Body & { ok: false; error: string; error_code: string } {
  return b.ok === false && typeof b.error === 'string' && b.error.length > 0 && typeof b.error_code === 'string'
}

async function isUp(): Promise<boolean> {
  try {
    const r = await fetch(`${SIDECAR}/health`)
    return r.ok
  } catch {
    return false
  }
}

async function post(path: string, body: unknown, signal?: AbortSignal) {
  const res = await fetch(`${SIDECAR}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: typeof body === 'string' ? body : JSON.stringify(body),
    signal,
  })
  return { status: res.status, json: (await res.json()) as Body }
}

function spawnSidecar(): ChildProcess {
  const child = spawn('python', ['sidecar-torch/main.py'], {
    stdio: ['ignore', 'pipe', 'pipe'],
    env: { ...process.env, SPINOML_TORCH_PORT: String(PORT), SPINOML_TORCH_TIMEOUT: '2' },
  })
  return child
}

async function waitUp(child: ChildProcess, deadlineMs: number): Promise<boolean> {
  const t0 = Date.now()
  while (Date.now() - t0 < deadlineMs) {
    if (child.exitCode !== null) return false
    if (await isUp()) return true
    await wait(200)
  }
  return false
}

// ── valid + invalid code payloads ────────────────────────────────────────
// exec namespace only gets __name__; the generated code imports torch itself,
// so these payloads mirror what generator.ts emits.
const VALID_LINEAR = `import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)
    def forward(self, x):
        return self.fc(x)
`

// Referencing the bogus layer at MODULE level fails during exec → COMPILE stage
// (a layer-name typo the generator would emit at import time).
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

// Undefined name at module level → exec fails to resolve the graph symbol →
// COMPILE stage (shapes for "invalid graph" that can't even be built).
const INVALID_GRAPH = `import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)
    def forward(self, x):
        return self.fc(x)
Model = missing_symbol
`

const CONSTRUCT_FAIL = `import torch
import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, explosion())
    def forward(self, x):
        return self.fc(x)
`

async function main() {
  // Refuse to run if the test port is occupied by something else — we must own
  // the instance to safely kill/restart it.
  if (await isUp()) {
    console.error(`test port ${PORT} already serving a sidecar — not touching it`)
    process.exit(1)
  }

  let child = spawnSidecar()
  ok('startup: sidecar spawns and becomes healthy', await waitUp(child, 8000))
  if (child.exitCode === null) {
    ok('startup: process stays alive', await isUp())
  }

  // ── Phase 11 — structured errors ──────────────────────────────────────
  console.log('\n— malformed / missing-field requests (HTTP 400) —')
  {
    const r = await post('/infer', '{this is not json')
    ok('malformed JSON → 400', r.status === 400, `status=${r.status}`)
    ok('malformed JSON → ok:false', r.json.ok === false)
    ok('malformed JSON → VALIDATION code', r.json.error_code === 'VALIDATION', `code=${r.json.error_code}`)
    ok('malformed JSON → non-empty error', typeof r.json.error === 'string' && r.json.error.length > 0)
  }
  {
    const r = await post('/infer', {})
    ok('missing everything → 400 + VALIDATION', r.status === 400 && r.json.error_code === 'VALIDATION')
  }
  {
    const r = await post('/infer', { input_shapes: [[1, 8]] })
    ok('missing code → 400 + VALIDATION', r.status === 400 && r.json.error_code === 'VALIDATION')
  }
  {
    const r = await post('/infer', { code: VALID_LINEAR })
    ok('missing shapes → 400 + VALIDATION', r.status === 400 && r.json.error_code === 'VALIDATION')
  }
  {
    const r = await post('/dataset/inspect', {})
    ok('dataset/inspect missing abspath → 400 + VALIDATION', r.status === 400 && r.json.error_code === 'VALIDATION')
  }
  {
    const r = await post('/deps/check', {})
    ok('deps/check missing specs → 400 + VALIDATION', r.status === 400 && r.json.error_code === 'VALIDATION')
  }
  {
    const r = await post('/does-not-exist', {})
    ok('unknown POST endpoint → 404 + UNKNOWN_ENDPOINT', r.status === 404 && r.json.error_code === 'UNKNOWN_ENDPOINT')
  }
  {
    const r = await fetch(`${SIDECAR}/does-not-exist`)
    const j = (await r.json()) as Body
    ok('unknown GET endpoint → 404 + UNKNOWN_ENDPOINT', r.status === 404 && j.error_code === 'UNKNOWN_ENDPOINT')
  }

  console.log('\n— invalid dtype / shape (HTTP 400) —')
  {
    const r = await post('/infer', { code: VALID_LINEAR, input_shapes: [[1, 8]], input_dtypes: ['float99'] })
    ok('unknown dtype → 400 + VALIDATION', r.status === 400 && r.json.error_code === 'VALIDATION',
       `code=${r.json.error_code}`)
  }
  {
    const r = await post('/infer', { code: VALID_LINEAR, input_shapes: [[1, 8]], input_dtypes: ['float32', 'int64'] })
    ok('dtype count mismatch → 400 + VALIDATION', r.status === 400 && r.json.error_code === 'VALIDATION')
  }
  {
    const r = await post('/infer', { code: VALID_LINEAR, input_shapes: [['a', 8]] })
    ok('non-int dim → 400 + VALIDATION (no crash)', r.status === 400 && r.json.error_code === 'VALIDATION',
       `status=${r.status}`)
  }
  {
    const r = await post('/infer', { code: VALID_LINEAR, input_shapes: [[1.5, 8]] })
    ok('float dim → 400 + VALIDATION', r.status === 400 && r.json.error_code === 'VALIDATION')
  }
  {
    const r = await post('/infer', { code: VALID_LINEAR, input_shapes: [[-1, 8]] })
    ok('negative dim → 400 + VALIDATION', r.status === 400 && r.json.error_code === 'VALIDATION')
  }
  {
    const r = await post('/infer', { code: VALID_LINEAR, input_shapes: [[]] })
    ok('empty shape → 400 + VALIDATION', r.status === 400 && r.json.error_code === 'VALIDATION')
  }
  {
    const r = await post('/infer', { code: VALID_LINEAR, input_shapes: [[1_000_000_000]] })
    ok('pathological (4B-element) shape → 400 + VALIDATION, no OOM', r.status === 400 && r.json.error_code === 'VALIDATION',
       `status=${r.status}`)
  }

  console.log('\n— business-logic errors (HTTP 200, ok:false, error_code set) —')
  {
    const r = await post('/infer', { code: UNKNOWN_LAYER, input_shapes: [[1, 8]] })
    ok('unknown layer/module → COMPILE code', r.status === 200 && r.json.error_code === 'COMPILE',
       `status=${r.status} code=${r.json.error_code}`)
    ok('unknown layer → error mentions layer', String(r.json.error).includes('BogusLayer'), String(r.json.error).slice(0, 80))
  }
  {
    const r = await post('/infer', { code: INVALID_GRAPH, input_shapes: [[1, 8]] })
    ok('invalid graph → COMPILE code', r.status === 200 && r.json.error_code === 'COMPILE',
       `status=${r.status}`)
    ok('invalid graph → clear error + trace', typeof r.json.trace === 'string' && r.json.trace.length > 0)
  }
  {
    const r = await post('/infer', { code: CONSTRUCT_FAIL, input_shapes: [[1, 8]] })
    ok('construct failure → CONSTRUCT code', r.status === 200 && r.json.error_code === 'CONSTRUCT',
       `status=${r.status} code=${r.json.error_code}`)
  }
  {
    const r = await post('/infer', { code: VALID_LINEAR, input_shapes: [[1, 16]] })
    ok('forward mismatch → FORWARD code', r.status === 200 && r.json.error_code === 'FORWARD',
       `status=${r.status} code=${r.json.error_code}`)
    ok('forward mismatch → error explains shape', /linear|matmul|size|expected|shape/i.test(String(r.json.error)),
       String(r.json.error).split('\n')[0])
  }

  // Also exercise /activations with the same validation gate.
  {
    const r = await post('/activations', { code: VALID_LINEAR, input_shapes: [[1.5, 8]] })
    ok('activations float dim → 400 + VALIDATION', r.status === 400 && r.json.error_code === 'VALIDATION')
  }

  console.log('\n— valid request still works after all the abuse —')
  {
    const r = await post('/infer', { code: VALID_LINEAR, input_shapes: [[1, 8]], input_dtypes: ['float32'] })
    const fc = r.json.shapes as Record<string, number[]> | undefined
    const out = fc?.['__output__'] ?? fc?.['fc']
    ok('valid model → ok:true', r.status === 200 && r.json.ok === true, `status=${r.status}`)
    ok('valid model → output shape present', !!out, `shapes=${JSON.stringify(fc)}`)
    ok('valid model → output shape [1,4]', out?.join(',') === '1,4', `got ${out}`)
    ok('valid model → n_params reported', typeof r.json.n_params === 'number' && r.json.n_params > 0)
  }

  // ── Phase 11 — timeout / slow-client semantics ────────────────────────
  console.log('\n— timeout: stalled request body must not pin a worker thread —')
  {
    const sock = await openStall()
    ok('stalled body connection accepted', sock !== null)
    await wait(1200)
    // Server must still answer while the stalled socket is held open.
    ok('health still answers mid-stall', await isUp())
    // The sidecar's inactivity cap (2s) should make the SERVER close the
    // stalled connection, proving the worker thread returned to the pool
    // instead of wedging the server.
    const closedByServer = await socketClosedByPeer(sock, 6000)
    ok('stalled connection closed server-side after cap', closedByServer, 'socket stayed open')
    if (sock) sock.destroy()
    ok('server still responsive after stall cap', await isUp())
    const r = await post('/infer', { code: VALID_LINEAR, input_shapes: [[1, 8]] })
    ok('valid request serves after stall', r.status === 200 && r.json.ok === true)
  }

  console.log('\n— timeout: client abort must not kill the sidecar or corrupt state —')
  {
    // Fire several requests and abort each immediately — outcome is irrelevant;
    // what matters is the sidecar survives the mid-flight disconnects.
    for (let i = 0; i < 4; i++) {
      const ac = new AbortController()
      void post('/infer', { code: VALID_LINEAR, input_shapes: [[1, 8]] }, ac.signal).catch(() => {})
      setTimeout(() => ac.abort(), 1)
    }
    await wait(300)
    ok('sidecar healthy after aborts', await isUp())
    const r2 = await post('/infer', { code: VALID_LINEAR, input_shapes: [[1, 8]] })
    ok('valid request succeeds after aborted neighbors', r2.status === 200 && r2.json.ok === true
       && (r2.json.shapes as Record<string, number[]>)?.fc?.join(',') === '1,4')
  }

  // ── Phase 12 — crash recovery ──────────────────────────────────────────
  console.log('\n— crash recovery: kill → detect unavailable → restart → recover —')
  {
    ok('sidecar alive before kill', await isUp())
    void killAfter(child, 250)
    // App-side caller behavior: a request must report unavailable, NOT stale data.
    let sawDown = 0
    const t0 = Date.now()
    while (Date.now() - t0 < 4000) {
      const up = await isUp()
      if (!up) sawDown++
      if (sawDown >= 2) break
      await wait(150)
    }
    ok('sidecar becomes unreachable after kill', sawDown >= 1, `unreachable-heartbeats=${sawDown}`)
    const tryStatus = await fetch(`${SIDECAR}/infer`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code: VALID_LINEAR, input_shapes: [[1, 8]] }),
    }).then((r) => r.status).catch(() => 0)
    ok('request against dead sidecar fails cleanly (no silent stale result)', tryStatus === 0 && !(await isUp()),
       `status=${tryStatus}`)

    // Restart: a fresh process on the same port must serve again.
    child = spawnSidecar()
    ok('restart: sidecar comes back healthy', await waitUp(child, 8000))
    const r = await post('/infer', { code: VALID_LINEAR, input_shapes: [[1, 8]] })
    const fc = r.json.shapes as Record<string, number[]> | undefined
    ok('restart: valid infer recovers fully', r.status === 200 && r.json.ok === true && fc?.['fc']?.join(',') === '1,4',
       `status=${r.status} shapes=${JSON.stringify(fc)}`)
  }

  // ── shutdown ─────────────────────────────────────────────────────────
  console.log('\n— shutdown —')
  ok('clean SIGTERM graceful shutdown', await gracefulStop(child), 'exit code non-zero/unexpected')
  await wait(300)
  ok('health fails after shutdown', !(await isUp()))

  // ── Phase 50: ESPF vocab fallback is truthful when the codebook is missing ──
  console.log('\n— ESPF vocab fallback (missing codebook) is truthful —')
  {
    const { spawnSync } = await import('node:child_process')
    const script = [
      'import sys, tempfile',
      'from pathlib import Path',
      "sys.path.insert(0, 'sidecar-torch')",
      'import dataset_handlers as dh',
      'dh.ESPF_DIR = Path(tempfile.mkdtemp())  # no codebook files here',
      'dh._ESPF_CACHE.clear()',
      "spec = {'kind': 'espf', 'codebook': 'drug'}",
      'got = dh.espf_vocab_size(spec)',
      "want = dh.seq_vocab_size({**spec, 'vocab': 'smiles'})",
      'assert got == want, (got, want)',
      "t = dh.tokenize_espf('CCOCC', spec)",
      'assert int(t.max()) < want, (int(t.max()), want)',
      "print('OK', got)",
    ].join('\n')
    const r = spawnSync('python', ['-c', script], { cwd: process.cwd(), encoding: 'utf8' })
    ok('missing ESPF codebook → vocab matches the char-level fallback',
       r.status === 0 && r.stdout.includes('OK'), (r.stderr || r.stdout || '').slice(0, 240))
  }

  console.log(`\n${failed === 0 ? '✓' : '✗'} ${passed} checks passed, ${failed} failed`)
  process.exit(failed === 0 ? 0 : 1)
}

// ── helpers ──────────────────────────────────────────────────────────────

async function socketClosedByPeer(sock: net.Socket | null, timeoutMs: number): Promise<boolean> {
  if (!sock) return false
  return new Promise((resolve) => {
    let done = false
    const finish = (v: boolean) => {
      if (done) return
      done = true
      resolve(v)
    }
    sock.once('end', () => finish(true))
    sock.once('close', () => finish(true))
    sock.once('error', () => finish(false))
    setTimeout(() => finish(false), timeoutMs)
  })
}

async function openStall(): Promise<net.Socket | null> {
  return new Promise((resolve) => {
    const sock = net.connect(PORT, '127.0.0.1', () => {
      // Send headers claiming a huge body, then stay silent — server-side
      // inactivity cap should eventually free the worker thread.
      sock.write(
        'POST /infer HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: 999999\r\n\r\npart',
      )
      resolve(sock)
    })
    sock.on('error', () => resolve(null))
  })
}

function killAfter(child: ChildProcess, ms: number): Promise<ChildProcess> {
  return new Promise((resolve) => {
    setTimeout(() => {
      child.kill('SIGKILL')
      child.on('exit', () => resolve(child))
    }, ms)
  })
}

async function gracefulStop(child: ChildProcess): Promise<boolean> {
  const exited = new Promise<boolean>((resolve) => {
    child.on('exit', () => resolve(true))
    setTimeout(() => resolve(false), 4000)
  })
  child.kill('SIGINT')
  return exited
}

void main()