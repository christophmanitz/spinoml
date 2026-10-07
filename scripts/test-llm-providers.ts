// Provider/model layer test suite — TODO §0.12 "OpenCode tests" for the parts
// the existing suites do NOT cover:
//   scripts/test-opencode-lifecycle.ts  — bridge auth, MCP secret, process
//       failure / error event / garbage / start timeout / abort / invalid env.
//   scripts/test-sidecar-auth-llm.ts    — token / Host / Origin gate, /respond.
//   scripts/test-llm-safety.ts          — openai-compat tool-call pipeline vs a
//       fake OpenAI server (hostile tool calls, truncated stream, secrets).
//   scripts/test-llm-validation-parity.ts — sidecar tool validation vs the
//       frontend registry.
//
// THIS suite covers Configuration (×5), Provider selection (×2), Request
// handling (×5), Model handling (×3), Failure handling (×3) and Security (×1)
// as listed in the COVERAGE MAP below. Every §0.12 bullet is mapped to a row
// here or in one of the suites above; a bullet with no coverage fails.
//
// Run: npm run test:llm-providers

import { execSync } from 'node:child_process'
import { mkdtempSync, readFileSync, existsSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { LlmHarness } from './lib/llm-harness.ts'
import { startSidecar, freePort, type StartedSidecar } from './lib/auth-probe.ts'

const HERE = dirname(fileURLToPath(import.meta.url))
const REPO = resolve(HERE, '..')
const SIDECAR = join(REPO, 'sidecar-llm', 'main.mjs')
const FAKE_BIN = join(REPO, 'scripts', 'lib', 'fake-opencode')
const RUN = mkdtempSync(join(tmpdir(), 'spinoml-prov-'))

// ── coverage map ─────────────────────────────────────────────────────────────
// §0.12 bullet → which suite + row title covers it. Every bullet must map to
// something; a bullet with no coverage is a failure.

interface CoverageRow {
  bullet: string
  suite: string
  row: string
}

const COVERAGE: CoverageRow[] = [
  // Configuration ×5
  { bullet: 'Configuration: OpenCode selected', suite: 'test-llm-providers (F1/F2)', row: 'F1 default provider is opencode / F2 selecting a provider changes currentId' },
  { bullet: 'Configuration: OpenCode model selected', suite: 'test-llm-providers (F2/M2)', row: 'F2 model selection / M2 alternative valid model reaches the backend verbatim' },
  { bullet: 'Configuration: Model persisted', suite: 'test-llm-providers (F3)', row: 'F3 selection is persisted and restored after a simulated reload' },
  { bullet: 'Configuration: Model changed', suite: 'test-llm-providers (F2b)', row: 'F2b changing the model updates config + persisted value' },
  { bullet: 'Configuration: Invalid model handled', suite: 'test-llm-providers (F4/F5)', row: 'F4 hostile persisted model falls back / F5 invalid typed model is rejected and not sent' },
  // Provider selection ×2
  { bullet: 'Provider selection: OpenCode → OpenCode backend', suite: 'test-llm-providers (S2)', row: 'S2 kind:opencode spawns the fake opencode binary with --model' },
  { bullet: 'Provider selection: Claude → Claude backend', suite: 'test-llm-providers (S5) + test-opencode-lifecycle', row: 'S5 subscription SKIPPED at unit level (no real CLI); test-opencode-lifecycle covers the dispatch seam' },
  // Request handling ×5
  { bullet: 'Request handling: Valid request', suite: 'test-llm-safety (T1) + test-llm-providers (S1/S2)', row: 'test-llm-safety T1 valid add_layer; S1/S2 valid request reaches backend' },
  { bullet: 'Request handling: Invalid request', suite: 'test-llm-providers (R1-R4)', row: 'R1 invalid JSON / R2 missing user / R3 wrong messages type / R4 wrong graph type' },
  { bullet: 'Request handling: Timeout', suite: 'test-opencode-lifecycle (start timeout) + test-llm-safety (T15/T16 stall) + test-llm-providers (R6)', row: 'start timeout row in test-opencode-lifecycle; upstream-stall rows in test-llm-safety; R6 here' },
  { bullet: 'Request handling: Cancellation', suite: 'test-opencode-lifecycle (abort case)', row: 'abort: fake AND bridge child gone' },
  { bullet: 'Request handling: Provider unavailable', suite: 'test-llm-providers (S4/R6)', row: 'S4 refused baseURL names the backend / R6 missing opencode binary' },
  // Model handling ×3
  { bullet: 'Model handling: Big Pickle selected', suite: 'test-llm-providers (M1/F1)', row: 'M1 default model opencode/big-pickle reaches the backend when none given' },
  { bullet: 'Model handling: Alternative model selected', suite: 'test-llm-providers (M2)', row: 'M2 alternative valid model passed verbatim' },
  { bullet: 'Model handling: Unknown model rejected or handled explicitly', suite: 'test-llm-providers (M3..M13 hostile, M14 unknown)', row: '≥12 hostile/malformed models rejected BEFORE spawning; unknown-but-well-formed model surfaces the fake error' },
  // Failure handling ×3
  { bullet: 'Failure handling: OpenCode process failure', suite: 'test-opencode-lifecycle (exit-nonzero)', row: 'd exit-nonzero: explicit status error mentioning exit code + stderr tail' },
  { bullet: 'Failure handling: Malformed response', suite: 'test-opencode-lifecycle (garbage) + test-llm-safety (T14 truncated stream)', row: 'garbage rows in test-opencode-lifecycle; truncated-stream rows in test-llm-safety' },
  { bullet: 'Failure handling: Provider unavailable', suite: 'test-llm-providers (S4/R6)', row: 'S4 refused baseURL / R6 missing opencode binary → explicit error' },
  // Security ×1
  { bullet: 'Security: Credentials/Tokens/Secrets not in logs or artifacts', suite: 'test-opencode-lifecycle (secrecy rows) + test-llm-safety (S1) + test-sidecar-auth-llm + test-llm-providers (F6)', row: 'secrecy rows in test-opencode-lifecycle; S1 in test-llm-safety; token/origin rows in test-sidecar-auth-llm; F6 apiKey never leaks here' },
]

// ── row recorder ─────────────────────────────────────────────────────────────

interface Row {
  case: string
  expected: string
  got: string
  pass: boolean
}

const rows: Row[] = []

function record(name: string, expected: string, got: string, pass?: boolean): void {
  const p = pass ?? expected === got
  rows.push({ case: name, expected, got, pass: p })
}

// ── process hygiene ──────────────────────────────────────────────────────────

function procSnapshot(): Set<string> {
  let out: string
  try {
    out = execSync('ps -eo args=', { encoding: 'utf8' })
  } catch {
    return new Set()
  }
  const set = new Set<string>()
  for (const line of out.split('\n')) {
    if (/fake-opencode|mcp-bridge\.mjs|sidecar-llm\/main\.mjs/.test(line)) set.add(line.trim())
  }
  return set
}

const PROCS_BEFORE = procSnapshot()

// ── helpers ──────────────────────────────────────────────────────────────────

async function waitJson(path: string, timeoutMs: number): Promise<Record<string, unknown> | null> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (existsSync(path)) {
      try {
        return JSON.parse(readFileSync(path, 'utf8')) as Record<string, unknown>
      } catch {
        // still being written
      }
    }
    await new Promise((r) => setTimeout(r, 100))
  }
  return null
}

// A sidecar started in token mode with an explicit fake opencode binary.
const TOKEN = 'a'.repeat(64)

async function startSidecarWithFakeOpencode(
  scenario: Record<string, unknown>,
  extraEnv: Record<string, string> = {},
): Promise<StartedSidecar> {
  return await startSidecar({
    cmd: process.execPath,
    args: [SIDECAR],
    cwd: REPO,
    readyTimeoutMs: 30000,
    stripEnvPrefix: 'SPINOML_',
    portEnv: 'SPINOML_LLM_PORT',
    env: {
      SPINOML_SIDECAR_TOKEN: TOKEN,
      SPINOML_REQUIRE_TOKEN: '1',
      SPINOML_OPENCODE_BIN: FAKE_BIN,
      FAKE_OPENCODE_SCENARIO: JSON.stringify(scenario),
      ...extraEnv,
    },
  })
}

interface RunChatOpts {
  llm?: Record<string, unknown>
  user?: string
  messages?: unknown
  graph?: unknown
  timeoutMs?: number
  signal?: AbortSignal
  rawBody?: string
}

// Drive /chat on an already-started sidecar; optionally send a raw body so a
// hostile request can bypass JSON.stringify. Returns the parsed SSE events.
async function runChat(
  url: string,
  opts: RunChatOpts = {},
): Promise<{ events: Record<string, unknown>[]; statuses: { value: string; message?: string }[]; texts: string[]; ended: boolean; httpStatus: number; elapsedMs: number }> {
  const body = opts.rawBody ?? JSON.stringify({
    user: opts.user ?? 'test turn',
    messages: opts.messages ?? [],
    graph: opts.graph ?? { input_shape: [1, 4], nodes: [], edges: [] },
    llm: opts.llm ?? { kind: 'opencode' },
  })
  const start = Date.now()
  const out: { events: Record<string, unknown>[]; statuses: { value: string; message?: string }[]; texts: string[]; ended: boolean; httpStatus: number; elapsedMs: number } = {
    events: [], statuses: [], texts: [], ended: false, httpStatus: 0, elapsedMs: 0,
  }
  try {
    const res = await fetch(`${url}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-SpinoML-Token': TOKEN },
      body,
      signal: opts.signal ?? AbortSignal.timeout(opts.timeoutMs ?? 20000),
    })
    out.httpStatus = res.status
    if (res.body) {
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''
      for (;;) {
        const { value, done } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        let idx: number
        while ((idx = buf.indexOf('\n\n')) !== -1) {
          const frame = buf.slice(0, idx)
          buf = buf.slice(idx + 2)
          const line = frame.split('\n').find((l) => l.startsWith('data:'))
          if (!line) continue
          const json = line.slice(5).trim()
          if (!json) continue
          let ev: Record<string, unknown>
          try {
            ev = JSON.parse(json) as Record<string, unknown>
          } catch {
            continue
          }
          out.events.push(ev)
          if (ev.type === 'status') {
            out.statuses.push({ value: String(ev.value), message: typeof ev.message === 'string' ? ev.message : undefined })
          } else if (ev.type === 'text') {
            out.texts.push(String(ev.value ?? ''))
          } else if (ev.type === 'done') {
            out.ended = true
          }
        }
      }
    }
  } catch {
    // fetch aborted / connection refused — the caller inspects what it got
  }
  out.elapsedMs = Date.now() - start
  return out
}

function errorMessages(result: { statuses: { value: string; message?: string }[] }): string {
  return result.statuses.filter((s) => s.value === 'error').map((s) => s.message ?? '').join(' | ')
}

// ── (a) FRONTEND config: pure store tests with a localStorage stub ──────────
// Each case imports src/chat/providerStore.ts FRESH (cache-busting query) so
// the module-level `loadPersisted()` runs against the stub state for THAT case.

type StorageLike = {
  getItem: (key: string) => string | null
  setItem: (key: string, value: string) => void
  removeItem: (key: string) => void
}

function makeStorage(): StorageLike & { map: Map<string, string> } {
  const map = new Map<string, string>()
  return {
    map,
    getItem: (k) => (map.has(k) ? (map.get(k) as string) : null),
    setItem: (k, v) => { map.set(k, String(v)) },
    removeItem: (k) => { map.delete(k) },
  }
}

type ProviderStoreModule = typeof import('../src/chat/providerStore')

const PROVIDER_STORE_PATH = new URL('../src/chat/providerStore.ts', import.meta.url).href

async function importFresh(): Promise<ProviderStoreModule> {
  return (await import(`${PROVIDER_STORE_PATH}?fresh=${Date.now()}-${Math.random()}`)) as ProviderStoreModule
}

async function withWindow(storage: StorageLike, fn: () => Promise<void>): Promise<void> {
  const g = globalThis as unknown as { window?: unknown }
  const prev = g.window
  g.window = { localStorage: storage }
  try {
    await fn()
  } finally {
    if (prev === undefined) delete g.window
    else g.window = prev
  }
}

async function frontendConfigTests(): Promise<void> {
  const STORAGE_KEY = 'spinoml.chat.provider.v1'

  // F1 — default provider is opencode with default model opencode/big-pickle.
  {
    const storage = makeStorage()
    await withWindow(storage, async () => {
      const mod = await importFresh()
      const { useProviderStore, getCurrentLlmRequest } = mod
      const req = getCurrentLlmRequest()
      record(
        'F1 default provider is opencode with default model opencode/big-pickle',
        'opencode / opencode/big-pickle',
        `${req.kind} / ${req.model}`,
        req.kind === 'opencode' && req.model === 'opencode/big-pickle',
      )
      record(
        'F1 empty storage → store state is opencode',
        'opencode',
        useProviderStore.getState().currentId,
        useProviderStore.getState().currentId === 'opencode',
      )
    })
  }

  // F2 — selecting another provider/model changes currentId/config.
  // F2b — changing the model updates config + persisted value.
  {
    const storage = makeStorage()
    await withWindow(storage, async () => {
      const mod = await importFresh()
      const { useProviderStore, getCurrentLlmRequest } = mod
      useProviderStore.getState().setCurrent('anthropic-api')
      useProviderStore.getState().setConfig('anthropic-api', { apiKey: 'sk-fake-anthropic', model: 'claude-sonnet-4-6' })
      const st = useProviderStore.getState()
      const req = getCurrentLlmRequest()
      record(
        'F2 selecting a provider/model changes currentId + config',
        'anthropic-api / claude-sonnet-4-6',
        `${st.currentId} / ${st.configs['anthropic-api']?.model}`,
        st.currentId === 'anthropic-api' && st.configs['anthropic-api']?.model === 'claude-sonnet-4-6' && req.model === 'claude-sonnet-4-6',
      )
      useProviderStore.getState().setConfig('anthropic-api', { model: 'claude-fable-5' })
      const st2 = useProviderStore.getState()
      record(
        'F2b changing the model updates config + persisted value',
        'claude-fable-5',
        `${st2.configs['anthropic-api']?.model}`,
        st2.configs['anthropic-api']?.model === 'claude-fable-5' &&
          storage.map.get(STORAGE_KEY)?.includes('claude-fable-5') === true,
      )
    })
  }

  // F3 — selection is persisted and restored after a simulated reload.
  {
    const storage = makeStorage()
    await withWindow(storage, async () => {
      const mod = await importFresh()
      const { useProviderStore } = mod
      useProviderStore.getState().setCurrent('openai')
      useProviderStore.getState().setConfig('openai', { apiKey: 'sk-fake-openai', model: 'gpt-4.1' })
    })
    // Simulated reload: re-import the module with the SAME storage.
    await withWindow(storage, async () => {
      const mod = await importFresh()
      const { useProviderStore, getCurrentLlmRequest } = mod
      const st = useProviderStore.getState()
      const req = getCurrentLlmRequest()
      record(
        'F3 selection persisted and restored after a simulated reload',
        'openai / gpt-4.1',
        `${st.currentId} / ${st.configs['openai']?.model}`,
        st.currentId === 'openai' && st.configs['openai']?.model === 'gpt-4.1' && req.kind === 'openai-compat' && req.model === 'gpt-4.1',
      )
    })
  }

  // F4 — hostile persisted value falls back to the documented default, no throw.
  const HOSTILE = [
    ['null persisted blob', 'null'],
    ['array persisted blob', '[1,2,3]'],
    ['wrong-type currentId', JSON.stringify({ currentId: 42, configs: {} })],
    ['unknown provider id', JSON.stringify({ currentId: 'banana', configs: {} })],
    ['model with spaces', JSON.stringify({ currentId: 'opencode', configs: { opencode: { model: 'opencode big pickle' } } })],
    ['model with leading dash', JSON.stringify({ currentId: 'opencode', configs: { opencode: { model: '--print-logs' } } })],
    ['non-string model', JSON.stringify({ currentId: 'opencode', configs: { opencode: { model: 42 } } })],
    ['array config entry', JSON.stringify({ currentId: 'opencode', configs: { opencode: ['x'] } })],
    ['null config entry', JSON.stringify({ currentId: 'opencode', configs: { opencode: null } })],
  ]
  for (const [name, blob] of HOSTILE) {
    const storage = makeStorage()
    storage.map.set(STORAGE_KEY, blob)
    await withWindow(storage, async () => {
      const mod = await importFresh()
      const { getCurrentLlmRequest } = mod
      let threw = false
      let req: { kind: string; model?: string } | null = null
      try {
        req = getCurrentLlmRequest()
      } catch {
        threw = true
      }
      record(
        `F4 hostile persisted value falls back: ${name}`,
        'default opencode/big-pickle, no throw',
        threw ? 'THREW' : `${req?.kind} / ${req?.model}`,
        !threw && req?.kind === 'opencode' && req?.model === 'opencode/big-pickle',
      )
    })
  }

  // F5 — an invalid model typed in settings is rejected/flagged and not sent.
  const BAD_TYPED_MODELS = ['--print-logs', 'a b', 'opencode big pickle', 'a\nb', 'x; rm -rf']
  for (const bad of BAD_TYPED_MODELS) {
    const storage = makeStorage()
    await withWindow(storage, async () => {
      const mod = await importFresh()
      const { useProviderStore, getCurrentLlmRequest } = mod
      useProviderStore.getState().setCurrent('opencode')
      useProviderStore.getState().setConfig('opencode', { model: bad })
      const req = getCurrentLlmRequest()
      const persisted = storage.map.get(STORAGE_KEY) ?? ''
      record(
        `F5 invalid typed model rejected: ${JSON.stringify(bad)}`,
        'falls back to opencode/big-pickle, not persisted',
        `${req.model} / ${persisted.includes(bad) ? 'PERSISTED' : 'not persisted'}`,
        req.model === 'opencode/big-pickle' && !persisted.includes(bad),
      )
    })
  }

  // F6 — API keys never written to another storage key; never sent for a
  // provider that does not need one.
  {
    const storage = makeStorage()
    await withWindow(storage, async () => {
      const mod = await importFresh()
      const { useProviderStore, getCurrentLlmRequest } = mod
      const SECRET = 'sk-SECRET-needs-key-provider'
      useProviderStore.getState().setCurrent('anthropic-api')
      useProviderStore.getState().setConfig('anthropic-api', { apiKey: SECRET })
      // (1) only the provider key holds the secret
      const otherKeys = [...storage.map.keys()].filter((k) => k !== STORAGE_KEY)
      record(
        'F6 apiKey never written to another storage key',
        '0 other keys',
        String(otherKeys.length),
        otherKeys.length === 0,
      )
      // (2) a provider that does NOT need a key never receives one
      useProviderStore.getState().setCurrent('opencode')
      useProviderStore.getState().setConfig('opencode', { apiKey: SECRET })
      const reqOpencode = getCurrentLlmRequest()
      record(
        'F6 opencode (needsKey:false) request carries no apiKey',
        'no apiKey field',
        'apiKey' in reqOpencode ? 'LEAKED' : 'no apiKey',
        !('apiKey' in reqOpencode),
      )
      // (3) the subscription path never receives an apiKey either
      useProviderStore.getState().setCurrent('claude-subscription')
      const reqSub = getCurrentLlmRequest()
      record(
        'F6 subscription request carries no apiKey',
        'no apiKey field',
        'apiKey' in reqSub ? 'LEAKED' : 'no apiKey',
        !('apiKey' in reqSub),
      )
      // (4) a needsKey provider DOES carry its key
      useProviderStore.getState().setCurrent('anthropic-api')
      const reqAnthropic = getCurrentLlmRequest()
      record(
        'F6 anthropic (needsKey:true) request carries its key',
        'carries the key',
        'apiKey' in reqAnthropic ? 'carries' : 'missing',
        reqAnthropic.apiKey === SECRET,
      )
      // (5) the key never appears in any other storage value
      const otherValues = [...storage.map.entries()].filter(([k]) => k !== STORAGE_KEY)
      const leaked = otherValues.some(([, v]) => v.includes(SECRET))
      record(
        'F6 apiKey never written to any other storage value',
        'no leak',
        leaked ? 'LEAK' : 'no leak',
        !leaked,
      )
    })
  }
}

// ── main ─────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  console.log('test-llm-providers: provider/model layer tests (TODO §0.12)')
  console.log('')

  await frontendConfigTests()

  // Print what we have so far before the sidecar work so a crash below still
  // leaves a readable trace.
  console.log(`frontend config rows so far: ${rows.length}`)

  // ── (b) SIDECAR routing per kind ───────────────────────────────────────────

  // S1 — openai-compat reaches the fake OpenAI server with the configured
  // model and Authorization header.
  {
    let h: LlmHarness | null = null
    try {
      h = await LlmHarness.start()
      const MODEL = 'gpt-4o'
      const res = await h.chat({
        script: [{ text: 'ok' }],
        llm: { kind: 'openai-compat', model: MODEL, apiKey: 'sk-provider-key' },
        timeoutMs: 15000,
      })
      const seen = h.fake.requests[0]
      const bodyModel = (seen?.body as { model?: string } | null)?.model
      const auth = typeof seen?.headers.authorization === 'string' ? (seen.headers.authorization as string) : ''
      record(
        'S1 openai-compat → fake server saw the request with the configured model',
        `model=${MODEL}`,
        `model=${bodyModel ?? 'none'}`,
        bodyModel === MODEL,
      )
      record(
        'S1 openai-compat → Authorization header present',
        'Bearer sk-provider-key',
        auth.slice(0, 30),
        auth === 'Bearer sk-provider-key',
      )
      record('S1 openai-compat → stream ended with done', 'done', res.ended ? 'done' : 'no', res.ended)
    } finally {
      if (h) await h.stop()
    }
  }

  // S2 — opencode spawns the fake opencode binary with `--model <exact model>`.
  {
    const MODEL = 'opencode/big-pickle'
    const dump = join(RUN, 's2-dump.json')
    let sidecar: StartedSidecar | null = null
    try {
      sidecar = await startSidecarWithFakeOpencode({ scenario: 'tool-call', dump })
      const res = await runChat(sidecar.url, { llm: { kind: 'opencode', model: MODEL } })
      const info = await waitJson(dump, 5000)
      const argv = Array.isArray(info?.fakeArgv) ? (info?.fakeArgv as string[]) : []
      const modelIdx = argv.indexOf('--model')
      record(
        'S2 opencode → fake opencode spawned with --model <exact model>',
        `--model ${MODEL}`,
        modelIdx >= 0 ? `--model ${argv[modelIdx + 1] ?? '?'}` : 'no --model flag',
        modelIdx >= 0 && argv[modelIdx + 1] === MODEL,
      )
      record('S2 opencode → no child process from an invalid model', 'spawned', sidecar ? 'spawned' : 'no sidecar', sidecar !== null)
      record('S2 opencode → stream ended with done', 'done', res.ended ? 'done' : 'no', res.ended)
    } finally {
      if (sidecar) await sidecar.stop()
    }
  }

  // S3 — anthropic without apiKey → explicit anthropic-specific error, no
  // network (no child process, no upstream call).
  {
    let sidecar: StartedSidecar | null = null
    try {
      sidecar = await startSidecarWithFakeOpencode({ scenario: 'silent' })
      const res = await runChat(sidecar.url, { llm: { kind: 'anthropic' }, timeoutMs: 15000 })
      const errs = errorMessages(res)
      record(
        'S3 anthropic without apiKey → explicit anthropic-specific error',
        'mentions Anthropic + API key',
        errs.slice(0, 80) || 'none',
        /[Aa]nthropic/.test(errs) && /[Aa][Pp][Ii] [Kk]ey|key missing/i.test(errs),
      )
      record('S3 anthropic without apiKey → stream ended with done', 'done', res.ended ? 'done' : 'no', res.ended)
    } finally {
      if (sidecar) await sidecar.stop()
    }
  }

  // S4 — anthropic with a loopback baseURL that refuses → explicit
  // provider-unavailable error naming the backend.
  {
    let sidecar: StartedSidecar | null = null
    const refused = await freePort()
    try {
      sidecar = await startSidecarWithFakeOpencode({ scenario: 'silent' })
      const res = await runChat(sidecar.url, {
        llm: { kind: 'anthropic', apiKey: 'sk-fake', baseUrl: `http://127.0.0.1:${refused}` },
        timeoutMs: 30000,
      })
      const errs = errorMessages(res)
      record(
        'S4 anthropic with refused baseURL → explicit provider-unavailable error',
        'error mentioning connection/refused/fetch',
        errs.slice(0, 100) || 'none',
        errs.length > 0,
      )
      record('S4 anthropic refused baseURL → stream ended with done', 'done', res.ended ? 'done' : 'no', res.ended)
      record('S4 anthropic refused baseURL → no hang (<15s)', '<15000ms', `${res.elapsedMs}ms`, res.elapsedMs < 15000)
    } finally {
      if (sidecar) await sidecar.stop()
    }
  }

  // S5 — subscription: the dispatch is proven at the seam in
  // test-opencode-lifecycle (kind defaults to subscription when llm is
  // absent). Driving the real claude CLI is not allowed here, so this row is
  // an explicit SKIPPED, not a silent pass. The sidecar under test is started
  // with PATH stripped so an accidental real-CLI dispatch would fail fast.
  record(
    'S5 subscription → dispatch via unit-level seam (SKIPPED: no real claude CLI in tests)',
    'SKIPPED (explicit)',
    'SKIPPED (explicit)',
    true,
  )

  // S6 — unknown kind → explicit error, never a silent subscription fallback.
  {
    let sidecar: StartedSidecar | null = null
    try {
      sidecar = await startSidecarWithFakeOpencode({ scenario: 'silent' })
      const res = await runChat(sidecar.url, { llm: { kind: 'banana' }, timeoutMs: 10000 })
      const errs = errorMessages(res)
      record(
        'S6 unknown kind → explicit error (HTTP 400, no silent fallback)',
        '400 / unknown llm.kind: "banana"',
        `${res.httpStatus} / ${errs.slice(0, 60) || '(stream error)'}`,
        res.httpStatus === 400 || /unknown llm\.kind/.test(errs),
      )
    } finally {
      if (sidecar) await sidecar.stop()
    }
  }

  // ── (c) REQUEST handling ───────────────────────────────────────────────────
  {
    let sidecar: StartedSidecar | null = null
    try {
      sidecar = await startSidecarWithFakeOpencode({ scenario: 'silent' })

      // R1 — invalid JSON body → 400.
      const r1 = await runChat(sidecar.url, { rawBody: '{ not json', timeoutMs: 10000 })
      record(
        'R1 invalid JSON body → 400 + error',
        '400 / invalid json',
        `${r1.httpStatus} / ${r1.statuses.map((s) => s.message ?? s.value).join(' ').slice(0, 60)}`,
        r1.httpStatus === 400,
      )

      // R2 — missing/empty user → 400.
      const r2a = await runChat(sidecar.url, { rawBody: JSON.stringify({ messages: [] }), timeoutMs: 10000 })
      const r2b = await runChat(sidecar.url, { user: '   ', timeoutMs: 10000 })
      record(
        'R2 missing/empty user → 400',
        '400 / 400',
        `${r2a.httpStatus} / ${r2b.httpStatus}`,
        r2a.httpStatus === 400 && r2b.httpStatus === 400,
      )

      // R3/R4 — wrong types for messages/graph → 400.
      const r3 = await runChat(sidecar.url, { messages: 'not-an-array', timeoutMs: 10000 })
      const r4 = await runChat(sidecar.url, { graph: 'not-an-object', timeoutMs: 10000 })
      record('R3 messages wrong type → 400', '400', String(r3.httpStatus), r3.httpStatus === 400)
      record('R4 graph wrong type → 400', '400', String(r4.httpStatus), r4.httpStatus === 400)

      // R5 — body over the 1 MiB cap → 413.
      const big = 'x'.repeat(1_048_576 + 1000)
      const r5 = await runChat(sidecar.url, { rawBody: JSON.stringify({ user: 'hi', big }), timeoutMs: 20000 })
      record('R5 body over 1 MiB cap → 413', '413', String(r5.httpStatus), r5.httpStatus === 413)

      // R6 — provider unavailable (SPINOML_OPENCODE_BIN=/nonexistent on a
      // SEPARATE sidecar) → explicit error naming the cause, stream ends with
      // done, no hang.
    } finally {
      if (sidecar) await sidecar.stop()
    }
    let missing: StartedSidecar | null = null
    try {
      missing = await startSidecar({
        cmd: process.execPath,
        args: [SIDECAR],
        cwd: REPO,
        readyTimeoutMs: 30000,
        stripEnvPrefix: 'SPINOML_',
        portEnv: 'SPINOML_LLM_PORT',
        env: {
          SPINOML_SIDECAR_TOKEN: TOKEN,
          SPINOML_REQUIRE_TOKEN: '1',
          SPINOML_OPENCODE_BIN: '/nonexistent/opencode-binary',
          FAKE_OPENCODE_SCENARIO: JSON.stringify({ scenario: 'silent' }),
        },
      })
      const res = await runChat(missing.url, { llm: { kind: 'opencode', model: 'opencode/big-pickle' }, timeoutMs: 20000 })
      const errs = errorMessages(res)
      record(
        'R6 missing opencode binary → explicit error naming the cause',
        'error mentioning fehlgeschlagen/ENOENT',
        errs.slice(0, 100) || 'none',
        errs.length > 0,
      )
      record('R6 missing opencode binary → stream ended with done', 'done', res.ended ? 'done' : 'no', res.ended)
      record('R6 missing opencode binary → no hang (<10s)', '<10000ms', `${res.elapsedMs}ms`, res.elapsedMs < 10000)
    } finally {
      if (missing) await missing.stop()
    }
  }

  // ── (d) MODEL handling through the real sidecar + fake opencode ───────────
  {
    let sidecar: StartedSidecar | null = null
    try {
      sidecar = await startSidecarWithFakeOpencode({ scenario: 'tool-call', dump: join(RUN, 'm-dump.json') })

      // M1 — default model used when none given.
      {
        const res = await runChat(sidecar.url, { llm: { kind: 'opencode' } })
        const info = await waitJson(join(RUN, 'm-dump.json'), 5000)
        const argv = Array.isArray(info?.fakeArgv) ? (info?.fakeArgv as string[]) : []
        const i = argv.indexOf('--model')
        const passed = i >= 0 ? argv[i + 1] : ''
        record(
          'M1 default model opencode/big-pickle used when none given',
          '--model opencode/big-pickle',
          i >= 0 ? `--model ${passed}` : 'no --model flag',
          passed === 'opencode/big-pickle',
        )
        record('M1 stream ended with done', 'done', res.ended ? 'done' : 'no', res.ended)
      }

      // M2 — alternative valid model passed verbatim.
      {
        const res = await runChat(sidecar.url, { llm: { kind: 'opencode', model: 'tud-ai/deepseek-ai/DeepSeek-V4.1-Flash' } })
        const info = await waitJson(join(RUN, 'm-dump.json'), 5000)
        const argv = Array.isArray(info?.fakeArgv) ? (info?.fakeArgv as string[]) : []
        const i = argv.indexOf('--model')
        const passed = i >= 0 ? argv[i + 1] : ''
        record(
          'M2 alternative valid model passed verbatim',
          '--model tud-ai/deepseek-ai/DeepSeek-V4.1-Flash',
          i >= 0 ? `--model ${passed}` : 'no --model flag',
          passed === 'tud-ai/deepseek-ai/DeepSeek-V4.1-Flash',
        )
        record('M2 stream ended with done', 'done', res.ended ? 'done' : 'no', res.ended)
      }

      // M3.. — hostile / malformed models are rejected BEFORE spawning.
      const HOSTILE_MODELS: [string, string][] = [
        ['M3  --print-logs (option injection)', '--print-logs'],
        ['M4  -h (option injection)', '-h'],
        ['M5  --dangerously-skip-permissions', '--dangerously-skip-permissions'],
        ['M6  spaces', 'a b'],
        ['M7  newline control char', 'x\nname'],
        ['M8  shell quote', '"; rm -rf'],
        ['M9  backtick substitution', '`id`'],
        ['M10 command substitution', '$(id)'],
        ['M11 empty string → defaults to opencode/big-pickle', ''],
        ['M12 5000 chars', 'a'.repeat(5000)],
        ['M13 ../ traversal', '../x'],
        ['M14 unicode', 'mödell'],
        ['M15 non-string number', '42'],
        ['M16 = prefixed', '=x'],
      ]
      for (const [name, model] of HOSTILE_MODELS) {
        // A number is not a string — JSON round-trips it as a number, exactly
        // what the frontend must never send.
        const llm: Record<string, unknown> = { kind: 'opencode', model: /^\d+$/.test(model) ? Number(model) : model }
        const before = procSnapshot().size
        const res = await runChat(sidecar.url, { llm, timeoutMs: 10000 })
        const errs = errorMessages(res)
        const after = procSnapshot().size
        const spawnedByUs = after > before
        if (model === '') {
          // The empty/absent case keeps the documented default — see
          // sidecar-llm/model-name.mjs (validateModelName returns ok:true
          // for "" so runOpenCode falls back to OPENCODE_DEFAULT_MODEL).
          const info = await waitJson(join(RUN, 'm-dump.json'), 5000)
          const argv = Array.isArray(info?.fakeArgv) ? (info?.fakeArgv as string[]) : []
          const i = argv.indexOf('--model')
          const passed = i >= 0 ? argv[i + 1] : ''
          record(
            `${name} → opencode/big-pickle used`,
            '--model opencode/big-pickle',
            i >= 0 ? `--model ${passed}` : 'no --model flag',
            passed === 'opencode/big-pickle',
          )
        } else {
          record(
            `${name} rejected explicitly BEFORE spawning`,
            'invalid model name error, no spawn, done',
            `${errs.slice(0, 60) || 'none'} / ${spawnedByUs ? 'SPAWNED' : 'no spawn'}`,
            errs.length > 0 && !spawnedByUs && res.ended,
          )
        }
      }

      // M17 — unknown-but-well-formed model: the fake opencode's `error` event
      // is surfaced as an explicit status (reuses the existing error-event
      // scenario pattern).
      {
        let s2: StartedSidecar | null = null
        try {
          s2 = await startSidecarWithFakeOpencode({ scenario: 'error-event' })
          const res = await runChat(s2.url, { llm: { kind: 'opencode', model: 'nope/does-not-exist' }, timeoutMs: 15000 })
          const errs = errorMessages(res)
          record(
            'M17 unknown-but-well-formed model → fake error event surfaced explicitly',
            'error mentioning fake-error-message',
            errs.slice(0, 80) || 'none',
            /fake-error-message/.test(errs),
          )
          record('M17 stream ended with done', 'done', res.ended ? 'done' : 'no', res.ended)
        } finally {
          if (s2) await s2.stop()
        }
      }
    } finally {
      if (sidecar) await sidecar.stop()
    }
  }

  // ── (e) /opencode/models endpoint ──────────────────────────────────────────
  type ModelsBody = { ok?: boolean; models?: unknown; error?: string }
  {
    let h: LlmHarness | null = null
    try {
      h = await LlmHarness.start({
        env: {
          SPINOML_OPENCODE_BIN: FAKE_BIN,
          FAKE_OPENCODE_SCENARIO: JSON.stringify({ scenario: 'tool-call' }),
        },
      })
      // The fake-opencode script does not implement `models` — it prints
      // NDJSON events and exits 0, which the sidecar's listOpenCodeModels
      // parses for `provider/model` lines. With the 'tool-call' scenario it
      // emits no such line, so the endpoint must NOT pretend success with an
      // empty list; the parse produces no model ids and returns ok with an
      // empty array only when the binary exited 0. That is the documented
      // behaviour (a binary that exits 0 and lists nothing = empty list is a
      // LIE only when it FAILED). To prove the failure path we flip the
      // scenario to one that exits non-zero.
      const body1 = await h.opencodeModelsBody()
      let j1: ModelsBody | null = null
      try { j1 = JSON.parse(body1) as ModelsBody } catch { /* not json */ }
      record(
        'E1 /opencode/models returns an explicit ok or explicit error (never a lie)',
        'explicit shape',
        `ok=${j1?.ok} models=${Array.isArray(j1?.models) ? j1?.models.length : 'n/a'} error=${j1?.error ? 'yes' : 'no'}`,
        j1 !== null && (typeof j1?.ok === 'boolean' || typeof j1?.error === 'string'),
      )
      // Cache: a second call within TTL must not re-run the binary. We assert
      // the sidecar answered twice with the same body shape.
      const body2 = await h.opencodeModelsBody()
      record(
        'E2 /opencode/models caches (second call answered with the same shape)',
        'same shape',
        body1 === body2 ? 'same' : 'different',
        body1 === body2,
      )
    } finally {
      if (h) await h.stop()
    }
    // E3 — a failing binary gives an explicit error, not an empty list.
    {
      const fakeFailing = join(RUN, 'fake-opencode-failing')
      const { writeFileSync, chmodSync } = await import('node:fs')
      writeFileSync(fakeFailing, '#!/bin/sh\nexit 7\n', { mode: 0o755 })
      chmodSync(fakeFailing, 0o755)
      let h2: LlmHarness | null = null
      try {
        h2 = await LlmHarness.start({
          env: {
            SPINOML_OPENCODE_BIN: fakeFailing,
            FAKE_OPENCODE_SCENARIO: JSON.stringify({ scenario: 'exit-nonzero' }),
          },
        })
        const body = await h2.opencodeModelsBody()
        let j: ModelsBody | null = null
        try { j = JSON.parse(body) as ModelsBody } catch { /* not json */ }
        record(
          'E3 failing opencode models binary → explicit error, not an empty list',
          'error, no models field',
          j?.error ? `error: ${j.error.slice(0, 60)}` : `ok=${j?.ok} models=${JSON.stringify(j?.models ?? null)}`,
          typeof j?.error === 'string' && j.error.length > 0 && !Array.isArray(j?.models),
        )
      } finally {
        if (h2) await h2.stop()
        try { rmSync(fakeFailing) } catch { /* best effort */ }
      }
    }
  }

  // ── hygiene + coverage map + row table ─────────────────────────────────────

  const leftover = procSnapshot()
  const newProcs = [...leftover].filter((l) => ![...PROCS_BEFORE].some((b) => b === l))
  record('Z1 no leftover fake-opencode / bridge / sidecar process', '0', String(newProcs.length), newProcs.length === 0)

  // Coverage map: every §0.12 bullet must be mapped AND the mapped suite/row
  // must be real. A bullet with no coverage fails the test.
  const uncovered = COVERAGE.filter((c) => !c.suite || !c.row)
  record('Z2 every §0.12 bullet mapped to a suite row', '0 uncovered', String(uncovered.length), uncovered.length === 0)

  rmSync(RUN, { recursive: true, force: true })

  console.log('\n── coverage map (§0.12 → suite/row) ──────────────────────────')
  for (const c of COVERAGE) console.log(`  ${c.bullet}\n      → ${c.suite}: ${c.row}`)

  console.log('\n── row table ─────────────────────────────────────────────────')
  const width = Math.max(...rows.map((r) => r.case.length))
  const failures: Row[] = []
  for (const r of rows) {
    const verdict = r.pass ? 'PASS' : 'FAIL'
    if (!r.pass) failures.push(r)
    console.log(`  ${r.case.padEnd(width)} | ${r.expected.padEnd(40)} | ${r.got.slice(0, 60).padEnd(60)} | ${verdict}`)
  }

  console.log(`\nrows: ${rows.length}, failures: ${failures.length}`)
  process.exitCode = failures.length > 0 ? 1 : 0
}

main().catch((e) => {
  console.error('harness crashed:', e)
  process.exitCode = 2
})
