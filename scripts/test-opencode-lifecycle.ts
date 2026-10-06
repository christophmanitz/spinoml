// Phase 79 — end-to-end lifecycle test for the opencode provider path of the
// LLM sidecar (the one path TODO Phase 0.12 had never tested). It runs the REAL
// sidecar (token mode) against a FAKE `opencode` binary that spawns the REAL
// `mcp-bridge.mjs` exactly as opencode does (`mcp.graph.command` + `environment`)
// and speaks MCP over its stdio. No real opencode, no network, no API key.
//
// It proves two things the previous auth test could not:
//   1. the per-turn MCP session secret authenticates the bridge end to end
//      (Phase 77/78; docs/engineering/SIDECAR_AUTH.md "LLM sidecar specifics"),
//      while the master token never reaches opencode or the bridge;
//   2. the opencode process lifecycle fails EXPLICITLY (non-zero exit, error
//      event, start timeout, client abort) — never silently — and leaves no
//      process or temp dir behind.
//
// Run: npm run test:opencode-lifecycle

import { execSync } from 'node:child_process'
import { randomBytes, randomUUID } from 'node:crypto'
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  expectExit,
  freePort,
  rawRequest,
  runMatrix,
  startSidecar,
  type MatrixRow,
  type RawResponse,
  type StartedSidecar,
} from './lib/auth-probe'
import { BASE_GRAPH, LlmHarness, type ChatResult, type GraphSnapshot } from './lib/llm-harness.ts'
import { decide, loadConfig } from '../sidecar-llm/auth.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const REPO = resolve(HERE, '..')
const SIDECAR = join(REPO, 'sidecar-llm', 'main.mjs')
const BRIDGE = join(REPO, 'sidecar-llm', 'mcp-bridge.mjs')
const FAKE_BIN = join(REPO, 'scripts', 'lib', 'fake-opencode')
const NODE = process.execPath
const TOKEN = randomBytes(32).toString('hex')
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
const HEX64_RE = /^[0-9a-f]{64}$/

const rows: MatrixRow[] = []
const RUN = mkdtempSync(join(tmpdir(), 'spinoml-oclc-'))
const DIRS_BEFORE = tmpDirs()
const PROCS_BEFORE = procSnapshot()

function record(name: string, expected: string, got: string, pass?: boolean): void {
  rows.push({ case: name, expected, got, pass })
}

// ── small typed accessors over the fake's JSON dump ────────────────────────

function strField(o: Record<string, unknown> | null, k: string): string {
  const v = o?.[k]
  return typeof v === 'string' ? v : ''
}

function numField(o: Record<string, unknown> | null, k: string): number {
  const v = o?.[k]
  return typeof v === 'number' ? v : 0
}

function boolField(o: Record<string, unknown> | null, k: string): boolean | null {
  const v = o?.[k]
  return typeof v === 'boolean' ? v : null
}

function strArray(o: Record<string, unknown> | null, k: string): string[] {
  const v = o?.[k]
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : []
}

function leaksToken(haystack: string): boolean {
  if (haystack.includes(TOKEN)) return true
  for (let i = 0; i + 16 <= TOKEN.length; i++) {
    if (haystack.includes(TOKEN.slice(i, i + 16))) return true
  }
  return false
}

function anyHex64(values: string[]): boolean {
  return values.some((v) => HEX64_RE.test(v))
}

// ── process / tmpdir hygiene ───────────────────────────────────────────────

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

function tmpDirs(): Set<string> {
  try {
    return new Set(readdirSync(tmpdir()).filter((n) => n.startsWith('spinoml-opencode-')))
  } catch {
    return new Set()
  }
}

function isPidAlive(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0) return false
  try {
    process.kill(pid, 0)
    return true
  } catch (e) {
    return (e as NodeJS.ErrnoException).code !== 'ESRCH'
  }
}

async function waitGone(pid: number, timeoutMs: number): Promise<boolean> {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    if (!isPidAlive(pid)) return true
    await new Promise((r) => setTimeout(r, 100))
  }
  return !isPidAlive(pid)
}

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

// ── sidecar + chat helpers ─────────────────────────────────────────────────

async function startTestSidecar(
  scenario: Record<string, unknown>,
  extraEnv: Record<string, string> = {},
): Promise<StartedSidecar> {
  return await startSidecar({
    cmd: NODE,
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
  signal?: AbortSignal
  timeoutMs?: number
  graph?: GraphSnapshot
}

async function runChat(url: string, opts: RunChatOpts = {}): Promise<ChatResult> {
  const harness = await LlmHarness.connect({ baseUrl: url })
  try {
    return await harness.chat({
      script: [],
      llm: { kind: 'opencode', model: 'fake', apiKey: 'sk-fake' },
      token: TOKEN,
      graph: opts.graph ?? BASE_GRAPH,
      timeoutMs: opts.timeoutMs ?? 15000,
      signal: opts.signal,
    })
  } finally {
    await harness.stop()
  }
}

function errorMessages(result: ChatResult): string {
  return result.statuses
    .filter((s) => s.value === 'error')
    .map((s) => s.message ?? '')
    .join(' | ')
}

// ── case (a): the happy path — bridge auth end to end ──────────────────────

async function caseToolCall(): Promise<void> {
  const dump = join(RUN, 'a-dump.json')
  const scenario = {
    scenario: 'tool-call',
    dump,
    tool: 'graph_add_layer',
    args: { layer_type: 'Linear', after: 'fc', params: { in_features: 4, out_features: 2 } },
  }
  const sidecar = await startTestSidecar(scenario)
  try {
    const result = await runChat(sidecar.url)
    const info = await waitJson(dump, 5000)
    const hasAction = result.actions.some((a) => a.op === 'add_layer')
    record('a tool-call: action add_layer emitted', 'yes', hasAction ? 'yes' : 'no', hasAction)
    record(
      'a tool-call: tool_use add_layer',
      'add_layer',
      result.toolUses[0]?.name ?? 'none',
      result.toolUses.some((t) => t.name === 'add_layer'),
    )
    record('a tool-call: tool_result ok', 'ok', result.toolResults[0] ? (result.toolResults[0].ok ? 'ok' : 'error') : 'none', result.toolResults.some((t) => t.ok))
    record('a tool-call: stream ended with done', 'done', result.ended ? 'done' : 'no', result.ended)
    record('a tool-call: requestId is a uuid', 'uuid', strField(info, 'requestId') || 'none', UUID_RE.test(strField(info, 'requestId')))
    record('a token secrecy: fake env has no master token', 'false', String(boolField(info, 'fakeEnvHasMasterToken')), boolField(info, 'fakeEnvHasMasterToken') === false)
    record('a token secrecy: bridge env has no master token', 'false', String(boolField(info, 'bridgeEnvHasMasterToken')), boolField(info, 'bridgeEnvHasMasterToken') === false)
    record('a secret: present in bridge env', 'true', String(boolField(info, 'bridgeEnvHasSecret')), boolField(info, 'bridgeEnvHasSecret') === true)
    record('a secret: not in bridge argv', 'false', String(boolField(info, 'bridgeSecretInArgv')), boolField(info, 'bridgeSecretInArgv') === false)
    record('a secret: not in fake argv', 'false', String(boolField(info, 'fakeSecretInArgv')), boolField(info, 'fakeSecretInArgv') === false)
    record('a secret: no 64-hex token in either argv', 'none', anyHex64([...strArray(info, 'fakeArgv'), ...strArray(info, 'bridgeArgv')]) ? 'hex64 found' : 'none', !anyHex64([...strArray(info, 'fakeArgv'), ...strArray(info, 'bridgeArgv')]))
    const raw = info ? JSON.stringify(info) : ''
    record('a secrecy: dump hides master token', 'clean', leaksToken(raw) ? 'LEAK' : 'clean', !leaksToken(raw))
    record('a secrecy: bridge argv leaks no secret', 'clean', leaksToken(strArray(info, 'bridgeArgv').join(' ')) ? 'LEAK' : 'clean', !leaksToken(strArray(info, 'bridgeArgv').join(' ')))
  } finally {
    await sidecar.stop()
  }
}

// ── case (b): bridge auth failures surface explicitly ──────────────────────

async function caseBridgeAuth(scenarioName: 'bridge-no-secret' | 'bridge-wrong-secret'): Promise<void> {
  const dump = join(RUN, `${scenarioName}-dump.json`)
  const sidecar = await startTestSidecar({
    scenario: scenarioName,
    dump,
    tool: 'graph_add_layer',
    args: { layer_type: 'ReLU', after: 'fc' },
  })
  try {
    const result = await runChat(sidecar.url)
    const visible = [
      ...result.toolResults.map((t) => t.error ?? t.result),
      ...result.texts,
      errorMessages(result),
    ].join(' | ')
    record(`${scenarioName}: no action emitted`, '0', String(result.actions.length), result.actions.length === 0)
    record(`${scenarioName}: tool_result reports error`, 'error', result.toolResults.some((t) => !t.ok) ? 'error' : 'none', result.toolResults.some((t) => !t.ok))
    record(`${scenarioName}: model-visible auth failure`, 'mentions authentication', visible.slice(0, 60) || 'none', /authenticat/i.test(visible))
    record(`${scenarioName}: no hang — stream ended`, 'done', result.ended ? 'done' : 'no', result.ended)
    const info = await waitJson(dump, 3000)
    record(`${scenarioName}: session secret was in bridge env`, 'true', String(boolField(info, 'bridgeEnvHasSecret')), boolField(info, 'bridgeEnvHasSecret') === true)
  } finally {
    await sidecar.stop()
  }
}

// ── case (c): hostile direct caller + decide() contract rows ───────────────

function headerGet(headers: Record<string, string>): (name: string) => string | null {
  const lower: Record<string, string> = {}
  for (const [k, v] of Object.entries(headers)) lower[k.toLowerCase()] = v
  return (name: string): string | null => lower[name.toLowerCase()] ?? null
}

function decisionContractRows(): void {
  const cfg = loadConfig({ SPINOML_SIDECAR_TOKEN: TOKEN, SPINOML_REQUIRE_TOKEN: '1' })
  const d1 = decide('POST', '/internal/mcp/x/call', headerGet({ Host: '127.0.0.1' }), cfg)
  record('decide: MCP route, token mode, no token → ok (master-token exempt)', 'ok', d1.kind, d1.kind === 'ok')
  const d2 = decide('POST', '/internal/mcp/x/call', headerGet({ Host: '127.0.0.1', Origin: 'https://evil.example' }), cfg)
  record('decide: MCP route + evil Origin → reject bad_origin', 'reject bad_origin', `${d2.kind} ${d2.kind === 'reject' ? d2.code : ''}`, d2.kind === 'reject' && d2.code === 'bad_origin')
  const d3 = decide('GET', '/internal/mcp/x/call', headerGet({ Host: '127.0.0.1' }), cfg)
  record('decide: GET MCP route, no token → reject unauthorized', 'reject unauthorized', `${d3.kind} ${d3.kind === 'reject' ? d3.code : ''}`, d3.kind === 'reject' && d3.code === 'unauthorized')
  const d4 = decide('POST', '/chat', headerGet({ Host: '127.0.0.1' }), cfg)
  record('decide: other path, no token → reject unauthorized', 'reject unauthorized', `${d4.kind} ${d4.kind === 'reject' ? d4.code : ''}`, d4.kind === 'reject' && d4.code === 'unauthorized')
  const d5 = decide('POST', '/chat', headerGet({ Host: '127.0.0.1', 'X-SpinoML-Token': TOKEN }), cfg)
  record('decide: /chat with master token → ok', 'ok', d5.kind, d5.kind === 'ok')
}

async function mcpCall(port: number, id: string, token: string | null): Promise<RawResponse> {
  const headers: Record<string, string> = { Host: '127.0.0.1', 'Content-Type': 'application/json' }
  if (token !== null) headers['X-SpinoML-Token'] = token
  return await rawRequest({
    port,
    method: 'POST',
    path: `/internal/mcp/${id}/call`,
    headers,
    body: JSON.stringify({ name: 'add_layer', args: { layer_type: 'ReLU' } }),
  })
}

function bodyCode(res: RawResponse): string {
  try {
    const j = JSON.parse(res.body) as Record<string, unknown>
    return typeof j.code === 'string' ? j.code : typeof j.error === 'string' ? j.error : ''
  } catch {
    return ''
  }
}

async function caseHostileDirect(): Promise<void> {
  decisionContractRows()
  const dump = join(RUN, 'c-dump.json')
  const pidFile = join(RUN, 'c-pid.json')
  const sidecar = await startTestSidecar({ scenario: 'silent', dump, pidFile })
  const harness = await LlmHarness.connect({ baseUrl: sidecar.url })
  const ctrl = new AbortController()
  const promise = harness.chat({
    script: [],
    llm: { kind: 'opencode', model: 'fake', apiKey: 'sk-fake' },
    token: TOKEN,
    graph: BASE_GRAPH,
    timeoutMs: 60000,
    signal: ctrl.signal,
  })
  let result: ChatResult | null = null
  try {
    const info = await waitJson(dump, 10000)
    const id = strField(info, 'requestId')
    record('c hostile: fake observed the live session id', 'uuid', id || 'none', UUID_RE.test(id))
    const probes: Array<{ name: string; id: string; token: string | null }> = [
      { name: 'no secret', id, token: null },
      { name: 'master token', id, token: TOKEN },
      { name: 'random id, no secret', id: randomUUID(), token: null },
      { name: 'random id, master token', id: randomUUID(), token: TOKEN },
    ]
    for (const p of probes) {
      const res = await mcpCall(sidecar.port, p.id, p.token)
      record(`c hostile: /internal/mcp/.../call [${p.name}]`, '401 unauthorized', `${res.status} ${bodyCode(res)}`, res.status === 401 && bodyCode(res) === 'unauthorized')
    }
    ctrl.abort()
    result = await promise
    record('c hostile: no mutation reached the turn', '0 actions', String(result.actions.length), result.actions.length === 0)
    // Wait for the aborted turn's processes to die so the sidecar's deterministic
    // session-dir cleanup can run before we tear the sidecar down.
    const pidInfo = await waitJson(pidFile, 3000)
    await waitGone(numField(pidInfo, 'pid'), 5000)
    await waitGone(numField(pidInfo, 'childPid'), 5000)
  } finally {
    if (result === null) ctrl.abort()
    await promise.catch(() => undefined)
    await harness.stop()
    await sidecar.stop()
  }
}

// ── case (d): lifecycle failures are explicit ──────────────────────────────

async function caseExitNonzero(): Promise<void> {
  const sidecar = await startTestSidecar({ scenario: 'exit-nonzero' })
  try {
    const result = await runChat(sidecar.url)
    const errs = errorMessages(result)
    record('d exit-nonzero: explicit status error', 'error', result.statuses.some((s) => s.value === 'error') ? 'error' : 'none', result.statuses.some((s) => s.value === 'error'))
    record('d exit-nonzero: mentions the exit code', 'exit 3', errs.slice(0, 80) || 'none', /exit 3/.test(errs))
    record('d exit-nonzero: includes stderr tail', 'stderr tail', errs.slice(0, 80) || 'none', /exit-nonzero stderr tail line/.test(errs))
    record('d exit-nonzero: stream ended with done', 'done', result.ended ? 'done' : 'no', result.ended)
  } finally {
    await sidecar.stop()
  }
}

async function caseErrorEvent(): Promise<void> {
  const sidecar = await startTestSidecar({ scenario: 'error-event' })
  try {
    const result = await runChat(sidecar.url)
    const errs = errorMessages(result)
    record('d error-event: explicit status error', 'error', errs.slice(0, 80) || 'none', result.statuses.some((s) => s.value === 'error'))
    record('d error-event: carries the message', 'fake-error-message', errs.slice(0, 80) || 'none', /fake-error-message/.test(errs))
    record('d error-event: stream ended with done', 'done', result.ended ? 'done' : 'no', result.ended)
  } finally {
    await sidecar.stop()
  }
}

async function caseGarbage(): Promise<void> {
  const sidecar = await startTestSidecar({ scenario: 'garbage' })
  try {
    const result = await runChat(sidecar.url)
    const text = result.texts.join('')
    record('d garbage: non-JSON noise ignored, text delivered', 'garbage-ok', text.slice(0, 80) || 'none', /garbage-ok/.test(text))
    record('d garbage: stream ended with done', 'done', result.ended ? 'done' : 'no', result.ended)
  } finally {
    await sidecar.stop()
  }
}

async function caseStartTimeout(): Promise<void> {
  const dump = join(RUN, 'd4-dump.json')
  const pidFile = join(RUN, 'd4-pid.json')
  const sidecar = await startTestSidecar({ scenario: 'silent', dump, pidFile }, { SPINOML_OPENCODE_START_TIMEOUT_MS: '1500' })
  try {
    const result = await runChat(sidecar.url, { timeoutMs: 8000 })
    const errs = errorMessages(result)
    record('d silent+start-timeout: explicit "antwortet nicht" error', 'antwortet nicht', errs.slice(0, 80) || 'none', /antwortet nicht/.test(errs))
    record('d silent+start-timeout: surfaced within 5s', '<5000ms', `${result.elapsedMs}ms`, result.elapsedMs < 5000)
    const pidInfo = await waitJson(pidFile, 3000)
    const pid = numField(pidInfo, 'pid')
    const childPid = numField(pidInfo, 'childPid')
    record('d silent+start-timeout: fake process gone', 'dead', (await waitGone(pid, 5000)) ? 'dead' : 'alive', await waitGone(pid, 100))
    record('d silent+start-timeout: bridge process gone', 'dead', (await waitGone(childPid, 5000)) ? 'dead' : 'alive', await waitGone(childPid, 100))
  } finally {
    await sidecar.stop()
  }
}

async function caseAbort(): Promise<void> {
  const dump = join(RUN, 'd5-dump.json')
  const pidFile = join(RUN, 'd5-pid.json')
  const sidecar = await startTestSidecar({ scenario: 'silent', dump, pidFile })
  const harness = await LlmHarness.connect({ baseUrl: sidecar.url })
  const ctrl = new AbortController()
  const promise = harness.chat({
    script: [],
    llm: { kind: 'opencode', model: 'fake', apiKey: 'sk-fake' },
    token: TOKEN,
    graph: BASE_GRAPH,
    timeoutMs: 60000,
    signal: ctrl.signal,
  })
  try {
    const pidInfo = await waitJson(pidFile, 10000)
    const pid = numField(pidInfo, 'pid')
    const childPid = numField(pidInfo, 'childPid')
    record('d abort: fake recorded its pids', 'pid+childPid', `${pid}/${childPid}`, pid > 0 && childPid > 0)
    ctrl.abort()
    const result = await promise
    record('d abort: client abort observed', 'aborted', String(result.aborted), result.aborted === true)
    const fakeGone = await waitGone(pid, 5000)
    const bridgeGone = await waitGone(childPid, 5000)
    record('d abort: fake dead within 5s', 'dead', fakeGone ? 'dead' : 'alive', fakeGone)
    record('d abort: bridge child dead within 5s', 'dead', bridgeGone ? 'dead' : 'alive', bridgeGone)
  } finally {
    await promise.catch(() => undefined)
    await harness.stop()
    await sidecar.stop()
  }
}

async function caseInvalidEnv(): Promise<void> {
  const cases: Array<{ name: string; env: Record<string, string>; varName: string }> = [
    { name: 'START_TIMEOUT_MS=abc', env: { SPINOML_OPENCODE_START_TIMEOUT_MS: 'abc' }, varName: 'SPINOML_OPENCODE_START_TIMEOUT_MS' },
    { name: 'START_TIMEOUT_MS=0', env: { SPINOML_OPENCODE_START_TIMEOUT_MS: '0' }, varName: 'SPINOML_OPENCODE_START_TIMEOUT_MS' },
    { name: 'TIMEOUT_MS=abc', env: { SPINOML_OPENCODE_TIMEOUT_MS: 'abc' }, varName: 'SPINOML_OPENCODE_TIMEOUT_MS' },
    { name: 'TIMEOUT_MS=-5', env: { SPINOML_OPENCODE_TIMEOUT_MS: '-5' }, varName: 'SPINOML_OPENCODE_TIMEOUT_MS' },
  ]
  for (const c of cases) {
    const port = await freePort()
    const r = await expectExit(
      {
        cmd: NODE,
        args: [SIDECAR],
        cwd: REPO,
        timeoutMs: 20000,
        stripEnvPrefix: 'SPINOML_',
        env: {
          SPINOML_SIDECAR_TOKEN: TOKEN,
          SPINOML_REQUIRE_TOKEN: '1',
          SPINOML_LLM_PORT: String(port),
          ...c.env,
        },
      },
      2,
    )
    record(`d startup ${c.name} exits 2`, 'exit 2', `exit ${r.code}`, r.ok)
    record(`d startup ${c.name} names the variable`, c.varName, r.stderr.slice(0, 80) || 'none', r.stderr.includes(c.varName))
    record(`d startup ${c.name} hides the token`, 'clean', r.stderr.includes(TOKEN) ? 'LEAK' : 'clean', !r.stderr.includes(TOKEN))
  }
}

// ── mutation proof ─────────────────────────────────────────────────────────

interface MutationProbe {
  observed: string
  invariantHeld: boolean
}

async function probeToolCallAction(): Promise<MutationProbe> {
  const sidecar = await startTestSidecar({
    scenario: 'tool-call',
    dump: join(RUN, 'mut-toolcall-dump.json'),
    tool: 'graph_add_layer',
    args: { layer_type: 'Linear', after: 'fc', params: { in_features: 4, out_features: 2 } },
  })
  try {
    const result = await runChat(sidecar.url)
    const n = result.actions.filter((a) => a.op === 'add_layer').length
    return { observed: `add_layer actions=${n}`, invariantHeld: n > 0 }
  } finally {
    await sidecar.stop()
  }
}

async function probeWrongSecretRejected(): Promise<MutationProbe> {
  const sidecar = await startTestSidecar({ scenario: 'bridge-wrong-secret', dump: join(RUN, 'mut-wrong-dump.json') })
  try {
    const result = await runChat(sidecar.url)
    const visible = [...result.toolResults.map((t) => t.error ?? t.result), ...result.texts].join(' ')
    const rejected = result.actions.length === 0 && /authenticat/i.test(visible)
    return { observed: `actions=${result.actions.length} authFail=${/authenticat/i.test(visible)}`, invariantHeld: rejected }
  } finally {
    await sidecar.stop()
  }
}

async function probeAbortKills(): Promise<MutationProbe> {
  const pidFile = join(RUN, 'mut-abort-pid.json')
  const dirsBeforeProbe = tmpDirs()
  const sidecar = await startTestSidecar({ scenario: 'silent', dump: join(RUN, 'mut-abort-dump.json'), pidFile })
  const harness = await LlmHarness.connect({ baseUrl: sidecar.url })
  const ctrl = new AbortController()
  let pid = 0
  let childPid = 0
  const promise = harness.chat({
    script: [],
    llm: { kind: 'opencode', model: 'fake', apiKey: 'sk-fake' },
    token: TOKEN,
    graph: BASE_GRAPH,
    timeoutMs: 60000,
    signal: ctrl.signal,
  })
  try {
    const pidInfo = await waitJson(pidFile, 10000)
    pid = numField(pidInfo, 'pid')
    childPid = numField(pidInfo, 'childPid')
    ctrl.abort()
    // Do not await the turn: with the kill removed it never resolves.
    void promise.catch(() => undefined)
    await new Promise((r) => setTimeout(r, 3000))
    const fakeAlive = isPidAlive(pid)
    const bridgeAlive = isPidAlive(childPid)
    return {
      observed: `fakeAlive=${fakeAlive} bridgeAlive=${bridgeAlive}`,
      invariantHeld: !fakeAlive && !bridgeAlive,
    }
  } finally {
    ctrl.abort()
    // Un-hang the mutated run: kill the fake so the sidecar's close handler runs
    // (and deterministically removes its temp session dir) before we tear down.
    for (const p of [pid, childPid]) {
      if (isPidAlive(p)) {
        try {
          process.kill(p, 'SIGKILL')
        } catch {
          // already gone
        }
      }
    }
    const dirDeadline = Date.now() + 3000
    while (Date.now() < dirDeadline) {
      const pending = [...tmpDirs()].filter((d) => !dirsBeforeProbe.has(d))
      if (pending.length === 0) break
      await new Promise((r) => setTimeout(r, 100))
    }
    await Promise.race([promise.catch(() => undefined), new Promise((r) => setTimeout(r, 1000))])
    await harness.stop()
    await sidecar.stop()
  }
}

async function mutationProof(): Promise<void> {
  const bridgePath = BRIDGE
  const mainPath = SIDECAR
  const files = [bridgePath, mainPath]
  const orig = new Map(files.map((f) => [f, readFileSync(f, 'utf8')]))
  const restore = (): void => {
    for (const [f, c] of orig) writeFileSync(f, c)
  }

  async function withMutation(
    label: string,
    file: string,
    from: RegExp,
    to: string,
    run: () => Promise<MutationProbe>,
  ): Promise<void> {
    restore()
    const src = readFileSync(file, 'utf8')
    const patched = src.replace(from, to)
    if (patched === src) {
      record(`mutation ${label}`, 'breaks (red)', `no change applied in ${basename(file)}`, false)
      restore()
      return
    }
    writeFileSync(file, patched)
    try {
      const { observed, invariantHeld } = await run()
      record(`mutation ${label}`, 'breaks (red)', observed, !invariantHeld)
    } finally {
      restore()
    }
  }

  // (i) The bridge never sends the per-turn secret → the happy path (a) breaks.
  await withMutation(
    'i) bridge sends no secret (case a goes red)',
    bridgePath,
    /function authHeader\(\) \{\n {2}if \(!MCP_SECRET\) return \{\}\n {2}return \{ 'X-SpinoML-Token': MCP_SECRET \}\n\}/,
    'function authHeader() {\n  return {}\n}',
    probeToolCallAction,
  )

  // (i2) The sidecar skips the secret check → a wrong secret is accepted, so
  // the explicit auth failure of case (b) disappears.
  await withMutation(
    'i2) sidecar skips the session-secret check (case b goes red)',
    mainPath,
    /const ok = !!session && sessionSecretMatches\(suppliedStr, session\.secret\)/,
    'const ok = !!session',
    probeWrongSecretRejected,
  )

  // (ii) Remove the abort kill (`signal: turnAbort.signal`) → abort no longer
  // terminates the fake or its bridge child.
  await withMutation(
    'ii) abort kill removed (abort case goes red)',
    mainPath,
    /cwd: sessionDir, stdio: \['ignore', 'pipe', 'pipe'\], env, signal: turnAbort\.signal,/,
    "cwd: sessionDir, stdio: ['ignore', 'pipe', 'pipe'], env,",
    probeAbortKills,
  )
}

// ── case (e): no leftovers ─────────────────────────────────────────────────

function leftoverRows(): void {
  const newProcs = [...procSnapshot()].filter((p) => !PROCS_BEFORE.has(p))
  record('e hygiene: no leftover fake/bridge/sidecar processes', 'none', newProcs.length ? newProcs.join(' ; ').slice(0, 100) : 'none', newProcs.length === 0)
  const newDirs = [...tmpDirs()].filter((d) => !DIRS_BEFORE.has(d))
  record('e hygiene: no leftover spinoml-opencode-* temp dirs', 'none', newDirs.length ? newDirs.join(', ') : 'none', newDirs.length === 0)
}

async function main(): Promise<void> {
  console.log(`opencode lifecycle sidecar: token mode, fake=${FAKE_BIN}`)
  await caseToolCall()
  await caseBridgeAuth('bridge-no-secret')
  await caseBridgeAuth('bridge-wrong-secret')
  await caseHostileDirect()
  await caseExitNonzero()
  await caseErrorEvent()
  await caseGarbage()
  await caseStartTimeout()
  await caseAbort()
  await caseInvalidEnv()
  await mutationProof()
  leftoverRows()

  const failures = runMatrix(rows)
  rmSync(RUN, { recursive: true, force: true })
  console.log(`\n${rows.length - failures.length}/${rows.length} rows passed`)
  if (failures.length) {
    console.log('\nFAILURES:')
    for (const f of failures) console.log(`  ✗ ${f.case}: expected ${f.expected}, got ${f.got}`)
    process.exit(1)
  }
}

void main()
