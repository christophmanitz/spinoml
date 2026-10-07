// scripts/test-llm-live.ts — gated LIVE provider suite (Phase condition 6)
//
// Run (human, with credentials):
//   SPINOML_LIVE_LLM=1 SPINOML_OPENCODE_TEST_MODEL=<provider/model> npm run test:llm-live
//
// Each provider is configured via env (table below); missing → SKIPPED.
// NEVER part of default CI. Refuses to run unless SPINOML_LIVE_LLM=1 (else
// exits 2 so a leaked call cannot quietly execute against a paid/quota
// model). The script never prints the apiKey and never puts it in argv.

import { randomBytes } from 'node:crypto'
import { spawnSync } from 'node:child_process'
import { LlmHarness, BASE_GRAPH } from './lib/llm-harness.ts'

if (process.env.SPINOML_LIVE_LLM !== '1') {
  process.stderr.write('BLOCKED  set SPINOML_LIVE_LLM=1\n')
  process.exit(2)
}

interface Cfg {
  kind: 'openai-compat' | 'anthropic' | 'opencode' | 'subscription'
  label: string
  configured: boolean
  reason: string
  llm: Record<string, unknown>
}

// `command` is a shell builtin, not an executable: ask a shell (execFileSync('command') is ENOENT).
function claudeOnPath(): boolean {
  return spawnSync('sh', ['-c', 'command -v claude'], { stdio: 'ignore' }).status === 0
}

const OAI_KEY = process.env.SPINOML_OPENAI_API_KEY ?? ''
const OAI_BASE = process.env.SPINOML_OPENAI_BASE_URL ?? ''
const OAI_MODEL = process.env.SPINOML_OPENAI_MODEL ?? ''
const ANT_KEY = process.env.ANTHROPIC_API_KEY ?? ''
const ANT_MODEL = process.env.SPINOML_ANTHROPIC_MODEL ?? 'claude-haiku-4-5-20251001'
const OC_MODEL = process.env.SPINOML_OPENCODE_TEST_MODEL ?? ''
const SUB_ON = process.env.SPINOML_LIVE_SUBSCRIPTION === '1' && claudeOnPath()

const cfgs: Cfg[] = [
  {
    kind: 'openai-compat', label: 'openai-compat',
    configured: !!(OAI_KEY && OAI_BASE && OAI_MODEL),
    reason: !(OAI_KEY && OAI_BASE && OAI_MODEL)
      ? 'env SPINOML_OPENAI_API_KEY + _BASE_URL + _MODEL missing' : '',
    llm: { kind: 'openai-compat', apiKey: OAI_KEY, baseUrl: OAI_BASE, model: OAI_MODEL },
  },
  {
    kind: 'anthropic', label: 'anthropic',
    configured: !!ANT_KEY,
    reason: !ANT_KEY ? 'env ANTHROPIC_API_KEY missing' : '',
    llm: { kind: 'anthropic', apiKey: ANT_KEY, model: ANT_MODEL },
  },
  {
    kind: 'opencode', label: 'opencode',
    configured: !!OC_MODEL,
    reason: !OC_MODEL ? 'env SPINOML_OPENCODE_TEST_MODEL missing' : '',
    llm: { kind: 'opencode', model: OC_MODEL },
  },
  {
    kind: 'subscription', label: 'subscription',
    configured: !!SUB_ON,
    reason: !SUB_ON ? 'env SPINOML_LIVE_SUBSCRIPTION=1 + `claude` on PATH required' : '',
    llm: { kind: 'subscription' },
  },
]

const configured = cfgs.filter((c) => c.configured)
if (configured.length === 0) {
  process.stderr.write('BLOCKED  no provider configured. Set one of:\n')
  for (const c of cfgs) process.stderr.write(`  ${c.label}: ${c.reason}\n`)
  process.exit(1)
}

const TOKEN = randomBytes(32).toString('hex')
const harness = await LlmHarness.start({
  env: { SPINOML_SIDECAR_TOKEN: TOKEN, SPINOML_REQUIRE_TOKEN: '1' },
})
const PORT = harness.port

async function runChat(
  llm: Record<string, unknown>,
  timeoutMs: number,
): Promise<{ events: Record<string, unknown>[]; raw: string; fetchError?: string; elapsedMs: number; timedOut: boolean }> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), timeoutMs)
  const raw: string[] = []
  const events: Record<string, unknown>[] = []
  let fetchError: string | undefined
  let timedOut = false
  const start = Date.now()
  try {
    const res = await fetch(`http://127.0.0.1:${PORT}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-SpinoML-Token': TOKEN },
      body: JSON.stringify({
        user: 'Add one Linear layer named probe to the graph, then answer "done".',
        messages: [],
        graph: BASE_GRAPH,
        training_graph: { nodes: [], edges: [] },
        data_graph: { nodes: [], edges: [] },
        llm,
      }),
      signal: ctrl.signal,
    })
    if (!res.body) throw new Error(`no body (HTTP ${res.status})`)
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buf = ''
    for (;;) {
      const { value, done } = await reader.read()
      if (done) break
      const chunk = decoder.decode(value, { stream: true })
      raw.push(chunk)
      buf += chunk
      let idx: number
      while ((idx = buf.indexOf('\n\n')) !== -1) {
        const frame = buf.slice(0, idx)
        buf = buf.slice(idx + 2)
        const line = frame.split('\n').find((l) => l.startsWith('data:'))
        if (!line) continue
        const json = line.slice(5).trim()
        if (!json) continue
        try { events.push(JSON.parse(json) as Record<string, unknown>) }
        catch { /* skip malformed — secret scan below uses raw bytes anyway */ }
      }
    }
  } catch (e) {
    if (ctrl.signal.aborted) timedOut = true
    else fetchError = e instanceof Error ? e.message : String(e)
  } finally { clearTimeout(timer) }
  return { events, raw: raw.join(''), fetchError, elapsedMs: Date.now() - start, timedOut }
}

const SECRETS = [OAI_KEY, ANT_KEY, TOKEN].filter((s) => s.length >= 16)

interface Row { provider: string; status: 'PASS' | 'FAIL' | 'SKIPPED'; detail: string }
const rows: Row[] = []

async function test(cfg: Cfg, timeoutMs: number, expectAction: boolean): Promise<Row> {
  if (!cfg.configured) return { provider: cfg.label, status: 'SKIPPED', detail: cfg.reason }
  const r = await runChat(cfg.llm, timeoutMs)
  const actions = r.events.filter((e) => e.type === 'action')
  const statuses = r.events.filter((e) => e.type === 'status')
  const hasAddLayer = actions.some(
    (a) => a.op === 'add_layer' && a.payload && typeof a.payload === 'object' && (a.payload as Record<string, unknown>).layer_type,
  )
  const doneEv = statuses.some((s) => s.value === 'done')
  const errorEv = statuses.some((s) => s.value === 'error')
  const leaks = SECRETS.filter((s) => r.raw.includes(s))
  const healthy = await harness.health()
  const ok = r.fetchError === undefined
    && !r.timedOut
    && healthy
    && leaks.length === 0
    && (expectAction ? hasAddLayer && doneEv : errorEv && !doneEv)
  const statusStr = statuses.map((s) => `${s.value}${s.message ? ':' + String(s.message).slice(0, 60) : ''}`).join(', ')
  return {
    provider: cfg.label, status: ok ? 'PASS' : 'FAIL',
    detail: `${r.elapsedMs}ms actions=[${actions.map((a) => a.op).join(',')}] statuses=[${statusStr}] leaked=${leaks.length} healthOK=${healthy}${r.timedOut ? ' TIMEOUT' : ''}${r.fetchError ? ' err=' + r.fetchError : ''}`,
  }
}

for (const cfg of cfgs) {
  rows.push(await test(cfg, 90_000, true))
}

if (OC_MODEL) {
  rows.push(await test(
    { ...cfgs[2]!, label: 'opencode bogus-model', llm: { kind: 'opencode', model: 'opencode/does-not-exist-live-test' } },
    120_000,
    false,
  ))
}

console.log('\nLLM live provider suite (gated: SPINOML_LIVE_LLM=1)')
console.log('-'.repeat(110))
for (const r of rows) console.log(`${r.provider.padEnd(22)} ${r.status.padEnd(8)} ${r.detail}`)
console.log('-'.repeat(110))
const failed = rows.filter((r) => r.status === 'FAIL').length
const skipped = rows.filter((r) => r.status === 'SKIPPED').length
console.log(`${configured.length} configured, ${skipped} skipped, ${failed} failed`)
await harness.stop()
process.exit(failed > 0 ? 1 : 0)
