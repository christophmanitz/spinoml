// LLM-safety harness: runs the REAL sidecar-llm/main.mjs against a fake
// OpenAI-compatible provider that scripts exactly what the "model" does, and
// records what the sidecar really does for each hostile/valid scenario.
//
// Run: npm run test:llm-safety
// Exits 1 while any ASSERT scenario fails (each failure is a real finding).

import { promises as fs } from 'node:fs'
import { readFileSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import os from 'node:os'
import path from 'node:path'
import { LlmHarness, BASE_GRAPH, type ChatResult, type GraphSnapshot } from './lib/llm-harness.ts'
import type { FakeStep, ToolCallSpec } from './lib/fake-openai.ts'

const tc = (name: string, args: unknown): ToolCallSpec => ({
  name,
  arguments: typeof args === 'string' ? args : JSON.stringify(args),
})
const calls = (...c: ToolCallSpec[]): FakeStep => ({ toolCalls: c })
const DONE: FakeStep = { text: 'done' }

const GRAPH_OUT: GraphSnapshot = {
  input_shape: [1, 4],
  nodes: [
    ...BASE_GRAPH.nodes,
    { id: 'out', layerType: 'Output', params: { name: 'y' } },
  ],
  edges: BASE_GRAPH.edges,
}

type Kind = 'ASSERT' | 'OBSERVE'
interface Row {
  id: string
  kind: Kind
  model: string
  expected: string
  observed: string
  pass: boolean
  failures: string[]
}
const rows: Row[] = []
let assertions = 0
let findings = 0

class Scenario {
  private readonly failures: string[] = []
  private readonly observedParts: string[] = []
  constructor(
    readonly id: string,
    readonly kind: Kind,
    readonly model: string,
    readonly expected: string,
  ) {}
  check(name: string, cond: boolean, detail = ''): void {
    assertions++
    if (!cond) this.failures.push(`${name}${detail ? ` (${detail})` : ''}`)
  }
  record(s: string): void {
    this.observedParts.push(s)
  }
  get passed(): boolean {
    return this.failures.length === 0
  }
  finish(observed?: string): void {
    const row: Row = {
      id: this.id,
      kind: this.kind,
      model: this.model,
      expected: this.expected,
      observed: observed ?? this.observedParts.join('; '),
      pass: this.passed,
      failures: this.failures,
    }
    rows.push(row)
    if (row.kind === 'ASSERT' && !row.pass) findings++
    const mark = row.kind === 'OBSERVE' ? '○' : row.pass ? '✓' : '✗'
    console.log(`  ${mark} ${row.id} — ${row.observed}`)
    for (const f of row.failures) console.log(`       ↳ ${f}`)
  }
}

const ops = (res: ChatResult): string[] => res.actions.map((a) => a.op)
const hasError = (res: ChatResult): boolean => res.statuses.some((s) => s.value === 'error')
const tr = (res: ChatResult, i = 0): string => {
  const t = res.toolResults[i]
  return t ? `ok:${t.ok} "${t.result}"` : 'none'
}

function badNumbers(v: unknown, trail = ''): string[] {
  const out: string[] = []
  if (typeof v === 'number') {
    if (!Number.isFinite(v) || v < 0) out.push(`${trail || 'value'}=${v}`)
    return out
  }
  if (v && typeof v === 'object') {
    for (const [k, val] of Object.entries(v)) out.push(...badNumbers(val, trail ? `${trail}.${k}` : k))
  }
  return out
}

// ── T1: valid add_layer with after ──────────────────────────────────────────
async function T1(h: LlmHarness): Promise<void> {
  const s = new Scenario('T1', 'ASSERT', 'add_layer{Linear,params,after:"fc"}', 'ok; exactly add_layer+connect actions; tool msg says added')
  const res = await h.chat({
    script: [calls(tc('add_layer', { layer_type: 'Linear', params: { in_features: 4, out_features: 2 }, after: 'fc' })), DONE],
  })
  s.check('tool_result ok', res.toolResults[0]?.ok === true, tr(res))
  const o = ops(res)
  s.check('actions == [add_layer, connect]', JSON.stringify(o) === JSON.stringify(['add_layer', 'connect']), o.join(','))
  const toolMsg = h.fake.toolResultsSeen()[0] ?? ''
  s.check('tool message says added', /added Linear/.test(toolMsg), toolMsg)
  s.record(`${tr(res)} actions=[${o.join(',')}] toolMsg="${toolMsg}" ended=${res.ended}`)
  s.finish()
}

// ── T2: malformed tool arguments ────────────────────────────────────────────
async function T2(h: LlmHarness): Promise<void> {
  const s = new Scenario('T2', 'ASSERT', 'add_layer with malformed JSON args', 'tool_result ok:false explicit; NO action')
  const res = await h.chat({ script: [calls(tc('add_layer', '{"layer_type": "Lin')), DONE] })
  s.check('tool_result ok:false', res.toolResults[0]?.ok === false, tr(res))
  s.check('no action emitted', res.actions.length === 0, `actions=[${ops(res).join(',')}]`)
  s.record(`tool_use.args=${JSON.stringify(res.toolUses[0]?.args)} tool_result=${tr(res)} actions=[${ops(res).join(',')}]`)
  s.finish()
}

// ── T3: unknown tool ────────────────────────────────────────────────────────
async function T3(h: LlmHarness): Promise<void> {
  const s = new Scenario('T3', 'ASSERT', 'unknown tool name', 'ok:false; no action')
  const res = await h.chat({ script: [calls(tc('bash', { cmd: 'rm -rf /' }), tc('nonsense_tool', {})), DONE] })
  s.check('unknown tool #1 ok:false', res.toolResults[0]?.ok === false, tr(res, 0))
  s.check('unknown tool #2 ok:false', res.toolResults[1]?.ok === false, tr(res, 1))
  s.check('no action emitted', res.actions.length === 0, `actions=[${ops(res).join(',')}]`)
  s.record(`${tr(res, 0)} / ${tr(res, 1)} actions=[${ops(res).join(',')}]`)
  s.finish()
}

// ── T4: missing required params ─────────────────────────────────────────────
async function T4(h: LlmHarness): Promise<void> {
  const s = new Scenario('T4', 'ASSERT', 'connect{} / connect{source} / add_layer{}', 'all ok:false; no action')
  const res = await h.chat({
    script: [calls(tc('connect', {}), tc('connect', { source: 'in' }), tc('add_layer', {})), DONE],
  })
  s.check('connect{} ok:false', res.toolResults[0]?.ok === false, tr(res, 0))
  s.check('connect{source} ok:false', res.toolResults[1]?.ok === false, tr(res, 1))
  s.check('add_layer{} ok:false', res.toolResults[2]?.ok === false, tr(res, 2))
  s.check('no action emitted', res.actions.length === 0, `actions=[${ops(res).join(',')}]`)
  s.record(`${tr(res, 0)} / ${tr(res, 1)} / ${tr(res, 2)} actions=[${ops(res).join(',')}]`)
  s.finish()
}

// ── T5: wrong types ─────────────────────────────────────────────────────────
async function T5(h: LlmHarness): Promise<void> {
  const s = new Scenario('T5', 'ASSERT', 'wrong param types', 'all ok:false; no action')
  const res = await h.chat({
    script: [
      calls(
        tc('add_layer', { layer_type: 5 }),
        tc('add_layer', { layer_type: 'Linear', params: 'abc' }),
        tc('add_layer', { layer_type: 'Linear', after: {} }),
        tc('update_params', { id: 'fc', params: 'abc' }),
      ),
      DONE,
    ],
  })
  for (let i = 0; i < 4; i++) s.check(`call#${i + 1} ok:false`, res.toolResults[i]?.ok === false, tr(res, i))
  s.check('no action emitted', res.actions.length === 0, `actions=[${ops(res).join(',')}]`)
  s.record(res.toolResults.map((t, i) => `#${i + 1}:${t.ok}`).join(' ') + ` actions=[${ops(res).join(',')}]`)
  s.finish()
}

// ── T6: unknown layer type ──────────────────────────────────────────────────
async function T6(h: LlmHarness): Promise<void> {
  const s = new Scenario('T6', 'ASSERT', 'add_layer{layer_type:"Banana"}', 'ok:false; no action')
  const res = await h.chat({ script: [calls(tc('add_layer', { layer_type: 'Banana' })), DONE] })
  s.check('ok:false', res.toolResults[0]?.ok === false, tr(res))
  s.check('no action emitted', res.actions.length === 0, `actions=[${ops(res).join(',')}]`)
  s.record(`${tr(res)} actions=[${ops(res).join(',')}] toolMsg="${h.fake.toolResultsSeen()[0] ?? ''}"`)
  s.finish()
}

// ── T7: hostile parameter values ────────────────────────────────────────────
async function T7(h: LlmHarness): Promise<void> {
  const s = new Scenario('T7', 'ASSERT', 'hostile add_layer params', 'ok:false OR sanitized; no action with non-finite/negative dims')
  const res = await h.chat({
    script: [
      calls(
        tc('add_layer', { layer_type: 'Linear', params: { in_features: -5, out_features: 2 } }),
        tc('add_layer', '{"layer_type":"Linear","params":{"in_features":1e999,"out_features":2}}'),
        tc('add_layer', { layer_type: 'Linear', params: { in_features: 'NaN', out_features: 2 } }),
        tc('add_layer', { layer_type: 'Linear', params: { in_features: 1e12, out_features: 2 } }),
        tc('add_layer', { layer_type: 'Linear', params: null }),
        tc('add_layer', { layer_type: 'Linear', params: { nested: { deep: { out_features: -3 } } } }),
        tc('add_layer', '{"layer_type":"Linear","params":{"__proto__":{"in_features":7},"in_features":8}}'),
      ),
      DONE,
    ],
  })
  const violations: string[] = []
  for (const a of res.actions) {
    if (a.op !== 'add_layer') continue
    const params = a.payload.params
    violations.push(...badNumbers(params))
    if (params && typeof params === 'object') {
      const p = params as Record<string, unknown>
      for (const key of ['in_features', 'out_features', 'in_channels', 'out_channels']) {
        const v = p[key]
        if (v !== undefined && (typeof v !== 'number' || !Number.isFinite(v) || v <= 0)) violations.push(`${key}=${JSON.stringify(v)} (not a positive number)`)
      }
    }
  }
  s.check('no action carries bad dimensions', violations.length === 0, violations.join(', '))
  const anyOk = res.toolResults.some((t) => t.ok)
  s.check('all hostile calls rejected OR sanitized', !anyOk || violations.length === 0, `ok flips=${res.toolResults.map((t) => t.ok).join(',')}`)
  s.record(`ok=[${res.toolResults.map((t) => t.ok).join(',')}] actions=[${ops(res).join(',')}] violations=[${violations.join(' | ')}]`)
  s.finish()
}

// ── T8: unknown `after` must not mutate state ───────────────────────────────
async function T8(h: LlmHarness): Promise<void> {
  const s = new Scenario('T8', 'ASSERT', 'add_layer{after:"ghost"} then connect llm1', 'ok:false; no add_layer action; llm1 unknown')
  const res = await h.chat({
    script: [
      calls(tc('add_layer', { layer_type: 'Linear', after: 'ghost' })),
      calls(tc('connect', { source: 'llm1', target: 'fc' })),
      DONE,
    ],
  })
  s.check('add_layer ok:false', res.toolResults[0]?.ok === false, tr(res, 0))
  s.check('no add_layer action', !ops(res).includes('add_layer'), `actions=[${ops(res).join(',')}]`)
  s.check('connect to llm1 ok:false (state unchanged)', res.toolResults[1]?.ok === false, tr(res, 1))
  s.record(`${tr(res, 0)} then ${tr(res, 1)} actions=[${ops(res).join(',')}]`)
  s.finish()
}

// ── T9: bad/duplicate/self/cyclic edges & unknown ids ───────────────────────
async function T9(h: LlmHarness): Promise<void> {
  const s = new Scenario('T9', 'ASSERT', 'connect unknown/dup/self/cycle; delete/update unknown', 'bad ops rejected; duplicate idempotent (ok:true, "already exists", no action)')
  const res = await h.chat({
    script: [
      calls(
        tc('connect', { source: 'ghost', target: 'fc' }),
        tc('connect', { source: 'in', target: 'ghost' }),
        tc('connect', { source: 'in', target: 'fc' }),
        tc('connect', { source: 'in', target: 'fc' }),
        tc('connect', { source: 'fc', target: 'fc' }),
        tc('connect', { source: 'fc', target: 'in' }),
        tc('delete_node', { id: 'ghost' }),
        tc('update_params', { id: 'ghost', params: { x: 1 } }),
      ),
      DONE,
    ],
  })
  s.check('connect ghost src ok:false', res.toolResults[0]?.ok === false, tr(res, 0))
  s.check('connect ghost dst ok:false', res.toolResults[1]?.ok === false, tr(res, 1))
  s.check('connect duplicate ok:true', res.toolResults[3]?.ok === true, tr(res, 3))
  s.check('duplicate message says already exists', /already exists/.test(res.toolResults[3]?.result ?? ''), tr(res, 3))
  s.check('self-connect ok:false', res.toolResults[4]?.ok === false, tr(res, 4))
  s.check('cycle-connect ok:false', res.toolResults[5]?.ok === false, tr(res, 5))
  s.check('delete unknown ok:false', res.toolResults[6]?.ok === false, tr(res, 6))
  s.check('update unknown ok:false', res.toolResults[7]?.ok === false, tr(res, 7))
  const connectActions = res.actions.filter((a) => a.op === 'connect')
  // in->fc already exists in BASE_GRAPH, so the two connect calls are no-ops.
  s.check('no connect action (duplicate is a no-op)', connectActions.length === 0, `connect actions=${connectActions.length}`)
  s.check('no self/cycle action', !ops(res).includes('connect'), `actions=[${ops(res).join(',')}]`)
  s.check('no delete/update action', !ops(res).some((o) => o === 'delete_node' || o === 'update_params'), `actions=[${ops(res).join(',')}]`)
  s.record(`ok=[${res.toolResults.map((t) => (t.ok ? 'T' : 'F')).join('')}] connectActions=${connectActions.length} actions=[${ops(res).join(',')}]`)
  s.finish()
}

// ── T10a: no non-graph powers / hostile paths ───────────────────────────────
async function T10a(h: LlmHarness, root: string): Promise<void> {
  const s = new Scenario('T10a', 'ASSERT', 'bash + read_file/write_file hostile paths', 'unknown tool or explicit path error; no action')
  const res = await h.chat({
    project: { root },
    script: [
      calls(
        tc('bash', { cmd: 'id' }),
        tc('read_file', { path: '../../etc/passwd' }),
        tc('read_file', { path: '/etc/shadow' }),
        tc('write_file', { path: '../../evil.sh', content: 'x' }),
      ),
      DONE,
    ],
  })
  s.check('bash ok:false', res.toolResults[0]?.ok === false, tr(res, 0))
  s.check('read_file ../../ ok:false', res.toolResults[1]?.ok === false, tr(res, 1))
  s.check('read_file /etc/shadow ok:false', res.toolResults[2]?.ok === false, tr(res, 2))
  s.check('write_file ../../ ok:false', res.toolResults[3]?.ok === false, tr(res, 3))
  s.check('no action emitted', res.actions.length === 0, `actions=[${ops(res).join(',')}]`)
  s.record(`ok=[${res.toolResults.map((t) => (t.ok ? 'T' : 'F')).join('')}] results=[${res.toolResults.map((t) => t.result).join(' | ')}]`)
  s.finish()
}

// ── T10b: run_script confirm gate ───────────────────────────────────────────
async function T10b(h: LlmHarness, root: string, marker: string): Promise<void> {
  const s = new Scenario('T10b', 'ASSERT', 'run_script hostile + valid, answer "no"', 'ask confirm before run; declined => nothing ran')
  const res = await h.chat({
    project: { root },
    onAsk: () => 'no',
    script: [
      calls(
        tc('run_script', { path: '../../etc/passwd', mode: 'shell' }),
        tc('run_script', { path: 'agent/probe.sh', mode: 'shell' }),
      ),
      DONE,
    ],
  })
  s.check('run_script hostile ok:false', res.toolResults[0]?.ok === false, tr(res, 0))
  s.check('run_script valid asked for confirm', res.asks.length >= 1, `asks=${res.asks.length}`)
  s.check('ask kind is confirm', res.asks.some((a) => a.kind === 'confirm'), res.asks.map((a) => a.kind).join(','))
  s.check('run_script valid ok:false after decline', res.toolResults[1]?.ok === false, tr(res, 1))
  const ran = await fs
    .access(marker)
    .then(() => true)
    .catch(() => false)
  s.check('declined script did NOT run (no marker)', !ran, `marker exists=${ran}`)
  s.record(`asks=${res.asks.length} ${tr(res, 0)} / ${tr(res, 1)} markerRan=${ran}`)
  s.finish()
}

// ── T11 OBSERVE: delete Input / Output ──────────────────────────────────────
async function T11(h: LlmHarness): Promise<void> {
  const s = new Scenario('T11', 'OBSERVE', 'delete_node{in}, delete_node{out}', 'record only: action emitted? graph-invalid admitted?')
  const res = await h.chat({
    graph: GRAPH_OUT,
    script: [calls(tc('delete_node', { id: 'in' }), tc('delete_node', { id: 'out' })), DONE],
  })
  s.record(`results=[${res.toolResults.map((t) => `${t.ok ? 'ok' : 'err'}:${t.result}`).join(' | ')}] actions=[${ops(res).join(',')}]`)
  s.finish()
}

// ── T12 OBSERVE: 30 mixed calls in one turn ─────────────────────────────────
async function T12(h: LlmHarness): Promise<void> {
  const s = new Scenario('T12', 'OBSERVE', '30 mixed valid/invalid calls in one turn', 'record only: final action sequence')
  const mixed: ToolCallSpec[] = []
  for (let i = 0; i < 30; i++) {
    if (i % 3 === 0) mixed.push(tc('add_layer', { layer_type: 'ReLU' }))
    else if (i % 3 === 1) mixed.push(tc('add_layer', {}))
    else mixed.push(tc('nonsense_tool', { i }))
  }
  const res = await h.chat({ script: [calls(...mixed), DONE] })
  s.record(`toolResults=${res.toolResults.length} okCount=${res.toolResults.filter((t) => t.ok).length} actions=[${ops(res).join(',')}]`)
  s.finish()
}

// ── T13: update_params with a junk value must not mutate ────────────────────
async function T13(h: LlmHarness): Promise<void> {
  const s = new Scenario('T13', 'ASSERT', 'update_params{fc, out_features:"junk"}', 'ok:false; no action')
  const res = await h.chat({
    script: [calls(tc('update_params', { id: 'fc', params: { out_features: 'not-a-number' } })), DONE],
  })
  s.check('ok:false', res.toolResults[0]?.ok === false, tr(res))
  s.check('no action emitted', res.actions.length === 0, `actions=[${ops(res).join(',')}]`)
  s.record(`${tr(res)} actions=[${ops(res).join(',')}]`)
  s.finish()
}

// ── T14: unknown parameter key is rejected with the valid keys ──────────────
async function T14(h: LlmHarness): Promise<void> {
  const s = new Scenario('T14', 'ASSERT', 'add_layer{Linear, params{... bogus}}', 'ok:false; error lists valid keys; no action')
  const res = await h.chat({
    script: [calls(tc('add_layer', { layer_type: 'Linear', params: { in_features: 4, out_features: 2, bogus: 1 } })), DONE],
  })
  const msg = res.toolResults[0]?.result ?? ''
  s.check('ok:false', res.toolResults[0]?.ok === false, tr(res))
  s.check('error names the bad key', /bogus/.test(msg), msg)
  s.check('error lists valid keys', /valid keys/.test(msg) && /in_features/.test(msg), msg)
  s.check('no action emitted', res.actions.length === 0, `actions=[${ops(res).join(',')}]`)
  s.record(`${tr(res)} actions=[${ops(res).join(',')}]`)
  s.finish()
}

// ── T14b: an unknown TOP-LEVEL argument is rejected, not silently stripped ───
async function T14b(h: LlmHarness): Promise<void> {
  const s = new Scenario('T14b', 'ASSERT', 'add_layer{layer_type, activation:"relu"} / connect{source,target,weight:1} / __proto__', 'ok:false; names the argument and the valid ones; no action')
  const res = await h.chat({
    script: [calls(
      tc('add_layer', { layer_type: 'Linear', activation: 'relu' }),
      tc('connect', { source: 'in', target: 'fc', weight: 1 }),
      { name: 'add_layer', arguments: '{"layer_type":"Linear","__proto__":{"x":1}}' },
    ), DONE],
  })
  const msgs = res.toolResults.map((r) => r.result ?? '')
  s.check('three tool results', res.toolResults.length === 3, tr(res))
  s.check('all rejected (ok:false)', res.toolResults.every((r) => r.ok === false), tr(res))
  s.check('error names the stripped argument', /"activation"/.test(msgs[0] ?? '') && /"weight"/.test(msgs[1] ?? ''), msgs.join(' | '))
  s.check('error lists the valid arguments', /valid: .*layer_type/.test(msgs[0] ?? ''), msgs[0] ?? '')
  s.check('prototype key rejected too', /unknown argument/.test(msgs[2] ?? ''), msgs[2] ?? '')
  s.check('no action emitted', res.actions.length === 0, `actions=[${ops(res).join(',')}]`)
  s.record(`${tr(res)} actions=[${ops(res).join(',')}]`)
  s.finish()
}

// ── T15: numeric strings are accepted and normalised to numbers ─────────────
async function T15(h: LlmHarness): Promise<void> {
  const s = new Scenario('T15', 'ASSERT', 'add_layer{Linear, params{in_features:"4", out_features:"64"}}', 'ok:true; action params normalised to numbers')
  const res = await h.chat({
    script: [calls(tc('add_layer', { layer_type: 'Linear', params: { in_features: '4', out_features: '64' } })), DONE],
  })
  s.check('ok:true', res.toolResults[0]?.ok === true, tr(res))
  const action = res.actions.find((a) => a.op === 'add_layer')
  const p = (action?.payload.params ?? {}) as Record<string, unknown>
  s.check('out_features normalised to number 64', p.out_features === 64 && typeof p.out_features === 'number', JSON.stringify(p))
  s.check('in_features normalised to number 4', p.in_features === 4 && typeof p.in_features === 'number', JSON.stringify(p))
  s.record(`${tr(res)} params=${JSON.stringify(p)} actions=[${ops(res).join(',')}]`)
  s.finish()
}

// ── T16: an Input node cannot be the target of a connect ────────────────────
async function T16(h: LlmHarness): Promise<void> {
  const s = new Scenario('T16', 'ASSERT', 'connect{fc -> in} (Input as target)', 'ok:false; no action')
  const graph: GraphSnapshot = {
    input_shape: [1, 4],
    nodes: [
      { id: 'in', layerType: 'Input', params: { name: 'x', shape: [1, 4], dtype: 'float32' } },
      { id: 'fc', layerType: 'Linear', params: { in_features: 4, out_features: 2 } },
    ],
    edges: [],
  }
  const res = await h.chat({ graph, script: [calls(tc('connect', { source: 'fc', target: 'in' })), DONE] })
  s.check('ok:false', res.toolResults[0]?.ok === false, tr(res))
  s.check('error mentions Input', /Input/.test(res.toolResults[0]?.result ?? ''), tr(res))
  s.check('no action emitted', res.actions.length === 0, `actions=[${ops(res).join(',')}]`)
  s.record(`${tr(res)} actions=[${ops(res).join(',')}]`)
  s.finish()
}

// ── T17: the same checks on a training-graph and a data-graph tool ──────────
async function T17(h: LlmHarness): Promise<void> {
  const s = new Scenario('T17', 'ASSERT', 'training + data add/update invalid values', 'bad calls ok:false, no action; valid adds ok')
  const res = await h.chat({
    script: [
      calls(
        tc('add_training_node', { node_type: 'Nope' }),
        tc('add_data_node', { node_type: 'Nope' }),
        tc('add_training_node', { node_type: 'Optimizer', params: { lr: -1 } }),
        tc('add_data_node', { node_type: 'TableSource', params: { dataset: 5 } }),
        tc('add_training_node', { node_type: 'Optimizer' }),
        tc('update_training_params', { id: 'tllm1', params: { lr: 'junk' } }),
        tc('add_data_node', { node_type: 'TableSource' }),
        tc('update_data_params', { id: 'dllm1', params: { dataset: 5 } }),
      ),
      DONE,
    ],
  })
  const ok = res.toolResults.map((t) => t.ok)
  s.check('no bad training/node type', ok[0] === false, tr(res, 0))
  s.check('no bad data/node type', ok[1] === false, tr(res, 1))
  s.check('bad training param rejected', ok[2] === false, tr(res, 2))
  s.check('bad data param rejected', ok[3] === false, tr(res, 3))
  s.check('valid training add ok', ok[4] === true, tr(res, 4))
  s.check('bad training update rejected', ok[5] === false, tr(res, 5))
  s.check('valid data add ok', ok[6] === true, tr(res, 6))
  s.check('bad data update rejected', ok[7] === false, tr(res, 7))
  const a = ops(res)
  s.check('only the two valid adds emitted actions', JSON.stringify(a) === JSON.stringify(['training:add_node', 'data:add_node']), a.join(','))
  s.record(`ok=[${ok.map((t) => (t ? 'T' : 'F')).join('')}] actions=[${a.join(',')}]`)
  s.finish()
}

// ── P1: provider failures ───────────────────────────────────────────────────
async function p1Case(h: LlmHarness, id: string, model: string, script: FakeStep[], llm?: { baseUrl: string }, expectNoAction = false): Promise<void> {
  const s = new Scenario(id, 'ASSERT', model, 'explicit error status within 20s; sidecar healthy; no partial tool run')
  const res = await h.chat({ script, llm, timeoutMs: 20000 })
  s.check('turn ended', res.ended, `ended=${res.ended} err=${res.fetchError ?? ''}`)
  s.check('explicit error status', hasError(res), `statuses=[${res.statuses.map((x) => x.value).join(',')}]`)
  if (expectNoAction) s.check('no partial tool executed', res.actions.length === 0, `actions=[${ops(res).join(',')}]`)
  s.check('within 20s', res.elapsedMs < 20000, `${res.elapsedMs}ms`)
  s.check('sidecar healthy after', await h.health())
  s.record(`ended=${res.ended} statuses=[${res.statuses.map((x) => x.value + (x.message ? `:${x.message.slice(0, 80)}` : '')).join(', ')}] ${res.elapsedMs}ms`)
  s.finish()
}

async function P1(h: LlmHarness): Promise<void> {
  await p1Case(h, 'P1.1', 'provider HTTP 500', [{ status: 500, repeat: true }])
  await p1Case(h, 'P1.2', 'provider HTTP 401 echoing apiKey', [{ status: 401, body: JSON.stringify({ error: { message: 'invalid api key sk-P1ECHO' } }), repeat: true }])
  await p1Case(h, 'P1.3', 'provider HTTP 429', [{ status: 429, repeat: true }])
  await p1Case(h, 'P1.4', 'connection refused', [], { baseUrl: 'http://127.0.0.1:1/v1' })
  await p1Case(h, 'P1.5', 'body not SSE (<html>)', [{ raw: '<html>oops</html>', repeat: true }])
  await p1Case(h, 'P1.6', 'invalid JSON in SSE data line', [{ raw: 'data: {not valid json}\n\ndn\n', repeat: true }])
  await p1Case(
    h,
    'P1.7',
    'stream truncated mid tool call',
    [{ toolCalls: [tc('add_layer', { layer_type: 'Linear', params: { in_features: 4, out_features: 2 } })], truncateAfterBytes: 350, repeat: true }],
    undefined,
    true,
  )
}

// ── P2: provider hangs; client aborts ───────────────────────────────────────
async function P2(h: LlmHarness): Promise<void> {
  const s = new Scenario('P2', 'ASSERT', 'provider hangs; client aborts /chat after 2s', 'upstream closed within 5s; explicit upstream timeout exists')
  const sidecarSrc = readFileSync('sidecar-llm/main.mjs', 'utf8')
  const hasUpstreamTimeout = sidecarSrc.includes('SPINOML_LLM_UPSTREAM_TIMEOUT_MS')
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), 2000)
  const res = await h.chat({ script: [{ hang: true }], signal: ctrl.signal, timeoutMs: 20000 })
  clearTimeout(timer)
  s.check('chat aborted by client', res.aborted, `aborted=${res.aborted}`)
  const closed = await h.fake.waitForHungClose(5000)
  const hangStarts = h.fake.hangStarts
  const hangCloses = h.fake.hangCloses
  s.check('sidecar closed upstream within 5s', closed, `hangStarts=${hangStarts} hangCloses=${hangCloses}`)
  s.check('upstream timeout env exists in sidecar', hasUpstreamTimeout, 'no SPINOML_LLM_UPSTREAM_TIMEOUT_MS in main.mjs')
  const next = await h.chat({ script: [calls(tc('add_layer', { layer_type: 'ReLU' })), DONE] })
  s.check('sidecar serves next chat', next.ended && next.toolResults[0]?.ok === true, `ended=${next.ended}`)
  s.check('sidecar healthy', await h.health())
  s.record(`aborted=${res.aborted} upstreamClosedIn5s=${closed} hangStarts=${hangStarts} hangCloses=${hangCloses} upstreamTimeoutEnv=${hasUpstreamTimeout}`)
  s.finish()
}

// ── P2b: idle timeout aborts a stalling provider ────────────────────────────
async function P2b(): Promise<void> {
  const s = new Scenario('P2b', 'ASSERT', 'provider sends one chunk then stalls (idle timeout 1.5s)', 'explicit "provider stalled" error within 5s')
  const h2 = await LlmHarness.start({ env: { SPINOML_LLM_UPSTREAM_TIMEOUT_MS: '1500' } })
  try {
    const res = await h2.chat({ script: [{ stall: true }], timeoutMs: 8000 })
    s.check('turn ended', res.ended, `ended=${res.ended} err=${res.fetchError ?? ''}`)
    s.check('explicit error status', hasError(res), `statuses=[${res.statuses.map((x) => x.value).join(',')}]`)
    s.check('error says provider stalled', res.statuses.some((x) => /provider stalled/.test(x.message ?? '')), `statuses=[${res.statuses.map((x) => `${x.value}:${x.message ?? ''}`).join(', ')}]`)
    s.check('within 5s', res.elapsedMs < 5000, `${res.elapsedMs}ms`)
    s.check('sidecar healthy after', await h2.health())
    s.record(`ended=${res.ended} statuses=[${res.statuses.map((x) => x.value + (x.message ? `:${x.message.slice(0, 60)}` : '')).join(', ')}] ${res.elapsedMs}ms`)
  } finally {
    await h2.stop()
  }
  s.finish()
}

// ── P3: runaway tool-call loop ──────────────────────────────────────────────
async function P3(h: LlmHarness): Promise<void> {
  const s = new Scenario('P3', 'ASSERT', 'provider requests tool calls forever', 'confirm ask at MAX_TOOL_TURNS; "no" stops; requests bounded')
  let asked = false
  const res = await h.chat({
    script: [calls(tc('add_layer', { layer_type: 'Linear' }))],
    loop: true,
    timeoutMs: 60000,
    onAsk: () => {
      asked = true
      return 'no'
    },
  })
  s.check('confirm ask appeared', asked && res.asks.length >= 1, `asks=${res.asks.length}`)
  s.check('ask is max_turns confirm', res.asks.some((a) => a.payload.reason === 'max_turns'), JSON.stringify(res.asks.map((a) => a.payload)))
  s.check('turn stopped with error status', hasError(res), `statuses=[${res.statuses.map((x) => x.value + (x.message ? `:${x.message}` : '')).join(',')}]`)
  s.check('requests bounded (~100)', h.fake.requests.length <= 110 && h.fake.requests.length >= 95, `requests=${h.fake.requests.length}`)
  s.record(`requests=${h.fake.requests.length} asks=${res.asks.length} statuses=[${res.statuses.map((x) => x.value).join(',')}] ended=${res.ended}`)
  s.finish()
}

// ── S1: secrets must never leak ─────────────────────────────────────────────
async function S1(h: LlmHarness, key: string): Promise<void> {
  const s = new Scenario('S1', 'ASSERT', 'apiKey sk-TESTSECRET-*', 'used in Authorization; absent from SSE, stdout/stderr, /health, /models')
  await h.chat({
    llm: { apiKey: key },
    script: [calls(tc('add_layer', { layer_type: 'ReLU' })), DONE],
  })
  const errRes = await h.chat({
    llm: { apiKey: key },
    script: [{ status: 401, body: JSON.stringify({ error: { message: `invalid api key ${key}` } }), repeat: true }],
  })
  const used = h.fake.authHeaders().some((a) => a.includes(key))
  s.check('apiKey present in Authorization header', used, h.fake.authHeaders().join(','))
  const surfaces: Record<string, string> = {
    sse: JSON.stringify(errRes.events),
    stdout: h.stdout,
    stderr: h.stderr,
    health: await h.healthBody(),
    models: await h.opencodeModelsBody(),
  }
  for (const [name, text] of Object.entries(surfaces)) {
    s.check(`apiKey absent from ${name}`, !text.includes(key), `found in ${name}`)
  }
  s.check('error turn reported explicitly', hasError(errRes), `statuses=[${errRes.statuses.map((x) => x.value).join(',')}]`)
  s.record(`usedInAuth=${used} leakedInto=[${Object.entries(surfaces).filter(([, t]) => t.includes(key)).map(([n]) => n).join(',') || 'none'}]`)
  s.finish()
}

// True only for a real `node …/sidecar-llm/main.mjs` process. A naive
// substring match would also hit unrelated agents whose command line mentions
// the sidecar path (e.g. this task's own prompt).
function psLeftover(): boolean {
  try {
    const out = execFileSync('ps', ['-eo', 'args'], { encoding: 'utf8' })
    return out.split('\n').some((line) => {
      const parts = line.trim().split(/\s+/)
      return parts.length >= 2 && /(^|\/)node$/.test(parts[0]) && parts[1].endsWith('sidecar-llm/main.mjs')
    })
  } catch {
    return false
  }
}

async function main(): Promise<void> {
  console.log('LLM safety harness — real sidecar vs. scripted fake provider\n')
  const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'spinoml-llm-safety-'))
  const marker = path.join(tmp, 'ran.marker')
  await fs.mkdir(path.join(tmp, 'agent'), { recursive: true })
  await fs.writeFile(path.join(tmp, 'agent', 'probe.sh'), `#!/bin/sh\ntouch "${marker}"\n`, 'utf8')

  let h: LlmHarness | null = null
  let healthAfter: boolean | undefined
  try {
    h = await LlmHarness.start()
    const key = `sk-TESTSECRET-${Math.random().toString(36).slice(2)}`

    await T1(h)
    await T2(h)
    await T3(h)
    await T4(h)
    await T5(h)
    await T6(h)
    await T7(h)
    await T8(h)
    await T9(h)
    await T10a(h, tmp)
    await T10b(h, tmp, marker)
    await T11(h)
    await T12(h)
    await T13(h)
    await T14(h)
    await T14b(h)
    await T15(h)
    await T16(h)
    await T17(h)
    await P1(h)
    await P2(h)
    await P2b()
    await P3(h)
    await S1(h, key)

    healthAfter = await h.health()
  } finally {
    if (h) await h.stop()
  }

  const z = new Scenario('Z1', 'ASSERT', 'process hygiene after the matrix', 'sidecar healthy; no leftover sidecar processes')
  const leftover = psLeftover()
  z.check('sidecar healthy after matrix', healthAfter)
  z.check('no leftover sidecar-llm process', !leftover, `psLeftover=${leftover}`)
  z.record(`healthAfter=${healthAfter} psLeftover=${leftover}`)
  z.finish()

  await fs.rm(tmp, { recursive: true, force: true })

  console.log('\n── scenario table ───────────────────────────────────────────────')
  for (const r of rows) {
    const verdict = r.kind === 'OBSERVE' ? 'OBSERVE' : r.pass ? 'PASS' : 'FINDING'
    console.log(`${r.id.padEnd(5)} ${verdict.padEnd(8)} ${r.model}`)
    console.log(`      expected: ${r.expected}`)
    console.log(`      observed: ${r.observed}`)
  }

  console.log(`\nassertions: ${assertions}, findings: ${findings}`)
  process.exitCode = findings > 0 ? 1 : 0
}

main().catch((e) => {
  console.error('harness crashed:', e)
  process.exitCode = 2
})
