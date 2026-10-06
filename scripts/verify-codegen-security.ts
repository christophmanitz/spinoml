#!/usr/bin/env tsx
// Phase 43 — Codegen security regression harness.
//
// Feeds adversarial payloads (newline/CR/NUL, raw backslash+quote, JS leaks,
// sentinel function call) into every string/numeric interpolation site of the
// three SpinoML code generators (model, data, training) and checks the produced
// Python via `python3 -I scripts/lib/py_inert_check.py`:
//   parses  = ast.parse succeeds
//   inert   = parses AND no tokenize NAME token equals PWN_SENTINEL AND no
//             NAME token is one of {NaN, Infinity, undefined, null}
//   PASS    = parses && inert (model generator may also pass via issues>0).
//
// Run against the unfixed code in step 1 (must fail) and against the fixed code
// in step 2 (must pass). Numeric hostile values must fall back to the default.

import { execFileSync } from 'node:child_process'
import type { Edge } from '@xyflow/react'
import {
  LAYERS,
  defaultParamsFor,
  coerceParams,
  type FieldSpec,
} from '../src/layers/registry'
import { DATA_NODES, defaultDataParams } from '../src/data/graph/registry'
import { generate } from '../src/codegen/generator'
import { generateNodeCode, generateDataCode } from '../src/codegen/dataCodegen'
import { generateTrainingCode } from '../src/codegen/trainingCodegen'
import type { TrainingPlan } from '../src/codegen/trainingGenerator'
import { defaultTrainingConfig } from '../src/training/types'
import type { LayerNode } from '../src/canvas/GraphStore'

const SENTINEL = 'PWN_SENTINEL'

// The exact 10 adversarial payloads from the brief (P1 … P10).
const PAYLOADS: string[] = [
  `x'); ${SENTINEL}(); ('`,                 // P1
  `x\n${SENTINEL}()\n#`,                   // P2
  `x\r${SENTINEL}()\r#`,                   // P3
  `x\u2028${SENTINEL}()`,                  // P4 (U+2028 LINE SEPARATOR)
  `x\\'); ${SENTINEL}(); ('`,              // P5 (backslash before quote)
  `x"""; ${SENTINEL}(); """`,              // P6
  `x\u0085${SENTINEL}()`,                  // P7 (U+0085 NEL)
  `x\0y`,                                  // P8 (NUL)
  '{' + SENTINEL + '()}${' + SENTINEL + '()}', // P9 (concat to keep `}${` literal)
  `x\x0b${SENTINEL}()\x0c#`,               // P10 (VT/FF)
]

const STRING_FIELD_TYPES = new Set<FieldSpec['type']>([
  'select', 'dataset-ref', 'column-single', 'columns-multi', 'text',
])
const NUMERIC_FIELD_TYPES = new Set<FieldSpec['type']>([
  'int', 'float', 'tuple-int', 'int-list', 'shape',
])

// By-design arbitrary-code sinks. Adding a new code-typed field requires
// editing this list (the harness fails loud if a new one sneaks in or an
// existing one is removed). `Custom.init_args` is code by design too (gated by
// the trust store, kind 'custom-init-args') — only STRUCTURAL safety is checked.
const INTENTIONAL_CODE_SINKS = [
  'Custom.source',     // code by design
  'CustomScript.code', // code by design
  'DataOp.script',     // code by design
  "Custom.init_args",  // code by design; gated by the trust store, kind 'custom-init-args'
]
// The subset that is a `code`-typed FieldSpec (init_args is `text` but emitted
// as code) — used for the exact "a new code field must be added deliberately"
// assertion.
const CODE_TYPED_SINKS = INTENTIONAL_CODE_SINKS.filter((s) => s !== 'Custom.init_args')

function findCodeSinks(): string[] {
  const sinks: string[] = []
  for (const [layerType, spec] of Object.entries(LAYERS)) {
    for (const f of spec.fields) if (f.type === 'code') sinks.push(`${layerType}.${f.name}`)
  }
  for (const [dataType, spec] of Object.entries(DATA_NODES)) {
    for (const f of spec.fields) if (f.type === 'code') sinks.push(`${dataType}.${f.name}`)
  }
  return sinks.sort()
}

type Sink = 'model' | 'data' | 'training' | 'literals' | 'numbers-model' | 'numbers-data' | 'numbers-training'

type Case = {
  id: string
  code: string
  sink: Sink
  payload: number   // 0..9 = payload index, -1 = baseline/no-payload
  issues?: string[]
  threw?: boolean
}

type Literal = { id: string; literal: string; expected: string; payload: number }

const cases: Case[] = []
const literals: Literal[] = []

// ── Helpers ───────────────────────────────────────────────────────────────

function mkLayerNode(
  id: string,
  layerType: string,
  params: Record<string, unknown>,
  extraData: Record<string, unknown> = {},
): LayerNode {
  return {
    id,
    type: 'layer',
    position: { x: 0, y: 0 },
    data: { layerType, params, ...extraData },
  } as LayerNode
}

function runModelCase(
  id: string,
  layerType: string,
  params: Record<string, unknown>,
  nodeId: string,
  extraData: Record<string, unknown> = {},
): Case {
  const nodes = [mkLayerNode('in', 'Input', defaultParamsFor('Input'))]
  const isInputKind = LAYERS[layerType]?.kind === 'input'
  nodes.push(mkLayerNode(nodeId, layerType, params, extraData))
  const edges: Edge[] = isInputKind ? [] : [{ id: 'e1', source: 'in', target: nodeId }]
  try {
    const r = generate(nodes, edges)
    return { id, code: r.code, issues: r.issues, sink: 'model', payload: -1 }
  } catch {
    return { id, code: '', issues: [], threw: true, sink: 'model', payload: -1 }
  }
}

function baseTrainingPlan(): TrainingPlan {
  return {
    modelRelpath: 'models/m.spinoml',
    datasetRelpath: 'datasets/ds.csv',
    target: 'y',
    features: ['f1', 'f2'],
    training: {
      ...defaultTrainingConfig(),
      epochs: 3,
      batch_size: 2,
      seed: 7,
      log_every_n_steps: 1,
      val_every_n_epochs: 1,
      gradient_accumulation_steps: 1,
      optimizer: { kind: 'Adam', lr: 0.01, weight_decay: 0 },
      loss: { kind: 'CrossEntropyLoss' },
      scheduler: { kind: 'none' },
      metrics: ['accuracy'],
      callbacks: [],
    },
  }
}

// ── 1. MODEL: every layer × every string field × every payload × coerced+raw

for (const [layerType, spec] of Object.entries(LAYERS)) {
  for (const field of spec.fields) {
    if (!STRING_FIELD_TYPES.has(field.type)) continue
    for (let pi = 0; pi < PAYLOADS.length; pi++) {
      const payload = PAYLOADS[pi]
      const value: unknown = field.type === 'columns-multi' ? [payload] : payload
      const defs = defaultParamsFor(layerType)

      // coerced (passes through coerceParams like the app does)
      {
        const params = coerceParams(layerType, { ...defs, [field.name]: value })
        const c = runModelCase(
          `M/${layerType}/${field.name}/coerced/P${pi + 1}`,
          layerType, params, 't',
        )
        c.payload = pi
        cases.push(c)
      }
      // raw (bypasses coerce to exercise serializeParam/escaping directly)
      {
        const params = { ...defs, [field.name]: value }
        const c = runModelCase(
          `M/${layerType}/${field.name}/raw/P${pi + 1}`,
          layerType, params, 't',
        )
        c.payload = pi
        cases.push(c)
      }
    }
  }
}

// 1a. Output `name` in a multi-output graph (exercises the dict-key emission
// path, which is only reached when ≥2 Output nodes share a forward predecessor).
for (let pi = 0; pi < PAYLOADS.length; pi++) {
  const payload = PAYLOADS[pi]
  const params = { ...defaultParamsFor('Output'), name: payload }
  const nodes = [
    mkLayerNode('in', 'Input', defaultParamsFor('Input')),
    mkLayerNode('o1', 'Output', params),
    mkLayerNode('o2', 'Output', params),
  ]
  const edges: Edge[] = [
    { id: 'e1', source: 'in', target: 'o1' },
    { id: 'e2', source: 'in', target: 'o2' },
  ]
  try {
    const r = generate(nodes, edges)
    cases.push({ id: `M/Output/name/multi-raw/P${pi + 1}`, code: r.code, issues: r.issues, sink: 'model', payload: pi })
  } catch {
    cases.push({ id: `M/Output/name/multi-raw/P${pi + 1}`, code: '', threw: true, sink: 'model', payload: pi })
  }
}

// 1b. extra unknown param keys (label, name-on-non-input, comment) per layer
for (const [layerType] of Object.entries(LAYERS)) {
  const defs = defaultParamsFor(layerType)
  const isInputKind = LAYERS[layerType]?.kind === 'input'
  for (let pi = 0; pi < PAYLOADS.length; pi++) {
    const payload = PAYLOADS[pi]
    for (const key of ['label', 'comment']) {
      const params = { ...defs, [key]: payload }
      const c = runModelCase(`M/${layerType}/extra.${key}/P${pi + 1}`, layerType, params, 't')
      c.payload = pi
      cases.push(c)
    }
    if (!isInputKind) {
      const params = { ...defs, name: payload }
      const c = runModelCase(`M/${layerType}/extra.name/P${pi + 1}`, layerType, params, 't')
      c.payload = pi
      cases.push(c)
    }
  }
}

// 1c. node `id` + node `data.label` per layer
for (const [layerType] of Object.entries(LAYERS)) {
  const defs = defaultParamsFor(layerType)
  for (let pi = 0; pi < PAYLOADS.length; pi++) {
    const payload = PAYLOADS[pi]
    // hostile id on a reachable node (id rarely interpolated when reachable)
    {
      const c = runModelCase(`M/${layerType}/nodeId.reachable/P${pi + 1}`, layerType, defs, payload)
      c.payload = pi
      cases.push(c)
    }
    // hostile id on an unreachable orphan → forces id into an issue-comment
    {
      const nodes = [mkLayerNode('in', 'Input', defaultParamsFor('Input'))]
      nodes.push(mkLayerNode(payload, layerType, defs))
      try {
        const r = generate(nodes, [])
        cases.push({ id: `M/${layerType}/nodeId.orphan/P${pi + 1}`, code: r.code, issues: r.issues, sink: 'model', payload: pi })
      } catch {
        cases.push({ id: `M/${layerType}/nodeId.orphan/P${pi + 1}`, code: '', threw: true, sink: 'model', payload: pi })
      }
    }
    // data.label as a stray key on the node
    {
      const c = runModelCase(`M/${layerType}/dataLabel/P${pi + 1}`, layerType, defs, 't', { label: payload })
      c.payload = pi
      cases.push(c)
    }
  }
}

// ── 2. DATA: generateNodeCode per data-type × field/payload; plan with ids+label

for (const [dataType, spec] of Object.entries(DATA_NODES)) {
  const fields = spec.fields.filter((f) => f.type !== 'code')
  for (let pi = 0; pi < PAYLOADS.length; pi++) {
    const payload = PAYLOADS[pi]
    // all non-code fields at once
    const allParams: Record<string, unknown> = {}
    for (const f of fields) allParams[f.name] = payload
    const code = generateNodeCode(dataType, allParams)
    cases.push({ id: `D/${dataType}/all/P${pi + 1}`, code, sink: 'data', payload: pi })

    // each field individually
    for (const f of fields) {
      const params: Record<string, unknown> = { ...defaultDataParams(dataType), [f.name]: payload }
      const c = generateNodeCode(dataType, params)
      cases.push({ id: `D/${dataType}/${f.name}/P${pi + 1}`, code: c, sink: 'data', payload: pi })
    }

    // extra stray keys
    for (const key of ['label', 'name']) {
      const params: Record<string, unknown> = { ...defaultDataParams(dataType), [key]: payload }
      const c = generateNodeCode(dataType, params)
      cases.push({ id: `D/${dataType}/extra.${key}/P${pi + 1}`, code: c, sink: 'data', payload: pi })
    }
  }
}

// generateDataCode(plan) with hostile node ids + CustomScript label + dataset/out_path
for (let pi = 0; pi < PAYLOADS.length; pi++) {
  const payload = PAYLOADS[pi]
  const plan = {
    order: [
      { id: payload + 'a', dataType: 'TableSource', params: { dataset: payload } },
      {
        id: payload + 'b',
        dataType: 'CustomScript',
        params: { label: payload, code: 'df = df  # benign template\n' },
      },
      { id: payload + 'c', dataType: 'WriteDataset', params: { out_path: payload, format: 'csv' } },
    ],
    hasSource: true,
    hasSink: true,
  }
  const code = generateDataCode(plan as unknown as Parameters<typeof generateDataCode>[0])
  cases.push({ id: `D/plan/ids+label+P${pi + 1}`, code, sink: 'data', payload: pi })
}

// ── 3. TRAINING: build a complete valid plan; replace one sink per case

for (let pi = 0; pi < PAYLOADS.length; pi++) {
  const payload = PAYLOADS[pi]
  const withMut = (mut: (p: TrainingPlan) => void): TrainingPlan => {
    const p = baseTrainingPlan()
    mut(p)
    return p
  }
  const trainingSinks: [string, TrainingPlan][] = [
    ['modelRelpath', withMut((p) => { p.modelRelpath = payload })],
    ['datasetRelpath', withMut((p) => { p.datasetRelpath = payload })],
    ['target', withMut((p) => { p.target = payload })],
    ['features', withMut((p) => { p.features = [payload] })],
    ['metric', withMut((p) => { p.training.metrics = [payload] })],
    ['loss.kind', withMut((p) => { (p.training.loss as { kind: string }).kind = payload })],
    ['optimizer.kind', withMut((p) => { (p.training.optimizer as { kind: string }).kind = payload })],
    ['scheduler.kind', withMut((p) => { (p.training.scheduler as { kind: string }).kind = payload })],
    ['head.output', withMut((p) => {
      p.training.heads = [{ output: payload, target: 'y', loss: 'CrossEntropyLoss', weight: 1 }]
    })],
  ]
  for (const [name, plan] of trainingSinks) {
    const code = generateTrainingCode(plan)
    cases.push({ id: `T/${name}/P${pi + 1}`, code, sink: 'training', payload: pi })
  }
}

// baseline benign plan must parse + be inert
{
  const code = generateTrainingCode(baseTrainingPlan())
  cases.push({ id: 'T/baseline', code, sink: 'training', payload: -1 })
}

// ── 5. NUMBERS: every numeric field/param across hostile value set

const NUM_HOSTILE: [string, unknown][] = [
  ['NaN', NaN],
  ['Infinity', Infinity],
  ['-Infinity', -Infinity],
  ['str', `1); ${SENTINEL}(); (`],
  ['null', null],
  ['undef', undefined],
  ['obj', {}],
]

for (const [layerType, spec] of Object.entries(LAYERS)) {
  for (const field of spec.fields) {
    if (!NUMERIC_FIELD_TYPES.has(field.type)) continue
    for (const [label, value] of NUM_HOSTILE) {
      const params = { ...defaultParamsFor(layerType), [field.name]: value }
      const c = runModelCase(
        `N/M/${layerType}/${field.name}/${label}`,
        layerType, params, 't',
      )
      c.payload = -1
      c.sink = 'numbers-model'
      cases.push(c)
    }
  }
}

for (const [dataType, spec] of Object.entries(DATA_NODES)) {
  for (const field of spec.fields) {
    if (!NUMERIC_FIELD_TYPES.has(field.type)) continue
    for (const [label, value] of NUM_HOSTILE) {
      const params: Record<string, unknown> = { ...defaultDataParams(dataType), [field.name]: value }
      const code = generateNodeCode(dataType, params)
      cases.push({ id: `N/D/${dataType}/${field.name}/${label}`, code, sink: 'numbers-data', payload: -1 })
    }
  }
}

const TRAINING_NUM_SINKS: [string, (p: TrainingPlan, v: unknown) => void][] = [
  ['training.epochs', (p, v) => { p.training.epochs = v as number }],
  ['training.batch_size', (p, v) => { p.training.batch_size = v as number }],
  ['training.val_split', (p, v) => { p.training.val_split = v as number }],
  ['training.seed', (p, v) => { p.training.seed = v as number }],
  ['training.optimizer.lr', (p, v) => { p.training.optimizer.lr = v as number }],
  ['training.optimizer.weight_decay', (p, v) => { p.training.optimizer.weight_decay = v as number }],
  ['training.optimizer.momentum', (p, v) => { p.training.optimizer.momentum = v as number }],
  ['training.loss.label_smoothing', (p, v) => { (p.training.loss as { label_smoothing?: number }).label_smoothing = v as number }],
  ['training.scheduler.step_size', (p, v) => { p.training.scheduler = { kind: 'StepLR', step_size: v as number, gamma: 0.1 } }],
  ['training.scheduler.gamma', (p, v) => { p.training.scheduler = { kind: 'StepLR', step_size: 10, gamma: v as number } }],
  ['training.scheduler.patience', (p, v) => { p.training.scheduler = { kind: 'ReduceLROnPlateau', patience: v as number } }],
  ['training.callback.max_norm', (p, v) => { p.training.callbacks = [{ kind: 'GradientClipping', max_norm: v as number }] }],
  ['training.num_workers', (p, v) => { p.training.num_workers = v as number }],
]

for (const [field, mut] of TRAINING_NUM_SINKS) {
  for (const [label, value] of NUM_HOSTILE) {
    const plan = baseTrainingPlan()
    mut(plan, value)
    const code = generateTrainingCode(plan)
    cases.push({ id: `N/T/${field}/${label}`, code, sink: 'numbers-training', payload: -1 })
  }
}

// ── Special: Custom.init_args is structural-only (code by design) ─────────

type SpecialCheck = { id: string; ok: boolean; reason: string }
const specialChecks: SpecialCheck[] = []

function customInitArgsCode(initArgs: string): { code: string; issues: string[] } {
  const params = { ...defaultParamsFor('Custom'), init_args: initArgs }
  const nodes = [mkLayerNode('in', 'Input', defaultParamsFor('Input')), mkLayerNode('t', 'Custom', params)]
  const edges: Edge[] = [{ id: 'e1', source: 'in', target: 't' }]
  try {
    const r = generate(nodes, edges)
    return { code: r.code, issues: r.issues }
  } catch {
    return { code: '', issues: ['generator threw'] }
  }
}

{
  // Benign args with a quoted string must be emitted byte-identically inside Cls(...).
  const benign = "384, 256, dropout=0.0, activation='relu'"
  const { code } = customInitArgsCode(benign)
  specialChecks.push({
    id: 'special/initargs-benign-quoted',
    ok: code.includes(`MyModule(${benign})`),
    reason: `expected \`MyModule(${benign})\` (quotes allowed, byte-identical)`,
  })
}
{
  // Length cap: 500 accepted, 501 rejected (raise instead of construct).
  const at500 = 'a'.repeat(500)
  const accepted = customInitArgsCode(at500)
  specialChecks.push({
    id: 'special/initargs-500-accepted',
    ok: accepted.code.includes(`MyModule(${at500})`) && accepted.issues.length === 0,
    reason: '500-char single-line args should be accepted verbatim',
  })
  const at501 = 'a'.repeat(501)
  const rejected = customInitArgsCode(at501)
  specialChecks.push({
    id: 'special/initargs-501-rejected',
    ok: rejected.code.includes("raise ValueError('invalid init_args')") && rejected.issues.length > 0,
    reason: '501-char args should be rejected (issue + raise, not construct)',
  })
}

// ── Intentional-code-sinks guard ──────────────────────────────────────────

const foundSinks = findCodeSinks()
const expectedSinks = [...CODE_TYPED_SINKS].sort()
const sinksMatch =
  foundSinks.length === expectedSinks.length &&
  foundSinks.every((s, i) => s === expectedSinks[i])

// ── Build python payload (literals filled lazily) ─────────────────────────

let pyStrFn: ((s: unknown) => string) | null = null
try {
  const mod = await import('../src/codegen/pyLiteral')
  pyStrFn = (mod.pyStr as (s: unknown) => string)
} catch (e) {
  // pyLiteral.ts is unusable: say so loudly; every literal below is then
  // registered with an empty literal and fails its round-trip check.
  console.log(`  ✗ cannot import src/codegen/pyLiteral: ${e instanceof Error ? e.message : String(e)}`)
}

for (let pi = 0; pi < PAYLOADS.length; pi++) {
  if (pyStrFn) {
    literals.push({ id: `lit/P${pi + 1}`, literal: pyStrFn(PAYLOADS[pi]), expected: PAYLOADS[pi], payload: pi })
  } else {
    // Step-1 only: pyLiteral.ts is missing. Mark every literal as FAIL with a
    // clear error so the evidence shows exactly what we're testing for.
    literals.push({ id: `lit/P${pi + 1}`, literal: '', expected: PAYLOADS[pi], payload: pi })
  }
}

// ── Run python helper ─────────────────────────────────────────────────────

const pythonInput = {
  cases: cases.map((c) => ({ id: c.id, code: c.code, sentinel: SENTINEL })),
  literals: literals.map((l) => ({ id: l.id, literal: l.literal, expected: l.expected })),
}

let pyOut: { cases: Array<{ id: string; parses: boolean; inert: boolean; error: string | null }>; literals: Array<{ id: string; ok: boolean; error: string | null }> }
try {
  const stdout = execFileSync('python3', ['-I', 'scripts/lib/py_inert_check.py'], {
    input: JSON.stringify(pythonInput),
    encoding: 'utf8',
    timeout: 60000,
  })
  pyOut = JSON.parse(stdout)
} catch (e) {
  const err = e as { stderr?: string; stdout?: string; message?: string }
  console.error('py_inert_check failed to run:', err.stderr ?? err.message)
  console.error(err.stdout ?? '')
  process.exit(2)
}

const byId = new Map<string, { parses: boolean; inert: boolean; error: string | null }>()
for (const r of pyOut.cases) byId.set(r.id, r)
const litById = new Map<string, { ok: boolean; error: string | null }>()
for (const r of pyOut.literals) litById.set(r.id, r)

// ── Determine pass/fail per case ──────────────────────────────────────────

type Verdict = { ok: boolean; reason: string }

function verdict(c: Case): Verdict {
  const r = byId.get(c.id)
  if (!r) return { ok: false, reason: 'no-python-result' }
  if (c.threw) return { ok: false, reason: 'generator threw' }
  // Custom.init_args is code by design (trust-gated): quotes / injection
  // payloads are EXEMPT from the inertness criterion; control-char / line
  // separator payloads must be structurally rejected (issue + raise).
  if (c.id.startsWith('M/Custom/init_args/')) {
    if ([0, 4, 5, 8].includes(c.payload)) return { ok: true, reason: 'init_args code-by-design (P1/P5/P6/P9 exempt)' }
    const raises = c.code.includes("raise ValueError('invalid init_args')")
    if (r.parses && raises && r.inert) return { ok: true, reason: 'rejected: raises + inert' }
    if ((c.issues?.length ?? 0) > 0 && raises) return { ok: true, reason: 'rejected: issue + raises' }
    return { ok: false, reason: r.error ? `python: ${r.error}` : 'init_args not structurally rejected' }
  }
  if (c.sink === 'model' || c.sink === 'numbers-model') {
    if (r.parses && r.inert) return { ok: true, reason: 'parses+inert' }
    if ((c.issues?.length ?? 0) > 0) return { ok: true, reason: 'issues>0 (fail-closed)' }
    return { ok: false, reason: r.error ? `python: ${r.error}` : (!r.parses ? 'does-not-parse' : 'not-inert') }
  }
  if (r.parses && r.inert) return { ok: true, reason: 'parses+inert' }
  return { ok: false, reason: r.error ? `python: ${r.error}` : (!r.parses ? 'does-not-parse' : 'not-inert') }
}

const verdicts = new Map<string, Verdict>()
const failing: { id: string; reason: string; sink: Sink }[] = []
let passCount = 0
for (const c of cases) {
  const v = verdict(c)
  verdicts.set(c.id, v)
  if (v.ok) passCount++
  else failing.push({ id: c.id, reason: v.reason, sink: c.sink })
}

// literal verdicts
const litFailing: string[] = []
for (const l of literals) {
  const r = litById.get(l.id)
  if (!r) { litFailing.push(l.id); continue }
  if (r.ok) passCount++
  else {
    failing.push({ id: l.id, reason: r.error ?? 'not-ok', sink: 'literals' })
    litFailing.push(l.id)
  }
}

// special Custom.init_args structural checks
for (const sc of specialChecks) {
  if (sc.ok) passCount++
  else failing.push({ id: sc.id, reason: sc.reason, sink: 'model' })
}

// ── Report ────────────────────────────────────────────────────────────────

console.log('phase 43: codegen security')
if (!sinksMatch) {
  console.log(`  ✗ INTENTIONAL_CODE_SINKS mismatch`)
  console.log(`    found:    ${JSON.stringify(foundSinks)}`)
  console.log(`    expected: ${JSON.stringify(expectedSinks)}`)
} else {
  console.log(`  ✓ INTENTIONAL_CODE_SINKS = ${expectedSinks.join(', ')}`)
  console.log(`  ✓ Custom.init_args exempt (code by design; trust store kind 'custom-init-args')`)
}

const totalCases = cases.length + literals.length + specialChecks.length
console.log(`\n${passCount}/${totalCases} cases pass, ${failing.length} failing`)

// sinks × payloads failing-count table
const SINKS: Sink[] = ['model', 'data', 'training', 'literals', 'numbers-model', 'numbers-data', 'numbers-training']
function failingCell(sink: Sink, pi: number): number {
  return failing.filter((f) => f.sink === sink && f.id.endsWith(`/P${pi + 1}`)).length
}
function totalCell(sink: Sink, pi: number): number {
  if (sink === 'literals') return literals.filter((l) => l.payload === pi).length
  return cases.filter((c) => c.sink === sink && c.payload === pi).length
}
function sinkTotal(sink: Sink): number {
  if (sink === 'literals') return literals.length
  return cases.filter((c) => c.sink === sink).length
}
function sinkFails(sink: Sink): number {
  return failing.filter((f) => f.sink === sink).length
}
const header = ['sink', ...PAYLOADS.map((_, i) => `P${i + 1}`), 'tot', 'fails']
const widths = header.map((h) => h.length)
const rows: string[][] = []
for (const sink of SINKS) {
  const row: string[] = [sink]
  for (let pi = 0; pi < PAYLOADS.length; pi++) {
    const t = totalCell(sink, pi)
    const f = failingCell(sink, pi)
    row.push(t ? `${t - f}/${t}` : '-')
  }
  row.push(String(sinkTotal(sink)))
  row.push(String(sinkFails(sink)))
  rows.push(row)
}
function pad(s: string, w: number): string { return s.length < w ? s + ' '.repeat(w - s.length) : s }
for (let i = 0; i < header.length; i++) widths[i] = Math.max(widths[i], ...rows.map((r) => r[i].length))
console.log('\n  sinks × payloads  (pass/total per cell):')
for (let i = 0; i < header.length; i++) process.stdout.write('  ' + pad(header[i], widths[i]))
console.log('')
for (const r of rows) {
  for (let i = 0; i < r.length; i++) process.stdout.write('  ' + pad(r[i], widths[i]))
  console.log('')
}

console.log('\n  failing case ids:')
const FAIL_CAP = 60
for (let i = 0; i < failing.length && i < FAIL_CAP; i++) {
  console.log(`    ${failing[i].id}  [${failing[i].sink}]  — ${failing[i].reason}`)
}
if (failing.length > FAIL_CAP) console.log(`    … and ${failing.length - FAIL_CAP} more`)

if (!sinksMatch || failing.length > 0) process.exit(1)
process.exit(0)
