// Model explanation — the deterministic half of the "explain this model" feature.
// PURE (like generator.ts): same nodes+edges → same explanation, no I/O. Walks
// the graph in the generator's own execution order and pairs each layer with its
// registry summary + the shape inference already computed on the node. The LLM
// "what does it do" one-liner is layered on top in modelIntent.ts — this module
// works fully offline and is the source of truth for the data-flow skeleton.

import type { Edge } from '@xyflow/react'
import type { LayerNode } from '../canvas/GraphStore'
import { LAYERS, type LayerSpec } from '../layers/registry'
import { generate } from './generator'

export type FlowStep = {
  id: string
  layerType: string
  /** PyTorch module path, e.g. "nn.Conv2d" (falls back to the layer type). */
  module: string
  /** The registry's compact per-layer summary, e.g. "3→64 k3x3". */
  summary: string
  /** Inferred output shape of this node, if shape inference has run. */
  outShape?: number[]
}

export type ModelExplanation = {
  inputs: { name: string; shape: number[]; isGraph?: boolean }[]
  steps: FlowStep[]
  output?: number[]
  code: string
  issues: string[]
  /** Compact one-line-per-step flow, reused for display and the LLM prompt. */
  flowText: string
}

function safeSummary(spec: LayerSpec, params: Record<string, unknown>): string {
  try {
    return spec.summary(params) ?? ''
  } catch {
    return ''
  }
}

function fmtShape(s: number[] | undefined): string {
  return s && s.length ? `[${s.join(', ')}]` : ''
}

function buildFlowText(
  inputs: ModelExplanation['inputs'],
  steps: FlowStep[],
  output: number[] | undefined,
): string {
  const lines: string[] = []
  for (const i of inputs) {
    lines.push(`Input ${i.name}${i.isGraph ? ' (graph)' : ''} ${fmtShape(i.shape)}`.trim())
  }
  for (const s of steps) {
    const bits = [s.module]
    if (s.summary) bits.push(s.summary)
    const head = bits.join('  ')
    const tail = s.outShape ? `  → ${fmtShape(s.outShape)}` : ''
    lines.push(`→ ${head}${tail}`)
  }
  if (output) lines.push(`Output ${fmtShape(output)}`)
  return lines.join('\n')
}

export function explainModel(nodes: LayerNode[], edges: Edge[]): ModelExplanation {
  const g = generate(nodes, edges)
  const byId = new Map(nodes.map((n) => [n.id, n]))

  const steps: FlowStep[] = []
  for (const id of g.order) {
    const n = byId.get(id)
    if (!n) continue
    const spec = LAYERS[n.data.layerType]
    if (!spec || spec.kind === 'input' || spec.kind === 'output') continue
    steps.push({
      id,
      layerType: n.data.layerType,
      module: spec.pytorchModule || n.data.layerType,
      summary: safeSummary(spec, n.data.params),
      outShape: n.data.inferredOutputShape,
    })
  }

  // Model output = last layer's inferred output (the Output node is identity).
  const output = steps.length ? steps[steps.length - 1].outShape : undefined
  const inputs = g.inputs.map((i) => ({ name: i.name, shape: i.shape, isGraph: i.isGraph }))
  const flowText = buildFlowText(inputs, steps, output)

  return { inputs, steps, output, code: g.code, issues: g.issues, flowText }
}
