import type { Edge } from '@xyflow/react'
import type { LayerNode, GraphSnapshot } from '../canvas/GraphStore'
import { LAYERS, defaultParamsFor, coerceParams, type FieldSpec } from '../layers/registry'

export type CodegenResult = {
  code: string
  issues: string[]
  attrMap: Record<string, string>
  inputShape: number[]
  /** Topological order of node IDs that carry a pytorchModule, in forward-pass order. */
  order: string[]
}

export function generatePyTorchCode(nodes: LayerNode[], edges: Edge[]): string {
  return generate(nodes, edges).code
}

/** Run the codegen against a serialized snapshot (e.g. a workspace file).
 *  Used to render the .py twin without having to load the file into the live store. */
export function generateFromSnapshot(snapshot: GraphSnapshot): CodegenResult {
  const nodes: LayerNode[] = snapshot.nodes.map((n) => ({
    id: n.id,
    type: 'layer',
    position: n.position ?? { x: 0, y: 0 },
    data: {
      layerType: n.layerType,
      params: coerceParams(n.layerType, { ...defaultParamsFor(n.layerType), ...n.params }),
    },
  }))
  const edges: Edge[] = snapshot.edges.map((e, i) => ({
    id: `e${i + 1}`, source: e.source, target: e.target,
  }))
  return generate(nodes, edges)
}

export function generate(nodes: LayerNode[], edges: Edge[]): CodegenResult {
  const issues: string[] = []
  const byId = new Map(nodes.map((n) => [n.id, n]))

  const succ = new Map<string, string[]>()
  const pred = new Map<string, string[]>()
  for (const n of nodes) {
    succ.set(n.id, [])
    pred.set(n.id, [])
  }
  for (const e of edges) {
    succ.get(e.source)?.push(e.target)
    pred.get(e.target)?.push(e.source)
  }

  const inputNode = nodes.find((n) => n.data.layerType === 'Input')
  if (!inputNode) {
    issues.push('No Input node — add one to begin.')
    return { code: emitStub(issues), issues, attrMap: {}, inputShape: [1, 3, 224, 224], order: [] }
  }

  const order: string[] = []
  const visiting = new Set<string>()
  const visited = new Set<string>()
  let cycleFound = false
  function visit(id: string): void {
    if (visited.has(id)) return
    if (visiting.has(id)) {
      cycleFound = true
      return
    }
    visiting.add(id)
    for (const nx of succ.get(id) ?? []) visit(nx)
    visiting.delete(id)
    visited.add(id)
    order.unshift(id)
  }
  visit(inputNode.id)

  if (cycleFound) issues.push('Cycle detected in graph — generation aborted.')

  const unreachable = nodes.filter((n) => !visited.has(n.id) && n.data.layerType !== 'Output')
  for (const n of unreachable) {
    issues.push(`Node ${n.id} (${n.data.layerType}) has no path from Input — skipped.`)
  }

  for (const id of order) {
    const ps = pred.get(id) ?? []
    if (ps.length > 1) {
      issues.push(`Node ${id}: multiple inputs not yet supported (Phase 5).`)
    }
    const ss = succ.get(id) ?? []
    if (ss.length > 1) {
      issues.push(`Node ${id}: fork to multiple successors — only the first path is wired.`)
    }
  }

  if (cycleFound) return { code: emitStub(issues), issues, attrMap: {}, inputShape: [1, 3, 224, 224], order: [] }

  const attrName = new Map<string, string>()
  const counter: Record<string, number> = {}
  for (const id of order) {
    const n = byId.get(id)!
    const spec = LAYERS[n.data.layerType]
    if (!spec || !spec.pytorchModule) continue
    const base = toSnake(n.data.layerType)
    counter[base] = (counter[base] ?? 0) + 1
    attrName.set(id, `${base}_${counter[base]}`)
  }

  const initLines: string[] = []
  for (const id of order) {
    const n = byId.get(id)!
    const spec = LAYERS[n.data.layerType]
    if (!spec || !spec.pytorchModule) continue
    const args = spec.fields
      .map((field) => `${field.name}=${serializeParam(field, n.data.params[field.name] ?? field.default)}`)
      .join(', ')
    initLines.push(`        self.${attrName.get(id)} = ${spec.pytorchModule}(${args})`)
  }

  const forwardLines: string[] = []
  for (const id of order) {
    const n = byId.get(id)!
    const spec = LAYERS[n.data.layerType]
    if (!spec || !spec.pytorchModule) continue
    forwardLines.push(`        x = self.${attrName.get(id)}(x)`)
  }

  const inputShape = (inputNode.data.params.shape as number[]) ?? [1, 3, 224, 224]
  const code = emitModule(initLines, forwardLines, inputShape, issues)
  const attrMap: Record<string, string> = {}
  for (const [id, attr] of attrName) attrMap[id] = attr
  const moduleOrder = order.filter((id) => attrMap[id] !== undefined)
  return { code, issues, attrMap, inputShape, order: moduleOrder }
}

function emitStub(issues: string[]): string {
  return `${issuesBlock(issues)}import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
`
}

function emitModule(
  initLines: string[],
  forwardLines: string[],
  inputShape: number[],
  issues: string[],
): string {
  const header = issuesBlock(issues)
  const init = initLines.length ? initLines.join('\n') : '        pass'
  const forward = forwardLines.length ? forwardLines.join('\n') : '        pass'

  return `${header}import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
${init}

    def forward(self, x):
${forward}
        return x


if __name__ == "__main__":
    model = Model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    sample = torch.zeros(${pyTuple(inputShape)})
    out = model(sample)
    print(f"Output shape: {tuple(out.shape)}")
`
}

function issuesBlock(issues: string[]): string {
  if (issues.length === 0) return ''
  return `# Graph issues:\n${issues.map((i) => `#   - ${i}`).join('\n')}\n\n`
}

function serializeParam(field: FieldSpec, value: unknown): string {
  switch (field.type) {
    case 'int':
      return String(Math.trunc(value as number))
    case 'float':
      return formatFloat(value as number)
    case 'bool':
      return (value as boolean) ? 'True' : 'False'
    case 'select':
      return `'${value as string}'`
    case 'tuple-int': {
      const arr = (Array.isArray(value) ? value : field.default) as number[]
      return pyTuple(arr.slice(0, field.arity))
    }
    case 'shape': {
      const arr = (Array.isArray(value) ? value : field.default) as number[]
      return pyTuple(arr)
    }
  }
}

function formatFloat(v: number): string {
  if (v === 0) return '0.0'
  const abs = Math.abs(v)
  if (abs < 1e-3 || abs >= 1e6) return v.toExponential()
  return Number.isInteger(v) ? `${v}.0` : String(v)
}

function pyTuple(arr: number[]): string {
  if (arr.length === 0) return '()'
  if (arr.length === 1) return `(${arr[0]},)`
  return `(${arr.join(', ')})`
}

function toSnake(name: string): string {
  return name
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1_$2')
    .toLowerCase()
}
