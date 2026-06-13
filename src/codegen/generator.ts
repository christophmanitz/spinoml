import type { Edge } from '@xyflow/react'
import type { LayerNode, GraphSnapshot } from '../canvas/GraphStore'
import { LAYERS, defaultParamsFor, coerceParams, type FieldSpec } from '../layers/registry'

export type CodegenResult = {
  code: string
  issues: string[]
  attrMap: Record<string, string>
  /** Legacy single-input shape — first Input node, for back-compat with the
   *  inference store / Smoke test path that still reads .inputShape. */
  inputShape: number[]
  /** All Input nodes in declaration order, each carrying its forward arg name
   *  and tensor shape. The sidecar uses this to build N zero tensors. */
  inputs: { id: string; name: string; shape: number[] }[]
  /** Topological order of node IDs that carry a pytorchModule, in init/forward order. */
  order: string[]
}

export function generatePyTorchCode(nodes: LayerNode[], edges: Edge[]): string {
  return generate(nodes, edges).code
}

/** Run the codegen against a serialized snapshot (e.g. a workspace file). */
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

  const kindOf = (id: string) => {
    const n = byId.get(id)
    const spec = n && LAYERS[n.data.layerType]
    return spec?.kind ?? 'module'
  }

  // ─── Inputs ─────────────────────────────────────────────────────────────
  const inputNodes = nodes.filter((n) => kindOf(n.id) === 'input')
  if (inputNodes.length === 0) {
    issues.push('No Input node — add one to begin.')
    return { code: emitStub(issues), issues, attrMap: {}, inputShape: [1, 3, 224, 224], inputs: [], order: [] }
  }
  // Deduplicate arg names — collisions silently get suffixed.
  const usedArgs = new Set<string>()
  const inputs = inputNodes.map((n, i) => {
    let name = String(n.data.params.name ?? `x${i + 1}`)
    if (usedArgs.has(name)) {
      let k = 2
      while (usedArgs.has(`${name}_${k}`)) k++
      name = `${name}_${k}`
    }
    usedArgs.add(name)
    return {
      id: n.id,
      name,
      shape: (n.data.params.shape as number[] | undefined) ?? [1, 3, 224, 224],
    }
  })

  // ─── Topological sort (Kahn's) ──────────────────────────────────────────
  const indeg = new Map<string, number>()
  for (const n of nodes) indeg.set(n.id, (pred.get(n.id) ?? []).length)
  const queue: string[] = []
  for (const i of inputs) queue.push(i.id) // start at all inputs
  const order: string[] = []
  let cycleFound = false
  while (queue.length) {
    const id = queue.shift()!
    if (order.includes(id)) continue
    order.push(id)
    for (const nx of succ.get(id) ?? []) {
      indeg.set(nx, (indeg.get(nx) ?? 0) - 1)
      if (indeg.get(nx) === 0) queue.push(nx)
    }
  }
  // Cycle detection: anything still has indeg > 0 → it's in a cycle.
  for (const [id, d] of indeg) {
    if (d > 0) {
      cycleFound = true
      issues.push(`Cycle through node ${id} (${byId.get(id)?.data.layerType ?? '?'}).`)
    }
  }
  if (cycleFound) {
    return { code: emitStub(issues), issues, attrMap: {}, inputShape: inputs[0]?.shape ?? [1, 3, 224, 224], inputs, order: [] }
  }

  // Unreachable nodes (none of the inputs reach them).
  const reachable = new Set(order)
  for (const n of nodes) {
    if (!reachable.has(n.id)) {
      issues.push(`Node ${n.id} (${n.data.layerType}) has no path from any Input — skipped.`)
    }
  }

  // ─── Variable + module attr names ───────────────────────────────────────
  const attrName = new Map<string, string>()
  const varName = new Map<string, string>()
  const counter: Record<string, number> = {}
  function uniq(base: string): string {
    counter[base] = (counter[base] ?? 0) + 1
    return counter[base] === 1 ? base : `${base}_${counter[base]}`
  }

  // Each input node gets its forward-arg name as its variable.
  for (const inp of inputs) varName.set(inp.id, inp.name)

  for (const id of order) {
    if (!reachable.has(id)) continue
    const k = kindOf(id)
    if (k === 'input') continue
    const n = byId.get(id)!
    const spec = LAYERS[n.data.layerType]
    if (!spec) continue
    if (k === 'module' && spec.pytorchModule) {
      const attr = uniq(toSnake(n.data.layerType))
      attrName.set(id, attr)
      varName.set(id, attr) // single-input modules reuse attr as var name
    } else if (k === 'merge') {
      varName.set(id, uniq('m_' + toSnake(n.data.layerType)))
    } else if (k === 'output') {
      // Output nodes reuse their predecessor's variable; no new var needed.
    }
  }

  // ─── init lines ─────────────────────────────────────────────────────────
  const initLines: string[] = []
  for (const id of order) {
    if (!reachable.has(id)) continue
    const n = byId.get(id)!
    const spec = LAYERS[n.data.layerType]
    if (!spec || !spec.pytorchModule) continue
    if (kindOf(id) !== 'module') continue
    const args = spec.fields
      .map((field) => `${field.name}=${serializeParam(field, n.data.params[field.name] ?? field.default)}`)
      .join(', ')
    initLines.push(`        self.${attrName.get(id)} = ${spec.pytorchModule}(${args})`)
  }

  // ─── forward lines ──────────────────────────────────────────────────────
  const forwardLines: string[] = []
  const outputCollect: { name: string; varName: string }[] = []
  for (const id of order) {
    if (!reachable.has(id)) continue
    const n = byId.get(id)!
    const spec = LAYERS[n.data.layerType]
    const k = kindOf(id)
    if (k === 'input') continue
    if (!spec) continue
    const preds = (pred.get(id) ?? []).map((pid) => varName.get(pid)).filter((v): v is string => !!v)
    if (k === 'module' && spec.pytorchModule) {
      if (preds.length === 0) {
        issues.push(`Node ${id} (${n.data.layerType}) has no upstream value.`)
        continue
      }
      if (preds.length > 1) {
        issues.push(`Node ${id} (${n.data.layerType}): module layer received ${preds.length} inputs — using first. Use a Merge layer (Concat/Add) to combine streams.`)
      }
      forwardLines.push(`        ${varName.get(id)} = self.${attrName.get(id)}(${preds[0]})`)
    } else if (k === 'merge') {
      if (preds.length < 2) {
        issues.push(`Node ${id} (${n.data.layerType}): merge layer needs ≥2 inputs (has ${preds.length}).`)
      }
      const expr = spec.forwardExpr ? spec.forwardExpr(preds, n.data.params) : preds[0] ?? ''
      forwardLines.push(`        ${varName.get(id)} = ${expr}`)
    } else if (k === 'output') {
      if (preds.length === 0) {
        issues.push(`Output node ${id} has no upstream value.`)
        continue
      }
      const outName = String(n.data.params.name ?? 'out')
      outputCollect.push({ name: outName, varName: preds[0] })
    }
  }

  // ─── return statement ───────────────────────────────────────────────────
  let returnLine: string
  if (outputCollect.length === 0) {
    // No explicit Output node — return the variable of whichever non-input node
    // has no successors. If none, return the last input.
    const sinks = order.filter((id) => kindOf(id) !== 'input' && (succ.get(id) ?? []).length === 0)
    if (sinks.length === 1) {
      returnLine = `        return ${varName.get(sinks[0]) ?? inputs[0].name}`
    } else if (sinks.length > 1) {
      returnLine = `        return ${sinks.map((id) => varName.get(id) ?? inputs[0].name).join(', ')}`
    } else {
      returnLine = `        return ${inputs[0].name}`
    }
  } else if (outputCollect.length === 1) {
    returnLine = `        return ${outputCollect[0].varName}`
  } else {
    const named = outputCollect.map((o) => `"${o.name}": ${o.varName}`).join(', ')
    returnLine = `        return { ${named} }`
  }

  const code = emitModule(initLines, forwardLines, returnLine, inputs, issues)
  const attrMap: Record<string, string> = {}
  for (const [id, attr] of attrName) attrMap[id] = attr
  const moduleOrder = order.filter((id) => attrMap[id] !== undefined)
  return {
    code, issues, attrMap,
    inputShape: inputs[0].shape,
    inputs,
    order: moduleOrder,
  }
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
  returnLine: string,
  inputs: { name: string; shape: number[] }[],
  issues: string[],
): string {
  const header = issuesBlock(issues)
  const init = initLines.length ? initLines.join('\n') : '        pass'
  const forward = forwardLines.length ? forwardLines.join('\n') : '        pass'
  const argSig = inputs.map((i) => i.name).join(', ')
  const sampleVars = inputs.map((i) => `${i.name} = torch.zeros(${pyTuple(i.shape)})`).join('\n    ')
  const callArgs = inputs.map((i) => i.name).join(', ')

  return `${header}import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
${init}

    def forward(self, ${argSig}):
${forward}
${returnLine}


if __name__ == "__main__":
    model = Model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    ${sampleVars}
    out = model(${callArgs})
    if isinstance(out, tuple):
        print("Output shapes:", [tuple(o.shape) for o in out])
    elif isinstance(out, dict):
        print("Output shapes:", {k: tuple(v.shape) for k, v in out.items()})
    else:
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
