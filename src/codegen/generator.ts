import type { Edge } from '@xyflow/react'
import type { LayerNode, GraphSnapshot } from '../canvas/GraphStore'
import { LAYERS, defaultParamsFor, coerceParams, classNameFromSource, type FieldSpec } from '../layers/registry'

export type CodegenResult = {
  code: string
  issues: string[]
  attrMap: Record<string, string>
  /** Legacy single-input shape — first Input node, for back-compat with the
   *  inference store / Smoke test path that still reads .inputShape. */
  inputShape: number[]
  /** All Input nodes in declaration order, each carrying its forward arg name,
   *  tensor shape and dtype. The sidecar uses this to build N zero tensors
   *  (dtype 'int64' → LongTensor, e.g. token ids for Embedding). */
  inputs: { id: string; name: string; shape: number[]; dtype: string }[]
  /** Topological order of node IDs that carry a pytorchModule, in init/forward order. */
  order: string[]
}

/** Internal, kind-agnostic node shape used by the builder so it can run over
 *  both the live LayerNode[] and a serialized subgraph snapshot. */
type GNode = { id: string; layerType: string; params: Record<string, unknown> }

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

/** Normalize a stored subgraph ({nodes,edges} of GraphSnapshot shape, or
 *  undefined) into GNodes with coerced params. */
function subgraphOf(params: Record<string, unknown>): { nodes: GNode[]; edges: { source: string; target: string }[] } {
  const sg = params.subgraph as GraphSnapshot | undefined
  if (!sg || !Array.isArray(sg.nodes)) return { nodes: [], edges: [] }
  return {
    nodes: sg.nodes.map((n) => ({
      id: n.id,
      layerType: n.layerType,
      params: coerceParams(n.layerType, { ...defaultParamsFor(n.layerType), ...n.params }),
    })),
    edges: sg.edges ?? [],
  }
}

function groupClassName(params: Record<string, unknown>): string {
  const raw = String(params.class_name ?? 'SubModule').trim() || 'SubModule'
  // Sanitise to a valid Python identifier.
  const id = raw.replace(/[^A-Za-z0-9_]/g, '_')
  return /^[A-Za-z_]/.test(id) ? id : `M_${id}`
}

type BuildCtx = {
  defs: Map<string, string>   // class name → class source (custom + group), deduped
  tg: Set<string>             // torch_geometric symbols needed anywhere
  hasCustom: { v: boolean }   // any Custom node anywhere → import F
}

export function generate(nodes: LayerNode[], edges: Edge[]): CodegenResult {
  const gnodes: GNode[] = nodes.map((n) => ({ id: n.id, layerType: n.data.layerType, params: n.data.params }))
  const ctx: BuildCtx = { defs: new Map(), tg: new Set(), hasCustom: { v: false } }
  const built = buildClass(gnodes, edges, 'Model', ctx, 0)

  const extraImports: string[] = []
  if (ctx.hasCustom.v) extraImports.push('import torch.nn.functional as F')
  if (ctx.tg.size) extraImports.push(`from torch_geometric.nn import ${[...ctx.tg].sort().join(', ')}`)

  const classBlocks = [...ctx.defs.values()].filter((t) => t && t.trim())
  const code = emitModule(built.classText, built.inputs, built.issues, extraImports, classBlocks)

  return {
    code,
    issues: built.issues,
    attrMap: built.attrMap,
    inputShape: built.inputs[0]?.shape ?? [1, 3, 224, 224],
    inputs: built.inputs,
    order: built.order,
  }
}

type BuiltClass = {
  classText: string
  inputs: { id: string; name: string; shape: number[]; dtype: string }[]
  order: string[]
  attrMap: Record<string, string>
  issues: string[]
}

/**
 * Build ONE nn.Module class from a graph. Recurses into `group` nodes (each
 * becomes its own nested class, registered in ctx.defs) and registers `custom`
 * node sources. Returns just the class text — the module wrapper (imports,
 * nested class defs, __main__) is assembled by generate().
 */
function buildClass(
  nodes: GNode[], edges: { source: string; target: string }[],
  className: string, ctx: BuildCtx, depth: number,
): BuiltClass {
  const issues: string[] = []
  const byId = new Map(nodes.map((n) => [n.id, n]))

  const succ = new Map<string, string[]>()
  const pred = new Map<string, string[]>()
  for (const n of nodes) { succ.set(n.id, []); pred.set(n.id, []) }
  for (const e of edges) {
    succ.get(e.source)?.push(e.target)
    pred.get(e.target)?.push(e.source)
  }

  const kindOf = (id: string) => LAYERS[byId.get(id)?.layerType ?? '']?.kind ?? 'module'

  // ─── Inputs ─────────────────────────────────────────────────────────────
  const inputNodes = nodes.filter((n) => kindOf(n.id) === 'input')
  if (inputNodes.length === 0) {
    issues.push(className === 'Model'
      ? 'No Input node — add one to begin.'
      : `${className}: subgraph has no Input node.`)
    return {
      classText: emitClass(className, [], [], '        return None', ''),
      inputs: [], order: [], attrMap: {}, issues,
    }
  }
  const usedArgs = new Set<string>()
  const inputs = inputNodes.map((n, i) => {
    let name = String(n.params.name ?? `x${i + 1}`)
    if (usedArgs.has(name)) {
      let k = 2
      while (usedArgs.has(`${name}_${k}`)) k++
      name = `${name}_${k}`
    }
    usedArgs.add(name)
    return {
      id: n.id,
      name,
      shape: (n.params.shape as number[] | undefined) ?? [1, 3, 224, 224],
      dtype: String(n.params.dtype ?? 'float32'),
    }
  })

  // ─── Topological sort (Kahn's) ──────────────────────────────────────────
  const indeg = new Map<string, number>()
  for (const n of nodes) indeg.set(n.id, (pred.get(n.id) ?? []).length)
  const queue: string[] = []
  for (const i of inputs) queue.push(i.id)
  const order: string[] = []
  while (queue.length) {
    const id = queue.shift()!
    if (order.includes(id)) continue
    order.push(id)
    for (const nx of succ.get(id) ?? []) {
      indeg.set(nx, (indeg.get(nx) ?? 0) - 1)
      if (indeg.get(nx) === 0) queue.push(nx)
    }
  }
  let cycleFound = false
  for (const [id, d] of indeg) {
    if (d > 0) {
      cycleFound = true
      issues.push(`Cycle through node ${id} (${byId.get(id)?.layerType ?? '?'}).`)
    }
  }
  if (cycleFound) {
    return {
      classText: emitClass(className, [], [], '        return None', inputs.map((i) => i.name).join(', ')),
      inputs, order: [], attrMap: {}, issues,
    }
  }

  const reachable = new Set(order)
  for (const n of nodes) {
    if (!reachable.has(n.id)) {
      issues.push(`Node ${n.id} (${n.layerType}) has no path from any Input — skipped.`)
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
  for (const inp of inputs) varName.set(inp.id, inp.name)

  for (const id of order) {
    if (!reachable.has(id)) continue
    const k = kindOf(id)
    if (k === 'input') continue
    const n = byId.get(id)!
    const spec = LAYERS[n.layerType]
    if (!spec) continue
    if (k === 'module' && spec.pytorchModule) {
      const attr = uniq(toSnake(n.layerType))
      attrName.set(id, attr)
      varName.set(id, attr)
    } else if (k === 'custom') {
      const cls = classNameFromSource(String(n.params.source ?? '')) ?? 'Custom'
      attrName.set(id, uniq(toSnake(cls)))
      varName.set(id, attrName.get(id)!)
    } else if (k === 'group') {
      attrName.set(id, uniq(toSnake(groupClassName(n.params))))
      varName.set(id, attrName.get(id)!)
    } else if (k === 'merge') {
      varName.set(id, uniq('m_' + toSnake(n.layerType)))
    } else if (k === 'function') {
      varName.set(id, uniq('fx_' + toSnake(n.layerType)))
    }
  }

  // ─── init lines (+ register custom/group class defs) ──────────────────────
  const initLines: string[] = []
  for (const id of order) {
    if (!reachable.has(id)) continue
    const n = byId.get(id)!
    const spec = LAYERS[n.layerType]
    if (!spec) continue
    const k = kindOf(id)
    if (k === 'module' && spec.pytorchModule) {
      const rhs = spec.initExpr
        ? spec.initExpr(n.params)
        : `${spec.pytorchModule}(${spec.fields
            .map((field) => `${field.name}=${serializeParam(field, n.params[field.name] ?? field.default)}`)
            .join(', ')})`
      initLines.push(`        self.${attrName.get(id)} = ${rhs}`)
    } else if (k === 'custom') {
      const src = String(n.params.source ?? '').trim()
      const cls = classNameFromSource(src)
      if (!src) {
        issues.push(`Custom node ${id} has no source code.`)
      } else if (!cls) {
        issues.push(`Custom node ${id}: source must define a class (e.g. "class MyModule(nn.Module):").`)
      } else {
        const args = String(n.params.init_args ?? '').trim()
        initLines.push(`        self.${attrName.get(id)} = ${cls}(${args})`)
        if (!ctx.defs.has(cls)) ctx.defs.set(cls, src)
        ctx.hasCustom.v = true
      }
    } else if (k === 'group') {
      const gcls = groupClassName(n.params)
      initLines.push(`        self.${attrName.get(id)} = ${gcls}()`)
      if (!ctx.defs.has(gcls)) {
        ctx.defs.set(gcls, '') // placeholder guards against re-entry
        const sub = subgraphOf(n.params)
        const child = buildClass(sub.nodes, sub.edges, gcls, ctx, depth + 1)
        ctx.defs.set(gcls, child.classText)
        // Surface subgraph problems on the parent, prefixed.
        for (const m of child.issues) issues.push(`${gcls}: ${m}`)
      }
    }
  }

  // ─── graph aux inputs (edge_index / batch) ──────────────────────────────
  const aux = {
    edgeIndex: inputs.find((i) => i.name === 'edge_index')?.name,
    batch: inputs.find((i) => i.name === 'batch')?.name,
  }

  // ─── forward lines ──────────────────────────────────────────────────────
  const forwardLines: string[] = []
  const outputCollect: { name: string; varName: string }[] = []
  for (const id of order) {
    if (!reachable.has(id)) continue
    const n = byId.get(id)!
    const spec = LAYERS[n.layerType]
    const k = kindOf(id)
    if (k === 'input') continue
    if (!spec) continue
    const preds = (pred.get(id) ?? []).map((pid) => varName.get(pid)).filter((v): v is string => !!v)
    if (k === 'module' && spec.pytorchModule) {
      if (preds.length === 0) { issues.push(`Node ${id} (${n.layerType}) has no upstream value.`); continue }
      if (preds.length > 1) {
        issues.push(`Node ${id} (${n.layerType}): module layer received ${preds.length} inputs — using first. Use a Merge layer (Concat/Add) to combine streams.`)
      }
      if (spec.needsEdgeIndex) {
        if (!aux.edgeIndex) issues.push(`Node ${id} (${n.layerType}) needs an Input named 'edge_index' (dtype int64, shape [2, E]).`)
        forwardLines.push(`        ${varName.get(id)} = self.${attrName.get(id)}(${preds[0]}, ${aux.edgeIndex ?? 'edge_index'})`)
      } else if (spec.tupleOutput) {
        forwardLines.push(`        ${varName.get(id)}, _ = self.${attrName.get(id)}(${preds[0]})`)
      } else {
        forwardLines.push(`        ${varName.get(id)} = self.${attrName.get(id)}(${preds[0]})`)
      }
    } else if (k === 'custom' || k === 'group') {
      // Skip nodes whose init was skipped (custom with no/invalid source) — the
      // attr won't exist, and the issue was already reported above.
      if (k === 'custom' && !classNameFromSource(String(n.params.source ?? ''))) continue
      const label = k === 'group' ? groupClassName(n.params) : (classNameFromSource(String(n.params.source ?? '')) ?? '?')
      if (preds.length === 0) {
        issues.push(`${k === 'group' ? 'Group' : 'Custom'} node ${id} (${label}) has no upstream value.`)
        continue
      }
      if (k === 'group') {
        const sub = subgraphOf(n.params)
        const nIn = sub.nodes.filter((s) => LAYERS[s.layerType]?.kind === 'input').length
        if (nIn > 0 && nIn !== preds.length) {
          issues.push(`Group node ${id} (${label}): subgraph takes ${nIn} input(s) but ${preds.length} are wired.`)
        }
      }
      forwardLines.push(`        ${varName.get(id)} = self.${attrName.get(id)}(${preds.join(', ')})`)
    } else if (k === 'merge') {
      if (preds.length < 2) issues.push(`Node ${id} (${n.layerType}): merge layer needs ≥2 inputs (has ${preds.length}).`)
      const expr = spec.forwardExpr ? spec.forwardExpr(preds, n.params, aux) : preds[0] ?? ''
      forwardLines.push(`        ${varName.get(id)} = ${expr}`)
    } else if (k === 'function') {
      if (preds.length === 0) { issues.push(`Node ${id} (${n.layerType}) has no upstream value.`); continue }
      if (preds.length > 1) issues.push(`Node ${id} (${n.layerType}): function layer uses first of ${preds.length} inputs.`)
      if (spec.pyImports?.some((s) => s.startsWith('global_')) && !aux.batch) {
        issues.push(`Node ${id} (${n.layerType}) needs an Input named 'batch' (dtype int64, shape [N_nodes]).`)
      }
      const expr = spec.forwardExpr ? spec.forwardExpr([preds[0]], n.params, aux) : preds[0]
      forwardLines.push(`        ${varName.get(id)} = ${expr}`)
    } else if (k === 'output') {
      if (preds.length === 0) { issues.push(`Output node ${id} has no upstream value.`); continue }
      outputCollect.push({ name: String(n.params.name ?? 'out'), varName: preds[0] })
    }
  }

  // ─── return statement ───────────────────────────────────────────────────
  let returnLine: string
  if (outputCollect.length === 0) {
    const sinks = order.filter((id) => kindOf(id) !== 'input' && (succ.get(id) ?? []).length === 0)
    if (sinks.length === 1) returnLine = `        return ${varName.get(sinks[0]) ?? inputs[0].name}`
    else if (sinks.length > 1) returnLine = `        return ${sinks.map((id) => varName.get(id) ?? inputs[0].name).join(', ')}`
    else returnLine = `        return ${inputs[0].name}`
  } else if (outputCollect.length === 1) {
    returnLine = `        return ${outputCollect[0].varName}`
  } else {
    returnLine = `        return { ${outputCollect.map((o) => `"${o.name}": ${o.varName}`).join(', ')} }`
  }

  // Collect torch_geometric symbols from this level's used nodes.
  for (const id of order) {
    if (!reachable.has(id)) continue
    LAYERS[byId.get(id)!.layerType]?.pyImports?.forEach((s) => ctx.tg.add(s))
  }

  const argSig = inputs.map((i) => i.name).join(', ')
  const classText = emitClass(className, initLines, forwardLines, returnLine, argSig)
  const attrMap: Record<string, string> = {}
  for (const [id, attr] of attrName) attrMap[id] = attr
  const moduleOrder = order.filter((id) => attrMap[id] !== undefined)
  return { classText, inputs, order: moduleOrder, attrMap, issues }
}

/** The `nn.Module(...)` constructor expression for a module-kind layer — used to
 *  "eject" a built-in node into an editable Custom-code node. null for
 *  non-module layers (functional / merge / io / group). */
export function layerInitExpr(layerType: string, params: Record<string, unknown>): string | null {
  const spec = LAYERS[layerType]
  if (!spec || !spec.pytorchModule) return null
  return spec.initExpr
    ? spec.initExpr(params)
    : `${spec.pytorchModule}(${spec.fields
        .map((field) => `${field.name}=${serializeParam(field, params[field.name] ?? field.default)}`)
        .join(', ')})`
}

function emitClass(
  className: string, initLines: string[], forwardLines: string[],
  returnLine: string, argSig: string,
): string {
  const init = initLines.length ? initLines.join('\n') : '        pass'
  const forward = forwardLines.length ? forwardLines.join('\n') : '        pass'
  return `class ${className}(nn.Module):
    def __init__(self):
        super().__init__()
${init}

    def forward(self${argSig ? ', ' + argSig : ''}):
${forward}
${returnLine}`
}

function emitModule(
  modelClassText: string,
  inputs: { name: string; shape: number[]; dtype: string }[],
  issues: string[],
  extraImports: string[] = [],
  classDefs: string[] = [],
): string {
  const header = issuesBlock(issues)
  const sampleVars = inputs.map((i) =>
    i.dtype === 'int64'
      ? `${i.name} = torch.zeros(${pyTuple(i.shape)}, dtype=torch.long)`
      : `${i.name} = torch.zeros(${pyTuple(i.shape)})`,
  ).join('\n    ')
  const callArgs = inputs.map((i) => i.name).join(', ')
  const imports = extraImports.length ? '\n' + extraImports.join('\n') : ''
  // Nested classes (custom + group) emitted verbatim, before Model.
  const defs = classDefs.length ? '\n\n' + classDefs.join('\n\n\n') + '\n' : ''

  return `${header}import torch
import torch.nn as nn${imports}
${defs}

${modelClassText}


if __name__ == "__main__":
    model = Model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    ${sampleVars || 'pass'}
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
    case 'int-list': {
      const arr = (Array.isArray(value) ? value : field.default) as number[]
      return `[${arr.join(', ')}]`
    }
    case 'shape': {
      const arr = (Array.isArray(value) ? value : field.default) as number[]
      return pyTuple(arr)
    }
    case 'dataset-ref':
      return `'${value as string}'`
    case 'columns-multi':
      return `[${(value as string[]).map((s) => `'${s}'`).join(', ')}]`
    case 'column-single':
      return `'${value as string}'`
    case 'text':
    case 'code':
      // Only used by Custom nodes, which emit init/source directly (never via
      // serializeParam). Present for switch exhaustiveness.
      return JSON.stringify(String(value))
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
