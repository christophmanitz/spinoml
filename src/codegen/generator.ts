import type { Edge } from '@xyflow/react'
import type { LayerNode, GraphSnapshot } from '../canvas/GraphStore'
import { LAYERS, defaultParamsFor, coerceParams, classNameFromSource, type FieldSpec } from '../layers/registry'
import { pyStr, pyComment, pyIdent, pyInt, pyFloat, pyIntList } from './pyLiteral'

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
  inputs: InputDesc[]
  /** Topological order of node IDs that carry a pytorchModule, in init/forward order. */
  order: string[]
}

/** One forward argument. For a graph input (`isGraph`), the arg is a PyG Data
 *  object built from `shape` ([N nodes, F features]), `nEdges` and `edgeDim`
 *  rather than a bare tensor; the sidecar/standalone harness assembles it. */
export type InputDesc = {
  id: string; name: string; shape: number[]; dtype: string
  isGraph?: boolean; nEdges?: number; edgeDim?: number
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

/** The subgraph's forward-arg names, in the SAME order buildClass derives them
 *  (input-kind nodes in array order, with the same duplicate-name suffixing).
 *  Used to map outer predecessors to inner inputs BY NAME instead of position.
 *  Independent of internal wiring — an input with no internal edge (e.g. a GNN
 *  encoder's `edge_index`/`batch`, picked up by aux) is still a forward arg. */
function subgraphInputNames(params: Record<string, unknown>): string[] {
  const sub = subgraphOf(params)
  const inputNodes = sub.nodes.filter((n) => LAYERS[n.layerType]?.kind === 'input')
  const used = new Set<string>()
  return inputNodes.map((n, i) => {
    let name = pyIdent(String(n.params.name ?? `x${i + 1}`), `x${i + 1}`)
    if (used.has(name)) {
      let k = 2
      while (used.has(`${name}_${k}`)) k++
      name = `${name}_${k}`
    }
    used.add(name)
    return name
  })
}

/** Map outer predecessor vars to a subgraph's inner inputs by NAME, returning
 *  args in inner-forward order. Matches an inner name `n` against a pred var
 *  that equals `n` or ends with `_n` (so outer `lig_x` feeds inner `x`). Returns
 *  null when the names don't cover every input uniquely — caller falls back to
 *  positional wiring (preserves existing graphs that rely on edge order). */
function nameAlignedArgs(innerNames: string[], preds: string[]): string[] | null {
  if (innerNames.length < 2 || innerNames.length !== preds.length) return null
  const used = new Set<number>()
  const out: string[] = []
  for (const inName of innerNames) {
    let idx = preds.findIndex((p, i) => !used.has(i) && p === inName)
    if (idx < 0) idx = preds.findIndex((p, i) => !used.has(i) && p.endsWith(`_${inName}`))
    if (idx < 0) return null
    used.add(idx)
    out.push(preds[idx])
  }
  return out
}

function groupClassName(params: Record<string, unknown>): string {
  return pyIdent(String(params.class_name ?? 'SubModule').trim(), 'SubModule')
}

/** `init_args` is arbitrary Python by design (gated by the trust store, kind
 *  'custom-init-args'), so a value whitelist is the wrong tool. We only enforce
 *  STRUCTURAL safety: single-line (no control chars / U+2028 / U+2029) and at
 *  most 500 chars, so it can never break out of this statement or smuggle a
 *  line terminator. Quotes are allowed (legitimate `activation='relu'`). */
const INIT_ARGS_MAX = 500
function isValidInitArgs(s: string): boolean {
  if (s.length > INIT_ARGS_MAX) return false
  for (const ch of s) {
    const cp = ch.codePointAt(0)!
    if (cp <= 0x1f || (cp >= 0x7f && cp <= 0x9f) || cp === 0x2028 || cp === 0x2029) return false
  }
  return true
}

function finiteIntArray(v: unknown, fallback: number[]): number[] {
  if (Array.isArray(v) && v.every((x) => typeof x === 'number' && Number.isFinite(x))) {
    return (v as number[]).map((x) => Math.trunc(x))
  }
  return fallback.map((x) => Math.trunc(x))
}

function finiteInt(v: unknown, fallback: number): number {
  const n = typeof v === 'number' ? v : Number(v)
  return Number.isFinite(n) ? Math.trunc(n) : Math.trunc(fallback)
}

type BuildCtx = {
  defs: Map<string, string>   // class name → class source (custom + group), deduped
  tg: Set<string>             // torch_geometric symbols needed anywhere
  hasCustom: { v: boolean }   // any Custom node anywhere → import F
}

export function generate(nodes: LayerNode[], edges: Edge[]): CodegenResult {
  // Determinism: canonicalize input order so the same graph always produces
  // identical code regardless of React Flow array order (ensures stable names
  // and attrMap for golden tests + replay). Node sort is a cheap tiebreak on
  // (layerType, id); edge sort is (source, target). Both are stable ties only.
  const canonNodes = [...nodes].sort((a, b) =>
    a.data.layerType.localeCompare(b.data.layerType) || a.id.localeCompare(b.id))
  const canonEdges = [...edges].sort((a, b) =>
    a.source.localeCompare(b.source) || a.target.localeCompare(b.target))
  const gnodes: GNode[] = canonNodes.map((n) => ({ id: n.id, layerType: n.data.layerType, params: n.data.params }))
  const ctx: BuildCtx = { defs: new Map(), tg: new Set(), hasCustom: { v: false } }
  const built = buildClass(gnodes, canonEdges, 'Model', ctx, 0)

  const extraImports: string[] = []
  if (ctx.hasCustom.v) extraImports.push('import torch.nn.functional as F')
  if (ctx.tg.size) extraImports.push(`from torch_geometric.nn import ${[...ctx.tg].sort().join(', ')}`)
  // Only the standalone __main__ harness constructs Data objects (top-level
  // graph args); attribute access inside classes needs no import.
  if (built.inputs.some((i) => i.isGraph)) extraImports.push('from torch_geometric.data import Data')

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
  inputs: InputDesc[]
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

  const kindOf = (id: string) => LAYERS[byId.get(id)?.layerType ?? '']?.kind ?? 'module'

  const succ = new Map<string, string[]>()
  const pred = new Map<string, string[]>()
  for (const n of nodes) { succ.set(n.id, []); pred.set(n.id, []) }
  for (const e of edges) {
    // A Manifest node is a data-layer DECLARATION (the pairing), not a forward
    // source; input nodes (Input/Graph/Sequence) are forward-arg ROOTS. Skip edges
    // FROM a manifest and edges INTO an input so neither perturbs the forward
    // graph — the input still becomes a plain forward argument.
    if (kindOf(e.source) === 'manifest' || kindOf(e.target) === 'input') continue
    succ.get(e.source)?.push(e.target)
    pred.get(e.target)?.push(e.source)
  }

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
  const inputs: InputDesc[] = inputNodes.map((n, i) => {
    const rawName = String(n.params.name ?? `x${i + 1}`)
    let name = pyIdent(rawName, `x${i + 1}`)
    if (name !== rawName) issues.push(`Node ${n.id}: name contained invalid characters and was sanitized.`)
    if (usedArgs.has(name)) {
      let k = 2
      while (usedArgs.has(`${name}_${k}`)) k++
      name = `${name}_${k}`
    }
    usedArgs.add(name)
    const isGraph = LAYERS[n.layerType]?.graphInput === true
    return {
      id: n.id,
      name,
      shape: finiteIntArray(n.params.shape, isGraph ? [32, 9] : [1, 3, 224, 224]),
      dtype: isGraph ? 'graph' : String(n.params.dtype ?? 'float32'),
      ...(isGraph ? { isGraph: true, nEdges: finiteInt(n.params.n_edges, 64), edgeDim: finiteInt(n.params.edge_dim, 0) } : {}),
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
    // DataOp nodes are data-stage, not model layers — a standalone one with no
    // wiring is expected, so don't warn that it's unreachable.
    if (!reachable.has(n.id) && kindOf(n.id) !== 'dataop' && kindOf(n.id) !== 'manifest') {
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
    } else if (k === 'dataop') {
      // Data-stage node: emits nothing into forward(). Alias its variable to its
      // single predecessor so `Input → DataOp → Layer` stays a valid chain.
      const p0 = (pred.get(id) ?? [])[0]
      const pv = p0 ? varName.get(p0) : undefined
      if (pv) varName.set(id, pv)
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
      // `initExpr` lives in the registry and builds its RHS from raw params —
      // coerce first so hostile select/number values can't reach the template.
      const rhs = spec.initExpr
        ? spec.initExpr(coerceParams(n.layerType, n.params))
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
        if (isValidInitArgs(args)) {
          initLines.push(`        self.${attrName.get(id)} = ${cls}(${args})`)
        } else {
          issues.push(`Custom node ${id}: init_args must be a single line without control characters (max 500 chars) and was ignored.`)
          // Hard block: emit `raise` instead of the constructor call so the
          // model can never be silently built with different arguments.
          initLines.push(`        raise ValueError('invalid init_args')`)
        }
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

  // ─── graph inputs (a whole PyG Data per node) + aux (edge_index / batch) ──
  // A `Graph` input's forward arg is a Data object. Where it feeds a built-in
  // GNN/pool (a tensor op), unpack it explicitly — `x, edge_index, batch =
  // g.x, g.edge_index, g.batch` — and drive aux from it. Where it feeds a
  // Subgraph or Custom node, the Data is passed WHOLE (graphDataVar), so that
  // code has full access (edge_attr, pos, anything) — like writing it by hand.
  const graphInputs = inputs.filter((i) => i.isGraph)
  const graphDataVar = new Map<string, string>()
  const graphUnpackLines: string[] = []
  const aux: { edgeIndex?: string; batch?: string } = {
    edgeIndex: inputs.find((i) => i.name === 'edge_index')?.name,
    batch: inputs.find((i) => i.name === 'batch')?.name,
  }
  let unpackedGraphs = 0
  for (const g of graphInputs) {
    graphDataVar.set(g.id, g.name)
    const feedsTensorOp = (succ.get(g.id) ?? []).some((sid) => {
      const sk = kindOf(sid)
      return sk === 'module' || sk === 'merge' || sk === 'function'
    })
    if (!feedsTensorOp) continue // only group/custom/output consumers → keep Data whole
    const base = graphInputs.length === 1 ? '' : `${toSnake(g.name)}_`
    const xv = `${base}x`, eiv = `${base}edge_index`, bv = `${base}batch`
    graphUnpackLines.push(`        ${xv}, ${eiv}, ${bv} = ${g.name}.x, ${g.name}.edge_index, ${g.name}.batch`)
    varName.set(g.id, xv) // GNN feature entry = node features
    if (unpackedGraphs === 0) { aux.edgeIndex = eiv; aux.batch = bv }
    unpackedGraphs++
  }
  if (unpackedGraphs > 1) {
    issues.push(`Multiple Graph inputs feed GNN layers in one scope — edge_index/batch resolve to the first. Put each GNN branch in its own Subgraph so each owns its graph.`)
  }

  // ─── forward lines ──────────────────────────────────────────────────────
  const forwardLines: string[] = [...graphUnpackLines]
  const outputCollect: { name: string; varName: string }[] = []
  for (const id of order) {
    if (!reachable.has(id)) continue
    const n = byId.get(id)!
    const spec = LAYERS[n.layerType]
    const k = kindOf(id)
    if (k === 'input') continue
    // DataOp is a data-stage passthrough — its var was aliased to its predecessor
    // above; it contributes no forward line.
    if (k === 'dataop') continue
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
      // Graph-input predecessors pass their WHOLE Data object here (full access
      // to edge_attr/pos/… in the consuming code), not the unpacked x tensor.
      const argVars = (pred.get(id) ?? [])
        .map((pid) => graphDataVar.get(pid) ?? varName.get(pid))
        .filter((v): v is string => !!v)
      if (argVars.length === 0) {
        issues.push(`${k === 'group' ? 'Group' : 'Custom'} node ${id} (${label}) has no upstream value.`)
        continue
      }
      let callArgs = argVars.join(', ')
      if (k === 'group') {
        const innerNames = subgraphInputNames(n.params)
        if (innerNames.length > 0 && innerNames.length !== argVars.length) {
          issues.push(`Group node ${id} (${label}): subgraph takes ${innerNames.length} input(s) but ${argVars.length} are wired.`)
        }
        // Prefer name-based wiring: outer `lig_x`/`lig_edge_index`/… → inner
        // `x`/`edge_index`/… regardless of edge order. Falls back to positional
        // when names don't line up (keeps older positional graphs working).
        const aligned = nameAlignedArgs(innerNames, argVars)
        if (aligned) callArgs = aligned.join(', ')
      }
      forwardLines.push(`        ${varName.get(id)} = self.${attrName.get(id)}(${callArgs})`)
    } else if (k === 'merge') {
      if (preds.length < 2) issues.push(`Node ${id} (${n.layerType}): merge layer needs ≥2 inputs (has ${preds.length}).`)
      const expr = spec.forwardExpr ? spec.forwardExpr(preds, coerceParams(n.layerType, n.params), aux) : preds[0] ?? ''
      forwardLines.push(`        ${varName.get(id)} = ${expr}`)
    } else if (k === 'function') {
      if (preds.length === 0) { issues.push(`Node ${id} (${n.layerType}) has no upstream value.`); continue }
      if (n.layerType === 'BuildGraph') {
        // Builds edge_index from the incoming node features and passes the
        // features through unchanged. Downstream GNN layers (later in topo
        // order, since they sit after this node) pick the edge_index up via
        // aux — no separate edge_index Input needed.
        const edgeVar = `${varName.get(id)}_ei`
        const gexpr = spec.forwardExpr ? spec.forwardExpr([preds[0]], coerceParams(n.layerType, n.params), aux) : preds[0]
        forwardLines.push(`        ${edgeVar} = ${gexpr}`)
        forwardLines.push(`        ${varName.get(id)} = ${preds[0]}`)
        aux.edgeIndex = edgeVar
      } else {
        if (preds.length > 1) issues.push(`Node ${id} (${n.layerType}): function layer uses first of ${preds.length} inputs.`)
        if (spec.pyImports?.some((s) => s.startsWith('global_')) && !aux.batch) {
          issues.push(`Node ${id} (${n.layerType}) needs an Input named 'batch' (dtype int64, shape [N_nodes]).`)
        }
        const expr = spec.forwardExpr ? spec.forwardExpr([preds[0]], coerceParams(n.layerType, n.params), aux) : preds[0]
        forwardLines.push(`        ${varName.get(id)} = ${expr}`)
      }
    } else if (k === 'output') {
      if (preds.length === 0) { issues.push(`Output node ${id} has no upstream value.`); continue }
      outputCollect.push({ name: String(n.params.name ?? 'out'), varName: preds[0] })
    }
  }

  // ─── return statement ───────────────────────────────────────────────────
  let returnLine: string
  if (outputCollect.length === 0) {
    const sinks = order.filter((id) => kindOf(id) !== 'input' && kindOf(id) !== 'dataop' && (succ.get(id) ?? []).length === 0)
    if (sinks.length === 1) returnLine = `        return ${varName.get(sinks[0]) ?? inputs[0].name}`
    else if (sinks.length > 1) returnLine = `        return ${sinks.map((id) => varName.get(id) ?? inputs[0].name).join(', ')}`
    else returnLine = `        return ${inputs[0].name}`
  } else if (outputCollect.length === 1) {
    returnLine = `        return ${outputCollect[0].varName}`
  } else {
    returnLine = `        return { ${outputCollect.map((o) => `${pyStr(o.name)}: ${o.varName}`).join(', ')} }`
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
  inputs: InputDesc[],
  issues: string[],
  extraImports: string[] = [],
  classDefs: string[] = [],
): string {
  const header = issuesBlock(issues)
  const sampleVars = inputs.map((i) => {
    if (i.isGraph) {
      // Build a real PyG Data: x [N, F], a small self-edge edge_index [2, E],
      // batch [N] (one graph), and edge_attr [E, De] when edge_dim>0.
      const [n, fdim] = [i.shape[0] ?? 1, i.shape[1] ?? 1]
      const e = i.nEdges ?? 0
      const parts = [
        `x=torch.zeros((${n}, ${fdim}))`,
        `edge_index=torch.zeros((2, ${e}), dtype=torch.long)`,
        `batch=torch.zeros((${n},), dtype=torch.long)`,
      ]
      if ((i.edgeDim ?? 0) > 0) parts.push(`edge_attr=torch.zeros((${e}, ${i.edgeDim}))`)
      return `${i.name} = Data(${parts.join(', ')})`
    }
    return i.dtype === 'int64'
      ? `${i.name} = torch.zeros(${pyTuple(i.shape)}, dtype=torch.long)`
      : `${i.name} = torch.zeros(${pyTuple(i.shape)})`
  }).join('\n    ')
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
    ${sampleVars || 'pass'}
    out = model(${callArgs})
    # Count params AFTER a forward so lazy (in_channels=-1) layers are initialized.
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
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
  return `# Graph issues:\n${issues.map((i) => `#   - ${pyComment(i)}`).join('\n')}\n\n`
}

function serializeParam(field: FieldSpec, value: unknown): string {
  switch (field.type) {
    case 'int':
      return pyInt(value, field.default)
    case 'float':
      return pyFloat(value, field.default)
    case 'bool':
      return (value as boolean) ? 'True' : 'False'
    case 'select':
      return pyStr(value)
    case 'tuple-int':
      return pyTuple(finiteIntArray(value, field.default).slice(0, field.arity))
    case 'int-list':
      return pyIntList(value, field.default)
    case 'shape':
      return pyTuple(finiteIntArray(value, field.default))
    case 'dataset-ref':
      return pyStr(value)
    case 'columns-multi':
      return `[${(Array.isArray(value) ? value : []).map((s) => pyStr(s)).join(', ')}]`
    case 'column-single':
      return pyStr(value)
    case 'text':
    case 'code':
      // Only used by Custom nodes, which emit init/source directly (never via
      // serializeParam). Present for switch exhaustiveness.
      return pyStr(value)
  }
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
