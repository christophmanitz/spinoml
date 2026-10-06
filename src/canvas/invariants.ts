// Graph-invariant validation for the architecture canvas (Phase 3 — TODO §4).
//
// The graph is the core model representation and must be treated as a critical
// data structure. This module (a) defines the invariant list, (b) validates any
// (nodes, edges) state against it, and (c) answers cycle queries so mutations
// can be rejected BEFORE the invalid state becomes authoritative (§4.3).
//
// Policy decisions (documented, enforced by validateGraphState):
//   - Node IDs must be unique. Edge IDs must be unique.
//   - Every edge references existing nodes; handles, when present, are non-empty
//     strings (LayerNode renders the default 'source'/'target' handles).
//   - Layer types must be known to the LAYERS registry (software guard for the
//     LLM path — the registry is the only authority on what a layer is).
//   - Parameters must satisfy their FieldSpec: right JS type, within min/max
//     bounds, select options, tuple arity (validated on the COERCED params that
//     actually live in the store).
//   - Cycles are NOT supported: the generator emits sequentially and the layout
//     is a longest-path DAG rank. A cycle is an ERROR and any mutation that
//     would create one is rejected (GRAPH_DAG_POLICY).
//   - Self-loops are errors.
//   - A single-input layer (everything except merge/custom/group) fed by more
//     than one edge is a WARN: the UI rewires such targets, so it should not
//     persist, but older saved files must still load (down-graded from error).

import type { Edge } from '@xyflow/react'
import type { LayerNode } from './GraphStore'
import { LAYERS } from '../layers/registry'

/** Cycles are not supported (generator emits sequentially; layout ranks by DAG
 *  longest path). Connect-mutations that would close a loop are rejected. */
export const GRAPH_DAG_POLICY = true

export type GraphIssueCode =
  | 'dup-node' | 'dup-edge' | 'unknown-node' | 'self-loop' | 'cycle'
  | 'unknown-layer' | 'param-missing' | 'param-type' | 'param-range'
  | 'tuple-arity' | 'select-out-of-range' | 'invalid-handle' | 'single-input'

export type GraphIssueSeverity = 'error' | 'warn'

export type GraphIssue = {
  code: GraphIssueCode
  severity: GraphIssueSeverity
  message: string
  nodeId?: string
  edgeId?: string
}

export type GraphValidation = { ok: boolean; issues: GraphIssue[] }

const err = (code: GraphIssueCode, message: string, c?: { nodeId?: string; edgeId?: string }): GraphIssue =>
  ({ code, severity: 'error', message, ...c } as GraphIssue)
const warn = (code: GraphIssueCode, message: string, c?: { nodeId?: string; edgeId?: string }): GraphIssue =>
  ({ code, severity: 'warn', message, ...c } as GraphIssue)

/** Is `target` reachable from `source` following existing edges? Used (with the
 *  self-loop case) to reject edges that would close a directed cycle. */
export function wouldCreateCycle(nodes: LayerNode[], edges: Edge[], source: string, target: string): boolean {
  if (source === target) return true
  const byId = new Set(nodes.map((n) => n.id))
  if (!byId.has(source) || !byId.has(target)) return true // unknown endpoint: treat as invalid
  const adj = new Map<string, string[]>()
  for (const e of edges) {
    if (!adj.has(e.source)) adj.set(e.source, [])
    adj.get(e.source)!.push(e.target)
  }
  const seen = new Set<string>([target])
  let frontier = [target]
  while (frontier.length > 0) {
    const next: string[] = []
    for (const id of frontier) {
      for (const t of adj.get(id) ?? []) {
        if (t === source) return true
        if (!seen.has(t)) { seen.add(t); next.push(t) }
      }
    }
    frontier = next
  }
  return false
}

function validateParams(layerType: string, params: Record<string, unknown>): GraphIssue[] {
  const spec = LAYERS[layerType]
  if (!spec) return []
  const out: GraphIssue[] = []
  for (const f of spec.fields) {
    const v = params[f.name]
    if (v === undefined || v === null) {
      out.push(warn('param-missing', `${layerType}: required parameter '${f.name}' is missing`))
      continue
    }
    switch (f.type) {
      case 'int': {
        if (!Number.isInteger(v)) out.push(err('param-type', `${layerType}.${f.name}: expected int, got ${JSON.stringify(v)}`))
        else if (f.min !== undefined && (v as number) < f.min) out.push(err('param-range', `${layerType}.${f.name}: ${v} < min ${f.min}`))
        else if (f.max !== undefined && (v as number) > f.max) out.push(err('param-range', `${layerType}.${f.name}: ${v} > max ${f.max}`))
        break
      }
      case 'float': {
        if (typeof v !== 'number' || !Number.isFinite(v)) out.push(err('param-type', `${layerType}.${f.name}: expected finite number, got ${JSON.stringify(v)}`))
        else if (f.min !== undefined && v < f.min) out.push(err('param-range', `${layerType}.${f.name}: ${v} < min ${f.min}`))
        else if (f.max !== undefined && v > f.max) out.push(err('param-range', `${layerType}.${f.name}: ${v} > max ${f.max}`))
        break
      }
      case 'bool':
        if (typeof v !== 'boolean') out.push(err('param-type', `${layerType}.${f.name}: expected boolean, got ${JSON.stringify(v)}`))
        break
      case 'select':
        if (typeof v !== 'string' || !f.options.includes(v))
          out.push(err('select-out-of-range', `${layerType}.${f.name}: '${String(v)}' not in ${JSON.stringify(f.options)}`))
        break
      case 'tuple-int': {
        if (!Array.isArray(v) || v.length !== f.arity || !v.every((x) => Number.isInteger(x)))
          out.push(err('tuple-arity', `${layerType}.${f.name}: expected ${f.arity} ints, got ${JSON.stringify(v)}`))
        break
      }
      case 'int-list':
      case 'shape': {
        if (!Array.isArray(v) || v.length === 0 || !v.every((x) => Number.isInteger(x)))
          out.push(err('param-type', `${layerType}.${f.name}: expected non-empty int array, got ${JSON.stringify(v)}`))
        break
      }
      case 'columns-multi':
        if (!Array.isArray(v) || !v.every((x) => typeof x === 'string'))
          out.push(err('param-type', `${layerType}.${f.name}: expected string array, got ${JSON.stringify(v)}`))
        break
      case 'dataset-ref':
      case 'column-single':
      case 'text':
      case 'code':
        if (typeof v !== 'string') out.push(err('param-type', `${layerType}.${f.name}: expected string, got ${JSON.stringify(v)}`))
        break
    }
  }
  return out
}

/** Full structural validation of a (nodes, edges) graph state. `ok` is false
 *  iff at least one ERROR-severity issue exists (WARNs load and persist). */
export function validateGraphState(nodes: LayerNode[], edges: Edge[]): GraphValidation {
  const issues: GraphIssue[] = []

  // Node ID uniqueness + known layer types.
  const seenNodes = new Set<string>()
  for (const n of nodes) {
    if (seenNodes.has(n.id)) {
      issues.push(err('dup-node', `node id '${n.id}' duplicated`, { nodeId: n.id }))
    } else {
      seenNodes.add(n.id)
    }
    const layerType = n.data?.layerType
    const spec = layerType ? LAYERS[layerType] : undefined
    if (!layerType || !spec) issues.push(err('unknown-layer', `layer '${String(layerType)}' (node '${n.id}') is not in the registry`, { nodeId: n.id }))
    else issues.push(...validateParams(layerType, n.data?.params ?? {}))
  }

  // Edge ID uniqueness, endpoint existence, self-loops, handles, single-input fan-in.
  const seenEdges = new Set<string>()
  const inDegree = new Map<string, number>()
  for (const e of edges) {
    if (seenEdges.has(e.id)) {
      issues.push(err('dup-edge', `edge id '${e.id}' duplicated (${e.source}→${e.target})`, { edgeId: e.id }))
    } else {
      seenEdges.add(e.id)
    }
    if (!seenNodes.has(e.source)) issues.push(err('unknown-node', `edge '${e.id}' source '${e.source}' does not exist`, { edgeId: e.id }))
    if (!seenNodes.has(e.target)) issues.push(err('unknown-node', `edge '${e.id}' target '${e.target}' does not exist`, { edgeId: e.id }))
    if (e.source === e.target) issues.push(err('self-loop', `edge '${e.id}' is a self-loop on '${e.source}'`, { edgeId: e.id }))
    for (const [side, handle] of [['source', e.sourceHandle], ['target', e.targetHandle]] as const) {
      if (handle !== undefined && handle !== null && (typeof handle !== 'string' || handle.length === 0))
        issues.push(err('invalid-handle', `edge '${e.id}' ${side}Handle is not a non-empty string`, { edgeId: e.id }))
    }
    inDegree.set(e.target, (inDegree.get(e.target) ?? 0) + 1)
  }
  const kinds = new Set(['merge', 'custom', 'group'])
  for (const n of nodes) {
    const deg = inDegree.get(n.id) ?? 0
    const layerType = n.data?.layerType
    if (deg > 1 && !(layerType && kinds.has(LAYERS[layerType]?.kind ?? ''))) {
      issues.push(warn('single-input', `node '${n.id}' (${layerType ?? '?'}) has ${deg} incoming edges`, { nodeId: n.id }))
    }
  }

  // DAG policy: a directed cycle anywhere is an error.
  if (GRAPH_DAG_POLICY && nodes.length > 0) {
    const adj = new Map<string, string[]>()
    for (const e of edges) {
      if (!adj.has(e.source)) adj.set(e.source, [])
      adj.get(e.source)!.push(e.target)
    }
    const WHITE = 0, GREY = 1, BLACK = 2
    const color = new Map<string, number>(nodes.map((n) => [n.id, WHITE]))
    // ITERATIVE depth-first search (explicit stack): a long valid chain — or a
    // hostile imported file — must never overflow the call stack (found by the
    // phase 66 fuzz test: a 10 000-node chain crashed the recursive version).
    let cyclic = false
    for (const start of nodes) {
      if (cyclic) break
      if (color.get(start.id) !== WHITE) continue
      const stack: { id: string; next: number }[] = [{ id: start.id, next: 0 }]
      color.set(start.id, GREY)
      while (stack.length > 0) {
        const frame = stack[stack.length - 1]
        const targets = adj.get(frame.id) ?? []
        if (frame.next >= targets.length) {
          color.set(frame.id, BLACK)
          stack.pop()
          continue
        }
        const t = targets[frame.next++]
        const c = color.get(t) ?? WHITE
        if (c === GREY) { cyclic = true; break } // back edge
        if (c === WHITE) {
          color.set(t, GREY)
          stack.push({ id: t, next: 0 })
        }
      }
    }
    if (cyclic) issues.push(err('cycle', 'graph contains a directed cycle (DAG required for codegen)', {}))
  }

  return { ok: !issues.some((i) => i.severity === 'error'), issues }
}