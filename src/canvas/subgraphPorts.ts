import type { Edge } from '@xyflow/react'
import type { LayerNode, GraphSnapshot } from './GraphStore'
import { LAYERS, defaultParamsFor, coerceParams } from '../layers/registry'

// ── Auto-port sync for Subgraph nodes ─────────────────────────────────────
// An input inside a subgraph that is fed FROM THE OUTSIDE is a *proxy*: it is
// configured by the connected outer node, not independently. So when an edge
// connects an outer node → a Subgraph node, we ensure the subgraph has one
// matching input node (a proxy) that inherits the source's type + config; when
// the edge is removed, an auto-created proxy is removed again (and a pre-existing
// hand-authored input it had claimed is released back to editable).
//
// Hybrid rule (preserves self-contained / template subgraphs):
//   • A free hand-authored input (no _proxyOf) gets CLAIMED by the first
//     unmatched source — keeping its id (and its internal wiring) intact.
//   • If no free input is left, a new proxy is CREATED (_autoCreated).
//   • On disconnect: claimed-but-now-orphaned proxies are released (hand) or
//     removed (auto). Removal also drops the proxy's internal edges.
// A proxy is marked read-only in the Inspector via params._proxyOf.

type SubNode = GraphSnapshot['nodes'][number]

// Config copied from the source onto its proxy, per source kind.
const COPY_FIELDS_GRAPH = ['name', 'shape', 'n_edges', 'edge_dim', 'dataset', 'branch']
const COPY_FIELDS_INPUT = ['name', 'shape', 'dtype', 'dataset', 'features', 'target']

function proxyFrom(source: LayerNode): { layerType: string; params: Record<string, unknown> } {
  const layerType = source.data.layerType === 'Graph' ? 'Graph' : 'Input'
  const fields = layerType === 'Graph' ? COPY_FIELDS_GRAPH : COPY_FIELDS_INPUT
  const params: Record<string, unknown> = {}
  for (const f of fields) if (source.data.params[f] !== undefined) params[f] = source.data.params[f]
  return { layerType, params }
}

function isInputKind(layerType: string): boolean {
  return LAYERS[layerType]?.kind === 'input'
}

/** Reconcile ONE subgraph's input proxies against the list of outer sources
 *  (distinct, in edge order) feeding the Subgraph node. */
function reconcileOne(sub: GraphSnapshot, sources: { sid: string; node: LayerNode }[]): GraphSnapshot {
  const wantSids = new Set(sources.map((s) => s.sid))
  let nodes: SubNode[] = sub.nodes.map((n) => ({ ...n, params: { ...n.params } }))
  let edges = sub.edges

  // 1. Release/remove proxies whose source edge is gone.
  const removeIds = new Set<string>()
  nodes = nodes.map((n) => {
    const proxyOf = n.params._proxyOf as string | undefined
    if (!proxyOf || wantSids.has(proxyOf)) return n
    if (n.params._autoCreated) { removeIds.add(n.id); return n } // auto → drop node
    // hand-created proxy → release back to an editable input (drop the markers)
    const rest = Object.fromEntries(
      Object.entries(n.params).filter(([k]) => k !== '_proxyOf' && k !== '_autoCreated'),
    )
    return { ...n, params: rest }
  })
  if (removeIds.size) {
    nodes = nodes.filter((n) => !removeIds.has(n.id))
    edges = edges.filter((e) => !removeIds.has(e.source) && !removeIds.has(e.target))
  }

  // 2. Claim / sync / create one proxy per source.
  let nextFree = 0
  const freeInputs = () => nodes.filter((n) => isInputKind(n.layerType) && n.params._proxyOf === undefined)
  for (const { sid, node: src } of sources) {
    const { layerType, params: copied } = proxyFrom(src)
    const existing = nodes.find((n) => n.params._proxyOf === sid)
    if (existing) {
      const auto = existing.params._autoCreated ? { _autoCreated: true } : {}
      nodes = nodes.map((n) => n.id === existing.id
        ? { ...n, layerType, params: coerceParams(layerType, { ...defaultParamsFor(layerType), ...copied, _proxyOf: sid, ...auto }) }
        : n)
      continue
    }
    const free = freeInputs()[nextFree]
    if (free) {
      nextFree++
      nodes = nodes.map((n) => n.id === free.id
        ? { ...n, layerType, params: coerceParams(layerType, { ...defaultParamsFor(layerType), ...copied, _proxyOf: sid }) }
        : n)
    } else {
      const id = `pin_${sid}`
      nodes = [
        { id, layerType, params: coerceParams(layerType, { ...defaultParamsFor(layerType), ...copied, _proxyOf: sid, _autoCreated: true }), position: { x: 40, y: 20 } },
        ...nodes,
      ]
    }
  }
  return { nodes, edges }
}

/** Recompute every Subgraph node's input proxies from the current edges. Returns
 *  the same array reference when nothing changed (safe to call after any edge or
 *  source-param mutation without churning React). Level-local: operates on the
 *  currently-focused graph's Subgraph nodes. */
export function reconcileSubgraphPorts(nodes: LayerNode[], edges: Edge[]): LayerNode[] {
  let changed = false
  const out = nodes.map((n) => {
    if (n.data.layerType !== 'Subgraph') return n
    const sub = n.data.params.subgraph as GraphSnapshot | undefined
    if (!sub || !Array.isArray(sub.nodes)) return n
    const seen = new Set<string>()
    const sources = edges
      .filter((e) => e.target === n.id)
      .map((e) => ({ sid: e.source, node: nodes.find((m) => m.id === e.source) }))
      .filter((s): s is { sid: string; node: LayerNode } => !!s.node && !seen.has(s.sid) && !!seen.add(s.sid))
    const next = reconcileOne(sub, sources)
    if (JSON.stringify(next) === JSON.stringify(sub)) return n
    changed = true
    return { ...n, data: { ...n.data, params: { ...n.data.params, subgraph: next } } }
  })
  return changed ? out : nodes
}
