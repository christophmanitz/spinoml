import { create } from 'zustand'
import {
  type Node,
  type Edge,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
  type XYPosition,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
} from '@xyflow/react'
import { defaultParamsFor, coerceParams, LAYERS } from '../layers/registry'
import { useLayoutStore, type FlowDir } from './layoutStore'
import { layeredLayout } from './layout'
import { reconcileSubgraphPorts } from './subgraphPorts'
import { validateGraphState, wouldCreateCycle, type GraphIssue } from './invariants'

export type LayerNodeData = {
  layerType: string
  params: Record<string, unknown>
  inferredInputShape?: number[]
  inferredOutputShape?: number[]
  hasError?: boolean
} & Record<string, unknown>

export type LayerNode = Node<LayerNodeData, 'layer'>

export type GraphSnapshot = {
  nodes: { id: string; layerType: string; params: Record<string, unknown>; position?: XYPosition }[]
  edges: { source: string; target: string }[]
}

let nextId = 1
const newNodeId = () => `n${nextId++}`
function bumpNextIdPast(ids: string[]) {
  for (const id of ids) {
    // Defensive: junk file entries may carry non-string ids — skip, never throw.
    // loadSnapshot validates ids afterwards and rejects cleanly.
    if (typeof id !== 'string') continue
    const m = id.match(/^n(\d+)$/)
    if (m) nextId = Math.max(nextId, parseInt(m[1], 10) + 1)
  }
}

type State = {
  nodes: LayerNode[]
  edges: Edge[]
  selectedNodeId: string | null
  /** Phase 40 — monotonic structural revision. Bumped on every committed
   *  structural change (addLayer/updateParams/replace/delete/connect/load/
   *  edges change) and on resetGraph. Pure position drags (onNodesChange
   *  containing only {type:'position'}) do NOT bump. Async callers record it
   *  as `const rev = get().revision` before the await and drop the response
   *  if `rev !== get().revision` — the response is stale. */
  revision: number
  /** Bumped by every DOCUMENT swap (`loadSnapshot` success, `resetGraph`) — not by edits. The undo
   *  history keys off it: a loaded/new document must start a fresh history, otherwise the first
   *  undo after opening file B would restore file A's graph into B's canvas (and autosave would
   *  write it into B's file). */
  loadEpoch: number

  onNodesChange: OnNodesChange<LayerNode>
  onEdgesChange: OnEdgesChange
  onConnect: OnConnect

  addLayer: (layerType: string, position: XYPosition, opts?: { id?: string; params?: Record<string, unknown> }) => string
  updateNodeParams: (id: string, params: Record<string, unknown>) => void
  /** Swap the layerType of an existing node in place. Edges, id, and position
   *  stay the same; params are reset to defaults for the new layer type and
   *  merged with the optional extraParams. */
  replaceNodeLayer: (id: string, newLayerType: string, extraParams?: Record<string, unknown>) => void
  setSelectedNodeId: (id: string | null) => void
  deleteNode: (id: string) => void
  /** Programmatic edge add. Returns true iff the edge was committed. Guards
   *  (Phase 3 §4.3 — validate BEFORE commit): both endpoints exist, no
   *  self-loop, and no directed cycle (DAG policy); existing identical edge is
   *  deduped; a single-input target is rewired (replacing its old incoming
   *  edge), matching canvas drag behaviour. */
  connectNodes: (source: string, target: string) => boolean
  /** Validate the CURRENT graph against the invariant list (§4.1). Pure read —
   *  no state write, no subscription, safe to call from handlers. */
  graphIssues: () => GraphIssue[]
  autoLayout: () => void
  /** Replace the whole graph. Returns false (and does NOT commit) when the
   *  snapshot violates error-level invariants — a corrupt/numbered file must
   *  never become the authoritative graph (§4.3, §5). */
  loadSnapshot: (snapshot: GraphSnapshot) => boolean
  resetGraph: () => void
}

export const useGraphStore = create<State>((set, get) => ({
  nodes: [
    {
      id: 'input',
      type: 'layer',
      position: { x: 250, y: 50 },
      data: { layerType: 'Input', params: defaultParamsFor('Input') },
    },
  ],
  edges: [],
  selectedNodeId: null,
  revision: 0,
  loadEpoch: 0,

  onNodesChange: (changes) => {
    // Phase 40 — pure position/dimensions drags are not structural; don't
    // bump the revision for them (avoids spurious staleness). Any other
    // change (add/remove/select) is structural.
    const structural = changes.some((c) => c.type !== 'position' && c.type !== 'dimensions' && c.type !== 'select')
    if (structural) {
      set({ nodes: applyNodeChanges(changes, get().nodes), revision: get().revision + 1 })
    } else {
      set({ nodes: applyNodeChanges(changes, get().nodes) })
    }
  },
  onEdgesChange: (changes) => {
    const edges = applyEdgeChanges(changes, get().edges)
    // Edge removed/added near a Subgraph node → re-sync its input proxies.
    set({ edges, nodes: reconcileSubgraphPorts(get().nodes, edges), revision: get().revision + 1 })
  },
  onConnect: (connection) => {
    if (!connection.source || !connection.target) return
    const { nodes, edges } = get()
    const g = validateGuard(nodes, edges, connection.source, connection.target)
    if (!g.ok) {
      console.debug('onConnect rejected:', g.reason)
      return
    }
    const base = baseForConnect(nodes, edges, connection.target)
    const nextEdges = addEdge({ ...connection, animated: true }, base)
    set({ edges: nextEdges, nodes: reconcileSubgraphPorts(nodes, nextEdges), revision: get().revision + 1 })
  },

  addLayer: (layerType, position, opts) => {
    // Unknown layer type → refuse BEFORE commit (registry is the only authority).
    if (!LAYERS[layerType]) {
      console.warn(`addLayer rejected: unknown layer type '${layerType}'`)
      return ''
    }
    const id = opts?.id && !get().nodes.some((n) => n.id === opts.id) ? opts.id : newNodeId()
    const merged = { ...defaultParamsFor(layerType), ...(opts?.params ?? {}) }
    const node: LayerNode = {
      id,
      type: 'layer',
      position,
      data: { layerType, params: coerceParams(layerType, merged) },
    }
    set({ nodes: [...get().nodes, node], selectedNodeId: id, revision: get().revision + 1 })
    return id
  },

  updateNodeParams: (id, params) => {
    const nodes = get().nodes.map((n) => {
      if (n.id !== id) return n
      const merged = { ...n.data.params, ...params }
      return { ...n, data: { ...n.data, params: coerceParams(n.data.layerType, merged) } }
    })
    // If the edited node feeds a Subgraph, its proxy mirrors the new config.
    set({ nodes: reconcileSubgraphPorts(nodes, get().edges), revision: get().revision + 1 })
  },

  replaceNodeLayer: (id, newLayerType, extraParams) => {
    if (!LAYERS[newLayerType]) {
      console.warn(`replaceNodeLayer rejected: unknown layer type '${newLayerType}'`)
      return
    }
    set({
      nodes: get().nodes.map((n) => {
        if (n.id !== id) return n
        const merged = { ...defaultParamsFor(newLayerType), ...(extraParams ?? {}) }
        return {
          ...n,
          data: {
            ...n.data,
            layerType: newLayerType,
            params: coerceParams(newLayerType, merged),
            inferredInputShape: undefined,
            inferredOutputShape: undefined,
            hasError: false,
          },
        }
      }),
      revision: get().revision + 1,
    })
  },

  setSelectedNodeId: (id) => set({ selectedNodeId: id }),

  deleteNode: (id) => {
    if (id === 'input') return
    const edges = get().edges.filter((e) => e.source !== id && e.target !== id)
    const nodes = get().nodes.filter((n) => n.id !== id)
    set({
      nodes: reconcileSubgraphPorts(nodes, edges),
      edges,
      selectedNodeId: get().selectedNodeId === id ? null : get().selectedNodeId,
      revision: get().revision + 1,
    })
  },

  connectNodes: (source, target) => {
    const { nodes, edges } = get()
    const g = validateGuard(nodes, edges, source, target)
    if (!g.ok) {
      console.debug('connectNodes rejected:', g.reason)
      return false
    }
    if (edges.some((e) => e.source === source && e.target === target)) return true // dedupe, already connected
    const base = baseForConnect(nodes, edges, target)
    const nextEdges = addEdge({ source, target, animated: true, id: nextEdgeId(edges) }, base)
    set({ edges: nextEdges, nodes: reconcileSubgraphPorts(nodes, nextEdges), revision: get().revision + 1 })
    return true
  },

  graphIssues: () => validateGraphState(get().nodes, get().edges).issues,

  autoLayout: () => {
    const { nodes, edges } = get()
    if (nodes.length === 0) return
    const positions = computeLayout(nodes, edges, useLayoutStore.getState().direction)
    const next = nodes.map((n) => {
      const pos = positions.get(n.id)
      if (!pos) return n
      if (pos.x === n.position.x && pos.y === n.position.y) return n
      return { ...n, position: pos }
    })
    set({ nodes: next })
  },

  loadSnapshot: (snapshot) => {
    bumpNextIdPast(snapshot.nodes.map((n) => n.id))
    const fallback = { x: 0, y: 0 }
    const nodes: LayerNode[] = snapshot.nodes.map((n) => ({
      id: n.id,
      type: 'layer' as const,
      position: n.position ?? fallback,
      data: {
        layerType: n.layerType,
        params: coerceParams(n.layerType, { ...defaultParamsFor(n.layerType), ...n.params }),
      },
    }))
    const edges: Edge[] = snapshot.edges.map((e, i) => ({
      id: `e${i + 1}`,
      source: e.source,
      target: e.target,
      animated: true,
    }))
    // §4.3 validate BEFORE commit: an error-level violation (unknown node ids,
    // unknown layer, dup ids, self-loop, cycle) must never become the graph.
    const v = validateGraphState(nodes, edges)
    if (!v.ok) {
      console.warn(`loadSnapshot rejected (${v.issues.length} issues):`, v.issues)
      return false
    }
    set({ nodes, edges, selectedNodeId: null, revision: get().revision + 1, loadEpoch: get().loadEpoch + 1 })
    const needsLayout = snapshot.nodes.some((n) => !n.position)
    if (needsLayout) get().autoLayout()
    return true
  },

  resetGraph: () => {
    nextId = 1
    set({
      nodes: [{
        id: 'input', type: 'layer',
        position: { x: 250, y: 50 },
        data: { layerType: 'Input', params: defaultParamsFor('Input') },
      }],
      edges: [],
      selectedNodeId: null,
      revision: get().revision + 1,
      loadEpoch: get().loadEpoch + 1,
    })
  },
}))

export function captureSnapshot(state: { nodes: LayerNode[]; edges: Edge[] }): GraphSnapshot {
  return {
    nodes: state.nodes.map((n) => ({
      id: n.id,
      layerType: n.data.layerType,
      params: n.data.params,
      position: { x: n.position.x, y: n.position.y },
    })),
    edges: state.edges.map((e) => ({ source: e.source, target: e.target })),
  }
}

export function captureStructuralSnapshot(state: { nodes: LayerNode[]; edges: Edge[] }) {
  return {
    nodes: state.nodes.map((n) => ({
      id: n.id,
      layerType: n.data.layerType,
      params: n.data.params,
    })),
    edges: state.edges.map((e) => ({ source: e.source, target: e.target })),
  }
}

// Layer kinds that genuinely fan in (accept ≥2 incoming edges): Merge
// (Concat/Add/Multiply/Stack), Custom (user-written forward) and Subgraph.
// Everything else (module/function/output/io) is single-input.
const MULTI_INPUT_KINDS = new Set(['merge', 'custom', 'group'])

/**
 * The edge list a new connection into `target` should be added on top of. For a
 * single-input target that's already fed, the existing incoming edge is dropped
 * so the new wire REPLACES it (rewire) — instead of the connection being
 * silently swallowed as a duplicate or piling up an invalid 2-input layer. This
 * is what lets you drag a fresh source into an already-connected Linear/
 * activation/etc. Merge/Custom/Subgraph keep accepting multiple inputs.
 */
function baseForConnect(nodes: LayerNode[], edges: Edge[], target: string): Edge[] {
  const tNode = nodes.find((n) => n.id === target)
  const tKind = tNode ? LAYERS[tNode.data.layerType]?.kind : undefined
  return tKind && MULTI_INPUT_KINDS.has(tKind) ? edges : edges.filter((e) => e.target !== target)
}

/**
 * Layered DAG layout (see canvas/layout.ts): ranks by longest path so edges
 * always flow with the chosen direction, barycenter-orders each rank to reduce
 * crossings, and gives every node a distinct cross slot so nothing overlaps.
 */
function computeLayout(nodes: LayerNode[], edges: Edge[], direction: FlowDir = 'TB'): Map<string, XYPosition> {
  return layeredLayout(
    nodes.map((n) => ({ id: n.id })),
    edges.map((e) => ({ source: e.source, target: e.target })),
    { direction },
  )
}

export function autoPositionAfter(nodes: LayerNode[], afterId?: string): XYPosition {
  if (afterId) {
    const ref = nodes.find((n) => n.id === afterId)
    if (ref) return { x: ref.position.x, y: ref.position.y + 110 }
  }
  if (nodes.length === 0) return { x: 250, y: 50 }
  const maxY = Math.max(...nodes.map((n) => n.position.y))
  const last = nodes.find((n) => n.position.y === maxY)
  return { x: last?.position.x ?? 250, y: maxY + 110 }
}

/**
 * Phase-3 edge-add guard (§4.3): a new source→target wire commits only when
 *   - both endpoints exist,
 *   - it is not a self-loop,
 *   - it does not close a directed cycle (GRAPH_DAG_POLICY).
 * Fan-in/single-input rules and dedupe are handled by the caller
 * (baseForConnect rewires single-input targets; connectNodes dedupes).
 */
function validateGuard(
  nodes: LayerNode[],
  edges: Edge[],
  source: string,
  target: string,
): { ok: boolean; reason?: string } {
  const byId = new Set(nodes.map((n) => n.id))
  if (!byId.has(source)) return { ok: false, reason: `unknown source '${source}'` }
  if (!byId.has(target)) return { ok: false, reason: `unknown target '${target}'` }
  if (source === target) return { ok: false, reason: 'self-loop' }
  if (wouldCreateCycle(nodes, edges, source, target)) {
    return { ok: false, reason: `edge ${source}→${target} would close a directed cycle (DAG policy)` }
  }
  return { ok: true }
}

/** Collision-safe `e<N>` id: max existing numeric suffix + 1. (The old
 *  `e${edges.length + 1}` scheme collided after edge deletions.) */
function nextEdgeId(edges: Edge[]): string {
  let max = 0
  for (const e of edges) {
    const m = e.id?.match(/^e(\d+)$/)
    if (m) max = Math.max(max, parseInt(m[1], 10))
  }
  return `e${max + 1}`
}
