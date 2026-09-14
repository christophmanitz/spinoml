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
    const m = id.match(/^n(\d+)$/)
    if (m) nextId = Math.max(nextId, parseInt(m[1], 10) + 1)
  }
}

type State = {
  nodes: LayerNode[]
  edges: Edge[]
  selectedNodeId: string | null

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
  connectNodes: (source: string, target: string) => void
  autoLayout: () => void
  loadSnapshot: (snapshot: GraphSnapshot) => void
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

  onNodesChange: (changes) => set({ nodes: applyNodeChanges(changes, get().nodes) }),
  onEdgesChange: (changes) => {
    const edges = applyEdgeChanges(changes, get().edges)
    // Edge removed/added near a Subgraph node → re-sync its input proxies.
    set({ edges, nodes: reconcileSubgraphPorts(get().nodes, edges) })
  },
  onConnect: (connection) => {
    if (!connection.source || !connection.target) return
    const base = baseForConnect(get().nodes, get().edges, connection.target)
    const edges = addEdge({ ...connection, animated: true }, base)
    set({ edges, nodes: reconcileSubgraphPorts(get().nodes, edges) })
  },

  addLayer: (layerType, position, opts) => {
    const id = opts?.id && !get().nodes.some((n) => n.id === opts.id) ? opts.id : newNodeId()
    const merged = { ...defaultParamsFor(layerType), ...(opts?.params ?? {}) }
    const node: LayerNode = {
      id,
      type: 'layer',
      position,
      data: { layerType, params: coerceParams(layerType, merged) },
    }
    set({ nodes: [...get().nodes, node], selectedNodeId: id })
    return id
  },

  updateNodeParams: (id, params) => {
    const nodes = get().nodes.map((n) => {
      if (n.id !== id) return n
      const merged = { ...n.data.params, ...params }
      return { ...n, data: { ...n.data, params: coerceParams(n.data.layerType, merged) } }
    })
    // If the edited node feeds a Subgraph, its proxy mirrors the new config.
    set({ nodes: reconcileSubgraphPorts(nodes, get().edges) })
  },

  replaceNodeLayer: (id, newLayerType, extraParams) => {
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
    })
  },

  connectNodes: (source, target) => {
    const cur = get().edges
    if (cur.some((e) => e.source === source && e.target === target)) return
    const base = baseForConnect(get().nodes, cur, target)
    const edges = addEdge({ source, target, animated: true, id: `e${cur.length + 1}` }, base)
    set({ edges, nodes: reconcileSubgraphPorts(get().nodes, edges) })
  },

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
    set({ nodes, edges, selectedNodeId: null })
    const needsLayout = snapshot.nodes.some((n) => !n.position)
    if (needsLayout) get().autoLayout()
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
