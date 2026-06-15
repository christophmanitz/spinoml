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
import { defaultParamsFor, coerceParams } from '../layers/registry'
import { useLayoutStore, type FlowDir } from './layoutStore'

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
  onEdgesChange: (changes) => set({ edges: applyEdgeChanges(changes, get().edges) }),
  onConnect: (connection) =>
    set({ edges: addEdge({ ...connection, animated: true }, get().edges) }),

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
    set({
      nodes: get().nodes.map((n) => {
        if (n.id !== id) return n
        const merged = { ...n.data.params, ...params }
        return { ...n, data: { ...n.data, params: coerceParams(n.data.layerType, merged) } }
      }),
    })
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
    set({
      nodes: get().nodes.filter((n) => n.id !== id),
      edges: get().edges.filter((e) => e.source !== id && e.target !== id),
      selectedNodeId: get().selectedNodeId === id ? null : get().selectedNodeId,
    })
  },

  connectNodes: (source, target) => {
    const edges = get().edges
    if (edges.some((e) => e.source === source && e.target === target)) return
    set({
      edges: addEdge({ source, target, animated: true, id: `e${edges.length + 1}` }, edges),
    })
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

const COL_W = 240
const ROW_H = 110
const MAX_ROWS = 8
const ORIGIN_X = 100
const ORIGIN_Y = 60

/**
 * Topo-walk from Input, packing into vertical columns of MAX_ROWS, wrapping
 * to the right when a column fills. Forks (multiple successors) keep the
 * first child in the same column and put siblings in adjacent columns at
 * the same depth. Nodes unreachable from Input get parked in a trailing
 * column so they're at least visible.
 */
function computeLayout(nodes: LayerNode[], edges: Edge[], direction: FlowDir = 'TB'): Map<string, XYPosition> {
  const out = new Map<string, XYPosition>()
  if (nodes.length === 0) return out

  // `p` = position along the chain, `s` = branch / wrap index. The two map to
  // x/y depending on direction: TB chains downward (p→y), LR rightward (p→x).
  const pos = (p: number, s: number): XYPosition =>
    direction === 'LR'
      ? { x: ORIGIN_X + p * COL_W, y: ORIGIN_Y + s * ROW_H }
      : { x: ORIGIN_X + s * COL_W, y: ORIGIN_Y + p * ROW_H }

  const succ = new Map<string, string[]>()
  for (const n of nodes) succ.set(n.id, [])
  for (const e of edges) succ.get(e.source)?.push(e.target)

  const input = nodes.find((n) => n.data.layerType === 'Input')
  const visited = new Set<string>()
  let col = 0
  let row = 0
  let maxCol = 0

  function place(id: string) {
    if (visited.has(id)) return
    visited.add(id)
    if (row >= MAX_ROWS) { row = 0; col++ }
    out.set(id, pos(row, col))
    row++
    if (col > maxCol) maxCol = col
    const next = succ.get(id) ?? []
    if (next.length === 0) return
    place(next[0])
    for (let i = 1; i < next.length; i++) {
      const branchCol = col + i
      const startRow = Math.max(0, row - 1)
      placeBranch(next[i], branchCol, startRow)
    }
  }

  function placeBranch(id: string, startCol: number, startRow: number) {
    if (visited.has(id)) return
    visited.add(id)
    let bcol = startCol
    let brow = startRow
    if (brow >= MAX_ROWS) { brow = 0; bcol++ }
    out.set(id, pos(brow, bcol))
    if (bcol > maxCol) maxCol = bcol
    const next = succ.get(id) ?? []
    for (const nxt of next) placeBranch(nxt, bcol, brow + 1)
  }

  if (input) place(input.id)

  let orphanCol = maxCol + 2
  let orphanRow = 0
  for (const n of nodes) {
    if (visited.has(n.id)) continue
    if (orphanRow >= MAX_ROWS) { orphanRow = 0; orphanCol++ }
    out.set(n.id, pos(orphanRow, orphanCol))
    orphanRow++
  }

  return out
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
