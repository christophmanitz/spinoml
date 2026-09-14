// Parallel-living Zustand store for the visual DATA-PROCESSING graph — the third
// canvas alongside canvas/GraphStore.ts (architecture) and training/graph/store.ts
// (training). Same React Flow plumbing, no shape inference, no protected node.
// A data graph is a DAG of data-prep steps (load → transform → fetch/graph-build
// → write) that compiles to ONE reproducible Python pipeline script
// (codegen/dataCodegen.ts). Persists to experiments/data-graphs/<name>.spinodata.

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

import { defaultDataParams, coerceDataParams } from './registry'
import { layeredLayout } from '../../canvas/layout'

export type DataNodeData = {
  dataType: string
  params: Record<string, unknown>
} & Record<string, unknown>

export type DataFlowNode = Node<DataNodeData, 'data'>

export type DataGraphSnapshot = {
  nodes: { id: string; dataType: string; params: Record<string, unknown>; position?: XYPosition }[]
  edges: { source: string; target: string }[]
}

let nextId = 1
const newNodeId = () => `d${nextId++}`
function bumpNextIdPast(ids: string[]) {
  for (const id of ids) {
    const m = id.match(/^d(\d+)$/)
    if (m) nextId = Math.max(nextId, parseInt(m[1], 10) + 1)
  }
}

// Collision-proof edge id (max existing `e<n>` + 1). Length-based ids collide
// after a delete-then-add, producing duplicate React keys.
function freshEdgeId(edges: Edge[]): string {
  let max = 0
  for (const e of edges) {
    const m = /^e(\d+)$/.exec(e.id)
    if (m) max = Math.max(max, parseInt(m[1], 10))
  }
  return `e${max + 1}`
}

type State = {
  nodes: DataFlowNode[]
  edges: Edge[]
  selectedNodeId: string | null

  onNodesChange: OnNodesChange<DataFlowNode>
  onEdgesChange: OnEdgesChange
  onConnect: OnConnect

  addNode: (dataType: string, position: XYPosition, opts?: { id?: string; params?: Record<string, unknown> }) => string
  updateNodeParams: (id: string, params: Record<string, unknown>) => void
  /** Turn a typed node into an editable CustomScript holding its generated code,
   *  in place (same id/position/edges). Manual or chatbot-driven. */
  convertToCustom: (id: string, code: string, label: string) => void
  setSelectedNodeId: (id: string | null) => void
  deleteNode: (id: string) => void
  connectNodes: (source: string, target: string) => void
  autoLayout: () => void
  loadSnapshot: (snapshot: DataGraphSnapshot) => void
  resetGraph: () => void
}

export const useDataGraphStore = create<State>((set, get) => ({
  nodes: [],
  edges: [],
  selectedNodeId: null,

  onNodesChange: (changes) => set({ nodes: applyNodeChanges(changes, get().nodes) }),
  onEdgesChange: (changes) => set({ edges: applyEdgeChanges(changes, get().edges) }),
  onConnect: (connection) => set({ edges: addEdge({ ...connection, animated: true, id: freshEdgeId(get().edges) }, get().edges) }),

  addNode: (dataType, position, opts) => {
    const id = opts?.id && !get().nodes.some((n) => n.id === opts.id) ? opts.id : newNodeId()
    const merged = { ...defaultDataParams(dataType), ...(opts?.params ?? {}) }
    const node: DataFlowNode = {
      id,
      type: 'data',
      position,
      data: { dataType, params: coerceDataParams(dataType, merged) },
    }
    set({ nodes: [...get().nodes, node], selectedNodeId: id })
    return id
  },

  updateNodeParams: (id, params) => {
    set({
      nodes: get().nodes.map((n) => {
        if (n.id !== id) return n
        const merged = { ...n.data.params, ...params }
        return { ...n, data: { ...n.data, params: coerceDataParams(n.data.dataType, merged) } }
      }),
    })
  },

  convertToCustom: (id, code, label) => {
    set({
      nodes: get().nodes.map((n) =>
        n.id === id
          ? { ...n, data: { dataType: 'CustomScript', params: coerceDataParams('CustomScript', { label, code }) } }
          : n,
      ),
      selectedNodeId: id,
    })
  },

  setSelectedNodeId: (id) => set({ selectedNodeId: id }),

  deleteNode: (id) => {
    set({
      nodes: get().nodes.filter((n) => n.id !== id),
      edges: get().edges.filter((e) => e.source !== id && e.target !== id),
      selectedNodeId: get().selectedNodeId === id ? null : get().selectedNodeId,
    })
  },

  connectNodes: (source, target) => {
    const edges = get().edges
    if (edges.some((e) => e.source === source && e.target === target)) return
    set({ edges: addEdge({ source, target, animated: true, id: freshEdgeId(edges) }, edges) })
  },

  autoLayout: () => {
    const { nodes, edges } = get()
    if (nodes.length === 0) return
    const positions = layoutLeftToRight(nodes, edges)
    set({
      nodes: nodes.map((n) => {
        const pos = positions.get(n.id)
        return pos ? { ...n, position: pos } : n
      }),
    })
  },

  loadSnapshot: (snapshot) => {
    bumpNextIdPast(snapshot.nodes.map((n) => n.id))
    const nodes: DataFlowNode[] = snapshot.nodes.map((n) => ({
      id: n.id,
      type: 'data' as const,
      position: n.position ?? { x: 0, y: 0 },
      data: {
        dataType: n.dataType,
        params: coerceDataParams(n.dataType, { ...defaultDataParams(n.dataType), ...n.params }),
      },
    }))
    const edges: Edge[] = snapshot.edges.map((e, i) => ({
      id: `e${i + 1}`, source: e.source, target: e.target, animated: true,
    }))
    set({ nodes, edges, selectedNodeId: null })
    if (snapshot.nodes.some((n) => !n.position)) get().autoLayout()
  },

  resetGraph: () => {
    nextId = 1
    set({ nodes: [], edges: [], selectedNodeId: null })
  },
}))

export function captureDataSnapshot(state: { nodes: DataFlowNode[]; edges: Edge[] }): DataGraphSnapshot {
  return {
    nodes: state.nodes.map((n) => ({
      id: n.id,
      dataType: n.data.dataType,
      params: n.data.params,
      position: { x: n.position.x, y: n.position.y },
    })),
    edges: state.edges.map((e) => ({ source: e.source, target: e.target })),
  }
}

// Left-to-right DAG layout: each node sits in the column = its longest distance
// from a root (Kahn layering), siblings stacked vertically. A pipeline reads
// naturally left→right (source → transforms → sink).
// Layered left→right layout (data nodes have fixed Left/Right handles). Shared
// with the architecture + training canvases — see canvas/layout.ts.
function layoutLeftToRight(nodes: DataFlowNode[], edges: Edge[]): Map<string, XYPosition> {
  return layeredLayout(
    nodes.map((n) => ({ id: n.id })),
    edges.map((e) => ({ source: e.source, target: e.target })),
    { direction: 'LR' },
  )
}
