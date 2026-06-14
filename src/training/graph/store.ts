// Parallel-living Zustand store for the visual TRAINING graph — the Phase-14
// analogue of canvas/GraphStore.ts. Same React Flow plumbing, but no shape
// inference and no protected Input node. Persists to experiments/
// training-graphs/<name>.mltrain (see graph/persist.ts).

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

import { defaultTrainingParams, coerceTrainingParams } from './registry'

export type TrainingNodeData = {
  trainingType: string
  params: Record<string, unknown>
} & Record<string, unknown>

export type TrainingFlowNode = Node<TrainingNodeData, 'training'>

export type TrainingGraphSnapshot = {
  nodes: { id: string; trainingType: string; params: Record<string, unknown>; position?: XYPosition }[]
  edges: { source: string; target: string }[]
}

let nextId = 1
const newNodeId = () => `t${nextId++}`
function bumpNextIdPast(ids: string[]) {
  for (const id of ids) {
    const m = id.match(/^t(\d+)$/)
    if (m) nextId = Math.max(nextId, parseInt(m[1], 10) + 1)
  }
}

type State = {
  nodes: TrainingFlowNode[]
  edges: Edge[]
  selectedNodeId: string | null

  onNodesChange: OnNodesChange<TrainingFlowNode>
  onEdgesChange: OnEdgesChange
  onConnect: OnConnect

  addNode: (trainingType: string, position: XYPosition, opts?: { id?: string; params?: Record<string, unknown> }) => string
  updateNodeParams: (id: string, params: Record<string, unknown>) => void
  setSelectedNodeId: (id: string | null) => void
  deleteNode: (id: string) => void
  connectNodes: (source: string, target: string) => void
  autoLayout: () => void
  loadSnapshot: (snapshot: TrainingGraphSnapshot) => void
  resetGraph: () => void
}

export const useTrainingGraphStore = create<State>((set, get) => ({
  nodes: [],
  edges: [],
  selectedNodeId: null,

  onNodesChange: (changes) => set({ nodes: applyNodeChanges(changes, get().nodes) }),
  onEdgesChange: (changes) => set({ edges: applyEdgeChanges(changes, get().edges) }),
  onConnect: (connection) => set({ edges: addEdge({ ...connection, animated: true }, get().edges) }),

  addNode: (trainingType, position, opts) => {
    const id = opts?.id && !get().nodes.some((n) => n.id === opts.id) ? opts.id : newNodeId()
    const merged = { ...defaultTrainingParams(trainingType), ...(opts?.params ?? {}) }
    const node: TrainingFlowNode = {
      id,
      type: 'training',
      position,
      data: { trainingType, params: coerceTrainingParams(trainingType, merged) },
    }
    set({ nodes: [...get().nodes, node], selectedNodeId: id })
    return id
  },

  updateNodeParams: (id, params) => {
    set({
      nodes: get().nodes.map((n) => {
        if (n.id !== id) return n
        const merged = { ...n.data.params, ...params }
        return { ...n, data: { ...n.data, params: coerceTrainingParams(n.data.trainingType, merged) } }
      }),
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
    set({ edges: addEdge({ source, target, animated: true, id: `e${edges.length + 1}` }, edges) })
  },

  autoLayout: () => {
    const { nodes, edges } = get()
    if (nodes.length === 0) return
    const positions = layoutTowardLoop(nodes, edges)
    set({
      nodes: nodes.map((n) => {
        const pos = positions.get(n.id)
        return pos ? { ...n, position: pos } : n
      }),
    })
  },

  loadSnapshot: (snapshot) => {
    bumpNextIdPast(snapshot.nodes.map((n) => n.id))
    const nodes: TrainingFlowNode[] = snapshot.nodes.map((n) => ({
      id: n.id,
      type: 'training' as const,
      position: n.position ?? { x: 0, y: 0 },
      data: {
        trainingType: n.trainingType,
        params: coerceTrainingParams(n.trainingType, { ...defaultTrainingParams(n.trainingType), ...n.params }),
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

export function captureTrainingSnapshot(state: { nodes: TrainingFlowNode[]; edges: Edge[] }): TrainingGraphSnapshot {
  return {
    nodes: state.nodes.map((n) => ({
      id: n.id,
      trainingType: n.data.trainingType,
      params: n.data.params,
      position: { x: n.position.x, y: n.position.y },
    })),
    edges: state.edges.map((e) => ({ source: e.source, target: e.target })),
  }
}

// Sources on the left, the TrainLoop on the right — a simple two-band layout
// (everything feeds the loop) is clearer here than the architecture topo-walk.
function layoutTowardLoop(nodes: TrainingFlowNode[], _edges: Edge[]): Map<string, XYPosition> {
  const out = new Map<string, XYPosition>()
  const loop = nodes.filter((n) => n.data.trainingType === 'TrainLoop')
  const rest = nodes.filter((n) => n.data.trainingType !== 'TrainLoop')
  rest.forEach((n, i) => out.set(n.id, { x: 80, y: 60 + i * 96 }))
  loop.forEach((n, i) => out.set(n.id, { x: 520, y: 200 + i * 140 }))
  return out
}
