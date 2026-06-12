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

export type LayerNodeData = {
  layerType: string
  params: Record<string, unknown>
  inferredInputShape?: number[]
  inferredOutputShape?: number[]
  hasError?: boolean
} & Record<string, unknown>

export type LayerNode = Node<LayerNodeData, 'layer'>

let nextId = 1
const newNodeId = () => `n${nextId++}`

type State = {
  nodes: LayerNode[]
  edges: Edge[]
  selectedNodeId: string | null

  onNodesChange: OnNodesChange<LayerNode>
  onEdgesChange: OnEdgesChange
  onConnect: OnConnect

  addLayer: (layerType: string, position: XYPosition, opts?: { id?: string; params?: Record<string, unknown> }) => string
  updateNodeParams: (id: string, params: Record<string, unknown>) => void
  setSelectedNodeId: (id: string | null) => void
  deleteNode: (id: string) => void
  connectNodes: (source: string, target: string) => void
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
}))

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
