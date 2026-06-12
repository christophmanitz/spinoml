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
import { defaultParamsFor } from '../layers/registry'

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

  addLayer: (layerType: string, position: XYPosition) => void
  updateNodeParams: (id: string, params: Record<string, unknown>) => void
  setSelectedNodeId: (id: string | null) => void
  deleteNode: (id: string) => void
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

  addLayer: (layerType, position) => {
    const id = newNodeId()
    const node: LayerNode = {
      id,
      type: 'layer',
      position,
      data: { layerType, params: defaultParamsFor(layerType) },
    }
    set({ nodes: [...get().nodes, node], selectedNodeId: id })
  },

  updateNodeParams: (id, params) => {
    set({
      nodes: get().nodes.map((n) =>
        n.id === id ? { ...n, data: { ...n.data, params: { ...n.data.params, ...params } } } : n,
      ),
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
}))
