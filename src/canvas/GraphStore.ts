import { create } from 'zustand'
import {
  type Node,
  type Edge,
  type OnNodesChange,
  type OnEdgesChange,
  type OnConnect,
  Position,
  applyNodeChanges,
  applyEdgeChanges,
  addEdge,
} from '@xyflow/react'

export type LayerNodeData = {
  layerType: string
  params: Record<string, unknown>
  inferredOutputShape?: number[]
} & Record<string, unknown>

type State = {
  nodes: Node<LayerNodeData>[]
  edges: Edge[]
  onNodesChange: OnNodesChange<Node<LayerNodeData>>
  onEdgesChange: OnEdgesChange
  onConnect: OnConnect
}

export const useGraphStore = create<State>((set, get) => ({
  nodes: [
    {
      id: 'input',
      type: 'default',
      position: { x: 50, y: 200 },
      data: { layerType: 'Input', params: { shape: [1, 3, 224, 224] } },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
    },
  ],
  edges: [],
  onNodesChange: (changes) => set({ nodes: applyNodeChanges(changes, get().nodes) }),
  onEdgesChange: (changes) => set({ edges: applyEdgeChanges(changes, get().edges) }),
  onConnect: (connection) => set({ edges: addEdge(connection, get().edges) }),
}))
