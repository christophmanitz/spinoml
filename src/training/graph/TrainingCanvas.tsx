import { useCallback, useMemo, useRef } from 'react'
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  Controls,
  MiniMap,
  Panel,
  useReactFlow,
  type NodeTypes,
  type OnSelectionChangeParams,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import { useTrainingGraphStore } from './store'
import TrainingNode from './TrainingNode'
import { TRAINING_NODES } from './registry'
import { TRAINING_DRAG_MIME } from './TrainingPalette'

function TrainingCanvasInner() {
  const nodes = useTrainingGraphStore((s) => s.nodes)
  const edges = useTrainingGraphStore((s) => s.edges)
  const onNodesChange = useTrainingGraphStore((s) => s.onNodesChange)
  const onEdgesChange = useTrainingGraphStore((s) => s.onEdgesChange)
  const onConnect = useTrainingGraphStore((s) => s.onConnect)
  const addNode = useTrainingGraphStore((s) => s.addNode)
  const setSelectedNodeId = useTrainingGraphStore((s) => s.setSelectedNodeId)
  const autoLayout = useTrainingGraphStore((s) => s.autoLayout)

  const { screenToFlowPosition, fitView } = useReactFlow()
  const wrapperRef = useRef<HTMLDivElement>(null)

  const nodeTypes = useMemo<NodeTypes>(() => ({ training: TrainingNode }), [])

  const onDragOver = useCallback((e: React.DragEvent) => {
    if (e.dataTransfer.types.includes(TRAINING_DRAG_MIME)) {
      e.preventDefault()
      e.dataTransfer.dropEffect = 'copy'
    }
  }, [])

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      const nodeType = e.dataTransfer.getData(TRAINING_DRAG_MIME)
      if (!nodeType || !TRAINING_NODES[nodeType]) return
      e.preventDefault()
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY })
      addNode(nodeType, position)
    },
    [addNode, screenToFlowPosition],
  )

  const onSelectionChange = useCallback(
    ({ nodes: selected }: OnSelectionChangeParams) => {
      setSelectedNodeId(selected.length === 1 ? selected[0].id : null)
    },
    [setSelectedNodeId],
  )

  return (
    <div ref={wrapperRef} className="h-full w-full" onDragOver={onDragOver} onDrop={onDrop}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onSelectionChange={onSelectionChange}
        fitView
        colorMode="dark"
        deleteKeyCode={['Backspace', 'Delete']}
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={16} size={1} />
        <Controls />
        <MiniMap pannable zoomable nodeColor="#3a4148" maskColor="#0b0d1099" />
        {nodes.length === 0 && (
          <Panel position="top-left" className="!m-3">
            <div className="max-w-xs rounded border border-[#1f2429] bg-[#13171b] p-3 text-[11px] text-[#7a8088]">
              Leerer Trainings-Graph. Ziehe Knoten aus der Palette: ein
              <span className="text-[#6ab7ff]"> DatasetSource</span>,
              <span className="text-[#b48ead]"> ModelSource</span>,
              <span className="text-[#5fd39a]"> Loss</span> +
              <span className="text-[#5fd39a]"> Optimizer</span> und einen
              <span className="text-[#ff7a85]"> TrainLoop</span>.
            </div>
          </Panel>
        )}
        <Panel position="top-right" className="!m-2">
          <button
            className="rounded border border-[#1f2429] bg-[#13171b] px-2 py-1 text-[11px] text-[#9aa1a8] hover:border-[#3a4148] hover:bg-[#1a1f24] hover:text-[#e6e8eb]"
            onClick={() => {
              autoLayout()
              setTimeout(() => fitView({ duration: 200, padding: 0.15 }), 0)
            }}
            title="Knoten neu anordnen"
          >
            ⊞ Auto layout
          </button>
        </Panel>
      </ReactFlow>
    </div>
  )
}

export default function TrainingCanvas() {
  return (
    <ReactFlowProvider>
      <TrainingCanvasInner />
    </ReactFlowProvider>
  )
}
