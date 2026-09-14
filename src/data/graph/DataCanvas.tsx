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

import { useDataGraphStore } from './store'
import DataNode from './DataNode'
import DataGraphBar from './DataGraphBar'
import { DATA_NODES } from './registry'
import { DATA_DRAG_MIME } from './DataPalette'
import CanvasFileGate from '../../canvasdoc/CanvasFileGate'
import ReloadCanvasButton from '../../canvasdoc/ReloadCanvasButton'
import { dataDocAdapter } from './doc'

function DataCanvasInner() {
  const nodes = useDataGraphStore((s) => s.nodes)
  const edges = useDataGraphStore((s) => s.edges)
  const onNodesChange = useDataGraphStore((s) => s.onNodesChange)
  const onEdgesChange = useDataGraphStore((s) => s.onEdgesChange)
  const onConnect = useDataGraphStore((s) => s.onConnect)
  const addNode = useDataGraphStore((s) => s.addNode)
  const setSelectedNodeId = useDataGraphStore((s) => s.setSelectedNodeId)
  const autoLayout = useDataGraphStore((s) => s.autoLayout)

  const { screenToFlowPosition, fitView } = useReactFlow()
  const wrapperRef = useRef<HTMLDivElement>(null)

  const nodeTypes = useMemo<NodeTypes>(() => ({ data: DataNode }), [])

  const onDragOver = useCallback((e: React.DragEvent) => {
    if (e.dataTransfer.types.includes(DATA_DRAG_MIME)) {
      e.preventDefault()
      e.dataTransfer.dropEffect = 'copy'
    }
  }, [])

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      const nodeType = e.dataTransfer.getData(DATA_DRAG_MIME)
      if (!nodeType || !DATA_NODES[nodeType]) return
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
        <MiniMap pannable zoomable nodeColor="#3a4148" maskColor="#0a0c0f99" />
        <Panel position="top-left" className="!m-2">
          <DataGraphBar />
        </Panel>
        <Panel position="top-right" className="!m-2 flex gap-1">
          <ReloadCanvasButton adapter={dataDocAdapter} />
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

export default function DataCanvas() {
  return (
    <ReactFlowProvider>
      <CanvasFileGate adapter={dataDocAdapter}>
        <DataCanvasInner />
      </CanvasFileGate>
    </ReactFlowProvider>
  )
}
