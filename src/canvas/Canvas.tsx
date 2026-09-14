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
import { useGraphStore } from './GraphStore'
import { useScopeStore } from './scopeStore'
import { useLayoutStore } from './layoutStore'
import LayerNode from './LayerNode'
import ShapeEdge from './ShapeEdge'
import { LAYERS } from '../layers/registry'
import { useVizStore } from '../visualization/store'
import type { EdgeTypes } from '@xyflow/react'
import CanvasFileGate from '../canvasdoc/CanvasFileGate'
import ReloadCanvasButton from '../canvasdoc/ReloadCanvasButton'
import { architectureDocAdapter } from './doc'

const DRAG_MIME = 'application/spinoml-layer'

function CanvasInner() {
  const nodes = useGraphStore((s) => s.nodes)
  const edges = useGraphStore((s) => s.edges)
  const onNodesChange = useGraphStore((s) => s.onNodesChange)
  const onEdgesChange = useGraphStore((s) => s.onEdgesChange)
  const onConnect = useGraphStore((s) => s.onConnect)
  const addLayer = useGraphStore((s) => s.addLayer)
  const setSelectedNodeId = useGraphStore((s) => s.setSelectedNodeId)
  const autoLayout = useGraphStore((s) => s.autoLayout)
  const direction = useLayoutStore((s) => s.direction)
  const explainMode = useVizStore((s) => s.explainMode)
  const arrived = useVizStore((s) => s.arrived)
  const flowingEdgeIds = useVizStore((s) => s.flowingEdgeIds)
  const byNode = useVizStore((s) => s.byNode)

  // In Explain mode each edge carries the tensor-shape glyph of the data flowing
  // through it (the source node's output shape + values), appearing as the
  // example reaches it; the edge being crossed is highlighted. Derived from the
  // viz store + node shapes, so the GraphStore edges (structural state) stay clean.
  const displayEdges = useMemo(() => {
    if (!explainMode) return edges
    const lit = new Set(flowingEdgeIds)
    return edges.map((e) => {
      const shape = arrived[e.source] ? nodes.find((n) => n.id === e.source)?.data.inferredOutputShape : undefined
      const isLit = lit.has(e.id)
      if (!shape && !isLit) return e
      const p = byNode[e.source]?.preview
      const face = p?.kind === 'maps' ? p.maps[0] : p?.kind === 'matrix' ? p.grid : undefined
      return { ...e, type: 'shape', data: { shape, face, lit: isLit } }
    })
  }, [edges, nodes, explainMode, arrived, flowingEdgeIds, byNode])

  const edgeTypes = useMemo<EdgeTypes>(() => ({ shape: ShapeEdge }), [])

  const { screenToFlowPosition, fitView } = useReactFlow()
  const wrapperRef = useRef<HTMLDivElement>(null)

  const nodeTypes = useMemo<NodeTypes>(() => ({ layer: LayerNode }), [])

  const onDragOver = useCallback((e: React.DragEvent) => {
    if (e.dataTransfer.types.includes(DRAG_MIME)) {
      e.preventDefault()
      e.dataTransfer.dropEffect = 'copy'
    }
  }, [])

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      const layerType = e.dataTransfer.getData(DRAG_MIME)
      if (!layerType || !LAYERS[layerType]) return
      e.preventDefault()
      const position = screenToFlowPosition({ x: e.clientX, y: e.clientY })
      addLayer(layerType, position)
    },
    [addLayer, screenToFlowPosition],
  )

  const onSelectionChange = useCallback(
    ({ nodes: selected }: OnSelectionChangeParams) => {
      setSelectedNodeId(selected.length === 1 ? selected[0].id : null)
    },
    [setSelectedNodeId],
  )

  // Double-click a Group node → descend into its subcanvas.
  const onNodeDoubleClick = useCallback(
    (_e: React.MouseEvent, node: { id: string; data: { layerType: string } }) => {
      if (node.data.layerType !== 'Subgraph') return
      useScopeStore.getState().enterGroup(node.id)
      setTimeout(() => fitView({ duration: 200, padding: 0.2 }), 0)
    },
    [fitView],
  )

  return (
    <div ref={wrapperRef} className="relative h-full w-full" onDragOver={onDragOver} onDrop={onDrop}>
      <Breadcrumb />
      <ReactFlow
        nodes={nodes}
        edges={displayEdges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onSelectionChange={onSelectionChange}
        onNodeDoubleClick={onNodeDoubleClick}
        fitView
        colorMode="dark"
        deleteKeyCode={['Backspace', 'Delete']}
        proOptions={{ hideAttribution: true }}
      >
        <Background gap={16} size={1} />
        <Controls />
        <MiniMap pannable zoomable nodeColor="#3a4148" maskColor="#0a0c0f99" />
        <Panel position="top-right" className="!m-2 flex gap-1">
          <ReloadCanvasButton adapter={architectureDocAdapter} />
          <button
            className="rounded border border-[#1f2429] bg-[#13171b] px-2 py-1 text-[11px] text-[#9aa1a8] hover:border-[#3a4148] hover:bg-[#1a1f24] hover:text-[#e6e8eb]"
            onClick={() => {
              useLayoutStore.getState().toggle()
              autoLayout()
              setTimeout(() => fitView({ duration: 200, padding: 0.15 }), 0)
            }}
            title="Flussrichtung umschalten (oben→unten / links→rechts)"
          >
            {direction === 'LR' ? '→ L→R' : '↓ O→U'}
          </button>
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

// Subcanvas breadcrumb — only shown when focused inside one or more Group nodes.
function Breadcrumb() {
  const stack = useScopeStore((s) => s.stack)
  const exitTo = useScopeStore((s) => s.exitTo)
  const { fitView } = useReactFlow()
  if (stack.length === 0) return null

  const go = (depth: number) => {
    exitTo(depth)
    setTimeout(() => fitView({ duration: 200, padding: 0.2 }), 0)
  }

  return (
    <div className="absolute left-2 top-2 z-10 flex items-center gap-1 rounded-md border border-[#1f2429] bg-[#0e1216]/95 px-2 py-1 text-[11px] shadow-lg backdrop-blur">
      <button onClick={() => go(0)} className="rounded px-1.5 py-0.5 text-[#9aa1a8] hover:bg-[#1a1e22] hover:text-[#e6e8eb]">
        Root
      </button>
      {stack.map((f, i) => (
        <span key={i} className="flex items-center gap-1">
          <span className="text-[#5b6168]">›</span>
          <button
            onClick={() => go(i + 1)}
            className={`rounded px-1.5 py-0.5 ${
              i === stack.length - 1 ? 'bg-[var(--accent-sel)] text-[var(--accent)]' : 'text-[#9aa1a8] hover:bg-[#1a1e22] hover:text-[#e6e8eb]'
            }`}
            title={`${f.label} — Subcanvas`}
          >
            {f.label}
          </button>
        </span>
      ))}
      <button
        onClick={() => go(stack.length - 1)}
        className="ml-1 rounded border border-[#1f2429] px-1.5 py-0.5 text-[#9aa1a8] hover:bg-[#1a1e22] hover:text-[#e6e8eb]"
        title="Eine Ebene zurück"
      >
        ↩ zurück
      </button>
    </div>
  )
}

export default function Canvas() {
  return (
    <ReactFlowProvider>
      <CanvasFileGate adapter={architectureDocAdapter}>
        <CanvasInner />
      </CanvasFileGate>
    </ReactFlowProvider>
  )
}
