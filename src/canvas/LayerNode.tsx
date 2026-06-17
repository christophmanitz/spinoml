import { useEffect } from 'react'
import { Handle, Position, useUpdateNodeInternals, type NodeProps, type Node } from '@xyflow/react'
import { LAYERS } from '../layers/registry'
import { colorForCategory } from '../layers/categories'
import CategoryIcon from '../layers/CategoryIcon'
import type { LayerNodeData } from './GraphStore'
import { useLayoutStore } from './layoutStore'
import { useVizStore } from '../visualization/store'
import MiniViz from '../visualization/MiniViz'

const ERROR_COLOR = '#f43f5e'

const handleStyle = { background: '#3a4148', width: 8, height: 8, border: 'none' }

export default function LayerNode({
  id,
  data,
  selected,
}: NodeProps<Node<LayerNodeData>>) {
  const spec = LAYERS[data.layerType]
  const catColor = spec ? colorForCategory(spec.category) : '#ff5555'
  const color = data.hasError ? ERROR_COLOR : catColor
  const summary = spec ? spec.summary(data.params) : '⚠ unknown layer'

  // Input-kind nodes (Input, Graph) have no incoming port; output-kind have no
  // outgoing port. Use the registry kind so new input/output types work too.
  const hasInput = spec?.kind !== 'input' && data.layerType !== 'Input'
  const hasOutput = spec?.kind !== 'output' && data.layerType !== 'Output'

  const dir = useLayoutStore((s) => s.direction)
  const targetPos = dir === 'LR' ? Position.Left : Position.Top
  const sourcePos = dir === 'LR' ? Position.Right : Position.Bottom

  // React Flow caches handle bounds; when the direction flips we must tell it to
  // re-measure, otherwise edges keep routing to the old anchor positions.
  const updateNodeInternals = useUpdateNodeInternals()
  useEffect(() => { updateNodeInternals(id) }, [dir, id, updateNodeInternals])

  const shapeText = data.inferredOutputShape ? `[${data.inferredOutputShape.join(', ')}]` : null

  const explainMode = useVizStore((s) => s.explainMode)
  const act = useVizStore((s) => s.byNode[id])
  const arrived = useVizStore((s) => s.arrived[id])
  const isHead = useVizStore((s) => s.flowHeadId === id)

  return (
    <div
      className={`min-w-[160px] rounded border bg-[#13171b] shadow-sm${isHead ? ' animate-pulse' : ''}`}
      style={{
        borderColor: data.hasError ? ERROR_COLOR : (isHead || arrived) ? catColor : selected ? catColor : '#1f2429',
        boxShadow: data.hasError
          ? `0 0 0 1px ${ERROR_COLOR}66`
          : isHead ? `0 0 0 2px ${catColor}, 0 0 18px ${catColor}99`
          : arrived ? `0 0 0 1px ${catColor}88`
          : selected ? `0 0 0 1px ${catColor}40` : undefined,
      }}
    >
      {hasInput && <Handle type="target" position={targetPos} style={handleStyle} />}

      <div
        className="flex items-center justify-between rounded-t px-2 py-1 text-[11px] font-medium"
        style={{ background: `${color}18`, color }}
      >
        <div className="flex items-center gap-1.5">
          <span
            aria-hidden
            className="inline-flex h-4 w-4 items-center justify-center rounded"
            style={{ background: `${color}33`, color }}
            title={spec?.category ?? 'unknown'}
          >
            <CategoryIcon cat={spec?.category} />
          </span>
          <span>{data.layerType}</span>
        </div>
        {data.hasError && <span className="text-[10px]" title="forward pass failed here">!</span>}
      </div>
      <div className="px-2 py-1.5 font-mono text-[10px] text-[#9aa1a8]">{summary}</div>
      {explainMode && act && arrived && (
        <div className="border-t border-[#1f2429] px-2 py-1.5">
          <MiniViz act={act} hex={catColor} />
        </div>
      )}
      {spec?.kind === 'group' && (
        <div className="border-t border-dashed border-[#2a2f36] px-2 py-1 text-[10px] text-[#7a8088]">
          ⤢ Doppelklick → Subcanvas
        </div>
      )}
      {shapeText && (
        <div className="border-t border-[#1f2429] px-2 py-1 font-mono text-[10px] text-[#7a8088]">
          out {shapeText}
        </div>
      )}

      {hasOutput && <Handle type="source" position={sourcePos} style={handleStyle} />}
    </div>
  )
}
