import { Handle, Position, type NodeProps, type Node } from '@xyflow/react'
import { LAYERS } from '../layers/registry'
import { iconForCategory, colorForCategory } from '../layers/categories'
import type { LayerNodeData } from './GraphStore'

const ERROR_COLOR = '#f43f5e'

const handleStyle = { background: '#3a4148', width: 8, height: 8, border: 'none' }

export default function LayerNode({
  data,
  selected,
}: NodeProps<Node<LayerNodeData>>) {
  const spec = LAYERS[data.layerType]
  const catColor = spec ? colorForCategory(spec.category) : '#ff5555'
  const color = data.hasError ? ERROR_COLOR : catColor
  const summary = spec ? spec.summary(data.params) : '⚠ unknown layer'
  const icon = spec ? iconForCategory(spec.category) : '?'

  const hasInput = data.layerType !== 'Input'
  const hasOutput = data.layerType !== 'Output'

  const shapeText = data.inferredOutputShape ? `[${data.inferredOutputShape.join(', ')}]` : null

  return (
    <div
      className="min-w-[160px] rounded border bg-[#13171b] shadow-sm"
      style={{
        borderColor: data.hasError ? ERROR_COLOR : selected ? catColor : '#1f2429',
        boxShadow: data.hasError
          ? `0 0 0 1px ${ERROR_COLOR}66`
          : selected ? `0 0 0 1px ${catColor}40` : undefined,
      }}
    >
      {hasInput && <Handle type="target" position={Position.Top} style={handleStyle} />}

      <div
        className="flex items-center justify-between rounded-t px-2 py-1 text-[11px] font-medium"
        style={{ background: `${color}18`, color }}
      >
        <div className="flex items-center gap-1.5">
          <span
            aria-hidden
            className="inline-flex h-4 w-4 items-center justify-center rounded text-[11px] leading-none"
            style={{ background: `${color}33`, color }}
            title={spec?.category ?? 'unknown'}
          >
            {icon}
          </span>
          <span>{data.layerType}</span>
        </div>
        {data.hasError && <span className="text-[10px]" title="forward pass failed here">!</span>}
      </div>
      <div className="px-2 py-1.5 font-mono text-[10px] text-[#9aa1a8]">{summary}</div>
      {shapeText && (
        <div className="border-t border-[#1f2429] px-2 py-1 font-mono text-[10px] text-[#7a8088]">
          out {shapeText}
        </div>
      )}

      {hasOutput && <Handle type="source" position={Position.Bottom} style={handleStyle} />}
    </div>
  )
}
