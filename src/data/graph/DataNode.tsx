import { Handle, Position, type NodeProps, type Node } from '@xyflow/react'

import { DATA_NODES } from './registry'
import { colorForDataCategory, iconForDataCategory } from './theme'
import type { DataNodeData } from './store'

const handleStyle = { width: 8, height: 8, background: '#3a4148', border: 'none' }

export default function DataNode({ data, selected }: NodeProps<Node<DataNodeData>>) {
  const spec = DATA_NODES[data.dataType]
  const color = spec ? colorForDataCategory(spec.category) : '#ff5555'
  const summary = spec ? spec.summary(data.params) : '⚠ unknown node'
  const icon = spec ? iconForDataCategory(spec.category) : '?'

  // Sources have no input; sinks have no output — everything else flows through.
  const isSource = spec?.category === 'Source'
  const isSink = data.dataType === 'WriteDataset'

  return (
    <div
      className="min-w-[160px] rounded border bg-[#13171b]"
      style={{
        borderColor: selected ? color : '#1f2429',
        boxShadow: selected ? `0 0 0 1px ${color}40` : undefined,
      }}
    >
      {!isSource && <Handle type="target" position={Position.Left} style={handleStyle} />}
      <div
        className="flex items-center gap-1.5 rounded-t px-2 py-1 text-[11px] font-medium"
        style={{ background: `${color}18`, color }}
      >
        <span
          className="inline-flex h-4 w-4 items-center justify-center rounded text-[11px] leading-none"
          style={{ background: `${color}33`, color }}
          title={spec?.category ?? 'unknown'}
        >
          {icon}
        </span>
        <span>{data.dataType}</span>
      </div>
      <div className="px-2 py-1.5 font-mono text-[10px] text-[#9aa1a8]">{summary}</div>
      {!isSink && <Handle type="source" position={Position.Right} style={handleStyle} />}
    </div>
  )
}
