import { Handle, Position, type NodeProps, type Node } from '@xyflow/react'

import { TRAINING_NODES } from './registry'
import { colorForTrainingCategory, iconForTrainingCategory } from './theme'
import type { TrainingNodeData } from './store'

const handleStyle = { width: 8, height: 8, background: '#3a4148', border: 'none' }

export default function TrainingNode({ data, selected }: NodeProps<Node<TrainingNodeData>>) {
  const spec = TRAINING_NODES[data.trainingType]
  const color = spec ? colorForTrainingCategory(spec.category) : '#ff5555'
  const summary = spec ? spec.summary(data.params) : '⚠ unknown node'
  const icon = spec ? iconForTrainingCategory(spec.category) : '?'

  // TrainLoop is the sink (only inputs); everything else can feed onward too.
  const isLoop = data.trainingType === 'TrainLoop'

  return (
    <div
      className="min-w-[150px] rounded border bg-[#13171b]"
      style={{
        borderColor: selected ? color : '#1f2429',
        boxShadow: selected ? `0 0 0 1px ${color}40` : undefined,
      }}
    >
      <Handle type="target" position={Position.Left} style={handleStyle} />
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
        <span>{data.trainingType}</span>
      </div>
      <div className="px-2 py-1.5 font-mono text-[10px] text-[#9aa1a8]">{summary}</div>
      {!isLoop && <Handle type="source" position={Position.Right} style={handleStyle} />}
    </div>
  )
}
