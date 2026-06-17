import { BaseEdge, EdgeLabelRenderer, getBezierPath, type EdgeProps } from '@xyflow/react'
import { TensorShape } from '../visualization/primitives'

// A custom edge that renders the tensor-shape glyph of the data flowing through
// it, at the edge midpoint. Used only in Explain mode; `data` is filled by
// Canvas from the source node's output shape + activations.
type ShapeEdgeData = { shape?: number[]; face?: number[][]; lit?: boolean }

export default function ShapeEdge({
  id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, markerEnd, data,
}: EdgeProps) {
  const [path, labelX, labelY] = getBezierPath({
    sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition,
  })
  const d = data as ShapeEdgeData | undefined
  const lit = !!d?.lit
  return (
    <>
      <BaseEdge id={id} path={path} markerEnd={markerEnd}
        style={lit ? { stroke: '#6ab7ff', strokeWidth: 2.5 } : undefined} />
      {d?.shape && d.shape.length > 0 && (
        <EdgeLabelRenderer>
          <div
            className="nodrag nopan"
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px) scale(0.92)`,
              pointerEvents: 'none',
            }}
          >
            <div style={{ background: '#0e1216ee', border: '1px solid #1f2429', borderRadius: 5, padding: 2 }}>
              <TensorShape shape={d.shape} face={d.face} hex={lit ? '#6ab7ff' : '#7a8088'} compact />
            </div>
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  )
}
