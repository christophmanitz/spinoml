// Dependency-free SVG drawing primitives for the activation visualisations.
// Same hand-rolled-SVG spirit as src/training/charts/LineChart.tsx.

import { diverging, intensity, gridExtent } from './primitiveHelpers'

export function Heatmap({
  grid, cell = 10, gap = 1, mode = 'mono', hex = 'var(--accent)', maxW,
}: {
  grid: number[][]
  cell?: number
  gap?: number
  mode?: 'mono' | 'diverging'
  hex?: string
  maxW?: number
}) {
  const rows = grid.length
  const cols = rows ? grid[0].length : 0
  if (!rows || !cols) return null
  let c = cell
  if (maxW) c = Math.max(2, Math.min(cell, Math.floor(maxW / cols)))
  const { min, max, absMax } = gridExtent(grid)
  const w = cols * (c + gap) - gap
  const h = rows * (c + gap) - gap
  return (
    <svg width={w} height={h} shapeRendering="crispEdges" style={{ display: 'block' }}>
      <rect x={0} y={0} width={w} height={h} fill="#0b0e11" />
      {grid.map((row, y) =>
        row.map((v, x) => (
          <rect
            key={`${x}-${y}`}
            x={x * (c + gap)}
            y={y * (c + gap)}
            width={c}
            height={c}
            fill={mode === 'diverging' ? diverging(v, absMax) : intensity(v, min, max, hex)}
          />
        )),
      )}
    </svg>
  )
}

/** Geometric tensor-shape glyph: makes the dimensionality tangible.
 *  1D → a strip of cells, 2D → a labelled matrix, 3D → a stack/cube (depth =
 *  channels), 4D → several cubes (batch). The front face is painted with the
 *  real values when a `face` grid is supplied. Axis sizes are drawn on the glyph. */
export function TensorShape({ shape, face, hex = 'var(--accent)', compact = false }: { shape: number[]; face?: number[][]; hex?: string; compact?: boolean }) {
  if (!shape || shape.length === 0) {
    return <div className="font-mono text-[11px]" style={{ color: hex }}>Skalar</div>
  }
  const d = shape.length
  let B = 1, C = 1, h = 1
  let w: number
  if (d === 1) { w = shape[0] }
  else if (d === 2) { h = shape[0]; w = shape[1] }
  else if (d === 3) { C = shape[0]; h = shape[1]; w = shape[2] }
  else { B = shape[0]; h = shape[d - 2]; w = shape[d - 1]; C = shape.slice(1, d - 2).reduce((a, b) => a * b, 1) }

  // Degenerate (a single vector / 1×n / n×1 / scalar-ish) → a strip of cells,
  // NOT a square matrix. Avoids showing e.g. [1] as a 2D grid.
  if (B === 1 && C === 1 && (h === 1 || w === 1)) {
    const n = Math.max(h, w, 1)
    const shown = Math.min(n, compact ? 10 : 16)
    const cs = compact ? 8 : 13
    const stripW = shown * cs
    return (
      <svg width={stripW + 14} height={cs + 14} style={{ display: 'block', overflow: 'visible' }}>
        {Array.from({ length: shown }).map((_, i) => (
          <rect key={i} x={i * cs} y={0} width={cs - 0.6} height={cs} fill={hex} fillOpacity={0.14} stroke={hex} strokeWidth={1} />
        ))}
        {n > shown && <text x={stripW + 2} y={cs - 2} fontSize={9} fill="#6f767e">…</text>}
        <text x={stripW / 2} y={cs + 10} textAnchor="middle" fontSize={9} fill="#9aa1a8">{n}</text>
      </svg>
    )
  }

  const M = compact ? 30 : 50
  const big = Math.max(h, w, 1)
  const sw = Math.max(compact ? 9 : 12, Math.round((w / big) * M))
  const sh = Math.max(compact ? 8 : 10, Math.round((h / big) * M))
  const dx = compact ? 3 : 4, dy = compact ? 3 : 4
  const drawnC = Math.min(C, 5)
  const depthW = (drawnC - 1) * dx
  const depthH = (drawnC - 1) * dy
  const blockW = sw + depthW
  const blockH = sh + depthH
  const drawnB = Math.min(B, 3)
  const gap = 14
  const padL = 16, padT = 14, padB = 16, padR = B > drawnB ? 26 : 8

  const fext = face ? gridExtent(face) : null

  function sheet(ox: number, oy: number, key: string) {
    const els = []
    for (let i = drawnC - 1; i >= 0; i--) {
      const x = ox + i * dx
      const y = oy + (drawnC - 1 - i) * dy
      const front = i === 0
      if (front && face && fext) {
        const rows = Math.min(face.length, 12)
        const cols = Math.min(face[0]?.length ?? 1, 12)
        const cw = sw / cols, ch = sh / rows
        for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
          els.push(<rect key={`${key}-${r}-${c}`} x={x + c * cw} y={y + r * ch} width={cw + 0.5} height={ch + 0.5}
            fill={intensity(face[r][c], fext.min, fext.max, hex)} />)
        }
        els.push(<rect key={`${key}-fr`} x={x} y={y} width={sw} height={sh} fill="none" stroke={hex} strokeWidth={1} />)
      } else {
        els.push(<rect key={`${key}-${i}`} x={x} y={y} width={sw} height={sh}
          fill={hex} fillOpacity={front ? 0.14 : 0.07} stroke={hex} strokeOpacity={front ? 1 : 0.5} strokeWidth={1} />)
        if (front && !face) {
          // hint a matrix with a few internal lines
          for (let g = 1; g < 4; g++) {
            els.push(<line key={`${key}-v${g}`} x1={x + (sw * g) / 4} y1={y} x2={x + (sw * g) / 4} y2={y + sh} stroke={hex} strokeOpacity={0.25} />)
            els.push(<line key={`${key}-h${g}`} x1={x} y1={y + (sh * g) / 4} x2={x + sw} y2={y + (sh * g) / 4} stroke={hex} strokeOpacity={0.25} />)
          }
        }
      }
    }
    return els
  }

  const blocks = []
  for (let b = 0; b < drawnB; b++) {
    const ox = padL + b * (blockW + gap)
    const oy = padT
    const fy = oy + depthH // front sheet top
    blocks.push(
      <g key={`b${b}`} opacity={b === 0 ? 1 : 0.85}>
        {sheet(ox, oy, `b${b}`)}
        {/* width label under the front sheet */}
        {b === 0 && <text x={ox + sw / 2} y={fy + sh + 10} textAnchor="middle" fontSize={9} fill="#9aa1a8">{w}</text>}
        {/* height label left of the front sheet */}
        {b === 0 && h > 1 && <text x={ox - 3} y={fy + sh / 2 + 3} textAnchor="end" fontSize={9} fill="#9aa1a8">{h}</text>}
        {/* channel/depth label near the top-right */}
        {b === 0 && C > 1 && <text x={ox + depthW + sw + 2} y={oy + 6} fontSize={9} fill="#9aa1a8">×{C}</text>}
      </g>,
    )
  }
  const totalW = padL + drawnB * blockW + (drawnB - 1) * gap + padR
  const totalH = padT + blockH + padB

  return (
    <svg width={totalW} height={totalH} style={{ display: 'block', overflow: 'visible' }}>
      {B > 1 && <text x={2} y={10} fontSize={9} fill="#9aa1a8">Batch {B}</text>}
      {blocks}
      {B > drawnB && (
        <text x={padL + drawnB * (blockW + gap) - gap + 4} y={padT + depthH + sh / 2} fontSize={9} fill="#6f767e">…</text>
      )}
    </svg>
  )
}

/** Histogram of a value list — shows the distribution shape (used for Norm
 *  "before vs after"). */
export function Histogram({
  values, bins = 24, width = 240, height = 60, hex = '#b39dff',
}: {
  values: number[]
  bins?: number
  width?: number
  height?: number
  hex?: string
}) {
  if (!values.length) return null
  let lo = Math.min(...values)
  let hi = Math.max(...values)
  if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5 }
  const counts = new Array(bins).fill(0)
  for (const v of values) {
    const b = Math.min(bins - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo)) * bins)))
    counts[b]++
  }
  const maxC = Math.max(...counts, 1)
  const bw = width / bins
  const zeroX = ((0 - lo) / (hi - lo)) * width
  const inRange = zeroX >= 0 && zeroX <= width
  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      {inRange && <line x1={zeroX} y1={0} x2={zeroX} y2={height} stroke="#2a2f36" strokeDasharray="2 2" />}
      {counts.map((c, i) => {
        const bh = (c / maxC) * (height - 2)
        return <rect key={i} x={i * bw} y={height - bh} width={Math.max(1, bw - 0.5)} height={bh} fill={hex} fillOpacity={0.8} />
      })}
    </svg>
  )
}

/** Node-link diagram for GNNs: nodes on a circle, edges as lines. Node fill
 *  intensity = that node's activation (when supplied). Makes a graph look like
 *  a graph instead of a matrix. */
export function NodeLinkGraph({
  edges, nNodes, activations, hex = '#34d399', size = 170,
}: {
  edges: [number, number][]
  nNodes: number
  activations?: number[]
  hex?: string
  size?: number
}) {
  const N = Math.min(Math.max(nNodes, 1), 24)
  const cx = size / 2, cy = size / 2, r = size / 2 - 16
  const pos = (i: number): [number, number] => {
    const a = (2 * Math.PI * i) / N - Math.PI / 2
    return [cx + r * Math.cos(a), cy + r * Math.sin(a)]
  }
  const absMax = activations && activations.length ? Math.max(1e-6, ...activations.map((v) => Math.abs(v))) : 1
  return (
    <svg width={size} height={size} style={{ display: 'block' }}>
      {edges.filter(([s, t]) => s < N && t < N).map(([s, t], i) => {
        const [x1, y1] = pos(s); const [x2, y2] = pos(t)
        return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#3a4148" strokeWidth={1} />
      })}
      {Array.from({ length: N }).map((_, i) => {
        const [x, y] = pos(i)
        const a = activations && i < activations.length ? Math.abs(activations[i]) / absMax : 0.55
        return <circle key={i} cx={x} cy={y} r={6} fill={hex} fillOpacity={0.2 + 0.8 * a} stroke="#0b0e11" strokeWidth={1} />
      })}
    </svg>
  )
}

/** Tiny colour legend explaining how to read a chart. */
export function Legend({ items }: { items: { color: string; label: string }[] }) {
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-[#6f767e]">
      {items.map((it, i) => (
        <span key={i} className="flex items-center gap-1">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: it.color }} />
          {it.label}
        </span>
      ))}
    </div>
  )
}

/** Vertical bars for a vector; baseline at zero so negatives drop below. */
export function VectorBars({
  values, width = 220, height = 56, hex = '#4dd0a8',
}: {
  values: number[]
  width?: number
  height?: number
  hex?: string
}) {
  if (!values.length) return null
  const absMax = Math.max(1e-6, ...values.map((v) => Math.abs(v)))
  const n = values.length
  const bw = Math.max(1, width / n)
  const mid = height / 2
  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <line x1={0} y1={mid} x2={width} y2={mid} stroke="#2a2f36" strokeWidth={1} />
      {values.map((v, i) => {
        const bh = (Math.abs(v) / absMax) * (mid - 1)
        return (
          <rect
            key={i}
            x={i * bw}
            y={v >= 0 ? mid - bh : mid}
            width={Math.max(1, bw - 0.5)}
            height={bh}
            fill={v >= 0 ? hex : 'var(--accent)'}
          />
        )
      })}
    </svg>
  )
}
