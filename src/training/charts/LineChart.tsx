import { useMemo, useState } from 'react'
import {
  type Series,
  computeStats,
  buildPath,
  VB_W,
  PAD_L,
  PAD_R,
  PAD_T,
  PAD_B,
  DEFAULT_HEIGHT,
} from '../../figures/lineFigure'

export type { Series } from '../../figures/lineFigure'

type Props = {
  series: Series[]
  height?: number
  yLog?: boolean
  xLabel?: string
  yFormat?: (v: number) => string
  xDomain?: [number, number]
}

export default function LineChart({ series, height = DEFAULT_HEIGHT, yLog = false, xLabel, yFormat, xDomain }: Props) {
  const [hoverX, setHoverX] = useState<number | null>(null)

  const fmt = yFormat ?? ((v: number) => (Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.001) ? v.toExponential(1) : Number(v.toFixed(4)).toString()))

  const stats = useMemo(() => computeStats(series, { yLog, yFormat: fmt, xDomain, height }), [series, yLog, fmt, xDomain, height])
  const { hasData, sx, sy, yTicks, xTicks, xMin, xMax } = stats

  if (!hasData) {
    return <div className="flex h-24 items-center justify-center text-[11px] text-[#5a6068]">keine Daten</div>
  }

  const allX = Array.from(new Set(series.flatMap((s) => s.points.map((p) => p.x)))).sort((a, b) => a - b)
  const nearest = hoverX == null ? null : allX.reduce((best, x) => Math.abs(x - hoverX) < Math.abs(best - hoverX) ? x : best, allX[0])

  return (
    <div>
      <svg
        viewBox={`0 0 ${VB_W} ${height}`}
        className="w-full"
        style={{ height }}
        preserveAspectRatio="none"
        onMouseMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect()
          const px = ((e.clientX - r.left) / r.width) * VB_W
          const xv = xMin + ((px - PAD_L) / (VB_W - PAD_L - PAD_R)) * (xMax - xMin)
          setHoverX(xv)
        }}
        onMouseLeave={() => setHoverX(null)}
      >
        {/* y grid + labels */}
        {yTicks.map(({ yv, label }, i) => {
          const y = sy(yv)
          if (y < PAD_T - 1 || y > height - PAD_B + 1) return null
          return (
            <g key={i}>
              <line x1={PAD_L} y1={y} x2={VB_W - PAD_R} y2={y} stroke="#1a1e22" strokeWidth={1} />
              <text x={PAD_L - 4} y={y + 3} textAnchor="end" fontSize={9} fill="#5a6068">{label}</text>
            </g>
          )
        })}
        {/* x labels */}
        {xTicks.map((t, i) => (
          <text key={i} x={sx(t)} y={height - PAD_B + 12} textAnchor="middle" fontSize={9} fill="#5a6068">{t}</text>
        ))}
        {/* crosshair */}
        {nearest != null && (
          <line x1={sx(nearest)} y1={PAD_T} x2={sx(nearest)} y2={height - PAD_B} stroke="#2c3238" strokeWidth={1} />
        )}
        {/* series */}
        {series.map((s) => {
          const d = buildPath(s.points, sx, sy, yLog)
          return <path key={s.label} d={d} fill="none" stroke={s.color} strokeWidth={1.5} />
        })}
        {/* hover dots */}
        {nearest != null && series.map((s) => {
          const p = s.points.find((q) => q.x === nearest)
          if (!p || p.y == null || (yLog && p.y <= 0)) return null
          return <circle key={s.label} cx={sx(nearest)} cy={sy(p.y)} r={2.5} fill={s.color} />
        })}
      </svg>

      {/* legend + hover readout */}
      <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 px-1 text-[10px]">
        {series.map((s) => {
          const hv = nearest != null ? s.points.find((q) => q.x === nearest)?.y : null
          return (
            <span key={s.label} className="flex items-center gap-1 text-[#9aa1a8]">
              <span className="inline-block h-2 w-2 rounded-sm" style={{ background: s.color }} />
              {s.label}
              {hv != null && <span className="font-mono text-[#cfd3d8]">{fmt(hv)}</span>}
            </span>
          )
        })}
        {xLabel && <span className="ml-auto text-[#5a6068]">{nearest != null ? `${xLabel} ${nearest}` : xLabel}</span>}
      </div>
    </div>
  )
}