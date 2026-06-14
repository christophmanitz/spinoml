import { useMemo, useState } from 'react'

export type Series = {
  label: string
  color: string
  /** points sorted by x; y may be null to leave a gap in the line */
  points: { x: number; y: number | null }[]
}

type Props = {
  series: Series[]
  height?: number
  yLog?: boolean
  xLabel?: string
  yFormat?: (v: number) => string
}

// Fixed viewBox; the SVG scales to its container width via w-full, so we get a
// responsive chart without measuring the DOM. Hand-rolled (no chart lib dep —
// see CLAUDE.md "Don't reach for new libraries").
const VB_W = 600
const PAD_L = 46
const PAD_R = 12
const PAD_T = 10
const PAD_B = 24

function niceTicks(min: number, max: number, count = 4): number[] {
  if (!isFinite(min) || !isFinite(max) || min === max) return [min]
  const span = max - min
  const step0 = span / count
  const mag = Math.pow(10, Math.floor(Math.log10(step0)))
  const norm = step0 / mag
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag
  const start = Math.ceil(min / step) * step
  const out: number[] = []
  for (let v = start; v <= max + step * 0.5; v += step) out.push(Number(v.toFixed(10)))
  return out
}

export default function LineChart({ series, height = 220, yLog = false, xLabel, yFormat }: Props) {
  const [hoverX, setHoverX] = useState<number | null>(null)
  const VB_H = height

  const fmt = yFormat ?? ((v: number) => (Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.001) ? v.toExponential(1) : Number(v.toFixed(4)).toString()))

  const stats = useMemo(() => {
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity
    for (const s of series) {
      for (const p of s.points) {
        if (p.x < xMin) xMin = p.x
        if (p.x > xMax) xMax = p.x
        if (p.y == null) continue
        const yv = yLog ? (p.y > 0 ? Math.log10(p.y) : NaN) : p.y
        if (!isFinite(yv)) continue
        if (yv < yMin) yMin = yv
        if (yv > yMax) yMax = yv
      }
    }
    if (!isFinite(xMin)) { xMin = 0; xMax = 1 }
    if (!isFinite(yMin)) { yMin = 0; yMax = 1 }
    if (xMin === xMax) xMax = xMin + 1
    if (yMin === yMax) { yMin -= 0.5; yMax += 0.5 }
    else { const pad = (yMax - yMin) * 0.06; yMin -= pad; yMax += pad }
    return { xMin, xMax, yMin, yMax }
  }, [series, yLog])

  const sx = (x: number) => PAD_L + ((x - stats.xMin) / (stats.xMax - stats.xMin)) * (VB_W - PAD_L - PAD_R)
  const sy = (y: number) => {
    const yv = yLog ? Math.log10(y) : y
    return VB_H - PAD_B - ((yv - stats.yMin) / (stats.yMax - stats.yMin)) * (VB_H - PAD_T - PAD_B)
  }

  const yTicks = useMemo(() => {
    const raw = niceTicks(stats.yMin, stats.yMax, 4)
    return raw.map((tv) => ({ tv, label: yLog ? fmt(Math.pow(10, tv)) : fmt(tv) }))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stats, yLog])

  const xTicks = useMemo(() => niceTicks(stats.xMin, stats.xMax, 6).filter((t) => Number.isInteger(t)), [stats])

  const hasData = series.some((s) => s.points.some((p) => p.y != null))
  if (!hasData) {
    return <div className="flex h-24 items-center justify-center text-[11px] text-[#5a6068]">keine Daten</div>
  }

  // nearest x index for the crosshair
  const allX = Array.from(new Set(series.flatMap((s) => s.points.map((p) => p.x)))).sort((a, b) => a - b)
  const nearest = hoverX == null ? null : allX.reduce((best, x) => Math.abs(x - hoverX) < Math.abs(best - hoverX) ? x : best, allX[0])

  return (
    <div>
      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        className="w-full"
        style={{ height }}
        preserveAspectRatio="none"
        onMouseMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect()
          const px = ((e.clientX - r.left) / r.width) * VB_W
          const xv = stats.xMin + ((px - PAD_L) / (VB_W - PAD_L - PAD_R)) * (stats.xMax - stats.xMin)
          setHoverX(xv)
        }}
        onMouseLeave={() => setHoverX(null)}
      >
        {/* y grid + labels */}
        {yTicks.map(({ tv, label }, i) => {
          const yv = yLog ? Math.pow(10, tv) : tv
          const y = sy(yv)
          if (y < PAD_T - 1 || y > VB_H - PAD_B + 1) return null
          return (
            <g key={i}>
              <line x1={PAD_L} y1={y} x2={VB_W - PAD_R} y2={y} stroke="#1a1e22" strokeWidth={1} />
              <text x={PAD_L - 4} y={y + 3} textAnchor="end" fontSize={9} fill="#5a6068">{label}</text>
            </g>
          )
        })}
        {/* x labels */}
        {xTicks.map((t, i) => (
          <text key={i} x={sx(t)} y={VB_H - PAD_B + 12} textAnchor="middle" fontSize={9} fill="#5a6068">{t}</text>
        ))}
        {/* crosshair */}
        {nearest != null && (
          <line x1={sx(nearest)} y1={PAD_T} x2={sx(nearest)} y2={VB_H - PAD_B} stroke="#2c3238" strokeWidth={1} />
        )}
        {/* series */}
        {series.map((s) => {
          // build path, breaking the line on null y
          let d = ''
          let pen = false
          for (const p of s.points) {
            if (p.y == null || (yLog && p.y <= 0)) { pen = false; continue }
            const cmd = pen ? 'L' : 'M'
            d += `${cmd}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)} `
            pen = true
          }
          return <path key={s.label} d={d.trim()} fill="none" stroke={s.color} strokeWidth={1.5} />
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
