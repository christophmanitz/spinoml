export const OKABE_ITO = [
  '#0072B2',
  '#E69F00',
  '#009E73',
  '#D55E00',
  '#CC79A7',
  '#56B4E9',
  '#F0E442',
  '#999999',
] as const

export type FigureTheme = 'screen' | 'print'

export type Series = {
  label: string
  color: string
  points: { x: number; y: number | null }[]
}

export type LineFigureOptions = {
  theme: FigureTheme
  width?: number
  height?: number
  xLabel?: string
  yLabel?: string
  title?: string
  yLog?: boolean
  yFormat?: (v: number) => string
  xDomain?: [number, number]
}

export const VB_W = 600
export const PAD_L = 46
export const PAD_R = 12
export const PAD_T = 10
export const PAD_B = 24
export const DEFAULT_WIDTH = 240
export const DEFAULT_HEIGHT = 170

export function niceTicks(min: number, max: number, count = 4): number[] {
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

export function escapeXml(s: string): string {
  return s
    .replace(/&/g, '&')
    .replace(/</g, '<')
    .replace(/>/g, '>')
    .replace(/"/g, '"')
    .replace(/'/g, '&apos;')
}

export function getSeriesColor(seriesColor: string, index: number, theme: FigureTheme): string {
  if (theme === 'print') {
    return OKABE_ITO[index % OKABE_ITO.length]
  }
  return seriesColor
}

export type FigureStats = {
  xMin: number
  xMax: number
  yMin: number
  yMax: number
  hasData: boolean
  sx: (x: number) => number
  sy: (y: number) => number
  yTicks: { tv: number; yv: number; label: string }[]
  xTicks: number[]
}

export function computeStats(
  series: Series[],
  opts: { yLog?: boolean; yFormat?: (v: number) => string; xDomain?: [number, number]; height: number }
): FigureStats {
  const { yLog = false, yFormat, xDomain, height } = opts

  const fmt = yFormat ?? ((v: number) =>
    Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.001) ? v.toExponential(1) : Number(v.toFixed(4)).toString()
  )

  let xMin = Infinity
  let xMax = -Infinity
  let yMin = Infinity
  let yMax = -Infinity
  let hasData = false

  for (const s of series) {
    for (const p of s.points) {
      if (p.x < xMin) xMin = p.x
      if (p.x > xMax) xMax = p.x
      if (p.y == null) continue
      const yv = yLog ? (p.y > 0 ? Math.log10(p.y) : NaN) : p.y
      if (!isFinite(yv)) continue
      if (yv < yMin) yMin = yv
      if (yv > yMax) yMax = yv
      hasData = true
    }
  }

  if (xDomain) {
    xMin = xDomain[0]
    xMax = xDomain[1]
  }
  if (!isFinite(xMin)) {
    xMin = 0
    xMax = 1
  }
  if (!isFinite(yMin)) {
    yMin = 0
    yMax = 1
  }
  if (xMin === xMax) xMax = xMin + 1
  if (yMin === yMax) {
    yMin -= 0.5
    yMax += 0.5
  } else {
    const pad = (yMax - yMin) * 0.06
    yMin -= pad
    yMax += pad
  }

  const sx = (x: number) => PAD_L + ((x - xMin) / (xMax - xMin)) * (VB_W - PAD_L - PAD_R)
  const sy = (y: number) => {
    const yv = yLog ? Math.log10(y) : y
    return height - PAD_B - ((yv - yMin) / (yMax - yMin)) * (height - PAD_T - PAD_B)
  }

  const yTicksRaw = niceTicks(yMin, yMax, 4)
  const yTicks = yTicksRaw.map((_tv) => {
    const yv = yLog ? Math.pow(10, _tv) : _tv
    return { tv: _tv, yv, label: fmt(yv) }
  })

  const xTicks = niceTicks(xMin, xMax, 6).filter((t) => Number.isInteger(t))

  return { xMin, xMax, yMin, yMax, hasData, sx, sy, yTicks, xTicks }
}

export function buildPath(
  points: { x: number; y: number | null }[],
  sx: (x: number) => number,
  sy: (y: number) => number,
  yLog: boolean
): string {
  let d = ''
  let pen = false
  for (const p of points) {
    if (p.y == null || (yLog && p.y <= 0)) {
      pen = false
      continue
    }
    const cmd = pen ? 'L' : 'M'
    d += `${cmd}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)} `
    pen = true
  }
  return d.trim()
}

export function lineFigureSvg(series: Series[], opts: LineFigureOptions): string {
  const {
    theme,
    width = DEFAULT_WIDTH,
    height = DEFAULT_HEIGHT,
    xLabel,
    yLabel,
    title,
    yLog = false,
    yFormat,
    xDomain,
  } = opts

  const stats = computeStats(series, { yLog, yFormat, xDomain, height })

  if (!stats.hasData) {
    return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${VB_W} ${height}" preserveAspectRatio="xMidYMid meet"><rect x="0" y="0" width="${VB_W}" height="${height}" fill="${theme === 'screen' ? 'transparent' : '#ffffff'}"/><text x="${VB_W / 2}" y="${height / 2}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="7" fill="#000000">keine Daten</text></svg>`
  }

  const isScreen = theme === 'screen'
  const bgColor = isScreen ? 'transparent' : '#ffffff'
  const axisColor = isScreen ? '#1a1e22' : '#000000'
  const textColor = isScreen ? '#5a6068' : '#000000'
  const gridColor = isScreen ? '#1a1e22' : '#cccccc'
  const fontFamily = isScreen ? 'inherit' : 'Arial, Helvetica, sans-serif'
  const fontSize = isScreen ? 9 : 7
  const titleFontSize = isScreen ? 11 : 9

  const vbHeight = height

  const parts: string[] = []
  parts.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${VB_W} ${vbHeight}" preserveAspectRatio="xMidYMid meet">`)
  parts.push(`  <rect x="0" y="0" width="${VB_W}" height="${vbHeight}" fill="${bgColor}"/>`)

  if (title) {
    parts.push(`  <text x="${VB_W / 2}" y="${PAD_T - 2}" text-anchor="middle" font-family="${fontFamily}" font-size="${titleFontSize}" fill="${textColor}">${escapeXml(title)}</text>`)
  }

  // y grid + labels
  for (const { yv, label } of stats.yTicks) {
    const y = stats.sy(yv)
    if (y < PAD_T - 1 || y > vbHeight - PAD_B + 1) continue
    parts.push(`  <line x1="${PAD_L}" y1="${y.toFixed(1)}" x2="${VB_W - PAD_R}" y2="${y.toFixed(1)}" stroke="${gridColor}" stroke-width="1"/>`)
    parts.push(`  <text x="${PAD_L - 4}" y="${(y + 3).toFixed(1)}" text-anchor="end" font-family="${fontFamily}" font-size="${fontSize}" fill="${textColor}">${escapeXml(label)}</text>`)
  }

  // x labels
  for (const t of stats.xTicks) {
    const x = stats.sx(t)
    parts.push(`  <text x="${x.toFixed(1)}" y="${vbHeight - PAD_B + 12}" text-anchor="middle" font-family="${fontFamily}" font-size="${fontSize}" fill="${textColor}">${escapeXml(String(t))}</text>`)
  }

  // series lines
  series.forEach((s, si) => {
    const color = getSeriesColor(s.color, si, theme)
    const d = buildPath(s.points, stats.sx, stats.sy, yLog)
    if (d) {
      parts.push(`  <path d="${d}" fill="none" stroke="${color}" stroke-width="1.5"/>`)
    }
  })

  // legend inside SVG
  const legendX = VB_W - PAD_R - 120
  const legendY = PAD_T + 4
  const legendLineH = fontSize + 2
  series.forEach((s, si) => {
    const color = getSeriesColor(s.color, si, theme)
    const ly = legendY + si * legendLineH
    parts.push(`  <rect x="${legendX}" y="${ly - fontSize + 1}" width="8" height="${fontSize - 1}" fill="${color}"/>`)
    parts.push(`  <text x="${legendX + 10}" y="${ly}" font-family="${fontFamily}" font-size="${fontSize}" fill="${textColor}">${escapeXml(s.label)}</text>`)
  })

  // axis labels
  if (xLabel) {
    parts.push(`  <text x="${VB_W / 2}" y="${vbHeight - 2}" text-anchor="middle" font-family="${fontFamily}" font-size="${fontSize}" fill="${textColor}">${escapeXml(xLabel)}</text>`)
  }
  if (yLabel) {
    parts.push(`  <text x="2" y="${vbHeight / 2}" text-anchor="middle" font-family="${fontFamily}" font-size="${fontSize}" fill="${textColor}" transform="rotate(-90, 2, ${vbHeight / 2})">${escapeXml(yLabel)}</text>`)
  }

  // axes
  parts.push(`  <line x1="${PAD_L}" y1="${PAD_T}" x2="${PAD_L}" y2="${vbHeight - PAD_B}" stroke="${axisColor}" stroke-width="1"/>`)
  parts.push(`  <line x1="${PAD_L}" y1="${vbHeight - PAD_B}" x2="${VB_W - PAD_R}" y2="${vbHeight - PAD_B}" stroke="${axisColor}" stroke-width="1"/>`)

  parts.push('</svg>')
  return parts.join('\n')
}