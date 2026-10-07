// Color / intensity helpers + legend palette. Pure, dependency-free, no JSX —
// kept out of primitives.tsx so that file only exports components (React
// Fast Refresh).

export function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '')
  const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

export const NEG: [number, number, number] = [106, 183, 255] // blue  (var(--accent))
export const POS: [number, number, number] = [255, 107, 107] // red   (#ff6b6b)

/** Diverging colour: negative → blue, 0 → transparent, positive → red. */
export function diverging(v: number, absMax: number): string {
  if (absMax <= 0) return 'transparent'
  const t = Math.max(-1, Math.min(1, v / absMax))
  const [r, g, b] = t < 0 ? NEG : POS
  return `rgba(${r}, ${g}, ${b}, ${Math.abs(t).toFixed(3)})`
}

/** Single-hue intensity: maps [min,max] → alpha over one colour. */
export function intensity(v: number, min: number, max: number, hex: string): string {
  const [r, g, b] = hexToRgb(hex)
  const t = max > min ? (v - min) / (max - min) : 0
  return `rgba(${r}, ${g}, ${b}, ${(0.08 + 0.92 * Math.max(0, Math.min(1, t))).toFixed(3)})`
}

export function gridExtent(grid: number[][]): { min: number; max: number; absMax: number } {
  let min = Infinity
  let max = -Infinity
  for (const row of grid) for (const v of row) {
    if (v < min) min = v
    if (v > max) max = v
  }
  if (!isFinite(min)) { min = 0; max = 0 }
  return { min, max, absMax: Math.max(Math.abs(min), Math.abs(max)) }
}

export const LEGEND_DIVERGING = [
  { color: '#ff6b6b', label: 'positiv (+)' },
  { color: 'var(--accent)', label: 'negativ (−)' },
]

export const legendMono = (hex: string) => [{ color: hex, label: 'heller = stärker / aktiver' }]
