import type { TrainingEvent } from '../types'
import type { Series } from './LineChart'
import { OKABE_ITO } from '../../figures/lineFigure'

// A small, stable palette so the same logical series gets the same colour
// across charts and across runs in compare-mode.
// Screen theme uses CSS variables (first entry is --accent); print theme uses
// OKABE_ITO literal hexes via lineFigure.ts.
export const CHART_COLORS = [
  'var(--accent)', '#5fd39a', '#e6c34a', '#ff7a85', '#b98cff', '#4ec9c9', '#ff9f5a', '#9aa1a8',
]

export { OKABE_ITO }

export function parseEventLines(text: string): TrainingEvent[] {
  const out: TrainingEvent[] = []
  for (const line of text.split('\n')) {
    const s = line.trim()
    if (!s) continue
    try { out.push(JSON.parse(s)) } catch { /* skip partial trailing line */ }
  }
  return out
}

export function epochEnds(events: TrainingEvent[]): TrainingEvent[] {
  return events.filter((e) => e.kind === 'epoch.end')
}

/** train_loss + val_loss over epochs. */
export function lossSeries(events: TrainingEvent[]): Series[] {
  const ep = epochEnds(events)
  return [
    { label: 'train', color: CHART_COLORS[0], points: ep.map((e) => ({ x: (e.epoch as number) + 1, y: numOrNull(e.train_loss) })) },
    { label: 'val', color: CHART_COLORS[3], points: ep.map((e) => ({ x: (e.epoch as number) + 1, y: numOrNull(e.val_loss) })) },
  ]
}

/** Learning rate over epochs (also carried per-batch but epoch granularity is enough). */
export function lrSeries(events: TrainingEvent[]): Series[] {
  const ep = epochEnds(events)
  return [{ label: 'lr', color: CHART_COLORS[2], points: ep.map((e) => ({ x: (e.epoch as number) + 1, y: numOrNull(e.lr) })) }]
}

/** One line per extra metric (accuracy/f1/mse/…) plus val_acc if present. */
export function metricSeries(events: TrainingEvent[]): Series[] {
  const ep = epochEnds(events)
  const keys = new Set<string>()
  for (const e of ep) {
    if (e.val_acc != null) keys.add('val_acc')
    const m = e.metrics as Record<string, unknown> | null | undefined
    if (m) for (const k of Object.keys(m)) keys.add(k)
  }
  let ci = 0
  return Array.from(keys).map((k) => ({
    label: k,
    color: CHART_COLORS[(ci++) % CHART_COLORS.length],
    points: ep.map((e) => ({
      x: (e.epoch as number) + 1,
      y: k === 'val_acc' ? numOrNull(e.val_acc) : numOrNull((e.metrics as Record<string, unknown> | undefined)?.[k]),
    })),
  }))
}

function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && isFinite(v) ? v : null
}
