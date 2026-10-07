// Run-evaluation diagrams, chosen by the run's task (the fixed schema emitted by
// the trainer as an `eval.summary` event): a confusion matrix for
// classification/binary, a predicted-vs-actual scatter for regression. Pure SVG
// + CSS so there's no chart-lib dependency; matches the dark Run-Detail theme.

import type { TrainingEvent } from '../types'

export type EvalSummary =
  | { task: 'classification' | 'binary'; confusion?: { labels: string[]; matrix: number[][] } }
  | { task: 'regression'; scatter?: { points: [number, number][]; n_total: number; pred_label?: string; truth_label?: string } }
  | { task: string }

/** A multitask eval.summary carries one entry per output head. */
export type HeadEval = { output: string } & EvalSummary

export function isEvalSummary(v: unknown): v is EvalSummary {
  if (typeof v !== 'object' || v === null) return false
  const o = v as Record<string, unknown>
  return typeof o.task === 'string'
}

/** Latest eval.summary snapshot (the trainer re-emits one on each new best).
 *  Single-task: the EvalSummary itself. Multitask: null here — use latestEvalHeads. */
export function latestEval(events: TrainingEvent[]): EvalSummary | null {
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i]
    if (e.kind === 'eval.summary') {
      // multitask payloads carry a `heads` array instead of a top-level task.
      if (Array.isArray((e as Record<string, unknown>).heads)) return null
      return isEvalSummary(e) ? e : null
    }
  }
  return null
}

/** Per-head eval summaries from the latest multitask eval.summary (null if the
 *  run is single-task). */
export function latestEvalHeads(events: TrainingEvent[]): HeadEval[] | null {
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i] as Record<string, unknown>
    if (e.kind === 'eval.summary' && Array.isArray(e.heads)) return e.heads as HeadEval[]
  }
  return null
}

/** Renders the schema-driven diagram for one eval summary: confusion matrix for
 *  classification/binary, predicted-vs-actual scatter for regression. */
export function EvalDiagram({ summary }: { summary: EvalSummary }) {
  if ((summary.task === 'classification' || summary.task === 'binary') && 'confusion' in summary && summary.confusion) {
    return <ConfusionMatrix labels={summary.confusion.labels} matrix={summary.confusion.matrix} />
  }
  if (summary.task === 'regression' && 'scatter' in summary && summary.scatter) {
    return (
      <ScatterPlot
        points={summary.scatter.points}
        nTotal={summary.scatter.n_total}
        predLabel={summary.scatter.pred_label}
        truthLabel={summary.scatter.truth_label}
      />
    )
  }
  return <div className="text-[11px] text-[#6f767e]">Für diese Aufgabe gibt es kein Standard-Diagramm.</div>
}

// ── Confusion matrix ────────────────────────────────────────────────────────

export function ConfusionMatrix({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  const n = matrix.length
  const rowSums = matrix.map((r) => r.reduce((a, b) => a + b, 0))
  const total = rowSums.reduce((a, b) => a + b, 0)
  const correct = matrix.reduce((acc, r, i) => acc + (r[i] ?? 0), 0)
  const acc = total > 0 ? correct / total : 0
  // Cell sizing: shrink as classes grow so a 10-class matrix still fits.
  const cell = n <= 6 ? 46 : n <= 12 ? 34 : 24
  const fontPx = n <= 12 ? 11 : 9

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3 text-[11px]">
        <span className="text-[#9aa1a8]">Konfusionsmatrix</span>
        <span className="text-[#6f767e]">Zeilen = Wahrheit · Spalten = Vorhersage</span>
        <span className="ml-auto text-[#5fd39a]">Accuracy {(acc * 100).toFixed(1)}%</span>
        <span className="text-[#6f767e]">n={total}</span>
      </div>
      <div className="overflow-auto">
        <div className="inline-grid" style={{ gridTemplateColumns: `auto repeat(${n}, ${cell}px)` }}>
          {/* top-left corner */}
          <div />
          {/* predicted-class header */}
          {labels.map((l, j) => (
            <div key={`h${j}`} className="flex items-end justify-center pb-1 text-[10px] text-[#6f767e]" title={`Vorhersage: ${l}`}>
              <span className="max-w-full truncate" style={{ maxWidth: cell }}>{l}</span>
            </div>
          ))}
          {matrix.map((row, i) => {
            const rs = rowSums[i] || 1
            return [
              // truth-class header
              <div key={`r${i}`} className="flex items-center justify-end pr-2 text-[10px] text-[#6f767e]" title={`Wahrheit: ${labels[i]}`}>
                <span className="max-w-[90px] truncate">{labels[i]}</span>
              </div>,
              ...row.map((v, j) => {
                const frac = v / rs // row-normalized intensity
                const onDiag = i === j
                // diagonal → green wash, off-diagonal errors → red wash
                const bg = onDiag
                  ? `rgba(95, 211, 154, ${0.12 + 0.6 * frac})`
                  : v > 0 ? `rgba(255, 122, 133, ${0.10 + 0.5 * frac})` : 'transparent'
                return (
                  <div
                    key={`c${i}-${j}`}
                    className="flex items-center justify-center border border-[#11151a]"
                    style={{ width: cell, height: cell, background: bg, fontSize: fontPx }}
                    title={`Wahrheit ${labels[i]} → Vorhersage ${labels[j]}: ${v} (${(frac * 100).toFixed(0)}%)`}
                  >
                    <span className={v > 0 ? 'text-[#e6e8eb]' : 'text-[#3a4148]'}>{v}</span>
                  </div>
                )
              }),
            ]
          })}
        </div>
      </div>
    </div>
  )
}

// ── Predicted-vs-actual scatter (regression) ──────────────────────────────────

export function ScatterPlot({
  points, nTotal, predLabel = 'Vorhersage', truthLabel = 'Wahrheit',
}: { points: [number, number][]; nTotal: number; predLabel?: string; truthLabel?: string }) {
  const W = 360, H = 300, padL = 44, padB = 32, padT = 10, padR = 10
  if (points.length === 0) return <div className="text-[11px] text-[#6f767e]">Keine Punkte.</div>
  // x = truth, y = pred. Shared domain so the y=x reference line is a true diagonal.
  const xs = points.map((p) => p[1])
  const ys = points.map((p) => p[0])
  let lo = Math.min(...xs, ...ys)
  let hi = Math.max(...xs, ...ys)
  if (lo === hi) { lo -= 1; hi += 1 }
  const pad = (hi - lo) * 0.05
  lo -= pad; hi += pad
  const sx = (v: number) => padL + ((v - lo) / (hi - lo)) * (W - padL - padR)
  const sy = (v: number) => H - padB - ((v - lo) / (hi - lo)) * (H - padB - padT)
  // Pearson correlation as a quick fit readout.
  const mx = xs.reduce((a, b) => a + b, 0) / xs.length
  const my = ys.reduce((a, b) => a + b, 0) / ys.length
  let sxy = 0, sxx = 0, syy = 0
  for (let i = 0; i < xs.length; i++) { const dx = xs[i] - mx, dy = ys[i] - my; sxy += dx * dy; sxx += dx * dx; syy += dy * dy }
  const r = sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : 0
  const ticks = niceTicks(lo, hi, 4)

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3 text-[11px]">
        <span className="text-[#9aa1a8]">Vorhersage vs. Wahrheit</span>
        <span className="ml-auto text-[#5fd39a]">r = {r.toFixed(3)}</span>
        <span className="text-[#6f767e]">{points.length < nTotal ? `${points.length} / ${nTotal}` : `n=${nTotal}`}</span>
      </div>
      <svg width={W} height={H} className="max-w-full">
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={sx(t)} y1={padT} x2={sx(t)} y2={H - padB} stroke="#171b1f" />
            <line x1={padL} y1={sy(t)} x2={W - padR} y2={sy(t)} stroke="#171b1f" />
            <text x={sx(t)} y={H - padB + 14} textAnchor="middle" fontSize="9" fill="#6f767e">{fmtTick(t)}</text>
            <text x={padL - 6} y={sy(t) + 3} textAnchor="end" fontSize="9" fill="#6f767e">{fmtTick(t)}</text>
          </g>
        ))}
        {/* y = x perfect-prediction reference */}
        <line x1={sx(lo)} y1={sy(lo)} x2={sx(hi)} y2={sy(hi)} stroke="#5fd39a" strokeDasharray="4 3" strokeOpacity="0.6" />
        {points.map((p, i) => (
          <circle key={i} cx={sx(p[1])} cy={sy(p[0])} r="2.4" fill="var(--accent)" fillOpacity="0.55" />
        ))}
        <text x={(W + padL) / 2} y={H - 2} textAnchor="middle" fontSize="10" fill="#6f767e">{truthLabel}</text>
        <text x={-((H - padB) / 2)} y={12} transform="rotate(-90)" textAnchor="middle" fontSize="10" fill="#6f767e">{predLabel}</text>
      </svg>
    </div>
  )
}

function niceTicks(lo: number, hi: number, count: number): number[] {
  const span = hi - lo
  if (span <= 0) return [lo]
  const step = niceStep(span / count)
  const start = Math.ceil(lo / step) * step
  const out: number[] = []
  for (let v = start; v <= hi + 1e-9; v += step) out.push(v)
  return out
}
function niceStep(raw: number): number {
  const mag = Math.pow(10, Math.floor(Math.log10(raw)))
  const norm = raw / mag
  const nice = norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10
  return nice * mag
}
function fmtTick(v: number): string {
  if (Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.01)) return v.toExponential(1)
  return Number(v.toFixed(2)).toString()
}
