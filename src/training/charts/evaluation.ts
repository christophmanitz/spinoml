// Eval-summary schema + selectors. Kept out of charts/Evaluation.tsx so that
// file only exports components (React Fast Refresh).

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
