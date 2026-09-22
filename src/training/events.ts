// Phase 31 — training event ordering (TODO §32).
//
// events.jsonl is append-only, fsynced per line and written by a single thread,
// so on-disk ordering is strict. The risk is at the CONSUMPTION boundary: an
// events read (especially over ssh) can complete out of order and a stale
// snapshot would overwrite a newer one in the UI — e.g. showing EPOCH 6 after
// run.failed was already rendered, i.e. "a late event must not overwrite a final
// state". This module is the pure, unit-tested vocabulary for that protection:
//   - `finalTerminal` / `truncateAtTerminal` — the FIRST terminal event
//     (run.done / run.failed / run.cancelled) decides the final state; anything
//     after it is a stale/out-of-order write and is dropped.
//   - `latestWinsGuard` — drops an older in-flight read whose response lands
//     after a newer read already applied (the inference/store staleness guard,
//     Phase 10, applied to events).
//   - `parseFinalEvents` — parse (tolerant of a partial trailing line after a
//     SIGKILL) + truncate in one step; what the modals feed setEvents.

import { parseEventLines } from './charts/series'
import type { TrainingEvent } from './types'

/** Terminal event kinds. A healthy run reaches exactly ONE of them; the first
 *  one seen in file order is the run's final state. */
export const TERMINAL_EVENT_KINDS = ['run.done', 'run.failed', 'run.cancelled'] as const

export function isTerminalEventKind(kind: string): boolean {
  return (TERMINAL_EVENT_KINDS as readonly string[]).includes(kind)
}

/** The FIRST terminal event in file order (the run's final state), or null
 *  while the run is still in progress. Later terminal events would be
 *  duplicates/stale and MUST NOT override the first one. */
export function finalTerminal(events: TrainingEvent[]): TrainingEvent | null {
  for (const e of events) if (isTerminalEventKind(e.kind)) return e
  return null
}

/** Events up to and including the first terminal event. Any event that appears
 *  after it (a late batch/epoch flush, a duplicate terminal write) is dropped —
 *  it must never surface in charts, duration or the failure/done banners. */
export function truncateAtTerminal(events: TrainingEvent[]): TrainingEvent[] {
  const i = events.findIndex((e) => isTerminalEventKind(e.kind))
  return i === -1 ? events : events.slice(0, i + 1)
}

/** True when the stream contains events AFTER the final one — i.e. the file
 *  (or the read) carried out-of-order data past a terminal state. */
export function hasStaleTrailing(events: TrainingEvent[]): boolean {
  return finalTerminal(events) !== null && truncateAtTerminal(events).length < events.length
}

/** Latest event of a kind, ignoring anything after the final terminal event —
 *  a trailing EPOCH after FAILED can't win the chart's last-point/curEpoch
 *  derivation. */
export function latestEvent(events: TrainingEvent[], kind: string): TrainingEvent | null {
  const prefix = truncateAtTerminal(events)
  for (let i = prefix.length - 1; i >= 0; i--) if (prefix[i].kind === kind) return prefix[i]
  return null
}

/** Parse an events.jsonl text (skipping a partial trailing line) AND truncate
 *  at the first terminal event — the single entry point the UI feeds setEvents. */
export function parseFinalEvents(text: string): TrainingEvent[] {
  return truncateAtTerminal(parseEventLines(text))
}

/** Monotonic "only the newest read may apply" guard. Begin a read (captures a
 *  sequence id), await it, then pass its result to `apply`: if a NEWER read has
 *  started in the meantime, the result is dropped. The events file only grows,
 *  so the newest-STARTED read always observes a superset, and an older response
 *  landing late must never overwrite it (that's the final-state protection). */
export function latestWinsGuard<T>(apply: (value: T) => void): (read: () => Promise<T>) => Promise<void> {
  let seq = 0
  return async (read) => {
    const id = ++seq
    const value = await read()
    if (id !== seq) return
    apply(value)
  }
}