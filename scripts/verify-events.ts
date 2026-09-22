#!/usr/bin/env tsx
// Phase 31 — training event ordering (TODO §32).
// "A late event must not overwrite a final state." events.jsonl is append-only
// + fsynced per line, so on-disk ordering is strict; the risk is at the
// consumption boundary — a stale read snapshot landing after the final one, or
// a trailing out-of-order line (EPOCH after FAILED). This harness proves the
// pure protection vocabulary in src/training/events.ts: terminal-state
// derivation, truncation, and the latest-read-wins guard. Pure TS, no python,
// no torch.

import { parseEventLines } from '../src/training/charts/series'
import {
  TERMINAL_EVENT_KINDS,
  finalTerminal,
  truncateAtTerminal,
  hasStaleTrailing,
  latestEvent,
  parseFinalEvents,
  latestWinsGuard,
} from '../src/training/events'
import type { TrainingEvent } from '../src/training/types'

let failures = 0
function check(name: string, cond: boolean, detail = '') {
  if (cond) console.log(`  ✓ ${name}`)
  else { failures++; console.log(`  ✗ ${name} ${detail}`) }
}

// A synthetic event with minimal fields.
function ev(kind: string, epoch?: number): TrainingEvent {
  return { t: '2026-09-22T00:00:00.000Z', kind: kind, ...(epoch != null ? { epoch } : {}) }
}

// Events a real trainer writes during the final epochs of a run.
function trainingRun(terminalKind: string | null): TrainingEvent[] {
  const base: TrainingEvent[] = [
    ev('run.start'),
    ev('dataset.loaded', 0),
    ev('epoch.start', 4),
    ev('epoch.end', 4),
    ev('epoch.start', 5),
    ev('epoch.end', 5),
  ]
  return terminalKind ? [...base, ev(terminalKind)] : base
}

// ── 1. Parsing: order preserved, partial trailing line skipped ──
console.log('  [parseEventLines: order + partial-line tolerance]')
{
  const text = [JSON.stringify(ev('run.start')), JSON.stringify(ev('epoch.end', 0)), '{"t": "2026-09-22", "kind": "epoch.end", "epoch"'].join('\n')
  const parsed = parseEventLines(text)
  check('valid lines parsed in order', parsed.length === 2 && parsed[0].kind === 'run.start' && parsed[1].kind === 'epoch.end')
  check('partial trailing line skipped (SIGKILL mid-write)', !parsed.some((e) => e.kind === 'epoch.end' && (e.epoch as number | undefined) === undefined))
}

// ── 2. Terminal-state derivation on a healthy run ──
console.log('  [finalTerminal while running vs after terminal]')
{
  const active = trainingRun(null)
  check('no terminal while run is active', finalTerminal(active) === null)
  check('no stale trailing while active', !hasStaleTrailing(active))
  check('truncate is a no-op while active', truncateAtTerminal(active).length === active.length)

  const done = trainingRun('run.done')
  const final = finalTerminal(done)
  check('run.done is the final state', final?.kind === 'run.done')
  check('truncate keeps everything when terminal is last', truncateAtTerminal(done).length === done.length)
  check('no stale trailing when file is clean', !hasStaleTrailing(done))
}

// ── 3. The Phase-31 scenario: EPOCH 6 lands AFTER run.failed ──
console.log('  [RUNNING → EPOCH 5 → FAILED → EPOCH 6 (late event)]')
{
  const stream = [...trainingRun('run.failed'), ev('epoch.end', 6)]
  const final = finalTerminal(stream)
  check('run.failed is the final state (not overwritten by EPOCH 6)', final?.kind === 'run.failed')
  const cut = truncateAtTerminal(stream)
  check('late EPOCH 6 dropped from the timeline', cut.length === 7 && !cut.some((e) => e.epoch === 6))
  check('hasStaleTrailing detects the out-of-order line', hasStaleTrailing(stream))
  const lastEpoch = latestEvent(stream, 'epoch.end')
  check('latest epoch.end is 5 (not the stale 6)', (lastEpoch?.epoch as number | undefined) === 5)
}

// ── 4. Terminal duplicates: DONE then FAILED/cancelled trailing writes ──
console.log('  [first terminal wins over trailing duplicates]')
{
  const stream = [...trainingRun('run.done'), ev('run.cancelled')]
  check('run.done wins over a trailing run.cancelled', finalTerminal(stream)?.kind === 'run.done')
  check('trailing cancelled dropped', truncateAtTerminal(stream).length === 7)
  check('stale trailing flagged', hasStaleTrailing(stream))
}

// ── 5. CANCELLED (the Phase-30 winner) is equally final at the event level ──
console.log('  [run.cancelled is terminal; no epoch after it surfaces]')
{
  const stream = [...trainingRun('run.cancelled'), ev('epoch.end', 9)]
  check('run.cancelled is the final state', finalTerminal(stream)?.kind === 'run.cancelled')
  check('late epoch dropped', truncateAtTerminal(stream).length === 7)
  check('no run.done present', finalTerminal(stream)?.kind !== 'run.done')
}

// ── 6. parseFinalEvents: file text → truncated events in one step ──
console.log('  [parseFinalEvents end-to-end (raw text incl. garbage tail)]')
{
  const text = [
    JSON.stringify(ev('run.start')),
    JSON.stringify(ev('epoch.end', 2)),
    JSON.stringify(ev('run.failed')),
    JSON.stringify(ev('epoch.end', 3)), // stale
    '{"t": "2026-09-22", "kind": "epoch.end", "epoch":', // partial
    '',
  ].join('\n')
  const events = parseFinalEvents(text)
  check('3 valid, ordered events survive', events.length === 3 && events[1].epoch === 2)
  check('stale EPOCH 3 truncated', !events.some((e) => e.epoch === 3))
  check('terminal kinds all recognized', TERMINAL_EVENT_KINDS.length === 3 && TERMINAL_EVENT_KINDS.every((k) => ['run.done', 'run.failed', 'run.cancelled'].includes(k)))
}

// ── 7. latestWinsGuard: an older read landing late is dropped ──
console.log('  [stale-read guard: newest-STARTED read wins]')
{
  const applied: string[] = []
  const guard = latestWinsGuard<string, void>((v) => { applied.push(v) })

  // Controllable fake reads: (name, latency) → returns name after latency ms.
  const makeRead = (name: string, ms: number): (() => Promise<string>) =>
    () => new Promise((res) => setTimeout(() => res(name), ms))

  // Older read started first, resolves LAST → must be dropped.
  void guard(makeRead('read-A', 30))
  await guard(makeRead('read-B', 5))
  await new Promise((r) => setTimeout(r, 40))
  check('newer (read-B) applied', applied.length === 1 && applied[0] === 'read-B')
  check('older (read-A) late response dropped', applied.length === 1)

  // Sequential reads (each awaited before the next begins) both apply in order.
  const seq = latestWinsGuard<string, void>((v) => { applied.push(`seq:${v}`) })
  await seq(makeRead('first', 1))
  await seq(makeRead('second', 1))
  check('sequential reads apply in order', applied.length === 3 && applied[2] === 'seq:second')
}

// ── 8. Integration feel: modal flow — status goes terminal, a late read fires ──
console.log('  [modal flow: terminal run + late tail read is inert]')
{
  // Emulates RunDetailModal: reload() (newest) + a slow tailReload() (older)
  // whose attempt started BEFORE reload — the guard drops the tail's response.
  const snapshotA = () => Promise.resolve([JSON.stringify(ev('epoch.end', 5)), JSON.stringify(ev('run.done'))].join('\n'))
  const snapshotB_stale = () => new Promise<string>((res) => setTimeout(() => res(JSON.stringify(ev('epoch.end', 5))), 20))
  const applied: TrainingEvent[][] = []
  const guard = latestWinsGuard<string, void>((t) => { applied.push(parseFinalEvents(t)) })

  void guard(snapshotB_stale) // older read, in flight
  await guard(snapshotA)      // new read after status flip → applies
  await new Promise((r) => setTimeout(r, 40))
  check('only the newest snapshot applied', applied.length === 1)
  check('applied snapshot includes the final run.done', applied[0].some((e) => e.kind === 'run.done'))
}

console.log(failures === 0 ? '\n✓ all event-ordering checks passed' : `\n✗ ${failures} check(s) failed`)
process.exit(failures === 0 ? 0 : 1)