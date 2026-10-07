// `${kind}:${relpath}` already loaded this session — so switching canvas modes
// doesn't reload (and clobber unsaved edits). Module-scoped on purpose.
// Kept out of CanvasFileGate.tsx so that file only exports components (React
// Fast Refresh).

import type { CanvasKind } from './store'

const hydrated = new Set<string>()

/** Mark a (kind, relpath) as already loaded so the gate won't reload it — used
 *  when we bind a file FROM the current store (the store already holds it). */
export function markCanvasHydrated(kind: CanvasKind, relpath: string): void {
  hydrated.add(`${kind}:${relpath}`)
}

export function isCanvasHydrated(kind: CanvasKind, relpath: string): boolean {
  return hydrated.has(`${kind}:${relpath}`)
}
