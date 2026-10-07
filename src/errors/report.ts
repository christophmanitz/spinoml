// Phase 49 — shared error reporting for the frontend.
//
// One helper (`reportError`) for explicit catches + a `fireAndForget` wrapper
// for the unavoidable promise-call statements (event handlers, store
// actions). Both push into the small zustand `useDiagnostics` store, which
// `DiagnosticsBanner.tsx` renders and which `globalHandlers.ts` also feeds
// from `window.onerror` / `unhandledrejection`.
//
// The store is in-memory only — it is deliberately NOT persisted so a single
// transient glitch doesn't haunt the next session. Phase 50 already
// establishes the principle that an explicit FAILED/UNKNOWN state is
// preferable to a default that looks like success; this module is the
// companion for asynchronous paths that the silent-catch audit doesn't see.

import { create } from 'zustand'

/** A single diagnostic record. `id` lets the banner dedupe / collapse. */
export type Diagnostic = {
  id: string
  /** where the error came from (component, store, sidecar name, …) */
  context: string
  /** short message — what the user sees */
  message: string
  /** original error object, kept for the console */
  error: unknown
  /** wall-clock ms — sort key + user-facing timestamp */
  at: number
  /** how many times the same (context, message) pair has re-occurred */
  count: number
}

type State = {
  /** ring buffer — last 20, oldest dropped */
  entries: Diagnostic[]
  push: (context: string, error: unknown) => Diagnostic
  clear: () => void
  dismiss: (id: string) => void
}

const MAX_ENTRIES = 20

function key(d: { context: string; message: string }): string {
  return `${d.context}\u0000${d.message}`
}

function safeMessage(err: unknown): string {
  if (err instanceof Error) return err.message
  if (typeof err === 'string') return err
  try {
    return JSON.stringify(err)
  } catch {
    return String(err)
  }
}

function safeStack(err: unknown): string {
  if (err instanceof Error && err.stack) return err.stack
  return ''
}

let counter = 0
function newId(): string {
  counter += 1
  return `${Date.now().toString(36)}-${counter.toString(36)}`
}

export const useDiagnostics = create<State>((set, get) => ({
  entries: [],
  push: (context, error) => {
    const message = safeMessage(error)
    const stack = safeStack(error)
    const dKey = key({ context, message })
    const existing = get().entries.find((e) => key(e) === dKey)
    if (existing) {
      // collapse duplicates: bump count, refresh the timestamp, keep the
      // newest error object so the console gets the latest stack
      const updated: Diagnostic = {
        ...existing,
        count: existing.count + 1,
        at: Date.now(),
        error: stack ? new Error(message, { cause: error }) : error,
      }
      set({
        entries: [
          updated,
          ...get().entries.filter((e) => e.id !== existing.id),
        ],
      })
      console.warn(`[${context}] (x${updated.count}) ${message}`, error)
      return updated
    }
    const entry: Diagnostic = {
      id: newId(),
      context,
      message,
      error,
      at: Date.now(),
      count: 1,
    }
    const next = [entry, ...get().entries]
    if (next.length > MAX_ENTRIES) next.length = MAX_ENTRIES
    set({ entries: next })
    console.warn(`[${context}] ${message}`, error)
    return entry
  },
  clear: () => set({ entries: [] }),
  dismiss: (id) => set({ entries: get().entries.filter((e) => e.id !== id) }),
}))

/** Record a caught error and return the same error so callers can
 *  re-throw if they want. Safe to call from any context. */
export function reportError(context: string, error: unknown): unknown {
  useDiagnostics.getState().push(context, error)
  return error
}

/** Wrap a fire-and-forget promise so its rejection is captured instead of
 *  vanishing into the runtime. The returned promise resolves to `undefined`
 *  once the inner one has settled (either way), so callers can keep using
 *  `void fireAndForget(…)` without changing the call sites.
 *
 *  Use ONLY when you genuinely cannot await or .catch() at the call site
 *  (e.g. the React `onClick={() => …}` shape). Anywhere else, await or
 *  .catch() inline. */
export function fireAndForget(context: string, p: Promise<unknown>): Promise<void> {
  return p.then(
    () => undefined,
    (err: unknown) => {
      useDiagnostics.getState().push(context, err)
    },
  )
}

/** Coerce an unknown error value to a human-readable string. Use this
 *  anywhere a `catch (e)` block reads `e.message` — non-Error throws
 *  (strings, plain objects, `throw "boom"`) crash with `Cannot read
 *  properties of undefined (reading 'message')`. */
export function errMessage(e: unknown): string {
  return safeMessage(e)
}

/** True iff `n` is a finite number. Use for external numerics parsed via
 *  `Number()` / `parseInt` that feed shapes, ids, epochs, etc. — a `NaN`
 *  would silently propagate into a graph, a run summary or a chart. */
export function isFiniteNumber(n: unknown): n is number {
  return typeof n === 'number' && Number.isFinite(n)
}
