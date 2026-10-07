// Phase 49 — install the two global error listeners once at startup.
//
// `unhandledrejection` catches a promise rejection that has no .catch()
// and no await chain reaching it. `error` catches a synchronous throw
// that escapes the React ErrorBoundary (e.g. inside a setTimeout or an
// event listener that bypasses React). Both feed the same `useDiagnostics`
// store so the banner can show them; without this module, those errors
// vanish into the console.
//
// Import this module exactly ONCE (src/main.tsx, after the other imports).

import { useDiagnostics } from './report'

let installed = false

export function installGlobalErrorHandlers(): void {
  if (installed) return
  installed = true
  if (typeof window === 'undefined') return

  const push = useDiagnostics.getState().push

  // Benign, expected noise that must not raise a user-facing banner:
  //  - an intentionally aborted request/operation (AbortController) rejects with AbortError;
  //  - browsers fire `error` for "ResizeObserver loop completed with undelivered notifications"
  //    during normal layout (React Flow / resizable panels) — no state is affected.
  const isBenign = (reason: unknown, message?: string): boolean => {
    if (reason instanceof DOMException && reason.name === 'AbortError') return true
    const text = message ?? (reason instanceof Error ? reason.message : '')
    return typeof text === 'string' && text.startsWith('ResizeObserver loop')
  }

  window.addEventListener('unhandledrejection', (ev) => {
    // ev.reason is `unknown` per the spec — we feed the same store entry
    // format as caught handlers.
    if (isBenign(ev.reason)) {
      ev.preventDefault()
      return
    }
    push('window.unhandledrejection', ev.reason)
    // prevent the default browser console warning (we already logged in
    // the store; a second noisy line is unhelpful)
    ev.preventDefault()
  })

  window.addEventListener('error', (ev) => {
    // ev.error is `unknown` too. `ev.message` + filename/line/column are
    // useful when the thrown value had no message of its own (e.g. an
    // object throw with no `.toString()`). Keep both.
    if (isBenign(ev.error, ev.message)) return
    const underlying = ev.error ?? `${ev.message} (${ev.filename}:${ev.lineno}:${ev.colno})`
    push('window.error', underlying)
  })
}

installGlobalErrorHandlers()
