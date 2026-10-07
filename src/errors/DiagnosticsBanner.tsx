// Phase 49 — dismissible rose banner for unhandled / fire-and-forget errors.
//
// Mounted once in src/App.tsx (no business logic — just listens to the
// `useDiagnostics` store). Hidden when the store is empty; auto-collapses
// duplicates via the store's count field; per-entry close button; "Alle
// schließen" wipes the buffer.

import { useDiagnostics } from './report'

function formatTime(at: number): string {
  const d = new Date(at)
  return d.toLocaleTimeString()
}

export default function DiagnosticsBanner() {
  const entries = useDiagnostics((s) => s.entries)
  const dismiss = useDiagnostics((s) => s.dismiss)
  const clear = useDiagnostics((s) => s.clear)
  if (entries.length === 0) return null
  return (
    <div
      role="status"
      aria-live="polite"
      className="pointer-events-none fixed left-1/2 top-12 z-50 -translate-x-1/2"
    >
      <div className="pointer-events-auto flex max-w-[min(720px,90vw)] flex-col gap-1 rounded border border-rose-800/70 bg-rose-950/95 px-3 py-2 text-[11px] text-rose-100 shadow-lg">
        <div className="flex items-center justify-between gap-3">
          <span className="font-semibold uppercase tracking-wide">
            Unbehandelte Fehler ({entries.length})
          </span>
          <button
            type="button"
            onClick={() => clear()}
            className="rounded px-1 text-[10px] text-rose-300 hover:bg-rose-900/60 hover:text-rose-100"
            title="Alle Einträge verwerfen"
          >
            Alle schließen
          </button>
        </div>
        <ul className="flex flex-col gap-1">
          {entries.map((d) => (
            <li
              key={d.id}
              className="flex items-start gap-2 rounded border border-rose-900/60 bg-rose-900/40 px-2 py-1"
            >
              <div className="min-w-0 flex-1">
                <div className="truncate">
                  <span className="font-mono text-rose-200">{d.context}</span>
                  {' — '}
                  <span>{d.message}</span>
                  {d.count > 1 && (
                    <span className="ml-2 rounded bg-rose-800/70 px-1 text-[10px] text-rose-200">
                      ×{d.count}
                    </span>
                  )}
                </div>
                <div className="text-[10px] text-rose-400">
                  {formatTime(d.at)} · Details in der Konsole
                </div>
              </div>
              <button
                type="button"
                onClick={() => dismiss(d.id)}
                className="shrink-0 rounded px-1 text-rose-300 hover:bg-rose-800/60 hover:text-rose-100"
                title="Eintrag schließen"
                aria-label="Eintrag schließen"
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
