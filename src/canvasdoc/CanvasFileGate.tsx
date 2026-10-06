// Wraps a canvas's React-Flow view. When the canvas is bound to a file it shows a
// thin header (which file + save status); when it isn't, it shows a chooser so the
// user opens an existing file or creates a new one — a canvas never displays an
// orphan, unsaved graph. A bound file is loaded into the store on mount (so a
// persisted binding reopens its graph on launch). Inactive (no Tauri workspace) →
// passthrough, so browser dev still works.

import { useEffect, useState } from 'react'
import { isTauri } from '../workspace/tauri-fs'
import { useWorkspaceStore } from '../workspace/store'
import { useCanvasDocStore, type CanvasDocAdapter, type CanvasKind } from './store'

// `${kind}:${relpath}` already loaded this session — so switching canvas modes
// doesn't reload (and clobber unsaved edits). Module-scoped on purpose.
const hydrated = new Set<string>()

/** Mark a (kind, relpath) as already loaded so the gate won't reload it — used
 *  when we bind a file FROM the current store (the store already holds it). */
export function markCanvasHydrated(kind: CanvasKind, relpath: string): void {
  hydrated.add(`${kind}:${relpath}`)
}

export default function CanvasFileGate({ adapter, children }: { adapter: CanvasDocAdapter; children: React.ReactNode }) {
  const doc = useCanvasDocStore((s) => s.docs[adapter.kind])
  const setBound = useCanvasDocStore((s) => s.setBound)
  const workspaceRoot = useWorkspaceStore((s) => s.workspaceRoot)
  const gated = isTauri() && !!workspaceRoot
  const [loading, setLoading] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const relpath = doc.relpath

  useEffect(() => {
    // External owners (workspace .spinoml) load the file themselves — don't reload.
    if (!gated || !relpath || adapter.external) return
    const key = `${adapter.kind}:${relpath}`
    if (hydrated.has(key)) return
    let cancelled = false
    setLoading(true)
    setLoadError(null)
    adapter.open(relpath)
      .then(() => { hydrated.add(key); if (!cancelled) setLoading(false) })
      .catch((e) => {
        // A bound file that no longer parses must be reported, not silently
        // dropped back to the chooser (which would imply "no file" rather than
        // "this file is unreadable").
        if (!cancelled) { setLoading(false); setLoadError(`${relpath}: ${e instanceof Error ? e.message : String(e)}`) }
      })
    return () => { cancelled = true }
  }, [gated, relpath, adapter, setBound])

  if (!gated) return <div className="flex h-full w-full flex-col">{children}</div>
  if (loadError) {
    return (
      <div className="flex h-full w-full flex-col">
        <div className="flex items-center gap-2 border-b border-rose-900/40 bg-rose-950/30 px-3 py-2 text-[11px] text-rose-300">
          <span>Datei konnte nicht geladen werden — {loadError}</span>
          <button
            className="ml-auto rounded border border-rose-900/60 px-1.5 py-0.5 text-[10px] hover:bg-rose-900/30"
            onClick={() => { setLoadError(null); setBound(adapter.kind, null) }}
          >Andere Datei wählen</button>
        </div>
        <div className="flex min-h-0 flex-1 items-center justify-center text-[12px] text-[#6f767e]">
          Der {adapter.label}-Graph wird nicht angezeigt, weil die Datei unlesbar ist.
        </div>
      </div>
    )
  }
  if (!relpath) return <FileChooser adapter={adapter} />
  if (loading) {
    return (
      <div className="flex h-full w-full items-center justify-center text-[12px] text-[#6f767e]">
        lade {relpath.split('/').pop()}…
      </div>
    )
  }
  return (
    <div className="flex h-full w-full flex-col">
      <FileHeader adapter={adapter} relpath={relpath} status={doc.status} error={doc.error} />
      <div className="relative min-h-0 flex-1">{children}</div>
    </div>
  )
}

function FileHeader({
  adapter, relpath, status, error,
}: {
  adapter: CanvasDocAdapter
  relpath: string
  status: 'idle' | 'saving' | 'saved' | 'error'
  error: string | null
}) {
  const setBound = useCanvasDocStore((s) => s.setBound)
  const base = relpath.split('/').pop() ?? relpath
  const statusText =
    status === 'saving' ? '… speichert' : status === 'saved' ? '✓ gespeichert' : status === 'error' ? `⚠ ${error ?? 'Fehler'}` : ''
  const statusColor = status === 'error' ? 'text-rose-400' : status === 'saving' ? 'text-[#6f767e]' : 'text-[#5fd39a]'
  return (
    <div className="flex h-7 shrink-0 items-center gap-2 border-b border-[#1f2429] bg-[#0e1216] px-3 text-[11px]">
      <span className="text-[var(--accent)]">▦</span>
      <span className="font-medium text-[#e6e8eb]" title={relpath}>{base}</span>
      <span className="truncate text-[10px] text-[#5a6068]">{relpath}</span>
      <span className={`ml-auto text-[10px] ${statusColor}`}>{statusText}</span>
      {adapter.save && (
        <button
          className="rounded border border-[#1f2429] px-1.5 py-0.5 text-[10px] text-[#9aa1a8] hover:border-[#3a4148] hover:text-[#e6e8eb]"
          onClick={() => void adapter.save!()}
          title="Jetzt speichern"
        >Speichern</button>
      )}
      <button
        className="rounded border border-[#1f2429] px-1.5 py-0.5 text-[10px] text-[#9aa1a8] hover:border-[#3a4148] hover:text-[#e6e8eb]"
        onClick={() => { if (adapter.unbind) adapter.unbind(); else setBound(adapter.kind, null) }}
        title="Andere Datei öffnen oder neue anlegen"
      >Datei wechseln</button>
    </div>
  )
}

function FileChooser({ adapter }: { adapter: CanvasDocAdapter }) {
  const setBound = useCanvasDocStore((s) => s.setBound)
  const [list, setList] = useState<{ relpath: string; name: string }[]>([])
  const [listErr, setListErr] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    void adapter.list()
      .then((l) => { if (!cancelled) { setList(l); setListErr(null) } })
      .catch((e) => {
        // Listing failed (unlike a genuinely empty directory): say so instead of
        // rendering "no files", which the user would read as "nothing exists".
        if (cancelled) return
        setList([])
        setListErr(e instanceof Error ? e.message : String(e))
      })
    return () => { cancelled = true }
  }, [adapter])

  const open = async (relpath: string) => {
    setBusy(true); setErr(null)
    try {
      await adapter.open(relpath)
      hydrated.add(`${adapter.kind}:${relpath}`)
      setBound(adapter.kind, relpath)
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false) }
  }
  const create = async () => {
    const trimmed = name.trim()
    if (!trimmed) return
    setBusy(true); setErr(null)
    try {
      const rel = await adapter.create(trimmed)
      hydrated.add(`${adapter.kind}:${rel}`)
      setBound(adapter.kind, rel)
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)) }
    finally { setBusy(false) }
  }

  return (
    <div className="flex h-full w-full items-center justify-center bg-[#0a0c0f] p-6">
      <div className="w-[440px] max-w-full rounded-lg border border-[#1f2429] bg-[#0e1216] p-5 shadow-xl">
        <div className="mb-1 text-sm font-medium text-[#e6e8eb]">Welche {adapter.label}-Datei?</div>
        <div className="mb-4 text-[11px] text-[#6f767e]">
          Dieser Canvas spiegelt immer eine <code className="text-[#9aa1a8]">{adapter.ext}</code>-Datei.
          Öffne eine vorhandene oder lege eine neue an — so gehört der Graph immer zu einer gespeicherten Datei.
        </div>

        <div className="mb-2 text-[10px] uppercase tracking-wide text-[#6f767e]">Vorhandene</div>
        <div className="mb-4 max-h-44 space-y-1 overflow-auto">
          {list.length === 0 && !listErr && <div className="text-[11px] text-[#5a6068]">— keine {adapter.ext}-Dateien —</div>}
          {listErr && <div className="text-[11px] text-rose-400">Liste konnte nicht geladen werden: {listErr}</div>}
          {list.map((file) => (
            <button
              key={file.relpath}
              disabled={busy}
              onClick={() => void open(file.relpath)}
              className="flex w-full items-center gap-2 rounded border border-[#1f2429] bg-[#13171b] px-2 py-1.5 text-left text-[12px] text-[#e6e8eb] hover:border-[#3a4148] hover:bg-[#181d22] disabled:opacity-50"
            >
              <span className="text-[var(--accent)]">▦</span>
              <span className="flex-1">{file.name}</span>
              <span className="text-[10px] text-[#5a6068]">{file.relpath}</span>
            </button>
          ))}
        </div>

        <div className="mb-2 text-[10px] uppercase tracking-wide text-[#6f767e]">Neu anlegen</div>
        <div className="flex gap-1">
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') void create() }}
            placeholder={`name${adapter.ext}`}
            className="flex-1 rounded border border-[#1f2429] bg-[#0b0e11] px-2 py-1.5 text-[12px] text-[#e6e8eb] focus:border-[var(--accent)] focus:outline-none"
          />
          <button
            disabled={busy || !name.trim()}
            onClick={() => void create()}
            className="rounded bg-[var(--accent-sel)] px-3 py-1.5 text-[12px] text-[var(--accent)] hover:bg-[var(--accent-sel-hover)] disabled:opacity-40"
          >Erstellen</button>
        </div>
        {err && <div className="mt-2 text-[11px] text-rose-400">{err}</div>}
      </div>
    </div>
  )
}
