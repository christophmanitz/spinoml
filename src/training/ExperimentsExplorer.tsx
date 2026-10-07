import { useEffect, useState } from 'react'

import { isTauri } from '../workspace/tauri-fs'
import { useConnectionsStore, remotePython, type RemoteSshConnection } from '../connections/store'
import { useTrainingStore } from './store'
import { isTerminal } from './types'
import { confirmDialog } from '../ui/confirm'
import StatusPill from './StatusPill'
import { fireAndForget } from '../errors/report'

export default function ExperimentsExplorer() {
  const runs = useTrainingStore((s) => s.runs)
  const loading = useTrainingStore((s) => s.listLoading)
  const error = useTrainingStore((s) => s.listError)
  const selectedRunId = useTrainingStore((s) => s.selectedRunId)
  const compareIds = useTrainingStore((s) => s.compareIds)
  const refresh = useTrainingStore((s) => s.refresh)
  const select = useTrainingStore((s) => s.select)
  const toggleCompare = useTrainingStore((s) => s.toggleCompare)
  const clearCompare = useTrainingStore((s) => s.clearCompare)
  const openCompare = useTrainingStore((s) => s.openCompare)
  const openNewRun = useTrainingStore((s) => s.openNewRun)
  const deleteRun = useTrainingStore((s) => s.deleteRun)
  // Re-refresh when the active connection changes (local↔remote).
  const currentId = useConnectionsStore((s) => s.currentId)
  const saved = useConnectionsStore((s) => s.saved)
  const remoteConn = saved.find((c) => c.id === currentId) ?? null

  // Status filter (client-side). 'all' = no filter.
  const [statusFilter, setStatusFilter] = useState<string>('all')

  useEffect(() => {
    if (isTauri()) void fireAndForget('refresh', refresh())
  }, [refresh, currentId])

  if (!isTauri()) {
    return (
      <div className="p-3 text-[11px] text-[#6f767e]">
        Training braucht Tauri (echtes Dateisystem). Im Browser-Dev nicht verfügbar.
      </div>
    )
  }

  // Per-status counts + the statuses actually present, in a canonical order so
  // the filter chips read running→queued→done→failed→cancelled (others appended).
  const counts: Record<string, number> = {}
  for (const r of runs) counts[r.status] = (counts[r.status] ?? 0) + 1
  const STATUS_ORDER = ['running', 'queued', 'done', 'failed', 'cancelled']
  const present = Object.keys(counts).sort((a, b) => {
    const ia = STATUS_ORDER.indexOf(a), ib = STATUS_ORDER.indexOf(b)
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib)
  })
  // If the active filter's status vanished after a refresh, fall back to all.
  const effFilter = statusFilter !== 'all' && !counts[statusFilter] ? 'all' : statusFilter
  const visibleRuns = effFilter === 'all' ? runs : runs.filter((r) => r.status === effFilter)

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 items-center gap-2 border-b border-[#1f2429] px-2 py-1.5">
        <span className="flex-1 text-[11px] uppercase tracking-wide text-[#6f767e]">Runs</span>
        <button
          onClick={() => void fireAndForget('refresh', refresh())}
          className="rounded px-1.5 py-0.5 text-[11px] text-[#6f767e] hover:bg-[#1a1e22] hover:text-[#e6e8eb]"
          title="Refresh"
        >
          ↻
        </button>
        <button
          onClick={() => openNewRun()}
          className="rounded bg-[var(--accent-sel)] px-2 py-0.5 text-[11px] text-[var(--accent)] hover:bg-[var(--accent-sel-hover)]"
          title="Neuen Trainings-Run starten"
        >
          + Run
        </button>
      </div>

      {remoteConn && <RemoteConfigStrip conn={remoteConn} />}

      {present.length > 1 && (
        <div className="flex shrink-0 flex-wrap items-center gap-1 border-b border-[#1f2429] px-2 py-1.5 text-[10px]">
          <FilterChip label="Alle" count={runs.length} active={effFilter === 'all'} onClick={() => setStatusFilter('all')} />
          {present.map((s) => (
            <FilterChip key={s} label={s} count={counts[s]} active={effFilter === s} onClick={() => setStatusFilter(s)} />
          ))}
        </div>
      )}

      {compareIds.length > 0 && (
        <div className="flex items-center gap-2 border-b border-[#1f2429] bg-[#0f1419] px-3 py-1.5 text-[11px]">
          <span className="text-[#9aa1a8]">{compareIds.length} ausgewählt</span>
          <button
            onClick={openCompare}
            disabled={compareIds.length < 2}
            className="ml-auto rounded bg-[var(--accent-sel)] px-2 py-0.5 text-[var(--accent)] hover:bg-[var(--accent-sel-hover)] disabled:opacity-40"
          >
            Vergleichen
          </button>
          <button onClick={clearCompare} className="rounded px-1.5 py-0.5 text-[#6f767e] hover:bg-[#1a1e22] hover:text-[#e6e8eb]">
            ×
          </button>
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-auto">
        {error && <div className="px-3 py-2 text-[11px] text-[#ff7a85]">{error}</div>}
        {!error && runs.length === 0 && (
          <div className="px-3 py-3 text-[11px] text-[#6f767e]">
            {loading ? 'lade…' : 'Noch keine Runs. „+ Run" startet den ersten.'}
          </div>
        )}
        {!error && runs.length > 0 && visibleRuns.length === 0 && (
          <div className="px-3 py-3 text-[11px] text-[#6f767e]">Keine Runs mit Status „{effFilter}".</div>
        )}
        {visibleRuns.map((r) => {
          const model = r.model_path.split('/').pop() ?? r.model_path
          const checked = compareIds.includes(r.run_id)
          return (
            <div
              key={r.run_id}
              className={`group flex items-center gap-2 border-b border-[#171b1f] pl-2 pr-3 hover:bg-[#14181c] ${
                selectedRunId === r.run_id ? 'bg-[#14181c]' : ''
              }`}
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={() => toggleCompare(r.run_id)}
                title="Für Vergleich auswählen"
                className="shrink-0 accent-[var(--accent)]"
              />
              <button
                onClick={() => select(r.run_id)}
                className="flex min-w-0 flex-1 flex-col gap-0.5 py-2 text-left"
              >
                <div className="flex items-center gap-2">
                  <StatusPill status={r.status} alive={r.alive} />
                  {r.eval_only && (
                    <span className="shrink-0 rounded bg-[var(--accent-sel)] px-1 py-0.5 text-[9px] font-semibold text-[var(--accent)]" title="Externe Validierung">VAL</span>
                  )}
                  <span className="flex-1 truncate text-[12px] text-[#e6e8eb]">
                    {r.run_label || r.run_id}
                  </span>
                </div>
                <div className="flex items-center gap-2 text-[10px] text-[#6f767e]">
                  <span className="truncate">{model}</span>
                  {r.best_val_loss != null && (
                    <span className="ml-auto shrink-0 text-[#5fd39a]">val {r.best_val_loss.toFixed(4)}</span>
                  )}
                </div>
              </button>
              {isTerminal(r.status) && (
                <button
                  onClick={async (e) => {
                    e.stopPropagation()
                    if (await confirmDialog(`Run „${r.run_label || r.run_id}" löschen?`)) await deleteRun(r.run_id)
                  }}
                  title="Run löschen"
                  className="shrink-0 rounded px-1 py-0.5 text-[#6f767e] opacity-0 hover:bg-[#1a1e22] hover:text-[#ff7a85] group-hover:opacity-100"
                >
                  ×
                </button>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

// A single status-filter chip: label + count, highlighted when active.
function FilterChip({ label, count, active, onClick }: { label: string; count: number; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`rounded px-1.5 py-0.5 ${
        active ? 'bg-[var(--accent-sel)] text-[var(--accent)]' : 'text-[#6f767e] hover:bg-[#1a1e22] hover:text-[#e6e8eb]'
      }`}
    >
      {label} <span className="opacity-60">{count}</span>
    </button>
  )
}

// Remote backend strip: confirms which host runs the training and lets the user
// point the run at the right python env (the #1 remote-direct footgun). Edits
// persist on the connection via updateRemote.
function RemoteConfigStrip({ conn }: { conn: RemoteSshConnection }) {
  const updateRemote = useConnectionsStore((s) => s.updateRemote)
  const [draft, setDraft] = useState(remotePython(conn))
  // Resync the draft when switching between remote connections (conn.id only —
  // conn is a stable snapshot for the lifetime of the modal). Mirrors the
  // previous effect but in render so no set-state-in-effect.
  const [lastConnId, setLastConnId] = useState(conn.id)
  if (conn.id !== lastConnId) {
    setLastConnId(conn.id)
    setDraft(remotePython(conn))
  }

  const commit = () => {
    const v = draft.trim()
    if (v !== remotePython(conn)) updateRemote(conn.id, { python: v || undefined })
  }

  return (
    <div className="flex flex-col gap-1 border-b border-[#1f2429] bg-[#0f1419] px-3 py-2 text-[11px]">
      <div className="flex items-center gap-1.5 text-[#6f767e]">
        <span className="text-[var(--accent)]">remote</span>
        <span className="truncate text-[#9aa1a8]">{conn.alias}:{conn.root}</span>
      </div>
      <label className="flex items-center gap-1.5">
        <span className="shrink-0 text-[#6f767e]">Python</span>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur() }}
          placeholder="python"
          spellCheck={false}
          className="min-w-0 flex-1 rounded border border-[#1f2429] bg-[#0b0e11] px-1.5 py-0.5 font-mono text-[10px] text-[#e6e8eb] focus:border-[var(--accent)] focus:outline-none"
          title="Pfad zum python mit torch (z.B. ~/miniconda3/envs/ml/bin/python)"
        />
      </label>
    </div>
  )
}
