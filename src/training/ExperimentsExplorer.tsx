import { useEffect, useState } from 'react'

import { isTauri } from '../workspace/tauri-fs'
import { useConnectionsStore, remotePython, type RemoteSshConnection } from '../connections/store'
import { useTrainingStore } from './store'
import { isTerminal } from './types'
import { confirmDialog } from '../ui/confirm'
import StatusPill from './StatusPill'

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

  useEffect(() => {
    if (isTauri()) void refresh()
  }, [refresh, currentId])

  if (!isTauri()) {
    return (
      <div className="p-3 text-[11px] text-[#7a8088]">
        Training braucht Tauri (echtes Dateisystem). Im Browser-Dev nicht verfügbar.
      </div>
    )
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 items-center gap-2 border-b border-[#1f2429] px-2 py-1.5">
        <span className="flex-1 text-[11px] uppercase tracking-wide text-[#7a8088]">Runs</span>
        <button
          onClick={() => void refresh()}
          className="rounded px-1.5 py-0.5 text-[11px] text-[#7a8088] hover:bg-[#1a1e22] hover:text-[#e6e8eb]"
          title="Refresh"
        >
          ↻
        </button>
        <button
          onClick={() => openNewRun()}
          className="rounded bg-[#13344f] px-2 py-0.5 text-[11px] text-[#6ab7ff] hover:bg-[#184466]"
          title="Neuen Trainings-Run starten"
        >
          + Run
        </button>
      </div>

      {remoteConn && <RemoteConfigStrip conn={remoteConn} />}

      {compareIds.length > 0 && (
        <div className="flex items-center gap-2 border-b border-[#1f2429] bg-[#0f1419] px-3 py-1.5 text-[11px]">
          <span className="text-[#9aa1a8]">{compareIds.length} ausgewählt</span>
          <button
            onClick={openCompare}
            disabled={compareIds.length < 2}
            className="ml-auto rounded bg-[#13344f] px-2 py-0.5 text-[#6ab7ff] hover:bg-[#184466] disabled:opacity-40"
          >
            Vergleichen
          </button>
          <button onClick={clearCompare} className="rounded px-1.5 py-0.5 text-[#7a8088] hover:bg-[#1a1e22] hover:text-[#e6e8eb]">
            ×
          </button>
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-auto">
        {error && <div className="px-3 py-2 text-[11px] text-[#ff7a85]">{error}</div>}
        {!error && runs.length === 0 && (
          <div className="px-3 py-3 text-[11px] text-[#7a8088]">
            {loading ? 'lade…' : 'Noch keine Runs. „+ Run" startet den ersten.'}
          </div>
        )}
        {runs.map((r) => {
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
                className="shrink-0 accent-[#6ab7ff]"
              />
              <button
                onClick={() => select(r.run_id)}
                className="flex min-w-0 flex-1 flex-col gap-0.5 py-2 text-left"
              >
                <div className="flex items-center gap-2">
                  <StatusPill status={r.status} alive={r.alive} />
                  <span className="flex-1 truncate text-[12px] text-[#e6e8eb]">
                    {r.run_label || r.run_id}
                  </span>
                </div>
                <div className="flex items-center gap-2 text-[10px] text-[#7a8088]">
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
                  className="shrink-0 rounded px-1 py-0.5 text-[#7a8088] opacity-0 hover:bg-[#1a1e22] hover:text-[#ff7a85] group-hover:opacity-100"
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

// Remote backend strip: confirms which host runs the training and lets the user
// point the run at the right python env (the #1 remote-direct footgun). Edits
// persist on the connection via updateRemote.
function RemoteConfigStrip({ conn }: { conn: RemoteSshConnection }) {
  const updateRemote = useConnectionsStore((s) => s.updateRemote)
  const [draft, setDraft] = useState(remotePython(conn))

  // resync when switching between remote connections
  useEffect(() => { setDraft(remotePython(conn)) }, [conn.id]) // eslint-disable-line react-hooks/exhaustive-deps

  const commit = () => {
    const v = draft.trim()
    if (v !== remotePython(conn)) updateRemote(conn.id, { python: v || undefined })
  }

  return (
    <div className="flex flex-col gap-1 border-b border-[#1f2429] bg-[#0f1419] px-3 py-2 text-[11px]">
      <div className="flex items-center gap-1.5 text-[#7a8088]">
        <span className="text-[#6ab7ff]">remote</span>
        <span className="truncate text-[#9aa1a8]">{conn.alias}:{conn.root}</span>
      </div>
      <label className="flex items-center gap-1.5">
        <span className="shrink-0 text-[#7a8088]">Python</span>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur() }}
          placeholder="python"
          spellCheck={false}
          className="min-w-0 flex-1 rounded border border-[#1f2429] bg-[#0b0e11] px-1.5 py-0.5 font-mono text-[10px] text-[#e6e8eb] focus:border-[#6ab7ff] focus:outline-none"
          title="Pfad zum python mit torch (z.B. ~/miniconda3/envs/ml/bin/python)"
        />
      </label>
    </div>
  )
}
