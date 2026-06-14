import { useEffect } from 'react'

import { isTauri } from '../workspace/tauri-fs'
import { useConnectionsStore } from '../connections/store'
import { useTrainingStore } from './store'
import { remoteTrainingBlocked, REMOTE_TRAINING_MSG } from './backend'
import StatusPill from './StatusPill'

export default function ExperimentsExplorer() {
  const runs = useTrainingStore((s) => s.runs)
  const loading = useTrainingStore((s) => s.listLoading)
  const error = useTrainingStore((s) => s.listError)
  const selectedRunId = useTrainingStore((s) => s.selectedRunId)
  const refresh = useTrainingStore((s) => s.refresh)
  const select = useTrainingStore((s) => s.select)
  const openNewRun = useTrainingStore((s) => s.openNewRun)
  // Re-refresh when the active connection changes (local↔remote).
  const currentId = useConnectionsStore((s) => s.currentId)

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
          onClick={openNewRun}
          disabled={remoteTrainingBlocked()}
          className="rounded bg-[#13344f] px-2 py-0.5 text-[11px] text-[#6ab7ff] hover:bg-[#184466] disabled:cursor-not-allowed disabled:opacity-40"
          title={remoteTrainingBlocked() ? REMOTE_TRAINING_MSG : 'Neuen Trainings-Run starten'}
        >
          + Run
        </button>
      </div>

      {remoteTrainingBlocked() && (
        <div className="border-b border-[#1f2429] bg-[#1a1410] px-3 py-2 text-[11px] text-[#e6c34a]">
          {REMOTE_TRAINING_MSG}
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
          return (
            <button
              key={r.run_id}
              onClick={() => select(r.run_id)}
              className={`flex w-full flex-col gap-0.5 border-b border-[#171b1f] px-3 py-2 text-left hover:bg-[#14181c] ${
                selectedRunId === r.run_id ? 'bg-[#14181c]' : ''
              }`}
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
          )
        })}
      </div>
    </div>
  )
}
