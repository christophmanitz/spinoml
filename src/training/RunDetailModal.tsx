import { useCallback, useEffect, useState, type ReactNode } from 'react'

import { training } from './backend'
import { useTrainingStore } from './store'
import { type RunConfig, type TrainingEvent, RUNNING_STATES } from './types'
import StatusPill from './StatusPill'
import LineChart from './charts/LineChart'
import { parseEventLines, lossSeries, lrSeries, metricSeries } from './charts/series'
import { runConfigToTrainingSnapshot } from './graph/fromRun'
import { useTrainingGraphStore } from './graph/store'
import { useViewModeStore } from './graph/viewMode'
import { getCurrentConnection } from '../connections/store'
import { confirmDialog } from '../ui/confirm'

type Tab = 'overview' | 'charts' | 'events' | 'logs' | 'script'

export default function RunDetailModal({ runId }: { runId: string }) {
  const close = useTrainingStore((s) => s.select)
  const stopRun = useTrainingStore((s) => s.stopRun)
  const deleteRun = useTrainingStore((s) => s.deleteRun)
  const summary = useTrainingStore((s) => s.runs.find((r) => r.run_id === runId))

  const [tab, setTab] = useState<Tab>('overview')
  const [events, setEvents] = useState<TrainingEvent[]>([])
  const [runJson, setRunJson] = useState('')
  const [trainPy, setTrainPy] = useState('')
  const [stdout, setStdout] = useState('')
  const [stderr, setStderr] = useState('')
  const [busy, setBusy] = useState(false)
  const [lossLog, setLossLog] = useState(false)
  const [trainSbatch, setTrainSbatch] = useState('')

  const status = summary?.status ?? 'unknown'
  const active = RUNNING_STATES.has(status) || (summary?.alive ?? false)

  // Backend (direct vs SLURM) is frozen in run.json.
  const backend = (() => {
    try { return (JSON.parse(runJson) as RunConfig).backend } catch { return null }
  })()
  const isSlurm = backend?.kind === 'slurm'

  // Rebuild this run's training graph onto the canvas, even if its .mltrain was
  // never saved — run.json carries the full frozen config.
  const openOnCanvas = () => {
    if (!runJson) return
    try {
      const config = JSON.parse(runJson) as RunConfig
      const snapshot = runConfigToTrainingSnapshot(config)
      useTrainingGraphStore.getState().loadSnapshot(snapshot)
      useViewModeStore.getState().setMode('training')
      close(null)
    } catch { /* run.json not ready / malformed */ }
  }

  // Full read — used on open + on status change. Includes the immutable files
  // (run.json, train.py) which never change after the run is created.
  const reload = useCallback(async () => {
    try {
      const [ev, rj, tp, sb, so, se] = await Promise.all([
        training.readFile(runId, 'events.jsonl'),
        training.readFile(runId, 'run.json'),
        training.readFile(runId, 'train.py'),
        training.readFile(runId, 'train.sbatch'),
        training.readFile(runId, 'stdout.log'),
        training.readFile(runId, 'stderr.log'),
      ])
      setEvents(parseEventLines(ev))
      setRunJson(rj)
      setTrainPy(tp)
      setTrainSbatch(sb)
      setStdout(so)
      setStderr(se)
    } catch { /* file may not exist yet */ }
  }, [runId])

  // Lightweight tail — only the file(s) that actually grow. On a remote (ssh)
  // connection every readFile is an ssh round-trip, so we DON'T re-fetch the
  // immutable run.json/train.py each tick, and only fetch the logs when the
  // logs tab is open. This is what keeps a remote run from saturating ssh.
  const tailReload = useCallback(async () => {
    try {
      setEvents(parseEventLines(await training.readFile(runId, 'events.jsonl')))
      if (tab === 'logs') {
        const [so, se] = await Promise.all([
          training.readFile(runId, 'stdout.log'),
          training.readFile(runId, 'stderr.log'),
        ])
        setStdout(so)
        setStderr(se)
      }
    } catch { /* file may not exist yet */ }
  }, [runId, tab])

  // Reload on open AND whenever the status changes. The status-change reload is
  // what catches the final epoch + run.done on the running→done transition: the
  // tail loop below stops the instant `active` flips false, so without this the
  // last update would sometimes be missing until the modal was reopened.
  useEffect(() => {
    void reload()
  }, [reload, status])

  // Tail while the run is alive — NON-overlapping (await before scheduling the
  // next tick) so a slow ssh call can't pile up. Slower cadence on remote.
  useEffect(() => {
    if (!active) return
    let stopped = false
    let timer: ReturnType<typeof setTimeout>
    const delay = getCurrentConnection().kind === 'remote-ssh' ? 5000 : 2000
    const tick = async () => {
      await tailReload()
      if (!stopped) timer = setTimeout(tick, delay)
    }
    timer = setTimeout(tick, delay)
    return () => { stopped = true; clearTimeout(timer) }
  }, [active, tailReload])

  useEffect(() => {
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape') close(null) }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [close])

  const epochEvents = events.filter((e) => e.kind === 'epoch.end')
  const last = epochEvents[epochEvents.length - 1]
  const failed = events.find((e) => e.kind === 'run.failed')
  const done = events.find((e) => e.kind === 'run.done')

  const totalEpochs = summary?.epochs ?? 0
  // Fix the chart x-axis to the planned epoch count so the curve fills
  // left→right; fall back to observed range if we don't know the total.
  const epochDomain: [number, number] | undefined =
    totalEpochs > 0 ? [1, totalEpochs] : undefined
  const curEpoch = last ? (last.epoch as number) + 1 : 0
  const progress = totalEpochs > 0 ? Math.min(1, curEpoch / totalEpochs) : 0
  // ETA from observed epoch cadence: mean wall-clock gap between epoch.end events.
  let etaSec: number | null = null
  let perEpochSec: number | null = null
  if (active && epochEvents.length >= 2 && curEpoch < totalEpochs) {
    const first = Date.parse(epochEvents[0].t)
    const lastT = Date.parse(epochEvents[epochEvents.length - 1].t)
    if (isFinite(first) && isFinite(lastT) && lastT > first) {
      perEpochSec = (lastT - first) / 1000 / (epochEvents.length - 1)
      etaSec = perEpochSec * (totalEpochs - curEpoch)
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6"
      onClick={(e) => { if (e.target === e.currentTarget) close(null) }}
    >
      <div className="flex h-full max-h-[92vh] w-full max-w-4xl flex-col overflow-hidden rounded-lg border border-[#1f2429] bg-[#0e1115] shadow-2xl">
        <div className="flex items-center gap-2 border-b border-[#1f2429] px-4 py-3">
          <StatusPill status={status} alive={summary?.alive} />
          <span className="truncate text-sm text-[#e6e8eb]">{summary?.run_label || runId}</span>
          <span
            className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${isSlurm ? 'bg-violet-900/30 text-violet-300' : 'bg-[#1f2429] text-[#7a8088]'}`}
            title={isSlurm ? `SLURM-Job · Partition ${backend?.kind === 'slurm' ? backend.slurm.partition || '—' : ''}` : 'Direkter Prozess (nohup setsid)'}
          >
            {isSlurm ? 'SLURM' : 'direct'}
          </span>
          <span className="ml-1 truncate font-mono text-[10px] text-[#5a6068]">{runId}</span>
          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={openOnCanvas}
              disabled={!runJson}
              className="rounded bg-[#13344f] px-2 py-0.5 text-[11px] text-[#6ab7ff] hover:bg-[#184466] disabled:opacity-40"
              title="Dieses Training als Graph im Training-Canvas öffnen"
            >
              → Training-Canvas
            </button>
            {active ? (
              <button
                onClick={async () => { setBusy(true); try { await stopRun(runId) } finally { setBusy(false) } }}
                disabled={busy}
                className="rounded bg-[#42191c] px-2 py-0.5 text-[11px] text-[#ff7a85] hover:bg-[#5a2227] disabled:opacity-40"
              >
                Stop
              </button>
            ) : (
              <button
                onClick={async () => { if (await confirmDialog('Run löschen?')) { await deleteRun(runId); close(null) } }}
                className="rounded px-2 py-0.5 text-[11px] text-[#7a8088] hover:bg-[#1a1e22] hover:text-[#ff7a85]"
              >
                Löschen
              </button>
            )}
            <button onClick={() => close(null)} className="rounded px-2 py-0.5 text-[#7a8088] hover:bg-[#1a1e22] hover:text-[#e6e8eb]">×</button>
          </div>
        </div>

        <div className="flex border-b border-[#1f2429] text-xs">
          {(['overview', 'charts', 'events', 'logs', 'script'] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-3 py-1.5 ${tab === t ? 'border-b border-[#6ab7ff] text-[#e6e8eb]' : 'text-[#7a8088] hover:text-[#9aa1a8]'}`}
            >
              {t}
            </button>
          ))}
        </div>

        <div className="min-h-0 flex-1 overflow-auto p-4 text-[12px] text-[#cfd3d8]">
          {tab === 'overview' && (
            <div className="space-y-4">
              {active && totalEpochs > 0 && (
                <div className="rounded border border-[#1f2429] bg-[#0a0d10] p-3">
                  <div className="mb-1.5 flex items-center text-[11px] text-[#9aa1a8]">
                    <span>epoch {curEpoch}/{totalEpochs}</span>
                    <span className="ml-auto font-mono text-[#7a8088]">
                      {etaSec != null ? `ETA ${fmtDuration(etaSec)}` : 'ETA …'}
                      {perEpochSec != null && <span className="ml-2">{perEpochSec.toFixed(1)}s/epoch</span>}
                    </span>
                  </div>
                  <div className="h-1.5 overflow-hidden rounded bg-[#1a1e22]">
                    <div className="h-full rounded bg-[#6ab7ff] transition-all" style={{ width: `${(progress * 100).toFixed(1)}%` }} />
                  </div>
                </div>
              )}
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                <Stat label="Status" value={status} />
                <Stat label="Epoch" value={last ? `${(last.epoch as number) + 1}/${summary?.epochs ?? '?'}` : '—'} />
                <Stat label="train loss" value={last?.train_loss != null ? Number(last.train_loss).toFixed(4) : '—'} />
                <Stat label="val loss" value={last?.val_loss != null ? Number(last.val_loss).toFixed(4) : '—'} />
                <Stat label="val acc" value={last?.val_acc != null ? Number(last.val_acc).toFixed(4) : '—'} />
                <Stat label="best val" value={summary?.best_val_loss != null ? summary.best_val_loss.toFixed(4) : '—'} />
                {done?.total_seconds != null && <Stat label="Dauer" value={`${Number(done.total_seconds).toFixed(0)}s`} />}
              </div>

              {failed && (
                <div className="rounded border border-[#42191c] bg-[#1a0f10] p-3 text-[11px] text-[#ff7a85]">
                  <div className="font-medium">Fehlgeschlagen in „{String(failed.stage)}"</div>
                  <div className="mt-1">{String(failed.error)}</div>
                  {!!failed.traceback && <pre className="mt-2 whitespace-pre-wrap text-[10px] text-[#b85a62]">{String(failed.traceback)}</pre>}
                </div>
              )}

              <div>
                <div className="mb-1 text-[11px] text-[#7a8088]">run.json</div>
                <pre className="max-h-64 overflow-auto rounded border border-[#1f2429] bg-[#0a0d10] p-2 text-[10px] text-[#9aa1a8]">{runJson || '—'}</pre>
              </div>
            </div>
          )}

          {tab === 'charts' && (
            epochEvents.length === 0 ? (
              <div className="text-[11px] text-[#7a8088]">Noch keine Epoch-Daten zum Plotten.</div>
            ) : (
              <div className="space-y-5">
                <ChartCard
                  title="Loss"
                  right={
                    <button
                      onClick={() => setLossLog((v) => !v)}
                      className={`rounded px-1.5 py-0.5 text-[10px] ${lossLog ? 'bg-[#13344f] text-[#6ab7ff]' : 'text-[#7a8088] hover:bg-[#1a1e22]'}`}
                    >
                      log
                    </button>
                  }
                >
                  <LineChart series={lossSeries(events)} yLog={lossLog} xLabel="epoch" xDomain={epochDomain} />
                </ChartCard>

                {metricSeries(events).length > 0 && (
                  <ChartCard title="Metriken">
                    <LineChart series={metricSeries(events)} xLabel="epoch" xDomain={epochDomain} />
                  </ChartCard>
                )}

                <ChartCard title="Learning rate">
                  <LineChart series={lrSeries(events)} xLabel="epoch" xDomain={epochDomain} yFormat={(v) => v.toExponential(1)} />
                </ChartCard>
              </div>
            )
          )}

          {tab === 'events' && (
            epochEvents.length === 0 ? (
              <div className="text-[11px] text-[#7a8088]">Noch keine Epoch-Events.</div>
            ) : (
              <table className="w-full text-left text-[11px]">
                <thead className="text-[#7a8088]">
                  <tr><th className="py-1 pr-3">epoch</th><th className="pr-3">train</th><th className="pr-3">val</th><th className="pr-3">acc</th><th>lr</th></tr>
                </thead>
                <tbody className="font-mono">
                  {epochEvents.slice().reverse().map((e, i) => (
                    <tr key={i} className="border-t border-[#171b1f]">
                      <td className="py-0.5 pr-3">{(e.epoch as number) + 1}</td>
                      <td className="pr-3">{e.train_loss != null ? Number(e.train_loss).toFixed(4) : '—'}</td>
                      <td className="pr-3">{e.val_loss != null ? Number(e.val_loss).toFixed(4) : '—'}</td>
                      <td className="pr-3">{e.val_acc != null ? Number(e.val_acc).toFixed(4) : '—'}</td>
                      <td>{e.lr != null ? Number(e.lr).toExponential(2) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )
          )}

          {tab === 'logs' && (
            <div className="space-y-3">
              <LogBlock title="stdout.log" text={stdout} />
              <LogBlock title="stderr.log" text={stderr} tone="err" />
            </div>
          )}

          {tab === 'script' && (
            <div className="space-y-3">
              {isSlurm && <LogBlock title="train.sbatch (an SLURM übergeben)" text={trainSbatch} />}
              <LogBlock title="train.py (ausgeführtes Trainings-Skript)" text={trainPy} />
              <LogBlock title="run.json (eingefrorene Config)" text={runJson} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function fmtDuration(sec: number): string {
  if (sec < 60) return `${Math.round(sec)}s`
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${Math.round(sec % 60)}s`
  return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`
}

function ChartCard({ title, right, children }: { title: string; right?: ReactNode; children: ReactNode }) {
  return (
    <div className="rounded border border-[#1f2429] bg-[#0a0d10] p-3">
      <div className="mb-2 flex items-center">
        <span className="text-[11px] text-[#9aa1a8]">{title}</span>
        {right && <span className="ml-auto">{right}</span>}
      </div>
      {children}
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-[#1f2429] bg-[#0a0d10] px-3 py-2">
      <div className="text-[10px] text-[#7a8088]">{label}</div>
      <div className="mt-0.5 font-mono text-[13px] text-[#e6e8eb]">{value}</div>
    </div>
  )
}

function LogBlock({ title, text, tone }: { title: string; text: string; tone?: 'err' }) {
  return (
    <div>
      <div className="mb-1 text-[11px] text-[#7a8088]">{title}</div>
      <pre className={`max-h-72 overflow-auto rounded border border-[#1f2429] bg-[#0a0d10] p-2 text-[10px] ${tone === 'err' ? 'text-[#b85a62]' : 'text-[#9aa1a8]'}`}>{text || '—'}</pre>
    </div>
  )
}
