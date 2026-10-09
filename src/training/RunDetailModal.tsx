import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react'

import { training } from './backend'
import { useTrainingStore } from './store'
import {
  type ResumableRecord,
  type RunConfig,
  type TrainingEvent,
  type GpuStat,
  RUNNING_STATES,
} from './types'
import StatusPill from './StatusPill'
import LineChart from './charts/LineChart'
import { lossSeries, lrSeries, metricSeries } from './charts/series'
import { latestWinsGuard, parseFinalEvents } from './events'
import { latestEval, latestEvalHeads } from './charts/evaluation'
import { EvalDiagram } from './charts/Evaluation'
import { parseRunConfig } from './parseRunConfig'
import { runConfigToTrainingSnapshot } from './graph/fromRun'
import { useTrainingGraphStore } from './graph/store'
import { useViewModeStore } from './graph/viewMode'
import { getCurrentConnection } from '../connections/store'
import { confirmDialog } from '../ui/confirm'
import { fireAndForget } from '../errors/report'
import { lineFigureSvg, type Series } from '../figures/lineFigure'
import { fs } from '../connections/backend'

type Tab = 'overview' | 'charts' | 'events' | 'predictions' | 'hardware' | 'logs' | 'script'

const TAB_LABELS: Record<Tab, string> = {
  overview: 'overview', charts: 'charts', events: 'events', predictions: 'Auswertung',
  hardware: 'hardware', logs: 'logs', script: 'script',
}

export default function RunDetailModal({ runId }: { runId: string }) {
  const close = useTrainingStore((s) => s.select)
  const stopRun = useTrainingStore((s) => s.stopRun)
  const deleteRun = useTrainingStore((s) => s.deleteRun)
  const openEvalRun = useTrainingStore((s) => s.openEvalRun)
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
  const [slurmJobId, setSlurmJobId] = useState<string | null>(null)
  const [gpu, setGpu] = useState<GpuStat[] | null>(null)
  const [gpuError, setGpuError] = useState<string | null>(null)
  const [promoteName, setPromoteName] = useState('')
  const [promoted, setPromoted] = useState<string | null>(null)
  const [promoteErr, setPromoteErr] = useState<string | null>(null)
  /** readFile failures (ssh drop, permission) — distinct from a missing file,
   *  which the backend returns as "". Surfaced so stale content isn't trusted. */
  const [readError, setReadError] = useState<string | null>(null)
  const [eventsError, setEventsError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  // Phase 74 — manifest.json content + parse state for the resumable banner.
  // `manifestError` records a real read/parse failure; an empty `manifestJson`
  // (file truly missing) renders as "Manifest nicht lesbar: <reason>", not as
  // a silent "no resumable flag" claim.
  const [manifestJson, setManifestJson] = useState('')
  const [manifestError, setManifestError] = useState<string | null>(null)
  const [exportMsg, setExportMsg] = useState<string | null>(null)

  const status = summary?.status ?? 'unknown'
  const active = RUNNING_STATES.has(status) || (summary?.alive ?? false)

  // Frozen run.json (backend, eval-validation metadata). A non-empty body that
  // fails to parse is reported, never silently treated as "no config".
  const parsed = useMemo(() => {
    if (!runJson) return { cfg: null as RunConfig | null, error: null as string | null }
    try { return { cfg: parseRunConfig(JSON.parse(runJson)), error: null } }
    catch (e) { return { cfg: null, error: e instanceof Error ? e.message : String(e) } }
  }, [runJson])
  const cfgError = parsed.error
  const backend = parsed.cfg?.backend ?? null
  const isSlurm = backend?.kind === 'slurm'
  const isEvalRun = (summary?.eval_only ?? false) || (parsed.cfg?.eval_only ?? false)
  const validate = parsed.cfg?.validate ?? null

  // Phase 31 — events JSONL is whole-file read + snapshotted. Two concurrent
  // reads can land out of order (esp. over ssh): the newest-STARTED read always
  // sees a superset of an older one, so an older response arriving late must be
  // DROPPED (a stale snapshot would "overwrite" the final state already shown).
  // parseFinalEvents additionally truncates at the first terminal event so a
  // trailing out-of-order line (EPOCH after FAILED) never surfaces.
  const eventsApply = useMemo(
    () => latestWinsGuard<string>((text) => setEvents(parseFinalEvents(text))),
    [],
  )

  // Frozen run.json (backend, eval-validation metadata).
  // (parsed above — cfgError carries a corruption message.)

  // Rebuild this run's training graph onto the canvas, even if its .spinotrain was
  // never saved — run.json carries the full frozen config.
  const openOnCanvas = () => {
    if (!runJson) return
    try {
      const config = parseRunConfig(JSON.parse(runJson))
      const snapshot = runConfigToTrainingSnapshot(config)
      useTrainingGraphStore.getState().loadSnapshot(snapshot)
      useViewModeStore.getState().setMode('training')
      close(null)
    } catch (e) {
      // A corrupt/frozen config must not make the button appear to do nothing.
      setActionError(`run.json konnte nicht geöffnet werden: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const exportChartSvg = async (chartName: string, series: Series[], opts: { yLog?: boolean; yFormat?: (v: number) => string; xDomain?: [number, number] } = {}) => {
    try {
      await fs.mkdir(`experiments/runs/${runId}/figures`)
    } catch {
      // directory may already exist
    }
    const svg = lineFigureSvg(series, {
      theme: 'print',
      width: 240,
      height: 170,
      xLabel: 'epoch',
      yLabel: chartName === 'lr' ? 'lr' : chartName === 'loss' ? 'loss' : 'value',
      title: chartName.charAt(0).toUpperCase() + chartName.slice(1),
      yLog: opts.yLog,
      yFormat: opts.yFormat,
      xDomain: opts.xDomain,
    })
    const relPath = `experiments/runs/${runId}/figures/${chartName}.svg`
    await fs.write(relPath, svg)
    setExportMsg(`Gespeichert: ${relPath}`)
    setTimeout(() => setExportMsg(null), 3000)
  }

  // A SLURM run's stdout/stderr land in slurm-<jobid>.out/.err (the #SBATCH
  // --output/--error targets), NOT stdout.log/stderr.log (those only exist for
  // direct/local launches). The job id is frozen in the `pid` file as
  // `slurm:<jobid>`. Returns the [out, err] file names for THIS run.
  const logFileNames = (jobId: string | null): [string, string] =>
    jobId ? [`slurm-${jobId}.out`, `slurm-${jobId}.err`] : ['stdout.log', 'stderr.log']

  // Full read — used on open + on status change. Includes the immutable files
  // (run.json, train.py) which never change after the run is created.
  const reload = useCallback(async () => {
    const errs: string[] = []
    // Resolve the log target first: read the pid file to learn whether this is
    // a SLURM run (`slurm:<jobid>`) so we tail the right files below. A missing
    // pid (normal for a brand-new run) reads as "" — only a real read failure
    // is recorded.
    let pidRaw = ''
    try { pidRaw = (await training.readFile(runId, 'pid')).trim() }
    catch (e) { errs.push(`pid: ${e instanceof Error ? e.message : String(e)}`) }
    const jobId = pidRaw.startsWith('slurm:') ? pidRaw.slice('slurm:'.length).trim() || null : null
    setSlurmJobId(jobId)
    const [outName, errName] = logFileNames(jobId)
    // Events go through the stale-read guard (never blocks the rest of the
    // load): an older read landing late is discarded, a terminal run's final
    // snapshot stays final.
    eventsApply(() => training.readFile(runId, 'events.jsonl'))
      .then(() => setEventsError(null))
      .catch((e) => setEventsError(`events.jsonl: ${e instanceof Error ? e.message : String(e)}`))
    const results = await Promise.all([
      readRunFile(runId, 'run.json'),
      readRunFile(runId, 'train.py'),
      readRunFile(runId, 'train.sbatch'),
      readRunFile(runId, outName),
      readRunFile(runId, errName),
      readRunFile(runId, 'manifest.json'),
    ])
    for (const r of results) if (r.error) errs.push(r.error)
    setRunJson(results[0].text)
    setTrainPy(results[1].text)
    setTrainSbatch(results[2].text)
    setStdout(results[3].text)
    setStderr(results[4].text)
    // Phase 74 — manifest.json: a real read failure is shown as the banner;
    // a missing file is the same state (the backend returns "" for missing).
    // Either way we never silently claim "not resumable".
    setManifestJson(results[5].text)
    setManifestError(results[5].error)
    setReadError(errs.length ? errs.join(' · ') : null)
  }, [runId, eventsApply])

  // Lightweight tail — only the file(s) that actually grow. On a remote (ssh)
  // connection every readFile is an ssh round-trip, so we DON'T re-fetch the
  // immutable run.json/train.py each tick, and only fetch the logs when the
  // logs tab is open. This is what keeps a remote run from saturating ssh.
  const tailReload = useCallback(async () => {
    eventsApply(() => training.readFile(runId, 'events.jsonl'))
      .then(() => setEventsError(null))
      .catch((e) => setEventsError(`events.jsonl: ${e instanceof Error ? e.message : String(e)}`))
    if (tab === 'logs') {
      const [outName, errName] = logFileNames(slurmJobId)
      const results = await Promise.all([
        readRunFile(runId, outName),
        readRunFile(runId, errName),
      ])
      setStdout(results[0].text)
      setStderr(results[1].text)
      const errs = results.map((r) => r.error).filter((x): x is string => !!x)
      if (errs.length) setReadError(errs.join(' · '))
    }
  }, [runId, tab, slurmJobId, eventsApply])

  // Reload on open AND whenever the status changes. The status-change reload is
  // what catches the final epoch + run.done on the running→done transition: the
  // tail loop below stops the instant `active` flips false, so without this the
  // last update would sometimes be missing until the modal was reopened.
  useEffect(() => {
    // reload() resets loading state synchronously before the async refetch; a
    // render-time adjustment would race the effect because reload() also runs on open.
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reload() synchronously resets loading before the async refetch
    void fireAndForget('reload', reload())
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

  // GPU snapshot polling — only while the hardware tab is open (it's a host-wide
  // nvidia-smi call, one ssh round-trip on remote). Non-overlapping.
  useEffect(() => {
    if (tab !== 'hardware') return
    let stopped = false
    let timer: ReturnType<typeof setTimeout>
    const delay = getCurrentConnection().kind === 'remote-ssh' ? 6000 : 3000
    const tick = async () => {
      try {
        setGpu(await training.gpuStats(runId))
        setGpuError(null)
      } catch (e) {
        // A failed probe must not render as "no GPU visible" (a false claim);
        // show an explicit unknown/error state instead.
        setGpu([])
        setGpuError(e instanceof Error ? e.message : String(e))
      }
      if (!stopped) timer = setTimeout(tick, delay)
    }
    void fireAndForget('tick', tick())
    return () => { stopped = true; clearTimeout(timer) }
  }, [tab, runId])

  // Prefill the promote name from the run label once run.json is loaded. Done in
  // render (guarded by !promoteName) so it converges after the first commit
  // without an effect.
  const runLabel = summary?.run_label
  if (!promoteName && runLabel) {
    setPromoteName(runLabel.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'model')
  }

  useEffect(() => {
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape') close(null) }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [close])

  const epochEvents = events.filter((e) => e.kind === 'epoch.end')
  const last = epochEvents[epochEvents.length - 1]
  const failed = events.find((e) => e.kind === 'run.failed')
  const done = events.find((e) => e.kind === 'run.done')
  // Latest sample-predictions snapshot (emitted by the trainer on each new best).
  const samplePreds = [...events].reverse().find((e) => e.kind === 'sample.preds')
  const predRows = (samplePreds?.rows as Array<Record<string, unknown>> | undefined) ?? []
  // Multitask: per-head prediction rows + per-head eval summaries.
  const predHeads = samplePreds?.heads as Array<{ output: string; task: string; rows: Array<Record<string, unknown>> }> | undefined
  const evalHeads = latestEvalHeads(events)
  // Fixed-schema evaluation payload → confusion matrix (classification/binary)
  // or pred-vs-actual scatter (regression), whichever the trainer emitted.
  const evalSummary = latestEval(events)
  const hasEval = !!evalSummary || !!evalHeads || predRows.length > 0 || !!predHeads?.length

  // Multitask/joint detection: run.json flag (chatbot-generated trainers set
  // `multitask: true`), the new Head-node `training.heads`, OR namespaced
  // "<output>/<metric>" keys in epoch.end metrics.
  const isMultitask = (() => {
    // parsed.cfg is null both when run.json is absent and when it is corrupt
    // (the latter is surfaced via cfgError); fall back to event heuristics.
    const c = parsed.cfg as (RunConfig & { multitask?: boolean }) | null
    if (c && (c.multitask || (c.training?.heads?.length ?? 0) > 0)) return true
    return epochEvents.some((e) => {
      const m = e.metrics as Record<string, unknown> | null | undefined
      return m && Object.keys(m).some((k) => k.includes('/'))
    })
  })()
  // Trainer-emitted context worth surfacing for joint runs.
  const dsLoaded = events.find((e) => e.kind === 'dataset.loaded')
  const regStd = events.find((e) => e.kind === 'reg.standardize')
  const lastMetrics = (last?.metrics as Record<string, unknown> | null | undefined) ?? null
  const canPromote = !active && (summary?.has_checkpoint ?? false)

  // Phase 74 — resumable banner for failed/cancelled runs. The flag is in
  // manifest.json (and mirrored in metrics.json); we read manifest.json because
  // it is the artifact the trainer writes atomically at every terminal state.
  // Never silently swallow a read/parse failure — `manifestReadProblem` is
  // rendered so a missing manifest renders as "Manifest nicht lesbar: …"
  // rather than a false "not resumable".
  const resumable = useMemo<ResumableRecord | null>(() => {
    if (!manifestJson) return null
    try {
      const m = JSON.parse(manifestJson) as Record<string, unknown>
      const r = m.resumable as Record<string, unknown> | undefined
      if (!r || typeof r !== 'object') return null
      return {
        resumable: r.resumable === true,
        resume_from: typeof r.resume_from === 'string' ? r.resume_from : null,
        epoch: typeof r.epoch === 'number' ? r.epoch : null,
        reason: typeof r.reason === 'string' ? r.reason : '',
      }
    } catch {
      // Corrupt manifest.json: the banner above falls back to "Manifest nicht
      // lesbar: <parse error>" via manifestReadProblem, which is rendered
      // separately. Returning null here is the documented "unknown" state,
      // not a claim of "not resumable".
      return null
    }
  }, [manifestJson])
  const manifestReadProblem = (status === 'failed' || status === 'cancelled')
    ? (manifestError ?? (manifestJson ? null : 'manifest.json nicht gefunden'))
    : null

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
      <div className="flex h-full max-h-[92vh] w-full max-w-4xl flex-col overflow-hidden rounded-lg border border-[#1f2429] bg-[#0e1216] shadow-2xl">
        <div className="flex items-center gap-2 border-b border-[#1f2429] px-4 py-3">
          <StatusPill status={status} alive={summary?.alive} />
          <span className="truncate text-sm text-[#e6e8eb]">{summary?.run_label || runId}</span>
          <span
            className={`shrink-0 rounded px-1.5 py-0.5 text-[10px] uppercase tracking-wide ${isSlurm ? 'bg-violet-900/30 text-violet-300' : 'bg-[#1f2429] text-[#6f767e]'}`}
            title={isSlurm ? `SLURM-Job · Partition ${backend?.kind === 'slurm' ? backend.slurm.partition || '—' : ''}` : 'Direkter Prozess (nohup setsid)'}
          >
            {isSlurm ? 'SLURM' : 'direct'}
          </span>
          {isMultitask && (
            <span
              className="shrink-0 rounded bg-[var(--accent-sel)] px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-[var(--accent)]"
              title="Multitask / Joint-Run — mehrere Aufgaben (z. B. Klassifikation + Regression) in einem Modell"
            >
              multitask
            </span>
          )}
          <span className="ml-1 truncate font-mono text-[10px] text-[#5a6068]">{runId}</span>
          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={openOnCanvas}
              disabled={!runJson}
              className="rounded bg-[var(--accent-sel)] px-2 py-0.5 text-[11px] text-[var(--accent)] hover:bg-[var(--accent-sel-hover)] disabled:opacity-40"
              title="Dieses Training als Graph im Training-Canvas öffnen"
            >
              → Training-Canvas
            </button>
            {!active && !isEvalRun && (summary?.has_checkpoint ?? false) && (
              <button
                onClick={() => { openEvalRun(runId); close(null) }}
                className="rounded bg-[var(--accent-sel)] px-2 py-0.5 text-[11px] text-[var(--accent)] hover:bg-[var(--accent-sel-hover)]"
                title="Dieses trainierte Modell auf einem externen/Benchmark-Datensatz validieren"
              >
                Externe Validierung
              </button>
            )}
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
                className="rounded px-2 py-0.5 text-[11px] text-[#6f767e] hover:bg-[#1a1e22] hover:text-[#ff7a85]"
              >
                Löschen
              </button>
            )}
            <button onClick={() => close(null)} className="rounded px-2 py-0.5 text-[#6f767e] hover:bg-[#1a1e22] hover:text-[#e6e8eb]">×</button>
          </div>
        </div>

        {(cfgError || readError || eventsError || gpuError || actionError) && (
          <div className="space-y-0.5 border-b border-rose-900/40 bg-rose-950/30 px-4 py-1.5 text-[10px] text-rose-300">
            {cfgError && <div>run.json beschädigt: {cfgError}</div>}
            {readError && <div>Dateien unvollständig gelesen: {readError}</div>}
            {eventsError && <div>{eventsError}</div>}
            {gpuError && <div>GPU-Status nicht abfragbar: {gpuError}</div>}
            {actionError && <div>{actionError}</div>}
          </div>
        )}

        {exportMsg && (
          <div className="border-b border-emerald-900/40 bg-emerald-950/30 px-4 py-1.5 text-[10px] text-emerald-300">
            {exportMsg}
          </div>
        )}

        {isEvalRun && validate && (
          <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 border-b border-[#1f2429] bg-[var(--accent-sel)]/20 px-4 py-1.5 text-[10px] text-[#9aa1a8]">
            <span className="rounded bg-[var(--accent-sel)] px-1.5 py-0.5 font-semibold text-[var(--accent)]">EXTERNE VALIDIERUNG</span>
            <span>Modell aus <span className="font-mono text-[#cfd3d8]">{validate.source_run ?? '?'}</span></span>
            <span>· Datensatz <span className="font-mono text-[#cfd3d8]">{parsed.cfg?.dataset.relpath}</span></span>
            {validate.adapter && Object.keys(validate.adapter.column_map).length > 0 && (
              <span className="text-[#5a6068]">· {Object.entries(validate.adapter.column_map).map(([role, col]) => `${role}→${col}`).join(', ')}</span>
            )}
          </div>
        )}

        {/* Phase 74 — resumable banner for failed/cancelled runs. Three states:
            (a) manifest.json present + resumable=true → amber banner
            (b) manifest.json present + resumable=false + reason → muted reason
            (c) manifest.json missing/corrupt → explicit "Manifest nicht lesbar"
            (d) status is not failed/cancelled → no banner. The status itself
            stays unchanged; the banner is purely informational and the user
            still opens "Neuer Run → Fortsetzen ab Checkpoint" themselves. */}
        {(status === 'failed' || status === 'cancelled') && resumable?.resumable && (
          <div className="border-b border-amber-900/40 bg-amber-950/30 px-4 py-2 text-[11px] text-amber-200">
            <div className="flex items-center gap-2">
              <span className="rounded bg-amber-900/50 px-1.5 py-0.5 font-semibold uppercase text-amber-100">Fortsetzbar</span>
              <span>
                Checkpoint nach Epoche {resumable.epoch ?? '?'} vorhanden
                (Status bleibt {status === 'failed' ? 'FEHLGESCHLAGEN' : 'ABGEBROCHEN'}).
                Zum Fortsetzen: Neuer Run → „Fortsetzen ab Checkpoint".
              </span>
            </div>
            {resumable.resume_from && (
              <div className="mt-1 font-mono text-[10px] text-amber-300/80">{resumable.resume_from}</div>
            )}
          </div>
        )}
        {(status === 'failed' || status === 'cancelled') && resumable && !resumable.resumable && resumable.reason && (
          <div className="border-b border-[#1f2429] bg-[#0a0d10] px-4 py-1.5 text-[10px] text-[#6f767e]">
            Nicht fortsetzbar: {resumable.reason}
          </div>
        )}
        {(status === 'failed' || status === 'cancelled') && manifestReadProblem && (
          <div className="border-b border-rose-900/40 bg-rose-950/30 px-4 py-1.5 text-[10px] text-rose-300">
            Manifest nicht lesbar: {manifestReadProblem}
          </div>
        )}

        <div className="flex border-b border-[#1f2429] text-xs">
          {(['overview', 'charts', 'events', 'predictions', 'hardware', 'logs', 'script'] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-3 py-1.5 ${tab === t ? 'border-b border-[var(--accent)] text-[#e6e8eb]' : 'text-[#6f767e] hover:text-[#9aa1a8]'}`}
            >
              {TAB_LABELS[t]}
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
                    <span className="ml-auto font-mono text-[#6f767e]">
                      {etaSec != null ? `ETA ${fmtDuration(etaSec)}` : 'ETA …'}
                      {perEpochSec != null && <span className="ml-2">{perEpochSec.toFixed(1)}s/epoch</span>}
                    </span>
                  </div>
                  <div className="h-1.5 overflow-hidden rounded bg-[#1a1e22]">
                    <div className="h-full rounded bg-[var(--accent)] transition-all" style={{ width: `${(progress * 100).toFixed(1)}%` }} />
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

              {/* Per-task metrics from the latest epoch (joint runs report e.g.
                  binder accuracy + affinity reg_mae_log10, or "<output>/<metric>"). */}
              {lastMetrics && Object.keys(lastMetrics).length > 0 && (
                <div>
                  <div className="mb-1 text-[11px] text-[#9aa1a8]">Metriken (letzte Epoch)</div>
                  <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                    {Object.entries(lastMetrics).map(([k, v]) => (
                      <Stat key={k} label={k} value={typeof v === 'number' ? v.toFixed(4) : String(v)} />
                    ))}
                  </div>
                </div>
              )}

              {/* Joint-run dataset context: branches + binder count + target
                  standardization the trainer emitted. */}
              {(dsLoaded || regStd) && (
                <div className="rounded border border-[#1f2429] bg-[#0a0d10] p-3 text-[11px] text-[#9aa1a8]">
                  {dsLoaded && (
                    <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                      {dsLoaded.n_rows != null && <span>{String(dsLoaded.n_rows)} Zeilen</span>}
                      {Array.isArray(dsLoaded.branches) && (dsLoaded.branches as unknown[]).length > 0 && (
                        <span>Branches: <span className="font-mono text-[#cfd3d8]">{(dsLoaded.branches as string[]).join(', ')}</span></span>
                      )}
                      {dsLoaded.n_binders != null && <span>{String(dsLoaded.n_binders)} Binder</span>}
                      {dsLoaded.skipped != null && Number(dsLoaded.skipped) > 0 && <span className="text-[#6f767e]">{String(dsLoaded.skipped)} übersprungen</span>}
                    </div>
                  )}
                  {regStd && (
                    <div className="mt-1 text-[10px] text-[#6f767e]">
                      Regressionsziel standardisiert: μ={Number(regStd.mean).toFixed(4)} · σ={Number(regStd.std).toFixed(4)}
                      {regStd.n_train_binders != null ? ` (n=${String(regStd.n_train_binders)})` : ''}
                    </div>
                  )}
                </div>
              )}

              {failed && (
                <div className="rounded border border-[#42191c] bg-[#1a0f10] p-3 text-[11px] text-[#ff7a85]">
                  <div className="font-medium">Fehlgeschlagen in „{String(failed.stage)}"</div>
                  <div className="mt-1">{String(failed.error)}</div>
                  {!!failed.traceback && <pre className="mt-2 whitespace-pre-wrap text-[10px] text-[#b85a62]">{String(failed.traceback)}</pre>}
                </div>
              )}

              {canPromote && (
                <div className="rounded border border-[#1f2429] bg-[#0a0d10] p-3">
                  <div className="mb-1.5 text-[11px] text-[#9aa1a8]">Bestes Modell übernehmen</div>
                  <div className="mb-2 text-[10px] text-[#6f767e]">
                    Kopiert <code>checkpoints/best.pt</code> nach <code>models/best/&lt;name&gt;.pt</code> —
                    von dort als Pretrained-Gewicht weiterverwendbar.
                  </div>
                  <div className="flex items-center gap-2">
                    <input
                      value={promoteName}
                      onChange={(e) => { setPromoteName(e.target.value); setPromoted(null); setPromoteErr(null) }}
                      placeholder="iris-mlp"
                      className="min-w-0 flex-1 rounded border border-[#2a3038] bg-[#0e1216] px-2 py-1 text-[11px] text-[#e6e8eb] focus:border-[var(--accent)] focus:outline-none"
                    />
                    <span className="text-[10px] text-[#5a6068]">.pt</span>
                    <button
                      disabled={busy || !promoteName.trim()}
                      onClick={async () => {
                        setBusy(true); setPromoted(null); setPromoteErr(null)
                        try {
                          const dest = await training.promote(runId, promoteName.trim())
                          setPromoted(dest)
                        } catch (e) {
                          setPromoteErr(e instanceof Error ? e.message : String(e))
                        } finally { setBusy(false) }
                      }}
                      className="rounded bg-[var(--accent-sel)] px-2 py-1 text-[11px] text-[var(--accent)] hover:bg-[var(--accent-sel-hover)] disabled:opacity-40"
                    >
                      Übernehmen
                    </button>
                  </div>
                  {promoted && <div className="mt-2 text-[10px] text-emerald-400">→ {promoted}</div>}
                  {promoteErr && <div className="mt-2 text-[10px] text-rose-400">{promoteErr}</div>}
                </div>
              )}

              <div>
                <div className="mb-1 text-[11px] text-[#6f767e]">run.json</div>
                <pre className="max-h-64 overflow-auto rounded border border-[#1f2429] bg-[#0a0d10] p-2 text-[10px] text-[#9aa1a8]">{runJson || '—'}</pre>
              </div>
            </div>
          )}

          {tab === 'charts' && (
            epochEvents.length === 0 ? (
              <div className="text-[11px] text-[#6f767e]">Noch keine Epoch-Daten zum Plotten.</div>
            ) : (
              <div className="space-y-5">
                <ChartCard
                  title="Loss"
                  right={
                    <>
                      <button
                        onClick={() => setLossLog((v) => !v)}
                        className={`rounded px-1.5 py-0.5 text-[10px] mr-1 ${lossLog ? 'bg-[var(--accent-sel)] text-[var(--accent)]' : 'text-[#6f767e] hover:bg-[#1a1e22]'}`}
                      >
                        log
                      </button>
                      <button
                        onClick={() => exportChartSvg('loss', lossSeries(events), { yLog: lossLog, xDomain: epochDomain })}
                        className="rounded px-1.5 py-0.5 text-[10px] text-[#6f767e] hover:bg-[#1a1e22]"
                        title="Als SVG exportieren"
                      >
                        SVG
                      </button>
                    </>
                  }
                >
                  <LineChart series={lossSeries(events)} yLog={lossLog} xLabel="epoch" xDomain={epochDomain} />
                </ChartCard>

                {metricSeries(events).length > 0 && (
                  <ChartCard
                    title="Metriken"
                    right={
                      <button
                        onClick={() => exportChartSvg('metrics', metricSeries(events), { xDomain: epochDomain })}
                        className="rounded px-1.5 py-0.5 text-[10px] text-[#6f767e] hover:bg-[#1a1e22]"
                        title="Als SVG exportieren"
                      >
                        SVG
                      </button>
                    }
                  >
                    <LineChart series={metricSeries(events)} xLabel="epoch" xDomain={epochDomain} />
                  </ChartCard>
                )}

                <ChartCard
                  title="Learning rate"
                  right={
                    <button
                      onClick={() => exportChartSvg('lr', lrSeries(events), { xDomain: epochDomain, yFormat: (v) => v.toExponential(1) })}
                      className="rounded px-1.5 py-0.5 text-[10px] text-[#6f767e] hover:bg-[#1a1e22]"
                      title="Als SVG exportieren"
                    >
                      SVG
                    </button>
                  }
                >
                  <LineChart series={lrSeries(events)} xLabel="epoch" xDomain={epochDomain} yFormat={(v) => v.toExponential(1)} />
                </ChartCard>
              </div>
            )
          )}

          {tab === 'events' && (
            epochEvents.length === 0 ? (
              <div className="text-[11px] text-[#6f767e]">Noch keine Epoch-Events.</div>
            ) : (
              <table className="w-full text-left text-[11px]">
                <thead className="text-[#6f767e]">
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

          {tab === 'predictions' && (
            !hasEval ? (
              <div className="space-y-2 text-[11px] text-[#6f767e]">
                <div>
                  Noch keine Auswertung — der Trainer schreibt Diagramme + Stichproben bei jedem neuen
                  Best-Checkpoint (braucht einen Validierungs-Split).
                </div>
                {isMultitask && epochEvents.length > 0 && (
                  <div className="rounded border border-[#1f2429] bg-[#0a0d10] p-2 text-[10px] text-[#9aa1a8]">
                    Dieser Joint-Run hat <span className="font-mono">keine eval.summary/sample.preds</span>-Events
                    geschrieben — Fortschritt + Metriken siehst du unter „charts". Für die Diagramme
                    (Konfusionsmatrix je Klassifikations-Kopf, Scatter je Regressions-Kopf) muss der
                    Trainer pro Kopf eine <span className="font-mono">eval.summary</span> emittieren.
                  </div>
                )}
              </div>
            ) : evalHeads || predHeads ? (
              // Multitask: one section per output head (diagram + sample table).
              <div className="space-y-6">
                {(evalHeads ?? (predHeads ?? []).map((p) => ({ output: p.output, task: p.task } as typeof p))).map((h) => {
                  const ph = predHeads?.find((p) => p.output === h.output)
                  return (
                    <div key={h.output} className="space-y-3">
                      <div className="flex items-center gap-2">
                        <span className="rounded bg-[var(--accent-sel)] px-1.5 py-0.5 text-[10px] text-[var(--accent)]">{h.output || 'out'}</span>
                        <span className="text-[11px] text-[#9aa1a8]">{h.task}</span>
                      </div>
                      {evalHeads && (
                        <div className="rounded border border-[#1f2429] bg-[#0a0d10] p-3">
                          <EvalDiagram summary={h as Parameters<typeof EvalDiagram>[0]['summary']} />
                        </div>
                      )}
                      {ph && ph.rows.length > 0 && <PredictionTable rows={ph.rows} epoch={samplePreds?.epoch as number | undefined} />}
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="space-y-5">
                {evalSummary && (
                  <div className="rounded border border-[#1f2429] bg-[#0a0d10] p-3">
                    <EvalDiagram summary={evalSummary} />
                  </div>
                )}
                {predRows.length > 0 && <PredictionTable rows={predRows} epoch={samplePreds?.epoch as number | undefined} />}
              </div>
            )
          )}

          {tab === 'hardware' && (
            gpu === null ? (
              <div className="text-[11px] text-[#6f767e]">GPU-Status wird abgefragt…</div>
            ) : gpu.length === 0 ? (
              <div className="text-[11px] text-[#6f767e]">
                Keine GPU sichtbar (kein <code>nvidia-smi</code> auf dem Ausführungs-Host, oder reines CPU-Training).
              </div>
            ) : (
              <div className="space-y-3">
                {gpu.map((g) => {
                  const memPct = g.mem_total_mb > 0 ? (g.mem_used_mb / g.mem_total_mb) * 100 : 0
                  return (
                    <div key={g.index} className="rounded border border-[#1f2429] bg-[#0a0d10] p-3">
                      <div className="mb-2 flex items-center text-[11px]">
                        <span className="text-[#e6e8eb]">GPU {g.index} · {g.name}</span>
                        <span className="ml-auto font-mono text-[#6f767e]">{g.temp_c.toFixed(0)}°C</span>
                      </div>
                      <Meter label="Auslastung" pct={g.util_pct} text={`${g.util_pct.toFixed(0)}%`} />
                      <div className="h-1" />
                      <Meter label="Speicher" pct={memPct} text={`${(g.mem_used_mb / 1024).toFixed(1)} / ${(g.mem_total_mb / 1024).toFixed(1)} GB`} />
                    </div>
                  )
                })}
                <div className="text-[10px] text-[#5a6068]">Aktualisiert automatisch, solange dieser Tab offen ist.</div>
              </div>
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

// Read one file out of a run dir, distinguishing a real read failure from a
// missing file (the backend returns "" for a missing file, never throws).
async function readRunFile(runId: string, name: string): Promise<{ text: string; error: string | null }> {
  try {
    return { text: await training.readFile(runId, name), error: null }
  } catch (e) {
    return { text: '', error: `${name}: ${e instanceof Error ? e.message : String(e)}` }
  }
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

function Meter({ label, pct, text }: { label: string; pct: number; text: string }) {
  const clamped = Math.max(0, Math.min(100, pct))
  return (
    <div>
      <div className="mb-0.5 flex items-center text-[10px] text-[#6f767e]">
        <span>{label}</span>
        <span className="ml-auto font-mono">{text}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded bg-[#1a1e22]">
        <div className="h-full rounded bg-[var(--accent)] transition-all" style={{ width: `${clamped.toFixed(1)}%` }} />
      </div>
    </div>
  )
}

function PredictionTable({ rows, epoch }: { rows: Array<Record<string, unknown>>; epoch?: number }) {
  return (
    <div className="space-y-2">
      <div className="text-[10px] text-[#6f767e]">
        Stichprobe aus dem Validierungs-Set beim besten Checkpoint
        {epoch != null ? ` (Epoch ${epoch + 1})` : ''}.
      </div>
      <table className="w-full text-left text-[11px]">
        <thead className="text-[#6f767e]">
          <tr><th className="py-1 pr-3">#</th><th className="pr-3">Vorhersage</th><th className="pr-3">Wahrheit</th><th className="pr-3">Konfidenz</th><th></th></tr>
        </thead>
        <tbody className="font-mono">
          {rows.map((r, i) => {
            const correct = r.correct as boolean | undefined
            return (
              <tr key={i} className="border-t border-[#171b1f]">
                <td className="py-0.5 pr-3 text-[#5a6068]">{i + 1}</td>
                <td className="pr-3 text-[#e6e8eb]">{String(r.pred)}</td>
                <td className="pr-3 text-[#9aa1a8]">{String(r.truth)}</td>
                <td className="pr-3">{r.conf != null ? Number(r.conf).toFixed(3) : '—'}</td>
                <td>{correct === undefined ? '' : correct ? <span className="text-emerald-400">✓</span> : <span className="text-rose-400">✗</span>}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-[#1f2429] bg-[#0a0d10] px-3 py-2">
      <div className="text-[10px] text-[#6f767e]">{label}</div>
      <div className="mt-0.5 font-mono text-[13px] text-[#e6e8eb]">{value}</div>
    </div>
  )
}

function LogBlock({ title, text, tone }: { title: string; text: string; tone?: 'err' }) {
  return (
    <div>
      <div className="mb-1 text-[11px] text-[#6f767e]">{title}</div>
      <pre className={`max-h-72 overflow-auto rounded border border-[#1f2429] bg-[#0a0d10] p-2 text-[10px] ${tone === 'err' ? 'text-[#b85a62]' : 'text-[#9aa1a8]'}`}>{text || '—'}</pre>
    </div>
  )
}
