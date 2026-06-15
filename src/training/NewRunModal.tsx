import { useEffect, useMemo, useState } from 'react'

import { fs, datasets as datasetsBackend } from '../connections/backend'
import { useDatasetsStore } from '../datasets/store'
import { useConnectionsStore, remotePython, sshTarget, type RemoteSshConnection } from '../connections/store'
import { useTrainingStore } from './store'
import { training } from './backend'
import {
  type LossKind,
  type OptimizerKind,
  type SchedulerKind,
  type TrainingConfig,
  type SlurmConfig,
  type RemoteTrainingCapabilities,
  type RunBackend,
  defaultTrainingConfig,
  defaultSlurmConfig,
} from './types'

const OPTIMIZERS: OptimizerKind[] = ['Adam', 'AdamW', 'SGD', 'RMSprop']
const LOSSES: LossKind[] = ['CrossEntropyLoss', 'BCEWithLogitsLoss', 'MSELoss', 'L1Loss']
const SCHEDULERS: SchedulerKind[] = ['none', 'StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau']

// ── hyperparameter sweep (Phase 18) ──
type SweepKey = 'lr' | 'batch_size' | 'weight_decay' | 'epochs' | 'seed'
const SWEEP_FIELDS: { key: SweepKey; label: string }[] = [
  { key: 'lr', label: 'Learning rate' },
  { key: 'batch_size', label: 'Batch size' },
  { key: 'weight_decay', label: 'Weight decay' },
  { key: 'epochs', label: 'Epochs' },
  { key: 'seed', label: 'Seed' },
]
const SWEEP_SHORT: Record<SweepKey, string> = { lr: 'lr', batch_size: 'bs', weight_decay: 'wd', epochs: 'ep', seed: 'seed' }

function parseValues(raw: string): number[] {
  return raw.split(',').map((s) => Number(s.trim())).filter((n) => Number.isFinite(n))
}

function cartesian(axes: { key: SweepKey; values: number[] }[]): Partial<Record<SweepKey, number>>[] {
  let combos: Partial<Record<SweepKey, number>>[] = [{}]
  for (const ax of axes) {
    const next: Partial<Record<SweepKey, number>>[] = []
    for (const c of combos) for (const v of ax.values) next.push({ ...c, [ax.key]: v })
    combos = next
  }
  return combos
}

function applyCombo(base: TrainingConfig, combo: Partial<Record<SweepKey, number>>): TrainingConfig {
  const t: TrainingConfig = { ...base, optimizer: { ...base.optimizer } }
  if (combo.lr != null) t.optimizer.lr = combo.lr
  if (combo.weight_decay != null) t.optimizer.weight_decay = combo.weight_decay
  if (combo.batch_size != null) t.batch_size = combo.batch_size
  if (combo.epochs != null) t.epochs = combo.epochs
  if (combo.seed != null) t.seed = combo.seed
  return t
}

function comboLabel(combo: Partial<Record<SweepKey, number>>): string {
  return (Object.entries(combo) as [SweepKey, number][])
    .map(([k, v]) => `${SWEEP_SHORT[k]}=${v}`)
    .join(' ')
}

function nextSweepKey(existing: { key: SweepKey }[]): SweepKey {
  const used = new Set(existing.map((s) => s.key))
  return SWEEP_FIELDS.find((f) => !used.has(f.key))?.key ?? 'lr'
}

export default function NewRunModal() {
  const close = useTrainingStore((s) => s.closeNewRun)
  const startRun = useTrainingStore((s) => s.startRun)
  const inspectDataset = useDatasetsStore((s) => s.inspect)

  // Current remote connection (for SLURM backend + persisting its config).
  const currentId = useConnectionsStore((s) => s.currentId)
  const remoteConn = useConnectionsStore((s) => s.saved.find((c) => c.id === s.currentId)) ?? null
  const updateRemote = useConnectionsStore((s) => s.updateRemote)

  const [caps, setCaps] = useState<RemoteTrainingCapabilities | null>(null)
  const [backendKind, setBackendKind] = useState<'local' | 'slurm'>('local')
  const [slurm, setSlurm] = useState<SlurmConfig>(remoteConn?.slurm ?? defaultSlurmConfig())

  // Phase 18 — hyperparameter sweep: each axis is a comma-separated value list;
  // the grid (cartesian product) launches one run per combination.
  const [sweeps, setSweeps] = useState<{ key: SweepKey; raw: string }[]>([])

  const [models, setModels] = useState<string[]>([])
  const [dsList, setDsList] = useState<{ relpath: string; name: string; is_dir: boolean }[]>([])
  const [listErr, setListErr] = useState<string | null>(null)

  const [label, setLabel] = useState('')
  const [modelRelpath, setModelRelpath] = useState('')
  const [datasetRelpath, setDatasetRelpath] = useState('')
  const [targetColumn, setTargetColumn] = useState('')
  const [cfg, setCfg] = useState<TrainingConfig>(defaultTrainingConfig())
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  // Phase 17 — resume: prior runs that have a checkpoints/best.pt to continue from.
  const [resumable, setResumable] = useState<{ run_id: string; run_label: string }[]>([])
  const [resumeId, setResumeId] = useState('')

  // dataset columns come from the datasets-store inspect cache
  const inspect = useDatasetsStore((s) => (datasetRelpath ? s.inspects[datasetRelpath] : undefined))
  const columns = useMemo(() => {
    const d = inspect?.data
    return d && d.kind === 'tabular' && d.ok ? d.columns : []
  }, [inspect])

  useEffect(() => {
    void (async () => {
      try {
        const entries = await fs.list()
        setModels(
          entries
            .filter((e) => !e.is_dir && e.relpath.toLowerCase().endsWith('.mlforge'))
            .map((e) => e.relpath)
            .sort(),
        )
        // Prime the datasets store so onPickDataset's inspect() can find the
        // entry (it looks up the store's entries, not this local list). Without
        // this, picking a dataset left the target dropdown empty.
        await useDatasetsStore.getState().refresh()
        const ds = useDatasetsStore.getState().entries
        setDsList(ds.map((d) => ({ relpath: d.relpath, name: d.name, is_dir: d.is_dir })))
        // Runs that have a best.pt → eligible as a resume source.
        const runs = await training.list()
        setResumable(runs.filter((r) => r.has_checkpoint).map((r) => ({ run_id: r.run_id, run_label: r.run_label })))
      } catch (e) {
        setListErr(e instanceof Error ? e.message : String(e))
      }
    })()
  }, [])

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') close()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [close])

  // Probe the remote host once (sbatch? partitions? gpus?) so we only offer
  // SLURM where it exists and can prefill the partition.
  useEffect(() => {
    let cancelled = false
    setCaps(null)
    void training.capabilities().then((c) => {
      if (cancelled) return
      setCaps(c)
      // No SLURM on this host (or it's not a cluster) → force direct backend so a
      // stale 'slurm' choice from a previous connection can't leak across hosts.
      if (!c.has_slurm) setBackendKind('local')
      setSlurm((prev) => (prev.partition || !c.partitions.length ? prev : { ...prev, partition: c.partitions[0] }))
    }).catch(() => { if (!cancelled) { setCaps(null); setBackendKind('local') } })
    return () => { cancelled = true }
  }, [currentId])

  function onPickDataset(rel: string) {
    setDatasetRelpath(rel)
    setTargetColumn('')
    if (rel) void inspectDataset(rel)
  }

  // default label from model name when nothing typed yet
  const effectiveLabel = label || (modelRelpath ? modelRelpath.split('/').pop()!.replace(/\.mlforge$/i, '') : '')

  const axes = sweeps
    .map((s) => ({ key: s.key, values: parseValues(s.raw) }))
    .filter((a) => a.values.length > 0)
  const combos = cartesian(axes)
  const sweepCount = combos.length

  const canSubmit =
    !!modelRelpath && !!datasetRelpath && !!targetColumn && !submitting && sweepCount <= 64

  async function submit() {
    setError(null)
    setSubmitting(true)
    try {
      const abspath = await datasetsBackend.abspath(datasetRelpath)
      const backend: RunBackend = backendKind === 'slurm' ? { kind: 'slurm', slurm } : { kind: 'local' }
      // Remember the SLURM config on the connection for next time.
      if (backendKind === 'slurm' && remoteConn) updateRemote(remoteConn.id, { slurm })
      const base = {
        modelRelpath,
        datasetRelpath,
        datasetAbspath: abspath,
        targetColumn,
        featureColumns: null, // null = all numeric cols except target
        backend,
        ...(resumeId ? { resumeFrom: `experiments/runs/${resumeId}/checkpoints/best.pt` } : {}),
      }
      if (axes.length === 0) {
        await startRun({ ...base, label: effectiveLabel || 'run', training: cfg })
      } else {
        // One run per grid point, started sequentially so run dirs / ssh don't
        // collide. Distinct labels make them legible in the list + compare view.
        for (const combo of combos) {
          await startRun({
            ...base,
            label: `${effectiveLabel || 'run'} [${comboLabel(combo)}]`,
            training: applyCombo(cfg, combo),
          })
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setSubmitting(false)
    }
  }

  const opt = cfg.optimizer
  const isTabular = inspect?.data?.kind === 'tabular'

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6"
      onClick={(e) => { if (e.target === e.currentTarget) close() }}
    >
      <div className="flex max-h-[92vh] w-full max-w-2xl flex-col overflow-hidden rounded-lg border border-[#1f2429] bg-[#0e1115] shadow-2xl">
        <div className="flex items-center gap-2 border-b border-[#1f2429] px-4 py-3">
          <span className="flex-1 text-sm text-[#e6e8eb]">Neuer Trainings-Run</span>
          <button onClick={close} className="rounded px-2 py-0.5 text-[#7a8088] hover:bg-[#1a1e22] hover:text-[#e6e8eb]">×</button>
        </div>

        <div className="min-h-0 flex-1 space-y-4 overflow-auto px-4 py-3 text-[12px] text-[#cfd3d8]">
          {listErr && <div className="text-[#ff7a85]">{listErr}</div>}

          <Field label="Modell">
            <select value={modelRelpath} onChange={(e) => setModelRelpath(e.target.value)} className={SELECT}>
              <option value="">— wählen —</option>
              {models.map((m) => <option key={m} value={m}>{m}</option>)}
            </select>
            {models.length === 0 && <Hint>Keine .mlforge-Modelle im Workspace. Erst ein Modell speichern.</Hint>}
          </Field>

          <Field label="Datensatz">
            <select value={datasetRelpath} onChange={(e) => onPickDataset(e.target.value)} className={SELECT}>
              <option value="">— wählen —</option>
              {dsList.map((d) => <option key={d.relpath} value={d.relpath}>{d.name}</option>)}
            </select>
            {datasetRelpath && !isTabular && inspect?.data && (
              <Hint warn>Phase 13 trainiert nur tabulare Datensätze (CSV/TSV/Parquet). Anderes Format folgt mit dem Trainings-Graph (Phase 14).</Hint>
            )}
          </Field>

          <Field label="Ziel-Spalte (target)">
            <select value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)} className={SELECT} disabled={!columns.length}>
              <option value="">{columns.length ? '— wählen —' : '(Datensatz wählen)'}</option>
              {columns.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
            <Hint>Features = alle übrigen numerischen Spalten. Verlust bestimmt Klassifikation vs. Regression.</Hint>
          </Field>

          <div className="grid grid-cols-3 gap-3">
            <NumField label="Epochs" value={cfg.epochs} onChange={(v) => setCfg({ ...cfg, epochs: v })} />
            <NumField label="Batch size" value={cfg.batch_size} onChange={(v) => setCfg({ ...cfg, batch_size: v })} />
            <NumField label="Val split" value={cfg.val_split} step={0.05} onChange={(v) => setCfg({ ...cfg, val_split: v })} />
          </div>

          <div className="grid grid-cols-3 gap-3">
            <Field label="Optimizer">
              <select value={opt.kind} onChange={(e) => setCfg({ ...cfg, optimizer: { ...opt, kind: e.target.value as OptimizerKind } })} className={SELECT}>
                {OPTIMIZERS.map((o) => <option key={o} value={o}>{o}</option>)}
              </select>
            </Field>
            <NumField label="Learning rate" value={opt.lr} step={0.0001} onChange={(v) => setCfg({ ...cfg, optimizer: { ...opt, lr: v } })} />
            <NumField label="Weight decay" value={opt.weight_decay} step={0.0001} onChange={(v) => setCfg({ ...cfg, optimizer: { ...opt, weight_decay: v } })} />
          </div>

          <div className="grid grid-cols-3 gap-3">
            <Field label="Loss">
              <select value={cfg.loss.kind} onChange={(e) => setCfg({ ...cfg, loss: { kind: e.target.value as LossKind } })} className={SELECT}>
                {LOSSES.map((l) => <option key={l} value={l}>{l}</option>)}
              </select>
            </Field>
            <Field label="Scheduler">
              <select value={cfg.scheduler.kind} onChange={(e) => setCfg({ ...cfg, scheduler: { kind: e.target.value as SchedulerKind } })} className={SELECT}>
                {SCHEDULERS.map((s) => <option key={s} value={s}>{s}</option>)}
              </select>
            </Field>
            <NumField label="Seed" value={cfg.seed} onChange={(v) => setCfg({ ...cfg, seed: v })} />
          </div>

          <Field label="Label (optional)">
            <input value={label} onChange={(e) => setLabel(e.target.value)} placeholder={effectiveLabel || 'run'} className={SELECT} />
          </Field>

          {resumable.length > 0 && (
            <Field label="Fortsetzen ab Checkpoint (optional)">
              <select value={resumeId} onChange={(e) => setResumeId(e.target.value)} className={SELECT}>
                <option value="">— von vorn trainieren —</option>
                {resumable.map((r) => <option key={r.run_id} value={r.run_id}>{r.run_label || r.run_id}</option>)}
              </select>
              <Hint>Lädt Gewichte + Optimizer aus <code>best.pt</code> des gewählten Runs und trainiert „Epochs" weitere Epochen. Modell-Architektur muss passen.</Hint>
            </Field>
          )}

          <div className="space-y-2 rounded border border-[#1f2429] bg-[#0a0d10] p-3">
            <div className="flex items-baseline gap-2">
              <span className="text-[11px] font-medium text-[#cfd3d8]">Hyperparameter-Tuning</span>
              <span className="text-[10px] text-[#7a8088]">(Grid Search, optional)</span>
            </div>
            <p className="text-[10px] leading-snug text-[#7a8088]">
              Mehrere Werte je Parameter durchprobieren statt einen festen. MLForge startet
              <strong className="text-[#9aa1a8]"> einen Run pro Kombination</strong> aller Parameter (Gitter) — danach im
              Vergleich gegenüberstellbar. Diese Werte überschreiben die Einzelwerte oben.
            </p>

            {sweeps.map((s, i) => {
              const n = parseValues(s.raw).length
              return (
                <div key={i} className="space-y-1.5 rounded border border-[#1f2429] bg-[#0b0e11] p-2">
                  <div className="flex items-center gap-2">
                    <select
                      value={s.key}
                      onChange={(e) => setSweeps(sweeps.map((x, j) => j === i ? { ...x, key: e.target.value as SweepKey } : x))}
                      className={SELECT}
                    >
                      {SWEEP_FIELDS.map((f) => <option key={f.key} value={f.key}>{f.label}</option>)}
                    </select>
                    <span className={`shrink-0 text-[10px] ${n > 0 ? 'text-[#6ab7ff]' : 'text-[#5a6068]'}`}>
                      {n > 0 ? `${n} Werte` : 'keine Werte'}
                    </span>
                    <button onClick={() => setSweeps(sweeps.filter((_, j) => j !== i))} className="shrink-0 rounded px-1.5 py-0.5 text-[11px] text-[#7a8088] hover:bg-[#1a1e22] hover:text-[#ff7a85]">×</button>
                  </div>
                  <label className="block">
                    <span className="mb-1 block text-[10px] text-[#7a8088]">zu testende Werte (komma-getrennt)</span>
                    <input
                      value={s.raw}
                      onChange={(e) => setSweeps(sweeps.map((x, j) => j === i ? { ...x, raw: e.target.value } : x))}
                      placeholder="z. B. 0.01, 0.001, 0.0001"
                      className={`${SELECT} font-mono`}
                    />
                  </label>
                </div>
              )
            })}

            <button
              onClick={() => setSweeps([...sweeps, { key: nextSweepKey(sweeps), raw: '' }])}
              disabled={sweeps.length >= SWEEP_FIELDS.length}
              className="rounded border border-[#1f2429] px-2 py-0.5 text-[10px] text-[#9aa1a8] hover:border-[#3a4148] hover:text-[#e6e8eb] disabled:opacity-40"
            >
              + Parameter hinzufügen
            </button>

            {axes.length === 0 && (
              <p className="text-[10px] text-[#5a6068]">
                Leer = ein einzelner Run mit den Werten oben. Es werden keine Werte automatisch gewählt.
              </p>
            )}

            {/* Live grid preview — count breakdown + first combos as chips. */}
            {axes.length > 0 && (
              <div className="space-y-1.5 rounded border border-[#1f2429] bg-[#0b0e11] p-2">
                <div className="flex items-baseline gap-2 text-[10px]">
                  <span className="font-mono text-[#9aa1a8]">
                    {axes.map((a) => `${a.values.length} ${SWEEP_FIELDS.find((f) => f.key === a.key)!.label}`).join('  ×  ')}
                  </span>
                  <span className={`ml-auto font-medium ${sweepCount > 64 ? 'text-[#ff7a85]' : 'text-[#6ab7ff]'}`}>
                    = {sweepCount} Run{sweepCount === 1 ? '' : 's'}{sweepCount > 64 ? ' · zu viele (max 64)' : ''}
                  </span>
                </div>
                {sweepCount <= 64 && (
                  <div className="flex flex-wrap gap-1">
                    {combos.slice(0, 10).map((c, i) => (
                      <span key={i} className="rounded bg-[#14181c] px-1.5 py-0.5 font-mono text-[9px] text-[#7a8088]">
                        {comboLabel(c)}
                      </span>
                    ))}
                    {combos.length > 10 && <span className="px-1 py-0.5 text-[9px] text-[#5a6068]">+{combos.length - 10} weitere</span>}
                  </div>
                )}
              </div>
            )}
          </div>

          {remoteConn && (
            <BackendSection
              conn={remoteConn}
              caps={caps}
              backendKind={backendKind}
              setBackendKind={setBackendKind}
              slurm={slurm}
              setSlurm={setSlurm}
            />
          )}

          {error && <div className="text-[#ff7a85]">{error}</div>}
        </div>

        <div className="flex items-center justify-end gap-2 border-t border-[#1f2429] px-4 py-3">
          <button onClick={close} className="rounded px-3 py-1 text-[12px] text-[#9aa1a8] hover:text-[#e6e8eb]">Abbrechen</button>
          <button
            onClick={() => void submit()}
            disabled={!canSubmit}
            className="rounded bg-[#13344f] px-3 py-1 text-[12px] text-[#6ab7ff] hover:bg-[#184466] disabled:cursor-not-allowed disabled:opacity-40"
          >
            {submitting ? 'starte…' : sweepCount > 1 ? `${sweepCount} Runs starten` : 'Run starten'}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Backend section: explains WHERE the run executes and WHERE its files live,
// adapting to whether the host is a SLURM cluster, a plain SSH box, or still
// being probed. Partitions are always whatever THIS host's `sinfo` reported —
// never a hardcoded list (cluster naming differs everywhere).
function BackendSection({
  conn, caps, backendKind, setBackendKind, slurm, setSlurm,
}: {
  conn: RemoteSshConnection
  caps: RemoteTrainingCapabilities | null
  backendKind: 'local' | 'slurm'
  setBackendKind: (k: 'local' | 'slurm') => void
  slurm: SlurmConfig
  setSlurm: (s: SlurmConfig) => void
}) {
  const host = sshTarget(conn)
  const root = conn.root.replace(/\/+$/, '')
  const runDir = `${root}/experiments/runs/<id>/`
  const python = remotePython(conn)
  const gpus = caps?.gpu_names ?? []

  return (
    <div className="space-y-3 rounded border border-[#1f2429] bg-[#0a0d10] p-3">
      <div className="flex items-center gap-2">
        <span className="text-[11px] font-medium text-[#cfd3d8]">Wo läuft das Training?</span>
        <span className="rounded bg-[#14181c] px-1.5 py-0.5 font-mono text-[10px] text-[#7a8088]">{host}</span>
      </div>

      {caps === null ? (
        <div className="text-[11px] text-[#7a8088]">Prüfe Fähigkeiten von <code className="text-[#9aa1a8]">{host}</code> (sbatch? GPUs?)…</div>
      ) : !caps.has_slurm ? (
        // Plain SSH host — no scheduler. Be explicit that this isn't a cluster.
        <div className="space-y-2">
          <div className="rounded border border-[#1f2429] bg-[#0b0e11] px-2 py-1.5 text-[11px] text-[#9aa1a8]">
            Kein SLURM auf diesem Host (kein <code>sbatch</code>) — also <strong className="text-[#cfd3d8]">kein HPC-Cluster</strong>.
            Das Training läuft direkt als losgelöster Hintergrund-Prozess (<code>nohup setsid</code>) auf <code className="text-[#9aa1a8]">{host}</code>
            und überlebt das Schließen von MLForge.
          </div>
          <FlowDiagram steps={directSteps(host, runDir, python)} />
          {gpus.length > 0 && <Hint>GPU am Host: {gpus.slice(0, 4).join(', ')} — wird automatisch genutzt, wenn torch CUDA sieht.</Hint>}
        </div>
      ) : (
        // SLURM cluster — let the user choose direct vs. batch, with a flow for each.
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <BackendCard
              active={backendKind === 'local'}
              onClick={() => setBackendKind('local')}
              title="Direkt"
              desc="Prozess auf dem Login-Knoten. Sofort, aber teilt sich die Login-Ressourcen."
            />
            <BackendCard
              active={backendKind === 'slurm'}
              onClick={() => setBackendKind('slurm')}
              title="SLURM-Job"
              desc="In die Queue (sbatch). Läuft auf einem Compute-Knoten mit eigenen Ressourcen."
            />
          </div>

          {backendKind === 'local' ? (
            <FlowDiagram steps={directSteps(host, runDir, python)} />
          ) : (
            <>
              <FlowDiagram steps={slurmSteps(host, runDir, slurm.partition || '<partition>')} />

              <div className="grid grid-cols-3 gap-3">
                <Field label={`Partition${caps.partitions.length ? ` (${caps.partitions.length} erkannt)` : ''}`}>
                  {caps.partitions.length ? (
                    <select value={slurm.partition} onChange={(e) => setSlurm({ ...slurm, partition: e.target.value })} className={SELECT}>
                      <option value="">— wählen —</option>
                      {caps.partitions.map((p) => <option key={p} value={p}>{p}</option>)}
                    </select>
                  ) : (
                    <input value={slurm.partition} onChange={(e) => setSlurm({ ...slurm, partition: e.target.value })} placeholder="sinfo lieferte nichts — manuell" className={SELECT} />
                  )}
                </Field>
                <Field label="Time (HH:MM:SS)">
                  <input value={slurm.time} onChange={(e) => setSlurm({ ...slurm, time: e.target.value })} className={SELECT} />
                </Field>
                <Field label="Memory">
                  <input value={slurm.mem} onChange={(e) => setSlurm({ ...slurm, mem: e.target.value })} className={SELECT} />
                </Field>
              </div>
              <div className="grid grid-cols-3 gap-3">
                <NumField label="CPUs/task" value={slurm.cpus_per_task} onChange={(v) => setSlurm({ ...slurm, cpus_per_task: v })} />
                <Field label="GRES (optional)">
                  <input value={slurm.gres ?? ''} onChange={(e) => setSlurm({ ...slurm, gres: e.target.value })} placeholder="gpu:1" className={SELECT} />
                </Field>
                <Field label="Account (optional)">
                  <input value={slurm.account ?? ''} onChange={(e) => setSlurm({ ...slurm, account: e.target.value })} className={SELECT} />
                </Field>
              </div>
              <Hint>
                GRES-Format ist clusterabhängig (z. B. <code>gpu:&lt;typ&gt;:&lt;n&gt;</code>) — nur für GPU-Partitionen nötig, sonst leer lassen.
                {gpus.length > 0 && <> Am Host gesehen: {gpus.slice(0, 4).join(', ')}.</>}
              </Hint>
              <Field label="module load (eine pro Zeile)">
                <textarea
                  value={slurm.modules.join('\n')}
                  onChange={(e) => setSlurm({ ...slurm, modules: e.target.value.split('\n').map((l) => l.trim()).filter(Boolean) })}
                  rows={2}
                  placeholder={'CUDA/12.4.0'}
                  className={`${SELECT} font-mono`}
                />
              </Field>
              <Field label="Pre-Run-Script (bash, optional)">
                <textarea
                  value={slurm.pre_run_script ?? ''}
                  onChange={(e) => setSlurm({ ...slurm, pre_run_script: e.target.value })}
                  rows={2}
                  placeholder={'export OMP_NUM_THREADS=8'}
                  className={`${SELECT} font-mono`}
                />
              </Field>
            </>
          )}
          <Hint>Python: <code className="text-[#9aa1a8]">{python}</code> — pro Verbindung editierbar (Stift in der Verbindungsliste). Muss torch (+pandas) haben.</Hint>
        </div>
      )}
    </div>
  )
}

function BackendCard({ active, onClick, title, desc }: { active: boolean; onClick: () => void; title: string; desc: string }) {
  return (
    <button
      onClick={onClick}
      className={`rounded border px-2.5 py-2 text-left transition-colors ${active ? 'border-[#6ab7ff] bg-[#13344f]/40' : 'border-[#1f2429] bg-[#0b0e11] hover:border-[#3a4148]'}`}
    >
      <div className="flex items-center gap-1.5">
        <span className={`inline-block h-2 w-2 rounded-full ${active ? 'bg-[#6ab7ff]' : 'bg-[#3a4148]'}`} />
        <span className={`text-[12px] font-medium ${active ? 'text-[#e6e8eb]' : 'text-[#cfd3d8]'}`}>{title}</span>
      </div>
      <div className="mt-1 text-[10px] leading-snug text-[#7a8088]">{desc}</div>
    </button>
  )
}

type FlowStep = { tag: string; title: string; lines: string[] }

function directSteps(host: string, runDir: string, python: string): FlowStep[] {
  return [
    { tag: 'lokal', title: 'Dein Rechner', lines: ['model.py + run.json', 'aus dem Graph generiert'] },
    { tag: host, title: 'Host (Prozess)', lines: [runDir, `${python.split('/').pop()} -u train.py`, 'nohup setsid'] },
    { tag: 'live', title: 'Ergebnis', lines: ['events.jsonl', 'best.pt · stdout/err', '→ Live-Status hier'] },
  ]
}

function slurmSteps(host: string, runDir: string, partition: string): FlowStep[] {
  return [
    { tag: 'lokal', title: 'Dein Rechner', lines: ['model.py + run.json', 'aus dem Graph generiert'] },
    { tag: host, title: 'Login-Knoten', lines: [runDir, 'train.sbatch', 'sbatch → Queue'] },
    { tag: partition, title: 'Compute-Knoten', lines: ['Job in der Partition', 'events.jsonl · best.pt', '→ Live-Status hier'] },
  ]
}

function FlowDiagram({ steps }: { steps: FlowStep[] }) {
  return (
    <div className="flex items-stretch gap-1 overflow-x-auto">
      {steps.map((s, i) => (
        <div key={i} className="flex items-stretch gap-1">
          <div className="min-w-[120px] flex-1 rounded border border-[#1f2429] bg-[#0b0e11] px-2 py-1.5">
            <div className="mb-1 truncate font-mono text-[9px] uppercase tracking-wide text-[#6ab7ff]">{s.tag}</div>
            <div className="text-[10px] font-medium text-[#cfd3d8]">{s.title}</div>
            {s.lines.map((l, j) => (
              <div key={j} className="truncate font-mono text-[9px] text-[#7a8088]" title={l}>{l}</div>
            ))}
          </div>
          {i < steps.length - 1 && <div className="flex items-center px-0.5 text-[#3a4148]">→</div>}
        </div>
      ))}
    </div>
  )
}

const SELECT = 'w-full rounded border border-[#1f2429] bg-[#14181c] px-2 py-1 text-[12px] text-[#e6e8eb] focus:border-[#6ab7ff] focus:outline-none'

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[11px] text-[#7a8088]">{label}</span>
      {children}
    </label>
  )
}

function NumField({ label, value, onChange, step = 1 }: { label: string; value: number; onChange: (v: number) => void; step?: number }) {
  return (
    <Field label={label}>
      <input
        type="number"
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className={SELECT}
      />
    </Field>
  )
}

function Hint({ children, warn }: { children: React.ReactNode; warn?: boolean }) {
  return <span className={`mt-1 block text-[10px] ${warn ? 'text-[#e6c34a]' : 'text-[#5a6068]'}`}>{children}</span>
}
