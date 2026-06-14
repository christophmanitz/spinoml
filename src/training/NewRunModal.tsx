import { useEffect, useMemo, useState } from 'react'

import { fs, datasets as datasetsBackend } from '../connections/backend'
import { useDatasetsStore } from '../datasets/store'
import { useConnectionsStore } from '../connections/store'
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
        const ds = await datasetsBackend.list()
        setDsList(ds.map((d) => ({ relpath: d.relpath, name: d.name, is_dir: d.is_dir })))
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
    void training.capabilities().then((c) => {
      if (cancelled) return
      setCaps(c)
      setSlurm((prev) => (prev.partition || !c.partitions.length ? prev : { ...prev, partition: c.partitions[0] }))
    }).catch(() => { if (!cancelled) setCaps(null) })
    return () => { cancelled = true }
  }, [currentId])

  function onPickDataset(rel: string) {
    setDatasetRelpath(rel)
    setTargetColumn('')
    if (rel) void inspectDataset(rel)
  }

  // default label from model name when nothing typed yet
  const effectiveLabel = label || (modelRelpath ? modelRelpath.split('/').pop()!.replace(/\.mlforge$/i, '') : '')

  const canSubmit =
    !!modelRelpath && !!datasetRelpath && !!targetColumn && !submitting

  async function submit() {
    setError(null)
    setSubmitting(true)
    try {
      const abspath = await datasetsBackend.abspath(datasetRelpath)
      const backend: RunBackend = backendKind === 'slurm' ? { kind: 'slurm', slurm } : { kind: 'local' }
      // Remember the SLURM config on the connection for next time.
      if (backendKind === 'slurm' && remoteConn) updateRemote(remoteConn.id, { slurm })
      await startRun({
        label: effectiveLabel || 'run',
        modelRelpath,
        datasetRelpath,
        datasetAbspath: abspath,
        targetColumn,
        featureColumns: null, // null = all numeric cols except target
        training: cfg,
        backend,
      })
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

          {caps?.has_slurm && (
            <div className="space-y-3 rounded border border-[#1f2429] bg-[#0a0d10] p-3">
              <Field label="Backend">
                <select value={backendKind} onChange={(e) => setBackendKind(e.target.value as 'local' | 'slurm')} className={SELECT}>
                  <option value="local">Direkt (ssh, nohup setsid)</option>
                  <option value="slurm">SLURM (sbatch)</option>
                </select>
                {caps.gpu_names.length > 0 && <Hint>GPUs erkannt: {caps.gpu_names.slice(0, 4).join(', ')}</Hint>}
              </Field>

              {backendKind === 'slurm' && (
                <>
                  <div className="grid grid-cols-3 gap-3">
                    <Field label="Partition">
                      {caps.partitions.length ? (
                        <select value={slurm.partition} onChange={(e) => setSlurm({ ...slurm, partition: e.target.value })} className={SELECT}>
                          <option value="">— wählen —</option>
                          {caps.partitions.map((p) => <option key={p} value={p}>{p}</option>)}
                        </select>
                      ) : (
                        <input value={slurm.partition} onChange={(e) => setSlurm({ ...slurm, partition: e.target.value })} className={SELECT} />
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
                    <Field label="GRES (z.B. gpu:1)">
                      <input value={slurm.gres ?? ''} onChange={(e) => setSlurm({ ...slurm, gres: e.target.value })} className={SELECT} />
                    </Field>
                    <Field label="Account">
                      <input value={slurm.account ?? ''} onChange={(e) => setSlurm({ ...slurm, account: e.target.value })} className={SELECT} />
                    </Field>
                  </div>
                  <Field label="module load (eine pro Zeile)">
                    <textarea
                      value={slurm.modules.join('\n')}
                      onChange={(e) => setSlurm({ ...slurm, modules: e.target.value.split('\n').map((l) => l.trim()).filter(Boolean) })}
                      rows={2}
                      placeholder={'Python/3.11.5\nCUDA/12.4.0'}
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
                  <Hint>Python: <code className="text-[#9aa1a8]">{remoteConn ? (remoteConn.python || 'python') : 'python'}</code> — im Runs-Tab editierbar. Muss torch (+pandas) haben (ggf. via module load).</Hint>
                </>
              )}
            </div>
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
            {submitting ? 'starte…' : 'Run starten'}
          </button>
        </div>
      </div>
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
