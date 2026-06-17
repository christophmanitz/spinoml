// Shared types for the Phase 13 training-run system. These mirror the
// run.json schema written into experiments/runs/<run_id>/ and the events.jsonl
// stream (see TODO.md "Phase 13").

export type OptimizerKind = 'Adam' | 'AdamW' | 'SGD' | 'RMSprop'
export type LossKind = 'CrossEntropyLoss' | 'BCEWithLogitsLoss' | 'MSELoss' | 'L1Loss'
export type SchedulerKind = 'none' | 'StepLR' | 'CosineAnnealingLR' | 'ReduceLROnPlateau'

export type OptimizerConfig = {
  kind: OptimizerKind
  lr: number
  weight_decay: number
  momentum?: number
}

export type CallbackConfig = { kind: string } & Record<string, unknown>

export type TrainingConfig = {
  epochs: number
  batch_size: number
  val_split: number
  seed: number
  log_every_n_steps: number
  optimizer: OptimizerConfig
  loss: { kind: LossKind }
  scheduler: { kind: SchedulerKind } & Record<string, unknown>
  /** Phase 14: extra metrics computed each val pass (accuracy/f1/mse/…). */
  metrics?: string[]
  /** Phase 14: early-stopping / grad-clip / AMP, emitted by the training graph. */
  callbacks?: CallbackConfig[]
}

/** Phase 17 — SLURM batch parameters, frozen into run.json + emitted as the
 *  #SBATCH header of train.sbatch on the remote. */
export type SlurmConfig = {
  partition: string
  /** wall-clock limit, HH:MM:SS */
  time: string
  /** e.g. "32G" */
  mem: string
  cpus_per_task: number
  /** e.g. "gpu:1" or "gpu:a100:1"; empty = no gres line */
  gres?: string
  account?: string
  qos?: string
  /** `module load …` lines run before python */
  modules: string[]
  /** free-text bash run before python (exports etc.) */
  pre_run_script?: string
}

/** How a run is launched. 'local' = direct (nohup setsid), whether the
 *  connection is local or remote-ssh. 'slurm' = sbatch on a remote cluster. */
export type RunBackend =
  | { kind: 'local' }
  | { kind: 'slurm'; slurm: SlurmConfig }

export type DatasetConfig = {
  /** Absolute path on the executor host (Phase 13 = local). */
  path: string
  /** Workspace-relative path, kept for display + future remote rsync. */
  relpath: string
  // 'manifest' = paired graph dataset (e.g. ligand+protein); target lives in the
  // .manifest itself, so feature_columns/target_column are unused for it.
  kind: 'tabular' | 'manifest'
  feature_columns: string[] | null
  target_column: string
}

export type RunConfig = {
  run_id: string
  run_label: string
  created_at: string
  status: 'queued'
  model_path: string
  backend: RunBackend
  dataset: DatasetConfig
  training: TrainingConfig
  /** Phase 17 — resume weights/optimizer from a prior run's checkpoint. A
   *  workspace-relative path (e.g. experiments/runs/<id>/checkpoints/best.pt)
   *  resolved on the executor host, or an absolute path. */
  resume_from?: string
}

export function defaultSlurmConfig(): SlurmConfig {
  return {
    partition: '',
    time: '04:00:00',
    mem: '32G',
    cpus_per_task: 8,
    gres: '',
    account: '',
    qos: '',
    modules: [],
    pre_run_script: '',
  }
}

// ── Returned by the Rust executor ──

export type RunSummary = {
  run_id: string
  run_label: string
  model_path: string
  dataset_path: string
  created_at: string
  status: string
  epochs: number
  best_val_loss: number | null
  alive: boolean
  /** Phase 17 — whether checkpoints/best.pt exists (→ resumable / promotable). */
  has_checkpoint: boolean
}

/** A snapshot from nvidia-smi on the executor host (hardware strip). */
export type GpuStat = {
  index: number
  name: string
  util_pct: number
  mem_used_mb: number
  mem_total_mb: number
  temp_c: number
}

export type RunStatus = {
  status: string
  alive: boolean
  pid: number | null
}

/** Phase 17 — what the remote host can do, probed once per connection. */
export type RemoteTrainingCapabilities = {
  has_slurm: boolean
  has_gpu: boolean
  partitions: string[]
  gpu_names: string[]
}

export type TrainingEvent = {
  t: string
  kind: string
} & Record<string, unknown>

export const RUNNING_STATES = new Set(['queued', 'running'])

export function isTerminal(status: string): boolean {
  return !RUNNING_STATES.has(status)
}

export function defaultTrainingConfig(): TrainingConfig {
  return {
    epochs: 50,
    batch_size: 32,
    val_split: 0.2,
    seed: 42,
    log_every_n_steps: 10,
    optimizer: { kind: 'Adam', lr: 1e-3, weight_decay: 0 },
    loss: { kind: 'CrossEntropyLoss' },
    scheduler: { kind: 'none' },
  }
}

/** `<iso-compact>_<slug>_<short-rand>` so FS sort == chronological. */
export function makeRunId(label: string): string {
  const iso = new Date().toISOString().replace(/[:.]/g, '-').replace('Z', '').slice(0, 19)
  const slug = (label || 'run')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 32) || 'run'
  const rand = Math.random().toString(36).slice(2, 6)
  return `${iso}_${slug}_${rand}`
}
