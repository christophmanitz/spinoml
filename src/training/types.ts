import type { DatasetFingerprint } from '../datasets/types'
import type { CodeTrustEntry } from '../trust/gate'

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

/** One output head of a multitask model. `output` matches a model `Output`
 *  node's name (the key the generated forward() returns in its dict); '' means
 *  the model's sole/default output. Each head has its own target column + loss,
 *  and contributes `weight * loss` to the combined objective. When
 *  TrainingConfig.heads is set the trainer runs multitask; otherwise it falls
 *  back to the single `loss` + dataset.target_column path. */
export type Head = {
  output: string
  target: string
  loss: LossKind
  weight: number
  label_smoothing?: number
}

export type SplitStrategy = 'random' | 'stratified' | 'grouped' | 'time-based' | 'predefined'

export const SPLIT_STRATEGIES: SplitStrategy[] = ['random', 'stratified', 'grouped', 'time-based', 'predefined']

export type TrainingConfig = {
  epochs: number
  batch_size: number
  val_split: number
  seed: number
  /** Phase 19: WHICH splitting method the run uses. Only 'random' is
   *  implemented by the trainer today; anything else is frozen into run.json
   *  AND the trainer fails loudly rather than silently falling back to random
   *  — we never silently change a user's chosen strategy. */
  split_strategy: SplitStrategy
  log_every_n_steps: number
  /** DataLoader knobs (from the DataLoader node; sensible defaults otherwise). */
  shuffle?: boolean
  num_workers?: number
  drop_last?: boolean
  /** Run validation every N epochs (TrainLoop node; default 1 = every epoch). */
  val_every_n_epochs?: number
  /** Accumulate grads over N batches before optimizer.step (TrainLoop node; default 1). */
  gradient_accumulation_steps?: number
  optimizer: OptimizerConfig
  loss: { kind: LossKind; label_smoothing?: number }
  /** Multitask: one entry per output head. When present (≥1) the trainer routes
   *  each model output to its own target + loss and optimizes their weighted sum;
   *  `loss` above is then only the single-task fallback. */
  heads?: Head[]
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
  /** Phase 18 — content-derived stable identifier captured at inspect time,
   *  frozen into run.json so the run records exactly WHICH data it trained on,
   *  independent of path/name. Absent when the dataset was never inspected or
   *  the workspace has no sidecar (remote pre-12b). */
  fingerprint?: DatasetFingerprint | null
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
  /** Phase 20 — immutable launch snapshot: content-hashes of the frozen
   *  model.spinoml (graph) + generated model.py + the preprocessing (DataOp)
   *  steps from the graph. train.py re-verifies the RUN-DIR copies against
   *  these hashes before training, so the running experiment cannot drift
   *  from the bytes that were locked in at launch. */
  snapshot?: RunSnapshot
  /** Phase 17 — resume weights/optimizer from a prior run's checkpoint. A
   *  workspace-relative path (e.g. experiments/runs/<id>/checkpoints/best.pt)
   *  resolved on the executor host, or an absolute path. */
  resume_from?: string
  /** External validation: when true, train.py loads `validate.checkpoint_from`
   *  and evaluates the WHOLE `dataset` once (no training) → eval.summary metrics. */
  eval_only?: boolean
  validate?: ValidateConfig
}

/** Phase 20 — content-addressed record of what a run will execute. */
export type RunSnapshot = {
  /** snapshot schema version (bump when the shape changes; train.py validates). */
  version: number
  /** sha256 of the model.spinoml bytes frozen into the run dir. */
  graph_sha256: string
  /** sha256 of the generated model.py bytes frozen into the run dir. */
  model_py_sha256: string
  /** DataOp preprocessing scripts extracted from the graph at launch. */
  preprocessing: PreprocessingStep[]
  /** Phase 43 — provenance of every intentional-arbitrary-code blob (Custom
   *  layer / DataOp script) that ran in this experiment, with the human approval
   *  origin + timestamp (or 'unrecorded'/null). Additive to the frozen bytes:
   *  train.py's `_verify_snapshot` reads only the two hashes, so this cannot
   *  break snapshot verification. */
  code_trust: CodeTrustEntry[]
}

/** A DataOp step baked into the graph (Phase 20): the script that produced
 *  the dataset the run consumes, plus its binding (input/output/mode/cache). */
export type PreprocessingStep = {
  node: string
  script: string
  input_dataset: string
  output_name: string
  mode: 'shell' | 'slurm'
  cache: boolean
}

/** External-validation config carried in an eval run's run.json. */
export type ValidateConfig = {
  /** Workspace-relative path to the trained checkpoint to validate (best.pt). */
  checkpoint_from: string
  /** The source run this checkpoint came from (for the UI banner). */
  source_run?: string
  /** How the external dataset was adapted to the model (for the UI summary). */
  adapter?: AdapterSpec
}

/** Maps an external dataset's columns/branches onto the model's trained schema.
 *  `feature_columns`/`target_column` (on the eval run's DatasetConfig) hold the
 *  EXTERNAL column names chosen here; the eval loader consumes them directly so a
 *  rename/select/reorder needs no materialization. `column_map` + `branch_map` are
 *  for display + the Data-canvas escalation. `unmatched` lists model roles with no
 *  external column (the user must map them or validation can't run). */
export type AdapterSpec = {
  /** model role (feature name / 'target' / branch) → chosen external column */
  column_map: Record<string, string>
  /** for manifest models: model branch → external manifest branch */
  branch_map?: Record<string, string>
  /** model roles with no external match yet */
  unmatched: string[]
  /** how the mapping was produced, for provenance */
  mode: 'auto' | 'manual' | 'hybrid' | 'pipeline'
  /** if the user escalated to a Data-canvas pipeline, the adapted dataset relpath */
  adapted_from?: string
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
  /** External-validation run (run.json eval_only) — badged distinctly in the list. */
  eval_only?: boolean
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
    split_strategy: 'random',
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
