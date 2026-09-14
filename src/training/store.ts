import { create } from 'zustand'

import { isTauri } from '../workspace/tauri-fs'
import { parseFile } from '../persistence/file'
import { generateFromSnapshot } from '../codegen/generator'
import { fs } from '../connections/backend'
import { getCurrentConnection } from '../connections/store'
import { useDatasetsStore } from '../datasets/store'
import type { DatasetFingerprint } from '../datasets/types'
import { training } from './backend'
import { buildRunSnapshot } from './snapshot'
import {
  type RunSummary,
  type RunConfig,
  type RunBackend,
  type DatasetConfig,
  type TrainingConfig,
  type AdapterSpec,
  type Head,
  makeRunId,
  RUNNING_STATES,
} from './types'

export type NewRunInput = {
  label: string
  /** Workspace-relative path to the .spinoml model. */
  modelRelpath: string
  /** Workspace-relative path to the dataset. */
  datasetRelpath: string
  /** Absolute dataset path on the executor host. */
  datasetAbspath: string
  targetColumn: string
  featureColumns: string[] | null
  training: TrainingConfig
  /** Launch backend; defaults to direct ({kind:'local'}). */
  backend?: RunBackend
  /** Phase 17 — resume weights/optimizer from a prior run's checkpoint
   *  (workspace-relative path). */
  resumeFrom?: string
}

/** External validation: evaluate a FINISHED run's checkpoint on a foreign dataset.
 *  Reuses the source run's model (architecture + generated module) so weights load
 *  into the exact same graph; the adapter chose which external columns play each
 *  feature/target role (in the trained order). */
export type EvalRunInput = {
  label: string
  /** The finished run whose model + best.pt we validate. */
  sourceRunId: string
  /** Workspace-relative external dataset + its absolute path on the executor. */
  datasetRelpath: string
  datasetAbspath: string
  /** External column names mapped to the model's features (in trained order). */
  featureColumns: string[] | null
  /** External target column (ground truth). */
  targetColumn: string
  /** Validate only THESE output heads (e.g. just classification when the external
   *  set has no affinity target). Defaults to the source run's full head set. */
  heads?: Head[]
  adapter?: AdapterSpec
  backend?: RunBackend
}

/** Prefill for the New-Run dialog when launching from the training graph: the
 *  compiled plan fills model/dataset/target/hyperparameters, and the user still
 *  gets the dialog's backend/SLURM/sweep/resume knobs on top. */
export type NewRunPrefill = {
  label?: string
  modelRelpath: string
  datasetRelpath: string
  targetColumn: string
  featureColumns: string[] | null
  training: TrainingConfig
}

type TrainingState = {
  runs: RunSummary[]
  listLoading: boolean
  listError: string | null
  selectedRunId: string | null
  newRunOpen: boolean
  /** Source run id for the External-Validation dialog (null = closed). */
  evalSourceId: string | null
  /** Set when the New-Run dialog was opened from the training graph. */
  newRunPrefill: NewRunPrefill | null
  /** Run ids selected for multi-run compare (Phase 15.3). */
  compareIds: string[]
  compareOpen: boolean

  refresh: () => Promise<void>
  select: (runId: string | null) => void
  toggleCompare: (runId: string) => void
  clearCompare: () => void
  openCompare: () => void
  closeCompare: () => void
  openNewRun: (prefill?: NewRunPrefill) => void
  closeNewRun: () => void
  openEvalRun: (sourceRunId: string) => void
  closeEvalRun: () => void
  startRun: (input: NewRunInput) => Promise<string>
  startEvalRun: (input: EvalRunInput) => Promise<string>
  stopRun: (runId: string) => Promise<void>
  deleteRun: (runId: string) => Promise<void>
}

/** Phase 18 — pull the cached content fingerprint for a dataset from the
 *  datasets store (populated by /dataset/inspect at DatasetExplorer open).
 *  Returns undefined when the dataset was never inspected (remote pre-12b,
 *  or user skipped the inspector) — run.json still carries it, just without
 *  a fingerprint. */
function cachedFingerprint(relpath: string): DatasetFingerprint | null {
  return useDatasetsStore.getState().inspects[relpath]?.data?.fingerprint ?? null
}

// Single shared poller — refreshes the list while any run is still alive so the
// UI tracks queued→running→done without the user clicking refresh. NON-overlapping
// (each refresh is awaited before the next is scheduled): on a remote ssh
// connection a refresh is an ssh round-trip that can take seconds, and a plain
// setInterval would stack those up and saturate the connection. Slower cadence
// on remote for the same reason.
let pollTimer: ReturnType<typeof setTimeout> | null = null

function pollDelay(): number {
  return getCurrentConnection().kind === 'remote-ssh' ? 5000 : 2000
}

// Called at the end of every refresh(). Keeps exactly one pending tick while a
// run is active. The tick nulls the timer THEN calls refresh(), so no new tick
// is scheduled until that refresh finishes and re-enters here — i.e. ssh calls
// can never overlap no matter how slow the connection is.
function syncPolling(get: () => TrainingState) {
  const anyActive = get().runs.some((r) => RUNNING_STATES.has(r.status) || r.alive)
  if (!anyActive) {
    if (pollTimer !== null) { clearTimeout(pollTimer); pollTimer = null }
    return
  }
  if (pollTimer !== null) return // a tick is already pending
  pollTimer = setTimeout(() => {
    pollTimer = null
    void get().refresh()
  }, pollDelay())
}

export const useTrainingStore = create<TrainingState>((set, get) => ({
  runs: [],
  listLoading: false,
  listError: null,
  selectedRunId: null,
  newRunOpen: false,
  evalSourceId: null,
  newRunPrefill: null,
  compareIds: [],
  compareOpen: false,

  refresh: async () => {
    if (!isTauri()) {
      set({ runs: [], listError: 'Training braucht Tauri (echtes Dateisystem).' })
      return
    }
    set({ listLoading: true, listError: null })
    try {
      const runs = await training.list()
      set({ runs, listLoading: false })
      syncPolling(get)
    } catch (e) {
      set({ listLoading: false, listError: e instanceof Error ? e.message : String(e) })
    }
  },

  select: (runId) => set({ selectedRunId: runId }),
  toggleCompare: (runId) => set((s) => ({
    compareIds: s.compareIds.includes(runId)
      ? s.compareIds.filter((id) => id !== runId)
      : [...s.compareIds, runId],
  })),
  clearCompare: () => set({ compareIds: [] }),
  openCompare: () => set({ compareOpen: true }),
  closeCompare: () => set({ compareOpen: false }),
  openNewRun: (prefill) => set({ newRunOpen: true, newRunPrefill: prefill ?? null }),
  closeNewRun: () => set({ newRunOpen: false, newRunPrefill: null }),
  openEvalRun: (sourceRunId) => set({ evalSourceId: sourceRunId }),
  closeEvalRun: () => set({ evalSourceId: null }),

  startRun: async (input) => {
    // Generate model.py from the frozen .spinoml snapshot (pure codegen).
    const modelContent = await fs.read(input.modelRelpath)
    const modelPy = generateFromSnapshot(parseFile(modelContent)).code

    const runId = makeRunId(input.label)
    // A .manifest is a paired graph dataset; everything else is tabular here.
    const fingerprint = cachedFingerprint(input.datasetRelpath)
    const dataset: DatasetConfig = {
      path: input.datasetAbspath,
      relpath: input.datasetRelpath,
      kind: input.datasetRelpath.toLowerCase().endsWith('.manifest') ? 'manifest' : 'tabular',
      feature_columns: input.featureColumns,
      target_column: input.targetColumn,
      ...(fingerprint ? { fingerprint } : {}),
    }
    const snapshot = await buildRunSnapshot(modelContent, modelPy)
    const config: RunConfig = {
      run_id: runId,
      run_label: input.label,
      created_at: new Date().toISOString(),
      status: 'queued',
      model_path: input.modelRelpath,
      backend: input.backend ?? { kind: 'local' },
      dataset,
      training: input.training,
      snapshot,
      ...(input.resumeFrom ? { resume_from: input.resumeFrom } : {}),
    }
    await training.start(runId, JSON.stringify(config, null, 2), modelContent, modelPy)
    await get().refresh()
    set({ selectedRunId: runId, newRunOpen: false })
    return runId
  },

  startEvalRun: async (input) => {
    // Reuse the SOURCE run's FROZEN model (architecture + generated module) so the
    // checkpoint loads into the exact same graph, and its training config (heads/
    // loss/metrics) so eval computes the same per-head metrics. No new Rust command:
    // an eval run is a normal run with eval_only + validate, launched via training.start.
    let modelPy: string
    let modelSpinoml: string
    let srcTraining: TrainingConfig
    try {
      modelPy = await training.readFile(input.sourceRunId, 'model.py')
      modelSpinoml = await training.readFile(input.sourceRunId, 'model.spinoml')
      const srcCfg = JSON.parse(await training.readFile(input.sourceRunId, 'run.json')) as RunConfig
      srcTraining = srcCfg.training
    } catch (e) {
      throw new Error(`could not read source run '${input.sourceRunId}': ${e instanceof Error ? e.message : String(e)}`, { cause: e })
    }

    const runId = makeRunId(`val ${input.label || 'run'}`)
    const isManifest = input.datasetRelpath.toLowerCase().endsWith('.manifest')
    const fingerprint = cachedFingerprint(input.datasetRelpath)
    const dataset: DatasetConfig = {
      path: input.datasetAbspath,
      relpath: input.datasetRelpath,
      kind: isManifest ? 'manifest' : 'tabular',
      feature_columns: input.featureColumns,
      target_column: input.targetColumn,
      ...(fingerprint ? { fingerprint } : {}),
    }
    const config: RunConfig = {
      run_id: runId,
      run_label: input.label || 'externe Validierung',
      created_at: new Date().toISOString(),
      status: 'queued',
      model_path: `experiments/runs/${input.sourceRunId}/model.spinoml`,
      backend: input.backend ?? { kind: 'local' },
      dataset,
      // Whole external set, no train/val split; epochs are ignored in eval-only.
      // `heads` (when given) restricts validation to outputs that have a target in
      // the external set — e.g. only the classification head when there's no affinity.
      training: { ...srcTraining, val_split: 0, ...(input.heads ? { heads: input.heads } : {}) },
      snapshot: await buildRunSnapshot(modelSpinoml, modelPy),
      eval_only: true,
      validate: {
        checkpoint_from: `experiments/runs/${input.sourceRunId}/checkpoints/best.pt`,
        source_run: input.sourceRunId,
        ...(input.adapter ? { adapter: input.adapter } : {}),
      },
    }
    await training.start(runId, JSON.stringify(config, null, 2), modelSpinoml, modelPy)
    await get().refresh()
    set({ selectedRunId: runId })
    return runId
  },

  stopRun: async (runId) => {
    await training.stop(runId)
    await get().refresh()
  },

  deleteRun: async (runId) => {
    await training.remove(runId)
    set((s) => ({
      selectedRunId: s.selectedRunId === runId ? null : s.selectedRunId,
      compareIds: s.compareIds.filter((id) => id !== runId),
    }))
    await get().refresh()
  },
}))
