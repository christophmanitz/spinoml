import { create } from 'zustand'

import { isTauri } from '../workspace/tauri-fs'
import { parseFile } from '../persistence/file'
import { generateFromSnapshot } from '../codegen/generator'
import { fs } from '../connections/backend'
import { getCurrentConnection } from '../connections/store'
import { training } from './backend'
import {
  type RunSummary,
  type RunConfig,
  type RunBackend,
  type DatasetConfig,
  type TrainingConfig,
  makeRunId,
  RUNNING_STATES,
} from './types'

export type NewRunInput = {
  label: string
  /** Workspace-relative path to the .mlforge model. */
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

type TrainingState = {
  runs: RunSummary[]
  listLoading: boolean
  listError: string | null
  selectedRunId: string | null
  newRunOpen: boolean
  /** Run ids selected for multi-run compare (Phase 15.3). */
  compareIds: string[]
  compareOpen: boolean

  refresh: () => Promise<void>
  select: (runId: string | null) => void
  toggleCompare: (runId: string) => void
  clearCompare: () => void
  openCompare: () => void
  closeCompare: () => void
  openNewRun: () => void
  closeNewRun: () => void
  startRun: (input: NewRunInput) => Promise<string>
  stopRun: (runId: string) => Promise<void>
  deleteRun: (runId: string) => Promise<void>
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
  openNewRun: () => set({ newRunOpen: true }),
  closeNewRun: () => set({ newRunOpen: false }),

  startRun: async (input) => {
    // Generate model.py from the frozen .mlforge snapshot (pure codegen).
    const modelContent = await fs.read(input.modelRelpath)
    const modelPy = generateFromSnapshot(parseFile(modelContent)).code

    const runId = makeRunId(input.label)
    const dataset: DatasetConfig = {
      path: input.datasetAbspath,
      relpath: input.datasetRelpath,
      kind: 'tabular',
      feature_columns: input.featureColumns,
      target_column: input.targetColumn,
    }
    const config: RunConfig = {
      run_id: runId,
      run_label: input.label,
      created_at: new Date().toISOString(),
      status: 'queued',
      model_path: input.modelRelpath,
      backend: input.backend ?? { kind: 'local' },
      dataset,
      training: input.training,
      ...(input.resumeFrom ? { resume_from: input.resumeFrom } : {}),
    }
    await training.start(runId, JSON.stringify(config, null, 2), modelContent, modelPy)
    await get().refresh()
    set({ selectedRunId: runId, newRunOpen: false })
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
