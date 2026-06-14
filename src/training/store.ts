import { create } from 'zustand'

import { isTauri } from '../workspace/tauri-fs'
import { parseFile } from '../persistence/file'
import { generateFromSnapshot } from '../codegen/generator'
import { fs } from '../connections/backend'
import { training, remoteTrainingBlocked } from './backend'
import {
  type RunSummary,
  type RunConfig,
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
// UI tracks queued→running→done without the user clicking refresh.
let pollTimer: ReturnType<typeof setInterval> | null = null

function syncPolling(get: () => TrainingState) {
  const anyActive = get().runs.some((r) => RUNNING_STATES.has(r.status) || r.alive)
  if (anyActive && pollTimer === null) {
    pollTimer = setInterval(() => void get().refresh(), 2000)
  } else if (!anyActive && pollTimer !== null) {
    clearInterval(pollTimer)
    pollTimer = null
  }
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
    if (remoteTrainingBlocked()) {
      set({ runs: [], listError: null })
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
      backend: { kind: 'local' },
      dataset,
      training: input.training,
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
