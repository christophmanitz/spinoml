// Typed wrappers around the local training_* Tauri commands. Local-only:
// remote training (ssh-direct / SLURM) arrives in Phase 16/17 and will get its
// own ssh_* mirror. Everything routes through training/backend.ts, never here
// directly (mirrors the workspace tauri-fs / connections tauri-ssh split).

import { invoke } from '@tauri-apps/api/core'
import type { RunSummary, RunStatus } from './types'

export const tauriTraining = {
  list: () => invoke<RunSummary[]>('list_training_runs'),
  status: (runId: string) => invoke<RunStatus>('training_run_status', { runId }),
  readFile: (runId: string, name: string) =>
    invoke<string>('read_training_run_file', { runId, name }),
  start: (runId: string, runJson: string, modelMlforge: string, modelPy: string) =>
    invoke<void>('start_training_run', { runId, runJson, modelMlforge, modelPy }),
  stop: (runId: string) => invoke<void>('stop_training_run', { runId }),
  remove: (runId: string) => invoke<void>('delete_training_run', { runId }),
}
