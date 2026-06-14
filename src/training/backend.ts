// Dispatch layer for training-run lifecycle calls, analogous to
// connections/backend.ts. Components/stores MUST go through this, not through
// tauri-training directly — that's where the local↔remote switch lives once
// remote training lands (Phase 16/17). For now remote is explicitly blocked.

import { getCurrentConnection } from '../connections/store'
import { tauriTraining } from './tauri-training'
import type { RunSummary, RunStatus } from './types'

export const REMOTE_TRAINING_MSG =
  'Remote-Training kommt in Phase 16 (ssh-direct) bzw. 17 (SLURM). ' +
  'Wechsle auf die lokale Verbindung, um einen Run zu starten.'

export function remoteTrainingBlocked(): boolean {
  return getCurrentConnection().kind === 'remote-ssh'
}

export const training = {
  list: (): Promise<RunSummary[]> => {
    if (remoteTrainingBlocked()) return Promise.resolve([])
    return tauriTraining.list()
  },
  status: (runId: string): Promise<RunStatus> => {
    if (remoteTrainingBlocked()) return Promise.reject(new Error(REMOTE_TRAINING_MSG))
    return tauriTraining.status(runId)
  },
  readFile: (runId: string, name: string): Promise<string> => {
    if (remoteTrainingBlocked()) return Promise.reject(new Error(REMOTE_TRAINING_MSG))
    return tauriTraining.readFile(runId, name)
  },
  start: (runId: string, runJson: string, modelMlforge: string, modelPy: string): Promise<void> => {
    if (remoteTrainingBlocked()) return Promise.reject(new Error(REMOTE_TRAINING_MSG))
    return tauriTraining.start(runId, runJson, modelMlforge, modelPy)
  },
  stop: (runId: string): Promise<void> => {
    if (remoteTrainingBlocked()) return Promise.reject(new Error(REMOTE_TRAINING_MSG))
    return tauriTraining.stop(runId)
  },
  remove: (runId: string): Promise<void> => {
    if (remoteTrainingBlocked()) return Promise.reject(new Error(REMOTE_TRAINING_MSG))
    return tauriTraining.remove(runId)
  },
}
