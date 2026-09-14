// Dispatch layer for training-run lifecycle calls, analogous to
// connections/backend.ts. Components/stores MUST go through this, not through
// tauri-training / tauri-ssh directly — this is where the local↔remote switch
// lives. Local = Phase 13 executor; remote-ssh = Phase 16 ssh-direct executor.

import { getCurrentConnection, sshTarget, remotePython, type RemoteSshConnection } from '../connections/store'
import { tauriTraining } from './tauri-training'
import { tauriSsh } from '../connections/tauri-ssh'
import type { RunSummary, RunStatus, RemoteTrainingCapabilities, GpuStat } from './types'

const NO_CAPS: RemoteTrainingCapabilities = { has_slurm: false, has_gpu: false, partitions: [], gpu_names: [] }

export const REMOTE_TRAINING_MSG =
  'Remote-Training (ssh-direct) läuft detached auf dem Host. ' +
  'Stelle sicher, dass der Python-Pfad unten auf eine Umgebung mit torch zeigt.'

// Kept for callers that still want to know whether we're on a remote backend.
// Remote training is supported since Phase 16, so this no longer blocks.
export function remoteTrainingBlocked(): boolean {
  return false
}

function remote(): RemoteSshConnection | null {
  const c = getCurrentConnection()
  return c.kind === 'remote-ssh' ? c : null
}

export const training = {
  list: (): Promise<RunSummary[]> => {
    const r = remote()
    return r ? tauriSsh.listTrainingRuns(sshTarget(r), r.root) : tauriTraining.list()
  },
  status: (runId: string): Promise<RunStatus> => {
    const r = remote()
    return r ? tauriSsh.trainingRunStatus(sshTarget(r), r.root, runId) : tauriTraining.status(runId)
  },
  readFile: (runId: string, name: string): Promise<string> => {
    const r = remote()
    return r ? tauriSsh.readTrainingRunFile(sshTarget(r), r.root, runId, name) : tauriTraining.readFile(runId, name)
  },
  start: (runId: string, runJson: string, modelSpinoml: string, modelPy: string): Promise<void> => {
    const r = remote()
    return r
      ? tauriSsh.startTrainingRun(sshTarget(r), r.root, runId, remotePython(r), runJson, modelSpinoml, modelPy)
      : tauriTraining.start(runId, runJson, modelSpinoml, modelPy)
  },
  stop: (runId: string): Promise<void> => {
    const r = remote()
    return r ? tauriSsh.stopTrainingRun(sshTarget(r), r.root, runId) : tauriTraining.stop(runId)
  },
  remove: (runId: string): Promise<void> => {
    const r = remote()
    return r ? tauriSsh.deleteTrainingRun(sshTarget(r), r.root, runId) : tauriTraining.remove(runId)
  },
  capabilities: (): Promise<RemoteTrainingCapabilities> => {
    const r = remote()
    // Local backend has no SLURM; only remote hosts are probed.
    return r ? tauriSsh.remoteTrainingCapabilities(sshTarget(r), r.root) : Promise.resolve(NO_CAPS)
  },
  /** Copy a run's best checkpoint to models/best/<name>.pt. Returns the dest relpath. */
  promote: (runId: string, destName: string): Promise<string> => {
    const r = remote()
    return r
      ? tauriSsh.promoteCheckpoint(sshTarget(r), r.root, runId, destName)
      : tauriTraining.promote(runId, destName)
  },
  /** GPU snapshot on the executor host (empty if no nvidia-smi). For a SLURM run
   *  the remote path uses the run's job id to probe the COMPUTE node (srun) rather
   *  than the login node, which has no GPU. */
  gpuStats: (runId?: string): Promise<GpuStat[]> => {
    const r = remote()
    return r ? tauriSsh.gpuStats(sshTarget(r), r.root, runId) : tauriTraining.gpuStats()
  },
}
