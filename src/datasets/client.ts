import type { InspectResult, StatsResult, SmokeResult } from './types'
import { torchFetch } from '../sidecars/torchUrl'
import { SidecarAuthError } from '../sidecars/auth'

async function post<T>(path: string, body: unknown): Promise<T | { ok: false; error: string; offline: boolean }> {
  try {
    const res = await torchFetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok) {
      return { ok: false, error: `sidecar HTTP ${res.status}`, offline: true } as const
    }
    return (await res.json()) as T
  } catch (e) {
    if (e instanceof SidecarAuthError) {
      // Reachable but rejected: `offline: false` keeps the store from labelling
      // it "unreachable" and surfaces the German auth message instead.
      return { ok: false, error: e.message, offline: false } as const
    }
    const msg = e instanceof Error ? e.message : String(e)
    return { ok: false, error: `sidecar unreachable: ${msg}`, offline: true } as const
  }
}

export async function inspectDataset(abspath: string) {
  return post<InspectResult>('/dataset/inspect', { abspath })
}

export async function statsDataset(abspath: string) {
  return post<StatsResult>('/dataset/stats', { abspath })
}

export type RunScriptResult = {
  ok: boolean
  mode?: 'shell' | 'slurm'
  code?: number
  stdout?: string
  stderr?: string
  job_id?: string | null
  timed_out?: boolean
  error?: string
}

/** Run a data-pipeline script in the workspace WITHOUT the chatbot — the torch
 *  sidecar (local, or the HPC one via the tunnel) writes `code` to <root>/<relpath>
 *  and runs it (shell now, or sbatch for slurm), returning captured output. */
export async function runWorkspaceScript(root: string, relpath: string, code: string, mode: 'shell' | 'slurm') {
  return post<RunScriptResult>('/run_script', { root, relpath, code, mode })
}

export type InputOption = {
  features?: string[]; target?: string; field?: string
  // A whole-graph (Graph node) input: assemble a PyG Data from one source;
  // `branch` selects the manifest branch (empty for single-graph datasets).
  graph?: true; branch?: string
}

export async function smokeDataset(
  code: string, abspath: string,
  inputShapes?: number[][], inputOptions?: InputOption[],
) {
  return post<SmokeResult>('/dataset/smoke', {
    code, abspath, input_shapes: inputShapes ?? null, input_options: inputOptions ?? null,
  })
}

export async function smokeDatasetMulti(
  code: string, abspaths: string[],
  inputShapes?: number[][], inputOptions?: InputOption[],
) {
  return post<SmokeResult>('/dataset/smoke', {
    code, abspaths, input_shapes: inputShapes ?? null, input_options: inputOptions ?? null,
  })
}
