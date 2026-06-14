import type { InspectResult, StatsResult, SmokeResult } from './types'
import { currentTorchUrl } from '../sidecars/torchUrl'

async function post<T>(path: string, body: unknown): Promise<T | { ok: false; error: string; offline: true }> {
  try {
    const res = await fetch(`${currentTorchUrl()}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok) {
      return { ok: false, error: `sidecar HTTP ${res.status}`, offline: true } as const
    }
    return (await res.json()) as T
  } catch (e) {
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

export type InputOption = { features?: string[]; target?: string }

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
