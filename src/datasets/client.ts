import type { InspectResult, StatsResult, SmokeResult } from './types'

const SIDECAR_URL = 'http://127.0.0.1:7421'

async function post<T>(path: string, body: unknown): Promise<T | { ok: false; error: string; offline: true }> {
  try {
    const res = await fetch(`${SIDECAR_URL}${path}`, {
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

export async function smokeDataset(code: string, abspath: string, inputShapes?: number[][]) {
  return post<SmokeResult>('/dataset/smoke', {
    code, abspath, input_shapes: inputShapes ?? null,
  })
}
