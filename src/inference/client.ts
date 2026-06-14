export type InferOk = {
  ok: true
  shapes: Record<string, number[]>
  n_params: number
}

export type InferErr = {
  ok: false
  stage?: 'compile' | 'construct' | 'input' | 'forward' | 'sidecar'
  error: string
  trace?: string
  shapes: Record<string, number[]>
  n_params?: number
}

export type InferResult = InferOk | InferErr

import { currentTorchUrl } from '../sidecars/torchUrl'

export async function inferShapes(
  code: string,
  inputShapes: number[][],
  signal?: AbortSignal,
): Promise<InferResult | { ok: false; error: string; offline: true; shapes: Record<string, number[]> }> {
  try {
    const res = await fetch(`${currentTorchUrl()}/infer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, input_shapes: inputShapes }),
      signal,
    })
    if (!res.ok) {
      return { ok: false, error: `sidecar HTTP ${res.status}`, shapes: {}, offline: true }
    }
    return (await res.json()) as InferResult
  } catch (e: unknown) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e
    const msg = e instanceof Error ? e.message : String(e)
    return { ok: false, error: `sidecar unreachable: ${msg}`, shapes: {}, offline: true }
  }
}

export async function sidecarHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${currentTorchUrl()}/health`)
    return res.ok
  } catch {
    return false
  }
}
