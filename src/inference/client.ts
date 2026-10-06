import { currentTorchEndpoint, currentTorchUrl, torchFetch } from '../sidecars/torchUrl'
import { SidecarAuthError, probeSidecarHealth, type SidecarHealth } from '../sidecars/auth'

export type InferOk = {
  ok: true
  shapes: Record<string, number[]>
  n_params: number
}

export type InferErr = {
  ok: false
  stage?: 'compile' | 'construct' | 'input' | 'forward' | 'sidecar'
  error: string
  error_code?: string
  trace?: string
  shapes: Record<string, number[]>
  n_params?: number
}

export type InferResult = InferOk | InferErr

/** Auth failure: the sidecar is reachable but rejected the request — reported
 *  with `offline: false` so it is never rendered as "unreachable". */
export type InferAuthErr = {
  ok: false
  error: string
  offline: false
  authFailed: true
  shapes: Record<string, number[]>
}

export async function inferShapes(
  code: string,
  inputShapes: number[][],
  inputDtypes?: string[],
  signal?: AbortSignal,
): Promise<InferResult | { ok: false; error: string; offline: true; shapes: Record<string, number[]> } | InferAuthErr> {
  try {
    const res = await torchFetch('/infer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code, input_shapes: inputShapes, input_dtypes: inputDtypes }),
      signal,
    })
    if (!res.ok) {
      return { ok: false, error: `sidecar HTTP ${res.status}`, shapes: {}, offline: true }
    }
    return (await res.json()) as InferResult
  } catch (e: unknown) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e
    if (e instanceof SidecarAuthError) {
      // Reachable but rejected (401/403): an explicit auth error, NOT offline.
      return { ok: false, error: e.message, shapes: {}, offline: false, authFailed: true }
    }
    const msg = e instanceof Error ? e.message : String(e)
    return { ok: false, error: `sidecar unreachable: ${msg}`, shapes: {}, offline: true }
  }
}

export async function sidecarHealth(): Promise<boolean> {
  return (await sidecarHealthState()).state === 'online'
}

/** Full health probe (online / offline / auth-failed + auth mode). */
export async function sidecarHealthState(): Promise<SidecarHealth> {
  return probeSidecarHealth(currentTorchEndpoint(), currentTorchUrl())
}
