import { currentTorchUrl } from '../sidecars/torchUrl'

// Mirrors the payload from sidecar-torch/main.py `activations()`.

export type ActStats = {
  min: number
  max: number
  mean: number
  std: number
  frac_zero: number
}

export type Preview =
  | { kind: 'scalar'; value: number }
  | { kind: 'vector'; values: number[] }
  | { kind: 'matrix'; grid: number[][] }
  | { kind: 'maps'; channels: number; shown: number; maps: number[][][] }
  | { kind: 'tokens'; values: number[] }
  | { kind: 'edges'; n_edges: number; n_nodes: number; edges: [number, number][] }

export type NodeActivation = { stats: ActStats; preview: Preview | null }

export type Weights =
  | { kind: 'matrix'; shape: number[]; grid: number[][] }
  | { kind: 'kernels'; out_channels: number; in_channels: number; shown: number; kernels: number[][][] }

export type ActivationsOk = {
  ok: true
  activations: Record<string, NodeActivation>
  weights: Record<string, Weights>
  n_params: number
  sample_note: string | null
  weights_source?: 'trained' | 'random'
  weights_note?: string | null
}
export type ActivationsErr = {
  ok: false
  stage?: string
  error: string
  trace?: string
  details?: unknown
}
export type ActivationsResult = ActivationsOk | ActivationsErr | { ok: false; error: string; offline: true }

export async function runActivationsReq(
  code: string,
  inputShapes: number[][],
  inputDtypes?: string[],
  abspaths?: string[] | null,
  checkpoint?: string | null,
  graphAttrs?: string[] | null,
  inputOptions?: Record<string, unknown>[] | null,
  signal?: AbortSignal,
): Promise<ActivationsResult> {
  try {
    const res = await fetch(`${currentTorchUrl()}/activations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        code,
        input_shapes: inputShapes,
        input_dtypes: inputDtypes,
        ...(abspaths && abspaths.length ? { abspaths } : {}),
        ...(checkpoint ? { checkpoint } : {}),
        ...(graphAttrs && graphAttrs.length ? { graph_attrs: graphAttrs } : {}),
        ...(inputOptions && inputOptions.length ? { input_options: inputOptions } : {}),
      }),
      signal,
    })
    if (!res.ok) return { ok: false, error: `sidecar HTTP ${res.status}`, offline: true }
    return (await res.json()) as ActivationsResult
  } catch (e: unknown) {
    if (e instanceof DOMException && e.name === 'AbortError') throw e
    const msg = e instanceof Error ? e.message : String(e)
    return { ok: false, error: `sidecar unreachable: ${msg}`, offline: true }
  }
}
