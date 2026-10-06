import type { GraphSnapshot } from '../../src/canvas/GraphStore'

/**
 * The three small reference graphs used by `scripts/verify-reference.ts`.
 *
 * These are REAL GraphSnapshot objects — the same shape the app persists and
 * `generateFromSnapshot()` consumes — not hand-rolled Python. The generator
 * turns each into a `Model(nn.Module)` that is then compared numerically
 * against a hand-written reference module (see `reference_models.py`).
 */

export const REFERENCE_NAMES = ['mlp', 'cnn', 'multi-input'] as const
export type ReferenceName = (typeof REFERENCE_NAMES)[number]

const pos = (x: number, y: number) => ({ x, y })

/**
 * A. MLP: Input x [1,10] float32 → Linear(10,16) → ReLU → Linear(16,2) → Output.
 * Expected parameters: 210 (Linear(10,16)=176 + Linear(16,2)=34), all trainable.
 */
export function mlpSnapshot(): GraphSnapshot {
  return {
    nodes: [
      { id: 'x', layerType: 'Input', params: { name: 'x', shape: [1, 10], dtype: 'float32' }, position: pos(0, 0) },
      { id: 'fc1', layerType: 'Linear', params: { in_features: 10, out_features: 16, bias: true }, position: pos(0, 140) },
      { id: 'act', layerType: 'ReLU', params: { inplace: false }, position: pos(0, 280) },
      { id: 'fc2', layerType: 'Linear', params: { in_features: 16, out_features: 2, bias: true }, position: pos(0, 420) },
      { id: 'out', layerType: 'Output', params: { name: 'out' }, position: pos(0, 560) },
    ],
    edges: [
      { source: 'x', target: 'fc1' },
      { source: 'fc1', target: 'act' },
      { source: 'act', target: 'fc2' },
      { source: 'fc2', target: 'out' },
    ],
  }
}

/**
 * B. CNN: Input x [1,64] float32 → Reshape [1,8,8] (the Reshape layer preserves
 * the batch dim automatically: `x.reshape(x.shape[0], 1, 8, 8)`) → Conv2d(1,4,k3,p1)
 * → ReLU → MaxPool2d(2) → Flatten → Linear(64,2) → Output.
 * Expected parameters: 170 (Conv2d(1,4,3)=40 + Linear(64,2)=130), all trainable.
 */
export function cnnSnapshot(): GraphSnapshot {
  return {
    nodes: [
      { id: 'x', layerType: 'Input', params: { name: 'x', shape: [1, 64], dtype: 'float32' }, position: pos(0, 0) },
      { id: 'reshape', layerType: 'Reshape', params: { shape: [1, 8, 8] }, position: pos(0, 140) },
      {
        id: 'conv',
        layerType: 'Conv2d',
        params: { in_channels: 1, out_channels: 4, kernel_size: [3, 3], stride: [1, 1], padding: [1, 1], bias: true },
        position: pos(0, 280),
      },
      { id: 'act', layerType: 'ReLU', params: { inplace: false }, position: pos(0, 420) },
      { id: 'pool', layerType: 'MaxPool2d', params: { kernel_size: [2, 2], stride: [2, 2], padding: [0, 0] }, position: pos(0, 560) },
      { id: 'flat', layerType: 'Flatten', params: { start_dim: 1, end_dim: -1 }, position: pos(0, 700) },
      { id: 'fc', layerType: 'Linear', params: { in_features: 64, out_features: 2, bias: true }, position: pos(0, 840) },
      { id: 'out', layerType: 'Output', params: { name: 'out' }, position: pos(0, 980) },
    ],
    edges: [
      { source: 'x', target: 'reshape' },
      { source: 'reshape', target: 'conv' },
      { source: 'conv', target: 'act' },
      { source: 'act', target: 'pool' },
      { source: 'pool', target: 'flat' },
      { source: 'flat', target: 'fc' },
      { source: 'fc', target: 'out' },
    ],
  }
}

/**
 * C. Multi-input: Input a [1,6] and Input b [1,4] (float32) → Linear(6,8)+ReLU on
 * a, Linear(4,8)+ReLU on b → Concat(dim=1) on the feature axis → Linear(16,2) → Output.
 * Expected parameters: 130 (Linear(6,8)=56 + Linear(4,8)=40 + Linear(16,2)=34), all trainable.
 *
 * The generated `forward(self, a, b)` argument order follows the input nodes'
 * canonical order (both are layerType `Input`, tie-broken by id → a, b).
 */
export function multiInputSnapshot(): GraphSnapshot {
  return {
    nodes: [
      { id: 'a', layerType: 'Input', params: { name: 'a', shape: [1, 6], dtype: 'float32' }, position: pos(0, 0) },
      { id: 'b', layerType: 'Input', params: { name: 'b', shape: [1, 4], dtype: 'float32' }, position: pos(360, 0) },
      { id: 'fc_a', layerType: 'Linear', params: { in_features: 6, out_features: 8, bias: true }, position: pos(0, 160) },
      { id: 'relu_a', layerType: 'ReLU', params: { inplace: false }, position: pos(0, 300) },
      { id: 'fc_b', layerType: 'Linear', params: { in_features: 4, out_features: 8, bias: true }, position: pos(360, 160) },
      { id: 'relu_b', layerType: 'ReLU', params: { inplace: false }, position: pos(360, 300) },
      { id: 'cat', layerType: 'Concat', params: { dim: 1 }, position: pos(180, 440) },
      { id: 'fc', layerType: 'Linear', params: { in_features: 16, out_features: 2, bias: true }, position: pos(180, 580) },
      { id: 'out', layerType: 'Output', params: { name: 'out' }, position: pos(180, 720) },
    ],
    edges: [
      { source: 'a', target: 'fc_a' },
      { source: 'fc_a', target: 'relu_a' },
      { source: 'b', target: 'fc_b' },
      { source: 'fc_b', target: 'relu_b' },
      { source: 'relu_a', target: 'cat' },
      { source: 'relu_b', target: 'cat' },
      { source: 'cat', target: 'fc' },
      { source: 'fc', target: 'out' },
    ],
  }
}
