import type { GraphSnapshot } from '../canvas/GraphStore'

export type Template = {
  id: string
  name: string
  description: string
  build: () => GraphSnapshot
}

function chain(
  inputShape: number[],
  steps: { layerType: string; params?: Record<string, unknown> }[],
): GraphSnapshot {
  const nodes: GraphSnapshot['nodes'] = [
    { id: 'input', layerType: 'Input', params: { shape: inputShape } },
  ]
  const edges: GraphSnapshot['edges'] = []
  let prev = 'input'
  steps.forEach((s, i) => {
    const id = `n${i + 1}`
    nodes.push({ id, layerType: s.layerType, params: s.params ?? {} })
    edges.push({ source: prev, target: id })
    prev = id
  })
  return { nodes, edges }
}

export const TEMPLATES: Template[] = [
  {
    id: 'empty',
    name: 'Empty',
    description: 'Just an Input node — start from scratch.',
    build: () => ({
      nodes: [{ id: 'input', layerType: 'Input', params: { shape: [1, 3, 224, 224] } }],
      edges: [],
    }),
  },
  {
    id: 'cnn-cifar',
    name: 'CNN classifier (CIFAR-style)',
    description: '3 conv blocks → adaptive pool → linear · 32×32 RGB, 10 classes',
    build: () =>
      chain([1, 3, 32, 32], [
        { layerType: 'Conv2d',      params: { in_channels: 3,  out_channels: 32, kernel_size: [3, 3], padding: [1, 1] } },
        { layerType: 'BatchNorm2d', params: { num_features: 32 } },
        { layerType: 'ReLU' },
        { layerType: 'MaxPool2d',   params: { kernel_size: [2, 2], stride: [2, 2] } },

        { layerType: 'Conv2d',      params: { in_channels: 32, out_channels: 64, kernel_size: [3, 3], padding: [1, 1] } },
        { layerType: 'BatchNorm2d', params: { num_features: 64 } },
        { layerType: 'ReLU' },
        { layerType: 'MaxPool2d',   params: { kernel_size: [2, 2], stride: [2, 2] } },

        { layerType: 'Conv2d',      params: { in_channels: 64, out_channels: 128, kernel_size: [3, 3], padding: [1, 1] } },
        { layerType: 'BatchNorm2d', params: { num_features: 128 } },
        { layerType: 'ReLU' },
        { layerType: 'AdaptiveAvgPool2d', params: { output_size: [1, 1] } },

        { layerType: 'Flatten' },
        { layerType: 'Dropout', params: { p: 0.3 } },
        { layerType: 'Linear',  params: { in_features: 128, out_features: 10 } },
      ]),
  },
  {
    id: 'mlp',
    name: 'MLP classifier',
    description: 'Flatten → 3 hidden Linear+ReLU → output · 28×28 grayscale, 10 classes',
    build: () =>
      chain([1, 1, 28, 28], [
        { layerType: 'Flatten' },
        { layerType: 'Linear', params: { in_features: 784, out_features: 256 } },
        { layerType: 'ReLU' },
        { layerType: 'Dropout', params: { p: 0.2 } },
        { layerType: 'Linear', params: { in_features: 256, out_features: 128 } },
        { layerType: 'ReLU' },
        { layerType: 'Linear', params: { in_features: 128, out_features: 10 } },
      ]),
  },
  {
    id: 'transformer-enc',
    name: 'Transformer encoder',
    description: '2× encoder layers + final norm · 16 tokens, d_model=512',
    build: () =>
      chain([1, 16, 512], [
        { layerType: 'TransformerEncoderLayer', params: { d_model: 512, nhead: 8, dim_feedforward: 2048, dropout: 0.1, activation: 'gelu', batch_first: true } },
        { layerType: 'TransformerEncoderLayer', params: { d_model: 512, nhead: 8, dim_feedforward: 2048, dropout: 0.1, activation: 'gelu', batch_first: true } },
        { layerType: 'LayerNorm', params: { normalized_shape: [512] } },
      ]),
  },
]
