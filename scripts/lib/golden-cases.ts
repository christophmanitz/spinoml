// Builders for the Phase-6 codegen-golden fixtures.
// Each entry becomes one <name>.graph.json + <name>.expected.py under scripts/golden/
// when the test runs with --update. The committed JSON is the source of truth the
// test loads on a normal run; this file is the boot path. node ids are stable
// human-readable names so the generated Python's forward-arg / attribute names
// are reviewable. Every graph is accepted by the generator without issues
// (the `attention` case is the regression test for the MultiheadAttention call signature).
import { defaultParamsFor } from '../../src/layers/registry'
import { defaultTrainingParams } from '../../src/training/graph/registry'
import { defaultDataParams } from '../../src/data/graph/registry'

export type GoldenKind = 'model' | 'training' | 'data'

export type GoldenInput = {
  name: string
  shape: number[]
  dtype: string
  isGraph?: boolean
  nEdges?: number
  edgeDim?: number
}

export type GoldenFixture = {
  name: string
  kind: GoldenKind
  nodes: { id: string; type: string; params: Record<string, unknown> }[]
  edges: { source: string; target: string }[]
  inputs?: GoldenInput[]
  expectedOutputShape?: number[] | Record<string, number[]>
  requires?: string[]
}

const M = (id: string, type: string, p: Record<string, unknown> = {}) => ({
  id, type, params: { ...defaultParamsFor(type), ...p },
})
const T = (id: string, type: string, p: Record<string, unknown> = {}) => ({
  id, type, params: { ...defaultTrainingParams(type), ...p },
})
const D = (id: string, type: string, p: Record<string, unknown> = {}) => ({
  id, type, params: { ...defaultDataParams(type), ...p },
})
const E = (source: string, target: string) => ({ source, target })
// Subgraph snapshots store their inner nodes in the persisted GraphSnapshot
// shape (key `layerType`), not the canvas node shape.
const SM = (id: string, layerType: string, p: Record<string, unknown> = {}) => ({
  id, layerType, params: { ...defaultParamsFor(layerType), ...p },
})

// Fixed harmless Custom source: y = factor * x. Deterministic, no imports.
const CUSTOM_SCALE_SRC = [
  'class Scale(nn.Module):',
  '    def __init__(self, factor):',
  '        super().__init__()',
  '        self.factor = factor',
  '',
  '    def forward(self, x):',
  '        return x * self.factor',
].join('\n')

// Fixed harmless CustomScript: makes a stable copy column.
const CUSTOM_DEDUP_SRC = [
  'df = df.drop_duplicates()',
  'print("after dedup", df.shape)',
].join('\n')

// Fixed harmless DataOp script: writes a marker file. Run-time irrelevant for codegen.
const DATAOP_SCRIPT = [
  'import sys, os, argparse',
  '',
  'def main():',
  '    ap = argparse.ArgumentParser()',
  '    ap.add_argument("--input", default="")',
  '    ap.add_argument("--output", default="processed/out")',
  '    args = ap.parse_args()',
  '    out_dir = os.path.join("datasets", args.output)',
  '    os.makedirs(out_dir, exist_ok=True)',
  '    with open(os.path.join(out_dir, ".marker"), "w") as f:',
  '        f.write("")',
  '    print(f"wrote marker {out_dir}")',
  '',
  'if __name__ == "__main__":',
  '    main()',
].join('\n')

export function buildGoldenCases(): GoldenFixture[] {
  return [
    // ── MODEL cases ─────────────────────────────────────────────────────────
    {
      kind: 'model',
      name: 'linear',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [1, 8], dtype: 'float32' }),
        M('lin', 'Linear', { in_features: 8, out_features: 4, bias: true }),
        M('out', 'Output', { name: 'logits' }),
      ],
      edges: [E('x', 'lin'), E('lin', 'out')],
      inputs: [{ name: 'x', shape: [1, 8], dtype: 'float32' }],
      expectedOutputShape: [1, 4],
    },
    {
      kind: 'model',
      name: 'mlp',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [1, 10], dtype: 'float32' }),
        M('fc1', 'Linear', { in_features: 10, out_features: 16, bias: true }),
        M('act1', 'ReLU', { inplace: false }),
        M('drp', 'Dropout', { p: 0.25 }),
        M('fc2', 'Linear', { in_features: 16, out_features: 2, bias: true }),
        M('out', 'Output', { name: 'out' }),
      ],
      edges: [E('x', 'fc1'), E('fc1', 'act1'), E('act1', 'drp'), E('drp', 'fc2'), E('fc2', 'out')],
      inputs: [{ name: 'x', shape: [1, 10], dtype: 'float32' }],
      expectedOutputShape: [1, 2],
    },
    {
      kind: 'model',
      name: 'cnn',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [1, 3, 16, 16], dtype: 'float32' }),
        M('conv', 'Conv2d', { in_channels: 3, out_channels: 8, kernel_size: [3, 3], stride: [1, 1], padding: [1, 1], bias: true }),
        M('bn', 'BatchNorm2d', { num_features: 8, eps: 1e-5, momentum: 0.1 }),
        M('relu', 'ReLU', { inplace: false }),
        M('drp2d', 'Dropout2d', { p: 0.1 }),
        M('pool', 'MaxPool2d', { kernel_size: [2, 2], stride: [2, 2], padding: [0, 0] }),
        M('flat', 'Flatten', { start_dim: 1, end_dim: -1 }),
        M('fc', 'Linear', { in_features: 512, out_features: 10, bias: true }),
        M('out', 'Output', { name: 'logits' }),
      ],
      edges: [E('x', 'conv'), E('conv', 'bn'), E('bn', 'relu'), E('relu', 'drp2d'), E('drp2d', 'pool'), E('pool', 'flat'), E('flat', 'fc'), E('fc', 'out')],
      inputs: [{ name: 'x', shape: [1, 3, 16, 16], dtype: 'float32' }],
      expectedOutputShape: [1, 10],
    },
    {
      kind: 'model',
      name: 'residual',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [1, 16], dtype: 'float32' }),
        M('l1', 'Linear', { in_features: 16, out_features: 16, bias: true }),
        M('act', 'ReLU', { inplace: false }),
        M('l2', 'Linear', { in_features: 16, out_features: 16, bias: true }),
        M('add', 'Add'),
        M('out', 'Output', { name: 'out' }),
      ],
      edges: [E('x', 'l1'), E('l1', 'act'), E('act', 'l2'), E('x', 'add'), E('l2', 'add'), E('add', 'out')],
      inputs: [{ name: 'x', shape: [1, 16], dtype: 'float32' }],
      expectedOutputShape: [1, 16],
    },
    {
      kind: 'model',
      name: 'branch-merge',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [1, 12], dtype: 'float32' }),
        M('la', 'Linear', { in_features: 12, out_features: 8, bias: true }),
        M('ra', 'ReLU', { inplace: false }),
        M('lb', 'Linear', { in_features: 12, out_features: 8, bias: true }),
        M('tb', 'Tanh'),
        M('add', 'Add'),
        M('out', 'Output', { name: 'logits' }),
      ],
      edges: [E('x', 'la'), E('la', 'ra'), E('x', 'lb'), E('lb', 'tb'), E('ra', 'add'), E('tb', 'add'), E('add', 'out')],
      inputs: [{ name: 'x', shape: [1, 12], dtype: 'float32' }],
      expectedOutputShape: [1, 8],
    },
    {
      kind: 'model',
      name: 'multi-input',
      nodes: [
        M('a', 'Input', { name: 'a', shape: [1, 6], dtype: 'float32' }),
        M('b', 'Input', { name: 'b', shape: [1, 4], dtype: 'float32' }),
        M('la', 'Linear', { in_features: 6, out_features: 8, bias: true }),
        M('ra', 'ReLU', { inplace: false }),
        M('lb', 'Linear', { in_features: 4, out_features: 8, bias: true }),
        M('rb', 'ReLU', { inplace: false }),
        M('cat', 'Concat', { dim: 1 }),
        M('fc', 'Linear', { in_features: 16, out_features: 2, bias: true }),
        M('out', 'Output', { name: 'out' }),
      ],
      edges: [E('a', 'la'), E('la', 'ra'), E('b', 'lb'), E('lb', 'rb'), E('ra', 'cat'), E('rb', 'cat'), E('cat', 'fc'), E('fc', 'out')],
      inputs: [{ name: 'a', shape: [1, 6], dtype: 'float32' }, { name: 'b', shape: [1, 4], dtype: 'float32' }],
      expectedOutputShape: [1, 2],
    },
    {
      kind: 'model',
      name: 'multi-output',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [1, 12], dtype: 'float32' }),
        M('la', 'Linear', { in_features: 12, out_features: 4, bias: true }),
        M('lb', 'Linear', { in_features: 12, out_features: 6, bias: true }),
        M('oa', 'Output', { name: 'logits' }),
        M('ob', 'Output', { name: 'embedding' }),
      ],
      edges: [E('x', 'la'), E('la', 'oa'), E('x', 'lb'), E('lb', 'ob')],
      inputs: [{ name: 'x', shape: [1, 12], dtype: 'float32' }],
      expectedOutputShape: { logits: [1, 4], embedding: [1, 6] },
    },
    {
      kind: 'model',
      name: 'concat',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [1, 10], dtype: 'float32' }),
        M('l1', 'Linear', { in_features: 10, out_features: 4, bias: true }),
        M('l2', 'Linear', { in_features: 10, out_features: 5, bias: true }),
        M('l3', 'Linear', { in_features: 10, out_features: 6, bias: true }),
        M('cat', 'Concat', { dim: 1 }),
        M('out', 'Output', { name: 'logits' }),
      ],
      edges: [E('x', 'l1'), E('x', 'l2'), E('x', 'l3'), E('l1', 'cat'), E('l2', 'cat'), E('l3', 'cat'), E('cat', 'out')],
      inputs: [{ name: 'x', shape: [1, 10], dtype: 'float32' }],
      expectedOutputShape: [1, 15],
    },
    {
      kind: 'model',
      name: 'flatten',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [1, 2, 3, 4], dtype: 'float32' }),
        M('flat', 'Flatten', { start_dim: 1, end_dim: -1 }),
        M('out', 'Output', { name: 'logits' }),
      ],
      edges: [E('x', 'flat'), E('flat', 'out')],
      inputs: [{ name: 'x', shape: [1, 2, 3, 4], dtype: 'float32' }],
      expectedOutputShape: [1, 24],
    },
    {
      kind: 'model',
      name: 'reshape',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [1, 24], dtype: 'float32' }),
        M('rsh', 'Reshape', { shape: [2, 12] }),
        M('out', 'Output', { name: 'logits' }),
      ],
      edges: [E('x', 'rsh'), E('rsh', 'out')],
      inputs: [{ name: 'x', shape: [1, 24], dtype: 'float32' }],
      expectedOutputShape: [1, 2, 12],
    },
    {
      kind: 'model',
      name: 'normalization',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [1, 16], dtype: 'float32' }),
        M('bn', 'BatchNorm1d', { num_features: 16, eps: 1e-5, momentum: 0.1 }),
        M('ln', 'LayerNorm', { normalized_shape: [16], eps: 1e-5 }),
        M('gn', 'GroupNorm', { num_groups: 4, num_channels: 16 }),
        M('out', 'Output', { name: 'out' }),
      ],
      edges: [E('x', 'bn'), E('bn', 'ln'), E('ln', 'gn'), E('gn', 'out')],
      inputs: [{ name: 'x', shape: [1, 16], dtype: 'float32' }],
      expectedOutputShape: [1, 16],
    },
    {
      kind: 'model',
      name: 'pooling',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [1, 1, 16, 16], dtype: 'float32' }),
        M('mp', 'MaxPool2d', { kernel_size: [2, 2], stride: [2, 2], padding: [0, 0] }),
        M('ap', 'AvgPool2d', { kernel_size: [2, 2], stride: [2, 2], padding: [0, 0] }),
        M('aap', 'AdaptiveAvgPool2d', { output_size: [2, 2] }),
        M('out', 'Output', { name: 'logits' }),
      ],
      edges: [E('x', 'mp'), E('mp', 'ap'), E('ap', 'aap'), E('aap', 'out')],
      inputs: [{ name: 'x', shape: [1, 1, 16, 16], dtype: 'float32' }],
      expectedOutputShape: [1, 1, 2, 2],
    },
    {
      kind: 'model',
      name: 'transformer',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [1, 16, 32], dtype: 'float32' }),
        M('el', 'TransformerEncoderLayer', { d_model: 32, nhead: 4, dim_feedforward: 64, dropout: 0.0, activation: 'gelu', batch_first: true }),
        M('enc', 'TransformerEncoder', { d_model: 32, nhead: 4, num_layers: 2, dim_feedforward: 64, dropout: 0.0, activation: 'gelu', batch_first: true }),
        M('out', 'Output', { name: 'embedding' }),
      ],
      edges: [E('x', 'el'), E('el', 'enc'), E('enc', 'out')],
      inputs: [{ name: 'x', shape: [1, 16, 32], dtype: 'float32' }],
      expectedOutputShape: [1, 16, 32],
    },
    {
      // Regression (Phase 6 review): nn.MultiheadAttention.forward needs (query, key, value)
      // and returns (output, weights); the generator used to emit a one-argument call that
      // raises TypeError at run time. Self-attention over the single incoming stream.
      kind: 'model',
      name: 'attention',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [1, 10, 32], dtype: 'float32' }),
        M('mha', 'MultiheadAttention', { embed_dim: 32, num_heads: 4, dropout: 0.0, batch_first: true }),
        M('out', 'Output', { name: 'attended' }),
      ],
      edges: [E('x', 'mha'), E('mha', 'out')],
      inputs: [{ name: 'x', shape: [1, 10, 32], dtype: 'float32' }],
      expectedOutputShape: [1, 10, 32],
    },
    {
      kind: 'model',
      name: 'recurrent',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [1, 8, 16], dtype: 'float32' }),
        M('lstm', 'LSTM', { input_size: 16, hidden_size: 16, num_layers: 1, batch_first: true, bidirectional: false, dropout: 0.0 }),
        M('gru', 'GRU', { input_size: 16, hidden_size: 16, num_layers: 1, batch_first: true, bidirectional: false, dropout: 0.0 }),
        M('rnn', 'RNN', { input_size: 16, hidden_size: 16, num_layers: 1, nonlinearity: 'tanh', batch_first: true, bidirectional: false }),
        M('out', 'Output', { name: 'out' }),
      ],
      edges: [E('x', 'lstm'), E('lstm', 'gru'), E('gru', 'rnn'), E('rnn', 'out')],
      inputs: [{ name: 'x', shape: [1, 8, 16], dtype: 'float32' }],
      expectedOutputShape: [1, 8, 16],
    },
    {
      kind: 'model',
      name: 'gnn',
      nodes: [
        M('data', 'Graph', { name: 'data', shape: [32, 9], n_edges: 64, edge_dim: 0 }),
        M('gcn', 'GCNConv', { in_channels: -1, out_channels: 16, improved: false, cached: false, add_self_loops: true, bias: true }),
        M('relu', 'ReLU', { inplace: false }),
        M('gmp', 'GlobalMeanPool'),
        M('out', 'Output', { name: 'logits' }),
      ],
      edges: [E('data', 'gcn'), E('gcn', 'relu'), E('relu', 'gmp'), E('gmp', 'out')],
      inputs: [{ name: 'data', shape: [32, 9], dtype: 'graph', isGraph: true, nEdges: 64, edgeDim: 0 }],
      expectedOutputShape: [1, 16],
      requires: ['torch_geometric'],
    },
    {
      kind: 'model',
      name: 'gnn-extra',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [8, 6], dtype: 'float32' }),
        M('ei', 'Input', { name: 'edge_index', shape: [2, 12], dtype: 'int64' }),
        M('batch', 'Input', { name: 'batch', shape: [8], dtype: 'int64' }),
        M('sage', 'SAGEConv', { in_channels: 6, out_channels: 5, aggr: 'mean', normalize: false, bias: true }),
        M('gconv', 'GraphConv', { in_channels: 5, out_channels: 4, aggr: 'add', bias: true }),
        M('gtrans', 'GraphTransformer', { in_channels: 4, out_channels: 4, heads: 2, concat: true, beta: false, dropout: 0.0, bias: true }),
        M('gap', 'GlobalAddPool'),
        M('out', 'Output', { name: 'logits' }),
      ],
      edges: [E('x', 'sage'), E('sage', 'gconv'), E('gconv', 'gtrans'), E('gtrans', 'gap'), E('gap', 'out')],
      inputs: [
        { name: 'x', shape: [8, 6], dtype: 'float32' },
        { name: 'edge_index', shape: [2, 12], dtype: 'int64' },
        { name: 'batch', shape: [8], dtype: 'int64' },
      ],
      expectedOutputShape: [1, 8],
      requires: ['torch_geometric'],
    },
    {
      kind: 'model',
      name: 'build-graph',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [8, 4], dtype: 'float32' }),
        M('batch', 'Input', { name: 'batch', shape: [8], dtype: 'int64' }),
        M('bg', 'BuildGraph', { method: 'knn', k: 3, radius: 1.0, dims: [], loop: false, cosine: false }),
        M('gat', 'GATConv', { in_channels: 4, out_channels: 4, heads: 2, concat: true, dropout: 0.0, bias: true }),
        M('relu', 'ReLU', { inplace: false }),
        M('gmp', 'GlobalMaxPool'),
        M('out', 'Output', { name: 'logits' }),
      ],
      edges: [E('x', 'bg'), E('bg', 'gat'), E('gat', 'relu'), E('relu', 'gmp'), E('gmp', 'out')],
      inputs: [{ name: 'x', shape: [8, 4], dtype: 'float32' }, { name: 'batch', shape: [8], dtype: 'int64' }],
      expectedOutputShape: [1, 8],
      requires: ['torch_geometric'],
    },
    {
      kind: 'model',
      name: 'sequence',
      nodes: [
        M('seq', 'Sequence', { name: 'seq', shape: [1, 8], dtype: 'int64' }),
        M('emb', 'Embedding', { num_embeddings: 50, embedding_dim: 16 }),
        M('flat', 'Flatten', { start_dim: 1, end_dim: -1 }),
        M('fc', 'Linear', { in_features: 128, out_features: 4, bias: true }),
        M('out', 'Output', { name: 'logits' }),
      ],
      edges: [E('seq', 'emb'), E('emb', 'flat'), E('flat', 'fc'), E('fc', 'out')],
      inputs: [{ name: 'seq', shape: [1, 8], dtype: 'int64' }],
      expectedOutputShape: [1, 4],
    },
    {
      kind: 'model',
      name: 'espf',
      nodes: [
        M('espf', 'ESPF', { name: 'espf', shape: [1, 8], dtype: 'int64', codebook: 'drug' }),
        M('emb', 'Embedding', { num_embeddings: 100, embedding_dim: 16 }),
        M('flat', 'Flatten', { start_dim: 1, end_dim: -1 }),
        M('fc', 'Linear', { in_features: 128, out_features: 4, bias: true }),
        M('out', 'Output', { name: 'logits' }),
      ],
      edges: [E('espf', 'emb'), E('emb', 'flat'), E('flat', 'fc'), E('fc', 'out')],
      inputs: [{ name: 'espf', shape: [1, 8], dtype: 'int64' }],
      expectedOutputShape: [1, 4],
    },
    {
      kind: 'model',
      name: 'manifest',
      nodes: [
        M('mani', 'Manifest', { dataset: 'datasets/pair.manifest' }),
        M('x', 'Input', { name: 'x', shape: [1, 8], dtype: 'float32' }),
        M('fc', 'Linear', { in_features: 8, out_features: 4, bias: true }),
        M('out', 'Output', { name: 'logits' }),
      ],
      edges: [E('mani', 'x'), E('x', 'fc'), E('fc', 'out')],
      inputs: [{ name: 'x', shape: [1, 8], dtype: 'float32' }],
      expectedOutputShape: [1, 4],
    },
    {
      kind: 'model',
      name: 'conv1d',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [1, 1, 16], dtype: 'float32' }),
        M('c1', 'Conv1d', { in_channels: 1, out_channels: 4, kernel_size: 3, stride: 1, padding: 1, bias: true }),
        M('out', 'Output', { name: 'logits' }),
      ],
      edges: [E('x', 'c1'), E('c1', 'out')],
      inputs: [{ name: 'x', shape: [1, 1, 16], dtype: 'float32' }],
      expectedOutputShape: [1, 4, 16],
    },
    {
      kind: 'model',
      name: 'convtranspose2d',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [1, 4, 8, 8], dtype: 'float32' }),
        M('ct', 'ConvTranspose2d', { in_channels: 4, out_channels: 2, kernel_size: [3, 3], stride: [2, 2], padding: [1, 1] }),
        M('out', 'Output', { name: 'logits' }),
      ],
      edges: [E('x', 'ct'), E('ct', 'out')],
      inputs: [{ name: 'x', shape: [1, 4, 8, 8], dtype: 'float32' }],
      expectedOutputShape: [1, 2, 15, 15],
    },
    {
      kind: 'model',
      name: 'conv3d',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [1, 1, 4, 4, 4], dtype: 'float32' }),
        M('c3', 'Conv3d', { in_channels: 1, out_channels: 2, kernel_size: [3, 3, 3], stride: [1, 1, 1], padding: [1, 1, 1], bias: true }),
        M('out', 'Output', { name: 'logits' }),
      ],
      edges: [E('x', 'c3'), E('c3', 'out')],
      inputs: [{ name: 'x', shape: [1, 1, 4, 4, 4], dtype: 'float32' }],
      expectedOutputShape: [1, 2, 4, 4, 4],
    },
    {
      kind: 'model',
      name: 'activations',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [1, 8], dtype: 'float32' }),
        M('a1', 'ReLU', { inplace: false }),
        M('a2', 'GELU'),
        M('a3', 'SiLU', { inplace: false }),
        M('a4', 'Sigmoid'),
        M('a5', 'Tanh'),
        M('a6', 'Softmax', { dim: -1 }),
        M('a7', 'LogSoftmax', { dim: -1 }),
        M('out', 'Output', { name: 'out' }),
      ],
      edges: [E('x', 'a1'), E('a1', 'a2'), E('a2', 'a3'), E('a3', 'a4'), E('a4', 'a5'), E('a5', 'a6'), E('a6', 'a7'), E('a7', 'out')],
      inputs: [{ name: 'x', shape: [1, 8], dtype: 'float32' }],
      expectedOutputShape: [1, 8],
    },
    {
      kind: 'model',
      name: 'reshape-extra',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [1, 2, 3, 4], dtype: 'float32' }),
        M('p', 'Permute', { dims: [0, 2, 1, 3] }),
        M('t', 'Transpose', { dim0: 1, dim1: 2 }),
        M('v', 'View', { shape: [-1] }),
        M('out', 'Output', { name: 'logits' }),
      ],
      edges: [E('x', 'p'), E('p', 't'), E('t', 'v'), E('v', 'out')],
      inputs: [{ name: 'x', shape: [1, 2, 3, 4], dtype: 'float32' }],
      expectedOutputShape: [1, 24],
    },
    {
      kind: 'model',
      name: 'merge-extra',
      nodes: [
        M('a', 'Input', { name: 'a', shape: [1, 4], dtype: 'float32' }),
        M('b', 'Input', { name: 'b', shape: [1, 4], dtype: 'float32' }),
        M('mul', 'Multiply'),
        M('add', 'Add'),
        M('stk', 'Stack', { dim: 0 }),
        M('om', 'Output', { name: 'aux' }),
        M('oa', 'Output', { name: 'mu' }),
        M('os', 'Output', { name: 'sigma' }),
      ],
      edges: [E('a', 'mul'), E('b', 'mul'), E('mul', 'om'), E('a', 'add'), E('b', 'add'), E('add', 'oa'), E('a', 'stk'), E('b', 'stk'), E('stk', 'os')],
      inputs: [{ name: 'a', shape: [1, 4], dtype: 'float32' }, { name: 'b', shape: [1, 4], dtype: 'float32' }],
      expectedOutputShape: { aux: [1, 4], mu: [1, 4], sigma: [2, 1, 4] },
    },
    {
      kind: 'model',
      name: 'custom',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [1, 8], dtype: 'float32' }),
        M('scale', 'Custom', { init_args: 'factor=2.0', source: CUSTOM_SCALE_SRC }),
        M('out', 'Output', { name: 'logits' }),
      ],
      edges: [E('x', 'scale'), E('scale', 'out')],
      inputs: [{ name: 'x', shape: [1, 8], dtype: 'float32' }],
      expectedOutputShape: [1, 8],
    },
    {
      kind: 'model',
      name: 'subgraph',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [1, 64], dtype: 'float32' }),
        M('block', 'Subgraph', {
          class_name: 'MlpBlock',
          subgraph: {
            nodes: [
              SM('in', 'Input', { name: 'x', shape: [1, 64], dtype: 'float32' }),
              SM('fc1', 'Linear', { in_features: 64, out_features: 32, bias: true }),
              SM('act', 'ReLU', { inplace: false }),
              SM('out', 'Output', { name: 'out' }),
            ],
            edges: [E('in', 'fc1'), E('fc1', 'act'), E('act', 'out')],
          },
        }),
        M('out', 'Output', { name: 'logits' }),
      ],
      edges: [E('x', 'block'), E('block', 'out')],
      inputs: [{ name: 'x', shape: [1, 64], dtype: 'float32' }],
      expectedOutputShape: [1, 32],
    },
    {
      kind: 'model',
      name: 'dataop',
      nodes: [
        M('x', 'Input', { name: 'x', shape: [1, 8], dtype: 'float32' }),
        M('dop', 'DataOp', { input_dataset: 'datasets/raw.csv', output_name: 'processed/out', mode: 'shell', cache: true, script: DATAOP_SCRIPT }),
        M('lin', 'Linear', { in_features: 8, out_features: 4, bias: true }),
        M('out', 'Output', { name: 'logits' }),
      ],
      edges: [E('x', 'dop'), E('dop', 'lin'), E('lin', 'out')],
      inputs: [{ name: 'x', shape: [1, 8], dtype: 'float32' }],
      expectedOutputShape: [1, 4],
    },

    // ── TRAINING cases ──────────────────────────────────────────────────────
    {
      kind: 'training',
      name: 'train-single',
      nodes: [
        T('ds', 'DatasetSource', { dataset: 'datasets/features.csv', target: 'label', features: ['a', 'b'] }),
        T('split', 'Split', { strategy: 'random', val_ratio: 0.2, seed: 7 }),
        T('dl', 'DataLoader', { batch_size: 16, shuffle: true, num_workers: 0, drop_last: false }),
        T('model', 'ModelSource', { model: 'models/mlp.spinoml' }),
        T('loss', 'Loss', { kind: 'CrossEntropyLoss', label_smoothing: 0.0 }),
        T('opt', 'Optimizer', { kind: 'Adam', lr: 0.001, weight_decay: 0.0, momentum: 0.9 }),
        T('met', 'Metric', { kind: 'accuracy' }),
        T('es', 'EarlyStopping', { monitor: 'val_loss', patience: 5, mode: 'min' }),
        T('loop', 'TrainLoop', { epochs: 10, seed: 42, log_every_n_steps: 5, val_every_n_epochs: 1, gradient_accumulation_steps: 1 }),
      ],
      edges: [E('ds', 'loop'), E('split', 'loop'), E('dl', 'loop'), E('model', 'loop'), E('loss', 'loop'), E('opt', 'loop'), E('met', 'loop'), E('es', 'loop')],
    },
    {
      kind: 'training',
      name: 'train-multitask',
      nodes: [
        T('ds', 'DatasetSource', { dataset: 'datasets/pairs.csv', target: '', features: [] }),
        T('model', 'ModelSource', { model: 'models/dual.spinoml' }),
        T('opt', 'Optimizer', { kind: 'AdamW', lr: 0.0005, weight_decay: 0.01, momentum: 0.9 }),
        T('h1', 'Head', { output: 'logits', target: 'label', loss: 'CrossEntropyLoss', weight: 1.0, label_smoothing: 0.0 }),
        T('h2', 'Head', { output: 'mu', target: 'value', loss: 'MSELoss', weight: 0.5, label_smoothing: 0.0 }),
        T('gc', 'GradientClipping', { max_norm: 1.0 }),
        T('amp', 'MixedPrecision', { dtype: 'bf16' }),
        T('met', 'Metric', { kind: 'accuracy' }),
        T('loop', 'TrainLoop', { epochs: 5, seed: 42, log_every_n_steps: 10, val_every_n_epochs: 1, gradient_accumulation_steps: 1 }),
      ],
      edges: [E('ds', 'loop'), E('model', 'loop'), E('opt', 'loop'), E('h1', 'loop'), E('h2', 'loop'), E('gc', 'loop'), E('amp', 'loop'), E('met', 'loop')],
    },
    {
      kind: 'training',
      name: 'train-scheduler',
      nodes: [
        T('ds', 'DatasetSource', { dataset: 'datasets/tab.csv', target: 'y', features: [] }),
        T('model', 'ModelSource', { model: 'models/mlp.spinoml' }),
        T('loss', 'Loss', { kind: 'MSELoss', label_smoothing: 0.0 }),
        T('opt', 'Optimizer', { kind: 'SGD', lr: 0.01, weight_decay: 0.0, momentum: 0.9 }),
        T('sch', 'Scheduler', { kind: 'StepLR', step_size: 20, gamma: 0.5, patience: 10 }),
        T('loop', 'TrainLoop', { epochs: 30, seed: 42, log_every_n_steps: 10, val_every_n_epochs: 1, gradient_accumulation_steps: 1 }),
      ],
      edges: [E('ds', 'loop'), E('model', 'loop'), E('loss', 'loop'), E('opt', 'loop'), E('sch', 'loop')],
    },
    {
      kind: 'training',
      name: 'train-manifest',
      nodes: [
        T('ds', 'DatasetSource', { dataset: 'datasets/pair.manifest', target: '', features: [] }),
        T('split', 'Split', { strategy: 'predefined', val_ratio: 0.2, seed: 1 }),
        T('model', 'ModelSource', { model: 'models/dual.spinoml' }),
        T('loss', 'Loss', { kind: 'BCEWithLogitsLoss', label_smoothing: 0.0 }),
        T('opt', 'Optimizer', { kind: 'RMSprop', lr: 0.001, weight_decay: 0.0, momentum: 0.9 }),
        T('sch', 'Scheduler', { kind: 'ReduceLROnPlateau', step_size: 30, gamma: 0.1, patience: 3 }),
        T('met', 'Metric', { kind: 'f1' }),
        T('loop', 'TrainLoop', { epochs: 20, seed: 42, log_every_n_steps: 10, val_every_n_epochs: 1, gradient_accumulation_steps: 1 }),
      ],
      edges: [E('ds', 'loop'), E('split', 'loop'), E('model', 'loop'), E('loss', 'loop'), E('opt', 'loop'), E('sch', 'loop'), E('met', 'loop')],
    },

    // ── DATA cases ──────────────────────────────────────────────────────────
    {
      kind: 'data',
      name: 'data-simple',
      nodes: [
        D('src', 'TableSource', { dataset: 'datasets/raw.csv' }),
        D('norm', 'Normalize', { columns: '', method: 'zscore' }),
        D('out', 'WriteDataset', { out_path: 'datasets/processed.csv', format: 'csv' }),
      ],
      edges: [E('src', 'norm'), E('norm', 'out')],
    },
    {
      kind: 'data',
      name: 'data-select',
      nodes: [
        D('src', 'TableSource', { dataset: 'datasets/raw.csv' }),
        D('sel', 'SelectColumns', { columns: 'id, label, seq' }),
        D('flt', 'FilterRows', { query: 'label == 1' }),
        D('drop', 'DropNA', { subset: 'seq' }),
        D('comp', 'ComputeColumn', { name: 'len', expr: 'a + b' }),
        D('out', 'WriteDataset', { out_path: 'datasets/processed.csv', format: 'csv' }),
      ],
      edges: [E('src', 'sel'), E('sel', 'flt'), E('flt', 'drop'), E('drop', 'comp'), E('comp', 'out')],
    },
    {
      kind: 'data',
      name: 'data-custom',
      nodes: [
        D('src', 'TableSource', { dataset: 'datasets/raw.csv' }),
        D('cs', 'CustomScript', { label: 'dedup', code: CUSTOM_DEDUP_SRC }),
        D('ren', 'RenameColumns', { mapping: 'old:new' }),
        D('out', 'WriteDataset', { out_path: 'datasets/processed.csv', format: 'csv' }),
      ],
      edges: [E('src', 'cs'), E('cs', 'ren'), E('ren', 'out')],
    },
    {
      kind: 'data',
      name: 'data-download',
      nodes: [
        D('src', 'TableSource', { dataset: 'datasets/raw.csv' }),
        D('dl', 'DownloadColumn', { id_column: 'uniprot', url_template: 'https://files.rcsb.org/download/{id}.pdb', out_dir: 'datasets/raw', filename_template: '{id}.pdb', add_path_column: true }),
        D('out', 'WriteDataset', { out_path: 'datasets/processed.csv', format: 'csv' }),
      ],
      edges: [E('src', 'dl'), E('dl', 'out')],
    },
    {
      kind: 'data',
      name: 'data-smiles',
      nodes: [
        D('src', 'TableSource', { dataset: 'datasets/mols.csv' }),
        D('s2g', 'SmilesToGraph', { smiles_column: 'smiles', out_name: 'datasets/graphs/mol_graphs.pt', atom_features: 'standard', bond_features: true, embed_3d: true, add_hydrogens: true }),
      ],
      edges: [E('src', 's2g')],
    },
    {
      kind: 'data',
      name: 'data-structure',
      nodes: [
        D('src', 'TableSource', { dataset: 'datasets/pdbs.csv' }),
        D('s2g', 'StructureToGraph', { path_column: 'file_path', out_name: 'datasets/graphs/protein_graphs.pt', contact_threshold: 8.0 }),
      ],
      edges: [E('src', 's2g')],
    },
  ]
}

/** Serialise a fixture to the committed `.graph.json` format (stable key order,
 *  2-space indent, trailing newline). The `type` key in the in-memory builder
 *  is renamed to `layerType` / `trainingType` / `dataType` per kind so the file
 *  matches the actual registry field name on each canvas. */
export function serializeFixture(fx: GoldenFixture): string {
  const typeKey = fx.kind === 'model' ? 'layerType' : fx.kind === 'training' ? 'trainingType' : 'dataType'
  const obj: Record<string, unknown> = {
    kind: fx.kind,
    nodes: fx.nodes.map((n) => ({ id: n.id, [typeKey]: n.type, params: n.params })),
    edges: fx.edges,
  }
  if (fx.inputs) obj.inputs = fx.inputs
  if (fx.expectedOutputShape !== undefined) obj.expectedOutputShape = fx.expectedOutputShape
  if (fx.requires) obj.requires = fx.requires
  return JSON.stringify(obj, null, 2) + '\n'
}

/** Fixture-shape indexer for the test file (parses `.graph.json` back). */
export type FixtureFile = GoldenFixture & { name: string }
