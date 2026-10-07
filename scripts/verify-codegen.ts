import { execSync } from 'node:child_process'
import { writeFileSync, mkdtempSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import type { Edge } from '@xyflow/react'
import { generate } from '../src/codegen/generator'
import { defaultParamsFor } from '../src/layers/registry'
import type { LayerNode } from '../src/canvas/GraphStore'
import {
  LIGAND_PROJECTOR_SRC, PROTEIN_PROJECTOR_SRC, BILINEAR_HEAD_SRC,
  buildDualEncoderGnn,
} from '../src/templates/templates'

function mkNode(id: string, layerType: string, overrides: Record<string, unknown> = {}): LayerNode {
  return {
    id,
    type: 'layer',
    position: { x: 0, y: 0 },
    data: { layerType, params: { ...defaultParamsFor(layerType), ...overrides } },
  }
}

type Case = { name: string; nodes: LayerNode[]; edges: Edge[]; expectIssues?: number; forbid?: RegExp }

const cases: Case[] = [
  {
    name: 'sequential conv-bn-relu-flatten-linear',
    nodes: [
      mkNode('input', 'Input', { shape: [1, 3, 32, 32] }),
      mkNode('n1', 'Conv2d', { in_channels: 3, out_channels: 16 }),
      mkNode('n2', 'BatchNorm2d', { num_features: 16 }),
      mkNode('n3', 'ReLU'),
      mkNode('n4', 'AdaptiveAvgPool2d', { output_size: [1, 1] }),
      mkNode('n5', 'Flatten'),
      mkNode('n6', 'Linear', { in_features: 16, out_features: 10 }),
    ],
    edges: [
      { id: 'e1', source: 'input', target: 'n1' },
      { id: 'e2', source: 'n1', target: 'n2' },
      { id: 'e3', source: 'n2', target: 'n3' },
      { id: 'e4', source: 'n3', target: 'n4' },
      { id: 'e5', source: 'n4', target: 'n5' },
      { id: 'e6', source: 'n5', target: 'n6' },
    ],
  },
  {
    name: 'transformer encoder single block',
    nodes: [
      mkNode('input', 'Input', { shape: [1, 16, 512] }),
      mkNode('n1', 'TransformerEncoderLayer', {
        d_model: 512, nhead: 8, dim_feedforward: 1024,
        dropout: 0.1, activation: 'gelu', batch_first: true,
      }),
      mkNode('n2', 'LayerNorm', { normalized_shape: [512] }),
    ],
    edges: [
      { id: 'e1', source: 'input', target: 'n1' },
      { id: 'e2', source: 'n1', target: 'n2' },
    ],
  },
  {
    name: 'conv -> reshape -> permute -> linear (conv to token bridge)',
    nodes: [
      mkNode('input', 'Input', { shape: [1, 3, 8, 8] }),
      mkNode('n1', 'Conv2d', { in_channels: 3, out_channels: 16 }),
      mkNode('n2', 'Reshape', { shape: [16, -1] }),     // [1,16,8,8] -> [1,16,64]
      mkNode('n3', 'Permute', { dims: [0, 2, 1] }),      // -> [1,64,16]
      mkNode('n4', 'Linear', { in_features: 16, out_features: 10 }),
    ],
    edges: [
      { id: 'e1', source: 'input', target: 'n1' },
      { id: 'e2', source: 'n1', target: 'n2' },
      { id: 'e3', source: 'n2', target: 'n3' },
      { id: 'e4', source: 'n3', target: 'n4' },
    ],
  },
  {
    name: 'lstm sequence model (tuple output)',
    nodes: [
      mkNode('input', 'Input', { shape: [1, 16, 32] }),  // [N, L, C], batch_first
      mkNode('n1', 'LSTM', { input_size: 32, hidden_size: 64, num_layers: 1, batch_first: true }),
      mkNode('n2', 'Linear', { in_features: 64, out_features: 5 }),
    ],
    edges: [
      { id: 'e1', source: 'input', target: 'n1' },
      { id: 'e2', source: 'n1', target: 'n2' },
    ],
  },
  {
    name: 'embedding from token ids (int64 input)',
    nodes: [
      mkNode('input', 'Input', { shape: [1, 16], dtype: 'int64' }),
      mkNode('n1', 'Embedding', { num_embeddings: 1000, embedding_dim: 64 }),
      mkNode('n2', 'TransformerEncoder', { d_model: 64, nhead: 8, num_layers: 2, dim_feedforward: 128, batch_first: true }),
    ],
    edges: [
      { id: 'e1', source: 'input', target: 'n1' },
      { id: 'e2', source: 'n1', target: 'n2' },
    ],
  },
  {
    name: 'gcn node classification (edge_index side input)',
    nodes: [
      mkNode('x', 'Input', { name: 'x', shape: [10, 16] }),               // [N, F]
      mkNode('ei', 'Input', { name: 'edge_index', dtype: 'int64', shape: [2, 20] }),
      mkNode('n1', 'GCNConv', { in_channels: 16, out_channels: 32 }),
      mkNode('n2', 'ReLU'),
      mkNode('n3', 'GCNConv', { in_channels: 32, out_channels: 7 }),
    ],
    edges: [
      { id: 'e1', source: 'x', target: 'n1' },
      { id: 'e2', source: 'n1', target: 'n2' },
      { id: 'e3', source: 'n2', target: 'n3' },
    ],
  },
  {
    name: 'gat + global pooling graph classification',
    nodes: [
      mkNode('x', 'Input', { name: 'x', shape: [10, 16] }),
      mkNode('ei', 'Input', { name: 'edge_index', dtype: 'int64', shape: [2, 20] }),
      mkNode('b', 'Input', { name: 'batch', dtype: 'int64', shape: [10] }),
      mkNode('n1', 'GATConv', { in_channels: 16, out_channels: 8, heads: 4 }), // -> [10, 32]
      mkNode('n2', 'ReLU'),
      mkNode('n3', 'GlobalMeanPool'),                                          // -> [1, 32]
      mkNode('n4', 'Linear', { in_features: 32, out_features: 3 }),
    ],
    edges: [
      { id: 'e1', source: 'x', target: 'n1' },
      { id: 'e2', source: 'n1', target: 'n2' },
      { id: 'e3', source: 'n2', target: 'n3' },
      { id: 'e4', source: 'n3', target: 'n4' },
    ],
  },
  {
    name: 'rankbind bilinear main model (custom code nodes, 2 inputs)',
    nodes: [
      mkNode('lig', 'Input', { name: 'lig_emb', shape: [1, 384], dtype: 'float32' }),
      mkNode('prot', 'Input', { name: 'prot_emb', shape: [1, 1280], dtype: 'float32' }),
      mkNode('lp', 'Custom', { class_name: 'LigandProjector', init_args: '384, 256, dropout=0.0', source: LIGAND_PROJECTOR_SRC }),
      mkNode('pp', 'Custom', { class_name: 'ProteinProjector', init_args: '1280, 256, dropout=0.0', source: PROTEIN_PROJECTOR_SRC }),
      mkNode('bh', 'Custom', { class_name: 'BilinearHead', init_args: '256, 256, rank=32', source: BILINEAR_HEAD_SRC }),
      mkNode('out', 'Output', { name: 'score' }),
    ],
    edges: [
      { id: 'e1', source: 'lig', target: 'lp' },
      { id: 'e2', source: 'prot', target: 'pp' },
      { id: 'e3', source: 'lp', target: 'bh' },
      { id: 'e4', source: 'pp', target: 'bh' },
      { id: 'e5', source: 'bh', target: 'out' },
    ],
  },
  {
    name: 'group node compiles to a nested nn.Module (subcanvas)',
    nodes: [
      mkNode('input', 'Input', { name: 'x', shape: [1, 64], dtype: 'float32' }),
      mkNode('g', 'Subgraph', {
        class_name: 'MLPBlock',
        subgraph: {
          nodes: [
            { id: 'in', layerType: 'Input', params: { name: 'x', shape: [1, 64], dtype: 'float32' } },
            { id: 'fc1', layerType: 'Linear', params: { in_features: 64, out_features: 128 } },
            { id: 'act', layerType: 'ReLU', params: {} },
            { id: 'fc2', layerType: 'Linear', params: { in_features: 128, out_features: 32 } },
            { id: 'out', layerType: 'Output', params: { name: 'out' } },
          ],
          edges: [
            { source: 'in', target: 'fc1' },
            { source: 'fc1', target: 'act' },
            { source: 'act', target: 'fc2' },
            { source: 'fc2', target: 'out' },
          ],
        },
      }),
      mkNode('head', 'Linear', { in_features: 32, out_features: 10 }),
      mkNode('out', 'Output', { name: 'logits' }),
    ],
    edges: [
      { id: 'e1', source: 'input', target: 'g' },
      { id: 'e2', source: 'g', target: 'head' },
      { id: 'e3', source: 'head', target: 'out' },
    ],
  },
  (() => {
    // Dual-encoder GNN template: two GNN encoder Subgraphs (each owns its own
    // edge_index/batch) → Concat → MLP head. Verifies multi-input subgraphs and
    // that the two branches DON'T share one global edge_index.
    const snap = buildDualEncoderGnn()
    return {
      name: 'dual-encoder GNN (two graph branches → concat → head)',
      nodes: snap.nodes.map((n) => ({
        id: n.id, type: 'layer', position: n.position ?? { x: 0, y: 0 },
        data: { layerType: n.layerType, params: n.params },
      })) as LayerNode[],
      edges: snap.edges.map((e, i) => ({ id: `e${i}`, source: e.source, target: e.target })),
    }
  })(),
  {
    // A DataOp node is a DATA-stage passthrough: it must NOT emit anything into
    // the model (invariant #1 stays pure) yet must not break the Input→Linear
    // chain it sits in.
    name: 'dataop node is a passthrough (emits nothing into forward)',
    nodes: [
      mkNode('input', 'Input', { shape: [1, 8], dtype: 'float32' }),
      mkNode('dop', 'DataOp', { output_name: 'tok', input_dataset: 'datasets/seqs.csv' }),
      mkNode('lin', 'Linear', { in_features: 8, out_features: 4 }),
      mkNode('out', 'Output', { name: 'y' }),
    ],
    edges: [
      { id: 'e1', source: 'input', target: 'dop' },
      { id: 'e2', source: 'dop', target: 'lin' },
      { id: 'e3', source: 'lin', target: 'out' },
    ],
    forbid: /DataOp|data_op/,
  },
  {
    name: 'graph with no input (expect 1 issue)',
    nodes: [mkNode('n1', 'Conv2d')],
    edges: [],
    expectIssues: 1,
  },
  {
    name: 'graph with unreachable node (expect 1 issue)',
    nodes: [
      mkNode('input', 'Input'),
      mkNode('n1', 'Conv2d'),
      mkNode('orphan', 'ReLU'),
    ],
    edges: [{ id: 'e1', source: 'input', target: 'n1' }],
    expectIssues: 1,
  },
]

const tmp = mkdtempSync(join(tmpdir(), 'spinoml-codegen-'))
let failed = 0

for (const c of cases) {
  const { code, issues } = generate(c.nodes, c.edges)
  console.log(`\n=== ${c.name} ===`)
  console.log(code)
  if (c.expectIssues !== undefined) {
    if (issues.length === c.expectIssues) {
      console.log(`  ✓ issues=${issues.length} (expected ${c.expectIssues})`)
    } else {
      console.log(`  ✗ issues=${issues.length}, expected ${c.expectIssues}: ${JSON.stringify(issues)}`)
      failed++
    }
    continue
  }
  if (issues.length > 0) {
    console.log(`  ✗ unexpected issues: ${JSON.stringify(issues)}`)
    failed++
    continue
  }
  if (c.forbid && c.forbid.test(code)) {
    console.log(`  ✗ generated code unexpectedly matched ${c.forbid}`)
    failed++
    continue
  }
  const file = join(tmp, `${c.name.replace(/\W+/g, '_')}.py`)
  writeFileSync(file, code)
  try {
    const out = execSync(`python ${file}`, { encoding: 'utf-8' })
    console.log(`  ✓ runs:\n${out.split('\n').map((l) => '      ' + l).join('\n')}`)
  } catch (e: unknown) {
    const f = e as { stdout?: { toString(): string } | string; stderr?: { toString(): string } | string }
    console.log(`  ✗ python failed:\n${(f.stdout?.toString() ?? '') + (f.stderr?.toString() ?? '')}`)
    failed++
  }
}

console.log(`\n${failed === 0 ? '✓' : '✗'} ${cases.length - failed}/${cases.length} passed`)
process.exit(failed === 0 ? 0 : 1)
