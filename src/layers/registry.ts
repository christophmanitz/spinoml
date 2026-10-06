export type FieldSpec =
  | { name: string; type: 'int'; min?: number; max?: number; step?: number; default: number }
  | { name: string; type: 'float'; min?: number; max?: number; step?: number; default: number }
  | { name: string; type: 'bool'; default: boolean }
  | { name: string; type: 'select'; options: string[]; default: string }
  | { name: string; type: 'tuple-int'; arity: 2 | 3; default: number[] }
  /** Comma-separated list of ints; negatives allowed (e.g. -1 for reshape).
   *  Used by functional reshape/permute nodes via forwardExpr. */
  | { name: string; type: 'int-list'; default: number[] }
  | { name: string; type: 'shape'; default: number[] }
  /** Runtime-populated dropdown of dataset relpaths from datasetsStore. Stored as string. */
  | { name: string; type: 'dataset-ref'; default: string }
  /** Runtime-populated multi-select of column names from the node's bound dataset
   *  (looks up its sibling 'dataset' param + datasetsStore.inspects). */
  | { name: string; type: 'columns-multi'; default: string[] }
  /** Single-select column. Same lookup as columns-multi, single string. */
  | { name: string; type: 'column-single'; default: string }
  /** Single-line free text (class name, constructor-arg string). Stored verbatim. */
  | { name: string; type: 'text'; default: string; placeholder?: string; datalist?: string[] }
  /** Multi-line source code (Python for a Custom node). Stored & emitted verbatim. */
  | { name: string; type: 'code'; default: string; placeholder?: string }

export type LayerKind = 'module' | 'input' | 'output' | 'merge' | 'function' | 'custom' | 'group' | 'dataop' | 'manifest'

export type LayerSpec = {
  type: string
  category: string
  /** PyTorch nn.Module path (e.g. "nn.Conv2d"). Empty for non-module kinds. */
  pytorchModule: string
  /** What kind of node this is. Controls codegen. Defaults to 'module' if absent.
   *  'merge'    — N→1 functional join (≥2 inputs), emits forwardExpr.
   *  'function' — 1→1 functional transform (reshape/permute), emits forwardExpr. */
  kind?: LayerKind
  /** For kind='merge'/'function': how to emit the forward expression given input
   *  variable names. `aux` carries the resolved var names of special graph inputs
   *  (an Input named 'edge_index' / 'batch') for GNN pooling ops.
   *  e.g. (xs) => `torch.cat([${xs.join(', ')}], dim=${dim})`. */
  forwardExpr?: (
    inputVars: string[],
    params: Record<string, unknown>,
    aux: { edgeIndex?: string; batch?: string },
  ) => string
  /** For module kinds whose constructor can't be expressed as flat kwargs
   *  (e.g. nn.TransformerEncoder wrapping a layer). Returns the full RHS of
   *  `self.attr = <expr>`. When present, the default kwargs emit is skipped. */
  initExpr?: (params: Record<string, unknown>) => string
  /** Module returns a tuple (output, state); codegen unpacks `var, _ = ...` and
   *  the sidecar forward-hook reads element 0. e.g. nn.LSTM/GRU/RNN. */
  tupleOutput?: boolean
  /** Self-attention module: forward needs (query, key, value) and returns
   *  (output, weights). Codegen emits `var = self.attr(x, x, x, need_weights=False)[0]`
   *  (nn.MultiheadAttention). A single-argument call is a TypeError at run time. */
  selfAttention?: boolean
  /** Message-passing module whose forward takes (x, edge_index). Codegen emits
   *  `var = self.attr(pred, edge_index)`, resolving edge_index from an Input
   *  node named 'edge_index'. e.g. GCNConv/GATConv/SAGEConv. */
  needsEdgeIndex?: boolean
  /** An Input node that carries a WHOLE PyG graph (a Data/Batch object), not a
   *  bare tensor. In codegen its forward arg is the Data object; where a GNN/pool
   *  consumes it, the generator emits `x, edge_index, batch = g.x, g.edge_index,
   *  g.batch` (explicit, nothing hidden) and feeds aux from it. One node binds
   *  1:1 to a .pt / graph_folder / pyg / molecule / manifest-branch. */
  graphInput?: boolean
  /** Symbols this layer needs from torch_geometric.nn. The generator unions
   *  these across all used nodes into one `from torch_geometric.nn import …`. */
  pyImports?: string[]
  fields: FieldSpec[]
  summary: (params: Record<string, unknown>) => string
}

const f = {
  int: (name: string, def: number, opts: { min?: number; max?: number; step?: number } = {}): FieldSpec => ({
    name, type: 'int', default: def, ...opts,
  }),
  float: (name: string, def: number, opts: { min?: number; max?: number; step?: number } = {}): FieldSpec => ({
    name, type: 'float', default: def, ...opts,
  }),
  bool: (name: string, def: boolean): FieldSpec => ({ name, type: 'bool', default: def }),
  select: (name: string, options: string[], def: string): FieldSpec => ({
    name, type: 'select', options, default: def,
  }),
  tuple2: (name: string, def: number[]): FieldSpec => ({ name, type: 'tuple-int', arity: 2, default: def }),
  tuple3: (name: string, def: number[]): FieldSpec => ({ name, type: 'tuple-int', arity: 3, default: def }),
  intList: (name: string, def: number[]): FieldSpec => ({ name, type: 'int-list', default: def }),
  shape: (name: string, def: number[]): FieldSpec => ({ name, type: 'shape', default: def }),
}

const get = <T>(p: Record<string, unknown>, k: string, fallback: T): T =>
  (p[k] as T) ?? fallback

/** Parse the first `class Name(...)` from Custom-node source. The Custom node
 *  has no separate class-name field — this IS its name, so it can't drift from
 *  what the codegen instantiates. */
export function classNameFromSource(source: string): string | null {
  const m = source.match(/^\s*class\s+([A-Za-z_]\w*)\s*[(:]/m)
  return m ? m[1] : null
}

export const LAYERS: Record<string, LayerSpec> = {
  Input: {
    type: 'Input', category: 'IO', pytorchModule: '', kind: 'input',
    fields: [
      // Free text (not a fixed select) so it can match ANY graph-dataset field
      // name — e.g. a custom 'coords' field; suggestions via datalist.
      { name: 'name', type: 'text', default: 'x', placeholder: 'x', datalist: ['x', 'x1', 'x2', 'x3', 'q', 'k', 'v', 'cond', 'edge_index', 'edge_attr', 'pos', 'batch', 'y'] } as FieldSpec,
      f.shape('shape', [1, 3, 224, 224]),
      // 'int64' makes the sample input a LongTensor — required by Embedding (token ids).
      { name: 'dtype', type: 'select', options: ['float32', 'int64'], default: 'float32' } as FieldSpec,
      { name: 'dataset', type: 'dataset-ref', default: '' } as FieldSpec,
      { name: 'features', type: 'columns-multi', default: [] } as FieldSpec,
      { name: 'target', type: 'column-single', default: '' } as FieldSpec,
    ],
    summary: (p) => {
      const ds = String(get(p, 'dataset', ''))
      const dsTag = ds ? ` ← ${ds.split('/').pop()}` : ''
      const feats = get(p, 'features', []) as string[]
      const featTag = ds && feats.length ? ` · ${feats.length} feat` : ''
      return `${get(p, 'name', 'x')} ∈ ${JSON.stringify(get(p, 'shape', [1, 3, 224, 224]))}${dsTag}${featTag}`
    },
  },
  Graph: {
    // ONE node = one whole PyG graph (Data/Batch). Feeds a GNN branch with its
    // node features + connectivity + graph membership in a single, intuitive
    // node. The generator unpacks it explicitly at the point of use, so the real
    // x / edge_index / batch mechanics stay visible — nothing is simplified away.
    type: 'Graph', category: 'IO', pytorchModule: '', kind: 'input', graphInput: true,
    fields: [
      { name: 'name', type: 'text', default: 'data', placeholder: 'data', datalist: ['data', 'graph', 'ligand', 'protein', 'mol', 'enzyme'] } as FieldSpec,
      // x shape [N nodes, F node-features]. edge_index is [2, n_edges];
      // edge_attr (if edge_dim>0) is [n_edges, edge_dim]; batch is [N].
      f.shape('shape', [32, 9]),
      f.int('n_edges', 64, { min: 0 }),
      f.int('edge_dim', 0, { min: 0 }),
      { name: 'dataset', type: 'dataset-ref', default: '' } as FieldSpec,
      // For a manifest dataset: which branch ('ligand'/'protein'/…) this graph is.
      { name: 'branch', type: 'text', default: '', placeholder: 'ligand' } as FieldSpec,
    ],
    summary: (p) => {
      const sh = get(p, 'shape', [32, 9]) as number[]
      const ds = String(get(p, 'dataset', ''))
      const branch = String(get(p, 'branch', ''))
      const tag = ds ? ` ← ${ds.split('/').pop()}${branch ? `:${branch}` : ''}` : ''
      return `${get(p, 'name', 'data')}: graph x${JSON.stringify(sh)}${tag}`
    },
  },
  Output: {
    type: 'Output', category: 'IO', pytorchModule: '', kind: 'output',
    fields: [
      { name: 'name', type: 'select', options: ['out', 'logits', 'embedding', 'mu', 'sigma', 'aux'], default: 'out' } as FieldSpec,
    ],
    summary: (p) => `→ ${get(p, 'name', 'out')}`,
  },
  Manifest: {
    // A paired-dataset DESCRIPTOR — NOT a tensor input. It declares a .manifest
    // (what belongs to what, row-by-row) and feeds its branches to typed input
    // nodes (Graph / Sequence / Input) via edges. Emits NO code: the pairing lives
    // in the data layer; each connected input node is still a forward arg, fed its
    // branch at smoke-test / training time. The manifest comes from THIS node, so
    // typed inputs no longer each load the manifest themselves.
    type: 'Manifest', category: 'IO', pytorchModule: '', kind: 'manifest',
    fields: [
      { name: 'dataset', type: 'dataset-ref', default: '' } as FieldSpec,
    ],
    summary: (p) => {
      const ds = String(get(p, 'dataset', ''))
      return ds ? `manifest ${ds.split('/').pop()}` : '(keine .manifest gewählt)'
    },
  },
  Sequence: {
    // A token-id sequence input (LongTensor) — e.g. a protein_seq manifest branch.
    // Distinct from Graph (whole PyG Data); feeds an Embedding/Transformer encoder.
    type: 'Sequence', category: 'IO', pytorchModule: '', kind: 'input',
    fields: [
      { name: 'name', type: 'text', default: 'seq', placeholder: 'seq', datalist: ['seq', 'tokens', 'protein_seq', 'smiles_ids'] } as FieldSpec,
      f.shape('shape', [512]),
      { name: 'dtype', type: 'select', options: ['int64', 'float32'], default: 'int64' } as FieldSpec,
      // Which manifest branch feeds this input (filled from the connected Manifest).
      { name: 'branch', type: 'text', default: '', placeholder: 'protein_seq' } as FieldSpec,
    ],
    summary: (p) => {
      const sh = get(p, 'shape', [512]) as number[]
      const br = String(get(p, 'branch', ''))
      return `${get(p, 'name', 'seq')}: tokens ${JSON.stringify(sh)}${br ? ` :${br}` : ''}`
    },
  },
  ESPF: {
    // ESPF (Explainable Substructure Partition Fingerprint, MolTrans): a SMILES
    // string tokenized into INTERPRETABLE SUBSTRUCTURE subword tokens (LongTensor)
    // via a BPE codebook — every token id ↔ a named substructure. Like Sequence it
    // is a token-id input that feeds an Embedding/Transformer, but bound to a SMILES
    // manifest branch declared `kind:"espf"`. The sidecar tokenizes inline at
    // sample/train time; num_embeddings = ESPF vocab (≈23.5k drug / shown in the
    // Inspector). Substructures surface in the Explain view.
    type: 'ESPF', category: 'IO', pytorchModule: '', kind: 'input',
    fields: [
      { name: 'name', type: 'text', default: 'smiles_espf', placeholder: 'smiles_espf', datalist: ['smiles_espf', 'ligand_espf', 'drug_tokens'] } as FieldSpec,
      // MolTrans uses max_d=50 ESPF tokens per drug; [L] is a 1-D token sequence.
      f.shape('shape', [50]),
      // Token ids → Embedding, so always a LongTensor.
      { name: 'dtype', type: 'select', options: ['int64'], default: 'int64' } as FieldSpec,
      // Which BPE codebook: 'drug' (SMILES, ChEMBL) or 'protein' (UniProt).
      f.select('codebook', ['drug', 'protein'], 'drug'),
      // Which manifest branch (kind:"espf") feeds this input.
      { name: 'branch', type: 'text', default: '', placeholder: 'ligand_smiles' } as FieldSpec,
    ],
    summary: (p) => {
      const sh = get(p, 'shape', [50]) as number[]
      const cb = String(get(p, 'codebook', 'drug'))
      const br = String(get(p, 'branch', ''))
      return `${get(p, 'name', 'smiles_espf')}: ESPF ${cb} ${JSON.stringify(sh)}${br ? ` :${br}` : ''}`
    },
  },

  Conv2d: {
    type: 'Conv2d', category: 'Conv', pytorchModule: 'nn.Conv2d',
    fields: [
      f.int('in_channels', 3, { min: 1 }),
      f.int('out_channels', 64, { min: 1 }),
      f.tuple2('kernel_size', [3, 3]),
      f.tuple2('stride', [1, 1]),
      f.tuple2('padding', [1, 1]),
      f.bool('bias', true),
    ],
    summary: (p) => `${get(p, 'in_channels', 3)}→${get(p, 'out_channels', 64)} k${(get(p, 'kernel_size', [3, 3]) as number[]).join('x')}`,
  },
  Conv1d: {
    type: 'Conv1d', category: 'Conv', pytorchModule: 'nn.Conv1d',
    fields: [
      f.int('in_channels', 1, { min: 1 }),
      f.int('out_channels', 16, { min: 1 }),
      f.int('kernel_size', 3, { min: 1 }),
      f.int('stride', 1, { min: 1 }),
      f.int('padding', 1, { min: 0 }),
      f.bool('bias', true),
    ],
    summary: (p) => `${get(p, 'in_channels', 1)}→${get(p, 'out_channels', 16)} k${get(p, 'kernel_size', 3)}`,
  },
  ConvTranspose2d: {
    type: 'ConvTranspose2d', category: 'Conv', pytorchModule: 'nn.ConvTranspose2d',
    fields: [
      f.int('in_channels', 64, { min: 1 }),
      f.int('out_channels', 32, { min: 1 }),
      f.tuple2('kernel_size', [3, 3]),
      f.tuple2('stride', [2, 2]),
      f.tuple2('padding', [1, 1]),
    ],
    summary: (p) => `${get(p, 'in_channels', 64)}→${get(p, 'out_channels', 32)} k${(get(p, 'kernel_size', [3, 3]) as number[]).join('x')} ↑`,
  },
  Conv3d: {
    type: 'Conv3d', category: 'Conv', pytorchModule: 'nn.Conv3d',
    fields: [
      f.int('in_channels', 3, { min: 1 }),
      f.int('out_channels', 16, { min: 1 }),
      f.tuple3('kernel_size', [3, 3, 3]),
      f.tuple3('stride', [1, 1, 1]),
      f.tuple3('padding', [1, 1, 1]),
      f.bool('bias', true),
    ],
    summary: (p) => `${get(p, 'in_channels', 3)}→${get(p, 'out_channels', 16)} k${(get(p, 'kernel_size', [3, 3, 3]) as number[]).join('x')}`,
  },

  Linear: {
    type: 'Linear', category: 'Linear', pytorchModule: 'nn.Linear',
    fields: [
      f.int('in_features', 512, { min: 1 }),
      f.int('out_features', 128, { min: 1 }),
      f.bool('bias', true),
    ],
    summary: (p) => `${get(p, 'in_features', 512)}→${get(p, 'out_features', 128)}`,
  },
  Flatten: {
    type: 'Flatten', category: 'Linear', pytorchModule: 'nn.Flatten',
    fields: [
      f.int('start_dim', 1, { min: 0 }),
      f.int('end_dim', -1),
    ],
    summary: (p) => `dims ${get(p, 'start_dim', 1)}…${get(p, 'end_dim', -1)}`,
  },
  Embedding: {
    // Maps integer token ids → dense vectors. Feed it from an Input whose
    // dtype is 'int64' (a LongTensor); a float Input will raise at forward.
    type: 'Embedding', category: 'Linear', pytorchModule: 'nn.Embedding',
    fields: [
      f.int('num_embeddings', 1000, { min: 1 }),
      f.int('embedding_dim', 128, { min: 1 }),
    ],
    summary: (p) => `emb ${get(p, 'num_embeddings', 1000)}×${get(p, 'embedding_dim', 128)}`,
  },

  BatchNorm2d: {
    type: 'BatchNorm2d', category: 'Norm', pytorchModule: 'nn.BatchNorm2d',
    fields: [
      f.int('num_features', 64, { min: 1 }),
      f.float('eps', 1e-5, { step: 1e-6 }),
      f.float('momentum', 0.1, { min: 0, max: 1, step: 0.01 }),
    ],
    summary: (p) => `bn ${get(p, 'num_features', 64)}`,
  },
  BatchNorm1d: {
    type: 'BatchNorm1d', category: 'Norm', pytorchModule: 'nn.BatchNorm1d',
    fields: [
      f.int('num_features', 64, { min: 1 }),
      f.float('eps', 1e-5, { step: 1e-6 }),
      f.float('momentum', 0.1, { min: 0, max: 1, step: 0.01 }),
    ],
    summary: (p) => `bn1d ${get(p, 'num_features', 64)}`,
  },
  LayerNorm: {
    type: 'LayerNorm', category: 'Norm', pytorchModule: 'nn.LayerNorm',
    fields: [
      f.shape('normalized_shape', [512]),
      f.float('eps', 1e-5, { step: 1e-6 }),
    ],
    summary: (p) => `ln ${JSON.stringify(get(p, 'normalized_shape', [512]))}`,
  },
  GroupNorm: {
    type: 'GroupNorm', category: 'Norm', pytorchModule: 'nn.GroupNorm',
    fields: [
      f.int('num_groups', 8, { min: 1 }),
      f.int('num_channels', 64, { min: 1 }),
    ],
    summary: (p) => `gn ${get(p, 'num_groups', 8)}g·${get(p, 'num_channels', 64)}c`,
  },

  ReLU: { type: 'ReLU', category: 'Activation', pytorchModule: 'nn.ReLU', fields: [f.bool('inplace', false)], summary: () => 'relu' },
  GELU: { type: 'GELU', category: 'Activation', pytorchModule: 'nn.GELU', fields: [], summary: () => 'gelu' },
  SiLU: { type: 'SiLU', category: 'Activation', pytorchModule: 'nn.SiLU', fields: [f.bool('inplace', false)], summary: () => 'silu' },
  Sigmoid: { type: 'Sigmoid', category: 'Activation', pytorchModule: 'nn.Sigmoid', fields: [], summary: () => 'sigmoid' },
  Tanh: { type: 'Tanh', category: 'Activation', pytorchModule: 'nn.Tanh', fields: [], summary: () => 'tanh' },
  Softmax: {
    type: 'Softmax', category: 'Activation', pytorchModule: 'nn.Softmax',
    fields: [f.int('dim', -1)],
    summary: (p) => `softmax dim=${get(p, 'dim', -1)}`,
  },
  LogSoftmax: {
    type: 'LogSoftmax', category: 'Activation', pytorchModule: 'nn.LogSoftmax',
    fields: [f.int('dim', -1)],
    summary: (p) => `log_softmax dim=${get(p, 'dim', -1)}`,
  },

  MaxPool2d: {
    type: 'MaxPool2d', category: 'Pool', pytorchModule: 'nn.MaxPool2d',
    fields: [
      f.tuple2('kernel_size', [2, 2]),
      f.tuple2('stride', [2, 2]),
      f.tuple2('padding', [0, 0]),
    ],
    summary: (p) => `max k${(get(p, 'kernel_size', [2, 2]) as number[]).join('x')}`,
  },
  AvgPool2d: {
    type: 'AvgPool2d', category: 'Pool', pytorchModule: 'nn.AvgPool2d',
    fields: [
      f.tuple2('kernel_size', [2, 2]),
      f.tuple2('stride', [2, 2]),
      f.tuple2('padding', [0, 0]),
    ],
    summary: (p) => `avg k${(get(p, 'kernel_size', [2, 2]) as number[]).join('x')}`,
  },
  AdaptiveAvgPool2d: {
    type: 'AdaptiveAvgPool2d', category: 'Pool', pytorchModule: 'nn.AdaptiveAvgPool2d',
    fields: [f.tuple2('output_size', [1, 1])],
    summary: (p) => `adaptive ${(get(p, 'output_size', [1, 1]) as number[]).join('x')}`,
  },

  Dropout: {
    type: 'Dropout', category: 'Regularize', pytorchModule: 'nn.Dropout',
    fields: [f.float('p', 0.5, { min: 0, max: 1, step: 0.05 })],
    summary: (p) => `p=${get(p, 'p', 0.5)}`,
  },
  Dropout2d: {
    type: 'Dropout2d', category: 'Regularize', pytorchModule: 'nn.Dropout2d',
    fields: [f.float('p', 0.5, { min: 0, max: 1, step: 0.05 })],
    summary: (p) => `2d p=${get(p, 'p', 0.5)}`,
  },

  MultiheadAttention: {
    type: 'MultiheadAttention', category: 'Attention', pytorchModule: 'nn.MultiheadAttention', selfAttention: true,
    fields: [
      f.int('embed_dim', 512, { min: 1 }),
      f.int('num_heads', 8, { min: 1 }),
      f.float('dropout', 0.0, { min: 0, max: 1, step: 0.05 }),
      f.bool('batch_first', true),
    ],
    summary: (p) => `mha d=${get(p, 'embed_dim', 512)} h=${get(p, 'num_heads', 8)}`,
  },
  TransformerEncoderLayer: {
    type: 'TransformerEncoderLayer', category: 'Attention', pytorchModule: 'nn.TransformerEncoderLayer',
    fields: [
      f.int('d_model', 512, { min: 1 }),
      f.int('nhead', 8, { min: 1 }),
      f.int('dim_feedforward', 2048, { min: 1 }),
      f.float('dropout', 0.1, { min: 0, max: 1, step: 0.05 }),
      f.select('activation', ['relu', 'gelu'], 'relu'),
      f.bool('batch_first', true),
    ],
    summary: (p) => `enc d=${get(p, 'd_model', 512)} h=${get(p, 'nhead', 8)}`,
  },
  TransformerEncoder: {
    // A stack of num_layers identical encoder blocks. The constructor wraps a
    // freshly-built TransformerEncoderLayer, so it needs a custom initExpr.
    type: 'TransformerEncoder', category: 'Attention', pytorchModule: 'nn.TransformerEncoder',
    fields: [
      f.int('d_model', 512, { min: 1 }),
      f.int('nhead', 8, { min: 1 }),
      f.int('num_layers', 6, { min: 1 }),
      f.int('dim_feedforward', 2048, { min: 1 }),
      f.float('dropout', 0.1, { min: 0, max: 1, step: 0.05 }),
      f.select('activation', ['relu', 'gelu'], 'gelu'),
      f.bool('batch_first', true),
    ],
    initExpr: (p) =>
      `nn.TransformerEncoder(nn.TransformerEncoderLayer(` +
      `d_model=${get(p, 'd_model', 512)}, nhead=${get(p, 'nhead', 8)}, ` +
      `dim_feedforward=${get(p, 'dim_feedforward', 2048)}, dropout=${get(p, 'dropout', 0.1)}, ` +
      `activation='${get(p, 'activation', 'gelu')}', batch_first=${get(p, 'batch_first', true) ? 'True' : 'False'}), ` +
      `num_layers=${get(p, 'num_layers', 6)})`,
    summary: (p) => `enc×${get(p, 'num_layers', 6)} d=${get(p, 'd_model', 512)}`,
  },

  // ─── Recurrent (return (output, state); codegen unpacks output) ─────────
  LSTM: {
    type: 'LSTM', category: 'Recurrent', pytorchModule: 'nn.LSTM', tupleOutput: true,
    fields: [
      f.int('input_size', 64, { min: 1 }),
      f.int('hidden_size', 128, { min: 1 }),
      f.int('num_layers', 1, { min: 1 }),
      f.bool('batch_first', true),
      f.bool('bidirectional', false),
      f.float('dropout', 0.0, { min: 0, max: 1, step: 0.05 }),
    ],
    summary: (p) => `lstm ${get(p, 'input_size', 64)}→${get(p, 'hidden_size', 128)}${get(p, 'bidirectional', false) ? ' ↔' : ''}`,
  },
  GRU: {
    type: 'GRU', category: 'Recurrent', pytorchModule: 'nn.GRU', tupleOutput: true,
    fields: [
      f.int('input_size', 64, { min: 1 }),
      f.int('hidden_size', 128, { min: 1 }),
      f.int('num_layers', 1, { min: 1 }),
      f.bool('batch_first', true),
      f.bool('bidirectional', false),
      f.float('dropout', 0.0, { min: 0, max: 1, step: 0.05 }),
    ],
    summary: (p) => `gru ${get(p, 'input_size', 64)}→${get(p, 'hidden_size', 128)}${get(p, 'bidirectional', false) ? ' ↔' : ''}`,
  },
  RNN: {
    type: 'RNN', category: 'Recurrent', pytorchModule: 'nn.RNN', tupleOutput: true,
    fields: [
      f.int('input_size', 64, { min: 1 }),
      f.int('hidden_size', 128, { min: 1 }),
      f.int('num_layers', 1, { min: 1 }),
      f.select('nonlinearity', ['tanh', 'relu'], 'tanh'),
      f.bool('batch_first', true),
      f.bool('bidirectional', false),
    ],
    summary: (p) => `rnn ${get(p, 'input_size', 64)}→${get(p, 'hidden_size', 128)}`,
  },

  // ─── Graph (torch_geometric; consume an Input named 'edge_index') ──────
  // Node features use the PyG convention [N_nodes, in_channels] (no batch
  // dim). edge_index is [2, N_edges] int64 — add an Input named 'edge_index'
  // with dtype 'int64'. Graph-level pooling additionally needs an Input
  // named 'batch' ([N_nodes] int64).
  GCNConv: {
    type: 'GCNConv', category: 'Graph', pytorchModule: 'GCNConv',
    needsEdgeIndex: true, pyImports: ['GCNConv'],
    fields: [
      // in_channels = -1 → PyG lazy init (inferred from the data on first forward),
      // so a GNN binds to any node-feature dimension without manual editing.
      f.int('in_channels', -1, { min: -1 }),
      f.int('out_channels', 32, { min: 1 }),
      f.bool('improved', false),
      f.bool('cached', false),
      f.bool('add_self_loops', true),
      f.bool('bias', true),
    ],
    summary: (p) => `gcn ${get(p, 'in_channels', -1) === -1 ? 'auto' : get(p, 'in_channels', -1)}→${get(p, 'out_channels', 32)}`,
  },
  GATConv: {
    type: 'GATConv', category: 'Graph', pytorchModule: 'GATConv',
    needsEdgeIndex: true, pyImports: ['GATConv'],
    fields: [
      f.int('in_channels', -1, { min: -1 }),
      f.int('out_channels', 32, { min: 1 }),
      f.int('heads', 1, { min: 1 }),
      f.bool('concat', true),
      f.float('dropout', 0.0, { min: 0, max: 1, step: 0.05 }),
      f.bool('bias', true),
    ],
    // out dim = out_channels * heads when concat, else out_channels.
    summary: (p) => `gat ${get(p, 'in_channels', -1) === -1 ? 'auto' : get(p, 'in_channels', -1)}→${get(p, 'out_channels', 32)}×${get(p, 'heads', 1)}h`,
  },
  SAGEConv: {
    type: 'SAGEConv', category: 'Graph', pytorchModule: 'SAGEConv',
    needsEdgeIndex: true, pyImports: ['SAGEConv'],
    fields: [
      f.int('in_channels', -1, { min: -1 }),
      f.int('out_channels', 32, { min: 1 }),
      f.select('aggr', ['mean', 'max', 'add', 'min'], 'mean'),
      f.bool('normalize', false),
      f.bool('bias', true),
    ],
    summary: (p) => `sage ${get(p, 'in_channels', -1) === -1 ? 'auto' : get(p, 'in_channels', -1)}→${get(p, 'out_channels', 32)}`,
  },
  GraphConv: {
    type: 'GraphConv', category: 'Graph', pytorchModule: 'GraphConv',
    needsEdgeIndex: true, pyImports: ['GraphConv'],
    fields: [
      f.int('in_channels', -1, { min: -1 }),
      f.int('out_channels', 32, { min: 1 }),
      f.select('aggr', ['add', 'mean', 'max'], 'add'),
      f.bool('bias', true),
    ],
    summary: (p) => `graphconv ${get(p, 'in_channels', -1) === -1 ? 'auto' : get(p, 'in_channels', -1)}→${get(p, 'out_channels', 32)}`,
  },
  GraphTransformer: {
    // PyG TransformerConv — multi-head graph attention à la the Graph Transformer.
    // out dim = out_channels * heads when concat, else out_channels (like GAT).
    type: 'GraphTransformer', category: 'Graph', pytorchModule: 'TransformerConv',
    needsEdgeIndex: true, pyImports: ['TransformerConv'],
    fields: [
      f.int('in_channels', 16, { min: 1 }),
      f.int('out_channels', 32, { min: 1 }),
      f.int('heads', 1, { min: 1 }),
      f.bool('concat', true),
      f.bool('beta', false),
      f.float('dropout', 0.0, { min: 0, max: 1, step: 0.05 }),
      f.bool('bias', true),
    ],
    summary: (p) => `gtrans ${get(p, 'in_channels', 16)}→${get(p, 'out_channels', 32)}×${get(p, 'heads', 1)}h`,
  },
  GlobalMeanPool: {
    // Graph-level readout: [N_nodes, F] → [N_graphs, F]. Needs a 'batch' Input.
    type: 'GlobalMeanPool', category: 'Graph', pytorchModule: '', kind: 'function',
    pyImports: ['global_mean_pool'], fields: [],
    forwardExpr: (xs, _p, aux) => `global_mean_pool(${xs[0]}, ${aux.batch ?? 'batch'})`,
    summary: () => 'mean pool → [B, F]',
  },
  GlobalMaxPool: {
    type: 'GlobalMaxPool', category: 'Graph', pytorchModule: '', kind: 'function',
    pyImports: ['global_max_pool'], fields: [],
    forwardExpr: (xs, _p, aux) => `global_max_pool(${xs[0]}, ${aux.batch ?? 'batch'})`,
    summary: () => 'max pool → [B, F]',
  },
  GlobalAddPool: {
    type: 'GlobalAddPool', category: 'Graph', pytorchModule: '', kind: 'function',
    pyImports: ['global_add_pool'], fields: [],
    forwardExpr: (xs, _p, aux) => `global_add_pool(${xs[0]}, ${aux.batch ?? 'batch'})`,
    summary: () => 'add pool → [B, F]',
  },

  // ─── Build a graph from node features at runtime (x → edge_index) ───────
  BuildGraph: {
    // Computes edge_index from the node features [N, F] so downstream GNN
    // layers don't need a separate edge_index Input. The generator picks the
    // produced edge_index up automatically (see codegen aux wiring).
    type: 'BuildGraph', category: 'IO', pytorchModule: '', kind: 'function',
    // Pure-torch (cdist + topk) — no torch-cluster / pyg-lib dependency, runs on CPU.
    // `dims` selects which columns of x define the distance (empty = all); the
    // node features passed downstream are always the full x.
    fields: [
      f.select('method', ['knn', 'radius', 'fully_connected'], 'knn'),
      f.int('k', 6, { min: 1 }),
      f.float('radius', 1.0, { min: 0, step: 0.1 }),
      f.intList('dims', []),
      f.bool('loop', false),
      f.bool('cosine', false),
    ],
    forwardExpr: (xs, p) => {
      const x = xs[0]
      const loop = get(p, 'loop', false)
      const method = String(get(p, 'method', 'knn'))
      if (method === 'fully_connected') {
        const cp = `torch.cartesian_prod(torch.arange(${x}.size(0), device=${x}.device), torch.arange(${x}.size(0), device=${x}.device)).t()`
        return loop ? cp : `(lambda _ei: _ei[:, _ei[0] != _ei[1]])(${cp})`
      }
      // The distance basis: a subset of x columns if `dims` is set, else all of x.
      const dims = get(p, 'dims', []) as number[]
      const xd = dims.length ? `${x}[:, [${dims.join(', ')}]]` : x
      if (method === 'radius') {
        const r = get(p, 'radius', 1.0)
        const filt = loop ? '_ei' : '_ei[:, _ei[0] != _ei[1]]'
        return `(lambda _ei: ${filt})((torch.cdist(${xd}, ${xd}) <= ${r}).nonzero().t())`
      }
      // knn: distances → k nearest per node (excluding self unless loop)
      const k = get(p, 'k', 6)
      const norm = `(${xd} / ${xd}.norm(dim=1, keepdim=True).clamp_min(1e-12))`
      const dist = get(p, 'cosine', false) ? `(1 - ${norm} @ ${norm}.t())` : `torch.cdist(${xd}, ${xd})`
      const nbr = loop
        ? `${dist}.topk(min(${k}, ${x}.size(0)), largest=False).indices`
        : `${dist}.topk(min(${k} + 1, ${x}.size(0)), largest=False).indices[:, 1:]`
      return `(lambda _nbr: torch.stack([torch.arange(${x}.size(0), device=${x}.device).repeat_interleave(_nbr.size(1)), _nbr.reshape(-1)]))(${nbr})`
    },
    summary: (p) => {
      const m = String(get(p, 'method', 'knn'))
      const dims = get(p, 'dims', []) as number[]
      const on = dims.length ? ` on dims [${dims.join(',')}]` : ''
      if (m === 'radius') return `graph: radius r=${get(p, 'radius', 1.0)}${on}`
      if (m === 'fully_connected') return 'graph: voll-verbunden'
      return `graph: kNN k=${get(p, 'k', 6)}${on}`
    },
  },

  // ─── Reshape (functional 1→1, no nn.Module) ────────────────────────────
  Reshape: {
    // Batch dim is preserved automatically; `shape` is the per-sample target
    // (use -1 to infer). e.g. shape=[16, -1] → x.reshape(x.shape[0], 16, -1).
    type: 'Reshape', category: 'Reshape', pytorchModule: '', kind: 'function',
    fields: [f.intList('shape', [-1])],
    forwardExpr: (xs, p) =>
      `${xs[0]}.reshape(${xs[0]}.shape[0], ${(get(p, 'shape', [-1]) as number[]).join(', ')})`,
    summary: (p) => `reshape [B, ${(get(p, 'shape', [-1]) as number[]).join(', ')}]`,
  },
  View: {
    type: 'View', category: 'Reshape', pytorchModule: '', kind: 'function',
    fields: [f.intList('shape', [-1])],
    forwardExpr: (xs, p) =>
      `${xs[0]}.reshape(${xs[0]}.shape[0], ${(get(p, 'shape', [-1]) as number[]).join(', ')}).contiguous()`,
    summary: (p) => `view [B, ${(get(p, 'shape', [-1]) as number[]).join(', ')}]`,
  },
  Permute: {
    // dims are the full permutation INCLUDING the batch dim, e.g. [0, 2, 1].
    type: 'Permute', category: 'Reshape', pytorchModule: '', kind: 'function',
    fields: [f.intList('dims', [0, 2, 1])],
    forwardExpr: (xs, p) => `${xs[0]}.permute(${(get(p, 'dims', [0, 2, 1]) as number[]).join(', ')})`,
    summary: (p) => `permute (${(get(p, 'dims', [0, 2, 1]) as number[]).join(', ')})`,
  },
  Transpose: {
    type: 'Transpose', category: 'Reshape', pytorchModule: '', kind: 'function',
    fields: [f.int('dim0', 1), f.int('dim1', 2)],
    forwardExpr: (xs, p) => `${xs[0]}.transpose(${get(p, 'dim0', 1)}, ${get(p, 'dim1', 2)})`,
    summary: (p) => `transpose ${get(p, 'dim0', 1)}↔${get(p, 'dim1', 2)}`,
  },

  // ─── Merge (functional, no nn.Module) ──────────────────────────────────
  Concat: {
    type: 'Concat', category: 'Merge', pytorchModule: '', kind: 'merge',
    fields: [f.int('dim', 1)],
    forwardExpr: (xs, p) => `torch.cat([${xs.join(', ')}], dim=${get(p, 'dim', 1)})`,
    summary: (p) => `cat dim=${get(p, 'dim', 1)}`,
  },
  Add: {
    type: 'Add', category: 'Merge', pytorchModule: '', kind: 'merge',
    fields: [],
    forwardExpr: (xs) => xs.length <= 1 ? (xs[0] ?? '0')
      : xs.reduce((acc, v) => `${acc} + ${v}`),
    summary: () => 'a + b',
  },
  Multiply: {
    type: 'Multiply', category: 'Merge', pytorchModule: '', kind: 'merge',
    fields: [],
    forwardExpr: (xs) => xs.length <= 1 ? (xs[0] ?? '1')
      : xs.reduce((acc, v) => `${acc} * ${v}`),
    summary: () => 'a · b',
  },
  Stack: {
    type: 'Stack', category: 'Merge', pytorchModule: '', kind: 'merge',
    fields: [f.int('dim', 0)],
    forwardExpr: (xs, p) => `torch.stack([${xs.join(', ')}], dim=${get(p, 'dim', 0)})`,
    summary: (p) => `stack dim=${get(p, 'dim', 0)}`,
  },

  // ─── Custom (free-form nn.Module — the escape hatch) ────────────────────
  // Write any nn.Module in `source`; the codegen emits the class verbatim at
  // module level, instantiates it as `self.<attr> = <class_name>(<init_args>)`,
  // and calls it in forward with ALL incoming edges as positional args (in edge
  // order). One forward output (tensor / tuple / dict). torch, nn and F
  // (torch.nn.functional) are imported for you; put any other imports at the top
  // of `source`. This is how non-graph ops (bilinear heads, custom attention,
  // relational message passing, …) get expressed.
  Custom: {
    type: 'Custom', category: 'Custom', pytorchModule: '', kind: 'custom',
    // The class name is the `class X(...)` in `source` — single source of truth,
    // so it can never drift from what gets instantiated. `init_args` is the
    // constructor arg string.
    fields: [
      { name: 'init_args', type: 'text', default: '', placeholder: 'z.B. 384, 256, dropout=0.0' } as FieldSpec,
      {
        name: 'source', type: 'code',
        placeholder: 'class MyModule(nn.Module): ...',
        // No-arg __init__ + LazyLinear → a freshly dropped node constructs and
        // runs immediately (init_args can stay empty). Edit freely.
        default: [
          'class MyModule(nn.Module):',
          '    def __init__(self):',
          '        super().__init__()',
          '        self.fc = nn.LazyLinear(32)',
          '',
          '    def forward(self, x):',
          '        return self.fc(x)',
        ].join('\n'),
      } as FieldSpec,
    ],
    summary: (p) => {
      const cls = classNameFromSource(String(get(p, 'source', ''))) ?? '?'
      const args = String(get(p, 'init_args', '')).trim()
      return `${cls}(${args})`
    },
  },

  // ─── DataOp (a DATA-stage node, not a model layer) ───────────────────────
  // Runs a self-written Python script ONCE (offline, via the agent run_script
  // path) to download / tokenize / transform / cache a dataset, materializing a
  // cached output (and optionally a manifest branch) that an Input then binds.
  // It is NOT part of the model forward pass: codegen treats it as a pure
  // passthrough (Input → DataOp → Layer keeps working; the DataOp emits nothing).
  // The script is executed from the Inspector's "Vorverarbeitung ausführen"
  // button, which writes it to agent/ and run_scripts it (with the confirm GUI).
  DataOp: {
    type: 'DataOp', category: 'Data', pytorchModule: '', kind: 'dataop',
    fields: [
      { name: 'input_dataset', type: 'dataset-ref', default: '' } as FieldSpec,
      { name: 'output_name', type: 'text', default: 'processed/out', placeholder: 'z.B. protein_seq_tokens' } as FieldSpec,
      { name: 'mode', type: 'select', options: ['shell', 'slurm'], default: 'shell' } as FieldSpec,
      { name: 'cache', type: 'bool', default: true } as FieldSpec,
      {
        name: 'script', type: 'code',
        placeholder: 'Python: read input_dataset → write output under datasets/',
        // A runnable-shaped template the user/agent edits. Reads its config from
        // argv (the Inspector passes --input/--output) and writes a cached result
        // under datasets/. Kept deliberately minimal — the agent fills the body.
        default: [
          'import sys, os, argparse',
          '',
          '# DataOp: transform an input dataset into a cached output under datasets/.',
          '# Invoked by SpinoML as:  python <this>.py --input <rel> --output <rel>',
          'def main():',
          '    ap = argparse.ArgumentParser()',
          '    ap.add_argument("--input", default="")',
          '    ap.add_argument("--output", default="processed/out")',
          '    args = ap.parse_args()',
          '    out_dir = os.path.join("datasets", args.output)',
          '    os.makedirs(out_dir, exist_ok=True)',
          '    # TODO: read args.input, transform it, write tensors/files into out_dir,',
          '    #       and (for a paired model) extend the .manifest with a new branch.',
          '    print(f"wrote nothing yet — fill in main(); input={args.input} output={out_dir}")',
          '',
          'if __name__ == "__main__":',
          '    main()',
        ].join('\n'),
      } as FieldSpec,
    ],
    summary: (p) => {
      const out = String(get(p, 'output_name', 'out')) || 'out'
      const mode = String(get(p, 'mode', 'shell'))
      const src = String(get(p, 'input_dataset', ''))
      const from = src ? `${src.split('/').pop()} → ` : ''
      return `${from}${out} · ${mode}`
    },
  },

  // ─── Subgraph (a node that is itself a graph — opens its own subcanvas) ───
  // Compiles to a nested `class <class_name>(nn.Module)` built from its
  // subgraph. The subgraph's Input nodes become the class's forward args (in
  // declaration order; the parent wires incoming edges positionally) and its
  // Output node(s) the return. Edit it by double-clicking the node.
  Subgraph: {
    type: 'Subgraph', category: 'Custom', pytorchModule: '', kind: 'group',
    fields: [
      { name: 'class_name', type: 'text', default: 'SubModule', placeholder: 'verschachtelter Modulname' } as FieldSpec,
    ],
    summary: (p) => {
      const cls = String(get(p, 'class_name', 'SubModule')) || 'SubModule'
      const sg = get(p, 'subgraph', undefined) as { nodes?: unknown[] } | undefined
      const n = Array.isArray(sg?.nodes) ? sg!.nodes!.length : 0
      return `${cls} · ${n} Knoten`
    },
  },
}

// Starter subgraph for a fresh Group: one Input → one Output (identity). Edited
// via the subcanvas. Kept as a plain GraphSnapshot-shaped object.
export const DEFAULT_SUBGRAPH = {
  nodes: [
    { id: 'in', layerType: 'Input', params: { name: 'x', shape: [1, 128], dtype: 'float32' }, position: { x: 80, y: 80 } },
    { id: 'out', layerType: 'Output', params: { name: 'out' }, position: { x: 80, y: 240 } },
  ],
  edges: [{ source: 'in', target: 'out' }],
}

export function defaultParamsFor(layerType: string): Record<string, unknown> {
  const spec = LAYERS[layerType]
  if (!spec) return {}
  const params: Record<string, unknown> = {}
  for (const field of spec.fields) {
    params[field.name] = field.default
  }
  // Group nodes carry a structural subgraph (not a FieldSpec) — seed a starter.
  if (spec.kind === 'group') {
    params.subgraph = structuredClone(DEFAULT_SUBGRAPH)
  }
  return params
}

/** Coerce a raw param map (e.g. from the LLM) to types matching the field schema.
 *  - tuple-int with arity N: number → [n, n, …], array of wrong length is padded/truncated
 *  - shape: number → [n], array kept (validated as positive ints)
 *  - int/float: strings parsed
 *  - select: only kept if in options
 *  Unknown keys pass through unchanged so future params don't get dropped silently.
 */
export function coerceParams(
  layerType: string,
  raw: Record<string, unknown>,
): Record<string, unknown> {
  const spec = LAYERS[layerType]
  if (!spec) return raw
  const out: Record<string, unknown> = { ...raw }
  for (const field of spec.fields) {
    if (!(field.name in raw)) continue
    out[field.name] = coerceField(field, raw[field.name])
  }
  return out
}

function coerceField(field: FieldSpec, value: unknown): unknown {
  switch (field.type) {
    case 'int': {
      const n = typeof value === 'string' ? parseInt(value, 10) : Number(value)
      return Number.isFinite(n) ? Math.trunc(n) : field.default
    }
    case 'float': {
      const n = typeof value === 'string' ? parseFloat(value) : Number(value)
      return Number.isFinite(n) ? n : field.default
    }
    case 'bool':
      if (typeof value === 'boolean') return value
      if (value === 'true') return true
      if (value === 'false') return false
      return Boolean(value)
    case 'select': {
      const s = String(value)
      return field.options.includes(s) ? s : field.default
    }
    case 'tuple-int': {
      const arr = toIntArray(value)
      if (arr.length === 0) return field.default
      if (arr.length === field.arity) return arr
      if (arr.length === 1) return Array(field.arity).fill(arr[0])
      if (arr.length > field.arity) return arr.slice(0, field.arity)
      const padded = [...arr]
      while (padded.length < field.arity) padded.push(arr[arr.length - 1])
      return padded
    }
    case 'int-list': {
      const arr = toIntArray(value)
      return arr.length ? arr : field.default
    }
    case 'shape': {
      const arr = toIntArray(value)
      return arr.length ? arr : field.default
    }
    case 'dataset-ref':
      return typeof value === 'string' ? value : field.default
    case 'columns-multi':
      return Array.isArray(value) ? value.filter((v) => typeof v === 'string') : field.default
    case 'column-single':
      return typeof value === 'string' ? value : field.default
    case 'text':
    case 'code':
      return typeof value === 'string' ? value : field.default
  }
}

function toIntArray(value: unknown): number[] {
  if (typeof value === 'number' && Number.isFinite(value)) return [Math.trunc(value)]
  if (typeof value === 'string') {
    const parts = value.split(/[,\s\[\]]+/).filter(Boolean)
    return parts
      .map((p) => parseInt(p, 10))
      .filter((n) => Number.isFinite(n))
  }
  if (Array.isArray(value)) {
    const out: number[] = []
    for (const v of value) {
      const n = typeof v === 'string' ? parseInt(v, 10) : Number(v)
      if (Number.isFinite(n)) out.push(Math.trunc(n))
    }
    return out
  }
  return []
}

export const LAYER_GROUPS: { name: string; layers: string[] }[] = (() => {
  const byCategory: Record<string, string[]> = {}
  for (const spec of Object.values(LAYERS)) {
    if (!byCategory[spec.category]) byCategory[spec.category] = []
    byCategory[spec.category].push(spec.type)
  }
  const order = ['IO', 'Data', 'Conv', 'Linear', 'Recurrent', 'Graph', 'Norm', 'Activation', 'Pool', 'Regularize', 'Attention', 'Reshape', 'Merge', 'Custom']
  return order.filter((c) => byCategory[c]).map((c) => ({ name: c, layers: byCategory[c] }))
})()
