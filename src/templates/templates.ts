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

// ─── RankBind (bilinear main model) Custom-node sources ─────────────────────
// Verbatim from v5_rankbind/model.py — the documented main scoring path
// score(L, P) = f(L)^T M g(P) + b. Pasted into Custom nodes so the graph
// reproduces the model 1:1 (projections + low-rank+diag bilinear head).

export const LIGAND_PROJECTOR_SRC = `class LigandProjector(nn.Module):
    """ChemBERTa mean-pool (384-d) -> d_lig."""
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)`

export const PROTEIN_PROJECTOR_SRC = `class ProteinProjector(nn.Module):
    """ESM2 mean-pool (1280-d) -> d_prot."""
    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)`

export const BILINEAR_HEAD_SRC = `class BilinearHead(nn.Module):
    """score = f(L)^T M g(P) + b, with M = U V^T + diag(d) (low-rank + diagonal)."""
    def __init__(self, d_lig, d_prot, rank=32):
        super().__init__()
        if d_lig != d_prot:
            raise ValueError("BilinearHead assumes d_lig == d_prot for diag term.")
        self.U = nn.Parameter(torch.empty(d_lig, rank))
        self.V = nn.Parameter(torch.empty(d_prot, rank))
        self.d = nn.Parameter(torch.zeros(d_lig))
        self.b = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)

    def forward(self, fL, gP):
        low = (fL @ self.U) * (gP @ self.V)
        lr = low.sum(dim=-1)
        diag = (fL * self.d * gP).sum(dim=-1)
        return lr + diag + self.b`

// A projector is a pure Sequential, so it's expressible as a Subgraph built from
// standard layer nodes: LayerNorm → Linear → GELU → Dropout → Linear.
function projectorSubgraph(inDim: number, outDim: number): GraphSnapshot {
  return {
    nodes: [
      { id: 'in', layerType: 'Input', params: { name: 'x', shape: [1, inDim], dtype: 'float32' }, position: { x: 40, y: 20 } },
      { id: 'ln', layerType: 'LayerNorm', params: { normalized_shape: [inDim] }, position: { x: 40, y: 120 } },
      { id: 'fc1', layerType: 'Linear', params: { in_features: inDim, out_features: outDim }, position: { x: 40, y: 220 } },
      { id: 'act', layerType: 'GELU', params: {}, position: { x: 40, y: 320 } },
      { id: 'drop', layerType: 'Dropout', params: { p: 0.0 }, position: { x: 40, y: 420 } },
      { id: 'fc2', layerType: 'Linear', params: { in_features: outDim, out_features: outDim }, position: { x: 40, y: 520 } },
      { id: 'out', layerType: 'Output', params: { name: 'out' }, position: { x: 40, y: 620 } },
    ],
    edges: [
      { source: 'in', target: 'ln' },
      { source: 'ln', target: 'fc1' },
      { source: 'fc1', target: 'act' },
      { source: 'act', target: 'drop' },
      { source: 'drop', target: 'fc2' },
      { source: 'fc2', target: 'out' },
    ],
  }
}

/** RankBind main bilinear model. The two projectors are Subgraph nodes (built
 *  from standard layers — double-click to open their subcanvas); the bilinear
 *  head stays a Custom code node (needs nn.Parameter + bilinear math). Edge
 *  order into BilinearHead matters — ligand first (fL), protein second (gP). */
export function buildRankBindBilinear(): GraphSnapshot {
  return {
    nodes: [
      { id: 'lig', layerType: 'Input', params: { name: 'lig_emb', shape: [1, 384], dtype: 'float32' }, position: { x: 0, y: 0 } },
      { id: 'prot', layerType: 'Input', params: { name: 'prot_emb', shape: [1, 1280], dtype: 'float32' }, position: { x: 0, y: 180 } },
      { id: 'lp', layerType: 'Subgraph', params: { class_name: 'LigandProjector', subgraph: projectorSubgraph(384, 256) }, position: { x: 280, y: 0 } },
      { id: 'pp', layerType: 'Subgraph', params: { class_name: 'ProteinProjector', subgraph: projectorSubgraph(1280, 256) }, position: { x: 280, y: 180 } },
      { id: 'bh', layerType: 'Custom', params: { init_args: '256, 256, rank=32', source: BILINEAR_HEAD_SRC }, position: { x: 580, y: 90 } },
      { id: 'out', layerType: 'Output', params: { name: 'score' }, position: { x: 840, y: 90 } },
    ],
    edges: [
      { source: 'lig', target: 'lp' },
      { source: 'prot', target: 'pp' },
      { source: 'lp', target: 'bh' },
      { source: 'pp', target: 'bh' },
      { source: 'bh', target: 'out' },
    ],
  }
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
  {
    id: 'rankbind-bilinear',
    name: 'RankBind (bilinear)',
    description: 'score = f(L)ᵀ M g(P) + b · two projectors + low-rank bilinear head, via Custom code nodes',
    build: buildRankBindBilinear,
  },
]
