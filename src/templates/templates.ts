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

/** GCN node classifier (Cora-style). Two GCNConv layers with ReLU + dropout.
 *  One `Graph` input carries the whole graph (x + edge_index + batch); the
 *  encoder unpacks it (`x, edge_index, batch = data.x, …`). [N, 16] → 7 logits. */
export function buildGcnNodeClassifier(): GraphSnapshot {
  return {
    nodes: [
      { id: 'data', layerType: 'Graph', params: { name: 'data', shape: [10, 16], n_edges: 40 }, position: { x: 0, y: 0 } },
      { id: 'g1', layerType: 'GCNConv', params: { in_channels: -1, out_channels: 32 }, position: { x: 280, y: 0 } },
      { id: 'a1', layerType: 'ReLU', params: {}, position: { x: 520, y: 0 } },
      { id: 'd1', layerType: 'Dropout', params: { p: 0.5 }, position: { x: 740, y: 0 } },
      { id: 'g2', layerType: 'GCNConv', params: { in_channels: 32, out_channels: 7 }, position: { x: 960, y: 0 } },
      { id: 'out', layerType: 'Output', params: { name: 'logits' }, position: { x: 1200, y: 0 } },
    ],
    edges: [
      { source: 'data', target: 'g1' },
      { source: 'g1', target: 'a1' },
      { source: 'a1', target: 'd1' },
      { source: 'd1', target: 'g2' },
      { source: 'g2', target: 'out' },
    ],
  }
}

/** GAT node classifier. First layer uses 8 attention heads (concat → 8×8=64),
 *  second collapses to the class count with a single head. Fed by one `Graph`
 *  input (unpacked to x/edge_index/batch in forward). */
export function buildGatNodeClassifier(): GraphSnapshot {
  return {
    nodes: [
      { id: 'data', layerType: 'Graph', params: { name: 'data', shape: [10, 16], n_edges: 40 }, position: { x: 0, y: 0 } },
      { id: 'g1', layerType: 'GATConv', params: { in_channels: -1, out_channels: 8, heads: 8, concat: true, dropout: 0.6 }, position: { x: 280, y: 0 } },
      { id: 'a1', layerType: 'ReLU', params: {}, position: { x: 540, y: 0 } },
      { id: 'd1', layerType: 'Dropout', params: { p: 0.6 }, position: { x: 760, y: 0 } },
      { id: 'g2', layerType: 'GATConv', params: { in_channels: 64, out_channels: 7, heads: 1, concat: false, dropout: 0.6 }, position: { x: 980, y: 0 } },
      { id: 'out', layerType: 'Output', params: { name: 'logits' }, position: { x: 1240, y: 0 } },
    ],
    edges: [
      { source: 'data', target: 'g1' },
      { source: 'g1', target: 'a1' },
      { source: 'a1', target: 'd1' },
      { source: 'd1', target: 'g2' },
      { source: 'g2', target: 'out' },
    ],
  }
}

/** GCN graph classifier. Two GCNConv blocks, then a GlobalMeanPool readout
 *  collapses each graph's nodes into one vector ([N_nodes,F] → [N_graphs,F]),
 *  then a Linear head → class logits per graph. One `Graph` input provides x +
 *  edge_index + batch (`batch` says which graph each node belongs to). */
export function buildGcnGraphClassifier(): GraphSnapshot {
  return {
    nodes: [
      { id: 'data', layerType: 'Graph', params: { name: 'data', shape: [30, 16], n_edges: 80 }, position: { x: 0, y: 0 } },
      { id: 'g1', layerType: 'GCNConv', params: { in_channels: -1, out_channels: 32 }, position: { x: 280, y: 0 } },
      { id: 'a1', layerType: 'ReLU', params: {}, position: { x: 500, y: 0 } },
      { id: 'g2', layerType: 'GCNConv', params: { in_channels: 32, out_channels: 64 }, position: { x: 700, y: 0 } },
      { id: 'a2', layerType: 'ReLU', params: {}, position: { x: 920, y: 0 } },
      { id: 'pool', layerType: 'GlobalMeanPool', params: {}, position: { x: 1120, y: 0 } },
      { id: 'fc', layerType: 'Linear', params: { in_features: 64, out_features: 6 }, position: { x: 1340, y: 0 } },
      { id: 'out', layerType: 'Output', params: { name: 'logits' }, position: { x: 1560, y: 0 } },
    ],
    edges: [
      { source: 'data', target: 'g1' },
      { source: 'g1', target: 'a1' },
      { source: 'a1', target: 'g2' },
      { source: 'g2', target: 'a2' },
      { source: 'a2', target: 'pool' },
      { source: 'pool', target: 'fc' },
      { source: 'fc', target: 'out' },
    ],
  }
}

/** kNN-graph classifier on plain features (no edge_index Input). A BuildGraph
 *  node connects each row to its k nearest neighbours, producing edge_index for
 *  the GCN layers automatically — turning tabular/feature data into a graph. */
export function buildKnnGraphClassifier(): GraphSnapshot {
  return {
    nodes: [
      { id: 'x', layerType: 'Input', params: { name: 'x', shape: [100, 16], dtype: 'float32' }, position: { x: 0, y: 0 } },
      { id: 'bg', layerType: 'BuildGraph', params: { method: 'knn', k: 6, loop: false, cosine: false }, position: { x: 280, y: 0 } },
      { id: 'g1', layerType: 'GCNConv', params: { in_channels: -1, out_channels: 32 }, position: { x: 540, y: 0 } },
      { id: 'a1', layerType: 'ReLU', params: {}, position: { x: 760, y: 0 } },
      { id: 'g2', layerType: 'GCNConv', params: { in_channels: 32, out_channels: 7 }, position: { x: 960, y: 0 } },
      { id: 'out', layerType: 'Output', params: { name: 'logits' }, position: { x: 1200, y: 0 } },
    ],
    edges: [
      { source: 'x', target: 'bg' },
      { source: 'bg', target: 'g1' },
      { source: 'g1', target: 'a1' },
      { source: 'a1', target: 'g2' },
      { source: 'g2', target: 'out' },
    ],
  }
}

/** A GNN encoder as a self-contained Subgraph: ONE `Graph` input (a whole PyG
 *  Data) → GCNConv → ReLU → GCNConv → GlobalMeanPool → one graph-level embedding.
 *  The encoder's forward is exactly idiomatic PyG —
 *    def forward(self, data):
 *      x, edge_index, batch = data.x, data.edge_index, data.batch
 *      ...
 *  Each encoder owns its own graph (its own Data), which is why the dual-encoder
 *  uses two of these: the two molecules never share connectivity. */
function gnnEncoderSubgraph(inDim: number, hidden: number, outDim: number): GraphSnapshot {
  return {
    nodes: [
      { id: 'data', layerType: 'Graph', params: { name: 'data', shape: [32, inDim], n_edges: 64 }, position: { x: 40, y: 20 } },
      // in_channels=-1 → lazy: binds to ANY node-feature dim without editing.
      { id: 'g1', layerType: 'GCNConv', params: { in_channels: -1, out_channels: hidden }, position: { x: 40, y: 140 } },
      { id: 'a1', layerType: 'ReLU', params: {}, position: { x: 40, y: 260 } },
      { id: 'g2', layerType: 'GCNConv', params: { in_channels: hidden, out_channels: outDim }, position: { x: 40, y: 380 } },
      { id: 'pool', layerType: 'GlobalMeanPool', params: {}, position: { x: 40, y: 500 } },
      { id: 'out', layerType: 'Output', params: { name: 'emb' }, position: { x: 40, y: 620 } },
    ],
    edges: [
      { source: 'data', target: 'g1' },
      { source: 'g1', target: 'a1' },
      { source: 'a1', target: 'g2' },
      { source: 'g2', target: 'pool' },
      { source: 'pool', target: 'out' },
    ],
  }
}

/** Dual-encoder for paired graph inputs (ligand + protein → affinity / EC).
 *  Two `Graph` inputs (each binds 1:1 to a graph dataset / manifest branch) feed
 *  two independent GNN encoder Subgraphs; their pooled embeddings concatenate
 *  into an MLP head. The model's forward is `forward(self, ligand, protein)` —
 *  two whole graphs in, one score out. The pairing (which ligand goes with which
 *  protein) comes from the manifest dataset: bind each Graph node to its branch. */
export function buildDualEncoderGnn(): GraphSnapshot {
  const LIG_IN = 9, PROT_IN = 20, HID = 64, EMB = 128
  return {
    nodes: [
      // The Manifest node declares the pairing (.manifest) and feeds each branch
      // to its Graph input via an edge — bind it to your .manifest; the branches
      // (ligand/protein) are already wired below.
      { id: 'manifest', layerType: 'Manifest', params: { dataset: '' }, position: { x: -340, y: 270 } },
      { id: 'ligand', layerType: 'Graph', params: { name: 'ligand', shape: [32, LIG_IN], n_edges: 64, branch: 'ligand' }, position: { x: 0, y: 80 } },
      { id: 'protein', layerType: 'Graph', params: { name: 'protein', shape: [128, PROT_IN], n_edges: 256, branch: 'protein' }, position: { x: 0, y: 460 } },
      { id: 'lig_enc', layerType: 'Subgraph', params: { class_name: 'LigandEncoder', subgraph: gnnEncoderSubgraph(LIG_IN, HID, EMB) }, position: { x: 320, y: 80 } },
      { id: 'prot_enc', layerType: 'Subgraph', params: { class_name: 'ProteinEncoder', subgraph: gnnEncoderSubgraph(PROT_IN, HID, EMB) }, position: { x: 320, y: 460 } },
      { id: 'merge', layerType: 'Concat', params: { dim: -1 }, position: { x: 620, y: 270 } },
      { id: 'fc1', layerType: 'Linear', params: { in_features: EMB * 2, out_features: HID }, position: { x: 840, y: 270 } },
      { id: 'act', layerType: 'ReLU', params: {}, position: { x: 1060, y: 270 } },
      { id: 'fc2', layerType: 'Linear', params: { in_features: HID, out_features: 1 }, position: { x: 1260, y: 270 } },
      { id: 'out', layerType: 'Output', params: { name: 'affinity' }, position: { x: 1480, y: 270 } },
    ],
    edges: [
      { source: 'manifest', target: 'ligand' },
      { source: 'manifest', target: 'protein' },
      { source: 'ligand', target: 'lig_enc' },
      { source: 'protein', target: 'prot_enc' },
      { source: 'lig_enc', target: 'merge' },
      { source: 'prot_enc', target: 'merge' },
      { source: 'merge', target: 'fc1' },
      { source: 'fc1', target: 'act' },
      { source: 'act', target: 'fc2' },
      { source: 'fc2', target: 'out' },
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
    id: 'gcn-node-classifier',
    name: 'GCN node classifier',
    description: '2× GCNConv + ReLU/Dropout · Knoten-Klassifikation · ein Graph-Knoten (x+edge_index+batch) · 16 Features → 7 Klassen',
    build: buildGcnNodeClassifier,
  },
  {
    id: 'gat-node-classifier',
    name: 'GAT node classifier',
    description: 'Attention-GNN · GATConv(8 Köpfe) → GATConv · Knoten-Klassifikation · 16 Features → 7 Klassen',
    build: buildGatNodeClassifier,
  },
  {
    id: 'gcn-graph-classifier',
    name: 'GCN graph classifier',
    description: '2× GCNConv → GlobalMeanPool → Linear · ganze Graphen klassifizieren · ein Graph-Knoten (x+edge_index+batch) · → 6 Klassen',
    build: buildGcnGraphClassifier,
  },
  {
    id: 'knn-graph-classifier',
    name: 'kNN graph (features→GCN)',
    description: 'BuildGraph(kNN) baut den Graphen aus Features → 2× GCNConv · KEIN edge_index-Input nötig · 16 Features → 7 Klassen',
    build: buildKnnGraphClassifier,
  },
  {
    id: 'rankbind-bilinear',
    name: 'RankBind (bilinear)',
    description: 'score = f(L)ᵀ M g(P) + b · two projectors + low-rank bilinear head, via Custom code nodes',
    build: buildRankBindBilinear,
  },
  {
    id: 'dual-encoder-gnn',
    name: 'Dual-Encoder GNN (Ligand + Protein)',
    description: '2 GNN-Encoder (eigener Graph je Branch, als Subgraph) → Concat → MLP · gepaarte Graph-Inputs → affinity/EC · jeden Branch ans Manifest binden',
    build: buildDualEncoderGnn,
  },
]
