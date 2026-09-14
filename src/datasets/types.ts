export type DatasetKind =
  | 'tabular'
  | 'image_folder'
  | 'tensor'
  | 'protein'
  | 'molecule'
  | 'huggingface'
  | 'pyg'
  | 'graph_folder'
  | 'manifest'
  | 'unknown'

/** Phase 18 — stable, content-derived dataset identifier computed by the torch
 *  sidecar at inspect time and frozen into run.json. Copy/rename-stable, never
 *  a human-readable name. `mode` says what the hash pins: 'content' = file
 *  bytes (tabular/tensor/molecule/protein), 'structure' = sorted (rel,size)
 *  listing (image/graph folders), 'config+content' = descriptor + referenced
 *  table (manifest), 'reference' = only the reference file (pyg/huggingface;
 *  the remote data itself is not pinned). */
export type DatasetFingerprint = {
  alg: 'sha256'
  mode: 'content' | 'structure' | 'config+content' | 'reference'
  hash: string
  size_bytes: number
  n_files?: number
}

/** Compact stable id string: `sha256:<hex>` — what goes into logs/events. */
export function fingerprintId(fp?: DatasetFingerprint | null): string | undefined {
  return fp && fp.hash ? `${fp.alg}:${fp.hash}` : undefined
}

export type GraphField = { name: string; shape: number[]; dtype: string }
export type XPreview = { rows: number; cols: number; grid: number[][] }

export type InspectBase = {
  kind: DatasetKind
  ok: boolean
  error?: string
  missing_dep?: string
  size_bytes?: number
  fingerprint?: DatasetFingerprint
}

export type TabularInspect = InspectBase & {
  kind: 'tabular'
  ok: true
  rows: number
  cols: number
  columns: string[]
  dtypes: string[]
  head: string[][]
  /** When the source was a prepared dataset DIRECTORY: the inner table read, and a
   *  summary of the bundle (side files / per-id embeddings / prep_card). */
  table?: string
  bundle?: {
    files: string[]
    subdirs: { name: string; entries: number }[]
    prep_card?: Record<string, unknown> | null
  }
}

export type ImageFolderInspect = InspectBase & {
  kind: 'image_folder'
  ok: true
  classes: { name: string; count: number }[]
  n_classes: number
  n_images: number
  sample_size: [number, number] | null
  thumbnails: { name: string; b64: string; w: number; h: number }[]
}

export type TensorInspect = InspectBase & {
  kind: 'tensor'
  ok: true
  container?: 'npz' | 'dict' | string
  shape?: number[]
  dtype?: string
  min?: number
  max?: number
  mean?: number
  arrays?: Record<string, number[]>
  keys?: { key: string; shape: number[] | null; dtype: string }[]
  // Present when the .pt holds a PyTorch-Geometric graph (Data object).
  is_graph?: boolean
  num_nodes?: number
  num_edges?: number
  num_node_features?: number
  edge_dim?: number
  fields?: GraphField[]
  x_preview?: XPreview | null
}

export type GraphFolderInspect = InspectBase & {
  kind: 'graph_folder'
  ok: true
  n_graphs: number
  example: string
  num_node_features: number
  edge_dim: number
  num_nodes: number
  num_edges: number
  fields: GraphField[]
  x_preview?: XPreview | null
  preview?: { n_nodes: number; edges: [number, number][] }
}

export type ProteinInspect = InspectBase & {
  kind: 'protein'
  ok: true
  parser: 'biopython' | 'fallback'
  chains: number
  residues: number
  atoms: number
  chain_info?: { id: string; residues: number; atoms: number }[]
}

export type MoleculeInspect = InspectBase & {
  kind: 'molecule'
  ok: true
  parser: 'rdkit' | 'fallback'
  n_molecules: number
  head: string[]
  sample_info?: {
    smiles: string
    valid: boolean
    canonical?: string
    atoms?: number
    bonds?: number
    mw?: number
  }[]
  graph0?: { smiles: string; n_nodes: number; edges: [number, number][] }
}

export type HuggingfaceInspect = InspectBase & {
  kind: 'huggingface'
  ok: true
  name: string
  description: string
  splits: Record<string, { num_examples: number | null }>
  features: Record<string, string>
}

export type PygInspect = InspectBase & {
  kind: 'pyg'
  ok: true
  name: string
  num_graphs: number
  num_nodes: number
  num_edges: number
  num_node_features: number
  num_classes: number
}

// One bindable slot of a manifest: a fully-qualified field key ('<branch>.x',
// '<branch>.edge_index', '<branch>.batch', or 'target') with its row-0 shape.
export type ManifestSlot = { field: string; shape: number[]; dtype: string }

export type ManifestInspect = InspectBase & {
  kind: 'manifest'
  ok: true
  n_rows: number
  table: string
  /** All columns of the manifest's table — feeds the Head node's target picker. */
  columns?: string[]
  branches: string[]
  target: { column: string; type: string } | null
  slots: ManifestSlot[]
  notes?: string[]
}

export type InspectResult =
  | TabularInspect
  | ImageFolderInspect
  | TensorInspect
  | ProteinInspect
  | MoleculeInspect
  | HuggingfaceInspect
  | PygInspect
  | GraphFolderInspect
  | ManifestInspect
  | (InspectBase & { ok: false })

export type ColumnSummary = {
  col: string
  dtype: string
  missing: number
  unique: number
  mean?: number
  std?: number
  min?: number
  max?: number
  hist?: { counts: number[]; edges: number[] }
}

export type TabularStats = {
  kind: 'tabular'
  ok: true
  rows: number
  cols: number
  summary: ColumnSummary[]
  corr: number[][] | null
  corr_cols: string[]
}

export type ImageFolderStats = {
  kind: 'image_folder'
  ok: true
  classes: { name: string; count: number }[]
  size_hist: { size: string; count: number }[]
  n_samples_for_size: number
}

export type TensorStats = TensorInspect & {
  std?: number
  zeros_frac?: number
  hist?: { counts: number[]; edges: number[] }
}

export type MoleculeStats = MoleculeInspect & {
  mw_mean?: number
  atom_mean?: number
  mw_hist?: { counts: number[]; edges: number[] }
  atom_hist?: { counts: number[]; edges: number[] }
  note?: string
}

export type StatsResult =
  | TabularStats
  | ImageFolderStats
  | TensorStats
  | MoleculeStats
  | { kind: DatasetKind; ok: true; note?: string }
  | { kind: DatasetKind; ok: false; error: string }

export type SmokeOk = {
  ok: true
  input_shape: number[] | number[][]
  output_shape: number[] | number[][] | null
  n_params: number
  sample_note?: string
  timings_ms: { sample: number; forward: number }
}

export type SmokeErr = {
  ok: false
  stage: 'sample' | 'compile' | 'construct' | 'forward'
  error: string
  trace?: string
  input_shape?: number[] | number[][]
  n_params?: number
  details?: unknown
}

export type SmokeResult = SmokeOk | SmokeErr
