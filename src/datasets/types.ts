export type DatasetKind =
  | 'tabular'
  | 'image_folder'
  | 'tensor'
  | 'protein'
  | 'molecule'
  | 'huggingface'
  | 'unknown'

export type InspectBase = {
  kind: DatasetKind
  ok: boolean
  error?: string
  missing_dep?: string
  size_bytes?: number
}

export type TabularInspect = InspectBase & {
  kind: 'tabular'
  ok: true
  rows: number
  cols: number
  columns: string[]
  dtypes: string[]
  head: string[][]
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
}

export type HuggingfaceInspect = InspectBase & {
  kind: 'huggingface'
  ok: true
  name: string
  description: string
  splits: Record<string, { num_examples: number | null }>
  features: Record<string, string>
}

export type InspectResult =
  | TabularInspect
  | ImageFolderInspect
  | TensorInspect
  | ProteinInspect
  | MoleculeInspect
  | HuggingfaceInspect
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
  input_shape: number[]
  output_shape: number[] | null
  n_params: number
  sample_note?: string
  timings_ms: { sample: number; forward: number }
}

export type SmokeErr = {
  ok: false
  stage: 'sample' | 'compile' | 'construct' | 'forward'
  error: string
  trace?: string
  input_shape?: number[]
  n_params?: number
  details?: unknown
}

export type SmokeResult = SmokeOk | SmokeErr
