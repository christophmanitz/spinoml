// THE data-node registry — the data-canvas analogue of training/graph/registry.ts.
// Adding a data-prep node type = editing this one file (spec + default + coerce is
// uniform). Each node compiles to a Python block in codegen/dataCodegen.ts that
// operates on a pandas DataFrame `df` threaded through the pipeline (some nodes
// also read a column to download files or build graphs as a side effect).

export type DataFieldSpec =
  | { name: string; type: 'int'; min?: number; max?: number; step?: number; default: number }
  | { name: string; type: 'float'; min?: number; max?: number; step?: number; default: number }
  | { name: string; type: 'bool'; default: boolean }
  | { name: string; type: 'select'; options: string[]; default: string }
  /** Free single-line string (column name, path, URL template, …). */
  | { name: string; type: 'text'; default: string; placeholder?: string }
  /** Multi-line Python (the CustomScript body). Edited in a code box. */
  | { name: string; type: 'code'; default: string }
  /** Runtime-populated dropdown of dataset relpaths (datasetsStore). */
  | { name: string; type: 'dataset-ref'; default: string }

/** Drives palette grouping + node theming. */
export type DataCategory =
  | 'Source'
  | 'Fetch'
  | 'Transform'
  | 'Graph'
  | 'Custom'
  | 'Sink'

export type DataNodeSpec = {
  type: string
  category: DataCategory
  fields: DataFieldSpec[]
  summary: (params: Record<string, unknown>) => string
  /** One-line hint shown in the inspector. */
  hint?: string
}

const get = <T>(p: Record<string, unknown>, k: string, fallback: T): T =>
  (p[k] as T) ?? fallback

const f = {
  int: (name: string, def: number, opts: { min?: number; max?: number; step?: number } = {}): DataFieldSpec => ({ name, type: 'int', default: def, ...opts }),
  float: (name: string, def: number, opts: { min?: number; max?: number; step?: number } = {}): DataFieldSpec => ({ name, type: 'float', default: def, ...opts }),
  bool: (name: string, def: boolean): DataFieldSpec => ({ name, type: 'bool', default: def }),
  select: (name: string, options: string[], def: string): DataFieldSpec => ({ name, type: 'select', options, default: def }),
  text: (name: string, def = '', placeholder?: string): DataFieldSpec => ({ name, type: 'text', default: def, placeholder }),
}

const CUSTOM_TEMPLATE = [
  '# Custom step. `df` is the pandas DataFrame coming from the previous node;',
  '# reassign `df` to pass your result on (or write files / save tensors here).',
  '# torch, pandas as pd and numpy as np are available if installed.',
  'df = df  # TODO: your transformation',
].join('\n')

export const DATA_NODES: Record<string, DataNodeSpec> = {
  // ── Source ──────────────────────────────────────────────────────────────
  TableSource: {
    type: 'TableSource', category: 'Source',
    fields: [{ name: 'dataset', type: 'dataset-ref', default: '' }],
    summary: (p) => {
      const ds = String(get(p, 'dataset', ''))
      return ds ? (ds.split('/').pop() ?? ds) : '(kein Datensatz)'
    },
    hint: 'Lädt eine Tabelle (csv/tsv/parquet) aus datasets/ als DataFrame `df`.',
  },
  // ── Fetch / download ──────────────────────────────────────────────────────
  DownloadColumn: {
    type: 'DownloadColumn', category: 'Fetch',
    fields: [
      f.text('id_column', 'uniprot', 'Spalte mit IDs/URLs'),
      f.text('url_template', 'https://files.rcsb.org/download/{id}.pdb', 'URL, {id} wird ersetzt'),
      f.text('out_dir', 'datasets/raw', 'Zielordner'),
      f.text('filename_template', '{id}.pdb', 'Dateiname, {id} wird ersetzt'),
      f.bool('add_path_column', true),
    ],
    summary: (p) => `${get(p, 'id_column', 'id')} → ${get(p, 'out_dir', 'datasets/raw')}`,
    hint: 'Lädt pro Zeile eine Datei (z.B. .pdb von RCSB oder .fasta von UniProt) anhand einer ID-/URL-Spalte.',
  },
  // ── Transform (pandas) ────────────────────────────────────────────────────
  RenameColumns: {
    type: 'RenameColumns', category: 'Transform',
    fields: [f.text('mapping', 'old:new, old2:new2', 'alt:neu, kommagetrennt')],
    summary: (p) => String(get(p, 'mapping', '')) || '(leer)',
    hint: 'Spalten umbenennen: "alt:neu" Paare, kommagetrennt.',
  },
  SelectColumns: {
    type: 'SelectColumns', category: 'Transform',
    fields: [f.text('columns', '', 'spalte1, spalte2, …')],
    summary: (p) => String(get(p, 'columns', '')) || '(alle)',
    hint: 'Nur diese Spalten behalten (kommagetrennt).',
  },
  FilterRows: {
    type: 'FilterRows', category: 'Transform',
    fields: [f.text('query', '', 'pandas query, z.B. label == 1')],
    summary: (p) => String(get(p, 'query', '')) || '(kein Filter)',
    hint: 'Zeilen filtern mit einem pandas .query()-Ausdruck.',
  },
  Normalize: {
    type: 'Normalize', category: 'Transform',
    fields: [
      f.text('columns', '', 'leer = alle numerischen'),
      f.select('method', ['zscore', 'minmax'], 'zscore'),
    ],
    summary: (p) => `${get(p, 'method', 'zscore')} · ${String(get(p, 'columns', '')) || 'alle num.'}`,
    hint: 'Spalten normalisieren (z-Score oder Min-Max).',
  },
  DropNA: {
    type: 'DropNA', category: 'Transform',
    fields: [f.text('subset', '', 'leer = alle Spalten')],
    summary: (p) => String(get(p, 'subset', '')) ? `subset: ${get(p, 'subset', '')}` : 'alle Spalten',
    hint: 'Zeilen mit fehlenden Werten entfernen.',
  },
  ComputeColumn: {
    type: 'ComputeColumn', category: 'Transform',
    fields: [
      f.text('name', 'new_col', 'neue Spalte'),
      f.text('expr', '', 'pandas eval, z.B. a + b'),
    ],
    summary: (p) => `${get(p, 'name', 'new_col')} = ${String(get(p, 'expr', '')) || '…'}`,
    hint: 'Neue Spalte aus einem pandas eval()-Ausdruck berechnen.',
  },
  // ── Graph building ─────────────────────────────────────────────────────────
  SmilesToGraph: {
    type: 'SmilesToGraph', category: 'Graph',
    fields: [
      f.text('smiles_column', 'smiles', 'Spalte mit SMILES'),
      f.text('out_name', 'datasets/graphs/mol_graphs.pt', 'Ziel (.pt)'),
      // Which physicochemical ATOM features become node features x:
      //  atomic_num → [Z]; standard → +[degree, charge, #H, hybridization, aromatic, in_ring];
      //  rich → + [mass, chirality, valence].
      f.select('atom_features', ['atomic_num', 'standard', 'rich'], 'standard'),
      // Bond features become edge_attr: one-hot bond type (4) + conjugated + in_ring.
      f.bool('bond_features', true),
      f.bool('embed_3d', true),
      f.bool('add_hydrogens', true),
    ],
    summary: (p) => `${get(p, 'smiles_column', 'smiles')} → ${(String(get(p, 'out_name', ''))).split('/').pop()} · ${get(p, 'atom_features', 'standard')}${get(p, 'bond_features', true) ? '+bonds' : ''}`,
    hint: 'SMILES → RDKit-Molekül → (3D) → PyG-Graph mit wählbaren physikochemischen Atom-Features (x) und optionalen Bindungs-Features (edge_attr); speichert eine .pt-Liste.',
  },
  StructureToGraph: {
    type: 'StructureToGraph', category: 'Graph',
    fields: [
      // Default matches DownloadColumn's output column ('file_path') so the
      // canonical Fetch → Graph chain (download .pdb → build contact graphs) connects.
      f.text('path_column', 'file_path', 'Spalte mit .pdb-Pfaden'),
      f.text('out_name', 'datasets/graphs/protein_graphs.pt', 'Ziel (.pt)'),
      f.float('contact_threshold', 8.0, { min: 1, step: 0.5 }),
    ],
    summary: (p) => `${get(p, 'path_column', 'file_path')} → ${(String(get(p, 'out_name', ''))).split('/').pop()}`,
    hint: '.pdb-Strukturen → Cα-Kontaktgraph (PyG); speichert eine .pt-Liste.',
  },
  // ── Custom escape hatch ─────────────────────────────────────────────────────
  CustomScript: {
    type: 'CustomScript', category: 'Custom',
    fields: [
      f.text('label', 'custom step', 'kurzer Name'),
      { name: 'code', type: 'code', default: CUSTOM_TEMPLATE },
    ],
    summary: (p) => String(get(p, 'label', 'custom step')) || 'custom step',
    hint: 'Freier Python-Code (du oder der Chatbot füllen ihn). `df` ist verfügbar.',
  },
  // ── Sink ─────────────────────────────────────────────────────────────────
  WriteDataset: {
    type: 'WriteDataset', category: 'Sink',
    fields: [
      f.text('out_path', 'datasets/processed.csv', 'Ziel unter datasets/'),
      f.select('format', ['csv', 'parquet', 'pt'], 'csv'),
    ],
    summary: (p) => String(get(p, 'out_path', 'datasets/processed.csv')),
    hint: 'Schreibt das Ergebnis nach datasets/ — erscheint live im Datasets-Tab und speist das Training.',
  },
}

export function defaultDataParams(nodeType: string): Record<string, unknown> {
  const spec = DATA_NODES[nodeType]
  if (!spec) return {}
  const params: Record<string, unknown> = {}
  for (const field of spec.fields) params[field.name] = field.default
  return params
}

function coerceField(field: DataFieldSpec, value: unknown): unknown {
  switch (field.type) {
    case 'int': {
      const n = Math.trunc(Number(value))
      return Number.isFinite(n) ? n : field.default
    }
    case 'float': {
      const n = Number(value)
      return Number.isFinite(n) ? n : field.default
    }
    case 'bool':
      return Boolean(value)
    case 'select':
      return field.options.includes(String(value)) ? String(value) : field.default
    case 'text':
    case 'code':
    case 'dataset-ref':
      return typeof value === 'string' ? value : field.default
  }
}

export function coerceDataParams(
  nodeType: string,
  raw: Record<string, unknown>,
): Record<string, unknown> {
  const spec = DATA_NODES[nodeType]
  if (!spec) return raw
  const out: Record<string, unknown> = { ...raw }
  for (const field of spec.fields) {
    if (!(field.name in raw)) continue
    out[field.name] = coerceField(field, raw[field.name])
  }
  return out
}

export const DATA_GROUPS: { name: DataCategory; nodes: string[] }[] = (() => {
  const byCat: Record<string, string[]> = {}
  for (const spec of Object.values(DATA_NODES)) {
    ;(byCat[spec.category] ??= []).push(spec.type)
  }
  const order: DataCategory[] = ['Source', 'Fetch', 'Transform', 'Graph', 'Custom', 'Sink']
  return order.filter((c) => byCat[c]).map((c) => ({ name: c, nodes: byCat[c] }))
})()
