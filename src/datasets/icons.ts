import type { DatasetKind } from './types'

export function iconFor(kind: DatasetKind): string {
  switch (kind) {
    case 'tabular': return 'TBL'
    case 'image_folder': return 'IMG'
    case 'tensor': return 'TEN'
    case 'protein': return 'PDB'
    case 'molecule': return 'SMI'
    case 'huggingface': return 'HF'
    case 'pyg': return 'PYG'
    case 'graph_folder': return 'GPH'
    case 'manifest': return 'PAIR'
    default: return '???'
  }
}

export function colorFor(kind: DatasetKind): string {
  switch (kind) {
    case 'tabular': return 'bg-sky-900/40 text-sky-300'
    case 'image_folder': return 'bg-emerald-900/40 text-emerald-300'
    case 'tensor': return 'bg-amber-900/40 text-amber-300'
    case 'protein': return 'bg-rose-900/40 text-rose-300'
    case 'molecule': return 'bg-violet-900/40 text-violet-300'
    case 'huggingface': return 'bg-fuchsia-900/40 text-fuchsia-300'
    case 'pyg': return 'bg-teal-900/40 text-teal-300'
    case 'graph_folder': return 'bg-teal-900/40 text-teal-300'
    case 'manifest': return 'bg-indigo-900/40 text-indigo-300'
    default: return 'bg-[#1f2429] text-[#6f767e]'
  }
}

export function guessKindFromName(name: string, isDir: boolean): DatasetKind {
  if (isDir) return 'image_folder' // best guess pre-inspect
  const lower = name.toLowerCase()
  if (lower.endsWith('.csv') || lower.endsWith('.tsv') || lower.endsWith('.parquet')) return 'tabular'
  if (lower.endsWith('.pt') || lower.endsWith('.pth') || lower.endsWith('.npy') || lower.endsWith('.npz')) return 'tensor'
  if (lower.endsWith('.pdb')) return 'protein'
  if (lower.endsWith('.smi') || lower.endsWith('.smiles')) return 'molecule'
  if (lower.endsWith('.hf')) return 'huggingface'
  if (lower.endsWith('.pyg')) return 'pyg'
  if (lower.endsWith('.manifest')) return 'manifest'
  return 'unknown'
}

export function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`
}
