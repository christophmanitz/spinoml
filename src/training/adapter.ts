// External-dataset → model adapter. Pure functions: derive the model's trained
// input contract from a finished run, then auto-suggest how an external dataset's
// columns/branches fill each role. The suggestion is editable in EvalRunModal
// (= hybrid); the eval loader consumes the chosen feature/target columns directly
// (rename/select/reorder needs no materialization). Complex transforms escalate to
// the Data canvas. See docs/FEATURES.md.

import type { RunConfig, AdapterSpec } from './types'
import type { InspectResult } from '../datasets/types'

export type ContractRole = {
  /** model role id: a trained feature/target column name, or a manifest branch. */
  key: string
  kind: 'feature' | 'target' | 'smiles' | 'branch'
  /** position among features — the eval loader reads features in THIS order. */
  order?: number
  /** dtype hint from the source dataset (for numeric-feature matching). */
  dtype?: string
  /** for the target role: classification | binary | regression (display). */
  task?: string
}

export type ModelContract = {
  datasetKind: 'tabular' | 'manifest'
  /** classification | binary | regression, from the source run's loss/heads. */
  task?: string
  roles: ContractRole[]
}

/** A concrete suggestion: the eval run's external feature/target columns plus the
 *  display/provenance AdapterSpec. */
export type Adaptation = {
  feature_columns: string[] | null
  target_column: string
  spec: AdapterSpec
}

function lossToTask(loss?: string): string | undefined {
  if (!loss) return undefined
  if (loss === 'MSELoss' || loss === 'L1Loss') return 'regression'
  if (loss === 'BCEWithLogitsLoss') return 'binary'
  if (loss === 'CrossEntropyLoss') return 'classification'
  return undefined
}

/** normalized column name for fuzzy matching: lowercase, strip non-alphanumerics. */
function norm(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]/g, '')
}

const SMILES_CHARS = new Set('CcNnOoSsPpFIBrClHcnos()[]=#@+-./\\1234567890'.split(''))
/** Mirror of dataset_handlers._looks_like_smiles for the auto-suggestion. */
export function looksLikeSmiles(token: string): boolean {
  const t = (token ?? '').trim()
  if (!t || t.includes(' ') || t.length < 2) return false
  let hits = 0
  for (const c of t) if (SMILES_CHARS.has(c)) hits++
  return hits / t.length > 0.85
}

/** Derive the model's trained input contract from the source run + (optionally) an
 *  inspect of the source dataset (used to recover feature columns when the run
 *  didn't pin them). */
export function modelContract(sourceCfg: RunConfig, sourceInspect: InspectResult | null): ModelContract {
  const ds = sourceCfg.dataset
  const task = lossToTask(sourceCfg.training?.heads?.[0]?.loss ?? sourceCfg.training?.loss?.kind)

  if (ds.kind === 'manifest') {
    const branches = sourceInspect && sourceInspect.ok && sourceInspect.kind === 'manifest'
      ? sourceInspect.branches : []
    const target = (sourceInspect && sourceInspect.ok && sourceInspect.kind === 'manifest'
      ? sourceInspect.target?.column : undefined) ?? ds.target_column
    const roles: ContractRole[] = branches.map((b) => ({ key: b, kind: 'branch' as const }))
    if (target) roles.push({ key: target, kind: 'target', task })
    return { datasetKind: 'manifest', task, roles }
  }

  // tabular: features in trained order; recover from inspect if the run didn't pin them.
  let features = ds.feature_columns
  if ((!features || features.length === 0) && sourceInspect && sourceInspect.ok && sourceInspect.kind === 'tabular') {
    features = sourceInspect.columns.filter((c, i) =>
      c !== ds.target_column && /int|float/.test(sourceInspect.dtypes[i] ?? ''))
  }
  const dtypeOf = (c: string): string | undefined => {
    if (sourceInspect && sourceInspect.ok && sourceInspect.kind === 'tabular') {
      const i = sourceInspect.columns.indexOf(c)
      return i >= 0 ? sourceInspect.dtypes[i] : undefined
    }
    return undefined
  }
  const roles: ContractRole[] = (features ?? []).map((c, i) => ({
    key: c, kind: 'feature' as const, order: i, dtype: dtypeOf(c),
  }))
  if (ds.target_column) roles.push({ key: ds.target_column, kind: 'target', task })
  return { datasetKind: 'tabular', task, roles }
}

/** Auto-suggest which external columns/branches fill each model role. Matching
 *  ladder: exact name → normalized/fuzzy name → dtype (numeric features) / SMILES
 *  detection. Unmatched roles are flagged for manual mapping. */
export function suggestAdapter(contract: ModelContract, external: InspectResult): Adaptation {
  const column_map: Record<string, string> = {}
  const branch_map: Record<string, string> = {}
  const unmatched: string[] = []

  // External candidates by kind.
  const extCols = external.ok && external.kind === 'tabular' ? external.columns : []
  const extDtypes = external.ok && external.kind === 'tabular' ? external.dtypes : []
  const extBranches = external.ok && external.kind === 'manifest' ? external.branches : []
  // First sample row per column → SMILES sniffing.
  const sample: Record<string, string> = {}
  if (external.ok && external.kind === 'tabular' && external.head?.length) {
    external.columns.forEach((c, i) => { sample[c] = external.head[0]?.[i] ?? '' })
  }
  const used = new Set<string>()

  const pickByName = (key: string, pool: string[]): string | undefined => {
    const exact = pool.find((c) => !used.has(c) && c === key)
    if (exact) return exact
    const nk = norm(key)
    const fuzzy = pool.find((c) => !used.has(c) && norm(c) === nk)
    return fuzzy
  }

  for (const role of contract.roles) {
    let chosen: string | undefined
    if (role.kind === 'branch') {
      chosen = pickByName(role.key, extBranches.filter((b) => !used.has(b)))
      if (chosen) { used.add(chosen); branch_map[role.key] = chosen; column_map[role.key] = chosen }
      else unmatched.push(role.key)
      continue
    }
    // tabular roles
    chosen = pickByName(role.key, extCols)
    if (!chosen && role.kind === 'feature') {
      // dtype fallback: first unused numeric column
      const idx = extCols.findIndex((c, i) => !used.has(c) && /int|float/.test(extDtypes[i] ?? ''))
      if (idx >= 0) chosen = extCols[idx]
    }
    if (!chosen && role.kind === 'smiles') {
      chosen = extCols.find((c) => !used.has(c) && looksLikeSmiles(sample[c] ?? ''))
    }
    if (!chosen && role.kind === 'target') {
      // common target names, then the last unused column
      chosen = extCols.find((c) => !used.has(c) && /^(y|label|target|class|activity|affinity|value)$/i.test(c))
        ?? [...extCols].reverse().find((c) => !used.has(c))
    }
    if (chosen) { used.add(chosen); column_map[role.key] = chosen }
    else unmatched.push(role.key)
  }

  const feature_columns = contract.roles
    .filter((r) => r.kind === 'feature')
    .sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
    .map((r) => column_map[r.key])
    .filter((c): c is string => !!c)
  const targetRole = contract.roles.find((r) => r.kind === 'target')
  const target_column = targetRole ? (column_map[targetRole.key] ?? '') : ''

  return {
    feature_columns: feature_columns.length ? feature_columns : null,
    target_column,
    spec: {
      column_map,
      ...(Object.keys(branch_map).length ? { branch_map } : {}),
      unmatched,
      mode: 'auto',
    },
  }
}

/** Build a starter ADAPTED MANIFEST for a manifest dual-encoder validated on a
 *  prepared dataset DIRECTORY. Reuses the source manifest's branch kinds/codebooks
 *  (so encoders match the trained model) and best-effort points each branch at a
 *  directory resource: drug/SMILES branches → the SMILES column; protein-sequence
 *  branches → a `lookup` join on the key column into a side CSV; file/graph branches
 *  → a per-id sub-dir (`dir`); ligand 3D graphs can be built inline from SMILES
 *  (`kind:"molecule"`). The result is EDITABLE in the modal — heuristics are a
 *  starting point, not a guarantee. `dirRel` is the directory's workspace relpath;
 *  the manifest is written next to datasets/, so `table` is `<dirName>/<table>`. */
/** Directory resources discovered via the filesystem (not the torch sidecar, so it
 *  works on a remote workspace without redeploying): the inner table + its columns,
 *  the side files (CSVs), and the sub-folders (precomputed per-id .pt). */
export type DirResources = {
  table: string
  columns: string[]
  files: string[]
  subdirs: string[]
  sample?: Record<string, string>
}

export function buildAdaptedManifest(
  sourceManifest: Record<string, unknown>,
  dirRel: string,
  res: DirResources,
): Record<string, unknown> {
  const dirName = dirRel.replace(/\/+$/, '').split('/').pop() ?? dirRel
  const table = res.table || 'pairs.csv'
  const cols = res.columns ?? []
  const bundleFiles = res.files ?? []
  const bundleDirs = res.subdirs ?? []
  const sample: Record<string, string> = res.sample ?? {}

  // Prefer NAME matching over value-sniffing — short id values (uniprot 'P12345')
  // can false-positive as SMILES.
  const smilesCol = cols.find((c) => /smiles|ligand|drug|compound|mol/i.test(c))
    ?? cols.find((c) => (sample[c] ?? '').length > 6 && looksLikeSmiles(sample[c] ?? '')) ?? ''
  const keyCol = cols.find((c) => /uniprot|target_id|protein_id|^id$|accession/i.test(c)) ?? cols.find((c) => c !== smilesCol) ?? ''
  const seqCsv = bundleFiles.find((f) => /seq/i.test(f)) ?? ''
  const embedDir = bundleDirs.find((d) => /emb|esm|graph|3d|struct|feat/i.test(d)) ?? bundleDirs[0] ?? ''

  const srcPairs = (sourceManifest.pairs ?? {}) as Record<string, Record<string, unknown>>
  const pairs: Record<string, Record<string, unknown>> = {}
  for (const [branch, spec] of Object.entries(srcPairs)) {
    const kind = String(spec.kind ?? '')
    const codebook = String(spec.codebook ?? '')
    const isProtein = codebook === 'protein' || /prot/i.test(branch)
    if (kind === 'espf' || kind === 'sequence') {
      if (isProtein) {
        // protein sequence lives in a side CSV → JOIN by the key column.
        pairs[branch] = {
          column: keyCol, lookup: `${dirName}/${seqCsv || 'sequences.csv'}`,
          lookup_key: keyCol, lookup_value: 'sequence',
          kind, ...(codebook ? { codebook } : {}), ...(spec.max_len ? { max_len: spec.max_len } : {}),
        }
      } else {
        pairs[branch] = { column: smilesCol, kind, ...(codebook ? { codebook } : {}), ...(spec.max_len ? { max_len: spec.max_len } : {}) }
      }
    } else if (kind === 'molecule') {
      pairs[branch] = { column: smilesCol, kind: 'molecule' }
    } else if (spec.dir) {
      // precomputed per-id graph/embedding files → a sub-dir matched by the key.
      const isLigand = /lig|drug|mol|compound/i.test(branch)
      if (isLigand && smilesCol) {
        // no ligand graphs in the benchmark → build them inline from SMILES.
        pairs[branch] = { column: smilesCol, kind: 'molecule' }
      } else {
        pairs[branch] = {
          column: keyCol, dir: `${dirName}/${embedDir || 'embeddings'}`,
          match: String(spec.match ?? 'exact'), ext: String(spec.ext ?? '.pt'),
        }
      }
    } else {
      pairs[branch] = { ...spec, column: smilesCol || keyCol }
    }
  }

  const tgt = (sourceManifest.target ?? {}) as Record<string, unknown>
  const labelCol = cols.find((c) => /^(label|y|class|active|binder)$/i.test(c)) ?? cols[cols.length - 1] ?? 'label'
  return {
    table: `${dirName}/${table}`,
    cache: false,
    pairs,
    target: { column: labelCol, type: String(tgt.type ?? 'classification') },
  }
}

/** One branch's mapping in the structured editor — how its source resolves to data. */
export type BranchCfg = {
  source: 'column' | 'molecule' | 'folder' | 'lookup'
  column: string        // the table column (data for column/molecule, key for folder/lookup)
  kind: string          // 'espf' | 'sequence' (column/lookup)
  codebook: string      // 'drug' | 'protein'
  max_len?: number
  dir: string           // folder of per-id .pt (source='folder')
  match: string         // 'exact' | 'contains'
  ext: string           // '.pt'
  lookupFile: string    // side CSV (source='lookup')
  lookupValue: string   // value column in the side CSV
}

/** Seed structured per-branch configs from the source manifest + a best-effort
 *  adapted manifest (so the editor opens pre-filled). */
export function seedBranchCfgs(
  sourceManifest: Record<string, unknown>,
  dirRel: string,
  res: DirResources,
): { branches: Record<string, BranchCfg>; targetColumn: string; table: string } {
  const adapted = buildAdaptedManifest(sourceManifest, dirRel, res) as {
    table: string; pairs: Record<string, Record<string, unknown>>; target: { column?: string }
  }
  const branches: Record<string, BranchCfg> = {}
  for (const [name, p] of Object.entries(adapted.pairs)) {
    const source: BranchCfg['source'] = p.lookup ? 'lookup' : p.dir ? 'folder' : p.kind === 'molecule' ? 'molecule' : 'column'
    branches[name] = {
      source,
      column: String(p.column ?? ''),
      kind: String(p.kind ?? 'espf'),
      codebook: String(p.codebook ?? (/prot/i.test(name) ? 'protein' : 'drug')),
      max_len: typeof p.max_len === 'number' ? p.max_len : undefined,
      dir: String(p.dir ?? ''),
      match: String(p.match ?? 'exact'),
      ext: String(p.ext ?? '.pt'),
      lookupFile: String(p.lookup ?? ''),
      lookupValue: String(p.lookup_value ?? 'sequence'),
    }
  }
  return { branches, targetColumn: adapted.target?.column ?? '', table: adapted.table }
}

/** Compile structured per-branch configs back into a manifest object. */
export function branchCfgsToManifest(
  table: string,
  branches: Record<string, BranchCfg>,
  targetColumn: string,
  targetType: string,
): Record<string, unknown> {
  const pairs: Record<string, Record<string, unknown>> = {}
  for (const [name, b] of Object.entries(branches)) {
    if (b.source === 'molecule') {
      pairs[name] = { column: b.column, kind: 'molecule' }
    } else if (b.source === 'folder') {
      pairs[name] = { column: b.column, dir: b.dir, match: b.match || 'exact', ext: b.ext || '.pt' }
    } else if (b.source === 'lookup') {
      pairs[name] = {
        column: b.column, lookup: b.lookupFile, lookup_key: b.column, lookup_value: b.lookupValue || 'sequence',
        kind: b.kind || 'espf', codebook: b.codebook, ...(b.max_len ? { max_len: b.max_len } : {}),
      }
    } else {
      pairs[name] = { column: b.column, kind: b.kind || 'espf', codebook: b.codebook, ...(b.max_len ? { max_len: b.max_len } : {}) }
    }
  }
  return { table, cache: false, pairs, target: { column: targetColumn, type: targetType } }
}

/** Recompute feature_columns/target_column from a (possibly user-edited) column_map
 *  — called when the modal commits a manual/hybrid mapping. */
export function adaptationFromMap(
  contract: ModelContract,
  column_map: Record<string, string>,
  mode: AdapterSpec['mode'],
): Adaptation {
  const branch_map: Record<string, string> = {}
  for (const r of contract.roles) if (r.kind === 'branch' && column_map[r.key]) branch_map[r.key] = column_map[r.key]
  const unmatched = contract.roles.filter((r) => !column_map[r.key]).map((r) => r.key)
  const feature_columns = contract.roles
    .filter((r) => r.kind === 'feature')
    .sort((a, b) => (a.order ?? 0) - (b.order ?? 0))
    .map((r) => column_map[r.key])
    .filter((c): c is string => !!c)
  const targetRole = contract.roles.find((r) => r.kind === 'target')
  return {
    feature_columns: feature_columns.length ? feature_columns : null,
    target_column: targetRole ? (column_map[targetRole.key] ?? '') : '',
    spec: { column_map, ...(Object.keys(branch_map).length ? { branch_map } : {}), unmatched, mode },
  }
}
