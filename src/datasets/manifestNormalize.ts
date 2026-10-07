// Manifest schema + normaliser. Kept out of DatasetDetail.tsx so that file
// only exports components (React Fast Refresh).

export type BranchMode = 'molecule' | 'dir' | 'path'

export type BranchCfg = {
  name: string
  column: string
  mode: BranchMode
  dir: string
  match: 'exact' | 'contains'
  ext: string
}

export type ManifestCfg = {
  table: string
  branches: BranchCfg[]
  target: { column: string; type: 'regression' | 'classification' }
  cache: boolean
}

export function normalizeManifest(raw: unknown): ManifestCfg {
  // Trust boundary: a manifest is an arbitrary JSON document read from disk
  // (or a JSON.parse of user input). Never throw — fall back to documented
  // defaults so a torn/malformed file still yields a usable editor.
  const isObj = (v: unknown): v is Record<string, unknown> =>
    typeof v === 'object' && v !== null && !Array.isArray(v)
  const root = isObj(raw) ? raw : {}
  const pairs = isObj(root.pairs) ? root.pairs : {}
  const branches: BranchCfg[] = Object.entries(pairs).map(([name, s]) => {
    const so = isObj(s) ? s : {}
    const dir = typeof so.dir === 'string' ? so.dir : ''
    return {
      name,
      column: typeof so.column === 'string' ? so.column : '',
      mode: so.kind === 'molecule' ? 'molecule' : dir ? 'dir' : 'path',
      dir,
      match: so.match === 'exact' ? 'exact' : 'contains',
      ext: typeof so.ext === 'string' ? so.ext : '.pt',
    }
  })
  const tgt = isObj(root.target) ? root.target : {}
  return {
    table: typeof root.table === 'string' ? root.table : '',
    branches: branches.length ? branches : [{ name: 'graph', column: '', mode: 'molecule', dir: '', match: 'contains', ext: '.pt' }],
    target: {
      column: typeof tgt.column === 'string' ? tgt.column : '',
      type: tgt.type === 'classification' ? 'classification' : 'regression',
    },
    cache: root.cache !== false, // default on
  }
}
