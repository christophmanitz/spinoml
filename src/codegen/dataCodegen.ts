// Data-graph → ONE reproducible Python pipeline script. PURE (same plan → same
// code, no I/O), like trainingCodegen.ts. Threads a pandas DataFrame `df` through
// the topologically-ordered nodes; download / graph-building / sink nodes use it
// for side effects (fetch files, save .pt, write datasets/). Heavy deps (rdkit,
// torch_geometric, biopython) are lazy-imported inside their block so a pipeline
// that doesn't need them stays light. This is exactly the script the "Pipeline
// ausführen" button hands to the chatbot to write to agent/ and run.

import type { DataPlan, DataPlanNode } from './dataGenerator'
import { pyStr, pyComment, pyList } from './pyLiteral'

function splitCsv(s: unknown): string[] {
  return String(s ?? '').split(',').map((x) => x.trim()).filter(Boolean)
}

const S = (p: Record<string, unknown>, k: string, d = ''): string => String(p[k] ?? d)
const B = (p: Record<string, unknown>, k: string): boolean => Boolean(p[k])
const N = (p: Record<string, unknown>, k: string, d: number): number => {
  const n = Number(p[k]); return Number.isFinite(n) ? n : d
}

function readerFor(relpath: string): string {
  const lower = relpath.toLowerCase()
  if (lower.endsWith('.parquet') || lower.endsWith('.pq')) return `pd.read_parquet(${pyStr(relpath)})`
  if (lower.endsWith('.tsv')) return `pd.read_csv(${pyStr(relpath)}, sep='\\t')`
  return `pd.read_csv(${pyStr(relpath)})`
}

function emitNode(n: DataPlanNode): string[] {
  const p = n.params
  const head = `# ${pyComment(`── ${n.dataType} (${n.id}) ──`)}`
  switch (n.dataType) {
    case 'TableSource': {
      const ds = S(p, 'dataset')
      if (!ds) return [head, `# ${pyComment('(kein Datensatz gewählt)')}`]
      return [head, `df = ${readerFor(ds)}`, `print('loaded', df.shape, 'from', ${pyStr(ds)})`]
    }
    case 'DownloadColumn': {
      const col = S(p, 'id_column', 'id')
      const tpl = S(p, 'url_template')
      const dir = S(p, 'out_dir', 'datasets/raw')
      const fn = S(p, 'filename_template', '{id}')
      const lines = [
        head, 'import os, urllib.request',
        `os.makedirs(${pyStr(dir)}, exist_ok=True)`,
        '_paths = {}',
        `for _id in df[${pyStr(col)}].dropna().astype(str).unique():`,
        `    _url = ${pyStr(tpl)}.replace('{id}', _id)`,
        `    _dest = os.path.join(${pyStr(dir)}, ${pyStr(fn)}.replace('{id}', _id))`,
        '    if not os.path.exists(_dest):',
        '        try:',
        '            urllib.request.urlretrieve(_url, _dest); print("downloaded", _dest)',
        '        except Exception as _e:',
        '            print("FAILED", _id, _e)',
        '    _paths[_id] = _dest',
      ]
      if (B(p, 'add_path_column')) {
        lines.push(`df['file_path'] = df[${pyStr(col)}].astype(str).map(_paths)`)
      }
      return lines
    }
    case 'RenameColumns': {
      const pairs = splitCsv(p.mapping).map((kv) => {
        const i = kv.indexOf(':')
        return i > 0 ? [kv.slice(0, i).trim(), kv.slice(i + 1).trim()] : null
      }).filter(Boolean) as string[][]
      if (!pairs.length) return [head, `# ${pyComment('(kein Mapping)')}`]
      const dict = pairs.map(([a, b]) => `${pyStr(a)}: ${pyStr(b)}`).join(', ')
      return [head, `df = df.rename(columns={${dict}})`]
    }
    case 'SelectColumns': {
      const cols = splitCsv(p.columns)
      if (!cols.length) return [head, `# ${pyComment('(alle Spalten behalten)')}`]
      return [head, `df = df[${pyList(cols)}]`]
    }
    case 'FilterRows': {
      const q = S(p, 'query')
      if (!q) return [head, `# ${pyComment('(kein Filter)')}`]
      return [head, `df = df.query(${pyStr(q)})`, `print('after filter', df.shape)`]
    }
    case 'Normalize': {
      const cols = splitCsv(p.columns)
      const method = S(p, 'method', 'zscore')
      const lines = [head]
      lines.push(cols.length
        ? `_cols = ${pyList(cols)}`
        : `_cols = df.select_dtypes(include='number').columns.tolist()`)
      lines.push('for _c in _cols:')
      if (method === 'minmax') {
        lines.push('    df[_c] = (df[_c] - df[_c].min()) / (df[_c].max() - df[_c].min() + 1e-8)')
      } else {
        lines.push('    df[_c] = (df[_c] - df[_c].mean()) / (df[_c].std() + 1e-8)')
      }
      return lines
    }
    case 'DropNA': {
      const subset = splitCsv(p.subset)
      return [head, subset.length ? `df = df.dropna(subset=${pyList(subset)})` : 'df = df.dropna()', `print('after dropna', df.shape)`]
    }
    case 'ComputeColumn': {
      const name = S(p, 'name', 'new_col')
      const expr = S(p, 'expr')
      if (!expr) return [head, '# (kein Ausdruck)']
      return [head, `df[${pyStr(name)}] = df.eval(${pyStr(expr)})`]
    }
    case 'SmilesToGraph': {
      const col = S(p, 'smiles_column', 'smiles')
      const out = S(p, 'out_name', 'datasets/graphs/mol_graphs.pt')
      const preset = S(p, 'atom_features', 'standard')
      const bonds = B(p, 'bond_features')
      const lines = [
        head, 'from rdkit import Chem', 'from rdkit.Chem import AllChem',
        'import torch, os', 'from torch_geometric.data import Data',
        '_HYB = {Chem.HybridizationType.SP: 1, Chem.HybridizationType.SP2: 2, Chem.HybridizationType.SP3: 3, Chem.HybridizationType.SP3D: 4, Chem.HybridizationType.SP3D2: 5}',
        '# Physicochemical ATOM features → node feature matrix x.',
        'def _atom_features(a):',
        '    f = [float(a.GetAtomicNum())]',
      ]
      if (preset === 'standard' || preset === 'rich') {
        lines.push('    f += [float(a.GetDegree()), float(a.GetFormalCharge()), float(a.GetTotalNumHs()), float(_HYB.get(a.GetHybridization(), 0)), float(a.GetIsAromatic()), float(a.IsInRing())]')
      }
      if (preset === 'rich') {
        lines.push('    f += [a.GetMass() * 0.01, float(int(a.GetChiralTag())), float(a.GetExplicitValence())]')
      }
      lines.push('    return f')
      if (bonds) {
        lines.push(
          '_BT = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}',
          '# BOND features → edge_attr: one-hot bond type (4) + conjugated + in_ring.',
          'def _bond_features(b):',
          '    oh = [0.0, 0.0, 0.0, 0.0]',
          '    oh[_BT.get(b.GetBondType(), 0)] = 1.0',
          '    return oh + [float(b.GetIsConjugated()), float(b.IsInRing())]',
        )
      }
      lines.push(
        '_graphs = []',
        `for _smi in df[${pyStr(col)}].dropna().astype(str):`,
        '    _m = Chem.MolFromSmiles(_smi)',
        '    if _m is None:',
        '        continue',
      )
      if (B(p, 'add_hydrogens')) lines.push('    _m = Chem.AddHs(_m)')
      if (B(p, 'embed_3d')) {
        lines.push(
          '    if AllChem.EmbedMolecule(_m, randomSeed=0) != 0:',
          '        continue',
          '    try:',
          '        AllChem.MMFFOptimizeMolecule(_m)',
          '    except Exception:',
          '        pass',
        )
      }
      lines.push(
        '    _x = torch.tensor([_atom_features(a) for a in _m.GetAtoms()], dtype=torch.float)',
        '    _ei = []',
        ...(bonds ? ['    _ea = []'] : []),
        '    for _b in _m.GetBonds():',
        '        _i, _j = _b.GetBeginAtomIdx(), _b.GetEndAtomIdx()',
        '        _ei += [[_i, _j], [_j, _i]]',
        ...(bonds ? ['        _bf = _bond_features(_b)', '        _ea += [_bf, _bf]'] : []),
        '    _edge_index = torch.tensor(_ei, dtype=torch.long).t().contiguous() if _ei else torch.empty((2, 0), dtype=torch.long)',
        '    _data = Data(x=_x, edge_index=_edge_index)',
        ...(bonds ? ['    _data.edge_attr = torch.tensor(_ea, dtype=torch.float) if _ea else torch.empty((0, 6), dtype=torch.float)'] : []),
        '    if _m.GetNumConformers() > 0:',
        '        _conf = _m.GetConformer()',
        '        _data.pos = torch.tensor([list(_conf.GetAtomPosition(i)) for i in range(_m.GetNumAtoms())], dtype=torch.float)',
        '    _graphs.append(_data)',
        `os.makedirs(os.path.dirname(${pyStr(out)}) or '.', exist_ok=True)`,
        `torch.save(_graphs, ${pyStr(out)})`,
        `print('built', len(_graphs), 'molecule graphs ->', ${pyStr(out)})`,
      )
      return lines
    }
    case 'StructureToGraph': {
      const col = S(p, 'path_column', 'pdb_path')
      const out = S(p, 'out_name', 'datasets/graphs/protein_graphs.pt')
      const thr = N(p, 'contact_threshold', 8.0)
      return [
        head, 'from Bio.PDB import PDBParser', 'import torch, os', 'from torch_geometric.data import Data',
        '_parser = PDBParser(QUIET=True)', '_graphs = []',
        `for _p in df[${pyStr(col)}].dropna().astype(str):`,
        '    try:',
        '        _s = _parser.get_structure("s", _p)',
        '    except Exception as _e:',
        '        print("skip", _p, _e); continue',
        '    _ca = [a for a in _s.get_atoms() if a.get_id() == "CA"]',
        '    if not _ca:',
        '        continue',
        '    _coords = torch.tensor([list(a.get_coord()) for a in _ca], dtype=torch.float)',
        '    _d = torch.cdist(_coords, _coords)',
        `    _mask = (_d < ${thr}) & (_d > 0)`,
        '    _edge_index = _mask.nonzero(as_tuple=False).t().contiguous()',
        '    _graphs.append(Data(x=torch.ones((_coords.shape[0], 1)), pos=_coords, edge_index=_edge_index))',
        `os.makedirs(os.path.dirname(${pyStr(out)}) or '.', exist_ok=True)`,
        `torch.save(_graphs, ${pyStr(out)})`,
        `print('built', len(_graphs), 'protein graphs ->', ${pyStr(out)})`,
      ]
    }
    case 'CustomScript': {
      const label = S(p, 'label', 'custom step')
      const code = S(p, 'code')
      return [`# ${pyComment(`── CustomScript (${n.id}): ${label} ──`)}`, ...code.split('\n')]
    }
    case 'WriteDataset': {
      const out = S(p, 'out_path', 'datasets/processed.csv')
      const fmt = S(p, 'format', 'csv')
      const lines = [head, 'import os', `os.makedirs(os.path.dirname(${pyStr(out)}) or '.', exist_ok=True)`]
      if (fmt === 'parquet') lines.push(`df.to_parquet(${pyStr(out)})`)
      else if (fmt === 'pt') lines.push('import torch', `torch.save(df, ${pyStr(out)})`)
      else lines.push(`df.to_csv(${pyStr(out)}, index=False)`)
      lines.push(`print('wrote', ${pyStr(out)})`)
      return lines
    }
    default:
      return [head, `# ${pyComment(`(unbekannter Knotentyp ${n.dataType})`)}`]
  }
}

export function generateDataCode(plan: DataPlan | null): string {
  if (!plan || plan.order.length === 0) {
    return [
      '# Daten-Pipeline noch nicht startklar.',
      '# Zieh Knoten auf den Canvas: TableSource → Transform/Fetch/Graph → WriteDataset,',
      '# verdrahte sie zu einer Kette, dann erscheint hier das Pipeline-Skript.',
    ].join('\n')
  }

  const L: string[] = []
  L.push('# Auto-generiert aus dem Daten-Canvas (SpinoML).')
  L.push('# Reproduzierbare Daten-Pipeline — läuft im Workspace-Root.')
  L.push('# Schwere Schritte (Graph-Bau, große Downloads) ggf. via SLURM ausführen.')
  L.push('')
  L.push('import pandas as pd')
  L.push('')
  L.push('df = None  # wird vom Quell-Knoten gesetzt')
  for (const n of plan.order) {
    L.push('')
    for (const line of emitNode(n)) L.push(line)
  }
  L.push('')
  L.push("print('pipeline done')")
  L.push('')
  return L.join('\n')
}

/** The Python block for ONE node — used to "convert to CustomScript" so a typed
 *  node's generated code can be edited by hand (or by the chatbot). The per-node
 *  header comment is dropped (the CustomScript carries its own label). */
export function generateNodeCode(dataType: string, params: Record<string, unknown>): string {
  return emitNode({ id: 'node', dataType, params }).filter((l) => !l.startsWith('# ── ')).join('\n')
}
