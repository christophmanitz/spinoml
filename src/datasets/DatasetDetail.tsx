import { useEffect, useState } from 'react'
import { useDatasetsStore, type SmokeHistoryEntry } from './store'
import { useGraphStore } from '../canvas/GraphStore'
import { iconFor, colorFor, formatSize } from './icons'
import { NodeLinkGraph } from '../visualization/primitives'
import type {
  InspectResult, StatsResult, SmokeResult,
  TabularInspect, ImageFolderInspect, TensorInspect, ProteinInspect, MoleculeInspect, HuggingfaceInspect, PygInspect, GraphFolderInspect, ManifestInspect, GraphField,
  TabularStats, ImageFolderStats, TensorStats, MoleculeStats,
} from './types'

type Tab = 'overview' | 'stats' | 'smoke'

export default function DatasetDetail({ relpath }: { relpath: string }) {
  const [tab, setTab] = useState<Tab>('overview')
  const entry = useDatasetsStore((s) => s.entries.find((e) => e.relpath === relpath))
  const inspect = useDatasetsStore((s) => s.inspects[relpath])
  const stats = useDatasetsStore((s) => s.stats[relpath])
  const smoke = useDatasetsStore((s) => s.smoke[relpath])
  const loadStats = useDatasetsStore((s) => s.loadStats)
  const runSmoke = useDatasetsStore((s) => s.runSmoke)
  const close = useDatasetsStore((s) => s.select)

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') close(null)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [close])

  if (!entry) return null

  const data = inspect?.data
  const kind = data?.kind ?? 'unknown'

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6"
      onClick={(e) => { if (e.target === e.currentTarget) close(null) }}
    >
      <div className="flex h-full max-h-[92vh] w-full max-w-6xl flex-col overflow-hidden rounded-lg border border-[#1f2429] bg-[#0e1216] shadow-2xl">
        <div className="flex items-center gap-2 border-b border-[#1f2429] px-4 py-3">
          <span className={`rounded px-1.5 py-0.5 font-mono text-[11px] ${colorFor(kind)}`}>
            {iconFor(kind)}
          </span>
          <span className="flex-1 truncate text-sm text-[#e6e8eb]">{entry.name}</span>
          <span className="text-[11px] text-[#6f767e]">{formatSize(entry.size_bytes)}</span>
          <button
            onClick={() => close(null)}
            className="ml-2 rounded px-2 py-0.5 text-[#6f767e] hover:bg-[#1a1e22] hover:text-[#e6e8eb]"
            title="close (Esc)"
          >
            ×
          </button>
        </div>
        <div className="flex border-b border-[#1f2429] text-xs">
          {(['overview', 'stats', 'smoke'] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => {
                setTab(t)
                if (t === 'stats') void loadStats(relpath)
              }}
              className={`px-4 py-2 ${
                tab === t ? 'border-b border-[var(--accent)] text-[#e6e8eb]' : 'text-[#6f767e] hover:text-[#9aa1a8]'
              }`}
            >
              {t === 'smoke' ? 'Smoke test' : t === 'overview' ? 'Overview' : 'Stats'}
            </button>
          ))}
        </div>
        <div className="min-h-0 flex-1 overflow-auto p-5 text-xs">
          {inspect?.loading && <div className="text-[#6f767e]">inspecting…</div>}
          {inspect?.error && <ErrorBox msg={inspect.error} />}
          {data && !data.ok && <InspectError data={data} />}
          {/* A broken/empty .manifest still gets the editor so it can be fixed/created. */}
          {tab === 'overview' && data && !data.ok && relpath.endsWith('.manifest') && (
            <div className="mt-2"><ManifestEditor relpath={relpath} /></div>
          )}
          {tab === 'overview' && data && data.ok && <OverviewBody data={data} relpath={relpath} />}
          {tab === 'stats' && (
            <StatsBody loading={stats?.loading} error={stats?.error} data={stats?.data ?? null} kind={kind} />
          )}
          {tab === 'smoke' && (
            <SmokeBody relpath={relpath} loading={smoke?.loading} error={smoke?.error} data={smoke?.data ?? null} runSmoke={runSmoke} />
          )}
        </div>
      </div>
    </div>
  )
}

function ErrorBox({ msg }: { msg: string }) {
  return <div className="rounded bg-rose-900/20 px-2 py-1.5 text-rose-300">{msg}</div>
}

function formatShape(shape: number[] | number[][]): string {
  if (shape.length === 0) return '()'
  if (typeof shape[0] === 'number') return `[${(shape as number[]).join(', ')}]`
  return (shape as number[][]).map((s) => `[${s.join(', ')}]`).join(' · ')
}

function InspectError({ data }: { data: InspectResult }) {
  if (data.ok) return null
  return (
    <div className="rounded bg-rose-900/20 px-2 py-1.5 text-rose-300">
      <div>{data.error ?? 'inspect failed'}</div>
      {data.missing_dep && (
        <div className="mt-1 text-[10px] text-rose-200/80">
          tip: <code>pip install {data.missing_dep}</code> in der spinoml-dev env
        </div>
      )}
    </div>
  )
}

function OverviewBody({ data, relpath }: { data: InspectResult; relpath: string }) {
  if (!data.ok) return null
  switch (data.kind) {
    case 'tabular': return <TabularOverview d={data} relpath={relpath} />
    case 'image_folder': return <ImageOverview d={data} relpath={relpath} />
    case 'tensor': return <TensorOverview d={data} relpath={relpath} />
    case 'protein': return <ProteinOverview d={data} relpath={relpath} />
    case 'molecule': return <MoleculeOverview d={data} relpath={relpath} />
    case 'huggingface': return <HfOverview d={data} />
    case 'pyg': return <PygOverview d={data} />
    case 'graph_folder': return <GraphFolderOverview d={data} />
    case 'manifest': return <ManifestOverview d={data} relpath={relpath} />
    default: return <div className="text-[#6f767e]">Unbekanntes Format.</div>
  }
}

function UseAsInputButton({ shape, label }: { shape: number[]; label?: string }) {
  const updateNodeParams = useGraphStore((s) => s.updateNodeParams)
  const inputNode = useGraphStore((s) => s.nodes.find((n) => n.data.layerType === 'Input'))
  const has = !!inputNode
  return (
    <button
      onClick={() => {
        if (!inputNode) return
        updateNodeParams(inputNode.id, { ...inputNode.data.params, shape })
      }}
      disabled={!has}
      title={has ? 'set shape on the Input node' : 'no Input node in graph'}
      className="rounded border border-[#2a3038] bg-[#1a1e22] px-2 py-0.5 text-[10px] text-[#9aa1a8] hover:border-[var(--accent)] hover:text-[#e6e8eb] disabled:opacity-40"
    >
      {label ?? `Use [${shape.join(', ')}] as input`}
    </button>
  )
}

function TabularOverview({ d, relpath: _relpath }: { d: TabularInspect; relpath: string }) {
  return (
    <div className="space-y-2">
      <div className="text-[#9aa1a8]">
        {d.rows.toLocaleString()} rows × {d.cols} cols
      </div>
      <div className="overflow-x-auto">
        <table className="text-[10px]">
          <thead className="text-[#6f767e]">
            <tr>
              {d.columns.map((c, i) => (
                <th key={c} className="border-b border-[#1f2429] px-2 py-1 text-left">
                  <div>{c}</div>
                  <div className="text-[#5a6068]">{d.dtypes[i]}</div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {d.head.map((row, ri) => (
              <tr key={ri} className="hover:bg-[#1a1e22]">
                {row.map((cell, ci) => (
                  <td key={ci} className="border-b border-[#1f2429]/40 px-2 py-1 text-[#e6e8eb]">
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <UseAsInputButton shape={[1, d.cols]} label={`Use [1, ${d.cols}] as input`} />
    </div>
  )
}

function ImageOverview({ d, relpath: _relpath }: { d: ImageFolderInspect; relpath: string }) {
  const w = d.sample_size?.[0] ?? 64
  const h = d.sample_size?.[1] ?? 64
  return (
    <div className="space-y-2">
      <div className="text-[#9aa1a8]">
        {d.n_images.toLocaleString()} images · {d.n_classes} classes
        {d.sample_size && ` · sample ${w}×${h}`}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {d.thumbnails.map((t) => (
          <div key={t.name} className="flex flex-col items-center gap-0.5">
            <img
              src={`data:image/jpeg;base64,${t.b64}`}
              alt={t.name}
              className="h-16 w-16 rounded border border-[#1f2429] object-cover"
            />
            <span className="text-[9px] text-[#6f767e]">{t.name.split('/')[0]}</span>
          </div>
        ))}
      </div>
      <div>
        <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">classes</div>
        <div className="space-y-0.5">
          {d.classes.map((c) => (
            <div key={c.name} className="flex items-center justify-between">
              <span className="text-[#e6e8eb]">{c.name}</span>
              <span className="text-[#6f767e]">{c.count}</span>
            </div>
          ))}
        </div>
      </div>
      <UseAsInputButton shape={[1, 3, h, w]} />
    </div>
  )
}

function TensorOverview({ d, relpath: _relpath }: { d: TensorInspect; relpath: string }) {
  if (d.is_graph) {
    return (
      <div className="space-y-2">
        <div className="text-[#9aa1a8]">PyTorch-Geometric-Graph · {d.num_nodes} Knoten · {d.num_edges} Kanten</div>
        <GraphFieldTable fields={d.fields ?? []} />
        {d.x_preview && <XPreviewView x={d.x_preview} />}
        <div className="text-[10px] text-[#6f767e]">
          Binde an GNN-Inputs mit passendem <strong>Namen</strong> (<code>x</code>, <code>edge_index</code>,{' '}
          <code>edge_attr</code>, …) — jeder Input zieht das gleichnamige Feld.
        </div>
      </div>
    )
  }
  return (
    <div className="space-y-2">
      {d.shape && (
        <div className="text-[#9aa1a8]">
          shape <code className="text-[#e6e8eb]">[{d.shape.join(', ')}]</code> · dtype{' '}
          <code className="text-[#e6e8eb]">{d.dtype}</code>
        </div>
      )}
      {d.mean != null && (
        <div className="text-[#6f767e]">
          mean {d.mean.toFixed(4)} · min {d.min?.toFixed(4)} · max {d.max?.toFixed(4)}
        </div>
      )}
      {d.container === 'dict' && d.keys && (
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">keys</div>
          <div className="space-y-0.5">
            {d.keys.map((k) => (
              <div key={k.key} className="flex justify-between">
                <span className="text-[#e6e8eb]">{k.key}</span>
                <span className="text-[#6f767e]">
                  {k.shape ? `[${k.shape.join(', ')}]` : k.dtype}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
      {d.container === 'npz' && d.arrays && (
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">arrays</div>
          {Object.entries(d.arrays).map(([k, v]) => (
            <div key={k} className="flex justify-between">
              <span className="text-[#e6e8eb]">{k}</span>
              <span className="text-[#6f767e]">[{v.join(', ')}]</span>
            </div>
          ))}
        </div>
      )}
      {d.shape && <UseAsInputButton shape={d.shape} />}
    </div>
  )
}

function ProteinOverview({ d, relpath: _relpath }: { d: ProteinInspect; relpath: string }) {
  return (
    <div className="space-y-2">
      <div className="text-[#9aa1a8]">
        {d.chains} chain{d.chains !== 1 ? 's' : ''} · {d.residues.toLocaleString()} residues · {d.atoms.toLocaleString()} atoms
      </div>
      <div className="text-[10px] text-[#5a6068]">parser: {d.parser}</div>
      {d.chain_info && (
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">chains</div>
          {d.chain_info.map((c) => (
            <div key={c.id} className="flex justify-between">
              <span className="text-[#e6e8eb]">{c.id || '(empty)'}</span>
              <span className="text-[#6f767e]">{c.residues} res · {c.atoms} atm</span>
            </div>
          ))}
        </div>
      )}
      <UseAsInputButton shape={[1, 256]} label="Use [1, 256] residue-seq as input" />
    </div>
  )
}

function MoleculeOverview({ d, relpath: _relpath }: { d: MoleculeInspect; relpath: string }) {
  return (
    <div className="space-y-2">
      <div className="text-[#9aa1a8]">
        {d.n_molecules.toLocaleString()} SMILES · parser: {d.parser}
      </div>
      <div>
        <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">head</div>
        <div className="space-y-0.5 font-mono text-[10px]">
          {d.head.slice(0, 8).map((s, i) => (
            <div key={i} className="truncate text-[#e6e8eb]">{s}</div>
          ))}
        </div>
      </div>
      {d.sample_info && d.sample_info.some((m) => m.valid) && (
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">rdkit info</div>
          {d.sample_info.filter((m) => m.valid).slice(0, 4).map((m, i) => (
            <div key={i} className="flex justify-between text-[10px]">
              <span className="truncate font-mono text-[#e6e8eb]">{m.canonical ?? m.smiles}</span>
              <span className="ml-2 text-[#6f767e]">{m.atoms}a · {m.bonds}b · MW {m.mw}</span>
            </div>
          ))}
        </div>
      )}
      {d.graph0 && (
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">
            erste Struktur als Graph · {d.graph0.n_nodes} Atome
          </div>
          <NodeLinkGraph edges={d.graph0.edges} nNodes={d.graph0.n_nodes} hex="#a78bfa" size={150} />
          <div className="truncate font-mono text-[10px] text-[#6f767e]">{d.graph0.smiles}</div>
        </div>
      )}
      <UseAsInputButton shape={[1, 64]} label="Use [1, 64] byte-encoded as input" />
      <div className="text-[10px] text-[#6f767e]">
        Als Graph nutzbar (RDKit): binde dieselbe Datei an Inputs <code>x</code> (Atom-Merkmale),
        <code> edge_index</code> (Bindungen) und <code>batch</code> für ein GNN.
      </div>
    </div>
  )
}

function HfOverview({ d }: { d: HuggingfaceInspect }) {
  return (
    <div className="space-y-2">
      <div className="text-[#9aa1a8]"><code className="text-[#e6e8eb]">{d.name}</code></div>
      {d.description && <div className="text-[#6f767e]">{d.description}</div>}
      <div>
        <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">splits</div>
        {Object.entries(d.splits).map(([k, v]) => (
          <div key={k} className="flex justify-between">
            <span className="text-[#e6e8eb]">{k}</span>
            <span className="text-[#6f767e]">{v.num_examples?.toLocaleString() ?? '?'}</span>
          </div>
        ))}
      </div>
      <div>
        <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">features</div>
        {Object.entries(d.features).map(([k, v]) => (
          <div key={k} className="flex justify-between text-[10px]">
            <span className="text-[#e6e8eb]">{k}</span>
            <span className="ml-2 truncate text-[#6f767e]">{v}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function XPreviewView({ x }: { x: { rows: number; cols: number; grid: number[][] } }) {
  const nCols = x.grid[0]?.length ?? 0
  return (
    <div>
      <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">
        Werte in x (erste {x.grid.length}/{x.rows} Knoten × {nCols}/{x.cols} Features)
      </div>
      <div className="max-h-48 overflow-auto rounded border border-[#1f2429]">
        <table className="font-mono text-[10px]">
          <thead className="sticky top-0 bg-[#0b0e11] text-[#5b6168]">
            <tr>
              <th className="px-1.5 py-0.5 text-left">Knoten</th>
              {Array.from({ length: nCols }).map((_, c) => (
                <th key={c} className="px-1.5 py-0.5 text-right">f{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {x.grid.map((row, r) => (
              <tr key={r} className="odd:bg-[#0e1216]">
                <td className="px-1.5 py-0.5 text-[#6f767e]">{r}</td>
                {row.map((v, c) => (
                  <td key={c} className="px-1.5 py-0.5 text-right text-[#e6e8eb]">{v}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-0.5 text-[10px] text-[#5b6168]">jede Zeile = ein Knoten · Werte gerundet</div>
    </div>
  )
}

function GraphFieldTable({ fields }: { fields: GraphField[] }) {
  if (!fields.length) return null
  return (
    <div>
      <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">Felder pro Graph</div>
      <div className="space-y-0.5 font-mono text-[10px]">
        {fields.map((f) => (
          <div key={f.name} className="flex justify-between">
            <span className="text-[#e6e8eb]">{f.name}</span>
            <span className="text-[#6f767e]">[{f.shape.join(', ')}] · {f.dtype.replace('torch.', '')}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function GraphFolderOverview({ d }: { d: GraphFolderInspect }) {
  return (
    <div className="space-y-2">
      <div className="text-[#9aa1a8]">
        {d.n_graphs.toLocaleString()} Graphen · {d.num_node_features} Knoten-Features
        {d.edge_dim ? ` · ${d.edge_dim} Kanten-Features` : ''}
      </div>
      <GraphFieldTable fields={d.fields} />
      {d.x_preview && <XPreviewView x={d.x_preview} />}
      <div className="text-[10px] text-[#5b6168]">
        3D-Koordinaten u. Ä. stecken in den Knoten-Features (<code>x</code>) — siehe Werte oben.
      </div>
      {d.preview && d.preview.edges.length > 0 && (
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">erster Graph ({d.example})</div>
          <NodeLinkGraph edges={d.preview.edges} nNodes={d.preview.n_nodes} hex="#34d399" size={160} />
        </div>
      )}
      <div className="text-[10px] text-[#6f767e]">
        Datensatz = dieser Ordner. Binde ihn an GNN-Inputs mit passendem <strong>Namen</strong>{' '}
        (<code>x</code>, <code>edge_index</code>, <code>edge_attr</code>, <code>y</code>) — jeder Input zieht sein Feld.
      </div>
    </div>
  )
}

function PygOverview({ d }: { d: PygInspect }) {
  return (
    <div className="space-y-2">
      <div className="text-[#9aa1a8]"><code className="text-[#e6e8eb]">{d.name}</code></div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-[11px]">
        <span className="text-[#6f767e]">Graphen</span><span className="text-[#e6e8eb]">{d.num_graphs.toLocaleString()}</span>
        <span className="text-[#6f767e]">Knoten</span><span className="text-[#e6e8eb]">{d.num_nodes.toLocaleString()}</span>
        <span className="text-[#6f767e]">Kanten</span><span className="text-[#e6e8eb]">{d.num_edges.toLocaleString()}</span>
        <span className="text-[#6f767e]">Merkmale/Knoten</span><span className="text-[#e6e8eb]">{d.num_node_features}</span>
        <span className="text-[#6f767e]">Klassen</span><span className="text-[#e6e8eb]">{d.num_classes}</span>
      </div>
      <div className="text-[10px] text-[#6f767e]">
        Als Graph nutzbar: binde dieselbe Datei an Inputs <code>x</code> (Knoten-Merkmale),
        <code> edge_index</code> (Kanten) und <code>batch</code> — jeder Input zieht sein Feld.
      </div>
    </div>
  )
}

// ── Graphical manifest editor: edit the .manifest JSON via a form, write to disk ──
type BranchMode = 'molecule' | 'dir' | 'path'
type BranchCfg = { name: string; column: string; mode: BranchMode; dir: string; match: 'exact' | 'contains'; ext: string }
type ManifestCfg = { table: string; branches: BranchCfg[]; target: { column: string; type: 'regression' | 'classification' }; cache: boolean }

function normalizeManifest(raw: any): ManifestCfg {
  const branches: BranchCfg[] = Object.entries(raw?.pairs ?? {}).map(([name, s]: [string, any]) => ({
    name,
    column: String(s?.column ?? ''),
    mode: s?.kind === 'molecule' ? 'molecule' : s?.dir ? 'dir' : 'path',
    dir: String(s?.dir ?? ''),
    match: s?.match === 'exact' ? 'exact' : 'contains',
    ext: String(s?.ext ?? '.pt'),
  }))
  return {
    table: String(raw?.table ?? ''),
    branches: branches.length ? branches : [{ name: 'graph', column: '', mode: 'molecule', dir: '', match: 'contains', ext: '.pt' }],
    target: { column: String(raw?.target?.column ?? ''), type: raw?.target?.type === 'classification' ? 'classification' : 'regression' },
    cache: raw?.cache !== false, // default on
  }
}

function buildManifestJson(cfg: ManifestCfg): unknown {
  const pairs: Record<string, unknown> = {}
  for (const b of cfg.branches) {
    if (!b.name) continue
    if (b.mode === 'molecule') pairs[b.name] = { column: b.column, kind: 'molecule' }
    else if (b.mode === 'dir') pairs[b.name] = { column: b.column, dir: b.dir, match: b.match, ext: b.ext }
    else pairs[b.name] = { column: b.column }
  }
  const out: Record<string, unknown> = { table: cfg.table, pairs }
  if (cfg.target.column) out.target = { column: cfg.target.column, type: cfg.target.type }
  if (!cfg.cache) out.cache = false // cache is on by default; only persist when off
  return out
}

function ManifestEditor({ relpath }: { relpath: string }) {
  const [cfg, setCfg] = useState<ManifestCfg | null>(null)
  const [status, setStatus] = useState<string>('')
  const inspectAction = useDatasetsStore((s) => s.inspect)
  const inspects = useDatasetsStore((s) => s.inspects)
  const datasets = useDatasetsStore((s) => s.entries)

  useEffect(() => {
    let alive = true
    void (async () => {
      try {
        const { fs } = await import('../connections/backend')
        const txt = await fs.read(relpath)
        if (alive) setCfg(normalizeManifest(JSON.parse(txt)))
      } catch {
        if (alive) setCfg(normalizeManifest({}))
      }
    })()
    return () => { alive = false }
  }, [relpath])

  // Column suggestions from the referenced table (inspected as tabular).
  const tableRel = cfg?.table ? `datasets/${cfg.table}` : ''
  useEffect(() => { if (tableRel) void inspectAction(tableRel) }, [tableRel, inspectAction])
  const tIns = tableRel ? inspects[tableRel]?.data : null
  const columns: string[] = tIns && tIns.ok && tIns.kind === 'tabular' ? tIns.columns : []
  const tableOptions = datasets.filter((e) => /\.(csv|tsv|parquet)$/i.test(e.name)).map((e) => e.name)

  if (!cfg) return <div className="text-[10px] text-[#6f767e]">lädt…</div>

  const up = (patch: Partial<ManifestCfg>) => { setCfg({ ...cfg, ...patch }); setStatus('') }
  const upBranch = (i: number, patch: Partial<BranchCfg>) =>
    up({ branches: cfg.branches.map((b, j) => (j === i ? { ...b, ...patch } : b)) })
  const addBranch = () => up({ branches: [...cfg.branches, { name: '', column: '', mode: 'dir', dir: '', match: 'contains', ext: '.pt' }] })
  const delBranch = (i: number) => up({ branches: cfg.branches.filter((_, j) => j !== i) })

  const save = async () => {
    try {
      const { fs } = await import('../connections/backend')
      await fs.write(relpath, JSON.stringify(buildManifestJson(cfg), null, 2) + '\n')
      setStatus('gespeichert ✓')
      void inspectAction(relpath, true) // refresh the slots view below
    } catch (e) {
      setStatus(`Fehler: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  const inp = 'rounded border border-[#1f2429] bg-[#0b0e11] px-1.5 py-1 text-[11px] text-[#e6e8eb]'
  return (
    <div className="space-y-2 rounded border border-[#1f2429] bg-[#0d1117] p-2">
      <div className="text-[10px] uppercase tracking-wider text-[#6f767e]">Manifest bearbeiten</div>

      <label className="flex items-center gap-2 text-[11px] text-[#9aa1a8]">
        <span className="w-16 shrink-0">Tabelle</span>
        <input className={`${inp} flex-1`} list="mf-tables" value={cfg.table}
          onChange={(e) => up({ table: e.target.value })} placeholder="reactions.csv" />
        <datalist id="mf-tables">{tableOptions.map((t) => <option key={t} value={t} />)}</datalist>
      </label>

      <datalist id="mf-cols">{columns.map((c) => <option key={c} value={c} />)}</datalist>

      <div className="text-[10px] uppercase tracking-wider text-[#6f767e]">Branches (= Encoder-Inputs)</div>
      {cfg.branches.map((b, i) => (
        <div key={i} className="space-y-1 rounded border border-[#1f2429] bg-[#0b0e11] p-1.5">
          <div className="flex items-center gap-1">
            <input className={`${inp} w-24`} value={b.name} onChange={(e) => upBranch(i, { name: e.target.value })} placeholder="ligand" />
            <input className={`${inp} flex-1`} list="mf-cols" value={b.column} onChange={(e) => upBranch(i, { column: e.target.value })} placeholder="Spalte" />
            <button onClick={() => delBranch(i)} className="px-1 text-[11px] text-rose-400 hover:text-rose-300" title="Branch entfernen">✕</button>
          </div>
          <div className="flex flex-wrap items-center gap-1">
            <select className={inp} value={b.mode} onChange={(e) => upBranch(i, { mode: e.target.value as BranchMode })}>
              <option value="molecule">SMILES → Graph (RDKit)</option>
              <option value="dir">ID → Datei in Verzeichnis</option>
              <option value="path">Spalte = Pfad</option>
            </select>
            {b.mode === 'dir' && <>
              <input className={`${inp} w-32`} value={b.dir} onChange={(e) => upBranch(i, { dir: e.target.value })} placeholder="graphs/proteins" />
              <select className={inp} value={b.match} onChange={(e) => upBranch(i, { match: e.target.value as 'exact' | 'contains' })}>
                <option value="contains">ID im Dateinamen</option>
                <option value="exact">exakt &lt;id&gt;&lt;ext&gt;</option>
              </select>
              <input className={`${inp} w-16`} value={b.ext} onChange={(e) => upBranch(i, { ext: e.target.value })} placeholder=".pt" />
            </>}
          </div>
        </div>
      ))}
      <button onClick={addBranch} className="rounded border border-[#1f2429] px-2 py-0.5 text-[11px] text-[#9aa1a8] hover:bg-[#13171b]">+ Branch</button>

      <div className="text-[10px] uppercase tracking-wider text-[#6f767e]">Target</div>
      <div className="flex items-center gap-1">
        <input className={`${inp} flex-1`} list="mf-cols" value={cfg.target.column}
          onChange={(e) => up({ target: { ...cfg.target, column: e.target.value } })} placeholder="affinity" />
        <select className={inp} value={cfg.target.type}
          onChange={(e) => up({ target: { ...cfg.target, type: e.target.value as 'regression' | 'classification' } })}>
          <option value="regression">Regression</option>
          <option value="classification">Klassifikation</option>
        </select>
      </div>

      <label className="flex items-center gap-1.5 text-[11px] text-[#9aa1a8]">
        <input type="checkbox" checked={cfg.cache} onChange={(e) => up({ cache: e.target.checked })} />
        Molekül-Graphen (RDKit) als <code>.pt</code> cachen (in <code>datasets/.graphcache/</code>)
      </label>

      <div className="flex items-center gap-2 pt-1">
        <button onClick={() => void save()} className="rounded bg-[var(--accent-sel)] px-3 py-1 text-[11px] text-[var(--accent)] hover:bg-[var(--accent-sel-hover)]">Speichern</button>
        {status && <span className="text-[10px] text-emerald-300/80">{status}</span>}
      </div>
    </div>
  )
}

function ManifestOverview({ d, relpath }: { d: ManifestInspect; relpath: string }) {
  const byBranch = new Map<string, typeof d.slots>()
  for (const s of d.slots) {
    if (s.field === 'target') continue
    const branch = s.field.split('.')[0]
    if (!byBranch.has(branch)) byBranch.set(branch, [])
    byBranch.get(branch)!.push(s)
  }
  return (
    <div className="space-y-2">
      <div className="text-[#9aa1a8]">
        {d.n_rows.toLocaleString()} Paare · Tabelle <code className="text-[#e6e8eb]">{d.table}</code>
      </div>
      <div className="text-[10px] text-[#6f767e]">
        Jede Zeile koppelt {d.branches.map((b, i) => (
          <span key={b}><code className="text-[#c8cdd3]">{b}</code>{i < d.branches.length - 1 ? ' + ' : ''}</span>
        ))}{d.target ? <> → Ziel <code className="text-[#c8cdd3]">{d.target.column}</code> ({d.target.type})</> : ''}.
        Bind jeden Branch-Slot unten an den passenden Encoder-Input (Inspector).
      </div>
      <ManifestEditor relpath={relpath} />
      {[...byBranch.entries()].map(([branch, slots]) => (
        <div key={branch}>
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">{branch}</div>
          <div className="space-y-0.5">
            {slots.map((s) => (
              <div key={s.field} className="flex items-center justify-between rounded bg-[#0b0e11] px-1.5 py-1 font-mono text-[10px]">
                <span className="text-[#c8cdd3]">{s.field}</span>
                <span className="text-[#6f767e]">[{s.shape.join(', ')}] · {s.dtype}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
      {d.target && (
        <div className="flex items-center justify-between rounded bg-[#0b0e11] px-1.5 py-1 font-mono text-[10px]">
          <span className="text-emerald-300/90">target</span>
          <span className="text-[#6f767e]">{d.target.column} · {d.target.type}</span>
        </div>
      )}
      {d.notes && d.notes.length > 0 && (
        <div className="space-y-0.5 rounded border border-amber-900/40 bg-amber-950/20 p-1.5">
          {d.notes.map((n, i) => <div key={i} className="text-[10px] text-amber-400/90">⚠ {n}</div>)}
        </div>
      )}
      <div className="text-[10px] text-[#5b6168]">
        Slots zeigen die Formen aus Zeile 0. „contains"-Auflösung matcht die ID im Dateinamen
        (z. B. <code>P12345</code> → <code>AF-P12345-F1-model_v4.pt</code>).
      </div>
    </div>
  )
}

function StatsBody({ loading, error, data, kind }: { loading?: boolean; error?: string | null; data: StatsResult | null; kind: string }) {
  if (loading) return <div className="text-[#6f767e]">computing stats…</div>
  if (error) return <ErrorBox msg={error} />
  if (!data) return <div className="text-[#6f767e]">no stats yet</div>
  if (!data.ok) return <ErrorBox msg={('error' in data && data.error) || 'stats failed'} />
  if (data.kind === 'tabular') return <TabularStatsView d={data as TabularStats} />
  if (data.kind === 'image_folder') return <ImageStatsView d={data as ImageFolderStats} />
  if (data.kind === 'tensor') return <TensorStatsView d={data as TensorStats} />
  if (data.kind === 'molecule') return <MoleculeStatsView d={data as MoleculeStats} />
  return <div className="text-[#6f767e]">Stats für {kind} sind aktuell nur in der Übersicht.</div>
}

function Bar({ frac, max = 1 }: { frac: number; max?: number }) {
  const pct = Math.max(0, Math.min(100, (frac / max) * 100))
  return (
    <div className="h-1.5 w-full overflow-hidden rounded bg-[#1a1e22]">
      <div className="h-full bg-[var(--accent)]/70" style={{ width: `${pct}%` }} />
    </div>
  )
}

function Hist({ counts, edges, color = 'var(--accent)' }: { counts: number[]; edges: number[]; color?: string }) {
  const max = Math.max(...counts, 1)
  return (
    <div>
      <div className="flex h-12 items-end gap-px">
        {counts.map((c, i) => (
          <div
            key={i}
            className="flex-1"
            style={{ height: `${(c / max) * 100}%`, background: `${color}b3` }}
            title={`${c} in [${edges[i]?.toFixed?.(2) ?? edges[i]}, ${edges[i + 1]?.toFixed?.(2) ?? edges[i + 1]}]`}
          />
        ))}
      </div>
      <div className="mt-0.5 flex justify-between text-[9px] text-[#5a6068]">
        <span>{Number(edges[0]).toFixed(2)}</span>
        <span>{Number(edges[edges.length - 1]).toFixed(2)}</span>
      </div>
    </div>
  )
}

function TabularStatsView({ d }: { d: TabularStats }) {
  return (
    <div className="space-y-3">
      <div className="text-[#9aa1a8]">{d.rows.toLocaleString()} rows × {d.cols} cols</div>
      <div className="space-y-2">
        {d.summary.map((s) => (
          <div key={s.col} className="rounded border border-[#1f2429] p-2">
            <div className="flex items-baseline justify-between">
              <span className="text-[#e6e8eb]">{s.col}</span>
              <span className="text-[10px] text-[#6f767e]">{s.dtype} · {s.unique} unique · {s.missing} missing</span>
            </div>
            {s.mean != null && (
              <div className="mt-0.5 text-[10px] text-[#6f767e]">
                μ {s.mean.toFixed(3)} · σ {s.std?.toFixed(3)} · [{s.min?.toFixed(2)}, {s.max?.toFixed(2)}]
              </div>
            )}
            {s.hist && <div className="mt-1"><Hist counts={s.hist.counts} edges={s.hist.edges} /></div>}
          </div>
        ))}
      </div>
      {d.corr && d.corr_cols.length > 0 && (
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">correlations</div>
          <CorrMatrix corr={d.corr} cols={d.corr_cols} />
        </div>
      )}
    </div>
  )
}

function CorrMatrix({ corr, cols }: { corr: number[][]; cols: string[] }) {
  return (
    <table className="text-[10px]">
      <thead>
        <tr>
          <th className="px-1"></th>
          {cols.map((c) => (
            <th key={c} className="rotate-[-30deg] px-1 text-[#6f767e]">{c}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {corr.map((row, i) => (
          <tr key={i}>
            <td className="pr-1 text-right text-[#6f767e]">{cols[i]}</td>
            {row.map((v, j) => {
              const a = Math.abs(v)
              const bg = v >= 0
                ? `rgba(36, 200, 219, ${a * 0.7})`
                : `rgba(248, 113, 113, ${a * 0.7})`
              return (
                <td key={j} className="border border-[#1f2429] text-center" style={{ background: bg, width: 28, height: 18 }}>
                  <span className={a > 0.5 ? 'text-[#e6e8eb]' : 'text-[#6f767e]'}>{v.toFixed(2)}</span>
                </td>
              )
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function ImageStatsView({ d }: { d: ImageFolderStats }) {
  const max = Math.max(...d.classes.map((c) => c.count), 1)
  return (
    <div className="space-y-3">
      <div>
        <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">class distribution</div>
        <div className="space-y-1">
          {d.classes.map((c) => (
            <div key={c.name}>
              <div className="flex justify-between text-[10px]">
                <span className="text-[#e6e8eb]">{c.name}</span>
                <span className="text-[#6f767e]">{c.count}</span>
              </div>
              <Bar frac={c.count} max={max} />
            </div>
          ))}
        </div>
      </div>
      <div>
        <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">image sizes (sample of {d.n_samples_for_size})</div>
        <div className="space-y-0.5 text-[10px]">
          {d.size_hist.map((s) => (
            <div key={s.size} className="flex justify-between">
              <span className="text-[#e6e8eb]">{s.size}</span>
              <span className="text-[#6f767e]">{s.count}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function TensorStatsView({ d }: { d: TensorStats }) {
  return (
    <div className="space-y-2">
      {d.shape && <div className="text-[#9aa1a8]">shape [{d.shape.join(', ')}] · dtype {d.dtype}</div>}
      <div className="text-[10px] text-[#6f767e]">
        mean {d.mean?.toFixed(4)} · std {d.std?.toFixed(4)} · min {d.min?.toFixed(4)} · max {d.max?.toFixed(4)}
        {d.zeros_frac != null && ` · zeros ${(d.zeros_frac * 100).toFixed(1)}%`}
      </div>
      {d.hist && <Hist counts={d.hist.counts} edges={d.hist.edges} color="#fbbf24" />}
    </div>
  )
}

function MoleculeStatsView({ d }: { d: MoleculeStats }) {
  return (
    <div className="space-y-3">
      {d.note && <div className="text-[10px] text-[#6f767e]">{d.note}</div>}
      {d.mw_mean != null && <div className="text-[#9aa1a8]">MW μ {d.mw_mean} · atoms μ {d.atom_mean}</div>}
      {d.mw_hist && (
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">molecular weight</div>
          <Hist counts={d.mw_hist.counts} edges={d.mw_hist.edges} color="#c084fc" />
        </div>
      )}
      {d.atom_hist && (
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">atom count</div>
          <Hist counts={d.atom_hist.counts} edges={d.atom_hist.edges} color="#c084fc" />
        </div>
      )}
    </div>
  )
}

function SmokeBody({
  relpath, loading, error, data, runSmoke,
}: {
  relpath: string
  loading?: boolean
  error?: string | null
  data: SmokeResult | null
  runSmoke: (relpath: string, inputShape?: number[]) => Promise<void>
}) {
  const inputShape = useGraphStore((s) => {
    const inputNode = s.nodes.find((n) => n.data.layerType === 'Input')
    const sh = inputNode?.data.params.shape as number[] | undefined
    return sh ?? null
  })
  const hasModel = useGraphStore((s) => s.nodes.length > 0)
  const history = useDatasetsStore((s) => s.history)
  const historyError = useDatasetsStore((s) => s.historyError)
  const loadHistory = useDatasetsStore((s) => s.loadHistory)

  useEffect(() => { void loadHistory() }, [loadHistory])

  const datasetHistory = history.filter((h) => h.dataset === relpath).slice(0, 12)
  return (
    <div className="space-y-2">
      <div className="text-[#9aa1a8]">
        Schickt einen Sample-Batch aus dem Datensatz durch das aktuelle Modell.
      </div>
      {inputShape && (
        <div className="text-[10px] text-[#6f767e]">
          input shape vom Graph: <code className="text-[#e6e8eb]">[{inputShape.join(', ')}]</code>
        </div>
      )}
      <button
        onClick={() => void runSmoke(relpath, inputShape ?? undefined)}
        disabled={loading || !hasModel}
        className="rounded border border-[#2a3038] bg-[#1a1e22] px-2 py-1 text-[11px] text-[#e6e8eb] hover:border-[var(--accent)] disabled:opacity-40"
      >
        {loading ? 'running…' : 'Run smoke test'}
      </button>
      {!hasModel && <div className="text-[10px] text-[#6f767e]">Zieh erst ein paar Layer auf den Canvas.</div>}
      {error && <ErrorBox msg={error} />}
      {data && data.ok && (
        <div className="rounded border border-emerald-900/40 bg-emerald-900/10 p-2">
          <div className="text-emerald-300">✓ forward pass succeeded</div>
          <div className="mt-1 space-y-0.5 text-[10px] text-[#9aa1a8]">
            <div>input: <code className="text-[#e6e8eb]">{formatShape(data.input_shape)}</code></div>
            <div>output: <code className="text-[#e6e8eb]">{data.output_shape ? formatShape(data.output_shape) : '(non-tensor)'}</code></div>
            <div>params: {data.n_params.toLocaleString()}</div>
            <div>timings: sample {data.timings_ms.sample.toFixed(1)}ms · forward {data.timings_ms.forward.toFixed(1)}ms</div>
            {data.sample_note && <div className="text-[#6f767e]">{data.sample_note}</div>}
          </div>
        </div>
      )}
      {data && !data.ok && (
        <SmokeError result={data} requestedShape={inputShape} relpath={relpath} />
      )}
      {datasetHistory.length > 0 && <SmokeHistory entries={datasetHistory} />}
      {historyError && <div className="text-[10px] text-rose-400">{historyError}</div>}
    </div>
  )
}

function SmokeError({
  result, requestedShape, relpath,
}: {
  result: Extract<SmokeResult, { ok: false }>
  requestedShape: number[] | null
  relpath: string
}) {
  const [showTrace, setShowTrace] = useState(false)
  const stageLabel: Record<string, string> = {
    sample: 'Sample-Build',
    compile: 'Code-Compile',
    construct: 'Modell-Init',
    forward: 'Forward-Pass',
  }
  const hint = buildHint(result, requestedShape)
  const updateNodeParams = useGraphStore((s) => s.updateNodeParams)
  // Select the stable nodes array, then filter in render — a `.filter()` INSIDE
  // the selector returns a fresh array each call ("getSnapshot should be cached"
  // → infinite loop → crash). Include Graph (kind input) too.
  const nodes = useGraphStore((s) => s.nodes)
  const inputNodes = nodes.filter((n) => n.data.layerType === 'Input' || n.data.layerType === 'Graph')
  const inspectData = useDatasetsStore((s) => s.inspects[relpath]?.data)
  const naturalShape = sampleNaturalShape(inspectData)

  return (
    <div className="rounded border border-rose-700/40 bg-rose-950/30">
      <div className="flex items-center gap-2 border-b border-rose-700/40 bg-rose-900/30 px-3 py-2">
        <span className="rounded bg-rose-900/60 px-1.5 py-0.5 font-mono text-[10px] text-rose-200">
          {stageLabel[result.stage] ?? result.stage}
        </span>
        <span className="text-[11px] text-rose-200">smoke test failed</span>
      </div>
      <div className="space-y-2 px-3 py-2 text-[11px] text-[#e6e8eb]">
        <div className="font-mono text-rose-300">{result.error}</div>

        <div className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[10px] text-[#9aa1a8]">
          {result.input_shape && (
            <>
              <span className="text-[#6f767e]">tatsächlicher Input:</span>
              <code className="text-[#e6e8eb]">{formatShape(result.input_shape)}</code>
            </>
          )}
          {requestedShape && (
            <>
              <span className="text-[#6f767e]">erwartete Input-Shape:</span>
              <code className="text-[#e6e8eb]">[{requestedShape.join(', ')}]</code>
            </>
          )}
          {result.n_params != null && (
            <>
              <span className="text-[#6f767e]">Modellgröße:</span>
              <span className="text-[#9aa1a8]">{result.n_params.toLocaleString()} Params</span>
            </>
          )}
        </div>

        {hint && (
          <div className="rounded border border-amber-700/40 bg-amber-950/20 p-2 text-[10px] leading-snug text-amber-200">
            <div className="font-semibold">Hinweis</div>
            <div className="mt-0.5 text-amber-100/80">{hint}</div>
            {naturalShape && naturalShape.length > 0 && inputNodes.length > 0 && (
              <button
                onClick={() => {
                  const first = inputNodes[0]
                  updateNodeParams(first.id, { ...first.data.params, shape: naturalShape })
                }}
                className="mt-1 rounded border border-amber-600/40 bg-amber-900/30 px-1.5 py-0.5 text-[10px] text-amber-100 hover:bg-amber-900/50"
                title="set Input.shape to the dataset's natural shape"
              >
                Input-Shape auf [{naturalShape.join(', ')}] setzen
              </button>
            )}
          </div>
        )}

        {result.trace && (
          <div>
            <button
              onClick={() => setShowTrace((s) => !s)}
              className="text-[10px] text-[#6f767e] underline-offset-2 hover:text-[#9aa1a8] hover:underline"
            >
              {showTrace ? 'Stacktrace ausblenden' : 'Stacktrace anzeigen'}
            </button>
            {showTrace && (
              <pre className="mt-1 max-h-72 overflow-auto whitespace-pre-wrap rounded bg-[#0a0c0f]/60 p-2 text-[10px] leading-snug text-rose-200/80">
                {result.trace}
              </pre>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function buildHint(
  result: Extract<SmokeResult, { ok: false }>,
  requestedShape: number[] | null,
): string | null {
  if (result.stage === 'sample') {
    if (result.error.includes('pandas') || result.error.includes('rdkit') || result.error.includes('Pillow') || result.error.includes('biopython')) {
      return 'Eine optionale Python-Lib fehlt. Tipp: in der spinoml-dev env nachinstallieren (z.B. pip install pandas pillow rdkit-pypi biopython datasets).'
    }
    return 'Der Datensatz konnte nicht geladen werden — Pfad oder Format-Erkennung prüfen.'
  }
  if (result.stage === 'compile') {
    return 'Der generierte PyTorch-Code lässt sich nicht ausführen. Meist liegt ein ungültiger Parameter an einem Layer vor — schau in den Code-Preview unten.'
  }
  if (result.stage === 'construct') {
    return 'Das Modell konnte nicht konstruiert werden — wahrscheinlich ungültige Layer-Parameter (z.B. negative Kernel-Größe, in_channels=0).'
  }
  if (result.stage === 'forward') {
    const actualShape = Array.isArray(result.input_shape?.[0])
      ? (result.input_shape as number[][])[0]
      : (result.input_shape as number[] | undefined)
    if (requestedShape && actualShape && !shapesEqual(requestedShape, actualShape)) {
      return `Das Modell erwartet Input-Shape [${requestedShape.join(', ')}], aus dem Datensatz kam aber [${actualShape.join(', ')}]. Entweder Input-Shape am Input-Node anpassen, oder ein zur Architektur passendes Dataset wählen (Tabular ↔ Linear/MLP, Bilder ↔ Conv2d).`
    }
    if (result.error.includes('mat1 and mat2 shapes cannot be multiplied')) {
      return 'Eine Linear-Schicht hat in_features falsch gesetzt. Schau in der Inspector-Spalte rechts: die FixHint zeigt dir den richtigen Wert basierend auf der eingehenden Shape.'
    }
    if (result.error.includes('Expected') && result.error.includes('conv2d')) {
      return 'Conv2d braucht einen 4D-Tensor [N, C, H, W]. Tabulardaten sind nur 2D — entweder die Architektur auf MLP umstellen oder den Datensatz reshapen.'
    }
    return 'Das Modell läuft auf einen Forward-Fehler — die Layer passen nicht zur Shape-Kette. Schau dir Stage und Stacktrace an, und nutze die Inspector-Hints zum Fixen.'
  }
  return null
}

function shapesEqual(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

function sampleNaturalShape(inspect: InspectResult | null | undefined): number[] | null {
  if (!inspect || !inspect.ok) return null
  if (inspect.kind === 'tabular') return [1, inspect.cols]
  if (inspect.kind === 'image_folder' && inspect.sample_size) {
    return [1, 3, inspect.sample_size[1], inspect.sample_size[0]]
  }
  if (inspect.kind === 'tensor' && inspect.shape) return inspect.shape
  return null
}

function SmokeHistory({ entries }: { entries: SmokeHistoryEntry[] }) {
  return (
    <div className="mt-3 border-t border-[#1f2429] pt-2">
      <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">previous runs</div>
      <div className="space-y-1">
        {entries.map((e, i) => (
          <div key={i} className="flex items-baseline gap-2 text-[10px]">
            <span className={e.ok ? 'text-emerald-300' : 'text-rose-300'}>{e.ok ? '✓' : '✗'}</span>
            <span className="text-[#5a6068]">{e.at.slice(5, 16).replace('T', ' ')}</span>
            <span className="flex-1 truncate text-[#9aa1a8]">
              {e.model ?? '(no model)'}
              {e.ok && e.output_shape ? ` → ${formatShape(e.output_shape)}` : ''}
              {!e.ok && e.error ? `: ${e.error.slice(0, 60)}` : ''}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
