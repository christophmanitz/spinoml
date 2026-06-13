import { useEffect, useState } from 'react'
import { useDatasetsStore, type SmokeHistoryEntry } from './store'
import { useGraphStore } from '../canvas/GraphStore'
import { iconFor, colorFor, formatSize } from './icons'
import type {
  InspectResult, StatsResult, SmokeResult,
  TabularInspect, ImageFolderInspect, TensorInspect, ProteinInspect, MoleculeInspect, HuggingfaceInspect,
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

  if (!entry) return null

  const data = inspect?.data
  const kind = data?.kind ?? 'unknown'

  return (
    <div className="flex max-h-[60%] min-h-0 flex-col border-t border-[#1f2429] bg-[#0e1115]">
      <div className="flex items-center gap-2 border-b border-[#1f2429] px-3 py-2">
        <span className={`rounded px-1 font-mono text-[10px] ${colorFor(kind)}`}>
          {iconFor(kind)}
        </span>
        <span className="flex-1 truncate text-xs text-[#e6e8eb]">{entry.name}</span>
        <span className="text-[10px] text-[#7a8088]">{formatSize(entry.size_bytes)}</span>
        <button
          onClick={() => close(null)}
          className="rounded px-1 text-[#7a8088] hover:bg-[#1a1e22]"
          title="close"
        >
          ×
        </button>
      </div>
      <div className="flex border-b border-[#1f2429] text-[11px]">
        {(['overview', 'stats', 'smoke'] as Tab[]).map((t) => (
          <button
            key={t}
            onClick={() => {
              setTab(t)
              if (t === 'stats') void loadStats(relpath)
            }}
            className={`px-3 py-1.5 ${
              tab === t ? 'border-b border-[#6ab7ff] text-[#e6e8eb]' : 'text-[#7a8088] hover:text-[#9aa1a8]'
            }`}
          >
            {t === 'smoke' ? 'Smoke test' : t}
          </button>
        ))}
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-3 text-xs">
        {inspect?.loading && <div className="text-[#7a8088]">inspecting…</div>}
        {inspect?.error && <ErrorBox msg={inspect.error} />}
        {data && !data.ok && <InspectError data={data} />}
        {tab === 'overview' && data && data.ok && <OverviewBody data={data} relpath={relpath} />}
        {tab === 'stats' && (
          <StatsBody loading={stats?.loading} error={stats?.error} data={stats?.data ?? null} kind={kind} />
        )}
        {tab === 'smoke' && (
          <SmokeBody relpath={relpath} loading={smoke?.loading} error={smoke?.error} data={smoke?.data ?? null} runSmoke={runSmoke} />
        )}
      </div>
    </div>
  )
}

function ErrorBox({ msg }: { msg: string }) {
  return <div className="rounded bg-rose-900/20 px-2 py-1.5 text-rose-300">{msg}</div>
}

function InspectError({ data }: { data: InspectResult }) {
  if (data.ok) return null
  return (
    <div className="rounded bg-rose-900/20 px-2 py-1.5 text-rose-300">
      <div>{data.error ?? 'inspect failed'}</div>
      {data.missing_dep && (
        <div className="mt-1 text-[10px] text-rose-200/80">
          tip: <code>pip install {data.missing_dep}</code> in der mlforge-dev env
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
    default: return <div className="text-[#7a8088]">Unbekanntes Format.</div>
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
      className="rounded border border-[#2a3038] bg-[#1a1e22] px-2 py-0.5 text-[10px] text-[#9aa1a8] hover:border-[#6ab7ff] hover:text-[#e6e8eb] disabled:opacity-40"
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
          <thead className="text-[#7a8088]">
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
            <span className="text-[9px] text-[#7a8088]">{t.name.split('/')[0]}</span>
          </div>
        ))}
      </div>
      <div>
        <div className="mb-1 text-[10px] uppercase tracking-wider text-[#7a8088]">classes</div>
        <div className="space-y-0.5">
          {d.classes.map((c) => (
            <div key={c.name} className="flex items-center justify-between">
              <span className="text-[#e6e8eb]">{c.name}</span>
              <span className="text-[#7a8088]">{c.count}</span>
            </div>
          ))}
        </div>
      </div>
      <UseAsInputButton shape={[1, 3, h, w]} />
    </div>
  )
}

function TensorOverview({ d, relpath: _relpath }: { d: TensorInspect; relpath: string }) {
  return (
    <div className="space-y-2">
      {d.shape && (
        <div className="text-[#9aa1a8]">
          shape <code className="text-[#e6e8eb]">[{d.shape.join(', ')}]</code> · dtype{' '}
          <code className="text-[#e6e8eb]">{d.dtype}</code>
        </div>
      )}
      {d.mean != null && (
        <div className="text-[#7a8088]">
          mean {d.mean.toFixed(4)} · min {d.min?.toFixed(4)} · max {d.max?.toFixed(4)}
        </div>
      )}
      {d.container === 'dict' && d.keys && (
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[#7a8088]">keys</div>
          <div className="space-y-0.5">
            {d.keys.map((k) => (
              <div key={k.key} className="flex justify-between">
                <span className="text-[#e6e8eb]">{k.key}</span>
                <span className="text-[#7a8088]">
                  {k.shape ? `[${k.shape.join(', ')}]` : k.dtype}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
      {d.container === 'npz' && d.arrays && (
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[#7a8088]">arrays</div>
          {Object.entries(d.arrays).map(([k, v]) => (
            <div key={k} className="flex justify-between">
              <span className="text-[#e6e8eb]">{k}</span>
              <span className="text-[#7a8088]">[{v.join(', ')}]</span>
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
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[#7a8088]">chains</div>
          {d.chain_info.map((c) => (
            <div key={c.id} className="flex justify-between">
              <span className="text-[#e6e8eb]">{c.id || '(empty)'}</span>
              <span className="text-[#7a8088]">{c.residues} res · {c.atoms} atm</span>
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
        <div className="mb-1 text-[10px] uppercase tracking-wider text-[#7a8088]">head</div>
        <div className="space-y-0.5 font-mono text-[10px]">
          {d.head.slice(0, 8).map((s, i) => (
            <div key={i} className="truncate text-[#e6e8eb]">{s}</div>
          ))}
        </div>
      </div>
      {d.sample_info && d.sample_info.some((m) => m.valid) && (
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[#7a8088]">rdkit info</div>
          {d.sample_info.filter((m) => m.valid).slice(0, 4).map((m, i) => (
            <div key={i} className="flex justify-between text-[10px]">
              <span className="truncate font-mono text-[#e6e8eb]">{m.canonical ?? m.smiles}</span>
              <span className="ml-2 text-[#7a8088]">{m.atoms}a · {m.bonds}b · MW {m.mw}</span>
            </div>
          ))}
        </div>
      )}
      <UseAsInputButton shape={[1, 64]} label="Use [1, 64] byte-encoded as input" />
    </div>
  )
}

function HfOverview({ d }: { d: HuggingfaceInspect }) {
  return (
    <div className="space-y-2">
      <div className="text-[#9aa1a8]"><code className="text-[#e6e8eb]">{d.name}</code></div>
      {d.description && <div className="text-[#7a8088]">{d.description}</div>}
      <div>
        <div className="mb-1 text-[10px] uppercase tracking-wider text-[#7a8088]">splits</div>
        {Object.entries(d.splits).map(([k, v]) => (
          <div key={k} className="flex justify-between">
            <span className="text-[#e6e8eb]">{k}</span>
            <span className="text-[#7a8088]">{v.num_examples?.toLocaleString() ?? '?'}</span>
          </div>
        ))}
      </div>
      <div>
        <div className="mb-1 text-[10px] uppercase tracking-wider text-[#7a8088]">features</div>
        {Object.entries(d.features).map(([k, v]) => (
          <div key={k} className="flex justify-between text-[10px]">
            <span className="text-[#e6e8eb]">{k}</span>
            <span className="ml-2 truncate text-[#7a8088]">{v}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function StatsBody({ loading, error, data, kind }: { loading?: boolean; error?: string | null; data: StatsResult | null; kind: string }) {
  if (loading) return <div className="text-[#7a8088]">computing stats…</div>
  if (error) return <ErrorBox msg={error} />
  if (!data) return <div className="text-[#7a8088]">no stats yet</div>
  if (!data.ok) return <ErrorBox msg={('error' in data && data.error) || 'stats failed'} />
  if (data.kind === 'tabular') return <TabularStatsView d={data as TabularStats} />
  if (data.kind === 'image_folder') return <ImageStatsView d={data as ImageFolderStats} />
  if (data.kind === 'tensor') return <TensorStatsView d={data as TensorStats} />
  if (data.kind === 'molecule') return <MoleculeStatsView d={data as MoleculeStats} />
  return <div className="text-[#7a8088]">Stats für {kind} sind aktuell nur in der Übersicht.</div>
}

function Bar({ frac, max = 1 }: { frac: number; max?: number }) {
  const pct = Math.max(0, Math.min(100, (frac / max) * 100))
  return (
    <div className="h-1.5 w-full overflow-hidden rounded bg-[#1a1e22]">
      <div className="h-full bg-[#6ab7ff]/70" style={{ width: `${pct}%` }} />
    </div>
  )
}

function Hist({ counts, edges, color = '#6ab7ff' }: { counts: number[]; edges: number[]; color?: string }) {
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
              <span className="text-[10px] text-[#7a8088]">{s.dtype} · {s.unique} unique · {s.missing} missing</span>
            </div>
            {s.mean != null && (
              <div className="mt-0.5 text-[10px] text-[#7a8088]">
                μ {s.mean.toFixed(3)} · σ {s.std?.toFixed(3)} · [{s.min?.toFixed(2)}, {s.max?.toFixed(2)}]
              </div>
            )}
            {s.hist && <div className="mt-1"><Hist counts={s.hist.counts} edges={s.hist.edges} /></div>}
          </div>
        ))}
      </div>
      {d.corr && d.corr_cols.length > 0 && (
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[#7a8088]">correlations</div>
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
            <th key={c} className="rotate-[-30deg] px-1 text-[#7a8088]">{c}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {corr.map((row, i) => (
          <tr key={i}>
            <td className="pr-1 text-right text-[#7a8088]">{cols[i]}</td>
            {row.map((v, j) => {
              const a = Math.abs(v)
              const bg = v >= 0
                ? `rgba(106, 183, 255, ${a * 0.7})`
                : `rgba(248, 113, 113, ${a * 0.7})`
              return (
                <td key={j} className="border border-[#1f2429] text-center" style={{ background: bg, width: 28, height: 18 }}>
                  <span className={a > 0.5 ? 'text-[#e6e8eb]' : 'text-[#7a8088]'}>{v.toFixed(2)}</span>
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
        <div className="mb-1 text-[10px] uppercase tracking-wider text-[#7a8088]">class distribution</div>
        <div className="space-y-1">
          {d.classes.map((c) => (
            <div key={c.name}>
              <div className="flex justify-between text-[10px]">
                <span className="text-[#e6e8eb]">{c.name}</span>
                <span className="text-[#7a8088]">{c.count}</span>
              </div>
              <Bar frac={c.count} max={max} />
            </div>
          ))}
        </div>
      </div>
      <div>
        <div className="mb-1 text-[10px] uppercase tracking-wider text-[#7a8088]">image sizes (sample of {d.n_samples_for_size})</div>
        <div className="space-y-0.5 text-[10px]">
          {d.size_hist.map((s) => (
            <div key={s.size} className="flex justify-between">
              <span className="text-[#e6e8eb]">{s.size}</span>
              <span className="text-[#7a8088]">{s.count}</span>
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
      <div className="text-[10px] text-[#7a8088]">
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
      {d.note && <div className="text-[10px] text-[#7a8088]">{d.note}</div>}
      {d.mw_mean != null && <div className="text-[#9aa1a8]">MW μ {d.mw_mean} · atoms μ {d.atom_mean}</div>}
      {d.mw_hist && (
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[#7a8088]">molecular weight</div>
          <Hist counts={d.mw_hist.counts} edges={d.mw_hist.edges} color="#c084fc" />
        </div>
      )}
      {d.atom_hist && (
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[#7a8088]">atom count</div>
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
  const loadHistory = useDatasetsStore((s) => s.loadHistory)

  useEffect(() => { void loadHistory() }, [loadHistory])

  const datasetHistory = history.filter((h) => h.dataset === relpath).slice(0, 12)
  return (
    <div className="space-y-2">
      <div className="text-[#9aa1a8]">
        Schickt einen Sample-Batch aus dem Datensatz durch das aktuelle Modell.
      </div>
      {inputShape && (
        <div className="text-[10px] text-[#7a8088]">
          input shape vom Graph: <code className="text-[#e6e8eb]">[{inputShape.join(', ')}]</code>
        </div>
      )}
      <button
        onClick={() => void runSmoke(relpath, inputShape ?? undefined)}
        disabled={loading || !hasModel}
        className="rounded border border-[#2a3038] bg-[#1a1e22] px-2 py-1 text-[11px] text-[#e6e8eb] hover:border-[#6ab7ff] disabled:opacity-40"
      >
        {loading ? 'running…' : 'Run smoke test'}
      </button>
      {!hasModel && <div className="text-[10px] text-[#7a8088]">Zieh erst ein paar Layer auf den Canvas.</div>}
      {error && <ErrorBox msg={error} />}
      {data && data.ok && (
        <div className="rounded border border-emerald-900/40 bg-emerald-900/10 p-2">
          <div className="text-emerald-300">✓ forward pass succeeded</div>
          <div className="mt-1 space-y-0.5 text-[10px] text-[#9aa1a8]">
            <div>input: <code className="text-[#e6e8eb]">[{data.input_shape.join(', ')}]</code></div>
            <div>output: <code className="text-[#e6e8eb]">{data.output_shape ? `[${data.output_shape.join(', ')}]` : '(non-tensor)'}</code></div>
            <div>params: {data.n_params.toLocaleString()}</div>
            <div>timings: sample {data.timings_ms.sample.toFixed(1)}ms · forward {data.timings_ms.forward.toFixed(1)}ms</div>
            {data.sample_note && <div className="text-[#7a8088]">{data.sample_note}</div>}
          </div>
        </div>
      )}
      {data && !data.ok && (
        <div className="rounded border border-rose-900/40 bg-rose-900/10 p-2">
          <div className="text-rose-300">✗ {data.stage}: {data.error}</div>
          {data.input_shape && (
            <div className="mt-1 text-[10px] text-[#7a8088]">
              input shape: <code>[{data.input_shape.join(', ')}]</code>
            </div>
          )}
          {data.trace && (
            <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap text-[10px] text-rose-200/70">{data.trace}</pre>
          )}
        </div>
      )}
      {datasetHistory.length > 0 && <SmokeHistory entries={datasetHistory} />}
    </div>
  )
}

function SmokeHistory({ entries }: { entries: SmokeHistoryEntry[] }) {
  return (
    <div className="mt-3 border-t border-[#1f2429] pt-2">
      <div className="mb-1 text-[10px] uppercase tracking-wider text-[#7a8088]">previous runs</div>
      <div className="space-y-1">
        {entries.map((e, i) => (
          <div key={i} className="flex items-baseline gap-2 text-[10px]">
            <span className={e.ok ? 'text-emerald-300' : 'text-rose-300'}>{e.ok ? '✓' : '✗'}</span>
            <span className="text-[#5a6068]">{e.at.slice(5, 16).replace('T', ' ')}</span>
            <span className="flex-1 truncate text-[#9aa1a8]">
              {e.model ?? '(no model)'}
              {e.ok && e.output_shape ? ` → [${e.output_shape.join(',')}]` : ''}
              {!e.ok && e.error ? `: ${e.error.slice(0, 60)}` : ''}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
