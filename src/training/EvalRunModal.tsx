import { useEffect, useMemo, useState } from 'react'

import { datasets as datasetsBackend, fs } from '../connections/backend'
import { useDatasetsStore } from '../datasets/store'
import { useConnectionsStore } from '../connections/store'
import { useViewModeStore } from './graph/viewMode'
import { useDataGraphStore } from '../data/graph/store'
import { useTrainingStore } from './store'
import { training } from './backend'
import {
  modelContract, suggestAdapter, adaptationFromMap, seedBranchCfgs, branchCfgsToManifest,
  type ModelContract, type Adaptation, type DirResources, type BranchCfg,
} from './adapter'
import {
  type RunConfig, type Head, type RunBackend, type SlurmConfig, type RemoteTrainingCapabilities,
  defaultSlurmConfig,
} from './types'
import { parseRunConfig } from './parseRunConfig'
import type { InspectResult } from '../datasets/types'

const SELECT = 'w-full rounded border border-[#1f2429] bg-[#14181c] px-2 py-1 text-[12px] text-[#e6e8eb] focus:border-[var(--accent)] focus:outline-none'
const INPUT = 'rounded border border-[#1f2429] bg-[#14181c] px-1.5 py-0.5 text-[11px] text-[#e6e8eb] focus:border-[var(--accent)] focus:outline-none'

function roleLabel(kind: string): string {
  return kind === 'feature' ? 'Feature' : kind === 'target' ? 'Ziel' : kind === 'smiles' ? 'SMILES' : kind === 'branch' ? 'Branch' : kind
}
function sourceHeads(cfg: RunConfig): Head[] {
  const h = cfg.training?.heads
  if (h && h.length) return h
  return [{ output: '', target: cfg.dataset?.target_column ?? '', loss: cfg.training?.loss?.kind ?? 'CrossEntropyLoss', weight: 1 }]
}
function dirResources(inspect: InspectResult | undefined): DirResources {
  if (!inspect || !inspect.ok || inspect.kind !== 'tabular') return { table: 'pairs.csv', columns: [], files: [], subdirs: [] }
  const sample: Record<string, string> = {}
  inspect.columns.forEach((c, i) => { sample[c] = inspect.head?.[0]?.[i] ?? '' })
  return {
    table: inspect.table ?? '', columns: inspect.columns,
    files: inspect.bundle?.files ?? [], subdirs: inspect.bundle?.subdirs.map((s) => s.name) ?? [], sample,
  }
}

const SOURCE_LABELS: Record<BranchCfg['source'], string> = {
  column: 'Spalte (tokenisieren)', molecule: 'SMILES → Graph (inline)', folder: 'Ordner mit .pt-Dateien', lookup: 'Join (Seiten-CSV)',
}

/** External-validation dialog. A manifest dual-encoder gets a per-branch editor
 *  (each branch → a column / inline molecule graph / a folder of .pt / a CSV join),
 *  plus per-output selection and local/SLURM execution. Tabular models get a plain
 *  column mapping. */
export default function EvalRunModal() {
  const sourceRunId = useTrainingStore((s) => s.evalSourceId)
  if (!sourceRunId) return null
  return <EvalRunModalInner sourceRunId={sourceRunId} />
}

function EvalRunModalInner({ sourceRunId }: { sourceRunId: string }) {
  const close = useTrainingStore((s) => s.closeEvalRun)
  const startEvalRun = useTrainingStore((s) => s.startEvalRun)
  const inspectDataset = useDatasetsStore((s) => s.inspect)
  const remoteConn = useConnectionsStore((s) => s.saved.find((c) => c.id === s.currentId && c.kind === 'remote-ssh')) ?? null
  const updateRemote = useConnectionsStore((s) => s.updateRemote)

  const [srcCfg, setSrcCfg] = useState<RunConfig | null>(null)
  const [srcManifest, setSrcManifest] = useState<Record<string, unknown> | null>(null)
  const [contract, setContract] = useState<ModelContract | null>(null)
  const [heads, setHeads] = useState<Head[]>([])
  const [selected, setSelected] = useState<Set<string>>(new Set())
  const [dsList, setDsList] = useState<{ relpath: string; name: string }[]>([])
  const [datasetRel, setDatasetRel] = useState('')
  const [colMap, setColMap] = useState<Record<string, string>>({})
  const [mode, setMode] = useState<Adaptation['spec']['mode']>('auto')
  // structured manifest editor state
  const [branches, setBranches] = useState<Record<string, BranchCfg>>({})
  const [table, setTable] = useState('')
  const [targetCol, setTargetCol] = useState('')
  const [label, setLabel] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [caps, setCaps] = useState<RemoteTrainingCapabilities | null>(null)
  const [capsError, setCapsError] = useState<string | null>(null)
  const [backendKind, setBackendKind] = useState<'local' | 'slurm'>('local')
  const [slurm, setSlurm] = useState<SlurmConfig>(remoteConn?.slurm ?? defaultSlurmConfig())
  const [srcWarn, setSrcWarn] = useState<string | null>(null)

  const externalInspect = useDatasetsStore((s) => (datasetRel ? s.inspects[datasetRel]?.data : undefined))
  const res = useMemo(() => dirResources(externalInspect ?? undefined), [externalInspect])
  const extColumns = res.columns
  const externalIsManifest = !!(externalInspect?.ok && externalInspect.kind === 'manifest')
  const manifestMode = !!srcManifest && !externalIsManifest

  useEffect(() => {
    void (async () => {
      try {
        const cfg = parseRunConfig(JSON.parse(await training.readFile(sourceRunId, 'run.json')))
        setSrcCfg(cfg)
        const hs = sourceHeads(cfg)
        setHeads(hs); setSelected(new Set(hs.map((h) => h.output)))
        setLabel(`${cfg.run_label || sourceRunId} extern`)
        let srcInspect: InspectResult | null = null
        try {
          await inspectDataset(cfg.dataset.relpath)
          srcInspect = useDatasetsStore.getState().inspects[cfg.dataset.relpath]?.data ?? null
          setSrcWarn(null)
        } catch (e) {
          // The source dataset is optional enrichment; if it can't be read the
          // contract is derived from run.json alone, but the user must be told.
          setSrcWarn(`Quell-Datensatz ${cfg.dataset.relpath} nicht ladbar: ${e instanceof Error ? e.message : String(e)}`)
        }
        setContract(modelContract(cfg, srcInspect))
        if (cfg.dataset.kind === 'manifest') {
          const raw = await fs.read(cfg.dataset.relpath)
          try { setSrcManifest(raw ? JSON.parse(raw) : null) }
          catch (e) {
            throw new Error(`Manifest ${cfg.dataset.relpath} ist beschädigt: ${e instanceof Error ? e.message : String(e)}`, { cause: e })
          }
        }
        await useDatasetsStore.getState().refresh()
        setDsList(useDatasetsStore.getState().entries.filter((d) => !d.name.startsWith('.'))
          .map((d) => ({ relpath: d.relpath, name: d.is_dir ? `${d.name}/` : d.name })))
      } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
    })()
  }, [sourceRunId, inspectDataset])

  useEffect(() => {
    if (!remoteConn) return
    let cancelled = false
    void training.capabilities()
      .then((c) => { if (!cancelled) { setCaps(c); setCapsError(null); setBackendKind(c.has_slurm ? 'slurm' : 'local') } })
      .catch((e) => {
        // A failed remote probe is NOT "no SLURM / local host" — keep caps null
        // and surface an explicit unknown state so the backend isn't guessed.
        if (!cancelled) { setCaps(null); setCapsError(e instanceof Error ? e.message : String(e)) }
      })
    return () => { cancelled = true }
  }, [remoteConn])

  // Seed the editor when a dataset is picked (manifest model) or the column map (tabular).
  useEffect(() => {
    if (!datasetRel) return
    if (srcManifest && !externalIsManifest) {
      const seeded = seedBranchCfgs(srcManifest, datasetRel, res)
      // Seeds the adapter form from the selected dataset/manifest. The seeds are
      // plain data derived from props; they cannot be computed during render
      // without re-running the (JSON-shaped) seed on every keystroke.
      // eslint-disable-next-line react-hooks/set-state-in-effect -- dataset/manifest selection seeds the editable adapter form
      setBranches(seeded.branches); setTargetCol(seeded.targetColumn); setTable(seeded.table)
    } else if (contract && externalInspect?.ok) {
      const a = suggestAdapter(contract, externalInspect); setColMap(a.spec.column_map); setMode('hybrid')
    }
  }, [datasetRel, srcManifest, externalIsManifest, res, contract, externalInspect])

  const onPickDataset = async (rel: string) => { setDatasetRel(rel); setColMap({}); setBranches({}); if (rel) await inspectDataset(rel) }
  const setBranch = (name: string, patch: Partial<BranchCfg>) => setBranches((b) => ({ ...b, [name]: { ...b[name], ...patch } }))

  const adaptation: Adaptation | null = useMemo(() => (contract ? adaptationFromMap(contract, colMap, mode) : null), [contract, colMap, mode])
  const selectedHeads = heads.filter((h) => selected.has(h.output))
  const canSubmit = !!datasetRel && selectedHeads.length > 0 && !submitting && (
    manifestMode ? (Object.keys(branches).length > 0 && !!targetCol)
    : externalIsManifest ? true
    : (!!adaptation && adaptation.spec.unmatched.length === 0 && !!adaptation.target_column)
  )

  const submit = async () => {
    if (!srcCfg) return
    setSubmitting(true); setError(null)
    try {
      const backend: RunBackend = backendKind === 'slurm' ? { kind: 'slurm', slurm } : { kind: 'local' }
      if (backendKind === 'slurm' && remoteConn) updateRemote(remoteConn.id, { slurm })
      const evalHeads = selectedHeads.length === heads.length ? undefined : selectedHeads
      const targetType = (srcManifest?.target as { type?: string } | undefined)?.type ?? 'classification'

      let datasetRelpath = datasetRel
      let featureColumns: string[] | null = null
      let targetColumn = ''
      let adapterSpec: Adaptation['spec'] = { column_map: {}, unmatched: [], mode: 'pipeline', adapted_from: datasetRel }

      if (manifestMode) {
        const dirName = datasetRel.replace(/\/+$/, '').split('/').pop() ?? 'external'
        datasetRelpath = `datasets/${dirName}.eval.manifest`
        const man = branchCfgsToManifest(table, branches, targetCol, targetType)
        await fs.write(datasetRelpath, JSON.stringify(man, null, 2))
        await inspectDataset(datasetRelpath) // prime ESPF cache the run needs
      } else if (externalIsManifest) {
        // external dataset is itself a manifest → validate directly
      } else if (adaptation) {
        featureColumns = adaptation.feature_columns; targetColumn = adaptation.target_column; adapterSpec = adaptation.spec
      }
      const abspath = await datasetsBackend.abspath(datasetRelpath)
      await startEvalRun({ label, sourceRunId, datasetRelpath, datasetAbspath: abspath, featureColumns, targetColumn, heads: evalHeads, adapter: adapterSpec, backend })
      close()
    } catch (e) { setError(e instanceof Error ? e.message : String(e)); setSubmitting(false) }
  }

  const escalate = () => {
    if (!datasetRel) return
    useViewModeStore.getState().setMode('data')
    const ds = useDataGraphStore.getState()
    const src = ds.addNode('TableSource', { x: 0, y: 0 }, { params: { dataset: datasetRel } })
    const sink = ds.addNode('WriteDataset', { x: 0, y: 160 }, { params: { out_path: 'datasets/adapted.csv', format: 'csv' } })
    ds.connectNodes(src, sink); ds.autoLayout(); close()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6" onClick={close}>
      <div className="flex max-h-[92vh] w-full max-w-2xl flex-col overflow-hidden rounded-lg border border-[#1f2429] bg-[#0e1216] shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-2 border-b border-[#1f2429] px-4 py-2.5">
          <span className="rounded bg-[var(--accent-sel)] px-1.5 py-0.5 text-[10px] font-semibold text-[var(--accent)]">VALIDIERUNG</span>
          <span className="text-[13px] text-[#e6e8eb]">Externer Datensatz auf «{sourceRunId}»</span>
          <button onClick={close} className="ml-auto rounded px-2 text-[#6f767e] hover:text-[#e6e8eb]">×</button>
        </div>

        {/* shared datalists for the per-branch inputs */}
        <datalist id="ev-cols">{extColumns.map((c) => <option key={c} value={c} />)}</datalist>
        <datalist id="ev-files">{res.files.map((f) => <option key={f} value={f} />)}</datalist>
        <datalist id="ev-dirs">{res.subdirs.map((d) => <option key={d} value={d} />)}</datalist>

        <div className="flex-1 space-y-3 overflow-y-auto p-4 text-[12px]">
          {error && <div className="rounded border border-rose-900/60 bg-rose-950/30 p-2 text-[11px] text-rose-300">{error}</div>}
          {capsError && <div className="rounded border border-amber-900/60 bg-amber-950/30 p-2 text-[11px] text-amber-200">Fähigkeiten des Remote-Hosts unbekannt (SLURM/GPU nicht geprüft): {capsError}. Backend wird nicht automatisch gewählt.</div>}
          {srcWarn && <div className="rounded border border-amber-900/60 bg-amber-950/30 p-2 text-[11px] text-amber-200">{srcWarn}</div>}

          <label className="block"><span className="mb-1 block text-[11px] text-[#6f767e]">Label</span>
            <input value={label} onChange={(e) => setLabel(e.target.value)} className={SELECT} /></label>

          <label className="block"><span className="mb-1 block text-[11px] text-[#6f767e]">Externer Datensatz (Benchmark/Validierung)</span>
            <select value={datasetRel} onChange={(e) => void onPickDataset(e.target.value)} className={SELECT}>
              <option value="">— wählen —</option>
              {dsList.map((d) => <option key={d.relpath} value={d.relpath}>{d.name}</option>)}
            </select></label>

          {heads.length > 1 && (
            <div className="space-y-1 rounded border border-[#1f2429] bg-[#0b0e11] p-2">
              <span className="text-[11px] font-semibold text-[#9aa1a8]">Outputs validieren</span>
              {heads.map((h) => (
                <label key={h.output} className="flex cursor-pointer items-center gap-2 text-[11px]">
                  <input type="checkbox" checked={selected.has(h.output)} onChange={() => setSelected((s) => { const n = new Set(s); if (n.has(h.output)) n.delete(h.output); else n.add(h.output); return n })} className="h-3 w-3 accent-[var(--accent)]" />
                  <span className="rounded bg-[var(--accent-sel)] px-1.5 py-0.5 text-[10px] text-[var(--accent)]">{h.output || 'out'}</span>
                  <span className="text-[#6f767e]">→ {h.target}</span>
                  <span className="ml-auto text-[10px] text-[#5a6068]">{h.loss}</span>
                </label>
              ))}
              <div className="text-[10px] text-[#5a6068]">Deaktiviere Outputs ohne Ziel-Spalte (z.B. Affinität).</div>
            </div>
          )}

          {/* Per-branch editor (manifest dual-encoder) */}
          {manifestMode && datasetRel && (
            <div className="space-y-2 rounded border border-[#1f2429] bg-[#0b0e11] p-2">
              <div className="flex items-center gap-2">
                <span className="text-[11px] font-semibold text-[#9aa1a8]">Tabelle</span>
                <input value={table} onChange={(e) => setTable(e.target.value)} list="ev-files" placeholder="bindingdb_kd/pairs.csv" className={`${INPUT} flex-1`} />
              </div>
              {Object.keys(branches).length === 0 && <div className="text-[10px] text-amber-400/80">Quell-Manifest des Modells nicht geladen — Branches manuell prüfen.</div>}
              {Object.entries(branches).map(([name, b]) => (
                <div key={name} className="space-y-1 rounded border border-[#1f2429] px-2 py-1.5">
                  <div className="flex items-center gap-2">
                    <span className="w-20 shrink-0 font-mono text-[11px] text-[#e6e8eb]">{name}</span>
                    <select value={b.source} onChange={(e) => setBranch(name, { source: e.target.value as BranchCfg['source'] })} className={INPUT}>
                      {(Object.keys(SOURCE_LABELS) as BranchCfg['source'][]).map((s) => <option key={s} value={s}>{SOURCE_LABELS[s]}</option>)}
                    </select>
                  </div>
                  <div className="flex flex-wrap items-center gap-1.5 pl-2">
                    {(b.source === 'column' || b.source === 'molecule') && (
                      <><span className="text-[10px] text-[#6f767e]">Spalte</span>
                        <input value={b.column} onChange={(e) => setBranch(name, { column: e.target.value })} list="ev-cols" placeholder="substrate_smiles" className={`${INPUT} w-40`} /></>
                    )}
                    {b.source === 'column' && (
                      <select value={b.codebook} onChange={(e) => setBranch(name, { codebook: e.target.value })} className={INPUT}>
                        <option value="drug">ESPF drug</option><option value="protein">ESPF protein</option>
                      </select>
                    )}
                    {b.source === 'folder' && (
                      <><span className="text-[10px] text-[#6f767e]">Schlüssel-Spalte</span>
                        <input value={b.column} onChange={(e) => setBranch(name, { column: e.target.value })} list="ev-cols" placeholder="uniprot" className={`${INPUT} w-28`} />
                        <span className="text-[10px] text-[#6f767e]">Ordner</span>
                        <input value={b.dir} onChange={(e) => setBranch(name, { dir: e.target.value })} list="ev-dirs" placeholder="bindingdb_kd/esm2_embeddings" className={`${INPUT} w-44`} />
                        <select value={b.match} onChange={(e) => setBranch(name, { match: e.target.value })} className={INPUT}><option value="exact">exact</option><option value="contains">contains</option></select>
                        <input value={b.ext} onChange={(e) => setBranch(name, { ext: e.target.value })} placeholder=".pt" className={`${INPUT} w-12`} /></>
                    )}
                    {b.source === 'lookup' && (
                      <><span className="text-[10px] text-[#6f767e]">Schlüssel</span>
                        <input value={b.column} onChange={(e) => setBranch(name, { column: e.target.value })} list="ev-cols" placeholder="uniprot" className={`${INPUT} w-24`} />
                        <span className="text-[10px] text-[#6f767e]">CSV</span>
                        <input value={b.lookupFile} onChange={(e) => setBranch(name, { lookupFile: e.target.value })} list="ev-files" placeholder="bindingdb_kd/sequences.csv" className={`${INPUT} w-44`} />
                        <span className="text-[10px] text-[#6f767e]">Wert-Spalte</span>
                        <input value={b.lookupValue} onChange={(e) => setBranch(name, { lookupValue: e.target.value })} placeholder="sequence" className={`${INPUT} w-24`} />
                        <select value={b.codebook} onChange={(e) => setBranch(name, { codebook: e.target.value })} className={INPUT}><option value="drug">ESPF drug</option><option value="protein">ESPF protein</option></select></>
                    )}
                  </div>
                </div>
              ))}
              <div className="flex items-center gap-2">
                <span className="text-[11px] font-semibold text-[#9aa1a8]">Ziel-Spalte</span>
                <input value={targetCol} onChange={(e) => setTargetCol(e.target.value)} list="ev-cols" placeholder="label" className={`${INPUT} w-40`} />
              </div>
              <div className="text-[10px] text-[#5a6068]">„SMILES → Graph (inline)" baut Liganden-Graphen aus SMILES (kein Precompute). „Ordner" nutzt vorberechnete .pt; fehlen sie, müssen sie erst erzeugt werden (Daten-Canvas).</div>
            </div>
          )}
          {externalIsManifest && <div className="rounded border border-[#1f2429] bg-[#0b0e11] p-2 text-[10px] text-[#9aa1a8]">Externer Datensatz ist bereits ein .manifest — wird direkt validiert.</div>}

          {!manifestMode && !externalIsManifest && contract && datasetRel && (
            <div className="space-y-1.5 rounded border border-[#1f2429] bg-[#0b0e11] p-2">
              <span className="text-[11px] font-semibold text-[#9aa1a8]">Zuordnung Modell-Rolle → externe Spalte</span>
              {contract.roles.map((r) => (
                <div key={r.key} className="flex items-center gap-2">
                  <span className="w-16 shrink-0 text-[10px] text-[#6f767e]">{roleLabel(r.kind)}</span>
                  <span className="w-28 shrink-0 truncate font-mono text-[11px] text-[#e6e8eb]" title={r.key}>{r.key}</span><span className="text-[#5a6068]">→</span>
                  <select value={colMap[r.key] ?? ''} onChange={(e) => { setColMap({ ...colMap, [r.key]: e.target.value }); setMode('manual') }} className={`${SELECT} ${!colMap[r.key] ? 'border-amber-700/70' : ''}`}>
                    <option value="">— (nicht zugeordnet) —</option>{extColumns.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
              ))}
            </div>
          )}

          {remoteConn && caps?.has_slurm && (
            <div className="space-y-1.5 rounded border border-[#1f2429] bg-[#0b0e11] p-2">
              <span className="text-[11px] font-semibold text-[#9aa1a8]">Ausführung</span>
              <div className="flex gap-2 text-[11px]">
                {(['slurm', 'local'] as const).map((k) => (
                  <button key={k} onClick={() => setBackendKind(k)} className={`rounded border px-2.5 py-1 ${backendKind === k ? 'border-[var(--accent)] bg-[var(--accent-sel)]/40 text-[var(--accent)]' : 'border-[#1f2429] text-[#9aa1a8]'}`}>{k === 'slurm' ? 'SLURM' : 'Login-Node'}</button>
                ))}
              </div>
              {backendKind === 'slurm' && (
                <div className="flex items-center gap-2"><span className="text-[10px] text-[#6f767e]">Partition</span>
                  <input value={slurm.partition} onChange={(e) => setSlurm({ ...slurm, partition: e.target.value })} list="ev-parts" placeholder={caps.partitions[0] ?? 'partition'} className={INPUT} />
                  <datalist id="ev-parts">{caps.partitions.map((p) => <option key={p} value={p} />)}</datalist></div>
              )}
            </div>
          )}
        </div>

        <div className="flex items-center gap-2 border-t border-[#1f2429] px-4 py-2.5">
          {datasetRel && <button onClick={escalate} className="rounded border border-[#1f2429] px-2.5 py-1 text-[11px] text-[#9aa1a8] hover:border-[#3a4148]">Im Daten-Canvas anpassen…</button>}
          <button onClick={close} className="ml-auto rounded px-2.5 py-1 text-[11px] text-[#6f767e]">Abbrechen</button>
          <button onClick={() => void submit()} disabled={!canSubmit} className="rounded bg-[var(--accent-sel)] px-3 py-1 text-[11px] text-[var(--accent)] disabled:opacity-40">
            {submitting ? 'Starte…' : backendKind === 'slurm' ? 'Validieren (SLURM)' : 'Validieren'}
          </button>
        </div>
      </div>
    </div>
  )
}
