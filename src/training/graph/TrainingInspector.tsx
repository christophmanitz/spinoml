import { useEffect, useMemo, useState } from 'react'

import { fs } from '../../connections/backend'
import { isTauri } from '../../workspace/tauri-fs'
import { useDatasetsStore } from '../../datasets/store'
import { compileTrainingGraph } from '../../codegen/trainingGenerator'
import { useTrainingStore } from '../store'
import { useTrainingGraphStore } from './store'
import { TRAINING_NODES, type TrainingFieldSpec } from './registry'

const INPUT = 'w-full rounded border border-[#1f2429] bg-[#0b0e11] px-2 py-1 text-[12px] text-[#e6e8eb] focus:border-[#6ab7ff] focus:outline-none'

export default function TrainingInspector() {
  const node = useTrainingGraphStore((s) => s.nodes.find((n) => n.id === s.selectedNodeId) ?? null)
  const updateNodeParams = useTrainingGraphStore((s) => s.updateNodeParams)
  const deleteNode = useTrainingGraphStore((s) => s.deleteNode)
  // Derive compile feedback via useMemo on the (stable) node/edge arrays.
  // NOT inside the selector — a selector returning a fresh object every call
  // trips useSyncExternalStore into an infinite re-render loop.
  const nodes = useTrainingGraphStore((s) => s.nodes)
  const edges = useTrainingGraphStore((s) => s.edges)
  const compile = useMemo(
    () => compileTrainingGraph({
      nodes: nodes.map((n) => ({ id: n.id, trainingType: n.data.trainingType, params: n.data.params })),
      edges: edges.map((e) => ({ source: e.source, target: e.target })),
    }),
    [nodes, edges],
  )

  if (!node) {
    return (
      <div className="flex h-full flex-col">
        <CompilePanel compile={compile} />
        <div className="p-3 text-[11px] text-[#7a8088]">
          Knoten auswählen, um Parameter zu bearbeiten.
        </div>
      </div>
    )
  }

  const spec = TRAINING_NODES[node.data.trainingType]
  if (!spec) return <div className="p-3 text-[11px] text-rose-300">Unbekannter Knoten: {node.data.trainingType}</div>

  return (
    <div className="flex h-full flex-col overflow-auto">
      <CompilePanel compile={compile} />
      <div className="flex items-center gap-2 border-b border-[#1f2429] px-3 py-2">
        <span className="flex-1 text-[12px] font-medium text-[#e6e8eb]">{node.data.trainingType}</span>
        <button
          onClick={() => deleteNode(node.id)}
          className="rounded px-1.5 py-0.5 text-[11px] text-[#7a8088] hover:bg-[#1a1e22] hover:text-[#ff7a85]"
        >
          löschen
        </button>
      </div>
      <div className="space-y-3 p-3">
        {spec.fields.map((field) => (
          <label key={field.name} className="block">
            <span className="mb-1 block text-[11px] text-[#7a8088]">{field.name}</span>
            <FieldInput
              field={field}
              value={node.data.params[field.name]}
              boundDataset={String(node.data.params.dataset ?? '')}
              onChange={(v) => updateNodeParams(node.id, { [field.name]: v })}
            />
          </label>
        ))}
      </div>
    </div>
  )
}

function CompilePanel({ compile }: { compile: ReturnType<typeof compileTrainingGraph> }) {
  const openNewRun = useTrainingStore((s) => s.openNewRun)
  const [error, setError] = useState<string | null>(null)

  // Compile the graph and hand the plan to the New-Run dialog pre-filled, so the
  // graph-driven launch still gets the dialog's backend/SLURM/sweep/resume knobs
  // (instead of a lesser, instant local-only start).
  function launch() {
    if (!compile.plan) return
    setError(null)
    const plan = compile.plan
    openNewRun({
      label: plan.modelRelpath.split('/').pop()!.replace(/\.spinoml$/i, ''),
      modelRelpath: plan.modelRelpath,
      datasetRelpath: plan.datasetRelpath,
      targetColumn: plan.target,
      featureColumns: plan.features,
      training: plan.training,
    })
  }

  return (
    <div className="space-y-1.5 border-b border-[#1f2429] px-3 py-2 text-[11px]">
      {compile.ok ? (
        <span className="text-[#5fd39a]">✓ Graph ist startklar</span>
      ) : (
        <div className="space-y-0.5">
          <span className="text-[#e6c34a]">Graph noch nicht startklar:</span>
          <ul className="ml-3 list-disc text-[#9aa1a8]">
            {compile.issues.map((m, i) => <li key={i}>{m}</li>)}
          </ul>
        </div>
      )}
      {compile.warnings.length > 0 && (
        <div className="space-y-0.5">
          <span className="text-[#c98b3a]">⚠ Verdrahtung prüfen:</span>
          <ul className="ml-3 list-disc text-[#9aa1a8]">
            {compile.warnings.map((m, i) => <li key={i}>{m}</li>)}
          </ul>
        </div>
      )}
      <button
        onClick={launch}
        disabled={!compile.ok}
        title="Aus diesem Graph einen Run vorbereiten — öffnet den Dialog mit Backend/SLURM, Sweep und Resume"
        className="w-full rounded bg-[#13344f] px-2 py-1 text-[11px] text-[#6ab7ff] hover:bg-[#184466] disabled:cursor-not-allowed disabled:opacity-40"
      >
        ▶ Run vorbereiten…
      </button>
      {error && <div className="text-[#ff7a85]">{error}</div>}
    </div>
  )
}

function FieldInput({
  field, value, boundDataset, onChange,
}: {
  field: TrainingFieldSpec
  value: unknown
  boundDataset: string
  onChange: (v: unknown) => void
}) {
  switch (field.type) {
    case 'int':
    case 'float':
      return (
        <input
          type="number"
          step={field.type === 'float' ? (field.step ?? 0.001) : (field.step ?? 1)}
          value={Number(value ?? field.default)}
          onChange={(e) => onChange(Number(e.target.value))}
          className={INPUT}
        />
      )
    case 'bool':
      return (
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={(e) => onChange(e.target.checked)}
          className="h-4 w-4 accent-[#6ab7ff]"
        />
      )
    case 'select':
      return (
        <select value={String(value ?? field.default)} onChange={(e) => onChange(e.target.value)} className={INPUT}>
          {field.options.map((o) => <option key={o} value={o}>{o}</option>)}
        </select>
      )
    case 'dataset-ref':
      return <DatasetRef value={String(value ?? '')} onChange={onChange} />
    case 'model-ref':
      return <ModelRef value={String(value ?? '')} onChange={onChange} />
    case 'column-single':
      return <ColumnSingle value={String(value ?? '')} dataset={boundDataset} onChange={onChange} />
    case 'columns-multi':
      return <ColumnsMulti value={(value as string[]) ?? []} dataset={boundDataset} onChange={onChange} />
  }
}

function DatasetRef({ value, onChange }: { value: string; onChange: (v: unknown) => void }) {
  const entries = useDatasetsStore((s) => s.entries)
  const refresh = useDatasetsStore((s) => s.refresh)
  const inspect = useDatasetsStore((s) => s.inspect)
  useEffect(() => {
    if (isTauri() && entries.length === 0) void refresh()
    if (value) void inspect(value)
  }, [refresh, inspect, value, entries.length])
  return (
    <select value={value} onChange={(e) => { onChange(e.target.value); if (e.target.value) void inspect(e.target.value) }} className={INPUT}>
      <option value="">— Datensatz —</option>
      {entries.map((e) => <option key={e.relpath} value={e.relpath}>{e.name}</option>)}
    </select>
  )
}

function ModelRef({ value, onChange }: { value: string; onChange: (v: unknown) => void }) {
  const [models, setModels] = useState<string[]>([])
  useEffect(() => {
    if (!isTauri()) return
    void fs.list().then((entries) =>
      setModels(entries.filter((e) => !e.is_dir && e.relpath.toLowerCase().endsWith('.spinoml')).map((e) => e.relpath).sort()),
    ).catch(() => {})
  }, [])
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className={INPUT}>
      <option value="">— Modell —</option>
      {models.map((m) => <option key={m} value={m}>{m}</option>)}
    </select>
  )
}

function useColumns(dataset: string): string[] {
  const data = useDatasetsStore((s) => (dataset ? s.inspects[dataset]?.data : undefined))
  const inspect = useDatasetsStore((s) => s.inspect)
  useEffect(() => { if (dataset) void inspect(dataset) }, [dataset, inspect])
  return data && data.kind === 'tabular' && data.ok ? data.columns : []
}

function ColumnSingle({ value, dataset, onChange }: { value: string; dataset: string; onChange: (v: unknown) => void }) {
  const columns = useColumns(dataset)
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className={INPUT} disabled={!columns.length}>
      <option value="">{columns.length ? '— Spalte —' : '(Datensatz wählen)'}</option>
      {columns.map((c) => <option key={c} value={c}>{c}</option>)}
    </select>
  )
}

function ColumnsMulti({ value, dataset, onChange }: { value: string[]; dataset: string; onChange: (v: unknown) => void }) {
  const columns = useColumns(dataset)
  if (!columns.length) return <div className="text-[10px] text-[#7a8088]">leer = alle numerischen Spalten</div>
  const toggle = (c: string) => onChange(value.includes(c) ? value.filter((x) => x !== c) : [...value, c])
  return (
    <div className="max-h-40 space-y-1 overflow-auto rounded border border-[#1f2429] bg-[#0b0e11] p-1.5">
      {columns.map((c) => (
        <label key={c} className="flex items-center gap-1.5 text-[11px] text-[#cfd3d8]">
          <input type="checkbox" checked={value.includes(c)} onChange={() => toggle(c)} className="h-3.5 w-3.5 accent-[#6ab7ff]" />
          {c}
        </label>
      ))}
    </div>
  )
}
