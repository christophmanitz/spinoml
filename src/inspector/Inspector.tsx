import { useEffect, useState } from 'react'
import { useGraphStore } from '../canvas/GraphStore'
import { LAYERS, type FieldSpec } from '../layers/registry'
import { useInferenceStore } from '../inference/store'
import { useDatasetsStore } from '../datasets/store'
import { isTauri } from '../workspace/tauri-fs'

export default function Inspector() {
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId)
  const node = useGraphStore((s) => s.nodes.find((n) => n.id === selectedNodeId))
  const updateNodeParams = useGraphStore((s) => s.updateNodeParams)
  const deleteNode = useGraphStore((s) => s.deleteNode)
  const failingNodeId = useInferenceStore((s) => s.failingNodeId)
  const inferenceError = useInferenceStore((s) => s.error)
  const inferenceStage = useInferenceStore((s) => s.errorStage)

  if (!node) {
    return (
      <div className="flex h-full min-h-0 flex-col p-3 text-sm">
        <div className="mb-2 text-xs uppercase tracking-wide text-[#7a8088]">Inspector</div>
        <div className="text-xs text-[#7a8088]">Select a node to edit its parameters.</div>
      </div>
    )
  }

  const spec = LAYERS[node.data.layerType]
  const params = node.data.params
  const isFailing = failingNodeId === node.id
  const inShape = node.data.inferredInputShape
  const outShape = node.data.inferredOutputShape

  return (
    <div className="flex h-full min-h-0 flex-col p-3 text-sm">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-xs uppercase tracking-wide text-[#7a8088]">Inspector</div>
        {node.id !== 'input' && (
          <button
            className="text-[10px] text-red-400 hover:text-red-300"
            onClick={() => deleteNode(node.id)}
          >
            delete
          </button>
        )}
      </div>

      <div className="mb-1 flex items-baseline gap-2">
        <span className="font-medium">{node.data.layerType}</span>
        <span className="text-[10px] text-[#7a8088]">#{node.id}</span>
      </div>

      <div className="mb-2 grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5 font-mono text-[10px] text-[#9aa1a8]">
        <span className="text-[#7a8088]">in</span>
        <span>{inShape ? `[${inShape.join(', ')}]` : <em className="text-[#5b6168]">unknown</em>}</span>
        <span className="text-[#7a8088]">out</span>
        <span>{outShape ? `[${outShape.join(', ')}]` : <em className="text-[#5b6168]">unknown</em>}</span>
      </div>

      {isFailing && inferenceError && (
        <div className="mb-2 rounded border border-rose-900/60 bg-rose-950/40 px-2 py-1.5 text-[10px] leading-snug text-rose-200">
          <div className="mb-0.5 font-medium text-rose-300">
            forward failed at this layer{inferenceStage ? ` (${inferenceStage})` : ''}
          </div>
          <div className="font-mono text-rose-200/80">{shortenError(inferenceError)}</div>
          <FixHints nodeId={node.id} layerType={node.data.layerType} params={params} inShape={inShape}
                    onPatch={(patch) => updateNodeParams(node.id, patch)} />
        </div>
      )}

      <div className="flex-1 overflow-y-auto pr-1">
        {spec?.fields.length === 0 && (
          <div className="text-xs text-[#7a8088]">No parameters.</div>
        )}
        {spec?.fields.map((field) => (
          <ParamField
            key={`${node.id}:${field.name}`}
            field={field}
            value={params[field.name] ?? field.default}
            inShape={inShape}
            onChange={(v) => updateNodeParams(node.id, { [field.name]: v })}
          />
        ))}
      </div>
    </div>
  )
}

function shortenError(s: string): string {
  return s.length > 240 ? s.slice(0, 240) + '…' : s
}

function FixHints({
  nodeId, layerType, params, inShape, onPatch,
}: {
  nodeId: string
  layerType: string
  params: Record<string, unknown>
  inShape: number[] | undefined
  onPatch: (patch: Record<string, unknown>) => void
}) {
  const replaceNodeLayer = useGraphStore((s) => s.replaceNodeLayer)
  if (!inShape) return null
  const hints: { label: string; patch: Record<string, unknown>; rationale: string }[] = []
  const swaps: { label: string; rationale: string; apply: () => void }[] = []

  const rank = inShape.length

  // ─── Rank-mismatch swaps: when the chosen layer fundamentally can't
  // accept the input rank (e.g. Conv2d on 2D tabular data). These are
  // bigger fixes than param patches — they replace the whole layer.
  if (layerType === 'Conv2d' && rank < 4) {
    const lastDim = inShape[rank - 1]
    swaps.push({
      label: `mit Linear ersetzen (in_features=${lastDim})`,
      rationale: `Conv2d braucht 4D-Input [N,C,H,W]; dein Input ist ${rank}D. Für Tabular/Sequenz nimm Linear.`,
      apply: () => replaceNodeLayer(nodeId, 'Linear', {
        in_features: lastDim,
        out_features: typeof params.out_channels === 'number' ? params.out_channels : 64,
      }),
    })
  }
  if (layerType === 'Conv1d' && rank < 3) {
    const lastDim = inShape[rank - 1]
    swaps.push({
      label: `mit Linear ersetzen (in_features=${lastDim})`,
      rationale: `Conv1d braucht 3D-Input [N,C,L]; dein Input ist ${rank}D.`,
      apply: () => replaceNodeLayer(nodeId, 'Linear', {
        in_features: lastDim,
        out_features: typeof params.out_channels === 'number' ? params.out_channels : 64,
      }),
    })
  }
  if ((layerType === 'BatchNorm2d' || layerType === 'GroupNorm') && rank < 4) {
    swaps.push({
      label: `mit BatchNorm1d ersetzen`,
      rationale: `${layerType} braucht 4D-Input; dein Input ist ${rank}D. Nimm LayerNorm/BatchNorm1d.`,
      apply: () => replaceNodeLayer(nodeId, 'LayerNorm', {
        normalized_shape: [inShape[rank - 1]],
      }),
    })
  }
  if ((layerType === 'MaxPool2d' || layerType === 'AvgPool2d' || layerType === 'AdaptiveAvgPool2d') && rank < 4) {
    swaps.push({
      label: `Pool entfernen — passt nicht zu ${rank}D-Input`,
      rationale: `${layerType} braucht einen 4D-Tensor; dein Input ist ${rank}D.`,
      apply: () => {
        // Replace pool with Identity-like Flatten so pipeline keeps moving.
        replaceNodeLayer(nodeId, 'Flatten')
      },
    })
  }
  if (layerType === 'Linear' && rank > 2) {
    swaps.push({
      label: `Flatten davor einfügen empfohlen`,
      rationale: `Linear arbeitet auf der letzten Dim; bei ${rank}D-Input hilft Flatten oder GlobalPool davor.`,
      apply: () => replaceNodeLayer(nodeId, 'Flatten'),
    })
  }

  if (layerType === 'LayerNorm') {
    const last = inShape[inShape.length - 1]
    const current = params.normalized_shape as number[] | undefined
    if (last && (!current || current[current.length - 1] !== last)) {
      hints.push({
        label: `set normalized_shape = [${last}]`,
        patch: { normalized_shape: [last] },
        rationale: 'LayerNorm normalizes over the trailing dim of the input',
      })
    }
  }
  if (layerType === 'Conv2d' || layerType === 'Conv1d') {
    const c = inShape[1]
    const current = params.in_channels as number | undefined
    if (typeof c === 'number' && current !== c) {
      hints.push({
        label: `set in_channels = ${c}`,
        patch: { in_channels: c },
        rationale: 'Conv expects in_channels to match the channel dim of the input',
      })
    }
  }
  if (layerType === 'BatchNorm2d') {
    const c = inShape[1]
    const current = params.num_features as number | undefined
    if (typeof c === 'number' && current !== c) {
      hints.push({
        label: `set num_features = ${c}`,
        patch: { num_features: c },
        rationale: 'BatchNorm2d num_features = channel dim',
      })
    }
  }
  if (layerType === 'Linear') {
    const last = inShape[inShape.length - 1]
    const current = params.in_features as number | undefined
    if (typeof last === 'number' && current !== last) {
      hints.push({
        label: `set in_features = ${last}`,
        patch: { in_features: last },
        rationale: 'Linear in_features = last input dim',
      })
    }
  }
  if (layerType === 'GroupNorm') {
    const c = inShape[1]
    const current = params.num_channels as number | undefined
    if (typeof c === 'number' && current !== c) {
      hints.push({
        label: `set num_channels = ${c}`,
        patch: { num_channels: c },
        rationale: 'GroupNorm num_channels = channel dim',
      })
    }
  }

  if (!hints.length && !swaps.length) return null
  return (
    <div className="mt-1.5 flex flex-col gap-1">
      {swaps.map((s, i) => (
        <button
          key={`swap${i}`}
          className="self-start rounded bg-amber-900/60 px-1.5 py-0.5 text-left text-[10px] text-amber-100 hover:bg-amber-800"
          onClick={s.apply}
          title={s.rationale}
        >
          ⇄ {s.label}
        </button>
      ))}
      {hints.map((h, i) => (
        <button
          key={i}
          className="self-start rounded bg-rose-900/60 px-1.5 py-0.5 text-left font-mono text-[10px] text-rose-100 hover:bg-rose-800"
          onClick={() => onPatch(h.patch)}
          title={h.rationale}
        >
          fix: {h.label}
        </button>
      ))}
    </div>
  )
}

function ParamField({
  field, value, inShape, onChange,
}: {
  field: FieldSpec
  value: unknown
  inShape: number[] | undefined
  onChange: (v: unknown) => void
}) {
  return (
    <label className="mb-2 flex flex-col gap-1">
      <span className="flex items-baseline justify-between text-[10px] uppercase tracking-wide text-[#7a8088]">
        <span>{field.name}</span>
        <FieldHint field={field} value={value} inShape={inShape} />
      </span>
      <FieldInput field={field} value={value} onChange={onChange} />
    </label>
  )
}

function FieldHint({
  field, value, inShape,
}: { field: FieldSpec; value: unknown; inShape: number[] | undefined }) {
  if (!inShape) return null
  const last = inShape[inShape.length - 1]
  const c = inShape[1]
  if (field.name === 'normalized_shape' && typeof last === 'number') {
    const cur = (value as number[] | undefined)?.[((value as number[] | undefined)?.length ?? 0) - 1]
    if (cur !== last) return <span className="text-[10px] text-amber-400">expects [{last}]</span>
  }
  if (field.name === 'in_channels' && typeof c === 'number') {
    if (value !== c) return <span className="text-[10px] text-amber-400">input has {c}</span>
  }
  if (field.name === 'num_features' && typeof c === 'number') {
    if (value !== c) return <span className="text-[10px] text-amber-400">input has {c}</span>
  }
  if (field.name === 'num_channels' && typeof c === 'number') {
    if (value !== c) return <span className="text-[10px] text-amber-400">input has {c}</span>
  }
  if (field.name === 'in_features' && typeof last === 'number') {
    if (value !== last) return <span className="text-[10px] text-amber-400">input has {last}</span>
  }
  return null
}

function FieldInput({
  field, value, onChange,
}: {
  field: FieldSpec
  value: unknown
  onChange: (v: unknown) => void
}) {
  const baseClass =
    'rounded border border-[#1f2429] bg-[#0e1216] px-2 py-1 text-xs outline-none focus:border-[#3a4148]'

  switch (field.type) {
    case 'int':
      return <IntInput field={field} value={value as number} onChange={onChange} baseClass={baseClass} />
    case 'float':
      return <FloatInput field={field} value={value as number} onChange={onChange} baseClass={baseClass} />
    case 'bool':
      return (
        <input
          type="checkbox"
          className="h-4 w-4 self-start accent-[#6ab7ff]"
          checked={value as boolean}
          onChange={(e) => onChange(e.target.checked)}
        />
      )
    case 'select':
      return (
        <select
          className={baseClass}
          value={value as string}
          onChange={(e) => onChange(e.target.value)}
        >
          {field.options.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      )
    case 'tuple-int':
      return <TupleIntInput field={field} value={value} onChange={onChange} baseClass={baseClass} />
    case 'shape':
      return <ShapeInput value={value} onChange={onChange} baseClass={baseClass} />
    case 'dataset-ref':
      return <DatasetRefInput value={value as string} onChange={onChange} baseClass={baseClass} />
  }
}

function DatasetRefInput({
  value, onChange, baseClass,
}: {
  value: string
  onChange: (v: unknown) => void
  baseClass: string
}) {
  const entries = useDatasetsStore((s) => s.entries)
  const inspects = useDatasetsStore((s) => s.inspects)
  const refresh = useDatasetsStore((s) => s.refresh)
  const select = useDatasetsStore((s) => s.select)
  const inspect = useDatasetsStore((s) => s.inspect)
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId)
  const updateNodeParams = useGraphStore((s) => s.updateNodeParams)
  const nodeParams = useGraphStore((s) => {
    const id = s.selectedNodeId
    if (!id) return null
    const node = s.nodes.find((n) => n.id === id)
    return node?.data.params ?? null
  })

  useEffect(() => {
    if (isTauri() && entries.length === 0) void refresh()
    if (value) void inspect(value)
  }, [refresh, inspect, value, entries.length])

  const meta = value ? inspects[value]?.data : null
  const naturalShape = meta && meta.ok ? sampleNaturalShapeFrom(meta) : null
  const currentShape = (nodeParams?.shape as number[] | undefined) ?? []
  const shapesMatch = naturalShape && shallowEqShape(currentShape, naturalShape)

  function pick(rel: string) {
    onChange(rel)
    if (!rel || !selectedNodeId) return
    // Fire inspect → as soon as we know the natural shape, auto-write it to
    // the Input node so the rest of the graph (shape inference, codegen) re-runs.
    void (async () => {
      await inspect(rel)
      const data = useDatasetsStore.getState().inspects[rel]?.data
      if (!data || !data.ok) return
      const shape = sampleNaturalShapeFrom(data)
      if (!shape) return
      const node = useGraphStore.getState().nodes.find((n) => n.id === selectedNodeId)
      if (!node) return
      updateNodeParams(selectedNodeId, { ...node.data.params, dataset: rel, shape })
    })()
  }

  function applyShape() {
    if (!selectedNodeId || !naturalShape) return
    updateNodeParams(selectedNodeId, { ...(nodeParams ?? {}), shape: naturalShape })
  }

  return (
    <div className="flex flex-col gap-1">
      <select
        className={baseClass}
        value={value}
        onChange={(e) => pick(e.target.value)}
      >
        <option value="">— kein Datensatz gebunden —</option>
        {entries.length === 0 && <option disabled value="">(leg Daten in datasets/ ab)</option>}
        {entries.map((e) => (
          <option key={e.relpath} value={e.relpath}>
            {e.name}
          </option>
        ))}
      </select>
      {value && (
        <div className="space-y-1 rounded border border-[#1f2429] bg-[#0b0e11] p-1.5 text-[10px]">
          {meta && !meta.ok && (
            <div className="text-rose-300">inspect: {meta.error ?? 'fehlgeschlagen'}</div>
          )}
          {naturalShape && (
            <div className="flex items-baseline justify-between gap-2">
              <span className="text-[#7a8088]">
                Dataset-Shape: <code className="text-[#9aa1a8]">[{naturalShape.join(', ')}]</code>
              </span>
              {!shapesMatch && (
                <button
                  onClick={applyShape}
                  className="rounded bg-amber-900/40 px-1.5 py-0.5 text-[10px] text-amber-200 hover:bg-amber-900/60"
                  title="set Input.shape to the dataset's natural shape"
                >
                  übernehmen
                </button>
              )}
              {shapesMatch && (
                <span className="text-emerald-400">✓ shapes match</span>
              )}
            </div>
          )}
          <div className="flex items-baseline justify-between">
            <span className="text-[#7a8088]">{value}</span>
            <button
              onClick={() => select(value)}
              className="text-[#6ab7ff] hover:underline"
              title="open dataset detail modal"
            >
              ansehen ↗
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function shallowEqShape(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

function sampleNaturalShapeFrom(meta: unknown): number[] | null {
  const m = meta as { kind?: string; cols?: number; sample_size?: [number, number]; shape?: number[] }
  if (!m || !m.kind) return null
  if (m.kind === 'tabular' && typeof m.cols === 'number') return [1, m.cols]
  if (m.kind === 'image_folder' && m.sample_size) return [1, 3, m.sample_size[1], m.sample_size[0]]
  if (m.kind === 'tensor' && Array.isArray(m.shape)) return m.shape
  return null
}

function IntInput({
  field, value, onChange, baseClass,
}: {
  field: Extract<FieldSpec, { type: 'int' }>
  value: number
  onChange: (v: unknown) => void
  baseClass: string
}) {
  const [draft, setDraft] = useState(String(value))
  useEffect(() => { setDraft(String(value)) }, [value])
  return (
    <input
      type="number"
      className={baseClass}
      value={draft}
      min={field.min}
      max={field.max}
      step={field.step ?? 1}
      onChange={(e) => {
        setDraft(e.target.value)
        const v = parseInt(e.target.value, 10)
        if (!Number.isNaN(v)) onChange(v)
      }}
      onBlur={() => setDraft(String(value))}
    />
  )
}

function FloatInput({
  field, value, onChange, baseClass,
}: {
  field: Extract<FieldSpec, { type: 'float' }>
  value: number
  onChange: (v: unknown) => void
  baseClass: string
}) {
  const [draft, setDraft] = useState(String(value))
  useEffect(() => { setDraft(String(value)) }, [value])
  return (
    <input
      type="number"
      className={baseClass}
      value={draft}
      min={field.min}
      max={field.max}
      step={field.step ?? 'any'}
      onChange={(e) => {
        setDraft(e.target.value)
        const v = parseFloat(e.target.value)
        if (!Number.isNaN(v)) onChange(v)
      }}
      onBlur={() => setDraft(String(value))}
    />
  )
}

function TupleIntInput({
  field, value, onChange, baseClass,
}: {
  field: Extract<FieldSpec, { type: 'tuple-int' }>
  value: unknown
  onChange: (v: unknown) => void
  baseClass: string
}) {
  const arr = (Array.isArray(value) ? value : field.default) as number[]
  const [drafts, setDrafts] = useState(() => arr.map((n) => String(n)))
  useEffect(() => { setDrafts(arr.map((n) => String(n))) }, [arr.join(',')])
  return (
    <div className="flex gap-1">
      {Array.from({ length: field.arity }).map((_, i) => (
        <input
          key={i}
          type="number"
          className={`${baseClass} w-full`}
          value={drafts[i] ?? ''}
          step={1}
          onChange={(e) => {
            const nextDrafts = [...drafts]
            nextDrafts[i] = e.target.value
            setDrafts(nextDrafts)
            const v = parseInt(e.target.value, 10)
            if (!Number.isNaN(v)) {
              const next = [...arr]
              next[i] = v
              onChange(next)
            }
          }}
          onBlur={() => setDrafts(arr.map((n) => String(n)))}
        />
      ))}
    </div>
  )
}

function ShapeInput({
  value, onChange, baseClass,
}: { value: unknown; onChange: (v: unknown) => void; baseClass: string }) {
  const arr = (Array.isArray(value) ? value : []) as number[]
  const canonical = arr.join(', ')
  const [draft, setDraft] = useState(canonical)
  useEffect(() => { setDraft(canonical) }, [canonical])

  const parsed = parseShape(draft)
  const valid = parsed !== null && parsed.length > 0

  return (
    <div className="flex flex-col gap-0.5">
      <input
        type="text"
        className={`${baseClass} ${valid ? '' : 'border-amber-700/60'}`}
        value={draft}
        placeholder="e.g. 1, 3, 224, 224"
        onChange={(e) => {
          setDraft(e.target.value)
          const p = parseShape(e.target.value)
          if (p && p.length > 0) onChange(p)
        }}
        onBlur={() => {
          const p = parseShape(draft)
          if (p && p.length > 0) {
            onChange(p)
            setDraft(p.join(', '))
          } else {
            setDraft(canonical)
          }
        }}
      />
      {!valid && draft.trim() !== '' && (
        <span className="text-[10px] text-amber-400">comma-separated positive ints</span>
      )}
    </div>
  )
}

function parseShape(s: string): number[] | null {
  const parts = s.split(/[,\s]+/).map((p) => p.trim()).filter(Boolean)
  if (parts.length === 0) return null
  const out: number[] = []
  for (const p of parts) {
    const n = parseInt(p, 10)
    if (!Number.isFinite(n) || n <= 0 || String(n) !== p) return null
    out.push(n)
  }
  return out
}
