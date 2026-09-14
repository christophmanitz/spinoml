import { useEffect, useMemo, useState } from 'react'
import { useGraphStore } from '../canvas/GraphStore'
import { LAYERS, type FieldSpec } from '../layers/registry'
import { useInferenceStore } from '../inference/store'
import { useDatasetsStore } from '../datasets/store'
import type { GraphField } from '../datasets/types'
import { isTauri } from '../workspace/tauri-fs'
import { useScopeStore } from '../canvas/scopeStore'
import { layerInitExpr } from '../codegen/generator'
import { useChatStore } from '../chat/store'
import CodeField from './CodeField'

export default function Inspector() {
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId)
  const node = useGraphStore((s) => s.nodes.find((n) => n.id === selectedNodeId))
  const updateNodeParams = useGraphStore((s) => s.updateNodeParams)
  const replaceNodeLayer = useGraphStore((s) => s.replaceNodeLayer)
  const deleteNode = useGraphStore((s) => s.deleteNode)
  const failingNodeId = useInferenceStore((s) => s.failingNodeId)
  const inferenceError = useInferenceStore((s) => s.error)
  const inferenceStage = useInferenceStore((s) => s.errorStage)

  // Is the selected node an Input bound to a graph dataset? Then we replace the
  // tabular column pickers with a graph-field binding panel.
  const boundDatasetRel = useGraphStore((s) => {
    const n = s.nodes.find((m) => m.id === s.selectedNodeId)
    return String(n?.data.params.dataset ?? '')
  })
  const boundData = useDatasetsStore((s) => (boundDatasetRel ? s.inspects[boundDatasetRel]?.data : null))
  const isGraphDataset = !!(boundData && boundData.ok
    && (boundData.kind === 'graph_folder' || (boundData.kind === 'tensor' && boundData.is_graph)))

  if (!node) {
    return (
      <div className="flex h-full min-h-0 flex-col p-3 text-sm">
        <div className="mb-2 text-xs uppercase tracking-wide text-[#6f767e]">Inspector</div>
        <div className="text-xs text-[#6f767e]">Select a node to edit its parameters.</div>
      </div>
    )
  }

  const spec = LAYERS[node.data.layerType]
  const params = node.data.params
  // A proxy input is configured by the outer node it's wired from — read-only here.
  const isProxy = !!params._proxyOf
  const isFailing = failingNodeId === node.id
  const inShape = node.data.inferredInputShape
  const outShape = node.data.inferredOutputShape

  return (
    <div className="flex h-full min-h-0 flex-col p-3 text-sm">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-xs uppercase tracking-wide text-[#6f767e]">Inspector</div>
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
        <span className="text-[10px] text-[#6f767e]">#{node.id}</span>
      </div>

      <div className="mb-2 grid grid-cols-[auto_1fr] gap-x-2 gap-y-0.5 font-mono text-[10px] text-[#9aa1a8]">
        <span className="text-[#6f767e]">in</span>
        <span>{inShape ? `[${inShape.join(', ')}]` : <em className="text-[#5b6168]">unknown</em>}</span>
        <span className="text-[#6f767e]">out</span>
        <span>{outShape ? `[${outShape.join(', ')}]` : <em className="text-[#5b6168]">unknown</em>}</span>
      </div>

      <NodeActions
        node={node}
        onEnterGroup={() => useScopeStore.getState().enterGroup(node.id)}
        onEject={() => ejectToCustom(node, replaceNodeLayer)}
      />

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
        {isProxy ? (
          <div className="space-y-2">
            <div className="rounded border border-[#243a52] bg-[#0d1722] p-2 text-[11px] text-[#8fb6e0]">
              🔗 Verbundener Eingang — konfiguriert vom Knoten <strong>„{String(params.name ?? '?')}"</strong> draußen.
              Hier nicht änderbar; Form/Bindung am äußeren Knoten setzen. Trennst du die Kante, verschwindet dieser Eingang.
            </div>
            <div className="space-y-0.5">
              {spec?.fields.filter((f) => params[f.name] !== undefined).map((f) => (
                <div key={f.name} className="flex items-center justify-between rounded bg-[#0b0e11] px-1.5 py-1 font-mono text-[10px]">
                  <span className="text-[#6f767e]">{f.name}</span>
                  <span className="text-[#9aa1a8]">{JSON.stringify(params[f.name])}</span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <>
            {spec?.fields.length === 0 && (
              <div className="text-xs text-[#6f767e]">No parameters.</div>
            )}
            {spec?.fields
              .filter((field) => !(isGraphDataset && node.data.layerType === 'Input'
                && (field.type === 'columns-multi' || field.type === 'column-single')))
              .map((field) => (
                <ParamField
                  key={`${node.id}:${field.name}`}
                  field={field}
                  value={params[field.name] ?? field.default}
                  inShape={inShape}
                  onChange={(v) => updateNodeParams(node.id, { [field.name]: v })}
                />
              ))}
            {node.data.layerType === 'Input' && <GraphBindingPanel node={node} />}
            {node.data.layerType === 'Graph' && <GraphNodeBindingPanel node={node} />}
            {node.data.layerType === 'Manifest' && <ManifestNodePanel node={node} />}
            {node.data.layerType === 'DataOp' && <DataOpPanel node={node} />}
          </>
        )}
      </div>
    </div>
  )
}

// DataOp = a data-stage node. Its preprocessing runs OFFLINE (not in the model
// forward), via the chatbot's write_file + run_script path so it reuses the same
// confirm GUI and local/ssh/SLURM execution. The button hands the agent the
// intent; the node's `script` param is already in the chat's graph snapshot.
function DataOpPanel({ node }: { node: { id: string; data: { params: Record<string, unknown> } } }) {
  const p = node.data.params
  const input = String(p.input_dataset ?? '')
  const output = String(p.output_name ?? 'out') || 'out'
  const mode = String(p.mode ?? 'shell')
  const online = useChatStore((s) => s.online)
  const status = useChatStore((s) => s.status)
  const busy = status === 'streaming'

  const run = () => {
    const scriptPath = `agent/${node.id}.py`
    const args = input ? `--input ${input} --output ${output}` : `--output ${output}`
    const msg = [
      `Führe die DataOp-Vorverarbeitung für Node "${node.id}" aus.`,
      `Schreibe dessen \`script\`-Parameter (steht im Graph-Snapshot) mit write_file nach ${scriptPath}.`,
      input ? `Eingabe-Dataset: ${input}.` : 'Es ist kein Eingabe-Dataset gesetzt.',
      `Ist das Skript noch ein Platzhalter/TODO, fülle es passend zum Eingabe-Dataset (lesen → transformieren → unter datasets/${output} cachen, ggf. die .manifest um einen Branch ergänzen); bei Unklarheiten frag mit ask_user.`,
      `Führe es dann aus mit run_script(path="${scriptPath}", mode="${mode}", args="${args}").`,
    ].filter(Boolean).join(' ')
    void useChatStore.getState().send(msg)
  }

  return (
    <div className="mt-3 space-y-2 border-t border-[#1f2429] pt-2">
      <div className="rounded border border-[#243a52] bg-[#0d1722] p-2 text-[11px] leading-snug text-[#8fb6e0]">
        Daten-Node: läuft einmal <strong>offline</strong> und cached das Ergebnis als Dataset
        (unter <code className="text-[#cdd3da]">datasets/{output}</code>). Nicht Teil des Modells —
        im Forward-Pass ist er ein Passthrough.
      </div>
      <button
        className="w-full rounded border border-emerald-800/60 bg-emerald-950/30 px-2 py-1.5 text-xs text-emerald-200 hover:bg-emerald-900/40 disabled:opacity-50"
        onClick={run}
        disabled={online === false || busy}
        title={online === false ? 'Chatbot-Sidecar offline' : 'Skript schreiben + ausführen (mit Bestätigung)'}
      >
        ▶ Vorverarbeitung ausführen{mode === 'slurm' ? ' (SLURM)' : ''}
      </button>
      {online === false && (
        <div className="text-[10px] text-amber-400">Chatbot offline — Sidecar starten (npm run sidecar:llm).</div>
      )}
      {busy && <div className="text-[10px] text-[#6f767e]">Chatbot arbeitet… Bestätigung erscheint im Chat.</div>}
    </div>
  )
}

// A `Graph` input binds to a WHOLE graph (dataset / manifest branch). This panel
// shows the graph's real fields (x [N,F] · edge_index [2,E] · batch · edge_attr)
// from the bound dataset — reality stays visible, nothing simplified — and
// auto-applies x's shape, n_edges and edge_dim to the node so codegen + smoke
// line up. For a manifest, pick which branch this node is.
function GraphNodeBindingPanel({ node }: { node: { id: string; data: { params: Record<string, unknown> } } }) {
  const branch = String(node.data.params.branch ?? '')
  const updateNodeParams = useGraphStore((s) => s.updateNodeParams)
  // Prefer a connected Manifest node's dataset (the manifest is its own node now);
  // fall back to this Graph node's own `dataset` for a standalone graph dataset.
  const allNodes = useGraphStore((s) => s.nodes)
  const allEdges = useGraphStore((s) => s.edges)
  const manifestRel = useMemo(() => {
    for (const e of allEdges) {
      if (e.target !== node.id) continue
      const src = allNodes.find((n) => n.id === e.source)
      if (src && src.data.layerType === 'Manifest') return String(src.data.params.dataset ?? '')
    }
    return ''
  }, [allNodes, allEdges, node.id])
  const datasetRel = manifestRel || String(node.data.params.dataset ?? '')
  const inspectAction = useDatasetsStore((s) => s.inspect)
  const cached = useDatasetsStore((s) => (datasetRel ? s.inspects[datasetRel] : undefined))
  const data = cached?.data ?? null
  const inspecting = !!cached?.loading
  // The inspect call returned an error (e.g. the HPC sidecar wasn't reachable yet,
  // or a path was missing). Surface it instead of the misleading "not a graph?".
  const inspectError = (!data && cached?.error) ? cached.error : null

  useEffect(() => { if (datasetRel) void inspectAction(datasetRel) }, [datasetRel, inspectAction])

  // Branches (manifest) + the fields for THIS graph (prefixed by branch in a
  // manifest, bare otherwise). Normalize to {name, shape, dtype}.
  const isManifest = !!(data && data.ok && data.kind === 'manifest')
  const branches: string[] = isManifest && data && data.ok && data.kind === 'manifest' ? data.branches : []
  const fields: GraphField[] | null = (() => {
    if (!data || !data.ok) return null
    if (data.kind === 'manifest') {
      const pre = branch ? `${branch}.` : ''
      const slots = data.slots.filter((s) => s.field !== 'target' && (!branch || s.field.startsWith(pre)))
      return slots.map((s) => ({ name: branch ? s.field.slice(pre.length) : s.field, shape: s.shape, dtype: s.dtype }))
    }
    if (data.kind === 'graph_folder') return data.fields
    if (data.kind === 'tensor' && data.is_graph) return data.fields ?? null
    return null
  })()

  const xField = fields?.find((f) => f.name === 'x') ?? null
  const eiField = fields?.find((f) => f.name === 'edge_index') ?? null
  const eaField = fields?.find((f) => f.name === 'edge_attr') ?? null
  const key = `${xField?.shape.join(',')}|${eiField?.shape.join(',')}|${eaField?.shape.join(',')}`
  // Per-branch resolution notes from a manifest inspect (e.g. a .pt that isn't a
  // PyG graph, a missing branch file) — shown so failures are explained, not hidden.
  const notes: string[] = isManifest && data && data.ok && data.kind === 'manifest' ? (data.notes ?? []) : []

  // Auto-apply x shape [N,F], n_edges (edge_index [2,E]) and edge_dim (edge_attr).
  useEffect(() => {
    if (!xField) return
    const patch: Record<string, unknown> = {}
    const curShape = node.data.params.shape as number[] | undefined
    const same = !!curShape && curShape.length === xField.shape.length && curShape.every((v, i) => v === xField.shape[i])
    if (!same) patch.shape = xField.shape
    if (eiField && eiField.shape.length === 2 && node.data.params.n_edges !== eiField.shape[1]) patch.n_edges = eiField.shape[1]
    // No edge_attr on this branch → edge_dim 0 (don't keep a stale value from a
    // previously-selected branch that did have edge features).
    const wantEdgeDim = (eaField && eaField.shape.length === 2) ? eaField.shape[1] : 0
    if (node.data.params.edge_dim !== wantEdgeDim) patch.edge_dim = wantEdgeDim
    if (Object.keys(patch).length) updateNodeParams(node.id, patch)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, node.id])

  if (!datasetRel) {
    return (
      <div className="mt-2 rounded border border-[#1f2429] bg-[#0b0e11] p-2 text-[10px] text-[#6f767e]">
        Binde diesen Graph an ein Dataset (oben „dataset") — ein <code>.pt</code>-Ordner, ein <code>pyg:</code>-Set,
        ein Molekül oder einen <strong>Manifest-Branch</strong>. Der ganze Graph (x · edge_index · batch · edge_attr) fließt als ein <code>Data</code>-Objekt.
      </div>
    )
  }

  return (
    <div className="mt-2 rounded border border-[#1f2429] bg-[#0b0e11] p-2">
      {branches.length > 0 && (
        <div className="mb-2">
          <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">Manifest-Branch dieses Graphen</div>
          <div className="flex flex-wrap gap-1">
            {branches.map((b) => (
              <button
                key={b}
                onClick={() => updateNodeParams(node.id, { branch: b })}
                className={`rounded px-1.5 py-0.5 font-mono text-[10px] ${b === branch ? 'bg-[var(--accent-sel)] text-[var(--accent)]' : 'text-[#9aa1a8] hover:bg-[#13171b]'}`}
              >{b}</button>
            ))}
          </div>
        </div>
      )}
      <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">Graph-Felder (Realität · read-only)</div>
      {fields && fields.length > 0 ? (
        <div className="space-y-0.5">
          {fields.map((f) => (
            <div key={f.name} className="flex items-center justify-between rounded px-1.5 py-1 font-mono text-[10px] text-[#9aa1a8]">
              <span>{f.name}</span>
              <span className="text-[#6f767e]">[{f.shape.join(', ')}] · {f.dtype.replace('torch.', '')}</span>
            </div>
          ))}
        </div>
      ) : inspecting ? (
        <div className="text-[10px] text-[#6f767e]">Dataset wird geprüft… (Torch-Sidecar)</div>
      ) : inspectError ? (
        <div className="space-y-1 text-[10px]">
          <div className="text-rose-400">Dataset konnte nicht gelesen werden: {inspectError}</div>
          <button
            onClick={() => datasetRel && void inspectAction(datasetRel, true)}
            className="rounded border border-[#1f2429] px-1.5 py-0.5 text-[#9aa1a8] hover:border-[#3a4148] hover:text-[#e6e8eb]"
          >Erneut versuchen</button>
        </div>
      ) : isManifest && !branch ? (
        <div className="text-[10px] text-amber-400/80">Branch oben wählen.</div>
      ) : isManifest && branch ? (
        <div className="text-[10px] text-amber-400/80">Branch „{branch}" hat keine Graph-Felder (x / edge_index) — siehe Hinweise unten.</div>
      ) : (
        <div className="text-[10px] text-amber-400/80">Keine Graph-Felder erkannt — ein Graph braucht ein .pt-Graph-/pyg:-/Molekül-Dataset oder einen Manifest-Branch.</div>
      )}
      {notes.length > 0 && (
        <div className="mt-1.5 space-y-0.5 border-t border-[#1f2429] pt-1.5 text-[10px] text-amber-400/70">
          {notes.map((n, i) => <div key={i}>⚠ {n}</div>)}
        </div>
      )}
      {xField && (
        <div className="mt-1 text-[10px] text-emerald-300/80">
          ✓ x [{xField.shape.join(', ')}] → Form gesetzt{eiField ? ` · ${eiField.shape[1]} Kanten` : ''}{eaField ? ` · edge_dim ${eaField.shape[1]}` : ''}
        </div>
      )}
    </div>
  )
}

// Panel for a Manifest node: shows the .manifest's branches grouped by TYPE
// (Graph / Sequence / Tensor / Target) so you see "what belongs to what" and
// connect each branch to the right typed input node via an edge.
function ManifestNodePanel({ node }: { node: { id: string; data: { params: Record<string, unknown> } } }) {
  const datasetRel = String(node.data.params.dataset ?? '')
  const inspectAction = useDatasetsStore((s) => s.inspect)
  const cached = useDatasetsStore((s) => (datasetRel ? s.inspects[datasetRel] : undefined))
  useEffect(() => { if (datasetRel) void inspectAction(datasetRel) }, [datasetRel, inspectAction])
  const data = cached?.data ?? null

  if (!datasetRel) {
    return (
      <div className="mt-2 rounded border border-[#1f2429] bg-[#0b0e11] p-2 text-[10px] text-[#6f767e]">
        Wähle oben eine <code>.manifest</code>. Verbinde diesen Node dann per Kante mit Graph-/Sequence-/Input-Nodes — je eine Kante pro Branch. Der Manifest-Node liefert die gepaarten Daten; er erzeugt selbst keinen Modell-Code.
      </div>
    )
  }
  if (cached?.loading) return <div className="mt-2 rounded border border-[#1f2429] bg-[#0b0e11] p-2 text-[10px] text-[#6f767e]">Manifest wird geprüft…</div>
  if (!data || !data.ok || data.kind !== 'manifest') {
    return <div className="mt-2 rounded border border-[#1f2429] bg-[#0b0e11] p-2 text-[10px] text-rose-400">{cached?.error ?? 'Keine gültige .manifest erkannt.'}</div>
  }

  const byBranch = new Map<string, { field: string; shape: number[]; dtype: string }[]>()
  for (const s of data.slots) {
    if (s.field === 'target') continue
    const dot = s.field.indexOf('.')
    const br = dot > 0 ? s.field.slice(0, dot) : s.field
    const fld = dot > 0 ? s.field.slice(dot + 1) : s.field
    if (!byBranch.has(br)) byBranch.set(br, [])
    byBranch.get(br)!.push({ field: fld, shape: s.shape, dtype: s.dtype })
  }
  // Branches the sidecar tokenizes as ESPF substructures (kind:"espf") — read off
  // the inspect notes, since an ESPF slot looks like a plain 1-D int Sequence slot.
  const espfBranches = new Set<string>()
  for (const n of data.notes ?? []) {
    const m = /branch '([^']+)': ESPF/.exec(n)
    if (m) espfBranches.add(m[1])
  }
  const typeOf = (br: string, flds: { field: string; shape: number[]; dtype: string }[]): string => {
    if (espfBranches.has(br)) return 'ESPF'
    if (!flds.length) return '?'
    const names = flds.map((f) => f.field)
    if (names.includes('x') && names.includes('edge_index')) return 'Graph'
    if (flds.length === 1 && flds[0].shape.length <= 1 && /int|long/.test(flds[0].dtype)) return 'Sequence'
    return 'Tensor'
  }
  const hint = (t: string) => t === 'Graph' ? ' → Graph-Node' : t === 'Sequence' ? ' → Sequence-Node' : t === 'ESPF' ? ' → ESPF-Node' : t === 'Tensor' ? ' → Input-Node' : ''

  return (
    <div className="mt-2 rounded border border-[#1f2429] bg-[#0b0e11] p-2">
      <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">Branches · verbinde je einen mit einem Input-Node</div>
      <div className="space-y-1.5">
        {data.branches.map((br) => {
          const flds = byBranch.get(br) ?? []
          const type = typeOf(br, flds)
          return (
            <div key={br} className="rounded border border-[#1f2429] px-1.5 py-1">
              <div className="flex items-center justify-between">
                <span className="font-mono text-[11px] text-[#e6e8eb]">{br}</span>
                <span className="text-[10px] text-[var(--accent)]">{type}{hint(type)}</span>
              </div>
              {flds.length ? (
                <div className="mt-0.5 flex flex-wrap gap-x-2 font-mono text-[10px] text-[#6f767e]">
                  {flds.map((f) => <span key={f.field}>{f.field}[{f.shape.join(',')}]</span>)}
                </div>
              ) : (
                <div className="mt-0.5 text-[10px] text-amber-400/70">nicht als Graph ladbar — siehe Hinweise (z.B. Token-Tensor → Sequence-Node)</div>
              )}
            </div>
          )
        })}
      </div>
      {data.target && <div className="mt-1.5 text-[10px] text-[#9aa1a8]">target: <span className="font-mono">{data.target.column}</span> ({data.target.type}) → Training</div>}
      {data.notes && data.notes.length > 0 && (
        <div className="mt-1.5 space-y-0.5 border-t border-[#1f2429] pt-1.5 text-[10px] text-amber-400/70">
          {data.notes.map((n, i) => <div key={i}>⚠ {n}</div>)}
        </div>
      )}
    </div>
  )
}

// When an Input is bound to a graph dataset, show its fields and auto-set the
// Input's shape + dtype from the field it binds to. Click a field to bind.
// Two modes:
//  • graph_folder / tensor-graph: the slot key IS the Input's `name` (x /
//    edge_index / …) — clicking sets `name` (which is also the codegen arg).
//  • manifest: slots are fully-qualified ('<branch>.x') and can't be a valid
//    codegen identifier, so the selection lives in `bind_field` and `name` is
//    left untouched (stays the Python-valid forward-arg name from the template).
function GraphBindingPanel({ node }: { node: { id: string; data: { params: Record<string, unknown> } } }) {
  const datasetRel = String(node.data.params.dataset ?? '')
  const name = String(node.data.params.name ?? 'x')
  const bindField = String(node.data.params.bind_field ?? '')
  const updateNodeParams = useGraphStore((s) => s.updateNodeParams)
  const inspectAction = useDatasetsStore((s) => s.inspect)
  const data = useDatasetsStore((s) => (datasetRel ? s.inspects[datasetRel]?.data : null))

  useEffect(() => { if (datasetRel) void inspectAction(datasetRel) }, [datasetRel, inspectAction])

  const isManifest = !!(data && data.ok && data.kind === 'manifest')
  const graphFields: GraphField[] | null =
    data && data.ok && data.kind === 'graph_folder' ? data.fields
    : data && data.ok && data.kind === 'tensor' && data.is_graph ? (data.fields ?? null)
    : null

  // Normalize both modes to {key, shape, dtype}. `sel` is the current binding
  // and `bindBy` is the param the click writes ('bind_field' vs 'name').
  const slots: { key: string; shape: number[]; dtype: string }[] | null =
    isManifest && data && data.ok && data.kind === 'manifest'
      ? data.slots.map((s) => ({ key: s.field, shape: s.shape, dtype: s.dtype }))
      : graphFields?.map((f) => ({ key: f.name, shape: f.shape, dtype: f.dtype })) ?? null
  const sel = isManifest ? bindField : name
  const notes = isManifest && data && data.ok && data.kind === 'manifest' ? data.notes : undefined

  const dtypeFor = (dt: string) => (/int|long|bool/.test(dt) ? 'int64' : 'float32')
  const pick = (key: string, shape: number[], dtype: string) =>
    updateNodeParams(node.id, isManifest
      ? { bind_field: key, shape, dtype }
      : { name: key, shape, dtype })

  const match = slots?.find((s) => s.key === sel) ?? null
  const matchKey = match ? `${match.shape.join(',')}:${match.dtype}` : ''

  // Auto-apply shape + dtype from the matching slot so /infer + smoke validate.
  useEffect(() => {
    if (!match) return
    const dt = dtypeFor(match.dtype)
    const cur = node.data.params.shape as number[] | undefined
    const same = !!cur && cur.length === match.shape.length && cur.every((v, i) => v === match.shape[i])
    if (!same || node.data.params.dtype !== dt) {
      updateNodeParams(node.id, { shape: match.shape, dtype: dt })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [matchKey, node.id])

  if (!datasetRel || !slots) return null

  return (
    <div className="mt-2 rounded border border-[#1f2429] bg-[#0b0e11] p-2">
      <div className="mb-1 text-[10px] uppercase tracking-wider text-[#6f767e]">
        {isManifest ? 'Manifest-Slots · klick = an Input binden' : 'Graph-Felder · klick = an Input binden'}
      </div>
      <div className="space-y-0.5">
        {slots.map((s) => {
          const active = s.key === sel
          return (
            <button
              key={s.key}
              onClick={() => pick(s.key, s.shape, dtypeFor(s.dtype))}
              className={`flex w-full items-center justify-between rounded px-1.5 py-1 font-mono text-[10px] ${active ? 'bg-[var(--accent-sel)] text-[var(--accent)]' : 'text-[#9aa1a8] hover:bg-[#13171b]'}`}
            >
              <span>{s.key}</span>
              <span className="text-[#6f767e]">[{s.shape.join(', ')}] · {s.dtype.replace('torch.', '')}</span>
            </button>
          )
        })}
      </div>
      {match
        ? <div className="mt-1 text-[10px] text-emerald-300/80">✓ {isManifest ? `gebunden an „${sel}" · ` : ''}Form [{match.shape.join(', ')}] + dtype automatisch gesetzt</div>
        : <div className="mt-1 text-[10px] text-amber-400/80">{isManifest ? 'Noch kein Slot gebunden — oben einen wählen.' : `Input-Name „${name}" passt zu keinem Feld — oben eins wählen.`}</div>}
      {!isManifest && slots.some((s) => s.key === 'y') && (
        <div className="mt-1.5 text-[10px] text-[#5b6168]">Ziel/Label = Feld <code>y</code> (im Training-Graph als Target wählbar).</div>
      )}
      {notes && notes.length > 0 && (
        <div className="mt-1.5 space-y-0.5 border-t border-[#1f2429] pt-1.5">
          {notes.map((n, i) => (
            <div key={i} className="text-[10px] text-amber-400/80">⚠ {n}</div>
          ))}
        </div>
      )}
    </div>
  )
}

function shortenError(s: string): string {
  return s.length > 240 ? s.slice(0, 240) + '…' : s
}

// Per-node actions: open a Group's subcanvas, or "eject" a built-in module node
// into an editable Custom-code node (take an existing node, tweak its code).
function NodeActions({
  node, onEnterGroup, onEject,
}: {
  node: { id: string; data: { layerType: string } }
  onEnterGroup: () => void
  onEject: () => void
}) {
  const lt = node.data.layerType
  const spec = LAYERS[lt]
  if (lt === 'Subgraph') {
    return (
      <button
        onClick={onEnterGroup}
        className="mb-2 w-full rounded bg-[var(--accent-sel)] px-2 py-1 text-[11px] text-[var(--accent)] hover:bg-[var(--accent-sel-hover)]"
        title="Subgraph dieses Knotens bearbeiten"
      >
        ⤢ Subcanvas öffnen
      </button>
    )
  }
  // Only module-kind layers (those with a pytorchModule) can be ejected.
  if (!spec?.pytorchModule || spec.kind === 'custom') return null
  return (
    <button
      onClick={onEject}
      className="mb-2 w-full rounded border border-[#1f2429] px-2 py-1 text-[11px] text-[#9aa1a8] hover:border-[#3a4148] hover:bg-[#13171b] hover:text-[#e6e8eb]"
      title="In einen editierbaren Custom-Code-Knoten umwandeln (Code dieses Layers als Startpunkt)"
    >
      ✎ Zu Custom-Code umwandeln
    </button>
  )
}

// Turn a built-in module node into a Custom node seeded with that layer's code,
// so you can take an existing node and lightly edit it.
function ejectToCustom(
  node: { id: string; data: { layerType: string; params: Record<string, unknown> } },
  replaceNodeLayer: (id: string, layerType: string, extra?: Record<string, unknown>) => void,
) {
  const lt = node.data.layerType
  const spec = LAYERS[lt]
  const expr = layerInitExpr(lt, node.data.params)
  if (!spec || !expr) return
  const cls = `${lt}Custom`
  const imports = spec.pyImports?.length
    ? `from torch_geometric.nn import ${spec.pyImports.join(', ')}\n\n\n`
    : ''
  const sig = spec.needsEdgeIndex ? 'x, edge_index' : 'x'
  const callArgs = spec.needsEdgeIndex ? 'x, edge_index' : 'x'
  const body = spec.tupleOutput
    ? `        out, _ = self.layer(${callArgs})\n        return out`
    : `        return self.layer(${callArgs})`
  const source = `${imports}class ${cls}(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = ${expr}

    def forward(self, ${sig}):
${body}`
  replaceNodeLayer(node.id, 'Custom', { class_name: cls, init_args: '', source })
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
  const GNN_CONV = layerType === 'GCNConv' || layerType === 'GATConv'
    || layerType === 'SAGEConv' || layerType === 'GraphConv'
  if (layerType === 'Conv2d' || layerType === 'Conv1d' || layerType === 'Conv3d' || GNN_CONV) {
    // For GNN node features [N, F] the feature dim is the last; for Conv it's dim 1.
    const c = GNN_CONV ? inShape[inShape.length - 1] : inShape[1]
    const current = params.in_channels as number | undefined
    if (typeof c === 'number' && current !== c) {
      hints.push({
        label: `set in_channels = ${c}`,
        patch: { in_channels: c },
        rationale: GNN_CONV
          ? 'GNN conv in_channels = node-feature dim (last dim of [N, F])'
          : 'Conv expects in_channels to match the channel dim of the input',
      })
    }
  }
  if (layerType === 'BatchNorm2d' || layerType === 'BatchNorm1d') {
    const c = inShape[1]
    const current = params.num_features as number | undefined
    if (typeof c === 'number' && current !== c) {
      hints.push({
        label: `set num_features = ${c}`,
        patch: { num_features: c },
        rationale: 'BatchNorm num_features = channel dim',
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
      <span className="flex items-baseline justify-between text-[10px] uppercase tracking-wide text-[#6f767e]">
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
          className="h-4 w-4 self-start accent-[var(--accent)]"
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
    case 'int-list':
      return <IntListInput value={value} onChange={onChange} baseClass={baseClass} />
    case 'shape':
      return <ShapeInput value={value} onChange={onChange} baseClass={baseClass} />
    case 'dataset-ref':
      return <DatasetRefInput value={value as string} onChange={onChange} baseClass={baseClass} />
    case 'columns-multi':
      return <ColumnsMultiInput value={value as string[]} onChange={onChange} />
    case 'column-single':
      return <ColumnSingleInput value={value as string} onChange={onChange} baseClass={baseClass} />
    case 'text':
      return <TextInput field={field} value={value as string} onChange={onChange} baseClass={baseClass} />
    case 'code':
      return <CodeInput field={field} value={value as string} onChange={onChange} />
  }
}

function TextInput({
  field, value, onChange, baseClass,
}: {
  field: Extract<FieldSpec, { type: 'text' }>
  value: string
  onChange: (v: unknown) => void
  baseClass: string
}) {
  const [draft, setDraft] = useState(value ?? '')
  useEffect(() => { setDraft(value ?? '') }, [value])
  const listId = field.datalist?.length ? `dl-${field.name}` : undefined
  return (
    <>
      <input
        type="text"
        className={`${baseClass} font-mono`}
        value={draft}
        placeholder={field.placeholder}
        spellCheck={false}
        list={listId}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={() => onChange(draft)}
        onKeyDown={(e) => { if (e.key === 'Enter') (e.target as HTMLInputElement).blur() }}
      />
      {listId && (
        <datalist id={listId}>
          {field.datalist!.map((o) => <option key={o} value={o} />)}
        </datalist>
      )}
    </>
  )
}

function CodeInput({
  field, value, onChange,
}: {
  field: Extract<FieldSpec, { type: 'code' }>
  value: string
  onChange: (v: unknown) => void
}) {
  return <CodeField value={value ?? ''} placeholder={field.placeholder} onChange={(v) => onChange(v)} />
}

function columnsFor(datasetRel: string | undefined): string[] | null {
  if (!datasetRel) return null
  const data = useDatasetsStore.getState().inspects[datasetRel]?.data
  if (!data || !data.ok) return null
  if (data.kind === 'tabular') return data.columns
  return null
}

function useBoundDatasetCols(): { cols: string[] | null; datasetRel: string } {
  const datasetRel = useGraphStore((s) => {
    const id = s.selectedNodeId
    if (!id) return ''
    const node = s.nodes.find((n) => n.id === id)
    return String(node?.data.params.dataset ?? '')
  })
  const inspectData = useDatasetsStore((s) => (datasetRel ? s.inspects[datasetRel]?.data : null))
  const cols = inspectData && inspectData.ok && inspectData.kind === 'tabular' ? inspectData.columns : null
  // Silence unused linter (helper API).
  void columnsFor
  return { cols, datasetRel }
}

function ColumnsMultiInput({
  value, onChange,
}: { value: string[]; onChange: (v: unknown) => void }) {
  const { cols, datasetRel } = useBoundDatasetCols()
  const selectedNodeId = useGraphStore((s) => s.selectedNodeId)
  const updateNodeParams = useGraphStore((s) => s.updateNodeParams)

  if (!datasetRel) {
    return <div className="text-[10px] italic text-[#5b6168]">(braucht ein gebundenes Dataset)</div>
  }
  if (!cols) {
    return <div className="text-[10px] italic text-[#5b6168]">(nur für tabular Datasets)</div>
  }

  function toggle(col: string) {
    const next = value.includes(col) ? value.filter((c) => c !== col) : [...value, col]
    onChange(next)
    if (selectedNodeId) {
      const node = useGraphStore.getState().nodes.find((n) => n.id === selectedNodeId)
      if (node) {
        const newShape = [(node.data.params.shape as number[] | undefined)?.[0] ?? 1, next.length]
        updateNodeParams(selectedNodeId, { ...node.data.params, features: next, shape: newShape })
      }
    }
  }

  function selectAllNumeric() {
    const data = useDatasetsStore.getState().inspects[datasetRel]?.data
    if (!data || !data.ok || data.kind !== 'tabular') return
    const numeric = data.columns.filter((_c, i) => /int|float/.test(data.dtypes[i] ?? ''))
    onChange(numeric)
    if (selectedNodeId) {
      const node = useGraphStore.getState().nodes.find((n) => n.id === selectedNodeId)
      if (node) {
        updateNodeParams(selectedNodeId, {
          ...node.data.params,
          features: numeric,
          shape: [(node.data.params.shape as number[] | undefined)?.[0] ?? 1, numeric.length],
        })
      }
    }
  }

  return (
    <div className="space-y-1 rounded border border-[#1f2429] bg-[#0b0e11] p-1.5">
      <div className="flex items-center justify-between">
        <span className="text-[10px] text-[#6f767e]">{value.length}/{cols.length} ausgewählt</span>
        <button
          onClick={selectAllNumeric}
          className="text-[10px] text-[var(--accent)] hover:underline"
        >
          alle numerischen
        </button>
      </div>
      <div className="max-h-32 space-y-0.5 overflow-y-auto">
        {cols.map((col) => (
          <label key={col} className="flex cursor-pointer items-center gap-1.5 text-[10px] text-[#e6e8eb] hover:bg-[#13171b]">
            <input
              type="checkbox"
              checked={value.includes(col)}
              onChange={() => toggle(col)}
              className="h-3 w-3 accent-[var(--accent)]"
            />
            <span className="truncate">{col}</span>
          </label>
        ))}
      </div>
    </div>
  )
}

function ColumnSingleInput({
  value, onChange, baseClass,
}: { value: string; onChange: (v: unknown) => void; baseClass: string }) {
  const { cols, datasetRel } = useBoundDatasetCols()
  if (!datasetRel) {
    return <div className="text-[10px] italic text-[#5b6168]">(braucht ein gebundenes Dataset)</div>
  }
  if (!cols) {
    return <div className="text-[10px] italic text-[#5b6168]">(nur für tabular Datasets)</div>
  }
  return (
    <select
      className={baseClass}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    >
      <option value="">— kein Target —</option>
      {cols.map((c) => (
        <option key={c} value={c}>{c}</option>
      ))}
    </select>
  )
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
    void (async () => {
      await inspect(rel, true)  // force: bust any stale cache from a previous dataset
      const data = useDatasetsStore.getState().inspects[rel]?.data
      if (!data || !data.ok) return
      const node = useGraphStore.getState().nodes.find((n) => n.id === selectedNodeId)
      if (!node) return

      // For tabular: default features = all numeric columns; shape = [1, n_numeric].
      if (data.kind === 'tabular') {
        const numeric = data.columns.filter((_c, i) => /int|float/.test(data.dtypes[i] ?? ''))
        updateNodeParams(selectedNodeId, {
          ...node.data.params,
          dataset: rel,
          features: numeric,
          target: '',
          shape: [1, numeric.length],
        })
        return
      }
      // Graph dataset: clear tabular leftovers (features/target); the
      // GraphBindingPanel sets shape + dtype per-field from the Input's name.
      const isGraph = data.kind === 'graph_folder' || data.kind === 'pyg'
        || (data.kind === 'tensor' && data.is_graph)
      if (isGraph) {
        updateNodeParams(selectedNodeId, { ...node.data.params, dataset: rel, features: [], target: '' })
        return
      }
      // Other non-tabular: natural shape, and clear stale tabular params.
      const shape = sampleNaturalShapeFrom(data)
      updateNodeParams(selectedNodeId, {
        ...node.data.params, dataset: rel, features: [], target: '',
        ...(shape ? { shape } : {}),
      })
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
              <span className="text-[#6f767e]">
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
            <span className="text-[#6f767e]">{value}</span>
            <button
              onClick={() => select(value)}
              className="text-[var(--accent)] hover:underline"
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

// Like ShapeInput, but allows negatives (e.g. -1 for reshape) and zero (dim
// indices for permute/transpose). Used by 'int-list' fields.
function IntListInput({
  value, onChange, baseClass,
}: { value: unknown; onChange: (v: unknown) => void; baseClass: string }) {
  const arr = (Array.isArray(value) ? value : []) as number[]
  const canonical = arr.join(', ')
  const [draft, setDraft] = useState(canonical)
  useEffect(() => { setDraft(canonical) }, [canonical])

  const parsed = parseIntList(draft)
  const valid = parsed !== null && parsed.length > 0

  return (
    <div className="flex flex-col gap-0.5">
      <input
        type="text"
        className={`${baseClass} ${valid ? '' : 'border-amber-700/60'}`}
        value={draft}
        placeholder="e.g. -1, 256"
        onChange={(e) => {
          setDraft(e.target.value)
          const p = parseIntList(e.target.value)
          if (p && p.length > 0) onChange(p)
        }}
        onBlur={() => {
          const p = parseIntList(draft)
          if (p && p.length > 0) {
            onChange(p)
            setDraft(p.join(', '))
          } else {
            setDraft(canonical)
          }
        }}
      />
      {!valid && draft.trim() !== '' && (
        <span className="text-[10px] text-amber-400">comma-separated ints (-1 allowed)</span>
      )}
    </div>
  )
}

function parseIntList(s: string): number[] | null {
  const parts = s.split(/[,\s]+/).map((p) => p.trim()).filter(Boolean)
  if (parts.length === 0) return null
  const out: number[] = []
  for (const p of parts) {
    const n = parseInt(p, 10)
    if (!Number.isFinite(n) || String(n) !== p) return null
    out.push(n)
  }
  return out
}
