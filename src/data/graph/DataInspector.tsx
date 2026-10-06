import { useEffect, useMemo, useRef } from 'react'

import { isTauri } from '../../workspace/tauri-fs'
import { useDatasetsStore } from '../../datasets/store'
import { compileDataGraph } from '../../codegen/dataGenerator'
import { generateNodeCode } from '../../codegen/dataCodegen'
import { shouldCommitText } from '../../inspector/editMeta'
import { hashBlob } from '../../trust/codeBlobs'
import { trust } from '../../trust/trustStore'
import { useDataGraphStore } from './store'
import { DATA_NODES, type DataFieldSpec } from './registry'

const INPUT = 'w-full rounded border border-[#1f2429] bg-[#0b0e11] px-2 py-1 text-[12px] text-[#e6e8eb] focus:border-[var(--accent)] focus:outline-none'

export default function DataInspector() {
  const node = useDataGraphStore((s) => s.nodes.find((n) => n.id === s.selectedNodeId) ?? null)
  const updateNodeParams = useDataGraphStore((s) => s.updateNodeParams)
  const deleteNode = useDataGraphStore((s) => s.deleteNode)
  const convertToCustom = useDataGraphStore((s) => s.convertToCustom)
  const nodes = useDataGraphStore((s) => s.nodes)
  const edges = useDataGraphStore((s) => s.edges)
  // useMemo (not selector) — a selector returning a fresh object loops useSyncExternalStore.
  const compile = useMemo(
    () => compileDataGraph({
      nodes: nodes.map((n) => ({ id: n.id, dataType: n.data.dataType, params: n.data.params })),
      edges: edges.map((e) => ({ source: e.source, target: e.target })),
    }),
    [nodes, edges],
  )

  if (!node) {
    return (
      <div className="flex h-full flex-col">
        <CompilePanel compile={compile} />
        <div className="p-3 text-[11px] text-[#6f767e]">Knoten auswählen, um Parameter zu bearbeiten.</div>
      </div>
    )
  }

  const spec = DATA_NODES[node.data.dataType]
  if (!spec) return <div className="p-3 text-[11px] text-rose-300">Unbekannter Knoten: {node.data.dataType}</div>

  return (
    <div className="flex h-full flex-col overflow-auto">
      <CompilePanel compile={compile} />
      <div className="flex items-center gap-2 border-b border-[#1f2429] px-3 py-2">
        <span className="flex-1 text-[12px] font-medium text-[#e6e8eb]">{node.data.dataType}</span>
        {node.data.dataType !== 'CustomScript' && (
          <button
            onClick={() => convertToCustom(node.id, generateNodeCode(node.data.dataType, node.data.params), `${node.data.dataType} (custom)`)}
            className="rounded px-1.5 py-0.5 text-[11px] text-[#6f767e] hover:bg-[#1a1e22] hover:text-[var(--accent)]"
            title="In ein editierbares CustomScript umwandeln (Code bearbeitbar)"
          >→ Custom</button>
        )}
        <button
          onClick={() => deleteNode(node.id)}
          className="rounded px-1.5 py-0.5 text-[11px] text-[#6f767e] hover:bg-[#1a1e22] hover:text-[#ff7a85]"
        >löschen</button>
      </div>
      {spec.hint && <div className="px-3 pt-2 text-[10px] leading-snug text-[#6f767e]">{spec.hint}</div>}
      <div className="space-y-3 p-3">
        {spec.fields.map((field) => (
          <label key={field.name} className="block">
            <span className="mb-1 block text-[11px] text-[#6f767e]">{field.name}</span>
            <FieldInput
              field={field}
              value={node.data.params[field.name]}
              onChange={(v) => updateNodeParams(node.id, { [field.name]: v })}
              onApproveCode={
                node.data.dataType === 'CustomScript' && field.type === 'code'
                  ? (text: string) => { void hashBlob('data-custom-script', text).then((h) => trust.approve(h, 'human-edit')) }
                  : undefined
              }
            />
          </label>
        ))}
      </div>
    </div>
  )
}

function CompilePanel({ compile }: { compile: ReturnType<typeof compileDataGraph> }) {
  return (
    <div className="space-y-1.5 border-b border-[#1f2429] px-3 py-2 text-[11px]">
      {compile.ok ? (
        <span className="text-[#5fd39a]">✓ Pipeline kompiliert</span>
      ) : (
        <div className="space-y-0.5">
          <span className="text-[#e6c34a]">Pipeline-Probleme:</span>
          <ul className="ml-3 list-disc text-[#9aa1a8]">
            {compile.issues.map((m, i) => <li key={i}>{m}</li>)}
          </ul>
        </div>
      )}
      {compile.warnings.length > 0 && (
        <div className="space-y-0.5">
          <span className="text-[#c98b3a]">⚠ Hinweise:</span>
          <ul className="ml-3 list-disc text-[#9aa1a8]">
            {compile.warnings.map((m, i) => <li key={i}>{m}</li>)}
          </ul>
        </div>
      )}
      <div className="text-[10px] text-[#5a6068]">Skript unten im Code-Panel · „Pipeline ausführen" oben links auf dem Canvas.</div>
    </div>
  )
}

function FieldInput({
  field, value, onChange, onApproveCode,
}: {
  field: DataFieldSpec
  value: unknown
  onChange: (v: unknown) => void
  onApproveCode?: (text: string) => void
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
          className="h-4 w-4 accent-[var(--accent)]"
        />
      )
    case 'select':
      return (
        <select value={String(value ?? field.default)} onChange={(e) => onChange(e.target.value)} className={INPUT}>
          {field.options.map((o) => <option key={o} value={o}>{o}</option>)}
        </select>
      )
    case 'text':
      return (
        <input
          type="text"
          value={String(value ?? '')}
          placeholder={field.placeholder}
          onChange={(e) => onChange(e.target.value)}
          className={INPUT}
        />
      )
    case 'code':
      return (
        <CodeTextarea
          value={String(value ?? '')}
          onChange={onChange}
          onApproveCode={onApproveCode}
        />
      )
    case 'dataset-ref':
      return <DatasetRef value={String(value ?? '')} onChange={onChange} />
  }
}

/** Phase 43 (fix round 1) — the data-graph CustomScript body is a CONTROLLED
 *  textarea, so React fires its DOM `onChange` only for real user typing (a
 *  programmatic store write just re-renders the value). That makes it a safe
 *  `human-edit` approval origin: no draft/blur mirror exists to re-commit an
 *  LLM-written value. The settled text is approved ~500 ms after typing stops,
 *  and only when it actually differs from the value the edit started from. */
function CodeTextarea({
  value, onChange, onApproveCode,
}: {
  value: string
  onChange: (v: unknown) => void
  onApproveCode?: (text: string) => void
}) {
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const base = useRef<string | null>(null)
  useEffect(() => () => { if (timer.current) clearTimeout(timer.current) }, [])
  return (
    <textarea
      value={value}
      onChange={(e) => {
        const next = e.target.value
        if (!shouldCommitText(next, value)) return
        if (base.current === null) base.current = value
        onChange(next)
        if (!onApproveCode) return
        if (timer.current) clearTimeout(timer.current)
        timer.current = setTimeout(() => {
          const from = base.current
          base.current = null
          timer.current = null
          if (from !== null && shouldCommitText(next, from)) onApproveCode(next)
        }, 500)
      }}
      spellCheck={false}
      rows={10}
      className={`${INPUT} resize-y font-mono text-[11px] leading-snug`}
    />
  )
}

function DatasetRef({ value, onChange }: { value: string; onChange: (v: unknown) => void }) {
  const entries = useDatasetsStore((s) => s.entries)
  const refresh = useDatasetsStore((s) => s.refresh)
  useEffect(() => {
    if (isTauri() && entries.length === 0) void refresh()
  }, [refresh, entries.length])
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className={INPUT}>
      <option value="">— Datensatz —</option>
      {entries.map((e) => <option key={e.relpath} value={e.relpath}>{e.name}</option>)}
    </select>
  )
}
