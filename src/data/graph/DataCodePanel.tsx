import { useMemo } from 'react'
import Editor from '@monaco-editor/react'
import { useDataGraphStore } from './store'
import { compileDataGraph } from '../../codegen/dataGenerator'
import { generateDataCode } from '../../codegen/dataCodegen'

// Panel-form code view for the DATA canvas — shown in the unified bottom Code tab
// when the data canvas is active (mirrors the architecture CodePreview layout).
export default function DataCodePanel() {
  const nodes = useDataGraphStore((s) => s.nodes)
  const edges = useDataGraphStore((s) => s.edges)

  const { code, ok, issues } = useMemo(() => {
    const compile = compileDataGraph({
      nodes: nodes.map((n) => ({ id: n.id, dataType: n.data.dataType, params: n.data.params })),
      edges: edges.map((e) => ({ source: e.source, target: e.target })),
    })
    return { code: generateDataCode(compile.plan), ok: compile.ok, issues: compile.issues }
  }, [nodes, edges])

  const statusLabel = ok ? 'clean' : `${issues.length} issue${issues.length === 1 ? '' : 's'}`
  const statusColor = ok ? 'text-emerald-400' : 'text-amber-400'

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex h-7 shrink-0 items-center justify-between border-b border-[#1f2429] px-3 text-xs uppercase tracking-wide text-[#6f767e]">
        <span>Data Pipeline (Python)</span>
        <span className={`text-[10px] normal-case ${statusColor}`}>{statusLabel}</span>
      </div>
      <div className="min-h-0 flex-1">
        <Editor
          height="100%"
          defaultLanguage="python"
          value={code}
          theme="vs-dark"
          options={{ readOnly: true, minimap: { enabled: false }, fontSize: 12, scrollBeyondLastLine: false, wordWrap: 'on' }}
        />
      </div>
    </div>
  )
}
