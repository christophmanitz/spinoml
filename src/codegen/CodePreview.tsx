import { useMemo } from 'react'
import Editor from '@monaco-editor/react'
import { useGraphStore } from '../canvas/GraphStore'
import { generate } from './generator'
import { useInferenceStore } from '../inference/store'

export default function CodePreview() {
  const nodes = useGraphStore((s) => s.nodes)
  const edges = useGraphStore((s) => s.edges)
  const inferenceError = useInferenceStore((s) => s.error)
  const inferenceStage = useInferenceStore((s) => s.errorStage)
  const inferenceStatus = useInferenceStore((s) => s.status)

  const { code, issues } = useMemo(() => generate(nodes, edges), [nodes, edges])

  const showRuntimeError = inferenceStatus === 'error' && inferenceError
  const statusLabel = showRuntimeError
    ? `runtime: ${inferenceStage ?? 'error'}`
    : issues.length
      ? `${issues.length} issue${issues.length > 1 ? 's' : ''}`
      : 'clean'

  const statusColor = showRuntimeError
    ? 'text-rose-400'
    : issues.length
      ? 'text-amber-400'
      : 'text-emerald-400'

  return (
    <div className="flex h-[260px] shrink-0 flex-col border-t border-[#1f2429]">
      <div className="flex h-7 shrink-0 items-center justify-between border-b border-[#1f2429] px-3 text-xs uppercase tracking-wide text-[#7a8088]">
        <span>Generated PyTorch</span>
        <span className={`text-[10px] normal-case ${statusColor}`}>{statusLabel}</span>
      </div>
      {showRuntimeError && (
        <div className="shrink-0 border-b border-[#1f2429] bg-rose-950/30 px-3 py-1.5 font-mono text-[10px] leading-snug text-rose-300">
          {inferenceError}
        </div>
      )}
      <div className="min-h-0 flex-1">
        <Editor
          height="100%"
          defaultLanguage="python"
          value={code}
          theme="vs-dark"
          options={{
            readOnly: true,
            minimap: { enabled: false },
            fontSize: 12,
            scrollBeyondLastLine: false,
            wordWrap: 'on',
          }}
        />
      </div>
    </div>
  )
}
