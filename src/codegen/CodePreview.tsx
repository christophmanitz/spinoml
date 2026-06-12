import { useMemo } from 'react'
import Editor from '@monaco-editor/react'
import { useGraphStore } from '../canvas/GraphStore'
import { generate } from './generator'
import { useInferenceStore } from '../inference/store'

export default function CodePreview() {
  const nodes = useGraphStore((s) => s.nodes)
  const edges = useGraphStore((s) => s.edges)
  const setSelectedNodeId = useGraphStore((s) => s.setSelectedNodeId)
  const inferenceError = useInferenceStore((s) => s.error)
  const inferenceStage = useInferenceStore((s) => s.errorStage)
  const inferenceStatus = useInferenceStore((s) => s.status)
  const failingNodeId = useInferenceStore((s) => s.failingNodeId)
  const failingLayerType = useInferenceStore((s) => s.failingNodeLayerType)

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
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex h-7 shrink-0 items-center justify-between border-b border-[#1f2429] px-3 text-xs uppercase tracking-wide text-[#7a8088]">
        <span>Generated PyTorch</span>
        <span className={`text-[10px] normal-case ${statusColor}`}>{statusLabel}</span>
      </div>
      {showRuntimeError && (
        <div className="shrink-0 border-b border-[#1f2429] bg-rose-950/30 px-3 py-1.5 font-mono text-[10px] leading-snug text-rose-300">
          {failingNodeId && (
            <button
              className="mr-2 rounded bg-rose-900/60 px-1.5 py-0.5 text-[10px] text-rose-200 hover:bg-rose-800"
              onClick={() => setSelectedNodeId(failingNodeId)}
              title="select the failing node"
            >
              {failingLayerType ?? 'node'} #{failingNodeId} →
            </button>
          )}
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
