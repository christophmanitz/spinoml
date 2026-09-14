import { useMemo } from 'react'
import Editor from '@monaco-editor/react'
import { useTrainingGraphStore, captureTrainingSnapshot } from './store'
import { compileTrainingGraph } from '../../codegen/trainingGenerator'
import { generateTrainingCode } from '../../codegen/trainingCodegen'

// Panel-form code view for the TRAINING canvas — shown in the unified bottom Code
// tab when the training canvas is active (replaces the old modal overlay).
export default function TrainingCodePanel() {
  const nodes = useTrainingGraphStore((s) => s.nodes)
  const edges = useTrainingGraphStore((s) => s.edges)

  const { code, ok, issues } = useMemo(() => {
    const snap = captureTrainingSnapshot({ nodes, edges })
    const compile = compileTrainingGraph(snap)
    return { code: generateTrainingCode(compile.plan), ok: compile.ok, issues: compile.issues }
  }, [nodes, edges])

  const statusLabel = ok ? 'startklar' : `${issues.length} offen`
  const statusColor = ok ? 'text-emerald-400' : 'text-amber-400'

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex h-7 shrink-0 items-center justify-between border-b border-[#1f2429] px-3 text-xs uppercase tracking-wide text-[#6f767e]">
        <span>Training Script (Python)</span>
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
