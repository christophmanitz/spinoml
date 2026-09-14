import { useMemo } from 'react'
import Editor from '@monaco-editor/react'
import { useTrainingGraphStore, captureTrainingSnapshot } from './store'
import { compileTrainingGraph } from '../../codegen/trainingGenerator'
import { generateTrainingCode } from '../../codegen/trainingCodegen'

export default function TrainingCodePreview({ onClose }: { onClose: () => void }) {
  const nodes = useTrainingGraphStore((s) => s.nodes)
  const edges = useTrainingGraphStore((s) => s.edges)

  const { code, ok, issues } = useMemo(() => {
    const snap = captureTrainingSnapshot({ nodes, edges })
    const compile = compileTrainingGraph(snap)
    return { code: generateTrainingCode(compile.plan), ok: compile.ok, issues: compile.issues }
  }, [nodes, edges])

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6" onClick={onClose}>
      <div
        className="flex h-[80vh] w-[760px] max-w-full flex-col overflow-hidden rounded-lg border border-[#262c33] bg-[#0e1216] shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-center justify-between border-b border-[#1f2429] px-4 py-2.5">
          <span className="text-sm font-medium text-[#e6e8eb]">Training-Code</span>
          <div className="flex items-center gap-3">
            <span className={`text-[10px] ${ok ? 'text-emerald-400' : 'text-amber-400'}`}>
              {ok ? 'startklar' : `${issues.length} offen`}
            </span>
            <button
              className="rounded px-2 py-0.5 text-xs text-[#9aa1a8] hover:bg-[#1f2429] hover:text-[#e6e8eb]"
              onClick={onClose}
            >
              schließen ✕
            </button>
          </div>
        </div>
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
    </div>
  )
}
