import { useMemo } from 'react'
import Editor from '@monaco-editor/react'
import { useGraphStore } from '../canvas/GraphStore'
import { generate } from './generator'

export default function CodePreview() {
  const nodes = useGraphStore((s) => s.nodes)
  const edges = useGraphStore((s) => s.edges)

  const { code, issues } = useMemo(() => generate(nodes, edges), [nodes, edges])

  return (
    <div className="h-[240px] shrink-0 border-t border-[#1f2429]">
      <div className="flex h-7 items-center justify-between border-b border-[#1f2429] px-3 text-xs uppercase tracking-wide text-[#7a8088]">
        <span>Generated PyTorch</span>
        <span
          className={`text-[10px] normal-case ${
            issues.length ? 'text-amber-400' : 'text-emerald-400'
          }`}
        >
          {issues.length
            ? `${issues.length} issue${issues.length > 1 ? 's' : ''}`
            : 'clean'}
        </span>
      </div>
      <Editor
        height="calc(100% - 28px)"
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
  )
}
