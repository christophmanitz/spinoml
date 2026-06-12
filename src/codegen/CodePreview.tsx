import Editor from '@monaco-editor/react'
import { useGraphStore } from '../canvas/GraphStore'
import { generatePyTorchCode } from './generator'

export default function CodePreview() {
  const nodes = useGraphStore((s) => s.nodes)
  const edges = useGraphStore((s) => s.edges)
  const code = generatePyTorchCode(nodes, edges)

  return (
    <div className="h-[220px] shrink-0 border-t border-[#1f2429]">
      <div className="flex h-7 items-center justify-between border-b border-[#1f2429] px-3 text-xs uppercase tracking-wide text-[#7a8088]">
        <span>Generated PyTorch</span>
        <span className="text-[10px] normal-case">phase 2 will fully populate this</span>
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
        }}
      />
    </div>
  )
}
