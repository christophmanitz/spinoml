import { useEffect, useMemo, useState } from 'react'
import Editor from '@monaco-editor/react'
import { useWorkspaceStore } from './store'
import { parseFile } from '../persistence/file'
import { generateFromSnapshot } from '../codegen/generator'

export type PyPreview = { fileId: string; pyName: string }

export default function PyCodeModal({
  preview, onClose,
}: { preview: PyPreview; onClose: () => void }) {
  const file = useWorkspaceStore((s) => s.entries[preview.fileId])
  const [copied, setCopied] = useState(false)

  const { code, issues } = useMemo(() => {
    if (!file || file.kind !== 'file') return { code: '', issues: [] }
    try {
      const snap = parseFile(file.content)
      return generateFromSnapshot(snap)
    } catch (e) {
      return { code: `# couldn't generate: ${(e as Error).message}\n`, issues: [] }
    }
  }, [file])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  if (!file || file.kind !== 'file') return null

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1200)
    } catch { /* ignore */ }
  }

  const download = () => {
    const blob = new Blob([code], { type: 'text/x-python' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = preview.pyName
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6"
      onClick={onClose}
    >
      <div
        className="flex h-[80vh] w-[min(960px,90vw)] flex-col overflow-hidden rounded border border-[#1f2429] bg-[#0e1216] shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-center justify-between border-b border-[#1f2429] px-3 py-2">
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-sm text-[#e6e8eb]">{preview.pyName}</span>
            <span className="text-[10px] text-[#7a8088]">generated from {file.name}</span>
            {issues.length > 0 && (
              <span className="text-[10px] text-amber-400">{issues.length} issue{issues.length > 1 ? 's' : ''}</span>
            )}
          </div>
          <div className="flex items-center gap-1 text-xs">
            <button
              className="rounded border border-[#1f2429] bg-[#13171b] px-2 py-1 hover:bg-[#1a1f24]"
              onClick={copy}
            >{copied ? 'copied' : 'Copy'}</button>
            <button
              className="rounded border border-[#1f2429] bg-[#13171b] px-2 py-1 hover:bg-[#1a1f24]"
              onClick={download}
            >Download</button>
            <button
              className="rounded border border-[#1f2429] bg-[#13171b] px-2 py-1 hover:bg-[#1a1f24]"
              onClick={onClose}
              title="Esc"
            >✕</button>
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
              fontSize: 13,
              scrollBeyondLastLine: false,
              wordWrap: 'on',
            }}
          />
        </div>
      </div>
    </div>
  )
}
