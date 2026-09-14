import { useEffect, useState } from 'react'
import Editor from '@monaco-editor/react'
import { useWorkspaceStore, type Entry } from './store'
import { isTauri } from './tauri-fs'
import { fs } from '../connections/backend'

// Generic read-only viewer for ANY workspace file (.py, .json, .jsonl, .log,
// .csv, .txt, …) — not just the generated PyTorch twin. Reads the real file
// content from disk (Tauri: fs.read by relpath; browser: the in-memory entry).

const BINARY_EXT = new Set([
  'pt', 'pth', 'ckpt', 'bin', 'npy', 'npz', 'pkl', 'pyc', 'so', 'o', 'a',
  'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp', 'ico', 'pdf', 'zip', 'gz',
  'tar', '7z', 'parquet', 'feather', 'arrow', 'wav', 'mp3', 'mp4',
])

const LANG: Record<string, string> = {
  py: 'python', pyi: 'python', js: 'javascript', mjs: 'javascript', cjs: 'javascript',
  ts: 'typescript', tsx: 'typescript', jsx: 'javascript', json: 'json',
  md: 'markdown', sh: 'shell', bash: 'shell', yaml: 'yaml', yml: 'yaml',
  toml: 'ini', ini: 'ini', cfg: 'ini', rs: 'rust', html: 'html', css: 'css',
  sql: 'sql', xml: 'xml', c: 'c', cpp: 'cpp', h: 'cpp',
}

function extOf(name: string): string {
  const i = name.lastIndexOf('.')
  return i === -1 ? '' : name.slice(i + 1).toLowerCase()
}

export default function FileViewerModal({ fileId, onClose }: { fileId: string; onClose: () => void }) {
  const entry = useWorkspaceStore((s) => s.entries[fileId]) as Entry | undefined
  const name = entry?.kind === 'file' ? entry.name : fileId.split('/').pop() ?? fileId
  const ext = extOf(name)
  const binary = BINARY_EXT.has(ext)

  const [text, setText] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  useEffect(() => {
    if (binary) return
    let cancelled = false
    ;(async () => {
      try {
        // Tauri: fileId is the workspace-relative path. Browser: read the
        // in-memory content (no disk).
        const content = isTauri()
          ? await fs.read(fileId)
          : (entry?.kind === 'file' ? entry.content : '')
        if (!cancelled) { setText(content); setError(null) }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e))
      }
    })()
    return () => { cancelled = true }
  }, [fileId, binary, entry])

  const copy = async () => {
    if (text == null) return
    try { await navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1200) } catch { /* ignore */ }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-6" onClick={onClose}>
      <div
        className="flex h-[80vh] w-[min(960px,90vw)] flex-col overflow-hidden rounded border border-[#1f2429] bg-[#0e1216] shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-center justify-between border-b border-[#1f2429] px-3 py-2">
          <div className="flex min-w-0 items-baseline gap-2">
            <span className="truncate font-mono text-sm text-[#e6e8eb]">{name}</span>
            {isTauri() && <span className="truncate text-[10px] text-[#5b6168]">{fileId}</span>}
          </div>
          <div className="flex items-center gap-1 text-xs">
            {!binary && (
              <button className="rounded border border-[#1f2429] bg-[#13171b] px-2 py-1 hover:bg-[#1a1f24]" onClick={copy}>
                {copied ? 'copied' : 'Copy'}
              </button>
            )}
            <button className="rounded border border-[#1f2429] bg-[#13171b] px-2 py-1 hover:bg-[#1a1f24]" onClick={onClose} title="Esc">✕</button>
          </div>
        </div>
        <div className="min-h-0 flex-1">
          {binary ? (
            <div className="flex h-full items-center justify-center px-6 text-center text-[12px] text-[#6f767e]">
              <div>
                <div className="mb-1 text-[#9aa1a8]">Binärdatei (.{ext})</div>
                <div>Keine Textvorschau verfügbar.</div>
              </div>
            </div>
          ) : error ? (
            <div className="flex h-full items-center justify-center px-6 text-center text-[12px] text-[#ff7a85]">{error}</div>
          ) : text == null ? (
            <div className="flex h-full items-center justify-center text-[12px] text-[#6f767e]">lade…</div>
          ) : (
            <Editor
              height="100%"
              language={LANG[ext] ?? 'plaintext'}
              value={text}
              theme="vs-dark"
              options={{ readOnly: true, minimap: { enabled: false }, fontSize: 13, scrollBeyondLastLine: false, wordWrap: 'on' }}
            />
          )}
        </div>
      </div>
    </div>
  )
}
