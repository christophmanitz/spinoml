import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import Editor, { type OnMount } from '@monaco-editor/react'

type MonacoEditor = Parameters<OnMount>[0]

// Monaco options tuned for writing a small nn.Module by hand: Python highlighting,
// 4-space indent, bracket auto-close, line numbers. Shared by inline + modal.
const MONACO_OPTIONS = {
  fontSize: 12,
  lineNumbers: 'on' as const,
  minimap: { enabled: false },
  scrollBeyondLastLine: false,
  wordWrap: 'off' as const,
  tabSize: 4,
  insertSpaces: true,
  detectIndentation: false,
  automaticLayout: true,
  autoClosingBrackets: 'always' as const,
  autoIndent: 'full' as const,
  bracketPairColorization: { enabled: true },
  renderWhitespace: 'selection' as const,
  scrollbar: { horizontalScrollbarSize: 8, verticalScrollbarSize: 8 },
  padding: { top: 8, bottom: 8 },
  fixedOverflowWidgets: true,
}

/**
 * Code editor for a Custom node's `source` field. UNCONTROLLED on purpose:
 * Monaco owns the buffer (defaultValue + refs, no per-keystroke React state),
 * which avoids the cursor-fighting / flicker you get from a controlled `value`
 * in a frequently re-rendering panel. The buffer commits to the GraphStore on
 * blur and on modal-close, so typing doesn't spam undo history or shape
 * inference.
 */
export default function CodeField({
  value, placeholder, onChange,
}: {
  value: string
  placeholder?: string
  onChange: (v: string) => void
}) {
  const editorRef = useRef<MonacoEditor | null>(null)
  // Seed text for the fullscreen modal, captured (in an event handler, not
  // render) when the user opens it. null = closed.
  const [modalSeed, setModalSeed] = useState<string | null>(null)

  const commit = () => {
    const v = editorRef.current?.getValue()
    if (v !== undefined && v !== value) onChange(v)
  }

  return (
    <div className="overflow-hidden rounded border border-[#1f2429] focus-within:border-[#3a4148]">
      <div className="flex items-center justify-between border-b border-[#1f2429] bg-[#0b0e11] px-2 py-1 text-[10px] text-[#7a8088]">
        <span className="font-mono">python</span>
        <button
          onClick={() => setModalSeed(editorRef.current?.getValue() ?? value ?? '')}
          className="rounded px-1.5 py-0.5 text-[#9aa1a8] hover:bg-[#1a1e22] hover:text-[#e6e8eb]"
          title="Im großen Editor öffnen"
        >
          ⤢ Vollbild
        </button>
      </div>
      <div className="h-[260px]">
        <Editor
          height="100%"
          defaultLanguage="python"
          theme="vs-dark"
          defaultValue={value ?? ''}
          options={MONACO_OPTIONS}
          onMount={(editor) => {
            editorRef.current = editor
            editor.onDidBlurEditorText(() => commit())
          }}
        />
      </div>
      {!value?.trim() && placeholder && (
        <div className="border-t border-[#1f2429] px-2 py-1 font-mono text-[10px] text-[#5b6168]">{placeholder}</div>
      )}

      {modalSeed !== null && (
        <CodeModal
          initial={modalSeed}
          onClose={(next) => {
            setModalSeed(null)
            if (next === undefined) return
            editorRef.current?.setValue(next) // keep the inline editor in sync
            if (next !== value) onChange(next)
          }}
        />
      )}
    </div>
  )
}

function CodeModal({
  initial, onClose,
}: {
  initial: string
  onClose: (next?: string) => void
}) {
  const editorRef = useRef<MonacoEditor | null>(null)
  const close = () => onClose(editorRef.current?.getValue() ?? initial)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close()
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 's') { e.preventDefault(); close() }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Portal to body so React Flow's transformed ancestors / panel overflow can't
  // clip or flicker the overlay.
  return createPortal(
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 p-6" onMouseDown={close}>
      <div
        className="flex h-[85vh] w-[85vw] max-w-[1100px] flex-col overflow-hidden rounded-lg border border-[#2a2f36] bg-[#0e1216] shadow-2xl"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-center justify-between border-b border-[#1f2429] px-4 py-2 text-xs text-[#9aa1a8]">
          <span className="font-mono">Custom code · python</span>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-[#5b6168]">⌘S / Esc zum Schließen</span>
            <button
              onClick={close}
              className="rounded bg-[#13344f] px-2.5 py-1 text-[11px] text-[#6ab7ff] hover:bg-[#184466]"
            >
              Fertig
            </button>
          </div>
        </div>
        <div className="min-h-0 flex-1">
          <Editor
            height="100%"
            defaultLanguage="python"
            theme="vs-dark"
            defaultValue={initial}
            options={{ ...MONACO_OPTIONS, fontSize: 13, minimap: { enabled: true } }}
            onMount={(editor) => { editorRef.current = editor; editor.focus() }}
          />
        </div>
      </div>
    </div>,
    document.body,
  )
}
