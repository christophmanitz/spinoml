import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import Editor, { type OnMount } from '@monaco-editor/react'
import { shouldFlushCode, modalEditedByUser, type EditMeta } from './editMeta'

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
 * Code editor for a Custom node's `source` field. UNCONTROLLED (defaultValue +
 * Monaco owns the buffer) so the cursor never fights React, BUT the live text is
 * mirrored into a ref via onChange — we never call getValue() on a disposing
 * editor (that returns '' and used to wipe the source on node switch). Commits
 * are debounced and flushed on unmount so edits survive switching nodes.
 */
export default function CodeField({
  value, placeholder, onChange,
}: {
  value: string
  placeholder?: string
  onChange: (v: string, meta?: EditMeta) => void
}) {
  const inlineRef = useRef<MonacoEditor | null>(null)
  const latest = useRef(value ?? '')          // live editor text (from onChange)
  const valueRef = useRef(value ?? '')        // last value we know the store holds
  // Keep the store-value mirror in sync AFTER commit (not during render, where
  // a concurrent/aborted render could leak an uncommitted prop value into a
  // flush decision).
  useEffect(() => { valueRef.current = value ?? '' }, [value])
  // Whether the CURRENT inline buffer came from a real user edit. A programmatic
  // setValue (isFlush === true) must never flip this — that is how an LLM/store
  // write under an uncontrolled editor could otherwise get auto-approved.
  const edited = useRef(false)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [modalSeed, setModalSeed] = useState<string | null>(null)

  const flush = () => {
    if (timer.current) { clearTimeout(timer.current); timer.current = null }
    if (shouldFlushCode(edited.current, latest.current, valueRef.current)) {
      onChange(latest.current, { userEdited: true })
    }
    edited.current = false
  }
  // Flush any pending USER edit when the field unmounts (e.g. selecting another
  // node). An unedited stale buffer is never written back (see editMeta).
  useEffect(() => () => flush(), []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleChange = (v: string | undefined, ev?: { isFlush?: boolean }) => {
    // Only a real content change (isFlush === false) counts as a user edit.
    if (ev && ev.isFlush === false) edited.current = true
    latest.current = v ?? ''
    if (timer.current) clearTimeout(timer.current)
    timer.current = setTimeout(flush, 500)
  }

  return (
    <div className="overflow-hidden rounded border border-[#1f2429] focus-within:border-[#3a4148]">
      <div className="flex items-center justify-between border-b border-[#1f2429] bg-[#0b0e11] px-2 py-1 text-[10px] text-[#6f767e]">
        <span className="font-mono">python</span>
        <button
          onClick={() => setModalSeed(latest.current)}
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
          onChange={handleChange}
          onMount={(editor) => { inlineRef.current = editor }}
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
            const userEdited = modalEditedByUser(modalSeed, next)
            latest.current = next
            // Sync the inline editor; this setValue is a flush (isFlush === true)
            // and therefore must NOT mark the buffer as user-edited.
            inlineRef.current?.setValue(next)
            // Only a real user change in the modal is committed — closing without
            // typing must not write anything back (H3).
            if (userEdited && next !== valueRef.current) onChange(next, { userEdited: true })
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
  const latest = useRef(initial)
  const close = () => onClose(latest.current)

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
              className="rounded bg-[var(--accent-sel)] px-2.5 py-1 text-[11px] text-[var(--accent)] hover:bg-[var(--accent-sel-hover)]"
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
            onChange={(v) => { latest.current = v ?? '' }}
            onMount={(editor) => editor.focus()}
          />
        </div>
      </div>
    </div>,
    document.body,
  )
}
