import { useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'

import { trust } from './trustStore'
import { useApproveDialog } from './useApproveDialog'
import type { CodeKind } from './codeBlobs'

const KIND_LABEL: Record<CodeKind, string> = {
  'custom-layer': 'Quellcode',
  'custom-init-args': 'Konstruktor-Argumente',
  'dataop-script': 'DataOp-Skript',
  'data-custom-script': 'Data-Pipeline-Skript',
}

const PREVIEW_LINES = 40

/** Phase 43 — the human approval surface. Lists every code blob that is about
 *  to run but has no local approval, shows its first 40 lines as PLAIN TEXT
 *  (never HTML), and is the only place that approves with 'user-approval'.
 *  "Abbrechen" is the default focused control; Esc closes. */
export default function ApproveCodeDialog() {
  const open = useApproveDialog((s) => s.open)
  const blobs = useApproveDialog((s) => s.blobs)
  const dismiss = useApproveDialog((s) => s.dismiss)
  const close = useApproveDialog((s) => s.close)
  const cancelRef = useRef<HTMLButtonElement | null>(null)

  useEffect(() => {
    if (!open) return
    cancelRef.current?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') close()
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [open, close])

  if (!open || blobs.length === 0) return null

  const approve = (sha256: string) => {
    trust.approve(sha256, 'user-approval')
    dismiss(sha256)
  }
  const approveAll = () => {
    for (const blob of blobs) trust.approve(blob.sha256, 'user-approval')
    close()
  }

  return createPortal(
    <div
      className="fixed inset-0 z-[110] flex items-center justify-center bg-black/60 p-6"
      onMouseDown={close}
    >
      <div
        className="flex max-h-[88vh] w-full max-w-3xl flex-col overflow-hidden rounded-lg border border-[#2a2f36] bg-[#0e1216] shadow-2xl"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="flex shrink-0 items-center justify-between border-b border-[#1f2429] px-4 py-2.5">
          <span className="text-sm font-medium text-amber-300">
            Code nicht freigegeben ({blobs.length})
          </span>
          <button
            onClick={close}
            className="rounded px-2 py-0.5 text-xs text-[#9aa1a8] hover:bg-[#1f2429] hover:text-[#e6e8eb]"
          >×</button>
        </div>

        <div className="min-h-0 flex-1 space-y-3 overflow-auto px-4 py-3">
          <div className="rounded border border-amber-900/60 bg-amber-950/30 px-3 py-2 text-[11px] leading-snug text-amber-200">
            Dieser Code wurde in SpinoML weder von dir geschrieben noch freigegeben und läuft
            mit deinen Rechten. Prüfe ihn, bevor du ihn freigibst.
          </div>

          {blobs.map((blob) => {
            const lines = blob.source.split('\n')
            const preview = lines.slice(0, PREVIEW_LINES).join('\n')
            return (
              <div key={blob.sha256} className="rounded border border-[#1f2429] bg-[#0b0e11]">
                <div className="flex flex-wrap items-center gap-x-2 gap-y-1 border-b border-[#1f2429] px-3 py-2 text-[11px]">
                  <span className="rounded bg-[#1a1e22] px-1.5 py-0.5 font-mono text-[10px] text-[#9aa1a8]">
                    {KIND_LABEL[blob.kind]}
                  </span>
                  <span className="font-mono text-[#cdd3da]">{blob.path}</span>
                  <span className="text-[#6f767e]">{lines.length} Zeile(n)</span>
                  <button
                    onClick={() => approve(blob.sha256)}
                    className="ml-auto rounded bg-amber-900/40 px-2.5 py-1 text-[11px] text-amber-200 hover:bg-amber-900/60"
                  >Freigeben</button>
                </div>
                <pre className="max-h-[280px] overflow-auto whitespace-pre px-3 py-2 font-mono text-[11px] leading-snug text-[#cfd3d8]">{preview}</pre>
                {lines.length > PREVIEW_LINES && (
                  <div className="border-t border-[#1f2429] px-3 py-1 text-[10px] text-[#6f767e]">
                    … {lines.length - PREVIEW_LINES} weitere Zeile(n) — vollständig im Editor sichtbar.
                  </div>
                )}
              </div>
            )
          })}
        </div>

        <div className="flex shrink-0 items-center justify-end gap-2 border-t border-[#1f2429] px-4 py-3">
          <button
            ref={cancelRef}
            onClick={close}
            className="rounded px-3 py-1 text-[12px] text-[#9aa1a8] hover:text-[#e6e8eb]"
          >Abbrechen</button>
          <button
            onClick={approveAll}
            className="rounded bg-amber-900/40 px-3 py-1 text-[12px] text-amber-200 hover:bg-amber-900/60"
          >Alle freigeben</button>
        </div>
      </div>
    </div>,
    document.body,
  )
}
