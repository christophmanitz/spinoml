import { useState } from 'react'
import { useReactFlow } from '@xyflow/react'

import { useCanvasDocStore, type CanvasDocAdapter } from './store'

// Re-reads the canvas's bound file from disk and reloads it into the graph.
// Use after something OUTSIDE the canvas changed the file — most often the
// chatbot writing/editing the .spinoml/.spinotrain/.spinodata via write_file.
// (Live in-graph tool mutations already show instantly; this is for on-disk
// changes the in-memory canvas hasn't picked up.)
export default function ReloadCanvasButton({ adapter }: { adapter: CanvasDocAdapter }) {
  const { fitView } = useReactFlow()
  const rel = useCanvasDocStore((s) => s.docs[adapter.kind].relpath)
  const [busy, setBusy] = useState(false)

  return (
    <button
      disabled={busy || !rel}
      onClick={async () => {
        if (!rel) return
        setBusy(true)
        try {
          await adapter.open(rel)
          setTimeout(() => fitView({ duration: 200, padding: 0.15 }), 0)
        } catch (e) {
          // adapter.open() re-reads the bound file; a read/parse failure must be
          // reported or the canvas silently keeps showing stale content.
          alert(`Neu laden fehlgeschlagen (${rel}):\n${e instanceof Error ? e.message : String(e)}`)
        }
        finally { setBusy(false) }
      }}
      className="rounded border border-[#1f2429] bg-[#13171b] px-2 py-1 text-[11px] text-[#9aa1a8] hover:border-[#3a4148] hover:bg-[#1a1f24] hover:text-[#e6e8eb] disabled:cursor-not-allowed disabled:opacity-40"
      title="Canvas aus der Datei neu laden (z. B. nachdem der Chatbot sie geändert hat)"
    >
      {busy ? '…' : '↻ Reload'}
    </button>
  )
}
