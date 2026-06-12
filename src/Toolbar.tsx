import { useEffect, useRef, useState } from 'react'
import { useHistoryStore } from './history/store'
import { useGraphStore } from './canvas/GraphStore'
import { downloadCurrent, pickAndLoad, clearAutosave } from './persistence/file'
import { TEMPLATES } from './templates/templates'
import { useWorkspaceStore, ROOT_ID } from './workspace/store'

export default function Toolbar() {
  const activeFileName = useWorkspaceStore((s) =>
    s.activeFileId ? s.entries[s.activeFileId]?.name ?? null : null,
  )
  const dirty = useWorkspaceStore((s) => s.dirty)

  const saveToWorkspace = () => {
    const ws = useWorkspaceStore.getState()
    if (ws.activeFileId) {
      ws.saveActive()
      return
    }
    const name = prompt('Save as (in models/):', 'untitled.mlforge')
    if (!name) return
    ws.saveAsNew(ROOT_ID, name)
  }

  return (
    <div className="flex items-center gap-1 text-xs">
      <Menu label="File">
        <Item onSelect={() => {
          if (confirm('Reset graph? Current model will be lost (Cmd+Z to undo).')) {
            useGraphStore.getState().resetGraph()
            useWorkspaceStore.getState().closeActive()
            clearAutosave()
          }
        }}>New</Item>
        <Item onSelect={saveToWorkspace} hint={activeFileName ? `→ ${activeFileName}` : 'new file'}>
          {activeFileName ? `Save${dirty ? ' (●)' : ''}` : 'Save…'}
        </Item>
        <div className="my-1 h-px bg-[#1f2429]" />
        <Item onSelect={() => pickAndLoad((snap) => {
          useGraphStore.getState().loadSnapshot(snap)
          useWorkspaceStore.getState().closeActive()
        })}>
          Open from disk…
        </Item>
        <Item onSelect={() => downloadCurrent(activeFileName ?? 'model.mlforge')}>
          Export to disk…
        </Item>
      </Menu>

      <Menu label="Edit">
        <Item
          onSelect={() => useHistoryStore.getState().undo()}
          disabled={!useHistoryStore((s) => s.canUndo)}
          hint="⌘Z"
        >Undo</Item>
        <Item
          onSelect={() => useHistoryStore.getState().redo()}
          disabled={!useHistoryStore((s) => s.canRedo)}
          hint="⌘⇧Z"
        >Redo</Item>
      </Menu>

      <Menu label="Templates">
        {TEMPLATES.map((t) => (
          <Item
            key={t.id}
            onSelect={() => {
              if (useGraphStore.getState().nodes.length > 1 &&
                  !confirm(`Replace current graph with "${t.name}"? (Cmd+Z to undo)`)) return
              useGraphStore.getState().loadSnapshot(t.build())
              useGraphStore.getState().autoLayout()
              useWorkspaceStore.getState().closeActive()
            }}
            hint={t.description}
          >{t.name}</Item>
        ))}
      </Menu>
    </div>
  )
}

function Menu({ label, children }: { label: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onDoc)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDoc)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  return (
    <div className="relative" ref={ref}>
      <button
        className={`rounded px-2 py-0.5 text-[#9aa1a8] hover:bg-[#1f2429] hover:text-[#e6e8eb] ${
          open ? 'bg-[#1f2429] text-[#e6e8eb]' : ''
        }`}
        onClick={() => setOpen((v) => !v)}
      >
        {label}
      </button>
      {open && (
        <div
          className="absolute left-0 top-full z-50 mt-1 min-w-[220px] rounded border border-[#1f2429] bg-[#13171b] p-1 shadow-lg"
          onClick={() => setOpen(false)}
        >
          {children}
        </div>
      )}
    </div>
  )
}

function Item({
  children, onSelect, disabled, hint,
}: { children: React.ReactNode; onSelect: () => void; disabled?: boolean; hint?: string }) {
  return (
    <button
      className="flex w-full items-center justify-between gap-3 rounded px-2 py-1 text-left text-xs text-[#e6e8eb] hover:bg-[#1f2429] disabled:cursor-not-allowed disabled:text-[#5b6168] disabled:hover:bg-transparent"
      onClick={(e) => { if (disabled) return; e.stopPropagation(); onSelect() }}
      disabled={disabled}
    >
      <span>{children}</span>
      {hint && <span className="text-[10px] text-[#7a8088]">{hint}</span>}
    </button>
  )
}
