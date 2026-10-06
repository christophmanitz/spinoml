import { useEffect, useRef, useState } from 'react'
import { useHistoryStore } from './history/store'
import { useGraphStore } from './canvas/GraphStore'
import { useScopeStore } from './canvas/scopeStore'
import { downloadCurrent, pickAndLoad, clearAutosave } from './persistence/file'
import { TEMPLATES } from './templates/templates'
import { collectCodeBlobs, hashBlob } from './trust/codeBlobs'
import { trust } from './trust/trustStore'
import { useWorkspaceStore, ROOT_ID } from './workspace/store'
import { isTauri } from './workspace/tauri-fs'
import { confirmDialog } from './ui/confirm'
import DependenciesModal from './deps/DependenciesModal'

export default function Toolbar() {
  const activeFileName = useWorkspaceStore((s) =>
    s.activeFileId ? s.entries[s.activeFileId]?.name ?? null : null,
  )
  const dirty = useWorkspaceStore((s) => s.dirty)
  const mode = useWorkspaceStore((s) => s.mode)
  const workspaceRoot = useWorkspaceStore((s) => s.workspaceRoot)
  const tauriAvailable = isTauri()
  const [depsOpen, setDepsOpen] = useState(false)

  const saveToWorkspace = async () => {
    const ws = useWorkspaceStore.getState()
    if (ws.activeFileId) {
      try { await ws.saveActive() }
      catch (e) { await reportError('Save failed', e) }
      return
    }
    // No active file: in Tauri webview window.prompt is unreliable, so just
    // auto-create with a default name. User can inline-rename in the tree.
    // When a project is loaded, default to models/; otherwise to root.
    const parentId = ws.entries['models']?.kind === 'folder' ? 'models' : ROOT_ID
    try {
      const id = await ws.saveAsNew(parentId, 'untitled.spinoml')
      if (!id) await reportError('Save failed', 'workspace did not return a file id')
    } catch (e) {
      await reportError('Save failed', e)
    }
  }

  return (
    <>
    <div className="flex items-center gap-1 text-xs">
      <Menu label="File">
        <Item onSelect={async () => {
          if (await confirmDialog('Reset graph? Current model will be lost (Cmd+Z to undo).')) {
            useScopeStore.getState().reset()
            useGraphStore.getState().resetGraph()
            useWorkspaceStore.getState().closeActive()
            clearAutosave()
          }
        }}>New</Item>
        <Item onSelect={saveToWorkspace} hint={activeFileName ? `→ ${activeFileName}` : 'new file'}>
          {activeFileName ? `Save${dirty ? ' (●)' : ''}` : 'Save…'}
        </Item>
        <div className="my-1 h-px bg-[#1f2429]" />
        {tauriAvailable && (
          <>
            <Item
              onSelect={() => { useWorkspaceStore.getState().openDirectory() }}
              hint={mode === 'tauri' && workspaceRoot ? truncatePath(workspaceRoot) : 'pick a folder'}
            >Open folder…</Item>
            {mode === 'tauri' && (
              <Item onSelect={() => { useWorkspaceStore.getState().closeDirectory() }}>
                Close folder
              </Item>
            )}
            <div className="my-1 h-px bg-[#1f2429]" />
          </>
        )}
        <Item onSelect={() => pickAndLoad((snap) => {
          useScopeStore.getState().reset()
          if (!useGraphStore.getState().loadSnapshot(snap)) {
            alert('Couldn\'t load: this file contains an invalid model (graph validation failed). It was not opened.')
            return
          }
          useWorkspaceStore.getState().closeActive()
        })}>
          Open file from disk…
        </Item>
        <Item onSelect={() => downloadCurrent(activeFileName ?? 'model.spinoml')}>
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
            onSelect={async () => {
              if (useGraphStore.getState().nodes.length > 1 &&
                  !(await confirmDialog(`Replace current graph with "${t.name}"? (Cmd+Z to undo)`))) return
              const snap = t.build()
              // Phase 43 — templates ship WITH the app, so inserting one is a
              // legitimate origin: approve each Custom/DataOp blob as 'template'.
              for (const blob of collectCodeBlobs(snap.nodes)) {
                trust.approve(await hashBlob(blob.kind, blob.source), 'template')
              }
              useScopeStore.getState().reset()
              useGraphStore.getState().loadSnapshot(snap)
              useGraphStore.getState().autoLayout()
              useWorkspaceStore.getState().closeActive()
            }}
            hint={t.description}
          >{t.name}</Item>
        ))}
      </Menu>

      <Menu label="Project">
        <Item onSelect={() => setDepsOpen(true)} hint="requirements.txt">
          Dependencies…
        </Item>
      </Menu>
    </div>
    {depsOpen && <DependenciesModal onClose={() => setDepsOpen(false)} />}
    </>
  )
}

function truncatePath(p: string): string {
  if (p.length <= 40) return p
  return '…' + p.slice(-37)
}

async function reportError(title: string, e: unknown): Promise<void> {
  const msg = e instanceof Error ? e.message : String(e)
  try {
    const { message } = await import('@tauri-apps/plugin-dialog')
    await message(msg, { title, kind: 'error' })
  } catch {
    alert(`${title}\n${msg}`)
  }
  console.error(title, e)
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
      {hint && <span className="text-[10px] text-[#6f767e]">{hint}</span>}
    </button>
  )
}
