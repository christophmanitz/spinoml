import { useEffect, useRef, useState } from 'react'
import { useWorkspaceStore, ROOT_ID, type Entry } from './store'

const DRAG_MIME = 'application/mlforge-workspace-entry'

export default function FileExplorer() {
  const root = useWorkspaceStore((s) => s.entries[ROOT_ID])
  const dirty = useWorkspaceStore((s) => s.dirty)
  const activeFileId = useWorkspaceStore((s) => s.activeFileId)
  const createFile = useWorkspaceStore((s) => s.createFile)
  const createFolder = useWorkspaceStore((s) => s.createFolder)
  const importFromText = useWorkspaceStore((s) => s.importFromText)
  const [rename, setRename] = useState<string | null>(null)
  const [menu, setMenu] = useState<{ id: string; x: number; y: number } | null>(null)
  const [dragOver, setDragOver] = useState<string | null>(null)

  useEffect(() => {
    if (!menu) return
    const close = () => setMenu(null)
    document.addEventListener('mousedown', close)
    document.addEventListener('scroll', close, true)
    return () => {
      document.removeEventListener('mousedown', close)
      document.removeEventListener('scroll', close, true)
    }
  }, [menu])

  const importFile = (parentId: string) => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.mlforge,.json,application/json'
    input.onchange = async () => {
      const file = input.files?.[0]
      if (!file) return
      try {
        importFromText(parentId, file.name, await file.text())
      } catch (e) {
        alert(`Couldn't import ${file.name}:\n${(e as Error).message}`)
      }
    }
    input.click()
  }

  const onRootDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(null)
    if (e.dataTransfer.types.includes(DRAG_MIME)) {
      const id = e.dataTransfer.getData(DRAG_MIME)
      if (id) useWorkspaceStore.getState().move(id, ROOT_ID)
      return
    }
    const file = e.dataTransfer.files?.[0]
    if (!file) return
    file.text().then((text) => {
      try { importFromText(ROOT_ID, file.name, text) }
      catch (err) { alert(`Couldn't import ${file.name}:\n${(err as Error).message}`) }
    })
  }

  return (
    <div className="flex h-full min-h-0 flex-col text-sm">
      <div className="flex shrink-0 items-center justify-between border-b border-[#1f2429] px-2 py-1">
        <div className="flex items-center gap-1.5 text-xs uppercase tracking-wide text-[#7a8088]">
          <span>Models</span>
          {dirty && activeFileId && <span className="text-amber-400" title="unsaved changes">●</span>}
        </div>
        <div className="flex items-center gap-1 text-[#7a8088]">
          <IconButton title="New file" onClick={() => {
            const id = createFile(ROOT_ID)
            if (id) setRename(id)
          }}>📄+</IconButton>
          <IconButton title="New folder" onClick={() => {
            const id = createFolder(ROOT_ID)
            if (id) setRename(id)
          }}>📁+</IconButton>
          <IconButton title="Import .mlforge" onClick={() => importFile(ROOT_ID)}>⇪</IconButton>
        </div>
      </div>

      <div
        className={`min-h-0 flex-1 overflow-y-auto py-1 ${
          dragOver === ROOT_ID ? 'bg-[#1f2429]/40' : ''
        }`}
        onDragOver={(e) => {
          if (e.dataTransfer.types.includes(DRAG_MIME) || e.dataTransfer.types.includes('Files')) {
            e.preventDefault()
            setDragOver(ROOT_ID)
          }
        }}
        onDragLeave={(e) => {
          if ((e.currentTarget as HTMLElement).contains(e.relatedTarget as Node)) return
          setDragOver(null)
        }}
        onDrop={onRootDrop}
      >
        {root && root.kind === 'folder' && root.childIds.length === 0 && (
          <div className="px-3 py-2 text-[10px] text-[#5b6168]">
            No saved models yet. Click <span className="text-[#9aa1a8]">📄+</span> to create
            one, or drop a <code>.mlforge</code> file here.
          </div>
        )}
        {root && root.kind === 'folder' && (
          <Tree
            parentId={ROOT_ID}
            depth={0}
            rename={rename}
            setRename={setRename}
            dragOver={dragOver}
            setDragOver={setDragOver}
            onContext={(id, x, y) => setMenu({ id, x, y })}
          />
        )}
      </div>

      {menu && <ContextMenu menu={menu}
        setRename={setRename}
        importFile={importFile}
        close={() => setMenu(null)} />}
    </div>
  )
}

function IconButton({ children, onClick, title }:
  { children: React.ReactNode; onClick: () => void; title: string }) {
  return (
    <button
      title={title}
      className="rounded px-1.5 py-0.5 text-[11px] hover:bg-[#1f2429] hover:text-[#e6e8eb]"
      onClick={onClick}
    >{children}</button>
  )
}

function Tree({
  parentId, depth, rename, setRename, dragOver, setDragOver, onContext,
}: {
  parentId: string
  depth: number
  rename: string | null
  setRename: (id: string | null) => void
  dragOver: string | null
  setDragOver: (id: string | null) => void
  onContext: (id: string, x: number, y: number) => void
}) {
  const parent = useWorkspaceStore((s) => s.entries[parentId])
  const expanded = useWorkspaceStore((s) => s.expanded)
  if (!parent || parent.kind !== 'folder') return null

  const childIds = [...parent.childIds].sort((a, b) => sortKey(a, b))

  return (
    <>
      {childIds.map((id) => {
        return (
          <Row
            key={id}
            id={id}
            depth={depth}
            renaming={rename === id}
            setRename={setRename}
            dragOver={dragOver}
            setDragOver={setDragOver}
            onContext={onContext}
            childTree={expanded.has(id) ? (
              <Tree
                parentId={id} depth={depth + 1}
                rename={rename} setRename={setRename}
                dragOver={dragOver} setDragOver={setDragOver}
                onContext={onContext}
              />
            ) : null}
          />
        )
      })}
    </>
  )
}

function sortKey(aId: string, bId: string): number {
  const a = useWorkspaceStore.getState().entries[aId]
  const b = useWorkspaceStore.getState().entries[bId]
  if (!a || !b) return 0
  if (a.kind !== b.kind) return a.kind === 'folder' ? -1 : 1
  return a.name.localeCompare(b.name)
}

function Row({
  id, depth, renaming, setRename, dragOver, setDragOver, onContext, childTree,
}: {
  id: string
  depth: number
  renaming: boolean
  setRename: (id: string | null) => void
  dragOver: string | null
  setDragOver: (id: string | null) => void
  onContext: (id: string, x: number, y: number) => void
  childTree: React.ReactNode
}) {
  const entry = useWorkspaceStore((s) => s.entries[id]) as Entry | undefined
  const isActive = useWorkspaceStore((s) => s.activeFileId === id)
  const expanded = useWorkspaceStore((s) => s.expanded.has(id))
  const toggle = useWorkspaceStore((s) => s.toggleExpanded)
  const open = useWorkspaceStore((s) => s.openFile)
  if (!entry) return null

  const isFolder = entry.kind === 'folder'
  const indent = 8 + depth * 12

  const onClick = () => {
    if (isFolder) toggle(id)
    else open(id)
  }

  const onContextMenu = (e: React.MouseEvent) => {
    e.preventDefault()
    onContext(id, e.clientX, e.clientY)
  }

  const isDragOver = dragOver === id && isFolder

  return (
    <>
      <div
        className={`group flex items-center justify-between gap-1 pr-2 ${
          isActive ? 'bg-[#1f2429] text-[#e6e8eb]'
          : isDragOver ? 'bg-[#1f2429]/60'
          : 'text-[#c0c5cc] hover:bg-[#13171b]'
        }`}
        style={{ paddingLeft: indent }}
        onClick={onClick}
        onDoubleClick={(e) => { e.stopPropagation(); setRename(id) }}
        onContextMenu={onContextMenu}
        draggable={!renaming}
        onDragStart={(e) => {
          e.dataTransfer.setData(DRAG_MIME, id)
          e.dataTransfer.effectAllowed = 'move'
        }}
        onDragOver={(e) => {
          if (!isFolder) return
          if (!e.dataTransfer.types.includes(DRAG_MIME) && !e.dataTransfer.types.includes('Files')) return
          e.preventDefault()
          e.stopPropagation()
          e.dataTransfer.dropEffect = 'move'
          setDragOver(id)
        }}
        onDragLeave={(e) => {
          if ((e.currentTarget as HTMLElement).contains(e.relatedTarget as Node)) return
          if (dragOver === id) setDragOver(null)
        }}
        onDrop={(e) => {
          if (!isFolder) return
          e.preventDefault()
          e.stopPropagation()
          setDragOver(null)
          if (e.dataTransfer.types.includes(DRAG_MIME)) {
            const movedId = e.dataTransfer.getData(DRAG_MIME)
            if (movedId) useWorkspaceStore.getState().move(movedId, id)
            return
          }
          const file = e.dataTransfer.files?.[0]
          if (file) {
            file.text().then((text) => {
              try { useWorkspaceStore.getState().importFromText(id, file.name, text) }
              catch (err) { alert(`Couldn't import ${file.name}:\n${(err as Error).message}`) }
            })
          }
        }}
      >
        <div className="flex min-w-0 items-center gap-0.5 py-[3px] text-xs">
          <span
            className="inline-block w-3 text-center text-[#5b6168]"
            onClick={(e) => { if (isFolder) { e.stopPropagation(); toggle(id) } }}
          >
            {isFolder ? (expanded ? '▾' : '▸') : ' '}
          </span>
          <span className="mr-1">{isFolder ? (expanded ? '📂' : '📁') : '📄'}</span>
          {renaming ? (
            <RenameInput initial={entry.name} commit={(name) => {
              useWorkspaceStore.getState().rename(id, name)
              setRename(null)
            }} cancel={() => setRename(null)} />
          ) : (
            <span className="truncate">{entry.name}</span>
          )}
        </div>
        {!renaming && (
          <div className="hidden gap-0.5 text-[#5b6168] group-hover:flex">
            {isFolder && (
              <IconButton title="New file" onClick={() => {
                const childId = useWorkspaceStore.getState().createFile(id)
                if (childId) setRename(childId)
              }}>＋</IconButton>
            )}
            <IconButton title="Rename" onClick={() => setRename(id)}>✎</IconButton>
            <IconButton title="Delete" onClick={() => {
              if (confirm(`Delete "${entry.name}"${isFolder ? ' and its contents' : ''}?`)) {
                useWorkspaceStore.getState().remove(id)
              }
            }}>✕</IconButton>
          </div>
        )}
      </div>
      {isFolder && expanded && childTree}
    </>
  )
}

function RenameInput({ initial, commit, cancel }: {
  initial: string
  commit: (name: string) => void
  cancel: () => void
}) {
  const [value, setValue] = useState(initial)
  const ref = useRef<HTMLInputElement>(null)
  useEffect(() => {
    ref.current?.focus()
    ref.current?.select()
  }, [])
  return (
    <input
      ref={ref}
      className="ml-1 w-full rounded border border-[#3a4148] bg-[#0e1216] px-1 py-0 text-xs text-[#e6e8eb] outline-none"
      value={value}
      onChange={(e) => setValue(e.target.value)}
      onClick={(e) => e.stopPropagation()}
      onKeyDown={(e) => {
        if (e.key === 'Enter') commit(value)
        if (e.key === 'Escape') cancel()
      }}
      onBlur={() => commit(value)}
    />
  )
}

function ContextMenu({
  menu, setRename, importFile, close,
}: {
  menu: { id: string; x: number; y: number }
  setRename: (id: string | null) => void
  importFile: (parentId: string) => void
  close: () => void
}) {
  const entry = useWorkspaceStore((s) => s.entries[menu.id])
  if (!entry) return null
  const isFolder = entry.kind === 'folder'

  return (
    <div
      className="fixed z-50 min-w-[180px] rounded border border-[#1f2429] bg-[#13171b] p-1 text-xs shadow-lg"
      style={{ left: menu.x, top: menu.y }}
      onMouseDown={(e) => e.stopPropagation()}
    >
      {entry.kind === 'file' && (
        <MenuItem onClick={() => { useWorkspaceStore.getState().openFile(menu.id); close() }}>
          Open
        </MenuItem>
      )}
      {isFolder && (
        <>
          <MenuItem onClick={() => {
            const id = useWorkspaceStore.getState().createFile(menu.id)
            if (id) setRename(id)
            close()
          }}>New file</MenuItem>
          <MenuItem onClick={() => {
            const id = useWorkspaceStore.getState().createFolder(menu.id)
            if (id) setRename(id)
            close()
          }}>New folder</MenuItem>
          <MenuItem onClick={() => { importFile(menu.id); close() }}>Import .mlforge…</MenuItem>
          <div className="my-1 h-px bg-[#1f2429]" />
        </>
      )}
      <MenuItem onClick={() => { setRename(menu.id); close() }}>Rename</MenuItem>
      {menu.id !== ROOT_ID && (
        <MenuItem
          className="text-rose-300 hover:bg-rose-950/40"
          onClick={() => {
            if (confirm(`Delete "${entry.name}"${isFolder ? ' and its contents' : ''}?`)) {
              useWorkspaceStore.getState().remove(menu.id)
            }
            close()
          }}
        >Delete</MenuItem>
      )}
    </div>
  )
}

function MenuItem({
  children, onClick, className,
}: { children: React.ReactNode; onClick: () => void; className?: string }) {
  return (
    <button
      className={`block w-full rounded px-2 py-1 text-left text-xs text-[#e6e8eb] hover:bg-[#1f2429] ${className ?? ''}`}
      onClick={onClick}
    >{children}</button>
  )
}
