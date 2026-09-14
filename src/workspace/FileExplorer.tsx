import { useEffect, useRef, useState } from 'react'
import { useWorkspaceStore, ROOT_ID, type Entry } from './store'
import { useDatasetsStore } from '../datasets/store'
import { useTrainingStore } from '../training/store'
import { useSidebarStore } from './sidebarStore'
import { classifyFile, isManagedFolder, type Section } from './fileKind'
import { iconFor, colorFor, guessKindFromName } from '../datasets/icons'
import StatusPill from '../training/StatusPill'
import PyCodeModal, { type PyPreview } from './PyCodeModal'
import FileViewerModal from './FileViewerModal'
import { confirmDialog } from '../ui/confirm'

const DRAG_MIME = 'application/spinoml-workspace-entry'

function pyNameFor(name: string): string {
  return name.replace(/\.spinoml$/i, '').replace(/\W+/g, '_') + '.py'
}

// Open a training/data graph file onto its file-bound canvas: load it into the
// store, mark it hydrated + bind it (so the CanvasFileGate shows it instead of the
// chooser), then switch to that view. Mirrors how .spinoml opens via the workspace.
async function openGraphOnCanvas(kind: 'training' | 'data', relpath: string, load: () => Promise<unknown>) {
  try {
    await load()
    const [{ useCanvasDocStore }, { markCanvasHydrated }, { useViewModeStore }] = await Promise.all([
      import('../canvasdoc/store'),
      import('../canvasdoc/CanvasFileGate'),
      import('../training/graph/viewMode'),
    ])
    markCanvasHydrated(kind, relpath)
    useCanvasDocStore.getState().setBound(kind, relpath)
    useViewModeStore.getState().setMode(kind)
  } catch { /* malformed / not on disk — ignore */ }
}

// Shared props threaded down to every row.
type RowCtx = {
  rename: string | null
  setRename: (id: string | null) => void
  dragOver: string | null
  setDragOver: (id: string | null) => void
  onContext: (id: string, x: number, y: number) => void
  openPy: (p: PyPreview) => void
  openFileView: (id: string) => void
}

export default function FileExplorer() {
  const root = useWorkspaceStore((s) => s.entries[ROOT_ID])
  const entries = useWorkspaceStore((s) => s.entries)
  const workspaceRoot = useWorkspaceStore((s) => s.workspaceRoot)
  const mode = useWorkspaceStore((s) => s.mode)
  const createFile = useWorkspaceStore((s) => s.createFile)
  const createFolder = useWorkspaceStore((s) => s.createFolder)
  const importFromText = useWorkspaceStore((s) => s.importFromText)
  const refreshFromDisk = useWorkspaceStore((s) => s.refreshFromDisk)
  const [rename, setRename] = useState<string | null>(null)
  const [menu, setMenu] = useState<{ id: string; x: number; y: number } | null>(null)
  const [dragOver, setDragOver] = useState<string | null>(null)
  const [pyPreview, setPyPreview] = useState<PyPreview | null>(null)
  const [fileView, setFileView] = useState<string | null>(null)

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
    input.accept = '.spinoml,.json,application/json'
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

  // Partition the workspace into role sections. We promote a conventional
  // top-level `models/` container's children so model files sit directly under
  // MODELLE, and we drop datasets/ and experiments/ entirely — they're shown by
  // their own sections (sourced from their stores), never as raw trees.
  const sections = root && root.kind === 'folder'
    ? partitionSections(entries, root.childIds)
    : { models: [], training: [], misc: [] }

  const ctx: RowCtx = {
    rename, setRename, dragOver, setDragOver,
    onContext: (id, x, y) => setMenu({ id, x, y }),
    openPy: setPyPreview, openFileView: setFileView,
  }

  const createIn = (mk: (id: string) => Promise<string>) => {
    const parentId = useWorkspaceStore.getState().entries['models']?.kind === 'folder' ? 'models' : ROOT_ID
    mk(parentId).then((id) => { if (id) setRename(id) })
  }

  return (
    <div className="flex h-full min-h-0 flex-col text-sm">
      <div className="flex shrink-0 items-center justify-between border-b border-[#1f2429] px-2 py-1">
        <div className="flex min-w-0 items-center gap-1.5 text-xs uppercase tracking-wide text-[#6f767e]">
          <span>{mode === 'tauri' ? 'Workspace' : 'Models'}</span>
          {mode === 'tauri' && workspaceRoot && (
            <span className="truncate font-mono text-[10px] normal-case text-[#5b6168]" title={workspaceRoot}>
              {workspaceRoot}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1 text-[#6f767e]">
          {mode === 'tauri' && (
            <IconButton title="Aktualisieren (von Disk neu laden)" onClick={() => { void refreshFromDisk() }}>↻</IconButton>
          )}
          <IconButton title="Neues Modell" onClick={() => createIn(createFile)}>📄+</IconButton>
          <IconButton title="Neuer Ordner" onClick={() => createIn(createFolder)}>📁+</IconButton>
          <IconButton title="Modell importieren (.spinoml)" onClick={() => {
            const parentId = useWorkspaceStore.getState().entries['models']?.kind === 'folder' ? 'models' : ROOT_ID
            importFile(parentId)
          }}>⇪</IconButton>
        </div>
      </div>

      <div
        className={`min-h-0 flex-1 overflow-y-auto py-1 ${dragOver === ROOT_ID ? 'bg-[#1f2429]/40' : ''}`}
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
        <Section sectionKey="models" title="Modelle" count={sections.models.length}>
          {sections.models.length > 0
            ? <EntryList ids={sections.models} depth={1} ctx={ctx} />
            : <Empty>Noch keine Modelle — „📄+" legt eins an.</Empty>}
        </Section>

        {sections.training.length > 0 && (
          <Section sectionKey="training" title="Training" count={sections.training.length}>
            <EntryList ids={sections.training} depth={1} ctx={ctx} />
          </Section>
        )}

        {mode === 'tauri' && <DataSection />}
        {mode === 'tauri' && <RunsSection />}

        {sections.misc.length > 0 && (
          <Section sectionKey="misc" title="Sonstiges" count={sections.misc.length}>
            <EntryList ids={sections.misc} depth={1} ctx={ctx} />
          </Section>
        )}
      </div>

      {menu && <ContextMenu menu={menu}
        setRename={setRename}
        importFile={importFile}
        openPy={setPyPreview}
        openFileView={setFileView}
        close={() => setMenu(null)} />}

      {pyPreview && <PyCodeModal preview={pyPreview} onClose={() => setPyPreview(null)} />}
      {fileView && <FileViewerModal fileId={fileView} onClose={() => setFileView(null)} />}
    </div>
  )
}

// ─── Sectioning ─────────────────────────────────────────────────────────────

function partitionSections(
  entries: Record<string, Entry>,
  rootChildIds: string[],
): Record<Section, string[]> {
  const out: Record<Section, string[]> = { models: [], training: [], misc: [] }
  // Promote the conventional `models/` container so its files appear directly.
  const top: string[] = []
  for (const id of rootChildIds) {
    const e = entries[id]
    if (!e) continue
    if (e.kind === 'folder' && id === 'models') { top.push(...e.childIds); continue }
    top.push(id)
  }
  for (const id of top) {
    const e = entries[id]
    if (!e) continue
    if (e.kind === 'folder') {
      if (isManagedFolder(id)) continue // datasets/ + experiments/ → own sections
      out[folderSection(entries, id)].push(id)
    } else {
      out[classifyFile(e.name).section].push(id)
    }
  }
  return out
}

// A folder belongs to the section of the artifacts it (recursively) holds.
function folderSection(entries: Record<string, Entry>, folderId: string): Section {
  let hasModel = false
  let hasTrain = false
  const stack = [folderId]
  while (stack.length) {
    const cur = entries[stack.pop()!]
    if (!cur || cur.kind !== 'folder') continue
    for (const cid of cur.childIds) {
      const c = entries[cid]
      if (!c) continue
      if (c.kind === 'folder') { stack.push(cid); continue }
      const sec = classifyFile(c.name).section
      if (sec === 'models') hasModel = true
      else if (sec === 'training') hasTrain = true
    }
  }
  if (hasModel) return 'models'
  if (hasTrain) return 'training'
  return 'misc'
}

function Section({
  sectionKey, title, count, children,
}: { sectionKey: string; title: string; count: number; children: React.ReactNode }) {
  const collapsed = useSidebarStore((s) => s.collapsed[sectionKey] ?? false)
  const toggle = useSidebarStore((s) => s.toggleSection)
  return (
    <div className="mb-0.5">
      <button
        onClick={() => toggle(sectionKey)}
        className="flex w-full items-center gap-1 px-2 py-1 text-[10px] font-semibold uppercase tracking-wider text-[#6f767e] hover:text-[#9aa1a8]"
      >
        <span className="inline-block w-3 text-center">{collapsed ? '▸' : '▾'}</span>
        <span className="flex-1 text-left">{title}</span>
        {count > 0 && <span className="rounded bg-[#1a1e22] px-1 text-[9px] text-[#6f767e]">{count}</span>}
      </button>
      {!collapsed && children}
    </div>
  )
}

function Empty({ children }: { children: React.ReactNode }) {
  return <div className="px-3 py-1.5 text-[10px] text-[#5b6168]">{children}</div>
}

// ─── Datasets section (sourced from the datasets store, jumps to its tab) ─────

function DataSection() {
  const entries = useDatasetsStore((s) => s.entries)
  const inspects = useDatasetsStore((s) => s.inspects)
  const refresh = useDatasetsStore((s) => s.refresh)
  const select = useDatasetsStore((s) => s.select)
  const setTab = useSidebarStore((s) => s.setTab)

  useEffect(() => { if (entries.length === 0) void refresh() }, [refresh]) // eslint-disable-line react-hooks/exhaustive-deps

  const open = (rel: string) => { select(rel); setTab('datasets') }

  return (
    <Section sectionKey="data" title="Daten" count={entries.length}>
      {entries.length === 0
        ? <Empty>Lege Dateien in <code>datasets/</code> ab.</Empty>
        : entries.map((e) => {
            const kind = inspects[e.relpath]?.data?.kind ?? guessKindFromName(e.name, e.is_dir)
            return (
              <button
                key={e.relpath}
                onClick={() => open(e.relpath)}
                className="flex w-full items-center gap-1.5 py-[3px] pr-2 text-left text-xs text-[#c0c5cc] hover:bg-[#13171b] hover:text-[#e6e8eb]"
                style={{ paddingLeft: 20 }}
                title={`${e.relpath} — im Datasets-Tab öffnen`}
              >
                <span className={`rounded px-1 font-mono text-[9px] ${colorFor(kind)}`}>{iconFor(kind)}</span>
                <span className="truncate">{e.name}</span>
              </button>
            )
          })}
    </Section>
  )
}

// ─── Experiments section (sourced from the training store, jumps to its tab) ──

function RunsSection() {
  const runs = useTrainingStore((s) => s.runs)
  const refresh = useTrainingStore((s) => s.refresh)
  const select = useTrainingStore((s) => s.select)
  const setTab = useSidebarStore((s) => s.setTab)

  useEffect(() => { if (runs.length === 0) void refresh() }, [refresh]) // eslint-disable-line react-hooks/exhaustive-deps

  const open = (runId: string) => { select(runId); setTab('experiments') }

  return (
    <Section sectionKey="experiments" title="Experimente" count={runs.length}>
      {runs.length === 0
        ? <Empty>Noch keine Runs.</Empty>
        : runs.slice(0, 12).map((r) => (
            <button
              key={r.run_id}
              onClick={() => open(r.run_id)}
              className="flex w-full items-center gap-1.5 py-[3px] pr-2 text-left text-xs text-[#c0c5cc] hover:bg-[#13171b] hover:text-[#e6e8eb]"
              style={{ paddingLeft: 18 }}
              title={`${r.run_label || r.run_id} — im Experiments-Tab öffnen`}
            >
              <StatusPill status={r.status} alive={r.alive} />
              <span className="min-w-0 flex-1 truncate">{r.run_label || r.run_id}</span>
              {r.best_val_loss != null && (
                <span className="shrink-0 text-[10px] text-[#5fd39a]">{r.best_val_loss.toFixed(4)}</span>
              )}
            </button>
          ))}
      {runs.length > 12 && (
        <button
          onClick={() => setTab('experiments')}
          className="px-3 py-1 text-left text-[10px] text-[var(--accent)] hover:underline"
          style={{ paddingLeft: 20 }}
        >
          +{runs.length - 12} weitere → Experiments-Tab
        </button>
      )}
    </Section>
  )
}

// ─── File tree ──────────────────────────────────────────────────────────────

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

// Renders an explicit, sibling-set list of entry ids (one section's contents, or
// a folder's children). Hides .py twins of sibling .spinoml files — those are
// shown as a child row under the model instead.
function EntryList({ ids, depth, ctx }: { ids: string[]; depth: number; ctx: RowCtx }) {
  const entries = useWorkspaceStore.getState().entries
  const sorted = [...ids].sort((a, b) => sortKey(a, b))
  const twinNames = new Set(
    sorted
      .map((id) => entries[id])
      .filter((e): e is Entry => !!e && e.kind === 'file' && e.name.toLowerCase().endsWith('.spinoml'))
      .map((e) => pyNameFor(e.name)),
  )
  return (
    <>
      {sorted.map((id) => {
        const e = entries[id]
        if (!e) return null
        if (e.kind === 'file' && e.name.toLowerCase().endsWith('.py') && twinNames.has(e.name)) return null
        return <EntryNode key={id} id={id} depth={depth} ctx={ctx} />
      })}
    </>
  )
}

function EntryNode({ id, depth, ctx }: { id: string; depth: number; ctx: RowCtx }) {
  const e = useWorkspaceStore((s) => s.entries[id])
  const expanded = useWorkspaceStore((s) => s.expanded.has(id))
  if (!e) return null
  const isFolder = e.kind === 'folder'
  const isFile = e.kind === 'file'
  const subTree = isFolder && expanded
    ? <EntryList ids={(e as Extract<Entry, { kind: 'folder' }>).childIds} depth={depth + 1} ctx={ctx} />
    : null
  return (
    <>
      <Row id={id} depth={depth} ctx={ctx} childTree={subTree} />
      {isFile && expanded && e.name.toLowerCase().endsWith('.spinoml') && (
        <PyChildRow fileId={id} fileName={e.name} depth={depth + 1} openPy={ctx.openPy} />
      )}
    </>
  )
}

function PyChildRow({
  fileId, fileName, depth, openPy,
}: { fileId: string; fileName: string; depth: number; openPy: (p: PyPreview) => void }) {
  const pyName = pyNameFor(fileName)
  return (
    <div
      className="flex items-center gap-1 py-[3px] text-xs text-[#5fd39a]/70 hover:bg-[#13171b] hover:text-[#5fd39a]"
      style={{ paddingLeft: 8 + depth * 12 }}
      onClick={() => openPy({ fileId, pyName })}
      title="Generierten PyTorch-Code ansehen"
    >
      <span className="inline-block w-3" />
      <span className="mr-0.5">🐍</span>
      <span className="truncate">{pyName}</span>
      <span className="ml-1 text-[9px] text-[#5b6168]">generiert</span>
    </div>
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
  id, depth, ctx, childTree,
}: { id: string; depth: number; ctx: RowCtx; childTree: React.ReactNode }) {
  const { rename, setRename, dragOver, setDragOver, onContext, openPy, openFileView } = ctx
  const entry = useWorkspaceStore((s) => s.entries[id]) as Entry | undefined
  const isActive = useWorkspaceStore((s) => s.activeFileId === id)
  const dirty = useWorkspaceStore((s) => s.dirty)
  const expanded = useWorkspaceStore((s) => s.expanded.has(id))
  const toggle = useWorkspaceStore((s) => s.toggleExpanded)
  const open = useWorkspaceStore((s) => s.openFile)
  if (!entry) return null

  const renaming = rename === id
  const isFolder = entry.kind === 'folder'
  const isFile = entry.kind === 'file'
  const indent = 8 + depth * 12
  const kind = isFile ? classifyFile(entry.name) : null
  const showDirty = isActive && dirty

  const onClick = () => {
    if (isFolder) { toggle(id); return }
    if (!isFile) return
    const lower = entry.name.toLowerCase()
    if (lower.endsWith('.spinoml')) { open(id); return }
    // .spinotrain / .spinodata: load the graph onto its canvas, BIND the file (so
    // the CanvasFileGate shows it instead of the chooser) + switch to that mode.
    if (lower.endsWith('.spinotrain')) {
      void openGraphOnCanvas('training', id, () =>
        import('../training/graph/files').then(({ loadTrainingGraph }) => loadTrainingGraph(id)))
      return
    }
    if (lower.endsWith('.spinodata')) {
      void openGraphOnCanvas('data', id, () =>
        import('../data/graph/files').then(({ openDataGraph }) => openDataGraph(id)))
      return
    }
    // datasets/: hand off to the dataset modal instead of trying to parse.
    if (id.startsWith('datasets/')) {
      const datasetRel = 'datasets/' + id.slice('datasets/'.length).split('/')[0]
      useDatasetsStore.getState().select(datasetRel)
      return
    }
    // Any other file: open the generic read-only viewer.
    openFileView(id)
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
        onContextMenu={(e) => { e.preventDefault(); onContext(id, e.clientX, e.clientY) }}
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
        <div className={`flex min-w-0 items-center gap-0.5 py-[3px] text-xs ${kind?.dim ? 'opacity-60' : ''}`}>
          <span
            className="inline-block w-3 text-center text-[#5b6168]"
            onClick={(e) => { e.stopPropagation(); toggle(id) }}
          >
            {(isFolder || isFile) ? (expanded ? '▾' : '▸') : ' '}
          </span>
          <span className={`mr-1 ${kind?.color ?? ''}`}>
            {isFolder ? (expanded ? '📂' : '📁') : kind?.icon}
          </span>
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
          <div className="flex shrink-0 items-center gap-1">
            {showDirty && <span className="text-amber-400" title="ungespeicherte Änderungen">●</span>}
            {kind?.label && !showDirty && (
              <span className="text-[9px] text-[#5b6168] group-hover:hidden">{kind.label}</span>
            )}
            <div className="hidden gap-0.5 text-[#5b6168] group-hover:flex">
              {isFile && entry.name.toLowerCase().endsWith('.spinoml') && (
                <IconButton title="Generierten PyTorch ansehen" onClick={() => {
                  openPy({ fileId: id, pyName: pyNameFor(entry.name) })
                }}>🐍</IconButton>
              )}
              {isFolder && (
                <IconButton title="Neue Datei" onClick={() => {
                  useWorkspaceStore.getState().createFile(id).then((childId) => {
                    if (childId) setRename(childId)
                  })
                }}>＋</IconButton>
              )}
              <IconButton title="Umbenennen" onClick={() => setRename(id)}>✎</IconButton>
              <IconButton title="Löschen" onClick={async () => {
                if (await confirmDialog(`„${entry.name}"${isFolder ? ' samt Inhalt' : ''} löschen?`)) {
                  useWorkspaceStore.getState().remove(id)
                }
              }}>✕</IconButton>
            </div>
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
  menu, setRename, importFile, openPy, openFileView, close,
}: {
  menu: { id: string; x: number; y: number }
  setRename: (id: string | null) => void
  importFile: (parentId: string) => void
  openPy: (p: PyPreview) => void
  openFileView: (id: string) => void
  close: () => void
}) {
  const entry = useWorkspaceStore((s) => s.entries[menu.id])
  if (!entry) return null
  const isFolder = entry.kind === 'folder'
  const isSpinoml = entry.kind === 'file' && entry.name.toLowerCase().endsWith('.spinoml')

  return (
    <div
      className="fixed z-50 min-w-[180px] rounded border border-[#1f2429] bg-[#13171b] p-1 text-xs shadow-lg"
      style={{ left: menu.x, top: menu.y }}
      onMouseDown={(e) => e.stopPropagation()}
    >
      {entry.kind === 'file' && (
        <>
          {isSpinoml && (
            <>
              <MenuItem onClick={() => { useWorkspaceStore.getState().openFile(menu.id); close() }}>
                Im Canvas öffnen
              </MenuItem>
              <MenuItem onClick={() => {
                openPy({ fileId: menu.id, pyName: pyNameFor(entry.name) })
                close()
              }}>Generierten PyTorch ansehen…</MenuItem>
            </>
          )}
          <MenuItem onClick={() => { openFileView(menu.id); close() }}>Datei ansehen…</MenuItem>
          <div className="my-1 h-px bg-[#1f2429]" />
        </>
      )}
      {isFolder && (
        <>
          <MenuItem onClick={() => {
            useWorkspaceStore.getState().createFile(menu.id).then((id) => {
              if (id) setRename(id)
            })
            close()
          }}>Neue Datei</MenuItem>
          <MenuItem onClick={() => {
            useWorkspaceStore.getState().createFolder(menu.id).then((id) => {
              if (id) setRename(id)
            })
            close()
          }}>Neuer Ordner</MenuItem>
          <MenuItem onClick={() => { importFile(menu.id); close() }}>Importieren (.spinoml)…</MenuItem>
          <div className="my-1 h-px bg-[#1f2429]" />
        </>
      )}
      <MenuItem onClick={() => { setRename(menu.id); close() }}>Umbenennen</MenuItem>
      {menu.id !== ROOT_ID && (
        <MenuItem
          className="text-rose-300 hover:bg-rose-950/40"
          onClick={async () => {
            if (await confirmDialog(`„${entry.name}"${isFolder ? ' samt Inhalt' : ''} löschen?`)) {
              useWorkspaceStore.getState().remove(menu.id)
            }
            close()
          }}
        >Löschen</MenuItem>
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
