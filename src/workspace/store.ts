import { create } from 'zustand'
import { useGraphStore } from '../canvas/GraphStore'
import { captureRootSnapshot, useScopeStore } from '../canvas/scopeStore'
import { parseFile, serializeCurrent } from '../persistence/file'
import { generateFromSnapshot } from '../codegen/generator'
import { isTauri, tauriFs } from './tauri-fs'
import { fs as fsBackend } from '../connections/backend'
import { useConnectionsStore } from '../connections/store'
import { confirmDialog } from '../ui/confirm'

// ─── Types ────────────────────────────────────────────────────────────────

export type Folder = {
  kind: 'folder'
  id: string
  name: string
  parentId: string | null
  childIds: string[]
}

export type File = {
  kind: 'file'
  id: string
  name: string
  parentId: string | null
  content: string
  savedAt: string
}

export type Entry = Folder | File

export const ROOT_ID = 'root'
const STORAGE_KEY = 'spinoml.workspace.v1'

type Mode = 'browser' | 'tauri'

type State = {
  mode: Mode
  workspaceRoot: string | null

  entries: Record<string, Entry>
  activeFileId: string | null
  expanded: Set<string>
  dirty: boolean

  createFile: (parentId: string, name?: string) => Promise<string>
  createFolder: (parentId: string, name?: string) => Promise<string>
  rename: (id: string, name: string) => Promise<void>
  remove: (id: string) => Promise<void>
  move: (id: string, newParentId: string) => Promise<void>

  toggleExpanded: (id: string) => void
  setExpanded: (id: string, value: boolean) => void

  openFile: (id: string) => Promise<boolean>
  saveActive: () => Promise<void>
  saveAsNew: (parentId: string, name: string) => Promise<string>
  closeActive: () => void
  importFromText: (parentId: string, name: string, text: string) => Promise<string>

  openDirectory: () => Promise<boolean>
  closeDirectory: () => Promise<void>
  refreshFromDisk: () => Promise<void>
}

function pyTwinPath(spinomlRel: string): string {
  return spinomlRel.replace(/\.spinoml$/i, '').replace(/[^\w\/]+/g, '_') + '.py'
}

function parentRelOf(relpath: string): string {
  const i = relpath.lastIndexOf('/')
  return i === -1 ? '' : relpath.slice(0, i)
}

function joinRel(parentRel: string, name: string): string {
  return parentRel ? `${parentRel}/${name}` : name
}

// ─── Helpers ──────────────────────────────────────────────────────────────

function newId(): string {
  return 'w' + Math.random().toString(36).slice(2, 10)
}

function uniqueName(base: string, taken: Set<string>): string {
  if (!taken.has(base)) return base
  const m = base.match(/^(.*?)(?:\((\d+)\))?(\.[^.]+)?$/)
  const stem = m?.[1]?.trim() ?? base
  const ext = m?.[3] ?? ''
  let i = 2
  while (taken.has(`${stem} (${i})${ext}`)) i++
  return `${stem} (${i})${ext}`
}

function siblingNames(entries: Record<string, Entry>, parentId: string): Set<string> {
  const parent = entries[parentId]
  if (!parent || parent.kind !== 'folder') return new Set()
  return new Set(parent.childIds.map((id) => entries[id]?.name).filter(Boolean) as string[])
}

function emptyWorkspace(): { entries: Record<string, Entry>; activeFileId: null; expanded: Set<string> } {
  return {
    entries: {
      [ROOT_ID]: {
        kind: 'folder', id: ROOT_ID, name: 'models', parentId: null, childIds: [],
      },
    },
    activeFileId: null,
    expanded: new Set([ROOT_ID]),
  }
}

// ─── Persistence ──────────────────────────────────────────────────────────

type Persisted = {
  entries: Record<string, Entry>
  activeFileId: string | null
  expanded: string[]
}

function hydrate(): { entries: Record<string, Entry>; activeFileId: string | null; expanded: Set<string> } {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return emptyWorkspace()
    const parsed = JSON.parse(raw) as Persisted
    if (!parsed.entries || !parsed.entries[ROOT_ID]) return emptyWorkspace()
    return {
      entries: parsed.entries,
      activeFileId: parsed.activeFileId ?? null,
      expanded: new Set(parsed.expanded ?? [ROOT_ID]),
    }
  } catch {
    return emptyWorkspace()
  }
}

function persist(state: State) {
  try {
    const data: Persisted = {
      entries: state.entries,
      activeFileId: state.activeFileId,
      expanded: [...state.expanded],
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
  } catch {
    /* quota — ignore */
  }
}

// ─── Store ────────────────────────────────────────────────────────────────

const initial = hydrate()

export const useWorkspaceStore = create<State>((set, get) => ({
  mode: 'browser',
  workspaceRoot: null,
  entries: initial.entries,
  activeFileId: initial.activeFileId,
  expanded: initial.expanded,
  dirty: false,

  createFile: async (parentId, name) => {
    if (get().mode === 'tauri') {
      const parent = get().entries[parentId]
      if (!parent || parent.kind !== 'folder') return ''
      const parentRel = parentId === ROOT_ID ? '' : parentId
      const wantName = uniqueName(name ?? 'untitled.spinoml', siblingNames(get().entries, parentId))
      const relpath = joinRel(parentRel, wantName)
      const content = serializeCurrent()
      await fsBackend.write(relpath, content)
      try {
        await fsBackend.write(pyTwinPath(relpath), generateFromSnapshot(parseFile(content)).code)
      } catch { /* ignore .py write errors */ }
      await get().refreshFromDisk()
      set({ activeFileId: relpath, dirty: false })
      return relpath
    }
    const parent = get().entries[parentId]
    if (!parent || parent.kind !== 'folder') return ''
    const id = newId()
    const wantName = uniqueName(name ?? 'untitled.spinoml', siblingNames(get().entries, parentId))
    const file: File = {
      kind: 'file', id, name: wantName, parentId,
      content: serializeCurrent(),
      savedAt: new Date().toISOString(),
    }
    set({
      entries: {
        ...get().entries,
        [id]: file,
        [parentId]: { ...parent, childIds: [...parent.childIds, id] },
      },
      activeFileId: id,
      dirty: false,
      expanded: new Set([...get().expanded, parentId]),
    })
    return id
  },

  createFolder: async (parentId, name) => {
    if (get().mode === 'tauri') {
      const parent = get().entries[parentId]
      if (!parent || parent.kind !== 'folder') return ''
      const parentRel = parentId === ROOT_ID ? '' : parentId
      const wantName = uniqueName(name ?? 'new folder', siblingNames(get().entries, parentId))
      const relpath = joinRel(parentRel, wantName)
      await fsBackend.mkdir(relpath)
      await get().refreshFromDisk()
      set({ expanded: new Set([...get().expanded, relpath]) })
      return relpath
    }
    const parent = get().entries[parentId]
    if (!parent || parent.kind !== 'folder') return ''
    const id = newId()
    const wantName = uniqueName(name ?? 'new folder', siblingNames(get().entries, parentId))
    const folder: Folder = { kind: 'folder', id, name: wantName, parentId, childIds: [] }
    set({
      entries: {
        ...get().entries,
        [id]: folder,
        [parentId]: { ...parent, childIds: [...parent.childIds, id] },
      },
      expanded: new Set([...get().expanded, parentId, id]),
    })
    return id
  },

  rename: async (id, name) => {
    if (get().mode === 'tauri') {
      if (id === ROOT_ID) return
      const e = get().entries[id]
      if (!e) return
      const trimmed = name.trim()
      if (!trimmed || trimmed === e.name) return
      const parentRel = (!e.parentId || e.parentId === ROOT_ID) ? '' : e.parentId
      const final = uniqueName(trimmed, siblingNames(get().entries, e.parentId ?? ROOT_ID))
      const newRel = joinRel(parentRel, final)
      await fsBackend.rename(id, newRel)
      if (e.kind === 'file' && id.toLowerCase().endsWith('.spinoml')) {
        try { await fsBackend.rename(pyTwinPath(id), pyTwinPath(newRel)) } catch { /* maybe absent */ }
      }
      const wasActive = get().activeFileId === id
      await get().refreshFromDisk()
      if (wasActive) set({ activeFileId: newRel })
      return
    }
    const e = get().entries[id]
    if (!e || id === ROOT_ID) return
    const trimmed = name.trim()
    if (!trimmed || trimmed === e.name) return
    const final = uniqueName(trimmed, siblingNames(get().entries, e.parentId ?? ROOT_ID))
    set({ entries: { ...get().entries, [id]: { ...e, name: final } } })
  },

  remove: async (id) => {
    if (id === ROOT_ID) return
    if (get().mode === 'tauri') {
      const e = get().entries[id]
      if (!e) return
      await fsBackend.remove(id)
      if (e.kind === 'file' && id.toLowerCase().endsWith('.spinoml')) {
        try { await fsBackend.remove(pyTwinPath(id)) } catch { /* maybe absent */ }
      }
      await get().refreshFromDisk()
      if (get().activeFileId === id) set({ activeFileId: null, dirty: false })
      return
    }
    const entries = { ...get().entries }
    const target = entries[id]
    if (!target) return
    const toDelete = new Set<string>([id])
    const stack = [id]
    while (stack.length) {
      const next = stack.pop()!
      const e = entries[next]
      if (e?.kind === 'folder') {
        for (const c of e.childIds) { toDelete.add(c); stack.push(c) }
      }
    }
    for (const tid of toDelete) delete entries[tid]
    const parentId = target.parentId
    if (parentId) {
      const parent = entries[parentId]
      if (parent?.kind === 'folder') {
        entries[parentId] = { ...parent, childIds: parent.childIds.filter((c) => c !== id) }
      }
    }
    const activeFileId = toDelete.has(get().activeFileId ?? '') ? null : get().activeFileId
    set({ entries, activeFileId })
  },

  move: async (id, newParentId) => {
    if (id === ROOT_ID || id === newParentId) return
    if (get().mode === 'tauri') {
      const node = get().entries[id]
      const dest = get().entries[newParentId]
      if (!node || !dest || dest.kind !== 'folder') return
      let cur: string | null = newParentId
      while (cur) {
        if (cur === id) return
        cur = get().entries[cur]?.parentId ?? null
      }
      const destRel = newParentId === ROOT_ID ? '' : newParentId
      const newRel = joinRel(destRel, node.name)
      await fsBackend.rename(id, newRel)
      if (node.kind === 'file' && id.toLowerCase().endsWith('.spinoml')) {
        try { await fsBackend.rename(pyTwinPath(id), pyTwinPath(newRel)) } catch { /* maybe absent */ }
      }
      const wasActive = get().activeFileId === id
      await get().refreshFromDisk()
      if (wasActive) set({ activeFileId: newRel })
      return
    }
    const entries = { ...get().entries }
    const node = entries[id]
    const dest = entries[newParentId]
    if (!node || !dest || dest.kind !== 'folder') return
    let cur: string | null = newParentId
    while (cur) {
      if (cur === id) return
      cur = entries[cur]?.parentId ?? null
    }
    const oldParentId = node.parentId ?? ROOT_ID
    const oldParent = entries[oldParentId]
    if (oldParent?.kind === 'folder') {
      entries[oldParentId] = { ...oldParent, childIds: oldParent.childIds.filter((c) => c !== id) }
    }
    const name = uniqueName(node.name, siblingNames(entries, newParentId))
    entries[id] = { ...node, parentId: newParentId, name }
    entries[newParentId] = { ...dest, childIds: [...dest.childIds, id] }
    set({ entries, expanded: new Set([...get().expanded, newParentId]) })
  },

  toggleExpanded: (id) => {
    const expanded = new Set(get().expanded)
    if (expanded.has(id)) expanded.delete(id)
    else expanded.add(id)
    set({ expanded })
  },

  setExpanded: (id, value) => {
    const expanded = new Set(get().expanded)
    if (value) expanded.add(id)
    else expanded.delete(id)
    set({ expanded })
  },

  openFile: async (id) => {
    const e = get().entries[id]
    if (!e || e.kind !== 'file') return false
    if (get().dirty) {
      const ok = await confirmDialog('Current model has unsaved changes. Discard and open this file?')
      if (!ok) return false
    }
    try {
      let content = e.content
      if (get().mode === 'tauri') {
        content = await fsBackend.read(id)
        set({ entries: { ...get().entries, [id]: { ...e, content } } })
      }
      const snap = parseFile(content)
      useScopeStore.getState().reset()
      useGraphStore.getState().loadSnapshot(snap)
      set({ activeFileId: id, dirty: false })
      return true
    } catch (err) {
      alert(`Couldn't open ${e.name}:\n${(err as Error).message}`)
      return false
    }
  },

  saveActive: async () => {
    const id = get().activeFileId
    if (!id) return
    const e = get().entries[id]
    if (!e || e.kind !== 'file') return
    const content = serializeCurrent()
    if (get().mode === 'tauri') {
      await fsBackend.write(id, content)
      try {
        await fsBackend.write(pyTwinPath(id), generateFromSnapshot(parseFile(content)).code)
      } catch { /* skip .py if codegen fails */ }
      set({
        entries: {
          ...get().entries,
          [id]: { ...e, content, savedAt: new Date().toISOString() },
        },
        dirty: false,
      })
      await get().refreshFromDisk()
      return
    }
    set({
      entries: {
        ...get().entries,
        [id]: { ...e, content, savedAt: new Date().toISOString() },
      },
      dirty: false,
    })
  },

  saveAsNew: async (parentId, name) => {
    return get().createFile(parentId, name.endsWith('.spinoml') ? name : `${name}.spinoml`)
  },

  closeActive: () => set({ activeFileId: null, dirty: false }),

  importFromText: async (parentId, name, text) => {
    const parent = get().entries[parentId]
    if (!parent || parent.kind !== 'folder') return ''
    parseFile(text) // validate
    if (get().mode === 'tauri') {
      const parentRel = parentId === ROOT_ID ? '' : parentId
      const wantName = uniqueName(
        name.endsWith('.spinoml') ? name : `${name}.spinoml`,
        siblingNames(get().entries, parentId),
      )
      const relpath = joinRel(parentRel, wantName)
      await fsBackend.write(relpath, text)
      try {
        await fsBackend.write(pyTwinPath(relpath), generateFromSnapshot(parseFile(text)).code)
      } catch { /* ignore */ }
      await get().refreshFromDisk()
      return relpath
    }
    const id = newId()
    const wantName = uniqueName(
      name.endsWith('.spinoml') ? name : `${name}.spinoml`,
      siblingNames(get().entries, parentId),
    )
    const file: File = {
      kind: 'file', id, name: wantName, parentId,
      content: text, savedAt: new Date().toISOString(),
    }
    set({
      entries: {
        ...get().entries,
        [id]: file,
        [parentId]: { ...parent, childIds: [...parent.childIds, id] },
      },
      expanded: new Set([...get().expanded, parentId]),
    })
    return id
  },

  openDirectory: async () => {
    if (!isTauri()) return false
    // openDirectory always means a LOCAL pick — reset connection so backend
    // calls don't keep routing through a remote one.
    useConnectionsStore.getState().setCurrent('local')
    const picked = await tauriFs.pickDir()
    if (!picked) return false
    const rootName = picked.split('/').pop() || picked.split('\\').pop() || 'workspace'
    set({
      mode: 'tauri',
      workspaceRoot: picked,
      entries: {
        [ROOT_ID]: { kind: 'folder', id: ROOT_ID, name: rootName, parentId: null, childIds: [] },
      },
      activeFileId: null,
      dirty: false,
      expanded: new Set([ROOT_ID]),
    })
    await get().refreshFromDisk()
    return true
  },

  closeDirectory: async () => {
    if (isTauri()) { try { await tauriFs.closeDir() } catch { /* ignore */ } }
    const re = hydrate()
    set({
      mode: 'browser',
      workspaceRoot: null,
      entries: re.entries,
      activeFileId: re.activeFileId,
      expanded: re.expanded,
      dirty: false,
    })
  },

  refreshFromDisk: async () => {
    if (get().mode !== 'tauri') return
    const list = await fsBackend.list()
    const rootName = get().workspaceRoot?.split(/[\\/]/).filter(Boolean).pop() ?? 'workspace'
    const entries: Record<string, Entry> = {
      [ROOT_ID]: { kind: 'folder', id: ROOT_ID, name: rootName, parentId: null, childIds: [] },
    }
    const sorted = [...list].sort(
      (a, b) => a.relpath.split('/').length - b.relpath.split('/').length,
    )
    for (const e of sorted) {
      const id = e.relpath
      const parentRel = parentRelOf(id)
      const parentId = parentRel === '' ? ROOT_ID : parentRel
      const parent = entries[parentId]
      if (!parent || parent.kind !== 'folder') continue
      parent.childIds.push(id)
      if (e.is_dir) {
        entries[id] = { kind: 'folder', id, name: e.name, parentId, childIds: [] }
      } else {
        const existing = get().entries[id]
        const content = existing?.kind === 'file' ? existing.content : ''
        entries[id] = { kind: 'file', id, name: e.name, parentId, content, savedAt: '' }
      }
    }
    set({ entries, expanded: new Set([ROOT_ID, ...get().expanded]) })
  },
}))

// Persist to localStorage only when in browser mode (Tauri mode lives on disk).
useWorkspaceStore.subscribe((s) => { if (s.mode === 'browser') persist(s) })

// Track dirty: any structural change to the graph after the last save/load
// marks the active file dirty. Compares fingerprints to the last persisted
// content so a no-op edit doesn't false-positive.
let lastFingerprint: string | null = null

function fingerprintCurrent(): string {
  // Fold to the root so edits made inside a subcanvas still register as dirty.
  const root = captureRootSnapshot()
  return JSON.stringify({
    nodes: root.nodes.map((n) => ({ id: n.id, layerType: n.layerType, params: n.params })),
    edges: root.edges.map((e) => ({ source: e.source, target: e.target })),
  })
}

function fingerprintFile(file: File): string | null {
  try {
    const snap = parseFile(file.content)
    return JSON.stringify({
      nodes: snap.nodes.map((n) => ({ id: n.id, layerType: n.layerType, params: n.params })),
      edges: snap.edges.map((e) => ({ source: e.source, target: e.target })),
    })
  } catch { return null }
}

function refreshFingerprintForActive(): void {
  const { activeFileId, entries } = useWorkspaceStore.getState()
  if (!activeFileId) { lastFingerprint = null; return }
  const file = entries[activeFileId]
  if (file?.kind === 'file') lastFingerprint = fingerprintFile(file)
  else lastFingerprint = null
}

// On activeFileId or save → reset baseline.
useWorkspaceStore.subscribe((s, prev) => {
  const becameClean = s.dirty !== prev.dirty && !s.dirty
  if (s.activeFileId !== prev.activeFileId || becameClean) {
    refreshFingerprintForActive()
  }
})

// On graph mutation → recompute dirty against baseline.
useGraphStore.subscribe((state, prev) => {
  if (state.nodes === prev.nodes && state.edges === prev.edges) return
  const ws = useWorkspaceStore.getState()
  if (!ws.activeFileId) return
  const now = fingerprintCurrent()
  const isDirty = lastFingerprint !== null && now !== lastFingerprint
  if (ws.dirty !== isDirty) useWorkspaceStore.setState({ dirty: isDirty })
})

refreshFingerprintForActive()
