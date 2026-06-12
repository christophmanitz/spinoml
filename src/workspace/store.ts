import { create } from 'zustand'
import { useGraphStore, captureStructuralSnapshot } from '../canvas/GraphStore'
import { parseFile, serializeCurrent } from '../persistence/file'

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
const STORAGE_KEY = 'mlforge.workspace.v1'

type State = {
  entries: Record<string, Entry>
  activeFileId: string | null
  expanded: Set<string>
  dirty: boolean

  createFile: (parentId: string, name?: string) => string
  createFolder: (parentId: string, name?: string) => string
  rename: (id: string, name: string) => void
  remove: (id: string) => void
  move: (id: string, newParentId: string) => void

  toggleExpanded: (id: string) => void
  setExpanded: (id: string, value: boolean) => void

  openFile: (id: string) => boolean        // returns true if loaded
  saveActive: () => void                   // writes serializeCurrent() into active file
  saveAsNew: (parentId: string, name: string) => string
  closeActive: () => void
  importFromText: (parentId: string, name: string, text: string) => string
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
  entries: initial.entries,
  activeFileId: initial.activeFileId,
  expanded: initial.expanded,
  dirty: false,

  createFile: (parentId, name) => {
    const parent = get().entries[parentId]
    if (!parent || parent.kind !== 'folder') return ''
    const id = newId()
    const wantName = uniqueName(name ?? 'untitled.mlforge', siblingNames(get().entries, parentId))
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

  createFolder: (parentId, name) => {
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

  rename: (id, name) => {
    const e = get().entries[id]
    if (!e || id === ROOT_ID) return
    const trimmed = name.trim()
    if (!trimmed || trimmed === e.name) return
    const final = uniqueName(trimmed, siblingNames(get().entries, e.parentId ?? ROOT_ID))
    set({ entries: { ...get().entries, [id]: { ...e, name: final } } })
  },

  remove: (id) => {
    if (id === ROOT_ID) return
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

  move: (id, newParentId) => {
    if (id === ROOT_ID || id === newParentId) return
    const entries = { ...get().entries }
    const node = entries[id]
    const dest = entries[newParentId]
    if (!node || !dest || dest.kind !== 'folder') return
    // no descending into self
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

  openFile: (id) => {
    const e = get().entries[id]
    if (!e || e.kind !== 'file') return false
    if (get().dirty) {
      const ok = confirm('Current model has unsaved changes. Discard and open this file?')
      if (!ok) return false
    }
    try {
      const snap = parseFile(e.content)
      useGraphStore.getState().loadSnapshot(snap)
      set({ activeFileId: id, dirty: false })
      return true
    } catch (err) {
      alert(`Couldn't open ${e.name}:\n${(err as Error).message}`)
      return false
    }
  },

  saveActive: () => {
    const id = get().activeFileId
    if (!id) return
    const e = get().entries[id]
    if (!e || e.kind !== 'file') return
    set({
      entries: {
        ...get().entries,
        [id]: { ...e, content: serializeCurrent(), savedAt: new Date().toISOString() },
      },
      dirty: false,
    })
  },

  saveAsNew: (parentId, name) => {
    return get().createFile(parentId, name.endsWith('.mlforge') ? name : `${name}.mlforge`)
  },

  closeActive: () => set({ activeFileId: null, dirty: false }),

  importFromText: (parentId, name, text) => {
    const parent = get().entries[parentId]
    if (!parent || parent.kind !== 'folder') return ''
    // validate the text parses; throws if not
    parseFile(text)
    const id = newId()
    const wantName = uniqueName(name.endsWith('.mlforge') ? name : `${name}.mlforge`,
                                siblingNames(get().entries, parentId))
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
}))

// Persist on every store change.
useWorkspaceStore.subscribe((s) => persist(s))

// Track dirty: any structural change to the graph after the last save/load
// marks the active file dirty. Compares fingerprints to the last persisted
// content so a no-op edit doesn't false-positive.
let lastFingerprint: string | null = null

function fingerprintCurrent(): string {
  return JSON.stringify(captureStructuralSnapshot(useGraphStore.getState()))
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
