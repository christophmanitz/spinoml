import { create } from 'zustand'
import { useGraphStore, captureStructuralSnapshot, captureSnapshot, type GraphSnapshot } from '../canvas/GraphStore'

const MAX_HISTORY = 50

type StructuralSnap = ReturnType<typeof captureStructuralSnapshot>

type HistoryState = {
  past: GraphSnapshot[]
  future: GraphSnapshot[]
  canUndo: boolean
  canRedo: boolean

  undo: () => void
  redo: () => void
  clear: () => void
}

let suppress = false
let lastStructural: string | null = null

function structuralKey(s: StructuralSnap): string {
  return JSON.stringify(s)
}

export const useHistoryStore = create<HistoryState>((set, get) => ({
  past: [],
  future: [],
  canUndo: false,
  canRedo: false,

  undo: () => {
    const { past } = get()
    if (past.length === 0) return
    const target = past[past.length - 1]
    const current = captureSnapshot(useGraphStore.getState())
    suppress = true
    try {
      useGraphStore.getState().loadSnapshot(target)
    } finally {
      lastStructural = structuralKey(captureStructuralSnapshot(useGraphStore.getState()))
      suppress = false
    }
    set({
      past: past.slice(0, -1),
      future: [current, ...get().future].slice(0, MAX_HISTORY),
      canUndo: past.length - 1 > 0,
      canRedo: true,
    })
  },

  redo: () => {
    const { future } = get()
    if (future.length === 0) return
    const target = future[0]
    const current = captureSnapshot(useGraphStore.getState())
    suppress = true
    try {
      useGraphStore.getState().loadSnapshot(target)
    } finally {
      lastStructural = structuralKey(captureStructuralSnapshot(useGraphStore.getState()))
      suppress = false
    }
    set({
      past: [...get().past, current].slice(-MAX_HISTORY),
      future: future.slice(1),
      canUndo: true,
      canRedo: future.length - 1 > 0,
    })
  },

  clear: () => set({ past: [], future: [], canUndo: false, canRedo: false }),
}))

/** Run a GraphStore mutation without it landing on the undo stack, and rebase
 *  the structural baseline afterwards. Used by subcanvas enter/exit, which swap
 *  the whole graph as a navigation (not an editable change). */
export function suspendHistory<T>(fn: () => T): T {
  suppress = true
  try {
    return fn()
  } finally {
    lastStructural = structuralKey(captureStructuralSnapshot(useGraphStore.getState()))
    suppress = false
  }
}

// Subscribe: push to history whenever the structural shape of the graph
// changes (ignore pure position drags, inferred shapes, selection).
useGraphStore.subscribe((state, prev) => {
  if (suppress) return
  if (state.nodes === prev.nodes && state.edges === prev.edges) return

  const snap = captureStructuralSnapshot(state)
  const key = structuralKey(snap)
  if (lastStructural === null) {
    lastStructural = key
    return
  }
  if (key === lastStructural) return
  lastStructural = key

  const full = captureSnapshot(prev)
  const h = useHistoryStore.getState()
  useHistoryStore.setState({
    past: [...h.past, full].slice(-MAX_HISTORY),
    future: [],
    canUndo: true,
    canRedo: false,
  })
})

// Keybindings: ⌘Z / Ctrl-Z, ⌘⇧Z / Ctrl-Y for redo.
if (typeof window !== 'undefined') {
  window.addEventListener('keydown', (e) => {
    const meta = e.metaKey || e.ctrlKey
    if (!meta) return
    const target = e.target as HTMLElement | null
    if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) return
    const key = e.key.toLowerCase()
    if (key === 'z' && !e.shiftKey) {
      e.preventDefault()
      useHistoryStore.getState().undo()
    } else if ((key === 'z' && e.shiftKey) || key === 'y') {
      e.preventDefault()
      useHistoryStore.getState().redo()
    }
  })
}
