// Shared "canvas ↔ file" binding. Every canvas (architecture / training / data)
// reflects ONE specific workspace file; this store tracks which file each canvas
// is bound to + a small save-status. When a canvas has no bound file the
// CanvasFileGate shows a chooser instead of an orphan, unsaved graph. The bound
// relpaths are persisted so the canvases reopen their last file on launch.

import { create } from 'zustand'

export type CanvasKind = 'architecture' | 'training' | 'data'
export type DocStatus = 'idle' | 'saving' | 'saved' | 'error'

type Doc = { relpath: string | null; status: DocStatus; error: string | null }

type State = {
  docs: Record<CanvasKind, Doc>
  setBound: (k: CanvasKind, relpath: string | null) => void
  setStatus: (k: CanvasKind, status: DocStatus, error?: string | null) => void
}

// An adapter teaches the gate how to list / open / create files for one canvas,
// and (optionally) how to save the current graph. Defined as module constants
// per canvas so the gate's effect deps stay stable.
export type CanvasDocAdapter = {
  kind: CanvasKind
  ext: string                                   // '.spinodata'
  label: string                                 // 'Daten-Graph'
  list: () => Promise<{ relpath: string; name: string }[]>
  open: (relpath: string) => Promise<void>      // load file into the canvas store
  create: (name: string) => Promise<string>     // create a file, return its relpath
  save?: () => Promise<void>                     // manual save of the current graph
  /** The file is loaded by an external owner (e.g. the workspace store for the
   *  architecture .spinoml), so the gate must NOT re-hydrate it on mount. */
  external?: boolean
  /** Called by "Datei wechseln" instead of clearing the binding — for an external
   *  owner this detaches at the source (e.g. workspace.closeActive). */
  unbind?: () => void
}

const KEY = 'spinoml.canvasdoc.bound.v1'

function loadBound(): Partial<Record<CanvasKind, string>> {
  try {
    const raw = localStorage.getItem(KEY)
    if (!raw) return {}
    const v: unknown = JSON.parse(raw)
    if (!v || typeof v !== 'object' || Array.isArray(v)) return {}
    return v as Partial<Record<CanvasKind, string>>
  } catch {
    // localStorage unavailable/corrupt: no persisted bindings, so each canvas
    // simply starts on its chooser (no false claim about a bound file).
    return {}
  }
}
function persistBound(docs: Record<CanvasKind, Doc>) {
  try {
    localStorage.setItem(KEY, JSON.stringify({
      architecture: docs.architecture.relpath,
      training: docs.training.relpath,
      data: docs.data.relpath,
    }))
  } catch {
    // Quota/private mode: only cross-reload persistence of the binding is lost;
    // the in-memory binding for this session is unaffected.
  }
}

const mk = (relpath: string | null): Doc => ({ relpath, status: 'idle', error: null })
const init = loadBound()

export const useCanvasDocStore = create<State>((set, get) => ({
  docs: {
    architecture: mk(init.architecture ?? null),
    training: mk(init.training ?? null),
    data: mk(init.data ?? null),
  },
  setBound: (k, relpath) => {
    const docs = { ...get().docs, [k]: { relpath, status: 'idle' as DocStatus, error: null } }
    set({ docs })
    persistBound(docs)
  },
  setStatus: (k, status, error = null) => {
    set({ docs: { ...get().docs, [k]: { ...get().docs[k], status, error } } })
  },
}))

export function boundRelpath(k: CanvasKind): string | null {
  return useCanvasDocStore.getState().docs[k].relpath
}
