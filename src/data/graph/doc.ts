// File-binding adapter + autosave-to-file for the DATA canvas. The canvas always
// reflects a .spinodata file (see CanvasFileGate); edits autosave to that file so
// the graph is never orphaned/unsaved.

import { isTauri } from '../../workspace/tauri-fs'
import { useWorkspaceStore } from '../../workspace/store'
import { useDataGraphStore } from './store'
import { useCanvasDocStore, boundRelpath, type CanvasDocAdapter } from '../../canvasdoc/store'
import { markCanvasHydrated } from '../../canvasdoc/CanvasFileGate'
import { listDataGraphs, openDataGraph, createDataGraph, writeDataGraph, bindNewDataGraphFromCurrent } from './files'

// True while we are LOADING a file into the store, so the autosave subscriber
// doesn't immediately write the just-loaded content straight back.
let hydrating = false

export const dataDocAdapter: CanvasDocAdapter = {
  kind: 'data',
  ext: '.spinodata',
  label: 'Daten-Graph',
  list: listDataGraphs,
  open: async (relpath) => {
    hydrating = true
    try { await openDataGraph(relpath) } finally { hydrating = false }
  },
  create: createDataGraph,
  save: async () => {
    const rel = boundRelpath('data')
    if (rel) await writeDataGraph(rel)
  },
}

// Debounced autosave to the bound file on every structural change.
let timer: ReturnType<typeof setTimeout> | null = null

export function startDataAutosaveToFile(): void {
  useDataGraphStore.subscribe((state, prev) => {
    if (hydrating) return
    if (state.nodes === prev.nodes && state.edges === prev.edges) return
    const rel = boundRelpath('data')
    if (!rel) return
    if (timer) clearTimeout(timer)
    timer = setTimeout(async () => {
      const { setStatus } = useCanvasDocStore.getState()
      setStatus('data', 'saving')
      try { await writeDataGraph(rel); setStatus('data', 'saved') }
      catch (e) { setStatus('data', 'error', e instanceof Error ? e.message : String(e)) }
    }, 700)
  })
}

// When the chatbot builds a data graph and no file is bound yet, save the CURRENT
// graph to a fresh .spinodata and bind it — so the work is shown + persisted
// instead of being hidden by the chooser and wiped. No-op if already bound, empty,
// or not in a Tauri workspace (the gate is passthrough there).
export async function ensureDataBound(): Promise<void> {
  try {
    if (boundRelpath('data')) return
    if (!isTauri() || !useWorkspaceStore.getState().workspaceRoot) return
    if (useDataGraphStore.getState().nodes.length === 0) return
    const rel = await bindNewDataGraphFromCurrent()
    markCanvasHydrated('data', rel)
    useCanvasDocStore.getState().setBound('data', rel)
  } catch (e) {
    // The chatbot just built a data graph but persisting it failed. Surface it
    // as an explicit canvas error instead of leaving the work unbound/unsaved.
    useCanvasDocStore.getState().setStatus('data', 'error', `Daten-Graph konnte nicht gespeichert werden: ${e instanceof Error ? e.message : String(e)}`)
  }
}
