// File-binding adapter + autosave-to-file for the TRAINING canvas. Mirrors
// data/graph/doc.ts. The canvas always reflects a .spinotrain file (see
// CanvasFileGate); edits autosave to it.

import { isTauri } from '../../workspace/tauri-fs'
import { useWorkspaceStore } from '../../workspace/store'
import { useTrainingGraphStore } from './store'
import { useCanvasDocStore, boundRelpath, type CanvasDocAdapter } from '../../canvasdoc/store'
import { markCanvasHydrated } from '../../canvasdoc/CanvasFileGate'
import { listTrainingGraphs, loadTrainingGraph, createTrainingGraph, writeTrainingGraph, bindNewTrainingGraphFromCurrent } from './files'

// True while LOADING a file into the store, so autosave doesn't write it back.
let hydrating = false

export const trainingDocAdapter: CanvasDocAdapter = {
  kind: 'training',
  ext: '.spinotrain',
  label: 'Training-Graph',
  list: async () => (await listTrainingGraphs()).map((relpath) => ({ relpath, name: relpath.split('/').pop() ?? relpath })),
  open: async (relpath) => {
    hydrating = true
    try { await loadTrainingGraph(relpath) } finally { hydrating = false }
  },
  create: createTrainingGraph,
  save: async () => {
    const rel = boundRelpath('training')
    if (rel) await writeTrainingGraph(rel)
  },
}

let timer: ReturnType<typeof setTimeout> | null = null

export function startTrainingAutosaveToFile(): void {
  useTrainingGraphStore.subscribe((state, prev) => {
    if (hydrating) return
    if (state.nodes === prev.nodes && state.edges === prev.edges) return
    const rel = boundRelpath('training')
    if (!rel) return
    if (timer) clearTimeout(timer)
    timer = setTimeout(async () => {
      const { setStatus } = useCanvasDocStore.getState()
      setStatus('training', 'saving')
      try { await writeTrainingGraph(rel); setStatus('training', 'saved') }
      catch (e) { setStatus('training', 'error', e instanceof Error ? e.message : String(e)) }
    }, 700)
  })
}

// Save a chatbot-built training graph to a fresh .spinotrain + bind it when none
// is bound (see data/graph/doc.ts ensureDataBound for the rationale).
export async function ensureTrainingBound(): Promise<void> {
  try {
    if (boundRelpath('training')) return
    if (!isTauri() || !useWorkspaceStore.getState().workspaceRoot) return
    if (useTrainingGraphStore.getState().nodes.length === 0) return
    const rel = await bindNewTrainingGraphFromCurrent()
    markCanvasHydrated('training', rel)
    useCanvasDocStore.getState().setBound('training', rel)
  } catch (e) {
    // The chatbot just built a training graph but persisting it failed. Surface
    // it as an explicit canvas error instead of leaving work unbound/unsaved.
    useCanvasDocStore.getState().setStatus('training', 'error', `Training-Graph konnte nicht gespeichert werden: ${e instanceof Error ? e.message : String(e)}`)
  }
}
