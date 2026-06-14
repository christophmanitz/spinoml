// Autosave/restore for the visual TRAINING graph — the analogue of
// persistence/file.ts for the architecture graph. The training canvas isn't
// bound to an open workspace file (saving .mltrain is explicit), so without
// this the canvas was empty on every app reopen. Debounced localStorage write
// on every structural change; restored at launch from main.tsx.

import { useTrainingGraphStore, captureTrainingSnapshot, type TrainingGraphSnapshot } from './store'

const KEY = 'mlforge.training-graph.autosave.v1'
let timer: ReturnType<typeof setTimeout> | null = null

function write() {
  try {
    const snap = captureTrainingSnapshot(useTrainingGraphStore.getState())
    localStorage.setItem(KEY, JSON.stringify(snap))
  } catch {
    /* quota / private mode — ignore */
  }
}

export function startTrainingAutosave(): void {
  useTrainingGraphStore.subscribe((state, prev) => {
    if (state.nodes === prev.nodes && state.edges === prev.edges) return
    if (timer) clearTimeout(timer)
    timer = setTimeout(write, 800)
  })
}

export function readTrainingAutosave(): TrainingGraphSnapshot | null {
  try {
    const text = localStorage.getItem(KEY)
    if (!text) return null
    const snap = JSON.parse(text) as TrainingGraphSnapshot
    if (!snap || !Array.isArray(snap.nodes) || !Array.isArray(snap.edges)) return null
    return snap
  } catch {
    return null
  }
}
