// Save/load training graphs as experiments/training-graphs/<name>.spinotrain via
// the connection backend (works local + remote). Kept out of the graph store
// so the store stays a pure in-memory model.

import { fs } from '../../connections/backend'
import { useTrainingGraphStore, captureTrainingSnapshot, type TrainingGraphSnapshot } from './store'
import { serializeTrainingSnapshot, parseTrainingFile } from './persist'

export const TRAINING_GRAPHS_DIR = 'experiments/training-graphs'

const EMPTY_TRAINING: TrainingGraphSnapshot = { nodes: [], edges: [] }

function sanitize(name: string): string {
  const base = name.trim().replace(/\.spinotrain$/i, '').replace(/[^a-zA-Z0-9._-]+/g, '-').replace(/^-+|-+$/g, '')
  return base || 'training'
}

export async function listTrainingGraphs(): Promise<string[]> {
  const entries = await fs.list()
  return entries
    .filter((e) => !e.is_dir && e.relpath.startsWith(TRAINING_GRAPHS_DIR + '/') && e.relpath.toLowerCase().endsWith('.spinotrain'))
    .map((e) => e.relpath)
    .sort()
}

export async function saveTrainingGraph(name: string): Promise<string> {
  const relpath = `${TRAINING_GRAPHS_DIR}/${sanitize(name)}.spinotrain`
  const snapshot = captureTrainingSnapshot(useTrainingGraphStore.getState())
  await fs.write(relpath, serializeTrainingSnapshot(snapshot))
  return relpath
}

export async function loadTrainingGraph(relpath: string): Promise<void> {
  const text = await fs.read(relpath)
  const snapshot = parseTrainingFile(text)
  useTrainingGraphStore.getState().loadSnapshot(snapshot)
}

/** Create a fresh empty .spinotrain, reset the store, return its relpath. */
export async function createTrainingGraph(name: string): Promise<string> {
  const relpath = `${TRAINING_GRAPHS_DIR}/${sanitize(name)}.spinotrain`
  useTrainingGraphStore.getState().resetGraph()
  await fs.write(relpath, serializeTrainingSnapshot(EMPTY_TRAINING))
  return relpath
}

/** Write the current store state to a .spinotrain (autosave + manual save). */
export async function writeTrainingGraph(relpath: string): Promise<void> {
  const snapshot = captureTrainingSnapshot(useTrainingGraphStore.getState())
  await fs.write(relpath, serializeTrainingSnapshot(snapshot))
}

/** Create a new .spinotrain holding the CURRENT store (does NOT reset) — used to
 *  auto-bind a graph the chatbot just built so it's saved + shown, not wiped. */
export async function bindNewTrainingGraphFromCurrent(): Promise<string> {
  const existing = new Set(await listTrainingGraphs())
  let rel = `${TRAINING_GRAPHS_DIR}/training.spinotrain`
  let i = 2
  while (existing.has(rel)) { rel = `${TRAINING_GRAPHS_DIR}/training-${i}.spinotrain`; i++ }
  await writeTrainingGraph(rel)
  return rel
}
