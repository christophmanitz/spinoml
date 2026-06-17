// Save/load training graphs as experiments/training-graphs/<name>.spinotrain via
// the connection backend (works local + remote). Kept out of the graph store
// so the store stays a pure in-memory model.

import { fs } from '../../connections/backend'
import { useTrainingGraphStore, captureTrainingSnapshot } from './store'
import { serializeTrainingSnapshot, parseTrainingFile } from './persist'

export const TRAINING_GRAPHS_DIR = 'experiments/training-graphs'

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
