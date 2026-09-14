// Save/load/list data graphs as experiments/data-graphs/<name>.spinodata via the
// connection backend (local + remote). Kept out of the store so the store stays a
// pure in-memory model.

import { fs } from '../../connections/backend'
import { useDataGraphStore, captureDataSnapshot, type DataGraphSnapshot } from './store'
import { serializeDataSnapshot, parseDataFile } from './persist'

export const DATA_GRAPHS_DIR = 'experiments/data-graphs'

function sanitize(name: string): string {
  const base = name.trim().replace(/\.spinodata$/i, '').replace(/[^a-zA-Z0-9._-]+/g, '-').replace(/^-+|-+$/g, '')
  return base || 'pipeline'
}

const EMPTY: DataGraphSnapshot = { nodes: [], edges: [] }

export async function listDataGraphs(): Promise<{ relpath: string; name: string }[]> {
  const entries = await fs.list()
  return entries
    .filter((e) => !e.is_dir && e.relpath.startsWith(DATA_GRAPHS_DIR + '/') && e.relpath.toLowerCase().endsWith('.spinodata'))
    .map((e) => ({ relpath: e.relpath, name: e.relpath.split('/').pop() ?? e.relpath }))
    .sort((a, b) => a.name.localeCompare(b.name))
}

/** Read a .spinodata and load it into the store. */
export async function openDataGraph(relpath: string): Promise<void> {
  const text = await fs.read(relpath)
  useDataGraphStore.getState().loadSnapshot(parseDataFile(text))
}

/** Create a fresh empty .spinodata, reset the store, return its relpath. */
export async function createDataGraph(name: string): Promise<string> {
  const relpath = `${DATA_GRAPHS_DIR}/${sanitize(name)}.spinodata`
  useDataGraphStore.getState().resetGraph()
  await fs.write(relpath, serializeDataSnapshot(EMPTY))
  return relpath
}

/** Write the current store state to a .spinodata (autosave + manual save). */
export async function writeDataGraph(relpath: string): Promise<void> {
  const snapshot = captureDataSnapshot(useDataGraphStore.getState())
  await fs.write(relpath, serializeDataSnapshot(snapshot))
}

/** Create a new .spinodata holding the CURRENT store (does NOT reset) — used to
 *  auto-bind a graph the chatbot just built so it's saved + shown, not wiped. */
export async function bindNewDataGraphFromCurrent(): Promise<string> {
  const existing = new Set((await listDataGraphs()).map((f) => f.relpath))
  let rel = `${DATA_GRAPHS_DIR}/pipeline.spinodata`
  let i = 2
  while (existing.has(rel)) { rel = `${DATA_GRAPHS_DIR}/pipeline-${i}.spinodata`; i++ }
  await writeDataGraph(rel)
  return rel
}
