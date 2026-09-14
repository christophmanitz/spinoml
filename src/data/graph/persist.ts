// .spinodata file format — same scheme as persist.ts (.spinotrain) / file.ts
// (.spinoml), different suffix + tag. Data graphs live under
// experiments/data-graphs/<name>.spinodata.

import type { DataGraphSnapshot } from './store'

export const DATA_FORMAT_VERSION = 1

export type SpinoDataFile = {
  format: 'spinodata'
  version: number
  savedAt: string
  graph: DataGraphSnapshot
}

export function serializeDataSnapshot(snapshot: DataGraphSnapshot): string {
  const file: SpinoDataFile = {
    format: 'spinodata',
    version: DATA_FORMAT_VERSION,
    savedAt: new Date().toISOString(),
    graph: snapshot,
  }
  return JSON.stringify(file, null, 2)
}

export function parseDataFile(text: string): DataGraphSnapshot {
  let obj: unknown
  try { obj = JSON.parse(text) } catch (e) { throw new Error(`not valid JSON: ${(e as Error).message}`) }
  if (!obj || typeof obj !== 'object') throw new Error('expected a JSON object')
  const file = obj as Partial<SpinoDataFile>
  if (file.format !== 'spinodata') throw new Error('not a spinodata file (missing format)')
  if (typeof file.version !== 'number') throw new Error('missing version')
  if (file.version > DATA_FORMAT_VERSION) {
    throw new Error(`file format v${file.version} is newer than this app (v${DATA_FORMAT_VERSION})`)
  }
  const graph = file.graph
  if (!graph || !Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) {
    throw new Error('graph payload missing nodes/edges')
  }
  return graph as DataGraphSnapshot
}
