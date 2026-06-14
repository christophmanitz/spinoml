// .mltrain file format — same scheme as persistence/file.ts (.mlforge), just a
// different suffix + format tag. Training graphs live under
// experiments/training-graphs/<name>.mltrain.

import type { TrainingGraphSnapshot } from './store'

export const TRAIN_FORMAT_VERSION = 1

export type MLTrainFile = {
  format: 'mltrain'
  version: number
  savedAt: string
  graph: TrainingGraphSnapshot
}

export function serializeTrainingSnapshot(snapshot: TrainingGraphSnapshot): string {
  const file: MLTrainFile = {
    format: 'mltrain',
    version: TRAIN_FORMAT_VERSION,
    savedAt: new Date().toISOString(),
    graph: snapshot,
  }
  return JSON.stringify(file, null, 2)
}

export function parseTrainingFile(text: string): TrainingGraphSnapshot {
  let obj: unknown
  try { obj = JSON.parse(text) } catch (e) { throw new Error(`not valid JSON: ${(e as Error).message}`) }
  if (!obj || typeof obj !== 'object') throw new Error('expected a JSON object')
  const file = obj as Partial<MLTrainFile>
  if (file.format !== 'mltrain') throw new Error('not a mltrain file (missing format)')
  if (typeof file.version !== 'number') throw new Error('missing version')
  if (file.version > TRAIN_FORMAT_VERSION) {
    throw new Error(`file format v${file.version} is newer than this app (v${TRAIN_FORMAT_VERSION})`)
  }
  const graph = file.graph
  if (!graph || !Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) {
    throw new Error('graph payload missing nodes/edges')
  }
  return graph as TrainingGraphSnapshot
}
