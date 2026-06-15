import { type GraphSnapshot } from '../canvas/GraphStore'
import { useGraphStore } from '../canvas/GraphStore'
import { captureRootSnapshot } from '../canvas/scopeStore'

export const FORMAT_VERSION = 1

export type MLForgeFile = {
  format: 'mlforge'
  version: number
  savedAt: string
  graph: GraphSnapshot
}

export function serializeCurrent(): string {
  const file: MLForgeFile = {
    format: 'mlforge',
    version: FORMAT_VERSION,
    savedAt: new Date().toISOString(),
    graph: captureRootSnapshot(),
  }
  return JSON.stringify(file, null, 2)
}

export function parseFile(text: string): GraphSnapshot {
  let obj: unknown
  try { obj = JSON.parse(text) }
  catch (e) { throw new Error(`not valid JSON: ${(e as Error).message}`) }
  if (!obj || typeof obj !== 'object') throw new Error('expected a JSON object')
  const file = obj as Partial<MLForgeFile>
  if (file.format !== 'mlforge') throw new Error('not a mlforge file (missing format)')
  if (typeof file.version !== 'number') throw new Error('missing version')
  if (file.version > FORMAT_VERSION) {
    throw new Error(`file format v${file.version} is newer than this app (v${FORMAT_VERSION})`)
  }
  const graph = file.graph
  if (!graph || !Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) {
    throw new Error('graph payload missing nodes/edges')
  }
  return graph as GraphSnapshot
}

export function downloadCurrent(filename = 'model.mlforge'): void {
  const blob = new Blob([serializeCurrent()], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export function pickAndLoad(onLoaded: (snapshot: GraphSnapshot, filename: string) => void): void {
  const input = document.createElement('input')
  input.type = 'file'
  input.accept = '.mlforge,.json,application/json'
  input.onchange = async () => {
    const file = input.files?.[0]
    if (!file) return
    const text = await file.text()
    try {
      const snap = parseFile(text)
      onLoaded(snap, file.name)
    } catch (e) {
      alert(`Couldn't load ${file.name}:\n${(e as Error).message}`)
    }
  }
  input.click()
}

// ────────────────────────────────────────────────────────────────────────────
// Auto-save to localStorage

const AUTOSAVE_KEY = 'mlforge.autosave.v1'
let autosaveTimer: ReturnType<typeof setTimeout> | null = null

function writeAutosave() {
  try {
    localStorage.setItem(AUTOSAVE_KEY, serializeCurrent())
  } catch {
    /* quota / private mode — ignore */
  }
}

export function startAutosave(): void {
  useGraphStore.subscribe((state, prev) => {
    if (state.nodes === prev.nodes && state.edges === prev.edges) return
    if (autosaveTimer) clearTimeout(autosaveTimer)
    autosaveTimer = setTimeout(writeAutosave, 800)
  })
}

export function readAutosave(): GraphSnapshot | null {
  try {
    const text = localStorage.getItem(AUTOSAVE_KEY)
    if (!text) return null
    return parseFile(text)
  } catch {
    return null
  }
}

export function clearAutosave(): void {
  try { localStorage.removeItem(AUTOSAVE_KEY) } catch { /* ignore */ }
}
