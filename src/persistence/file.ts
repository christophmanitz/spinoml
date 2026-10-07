import { type GraphSnapshot } from '../canvas/GraphStore'
import { useGraphStore } from '../canvas/GraphStore'
import { captureRootSnapshot } from '../canvas/scopeStore'

export const FORMAT_VERSION = 1

// Explicit schema migrations, keyed by the version they upgrade FROM.
// Future schema changes add an entry here (v1 → v2, …) instead of branching
// in parseFile. The graph payload shape itself stays a GraphSnapshot; anything
// that reshapes it goes through this map so old files keep opening.
const MIGRATIONS: Record<number, (file: unknown) => unknown> = {}

export type SpinoMLFile = {
  format: 'spinoml'
  version: number
  savedAt: string
  graph: GraphSnapshot
}

function migrateFile(file: SpinoMLFile): SpinoMLFile {
  let version = file.version
  let out: unknown = file
  while (version < FORMAT_VERSION) {
    const migrate = MIGRATIONS[version]
    if (!migrate) {
      throw new Error(`file format v${version} is too old to open (no migration path to v${FORMAT_VERSION})`)
    }
    out = migrate(out)
    version += 1
  }
  return out as SpinoMLFile
}

export function serializeCurrent(): string {
  const file: SpinoMLFile = {
    format: 'spinoml',
    version: FORMAT_VERSION,
    savedAt: new Date().toISOString(),
    graph: captureRootSnapshot(),
  }
  return JSON.stringify(file, null, 2)
}

export function parseFile(text: string): GraphSnapshot {
  let obj: unknown
  try { obj = JSON.parse(text) }
  catch (e) { throw new Error(`not valid JSON: ${(e as Error).message}`, { cause: e }) }
  if (!obj || typeof obj !== 'object') throw new Error('expected a JSON object')
  const file = obj as Partial<SpinoMLFile>
  if (file.format !== 'spinoml') throw new Error('not a spinoml file (missing format)')
  if (typeof file.version !== 'number' || !Number.isInteger(file.version)) {
    throw new Error(`invalid version: expected an integer, got ${String(file.version)}`)
  }
  if (file.version > FORMAT_VERSION) {
    throw new Error(`file format v${file.version} is newer than this app (v${FORMAT_VERSION})`)
  }
  const current = migrateFile(file as SpinoMLFile)
  const graph = current.graph
  if (!graph || !Array.isArray(graph.nodes) || !Array.isArray(graph.edges)) {
    throw new Error('graph payload missing nodes/edges')
  }
  return graph as GraphSnapshot
}

export function downloadCurrent(filename = 'model.spinoml'): void {
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
  input.accept = '.spinoml,.json,application/json'
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

const AUTOSAVE_KEY = 'spinoml.autosave.v1'
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
    // Autosave is best-effort recovery: a missing or corrupt entry both mean
    // "nothing recoverable", and the canvas reports no restored state either way.
    return null
  }
}

export function clearAutosave(): void {
  try { localStorage.removeItem(AUTOSAVE_KEY) }
  catch {
    // Best-effort cleanup; failing to remove the key cannot create a false state.
  }
}
