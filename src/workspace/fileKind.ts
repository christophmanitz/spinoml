// Classifies a workspace entry into a role: which section it belongs to, what
// icon/color/label to show, and whether it's a secondary (dimmed) artifact.
// This is THE place that decides "what is this file" for the explorer.

export type Section = 'models' | 'training' | 'misc'

export type FileKind = {
  /** which top-level section the entry lives under (datasets/experiments are
   *  handled separately — they come from their own stores, not the file tree). */
  section: Section
  /** glyph shown before the name. */
  icon: string
  /** faint role label shown after the name (empty = none). */
  label: string
  /** tailwind text color for the icon chip. */
  color: string
  /** secondary artifact (generated code, logs) → render dimmed. */
  dim: boolean
}

const LOG_NAMES = new Set([
  'events.jsonl', 'status', 'pid', 'metrics.json', 'run.json',
  'stdout.log', 'stderr.log',
])

function ext(name: string): string {
  const i = name.lastIndexOf('.')
  return i === -1 ? '' : name.slice(i).toLowerCase()
}

/** Classify a file by name. Folders are classified separately (see below). */
export function classifyFile(name: string): FileKind {
  const e = ext(name)
  if (e === '.spinoml')
    return { section: 'models', icon: '▦', label: 'Modell', color: 'text-[#6ab7ff]', dim: false }
  if (e === '.spinotrain')
    return { section: 'training', icon: '⚙', label: 'Training', color: 'text-[#b98bff]', dim: false }
  if (e === '.py')
    return { section: 'models', icon: '🐍', label: 'generiert', color: 'text-[#5fd39a]', dim: true }
  if (e === '.pt' || e === '.pth' || e === '.ckpt')
    return { section: 'misc', icon: '💾', label: 'Checkpoint', color: 'text-[#e6c34a]', dim: false }
  if (e === '.md' || e === '.txt')
    return { section: 'misc', icon: '📝', label: '', color: 'text-[#9aa1a8]', dim: false }
  if (LOG_NAMES.has(name.toLowerCase()))
    return { section: 'misc', icon: '▫', label: 'log', color: 'text-[#5b6168]', dim: true }
  return { section: 'misc', icon: '📄', label: '', color: 'text-[#9aa1a8]', dim: false }
}

/** Top-level folders that are surfaced by their own dedicated section/store and
 *  must NOT be walked as raw trees in the Files explorer. Matches the run/dataset
 *  machinery (events.jsonl, pid, run dirs, …) that used to leak into the tree. */
export function isManagedFolder(relpath: string): boolean {
  return relpath === 'datasets' || relpath === 'experiments'
}
