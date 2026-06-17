// Remembers recently-opened LOCAL workspace folders so the Welcome screen can
// offer them as one-click choices. Local-only (remote workspaces live in the
// connections store). Persisted to localStorage.

export type RecentWorkspace = { path: string; name: string; openedAt: number }

const STORAGE_KEY = 'spinoml.recent-workspaces.v1'
const MAX = 8

function nameOf(path: string): string {
  return path.split(/[\\/]/).filter(Boolean).pop() ?? path
}

export function getRecentWorkspaces(): RecentWorkspace[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed
      .filter((r): r is RecentWorkspace => !!r && typeof r.path === 'string')
      .map((r) => ({ path: r.path, name: r.name || nameOf(r.path), openedAt: Number(r.openedAt) || 0 }))
  } catch {
    return []
  }
}

function write(list: RecentWorkspace[]): void {
  if (typeof window === 'undefined') return
  try { window.localStorage.setItem(STORAGE_KEY, JSON.stringify(list.slice(0, MAX))) } catch { /* quota */ }
}

/** Record (or bump to top) a freshly-opened local workspace. */
export function addRecentWorkspace(path: string): void {
  if (!path) return
  const rest = getRecentWorkspaces().filter((r) => r.path !== path)
  write([{ path, name: nameOf(path), openedAt: Date.now() }, ...rest])
}

export function removeRecentWorkspace(path: string): void {
  write(getRecentWorkspaces().filter((r) => r.path !== path))
}
