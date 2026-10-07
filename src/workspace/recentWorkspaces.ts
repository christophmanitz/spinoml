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
    // localStorage unavailable/corrupt: the Recent list is empty. It is a
    // convenience list, not a claim that no workspace exists.
    return []
  }
}

function write(list: RecentWorkspace[]): void {
  if (typeof window === 'undefined') return
  try { window.localStorage.setItem(STORAGE_KEY, JSON.stringify(list.slice(0, MAX))) }
  catch {
    // Quota/private mode: only persistence of the convenience list is skipped.
  }
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

// The LOCAL workspace currently open, persisted separately from the recents
// list. Rust's WorkspaceState (the live currentDir) is in-memory only, so a full
// app restart — or, in `tauri dev`, a rebuild relaunch — loses which local
// folder was open and drops the user to the Welcome screen. This pointer lets
// startup re-open it automatically (the local equivalent of how a remote
// workspace auto-restores from the persisted connection). It is set whenever a
// local project loads and CLEARED on an explicit close, so an intentional close
// does NOT auto-reopen (unlike recents, which persist for the Welcome list).
const ACTIVE_KEY = 'spinoml.active-workspace.v1'

export function getActiveWorkspace(): string | null {
  if (typeof window === 'undefined') return null
  try { return window.localStorage.getItem(ACTIVE_KEY) || null }
  catch {
    // localStorage unavailable: no remembered workspace; startup falls back to
    // the Welcome screen, which is an accurate "nothing is open" state.
    return null
  }
}

export function setActiveWorkspace(path: string | null): void {
  if (typeof window === 'undefined') return
  try {
    if (path) window.localStorage.setItem(ACTIVE_KEY, path)
    else window.localStorage.removeItem(ACTIVE_KEY)
  } catch {
    // Quota/private mode: only auto-reopen of the last local workspace is lost.
  }
}

// The open .spinoml RELPATH, keyed by workspace root. In Tauri mode the workspace
// store is NOT persisted to localStorage (it's rebuilt from disk on load), so a
// webview reload / project re-check would otherwise drop the file BINDING and the
// canvas falls back to the file chooser. We persist just the relpath here and
// restore the binding after bootstrap (graph content is restored separately by the
// autosave) so the open model stays open. Keyed by root so switching workspaces
// never reopens the wrong file.
const ACTIVE_FILE_KEY = 'spinoml.active-file.v1'

export function getActiveFile(root: string): string | null {
  if (typeof window === 'undefined') return null
  try {
    const raw = window.localStorage.getItem(ACTIVE_FILE_KEY)
    if (!raw) return null
    const v: unknown = JSON.parse(raw)
    if (!v || typeof v !== 'object' || Array.isArray(v)) return null
    const o = v as { root?: unknown; relpath?: unknown }
    return o.root === root && typeof o.relpath === 'string' ? o.relpath : null
  } catch {
    // localStorage unavailable/corrupt: no remembered open file; the canvas
    // falls back to its chooser rather than reopening the wrong file.
    return null
  }
}

export function setActiveFile(root: string | null, relpath: string | null): void {
  if (typeof window === 'undefined') return
  try {
    if (root && relpath) window.localStorage.setItem(ACTIVE_FILE_KEY, JSON.stringify({ root, relpath }))
    else window.localStorage.removeItem(ACTIVE_FILE_KEY)
  } catch {
    // Quota/private mode: only the remembered file binding across reloads is lost.
  }
}
