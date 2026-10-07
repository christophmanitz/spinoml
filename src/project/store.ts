import { create } from 'zustand'
import { isTauri, tauriFs, type ProjectMeta, type ProjectMetaPatch } from '../workspace/tauri-fs'
import { useWorkspaceStore, ROOT_ID } from '../workspace/store'
import {
  project as projectBackend,
} from '../connections/backend'
import {
  getCurrentConnection,
  useConnectionsStore,
} from '../connections/store'
import { tauriSsh } from '../connections/tauri-ssh'
import { sshTarget } from '../connections/store'
import { useRemoteSidecarStore } from '../sidecars/remoteSidecar'
import { addRecentWorkspace, getActiveWorkspace, setActiveWorkspace, getActiveFile } from '../workspace/recentWorkspaces'

export type ProjectStatus =
  | { kind: 'none' }
  | { kind: 'loading' }
  | { kind: 'legacy'; root: string; legacy_spinoml_count: number }
  | { kind: 'remote-missing'; root: string; alias: string }  // user pointed to a remote dir that doesn't exist yet
  | { kind: 'loaded'; root: string; meta: ProjectMeta }
  | { kind: 'error'; error: string }

type State = {
  status: ProjectStatus
  refresh: () => Promise<void>
  init: (name: string, description: string, goal: string) => Promise<void>
  migrate: (name: string, description: string, goal: string) => Promise<void>
  patch: (p: ProjectMetaPatch) => Promise<void>
  pickFolder: () => Promise<void>
  openLocalPath: (path: string) => Promise<void>
  openConnection: (connectionId: string) => Promise<void>
  closeProject: () => Promise<void>
}

export const useProjectStore = create<State>((set, get) => ({
  status: { kind: 'none' },

  refresh: async () => {
    if (!isTauri()) {
      set({ status: { kind: 'none' } })
      return
    }
    const conn = getCurrentConnection()
    // Only blank to the full-screen loading/Welcome on a FIRST load. Re-refreshing
    // an ALREADY-loaded project (e.g. a background re-check on a remote connection)
    // keeps the current UI mounted, so the canvas + chat panel don't unmount and
    // lose state. Connection switches go through closeProject (status→'none'), so
    // those still show the loading screen as expected.
    if (get().status.kind !== 'loaded') set({ status: { kind: 'loading' } })
    try {
      if (conn.kind === 'remote-ssh') {
        const load = await projectBackend.load()
        if (load.meta) {
          set({ status: { kind: 'loaded', root: load.root, meta: load.meta } })
          await bootstrapWorkspace(load.root)
          // Fire-and-forget the remote-sidecar bootstrap so smoke tests and
          // shape inference work against the HPC's filesystem. UI subscribes
          // to remote-sidecar:status events for progress; we don't block the
          // workspace load on it.
          void useRemoteSidecarStore.getState().ensure(sshTarget(conn), load.root)
        } else if (!load.rootExists) {
          set({ status: { kind: 'remote-missing', root: load.root, alias: conn.alias } })
        } else {
          // Root exists but no project.json: treat as needing init. We don't
          // currently surface "legacy" on remote (no .spinoml migration story).
          set({ status: { kind: 'remote-missing', root: load.root, alias: conn.alias } })
        }
        return
      }
      let current = await tauriFs.currentDir()
      if (!current) {
        // Rust's WorkspaceState is in-memory only, so after an app restart
        // (incl. a `tauri dev` rebuild relaunch) currentDir is empty even though
        // a local workspace was open. Re-point it at the last active folder so
        // the user isn't dropped back to Welcome. Cleared on explicit close, so
        // this only fires when the app went away unexpectedly.
        const active = getActiveWorkspace()
        if (active) {
          try { current = await tauriFs.setDir(active) }
          catch { setActiveWorkspace(null); current = null }
        }
      }
      if (!current) {
        set({ status: { kind: 'none' } })
        return
      }
      const load = await projectBackend.load()
      if (load.meta) {
        set({ status: { kind: 'loaded', root: load.root, meta: load.meta } })
        setActiveWorkspace(load.root)
        await bootstrapWorkspace(load.root)
      } else if (load.hasLegacyFiles) {
        set({ status: { kind: 'legacy', root: load.root, legacy_spinoml_count: load.legacySpinomlCount } })
      } else {
        set({ status: { kind: 'legacy', root: load.root, legacy_spinoml_count: 0 } })
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      set({ status: { kind: 'error', error: msg } })
    }
  },

  init: async (name, description, goal) => {
    const meta = await projectBackend.init(name, description, goal)
    const conn = getCurrentConnection()
    const root = conn.kind === 'remote-ssh' ? conn.root : await tauriFs.currentDir()
    if (!root) {
      set({ status: { kind: 'error', error: 'Tauri-Laufwerk konnte nicht gelesen werden — Ordner geöffnet?' } })
      return
    }
    set({ status: { kind: 'loaded', root, meta } })
    if (conn.kind !== 'remote-ssh') setActiveWorkspace(root)
    await bootstrapWorkspace(root)
    if (conn.kind === 'remote-ssh') {
      void useRemoteSidecarStore.getState().ensure(sshTarget(conn), root)
    }
  },

  migrate: async (name, description, goal) => {
    // Legacy migration only makes sense for local — remote starts fresh.
    const meta = await projectBackend.migrateLocal(name, description, goal)
    const current = await tauriFs.currentDir()
    if (!current) {
      set({ status: { kind: 'error', error: 'Tauri-Laufwerk konnte nicht gelesen werden — Ordner geöffnet?' } })
      return
    }
    set({ status: { kind: 'loaded', root: current, meta } })
    setActiveWorkspace(current)
    await bootstrapWorkspace(current)
  },

  patch: async (p) => {
    const meta = await projectBackend.update(p)
    const cur = get().status
    if (cur.kind === 'loaded') {
      set({ status: { ...cur, meta } })
    }
  },

  pickFolder: async () => {
    // Switch to local connection so the file dialog and downstream calls
    // route through the local backend even if remote was active before.
    useConnectionsStore.getState().setCurrent('local')
    const root = await tauriFs.pickDir()
    if (!root) return
    await get().refresh()
    if (get().status.kind !== 'error') addRecentWorkspace(root)
  },

  openLocalPath: async (path) => {
    useConnectionsStore.getState().setCurrent('local')
    try {
      const root = await tauriFs.setDir(path)
      await get().refresh()
      if (get().status.kind !== 'error') addRecentWorkspace(root)
    } catch (e) {
      set({ status: { kind: 'error', error: e instanceof Error ? e.message : String(e) } })
    }
  },

  openConnection: async (connectionId) => {
    useConnectionsStore.getState().setCurrent(connectionId)
    await get().refresh()
  },

  closeProject: async () => {
    const conn = getCurrentConnection()
    if (conn.kind === 'remote-ssh') {
      try { await tauriSsh.close() }
      catch {
        // Best-effort cleanup while closing the project; the connection target
        // is being dropped anyway, so a failure cannot make the UI lie.
      }
      try { await useRemoteSidecarStore.getState().stop() }
      catch {
        // Best-effort sidecar teardown on close; the store flips to 'stopped'
        // regardless and no stale "running" status is presented.
      }
    } else {
      try { await tauriFs.closeDir() }
      catch {
        // Best-effort local directory cleanup on close; the workspace binding is
        // cleared below so no directory is falsely reported as open.
      }
    }
    set({ status: { kind: 'none' } })
    // An explicit close must NOT auto-reopen on the next launch.
    setActiveWorkspace(null)
    useWorkspaceStore.setState({ mode: 'browser', workspaceRoot: null })
    useConnectionsStore.getState().setCurrent('local')
  },
}))

async function bootstrapWorkspace(root: string) {
  const ws = useWorkspaceStore.getState()
  // Re-bootstrapping the SAME workspace (e.g. a project refresh that fires while
  // the window is backgrounded on a remote/HPC connection) must NOT close the open
  // .spinoml or wipe the tree — just re-sync entries from disk and KEEP the active
  // file. Only a genuine workspace switch (different root) resets activeFileId.
  // Without this, a re-check closed the open model on the canvas (and the chat
  // panel unmounted as the app fell back to Welcome).
  if (ws.mode === 'tauri' && ws.workspaceRoot === root && ws.entries[ROOT_ID]) {
    await ws.refreshFromDisk()
    const after = useWorkspaceStore.getState()
    // Invariant: activeFileId must reference an existing entry (or be null) — if
    // the open file vanished on disk between checks, drop the binding.
    if (after.activeFileId && !after.entries[after.activeFileId]) {
      useWorkspaceStore.setState({ activeFileId: null, dirty: false })
    }
    return
  }
  // Capture the persisted open-file BEFORE the reset below — setting activeFileId
  // to null fires the persist subscription, which would otherwise clear it first.
  const last = getActiveFile(root)
  const rootName = root.split('/').filter(Boolean).pop() || 'workspace'
  useWorkspaceStore.setState({
    mode: 'tauri',
    workspaceRoot: root,
    entries: {
      [ROOT_ID]: { kind: 'folder', id: ROOT_ID, name: rootName, parentId: null, childIds: [] },
    },
    activeFileId: null,
    dirty: false,
    expanded: new Set([ROOT_ID]),
  })
  await useWorkspaceStore.getState().refreshFromDisk()
  // Restore the file BINDING the user had open in this workspace (survives a
  // webview reload / remote re-check) so the canvas shows the model instead of the
  // chooser. Binding only — the graph CONTENT (incl. unsaved edits) is restored
  // separately by the autosave, so we must NOT reload the file here and clobber it.
  if (last && useWorkspaceStore.getState().entries[last]?.kind === 'file') {
    useWorkspaceStore.setState({ activeFileId: last, dirty: false })
  }
}
