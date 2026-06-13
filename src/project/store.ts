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

export type ProjectStatus =
  | { kind: 'none' }
  | { kind: 'loading' }
  | { kind: 'legacy'; root: string; legacy_mlforge_count: number }
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
    set({ status: { kind: 'loading' } })
    try {
      if (conn.kind === 'remote-ssh') {
        const load = await projectBackend.load()
        if (load.meta) {
          set({ status: { kind: 'loaded', root: load.root, meta: load.meta } })
          await bootstrapWorkspace(load.root)
        } else if (!load.rootExists) {
          set({ status: { kind: 'remote-missing', root: load.root, alias: conn.alias } })
        } else {
          // Root exists but no project.json: treat as needing init. We don't
          // currently surface "legacy" on remote (no .mlforge migration story).
          set({ status: { kind: 'remote-missing', root: load.root, alias: conn.alias } })
        }
        return
      }
      const current = await tauriFs.currentDir()
      if (!current) {
        set({ status: { kind: 'none' } })
        return
      }
      const load = await projectBackend.load()
      if (load.meta) {
        set({ status: { kind: 'loaded', root: load.root, meta: load.meta } })
        await bootstrapWorkspace(load.root)
      } else if (load.hasLegacyFiles) {
        set({ status: { kind: 'legacy', root: load.root, legacy_mlforge_count: load.legacyMlforgeCount } })
      } else {
        set({ status: { kind: 'legacy', root: load.root, legacy_mlforge_count: 0 } })
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      set({ status: { kind: 'error', error: msg } })
    }
  },

  init: async (name, description, goal) => {
    const meta = await projectBackend.init(name, description, goal)
    const conn = getCurrentConnection()
    const root = conn.kind === 'remote-ssh' ? conn.root : (await tauriFs.currentDir())!
    set({ status: { kind: 'loaded', root, meta } })
    await bootstrapWorkspace(root)
  },

  migrate: async (name, description, goal) => {
    // Legacy migration only makes sense for local — remote starts fresh.
    const meta = await projectBackend.migrateLocal(name, description, goal)
    const current = (await tauriFs.currentDir())!
    set({ status: { kind: 'loaded', root: current, meta } })
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
  },

  openConnection: async (connectionId) => {
    useConnectionsStore.getState().setCurrent(connectionId)
    await get().refresh()
  },

  closeProject: async () => {
    const conn = getCurrentConnection()
    if (conn.kind === 'remote-ssh') {
      try { await tauriSsh.close() } catch { /* ignore */ }
    } else {
      try { await tauriFs.closeDir() } catch { /* ignore */ }
    }
    set({ status: { kind: 'none' } })
    useWorkspaceStore.setState({ mode: 'browser', workspaceRoot: null })
    useConnectionsStore.getState().setCurrent('local')
  },
}))

async function bootstrapWorkspace(root: string) {
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
}
