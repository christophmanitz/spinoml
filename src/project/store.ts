import { create } from 'zustand'
import { isTauri, tauriFs, type ProjectMeta, type ProjectMetaPatch } from '../workspace/tauri-fs'
import { useWorkspaceStore, ROOT_ID } from '../workspace/store'

export type ProjectStatus =
  | { kind: 'none' }
  | { kind: 'loading' }
  | { kind: 'legacy'; root: string; legacy_mlforge_count: number }
  | { kind: 'loaded'; root: string; meta: ProjectMeta }
  | { kind: 'error'; error: string }

type State = {
  status: ProjectStatus
  refresh: () => Promise<void>
  init: (name: string, description: string, goal: string) => Promise<void>
  migrate: (name: string, description: string, goal: string) => Promise<void>
  patch: (p: ProjectMetaPatch) => Promise<void>
  pickFolder: () => Promise<void>
  closeProject: () => Promise<void>
}

export const useProjectStore = create<State>((set, get) => ({
  status: { kind: 'none' },

  refresh: async () => {
    if (!isTauri()) {
      set({ status: { kind: 'none' } })
      return
    }
    set({ status: { kind: 'loading' } })
    try {
      const current = await tauriFs.currentDir()
      if (!current) {
        set({ status: { kind: 'none' } })
        return
      }
      const load = await tauriFs.loadProject()
      if (load.meta) {
        set({ status: { kind: 'loaded', root: load.root, meta: load.meta } })
        await bootstrapWorkspace(load.root)
      } else if (load.has_legacy_files) {
        set({ status: { kind: 'legacy', root: load.root, legacy_mlforge_count: load.legacy_mlforge_count } })
      } else {
        set({ status: { kind: 'legacy', root: load.root, legacy_mlforge_count: 0 } })
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      set({ status: { kind: 'error', error: msg } })
    }
  },

  init: async (name, description, goal) => {
    const meta = await tauriFs.initProject(name, description, goal)
    const current = (await tauriFs.currentDir())!
    set({ status: { kind: 'loaded', root: current, meta } })
    await bootstrapWorkspace(current)
  },

  migrate: async (name, description, goal) => {
    const meta = await tauriFs.migrateLegacyProject(name, description, goal)
    const current = (await tauriFs.currentDir())!
    set({ status: { kind: 'loaded', root: current, meta } })
    await bootstrapWorkspace(current)
  },

  patch: async (p) => {
    const meta = await tauriFs.updateProjectMeta(p)
    const cur = get().status
    if (cur.kind === 'loaded') {
      set({ status: { ...cur, meta } })
    }
  },

  pickFolder: async () => {
    const root = await tauriFs.pickDir()
    if (!root) return
    await get().refresh()
  },

  closeProject: async () => {
    await tauriFs.closeDir()
    set({ status: { kind: 'none' } })
    // Hand workspace store back to browser mode without going through its picker.
    useWorkspaceStore.setState({ mode: 'browser', workspaceRoot: null })
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
