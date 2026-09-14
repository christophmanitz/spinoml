// File-binding adapter for the ARCHITECTURE canvas. Unlike training/data, the
// .spinoml is owned by the workspace store (open/save/close + the .py twin + dirty
// tracking already live there), so this adapter just delegates and is marked
// `external` — the gate shows the header/chooser but never reloads the file.

import { fs } from '../connections/backend'
import { useWorkspaceStore, ROOT_ID } from '../workspace/store'
import { useCanvasDocStore, type CanvasDocAdapter } from '../canvasdoc/store'

export const architectureDocAdapter: CanvasDocAdapter = {
  kind: 'architecture',
  ext: '.spinoml',
  label: 'Modell',
  external: true,
  list: async () => {
    const entries = await fs.list()
    return entries
      .filter((e) => !e.is_dir && e.relpath.toLowerCase().endsWith('.spinoml'))
      .map((e) => ({ relpath: e.relpath, name: e.name }))
      .sort((a, b) => a.relpath.localeCompare(b.relpath))
  },
  open: async (relpath) => { await useWorkspaceStore.getState().openFile(relpath) },
  create: async (name) => {
    const file = name.toLowerCase().endsWith('.spinoml') ? name : `${name}.spinoml`
    return useWorkspaceStore.getState().createFile(ROOT_ID, file)
  },
  save: async () => { await useWorkspaceStore.getState().saveActive() },
  unbind: () => useWorkspaceStore.getState().closeActive(),
}

// Keep the architecture binding mirrored to whatever the workspace has open, so
// the canvas header is correct no matter who opened the file (FileExplorer,
// Toolbar, chat, launch-restore).
export function startArchitectureDocSync(): void {
  const sync = () => useCanvasDocStore.getState().setBound('architecture', useWorkspaceStore.getState().activeFileId)
  sync()
  useWorkspaceStore.subscribe((s, prev) => {
    if (s.activeFileId !== prev.activeFileId) {
      useCanvasDocStore.getState().setBound('architecture', s.activeFileId)
    }
  })
}
