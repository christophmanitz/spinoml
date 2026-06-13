// Single dispatch layer between higher-level stores and either the local
// Tauri filesystem (tauriFs) or the remote SSH filesystem (tauriSsh).
// Other code MUST NOT import tauri-fs or tauri-ssh directly for these
// concerns — go through `fs`, `project`, `notes`, `experiments`, `datasets`
// here. That way adding a third backend later (e.g. sftp-mounted, or a
// remote sidecar with side-channel writes) is one file.

import {
  tauriFs,
  type FsEntry,
  type ProjectMeta,
  type ProjectMetaPatch,
  type NoteEntry,
  type DatasetEntry,
} from '../workspace/tauri-fs'
import { tauriSsh } from './tauri-ssh'
import { getCurrentConnection } from './store'

export type ProjectLoadResult = {
  root: string
  meta: ProjectMeta | null
  rootExists: boolean
  hasLegacyFiles: boolean
  legacyMlforgeCount: number
}

export const fs = {
  list: (): Promise<FsEntry[]> => {
    const c = getCurrentConnection()
    return c.kind === 'remote-ssh' ? tauriSsh.walk(c.alias, c.root) : tauriFs.list()
  },
  read: (relpath: string): Promise<string> => {
    const c = getCurrentConnection()
    return c.kind === 'remote-ssh'
      ? tauriSsh.readFile(c.alias, c.root, relpath)
      : tauriFs.read(relpath)
  },
  write: (relpath: string, content: string): Promise<void> => {
    const c = getCurrentConnection()
    return c.kind === 'remote-ssh'
      ? tauriSsh.writeFile(c.alias, c.root, relpath, content)
      : tauriFs.write(relpath, content)
  },
  remove: (relpath: string): Promise<void> => {
    const c = getCurrentConnection()
    return c.kind === 'remote-ssh'
      ? tauriSsh.deletePath(c.alias, c.root, relpath)
      : tauriFs.remove(relpath)
  },
  mkdir: (relpath: string): Promise<void> => {
    const c = getCurrentConnection()
    return c.kind === 'remote-ssh'
      ? tauriSsh.mkdir(c.alias, c.root, relpath)
      : tauriFs.mkdir(relpath)
  },
  rename: (fromRel: string, toRel: string): Promise<void> => {
    const c = getCurrentConnection()
    return c.kind === 'remote-ssh'
      ? tauriSsh.rename(c.alias, c.root, fromRel, toRel)
      : tauriFs.rename(fromRel, toRel)
  },
}

export const project = {
  load: async (): Promise<ProjectLoadResult> => {
    const c = getCurrentConnection()
    if (c.kind === 'remote-ssh') {
      const r = await tauriSsh.loadProject(c.alias, c.root)
      return {
        root: r.root,
        meta: r.meta,
        rootExists: r.root_exists,
        hasLegacyFiles: r.has_legacy_files,
        legacyMlforgeCount: r.legacy_mlforge_count,
      }
    }
    const r = await tauriFs.loadProject()
    return {
      root: r.root,
      meta: r.meta,
      rootExists: true,
      hasLegacyFiles: r.has_legacy_files,
      legacyMlforgeCount: r.legacy_mlforge_count,
    }
  },
  init: (name: string, description: string, goal: string): Promise<ProjectMeta> => {
    const c = getCurrentConnection()
    if (c.kind === 'remote-ssh') {
      return tauriSsh.initProject(c.alias, c.root, name, description, goal)
    }
    return tauriFs.initProject(name, description, goal)
  },
  update: (patch: ProjectMetaPatch): Promise<ProjectMeta> => {
    const c = getCurrentConnection()
    if (c.kind === 'remote-ssh') {
      return tauriSsh.updateProjectMeta(c.alias, c.root, patch as Record<string, unknown>)
    }
    return tauriFs.updateProjectMeta(patch)
  },
  // Legacy migration is local-only — remote projects start fresh.
  migrateLocal: (name: string, description: string, goal: string): Promise<ProjectMeta> =>
    tauriFs.migrateLegacyProject(name, description, goal),
}

export const notes = {
  list: (): Promise<NoteEntry[]> => {
    const c = getCurrentConnection()
    return c.kind === 'remote-ssh' ? tauriSsh.listNotes(c.alias, c.root) : tauriFs.listNotes()
  },
  read: (name: string): Promise<string> => {
    const c = getCurrentConnection()
    return c.kind === 'remote-ssh'
      ? tauriSsh.readNote(c.alias, c.root, name)
      : tauriFs.readNote(name)
  },
  write: (name: string, content: string): Promise<void> => {
    const c = getCurrentConnection()
    return c.kind === 'remote-ssh'
      ? tauriSsh.writeNote(c.alias, c.root, name, content)
      : tauriFs.writeNote(name, content)
  },
  append: (name: string, content: string): Promise<void> => {
    const c = getCurrentConnection()
    return c.kind === 'remote-ssh'
      ? tauriSsh.appendNote(c.alias, c.root, name, content)
      : tauriFs.appendNote(name, content)
  },
}

export const experiments = {
  read: (filename: string): Promise<string> => {
    const c = getCurrentConnection()
    return c.kind === 'remote-ssh'
      ? tauriSsh.readExperiment(c.alias, c.root, filename)
      : tauriFs.readExperiment(filename)
  },
  append: (filename: string, line: string): Promise<void> => {
    const c = getCurrentConnection()
    return c.kind === 'remote-ssh'
      ? tauriSsh.appendExperiment(c.alias, c.root, filename, line)
      : tauriFs.appendExperiment(filename, line)
  },
}

export const datasets = {
  list: (): Promise<DatasetEntry[]> => {
    const c = getCurrentConnection()
    return c.kind === 'remote-ssh' ? tauriSsh.listDatasets(c.alias, c.root) : tauriFs.listDatasets()
  },
  abspath: (relpath: string): Promise<string> => {
    const c = getCurrentConnection()
    if (c.kind === 'remote-ssh') {
      // Remote datasets are not addressable by the local sidecar. The frontend
      // should detect this case before dispatching a smoke test and surface a
      // clear message. We still return a path-looking string so callers don't
      // crash on undefined.
      const root = c.root.endsWith('/') ? c.root.slice(0, -1) : c.root
      return Promise.resolve(`${root}/${relpath}`)
    }
    return tauriFs.datasetAbspath(relpath)
  },
}
