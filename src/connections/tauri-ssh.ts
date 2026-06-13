// Typed wrappers around the ssh_* Tauri commands. These are the remote
// mirror of `workspace/tauri-fs.ts`. Whether the app routes through these
// or through tauri-fs depends on the current Connection (see store.ts).

import { invoke } from '@tauri-apps/api/core'
import type { ProjectMeta, NoteEntry, DatasetEntry, FsEntry } from '../workspace/tauri-fs'

export type SshTestResult = {
  ok: boolean
  uname: string
  home: string
}

export type RemoteProjectLoad = {
  root: string
  meta: ProjectMeta | null
  root_exists: boolean
  has_legacy_files: boolean
  legacy_mlforge_count: number
}

export type CurrentRemote = {
  alias: string
  root: string
}

type Patch = Record<string, unknown>

export const tauriSsh = {
  testConnection: (alias: string) =>
    invoke<SshTestResult>('ssh_test_connection', { alias }),

  current: () => invoke<CurrentRemote | null>('ssh_current'),
  close: () => invoke<void>('ssh_close'),

  loadProject: (alias: string, root: string) =>
    invoke<RemoteProjectLoad>('ssh_load_project', { alias, root }),

  initProject: (alias: string, root: string, name: string, description: string, goal: string) =>
    invoke<ProjectMeta>('ssh_init_project', { alias, root, name, description, goal }),

  updateProjectMeta: (alias: string, root: string, patch: Patch) =>
    invoke<ProjectMeta>('ssh_update_project_meta', { alias, root, patch }),

  walk: (alias: string, root: string) =>
    invoke<FsEntry[]>('ssh_walk', { alias, root }),

  readFile: (alias: string, root: string, relpath: string) =>
    invoke<string>('ssh_read_file', { alias, root, relpath }),

  writeFile: (alias: string, root: string, relpath: string, content: string) =>
    invoke<void>('ssh_write_file', { alias, root, relpath, content }),

  deletePath: (alias: string, root: string, relpath: string) =>
    invoke<void>('ssh_delete_path', { alias, root, relpath }),

  mkdir: (alias: string, root: string, relpath: string) =>
    invoke<void>('ssh_mkdir', { alias, root, relpath }),

  rename: (alias: string, root: string, fromRel: string, toRel: string) =>
    invoke<void>('ssh_rename', { alias, root, fromRel, toRel }),

  listNotes: (alias: string, root: string) =>
    invoke<NoteEntry[]>('ssh_list_notes', { alias, root }),

  readNote: (alias: string, root: string, name: string) =>
    invoke<string>('ssh_read_note', { alias, root, name }),

  writeNote: (alias: string, root: string, name: string, content: string) =>
    invoke<void>('ssh_write_note', { alias, root, name, content }),

  appendNote: (alias: string, root: string, name: string, content: string) =>
    invoke<void>('ssh_append_note', { alias, root, name, content }),

  appendExperiment: (alias: string, root: string, filename: string, line: string) =>
    invoke<void>('ssh_append_experiment', { alias, root, filename, line }),

  readExperiment: (alias: string, root: string, filename: string) =>
    invoke<string>('ssh_read_experiment', { alias, root, filename }),

  listDatasets: (alias: string, root: string) =>
    invoke<DatasetEntry[]>('ssh_list_datasets', { alias, root }),
}
