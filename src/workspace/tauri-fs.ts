// Tauri-only filesystem adapter. The frontend can detect at runtime whether
// we're running inside the Tauri webview (window.__TAURI_INTERNALS__) and only
// import this module's invokers when so. The Rust side scopes everything to a
// single user-picked workspace root.

import { invoke } from '@tauri-apps/api/core'

export type FsEntry = {
  name: string
  relpath: string
  is_dir: boolean
}

export function isTauri(): boolean {
  return typeof window !== 'undefined'
    && typeof (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__ !== 'undefined'
}

export type DatasetEntry = {
  name: string
  relpath: string
  abspath: string
  is_dir: boolean
  size_bytes: number
}

export const tauriFs = {
  pickDir: () => invoke<string | null>('pick_workspace_dir'),
  setDir: (path: string) => invoke<string>('set_workspace_dir', { path }),
  currentDir: () => invoke<string | null>('current_workspace_dir'),
  closeDir: () => invoke<void>('close_workspace_dir'),
  list: () => invoke<FsEntry[]>('list_workspace'),
  read: (relpath: string) => invoke<string>('read_workspace_file', { relpath }),
  write: (relpath: string, content: string) =>
    invoke<void>('write_workspace_file', { relpath, content }),
  remove: (relpath: string) => invoke<void>('delete_workspace_path', { relpath }),
  mkdir: (relpath: string) => invoke<void>('mkdir_workspace', { relpath }),
  rename: (fromRel: string, toRel: string) =>
    invoke<void>('rename_workspace_path', { fromRel, toRel }),
  sidecarManagedStatus: () =>
    invoke<{ torch: boolean; llm: boolean }>('sidecar_managed_status'),
  listDatasets: () => invoke<DatasetEntry[]>('list_datasets'),
  datasetAbspath: (relpath: string) =>
    invoke<string>('dataset_abspath', { relpath }),
  loadProject: () => invoke<ProjectLoad>('load_project'),
  initProject: (name: string, description: string, goal: string) =>
    invoke<ProjectMeta>('init_project', { name, description, goal }),
  updateProjectMeta: (patch: ProjectMetaPatch) =>
    invoke<ProjectMeta>('update_project_meta', { patch }),
  migrateLegacyProject: (name: string, description: string, goal: string) =>
    invoke<ProjectMeta>('migrate_legacy_project', { name, description, goal }),
  listNotes: () => invoke<NoteEntry[]>('list_notes'),
  readNote: (name: string) => invoke<string>('read_note', { name }),
  writeNote: (name: string, content: string) =>
    invoke<void>('write_note', { name, content }),
  appendNote: (name: string, content: string) =>
    invoke<void>('append_note', { name, content }),
  appendExperiment: (filename: string, line: string) =>
    invoke<void>('append_experiment', { filename, line }),
  readExperiment: (filename: string) =>
    invoke<string>('read_experiment', { filename }),
}

export type ProjectMeta = {
  name: string
  description: string
  goal: string
  active_model: string | null
  active_dataset: string | null
  created_at: string
  updated_at: string
  schema_version: number
}

export type ProjectMetaPatch = Partial<{
  name: string
  description: string
  goal: string
  active_model: string | null
  active_dataset: string | null
}>

export type ProjectLoad = {
  root: string
  meta: ProjectMeta | null
  has_legacy_files: boolean
  legacy_spinoml_count: number
}

export type NoteEntry = {
  name: string
  relpath: string
  size_bytes: number
  modified_at: string
}
