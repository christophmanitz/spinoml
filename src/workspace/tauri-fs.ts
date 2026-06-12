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

export const tauriFs = {
  pickDir: () => invoke<string | null>('pick_workspace_dir'),
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
}
