// Typed wrappers + live status for the Phase 12b remote torch-sidecar.
// The Rust side emits `remote-sidecar:status` events through every phase of
// bootstrap (probe → install → deploy → starting → running). We mirror that
// into a Zustand store so the UI can show progress and disable smoke tests
// while it's still booting.

import { invoke } from '@tauri-apps/api/core'
import { listen, type UnlistenFn } from '@tauri-apps/api/event'
import { create } from 'zustand'
import { isTauri } from '../workspace/tauri-fs'

export type RemoteSidecarStatus =
  | { kind: 'idle' }
  | { kind: 'preparing'; phase: string; message: string }
  | { kind: 'starting' }
  | { kind: 'running'; local_port: number; remote_port: number; alias: string; root: string }
  | { kind: 'stopped' }
  | { kind: 'error'; message: string }

type State = {
  status: RemoteSidecarStatus
  ensure: (alias: string, root: string, force?: boolean) => Promise<void>
  stop: () => Promise<void>
  refresh: () => Promise<void>
}

export const useRemoteSidecarStore = create<State>((set) => ({
  status: { kind: 'idle' },

  ensure: async (alias, root, force = false) => {
    if (!isTauri()) return
    if (force) set({ status: { kind: 'preparing', phase: 'probe', message: 'Neu verbinden…' } })
    try {
      const s = await invoke<RemoteSidecarStatus>('ensure_remote_sidecar', { alias, root, force })
      set({ status: s })
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      set({ status: { kind: 'error', message: msg } })
    }
  },

  stop: async () => {
    if (!isTauri()) return
    try { await invoke('stop_remote_sidecar') }
    catch {
      // Best-effort teardown: we still mark the sidecar stopped so no stale
      // running tunnel is shown as current.
    }
    set({ status: { kind: 'stopped' } })
  },

  refresh: async () => {
    if (!isTauri()) return
    try {
      const s = await invoke<RemoteSidecarStatus>('remote_sidecar_status')
      set({ status: s })
    } catch (e) {
      // Can't confirm the sidecar state → do not keep displaying the previous
      // status as if it were current; show an explicit unknown/error instead.
      set({ status: { kind: 'error', message: `Sidecar-Status unbekannt: ${e instanceof Error ? e.message : String(e)}` } })
    }
  },
}))

// Subscribe to Rust event stream once at module load.
let unlisten: UnlistenFn | null = null
if (isTauri()) {
  void (async () => {
    unlisten = await listen<RemoteSidecarStatus>('remote-sidecar:status', (e) => {
      useRemoteSidecarStore.setState({ status: e.payload })
    })
  })()
}

if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => { unlisten?.() })
}
