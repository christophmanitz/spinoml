// Connection model. The app can target one of:
//   - LOCAL: existing local workspace via Tauri FS commands.
//   - REMOTE-SSH: a workspace on a remote host reached via system `ssh`.
//
// The current connection determines which set of commands the workspace,
// project, datasets and (later) terminal stores dispatch to. Connections
// are persisted to localStorage; secrets are NOT stored — auth is delegated
// to the user's ~/.ssh/config + ssh-agent (or whatever method the alias
// implies on their machine).

import { create } from 'zustand'
import { isTauri } from '../workspace/tauri-fs'
import { tauriSsh, type SshTestResult } from './tauri-ssh'

export type LocalConnection = {
  id: 'local'
  kind: 'local'
  label: string
}

export type RemoteSshConnection = {
  id: string
  kind: 'remote-ssh'
  label: string
  alias: string
  root: string
}

export type Connection = LocalConnection | RemoteSshConnection

export const LOCAL_CONNECTION: LocalConnection = {
  id: 'local',
  kind: 'local',
  label: 'Lokal',
}

const STORAGE_KEY = 'mlforge.connections.v1'

type Persisted = {
  saved: RemoteSshConnection[]
  currentId: string
}

function loadPersisted(): Persisted {
  if (typeof window === 'undefined') return { saved: [], currentId: 'local' }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return { saved: [], currentId: 'local' }
    const parsed = JSON.parse(raw)
    const saved: RemoteSshConnection[] = Array.isArray(parsed?.saved)
      ? parsed.saved.filter(
          (c: unknown): c is RemoteSshConnection =>
            !!c
            && typeof c === 'object'
            && (c as RemoteSshConnection).kind === 'remote-ssh'
            && typeof (c as RemoteSshConnection).id === 'string'
            && typeof (c as RemoteSshConnection).alias === 'string'
            && typeof (c as RemoteSshConnection).root === 'string',
        )
      : []
    const currentId = typeof parsed?.currentId === 'string' ? parsed.currentId : 'local'
    return { saved, currentId }
  } catch {
    return { saved: [], currentId: 'local' }
  }
}

function persist(state: Persisted): void {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state))
  } catch {
    /* quota / private-mode — best effort */
  }
}

function randomId(): string {
  return `rmt-${Math.random().toString(36).slice(2, 10)}`
}

type State = {
  saved: RemoteSshConnection[]
  currentId: string

  setCurrent: (id: string) => void
  addRemote: (label: string, alias: string, root: string) => RemoteSshConnection
  updateRemote: (id: string, patch: Partial<Pick<RemoteSshConnection, 'label' | 'alias' | 'root'>>) => void
  removeRemote: (id: string) => void

  testConnection: (alias: string) => Promise<SshTestResult>
}

const initial = loadPersisted()

export const useConnectionsStore = create<State>((set, get) => ({
  saved: initial.saved,
  currentId: initial.currentId,

  setCurrent: (id) => {
    set({ currentId: id })
    persist({ saved: get().saved, currentId: id })
  },

  addRemote: (label, alias, root) => {
    const conn: RemoteSshConnection = {
      id: randomId(),
      kind: 'remote-ssh',
      label: label || alias,
      alias,
      root,
    }
    const saved = [...get().saved, conn]
    set({ saved })
    persist({ saved, currentId: get().currentId })
    return conn
  },

  updateRemote: (id, patch) => {
    const saved = get().saved.map((c) => (c.id === id ? { ...c, ...patch } : c))
    set({ saved })
    persist({ saved, currentId: get().currentId })
  },

  removeRemote: (id) => {
    const saved = get().saved.filter((c) => c.id !== id)
    // If we just removed the current connection, fall back to local.
    const currentId = get().currentId === id ? 'local' : get().currentId
    set({ saved, currentId })
    persist({ saved, currentId })
  },

  testConnection: async (alias) => {
    if (!isTauri()) {
      throw new Error('SSH-Verbindungen brauchen Tauri (kein Browser-Modus).')
    }
    return tauriSsh.testConnection(alias)
  },
}))

// Helpers used by other stores to read the active backend without subscribing.

export function getCurrentConnection(): Connection {
  const s = useConnectionsStore.getState()
  return s.saved.find((c) => c.id === s.currentId) ?? LOCAL_CONNECTION
}

export function isRemoteActive(): boolean {
  return getCurrentConnection().kind === 'remote-ssh'
}

export function getActiveRemote(): RemoteSshConnection | null {
  const c = getCurrentConnection()
  return c.kind === 'remote-ssh' ? c : null
}
