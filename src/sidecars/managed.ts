import { create } from 'zustand'
import { isTauri, tauriFs } from '../workspace/tauri-fs'

type ManagedState = {
  /** Did the Rust shell spawn the torch sidecar at boot? */
  torch: boolean
  /** Did the Rust shell spawn the LLM sidecar at boot? */
  llm: boolean
  /** True while we're still waiting for the initial status response. */
  unknown: boolean
}

export const useManagedSidecars = create<ManagedState>(() => ({
  torch: false,
  llm: false,
  unknown: isTauri(),
}))

if (isTauri()) {
  tauriFs
    .sidecarManagedStatus()
    .then((s) => useManagedSidecars.setState({ ...s, unknown: false }))
    .catch(() => useManagedSidecars.setState({ unknown: false }))
}
