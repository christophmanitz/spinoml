import { create } from 'zustand'

// Flow direction for the canvas: 'TB' = top→bottom, 'LR' = left→right. Drives
// both auto-layout axes and node handle positions. Persisted.
export type FlowDir = 'TB' | 'LR'

const KEY = 'spinoml.flowdir.v1'

function load(): FlowDir {
  try { return localStorage.getItem(KEY) === 'LR' ? 'LR' : 'TB' }
  catch {
    // localStorage unavailable (private mode/SSR): default flow direction is
    // harmless and is not a claim about any stored state.
    return 'TB'
  }
}

type State = {
  direction: FlowDir
  setDirection: (d: FlowDir) => void
  toggle: () => void
}

export const useLayoutStore = create<State>((set, get) => ({
  direction: load(),
  setDirection: (d) => {
    try { localStorage.setItem(KEY, d) }
    catch {
      // Quota/private mode: only the cross-reload persistence of the flow
      // direction is lost; the live direction is still applied below.
    }
    set({ direction: d })
  },
  toggle: () => get().setDirection(get().direction === 'TB' ? 'LR' : 'TB'),
}))
