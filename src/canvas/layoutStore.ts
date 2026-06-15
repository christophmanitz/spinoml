import { create } from 'zustand'

// Flow direction for the canvas: 'TB' = top→bottom, 'LR' = left→right. Drives
// both auto-layout axes and node handle positions. Persisted.
export type FlowDir = 'TB' | 'LR'

const KEY = 'mlforge.flowdir.v1'

function load(): FlowDir {
  try { return localStorage.getItem(KEY) === 'LR' ? 'LR' : 'TB' } catch { return 'TB' }
}

type State = {
  direction: FlowDir
  setDirection: (d: FlowDir) => void
  toggle: () => void
}

export const useLayoutStore = create<State>((set, get) => ({
  direction: load(),
  setDirection: (d) => {
    try { localStorage.setItem(KEY, d) } catch { /* quota */ }
    set({ direction: d })
  },
  toggle: () => get().setDirection(get().direction === 'TB' ? 'LR' : 'TB'),
}))
