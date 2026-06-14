// Tiny global UI store for the Architecture ↔ Training canvas switch. Lives on
// its own (cross-cutting view state) so App.tsx can swap the palette / canvas /
// inspector panels without threading props.

import { create } from 'zustand'

export type ViewMode = 'architecture' | 'training'

type State = {
  mode: ViewMode
  setMode: (m: ViewMode) => void
}

export const useViewModeStore = create<State>((set) => ({
  mode: 'architecture',
  setMode: (mode) => set({ mode }),
}))
