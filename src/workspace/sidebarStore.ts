import { create } from 'zustand'

// The left sidebar's active tab, lifted out of <LeftSidebar/> so cross-links
// (e.g. a dataset row in the Files tab) can jump to another tab. Also holds the
// Files-explorer section collapse state, which must survive remounts (the
// explorer unmounts whenever you switch tabs).
export type SidebarTab = 'files' | 'datasets' | 'experiments'

type SidebarState = {
  tab: SidebarTab
  setTab: (tab: SidebarTab) => void
  collapsed: Record<string, boolean>
  toggleSection: (key: string) => void
}

export const useSidebarStore = create<SidebarState>((set) => ({
  tab: 'files',
  setTab: (tab) => set({ tab }),
  collapsed: { misc: true }, // SONSTIGES starts collapsed
  toggleSection: (key) =>
    set((s) => ({ collapsed: { ...s.collapsed, [key]: !s.collapsed[key] } })),
}))
