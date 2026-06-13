import { create } from 'zustand'
import { isTauri, tauriFs, type DatasetEntry } from '../workspace/tauri-fs'
import { inspectDataset, statsDataset, smokeDataset } from './client'
import type { InspectResult, StatsResult, SmokeResult } from './types'
import { useGraphStore } from '../canvas/GraphStore'
import { generate } from '../codegen/generator'

type Cached<T> = {
  loading: boolean
  data: T | null
  error: string | null
}

type DatasetsState = {
  entries: DatasetEntry[]
  selectedRel: string | null
  listLoading: boolean
  listError: string | null

  inspects: Record<string, Cached<InspectResult>>
  stats: Record<string, Cached<StatsResult>>
  smoke: Record<string, Cached<SmokeResult>>

  refresh: () => Promise<void>
  select: (relpath: string | null) => void
  inspect: (relpath: string, force?: boolean) => Promise<void>
  loadStats: (relpath: string, force?: boolean) => Promise<void>
  runSmoke: (relpath: string, inputShape?: number[]) => Promise<void>
}

function entryByRel(entries: DatasetEntry[], rel: string): DatasetEntry | undefined {
  return entries.find((e) => e.relpath === rel)
}

export const useDatasetsStore = create<DatasetsState>((set, get) => ({
  entries: [],
  selectedRel: null,
  listLoading: false,
  listError: null,
  inspects: {},
  stats: {},
  smoke: {},

  refresh: async () => {
    if (!isTauri()) {
      set({ entries: [], listError: 'datasets require Tauri (real filesystem)' })
      return
    }
    set({ listLoading: true, listError: null })
    try {
      const entries = await tauriFs.listDatasets()
      set({ entries, listLoading: false })
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e)
      set({ listLoading: false, listError: msg })
    }
  },

  select: (relpath) => {
    set({ selectedRel: relpath })
    if (relpath) {
      void get().inspect(relpath)
    }
  },

  inspect: async (relpath, force = false) => {
    const cached = get().inspects[relpath]
    if (!force && cached?.data) return
    const entry = entryByRel(get().entries, relpath)
    if (!entry) {
      set({ inspects: { ...get().inspects, [relpath]: { loading: false, data: null, error: 'no such dataset' } } })
      return
    }
    set({ inspects: { ...get().inspects, [relpath]: { loading: true, data: null, error: null } } })
    const result = await inspectDataset(entry.abspath)
    if ('offline' in result && result.offline) {
      set({ inspects: { ...get().inspects, [relpath]: { loading: false, data: null, error: result.error } } })
      return
    }
    set({ inspects: { ...get().inspects, [relpath]: { loading: false, data: result as InspectResult, error: null } } })
  },

  loadStats: async (relpath, force = false) => {
    const cached = get().stats[relpath]
    if (!force && cached?.data) return
    const entry = entryByRel(get().entries, relpath)
    if (!entry) return
    set({ stats: { ...get().stats, [relpath]: { loading: true, data: null, error: null } } })
    const result = await statsDataset(entry.abspath)
    if ('offline' in result && result.offline) {
      set({ stats: { ...get().stats, [relpath]: { loading: false, data: null, error: result.error } } })
      return
    }
    set({ stats: { ...get().stats, [relpath]: { loading: false, data: result as StatsResult, error: null } } })
  },

  runSmoke: async (relpath, inputShape) => {
    const entry = entryByRel(get().entries, relpath)
    if (!entry) return
    const { nodes, edges } = useGraphStore.getState()
    const { code, inputShape: defaultShape } = generate(nodes, edges)
    const shape = inputShape ?? defaultShape ?? undefined
    set({ smoke: { ...get().smoke, [relpath]: { loading: true, data: null, error: null } } })
    const result = await smokeDataset(code, entry.abspath, shape)
    if ('offline' in result && result.offline) {
      set({ smoke: { ...get().smoke, [relpath]: { loading: false, data: null, error: result.error } } })
      return
    }
    set({ smoke: { ...get().smoke, [relpath]: { loading: false, data: result as SmokeResult, error: null } } })
  },
}))
