import { create } from 'zustand'
import { isTauri, type DatasetEntry } from '../workspace/tauri-fs'
import { inspectDataset, statsDataset, smokeDataset } from './client'
import type { InspectResult, StatsResult, SmokeResult } from './types'
import { useGraphStore } from '../canvas/GraphStore'
import { generate } from '../codegen/generator'
import { useProjectStore } from '../project/store'
import { useWorkspaceStore } from '../workspace/store'
import { datasets as datasetsBackend, experiments as experimentsBackend } from '../connections/backend'
import { isRemoteActive } from '../connections/store'

type Cached<T> = {
  loading: boolean
  data: T | null
  error: string | null
}

export type SmokeHistoryEntry = {
  at: string
  ok: boolean
  dataset: string
  model: string | null
  input_shape?: number[] | number[][]
  output_shape?: number[] | number[][] | null
  n_params?: number
  error?: string
  stage?: string
}

type DatasetsState = {
  entries: DatasetEntry[]
  selectedRel: string | null
  listLoading: boolean
  listError: string | null

  inspects: Record<string, Cached<InspectResult>>
  stats: Record<string, Cached<StatsResult>>
  smoke: Record<string, Cached<SmokeResult>>
  history: SmokeHistoryEntry[]

  refresh: () => Promise<void>
  select: (relpath: string | null) => void
  inspect: (relpath: string, force?: boolean) => Promise<void>
  loadStats: (relpath: string, force?: boolean) => Promise<void>
  runSmoke: (relpath: string, inputShape?: number[]) => Promise<void>
  loadHistory: () => Promise<void>
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
  history: [],

  refresh: async () => {
    if (!isTauri()) {
      set({ entries: [], listError: 'datasets require Tauri (real filesystem)' })
      return
    }
    set({ listLoading: true, listError: null })
    try {
      const entries = await datasetsBackend.list()
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
    if (isRemoteActive()) {
      set({ smoke: { ...get().smoke, [relpath]: {
        loading: false, data: null,
        error: 'Smoke-Tests gegen Remote-Datensätze brauchen einen Sidecar auf dem HPC (Phase 12b). Kopiere den Datensatz lokal um sofort zu testen.',
      } } })
      return
    }
    const { nodes, edges } = useGraphStore.getState()
    const { code, inputs } = generate(nodes, edges)
    // Per-input dataset binding: if every Input node has a bound dataset (via
    // its 'dataset' param), use the multi-dataset smoke endpoint. Otherwise
    // fall back to broadcasting the clicked dataset to every input.
    const inputNodes = nodes.filter((n) => n.data.layerType === 'Input')
    const perInputDatasets = inputNodes.map((n) => String(n.data.params.dataset ?? ''))
    const perInputOptions = inputNodes.map((n) => {
      const feats = n.data.params.features as string[] | undefined
      const target = n.data.params.target as string | undefined
      const opt: { features?: string[]; target?: string } = {}
      if (Array.isArray(feats) && feats.length) opt.features = feats
      if (target) opt.target = target
      return opt
    })
    const allBound = inputs.length > 1 && perInputDatasets.every((d) => d.length > 0)
    const shapes = inputShape ? [inputShape] : inputs.map((i) => i.shape)
    set({ smoke: { ...get().smoke, [relpath]: { loading: true, data: null, error: null } } })

    let result
    if (allBound) {
      const abspaths: string[] = []
      for (const rel of perInputDatasets) {
        const e = entryByRel(get().entries, rel)
        if (!e) {
          set({ smoke: { ...get().smoke, [relpath]: {
            loading: false, data: null,
            error: `Input bound to '${rel}' but dataset not found in workspace.`,
          } } })
          return
        }
        abspaths.push(e.abspath)
      }
      const { smokeDatasetMulti } = await import('./client')
      result = await smokeDatasetMulti(code, abspaths, shapes, perInputOptions)
    } else {
      // Single-dataset broadcast: use options for the single input (if any).
      result = await smokeDataset(code, entry.abspath, shapes, perInputOptions.slice(0, 1))
    }
    if ('offline' in result && result.offline) {
      set({ smoke: { ...get().smoke, [relpath]: { loading: false, data: null, error: result.error } } })
      return
    }
    const final = result as SmokeResult
    set({ smoke: { ...get().smoke, [relpath]: { loading: false, data: final, error: null } } })

    // Persist to experiments/smoke-results.jsonl if a project is loaded.
    if (isTauri() && useProjectStore.getState().status.kind === 'loaded') {
      const modelName = pickActiveModelName()
      const histEntry: SmokeHistoryEntry = final.ok ? {
        at: new Date().toISOString(),
        ok: true,
        dataset: relpath,
        model: modelName,
        input_shape: final.input_shape,
        output_shape: final.output_shape,
        n_params: final.n_params,
      } : {
        at: new Date().toISOString(),
        ok: false,
        dataset: relpath,
        model: modelName,
        input_shape: final.input_shape,
        stage: final.stage,
        error: final.error,
      }
      try {
        await experimentsBackend.append('smoke-results.jsonl', JSON.stringify(histEntry))
        set({ history: [histEntry, ...get().history].slice(0, 50) })
      } catch { /* logging is best-effort */ }
    }
  },

  loadHistory: async () => {
    if (!isTauri() || useProjectStore.getState().status.kind !== 'loaded') return
    try {
      const text = await experimentsBackend.read('smoke-results.jsonl')
      const lines = text.split('\n').filter((l) => l.trim())
      const items: SmokeHistoryEntry[] = []
      for (const line of lines) {
        try { items.push(JSON.parse(line)) } catch { /* skip bad lines */ }
      }
      set({ history: items.reverse().slice(0, 50) })
    } catch { /* file may not exist yet */ }
  },
}))

function pickActiveModelName(): string | null {
  const ws = useWorkspaceStore.getState()
  if (!ws.activeFileId) return null
  const e = ws.entries[ws.activeFileId]
  return e?.kind === 'file' ? e.name : null
}
