import { create } from 'zustand'
import { isTauri, type DatasetEntry } from '../workspace/tauri-fs'
import { inspectDataset, statsDataset, smokeDataset } from './client'
import type { InspectResult, StatsResult, SmokeResult } from './types'
import { useGraphStore } from '../canvas/GraphStore'
import { LAYERS } from '../layers/registry'
import { generate } from '../codegen/generator'
import { useProjectStore } from '../project/store'
import { useWorkspaceStore } from '../workspace/store'
import { datasets as datasetsBackend, experiments as experimentsBackend } from '../connections/backend'
import { assertTrusted, UntrustedCodeError } from '../trust/guard'

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

// Phase 40 — per-dataset monotonic sequence guards. A later inspect/stats/
// smoke request must never be overwritten by an earlier one that resolves
// late (e.g. dsA→dsB→dsA before dsA#1 returns). Each relpath has its own
// sequence; the response is applied only if it is still the latest.
const inspectSeq = new Map<string, number>()
const statsSeq = new Map<string, number>()
const smokeSeq = new Map<string, number>()
// Phase 41 — refresh() is a list operation that can race with itself and
// with inspect/stats/smoke (which capture entryByRel before the await).
let refreshSeq = 0

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
    const seq = ++refreshSeq
    set({ listLoading: true, listError: null })
    try {
      const entries = await datasetsBackend.list()
      if (seq !== refreshSeq) return // stale — a newer refresh() started
      set({ entries, listLoading: false, listError: null })
    } catch (e) {
      if (seq !== refreshSeq) return
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
    const seq = (inspectSeq.get(relpath) ?? 0) + 1
    inspectSeq.set(relpath, seq)
    set({ inspects: { ...get().inspects, [relpath]: { loading: true, data: null, error: null } } })
    const result = await inspectDataset(entry.abspath)
    if (seq !== inspectSeq.get(relpath)) return // stale — a newer inspect started
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
    const seq = (statsSeq.get(relpath) ?? 0) + 1
    statsSeq.set(relpath, seq)
    set({ stats: { ...get().stats, [relpath]: { loading: true, data: null, error: null } } })
    const result = await statsDataset(entry.abspath)
    if (seq !== statsSeq.get(relpath)) return // stale
    if ('offline' in result && result.offline) {
      set({ stats: { ...get().stats, [relpath]: { loading: false, data: null, error: result.error } } })
      return
    }
    set({ stats: { ...get().stats, [relpath]: { loading: false, data: result as StatsResult, error: null } } })
  },

  runSmoke: async (relpath, inputShape) => {
    const entry = entryByRel(get().entries, relpath)
    if (!entry) return
    const seq = (smokeSeq.get(relpath) ?? 0) + 1
    smokeSeq.set(relpath, seq)
    const graphRev = useGraphStore.getState().revision
    const { nodes, edges } = useGraphStore.getState()
    const { code, inputs } = generate(nodes, edges)
    // Per-input dataset binding: if every Input node has a bound dataset (via
    // its 'dataset' param), use the multi-dataset smoke endpoint. Otherwise
    // fall back to broadcasting the clicked dataset to every input.
    // Input-kind nodes in node order = the codegen forward-arg order. Includes
    // the whole-graph `Graph` node (kind input), which binds 1:1 to a graph
    // dataset / manifest branch.
    // Kind-driven so every input-kind node (Input/Graph/Sequence/ESPF/…) is covered.
    const inputNodes = nodes.filter((n) => LAYERS[n.data.layerType]?.kind === 'input')
    // A connected Manifest node's dataset (the manifest is its own node now) wins
    // over the input's own `dataset` (standalone graph dataset).
    const manifestRelFor = (id: string): string => {
      for (const e of edges) {
        if (e.target !== id) continue
        const src = nodes.find((n) => n.id === e.source)
        if (src && src.data.layerType === 'Manifest') return String(src.data.params.dataset ?? '')
      }
      return ''
    }
    const perInputDatasets = inputNodes.map((n) => manifestRelFor(n.id) || String(n.data.params.dataset ?? ''))
    const perInputOptions = inputNodes.map((n) => {
      // A Graph node pulls the WHOLE graph (x/edge_index/batch/edge_attr) from
      // one source; the sidecar assembles a Data. `branch` selects the manifest
      // branch ('ligand' → 'ligand.x', …); empty for single-graph datasets.
      if (n.data.layerType === 'Graph') {
        const branch = String(n.data.params.branch ?? '')
        const opt: { graph: true; branch?: string } = { graph: true }
        if (branch) opt.branch = branch
        return opt
      }
      const feats = n.data.params.features as string[] | undefined
      const target = n.data.params.target as string | undefined
      const opt: { features?: string[]; target?: string; field?: string } = {}
      if (Array.isArray(feats) && feats.length) opt.features = feats
      if (target) opt.target = target
      // A non-graph input fed by a manifest branch reads '<branch>.x'; else an
      // explicit bind_field slot, else the Input's name (graph-dataset field).
      const branch = String(n.data.params.branch ?? '')
      const bindField = n.data.params.bind_field
      opt.field = bindField ? String(bindField) : branch ? `${branch}.x` : String(n.data.params.name ?? 'x')
      return opt
    })
    const allBound = inputs.length > 1 && perInputDatasets.every((d) => d.length > 0)
    const shapes = inputShape ? [inputShape] : inputs.map((i) => i.shape)
    set({ smoke: { ...get().smoke, [relpath]: { loading: true, data: null, error: null } } })

    // Phase 43 — never ship unapproved generated code to /dataset/smoke.
    try {
      await assertTrusted(nodes)
    } catch (e) {
      if (e instanceof UntrustedCodeError) {
        if (seq !== smokeSeq.get(relpath)) return
        set({ smoke: { ...get().smoke, [relpath]: { loading: false, data: null, error: e.message } } })
        return
      }
      throw e
    }

    let result
    if (allBound) {
      const abspaths: string[] = []
      for (const rel of perInputDatasets) {
        const e = entryByRel(get().entries, rel)
        if (!e) {
          if (seq !== smokeSeq.get(relpath)) return
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
    // Phase 40 — stale guard: graph moved or newer smoke started → drop
    if (seq !== smokeSeq.get(relpath) || graphRev !== useGraphStore.getState().revision) return
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
