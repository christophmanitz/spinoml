import { create } from 'zustand'
import { runActivationsReq, type NodeActivation, type Weights, type ActivationsResult } from './client'
import { useGraphStore } from '../canvas/GraphStore'
import { generate } from '../codegen/generator'
import { LAYERS } from '../layers/registry'
import { useDatasetsStore } from '../datasets/store'
import { useProjectStore } from '../project/store'
import { getCurrentConnection } from '../connections/store'

// Ephemeral "Explain" state — what flows through the model when one example is
// pushed through. Kept OUT of GraphStore (heavy + non-structural; would pollute
// undo/redo + persistence, cf. CLAUDE.md invariant 6). Discarded whenever the
// graph's STRUCTURE changes (not on shape-only re-inference writes).

type VizState = {
  explainMode: boolean
  running: boolean
  error: string | null
  source: 'dataset' | 'synthetic' | null
  sampleNote: string | null
  /** Per-node activations, keyed by graph node id. */
  byNode: Record<string, NodeActivation>
  /** Per-node static weights (Linear matrix / Conv kernels), keyed by node id. */
  weightsByNode: Record<string, Weights>
  /** The example fed in (mapped to the Input node) + the final output. */
  inputNodeId: string | null

  // ─── Flow animation (the packet travelling edge-by-edge) ───
  /** Nodes the example has already reached (steady glow). */
  arrived: Record<string, boolean>
  /** The node the packet is at right now (strong pulse). */
  flowHeadId: string | null
  /** Edge ids lit while the packet crosses them. */
  flowingEdgeIds: string[]
  playing: boolean

  // ─── Trained weights (Phase D) ───
  /** Run id whose checkpoint to load; null = random init (untrained). */
  weightsRunId: string | null
  weightsSource: 'trained' | 'random' | null
  weightsNote: string | null

  setExplain: (on: boolean) => void
  toggleExplain: () => void
  setWeightsRun: (runId: string | null) => void
  run: () => Promise<void>
  playFlow: () => void
  clear: () => void
}

let inFlight: AbortController | null = null
let flowTimer: ReturnType<typeof setInterval> | null = null
function stopFlowTimer() {
  if (flowTimer) clearInterval(flowTimer)
  flowTimer = null
}

export const useVizStore = create<VizState>((set, get) => ({
  explainMode: false,
  running: false,
  error: null,
  source: null,
  sampleNote: null,
  byNode: {},
  weightsByNode: {},
  inputNodeId: null,
  arrived: {},
  flowHeadId: null,
  flowingEdgeIds: [],
  playing: false,
  weightsRunId: null,
  weightsSource: null,
  weightsNote: null,

  setExplain: (on) => {
    set({ explainMode: on })
    if (!on) get().clear()
  },
  toggleExplain: () => get().setExplain(!get().explainMode),

  setWeightsRun: (runId) => {
    set({ weightsRunId: runId })
    if (Object.keys(get().byNode).length) get().run()
  },

  clear: () => {
    if (inFlight) inFlight.abort()
    inFlight = null
    stopFlowTimer()
    set({
      byNode: {}, weightsByNode: {}, inputNodeId: null, error: null, source: null, sampleNote: null,
      running: false, arrived: {}, flowHeadId: null, flowingEdgeIds: [], playing: false,
      weightsSource: null, weightsNote: null,
    })
  },

  // Animate the example flowing through the graph: reveal nodes in topological
  // order, lighting the incoming edge as the packet arrives at each node.
  playFlow: () => {
    stopFlowTimer()
    const { nodes, edges } = useGraphStore.getState()
    const indeg = new Map(nodes.map((n) => [n.id, 0]))
    const adj = new Map<string, string[]>(nodes.map((n) => [n.id, []]))
    for (const e of edges) {
      if (indeg.has(e.target)) indeg.set(e.target, (indeg.get(e.target) ?? 0) + 1)
      adj.get(e.source)?.push(e.target)
    }
    const queue = nodes.filter((n) => (indeg.get(n.id) ?? 0) === 0).map((n) => n.id)
    const order: string[] = []
    while (queue.length) {
      const id = queue.shift()!
      order.push(id)
      for (const t of adj.get(id) ?? []) {
        indeg.set(t, (indeg.get(t) ?? 1) - 1)
        if ((indeg.get(t) ?? 0) === 0) queue.push(t)
      }
    }
    for (const n of nodes) if (!order.includes(n.id)) order.push(n.id) // cycle leftovers

    set({ arrived: {}, flowHeadId: null, flowingEdgeIds: [], playing: true })
    let k = 0
    flowTimer = setInterval(() => {
      if (k >= order.length) {
        stopFlowTimer()
        set({ flowHeadId: null, flowingEdgeIds: [], playing: false })
        return
      }
      const id = order[k]
      const arrived = { ...get().arrived, [id]: true }
      const lit = edges.filter((e) => e.target === id && arrived[e.source]).map((e) => e.id)
      set({ arrived, flowHeadId: id, flowingEdgeIds: lit })
      k++
    }, 750)
  },

  run: async () => {
    const { nodes, edges } = useGraphStore.getState()
    const { code, issues, attrMap, inputs } = generate(nodes, edges)

    if (!inputs.length || issues.some((i) => i.startsWith('Cycle'))) {
      set({ error: 'Graph braucht einen Input und darf keinen Zyklus haben.', running: false })
      return
    }

    const inputShapes = inputs.map((i) => i.shape)
    const inputDtypes = inputs.map((i) => i.dtype)

    // Graph-layer attrs: their activations are node×feature ([N,F], no batch
    // dim) — tell the sidecar to keep the full tensor, not just node 0.
    const graphAttrs = nodes
      .filter((n) => LAYERS[n.data.layerType]?.category === 'Graph')
      .map((n) => attrMap[n.id])
      .filter(Boolean) as string[]

    // Dataset binding for a REAL example (incl. real molecule/PyG graphs):
    //  - if every Input node has a `dataset` bound → one dataset per input, with
    //    field/target options (so a molecule .smi/.csv yields x + edge_index + batch);
    //  - else single selected dataset for a 1-input model;
    //  - else synthetic (the sidecar fabricates a random graph for edge_index).
    const ds = useDatasetsStore.getState()
    let abspaths: string[] | null = null
    let inputOptions: Record<string, unknown>[] | null = null

    const perInputRel = inputs.map((inp) => {
      const n = nodes.find((m) => m.id === inp.id)
      return String(n?.data.params.dataset ?? '')
    })
    if (perInputRel.every((r) => r.length > 0)) {
      const paths: string[] = []
      let ok = true
      for (const rel of perInputRel) {
        const e = ds.entries.find((x) => x.relpath === rel)
        if (!e?.abspath) { ok = false; break }
        paths.push(e.abspath)
      }
      if (ok) {
        abspaths = paths
        inputOptions = inputs.map((inp) => {
          const n = nodes.find((m) => m.id === inp.id)
          // Whole-graph (Graph node): assemble a Data from one source; branch
          // selects the manifest branch (empty for single-graph datasets).
          if (inp.isGraph) {
            const branch = String(n?.data.params.branch ?? '')
            const opt: Record<string, unknown> = { graph: true }
            if (branch) opt.branch = branch
            return opt
          }
          const opt: Record<string, unknown> = { field: String(n?.data.params.name ?? inp.name) }
          const target = n?.data.params.target as string | undefined
          const feats = n?.data.params.features as string[] | undefined
          if (target) opt.target = target
          if (Array.isArray(feats) && feats.length) opt.features = feats
          return opt
        })
      }
    }
    if (!abspaths && inputs.length === 1) {
      const entry = ds.selectedRel ? ds.entries.find((e) => e.relpath === ds.selectedRel) : undefined
      if (entry?.abspath) abspaths = [entry.abspath]
    }

    // Trained-weights checkpoint (local workspace only — the local sidecar
    // can't read a remote HPC path). Built from the project root + run id.
    let checkpoint: string | null = null
    const wid = get().weightsRunId
    if (wid) {
      const conn = getCurrentConnection()
      const ps = useProjectStore.getState()
      if (conn.kind === 'local' && ps.status.kind === 'loaded') {
        checkpoint = `${ps.status.root.replace(/\/$/, '')}/experiments/runs/${wid}/checkpoints/best.pt`
      }
    }

    if (inFlight) inFlight.abort()
    const ctrl = new AbortController()
    inFlight = ctrl
    set({ running: true, error: null })

    let result: ActivationsResult
    try {
      result = await runActivationsReq(code, inputShapes, inputDtypes, abspaths, checkpoint, graphAttrs, inputOptions, ctrl.signal)
      // Auto-fallback to synthetic if the real dataset sample couldn't be built.
      if (abspaths && !result.ok && 'stage' in result && result.stage === 'sample') {
        abspaths = null
        inputOptions = null
        result = await runActivationsReq(code, inputShapes, inputDtypes, null, checkpoint, graphAttrs, null, ctrl.signal)
      }
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') return
      set({ running: false, error: e instanceof Error ? e.message : String(e) })
      return
    }
    if (inFlight !== ctrl) return
    inFlight = null

    if (!result.ok) {
      set({ running: false, error: result.error, byNode: {}, weightsByNode: {} })
      return
    }

    // Map attr-keyed payload back onto graph node ids via attrMap.
    const byNode: Record<string, NodeActivation> = {}
    const weightsByNode: Record<string, Weights> = {}
    for (const [nodeId, attr] of Object.entries(attrMap)) {
      if (result.activations[attr]) byNode[nodeId] = result.activations[attr]
      if (result.weights[attr]) weightsByNode[nodeId] = result.weights[attr]
    }
    // Each model input → its Input node (multi-input: x, edge_index, batch, …).
    inputs.forEach((inp, i) => {
      const a = result.activations[`__input_${i}__`] ?? (i === 0 ? result.activations['__input__'] : undefined)
      if (a) byNode[inp.id] = a
    })
    const inputNodeId = inputs[0].id
    const outNode = nodes.find((n) => n.data.layerType === 'Output')
    if (outNode && result.activations['__output__']) byNode[outNode.id] = result.activations['__output__']

    set({
      running: false,
      error: null,
      byNode,
      weightsByNode,
      inputNodeId,
      source: abspaths ? 'dataset' : 'synthetic',
      sampleNote: result.sample_note,
      weightsSource: result.weights_source ?? 'random',
      weightsNote: result.weights_note ?? null,
    })
    get().playFlow()
  },
}))

// Discard activations only when the graph STRUCTURE changes — not on the
// debounced shape-inference writes (which replace the nodes array but keep ids
// + layer types). A cheap structural fingerprint avoids wiping on every keystroke.
function structuralKey(
  nodes: { id: string; data: { layerType: string } }[],
  edges: { source: string; target: string }[],
): string {
  const ns = nodes.map((n) => `${n.id}:${n.data.layerType}`).join(',')
  const es = edges.map((e) => `${e.source}>${e.target}`).join(',')
  return `${ns}|${es}`
}

let lastKey = structuralKey(useGraphStore.getState().nodes, useGraphStore.getState().edges)
useGraphStore.subscribe((state) => {
  const key = structuralKey(state.nodes, state.edges)
  if (key === lastKey) return
  lastKey = key
  const viz = useVizStore.getState()
  if (Object.keys(viz.byNode).length || viz.error) viz.clear()
})
