import { create } from 'zustand'
import { inferShapes, type InferResult } from './client'
import { useGraphStore } from '../canvas/GraphStore'
import { generate } from '../codegen/generator'
import { LAYERS } from '../layers/registry'

type Status = 'idle' | 'inferring' | 'ok' | 'error' | 'offline'

type InferenceState = {
  status: Status
  error: string | null
  errorStage: string | null
  errorTrace: string | null
  failingNodeId: string | null
  failingNodeLayerType: string | null
  nParams: number | null
  attrShapes: Record<string, number[]>
  lastRunAt: number | null

  kick: () => void
}

let timer: ReturnType<typeof setTimeout> | null = null
let inFlight: AbortController | null = null
let runCounter = 0
let applyingShapes = false

export const useInferenceStore = create<InferenceState>((set) => ({
  status: 'idle',
  error: null,
  errorStage: null,
  errorTrace: null,
  failingNodeId: null,
  failingNodeLayerType: null,
  nParams: null,
  attrShapes: {},
  lastRunAt: null,

  kick: () => {
    if (timer) clearTimeout(timer)
    timer = setTimeout(async () => {
      timer = null
      if (inFlight) inFlight.abort()
      const ctrl = new AbortController()
      inFlight = ctrl
      const runId = ++runCounter
      const graphRev = useGraphStore.getState().revision

      const { nodes, edges } = useGraphStore.getState()
      const { code, issues, attrMap, inputs, order } = generate(nodes, edges)
      const inputShapes = inputs.map((i) => i.shape)
      const inputDtypes = inputs.map((i) => i.dtype)

      const hasInput = nodes.some((n) => LAYERS[n.data.layerType]?.kind === 'input')
      if (!hasInput || issues.some((i) => i.startsWith('Cycle'))) {
        set({
          status: 'idle',
          error: null,
          errorStage: null,
          errorTrace: null,
          failingNodeId: null,
          failingNodeLayerType: null,
          attrShapes: {},
        })
        clearShapesOnNodes()
        return
      }

      set({ status: 'inferring' })

      let result: InferResult | { ok: false; error: string; offline: true; shapes: Record<string, number[]> }
      try {
        result = await inferShapes(code, inputShapes, inputDtypes, ctrl.signal)
      } catch (e) {
        if (e instanceof DOMException && e.name === 'AbortError') {
          // Phase 39 — aborted by a newer kick() (stale) → let the newer run
          // decide the final status. If this was the latest run and was
          // cancelled without a successor, leave `inferring` would hang the
          // badge; return to the previous truthful state (idle if nothing was
          // valid before, otherwise keep prior ok/error/offline).
          if (runId !== runCounter) return
          // Latest was cancelled — keep prior state if it was already truthful
          // (ok/error/offline), otherwise drop inferring to idle.
          const cur = useInferenceStore.getState().status
          if (cur === 'inferring') {
            set({
              status: 'idle',
              error: null,
              errorStage: null,
              errorTrace: null,
              failingNodeId: null,
              failingNodeLayerType: null,
              attrShapes: {},
            })
            clearShapesOnNodes()
          }
          return
        }
        // Phase 38 — any other throw (network, sidecar crash, unhandled
        // fetch error) must leave `inferring`, not hang the badge forever.
        // Also guard graph revision: if the graph moved since we started,
        // dropping the response is the correct staleness behavior.
        if (graphRev !== useGraphStore.getState().revision) return
        const msg = e instanceof Error ? e.message : String(e)
        const offline = msg.toLowerCase().includes('fetch') || msg.toLowerCase().includes('network') || msg.toLowerCase().includes('offline')
        set({
          status: offline ? 'offline' : 'error',
          error: msg,
          errorStage: null,
          errorTrace: null,
          failingNodeId: null,
          failingNodeLayerType: null,
          attrShapes: {},
        })
        clearShapesOnNodes()
        return
      }
      // Phase 40 — stale-response guard: graph changed since we started
      // (runCounter covers ordering, revision covers structural move).
      if (runId !== runCounter || graphRev !== useGraphStore.getState().revision) return

      if ('offline' in result && result.offline) {
        set({
          status: 'offline',
          error: result.error,
          errorStage: null,
          errorTrace: null,
          failingNodeId: null,
          failingNodeLayerType: null,
          attrShapes: {},
        })
        clearShapesOnNodes()
        return
      }
      const r: InferResult = result

      if (r.ok) {
        set({
          status: 'ok',
          error: null,
          errorStage: null,
          errorTrace: null,
          failingNodeId: null,
          failingNodeLayerType: null,
          nParams: r.n_params,
          attrShapes: r.shapes,
          lastRunAt: Date.now(),
        })
        applyShapesToNodes(attrMap, r.shapes, edges, null)
      } else {
        const failingId = findFailingNode(order, attrMap, r.shapes)
        const failingNode = failingId ? nodes.find((n) => n.id === failingId) : undefined
        set({
          status: 'error',
          error: r.error,
          errorStage: r.stage ?? null,
          errorTrace: r.trace ?? null,
          failingNodeId: failingId,
          failingNodeLayerType: failingNode?.data.layerType ?? null,
          nParams: typeof r.n_params === 'number' ? r.n_params : null,
          attrShapes: r.shapes,
          lastRunAt: Date.now(),
        })
        applyShapesToNodes(attrMap, r.shapes, edges, failingId)
      }
    }, 200)
  },
}))

function findFailingNode(
  order: string[],
  attrMap: Record<string, string>,
  shapes: Record<string, number[]>,
): string | null {
  for (const id of order) {
    const attr = attrMap[id]
    if (attr && shapes[attr] === undefined) return id
  }
  return null
}

function shapesEqual(a: number[] | undefined, b: number[] | undefined): boolean {
  if (a === b) return true
  if (!a || !b) return false
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

function applyShapesToNodes(
  attrMap: Record<string, string>,
  shapes: Record<string, number[]>,
  edges: { source: string; target: string }[],
  failingId: string | null,
) {
  const graph = useGraphStore.getState()

  const predOf = new Map<string, string>()
  for (const e of edges) if (!predOf.has(e.target)) predOf.set(e.target, e.source)

  // Input-kind nodes (Input, Graph, Sequence) carry their shape directly. A Graph
  // node's shape is its node-feature matrix [N, F] — what the first GNN layer sees.
  // Kind-driven so any input-kind node (incl. Sequence) is covered.
  const isInputKind = (lt: string) => LAYERS[lt]?.kind === 'input'
  const inputShapeFor = (id: string): number[] | undefined => {
    const n = graph.nodes.find((m) => m.id === id)
    if (!n || !isInputKind(n.data.layerType)) return undefined
    return (n.data.params.shape as number[] | undefined) ?? undefined
  }

  function outputOf(nodeId: string): number[] | undefined {
    const inShape = inputShapeFor(nodeId)
    if (inShape) return inShape
    const attr = attrMap[nodeId]
    return attr ? shapes[attr] : undefined
  }

  let dirty = false
  const nextNodes = graph.nodes.map((n) => {
    const data = { ...n.data }
    let changed = false

    const inShape = isInputKind(n.data.layerType) ? inputShapeFor(n.id) : (() => {
      const pid = predOf.get(n.id)
      return pid ? outputOf(pid) : undefined
    })()
    const outShape = isInputKind(n.data.layerType) ? inputShapeFor(n.id) : outputOf(n.id)
    const isFailing = failingId === n.id

    if (!shapesEqual(data.inferredInputShape, inShape)) {
      if (inShape) data.inferredInputShape = inShape
      else delete data.inferredInputShape
      changed = true
    }
    if (!shapesEqual(data.inferredOutputShape, outShape)) {
      if (outShape) data.inferredOutputShape = outShape
      else delete data.inferredOutputShape
      changed = true
    }
    if (Boolean(data.hasError) !== isFailing) {
      if (isFailing) data.hasError = true
      else delete data.hasError
      changed = true
    }

    if (!changed) return n
    dirty = true
    return { ...n, data }
  })

  if (!dirty) return
  applyingShapes = true
  try {
    useGraphStore.setState({ nodes: nextNodes })
  } finally {
    applyingShapes = false
  }
}

function clearShapesOnNodes() {
  const graph = useGraphStore.getState()
  let dirty = false
  const nodes = graph.nodes.map((n) => {
    if (!n.data.inferredInputShape && !n.data.inferredOutputShape && !n.data.hasError) return n
    if (LAYERS[n.data.layerType]?.kind === 'input' && !n.data.hasError && !n.data.inferredInputShape) return n
    dirty = true
    const data = { ...n.data }
    delete data.inferredInputShape
    if (LAYERS[n.data.layerType]?.kind !== 'input') delete data.inferredOutputShape
    delete data.hasError
    return { ...n, data }
  })
  if (!dirty) return
  applyingShapes = true
  try {
    useGraphStore.setState({ nodes })
  } finally {
    applyingShapes = false
  }
}

useGraphStore.subscribe((state, prev) => {
  if (applyingShapes) return
  if (state.nodes !== prev.nodes || state.edges !== prev.edges) {
    useInferenceStore.getState().kick()
  }
})

useInferenceStore.getState().kick()
