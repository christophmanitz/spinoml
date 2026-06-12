import { create } from 'zustand'
import { inferShapes, type InferResult } from './client'
import { useGraphStore } from '../canvas/GraphStore'
import { generate } from '../codegen/generator'

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

      const { nodes, edges } = useGraphStore.getState()
      const { code, issues, attrMap, inputShape, order } = generate(nodes, edges)

      const hasInput = nodes.some((n) => n.data.layerType === 'Input')
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
        result = await inferShapes(code, inputShape, ctrl.signal)
      } catch (e) {
        if (e instanceof DOMException && e.name === 'AbortError') return
        throw e
      }
      if (runId !== runCounter) return

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

  const inputNode = graph.nodes.find((n) => n.data.layerType === 'Input')
  const inputShape = (inputNode?.data.params.shape as number[] | undefined) ?? undefined

  function outputOf(nodeId: string): number[] | undefined {
    if (nodeId === inputNode?.id) return inputShape
    const attr = attrMap[nodeId]
    return attr ? shapes[attr] : undefined
  }

  let dirty = false
  const nextNodes = graph.nodes.map((n) => {
    const data = { ...n.data }
    let changed = false

    const inShape = n.data.layerType === 'Input' ? inputShape : (() => {
      const pid = predOf.get(n.id)
      return pid ? outputOf(pid) : undefined
    })()
    const outShape = n.data.layerType === 'Input' ? inputShape : outputOf(n.id)
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
    if (n.data.layerType === 'Input' && !n.data.hasError && !n.data.inferredInputShape) return n
    dirty = true
    const data = { ...n.data }
    delete data.inferredInputShape
    if (n.data.layerType !== 'Input') delete data.inferredOutputShape
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
