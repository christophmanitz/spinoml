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
      const { code, issues, attrMap, inputShape } = generate(nodes, edges)

      const hasInput = nodes.some((n) => n.data.layerType === 'Input')
      if (!hasInput || issues.some((i) => i.startsWith('Cycle'))) {
        set({ status: 'idle', error: null, errorStage: null, errorTrace: null, attrShapes: {} })
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
          nParams: r.n_params,
          attrShapes: r.shapes,
          lastRunAt: Date.now(),
        })
      } else {
        set({
          status: 'error',
          error: r.error,
          errorStage: r.stage ?? null,
          errorTrace: r.trace ?? null,
          nParams: typeof r.n_params === 'number' ? r.n_params : null,
          attrShapes: r.shapes,
          lastRunAt: Date.now(),
        })
      }
      applyShapesToNodes(attrMap, r.shapes)
    }, 200)
  },
}))

function shapesEqual(a: number[] | undefined, b: number[] | undefined): boolean {
  if (a === b) return true
  if (!a || !b) return false
  if (a.length !== b.length) return false
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
  return true
}

function applyShapesToNodes(attrMap: Record<string, string>, shapes: Record<string, number[]>) {
  const graph = useGraphStore.getState()
  let dirty = false
  const nodes = graph.nodes.map((n) => {
    if (n.data.layerType === 'Input') {
      const wanted = (n.data.params.shape as number[] | undefined) ?? n.data.inferredOutputShape
      if (!wanted) return n
      if (shapesEqual(n.data.inferredOutputShape, wanted)) return n
      dirty = true
      return { ...n, data: { ...n.data, inferredOutputShape: wanted } }
    }
    const attr = attrMap[n.id]
    const shape = attr ? shapes[attr] : undefined
    if (shape) {
      if (shapesEqual(n.data.inferredOutputShape, shape)) return n
      dirty = true
      return { ...n, data: { ...n.data, inferredOutputShape: shape } }
    }
    if (n.data.inferredOutputShape) {
      dirty = true
      const next = { ...n, data: { ...n.data } }
      delete (next.data as { inferredOutputShape?: number[] }).inferredOutputShape
      return next
    }
    return n
  })
  if (!dirty) return
  applyingShapes = true
  try {
    useGraphStore.setState({ nodes })
  } finally {
    applyingShapes = false
  }
}

function clearShapesOnNodes() {
  const graph = useGraphStore.getState()
  let dirty = false
  const nodes = graph.nodes.map((n) => {
    if (!n.data.inferredOutputShape) return n
    if (n.data.layerType === 'Input') return n
    dirty = true
    const next = { ...n, data: { ...n.data } }
    delete (next.data as { inferredOutputShape?: number[] }).inferredOutputShape
    return next
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
