import { create } from 'zustand'
import { llmHealth, streamChat, type ChatEvent } from './client'
import { useGraphStore, autoPositionAfter } from '../canvas/GraphStore'
import { useInferenceStore } from '../inference/store'
import { useProjectStore } from '../project/store'
import { useDatasetsStore } from '../datasets/store'
import { useWorkspaceStore } from '../workspace/store'
import { isTauri, tauriFs } from '../workspace/tauri-fs'

export type ToolCall = {
  id: string
  name: string
  args: Record<string, unknown>
  status: 'running' | 'ok' | 'error'
  result?: string
  error?: string
}

export type ChatMessage =
  | { id: string; role: 'user'; content: string }
  | { id: string; role: 'assistant'; content: string; toolCalls: ToolCall[]; status: 'streaming' | 'done' | 'error'; error?: string }

type Status = 'idle' | 'streaming' | 'offline'

type ChatState = {
  messages: ChatMessage[]
  status: Status
  online: boolean | null
  send: (text: string) => Promise<void>
  reset: () => void
  refreshHealth: () => Promise<void>
}

let inflight: AbortController | null = null
let assistantSeq = 0
let userSeq = 0

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  status: 'idle',
  online: null,

  reset: () => {
    if (inflight) inflight.abort()
    inflight = null
    set({ messages: [], status: 'idle' })
  },

  refreshHealth: async () => {
    const ok = await llmHealth()
    set({ online: ok })
  },

  send: async (text) => {
    const trimmed = text.trim()
    if (!trimmed) return
    if (get().status === 'streaming') return

    const userMsg: ChatMessage = { id: `u${++userSeq}`, role: 'user', content: trimmed }
    const assistantId = `a${++assistantSeq}`
    const assistantMsg: ChatMessage = {
      id: assistantId, role: 'assistant', content: '', toolCalls: [], status: 'streaming',
    }
    set({ messages: [...get().messages, userMsg, assistantMsg], status: 'streaming' })

    const history = get().messages
      .filter((m) => m.id !== assistantId)
      .map((m) => ({ role: m.role, content: m.content }))

    const graph = snapshotGraph()
    const error = snapshotError()
    const project = await snapshotProject()

    inflight = new AbortController()
    let mutatedGraph = false

    try {
      await streamChat(
        { user: trimmed, messages: history, graph, error: error ?? undefined, project: project ?? undefined },
        (ev) => {
          if (ev.type === 'action') mutatedGraph = true
          applyEvent(assistantId, ev, set, get)
        },
        inflight.signal,
      )
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') return
      const msg = e instanceof Error ? e.message : String(e)
      patchAssistant(assistantId, set, get, (a) => ({ ...a, status: 'error', error: msg }))
    } finally {
      inflight = null
      patchAssistant(assistantId, set, get, (a) => a.status === 'streaming' ? { ...a, status: 'done' } : a)
      set({ status: 'idle' })
      if (mutatedGraph) useGraphStore.getState().autoLayout()
    }
  },
}))

function applyEvent(
  assistantId: string,
  ev: ChatEvent,
  set: (p: Partial<ChatState> | ((s: ChatState) => Partial<ChatState>)) => void,
  get: () => ChatState,
) {
  switch (ev.type) {
    case 'text':
      patchAssistant(assistantId, set, get, (a) => ({ ...a, content: a.content + ev.value }))
      break
    case 'tool_use':
      patchAssistant(assistantId, set, get, (a) => ({
        ...a,
        toolCalls: [...a.toolCalls, {
          id: ev.id, name: ev.name, args: ev.args, status: 'running',
        }],
      }))
      break
    case 'tool_result':
      patchAssistant(assistantId, set, get, (a) => ({
        ...a,
        toolCalls: a.toolCalls.map((t) => t.id === ev.id
          ? { ...t, status: ev.ok ? 'ok' : 'error', result: ev.result, error: ev.error }
          : t),
      }))
      break
    case 'action':
      dispatchAction(ev.op, ev.payload)
      break
    case 'status':
      if (ev.value === 'error') {
        patchAssistant(assistantId, set, get, (a) => ({ ...a, status: 'error', error: ev.message }))
      }
      break
    case 'done':
      patchAssistant(assistantId, set, get, (a) => a.status === 'streaming' ? { ...a, status: 'done' } : a)
      break
  }
}

function patchAssistant(
  id: string,
  set: (p: Partial<ChatState> | ((s: ChatState) => Partial<ChatState>)) => void,
  get: () => ChatState,
  patcher: (m: Extract<ChatMessage, { role: 'assistant' }>) => ChatMessage,
) {
  set({
    messages: get().messages.map((m) => (m.id === id && m.role === 'assistant') ? patcher(m) : m),
  })
}

function snapshotGraph() {
  const { nodes, edges } = useGraphStore.getState()
  const inputNode = nodes.find((n) => n.data.layerType === 'Input')
  return {
    input_shape: (inputNode?.data.params.shape as number[] | undefined) ?? [1, 3, 224, 224],
    nodes: nodes.map((n) => ({
      id: n.id,
      layerType: n.data.layerType,
      params: n.data.params,
      ...(n.data.inferredOutputShape ? { inferred_output_shape: n.data.inferredOutputShape } : {}),
    })),
    edges: edges.map((e) => ({ source: e.source, target: e.target })),
  }
}

function snapshotError() {
  const inf = useInferenceStore.getState()
  if (inf.status !== 'error' || !inf.error) return null
  return {
    message: inf.error,
    failingNodeId: inf.failingNodeId,
    failingNodeLayerType: inf.failingNodeLayerType,
  }
}

async function snapshotProject() {
  if (!isTauri()) return null
  const ps = useProjectStore.getState()
  if (ps.status.kind !== 'loaded') return null
  const { root, meta } = ps.status

  let active_model: string | null = meta.active_model ?? null
  const ws = useWorkspaceStore.getState()
  if (ws.activeFileId) {
    const entry = ws.entries[ws.activeFileId]
    if (entry && entry.kind === 'file') active_model = entry.name
  }

  let active_dataset_inspect: unknown
  const active_dataset = meta.active_dataset ?? useDatasetsStore.getState().selectedRel
  if (active_dataset) {
    const cached = useDatasetsStore.getState().inspects[active_dataset]
    active_dataset_inspect = cached?.data ?? undefined
  }

  let recent_notes: { name: string; excerpt: string }[] = []
  try {
    const notes = await tauriFs.listNotes()
    const top = notes.slice(0, 3)
    for (const n of top) {
      try {
        const full = await tauriFs.readNote(n.name)
        recent_notes.push({ name: n.name, excerpt: full.slice(0, 1500) })
      } catch { /* skip individual failures */ }
    }
  } catch { /* notes optional */ }

  return {
    root,
    name: meta.name,
    description: meta.description,
    goal: meta.goal,
    active_model,
    active_dataset,
    active_dataset_inspect,
    recent_notes,
  }
}

function dispatchAction(op: string, p: Record<string, unknown>) {
  const g = useGraphStore.getState()
  switch (op) {
    case 'set_input_shape': {
      const shape = p.shape as number[]
      g.updateNodeParams('input', { shape })
      break
    }
    case 'add_layer': {
      const id = p.id as string
      const layerType = p.layer_type as string
      const params = (p.params ?? {}) as Record<string, unknown>
      const pos = autoPositionAfter(g.nodes)
      g.addLayer(layerType, pos, { id, params })
      break
    }
    case 'connect': {
      g.connectNodes(p.source as string, p.target as string)
      break
    }
    case 'update_params': {
      g.updateNodeParams(p.id as string, p.params as Record<string, unknown>)
      break
    }
    case 'delete_node': {
      g.deleteNode(p.id as string)
      break
    }
    default:
      console.warn('unknown action op', op, p)
  }
}

useChatStore.getState().refreshHealth()
setInterval(() => useChatStore.getState().refreshHealth(), 5000)
