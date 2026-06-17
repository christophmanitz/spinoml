import { create } from 'zustand'
import { llmHealth, streamChat, type ChatEvent } from './client'
import { useGraphStore, autoPositionAfter } from '../canvas/GraphStore'
import { useTrainingGraphStore } from '../training/graph/store'
import { useViewModeStore } from '../training/graph/viewMode'
import { useInferenceStore } from '../inference/store'
import { useProjectStore } from '../project/store'
import { useDatasetsStore } from '../datasets/store'
import { useWorkspaceStore } from '../workspace/store'
import { isTauri } from '../workspace/tauri-fs'
import { notes as notesBackend } from '../connections/backend'
import { getCurrentConnection, sshTarget } from '../connections/store'
import { getCurrentLlmRequest } from './providerStore'

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
    const training_graph = snapshotTrainingGraph()
    const error = snapshotError()
    const project = await snapshotProject()

    inflight = new AbortController()
    let mutatedGraph = false
    let mutatedTraining = false

    const llm = getCurrentLlmRequest()

    try {
      await streamChat(
        { user: trimmed, messages: history, graph, training_graph, error: error ?? undefined, project: project ?? undefined, llm },
        (ev) => {
          if (ev.type === 'action') {
            if (ev.op.startsWith('training:')) mutatedTraining = true
            else mutatedGraph = true
          }
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
      if (mutatedTraining) {
        useTrainingGraphStore.getState().autoLayout()
        // Surface the chatbot's training-graph edits: switch to the training view
        // so the user actually SEES what changed (it lives on a separate canvas).
        useViewModeStore.getState().setMode('training')
      }
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
  const inputNodes = nodes.filter((n) => n.data.layerType === 'Input')
  const firstShape = (inputNodes[0]?.data.params.shape as number[] | undefined) ?? [1, 3, 224, 224]
  const inputs = inputNodes.map((n) => ({
    id: n.id,
    name: String(n.data.params.name ?? 'x'),
    shape: (n.data.params.shape as number[] | undefined) ?? firstShape,
  }))
  return {
    input_shape: firstShape,
    inputs,
    nodes: nodes.map((n) => ({
      id: n.id,
      layerType: n.data.layerType,
      params: n.data.params,
      ...(n.data.inferredOutputShape ? { inferred_output_shape: n.data.inferredOutputShape } : {}),
    })),
    edges: edges.map((e) => ({ source: e.source, target: e.target })),
  }
}

function snapshotTrainingGraph() {
  const { nodes, edges } = useTrainingGraphStore.getState()
  return {
    nodes: nodes.map((n) => ({ id: n.id, trainingType: n.data.trainingType, params: n.data.params })),
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
    const list = await notesBackend.list()
    const top = list.slice(0, 3)
    for (const n of top) {
      try {
        const full = await notesBackend.read(n.name)
        recent_notes.push({ name: n.name, excerpt: full.slice(0, 1500) })
      } catch { /* skip individual failures */ }
    }
  } catch { /* notes optional */ }

  // Tell the sidecar whether to use local fs or shell out to ssh for file
  // operations + dataset downloads. ssh_target is the same string Tauri
  // uses (user@host or a plain alias from ~/.ssh/config).
  const conn = getCurrentConnection()
  const ssh_target = conn.kind === 'remote-ssh' ? sshTarget(conn) : null

  return {
    root,
    ssh_target,
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
    case 'dataset-added': {
      // sidecar downloaded a file into <root>/datasets/ — pull the new
      // dataset list and auto-select the freshly added entry so it shows
      // up in the right-side modal.
      void useDatasetsStore.getState().refresh().then(() => {
        const rel = (p.relpath as string) || ''
        if (rel) useDatasetsStore.getState().select(rel)
      })
      break
    }
    // ─── Training-graph actions (Phase 14) ───────────────────────────────
    case 'training:add_node': {
      const t = useTrainingGraphStore.getState()
      const pos = { x: 80 + (t.nodes.length % 3) * 220, y: 60 + t.nodes.length * 70 }
      t.addNode(p.node_type as string, pos, { id: p.id as string, params: (p.params ?? {}) as Record<string, unknown> })
      break
    }
    case 'training:connect': {
      useTrainingGraphStore.getState().connectNodes(p.source as string, p.target as string)
      break
    }
    case 'training:update_params': {
      useTrainingGraphStore.getState().updateNodeParams(p.id as string, p.params as Record<string, unknown>)
      break
    }
    case 'training:delete_node': {
      useTrainingGraphStore.getState().deleteNode(p.id as string)
      break
    }
    case 'training:clear': {
      useTrainingGraphStore.getState().resetGraph()
      break
    }
    default:
      console.warn('unknown action op', op, p)
  }
}

useChatStore.getState().refreshHealth()
setInterval(() => useChatStore.getState().refreshHealth(), 5000)
