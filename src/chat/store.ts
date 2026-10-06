import { create } from 'zustand'
import { llmHealthState, streamChat, respondToChat, type ChatEvent, type AskKind } from './client'
import { useGraphStore, autoPositionAfter } from '../canvas/GraphStore'
import { useTrainingGraphStore } from '../training/graph/store'
import { useDataGraphStore } from '../data/graph/store'
import { ensureTrainingBound } from '../training/graph/doc'
import { ensureDataBound } from '../data/graph/doc'
import { useViewModeStore } from '../training/graph/viewMode'
import { useInferenceStore } from '../inference/store'
import { useProjectStore } from '../project/store'
import { useDatasetsStore } from '../datasets/store'
import { useWorkspaceStore } from '../workspace/store'
import { isTauri } from '../workspace/tauri-fs'
import { notes as notesBackend } from '../connections/backend'
import { getCurrentConnection, sshTarget } from '../connections/store'
import { getCurrentLlmRequest } from './providerStore'
import { useChatUi } from './uiStore'

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
  | { id: string; role: 'assistant'; content: string; toolCalls: ToolCall[]; status: 'streaming' | 'done' | 'error'; error?: string; log?: string }

type Status = 'idle' | 'streaming' | 'offline'

export type PendingAsk = {
  id: string
  kind: AskKind
  prompt: string
  options?: string[] | null
  payload?: Record<string, unknown> | null
}

type ChatState = {
  messages: ChatMessage[]
  status: Status
  online: boolean | null
  /** Reachable but rejected (401/403) — a distinct state from "offline". The
   *  choice: `online` stays false for an auth failure (the model is NOT usable),
   *  and `authFailed` tells the badge to show the rose "auth failed" style. */
  authFailed: boolean
  authMessage: string | null
  /** Auth mode reported by the last successful probe (null = older sidecar). */
  llmAuth: 'token' | 'unauthenticated-dev' | null
  /** Set while the LLM/run_script is waiting for a GUI answer. */
  pendingAsk: PendingAsk | null
  send: (text: string) => Promise<void>
  reset: () => void
  /** Trim the chat to the last few messages so the context stops growing. Safe:
   *  the live graph/dataset/run state is re-sent in the system prompt each turn. */
  compact: () => void
  /** Abort the in-flight turn (and any running script) but KEEP the history. */
  stop: () => void
  refreshHealth: () => Promise<void>
  answerAsk: (answer: unknown) => void
}

let inflight: AbortController | null = null
let assistantSeq = 0
let userSeq = 0

function bumpSeqFrom(messages: ChatMessage[]): void {
  // Restored ids look like `u3` / `a5`; continue the counters past them so new
  // messages don't collide with persisted ones after a reload.
  for (const m of messages) {
    const n = parseInt(m.id.slice(1), 10)
    if (Number.isFinite(n)) {
      if (m.role === 'user') userSeq = Math.max(userSeq, n)
      else assistantSeq = Math.max(assistantSeq, n)
    }
  }
}

// Chat history is persisted to localStorage so it survives a webview reload /
// backgrounding (the store is otherwise in-memory and would re-init empty). On an
// EXPLICIT project close we clear it (clearPersistedChat) so a different project
// starts fresh — a reload of the SAME session restores it.
const CHAT_KEY = 'spinoml.chat.v1'

function loadPersistedMessages(): ChatMessage[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = window.localStorage.getItem(CHAT_KEY)
    if (!raw) return []
    const arr = JSON.parse(raw) as ChatMessage[]
    if (!Array.isArray(arr)) return []
    // Coerce any message that was mid-stream when we were interrupted to a final
    // state so it doesn't render as a stuck spinner.
    const msgs = arr.map((m) =>
      m.role === 'assistant' && m.status === 'streaming' ? { ...m, status: 'done' as const } : m,
    )
    bumpSeqFrom(msgs)
    return msgs
  } catch {
    // localStorage unavailable/corrupt: start with an empty chat. This is
    // browser-session history only; it is not a claim about the model or runs.
    return []
  }
}

function persistMessages(messages: ChatMessage[]): void {
  if (typeof window === 'undefined') return
  try { window.localStorage.setItem(CHAT_KEY, JSON.stringify(messages.slice(-100))) }
  catch {
    // Quota/private mode: only cross-reload persistence of the chat is skipped;
    // the live conversation for this session is unaffected.
  }
}

export function clearPersistedChat(): void {
  if (typeof window === 'undefined') return
  try { window.localStorage.removeItem(CHAT_KEY) }
  catch {
    // Best-effort cleanup on explicit project close; if it fails the worst case
    // is the previous project's chat is restored next launch, not a false state.
  }
  useChatStore.getState().reset()
}

export const useChatStore = create<ChatState>((set, get) => ({
  messages: loadPersistedMessages(),
  status: 'idle',
  online: null,
  authFailed: false,
  authMessage: null,
  llmAuth: null,
  pendingAsk: null,

  reset: () => {
    if (inflight) inflight.abort()
    inflight = null
    set({ messages: [], status: 'idle', pendingAsk: null })
  },

  compact: () => {
    if (get().status === 'streaming') return
    const msgs = get().messages
    const KEEP = 6
    if (msgs.length <= KEEP) return
    const recap: ChatMessage = {
      id: `a${++assistantSeq}`, role: 'assistant', status: 'done', toolCalls: [],
      content: `📝 *Verlauf gekürzt — ${msgs.length - KEEP} ältere Nachrichten ausgeblendet. Der aktuelle Modell-/Daten-/Run-Zustand wird ohnehin jede Runde frisch mitgeschickt.*`,
    }
    set({ messages: [recap, ...msgs.slice(-KEEP)] })
  },

  stop: () => {
    // Aborts the fetch → the sidecar sees the disconnect, kills any running
    // run_script child, and ends the turn. History is preserved; send()'s
    // finally marks the streaming message done.
    if (inflight) inflight.abort()
    inflight = null
    set({ pendingAsk: null })
  },

  refreshHealth: async () => {
    const h = await llmHealthState()
    set({
      online: h.state === 'online',
      authFailed: h.state === 'auth-failed',
      authMessage: h.state === 'auth-failed' ? (h.message ?? 'Sidecar-Authentifizierung fehlgeschlagen.') : null,
      llmAuth: h.auth ?? null,
    })
  },

  answerAsk: (answer) => {
    const ask = get().pendingAsk
    if (!ask) return
    set({ pendingAsk: null })
    // A rejected approval the user clicked must be visible, not swallowed by
    // the fire-and-forget POST: append an explicit error line to the chat.
    void respondToChat(ask.id, answer).catch((e: unknown) => {
      const msg = e instanceof Error ? e.message : String(e)
      set({
        messages: [...get().messages, {
          id: `a${++assistantSeq}`, role: 'assistant', content: '', toolCalls: [],
          status: 'error', error: msg,
        }],
      })
    })
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
    set({ messages: [...get().messages, userMsg, assistantMsg], status: 'streaming', pendingAsk: null })

    const history = get().messages
      .filter((m) => m.id !== assistantId)
      .map((m) => ({ role: m.role, content: m.content }))

    const graph = snapshotGraph()
    const training_graph = snapshotTrainingGraph()
    const data_graph = snapshotDataGraph()
    const error = snapshotError()
    const project = await snapshotProject()

    inflight = new AbortController()
    let mutatedGraph = false
    let mutatedTraining = false
    let mutatedData = false

    const llm = getCurrentLlmRequest()
    const { autoMode, docMode } = useChatUi.getState()

    try {
      await streamChat(
        { user: trimmed, messages: history, graph, training_graph, data_graph, error: error ?? undefined, project: project ?? undefined, llm, autoMode, docMode },
        (ev) => {
          if (ev.type === 'action') {
            if (ev.op.startsWith('training:')) mutatedTraining = true
            else if (ev.op.startsWith('data:')) mutatedData = true
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
      // The turn is over; any dangling question can no longer be answered.
      set({ status: 'idle', pendingAsk: null })
      if (mutatedGraph) useGraphStore.getState().autoLayout()
      if (mutatedTraining) {
        useTrainingGraphStore.getState().autoLayout()
        // Bind the chatbot's work to a file BEFORE switching, so the file-bound
        // canvas shows it (not the chooser) and autosaves it.
        await ensureTrainingBound()
        // Surface the chatbot's training-graph edits: switch to the training view
        // so the user actually SEES what changed (it lives on a separate canvas).
        useViewModeStore.getState().setMode('training')
      }
      if (mutatedData) {
        useDataGraphStore.getState().autoLayout()
        await ensureDataBound()
        useViewModeStore.getState().setMode('data')
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
    case 'log':
      // Append live script output, keeping only the tail so a chatty run can't
      // grow the message unboundedly.
      patchAssistant(assistantId, set, get, (a) => {
        const next = (a.log ?? '') + ev.value
        return { ...a, log: next.length > 20000 ? next.slice(next.length - 20000) : next }
      })
      break
    case 'ask':
      set({ pendingAsk: { id: ev.id, kind: ev.kind, prompt: ev.prompt, options: ev.options, payload: ev.payload } })
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

function snapshotDataGraph() {
  const { nodes, edges } = useDataGraphStore.getState()
  return {
    nodes: nodes.map((n) => ({ id: n.id, dataType: n.data.dataType, params: n.data.params })),
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
      } catch {
        // A single unreadable note is skipped; recent_notes is optional context
        // for the LLM, not a claim shown to the user.
      }
    }
  } catch {
    // The whole notes listing is optional LLM context; its absence cannot make
    // the UI or the model's answer claim anything untrue.
  }

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
    // ─── Data-processing-graph actions (the third canvas) ────────────────
    case 'data:add_node': {
      const d = useDataGraphStore.getState()
      const pos = { x: 80 + (d.nodes.length % 3) * 230, y: 60 + d.nodes.length * 70 }
      d.addNode(p.node_type as string, pos, { id: p.id as string, params: (p.params ?? {}) as Record<string, unknown> })
      break
    }
    case 'data:connect': {
      useDataGraphStore.getState().connectNodes(p.source as string, p.target as string)
      break
    }
    case 'data:update_params': {
      useDataGraphStore.getState().updateNodeParams(p.id as string, p.params as Record<string, unknown>)
      break
    }
    case 'data:delete_node': {
      useDataGraphStore.getState().deleteNode(p.id as string)
      break
    }
    case 'data:clear': {
      useDataGraphStore.getState().resetGraph()
      break
    }
    default:
      console.warn('unknown action op', op, p)
  }
}

useChatStore.getState().refreshHealth()
setInterval(() => useChatStore.getState().refreshHealth(), 5000)

// Persist chat history (debounced) so it survives a reload / backgrounding without
// thrashing localStorage on every streamed token.
let chatPersistTimer: ReturnType<typeof setTimeout> | null = null
useChatStore.subscribe((s, prev) => {
  if (s.messages === prev.messages) return
  if (chatPersistTimer) clearTimeout(chatPersistTimer)
  chatPersistTimer = setTimeout(() => persistMessages(useChatStore.getState().messages), 600)
})

// Clear the chat only on an EXPLICIT project close (status loaded→none) so a
// different project starts fresh. A reload/re-check goes none→loading→loaded (never
// loaded→none), so the history is kept across those — which is the whole point.
let lastProjectKind = useProjectStore.getState().status.kind
useProjectStore.subscribe((s) => {
  const k = s.status.kind
  if (lastProjectKind !== 'none' && k === 'none') clearPersistedChat()
  lastProjectKind = k
})
