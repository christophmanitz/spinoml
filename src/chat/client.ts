export type AskKind = 'confirm' | 'select' | 'text'

export type ChatEvent =
  | { type: 'status'; value: 'thinking' | 'done' | 'error'; message?: string }
  | { type: 'text'; value: string }
  | { type: 'tool_use'; id: string; name: string; args: Record<string, unknown> }
  | { type: 'tool_result'; id: string; ok: boolean; result?: string; error?: string }
  | { type: 'action'; op: string; payload: Record<string, unknown> }
  // Live stdout/stderr from a running script (run_script) — streamed so the
  // user can watch the run instead of staring at a frozen "running".
  | { type: 'log'; value: string }
  // The LLM (or a gated run_script) is asking the user something and is WAITING
  // for an answer posted back via respondToChat. Rendered as a QuestionCard.
  | { type: 'ask'; id: string; kind: AskKind; prompt: string; options?: string[] | null; payload?: Record<string, unknown> | null }
  | { type: 'done' }

export type ChatRequest = {
  user: string
  messages: { role: 'user' | 'assistant'; content: string }[]
  graph: {
    input_shape: number[]
    inputs?: { id: string; name: string; shape: number[] }[]
    nodes: { id: string; layerType: string; params: Record<string, unknown> }[]
    edges: { source: string; target: string }[]
  }
  training_graph?: {
    nodes: { id: string; trainingType: string; params: Record<string, unknown> }[]
    edges: { source: string; target: string }[]
  }
  data_graph?: {
    nodes: { id: string; dataType: string; params: Record<string, unknown> }[]
    edges: { source: string; target: string }[]
  }
  error?: { message: string; failingNodeId?: string | null; failingNodeLayerType?: string | null }
  project?: {
    root: string
    name: string
    description: string
    goal: string
    active_model: string | null
    active_dataset: string | null
    active_dataset_inspect?: unknown
    recent_notes?: { name: string; excerpt: string }[]
  }
  /** LLM source selection. Omitted → sidecar defaults to the subscription path. */
  llm?: {
    kind: 'opencode' | 'subscription' | 'anthropic' | 'openai-compat'
    model?: string
    apiKey?: string
    baseUrl?: string
  }
  /** FEAT-3 — auto-approve shell run_script (SLURM still confirms). */
  autoMode?: boolean
  /** FEAT-4 — documentation verbosity for the chatbot. */
  docMode?: 'off' | 'compact' | 'verbose'
}

const SIDECAR_URL = 'http://127.0.0.1:7422'

/** Answer a pending `ask` event mid-turn. The sidecar correlates by the
 *  globally-unique askId and resolves the awaiting tool handler, so the same
 *  /chat turn continues. Best-effort: a missing/expired ask just no-ops. */
export async function respondToChat(askId: string, answer: unknown): Promise<void> {
  try {
    await fetch(`${SIDECAR_URL}/respond`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ askId, answer }),
    })
  } catch {
    /* sidecar gone — the turn will time out on its side */
  }
}

export async function llmHealth(): Promise<boolean> {
  try {
    const r = await fetch(`${SIDECAR_URL}/health`)
    return r.ok
  } catch {
    return false
  }
}

/** Live provider/model list from the opencode CLI (`opencode models`, cached
 *  in the sidecar). Empty on any error — the UI falls back to suggestions. */
export async function fetchOpenCodeModels(): Promise<string[]> {
  try {
    const r = await fetch(`${SIDECAR_URL}/opencode/models`)
    if (!r.ok) return []
    const j = (await r.json()) as { ok?: boolean; models?: string[] }
    return Array.isArray(j.models) ? j.models : []
  } catch {
    return []
  }
}

export async function streamChat(
  req: ChatRequest,
  onEvent: (e: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${SIDECAR_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal,
  })
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => '')
    throw new Error(`sidecar HTTP ${res.status}: ${text || '(no body)'}`)
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let idx: number
    while ((idx = buf.indexOf('\n\n')) !== -1) {
      const raw = buf.slice(0, idx)
      buf = buf.slice(idx + 2)
      const line = raw.split('\n').find((l) => l.startsWith('data:'))
      if (!line) continue
      const json = line.slice(5).trim()
      if (!json) continue
      try {
        onEvent(JSON.parse(json) as ChatEvent)
      } catch (e) {
        console.warn('bad SSE chunk', json, e)
      }
    }
  }
}
