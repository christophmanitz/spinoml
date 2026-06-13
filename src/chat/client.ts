export type ChatEvent =
  | { type: 'status'; value: 'thinking' | 'done' | 'error'; message?: string }
  | { type: 'text'; value: string }
  | { type: 'tool_use'; id: string; name: string; args: Record<string, unknown> }
  | { type: 'tool_result'; id: string; ok: boolean; result?: string; error?: string }
  | { type: 'action'; op: string; payload: Record<string, unknown> }
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
}

const SIDECAR_URL = 'http://127.0.0.1:7422'

export async function llmHealth(): Promise<boolean> {
  try {
    const r = await fetch(`${SIDECAR_URL}/health`)
    return r.ok
  } catch {
    return false
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
