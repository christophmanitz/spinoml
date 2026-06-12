// MLForge LLM sidecar.
//
// Bridges the React frontend and Claude. Hosts an HTTP server on
// 127.0.0.1:7422; the only meaningful endpoint is POST /chat which streams
// SSE events.
//
// The SDK uses the user's local `claude` CLI which provides OAuth via
// `claude setup-token` — no API key, billed against the Max subscription.
//
// Each /chat request carries the full conversation history plus a snapshot
// of the user's current architecture. We rebuild a transient graph state
// here so the tool handlers can mutate and reason about it during the turn;
// each mutation is also pushed back to the frontend as an SSE action event
// so the GraphStore stays in lockstep.
//
// SSE event shapes:
//   {type: "status", value: "thinking"|"done"|"error", message?}
//   {type: "text",   value: "<assistant token chunk>"}
//   {type: "tool_use",    id, name, args}
//   {type: "tool_result", id, ok, result?, error?}
//   {type: "action", op, payload}     // mirror of the mutation, for the GraphStore
//   {type: "done"}

import { createServer } from 'node:http'
import { z } from 'zod'
import { query, tool, createSdkMcpServer } from '@anthropic-ai/claude-agent-sdk'

const PORT = 7422

// ────────────────────────────────────────────────────────────────────────────
// Graph state held only for the duration of one /chat turn.

function makeGraphContext(initial) {
  const nodes = new Map() // id → { id, layerType, params }
  const edges = new Map() // edge_key → { source, target }
  let counter = 0
  let inputShape = initial?.inputShape ?? [1, 3, 224, 224]

  for (const n of initial?.nodes ?? []) {
    nodes.set(n.id, { id: n.id, layerType: n.layerType, params: { ...(n.params ?? {}) } })
  }
  for (const e of initial?.edges ?? []) {
    edges.set(`${e.source}->${e.target}`, { source: e.source, target: e.target })
  }

  function nextId() {
    counter++
    while (nodes.has(`llm${counter}`)) counter++
    return `llm${counter}`
  }

  return {
    nodes,
    edges,
    nextId,
    setInputShape(s) { inputShape = s },
    snapshot() {
      return {
        input_shape: inputShape,
        nodes: [...nodes.values()],
        edges: [...edges.values()],
      }
    },
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Per-turn action queue. Tool handlers push actions; the SSE writer drains.

function makeActionStream() {
  const queue = []
  let resolveNext = null
  let closed = false

  return {
    push(action) {
      if (closed) return
      queue.push(action)
      if (resolveNext) { const r = resolveNext; resolveNext = null; r() }
    },
    close() { closed = true; if (resolveNext) { const r = resolveNext; resolveNext = null; r() } },
    async *drain() {
      while (true) {
        while (queue.length) yield queue.shift()
        if (closed) return
        await new Promise((r) => { resolveNext = r })
      }
    },
  }
}

// ────────────────────────────────────────────────────────────────────────────
// MCP tool surface.

function buildMcpServer(ctx, actions) {
  const tools = [
    tool(
      'set_input_shape',
      'Change the model input tensor shape. Example shapes: [1, 3, 224, 224] for ImageNet RGB, [1, 16, 512] for a sequence of 16 tokens with 512 features.',
      { shape: z.array(z.number().int().positive()).min(2).max(6) },
      async ({ shape }) => {
        ctx.setInputShape(shape)
        actions.push({ op: 'set_input_shape', payload: { shape } })
        return content(`input shape set to [${shape.join(', ')}]`)
      },
    ),
    tool(
      'add_layer',
      'Add a new layer to the architecture. Use "after" to wire it after an existing node id. Supported layer_type values are: Conv2d, Conv1d, ConvTranspose2d, Linear, Flatten, BatchNorm2d, LayerNorm, GroupNorm, ReLU, GELU, SiLU, Sigmoid, Tanh, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, Dropout, Dropout2d, MultiheadAttention, TransformerEncoderLayer, Output. Pass params as a JSON object of layer-specific fields (e.g. {in_channels: 3, out_channels: 64} for Conv2d).',
      {
        layer_type: z.string(),
        after: z.string().optional().describe('Optional source node id to connect from'),
        params: z.record(z.string(), z.unknown()).optional(),
      },
      async ({ layer_type, after, params }) => {
        const id = ctx.nextId()
        ctx.nodes.set(id, { id, layerType: layer_type, params: params ?? {} })
        actions.push({ op: 'add_layer', payload: { id, layer_type, params: params ?? {} } })
        if (after) {
          if (!ctx.nodes.has(after)) {
            return content(`error: source node "${after}" not found`, true)
          }
          const key = `${after}->${id}`
          ctx.edges.set(key, { source: after, target: id })
          actions.push({ op: 'connect', payload: { source: after, target: id } })
        }
        return content(`added ${layer_type} as ${id}${after ? ` after ${after}` : ''}`)
      },
    ),
    tool(
      'connect',
      'Wire one node\'s output into another node\'s input.',
      { source: z.string(), target: z.string() },
      async ({ source, target }) => {
        if (!ctx.nodes.has(source)) return content(`error: source "${source}" not found`, true)
        if (!ctx.nodes.has(target)) return content(`error: target "${target}" not found`, true)
        const key = `${source}->${target}`
        if (ctx.edges.has(key)) return content(`edge ${source}->${target} already exists`)
        ctx.edges.set(key, { source, target })
        actions.push({ op: 'connect', payload: { source, target } })
        return content(`connected ${source} → ${target}`)
      },
    ),
    tool(
      'update_params',
      'Patch parameters on an existing node. Only the supplied keys change. For shape-typed params like LayerNorm.normalized_shape, pass a list of ints.',
      { id: z.string(), params: z.record(z.string(), z.unknown()) },
      async ({ id, params }) => {
        const n = ctx.nodes.get(id)
        if (!n) return content(`error: node "${id}" not found`, true)
        n.params = { ...n.params, ...params }
        actions.push({ op: 'update_params', payload: { id, params } })
        return content(`patched ${id}: ${JSON.stringify(params)}`)
      },
    ),
    tool(
      'delete_node',
      'Remove a node and its incident edges. The Input node cannot be deleted.',
      { id: z.string() },
      async ({ id }) => {
        if (id === 'input') return content(`error: cannot delete the Input node`, true)
        if (!ctx.nodes.has(id)) return content(`error: node "${id}" not found`, true)
        ctx.nodes.delete(id)
        for (const [key, e] of ctx.edges) {
          if (e.source === id || e.target === id) ctx.edges.delete(key)
        }
        actions.push({ op: 'delete_node', payload: { id } })
        return content(`deleted ${id}`)
      },
    ),
  ]

  return createSdkMcpServer({ name: 'mlforge-graph', version: '0.1.0', tools })
}

function content(text, isError = false) {
  return {
    content: [{ type: 'text', text }],
    ...(isError ? { isError: true } : {}),
  }
}

// ────────────────────────────────────────────────────────────────────────────
// HTTP layer.

function cors(res) {
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type')
}

function sendJson(res, status, obj) {
  cors(res)
  res.writeHead(status, { 'Content-Type': 'application/json' })
  res.end(JSON.stringify(obj))
}

function buildSystemPrompt(snapshot, error) {
  const lines = [
    'You are an expert PyTorch architect embedded in MLForge, a drag-and-drop GUI for building nn.Module architectures.',
    '',
    'You can SEE the user\'s current graph (below) and can MUTATE it via tools. Prefer tools over describing changes in prose — the user wants you to actually build, not just suggest.',
    '',
    'When the user describes a goal, take the smallest sequence of tool calls that reaches it. Common patterns:',
    ' - "make my CNN deeper" → repeated add_layer(Conv2d, after=...), add_layer(BatchNorm2d, ...), add_layer(ReLU, ...)',
    ' - "fix the failing layer" → inspect the error + shapes shown below, then update_params (or delete_node + add_layer when the layer choice itself is wrong, e.g. LayerNorm on a CNN body → swap to GroupNorm or BatchNorm2d).',
    ' - "give me a classification head" → add_layer(AdaptiveAvgPool2d) + add_layer(Flatten) + add_layer(Linear, params={in_features: <channels>, out_features: <num_classes>}).',
    '',
    'Layer params must match the input shape: Conv2d.in_channels = channel dim of input, BatchNorm2d.num_features = channel dim, Linear.in_features = last dim, LayerNorm.normalized_shape = trailing dims. Inspect the shapes shown for each node before choosing parameters.',
    '',
    'After tool calls, briefly tell the user what you changed (one short sentence) — they can see the result on the canvas.',
    '',
    'Current architecture snapshot:',
    '```json',
    JSON.stringify(snapshot, null, 2),
    '```',
  ]
  if (error) {
    lines.push('', `Current forward-pass error: ${error.message}`)
    if (error.failingNodeId) lines.push(`Failing node: ${error.failingNodeId} (${error.failingNodeLayerType ?? 'unknown'})`)
  }
  return lines.join('\n')
}

function formatHistoryAsPrompt(messages, latestUser) {
  const turns = []
  for (const m of messages ?? []) {
    if (m.role === 'user') turns.push(`User: ${m.content}`)
    else if (m.role === 'assistant') turns.push(`Assistant: ${m.content}`)
  }
  turns.push(`User: ${latestUser}`)
  turns.push('Assistant:')
  return turns.join('\n\n')
}

async function handleChat(req, res) {
  let body = ''
  for await (const chunk of req) body += chunk
  let payload
  try { payload = JSON.parse(body || '{}') }
  catch (e) { return sendJson(res, 400, { error: `invalid json: ${e.message}` }) }

  const { user, messages, graph, error } = payload
  if (typeof user !== 'string' || !user.trim()) {
    return sendJson(res, 400, { error: 'missing "user" string' })
  }

  cors(res)
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  })

  function emit(ev) { res.write(`data: ${JSON.stringify(ev)}\n\n`) }

  const ctx = makeGraphContext({
    inputShape: graph?.input_shape ?? graph?.inputShape,
    nodes: graph?.nodes ?? [],
    edges: graph?.edges ?? [],
  })
  const actions = makeActionStream()

  // Drain actions to SSE as they're pushed, in parallel with the SDK.
  const actionPump = (async () => {
    for await (const a of actions.drain()) emit({ type: 'action', op: a.op, payload: a.payload })
  })()

  const mcp = buildMcpServer(ctx, actions)
  const systemPrompt = buildSystemPrompt(ctx.snapshot(), error)
  const prompt = formatHistoryAsPrompt(messages, user)

  emit({ type: 'status', value: 'thinking' })

  try {
    for await (const m of query({
      prompt,
      options: {
        systemPrompt: { type: 'preset', preset: 'claude_code', append: systemPrompt },
        mcpServers: { graph: mcp },
        allowedTools: [
          'mcp__graph__set_input_shape',
          'mcp__graph__add_layer',
          'mcp__graph__connect',
          'mcp__graph__update_params',
          'mcp__graph__delete_node',
        ],
        permissionMode: 'bypassPermissions',
        maxTurns: 60,
      },
    })) {
      handleSdkMessage(m, emit)
    }
    emit({ type: 'status', value: 'done' })
  } catch (e) {
    emit({ type: 'status', value: 'error', message: `${e.name}: ${e.message}` })
  } finally {
    actions.close()
    await actionPump
    emit({ type: 'done' })
    res.end()
  }
}

function handleSdkMessage(m, emit) {
  // The SDK emits typed messages — we forward what the frontend cares about.
  if (m.type === 'assistant' && m.message?.content) {
    for (const block of m.message.content) {
      if (block.type === 'text' && typeof block.text === 'string' && block.text.length) {
        emit({ type: 'text', value: block.text })
      } else if (block.type === 'tool_use') {
        emit({ type: 'tool_use', id: block.id, name: block.name, args: block.input ?? {} })
      }
    }
  } else if (m.type === 'user' && m.message?.content) {
    for (const block of m.message.content) {
      if (block.type === 'tool_result') {
        const text = Array.isArray(block.content)
          ? block.content.map((c) => c.text ?? '').join('').trim()
          : typeof block.content === 'string' ? block.content : ''
        emit({
          type: 'tool_result',
          id: block.tool_use_id,
          ok: !block.is_error,
          result: text,
          error: block.is_error ? text : undefined,
        })
      }
    }
  } else if (m.type === 'result' && m.subtype === 'error_max_turns') {
    emit({ type: 'status', value: 'error', message: 'hit max turns' })
  }
}

const server = createServer(async (req, res) => {
  if (req.method === 'OPTIONS') { cors(res); res.writeHead(204); res.end(); return }
  if (req.method === 'GET' && req.url === '/health') {
    return sendJson(res, 200, { ok: true })
  }
  if (req.method === 'POST' && req.url === '/chat') {
    try { await handleChat(req, res) }
    catch (e) {
      if (!res.headersSent) sendJson(res, 500, { error: `${e.name}: ${e.message}` })
      else { try { res.write(`data: ${JSON.stringify({ type: 'status', value: 'error', message: e.message })}\n\n`); res.end() } catch {} }
    }
    return
  }
  sendJson(res, 404, { error: 'not found' })
})

server.listen(PORT, '127.0.0.1', () => {
  console.log(`[mlforge-llm] listening on http://127.0.0.1:${PORT}`)
})
