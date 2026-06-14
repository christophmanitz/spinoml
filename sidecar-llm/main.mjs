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
import { promises as fs } from 'node:fs'
import { spawn } from 'node:child_process'
import path from 'node:path'
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
// Workspace abstraction. The frontend tells us per turn whether file ops
// should hit the laptop's filesystem (local workspace) or an SSH target
// (remote-ssh workspace, root path is raw — may be tilde-prefixed). The
// helpers below dispatch on `ws.isRemote`. No new HTTP server: we just
// shell out to ssh from Node, which keeps auth + key handling identical
// to what Tauri does.

const SSH_OPTS = [
  '-o', 'BatchMode=yes',
  '-o', 'ConnectTimeout=10',
  '-o', 'ServerAliveInterval=20',
  '-o', 'ServerAliveCountMax=3',
]

function shellQuote(s) {
  return "'" + String(s).replace(/'/g, "'\\''") + "'"
}

/** Quote a path, expanding a leading `~/` to "$HOME" so the remote shell
 *  resolves it. Mirrors Rust's ssh.rs::shell_quote_path exactly. */
function shellQuotePath(s) {
  if (typeof s !== 'string' || s.length === 0) return shellQuote(s ?? '')
  if (s === '~') return '"$HOME"'
  if (s.startsWith('~/')) {
    return `"$HOME"${shellQuote('/' + s.slice(2))}`
  }
  return shellQuote(s)
}

function runSsh(target, remoteCmd, stdin) {
  return new Promise((resolve, reject) => {
    const child = spawn('ssh', [...SSH_OPTS, target, remoteCmd], {
      stdio: ['pipe', 'pipe', 'pipe'],
    })
    let stdout = ''
    let stderr = ''
    child.stdout.on('data', (d) => { stdout += d.toString() })
    child.stderr.on('data', (d) => { stderr += d.toString() })
    child.on('error', reject)
    child.on('close', (code) => {
      if (code === 0) {
        resolve(stdout)
      } else {
        const trimmed = stderr.trim()
        const hint = code === 255 ? ' (255 = connection/auth; check ~/.ssh/config + agent)' : ''
        reject(new Error(`ssh exit ${code}: ${trimmed}${hint}`))
      }
    })
    if (stdin !== undefined && stdin !== null) {
      child.stdin.write(stdin)
    }
    child.stdin.end()
  })
}

function makeWorkspace(project) {
  if (!project || !project.root) return null
  return {
    root: String(project.root).replace(/\/$/, ''),
    sshTarget: project.ssh_target || null,
    isRemote: !!project.ssh_target,
  }
}

// ────────────────────────────────────────────────────────────────────────────
// MCP tool surface.

// Safe notes helpers — scoped to a single project root passed per-turn.
function safeNoteFilename(name) {
  const base = path.basename(name || '')
  if (!base || base.startsWith('.')) throw new Error('invalid note name')
  if (!/\.(md|txt)$/i.test(base)) throw new Error('note name must end in .md or .txt')
  return base
}

function safeDatasetFilename(name) {
  const base = path.basename(name || '')
  if (!base || base.startsWith('.') || base.includes('/') || base.includes('\\')) {
    throw new Error('invalid filename')
  }
  if (!/\.(csv|tsv|parquet|pq|json|jsonl|npy|npz|pt|pth|zip|tar|gz|tgz|pdb|sdf|smi|smiles|txt)$/i.test(base)) {
    throw new Error('unsupported extension; allowed: csv, tsv, parquet, pq, json, jsonl, npy, npz, pt, pth, zip, tar(.gz), pdb, sdf, smi, txt')
  }
  return base
}

async function notesList(ws) {
  if (ws.isRemote) {
    const dir = `${ws.root}/notes`
    const dirQ = shellQuotePath(dir)
    const out = await runSsh(
      ws.sshTarget,
      `mkdir -p ${dirQ} && find ${dirQ} -mindepth 1 -maxdepth 1 -type f ` +
      `\\( -name '*.md' -o -name '*.txt' \\) -printf '%f\\t%s\\n' 2>/dev/null`,
    )
    return out
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean)
      .map((line) => {
        const [name, size] = line.split('\t')
        return { name, size: Number(size) || 0 }
      })
  }
  const dir = path.join(ws.root, 'notes')
  const items = await fs.readdir(dir).catch(() => [])
  const filtered = items.filter((f) => /\.(md|txt)$/i.test(f) && !f.startsWith('.'))
  const out = []
  for (const f of filtered.sort()) {
    try {
      const st = await fs.stat(path.join(dir, f))
      out.push({ name: f, size: st.size, mtime: new Date(st.mtimeMs).toISOString() })
    } catch { out.push({ name: f, size: 0 }) }
  }
  return out
}

async function notesRead(ws, name) {
  const safe = safeNoteFilename(name)
  if (ws.isRemote) {
    const p = `${ws.root}/notes/${safe}`
    return await runSsh(ws.sshTarget, `cat ${shellQuotePath(p)}`)
  }
  return await fs.readFile(path.join(ws.root, 'notes', safe), 'utf8')
}

async function notesAppend(ws, name, body) {
  const safe = safeNoteFilename(name)
  const text = body.endsWith('\n') ? body : body + '\n'
  if (ws.isRemote) {
    const dir = `${ws.root}/notes`
    const p = `${dir}/${safe}`
    await runSsh(
      ws.sshTarget,
      `mkdir -p ${shellQuotePath(dir)} && cat >> ${shellQuotePath(p)}`,
      text,
    )
  } else {
    const dir = path.join(ws.root, 'notes')
    await fs.mkdir(dir, { recursive: true })
    await fs.appendFile(path.join(dir, safe), text, 'utf8')
  }
  return text.length
}

async function downloadToDatasets(ws, url, filename) {
  if (!/^https?:\/\//i.test(url)) {
    throw new Error('url must start with http:// or https://')
  }
  const safe = safeDatasetFilename(filename)
  if (ws.isRemote) {
    const dir = `${ws.root}/datasets`
    const p = `${dir}/${safe}`
    // -fsSL → fail on HTTP errors, silent progress, follow redirects.
    // --max-time 300s keeps a hung download from hanging the chat turn.
    const out = await runSsh(
      ws.sshTarget,
      `mkdir -p ${shellQuotePath(dir)} && \
       curl -fsSL --max-time 300 ${shellQuote(url)} -o ${shellQuotePath(p)} && \
       wc -c < ${shellQuotePath(p)}`,
    )
    const bytes = parseInt(out.trim(), 10) || 0
    return { relpath: `datasets/${safe}`, bytes }
  }
  const dir = path.join(ws.root, 'datasets')
  await fs.mkdir(dir, { recursive: true })
  const dest = path.join(dir, safe)
  const res = await fetch(url, { redirect: 'follow' })
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  const buf = Buffer.from(await res.arrayBuffer())
  await fs.writeFile(dest, buf)
  return { relpath: `datasets/${safe}`, bytes: buf.length }
}

function buildMcpServer(ctx, actions, workspace) {
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

  if (workspace) {
    tools.push(
      tool(
        'list_notes',
        'List markdown notes in the project\'s notes/ folder. Use this to discover what context already exists before making suggestions.',
        {},
        async () => {
          try {
            const items = await notesList(workspace)
            if (!items.length) return content('(no notes yet)')
            return content(items.map((n) => `${n.name} — ${n.size} bytes`).join('\n'))
          } catch (e) {
            return content(`error: ${e.message}`, true)
          }
        },
      ),
      tool(
        'read_note',
        'Read the full contents of a project note (markdown or text). Use to load decision logs, prior session notes, or the project README before suggesting changes.',
        { name: z.string() },
        async ({ name }) => {
          try {
            return content(await notesRead(workspace, name))
          } catch (e) {
            return content(`error: ${e.message}`, true)
          }
        },
      ),
      tool(
        'append_note',
        'Append text to a project note (creates the file if missing). Use to record decisions, things you tried, or hand-offs for the next session. Always include a timestamp header. Prefer a small set of notes — decisions.md, session-YYYY-MM-DD.md — over one note per turn.',
        { name: z.string(), content: z.string() },
        async ({ name, content: body }) => {
          try {
            const written = await notesAppend(workspace, name, body)
            return content(`appended ${written} chars to notes/${safeNoteFilename(name)}`)
          } catch (e) {
            return content(`error: ${e.message}`, true)
          }
        },
      ),
      tool(
        'download_to_datasets',
        'Download a URL into the project\'s datasets/ folder. Use this when the user asks for a standard dataset by name (iris, MNIST, california housing, fashion-mnist, etc.). Pick a known stable mirror — UCI archive raw, scikit-learn raw, sklearn-datasets GitHub, HuggingFace datasets resolve URLs, common tutorial GitHub repos — and a filename ending in .csv/.parquet/.npy/.json/.zip/etc. The dataset shows up live in MLForge\'s Datasets tab the moment the download finishes. For binary archives (tar.gz, zip), let the user know they\'ll need to unpack — you can do this with a follow-up shell tool if one exists, otherwise tell them to extract via the Terminal tab.',
        {
          url: z.string().url(),
          filename: z.string().describe('Basename only (e.g. "iris.csv"), saved under datasets/'),
        },
        async ({ url, filename }) => {
          try {
            const r = await downloadToDatasets(workspace, url, filename)
            actions.push({ op: 'dataset-added', payload: { relpath: r.relpath, bytes: r.bytes } })
            const kb = (r.bytes / 1024).toFixed(1)
            return content(`saved ${kb} KB to ${r.relpath}`)
          } catch (e) {
            return content(`download failed: ${e.message}`, true)
          }
        },
      ),
    )
  }

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

function buildSystemPrompt(snapshot, error, project) {
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
  ]
  if (project) {
    lines.push(
      '',
      '═══ Project context ═══',
      `Name: ${project.name || '(unnamed)'}`,
    )
    if (project.description) lines.push(`Description: ${project.description}`)
    if (project.goal) lines.push(`Goal: ${project.goal}`)
    if (project.active_model) lines.push(`Active model: ${project.active_model}`)
    if (project.active_dataset) {
      lines.push(`Active dataset: ${project.active_dataset}`)
      if (project.active_dataset_inspect) {
        lines.push('Active dataset summary:', '```json', JSON.stringify(project.active_dataset_inspect, null, 2), '```')
      }
    }
    if (Array.isArray(project.recent_notes) && project.recent_notes.length) {
      lines.push('', 'Recent project notes (excerpts):')
      for (const n of project.recent_notes) {
        lines.push(`--- ${n.name} ---`, n.excerpt, '')
      }
    }
    lines.push(
      '',
      'You have notes tools: list_notes, read_note, append_note. Use append_note to record',
      'non-obvious decisions, dead ends, or hand-offs for the next session — e.g.',
      'append to "decisions.md" with a dated entry. Do this sparingly, only when the',
      'information is worth keeping across sessions.',
      '',
      'You also have download_to_datasets(url, filename). When the user asks for a',
      'standard dataset by name (iris, MNIST, california housing, boston, wine, etc.)',
      'pick a stable raw mirror and download it. Iris CSV with header is at',
      'https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv',
      'or https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
      '(headerless). After download, mention the dataset appears in MLForge\'s Datasets',
      'tab and suggest the next concrete step (e.g. "build an MLP with 4-input Input").',
    )
    if (project.ssh_target) {
      lines.push(
        '',
        `This workspace lives on REMOTE host \`${project.ssh_target}\` at \`${project.root}\`. All`,
        'note + dataset operations route via ssh. File-system reads/writes you do',
        'through other tools (Bash, Read, Edit, Write) hit your LAPTOP, NOT the HPC.',
        'For HPC-side operations beyond notes/datasets, tell the user to use the',
        'Terminal tab at the bottom — that\'s an ssh -tt session in the workspace.',
      )
    }
    lines.push('═══════════════════════')
  }
  lines.push(
    '',
    'Current architecture snapshot:',
    '```json',
    JSON.stringify(snapshot, null, 2),
    '```',
  )
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

  const { user, messages, graph, error, project } = payload
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

  const workspace = makeWorkspace(project)
  const mcp = buildMcpServer(ctx, actions, workspace)
  const systemPrompt = buildSystemPrompt(ctx.snapshot(), error, project)
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
          ...(workspace ? [
            'mcp__graph__list_notes',
            'mcp__graph__read_note',
            'mcp__graph__append_note',
            'mcp__graph__download_to_datasets',
          ] : []),
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
