#!/usr/bin/env node
// SpinoML → OpenCode MCP tool bridge.
//
// OpenCode's CLI (`opencode run --format json`) loads tools from MCP servers
// declared in its config (`opencode.json` / `OPENCODE_CONFIG_CONTENT`). This
// process is that server: a stdio MCP server that proxies every tools/list /
// tools/call to the SpinoML LLM sidecar over HTTP on 127.0.0.1:7422.
//
// All per-turn state (graph context, tool handlers, the SSE action stream,
// ask/answer) lives in the sidecar; the bridge is deliberately stateless so it
// can never become a second source of truth. The model proposes operations;
// the sidecar's execTool + the GraphStore validation gate apply them.
//
// The sidecar spawns opencode with one MCP server named "graph", so the model
// sees tools as `<mcpname>_<tool>` (e.g. `graph_add_layer`). This bridge maps
// between that prefixed name and the sidecar's bare tool name.
//
// Usage: node mcp-bridge.mjs <requestId>

const SIDECAR = process.env.SPINOML_SIDECAR || 'http://127.0.0.1:7422'
const MCP_NAME = 'graph'
const requestId = process.argv[2]
if (!requestId) {
  console.error('[spinoml-mcp] missing requestId argument')
  process.exit(1)
}

let seq = 0

function send(obj) {
  process.stdout.write(JSON.stringify(obj) + '\n')
}

function sendResult(id, result) {
  send({ jsonrpc: '2.0', id, result })
}

async function sidecarJson(pathname, body, timeoutMs = 30000) {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), timeoutMs)
  try {
    const r = await fetch(`${SIDECAR}${pathname}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    })
    const json = await r.json().catch(() => null)
    if (!r.ok || !json) {
      return { error: `sidecar HTTP ${r.status}: ${JSON.stringify(json)}` }
    }
    return json
  } catch (e) {
    return { error: e.name === 'AbortError' ? 'sidecar call timed out' : String(e?.message ?? e) }
  } finally {
    clearTimeout(timer)
  }
}

async function handle(msg) {
  switch (msg.method) {
    case 'initialize':
      sendResult(msg.id, {
        protocolVersion: msg.params?.protocolVersion || '2025-03-26',
        capabilities: { tools: { listChanged: false } },
        serverInfo: { name: `spinoml-${MCP_NAME}-mcp`, version: '1.0.0' },
      })
      break

    case 'notifications/initialized':
    case 'notifications/cancelled':
      // No reply for notifications.
      break

    case 'tools/list': {
      const { tools, error } = await sidecarJson(`/internal/mcp/${requestId}/list`, {})
      if (error) {
        sendResult(msg.id, { tools: [], error: { code: 'MCP_PASSTHROUGH_ERROR', message: error } })
        break
      }
      const mapped = (tools ?? []).map((t) => ({
        name: `${MCP_NAME}_${t.name}`,
        description: t.description,
        inputSchema: t.inputSchema,
      }))
      sendResult(msg.id, { tools: mapped })
      break
    }

    case 'tools/call': {
      const { name, arguments: args } = msg.params || {}
      seq += 1
      const callId = `mcp_${requestId}_${seq}`
      if (!name || !name.startsWith(`${MCP_NAME}_`)) {
        sendResult(msg.id, {
          content: [{ type: 'text', text: `unknown tool: ${name}` }],
          isError: true,
        })
        break
      }
      const toolName = name.slice(MCP_NAME.length + 1)
      const { ok, result, error } = await sidecarJson(
        `/internal/mcp/${requestId}/call`,
        { id: callId, name: toolName, args: args ?? {} },
        // The sidecar may BLOCK on ask_user (GUI confirm) for up to the ask
        // timeout; opencode waits for the tool result regardless.
        11 * 60 * 1000,
      )
      if (error) {
        sendResult(msg.id, {
          content: [{ type: 'text', text: error }],
          isError: true,
        })
        break
      }
      sendResult(msg.id, {
        content: [{ type: 'text', text: result ?? '' }],
        ...(ok ? {} : { isError: true }),
      })
      break
    }

    default:
      sendResult(msg.id, { error: { code: 'METHOD_NOT_FOUND', message: `unknown method: ${msg.method}` } })
  }
}

let buf = ''
process.stdin.setEncoding('utf8')
process.stdin.on('data', (chunk) => {
  buf += chunk
  let idx
  while ((idx = buf.indexOf('\n')) !== -1) {
    const line = buf.slice(0, idx).trim()
    buf = buf.slice(idx + 1)
    if (!line) continue
    let msg
    try { msg = JSON.parse(line) } catch { continue }
    handle(msg).catch(() => { /* the async handlers resolve their own errors */ })
  }
})
process.stdin.on('end', () => process.exit(0))