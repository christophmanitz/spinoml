# sidecar-llm

Node.js process that bridges the Tauri app and Claude.

**Status**: Phase 4 implemented (with Phase 5 graph mutation folded in).

Tiny HTTP server on `127.0.0.1:7422`. The frontend POSTs the conversation
history plus a snapshot of the current architecture; we run a Claude turn
via `@anthropic-ai/claude-agent-sdk` and stream tokens + tool calls back as
SSE. The SDK spawns `claude` under the hood and uses its OAuth, so this
consumes the user's Max subscription instead of API credit.

Graph mutation is wired as MCP tools (in-process, via
`createSdkMcpServer`). When Claude calls one, the handler:

1. mutates a transient per-turn graph held inside the sidecar (so subsequent
   tool calls in the same turn see the up-to-date state and can reference
   newly created node ids),
2. pushes an `action` event on the SSE stream so the React GraphStore can
   apply the same mutation to the canvas live.

Tool surface:

```
set_input_shape(shape)
add_layer(layer_type, after?, params?) → {id}
connect(source, target)
update_params(id, params)
delete_node(id)
```

SSE event shapes (one JSON object per `data:` line):

```
{type:"status", value:"thinking"|"done"|"error", message?}
{type:"text",   value:"<chunk>"}
{type:"tool_use",    id, name, args}
{type:"tool_result", id, ok, result?, error?}
{type:"action", op, payload}
{type:"done"}
```

Start:

```bash
conda activate mlforge-dev
node sidecar-llm/main.mjs       # or: npm run sidecar:llm
```
