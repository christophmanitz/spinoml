# sidecar-llm

Node.js process that bridges the Tauri app and Claude.

**Status**: stub, populated in Phase 4.

Runs `@anthropic-ai/claude-agent-sdk`, which spawns `claude` under the hood and
inherits the user's Max-subscription OAuth (no API key needed).

Communicates with the frontend via Tauri IPC. Exposes graph-mutation MCP tools
to the model in Phase 5 so it can `get_graph`, `add_node`, `connect`,
`update_node`, `delete_node`, `validate(input_shape)`.
