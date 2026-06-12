# MLForge

Drag-and-drop PyTorch architecture builder, steered by Claude.

Build `nn.Module` architectures by dragging layers onto a canvas. A connected LLM
(Claude, via the Agent SDK) observes the graph through MCP tools and can edit it
live or on demand. Live shape inference validates the graph against a sample input
shape and surfaces errors inline.

## Stack

- **Shell**: Tauri 2 (Rust + system webview)
- **Frontend**: React 19 + TypeScript + Vite, React Flow for the canvas,
  Zustand for graph state, Tailwind v4, Monaco for code preview
- **LLM bridge**: Node.js sidecar running `@anthropic-ai/claude-agent-sdk`
  (uses Claude Code's OAuth → consumes Max subscription, not pay-per-use)
- **Shape inference**: Python sidecar running PyTorch, HTTP on 127.0.0.1:7421

## Phase status

- [x] **Phase 0** — Scaffold: window, panels, dependencies wired
- [x] **Phase 1** — Layer palette, canvas with typed nodes, inspector forms
- [x] **Phase 2** — Graph → PyTorch code generator
- [x] **Phase 3** — Python sidecar with shape inference
- [x] **Phase 4** — Claude Agent SDK sidecar + chat UI with live graph mutation
- [x] **Phase 5** — Save/load `.mlforge` files, undo/redo, templates
- [ ] **Phase 6** — Tauri packaging + sidecar lifecycle management

## Repo layout

```
src/                    React frontend
  palette/              Draggable layer types (Conv, Linear, …)
  canvas/               React Flow canvas + Zustand GraphStore
  inspector/            Per-node parameter editor
  chat/                 Claude chat panel
  codegen/              Graph → PyTorch nn.Module source
src-tauri/              Rust shell, sidecar lifecycle
sidecar-llm/            Node.js, runs Claude Agent SDK (phase 4+)
sidecar-torch/          Python, shape inference (phase 3+)
```

## Develop

Everything runs inside the `mlforge-dev` conda env (node 20, rust 1.96, python 3.12).

```bash
conda activate mlforge-dev
npm install
npm run tauri dev          # opens the window
# or vite-only (no rust build needed):
npm run dev                # browser at http://localhost:5173

# in two extra terminals (both inside the conda env):
npm run sidecar:torch      # shape inference on 127.0.0.1:7421
npm run sidecar:llm        # Claude bridge      on 127.0.0.1:7422
```

The LLM sidecar spawns `claude` under the hood — make sure
`claude setup-token` was run once so it can hit your Max subscription.

Verification harnesses:

```bash
npm run verify:codegen     # generates 4 graphs → writes .py → runs python on each
npm run verify:sidecar     # autostarts sidecar, posts generated code, checks shapes & errors
```

### Tauri Linux system deps (one-time)

```bash
sudo apt install -y \
  libwebkit2gtk-4.1-dev \
  libjavascriptcoregtk-4.1-dev \
  libsoup-3.0-dev \
  libgtk-3-dev \
  librsvg2-dev \
  libayatana-appindicator3-dev \
  pkg-config
```

## Claude integration

Requires Claude Code installed locally and authenticated via
`claude setup-token` (Max subscription). The Node sidecar spawned by Tauri
inherits that auth, so no API key is needed.
