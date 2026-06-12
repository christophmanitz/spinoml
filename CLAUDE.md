# CLAUDE.md — working notes for future Claude sessions

This file is the runbook for any future Claude (Code, Sonnet, Opus, …)
session that touches this repo. It assumes you've read README.md once for
context. Everything below is operational: how to make changes, where things
live, what to verify, what not to break.

## TL;DR project map

```
src/
  canvas/         React Flow canvas + Zustand GraphStore (the single source
                  of truth for the in-memory graph: nodes, edges, selection,
                  inferred shapes, autoLayout, undo target state)
  layers/         registry.ts — THE registry. Adding a layer type = editing
                  this one file. defaultParamsFor + coerceParams live here.
  codegen/        generator.ts — graph → PyTorch nn.Module source.
                  CodePreview.tsx — Monaco editor binding.
  inference/      client.ts (HTTP to torch sidecar) + store.ts (debounced,
                  re-entrancy-guarded subscription to GraphStore).
  chat/           client.ts (SSE parser) + store.ts (chat history, dispatches
                  LLM actions to GraphStore) + ChatPanel.tsx.
  inspector/      Inspector.tsx — per-node form. Controlled inputs with draft
                  buffers; FixHints suggests param patches by reading the
                  inferred input shape.
  workspace/      store.ts (two modes: 'browser' = localStorage virtual FS,
                  'tauri' = real disk via Rust commands).
                  FileExplorer.tsx (the VSCode-like tree).
                  tauri-fs.ts (typed invoke wrappers).
                  PyCodeModal.tsx (generated .py preview).
  templates/      Built-in architecture starters.
  history/        Undo/redo subscribing to GraphStore structural changes.
  persistence/    .mlforge file format, autosave to localStorage.
  sidecars/       managed.ts — query whether Rust spawned the sidecars.
  ErrorBoundary.tsx   Wraps <App/> — catches render-time crashes.
  Toolbar.tsx     Top-bar File / Edit / Templates menus.
  App.tsx         Resizable layout, header badges.
  index.css       Tailwind + a few CSS hacks (e.g. GPU layers for nodes).

src-tauri/
  src/lib.rs      Tauri Builder + filesystem commands + sidecar lifecycle.
                  ALL invocable Rust commands are registered in
                  invoke_handler! here.
  tauri.conf.json bundle config, window config, dev URL.
  capabilities/default.json   Plugin permissions.

sidecar-torch/main.py   tiny HTTP/JSON server on 127.0.0.1:7421, runs
                        PyTorch forward hooks for shape inference.
sidecar-llm/main.mjs    HTTP/SSE server on 127.0.0.1:7422, runs Claude
                        Agent SDK + in-process MCP server with the
                        graph-mutation tools.

scripts/verify-codegen.ts    runs generator over 4 graphs and execs the
                             generated Python to confirm shape/output.
scripts/verify-sidecar.ts    autostarts the torch sidecar and asserts
                             happy-path + intentional-error responses.
```

## Two execution modes — keep them straight

| | browser dev | Tauri dev | installed .deb |
|-|-|-|-|
| launch | `npm run dev` + http://localhost:5173 | `npm run tauri dev` | `mlforge` from launcher |
| filesystem | localStorage virtual FS only | localStorage *or* picked folder | localStorage *or* picked folder |
| sidecars | manual (`npm run sidecar:torch` + `…:llm`) | spawned by Rust | spawned by Rust |
| `isTauri()` | false | true | true |

`workspace/store.ts` branches on `mode === 'tauri'`. EVERY mutating action
has two implementations. When you add a workspace action, ALWAYS implement
both branches.

`isTauri()` reads `window.__TAURI_INTERNALS__`. Don't `import` Tauri APIs
at module top-level when they'd be a no-op in browser — guard the side
effect with `if (isTauri())`.

## Verification commands you should run

```bash
conda activate mlforge-dev          # always start here
npm run build                       # tsc + vite, must be green
npm run verify:codegen              # 4 codegen cases, runs python on each
npm run verify:sidecar              # autostarts torch sidecar + asserts
```

After Rust changes:
```bash
( cd src-tauri && cargo check )     # cheap, catches most type errors
npm run tauri dev                   # full path: link + window + sidecars
```

After a workspace/persistence change, manual smoke is mandatory: open dev,
File → New, build a CNN, Save, check the file appears in the explorer,
Cmd+Z, refresh — the autosave must restore.

## Common change recipes

### Add a new layer type

1. `src/layers/registry.ts`: extend `LAYERS` with a `LayerSpec` (type,
   category, `pytorchModule`, fields, `summary`). Stick to existing
   `FieldSpec` kinds; if you need a new field kind, see
   "Add a new FieldSpec" below.
2. If the layer can break sequential codegen (multi-input, branching),
   you'll need to teach `generator.ts` about it. Otherwise it Just Works
   because emit is uniform: `self.<attr> = nn.<PytorchModule>(<args>)`.
3. Add a FixHint in `inspector/Inspector.tsx`'s `FixHints()` if there's
   an obvious one-click correction tied to input shape (e.g.
   `in_channels` from channel dim).
4. Run `npm run verify:codegen` and `npm run verify:sidecar`. If your
   layer is sensitive to params, add a case to one of the harnesses.
5. Inform Claude (the LLM tool registry in `sidecar-llm/main.mjs`)? Only
   if the layer is rare enough that Claude won't guess the name — the
   `add_layer` tool accepts any string and the registry validates.

### Add a new FieldSpec kind

Adding e.g. a `list-of-int` field touches:
- `layers/registry.ts` — type definition, `defaultParamsFor`,
  `coerceParams` (the coercion case is mandatory: LLM-supplied values
  must be sanitised before they hit the GraphStore).
- `inspector/Inspector.tsx` — render case in `FieldInput` AND a draft
  buffer (look at `ShapeInput` for the pattern: controlled value, parse
  on every keystroke, commit only when valid).
- `codegen/generator.ts` `serializeParam` — emit Python syntax.

### Add a new LLM tool

1. `sidecar-llm/main.mjs` `buildMcpServer`: add another `tool(name, desc,
   zodSchema, handler)`. The handler mutates the in-memory `ctx`
   (so subsequent tool calls in the same turn see the new state) AND
   pushes an `action` event via `actions.push(...)`.
2. `invoke_handler` (Rust) — no change; LLM tools live in node.
3. `src/chat/store.ts` `dispatchAction`: add a case that maps the
   `action` op to a GraphStore mutation.
4. Update `allowedTools` in `query()` config in main.mjs.
5. Bump `maxTurns` if your tool takes many calls per request.
6. Update CLAUDE-the-model's awareness via `buildSystemPrompt`: mention
   the new tool, its idiomatic use, and dim-correctness rules.

### Add a Rust filesystem command

1. `src-tauri/src/lib.rs` — write `#[tauri::command] fn foo(...)`, scope
   to `WorkspaceState` so paths are validated under the workspace root.
   ALWAYS go through `resolve(&root, &relpath)` — that's the path-sanity
   gate (rejects `..` and absolute paths).
2. Add to `tauri::generate_handler![...]` at the bottom.
3. `src/workspace/tauri-fs.ts` — add a typed wrapper.
4. `src/workspace/store.ts` — branch the action on `mode === 'tauri'`.
5. `cargo check` then `npm run tauri dev` and exercise it.

### Bug: shape inference flickers or loops

This has bitten us twice. The story:
- `inference/store.ts` writes `inferredOutputShape` into GraphStore.
- GraphStore change triggers `inference/store.ts` subscriber.
- Subscriber re-runs inference → writes shapes → triggers itself.
- Two guards: an identity-check (skip setState if every shape is
  unchanged) and the `applyingShapes` boolean (skip subscription firing
  while WE are writing).

If you add any new derived-state writeback path, replicate both guards.

### Bug: Tauri webview shows blank canvas

`flex-1` doesn't work inside a `<Panel>` from react-resizable-panels —
they don't provide a flex parent. Use `h-full w-full`. We hit this on
the Canvas after Phase 4.5.

### Bug: page goes blank when LLM mutates the graph

99% chance it's a coercion miss. Claude returned `kernel_size: 3` instead
of `[3, 3]`, the field-renderer threw, and the ErrorBoundary may or may
not catch it cleanly. Fix path:
1. Check `coerceParams` in registry.ts has a branch for the involved
   FieldSpec kind.
2. Reproduce by hand-crafting a chat: `add_layer({layer_type: "...",
   params: {bad: 3}})`.
3. The ErrorBoundary in src/ErrorBoundary.tsx is the safety net — never
   remove it.

## Invariants — do not break these

1. **`generator.ts` is pure**: same nodes+edges → same code, no I/O, no
   randomness. The verify-codegen harness relies on this. Don't move
   randomness or `Date.now()` calls in.
2. **`coerceParams` runs on every write to a node's params**: GraphStore
   addLayer and updateNodeParams both pass through it. If you add a new
   way to set params, route through one of those.
3. **Sidecar URLs are localhost-only** (`127.0.0.1`). The Rust scope on
   the workspace root is the file-side counterpart. Don't expose either
   to the network.
4. **`activeFileId` must always reference an entry that exists** (or be
   null). When you delete the entry that's active, set activeFileId
   to null in the same setState. workspace/store handles this — match
   the pattern if you write new actions.
5. **Tauri invoke_handler! list is the gate**: any Rust command not in
   that macro is unreachable from JS. Easy to forget when adding one.
6. **Undo/redo only tracks structural changes** (`nodes.id +
   layerType + params`, `edges.source/target`). Position changes are
   intentionally not undoable. autoLayout writes new positions but its
   structural fingerprint is unchanged, so it doesn't pollute the stack.
7. **`pyTwinPath(relpath)` mapping**: `.mlforge` → `.py` with the same
   stem, sanitised. Save in Tauri mode writes both atomically (well,
   sequentially with no rollback — best-effort). Don't introduce a
   second naming scheme.

## Patterns that work

- **Async store actions returning Promise<string|void>**: callers use
  `.then((id) => setRename(id))` instead of awaiting. Keep this — it's
  how the explorer's create-then-rename interleaves with disk I/O
  without leaking awaits to render code.
- **`useGraphStore.getState()` for one-shot reads inside event handlers**:
  never subscribe via the hook for transient reads inside callbacks.
- **`captureSnapshot` vs `captureStructuralSnapshot`**: snapshot
  preserves positions (used for persistence + history rebound); structural
  excludes positions (used for dirty-detection + history-trigger).
- **SSE parsing**: `streamChat` (chat/client.ts) is the reference. The
  `\n\n`-delimited buffer pattern handles partial frames. Reuse it if
  you add another streaming endpoint.

## Things that look like bugs but aren't

- Header LLM badge says "LLM: offline" even though `npm run sidecar:llm`
  is running externally → the badge polls /health on 7422 every 5s; if
  *that* fails the badge is right. If the sidecar IS up, check its
  console (it may have crashed silently on a bad chat).
- "Shape (auto)" but inference shows offline → Rust spawned the
  sidecars, but one died on first request. `npm run tauri dev` console
  carries their stderr.
- Save in browser mode appears not to write to disk → correct, in
  browser mode "Save" goes to localStorage. Use Export to disk for a
  real file.

## When you don't know

- The README has the "happy path" for setup.
- `git log --oneline` over the phase-* commits is a fast tour of how
  features came in; each commit message lists the actual design moves,
  not just the change.
- The verify-codegen / verify-sidecar harnesses are the cheapest way to
  prove a change didn't regress generation or shape inference.

## Don't

- Don't add a third execution mode (Electron, web worker, …) without
  factoring the workspace + sidecar abstractions further. Two modes is
  already a tax; three doubles every new action.
- Don't reach for new state libraries. Zustand is the convention. If
  some state belongs everywhere (selection, hovered node) put it in
  GraphStore; if it's domain-specific, give it its own small store and
  cross-subscribe.
- Don't introduce a global mutable map keyed by node ID outside
  GraphStore — undo/redo, persistence and live inference all depend
  on `nodes` being THE list.
- Don't bypass `coerceParams` because "the user typed it correctly".
  The LLM is also a user.

## Commit hygiene

- One phase per commit. Subject is `phase N: short verb`. Body
  documents WHY and any non-obvious design moves (look at the recent
  commits — they're long for a reason).
- Never amend pushed commits.
- If a change spans a phase boundary, split it into two commits even
  if they pass together. Future-you will thank you.
