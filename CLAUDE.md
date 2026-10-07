# CLAUDE.md — working notes for future Claude sessions

This file is the runbook for any future Claude (Code, Sonnet, Opus, …)
session that touches this repo. It assumes you've read README.md once for
context. Everything below is operational: how to make changes, where things
live, what to verify, what not to break.

> **READ `docs/FEATURES.md` FIRST when implementing or operating SpinoML.**
> It is the capability catalog — what features exist, how a user/the chatbot
> drives them, and which source file is the truth for each — so you can act
> without re-deriving the app. CLAUDE.md (this file) is the *code-change*
> runbook (how/where to edit + invariants); FEATURES.md is *what it does*.
> **MANDATE:** whenever you add or change a user-facing feature, update the
> relevant section of `docs/FEATURES.md` AND add a dated line to its Changelog,
> in the same change. Keep it concepts+pointers — never duplicate code there.

## TL;DR project map

```
src/
  canvas/         React Flow canvas + Zustand GraphStore (the single source
                  of truth for the in-memory graph: nodes, edges, selection,
                  inferred shapes, autoLayout, undo target state).
                  subgraphPorts.ts — auto-port sync: an edge into a Subgraph
                  node creates/claims a read-only PROXY input inside (marked
                  params._proxyOf), inheriting the outer source's config;
                  removing the edge releases (hand) or deletes (auto) it.
                  Runs in GraphStore onConnect/onEdgesChange/updateNodeParams.
  layers/         registry.ts — THE registry. Adding a layer type = editing
                  this one file. defaultParamsFor + coerceParams live here.
                  The `Graph` node (kind input, graphInput:true) is ONE whole
                  PyG Data (x+edge_index+batch+edge_attr) — binds 1:1 to a graph
                  dataset / manifest branch; one intuitive node per GNN input.
                  The `Sequence` node (kind input, int64) is a token-id input
                  (char/byte vocab) bound to a manifest branch `kind:"sequence"`.
                  The `ESPF` node (kind input, int64) tokenizes a SMILES branch
                  (`kind:"espf"`) into INTERPRETABLE substructure subword tokens
                  via the vendored MolTrans ESPF BPE codebook (sidecar-torch/espf/,
                  BSD-3) — feeds an Embedding like Sequence but chemically
                  meaningful (token id ↔ named substructure; surfaced as labels in
                  the Explain view). Sidecar tokenizes inline (dataset_handlers.
                  tokenize_espf); training reads a compact codebook the sidecar
                  caches next to the manifest in <manifest_dir>/.espf/ (primed on
                  inspect — the .graphcache analog). No `subword-nmt` dep: the BPE
                  apply is pure-Python (parity-tested vs the reference).
                  The `DataOp` node (kind 'dataop', category 'Data') is a
                  DATA-stage node, NOT a model layer: it carries a Python
                  `script` run OFFLINE (via the chatbot's write_file+run_script)
                  to download/tokenize/cache a dataset; in generator.ts it is a
                  pure passthrough (aliases its predecessor's var, emits nothing
                  into forward) so it never perturbs model codegen. Inspector's
                  DataOpPanel "Vorverarbeitung ausführen" kicks off that run.
  codegen/        pyLiteral.ts — the ONLY way to put a string/number/identifier/comment
                  text into generated Python (pyStr/pyComment/pyIdent/pyFloat/pyInt).
                  verify:codegen-security fuzzes every sink with hostile payloads.
  codegen/        generator.ts — graph → PyTorch nn.Module source. A Graph
                  input becomes a Data forward-arg; where a built-in GNN/pool
                  consumes it the generator emits `x, edge_index, batch =
                  g.x, …` (explicit), but a Custom/Subgraph consumer gets the
                  WHOLE Data (full access — like hand-written code).
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
                  LeftSidebar.tsx (Files/Datasets tab switcher).
                  tauri-fs.ts (typed invoke wrappers — LOCAL FS only).
                  PyCodeModal.tsx (generated .py preview).
  connections/    store.ts (saved SSH connections + current backend selection,
                  persisted to localStorage; secrets stay in ~/.ssh/config).
                  tauri-ssh.ts (typed invoke wrappers — remote FS over ssh).
                  backend.ts (DISPATCH LAYER — fs/project/notes/experiments/
                  datasets route to local or remote based on getCurrentConnection().
                  Other code MUST go through these, NOT tauri-fs/tauri-ssh).
  terminal/       Terminal.tsx — xterm.js bound to a Rust-side PTY session
                  (local bash or `ssh -tt <alias>`).
  datasets/       client.ts (HTTP to torch sidecar /dataset/*)
                  store.ts (per-relpath inspect/stats/smoke cache)
                  DatasetExplorer.tsx (list under workspace/datasets/)
                  DatasetDetail.tsx (Overview/Stats/SmokeTest tabs)
                  types.ts (shared kinds for all dataset formats, incl. the
                  `manifest` kind — a JSON descriptor pairing a table to per-
                  branch graph sources row-by-row, e.g. ligand SMILES + protein
                  .pt + target column, for dual-encoders. Each Input binds a
                  '<branch>.<field>' slot via `bind_field` in the Inspector).
  templates/      Built-in architecture starters.
  visualization/  "Explain"-Modus (3Blue1Brown-Stil). store.ts = ephemerer
                  Viz-Store (per-Node Aktivierungen + Gewichte, gekeyed by
                  nodeId; NIE in GraphStore — wird bei Struktur-Änderung
                  verworfen). client.ts → POST /activations am torch-Sidecar.
                  LayerExplain.tsx = kategorie-getriebenes Detail-Panel (ersetzt
                  den Inspector wenn explainMode an); MiniViz.tsx = Node-Vorschau;
                  primitives.tsx = SVG-Heatmap/Bars. Run ist on-demand (▶ im
                  Header), NICHT bei jedem Tastendruck.
  trust/          Code-trust gate (Phase 43). trustStore.ts = content-addressed set of
                  sha256(kind\0source) that the HUMAN approved (localStorage, OUTSIDE
                  params/.spinoml: coerceParams passes unknown keys, so a flag in params
                  would be forgeable). codeBlobs.ts collects every code-bearing node
                  (Custom.source + Custom.init_args, DataOp.script, data CustomScript.code,
                  recursing into Subgraphs). guard.ts assertTrusted/listUntrusted is called
                  by EVERY executor (inference, activations, dataset smoke, training
                  verifier, startRun/startEvalRun) — a new executor MUST call it too.
                  ApproveCodeDialog.tsx = the explicit approval UI. scripts/verify-code-trust.ts
                  enforces ALLOWED_APPROVERS (who may call trust.approve).
  history/        Undo/redo subscribing to GraphStore structural changes.
  persistence/    .spinoml file format, autosave to localStorage.
  sidecars/       managed.ts — query whether Rust spawned the sidecars.
  ErrorBoundary.tsx   Wraps <App/> — catches render-time crashes.
  Toolbar.tsx     Top-bar File / Edit / Templates menus.
  App.tsx         Resizable layout, header badges.
  index.css       Tailwind + a few CSS hacks (e.g. GPU layers for nodes).

src-tauri/
  src/lib.rs      Tauri Builder + LOCAL filesystem commands + sidecar lifecycle.
                  ALL invocable Rust commands are registered in
                  invoke_handler! here (local + ssh:: + pty::).
  src/ssh.rs      Mirror of every local FS command as an `ssh_*` variant.
                  Transport: system `ssh` subprocess (uses ~/.ssh/config,
                  agent, ProxyJump, GSSAPI — no secrets stored). Path
                  quoting in shell_quote_path() handles `~/` → `"$HOME"`.
  src/pty.rs      PTY sessions for the terminal tab. portable-pty crate.
                  Local: spawns $SHELL -l with cwd=workspaceRoot.
                  Remote: spawns `ssh -tt <alias>` then `cd <root>; exec $SHELL`.
                  Emits Tauri events `pty:<id>:data` / `pty:<id>:exit`.
  tauri.conf.json bundle config, window config, dev URL.
  capabilities/default.json   Plugin permissions.

sidecar-torch/main.py             HTTP/JSON server on 127.0.0.1:7421.
                                  Endpoints: /infer, /dataset/inspect,
                                  /dataset/stats, /dataset/smoke, /activations
                                  (real forward pass → per-module activations +
                                  weights, downsampled; feeds the Explain viz.
                                  Optional `checkpoint` abspath → loads trained
                                  weights from a run's checkpoints/best.pt
                                  (strict=False, falls back to random + note on
                                  mismatch); returns weights_source). GNN-aware:
                                  `graph_attrs` keeps node×feature tensors un-
                                  stripped, an int [2,E] input is previewed as a
                                  graph (edges), and a synthetic edge_index is a
                                  random graph so the GNN viz is meaningful.
sidecar-torch/dataset_handlers.py per-kind inspect/stats/sample_tensor for
                                  tabular, image_folder, tensor, protein,
                                  molecule, huggingface, pyg. Lazy-imports heavy
                                  deps so missing pandas/PIL/rdkit/biopython/
                                  torch_geometric gracefully degrades to a
                                  missing_dep error. sample_tensor takes an
                                  options.field (x/edge_index/edge_attr/pos/batch/y)
                                  so ONE graph dataset feeds all GNN inputs:
                                  molecule builds an atom/bond graph (RDKit, also
                                  from a SMILES COLUMN of a tabular file via
                                  options.target), pyg loads a PyG dataset (`.pyg`
                                  = `pyg:Planetoid/Cora`), and the tensor kind
                                  loads a saved PyG Data `.pt` (their graphein
                                  pipeline output) exposing all fields.
sidecar-llm/main.mjs    HTTP/SSE server on 127.0.0.1:7422. LLM source paths chosen
                        per /chat request via `payload.llm.kind`: 'opencode' (the
                        DEFAULT — spawns `opencode run --format json -m <model>`
                        per turn; the model is exposed ONLY the `graph` MCP tool
                        server `sidecar-llm/mcp-bridge.mjs` (stdio → `POST
                        /internal/mcp/<requestId>/list|call`) so all its operations
                        flow through the same validation gate as Claude. Built-in
                        opencode tools (read/bash/write/…) disabled via inline
                        config. Frontend model list: `GET /opencode/models`.),
                        'subscription' (claude-agent-sdk + OAuth, in-process MCP
                        server), 'anthropic' (@anthropic-ai/sdk Messages API),
                        'openai-compat' (openai SDK — OpenAI, Gemini, Ollama).
                        buildToolSpecs() is the single
                        provider-neutral tool registry; all three paths share
                        the same handlers + `actions` SSE mirror. Frontend
                        provider/key selection lives in src/chat/providerStore.ts
                        (localStorage) + ProviderSettings.tsx.
                        Agent tools beyond graph mutation: read_file / write_file
                        (incl. .py/.sh under agent/, datasets/, notes/) / list_dir
                        / run_script (mode 'shell' = run now where the workspace
                        lives, HARD-CAPPED at 2 min on a remote login node then
                        killed + told to use SLURM; 'slurm' = sbatch for heavy
                        compute; shell stdout/stderr STREAM live to the chat via
                        the `log` SSE event) / slurm_status,
                        all routed local-or-ssh like writeDatasetFile. `agent/`
                        is the scratch dir. run_script is GUI-GATED: it calls
                        askUser({kind:'confirm'}) and only runs on approval.
                        ASK/ANSWER CHANNEL: askUser() emits an SSE `ask` event and
                        AWAITS the answer POSTed to the new `POST /respond`
                        endpoint (pendingAsks map, per-turn registry rejected on
                        turn-end/disconnect). Backs both the ask_user tool and the
                        run confirm. Frontend: client.ts `ask`+respondToChat,
                        store.ts pendingAsk/answerAsk, ChatPanel QuestionCard.
                        Chat font scale = src/chat/uiStore.ts (A−/A+ in header).

scripts/verify-codegen.ts    runs generator over ~13 graphs and execs the
                             generated Python to confirm shape/output (incl. a
                             DataOp-passthrough case asserting it emits nothing).
scripts/verify-sidecar.ts    autostarts the torch sidecar and asserts
                             happy-path + intentional-error responses.
sidecar-llm/shell-safety.mjs   pure helpers: splitArgs/quoteArgv (run_script args), checkDownloadUrl +
                               isBlockedAddress + safeFetch (SSRF), checkSshTarget. Tested against a
                               real shell by scripts/verify-command-injection.ts.
sidecar-torch/deps_policy.py   validate_specs: /deps/* accept only plain PyPI requirements.
sidecar-torch/scope.py         check_path: every path a request (or a manifest/table cell) makes the sidecar
                               open must resolve inside an allowed root / symlink target. Config via env
                               SPINOML_ALLOWED_ROOTS / SPINOML_SYMLINK_TARGETS / ~/.cache/spinoml/scope.json;
                               mode shown in /health.scope. Use the RETURNED resolved path for every open.
sidecar-torch/safe_load.py     safe_torch_load — the ONLY torch.load. The block between the `# >>> safe_load`
                               markers is duplicated byte-for-byte in training_template.py (standalone
                               train.py); scripts/test-safe-load.py enforces equality.
sidecar-llm/path-scope.mjs     resolveInWorkspace: symlink-aware containment for the LLM tools' local paths.
sidecar-torch/auth.py          Token + Host + Origin gate (`decide`, called BEFORE the body is read); sidecar-llm/auth.mjs is
sidecar-llm/auth.mjs           its twin (same order/codes). Spec: docs/engineering/SIDECAR_AUTH.md. The token is read from
                               SPINOML_SIDECAR_TOKEN then deleted from the process env.
src-tauri/src/sidecar_auth.rs  Per-launch token (getrandom), `sidecar_token` command; remote token per ssh session.
src/sidecars/auth.ts           `sidecarFetch`/`torchFetch` — the ONLY way to call a sidecar from src/ (verify:sidecar-fetch).
```

## Two execution modes + two FS backends — keep them straight

**Execution mode** (`workspace/store.ts.mode`, decided at runtime):

| | browser dev | Tauri dev | installed .deb |
|-|-|-|-|
| launch | `npm run dev` + http://localhost:5173 | `npm run tauri dev` | `spinoml` from launcher |
| filesystem | localStorage virtual FS only | localStorage *or* a workspace | localStorage *or* a workspace |
| sidecars | manual (`npm run sidecar:torch` + `…:llm`) | spawned by Rust | spawned by Rust |
| `isTauri()` | false | true | true |

`workspace/store.ts` branches on `mode === 'tauri'`. EVERY mutating action
has two implementations (browser virtual-FS vs Tauri-backed). When you add
a workspace action, ALWAYS implement both branches.

`isTauri()` reads `window.__TAURI_INTERNALS__`. Don't `import` Tauri APIs
at module top-level when they'd be a no-op in browser — guard the side
effect with `if (isTauri())`.

**FS backend** (within Tauri mode only, decided by `connections/store.ts`):

| | local | remote-ssh |
|-|-|-|
| workspace root | a folder picked via dialog | an alias + path on a remote host |
| transport | `fs::*` in Rust | system `ssh` / `sftp`-style writes |
| sidecars | localhost (7421 torch, 7422 llm) | local LLM sidecar + a remote torch sidecar bootstrapped over ssh and reached through the tunnel on 7424 (`remote_sidecar.rs`) |
| dataset smoke test | works | works through the remote sidecar once `ensure_remote_sidecar` ran (NOT verified against a real cluster — docs/engineering/REMOTE_TRAINING.md §9) |

The dispatch happens in `src/connections/backend.ts` — every higher-level
store (workspace, project, datasets, chat-snapshot) imports `fs` / `project`
/ `notes` / `experiments` / `datasets` from there. **Do NOT import
`tauri-fs` or `tauri-ssh` directly** outside that file; if you do, you've
created a backend leak and remote workspaces will silently call the local
FS (or vice versa).

A remote connection is a `{ alias, root }` pair. `alias` must exist in
`~/.ssh/config` and be reachable WITHOUT a password prompt
(`BatchMode=yes` is set for all ssh_* commands except the terminal).
`root` is either absolute (`/scratch/.../spinoml`) or tilde-prefixed
(`~/spinoml`); `shell_quote_path()` in `ssh.rs` handles the tilde
substitution to `"$HOME"`.

## Verification commands you should run

```bash
conda activate spinoml-dev          # always start here
npm run build                       # tsc + vite, must be green
npm run verify:codegen              # 4 codegen cases, runs python on each
npm run verify:sidecar              # autostarts torch sidecar + asserts
npm run verify:codegen-security     # 5362 hostile-payload cases through Python ast/tokenize
npm run verify:command-injection    # args quoting vs a real sh, SSRF policy, ssh target policy
npm run verify:code-trust           # trust store + collector + ALLOWED_APPROVERS invariants
npm run verify:code-trust-wiring    # untrusted code never reaches /infer, smoke, startRun
npm run test:deps-policy            # pip spec policy   (run inside the conda env)
npm run test:run-script             # torch /run_script relpath policy (inside the conda env)
npm run verify:paths                # Node symlink-aware path containment (48)
npm run verify:reference            # generated model == hand-written PyTorch (params/forward/loss/grads)
npm run verify:reference-train      # reference experiments through the REAL trainer (inside the conda env)
npm run verify:manifest             # manifest.json + git state + config identity (inside the conda env)
npm run test:property               # 200 random valid graphs vs an independent oracle (inside the conda env)
npm run test:fuzz                   # 948 invalid mutants must be rejected cleanly (inside the conda env)
npm run verify:silent-catch         # no undocumented swallowing catch in src/ or sidecar-llm/
npm run test:llm-safety             # real LLM sidecar vs a fake OpenAI server (hostile tool calls, failures, secrets)
npm run test:llm-validation-parity  # sidecar tool validation vs the frontend registry; run `npm run gen:layer-catalog` after registry changes
npm run typecheck:scripts           # tsc over scripts/ — tsx does NOT type-check
npm run ci                          # every suite with timeouts/clean env (`npm run suites` lists them; run it with no app/sidecar running)
npm run test:scope                  # Python scope + hostile manifests + real HTTP (inside the conda env)
npm run test:safe-load              # real malicious-pickle attempts (inside the conda env)
npm run test:sidecar-auth-torch     # token/Host/Origin vs a REAL torch sidecar process (inside the conda env)
npm run test:sidecar-auth-llm       # same for the LLM sidecar + /respond + MCP bridge secret
npm run test:sidecar-auth-frontend  # sidecarFetch retry/401/403/health states vs a fake sidecar
npm run verify:sidecar-fetch        # no bare fetch( in src/ outside src/sidecars/auth.ts
npm run verify:remote-deploy-files  # Rust SIDECAR_FILES covers the sidecar-torch import closure + espf data
npm run test:opencode-lifecycle     # opencode provider vs a fake opencode binary (bridge auth e2e, exit/abort/timeout)
npm run test:codegen-golden         # 40 byte-exact generated-Python cases + coverage guard (`--update` for deliberate changes)
npm run test:process-lifecycle      # both sidecars: port busy/invalid port/SIGTERM without orphans/start-stop cycles (inside the conda env)
npm run verify:silent-except-py     # Python swallowing-except guard (inside the conda env); verify:rust-panics = Rust twin
npm run verify:ts-safety            # ratchet: no new any / unchecked cast / non-null assertion / floating promise in src/
npm run test:resource-leaks         # fd/thread/RSS/child/in-process-state bounds under sustained use (inside the conda env)
npm run test:soak -- --seconds 3600 # long mixed-load run with real trainer runs (default 90 s is what CI runs)
npm run verify:doc-refs             # every path / npm script / link in docs/engineering, FEATURES.md, CLAUDE.md exists
npm run test:webview-csp            # real headless Chromium on dist/ with the CSP of tauri.conf.json (SKIPPED without a browser)
npm run verify:opencode             # LLM sidecar must be up; asserts /opencode/models
                                    # + a real opencode chat + clean bogus-model error
```

After Rust changes:
```bash
conda run --no-capture-output -n mlforge-dev bash -c 'cd src-tauri && cargo check && cargo test'   # the toolchain is in the conda env, not on the base PATH
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

1. `sidecar-llm/main.mjs` `buildToolSpecs`: add another `tool(name, desc,
   zodSchema, handler)` (the local shadow returns a spec, not an SDK tool).
   The handler mutates the in-memory `ctx` (so subsequent tool calls in the
   same turn see the new state) AND pushes an `action` event via
   `actions.push(...)`. All three provider paths reuse the spec automatically.
2. `invoke_handler` (Rust) — no change; LLM tools live in node.
3. `src/chat/store.ts` `dispatchAction`: add a case that maps the
   `action` op to a GraphStore mutation.
4. Update `allowedTools` in the subscription `query()` config in main.mjs
   (only the subscription path needs the `mcp__graph__<name>` allow-list; the
   anthropic / openai-compat loops expose every spec automatically, and the
   opencode path serves every spec via the MCP bridge `mcp-bridge.mjs` without
   an allow-list).
5. Bump `maxTurns` (subscription) / `MAX_TOOL_TURNS` (direct-API loops) if your
   tool takes many calls per request.
6. Update CLAUDE-the-model's awareness via `buildSystemPrompt`: mention
   the new tool, its idiomatic use, and dim-correctness rules.

### Add a new dataset kind

1. `sidecar-torch/dataset_handlers.py`:
   - extend `detect_kind()` with new ext/heuristic
   - add `_inspect_<kind>(abspath)` returning `{kind, ok, ...}`
   - add `_stats_<kind>(abspath)` (may just return `{ok: True}` if same as inspect)
   - add `_sample_<kind>(abspath, target_shape)` returning `{ok, tensor, natural_shape, note}`
     so smoke_test can feed real data through the model
   - wire all three into the dispatch (`inspect()`, `stats()`, `sample_tensor()`)
2. `src/datasets/types.ts` — add an `<Kind>Inspect` / `<Kind>Stats` type and union it.
3. `src/datasets/icons.ts` — `iconFor()` + `colorFor()` cases + `guessKindFromName()`.
4. `src/datasets/DatasetDetail.tsx` — a `<Kind>Overview` component (Use-as-input
   button with a sensible default shape), a `<Kind>StatsView` if stats differ
   from inspect, and switch cases in `OverviewBody`/`StatsBody`.
5. Optional Python deps: lazy-import inside the handler; on `ImportError` return
   `_missing_dep(kind, "pip-name")` — the UI shows a hint with the pip command.

### Multitask training (multiple output heads)

The trainer is uniformly multi-head; single-task is just one head.
- **Data model**: `TrainingConfig.heads?: Head[]` (types.ts). Each `Head` =
  `{output, target, loss, weight, label_smoothing?}`. `output` matches a model
  `Output` node's name (the dict key `forward()` returns); '' = the sole output.
  When `heads` is present (≥1) the run is multitask; otherwise the legacy
  `loss` + `dataset.target_column` path is used.
- **Graph**: a `Head` node (training registry, category Objective, `multi`).
  ≥1 Head node → multitask; the single `Loss` node + `DatasetSource.target`
  become optional. Compiled in `trainingGenerator.ts` (`byType('Head')`); the
  Head node's `target` column field resolves against the graph's DatasetSource
  (TrainingInspector computes `boundDataset` from it since a Head has no
  `dataset` param of its own).
- **Python** (`training_template.py`): `resolve_heads()` builds the head list
  (single-task synthesises one head); `load_tabular`/`load_manifest_graphs`
  return per-head targets keyed by output name; `MultiTaskDataset` +
  `make_collate` yield `(xb, {output → y})`; `resolve_outputs()` maps the model
  return (dict by name / tuple by order / single tensor) to per-head tensors;
  the objective is the weighted sum of per-head losses. Per-head metrics are
  emitted namespaced `"<output>/<metric>"` (so `metricSeries` auto-charts them),
  and `eval.summary`/`sample.preds` carry a per-head `heads:[…]` array.
- **UI**: `RunDetailModal` renders one eval section per head (multitask) via
  `latestEvalHeads`/`EvalDiagram`; `NewRunModal` shows a read-only heads summary
  instead of the single target picker when `cfg.heads` is set.
- **Verify**: `npm run verify:traingen` covers both compile and an end-to-end
  2-head (classification + regression) run.

### Add a Rust filesystem command

1. `src-tauri/src/lib.rs` — write `#[tauri::command] fn foo(...)`, scope
   to `WorkspaceState` so paths are validated under the workspace root.
   ALWAYS go through `resolve(&root, &relpath)` — that's the path-sanity
   gate (rejects `..` and absolute paths).
2. Add to `tauri::generate_handler![...]` at the bottom.
3. `src/workspace/tauri-fs.ts` — add a typed wrapper.
4. `src/workspace/store.ts` — branch the action on `mode === 'tauri'`.
5. **Mirror it on the remote side** if it's a FS-touching command —
   see "Add an ssh_* mirror" below. Skipping this is what breaks remote
   workspaces silently.
6. `cargo check` then `npm run tauri dev` and exercise it.

### Add an ssh_* mirror for a remote-mirrored command

When you add a local `foo(state, …)` that reads/writes the workspace,
also add the remote-mirrored variant:

1. `src-tauri/src/ssh.rs` — write `#[tauri::command] pub fn ssh_foo(alias,
   root, …)`. Use `validate_alias` + `validate_remote_root` +
   `validate_relpath` for every input that touches the shell. NEVER
   `format!()` user input into a remote command without
   `shell_quote_path` (for paths) or `shell_quote` (for non-path strings).
   Tilde-prefixed roots are handled by `shell_quote_path` — `'~/foo'`
   becomes `"$HOME"'/foo'` so the remote shell expands it.
2. Register in `lib.rs` `generate_handler![…, ssh::ssh_foo, …]`.
3. `src/connections/tauri-ssh.ts` — typed wrapper `ssh_foo(alias, root, …)`.
4. `src/connections/backend.ts` — add the operation to the right backend
   object (`fs.foo`, `project.foo`, `notes.foo`, …) and dispatch on
   `getCurrentConnection().kind`. **Every consumer goes through this**;
   that's where the local↔remote switch happens.
5. If the result shape differs from local, normalize at the backend
   boundary so consumers see one shape (cf. `ProjectLoadResult`).
6. Test with a real SSH alias: `ssh <alias> echo ok` first, then exercise
   the UI path.

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
7. **`pyTwinPath(relpath)` mapping**: `.spinoml` → `.py` with the same
   stem, sanitised. Save in Tauri mode writes both atomically (well,
   sequentially with no rollback — best-effort). Don't introduce a
   second naming scheme.

8. **Generated Python is built only from `pyLiteral.ts` values.** Never write
   `'${userString}'` or a user string inside a `# comment` in a generator. The four
   code-by-design sinks (Custom.source, Custom.init_args, DataOp.script,
   CustomScript.code) are the only exceptions and are listed in the harness.
9. **Code-bearing nodes only execute if approved by a human.** `trust.approve` may be
   called only from the files in `ALLOWED_APPROVERS` (scripts/verify-code-trust.ts), only
   from a user-initiated DOM/UI handler and only with `userEdited === true` — never from
   `src/chat/`, persistence, workspace or `GraphStore` (LLM path and file load must not
   approve). Any new place that sends generated code to a sidecar or executor must call
   `assertTrusted`/`listUntrusted` first.
10. **No shell command is built by concatenating LLM/user values.** Use
    `splitArgs`+`quoteArgv`/`shellQuote`, a `./` prefix for script paths, `validate_alias`
    plus `--` for ssh targets, `deps_policy.validate_specs` for pip. After any Rust change run
    `cargo check && cargo test` inside the conda env (a bare shell has no `cargo`) and say what you ran.
11. **Never open a request- or file-content-derived path without `scope.check_path`** (Python) /
    `resolveInWorkspace` (Node), and never `torch.load` directly — `safe_torch_load` only. New
    dataset kinds must add their open sites to the audit list at the top of `dataset_handlers.py`
    and to `scripts/test-scope.py`.
12. **LLM tool handlers validate first, mutate last.** Every provider funnels through `invokeTool` (schema + strict top-level
    arguments); node params must pass `validateNodeParams` (catalog generated from the frontend registries). A handler may not push an action
    or touch `ctx` before all checks passed, and a tool result must never say "added" for something the frontend will change or reject.
    New registry fields/layers: `npm run gen:layer-catalog`, then `test:llm-validation-parity`. New scripts must be registered in
    `scripts/suites.ts` (`npm run ci -- --check` fails otherwise).

13. **Every sidecar request goes through `sidecarFetch`/`torchFetch`; every sidecar endpoint goes through `decide`.** A new
    HTTP endpoint inherits the token/Host/Origin gate automatically only if it is added inside the existing handlers — never
    create a second server/listener, never answer before `decide` ran, never emit `Access-Control-Allow-Origin: *`, never put the
    token in argv, a log line, an error message or a file, and never hand the master token to a child process (the opencode MCP
    bridge uses its own per-turn secret). A new `sidecar-torch/*.py` module must be added to `SIDECAR_FILES` in
    `remote_sidecar.rs` (`verify:remote-deploy-files`).

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
- "Verbindung testen" returns SSH exit 255 → BatchMode is on for
  non-terminal ssh calls, so password prompts auto-fail. Fix: configure
  key-based auth (or GSSAPI/Kerberos) for that alias. The terminal
  itself uses `SSH_OPTS_INTERACTIVE` and CAN prompt — try opening the
  Terminal tab first to accept host keys / enter 2FA.
- Remote dataset smoke test fails or is unavailable → the remote torch sidecar is probably not running: the
  header badge shows its state; start it from the remote-sidecar badge (it bootstraps over ssh, see
  docs/engineering/REMOTE_TRAINING.md §3). The local torch sidecar cannot open paths like `/scratch/...`.
- Terminal shows "[terminal exited]" immediately on remote connect →
  `ssh -tt` was rejected (255 = auth/network, other codes = remote
  shell). The exit line includes the captured stderr.

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
  already a tax.
- Don't add a new FS backend by branching inside each store. The
  branch lives in `src/connections/backend.ts` — everyone else dispatches
  through it. If you find yourself writing `if (conn.kind === 'remote-ssh')`
  in a store, that's a smell.
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
