# SpinoML — feature & operation guide (read this first)

> **Purpose.** This is the fast, LLM-oriented map of *what SpinoML can do and how
> to operate it* — so you (a future Claude session, or the in-app chatbot) can act
> immediately without re-deriving the whole app. CLAUDE.md is the *code-change
> runbook* ("how/where to edit"); this file is the *capability catalog* ("what
> exists, how it's used, where the source of truth lives").
>
> **Maintenance rule (do not skip).** This doc is **concepts + pointers**, never a
> copy of code. When you add/change a user-facing feature, (1) update the relevant
> section here and (2) add a dated line to the **Changelog** at the bottom. Exact
> enumerations (every layer, every dataset kind) live in the cited source files —
> link to them, don't duplicate them (duplicates rot).

---

## 1. What SpinoML is

A desktop (Tauri) + browser app for building, inspecting, training, and explaining
PyTorch models **visually on a canvas**, with an LLM chatbot that can drive every
operation. Target user builds paper-grade ML (graph/GNN + molecular/protein
dual-encoders). The canvas → PyTorch codegen is **pure** (same graph → same code).

Three runtimes (see CLAUDE.md "Two execution modes"): browser dev, Tauri dev,
installed `.deb`. Two FS backends in Tauri: **local** folder or **remote-ssh** HPC
workspace (dispatched in `src/connections/backend.ts`).

## 2. The three canvases

Each is a separate Zustand store + registry + codegen, file-bound via
`canvasdoc/CanvasFileGate`:

| Canvas | Builds | Registry (source of truth) | Codegen | File |
|---|---|---|---|---|
| **Architecture** | a `nn.Module` | `src/layers/registry.ts` | `src/codegen/generator.ts` | `*.spinoml` |
| **Training** | a training config | `src/training/graph/registry.ts` | `src/codegen/trainingGenerator.ts` | `*.spinotrain` |
| **Data** | one linear pandas pipeline | `src/data/graph/registry.ts` | `src/codegen/dataGenerator.ts` | `*.spinodata` |

## 3. Architecture nodes (the model)

The full list + every param lives in **`src/layers/registry.ts`** (`LAYERS`). Mental
model by `kind`:

- **`input`** — becomes a `forward()` argument. Types:
  - `Input` — a plain tensor (set `shape`, `dtype`).
  - `Graph` (`graphInput:true`) — ONE whole PyG `Data` (x+edge_index+batch+edge_attr);
    feed a GNN/pool and the generator unpacks `x, edge_index, batch = g.x, …`.
  - `Sequence` — token-id `LongTensor` (int64) → Embedding/Transformer. Bound to a
    manifest branch `kind:"sequence"` (char/byte tokenizer).
  - `ESPF` — SMILES tokenized into **interpretable substructure** subword tokens
    (MolTrans ESPF codebook) → Embedding. Bound to a manifest branch `kind:"espf"`.
    Params: `codebook` (`drug`/`protein`), `shape` (max tokens, default 50). Token
    id ↔ named substructure; substructures surface in the Explain view.
- **`output`** — `Output` node; its `name` is the dict key `forward()` returns
  (multi-output models return a dict; ≥1 `Head` in training → multitask).
- **`manifest`** — `Manifest` node: a **non-emitting** pairing descriptor. Bind it
  to a `.manifest` and draw one edge → each typed input to declare which branch
  feeds which. Generates no model code (see §5).
- **`module`** — standard `nn.*` layers (Conv/Linear/Norm/Activation/Pool/Attention/
  Recurrent) + GNN (`GCNConv`/`GATConv`/`SAGEConv`/`GraphConv`/`GraphTransformer`,
  `GlobalMean/Max/AddPool`). GNN layers act on node features `[N, in_ch]` (no batch).
- **`merge`** (`Concat`/`Add`/`Multiply`/`Stack`) — N→1; **`function`**
  (`Reshape`/`View`/`Permute`/`Transpose`, global pools) — 1→1.
- **`custom`** — free-form `nn.Module` from source. **`group`** (`Subgraph`) — nested
  reusable class (double-click → subcanvas); put each dual-encoder branch in its own.
- **`dataop`** — `DataOp` (category Data): carries a Python `script` run OFFLINE via
  the chatbot (`write_file`+`run_script`) to download/tokenize/cache a dataset; a
  pure passthrough in codegen.

**Adding a layer = editing `registry.ts` only** (recipe in CLAUDE.md). Codegen is
uniform; multi-input/branching layers need teaching `generator.ts`.

## 3. Training Strategy & Integrity (Phase 19)

The Split node's `strategy` (`random`/`stratified`/`grouped`/`time-based`/
`predefined`) is frozen into `run.json` (`training.split_strategy`) at launch and
**never silently changed**. The trainer only implements `random` today; any other
strategy in run.json produces an immediate loud failure rather than a silent random
fallback. After splitting, the trainer asserts zero overlap between train and
validation sets (`split.integrity` event with `overlap=0`) and records the
strategy used, so every run documents exactly HOW its splits were formed. A
non-zero overlap (from a future custom split method) is a fail-closed refusal.

### Run snapshot (Phase 20) — the experiment never touches mutable UI state

At launch the executor freezes FOUR files into the run dir: `run.json`, the
`model.spinoml` graph, the generated `model.py`, and the `train.py` template —
the trainer runs detached on those copies, so editing the graph in the UI after
launch cannot change a running experiment. Since Phase 20 each `run.json` ALSO
carries a `snapshot` section: sha256 of the frozen graph (`graph_sha256`) and of
the generated model (`model_py_sha256`), plus the graph's DataOp preprocessing
scripts. `train.py` re-hashes the run-dir copies at startup and emits
`run.snapshot`; any drift (e.g. a hand-edited model.py in the run dir) is a loud,
halting failure BEFORE training — the executed artifacts are provably the
launch-time bytes.

### Training configuration & environment record (Phase 21)

`run.json` already froze the full training configuration at launch (graph,
generated model, dataset identifier + split + preprocessing + seed + optimizer/
lr/scheduler/batch/epochs/loss/metrics — see Phases 18–20). Phase 21 adds the
RUNTIME half of the record: once the model builds, `train.py` emits a
`config.env` event with software versions (python/torch/cuda/numpy), compute
device + GPU/VRAM, precision dtype (fp32/fp16/bf16), CPU count, RAM, and the
workspace git commit when it's a repo. The same summary is written into
`metrics.json` at the end, so a run's outcome can be attributed to the exact
stack it executed on — not just the config the user set.

### Seeds and determinism (Phase 22)

The trainer seeds EVERY random source: Python `random`, NumPy, `torch.manual_seed`,
`torch.cuda.manual_seed_all`, sets `cudnn.deterministic=True` + `cudnn.benchmark=False`,
and calls `torch.use_deterministic_algorithms(True, warn_only=True)` so
nondeterministic ops warn instead of silently diverging. DataLoader gets an
independent seeded `torch.Generator` + `worker_init_fn` so multi-worker
shuffling is reproducible too. All seed state is exposed in a `run.determinism`
event (`seed`, which backends were seeded, cuDNN settings). **Caveat:** CUDA
`atomicAdd`-based reductions remain nondeterministic even with these flags, so
we explicitly do NOT claim bit-level reproducibility on GPU — the event records
when that caveat applies (CPU runs are reproducible with the same seed + stack).

## 4. Datasets

Kinds + handlers live in **`sidecar-torch/dataset_handlers.py`** (and the UI types in
`src/datasets/types.ts`). Each kind has inspect / stats / sample. Kinds: tabular,
image_folder, tensor (`.pt`/`.npy`), protein (`.pdb`), molecule (SMILES), pyg,
huggingface, **manifest** (§5). A **prepared dataset DIRECTORY** (a folder whose primary
content is a table — `pairs.csv`/`data.csv`/the main CSV — optionally with side files like
`sequences.csv`, per-id embeddings, a `prep_card.json`; e.g. a TDC BindingDB export) is read
as **tabular on its inner table** (`detect_kind`/`_table_path`/`_dir_table`; inspect surfaces
the `bundle`), so such dirs are selectable for external validation (§6b). Heavy deps
(pandas/PIL/rdkit/biopython/torch_geometric) are lazy-imported → graceful `missing_dep`
instead of a crash. Datasets appear live in the Datasets tab when written under `datasets/`.
Every inspect result additionally carries a **content fingerprint** (Phase 18) —
a stable SHA-256 id that survives being copied/replaced/renamed and is determined by
bytes — never the human-readable
name. Modes: `content` (file bytes: tabular/tensor/molecule/protein), `structure`
(sorted relpath+size listing: image/graph folders), `config+content` (manifest +
its referenced table), `reference` (pyg/huggingface ref file only — the remote
data itself is NOT pinned). The fingerprint is attached to every `inspect`
result, cached by relpath, and **frozen into `run.json` (`dataset.fingerprint`)**
at launch with `feature_columns`/`target_column`/split config (the `Split` node's
`val_ratio`/`seed` inside `training`). The trainer re-emits it as
`run.provenance` + `dataset.fingerprint.check` (it REHASHES the primary file just
before loading, so a run frozen against one dataset but executed later on
different bytes fails loudly instead of silently training on the wrong data).
Only when the dataset was never inspected (e.g. remote workspaces pre-12b) is the
key absent. (Phase 18 — `sidecar-torch/dataset_handlers.py: _fingerprint_for` /
`training/types.ts DatasetConfig.fingerprint` / `training_template.py`.)

## 5. Manifests (paired / dual-encoder datasets)

A `.manifest` is a JSON descriptor pairing a **table** to **per-branch sources**
row-by-row (e.g. ligand SMILES + protein graph + target). Authoring: the chatbot's
`write_dataset_file`. Schema + branch resolution doc: the comment block above
`_read_manifest` in `sidecar-torch/dataset_handlers.py`.

Branch `kind` (HOW a cell becomes a tensor, and which input node consumes it):

| `kind` | cell → tensor | input node | key fields |
|---|---|---|---|
| `molecule` | RDKit atom/bond graph from SMILES | `Graph` | `column` |
| `espf` | ESPF substructure subword ids (MolTrans) | `ESPF` | `column`, `codebook` (`drug`/`protein`), `max_len` |
| `sequence` | char/byte token ids | `Sequence` | `column`, `vocab` (`protein`/`smiles`/chars/omit), `max_len` |
| *(file)* | `torch.load` a `.pt` | `Graph` | `column`, `dir` (ABSOLUTE), `match` (`exact`/`contains`), `ext` |

Any branch may also carry **`lookup`** — JOIN a side table by key to resolve the real
cell value before tokenizing/loading (e.g. a `prot_seq` branch keyed by `uniprot`
pulls the sequence from `sequences.csv`): `{column: "uniprot", lookup: "sequences.csv",
lookup_key: "uniprot", lookup_value: "sequence", kind: "espf", codebook: "protein"}`.
Tensor branches batch by N-D padding (`_pad_stack`): 1-D tokens → `[B, Lmax]`, 2-D
per-residue/atom features/embeddings → `[B, Lmax, D]`.

`target`: `{column, type:"classification"|"regression"}`. Inspecting a manifest in
the Datasets tab shows each branch's suggested node type + the Embedding
`num_embeddings` for sequence/ESPF branches. Sidecar tokenizes inline at
sample/train; ESPF caches a compact codebook to `<manifest_dir>/.espf/` (the
`.graphcache` analog) so the standalone training snapshot is self-contained.

## 6. Training

- Uniform **multitask**: `TrainingConfig.heads?: Head[]` (one head per model output);
  single-task = one head. Graph: a `Head` node (Objective). Compiled in
  `trainingGenerator.ts`; executed by `sidecar-torch/training_template.py`.
- A **run** = a self-contained dir `experiments/runs/<id>/` (frozen `model.py` +
  `train.py` snapshot + `run.json`); see `src-tauri/src/training.rs`. Runs locally
  (`setsid python train.py`) or via SLURM on remote. The training store polls run
  status (slower cadence on remote ssh).
- **Run state machine (Phase 30)**: the `status` file is governed by a real state
  machine — `queued → running → done|failed|cancelled`, terminal states FINAL.
  The trainer applies every transition via `transition_status()` (queued→running
  on start, running→terminal on finish/fail/cancel); `stop_training_run` (local +
  ssh) only flips a non-terminal status to `cancelled`. A late write — a cancel
  racing a `done`, a `running` after a crash, or a stop on a finished run — is
  REJECTED, so `CANCELLED → SUCCEEDED`, `FAILED → RUNNING`, `SUCCEEDED → RUNNING`
  cannot happen; a cancel landing in the finalisation window emits
  `run.cancelled` instead of `run.done`. Verify: `npm run verify:states`
  (transition table + cancel-before-start / mid-training / race / double-cancel
  / post-done guard).
- **Event ordering (Phase 31)**: `events.jsonl` is append-only + fsynced per
  line, so on-disk ordering is strict; the protection is at the read boundary.
  `src/training/events.ts` owns the pure vocabulary: the FIRST terminal event
  (run.done/failed/cancelled) decides the final state, anything after it — a
  trailing EPOCH after FAILED, a duplicate terminal write — is truncated away
  (`parseFinalEvents` feeds every `setEvents` in RunDetail/CompareModal), and a
  `latestWinsGuard` drops an older in-flight events read whose response lands
  after a newer one (the inference/store staleness guard, applied to events, so
  a remote-tail response can never overwrite an already-final view). Verify:
  `npm run verify:events` (parse tolerance, out-of-order EPOCH-after-FAILED,
  first-terminal-wins over duplicates, stale-read drop, modal flow).
- **Cancellation (Phase 32)**: SIGTERM and SIGINT unwind through one terminal-shielded cancellation path. It records exactly one `run.cancelled` event, writes cancelled metrics, and leaves `last.pt` at the final completed epoch; a first-epoch stop stores epoch -1 so resume starts at epoch 0. A late signal cannot change done or failed. Verify: `npm run verify:cancel`.
- **Submission idempotency (Phase 33)**: remote starts claim `experiments/runs/<run_id>` with one atomic `mkdir`. Concurrent starts have one owner; retries after a lost response detect the recorded `pid` and return success without a second launch. An existing run without a PID remains an explicit incomplete-submission error. Verify: `npm run verify:submission`.
- **SSH reliability (Phase 34)**: transport failures now report their actual cause: authentication or host-key failure, DNS/host unavailable, timeout, connection loss, missing remote path, permission denied, disk full, SFTP failure, or remote command failure. SSH uses a 10-second connect timeout, keepalives, and multiplexed connections that reopen after loss. Verify: `npm run verify:ssh` (and `cargo test --manifest-path src-tauri/Cargo.toml ssh_failure_tests`).
- **Credential safety (Phase 35)**: passwords, private keys, API tokens, OAuth secrets, and SSH credentials are strictly banned from experiment artifacts, logs, and frontend stores. The connection store delegates auth entirely to the user's `~/.ssh/config` and ssh-agent without persisting credentials. Runtime environment recording (`_env_info`) captures only whitelisted platform/torch diagnostics (never dumping `os.environ`), and `sanitize_credentials` strips private keys, URL passwords, and secret tokens from all SSH diagnostics. Verify: `npm run verify:credentials`.
- **SLURM reliability (Phase 36)**: every scheduler state is mapped explicitly — `sbatch` success writes `slurm:<jobid>` to `pid` + `train.sbatch` frozen, failure surfaces `sbatch failed:` with the tool's own error text; `squeue %T` while queued (PENDING→queued, RUNNING→running) and `sacct State` after the queue (COMPLETED→done, CANCELLED→cancelled, FAILED/TIMEOUT/OOM/NODE_FAIL→failed) via `reconcile_slurm_status`; UNKNOWN/communication loss falls back to `failed` for stale queued/running, never silently `running`. `build_sbatch` emits correct `#SBATCH` directives for partition/time/mem/cpus/gres/account/qos/modules/pre_run_script. Verify: `npm run verify:slurm` (33 checks: 4 sbatch + 9 reconciliation Rust tests + probe/persistence/markers).
- **Training immutability (Phase 42)**: once `training/store.ts:217` `startRun` is called, `RunConfig` freezes `modelContent` (via `fs.read`), `modelPy` (via `generateFromSnapshot`), `snapshot` (`graph_sha256`+`model_py_sha256` via `snapshot.ts:43`), `dataset`+`fingerprint`+`training`+`backend`; `training.rs:447`/`ssh.rs:919` write `run.json`/`model.spinoml`/`model.py`/`train.py` atomically into `experiments/runs/<id>/` and launch detached (`setsid`/`sbatch`, survives app close) — the run never reads `GraphStore` after launch. `training_template.py:342` `_verify_snapshot` re-hashes `RUN_DIR` copies and `fail`s loudly on drift; later UI edits bump `GraphStore.revision` but never touch the run dir. Verify: `npm run verify:immutability` (27 checks).
- **Concurrent operations (Phase 41)**: `workspace/store.ts:529` `saveSeq`+`revAtStart` → concurrent `saveActive` last-wins and an edit that raced the write is not lost (dirty corrected to `true`); `inference/store.ts:39` Edit+inference is debounced+`runCounter`+`graphRev`+`AbortController` (stale shape dropped, `test:races` proves it); `chat/store.ts:158` `if streaming return` blocks second LLM turn while one streams and `dispatchAction` goes through `validateGuard`/`coerceParams`; `training/store.ts:215` Training start freezes `modelContent`+`modelPy`+`snapshot` sha256 (run dir immutable, drift fails loudly); `datasets/store.ts:57` `inspectSeq`/`statsSeq`/`smokeSeq`+`smoke` `graphRev`+`refreshSeq` make Dataset reload+inspect latest-wins. All 5 concurrent pairs are allowed but stales are dropped, never silently corrupt. Verify: `npm run verify:concurrent` (44 checks) + `verify:graph-revision`.
- **Graph revision system (Phase 40)**: `GraphStore.revision` (monotonic, structural only — position drags don't bump) and per-dataset `inspectSeq`/`statsSeq`/`smokeSeq` plus `training refreshSeq` implement `Response revision == current revision?` → drop stale. `inference/store.ts:43` captures `graphRev`+`runId`, `datasets/store.ts:14` per-relpath seq + `smoke` graphRev, `training/store.ts:121` `refreshSeq` (like `events.ts:latestWinsGuard`). Predictable `latest-wins`, no stale shape/inspect/smoke/list overwrites. Verify: `npm run verify:graph-revision` (32 checks) + `test:races` (replay).
- **Frontend error states (Phase 39)**: Loading/Success/Error/Unavailable are distinct everywhere — `inference/store.ts:7` `idle`/`inferring`/`ok`/`error`/`offline` (+ stale `runCounter`, AbortError no longer hangs `inferring…`), `training/store.ts:76` `listLoading` vs `listError`+`unknown` vs `StatusPill` 6-way, `datasets/store.ts:12` `Cached<T>`, `project/store.ts:16` `remote-missing`, `chat` `streaming`/`done`/`error`+`online`. Timeout/Cancelled are still string-typed (SSH `ConnectTimeout`→`listError`, fetch timeout→`offline`/`error`, no `AbortSignal.timeout` or `cancelled` enum) — they never masquerade as Success. Verify: `npm run verify:frontend-errors` (42 checks).
- **UI state must not lie (Phase 38)**: `training/store.ts:155` `refresh()` degrades stale `running`/`queued` to `unknown`+`alive:false` on `listError` and clears `pollTimer` — a transient SSH loss never leaves a green `running` pill; `App.tsx:154` `ProjectHeader` turns violet `ssh · alias` into rose `ssh · alias — nicht verbunden` when `listError` or `status.kind==='error'`. `inference/store.ts:70` no longer `throw e` inside the debounced `kick` — non-abort fetch/sidecar errors now set `offline`/`error` + `clearShapesOnNodes()`, so `inferring…` cannot hang forever; `InferenceBadge` shows `ok` only on `ok`. Save paths are truthful: `workspace/store.ts:375` awaits `fs.write` before `dirty:false`, `training/graph/doc.ts:33` + `data/graph/doc.ts:41` set `saved` only on success. Verify: `npm run verify:ui-state` (33 checks).
- **Remote job recovery (Phase 37)**: close → restart → reconnect never trusts UI cache — `useTrainingStore.runs` is memory-only, `App.tsx:257` blanks `runs:[]` on workspace reload, `training.list()` then re-queries the actual remote `experiments/runs/<id>/{status,pid,run.json,metrics.json,events.jsonl}` plus live `squeue`/`sacct`/`kill -0` per run and reconciles (`reconcile_status`/`reconcile_slurm_status`); a stale `running` without a live pid/squeue becomes `failed`. Detached `setsid`/`nohup setsid` (direct) and `sbatch` `slurm:<jid>` (SLURM) survive app close (reparent to init); atomic `mkdir` claim makes a lost-response retry idempotent. `connections/store.ts` persists only `alias/root`, never run state; `getCurrentConnection()` and `training/backend.ts` dispatch live per call. Verify: `npm run verify:recovery` (38 checks).
- Trainable dataset kinds: **tabular** + **manifest**. Manifest branches batch as
  PyG `Batch` (graphs) or padded `[B, Lmax]` LongTensors (sequence/ESPF).
- **GPU**: `training_template.py` picks `cuda` when available, moves the model AND
  every batch (`to_device`, recurses tuples/dicts/PyG Batch) onto it; eval tensors
  come back to CPU for the JSON summaries. The actual device is recorded in
  `metrics.json` (`device`/`gpu`) and the `model.built` event. The SLURM form's
  `#SBATCH --gres=gpu:1` + `module load <cuda>` (ssh.rs) now actually trains on the
  GPU — before, the template never left the CPU regardless of the allocation.
- Verify: `npm run verify:traingen` (compile + 2-head e2e).

## 6b. External validation (run a trained model on a foreign/benchmark dataset)

Validate a FINISHED run's checkpoint on a NEW labeled dataset → eval metrics, no
retraining. An **eval run is a normal run** with `run.json` `eval_only:true` +
`validate:{checkpoint_from, source_run, adapter}` — launched via the same
`training.start` (no new Rust command). `training_template.py` `main()` branches to
the eval-only path: load the source `best.pt`, evaluate the WHOLE external dataset
once via the shared `evaluate()`/`emit_eval()` (same `eval.summary`/`sample.preds`
the Run-Detail UI renders), write `metrics.json`, done. Classification targets are
encoded against the checkpoint's **trained** `head_classes` (correctness — indices
must match the model).

- **Entry**: "Externe Validierung" button on a finished run (`RunDetailModal`,
  shown when `has_checkpoint && !eval_only`). Eval runs get a **VAL** badge in
  `ExperimentsExplorer` and an "EXTERNE VALIDIERUNG" banner in the detail modal.
- **Adapter** (`src/training/adapter.ts`, pure + unit-tested): `modelContract`
  derives the model's trained schema (features-in-order / target / branches) from
  the source run + its dataset inspect; `suggestAdapter` auto-maps the external
  columns (exact → fuzzy name → numeric dtype → SMILES sniff), flagging unmatched
  roles. `EvalRunModal.tsx` shows the mapping as an editable table (auto = the
  suggestion, edit = manual → **hybrid**). The eval loader consumes the chosen
  external `feature_columns`/`target_column` directly — rename/select/reorder needs
  **no materialization**.
- **Per-output selection**: a multi-output model can be validated on a subset of its
  heads — de-select outputs the external set has no target for (e.g. validate only the
  classification head when there's no affinity column). `EvalRunModal` checkboxes →
  `EvalRunInput.heads` → `training.heads` for the eval run.
- **Manifest model on a prepared DIRECTORY**: `EvalRunModal` generates an *editable
  adapted manifest* (`buildAdaptedManifest`) — reuses the source model's branch
  kinds/codebooks and points each branch at a directory resource: SMILES column →
  drug-ESPF; ligand 3D → inline `molecule` graph (no precompute); protein sequence →
  `lookup` join into `sequences.csv`; per-protein `.pt` → `dir`. The modal **inspects
  the written manifest first** so the sidecar materializes the `.espf` cache the run
  needs, then validates the selected output(s).
- **Complex adaptation** (units, log-scale, computed cols, structure→graph, building
  a matching manifest): the "Im Daten-Canvas anpassen" button seeds the Data canvas
  (§2) on the external dataset → user builds a pandas pipeline → `WriteDataset` →
  validate on the adapted output. The eval engine is kind-agnostic (tabular,
  molecule/SMILES, manifest, structure files) — it runs the model over whatever the
  loader yields.
- Verify: `npm run verify:traingen` (case 5: train → eval-only on a RENAMED dataset
  → asserts no training loop, trained-class confusion, `metrics.json` eval_only).

## 7. The in-app chatbot (`sidecar-llm/main.mjs`)

SSE server on :7422. Four provider paths (`opencode` [DEFAULT] / `subscription` /
`anthropic` / `openai-compat`) share one tool registry (`buildToolSpecs`). The
`opencode` path spawns the `opencode` CLI per chat turn (`opencode run --format
json --model <model>`) with the model driven ONLY through a stdio MCP tool
server (`sidecar-llm/mcp-bridge.mjs`, named `graph` → tools appear as
`graph_<name>`); every opencode built-in tool is disabled so nothing can bypass
the graph-validation gate. Tool calls proxy via `POST /internal/mcp/<requestId>/
list|call` to the SAME handlers as the other paths. Model list for the settings
UI: `GET /opencode/models` (`opencode models`, cached 60s). System prompt:
`buildSystemPrompt`. Tools (each mirrors an `action` to the frontend store):
- Graph mutation: `add_layer` / `connect` / `update_params` / `add_subgraph` / …
  (architecture), `add_data_node` (data canvas), training-param tools.
- Files/exec: `read_file` / `write_file` / `list_dir` / `write_dataset_file`
  (manifests/CSVs) / `run_script` (mode `shell` = run now, GUI-gated; `slurm` =
  sbatch) / `slurm_status`. All routed local-or-ssh. `agent/` is the scratch dir.
- Ask/answer: `ask_user` + run confirm via the `ask`/`POST /respond` channel.
- Provenance: `record_step` → `notes/lab-notebook.md` (reproducibility).

When you add an LLM tool or a node, update the relevant strings in `main.mjs`
(`add_layer` list/prose, `write_dataset_file` schema, `buildSystemPrompt`) so the
chatbot knows about it — recipe in CLAUDE.md "Add a new LLM tool".

## 8. Inspect / smoke-test / Explain

- **Shape inference**: `src/inference/store.ts` (debounced, re-entrancy-guarded) →
  `inferredOutputShape` on nodes.
- **Smoke test**: `src/datasets/store.ts` `runSmoke` → POST `/dataset/*` and
  `/infer`; feeds real sampled data through the model (kind-driven over input nodes).
- **Explain mode** (`src/visualization/`): on-demand forward pass → POST
  `/activations` → per-node activations + weights (downsampled). The `tokens`
  preview shows ESPF substructure **labels** when the input is an ESPF node.

## 8b. Code trust (execution gate)

Arbitrary Python in a graph (Custom `source`/`init_args`, DataOp `script`, data
`CustomScript` `code`) may only run if its content hash is in a local trust
store (`src/trust/trustStore.ts`, sha256 over `kind\0source`). Every executor
asks the gate first: `src/trust/guard.ts` (`listUntrusted`/`assertTrusted`) is
called by inference, visualization, dataset smoke, the training verifier and
`training/store.ts` before launch; fail-closed leaves
`useInferenceStore.status === 'untrusted'` with the offending blobs.
The ONLY approval paths are human-initiated: editing a code field / `init_args`
in the Inspector (`human-edit`), "Zu Custom-Code umwandeln" (`eject`), inserting
a built-in template (`template`), and the explicit click in the approval dialog
(`user-approval`, `src/trust/ApproveCodeDialog.tsx`). The chat/LLM path and file
load/restore never approve. The amber `InferenceBadge`, the blocked
"Vorverarbeitung ausführen" (DataOp), "Pipeline ausführen"/"via Chat" (data
canvas) and the blocked training launch open the dialog via
`src/trust/useApproveDialog.ts`. `scripts/verify-code-trust.ts` enforces
`ALLOWED_APPROVERS` + origin/HTML static invariants.

## 8c. Command, argument and generated-code safety (Phase 43/44)

- **Generated Python is inert for hostile values**: every string/number the three
  generators interpolate goes through `src/codegen/pyLiteral.ts` (`pyStr`,
  `pyComment`, `pyIdent`, `pyFloat/pyInt/pyIntList`). Only four sinks are code by
  design (`Custom.source`, `Custom.init_args`, `DataOp.script`, `CustomScript.code`);
  they are gated by §8b. Proof: `npm run verify:codegen-security`.
- **Chatbot `run_script`**: `args` are split by a POSIX-like tokenizer with no
  expansion (`sidecar-llm/shell-safety.mjs` `splitArgs`/`quoteArgv`), the confirm
  dialog shows the quoted argv that really runs, and the script path is passed as
  `./<path>` so it can never be read as an option. Hostile args are rejected
  pre-confirm. Proof: `npm run verify:command-injection`.
- **`download_to_datasets`**: URL policy (`checkDownloadUrl`: http/https only, no
  credentials, ports 80/443/8080/8443, no internal names/private/loopback/
  link-local addresses incl. IPv6 mapped/NAT64/6to4/Teredo, fail-closed on invalid
  input) + local `safeFetch` (DNS resolution check and a re-check of every redirect
  hop). Remote downloads use curl with `--proto`/`--max-redirs`; DNS-name SSRF on
  the remote network is not detected (docs/engineering/LIMITATIONS.md §2).
- **Torch sidecar**: `/deps/check`/`/deps/install` accept only plain PyPI
  requirements (`sidecar-torch/deps_policy.py`; `torch`/`pip`/`setuptools` refused;
  `error_code:"INVALID_SPEC"`); `/run_script` accepts only whitelisted relative
  script names (`error_code:"INVALID_RELPATH"`). `root` is still taken from the
  request (path scoping = Phase 45/46).
- **ssh targets** (Rust): `validate_alias` rejects a leading `-` and is shared by
  `ssh.rs`, `pty.rs`, `remote_sidecar.rs`; every spawn passes `--` before the
  target. (`cargo check` and `cargo test` pass in the `mlforge-dev` conda env.)

## 8d. Filesystem scope and safe unpickling (Phase 45–47)

- **Sidecar file scope** (`sidecar-torch/scope.py`, Node `sidecar-llm/path-scope.mjs`): a
  path is served only if its fully resolved location lies under an allowed root or a
  user-listed symlink target. Configure with `SPINOML_ALLOWED_ROOTS` (workspace root),
  `SPINOML_SYMLINK_TARGETS` (e.g. `/work2/...` when `datasets` is a link to scratch) or
  `~/.cache/spinoml/scope.json` (`{"version":1,"roots":[...],"symlink_targets":[...]}`, owner-only,
  re-read on change). `GET /health` → `scope.mode`: `enforced` | `unconfigured-open` (no
  restriction, warning — the local app today) | `unconfigured-closed` (`SPINOML_REQUIRE_SCOPE=1`).
  Errors: `SCOPE_DENIED`, `PATH_SYMLINK_OUTSIDE`, `SCOPE_UNCONFIGURED`, `PATH_INVALID`, each with a
  one-line fix. Manifests/table cells that point outside are refused the same way.
  The Rust shell writes this file itself (`src-tauri/src/scope_file.rs`) whenever a local workspace is
  picked/opened (and clears the roots on close), so a workspace turns the managed sidecars `enforced`;
  the Rust file commands check the fully resolved path the same way (symlinks out of the workspace need
  an entry in `symlink_targets`).
- **Safe `.pt` loading** (`sidecar-torch/safe_load.py`, embedded block in `training_template.py`):
  `weights_only=True` + PyG/numpy allow-list; a file needing arbitrary unpickling is refused with
  `UNSAFE_PICKLE` (datasets), a note (activations) or a failed run (trainer). Escape hatch for
  files you trust: start the sidecar/trainer with `SPINOML_ALLOW_UNSAFE_PICKLE=1` (recorded in
  the run as `unsafe_pickle`). Proof: `npm run test:safe-load`, `test:scope`, `verify:paths`.

## 8e. Sidecar authentication (Phase 77/78)

- **What the user sees**: nothing in normal use. The app generates a random token per launch
  and attaches it to every sidecar request. Header badges show `auth failed` (rose, tooltip =
  German message) instead of `offline` when a sidecar rejects the token; an amber
  `ungesichert` chip appears when a Tauri build talks to a sidecar that runs without a token
  (e.g. one started by hand). Browser-dev (`npm run dev` + manual sidecars) runs tokenless
  on purpose and is marked `unauthenticated-dev` in `/health`.
- **Wire protocol** (`docs/engineering/SIDECAR_AUTH.md` is the spec): header
  `X-SpinoML-Token` on everything except `OPTIONS` and `GET /health`; Host must be a loopback
  name; `Origin`, when present, must be in the exact allow-list (`tauri://localhost`,
  `http://tauri.localhost`, `https://tauri.localhost`, `http://localhost:5173`,
  `http://127.0.0.1:5173`, + `SPINOML_ALLOWED_ORIGINS`); CORS echoes the allowed Origin
  (never `*`). `/health` without a token returns only `{ok, auth, requiresAuth, tokenOk:false}`.
- **Where it lives**: `sidecar-torch/auth.py`, `sidecar-llm/auth.mjs` (same decision order),
  `src-tauri/src/sidecar_auth.rs` (token generation + the `sidecar_token` command),
  `src/sidecars/auth.ts` (`sidecarFetch`: header, retry once on 401, `SidecarAuthError`,
  `probeSidecarHealth`). `scripts/verify-sidecar-fetch.ts` fails if any `src/` file calls bare
  `fetch` — a new client MUST use `sidecarFetch`/`torchFetch`.
- **Environment** (sidecars): `SPINOML_SIDECAR_TOKEN` (≥ 32 chars `[A-Za-z0-9_-]`, else exit 2),
  `SPINOML_REQUIRE_TOKEN=1` (token mandatory; the Rust shell sets both for managed sidecars),
  `SPINOML_ALLOWED_ORIGINS`. The sidecar deletes the token from its own environment after
  reading it, so children (pip, scripts, opencode) never inherit it.
- **Remote sidecar**: a fresh token per session, delivered over ssh stdin (never argv).
  `deploy()` ships every file in `SIDECAR_FILES` (`remote_sidecar.rs`) — add a new
  `sidecar-torch/*.py` module there (`npm run verify:remote-deploy-files` tells you).
- **opencode MCP bridge**: authenticated with a per-turn session secret (env
  `SPINOML_MCP_SECRET` via the opencode `environment`), never the master token; ask/request ids
  are random UUIDs.
- **Proof**: `test:sidecar-auth-torch`, `test:sidecar-auth-llm`, `test:sidecar-auth-frontend`,
  `test:opencode-lifecycle`, `verify:sidecar-fetch`, `verify:remote-deploy-files`, `cargo test`.
  Limits (webview trust, same-user processes, CSP/Monaco CDN): LIMITATIONS.md §2, R052.

## 9. Common tasks (how to do X)

- **Build a dual-encoder (ligand + protein → affinity)**: `Manifest` → `Graph`
  (ligand `kind:"molecule"`) + `Graph`/`Sequence`/`ESPF` (protein) → per-branch
  encoder (own `Subgraph`) → `Concat`/bilinear head → `Output`; manifest `target` =
  affinity. Templates: `src/templates/templates.ts` (`buildDualEncoderGnn`).
- **Tokenize SMILES interpretably**: manifest branch `kind:"espf"` + an `ESPF` node
  → `Embedding(num_embeddings=ESPF vocab)` → 1D-CNN/Transformer.
- **Offline preprocessing**: a `DataOp` node (or the Data canvas), script under
  `agent/`, run via `run_script`; write outputs under `datasets/`.

## Changelog (append one dated line per feature; newest first)

- 2026-10-06 — **Rust panic/swallow audit (Phase 48 + 50 Rust half)**: no `unwrap`/`panic` in non-test Rust, 112 best-effort sites classified and guarded (`npm run verify:rust-panics`); fixes: u32-epoch overflow in the run list, a false "no runs" list on a permission error, terminal-session id collisions.
- 2026-10-06 — **Codegen golden tests + MultiheadAttention fix (Phase 6)**: the generated Python of 40 reference graphs (model, training, data) is pinned byte for byte (`npm run test:codegen-golden`, `scripts/golden/`; a new layer/node needs a golden case or an explicit exclusion); the `MultiheadAttention` layer now generates a valid self-attention call (`self.mha(x, x, x, need_weights=False)[0]`) — before, the layer raised a TypeError.
- 2026-10-06 — **Scope file written by the app + symlink-aware Rust FS (R016)**: opening a local workspace now enforces the sidecar path scope (`src-tauri/src/scope_file.rs` writes `scope.json`); `resolve()` rejects escaping/dangling symlinks and delete/rename act on a link, never its target; `list_workspace` follows allowed out-of-tree links with cycle protection (see §8d).
- 2026-10-06 — **Python silent-exception audit (Phase 50)**: `npm run verify:silent-except-py` fails any undocumented swallowing `except` in `sidecar-torch/*.py`; resume records `rng_restore` per random stream; a status file that cannot be read, an empty training loader and a corrupt manifest no longer produce false `done`/`resumable` (see `docs/engineering/SILENT_EXCEPTIONS.md`).
- 2026-10-06 — **Sidecar authentication (Phase 77/78)**: per-launch token + Host/Origin allow-list on both sidecars (no more `CORS *`), `sidecarFetch` wrapper with an `auth failed` UI state, per-session remote token over ssh stdin, per-turn MCP bridge secret, random ask/request ids; also fixes the remote deploy that shipped only 2 of the sidecar's files (see §8e, `docs/engineering/SIDECAR_AUTH.md`).
- 2026-10-06 — **Test infrastructure (Phase 67–72)**: `npm run suites` / `npm run ci` — one registry of 54 suites by test-pyramid category with hard per-suite timeouts, scrubbed environment (no API keys/ssh agent), port preflight, scoped leak detection and honest PASS/FAIL/SKIPPED/BLOCKED/TIMEOUT statuses; `npm run typecheck:scripts`; `.github/workflows/ci.yml` (defined, not yet run on GitHub); pinned `sidecar-torch/requirements.txt`.
- 2026-10-06 — **LLM tool-call safety (Phase 14–16, 76)**: every provider's tool calls pass one gate (argument schema, strict top-level args, JSON errors) and a catalog-based validator that accepts only values the frontend would store unchanged; handlers are atomic; connect rejects self-loops/cycles; broken provider streams end as explicit errors with abort + idle timeout (`SPINOML_LLM_UPSTREAM_TIMEOUT_MS`); API keys are redacted from error events. `SPINOML_LLM_PORT` selects the sidecar port. `npm run test:llm-safety` (fake OpenAI server, 124 assertions), `npm run test:llm-validation-parity`, `npm run gen:layer-catalog`. See `docs/engineering/LLM_SAFETY.md`.

- 2026-10-06 — **Result integrity + resumable runs (Phase 73/74)**: `done` requires an integrity gate (valid metrics, loadable `best.pt`/`last.pt`, required events, valid manifest, executor logs); otherwise the run ends FAILED stage `integrity`. Failed/cancelled runs with a valid checkpoint carry `resumable` in `metrics.json`/`manifest.json` and show a "Fortsetzbar" banner in the run detail; nothing resumes automatically. `npm run verify:integrity`.

- 2026-10-06 — **Property, random-graph and fuzz tests (Phase 64–66)**: `npm run test:property` (200 seeded random valid graphs over 6 families → accepted, code runs forward+backward, shapes/param counts equal an independent oracle, 40 via the real sidecar vs forward hooks) and `npm run test:fuzz` (948 invalid mutants/run: rejected cleanly, store unchanged, semantic errors come back structured and classified `invalid`). Found and fixed: `validateGraphState` stack overflow on a 10 000-node chain (now iterative).

- 2026-10-06 — **Run manifest + git state + canonical hashes (Phase 58–63)**: every run writes `manifest.json` (`spinoml.run-manifest/1`: experiment id, git commit/branch/dirty files/untracked count, hashes incl. `config_identity_sha256`, seed, software incl. torch_geometric + OS, hardware, notes) atomically at start and at every terminal state; `reproducible_from_git` is true only for a known commit with no modified tracked file. See `docs/engineering/REPRODUCIBILITY.md`. `npm run verify:manifest` (76).
- 2026-10-06 — **Silent-exception audit, Node sidecar (Phase 50)**: guard scans `sidecar-llm/*.mjs` too; 10 hidden failures became explicit tool errors (notes/list/read probes, run.json/metrics.json readers, `slurm_status`, model listing, mcp-bridge).

- 2026-10-06 — **Silent-exception audit, frontend (Phase 50)**: 24 hidden failures made visible (`.py`-twin write/rename/remove errors banner `pyTwinError`, malformed run.json/events in run detail/compare, canvas bind failures, a failed remote capability probe no longer shows the local backend and blocks the training start); the other 56 swallows are documented in `docs/engineering/SILENT_EXCEPTIONS.md`. `npm run verify:silent-catch`, `npm run verify:silent-fixes` (33).
- 2026-10-06 — **Reference experiments + generated-vs-hand-written equivalence (Phase 53–57)**: `examples/reference-experiments/{mlp,cnn,multi-input}/` (graph fixtures + README: how to run). `npm run verify:reference` (generated model == hand-written PyTorch: params, forward f32/f64, loss, gradients, negative control), `npm run verify:reference-train` (real trainer: complete artifact set, checkpoint↔model, same-seed rerun identical). CUDA is reported as SKIPPED where unavailable.

- 2026-10-06 — **Unsafe deserialization closed (Phase 47)**: every `torch.load` goes through `safe_load.py` (`weights_only=True` + PyG/numpy allow-list, `UNSAFE_PICKLE` errors, `SPINOML_ALLOW_UNSAFE_PICKLE` escape hatch recorded per run); trainer carries a byte-identical embedded copy. `npm run test:safe-load` (85, real RCE attempts).
- 2026-10-06 — **Filesystem scope (Phase 45/46)**: symlink-aware realpath containment for the torch sidecar (`scope.py`: roots, symlink targets, scope.json, `/health.scope`, explicit 403 codes; manifest/table-derived paths included) and the Node sidecar (`path-scope.mjs`); remote HPC sidecar launched with `SPINOML_ALLOWED_ROOTS`. Local default is the visible `unconfigured-open` mode until Rust writes `scope.json`. `npm run test:scope` (88), `npm run verify:paths` (48).

- 2026-10-06 — **Command/argument injection hardening (Phase 44)**: `run_script` args quoted via a real tokenizer + `./`-prefixed targets (`sidecar-llm/shell-safety.mjs`); torch `/run_script` relpath whitelist (removes the `bash -lc <filename>` branch); `/deps/*` pip-spec policy (`sidecar-torch/deps_policy.py`); `download_to_datasets` SSRF policy + DNS-checking `safeFetch`; ssh `validate_alias` rejects leading `-`, `--` before every ssh target, `remote_sidecar` alias now validated (Rust edits compiled; `cargo test` passes). New checks: `npm run verify:command-injection` (162), `npm run test:deps-policy` (34), `npm run test:run-script` (125), extra rejects in `verify:sidecar`. See §8c.
- 2026-10-06 — **Generated-Python safety + code-trust gate (Phase 43)**: shared `src/codegen/pyLiteral.ts` makes every interpolated value inert (`npm run verify:codegen-security`, 5362 adversarial cases checked with Python `ast`/`tokenize`; 620 failed before); LLM-written or imported `Custom`/`init_args`/`DataOp`/`CustomScript` code no longer runs without an explicit user decision — content-addressed trust store, approval dialog, `status:'untrusted'`, run.json `snapshot.code_trust`. `npm run verify:code-trust` (110), `npm run verify:code-trust-wiring` (78). See §8b/§8c.

- 2026-10-06 — **Code trust UI (Phase 43, user-facing half)**: the execution gate gets its human surface — `src/trust/ApproveCodeDialog.tsx` (per-blob label/path/line-count + first 40 lines as plain text, per-blob "Freigeben" and "Alle freigeben" → `trust.approve(hash, 'user-approval')`, Esc/"Abbrechen" default focus) mounted once in `App.tsx` and opened via `src/trust/useApproveDialog.ts` from the amber `InferenceBadge`, the blocked `NewRunModal` launch (`UntrustedCodeError` or `listUntrusted` pre-check), DataOp "Vorverarbeitung ausführen" and the data-canvas "Pipeline ausführen"/"via Chat". The only other approvers are the Inspector's human-edit (`human-edit`, incl. `init_args`) + eject (`eject`) and built-in template insertion (`template`); chat/load/restore still never approve. `scripts/verify-code-trust.ts` gains `ALLOWED_APPROVERS` + explicit-origin-literal, no-`dangerouslySetInnerHTML`, and `'user-approval'`-origin-only-in-dialog static checks.

- 2026-09-22 — **Training immutability (Phase 42)**: once `startRun` is called, the run is fully frozen by `snapshot` sha256 (`modelContent`+`modelPy` via `generateFromSnapshot`+`buildRunSnapshot`) and `training.rs`/`ssh.rs` write `run.json`/`model.spinoml`/`model.py`/`train.py` atomically into `experiments/runs/<id>/` and launch detached (`setsid`/`sbatch`, survives app close); the run never reads live `GraphStore`. `train.py` `_verify_snapshot` re-hashes and `fail`s loudly on drift; later UI edits bump `GraphStore.revision` but cannot mutate the running experiment. New `npm run verify:immutability` (27 checks) + prior `verify:traingen` snapshot section lock immutability.

- 2026-09-22 — **Concurrent operations (Phase 41)**: `workspace/store.ts` now guards `saveActive` with `saveSeq`+`revAtStart` and corrects `dirty` if an edit raced the write (no silent loss); `inference` is debounced+`runCounter`+`graphRev`+`AbortController` (stale shape dropped); `datasets` has `inspectSeq`/`statsSeq`/`smokeSeq`+`smoke` `graphRev`+`refreshSeq`; `training` start is snapshot-frozen (drift fails loudly) and `chat` dispatch goes through `validateGuard`. All 5 concurrent pairs are allowed but stales are dropped. New `npm run verify:concurrent` (44 checks) proves Save+edit, Edit+inference, LLM+user, Training+edit, Dataset reload+inspect guards.

- 2026-09-22 — **Graph revision system (Phase 40)**: `GraphStore.revision: number` (0→monotonic) bumped on every structural commit (add/update/replace/delete/connect/load/reset; position drags don't bump) and checked by every async consumer: `inference/store.ts:43` (`graphRev`+`runCounter`), `datasets/store.ts:14` (per-dataset `inspectSeq`/`statsSeq`/`smokeSeq` + `smoke` `graphRev`), `training/store.ts:121` (`refreshSeq` latest-wins). A stale response whose graph moved since the request is dropped — no overwriting current shapes/inspects/lists. New `npm run verify:graph-revision` (32 checks) proves bumps, drag not bump, all guards, and the prior `latestWinsGuard` reference.

- 2026-09-22 — **Frontend error states (Phase 39)**: Loading/Success/Error/Unavailable are now distinct — `inference` `idle`/`inferring`/`ok`/`error`/`offline`, `training` `listLoading`/`listError`+`unknown`, `datasets` `Cached<T>`, `project` `remote-missing`, `chat` `streaming`/`done`/`error`+`online`. Timeout/Cancelled remain string-typed (known gap → `LIMITATIONS.md`): SSH timeout → `listError`+`unknown`, fetch abort → `idle` not hang, but no `timeout`/`cancelled` enum. Never false Success. New `npm run verify:frontend-errors` (42 checks) + prior `verify:ui-state` prove Loading/Success/Error/Unavailable distinct.

- 2026-09-22 — **UI state must not lie (Phase 38)**: `training/store.ts` now degrades stale `running`/`queued` to `unknown`+`alive:false` on `listError` (SSH loss) and clears `pollTimer`; `App.tsx` `ProjectHeader` shows violet `ssh · alias` only while connected, rose `ssh · alias — nicht verbunden` when `listError` or `status.kind==='error'`; `inference/store.ts` `throw e` → `offline`/`error` + `clearShapes`, so `inferring…` cannot hang; save paths remain truthful (await `fs.write` before `dirty:false`, `saved` only on success). New `npm run verify:ui-state` (33 checks) proves training/SSH/SLURM/inference/save truthfulness.

- 2026-09-22 — **Remote job recovery (Phase 37)**: closing SpinoML and reopening never trusts UI cache — `useTrainingStore.runs` is memory-only, `App.tsx:257` blanks `runs:[]` on every `workspaceRoot` reconnect, `ExperimentsExplorer:32` re-refreshes on `currentId` change, and `training.list()` then re-queries the actual remote `experiments/runs/<id>/{status,pid,run.json,metrics.json,events.jsonl}` plus live `squeue`/`sacct`/`kill -0` per run and reconciles (`reconcile_status`/`reconcile_slurm_status` → stale `running` without alive → `failed`). Detached `setsid`/`nohup setsid` (direct) and `sbatch` `slurm:<jid>` survive app/SSh close; the atomic `mkdir` claim (Phase 33) makes a lost-response retry after restart idempotent (`MLF_ALREADY_LAUNCHED` vs `MLF_EXISTS_INCOMPLETE`). `connections/store.ts` persists only `alias/root` and `training/backend.ts` dispatches on live `getCurrentConnection()`, never cached run state. New `npm run verify:recovery` (38 checks) + Rust reconciliation reuse proves close→restart→reconnect recovers the actual remote job.

- 2026-09-22 — **SLURM reliability (Phase 36)**: every scheduler state — submit success (pid `slurm:<jid>` + `train.sbatch` frozen), submit failure (`sbatch failed:`), PENDING→queued, RUNNING→running, COMPLETED→done, FAILED/TIMEOUT/OOM/NODE_FAIL→failed, CANCELLED (`CANCELLED by 123` stripped)→cancelled, UNKNOWN/communication loss→failed/unknown — is mapped explicitly via `training.rs:reconcile_slurm_status` (squeue dominates while queued, sacct after, terminal file wins) and `ssh.rs:build_sbatch` (correct `#SBATCH` for partition/time/mem/cpus/gres/account/qos/modules/pre_run_script, job-name sanitized). `ssh_stop_training_run` uses `scancel <jid>` for SLURM vs `kill -TERM -<pid>` for direct. New `npm run verify:slurm` (33 checks) + Rust tests `slurm_tests` (4) + 9 reconciliation cases prove submit, all 8 scheduler states, job-ID persistence, and comm-loss fallback.

- 2026-09-22 — **Credential safety (Phase 35)**: passwords, private keys, API tokens, OAuth secrets, and SSH credentials are confirmed absent from all experiment artifacts, logs, and config files. SpinoML delegates remote authentication entirely to system `~/.ssh/config` and `ssh-agent` without storing secrets in local storage. Runtime environment recording (`_env_info`) captures platform diagnostics without dumping `os.environ`. All SSH error outputs are sanitized via `sanitize_credentials` before being reported. New `npm run verify:credentials` proves leak-free artifacts, logs, error reporting, and training templates.

- 2026-09-22 — **SSH reliability (Phase 34)**: transport and remote failures now report distinct, explicit error causes: authentication/host-key mismatch, DNS host resolution failure, connection timeout, connection reset/loss, missing remote paths, permission denied, filesystem full, or SFTP failures. Multiplexed ControlMaster sockets ensure fast, keepalive-backed re-use and automatic re-establishment. New `npm run verify:ssh` and Rust unit tests verify error classification across all scenarios.

- 2026-09-22 — **Submission idempotency (Phase 33)**: remote SSH start now claims the run directory atomically. One concurrent client can submit; a retry after a lost response sees the durable PID and returns success without launching a duplicate. An incomplete prior attempt fails explicitly before launch. New `npm run verify:submission` covers first submit, retry before/after PID, and 16 concurrent claimers.

- 2026-09-22 — **Cancellation (Phase 32)**: SIGTERM/SIGINT now unwind through one terminal-shielded trainer path, rather than terminating the process mid-epoch. Cancellation produces exactly one `run.cancelled`, cancelled metrics, and a resumable checkpoint at the last completed epoch (epoch -1 during the first epoch, so resume starts at 0); a late signal cannot change done/failed. New `npm run verify:cancel` exercises cancel-before-start, startup, first-epoch signal, cooperative stop, double SIGTERM, and post-completion shielding.

- 2026-09-22 — **Event ordering (Phase 31)**: final-state protection moved to the
  events-read boundary. On-disk `events.jsonl` is already append-only + fsynced
  per line; the risk was a stale read snapshot landing over a newer one (esp.
  over ssh) or a trailing out-of-order line (EPOCH after FAILED). New pure module
  `src/training/events.ts` — `finalTerminal`/`truncateAtTerminal` let the first
  terminal event (run.done/failed/cancelled) decide the final state and drop
  everything after it; `latestWinsGuard` discards an older in-flight events read
  whose response arrives after a newer one (the Phase-10 inference guard applied
  to events). RunDetailModal + CompareModal now feed every `setEvents` through
  `parseFinalEvents` + the guard. New `npm run verify:events` proves parsing
  tolerance, EPOCH-after-FAILED truncation, first-terminal-wins over duplicate
  terminal writes, stale-read dropping and the modal flow (26 checks).

- 2026-09-22 — **Run state machine (Phase 30)**: the run `status` file is now a
  real state machine — `queued → running → done|failed|cancelled` with terminal
  states FINAL. `transition_status()` in `training_template.py` gates every
  trainer write (rejecting late `running`/`done` after a cancel or crash); local
  `stop_training_run` and remote `ssh_stop_training_run` only flip a
  non-terminal status to `cancelled`, so stopping a SUCCEEDED/FAILED run is a
  no-op. A cancel racing the finalisation window emits `run.cancelled` instead
  of `run.done` — `CANCELLED → SUCCEEDED` is impossible. New
  `npm run verify:states` proves the full transition table plus cancel-before-
  start, cancel-mid-training, CANCELLED→SUCCEEDED race, double cancellation and
  the post-done cancel guard (43 checks).
- 2026-09-14 — **Seed + determinism record (Phase 22)**: `train.py` now seeds
  every random source (Python `random`, NumPy, torch CPU + all CUDA devices),
  enforces `cudnn.deterministic` + `benchmark=False`, enables
  `use_deterministic_algorithms(warn_only=True)`, and gives DataLoaders a
  seeded `torch.Generator` + `worker_init_fn`. Seed state is documented in a
  `run.determinism` event; the CUDA `atomicAdd` caveat is recorded rather than
  silently claiming bit-level reproducibility.

- 2026-09-14 — **Scientific smoke test (Phase 23)**: new `npm run verify:smoke`
  script trains a tiny MLP (10→64→2, Linear-ReLU-Linear) on a synthetic 100-sample
  10-feature binary-classification dataset for 5 epochs and asserts: loss
  decreasing + finite, accuracy finite ∈ [0,1], checkpoint/best.pt exists,
  metrics.json present with correct fields, and `run.provenance` +
  `run.determinism` events emitted — a fast CI-compatible end-to-end sanity
  gate.

- 2026-09-14 — **Metric correctness tests (Phase 29)**: new `npm run verify:metrics`
  proves loss/metric aggregation is correctly batch-size-weighted: a frozen
  model on a 97-row dataset with batch 32 (last batch = 1) is compared against
  an independent per-sample Python reference over the identical split — epoch
  train loss, val loss and accuracy match within 1e-4, while naive
  mean-of-batch-means demonstrably differs (the test would catch an
  unweighted-averaging regression).

- 2026-09-14 — **Atomic + crash-safe checkpoints (Phases 27+28)**: `_atomic_save`
  writes checkpoints via tmp-file → fsync → `os.replace()` → dir-fsync, so a
  crash mid-save never truncates the previous valid checkpoint. verify:checkpoint
  now SIGKILLs a live trainer right after a checkpoint event and asserts every
  .pt on disk still loads (no .tmp leftovers) and a restart resumes cleanly;
  a deliberately corrupted .pt is rejected loudly (`run.failed` stage `resume`).

- 2026-09-14 — **Checkpoint correctness (Phase 26)**: checkpoints now preserve
  the full resumable state — model/optimizer/scheduler state, epoch, a new
  `global_step` counter (tracked across epochs, restored on resume), RNG
  streams (`torch` CPU + all CUDA devices, NumPy, Python stdlib) restored via
  `_restore_rng()` so a resumed run continues from the saved random streams,
  and the frozen experiment `config`. A cancelled run saves `last.pt` for the
  last completed epoch, making stop → load → resume a first-class flow. New
  `npm run verify:checkpoint` covers train→save→resume→cancel (31 checks).

- 2026-09-14 — **Numerical failure detection (Phase 25)**: the trainer monitors
  train loss (per batch), parameter gradients (before every optimizer step),
  and val loss/accuracy/metrics (after every evaluate pass, incl. eval-only)
  for NaN/inf. First non-finite value → `run.failed` with stage `numeric` and
  an explicit reason naming the offending value/parameter — a numerically
  unstable run is NEVER reported as success. verify:failures gained a
  NaN-mid-training model case asserting stage `numeric`.

- 2026-09-14 — **Training failure tests (Phase 24)**: new `npm run verify:failures`
  harness deliberately causes each listed failure mode against the REAL trainer
  and asserts FAILURE (exit≠0, `status`≠done, `run.failed` event), never
  SUCCESS and never stuck in RUNNING: invalid dataset, invalid model, invalid
  optimizer, invalid (negative) learning rate, missing output dir
  (checkpoints-as-file), unwritable output dir (chmod 555), NaN input, and
  mid-training SIGKILL. Also HARDENS the trainer: `load_tabular` now FAILS
  LOUDLY on NaN/inf in the feature matrix instead of silently `fillna(0.0)` —
  a NaN/inf dataset can never be trained on or produce believable metrics.

- 2026-09-14 — **Training environment record (Phase 21)**: `train.py` emits a
  `config.env` event at launch recording the runtime stack (python/torch/cuda/
  numpy versions, device + GPU/VRAM, compute dtype, CPU count, RAM, workspace
  git commit) and copies it into `metrics.json` at run end.

- 2026-09-14 — **Training snapshot (Phase 20)**: every `run.json` now carries an
  immutable `snapshot` section — sha256 of the frozen `model.spinoml` graph +
  generated `model.py` + the graph's DataOp preprocessing scripts. `train.py`
  re-verifies the run-dir copies against it (`run.snapshot` event) and refuses to
  start if any artifact drifted, so a running experiment provably never depends
  on mutable UI state.

- 2026-09-14 — **Split integrity + strategy guard (Phase 19)**: the Split node now
  records a `strategy` (`random`/`stratified`/`grouped`/`time-based`/`predefined`),
  frozen into `run.json` (`training.split_strategy`) and never silently changed —
  an unimplemented strategy makes `train.py` fail loudly instead of falling back to
  random. The trainer asserts zero train/val overlap (`split.integrity` event) and
  fingerprints the strategy into `run.provenance`.

- 2026-09-14 — **Dataset fingerprinting (Phase 18)**: every dataset `inspect` now
  returns a stable SHA-256 `fingerprint` (`content`/`structure`/`config+content`/
  `reference` modes, copy- & rename-stable), which is frozen into `run.json`
  (`dataset.fingerprint`) and re-checked by `train.py` before loading
  (`run.provenance` + `dataset.fingerprint.check` events). A run trained later on
  different bytes now fails loudly. `test:datasets` covers determinism,
  copy-stability, and content-change detection (124 checks).

- 2026-09-14 — **OpenCode as first-class LLM provider** (default): `kind:'opencode'` in
  the sidecar spawns `opencode run --format json` per chat turn; the model is exposed
  ONLY the `graph` MCP tool server (`sidecar-llm/mcp-bridge.mjs`, stdio → `POST
  /internal/mcp/<requestId>/list|call`) → same `execTool` validation gate as Claude.
  Built-in opencode tools (read/bash/write/…) disabled via inline config
  (`OPENCODE_CONFIG_CONTENT`). `GET /opencode/models` (cached). Frontend:
  `providerStore.ts` adds an `opencode` provider (first, default `opencode/big-pickle`),
  `ProviderSettings.tsx` model datalist fed from the live list, badge shows the model.
  Verify: `npm run verify:opencode`. Docs/CLAUDE.md updated.
- 2026-06-26 — **Duplicate edge-id fix** (`training/graph/store.ts`, `data/graph/store.ts`):
  edge ids were `e${edges.length+1}`, which collides after a delete-then-add (load
  e1..e10, delete one, add → e10 again) → duplicate React keys that destabilize
  React Flow (observed as a frozen webview). Added `freshEdgeId` (max existing
  `e<n>` + 1), used in both onConnect and connectNodes.
- 2026-06-24 — **Manifest preprocessing speedup** (`training_template.py`):
  `load_manifest_graphs` now memoizes each branch's result by raw cell value
  (in-memory dedup — collapses ESPF tokenization / `torch.load` / lookup from
  #rows to #unique values, 10-50× on dual-encoder sets like BindingDB) AND caches
  ESPF token ids to disk (`.graphcache/espf_<codebook>_<hash>.pt`, deterministic
  per codebook+max_len+value) so re-runs start fast. Emits periodic
  `dataset.preprocess {done,total,kept,skipped}` events so a long single-threaded
  preprocess no longer looks frozen (no event between run.start and dataset.loaded).
- 2026-06-24 — **SLURM run logs + hardware fix** (`RunDetailModal.tsx`, `ssh.rs`,
  `training.rs`): the Logs tab now reads `slurm-<jobid>.out/.err` for SLURM runs
  (job id from the `pid` file `slurm:<id>`) instead of the non-existent
  `stdout.log/stderr.log` — `is_readable()` whitelists those names. The Hardware tab
  passes the run id to `gpuStats`; `ssh_gpu_stats` probes the COMPUTE node via
  `srun --overlap --jobid=<id> nvidia-smi` (was hitting the login node, which has no
  GPU), falling back to the login-node probe when the alloc is gone.
- 2026-06-24 — **Run status filter**: the Experiments browser (`ExperimentsExplorer.tsx`)
  gained client-side status-filter chips (Alle / running / queued / done / failed /
  cancelled, with counts; only statuses present are shown). Also removed the redundant
  Auto-Modus info banner in the chat panel (the toggle button stays).
- 2026-06-24 — **GPU training fix**: `training_template.py` now moves the model and
  every batch onto `cuda` when available (new `to_device` helper; eval tensors return
  to CPU for summaries). Previously the run stayed on CPU even with `#SBATCH --gres=gpu:1`
  + a CUDA module loaded — `device_type` was computed but never applied. Records the
  real `device`/`gpu` in `metrics.json` + `model.built`. (Was filed as the "cosmetic"
  Bug #7 in the 2026-06-19 handoff; it was the actual GPU-placement bug.)
- 2026-06-23 — **External-validation fixes**: a manifest model ALWAYS uses the
  manifest-builder (not column-mapping) so it loads as a manifest (4 branch args,
  not a flat tensor); EvalRunModal gained **SLURM/local backend selection** (defaults
  to SLURM on a remote HPC); directory resources (table columns, side CSVs, sub-dirs)
  are shown for reference. `buildAdaptedManifest` now takes fs-derived `DirResources`.
- 2026-06-23 — **Dual-encoder external validation**: manifest `lookup` branch (JOIN a
  side table by key, e.g. protein sequence by uniprot), N-D `_pad_stack` (2-D
  embeddings → `[B,Lmax,D]`), `batch_len` handles mixed graph/tensor branches,
  per-output head selection, and a directory→adapted-manifest builder
  (`buildAdaptedManifest`) in EvalRunModal. Validates a manifest dual-encoder on a
  prepared directory (e.g. TDC BindingDB) using only the available output(s).
- 2026-06-23 — **Prepared dataset directories**: a folder with a primary table
  (`pairs.csv`/…) + side files (sequences/embeddings/`prep_card.json`) is recognized
  as tabular on its inner table; selectable for external validation. (sidecar
  `detect_kind`/`_table_path`, `load_tabular`; EvalRunModal includes dirs.) NEEDS
  remote-sidecar redeploy for a remote workspace.
- 2026-06-23 — **External validation + data adapter**: validate a trained run's
  checkpoint on a foreign/benchmark dataset (eval-only run via `training.start`,
  reusing the eval engine + UI). Auto/manual/hybrid column-mapping adapter
  (`adapter.ts`, `EvalRunModal.tsx`) fits the external dataset to the model's
  trained schema; Data-canvas escalation for complex transforms. Trained-class
  encoding preserved. See §6b.

- 2026-06-23 — **Fix (canvas edges)**: nodes re-measure handle bounds on any size
  change (ResizeObserver in `LayerNode.tsx`), so edges no longer render at stale
  positions when a node grows (summary/shape line/Explain preview).
- 2026-06-23 — **Fix (HPC state persistence)**: backgrounding/reloading a remote
  project no longer resets the chatbot or closes the open `.spinoml`. (a) Same-root
  re-bootstrap preserves `activeFileId` and `refresh()` doesn't blank to Welcome
  when already loaded (`project/store.ts`); (b) the open-file BINDING is persisted
  per workspace and restored after bootstrap — content comes from the autosave, so
  unsaved edits survive (`recentWorkspaces.ts`, `workspace/store.ts`); (c) chat
  history is persisted to localStorage and restored on reload, cleared only on an
  explicit project close (`chat/store.ts`).
- 2026-06-23 — **ESPF node**: `ESPF` input + manifest `kind:"espf"` (MolTrans
  substructure tokenizer, vendored codebook `sidecar-torch/espf/`, pure-Python BPE,
  interpretable labels in Explain). See `memory/espf-tokenization-node.md`.
- 2026-06-23 — **Sequence branches**: manifest `kind:"sequence"` (char/byte
  tokenizer) feeding the `Sequence` input node, end-to-end sample + train.
