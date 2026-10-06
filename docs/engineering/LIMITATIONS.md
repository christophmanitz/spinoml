# SpinoML — Known limitations

> Honest list of what the hardening phases did **not** (or could not) close.
> Written 2026-10-06 (Phase 43/44 batch); it describes actual behaviour, not intent.
> Each item names its source (a test, a file, or a TODO phase). When an item is
> fixed, move it to the changelog in `docs/FEATURES.md` and delete it here.

## 1. Verification environment (this machine)

- **Rust toolchain lives in the conda env, not on the base PATH.** `cargo 1.96.0` / `rustc 1.96.0` are installed in
  `mlforge-dev`; run Rust checks as `conda run --no-capture-output -n mlforge-dev cargo check` / `cargo test` from
  `src-tauri/` (a bare shell has no `cargo`, which fooled earlier sessions and `BASELINE.md`). Verified 2026-10-06:
  `cargo check` clean, `cargo test` 22 passed (incl. the Phase 44 `alias_validation_tests`). Not covered: `npm run
  tauri dev`/packaging was not exercised in this session.
- `verify-*`/`test-*` scripts that need Python must run inside the conda env
  (`conda run -n mlforge-dev …`); the env is still named `mlforge-dev`
  (rename leftover, see memory). Plain `python` is not on PATH in a bare shell.
- Lint is not a green gate: at HEAD before this batch `npm run lint` reports
  82 problems (78 errors / 4 warnings); the working tree additionally counts
  build artifacts under `src-tauri/target/` (R017).

## 2. Security — open items (as of this batch)

- **Sidecars are unauthenticated and send `Access-Control-Allow-Origin: *`**
  (torch `sidecar-torch/main.py` `_cors`, LLM `sidecar-llm/main.mjs` `cors`).
  Any web page in the user's browser, or any local process, can call
  `/infer`, `/run_script`, `/deps/install`, `/respond`. Consequently:
  - the **code-trust gate is a frontend gate only** — it stops the app from
    sending unapproved code, but does not stop a direct HTTP call to `/infer`;
  - `autoMode: true` in a chat request body still disables the shell-run
    confirmation (R013, R014 — planned in Phase 77/78).
- **Unsafe unpickling (fixed in Phase 47, residual risks)**: all `.pt` loads use
  `weights_only=True` plus an allow-list (`sidecar-torch/safe_load.py`). Residual:
  `SPINOML_ALLOW_UNSAFE_PICKLE=1` re-enables full unpickling for the whole process
  (loud warning, recorded as `unsafe_pickle` in the run's `config.env`/`provenance`);
  the allow-list contains PyG and numpy array types, so other custom classes in a
  `.pt` are refused until the file is re-saved as tensors/dicts/PyG `Data`; on
  torch < 2.4 there is no allow-list API (only plain tensors/dicts load).
- **Torch sidecar path scoping exists but is not enforcing on the local app yet**
  (Phase 45/46, R016): `scope.py` enforces realpath containment (roots +
  user-listed symlink targets) for every request path and every path derived from
  file content, *when a root is configured* — `SPINOML_ALLOWED_ROOTS`,
  `~/.cache/spinoml/scope.json`, or (remote HPC) the launcher's export. The local
  managed sidecar is started before a workspace exists and Rust cannot tell it
  the root (and no Rust writer exists yet), so without configuration it runs in the
  visible mode `unconfigured-open` (stderr warning, `/health.scope.mode`).
  `SPINOML_REQUIRE_SCOPE=1` fails closed. Remote workspaces that symlink data out
  of the project root (e.g. `datasets -> /work2/...`) must export
  `SPINOML_SYMLINK_TARGETS=/work2/...` in `<root>/.spinoml/env.sh`, otherwise the
  request is refused with `PATH_SYMLINK_OUTSIDE` and a one-line fix.
- **Scoping does not protect the exec endpoints**: `/infer`, `/dataset/smoke` and
  `/activations` execute model `code`, which can open any file the process can and
  returns exception text. Only the sidecar token (Phase 77/78) closes that.
- **Rust `resolve()` is purely lexical** (no symlink resolution) and the Node
  sidecar's remote (ssh) workspaces only get lexical checks; the local Node path is
  symlink-aware (`path-scope.mjs`).
- Minor: `scope.py` rejects a path that traverses the *same* symlink twice (e.g.
  `link -> .`, `link/link/x`) as a loop (fail-closed); the Node resolver allows it.
- **SSRF policy of `download_to_datasets` is best-effort**:
  - local path: literal-IP blocklist + DNS resolution check on every hop with
    manual redirects; a DNS-rebinding race between our lookup and the actual
    connection remains;
  - remote path (curl on the login node): only the literal-IP/hostname blocklist
    plus `--proto`/`--max-redirs`; a hostname that resolves to a private address
    on the remote network is **not** detected;
  - the tool has no confirmation dialog.
- **Expression nodes are code-equivalent**: data nodes `FilterRows.query` and
  `ComputeColumn.expr` are passed to pandas `query`/`eval`; they are emitted
  as inert string literals but are evaluated by pandas. They are not part of
  the code-trust gate.
- `opencode` is invoked with the prompt as a positional argument; a prompt that
  starts with `-` is not separated by `--` (low risk: the prompt is the user's
  own message).

## 3. Code-trust gate (Phase 43) — what it does not protect against

- Trust is **per installation** (browser `localStorage`, key `spinoml.codeTrust.v1`,
  cap 5000 records), not per workspace and not synced.
- **Editing counts as review.** Typing or pasting into a code field approves the
  settled text. A user who pastes malicious code, or adds one character to
  LLM-written code they did not read, approves it. The dialog is the only place
  that shows the code before approving.
- `train.py` does **not** re-check `code_trust`; `run.json` is a provenance
  record written by the same app, not an authentication boundary. The decision
  is recorded (`snapshot.code_trust`: node, path, kind, sha256, origin,
  approved_at) but not enforced on the cluster.
- The Inspector code editor is uncontrolled (Monaco `defaultValue`): an
  external change (e.g. by the chatbot) is not shown until the node is
  re-selected. After the Phase 43 fix a stale unedited buffer is no longer
  written back, but the displayed text can be out of date.
- `init_args` of a `Custom` node is Python by design (quotes allowed); it is
  gated as its own trust blob (`custom-init-args`), not sanitised.

## 4. Frontend state model (from Phases 39/40)

- Timeout and Cancelled are not separate enum states: an inference timeout is
  reported as `offline`/`error` text, a chat `AbortError` ends as `done`. They
  are never shown as Success, but the UI cannot tell them apart (Phase 39).
- Only the architecture graph store has a revision counter; training-graph and
  data-graph stores do not (Phase 40).

## 5. Reproducibility

- Bit-level GPU reproducibility is **not** claimed: CUDA `atomicAdd` based
  reductions are nondeterministic even with the seeds and deterministic flags set
  (Phase 22). CPU runs are reproducible for equal seed + software stack + inputs.
- Dataset fingerprints are absent for remote workspaces before the remote
  inspector exists (Phase 12b); `reference` mode (pyg/huggingface) pins only the
  reference file, not the downloaded data (Phase 18).
- Only the `random` split strategy is implemented; the other strategies fail
  loudly (Phase 19).

## 6. Not yet audited

Phases of `TODO.md` that have no "implemented" note are not done. In particular:
Rust `unwrap`/`panic` audit (48), the Python and Rust halves of the silent-exception
audit (50; TypeScript and Node are done), resource leak and long-run tests (51/52),
CI (69),
retries/timeouts (71/72), the
sidecar token + localhost review (77/78), dependency audit (79) and the final
reliability report (90).

**CUDA is unverified.** `torch 2.12.0+cpu` is installed here: every CUDA branch of
`verify:reference`/`verify:reference-train` prints `SKIPPED  CUDA` (Phase 56). The
generated-model equivalence and the same-seed reproducibility were measured on CPU only;
no GPU claim is made (R044).

**Reproducibility limits (Phase 59–63).** `reproducible_from_git` needs `git` with `-C` (≥ 1.8.5) on the machine that
runs `train.py`; older git (some login nodes) yields an explicit reason and `false`. Node/Rust versions and the
SpinoML app version are not recorded in the manifest. See `REPRODUCIBILITY.md` §5.

**Property/fuzz coverage (Phase 64–66).** The random-graph generator covers MLP, CNN (Conv2d/MaxPool2d/BatchNorm2d), residual,
branch+Concat, multi-input and sequence-embedding families only. Not generated yet: attention, recurrent, GNN/PyG, Conv1d/3d,
other pooling and normalisation layers, Subgraph/Custom nodes. Shape agreement for those rests on the hand-written reference tests
(`verify:reference`) and the existing sidecar tests, not on property tests.

**Run integrity (Phase 73/74).** Checkpoints larger than 256 MB are not load-verified by the gate (header + size check, a note)
and are therefore reported as not resumable. A manifest that cannot be written fails the run (it is required metadata).
`stdout.log`/`stderr.log` are only required when the executor launched the run (`pid` file present).
