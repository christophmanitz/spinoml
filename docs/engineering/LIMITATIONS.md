# SpinoML — Known limitations

> Honest list of what the hardening phases did **not** (or could not) close.
> Written 2026-10-06 (Phase 43/44 batch); it describes actual behaviour, not intent.
> Each item names its source (a test, a file, or a TODO phase). When an item is
> fixed, move it to the changelog in `docs/FEATURES.md` and delete it here.

## 1. Verification environment (this machine)

- **No Rust toolchain.** `cargo`/`rustc` are not installed (`which cargo` empty,
  no `~/.rustup`). `~/.cargo/registry` and `src-tauri/target/release` carry
  2026-09-22 timestamps, so a toolchain existed when Phases 30–37 were written,
  but `cargo check` / `cargo test` cannot be re-run today.
  - **The Rust edits of 2026-10-06 are UNCOMPILED and their tests unexecuted**:
    `ssh.rs` (`validate_alias` is now `pub(crate)`, rejects a leading `-`, new
    `alias_validation_tests`; `--` before the target), `remote_sidecar.rs`
    (`ensure_remote_sidecar` now validates the alias; `--`), `pty.rs` (uses
    `validate_alias`; `--`), `training.rs` (`shell_quote` on `SPINOML_PYTHON`).
    They were reviewed by hand only. Run `cd src-tauri && cargo check && cargo test`
    before releasing. (R023/R042)
  - Earlier "cargo ✓" statements in `TEST_MATRIX.md` date from the toolchain era
    and are not reproducible here.
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
  the root (and cannot be compiled here), so without configuration it runs in the
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

Phases 45–82 of `TODO.md` that have no "implemented" note are not done. In
particular Rust `unwrap`/`panic` audit (48), silent-exception audit (50), resource
leak and long-run tests (51/52), reference experiments and hand-written PyTorch
comparison (53/54), run manifest + git dirty state (59/60), CI (69), dependency
audit (79) and the final reliability report (90).
