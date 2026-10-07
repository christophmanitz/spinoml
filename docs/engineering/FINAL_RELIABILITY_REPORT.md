# SpinoML — Final reliability report (TODO.md §90)

Date of the final run: 2026-10-07. Every statement below is backed by a suite, a document or a commit named in the
same row; where nothing backs it, it is marked as not verified.

## 90.1 Executive summary

**CONDITIONAL**

SpinoML is **not** declared "Production Ready for Scientific Work". On the machine this work was done on, every CRITICAL
requirement that can be evaluated there passes with a named, mutation-checked suite (`RELEASE_GATE.md` evaluates the
84-item checklist row by row and the twelve §85 stop conditions one by one). Six applicable requirements could not be
verified here; the verdict stays CONDITIONAL until they are:

1. **CUDA** — torch here is `2.12.0+cpu`; every CUDA branch reports `SKIPPED  CUDA` and bit-level reproducibility on a GPU is not claimed.
2. **A real cluster** — SSH/SLURM handling, remote job reconciliation and the remote-sidecar bootstrap were verified against local fakes only (`remote-live` is BLOCKED).
3. **The real Tauri window** — the strict CSP, the sidecar authentication wiring and the scope-file writer were verified in a real Chromium and with unit/real-process tests, never in the WebKitGTK window. Fallback if it misbehaves: `app.security.csp: null`.
4. **CI on GitHub** — `.github/workflows/ci.yml` has never run.
5. **A long soak** — the CI soak is 90 s; an hour (`npm run test:soak -- --seconds 3600`) is needed to say anything about slow leaks.
6. **The Claude subscription provider** and the real opencode CLI are not driven by any suite.

What the hardening changed, in one paragraph: the application now fails closed where it used to lie. A training run
reaches `done` only through an integrity gate (valid metrics, loadable checkpoints, required events, a valid manifest);
generated code is pinned byte for byte by 40 golden cases and compared numerically with hand-written PyTorch; the two
local sidecars authenticate every request (token, Host, Origin) and scope every file access; LLM tool calls are validated
against the frontend registry before any state changes; unhandled errors reach the UI instead of the console; and 89
defects (29 CRITICAL) were found and fixed — 17 of them only because an independent review caught what green worker tests
had encoded or missed (`BUGS_FIXED.md`).

## 90.2 Environment

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.5 LTS, Linux 7.0.0-34-generic x86-64 |
| CPU | Intel Core Ultra 7 255H, 16 threads |
| RAM | 30 GiB |
| GPU | none (`nvidia-smi` absent) |
| CUDA | none (`torch.version.cuda = None`, `cuda available = False`) |
| Python | 3.12.13 (conda env `mlforge-dev`) |
| PyTorch | 2.12.0+cpu; torch_geometric 2.8.0, numpy 2.5.3, pandas 3.0.2 |
| Node | v22.23.2 in an interactive shell; **v20.20.2** inside the conda env that the suite runner uses (Node-22-only APIs fail only there — learned the hard way) |
| npm | 10.9.8 |
| Rust | rustc / cargo 1.96.0 (inside the conda env, not on the bare PATH) |
| Git commit | the commit that contains this file (`git log -1`); the suite results below were produced on the tree of the immediately preceding commit plus the documentation files |

## 90.3 Test results

Source: `npm run ci` (the registry in `scripts/suites.ts`; every suite has a hard timeout, a scrubbed environment, a
private `TMPDIR` and a post-suite leak sweep). `FAIL` for the lint step cannot occur at baseline 0 any more.

Final run: **PASS 69 · SKIPPED 4 · FAIL 0 · TIMEOUT 0 · BLOCKED 3**, total 961 s, lint **0 problems**; started 2026-10-07T03:39:24.731Z; git {"commit": "68b8a282a08a4b918b2426863c8536bd5ddc030e", "dirty": true}.

SKIPPED = the suite ran but skipped a branch (CUDA, an end-to-end part that needs a running torch sidecar) and says so; BLOCKED = the required capability is absent (no real SLURM/ssh host, no CUDA, no live LLM key) — these are NOT passes.

| Test | Result | Duration | Notes |
|---|---|---:|---|
| Lint | PASS | — | 0 problems (baseline 0: 0e/0w), no regression |
| Build | PASS | 14 s | `tsc -b && vite build`; test scripts are type-checked too |
| Codegen | PASS | 34 s | 13 graphs executed; 40 byte-exact golden cases; 5362 hostile payloads; determinism |
| Sidecar | PASS (1 SKIPPED) | 258 s | request/response + activations, crash recovery, ports/signals/orphans, leak bounds, 90 s soak (GPU memory SKIPPED) |
| Training generation | PASS | 25 s | training codegen compile + multitask + eval-only |
| Rust | PASS | 5 s | 88 tests (ssh/slurm/training/auth/scope/pty/remote script) |
| Unit | PASS | 23 s | stores, policies, guards (test:verifier SKIPPED its e2e part: no torch sidecar running) |
| Integration | PASS | 28 s | concurrent ops, stale responses, UI truthfulness, real-Chromium app + CSP smoke |
| Scientific smoke | PASS (2 SKIPPED) | 332 s | real trainer runs; `verify:reference*` skip their CUDA branch (torch is CPU-only here) |
| Persistence | PASS | 3 s | round trips, torn/corrupt files, hostile persisted state |
| Recovery | PASS | 28 s | fake ssh / SLURM only — `remote-live` is BLOCKED |
| Security | PASS | 141 s | token/Host/Origin vs real processes, path scope, safe unpickling, LLM gate, MCP bridge, injection |
| Fuzz/property | PASS | 13 s | 200 random graphs vs an oracle; 948 invalid mutants; 8284 validation-parity cases |

### Every suite of the final run

| Suite | Category | Status | Duration | Note |
|---|---|---|---:|---|
| `test:graphstore` | unit | PASS | 1 s |  |
| `test:persistence` | unit | PASS | 1 s |  |
| `test:determinism` | unit | PASS | 1 s |  |
| `test:verifier` | unit | SKIPPED | 3 s | SKIPPED: valid model → status valid (torch sidecar offline); SKIPPED: broken model → status invalid (torch sidecar offline) |
| `test:races` | unit | PASS | 3 s |  |
| `test:deps-policy` | unit | PASS | 0 s |  |
| `test:run-script` | unit | PASS | 2 s |  |
| `test:llm-validation-parity` | contract | PASS | 1 s |  |
| `test:llm-safety` | contract | PASS | 18 s |  |
| `verify:command-injection` | unit | PASS | 1 s |  |
| `verify:paths` | unit | PASS | 1 s |  |
| `verify:code-trust` | unit | PASS | 1 s |  |
| `verify:silent-catch` | unit | PASS | 2 s |  |
| `verify:silent-fixes` | unit | PASS | 1 s |  |
| `verify:sidecar` | contract | PASS | 4 s |  |
| `test:safe-load` | contract | PASS | 4 s |  |
| `test:scope` | contract | PASS | 8 s |  |
| `test:sidecar-auth-torch` | contract | PASS | 6 s |  |
| `test:sidecar-auth-llm` | contract | PASS | 66 s |  |
| `test:sidecar-auth-frontend` | contract | PASS | 1 s |  |
| `test:codegen-golden` | integration | PASS | 7 s |  |
| `verify:silent-except-py` | unit | PASS | 1 s |  |
| `test:silent-except-py` | unit | PASS | 0 s |  |
| `test:process-lifecycle` | contract | PASS | 104 s |  |
| `test:webview-csp` | integration | PASS | 14 s |  |
| `verify:ts-safety` | unit | PASS | 2 s |  |
| `test:ts-safety-selftest` | unit | PASS | 2 s |  |
| `test:ts-safety` | unit | PASS | 4 s |  |
| `verify:doc-refs` | unit | PASS | 1 s |  |
| `test:resource-leaks` | integration | PASS | 34 s |  |
| `test:soak` | integration | SKIPPED | 107 s | SKIPPED  CUDA — no GPU memory sampling on this host (torch.cuda is never probed); SKIPPED  UI/Rust — saves and metric polling are not exerci |
| `test:history-store` | unit | PASS | 1 s |  |
| `test:workspace-store` | unit | PASS | 1 s |  |
| `test:llm-providers` | contract | PASS | 16 s |  |
| `verify:rust-panics` | unit | PASS | 1 s |  |
| `test:rust-panics-selftest` | unit | PASS | 1 s |  |
| `verify:sidecar-fetch` | unit | PASS | 1 s |  |
| `verify:remote-deploy-files` | unit | PASS | 1 s |  |
| `test:opencode-lifecycle` | contract | PASS | 20 s |  |
| `verify:ssh` | contract | PASS | 6 s |  |
| `verify:slurm` | contract | PASS | 2 s |  |
| `verify:credentials` | contract | PASS | 1 s |  |
| `verify:submission` | contract | PASS | 1 s |  |
| `verify:recovery` | contract | PASS | 1 s |  |
| `verify:states` | contract | PASS | 16 s |  |
| `verify:events` | contract | PASS | 1 s |  |
| `verify:codegen` | integration | PASS | 23 s |  |
| `verify:codegen-security` | integration | PASS | 3 s |  |
| `verify:traingen` | integration | PASS | 25 s |  |
| `test:robustness` | integration | PASS | 9 s |  |
| `test:datasets` | integration | PASS | 4 s |  |
| `verify:concurrent` | integration | PASS | 1 s |  |
| `verify:graph-revision` | integration | PASS | 1 s |  |
| `verify:ui-state` | integration | PASS | 1 s |  |
| `verify:frontend-errors` | integration | PASS | 1 s |  |
| `verify:immutability` | integration | PASS | 1 s |  |
| `verify:code-trust-wiring` | integration | PASS | 5 s |  |
| `verify:smoke` | e2e | PASS | 5 s |  |
| `verify:failures` | e2e | PASS | 29 s |  |
| `verify:checkpoint` | e2e | PASS | 43 s |  |
| `verify:cancel` | e2e | PASS | 26 s |  |
| `verify:metrics` | e2e | PASS | 7 s |  |
| `verify:manifest` | e2e | PASS | 51 s |  |
| `verify:integrity` | e2e | PASS | 111 s |  |
| `verify:reference` | scientific | SKIPPED | 4 s | SKIPPED  CUDA — torch.cuda.is_available() is False |
| `verify:reference-train` | scientific | SKIPPED | 56 s | SKIPPED  CUDA — torch.cuda.is_available() is False |
| `test:property` | scientific | PASS | 8 s |  |
| `test:fuzz` | scientific | PASS | 4 s |  |
| `remote-live` | remote | BLOCKED | 0 s | ssh-host: set SPINOML_REMOTE_TESTS=1 to enable live ssh-host suites; slurm: set SPINOML_REMOTE_TESTS=1 to enable live SLURM suites |
| `hardware-cuda` | hardware | BLOCKED | 0 s | cuda: no torch.cuda.is_available() == True probe possible |
| `build` | infrastructure | PASS | 10 s |  |
| `typecheck-scripts` | infrastructure | PASS | 4 s |  |
| `lint` | infrastructure | PASS | 16 s |  |
| `cargo-check` | infrastructure | PASS | 2 s |  |
| `cargo-test` | infrastructure | PASS | 1 s |  |
| `verify-opencode` | infrastructure | BLOCKED | 0 s | llm-key: set SPINOML_LIVE_LLM=1 to exercise the LLM path |

## 90.4 Bugs fixed

The full catalogue — **89 defects**, one row each with ID, who found it, description, root cause, fix, regression test,
severity and fixing commit — is `docs/engineering/BUGS_FIXED.md`; it is part of this report. Summary:

- By severity: **CRITICAL 29, HIGH 37, MEDIUM 21, LOW 2** (43 severities are the catalogue's own reading of the TODO §86
  scale and are marked `*`; the rest come from `RISK_REGISTER.md`).
- By who found it: audit 44, test 23, **review of worker output 17**, manual probe 5.
- **9 rows have no recorded regression test** — named in the catalogue as weaknesses, not hidden.
- Defects found in the last stretch (after the first draft of this report's catalogue): the cross-document undo bug
  (R060, CRITICAL: the first undo after opening a file restored the previous file's graph into it and autosave wrote it
  into the file), undo history aliasing live params, corrupt persisted workspace state that hung the tab, an unknown LLM
  `kind` silently falling back to the Claude subscription path, and unvalidated opencode model names.

## 90.5 Remaining risks

Nothing is hidden here; the detail and the evidence are in `LIMITATIONS.md` (§1–§7) and `RISK_REGISTER.md` (OPEN rows R021, R044).

**Verification gaps (the CONDITIONAL reasons)**
- No CUDA evidence (R044): results on a GPU may differ, bit-level reproducibility on CUDA is not claimed.
- No real cluster: SSH/SLURM/remote-sidecar behaviour is verified against fakes; a site quirk (RHEL `systemd-logind`, shell, quotas, `git` version) can still break it. Cluster scratch paths are site-specific (R021).
- The Tauri/WebKitGTK window was never run: CSP, auth and scope wiring are verified elsewhere only.
- `ci.yml` has never run on GitHub; `test:verifier` skips its end-to-end part without a torch sidecar; `verify:reference*` skip CUDA.
- No hour-scale soak; GPU memory and the memory of the Rust shell/webview are not measured.

**Security (documented limits, none CRITICAL)**
- The webview is trusted: a strict CSP is shipped, but a bug in the app's own HTML injection would still run with Tauri IPC. A same-user process can read the sidecar token. Browser-dev mode is tokenless (Host/Origin still enforced). No token rotation, no TLS (loopback / ssh tunnel only).
- `/infer`, dataset smoke and activations execute user-approved model code with the sidecar's file access; scope protects the file endpoints, not code that is approved by design. The code-trust gate is a frontend gate.
- Mode `unconfigured-open` until the first workspace is chosen (`SPINOML_REQUIRE_SCOPE` is not set by the app because GUI flows could not be tested).
- Dependencies: a point-in-time audit only (2 low npm advisories, a lock-only Rust `rkyv` entry, unmaintained transitive crates pulled in by Tauri); nothing runs the audit on a schedule. The Claude subscription provider was not exercised after the MCP SDK update.

**Scientific/product limits**
- Only the `random` split strategy is implemented (the others fail the run explicitly). Datasets of fingerprint mode `reference` (PyG/HuggingFace names) are not pinned by content; dataset *bytes* are not stored, only a fingerprint.
- Gradient comparison with hand-written PyTorch exists for three reference graphs; other layer families rely on golden text, a forward pass and the random-graph oracle. The random-graph generator covers MLP/CNN/residual/branch/multi-input/sequence families (not attention/recurrent/GNN).
- Checkpoints larger than 256 MB get header/size checks only. Resume is data (`resumable`), never automatic.
- Layers and dataset kinds that are not in the registry / `detect_kind` are listed in `LIMITATIONS.md` §7; `Custom`/`Subgraph` are the escape hatch.
- Browser-mode workspace names are not validated; the Tauri branch of the workspace store has no test.
- The documentation cites `file:line`; those drift with every edit (the checker verifies paths, not line numbers).

## 90.6 Reproducibility assessment

Detail and the test behind every claim: `REPRODUCIBILITY.md` (§1–§7).

- **Deterministic (measured):** CPU training with the same seed, software stack and inputs gives bit-identical per-epoch losses (maximum difference 0.0 for the three reference experiments, `verify:reference-train`; a different seed differs); code generation is byte-identical across repeated calls, save/load round trips and fresh processes (`test:determinism`, `test:codegen-golden`); shape inference is a pure function of the generated code.
- **Partially deterministic:** GPU training — seeds, cuDNN deterministic mode, `use_deterministic_algorithms(warn_only=True)` and seeded DataLoader workers are set and recorded, but `atomicAdd` reductions and `scatter_add` stay nondeterministic and the app claims nothing for CUDA; a different PyTorch/CUDA/PyG/BLAS/CPU can change results (the manifest records the versions so a difference is explainable); a resumed run restores optimizer, scheduler, epoch, global step and every RNG stream it can — `run.resumed.rng_restore` records which streams really were restored.
- **Nondeterministic by nature:** LLM provider output, wall-clock timestamps, scheduler queueing on a cluster.
- **Metadata stored per run** (`manifest.json`, `run.json`, `events.jsonl`): git commit and dirty state (`reproducible_from_git` only for a known commit with no tracked changes), the frozen graph and the exact generated `model.py` with their hashes, the full training configuration, dataset identity (fingerprint modes `content` / `structure` / `config+content` / `reference`), the split strategy with a zero-overlap assertion, the seed, software versions (python, torch, torch_geometric, numpy, CUDA/cuDNN, OS), hardware (device, GPU, CPU count, RAM — no hostname), approved custom-code hashes and whether unsafe unpickling was enabled, and a canonical `config_identity_sha256`.
- **Required to reproduce an experiment:** the run directory; the git commit when `reproducible_from_git` is true; data whose fingerprint equals the recorded one (the trainer refuses to start on changed bytes); the recorded software versions; then `python train.py` in a copy of the run directory with the same seed, comparing `config_identity_sha256` and the per-epoch losses. The data bytes themselves are not part of the run directory.
