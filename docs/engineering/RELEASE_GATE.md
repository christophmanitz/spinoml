# Release gate (TODO.md §84/§85) — evaluated 2026-10-07

> The repository may only be described as **"Production Ready for Scientific Work"** if every applicable
> CRITICAL requirement passes. This file evaluates the checklist item by item against evidence that exists in
> this repository, and states what could NOT be evaluated here. **Verdict: CONDITIONAL** — see the end.
>
> Legend: `[x]` met and exercised by a suite named in the row · `[~]` met only partly or only against fakes —
> the limit is stated · `[ ]` not met / not verifiable on this machine. Suite names are `npm run <name>`
> scripts (registry: `scripts/suites.ts`, run everything with `npm run ci`).

Verification environment: Ubuntu 24.04, x86-64, 16 threads, no GPU, torch `2.12.0+cpu`, Python 3.12, the
conda env's Node 20 for suites (Node 22 in an interactive shell), Rust 1.96. See `FINAL_RELIABILITY_REPORT.md` §90.2.

## Code

* [x] TypeScript build passes — `build` (`tsc -b && vite build`), `typecheck-scripts` (the test scripts are type-checked too)
* [x] Lint passes — `npx eslint .` reports **0 problems** (was 81); `scripts/lint-baseline.json` is zero, so the runner fails on any new one
* [x] Rust checks pass — `cargo-check`, `cargo-test` (88 tests), `verify:rust-panics`
* [x] Python tests pass — `verify:sidecar`, `test:robustness`, `test:scope`, `test:safe-load`, `test:deps-policy`, `test:run-script`, `verify:silent-except-py`, the trainer suites below. **Limit:** CUDA branches report `SKIPPED  CUDA`; `test:verifier` skips its end-to-end part without a running torch sidecar
* [x] No known CRITICAL bugs — no CRITICAL row of `RISK_REGISTER.md` is OPEN (after the 2026-10-07 reconciliation; R021 = site-specific cluster path and R044 = no CUDA evidence are now ADDRESSED for one site / one GPU, with the scope limits stated there). "Known" means found by the audits/tests/reviews described in `BUGS_FIXED.md`; nothing here proves the absence of unknown ones

## Graph

* [x] Graph invariants are enforced — `test:graphstore`, `test:fuzz` (948 invalid mutants rejected), `test:property`
* [x] Invalid mutations are rejected — `test:graphstore`, `test:fuzz`, `test:llm-safety` (hostile tool calls change nothing)
* [x] Persistence roundtrip works — `test:persistence`, `verify:codegen`
* [x] Corrupt files fail safely — `test:persistence` (torn/garbage `.spinoml`, fail-closed), `verify:frontend-errors`, `test:workspace-store` (hostile persisted state is repaired into a tree), `verify:failures` (corrupt run files)
* [x] Concurrent mutations are safe — `verify:concurrent`, `test:races`, `verify:graph-revision`, `test:history-store` (cross-document undo corruption found and fixed, R060)

## Code Generation

* [x] Code generation is deterministic — `test:determinism`, `test:codegen-golden` (twice in-process + once in a fresh process)
* [x] Golden tests exist — `test:codegen-golden` (40 byte-exact cases, coverage guard over every registry entry)
* [x] Generated code compiles — `test:codegen-golden` (`ast.parse` of every case), `verify:codegen`
* [x] Generated code imports — `test:codegen-golden` imports each model case
* [x] Generated models instantiate — `test:codegen-golden`, `verify:codegen` (13 graphs), `verify:reference`
* [x] Forward pass works — `test:codegen-golden` runs one forward per model case and checks the output shape
* [x] Backward pass works — `verify:reference` (loss + backward), `verify:reference-train`
* [~] Gradient tests exist — `verify:reference` compares **every gradient** with hand-written PyTorch for three reference graphs (MLP, CNN, multi-input), with a mutation-verified negative control. The other layer families (attention, recurrent, GNN, …) are checked by golden text, one forward pass and the random-graph oracle, **not** by a gradient comparison. Verified on CPU and (for the three reference graphs only) on one GPU, `test:hardware-cuda` (RTX 2080 Ti): forward ≤ 1.2e-7, gradients ≤ 8.9e-8; other GPU architectures not tested

## Shape Inference

* [x] Positive tests — `verify:sidecar`, `verify:codegen`, `test:property` (200 random valid graphs vs an independent oracle)
* [x] Negative tests — `verify:sidecar` (intentional error responses), `test:fuzz`, `test:robustness` (structured errors on every path)
* [x] Multi-input tests — `verify:sidecar`, `verify:codegen`, `test:property`
* [x] Multi-output tests — `verify:codegen`, `test:codegen-golden`, `verify:traingen` (multitask)
* [x] Shape mismatch detection — `verify:sidecar`, `verify:ui-state`, `test:fuzz`
* [x] Stale-response protection — `test:races`, `verify:graph-revision`
* [x] Failed inference cannot silently produce a valid state — `verify:ui-state`, `verify:frontend-errors`

## Sidecars

* [x] Startup tested — `test:process-lifecycle` (busy port exit 3, invalid port exit 2), `test:sidecar-auth-torch`, `test:sidecar-auth-llm`
* [x] Shutdown tested — `test:process-lifecycle` (SIGTERM/SIGINT without orphans, grandchildren, zombies)
* [x] Restart tested — `test:process-lifecycle` (25 cycles, same-port restart after SIGTERM and SIGKILL), `test:robustness`
* [x] Invalid requests tested — `test:robustness`, `test:fuzz`, `test:llm-safety`, `test:llm-providers`, the auth matrices
* [x] Timeout tested — `test:robustness` (slow-client stall cap), `test:llm-safety` (upstream stall), `test:opencode-lifecycle` (start timeout)
* [x] Crash recovery tested — `test:robustness` (kill → restart), `test:process-lifecycle`
* [x] Error states are visible — `verify:ui-state`, header badges (`offline` vs `auth failed`), `test:sidecar-auth-frontend`. **Not exercised:** the Rust parent (`spawn_managed`, `PR_SET_PDEATHSIG`) and the real window

## Training

* [x] Seed handling — `verify:traingen`, `test:determinism`
* [x] Training snapshot — `verify:immutability`, `verify:traingen` (frozen graph/model hashes; drift → stage `snapshot`)
* [x] Checkpointing — `verify:checkpoint` (atomic writes, SIGKILL mid-training)
* [x] Resume — `verify:checkpoint`, `verify:integrity` (`rng_restore` recorded)
* [x] NaN/Inf detection — `verify:failures`
* [x] Failure state — `verify:failures`, `verify:states`
* [x] Success state — `verify:integrity` (the integrity gate: `done` needs valid metrics, loadable checkpoints, required events, a valid manifest), `verify:smoke` (synthetic data, 5 epochs, checkpoint + metrics)
* [x] Cancellation — `verify:cancel`, `verify:states`
* [x] Metric correctness — `verify:metrics`
* [x] Artifact integrity — `verify:integrity`, `verify:manifest`

## Scientific Reproducibility

* [x] Git commit recorded · [x] Dirty Git state recorded — `verify:manifest` (`reproducible_from_git` only for a known commit without tracked changes)
* [x] Graph stored · [x] Generated model stored · [x] Training configuration stored — run dir `run.json` / `model.spinoml` / `model.py` (`verify:traingen`, `verify:immutability`)
* [x] Dataset identity stored · [x] Dataset split stored — fingerprints + frozen split strategy with a zero-overlap assertion (`verify:traingen`, `test:datasets`)
* [x] Seed stored · [x] Software versions stored · [x] Hardware information stored — `config.env` in `manifest.json` (`verify:manifest`)
* [x] Experiment manifest stored — `verify:manifest`, `verify:integrity`
* **What reproducibility is NOT claimed for:** bitwise equality on CUDA (nondeterministic reductions), runs after the dataset's *content* changed (only a fingerprint is stored), runs on a different software stack. See `REPRODUCIBILITY.md` and `LIMITATIONS.md` §7.1

## Remote Execution

* [~] SSH failure handling — `verify:ssh`, `verify:credentials` (failure classification, no secrets in diagnostics) against a local fake `ssh`
* [~] SSH reconnect — recovery after a lost connection is re-reading the run dir (`verify:recovery`); no live-connection resumption exists
* [~] SFTP failure handling — the app uses no SFTP: file transfer is `ssh … cat`/stdin, covered by the same failure classification
* [~] SLURM submission · [~] SLURM status tracking · [~] SLURM failure handling — `verify:slurm` (submit parsing, every squeue/sacct mapping incl. communication loss, `cargo test`), against fakes
* [~] Job ID persistence — `verify:submission` (claim directory idempotency, retry recognises the durable pid)
* [~] Application restart recovery — `verify:recovery`, `verify:states`
* **Verification update 2026-10-07:** `test:remote-live` drove the real `ssh_*`/launch code against SC Leipzig (`login01`, alias `leipzig-hpc`, SLURM): live connection, a direct run, a SLURM run (done) plus a run cancelled via `ssh_stop_training_run` (`sacct` CANCELLED), recovery after dropping in-memory state and a GPU run (`env.device == cuda`) all pass (PASS 5 / FAIL 0 / SKIPPED 1); `live_bootstrap` is SKIPPED because `run_bootstrap` takes an `AppHandle`. The rows above were otherwise verified against fakes; the bootstrap/tunnel/remote-sidecar through the real window, other clusters, other GPUs, long runs and multi-node remain unverified — see `REMOTE_TRAINING.md` §9

## Security

* [x] No known command injection — `verify:command-injection` (quoting round-trip against a real shell, SSRF policy, ssh target policy), `cargo test` (`validate_alias`)
* [x] No known path traversal — `verify:paths`, `test:scope`, `cargo test` (symlink-aware `resolve()`, scope file)
* [~] No unsafe Python execution — generated text is built only from escaped literals (`verify:codegen-security`, 5362 hostile cases); code-bearing nodes (Custom/DataOp/CustomScript) execute **only after a human approves them** (`verify:code-trust`, `verify:code-trust-wiring`) — by design they DO execute; the gate is a frontend gate, the sidecars are protected by the token
* [x] Unsafe deserialization reviewed — `test:safe-load` (`weights_only=True` + allow-list; malicious pickles refused)
* [x] No secrets in logs — `test:llm-safety` (key never reaches the event stream), the auth suites (token never in stdout/stderr/SSE), `verify:silent-catch`
* [x] Local services are appropriately bound — 127.0.0.1 only (asserted: a non-loopback interface refuses), token + Host + Origin on every request (`test:sidecar-auth-torch`, `test:sidecar-auth-llm`), scope enforced once a workspace exists
* [x] LLM/MCP actions are validated — `test:llm-safety`, `test:llm-validation-parity`, `test:opencode-lifecycle` (real MCP bridge), `test:llm-providers`
* **Residual (documented in `LIMITATIONS.md` §2):** the webview is trusted (a strict CSP is shipped but only verified in a real Chromium, not in the Tauri window), a same-user process can read the token, browser-dev mode is tokenless, `/infer` runs approved model code with the sidecar's file access

## §85 Critical stop conditions — can any of these still happen?

| Condition | Assessment |
|---|---|
| Data can be silently corrupted | **No known path.** The one found (cross-document undo, R060) is fixed and tested; corrupt persisted/run files are rejected or repaired and reported. Unknown ones cannot be excluded |
| A wrong model can be generated and accepted as valid | Not for the reference/golden/random families on CPU (`verify:reference`, `test:codegen-golden`, `test:property`) nor for the three reference graphs on one GPU (`test:hardware-cuda`, RTX 2080 Ti). **Other GPU architectures are unverified**; layer families without a gradient comparison rely on golden + forward checks |
| Shape inference can silently be wrong | Checked against an independent oracle for the random-graph families (`test:property`); families not generated are listed in `LIMITATIONS.md` §6 |
| A failed training run can be reported as successful | No: the integrity gate refuses `done`; unreadable status/empty loader/corrupt manifest were fixed (Phase 50) |
| A corrupted checkpoint can be silently accepted | No, except checkpoints > 256 MB get header/size checks only (`LIMITATIONS.md` §5) |
| Dataset leakage without detection where the system manages the split | No: strategy frozen, zero-overlap asserted, overlap fails the run. Only `random` is implemented — the other selectable strategies fail the run (stage `split`) rather than falling back silently |
| A stale asynchronous response can overwrite current state | No (`test:races`, `verify:graph-revision`) |
| An invalid graph can become the authoritative graph | No: validated before commit (`loadSnapshot`, `connectNodes`, LLM tools) |
| A remote job can be reported with an incorrect final status | Reconciliation is unit/fake-tested for every state and was exercised once against one real SLURM site (`test:remote-live`: a run cancelled → `sacct` CANCELLED; direct + SLURM runs reach `done`); not exhaustively on other schedulers |
| A running experiment can be silently changed by later UI edits | No: frozen run snapshot with hashes (`verify:immutability`) |
| An experiment cannot be reconstructed from its artifacts | Reconstructible from git state + manifest + frozen graph/model + config + seed; dataset *content* is not stored (fingerprint only) |
| Critical security vulnerabilities remain unresolved | None known after Phases 43–47, 77/78, 79 (dependency audit: 0 critical, 2 low npm advisories, 1 lock-only Rust entry, unmaintained transitive crates) |

## Verdict: **CONDITIONAL**

Every CRITICAL item that can be evaluated on this machine passes with a named, mutation-checked suite. Four requirements
were closed on 2026-10-07, each with the scope limits stated below (GitHub CI green; CUDA on one GPU; a real cluster at one site; a 1 h soak). The label
**"Production Ready for Scientific Work" is NOT claimed**, because these applicable requirements remain:

1. **The real Tauri window** — launch the built app once and confirm the strict CSP, the auth wiring and the scope writer behave (they are verified in a real Chromium and by unit/real-process tests, not in WebKitGTK). Checklist: `docs/engineering/TAURI_SMOKE.md`. Fallback if the window misbehaves: `app.security.csp: null`.
2. **The Claude subscription provider** (OAuth) and the real opencode CLI are not driven by any suite (`scripts/test-llm-live.ts` exists, verified only against a local fake provider; needs credentials and `SPINOML_LIVE_LLM=1`).
3. **The in-window cluster path** — `test:remote-live` passed once, but `run_bootstrap` takes an `AppHandle`, so the bootstrap/tunnel/remote-sidecar through the real Tauri window is still a manual step; other clusters, other GPU architectures, multi-GPU, long runs and multi-node are untested.
