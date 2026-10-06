# SpinoML — Test Matrix

> Phase 1 (TODO §2). Maps each component to the tests/harnesses that verify it.
> Baseline (docs/engineering/BASELINE.md, 2026-09-14): build ✓, codegen 13/13 ✓,
> sidecar 6/6+activations ✓, traingen ✓, opencode ✓, lint ✗ (49e/4w), cargo/pytest/npm-test n/a.

Component              | Automated tests today                            | Gap                          | Planned (TODO §)
---------------------- | ---------------------------------------------- | ---------------------------- | -----------------
Frontend shell         | `npm run build` (tsc+vite) + `verify:ui-state` ✓ + `verify:frontend-errors` ✓ (loading/success/error/offline distinct; timeout/cancelled string-typed, not Success) + `verify:silent-catch` ✓ + `verify:silent-fixes` ✓ (every swallowing handler documented; fixed hidden failures asserted) | no e2e/render tests          | §38/39 UI state + error states
GraphStore             | `npm run test:graphstore` ✓ + `verify:graph-revision` ✓ + `verify:concurrent` ✓ (saveSeq+revAtStart dirty correction; last-wins, no silent loss) + `test:fuzz` ✓ (948 invalid mutants/run, 28 operators: rejected, store unchanged, sidecar structured errors, 10 000-node stress) | history/dirty-snapshot UT   | §4.2/40/41
Layer registry         | `verify-codegen` (13 graphs via generator) + G001 param validation | coerceParams per-kind UT  | §4.3 validate-before-commit
Code generation        | `verify-codegen` 13/13 ✓ + `test:determinism` ✓ (repeatability, shuffled arrays) + `verify:codegen-security` ✓ (5362 adversarial cases: 10 hostile payloads × every string sink of model/data/training codegen + numeric NaN/Infinity/string/null, checked with Python `ast`+`tokenize`; `pyStr` round-trip via `ast.literal_eval`; by-design code sinks asserted exactly) + `verify:reference` ✓ (3 reference graphs vs hand-written PyTorch: params/forward f32+f64/loss/grads + negative control) + `verify:reference-train` ✓ (real trainer: artifact set, checkpoint↔model, same-seed rerun identical, other seed differs) | golden outputs for all layer families, random graphs  | §6 golden tests
Training codegen       | `verify-traingen` ✓ (compile, multitask, eval-only) | immersive failure tests | §24 training failure tests
Shape inference        | `verify-sidecar` ✓ + `test:verifier` ✓ (fail-closed gate) + `test:races` ✓ (stale-response replay) + `verify:graph-revision` ✓ (graph revision + runCounter guard, stale shape drop) | live multi-request on real sidecar | §8/9/10/40 + `test:property` ✓ (200 random valid graphs × 6 families vs an independent oracle; 40 via the real sidecar vs forward hooks)
Dataset handling       | `verify-sidecar` ✓ + `test:datasets` ✓ + `verify:graph-revision` ✓ + `verify:concurrent` ✓ (per-dataset seq + refreshSeq + smoke graphRev, no stale overwrite) | leak tests, parquet (pyarrow absent), remote pre-12b | §19/40/41 + `test:safe-load` ✓ (85; real RCE attempts, all 18 example artifacts load) + `test:scope` ✓ (manifest escapes)
Training runs (local)  | `verify-traingen` e2e ✓ + `verify:states` ✓ + `verify:events` ✓ + `verify:cancel` ✓ + `verify:manifest` ✓ (76: git clean/dirty/staged/untracked/no-repo/no-git/detached, hashes, config identity invariants, failed/cancelled/eval-only manifests, no paths/env/hostname, atomicity) | — | §30/31/32
Training runs (remote) | `verify:submission` ✓ + `verify:slurm` ✓ + `verify:recovery` ✓ (close→restart→reconnect re-queries live pid/status/squeue/sacct, not cache; detached survival; atomic claim) | — | §33/36/37
Persistence            | `npm run test:persistence` ✓ (P001/P011: round-trip, malformed-file matrix, schema versions, fail-closed) | autosave (localStorage) browser smoke | §4.3, §5
Torch sidecar          | `verify-sidecar` 6/6+activations ✓ + `test:robustness` ✓ (11-case matrix: startup, structured errors with `error_code` on every path, slow-client stall cap, abort resistance, kill→restart recovery) | auth/path-scope tests        | §77/78 security
LLM sidecar            | `verify-opencode` ✓ (+manual ask/confirm) + `verify:command-injection` ✓ (args split/quote round-tripped through a real `sh`, SSRF blocklist + DNS-resolving `safeFetch` with fake lookup/fetch, ssh target policy, runScript `./` prefix)      | `/respond`/auto-approve auth | §14/15, §77 + `verify:silent-catch`/`verify:silent-fixes` ✓ (Node half of phase 50)
MCP                    | `verify-opencode` (via opencode bridge) ✓      | spec-parity across providers | §16 MCP validation
Tauri/Rust             | `cargo check` ✓ + `cargo test` (ssh failures + slurm 13 tests) ✓ + `alias_validation_tests` (Phase 44, **written but UNCOMPILED here — no toolchain, see LIMITATIONS §1**) | command integration tests    | §1.6 (env), §48 Rust errors
Filesystem (local)     | `verify:paths` ✓ (48, Node resolver) + `test:scope` ✓ (88, Python scope: path matrix, modes/config, dataset_handlers incl. hostile manifests, real-HTTP) | Rust `resolve()` symlinks (lexical only, no toolchain) | §45/46
Filesystem (remote)    | `verify:ssh` ✓                                | ssh mirror parity tests      | §34/35
SSH                    | `verify:ssh` ✓ + `verify:credentials` ✓       | live cluster integration     | §34/35 (+ R003 alias)
PTY/Terminal           | none                                          | interactive smoke            | manual
SLURM                  | `verify:slurm` ✓ + `verify:recovery` ✓ (live squeue/sacct/kill-0 + reconcile, detached setsid/sbatch, app blanks+refresh chain) | live sbatch/squeue smoke       | §36/37
Workspace (browser)    | none                                          | virtual-FS UT                | §38/39
Security (sidecars)   | `verify:command-injection` ✓ + `test:deps-policy` ✓ (34) + `test:run-script` ✓ (125; mocked `subprocess.run`, no file written on rejection) + `verify:code-trust` ✓ + `verify:code-trust-wiring` ✓ (LLM-path can't approve; untrusted code never reaches `/infer`, activations, smoke, `startRun`) | CORS `*`/no token (R013/R014), `torch.load` pickle (R015), path scoping (R016) still untested/open | §14/45/46/47, §77/78

## Harnesses available today

| Command                 | Component(s)                        | Baseline result |
| ----------------------- | ----------------------------------- | --------------- |
| `npm run build`         | all TS                              | PASS (2 warnings) |
| `npm run test:graphstore` | GraphStore invariants + mutation guards        | PASS         |
| `npm run test:persistence` | persistence round-trip + malformed-file fail-safety | PASS         |
| `npm run test:determinism` | codegen determinism (repeat + shuffled arrays)  | PASS         |
| `npm run test:verifier` | fail-closed model gate (decision matrix + e2e) | PASS         |
| `npm run test:races`  | async inference staleness guard (fetch mock) | PASS         |
| `npm run test:robustness` | torch sidecar robustness + crash recovery (isolated instance) | PASS         |
| `npm run test:datasets` | dataset handlers per-kind reliability matrix + hang regression + Phase-18 fingerprint (Python) | PASS 124 checks |
| `npm run lint`          | all TS (eslint)                     | FAIL 49e/4w (gate no-op) |
| `npm run verify:codegen`| generator (13 graphs, exec-ed)      | PASS 13/13  |
| `npm run verify:traingen`| training codegen (compile + 2 e2e + strategy guard + snapshot + env + determinism) | PASS         |
| `npm run verify:smoke`| scientific smoke test (synthetic data, 5 epochs, checkpoint + metrics) | PASS 15 checks |
| `npm run verify:failures`| training failure tests (9 failure modes incl. NaN input + NaN-loss→`numeric` hardening) | PASS 37 checks |
| `npm run verify:checkpoint`| checkpoint correctness (train/save/resume/cancel, full state incl. rng + config) + atomic-write crash sim + corrupted-ckpt rejection | PASS 45 checks |
| `npm run verify:metrics`| metric correctness (batch-size-weighted loss/metric aggregation vs Python reference) | PASS 11 checks |
| `npm run verify:states` | run state machine transitions and invalid-transition guards | PASS 43 checks |
| `npm run verify:events` | event ordering, terminal truncation, stale-read dropping   | PASS 26 checks |
| `npm run verify:cancel` | SIGTERM/SIGINT signal handling, resume checkpoint, shielding| PASS 28 checks |
| `npm run verify:submission`| atomic remote run directory claim and retry idempotency   | PASS 7 checks |
| `npm run verify:ssh`    | SSH failure classification, transport options, error detail | PASS 17 checks |
| `npm run verify:credentials`| secret/credential scan across artifacts, logs, error sanitize | PASS 8 checks |
| `npm run verify:slurm`     | SLURM reliability (sbatch 4 tests + 9 scheduler-state reconciliation, submit success/failure, squeue/sacct probes, job-ID persistence) | PASS 33 checks |
| `npm run verify:recovery`  | remote job recovery (close→restart→reconnect live re-query, no localStorage cache; detached/sbatch survival; reconcile gates) | PASS 38 checks |
| `npm run verify:ui-state`  | UI state must not lie (training stale→unknown on SSH loss, SSH badge disconnect, inference not hanging, save truthfulness) | PASS 33 checks |
| `npm run verify:frontend-errors` | frontend error states (loading/success/error/offline distinct; timeout/cancelled string-typed) | PASS 42 checks |
| `npm run verify:graph-revision`  | graph revision system (structural revision bump, stale shape/dataset/smoke/refresh drop) | PASS 32 checks |
| `npm run verify:concurrent`     | concurrent operations (Save+edit dirty correction, Edit+inference debounced+revision, LLM+user validate, Training snapshot frozen, Dataset seq) | PASS 44 checks |
| `npm run verify:immutability`   | training immutability (frozen run snapshot, detached launch, re-verified drift, no live GraphStore read) | PASS 27 checks |
| `npm run verify:sidecar`| torch sidecar (autostart)           | PASS 6/6+activations |
| `npm run verify:opencode`| LLM sidecar opencode provider      | PASS         |
| `cargo check` (src-tauri)| Rust compilation                    | PASS         |
| `pytest` / `npm test`   | —                                   | SKIPPED (no suites) |

Prereq for all `verify:*` that exec Python: conda env `mlforge-dev` active
(`python` is not on the bare PATH). `verify:opencode` additionally needs the
LLM sidecar on 127.0.0.1:7422 and the OpenCode CLI. + `verify:silent-catch`/`verify:silent-fixes` ✓ (Node half of phase 50)

## Test-ID cross-reference

Planned mutation/unit tests referenced by RISK_REGISTER.md:

- G001 — graph invariants + mutation tests (§4.1/4.2)
- C001 — codegen golden tests (§6), D001/… data-canvas twin
- S001 — shape-inference contract tests (§8/9); S002 — fail-closed training-launch gate (§10, test:verifier)
- S003 — sidecar HTTP robustness matrix (§11/12, test:robustness: malformed/missing/invalid-payload → 400 `error_code`, business errors → 200 `ok:false`+code, stall cap, abort, kill→restart)
- T001–T004 — training: job state machine (§30), metric correctness (§29),
  event ordering (§31), immutability (§42)
- D001 — split leakage (§19); D002 — dataset fingerprinting (§18)
- P001 — persistence round-trip + corruption (§5)
- A001 — async race tests (§10)
- R001–R007 — SSH/SLURM reliability (§34–37), remote smoke (§12b), env paths
- SEC001–SEC008 — injection/path/auth/code-trust/SSRF (§14/43/44/45/46/47, §77/78); SEC006 = code-trust gate, SEC007 = option/argument injection, SEC008 = SSRF
- Q001/Q002 — lint gate + store unit tests (§3, §4)

## Verification workflow

1. Any code change: `npm run build` (lint too once §3 is fixed).
2. Generator/sidecar-sensitive change: full `verify:*` suite.
3. Rust change: `cargo check` (requires toolchain — see R023).
4. Workspace/persistence change: manual smoke per CLAUDE.md.
5. Remote/SSH/SLURM change: `ssh zw93onug@login01.sc.uni-leipzig.de echo ok`
   first, then a real sbatch smoke on `paula` (never run training on the login node).