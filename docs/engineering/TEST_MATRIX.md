# SpinoML — Test Matrix

> Phase 1 (TODO §2). Maps each component to the tests/harnesses that verify it.
> Baseline (docs/engineering/BASELINE.md, 2026-09-14): build ✓, codegen 13/13 ✓,
> sidecar 6/6+activations ✓, traingen ✓, opencode ✓, lint ✗ (49e/4w), cargo/pytest/npm-test n/a.

Component              | Automated tests today                            | Gap                          | Planned (TODO §)
---------------------- | ---------------------------------------------- | ---------------------------- | -----------------
Frontend shell         | `npm run build` (tsc+vite)                     | no e2e/render tests          | §38/39 UI state + error states
GraphStore             | `npm run test:graphstore` ✓ (G001: invariants, mutation guards, validate-before-commit, edge-id gen) | history/dirty-snapshot UT   | §4.2
Layer registry         | `verify-codegen` (13 graphs via generator) + G001 param validation | coerceParams per-kind UT  | §4.3 validate-before-commit
Code generation        | `verify-codegen` 13/13 ✓ + `test:determinism` ✓ (repeatability, shuffled arrays) | golden outputs, multi-input  | §6 golden tests
Training codegen       | `verify-traingen` ✓ (compile, multitask, eval-only) | immersive failure tests | §24 training failure tests
Shape inference        | `verify-sidecar` (synthetic /infer) ✓ + `test:verifier` ✓ (fail-closed gate, INVALID/UNKNOWN/VALID decision matrix, end-to-end with sidecar when reachable) + `test:races` ✓ (stale-response replay via fetch mock) | live multi-request interleaving on real sidecar | §8/9/10
Dataset handling       | `verify-sidecar` (inspect/stats/smoke) ✓ + `test:datasets` ✓ (per-kind §18 matrix: valid/empty/missing/corrupt/wrong-dtype/NaN/Inf/single/large/unicode/spaces/relative — explicit-error, never-silently-empty + Phase-18 fingerprint: determinism, copy-stability, content-change) | leak tests, parquet (pyarrow absent env), remote pre-12b (no sidecar → no fingerprint) | §19
Training runs (local)  | `verify-traingen` e2e (2-head, external-val) ✓ | local job state machine     | §30/31/32/33
Training runs (remote) | none                                          | ssh + slurm run tests        | §34/35/36/37
Persistence            | `npm run test:persistence` ✓ (P001/P011: round-trip, malformed-file matrix, schema versions, fail-closed) | autosave (localStorage) browser smoke | §4.3, §5
Torch sidecar          | `verify-sidecar` 6/6+activations ✓ + `test:robustness` ✓ (11-case matrix: startup, structured errors with `error_code` on every path, slow-client stall cap, abort resistance, kill→restart recovery) | auth/path-scope tests        | §77/78 security
LLM sidecar            | `verify-opencode` ✓ (+manual ask/confirm)      | `/respond`/auto-approve auth | §14/15, §77
MCP                    | `verify-opencode` (via opencode bridge) ✓      | spec-parity across providers | §16 MCP validation
Tauri/Rust             | none (rust toolchain absent locally)           | cargo check + command tests  | §1.6 (env), §48 Rust errors
Filesystem (local)     | none                                          | path-sanity tests            | §45/46
Filesystem (remote)    | none                                          | ssh mirror parity tests      | §34/35
SSH                    | none                                          | state-on-failure tests       | §34/35 (+ R003 alias)
PTY/Terminal           | none                                          | interactive smoke            | manual
SLURM                  | none (probed live: partitions exist)           | sbatch/squeue/cancel tests   | §36/37
Workspace (browser)    | none                                          | virtual-FS UT                | §38/39
Security (sidecars)    | none (known open: CORS `*`, no auth)           | auth + injection tests       | §14/44/47, §77/78

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
| `npm run verify:failures`| training failure tests (8 failure modes, NaN input hardening) | PASS 33 checks |
| `npm run verify:sidecar`| torch sidecar (autostart)           | PASS 6/6+activations |
| `npm run verify:opencode`| LLM sidecar opencode provider      | PASS         |
| `cargo check` (src-tauri)| Rust                                | SKIPPED (no toolchain) |
| `pytest` / `npm test`   | —                                   | SKIPPED (no suites) |

Prereq for all `verify:*` that exec Python: conda env `mlforge-dev` active
(`python` is not on the bare PATH). `verify:opencode` additionally needs the
LLM sidecar on 127.0.0.1:7422 and the OpenCode CLI.

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
- SEC001–SEC006 — injection/path/auth (§14/44/45/46/47, §77/78)
- Q001/Q002 — lint gate + store unit tests (§3, §4)

## Verification workflow

1. Any code change: `npm run build` (lint too once §3 is fixed).
2. Generator/sidecar-sensitive change: full `verify:*` suite.
3. Rust change: `cargo check` (requires toolchain — see R023).
4. Workspace/persistence change: manual smoke per CLAUDE.md.
5. Remote/SSH/SLURM change: `ssh zw93onug@login01.sc.uni-leipzig.de echo ok`
   first, then a real sbatch smoke on `paula` (never run training on the login node).