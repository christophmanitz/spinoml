# SpinoML — Test Matrix

> Phase 1 (TODO §2). Maps each component to the tests/harnesses that verify it.
> Baseline (docs/engineering/BASELINE.md, 2026-09-14): build ✓, codegen 13/13 ✓,
> sidecar 6/6+activations ✓, traingen ✓, opencode ✓, lint ✗ (49e/4w), cargo/pytest/npm-test n/a.

Component              | Automated tests today                            | Gap                          | Planned (TODO §)
---------------------- | ---------------------------------------------- | ---------------------------- | -----------------
Frontend shell         | `npm run build` (tsc+vite)                     | no e2e/render tests          | §38/39 UI state + error states
GraphStore             | none (only type-imports in verify:*)           | mutation/history/ports UT    | §4 GraphStore correctness, §4.2
Layer registry         | `verify-codegen` (13 graphs via generator)     | coerceParams per-kind UT     | §4.3 validate-before-commit
Code generation        | `verify-codegen` 13/13 ✓                       | golden outputs, multi-input  | §6 golden tests
Training codegen       | `verify-traingen` ✓ (compile, multitask, eval-only) | immersive failure tests | §24 training failure tests
Shape inference        | `verify-sidecar` (synthetic /infer) ✓          | fail-closed + race tests     | §8/9/10
Dataset handling       | `verify-sidecar` (inspect/stats/smoke) ✓       | per-kind handlers, leak tests| §17/18, §19
Training runs (local)  | `verify-traingen` e2e (2-head, external-val) ✓ | local job state machine     | §30/31/32/33
Training runs (remote) | none                                          | ssh + slurm run tests        | §34/35/36/37
Persistence            | none (manual smoke mandated in CLAUDE.md)      | round-trip + schema tests    | §4.3, §5
Torch sidecar          | `verify-sidecar` 6/6+activations ✓             | auth/path-scope tests        | §77/78 security
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
| `npm run lint`          | all TS (eslint)                     | FAIL 49e/4w (gate no-op) |
| `npm run verify:codegen`| generator (13 graphs, exec-ed)      | PASS 13/13  |
| `npm run verify:traingen`| training codegen (compile + 2 e2e) | PASS         |
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
- S001 — shape-inference contract tests (§8/9)
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