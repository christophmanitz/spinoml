# SpinoML — Risk Register

> Phase 1 (TODO §2) + Phase 2 (TODO §3) seed. This table is the living register;
> it is updated as hardening phases close items. Severity: CRITICAL (must fix
> before release), HIGH, MEDIUM. Status: OPEN / ADDRESSED / VERIFIED / CLOSED.
> Each risk references the hardening phase(s) in TODO.md that own it.

| ID     | Area      | Risk                                                                | Severity | Test    | Status   | Owned by (TODO §) |
| ------ | --------- | ------------------------------------------------------------------- | -------- | ------- | -------- | ----------------- |
| R001   | Graph     | Invalid graph can be committed                                       | CRITICAL | G001    | ADDRESSED | §4 GraphStore correctness (validate-before-commit + mutation guards landed 2026-09-14) |
| R002   | Codegen   | Generated code differs from graph                                   | CRITICAL | C001    | OPEN     | §5/6/7 codegen + golden tests |
| R003   | Shape     | Incorrect shape accepted                                             | CRITICAL | S001    | OPEN     | §8/9/10 shape inference (training-launch gate landed 2026-09-14, see R028) |
| R004   | Training  | Failed training reported successful                                 | CRITICAL | T001    | OPEN     | §30 local job state machine |
| R005   | Checkpoint| Corrupted checkpoint accepted                                        | CRITICAL | T002    | OPEN     | §26/27/28 checkpoint correctness/atomicity/crash |
| R006   | Dataset   | Split leakage (train/val/test)                                       | CRITICAL | D001    | OPEN     | §19 train/val/test integrity |
| R007   | Async     | Stale inference/response overwrites state                            | HIGH     | A001    | ADDRESSED | §10 async races (staleness guard + abort verified by test:races 2026-09-14) |
| R008   | Sidecar   | Sidecar crash leaves application inconsistent                        | HIGH     | S003    | ADDRESSED | §11/12/13 sidecar robustness/crash/ports (structured `error_code` on every path, stall cap, abort, kill→restart verified by test:robustness 2026-09-14; §13 port supervision = Rust/managed.ts remainder) |
| R009   | SSH       | Connection failure produces wrong state (silent local fallback)      | HIGH     | R001    | OPEN     | §34/35 SSH reliability & credential safety |
| R010   | SLURM     | Wrong remote job state reported                                       | HIGH     | R002    | OPEN     | §36/37 SLURM reliability & job recovery |
| R011   | Persistence| Graph corruption on save/autosave                                    | CRITICAL | P001    | ADDRESSED | §4.3/5 persistence + schema versioning (round-trip + fail-closed tests landed 2026-09-14) |
| R012   | Security  | Command/path injection (ssh, run_script)                             | CRITICAL | SEC001  | OPEN     | §44/45/46/47 injection + path security |
| R013   | Security  | Sidecar CSRF via CORS `*` — arbitrary website executes code/files      | CRITICAL | SEC002  | OPEN     | AUDIT §1.1/1.3, plan §77 |
| R014   | Security  | LLM `/respond` + auto-approve paths unauthenticated                   | CRITICAL | SEC003  | OPEN     | AUDIT §1.2, plan §14/15 |
| R015   | Security  | `torch.load(weights_only=False)` pickle RCE on foreign `.pt`          | CRITICAL | SEC004  | OPEN     | AUDIT §1.4, plan §47 |
| R016   | Security  | Missing sidecar path scoping — arbitrary file read/write/exec         | CRITICAL | SEC005  | OPEN     | AUDIT §1.5 |
| R017   | Quality   | Lint debt (49e/4w) hides real defects; eslint exits 0 (gate no-op)    | MEDIUM   | Q001    | OPEN     | AUDIT §3.1 |
| R018   | Quality   | No unit tests for state stores (GraphStore/history/workspace)         | HIGH     | Q002    | OPEN     | §4.2 graph mutations (GraphStore done 2026-09-14 via G001; history/workspace open) |
| R019   | Remote    | SSH alias `leipzig-hpc` broken in `~/.ssh/config` (CLI/terminal only)  | MEDIUM   | R003    | VERIFIED | fixed 2026-09-14 (`~/.ssh/config` consolidated) |
| R020   | Remote    | Remote smoke blocked until Phase 12b (abspath not sidecar-addressable)| MEDIUM   | R004    | OPEN     | §12b |
| R021   | Remote    | Cluster scratch path unknown; `/scratch/<user>` absent (network home) | MEDIUM   | R005    | OPEN     | Phase 12b site prep |
| R022   | Remote    | `python: not found` on bare shell breaks verify harnesses (env gap)   | MEDIUM   | R006    | OPEN     | ops: activate `mlforge-dev` |
| R023   | Remote    | Rust toolchain absent → `cargo check`/Tauri packaging impossible      | HIGH     | R007    | OPEN     | §1.6 env gap |
| R024   | Dataset   | Manifest/branch binding mismatches (graph datasets per branch)        | HIGH     | D002    | OPEN     | §17/18 dataset reliability & fingerprinting |
| R025   | Training  | Metric correctness / per-head eval drift (multitask)                  | HIGH     | T003    | OPEN     | §29/31 metric + event ordering |
| R026   | Training  | run.json immutability violated (post-launch mutation)                 | HIGH     | T004    | OPEN     | §42 training immutability |
| R027   | LLM       | Model config scattered through code (provider drift)                  | MEDIUM   | L001    | OPEN     | §0.4 (Phase 0 addressed: providerStore centralized — VERIFY once) |
| R028   | Training  | Training launched on a model never verified (shapes unknown/invalid) | HIGH     | S002    | ADDRESSED | §10 fail-closed (verifyModelForTraining gates NewRunModal submit 2026-09-14) |

## Notes on the live environment (2026-09-14)

- **Prod remote state verified:** `zw93onug@login01.sc.uni-leipzig.de`, root
  `~/spinoml_gnn` (current), SLURM partition `paula` + `gres gpu:1`; remote
  sidecar venv present at `<root>/.spinoml/venv`. Partitions observed live:
  sirius, polaris, clara, paula/gpu-a30, paul, t3000, gpu-v100, gpu-rtx2080ti.
- **Invariants at stake for R009:** `getCurrentConnection()` falls back to LOCAL
  when the current remote id is not found — a wrongly-persisted id silently
  switches the app to local FS (the documented remote-connection failure mode).
- Baseline (docs/engineering/BASELINE.md) records all verification statuses that
  back the "Addressable" column of TEST_MATRIX.md.

## How to update

- Add a row when a new hazard is identified in any component inventory entry.
- Flip `Status` to ADDRESSED when the fix lands; VERIFIED when the corresponding
  test in TEST_MATRIX.md passes; CLOSED once the owning phase is done.
- Never delete a row — mark it CLOSED (audit trail).