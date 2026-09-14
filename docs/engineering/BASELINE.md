# SpinoML — Hardening baseline

Established: 2026-09-14
Purpose (TODO §1.9): record exactly what works **before** hardening begins.

## Repository state

| | |
|---|---|
| Remote | `https://github.com/christophmanitz/spinoml.git` |
| HEAD | `af19e06f18eb4742b06d3a5e837ea356f18055fa` — *checkpoint: rest of the hardening WIP* |
| Includes | Phase 0 (OpenCode provider, `e5a2a7c`) + pre-existing hardening WIP layer that could not be separated |
| Working tree | clean |

Note on phase ordering: TODO §0 (OpenCode) was executed and committed before this
baseline, per the "CRITICAL FIRST PHASE" instruction. Everything below therefore
baselines the **post-Phase-0** tree, which is the state hardening continues from.

## Environment

| | |
|---|---|
| OS | Ubuntu 24.04.4 LTS, x86_64 |
| CPU | 16 cores (AMD/Intel) |
| RAM | 30 GiB (23 GiB available at baseline time) |
| GPU | none (no `nvidia-smi`, no CUDA driver in `ldconfig`) |
| CUDA | not available — PyTorch is a CPU build |
| Python | 3.12.13 (conda env `mlforge-dev`) |
| Node | v22.23.2 (nvm) |
| npm | 10.9.8 |
| Rust | **not installed** (`rustc`/`cargo` absent from PATH and `~/.cargo`) |
| Git | 2.43.0 |
| PyTorch | 2.12.0+cpu (`torch.cuda.is_available()` → False) |

The documented dev env name in CLAUDE.md/TODO is `spinoml-dev`; on this machine
it is `mlforge-dev`. Harnesses that exec `python` require that env active in the
shell — see failure notes below.

## Verification matrix

Run: 2026-09-14, conda env `mlforge-dev` active for all Python-executing harnesses.

| Check | Command | Result |
|---|---|---|
| Build | `npm run build` (tsc -b && vite build) | **PASS** (2 warnings, below) |
| Lint | `npm run lint` (eslint .) | **FAIL** — 49 errors / 4 warnings (below) |
| Codegen | `npm run verify:codegen` | **PASS** — 13/13 graphs |
| Torch sidecar | `npm run verify:sidecar` | **PASS** — 6/6 cases + activations |
| Training codegen | `npm run verify:traingen` | **PASS** — compile + e2e multitask + e2e external validation |
| Scientific smoke | `npm run verify:smoke` | **PASS** — 15 checks (synthetic data, 5 epochs, checkpoint + metrics) |
| Training failures | `npm run verify:failures` | **PASS** — 9 failure modes (invalid ds/model/optimizer/lr, missing/unwritable out dir, NaN, NaN-loss→numeric, SIGKILL) all FAIL loudly |
| Checkpoints | `npm run verify:checkpoint` | **PASS** — 45 checks (full state incl. rng + config, resume, cancel-saves, atomic-write crash sim, corrupted-ckpt rejection) |
| OpenCode provider | `npm run verify:opencode` | **PASS** — models list, /chat stream, bogus-model clean error |
| Rust | `cargo check` in src-tauri | **SKIPPED** — Rust toolchain not installed on this machine |
| Python tests | `python -m pytest` | **SKIPPED** — no Python test suite in repo, pytest not installed |
| JS/TS tests | `npm test` | **SKIPPED** — no `test` script in package.json |

## Failures & notes

### Lint — FAIL (known debt, unchanged by Phase 0)

- **Command:** `npm run lint`
- **Symptom:** 53 problems — 49 errors, 4 warnings
- **Component:** pre-existing code, catalogued in full in `AUDIT.md` §3.1. The
  count is **identical** to the audit's recorded state; Phase 0 added zero new
  findings. Categories: `react-refresh/only-export-components`, `react-hooks/`
  (set-state-in-effect, exhaustive-deps), `preserve-caught-error`,
  `no-useless-escape`, `no-useless-assignment`.
- **Exit code:** 0 (eslint exits 0 under the current config despite findings),
  so CI-style gate is currently a no-op; treat the error count as the signal.
- **Fix target:** Phase "lint grün" / AUDIT §3.1 — deferred to hardening.

### Build — PASS with 2 warnings

- `[INEFFECTIVE_DYNAMIC_IMPORT]` for `src/data/graph/files.ts` and
  `src/training/graph/viewMode.ts` — modules both statically and dynamically
  imported (FileExplorer). Pre-existing, cosmetic (dead chunk-split intent).

### Environmental: harnesses require the conda env

- Running the verify scripts from a bare shell (no conda env) fails with
  `/bin/sh: 1: python: not found` in `verify:traingen` (and would hit
  `verify:codegen` / `verify:sidecar` the same way). There is no `python`
  binary outside the `mlforge-dev` env. Not a code defect — a documented
  environment prerequisite for all future sessions: **activate the env first.**
- LLM-dependent check `verify:opencode` additionally requires the LLM sidecar
  on `127.0.0.1:7422` (spawned by Rust in Tauri mode; `npm run sidecar:llm`
  in browser dev). OpenCode CLI must also be installed there.

## What this means for hardening

1. **Green today:** build, codegen, torch sidecar, training codegen, OpenCode
   provider path — the core generation + inference + training pipelines hold.
2. **Only blocker-class finding:** lint posture (49 errors) — already itemised,
   zero diff from Phase 0.
3. **Environment gaps to fix before Rust/remote phases:** Rust toolchain
   missing (blocks `cargo check`, Tauri packaging); CPU-only PyTorch (remote
   GPU phases will need the HPC sidecar, as already planned in TODO §12b).