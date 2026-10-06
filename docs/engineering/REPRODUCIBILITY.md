# SpinoML — Reproducibility

> What a run records, what is deterministic, how to reproduce an experiment, and
> what SpinoML does **not** promise. Every claim names the test that backs it.
> Written 2026-10-06 (Phases 18–22, 58–63); if a claim and the code disagree, the
> code and its test win — fix this file.

## 1. The run directory is the experiment

A run lives in `<workspace>/experiments/runs/<run id>/`. It is frozen at launch
(Phase 20/42): the trainer never reads the live UI state again.

| File | Written by | Content |
|---|---|---|
| `run.json` | app, at launch | the full frozen configuration: training config (optimizer, lr, epochs, batch size, seed, split strategy/ratio, heads), dataset config + **fingerprint**, backend/SLURM settings, `snapshot` (graph and model hashes, preprocessing scripts, `code_trust` approvals) |
| `model.spinoml` | app, at launch | the exact graph that was trained |
| `model.py` | app, at launch | the **exact generated PyTorch code** (kept because the code generator may change later) |
| `train.py` | app, at launch | the trainer, copied standalone (it cannot depend on sidecar modules) |
| `manifest.json` | `train.py` | machine-readable identity of the run, see §3 |
| `metrics.json` | `train.py` | final summary (`status`, `best_val_loss`, `epochs`, `n_params`, …) |
| `events.jsonl` | `train.py` | append-only log: `run.provenance`, `dataset.fingerprint`, `config.env`, `run.determinism`, `run.snapshot`, `split.integrity`, per-epoch events, terminal `run.done`/`run.failed`/`run.cancelled` |
| `checkpoints/best.pt`, `last.pt` | `train.py` | model/optimizer/scheduler state, epoch, global step, RNG streams, frozen config; written atomically (tmp + fsync + rename) |
| `stdout.log`, `stderr.log`, `status`, `pid` | executor | process output and job state |

`train.py` re-hashes the run-dir copies of the graph and `model.py` against
`run.json.snapshot` before training and **fails loudly on drift** (`verify:traingen`,
`verify:immutability`).

## 2. What is recorded (and the test that proves it)

| Item | Where | Test |
|---|---|---|
| Graph, generated model | `model.spinoml`, `model.py`, `snapshot.*_sha256` | `verify:immutability`, `verify:reference-train` (byte-equal to fresh codegen) |
| Dataset identity | `dataset.fingerprint` (SHA-256; modes `content`, `structure`, `config+content`, `reference`) re-verified by `train.py` before loading | `test:datasets` |
| Split | `training.split_strategy/val_split/seed`; zero-overlap asserted (`split.integrity`) | `verify:traingen` |
| Seed | `run.json`, `run.determinism` event, manifest | `verify:smoke`, `verify:reference-train` |
| Software / hardware | `config.env` event and `manifest.software/hardware`: python, torch, torch_geometric (or null), numpy, CUDA/cuDNN, OS string, device, dtype, GPU, CPU count, RAM. **No hostname, user name or absolute path.** | `verify:manifest` |
| Git state | `manifest.git` (§3) | `verify:manifest` |
| Approved custom code | `snapshot.code_trust` (node, kind, sha256, origin, approved_at) | `verify:code-trust-wiring` |
| Whether unsafe unpickling was enabled | `unsafe_pickle` in `config.env`, `run.provenance`, manifest | `test:safe-load` |

Not recorded, on purpose: Node and Rust versions (a training run uses neither), the
SpinoML app version, environment variables.

## 3. `manifest.json` and the git claim

Schema `spinoml.run-manifest/1`, written atomically at start (`status: running`,
before data loading, so a run that fails while loading data still has one), updated
after the environment is known, and rewritten at every terminal state
(`done` incl. eval-only, `failed` with stage/message, `cancelled`).
A failure to write it never changes the training outcome but is recorded as a
`manifest.error` event (`verify:manifest`, case 10).

`git` is read from the workspace root with argv lists (never a shell string):
full commit, branch (null on detached HEAD), tracked changes (staged or unstaged,
`dirty_files` ≤ 50 relative paths) and the number of untracked files **excluding
this run's own directory**.

**`reproducible_from_git` is `true` only if the commit is known and no tracked file
is modified.** Anything else — not a repository, git missing or timed out, any failed
git call, uncommitted changes — gives `false` and a `reason`/`notes` entry; a failed
git call can never look "clean" (`dirty_tracked` stays `null`). `notes` also state
when untracked files exist (not part of the commit), when the dataset fingerprint
mode is `reference` (pins only the reference file), when the device is CUDA and when
unsafe unpickling was enabled.

`hashes.config_identity_sha256` identifies the *inputs* of an experiment. It is the
SHA-256 of canonical JSON (sorted keys, no whitespace) over exactly the fields listed
in the manifest's `identity_fields`: graph hash, generated-model hash, preprocessing
script hashes, code-trust hashes, the identity-relevant training settings, dataset
fingerprint + kind + columns + heads + adapter, split settings and the seed. It
excludes run id, label, timestamps, paths and submission/backend settings, so two
runs with identical inputs get the same hash from different directories or machines,
and changing the learning rate, seed, dataset content or graph changes it
(`verify:manifest`, case 7).

## 4. What is deterministic

Backed by measurement, not by intent:

- **CPU, same seed, same software stack, same inputs → bit-identical per-epoch
  losses.** Measured max difference 0.0 for the three reference experiments
  (`verify:reference-train`); a different seed gives different losses.
- **Code generation is deterministic**: identical text on repeated calls and after a
  save/load round trip (`test:determinism`, `verify:reference`).
- **Generated model = hand-written PyTorch** for the three reference graphs:
  parameter counts, forward (float64 rtol 1e-10, float32 rtol 1e-5), loss and every
  gradient (`verify:reference`).
- **Seeding** covers Python `random`, NumPy, torch CPU and all CUDA devices, cuDNN
  deterministic/benchmark flags, `use_deterministic_algorithms(warn_only=True)` and
  seeded DataLoader workers (Phase 22, `run.determinism`).
- **Resume** restores optimizer, scheduler, epoch, global step and RNG streams
  (`verify:checkpoint`).

## 5. What is only partly deterministic or not at all

- **GPU runs are not claimed to be bit-reproducible.** CUDA reductions that use
  `atomicAdd` are nondeterministic even with all flags set. CUDA was **not tested on
  this machine** (`torch 2.12.0+cpu`; both harnesses print `SKIPPED  CUDA`).
- A different PyTorch/CUDA/PyG version, BLAS or CPU architecture can change results;
  the manifest records versions so the difference is at least explainable.
- Datasets with fingerprint mode `reference` (PyG/HuggingFace names) are not pinned
  by content; the manifest says so.
- Only the `random` split strategy is implemented.
- Remote workspaces had no dataset fingerprint before the remote inspector exists.

## 6. How to reproduce an experiment

1. Open the run directory; read `manifest.json`: `reproducible_from_git`, `git.commit`,
   `notes`, `software`, `dataset.fingerprint`.
2. Check out `git.commit` if `reproducible_from_git` is true; otherwise rely on the
   frozen files in the run directory (they contain everything except the data bytes).
3. Provide data whose fingerprint equals `hashes.dataset_fingerprint` (the trainer
   refuses to start on changed bytes).
4. Install the recorded software versions (`manifest.software`).
5. Run `python train.py` in a copy of the run directory (same seed). Compare
   `config_identity_sha256` to confirm the inputs are identical, then compare the
   per-epoch losses in `events.jsonl`.
6. Reload the model: `model.py` + `checkpoints/best.pt` (`safe_torch_load`, see
   `LIMITATIONS.md`).

## 7. Known gaps

Listed in `LIMITATIONS.md` (CUDA unverified, git not available on some login nodes,
`reference`-mode datasets, no enforcement of `code_trust` on the cluster).
