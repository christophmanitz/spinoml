# Remaining work — plan and resources checklist

Status: 2026-10-07, after the push of `52df750`. The verdict in `RELEASE_GATE.md` / `FINAL_RELIABILITY_REPORT.md` is
**CONDITIONAL** because six requirements could not be verified on the development machine. This file turns them into a
plan: what closes each one (exact evidence), what it needs (resources checklist), who does what, and in which order.

Legend: **[me]** = work Claude can do in this repository without your hardware/accounts · **[you]** = needs your machine,
account, credentials or a decision · `[ ]` open · `[x]` done.

## 0. Overview and order

| # | Condition (from `RELEASE_GATE.md`) | Blocking resource | Effort | Order |
|---|---|---|---|---|
| 4 | CI green on GitHub | none (GitHub Actions is already running — **first run: rust green, node and python jobs red**, see §1) | 0.5–1 day [me] | **1st** |
| 3 | Real Tauri window smoke (CSP, auth wiring, scope writer) | a desktop session with WebKitGTK | 0.5–1 h [you] (+ 1 h prep [me]) | 2nd |
| 5 | Hour-scale soak | an idle machine for 1 h | 1 h unattended [you or me] | 3rd (parallel to 2) |
| 6 | Claude subscription provider + real opencode CLI | CLI logins / a model with quota | 1–2 h [you] (+ 2 h prep [me]) | 4th |
| 2 | Real cluster (SSH, SLURM, remote sidecar) | HPC login + SLURM allocation | 0.5 day [you] (+ 1 day prep [me]) | 5th |
| 1 | CUDA | a machine with an NVIDIA GPU | 2–3 h [you] (+ 0.5 day prep [me]) | 5th, **combine with #2** (GPU partition) |

Dependencies: #4 first — it is free and exposes environment assumptions (see §1) that would otherwise also bite #2 and #1.
#2 and #1 share a prerequisite (the live suites in §7, which do not exist yet) and, if your cluster has a GPU partition, one
allocation. #3 and #5 are independent of everything else.

## 1. Condition 4 — CI green on GitHub

**Evidence so far.** The push of `52df750` triggered the first CI run ever
(<https://github.com/christophmanitz/spinoml/actions/runs/37588448387>): job `rust` **passed**; job `node` failed 11 suites,
job `python` failed 8. These are environment assumptions of the suites/workflow, not product defects — exactly what a first run
on a clean machine is for. Diagnosed causes (from the job logs):

| # | Symptom | Cause | Fix |
|---|---|---|---|
| a | `ERR_MODULE_NOT_FOUND` in `test:llm-safety`, `test:sidecar-auth-llm`, `test:llm-providers`, `test:opencode-lifecycle`, `test:process-lifecycle`, `test:resource-leaks`, `test:soak` | `sidecar-llm/` has its own `package.json`; the workflow installs only the root dependencies | add `npm ci --prefix sidecar-llm` (node and python jobs) |
| b | `cargo-check`, `cargo-test`, `verify:ssh`, `verify:slurm`, `verify:credentials`, `verify:recovery` fail in the node/python jobs (`glib-2.0.pc` not found) | these suites need the Rust toolchain AND the Tauri system libraries (`libwebkit2gtk-4.1-dev`, `libglib2.0-dev`, …), which only the `rust` job installs | run the cargo-backed suites only in the `rust` job (filter by `needs: ['cargo']`), or install the apt packages in the other jobs |
| c | `test:webview-csp`: the headless browser exits at start | GitHub's Ubuntu runner restricts user namespaces, Chrome needs `--no-sandbox` there | add `--no-sandbox` when `CI=1` (or an env override such as `SPINOML_CHROMIUM_ARGS`) and keep the explicit `SKIPPED` line if no browser starts |
| d | `verify:reference`, `verify:reference-train`, `test:property`: "Not a conda environment: …/envs/mlforge-dev" / 201 property failures | the scripts shell out to `conda run -n mlforge-dev python`; the CI python job uses a plain `pip` environment | use the runner's resolved interpreter (`PYTHON`/`pythonCmd`) instead of a hard-coded conda env name |
| e | `verify:checkpoint`: 1 check failed | not yet diagnosed (the log shows only the summary line) | read `.test-results/verify:checkpoint.log` from the artifact, reproduce with the job's Python/torch versions |
| f | 33–42 suites BLOCKED in a job | by design (each job runs its slice) — but make sure every suite runs in exactly one job | audit the three job filters against `scripts/suites.ts` |

**Done when**
- [ ] a run on `main` shows all three jobs green (BLOCKED only for `remote-live`, `hardware-cuda`, `verify-opencode`);
- [ ] the same workflow is green on a pull request (open a throw-away PR);
- [ ] `ci.yml` header comments are corrected (they still say the Rust toolchain is absent on the dev box; it is in the conda env);
- [ ] `RELEASE_GATE.md` verdict item 4 and `LIMITATIONS.md` §6 updated with the run URL.

**Resources**
- [x] GitHub repo write access and Actions enabled (the run exists)
- [ ] `gh` CLI authenticated (it is: account `christophmanitz`) for `gh run view --log-failed`
- [ ] nothing else — all of it is repository work **[me]**; iterate with `git push` + `gh run watch`

## 2. Condition 3 — the real Tauri/WebKitGTK window

What has never been exercised: the built app's strict CSP, `sidecar_token` over IPC, the scope-file writer on workspace open,
bundled Monaco offline, the badges' `auth failed` vs `offline` states, the diagnostics banner. Verified elsewhere only in real
Chromium and with unit/real-process tests.

**Resources checklist**
- [ ] a desktop session (X11/Wayland) on this machine or another Linux box
- [ ] Tauri build prerequisites installed (`libwebkit2gtk-4.1-dev`, `libgtk-3-dev`, `libayatana-appindicator3-dev`, `librsvg2-dev`, `patchelf`; see https://v2.tauri.app/start/prerequisites/) — `cargo check` already compiles here, so most are present
- [ ] the conda env `mlforge-dev` activated in the shell that launches the app (an installed `.deb` started from the launcher needs it on `PATH`, otherwise both sidecars stay offline — known, see README)
- [ ] ~10 GB free disk for `npm run tauri build`; ports 7421/7422/7424 free (stop `npm run sidecar:*` first)
- [ ] a scratch workspace folder and, for the offline check, the ability to disable the network (`nmcli networking off` or unplug)

**Plan**
1. **[me]** write docs/engineering/TAURI_SMOKE (new, to be written): a numbered manual checklist with the exact expected result of every step (below), and an optional helper that prints the evidence (scope file, `/health` bodies with/without token).
2. **[you]** `npm run tauri dev` first (dev CSP is `null`, proves the app itself), then `npm run tauri build` and run the built binary / install the `.deb` (this is the one with the strict CSP).
3. **[you]** walk the checklist; open the webview devtools (dev build or `--debug` build) and read the console:
   - [ ] window renders, no console error containing "Content Security Policy" / "Refused to"
   - [ ] header badges: torch and LLM sidecar show **online**, no `ungesichert` chip (managed sidecars run with a token); stop one sidecar → **offline**; start it by hand without the token → `auth failed` / `ungesichert`
   - [ ] choose a workspace → `~/.cache/spinoml/scope.json` (or `$XDG_RUNTIME_DIR/spinoml/scope.json`) now lists the canonical root, mode 0600; `GET /health` with the token shows `scope.mode: enforced`
   - [ ] a code editor (Code panel, a Custom layer) mounts **with the network off** and highlights Python
   - [ ] build a small model, shape inference works, save, reopen the file, press Ctrl+Z immediately → nothing changes (R060)
   - [ ] open a dataset under `datasets/` that is a symlink to another directory → still listed (allowed target configured) / refused with the one-line fix (not configured)
   - [ ] chat panel: opencode provider answers; tool call changes the graph; the `run_script` confirmation appears and a click approves
   - [ ] start a 2-epoch local training run; Run-Detail shows epochs, `done`, integrity ok
   - [ ] cause an unhandled error on purpose (devtools: `Promise.reject(new Error('x'))`) → the rose "Unbehandelte Fehler" banner appears
4. **Fallback** if the window is blank or an editor fails: set `app.security.csp` to `null` in `src-tauri/tauri.conf.json`, rebuild, confirm the app works, and send me the console output — then the policy needs a Tauri-specific directive (`ipc:`, `asset:`, a nonce) and I fix it.

**Done when**: every box above is ticked; results pasted into the new TAURI_SMOKE document (date, OS, WebKitGTK version); `RELEASE_GATE.md` §Security wording `[~]` → `[x]` for the window, `LIMITATIONS.md` §2 item 1 updated.

## 3. Condition 5 — hour-scale soak

**Resources checklist**
- [ ] an idle machine for ≥ 1 h (no other test runs, no `npm run ci`), ≥ 4 GB free RAM, ports 7421/7422 free
- [ ] conda env `mlforge-dev` (torch, pyg, numpy) — the soak starts both sidecars and real trainer runs itself
- [ ] ~1 GB free disk for the per-run `TMPDIR` and `.test-results/soak-*.json`

**Plan**
1. **[you or me]** `conda run --no-capture-output -n mlforge-dev npm run test:soak -- --seconds 3600` (run it detached: `nohup setsid … &`, then `tail`). Optionally a second run at `--seconds 14400` overnight.
2. **[me]** read `.test-results/soak-<timestamp>.json`/`.md`: projected RSS growth per hour (fail > 200 MB/h), fd/thread slope (fail > 0.05/min), trainer runs all `done`, no 5xx, no died process.
3. **[me]** if the LLM sidecar's projected growth (25–75 MB/h at 90 s) persists at one hour, bisect with `/health.diag` (which map/handle grows) and fix; record the numbers in `LIMITATIONS.md` §5.

**Done when**: a ≥ 3600 s run passes; the measured slope replaces the "25–75 MB/h on 90 s" caveat in `LIMITATIONS.md` and `TODO.md` Phase 52.

## 4. Condition 6 — Claude subscription provider and real opencode

**Resources checklist**
- [ ] `opencode` CLI installed (tested with 1.18.15) and logged in; a model with quota, e.g. a ScaDS/`tud-ai/...` model via the key in `~/.local/share/opencode/auth.json` (the free `opencode/*` tier refuses non-OpenCode callers with HTTP 403 — verified)
- [ ] `claude` CLI installed and logged in (OAuth / Max subscription) for kind `subscription`
- [ ] optional: an Anthropic API key for kind `anthropic`, an OpenAI-compatible key/URL for `openai-compat`
- [ ] awareness: these tests spend quota / tokens; keep prompts tiny; no secrets in prompts or logs

**Plan**
1. **[me]** add the new live-provider suite (a script plus an npm script, both to be added; registered with `needs: ['llm-key']`, i.e. BLOCKED unless `SPINOML_LIVE_LLM=1`): for each configured live provider run one tiny turn that must call `add_layer` and assert the `action` event, the tool result, `done`, no secret in the SSE stream; for opencode also an unknown model → explicit error; `subscription` only if `claude` is on `PATH`. Never part of default CI.
2. **[you]** `SPINOML_LIVE_LLM=1 SPINOML_OPENCODE_TEST_MODEL=<provider/model> npm run verify:opencode` (existing: real opencode chat + clean bogus-model error), then the new live-provider suite with SPINOML_LIVE_LLM=1 once it exists.
3. **[you]** in the real app (see §2): one chat turn on each provider you actually use.

**Done when**: `verify:opencode` and the new live-provider suite pass on your machine (paste the table into `FINAL_RELIABILITY_REPORT.md` §90.3); `LIMITATIONS.md` §7.8 rows "NOT driven end-to-end" updated per provider.

## 5. Conditions 2 and 1 — real cluster and CUDA

The two suites that should prove these exist only as placeholders (`remote-live`: "no live-cluster suite exists";
`hardware-cuda`: "no dedicated CUDA suite"). **They have to be written first** (§7). If your cluster has a GPU partition, run
both in the same allocation.

### Resources checklist — cluster
- [ ] an ssh alias in `~/.ssh/config` that works with `BatchMode=yes` (key/agent/GSSAPI, **no password or 2FA prompt**): `ssh -o BatchMode=yes <alias> true` must exit 0
- [ ] a remote root with quota (per your notes: data lives in the SC home; `/work2` only when the quota is full) — the live suite creates and deletes `spinoml-live-<timestamp>/` inside it
- [ ] on the login node: `python3` ≥ 3.10 with `venv`, outbound pip access (or a mirror/module), `git` ≥ 1.8.5 (manifest git claim), `sbatch`/`squeue`/`sacct`/`scancel`, `setsid`, `nohup`
- [ ] SLURM: account, partition (and a GPU partition/`--gres` string for #1), QoS, a time limit (≤ 15 min per job is enough), a CPU/memory request the site allows
- [ ] permission to start a small python process on the login node for the sidecar (the 2-minute shell cap applies) and to open a local ssh tunnel on port 7424
- [ ] a note of what the suite may write and how to clean up (`rm -rf <root>/spinoml-live-*`, `scancel -n spinoml-live-*`)

### Resources checklist — CUDA
- [ ] an NVIDIA GPU (≥ 8 GB VRAM recommended), driver installed, `nvidia-smi` works
- [ ] a CUDA-enabled torch wheel matching the driver in an environment (`python -c "import torch; print(torch.cuda.is_available())"` → `True`); the same environment for `pyg`
- [ ] ideally a second GPU model/driver (or a second node) to measure cross-device variance — optional
- [ ] 30–60 min of GPU time

### Plan
1. **[me]** write the live suites (§7), dry-run them locally against the existing fakes so the harness itself is proven (the CUDA suite prints `SKIPPED  CUDA` on a CPU box exactly like `verify:reference`).
2. **[you]** `SPINOML_REMOTE_TESTS=1 SPINOML_REMOTE_ALIAS=<alias> SPINOML_REMOTE_ROOT=<dir>` plus the command of the new remote-live suite (named when it is written; today `remote-live` is only a BLOCKED placeholder), in the GPU allocation also the CUDA suite.
3. **[you]** launch the real app once against the same alias (§2 checklist step "remote sidecar badge"): bootstrap, tunnel on 7424, dataset smoke on the remote sidecar, a remote run, stop, restart the app, find the run again.
4. **[me]** read the outputs, fix what breaks (expect site quirks: shell, quotas, `git` version, `systemd-logind`), update docs.

**Done when**
- [ ] `remote-live` passes: connection test, probe, bootstrap/deploy list, a direct remote run and a SLURM run (submit → squeue/sacct states → `done` with a valid integrity block), cancel, delete, app-restart recovery of both, cleanup leaves nothing behind;
- [ ] `hardware-cuda` passes: the three reference experiments on CUDA match CPU within the documented tolerance, two same-seed CUDA runs differ by a measured, recorded amount (this number replaces "not claimed" in `REPRODUCIBILITY.md` §5), mixed precision (bf16/fp16) smoke, `cudnn` flags recorded in the manifest;
- [ ] `RISK_REGISTER.md` R044 and R021 closed with evidence, `RELEASE_GATE.md` §Remote Execution `[~]` → `[x]`, `REMOTE_TRAINING.md` §9 rewritten with what was actually verified.

## 6. Shared resources checklist (one place)

**Accounts / access**
- [x] GitHub write access to `christophmanitz/spinoml`
- [ ] HPC login (ssh alias, key auth, no prompts) + SLURM account/partition/QoS (+ GPU partition)
- [ ] opencode model with quota; `claude` CLI login; optional Anthropic / OpenAI-compatible keys

**Hardware**
- [ ] desktop session with WebKitGTK for the Tauri window
- [ ] idle machine for 1 h (soak)
- [ ] NVIDIA GPU (own box or the cluster's GPU partition)

**Software on the machine that runs the checks**
- [x] conda env `mlforge-dev` (Python 3.12, torch 2.12 CPU, pyg 2.8, Node 20, Rust 1.96) — for CUDA a second env with a CUDA torch build
- [ ] Tauri system libraries (see §2); a Chromium/Chrome for `test:webview-csp` (present: `/snap/bin/chromium`)
- [x] `gh` CLI, `git`, `npm`; `opencode` (installed), `claude` (installed here — it is the CLI running this session)
- [ ] cargo-audit / pip-audit are NOT installed (the Phase 79 audit used throw-away installs) — decide whether to schedule `npm audit` / `cargo audit` in CI

**Decisions only you can make**
- [ ] which cluster/alias/partition is the acceptance target; which GPU counts as "the" CUDA reference
- [ ] whether `SPINOML_REQUIRE_SCOPE=1` should be set by the app once the window smoke is green (fail-closed before a workspace is chosen)
- [ ] whether browser-dev mode should also require a token (`SPINOML_REQUIRE_TOKEN=1` is available)
- [ ] whether to add a scheduled (weekly) dependency-audit workflow

## 7. Work items I can do now (no hardware needed)

| # | Item | Output | Closes / prepares |
|---|---|---|---|
| 7.1 | Fix the six CI findings of §1 | green `ci.yml` run, corrected header comments | condition 4 |
| 7.2 | a new remote-live script or Rust `#[ignore]` live tests calling the plain `ssh_*` async functions (`ssh_start_training_run`, `ssh_training_run_status`, `ssh_stop_training_run`, `ssh_delete_training_run`, `ssh_remote_training_capabilities`, SLURM submit/reconcile) from env-gated tests; the bootstrap (`run_bootstrap` needs an `AppHandle`) either through Tauri's mock app (`tauri` `test` feature) or stays a manual step in §2 | a new npm script replacing the BLOCKED placeholder, env: `SPINOML_REMOTE_TESTS=1`, `SPINOML_REMOTE_ALIAS`, `SPINOML_REMOTE_ROOT`, optional `SPINOML_REMOTE_SLURM_*` | condition 2 |
| 7.3 | a new hardware-cuda script: CPU-vs-CUDA numerical comparison of the three reference experiments, same-seed repeatability on CUDA with the measured spread, bf16/fp16 smoke, `cudnn` flags, GPU memory sample; prints `SKIPPED  CUDA` without a GPU | replaces the BLOCKED placeholder | condition 1 |
| 7.4 | a new live-provider test script (§4) | gated live provider suite | condition 6 |
| 7.5 | new document TAURI_SMOKE under docs/engineering (§2) | the manual checklist + an evidence helper | condition 3 |
| 7.6 | Register the new suites in `scripts/suites.ts` with `needs` so they stay BLOCKED (not failing) in CI | `npm run ci -- --check` green | all |

## 8. Definition of done for the gate

The verdict may change from CONDITIONAL to **PASS** only when all of these hold, each with evidence committed:

- [ ] CI green on `main` and on a PR (§1)
- [ ] Tauri window checklist ticked, fallback not needed or the CSP fixed (§2)
- [ ] ≥ 1 h soak within the bounds (§3)
- [ ] live LLM providers you rely on pass (§4)
- [ ] `remote-live` and `hardware-cuda` pass on the target site / GPU (§5)
- [ ] `RELEASE_GATE.md` rows flipped with the evidence, `RISK_REGISTER.md` R021/R044 closed, `FINAL_RELIABILITY_REPORT.md` §90.1 re-issued with a new date and test table, `LIMITATIONS.md` §6 emptied of the items above

Suggested calendar: day 1 — §1 and §7.1/7.5 (me), §3 started in the evening; day 2 — §2 and §4 (you), §7.2–7.4 (me); day 3 — §5 on the cluster
(you), fixes (me), re-issue of the report.
