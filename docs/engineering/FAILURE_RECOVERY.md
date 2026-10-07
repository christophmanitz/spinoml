# SpinoML — Failure & Recovery

> Phase 80 deliverable. State-by-state: what happens during a crash, a kill,
> a remote disconnect, an OOM, or a stale read, and what the user / app
> must do. Every claim is anchored in code (file:line) or in a real npm
> test script (the only thing you can trust when a run disappears).
> Written 2026-10-07 against HEAD; the run/state machine lives in
> `sidecar-torch/training_template.py`, `src-tauri/src/training.rs` and
> `src-tauri/src/ssh.rs`.

---

## 1. The state machine (the truth)

A run dir (`experiments/runs/<run_id>/`) holds the **single source of truth**
as files: `status` (one word), `events.jsonl` (append-only JSONL),
`metrics.json`, `checkpoints/best.pt` + `last.pt`, `run.json`, `model.py`,
`model.spinoml`, `manifest.json`, optionally `pid`, `stdout.log`,
`stderr.log` (executor launches) or `slurm-<jid>.out`/`slurm-<jid>.err`
(SLURM launches). After the app exits, the run keeps going (setsid/nohup
+ detached `setsid python -u train.py …` in `src-tauri/src/training.rs:548`).
Re-opening the app re-reads the files — there is no live IPC.

`status` is governed by `transition_status(next)` in
`sidecar-torch/training_template.py:238` and the Rust reader
`reconcile_status` in `src-tauri/src/training.rs:98`. The legal transitions
are exactly:

| current    | → next          | who writes it                                              |
| ---------- | --------------- | ---------------------------------------------------------- |
| (absent)   | `queued`        | Rust executor on launch (`start_training_run:539`)         |
| `queued`   | `running`       | trainer, first line of `main()` (`training_template.py:2124`) |
| `queued`   | `cancelled`     | user pressed Stop before the trainer started (Rust: `stop_training_run:581`, ssh mirror: `ssh_stop_training_run:1328`) |
| `running`  | `done`          | trainer, after the integrity gate passes (`training_template.py:2768`) |
| `running`  | `failed`        | trainer, `fail(stage, msg)` (`training_template.py:307`)   |
| `running`  | `cancelled`     | (a) cooperative: epoch-boundary cancel check (`:2586`), or (b) cooperative: signal `_on_cancel` (`:2111`), or (c) Rust `stop_training_run:581` + SIGTERM (`:585-589`) |
| `done` / `failed` / `cancelled` | anything | **REJECTED.** Terminal states are final; `_TERMINAL_STATUSES` (`:219`) and `reconcile_status` (Rust: `:98`) both guard them. A late event cannot resurrect a finished run. |
| `done`/`failed`/`cancelled` | same word (idempotent rewrite) | the writer of that word (e.g. cooperative cancel arriving after `done`) — accepted but a no-op |
| _STATUS_UNREADABLE | anything | rejected — an unreadable status file means "unknown state", and we never invent a transition (`:245`). |

If the status file is missing, the local reader treats it as "not started
yet" (`<unreadable>` only fires on a real read error, see
`training_template.py:226-235`); the Rust reader renders the absent state
as `"unknown"` (`training.rs:99-100`). A `running` or `queued` status with
no live process is reported as `failed` by `reconcile_status` (`:101-105`)
— this is how a half-launched run (setsid succeeded but python died before
the first write) shows up.

Failure stages the trainer writes into the `run.failed` event:

| stage           | where (`training_template.py`)                                  | meaning |
| --------------- | --------------------------------------------------------------- | ------- |
| `numeric`       | `:342` (loss/metric finiteness), `:2622` (gradient finiteness)   | Non-finite loss / gradient / metric. The run is unusable; the trainer refuses to report success (Phase 25). |
| `config`        | `:2131`, `:2139`, `:2787`                                        | Cannot read `run.json`, or the status file became unreadable before finalisation. |
| `import-torch`  | `:2171`                                                          | The python interpreter could not import torch. |
| `snapshot`      | `:2232`                                                          | The frozen `model.spinoml`/`model.py` in the run dir drifted from the hashes recorded at launch (Phase 20). |
| `split`         | `:2154`, `:2314`, `:2333`, `:2377`                               | Unknown `split_strategy`, not enough rows, train/val overlap, or empty training loader. |
| `dataset`       | `:2308`                                                          | Dataset load failed (corrupt file, missing target/feature column, NaN/inf inputs — see `load_tabular:1386`). |
| `model`         | `:2385`                                                          | `Model()` raised (bad param, dimension mismatch at first forward). |
| `resume`        | `:2558`                                                          | `safe_torch_load(resume_from)` raised; the resume target is unreadable / incompatible. |
| `train`         | `:2730`                                                          | Anything the training loop raised — DataLoader worker crash, OOM, etc. |
| `validate`      | `:2247`, `:2255`, `:2530`                                        | eval-only run: missing `checkpoint_from`, can't load checkpoint, eval raised. |
| `integrity`     | `:2519`, `:2766`                                                 | After the loop finishes and the integrity gate runs (Phase 73, see §6 below): a required artifact is missing/invalid. The run NEVER ends `done`. |
| `fatal`         | `:2813`                                                          | Last-resort catch in `__main__` — only reached if an exception escapes `main()` itself. |

Evidence (the suites that hit each path): `verify:failures` (NaN loss →
`numeric`; SIGKILL → `fatal`/`train`), `verify:checkpoint` (corrupt
checkpoint → `resume`), `verify:integrity` (sabotaged artifact → `integrity`),
`verify:states` (transition matrix), `verify:cancel` (terminal-state
shielding), `verify:traingen` (snapshot drift → `snapshot`).

## 2. Local training crash / kill / OOM / power loss

### What happens

The detached launch (`training.rs:541-557`, `setsid python -u train.py
> stdout.log 2> stderr.log < /dev/null & echo $! > pid`) keeps the trainer
alive after the app exits, so an **app close, app crash, or app restart
does NOT stop the run** — the python process is reparented to init. Only
SIGTERM/SIGKILL of the recorded pid (or a system reboot) takes it down.

| Cause                              | Visible effect                                                  | How the run is reported next open |
| ---------------------------------- | --------------------------------------------------------------- | --------------------------------- |
| App closed / restarted             | nothing (setsid detached)                                       | `training_run_status` reads `status` + `pid` and reports live/queued/done correctly (`training.rs:468-478`). |
| Process SIGKILL (OOM, `kill -9`)   | no `status` rewrite; `events.jsonl` stops mid-line; `checkpoints/` may contain a half-written `last.pt`. | `reconcile_status` (`training.rs:101-105`) reports `running` (or `queued`) with `alive=false` as `failed`. |
| Power loss mid-epoch               | as SIGKILL; same `status` is whatever was written before the loss; `last.pt` either exists (last completed epoch) or not. | `failed` if the status said `running`/`queued`; otherwise the terminal status wins. |
| Power loss mid-checkpoint          | `_atomic_save` (`:404-430`) writes to `last.pt.tmp` then `os.replace`; a power loss between the two leaves the OLD valid `last.pt` in place and a leftover `last.pt.tmp` (a crashed save never silently corrupts the previous good checkpoint). | `failed`, last good epoch is resumable. |
| OOM (kernel kill)                  | as SIGKILL. | `failed`. |
| Internal exception (e.g. `RuntimeError` in the training loop) | `except Exception` in the epoch loop (`training_template.py:2717-2730`) saves the LAST COMPLETED epoch's checkpoint, calls `fail("train", …)` and `sys.exit(1)`. The terminal state is `failed`. | `failed`, resumable (see §5). |

### What the user must do

1. **Re-open the run in the UI.** The Run-Detail modal reads
   `events.jsonl` + `metrics.json` + `status` (via
   `read_training_run_file`, `training.rs:481-497`) and shows the
   loss curve up to the point of death, plus the failure `stage`
   recorded in the last `run.failed` event.
2. **Inspect the cause.** The `stage` field is the primary diagnostic;
   `stderr.log` (when present, i.e. executor-launched runs) has the
   python traceback up to the kill. `events.jsonl` is your honest
   record — it is `fsync`-flushed per line (`:202-205`) and never lies
   about what happened.
3. **Resume if `run.resumable` is true.** See §5; the banner in
   Run-Detail offers "Resume run" only when the metrics.json
   `resumable: true` entry is set (`_compute_resumable`,
   `training_template.py:1115-1195`).
4. **If the run is `failed` with no `last.pt`**, nothing to resume.
   Adjust the graph (smaller model, fewer workers, lower batch size)
   and start a new run. The failed dir stays on disk until you delete
   it (`delete_training_run`, `training.rs:636-648`) — it cannot be
   deleted while `pid_of(dir).map(is_alive)` is true (`training.rs:644-646`).

### What the system does for you

- `_atomic_save` (`:404-430`) makes a power loss mid-write a non-event:
  the previous good `last.pt` survives. A leftover `last.pt.tmp` is
  removed in the `finally` block; it is the only orphaned temp file
  the trainer leaves behind.
- A SIGTERM-driven cooperative cancel saves `last.pt` at the last
  COMPLETED epoch before unwinding (`_Cancelled` handler, `:2704-2716`),
  so a stopped run is resumable too.
- A failing `fail(stage)` path tries to save `last.pt` before the
  terminal write (`training_template.py:2718-2730`), so an
  exception-driven failure is ALSO resumable whenever at least one
  epoch completed.

## 3. Cancellation (user-pressed Stop)

### Two paths, one outcome

The UI calls either `stop_training_run` (local, `training.rs:566-592`)
or `ssh_stop_training_run` (remote, `ssh.rs:1319-1343`). Both do the
same two things in order:

1. **Write `cancelled` into `status` BEFORE any signal** (cooperative
   cancel — the trainer checks `status` at every epoch boundary,
   `training_template.py:2586`, and unwinds to `_finish_cancel`).
2. **SIGTERM the process group** (forceful — `setsid` made python the
   session leader so `kill -TERM -<pid>` reaches dataloader workers
   too). The ssh mirror does `kill -TERM -"$pid"` then `kill -TERM "$pid"`
   (`ssh.rs:1336-1338`).

The cooperative path is the one that lands in `cancelled` cleanly; the
forceful path is the safety net for a stuck trainer. Both routes
share the same terminal-shielded recorder `_finish_cancel`
(`training_template.py:281-298`) which:

- Goes through `transition_status("cancelled")` (so it can NEVER
  overwrite a `done`/`failed` status — a stop pressed AFTER `done` is
  a silent no-op and the run stays `done`).
- Emits exactly one `run.cancelled` event (the `_CANCEL_RECORDED`
  sentinel at `:271` makes duplicate signals idempotent).
- Writes `metrics.json` with `status: "cancelled"` and the epoch at
  cancellation.
- Marks the run `resumable` via `_record_terminal_resumable` (`:297`)
  IF `last.pt` exists, is valid and matches this run's model.

### The cancellation state machine — exhaustive

| Trainer state             | User presses Stop | Result                                                |
| ------------------------- | ----------------- | ----------------------------------------------------- |
| pre-launch (`queued`)     | Stop              | `cancelled` written, python never started → exit 0, `run.cancelled` event recorded (`main()` at `:2124-2127`) |
| loading dataset, building model | Stop         | trainer not in epoch loop yet; the next cancel check at the first epoch boundary sees `cancelled` and unwinds; OR the SIGTERM lands during `import torch`/model construction — `_Cancelled` is caught by `__main__` (`:2803-2808`), `_finish_cancel(0, …)` runs |
| mid-epoch (training)      | Stop              | SIGTERM → `_on_cancel` → `raise _Cancelled` → caught at `:2704` → save last completed epoch's checkpoint, `_finish_cancel(epoch, "cancelled (signal N)")` |
| at epoch boundary         | Stop              | cooperative check at `:2586` wins first; SIGTERM lands on an already-cancelled trainer and is ignored by `_SHUTDOWN["v"]` (`:274-278`) |
| post-`done`               | Stop              | `_read_status() == "done"` → `stop_training_run` returns OK without writing (`training.rs:577-578`); ssh mirror same (`ssh.rs:1330-1333`) |
| post-`failed`             | Stop              | as `done` (terminal is final) |
| post-`cancelled`          | Stop (double)     | idempotent — `_CANCEL_RECORDED` suppresses the duplicate event (`training_template.py:290-292`), the metrics rewrite is idempotent |

Evidence: `verify:cancel` (28 checks: signal-at-boundary, double-stop,
post-done shielding), `verify:states` (transition matrix + terminal-state
guards), `verify:events` (no stale trailing events, first-terminal-wins).

### What the user must do

Nothing — cancellation is fire-and-forget. The UI's Run-Detail will
update the status on the next poll. To inspect: read
`experiments/runs/<run_id>/status` (`cancelled`), the last event in
`events.jsonl` (`run.cancelled`), and `metrics.json`
(`{"status": "cancelled", "epoch": …, "resumable": {...}}`).

If you need to hard-kill a hung trainer that ignored SIGTERM:
`kill -KILL <pid>` (the pid is in `<run_id>/pid` and visible via
`kill -0 <pid>` in `is_alive`, `training.rs:276-285`). The run will
then look `running` on disk but `alive=false`; the next
`training_run_status` call reconciles that to `failed` via
`reconcile_status` (`training.rs:101-105`).

## 4. Torch sidecar crash / hang

### What happens

The torch sidecar is a `ThreadingHTTPServer` (`sidecar-torch/main.py:32`)
on `127.0.0.1:7421`. Per-connection socket inactivity cap is
`REQUEST_TIMEOUT = float(os.environ.get("SPINOML_TORCH_TIMEOUT", "30"))`
seconds (`main.py:108`, applied at `:1139`). Every request is processed
on a worker thread — one hung request does NOT block another, but a
true process crash takes the whole sidecar down.

The Rust shell manages the lifecycle (`spawn_managed` in
`src-tauri/src/lib.rs:108-140`): on launch it generates a per-launch
token and writes it as `SPINOML_SIDECAR_TOKEN`/`SPINOML_REQUIRE_TOKEN=1`
into the child's env. On exit (`shutdown_sidecars`, `:142-157`) it
SIGKILLs both managed children.

### Bad-arg / startup failures

- Busy port: sidecar exits 3 (`main.py:1139` → `EADDRINUSE`).
- Invalid `SPINOML_TORCH_PORT`: sidecar exits 2
  (`int('abc')` → uncaught `ValueError`, surfaced in stderr; the Rust
  shell reports "spawn failed").
- Missing `python`: same — Rust can't spawn, the badge shows
  `LLM: offline` / `torch: offline`.

### Process crash mid-request

A worker thread can crash on a bad payload; the server's exception
handling in `do_POST` (`:1213-1240`) catches and returns a structured
error JSON (`error_code` populated from `_err`, `main.py:432`). The
TCP socket closes; the client sees a connection-reset. The sidecar
process itself stays alive — other requests are unaffected.

A python crash that takes the whole process down (e.g. a segfault,
OOM-killed by the OS, or `os._exit`) is NOT auto-restarted by the Rust
shell in production. The user must restart the app, or in Tauri dev
press the "Restart sidecars" affordance (the Rust status re-polls).

### Hang / unresponsive

The 30s `REQUEST_TIMEOUT` (`:108`) closes the socket on inactivity.
The frontend's `sidecarFetch` (in `src/sidecars/auth.ts`) has its own
retry-once policy and surfaces a structured error. The UI's
"Inference: offline" badge reflects `/health` polling; an actually-hung
sidecar (server thread blocked on a slow `exec`) keeps `/health`
responsive until the OS scheduler gives it a turn — a stuck
`/infer` does NOT make `/health` look offline.

### What the user must do

| Symptom                                              | Action |
| ---------------------------------------------------- | ------ |
| `LLM: offline` / `torch: offline` in the header      | restart the app. In Tauri mode, the sidecars respawn automatically on the next launch (`spawn_managed`); in browser dev, run `npm run sidecar:torch` and `npm run sidecar:llm`. |
| `/infer` returns 500/structured error                 | fix the model / dataset per the `error_code`; the sidecar is still alive. |
| `/infer` hangs > 30s                                  | socket closes; the request fails client-side; the worker thread is freed. If the hang is recurring, the sidecar has a real bug — capture `stderr.log` and report. |
| Sidecar exited (port busy / bad port)                 | kill the lingering process (`fuser -k 7421/tcp` on Linux) or pick a different `SPINOML_TORCH_PORT`. |

Evidence: `test:process-lifecycle` (75 rows: port busy exit 3, invalid
port exit 2, SIGTERM/SIGINT without orphans, 25 start/stop cycles
fd-stable, same-port restart), `test:sidecar-auth-torch` (144 rows
vs a real process), `test:robustness` (11-case matrix: stall cap,
abort resistance, kill→restart recovery).

## 5. Resumable runs — what makes a run resumable (and what blocks it)

**Important:** SpinoML does **not** auto-resume. The user must start a
new run with `resume_from = experiments/runs/<run_id>/checkpoints/last.pt`.
The terminal status stays `failed`/`cancelled`; the
`metrics.json["resumable"]` entry is a flag the UI banner reads, not a
new status value (Phase 74, `_record_terminal_resumable`,
`training_template.py:1198-1234`).

### Conditions for `resumable: true`

Computed in `_compute_resumable(final_status)`
(`training_template.py:1115-1195`). ALL must hold:

1. The final status is `failed` or `cancelled` (a `done` run is never
   resumable — it already finished).
2. `checkpoints/last.pt` exists, non-empty, is a valid zip.
3. `last.pt` ≤ 256 MB (skip threshold; see `_CKPT_LOAD_SKIP_BYTES =
   256 * 1024 * 1024`, `training_template.py:928`). Larger files skip
   the deep load+keys check; the deep check is reported as "checkpoint
   load verification skipped (size)" in the gate, but the resumable
   verdict is `false` (`:1139-1141`).
4. `safe_torch_load(last.pt)` succeeds — i.e. `weights_only=True`
   accepts the contents (Phase 47; PyG/numpy allow-list is
   pre-registered in `register_safe_globals`, `:80-148`).
5. The checkpoint is a dict with keys `{"model_state", "optim_state",
   "epoch", "global_step", "config"}`.
6. **The checkpoint's `config.snapshot.graph_sha256` and
   `model_py_sha256` match THIS run's `manifest.json["hashes"]`** —
   the resume target must belong to this run, not to a different model.
   If the manifest is unreadable, the run is reported as `false`
   with `reason: "run hashes unavailable (manifest unreadable)"`
   rather than falsely claiming resumability (`:1179-1184`).

### Conditions that block resume (and the reason string)

| reason                                                | cause |
| ----------------------------------------------------- | ----- |
| `"run completed"`                                     | final status was `done`. |
| `"no checkpoint"`                                     | `last.pt` missing or zero bytes. |
| `"checkpoint corrupt"`                                | not a zip, load failed, missing keys, not a dict. |
| `"checkpoint load verification skipped (size)"`       | `last.pt` > 256 MB (skip threshold). |
| `"run hashes unavailable (manifest unreadable)"`      | `manifest.json` is unreadable / missing. |
| `"checkpoint belongs to a different model"`           | The hash in `config.snapshot.*` does NOT match this run's manifest — typically the checkpoint was copied from another run dir. |
| `"resumable computation failed: <ExceptionType>"`     | The function itself raised; recorded as `false`, never crashes. |

### How `resume_from` is consumed

`resume_from` is read from `cfg.get("resume_from")` in
`main()` (`:2536`). It can be:

- A workspace-relative path (`experiments/runs/.../checkpoints/last.pt`).
- An absolute path (`/scratch/.../last.pt`).
- It is loaded via `safe_torch_load`; the loaded dict must carry
  `model_state` (required), `optim_state` (required for optimizer
  continuity), `sched_state` (optional), `best_val` (default `inf`),
  `epoch` (int — start_epoch = epoch + 1), `global_step` (int), and
  `rng` (the dict from `_rng_state()` at the time of save — restored
  best-effort, see below).

The fail-loud on bad resume is `fail("resume", ...)` at
`:2558`, which writes `run.failed stage=resume`. There is NO silent
fresh-start from a broken resume target — by design, a failed resume
must be visible.

### RNG restoration — what is and is not recovered

Captured at every checkpoint (`_rng_state`, `:346-366`):
- `torch.get_rng_state()` — torch CPU RNG
- `torch.cuda.get_rng_state_all()` — every visible CUDA device, when
  available (silently skipped on a CPU build)
- `numpy.random.get_state()` — NumPy (if installed)
- `python` (stdlib `random`) — always

Restored best-effort at resume (`_restore_rng`, `:369-401`); the
record is emitted as `run.resumed` with `rng_restore`:

```
{"torch": "restored" | "absent in checkpoint" | "failed: <reason>",
 "torch_cuda": "...", "numpy": "...", "python": "..."}
```

A resume that could not restore one or more streams is NOT a bitwise
continuation — the metrics will differ. The user MUST look at
`rng_restore` to know whether their resumed run is reproducible.

Evidence: `verify:checkpoint` (45 checks: full state incl. rng +
config, cancel-saves, atomic-write crash sim, corrupted-ckpt rejection,
resume continuity).

## 6. Numerical failures

`require_finite(name, value, where)` at `training_template.py:333-343`
is called at four points:

1. **train loss, every batch** (`:2611`): non-finite → `fail("numeric",
   "non-finite train loss at epoch=N step=M")`.
2. **gradients, before every optimizer.step()** (`:2618-2625`): any
   `p.grad` with a NaN/Inf component → `fail("numeric",
   "non-finite gradient after backward for '<param>' at epoch=N step=M")`.
3. **val loss / val acc / per-metric** during the periodic validation
   pass (`:2660-2665`): non-finite → `fail("numeric", "non-finite val
   loss at epoch=N")`. The monitored curve (early stop, best-val,
   checkpointing) is unusable after this, so the run fails loudly
   instead of reporting success on garbage.
4. **eval-only validation** (`/validate`, `:2490-2494`): same gate.

A NaN that slips past the gate (e.g. in an unsupported third-party
op) is impossible to catch — by design, anything that escapes
`require_finite` is a separate bug.

What the user sees: `run.failed` with `stage="numeric"`. The checkpoint
at the last COMPLETED epoch is saved (`training_template.py:2718-2730`),
so the run is resumable AFTER fixing the cause (lower lr, gradient
clipping via the `GradientClipping` callback, fp16 → bf16, fix the
data pipeline). Numeric failures are NEVER `done` (Phase 25 invariant;
see `verify:failures`).

## 7. Corrupt / missing run-dir files

Each reader is fail-soft (returns empty / `unknown`) rather than
crashing the UI:

| File                 | Who reads                                                | On missing/corrupt                                                       |
| -------------------- | -------------------------------------------------------- | ------------------------------------------------------------------------ |
| `status`             | `training_run_status` (`training.rs:468-478`); `reconcile_status` for local reads; the ssh mirror at `ssh.rs:1247-1258` | absent → `"unknown"` (`training.rs:100, 290`); corrupt (any IO error) → `"unknown"` and `reconcile_status` reports `running`/`queued` with `alive=false` as `failed`. |
| `events.jsonl`       | `list_training_runs` (filtered to `epoch.end`/`run.done` only — `filter_summary_events`, `training.rs:352-357`); `read_training_run_file`; `verify:events` | absent → empty string; corrupt lines are skipped line-by-line (`scan_events`, `training.rs:180-212`); the run still shows the loss/epochs it could parse. |
| `metrics.json`       | `summarize` (`training.rs:293-311`), `RunSummary::from_parts` (`:118-175`) | absent → loss fallback to `events.jsonl` (`:159-169`); corrupt → parse returns Null and the field defaults to absent — the run is reported without a `best_val_loss` rather than crashing. |
| `checkpoints/best.pt` | `safe_exists` (`training.rs:336-347`); `_checkpoint_ok` (`training_template.py:946-979`) | missing → `has_checkpoint=false` in the UI; ≥256 MB → gate note "checkpoint load verification skipped (size)"; corrupt zip → `_checkpoint_ok` returns `(False, "not a valid zip" | "load failed: …")`, the integrity gate adds it to `invalid`, the run ends `failed` stage `integrity` (Phase 73). |
| `run.json`           | `summarize` + the trainer itself                         | trainer-side: corrupt → `fail("config", "cannot read run.json: …")` (`training_template.py:2139`). |
| `model.py` / `model.spinoml` | snapshot check, `:2229-2236` (`_verify_snapshot`)    | hash mismatch → `fail("snapshot", "run dir artifacts mutated after launch — …")` (Phase 20, REFUSE to train on drifted code). |
| `manifest.json`      | the trainer's manifest pipeline                          | unreadable → recorded as `manifest.error` event, the run continues (manifest is best-effort documentation, NOT a gate); the run is still `done` if everything else passes. |
| `pid`                | `pid_of` (`training.rs:269-273`); `is_alive` (`:276-285`)| absent → `alive=false`, status reconciled to `failed` if it claimed `running`/`queued`. |
| `stdout.log` / `stderr.log` | only required when `pid` exists (`_verify_run_integrity:1084-1096`) | direct `python train.py` launches (the verify harnesses) carry no `pid` and therefore need no logs. Executor launches with `pid` MUST have non-empty `stdout.log`+`stderr.log`. |

Evidence: `verify:integrity` (each sabotage path: missing/empty/
corrupt checkpoint, bad metrics, missing events/manifest end FAILED
stage `integrity`, never done), `verify:events` (terminal truncation,
stale-read dropping), `verify:states` (state machine).

## 8. SSH drop during a remote run / app restart during a remote run

### How the run is re-found

Remote launches leave the same files on the remote host that local
launches do (`run.json`, `model.spinoml`, `model.py`, `train.py`,
`status`, `pid`, `events.jsonl`, `manifest.json`, `checkpoints/`).
The `pid` file holds either the linux pid (setsid launcher,
`ssh.rs:1041-1044`) or `slurm:<jobid>` (sbatch launcher,
`ssh.rs:1011-1015`). After an SSH drop or app restart, re-opening the
workspace triggers:

- `ssh_list_training_runs` (`ssh.rs:1128-1231`) — one ssh round-trip
  per run dir, reading `pid`, `checkpoints/best.pt`, `status`, `run.json`,
  `metrics.json`, and `epoch.end|run.done` lines from `events.jsonl`.
  Pid liveness is checked in-band:
  - `slurm:<jid>` → `squeue -j <jid> -h -o '%T'`; if squeue prints a
    state the job is alive, otherwise the ssh mirror falls through to
    sacct.
  - bare numeric pid → `kill -0 <pid>` on the remote.
- `ssh_training_run_status` (`ssh.rs:1234-1299`) — for SLURM jobs
  this additionally fetches the live `squeue` `%T` state, and once the
  job has left the queue, the `sacct` state. The result is run through
  `training::reconcile_slurm_status` (`training.rs:430-465`) to get
  the user-visible status.

There is no localStorage cache of remote run state — `verify:recovery`
asserts this explicitly (38 checks).

### SSH dropped while a run is running

| Backend | Live status recovery after ssh returns                                          |
| ------- | ------------------------------------------------------------------------------- |
| direct (setsid) | The python process is reparented to init on the HPC login node (`nohup setsid`) — it keeps running. On reconnect, `kill -0 <pid>` succeeds, `is_alive=true`, status reads as `running`. |
| SLURM   | The job is still in the scheduler — `squeue -j <jid> -o '%T'` shows `RUNNING`/`PENDING`, status reads as `running`/`queued` via `reconcile_slurm_status`. |
| Both    | The ssh round-trip fails (`ssh_exec` returns an error string). The UI shows the SSH badge as disconnected, the run list shows `unknown` (no live info available). Once the user reconnects, the live state reappears. The run itself was NEVER affected by the drop. |

### App closed / restarted

Same — `setsid`/`nohup` (direct) or sbatch (SLURM) survive an app
exit. Re-opening the app re-reads the on-host files; the run keeps
going. `verify:recovery` exercises close→restart→reconnect.

### SSH lost mid-submission

The submission path is idempotent (Phase 33): `ssh_start_training_run`
(`ssh.rs:927-1048`) creates the run dir as the SUBMISSION CLAIM
(`mkdir` of the run dir). The four markers it may see:

| marker                    | meaning                                                              |
| ------------------------- | -------------------------------------------------------------------- |
| `MLF_CLAIM_FAILED`        | `experiments/runs` could not be created — fatal, no retry            |
| `MLF_CREATED`             | fresh claim, proceed to write files + launch                        |
| `MLF_ALREADY_LAUNCHED`    | the dir exists AND has a non-empty `pid` — assume the previous submission succeeded (the response was lost in transit); return OK without re-launching |
| `MLF_EXISTS_INCOMPLETE`   | the dir exists but `pid` is empty — refuse duplicate launch (a prior attempt died before writing its pid); let the user delete the orphan dir or finish it manually |

The ssh command ordering is `mkdir -p <runs>; mkdir <dir>; …` so the
two makedirs together act as a single atomic claim (`:957-976`).

Evidence: `verify:submission` (7 checks: idempotency under retries),
`verify:recovery` (38 checks), `verify:slurm` (33 checks: squeue
mapping, sacct handoff, scheduler kills).

## 9. SLURM state → app status reconciliation

`reconcile_slurm_status(status_raw, squeue_state, sacct_state)`
(`training.rs:430-465`):

| `squeue %T` | `sacct State` | trainer `status` file | app shows       |
| ----------- | ------------- | --------------------- | --------------- |
| `PENDING`/`CONFIGURING` | (any) | (any)        | `queued`        |
| `RUNNING`/`COMPLETING` (or any non-empty live state) | (any) | `done`/`failed`/`cancelled` | the trainer's terminal word WINS (the file is the source of truth, even if squeue hasn't caught up) |
| `RUNNING`/`COMPLETING` | (any) | `running` (or empty) | `running` |
| (empty — job left queue) | (any) | `done`/`failed`/`cancelled` | the trainer's terminal word WINS |
| (empty) | `COMPLETED` | (any but terminal) | `done` |
| (empty) | `CANCELLED [+ by <uid>]` | (any but terminal) | `cancelled` |
| (empty) | `FAILED`/`TIMEOUT`/`OUT_OF_MEMORY`/`NODE_FAIL`/`BOOT_FAIL`/`DEADLINE`/`PREEMPTED` | (any but terminal) | `failed` |
| (empty) | (empty) | `done`/`failed`/`cancelled` | pass-through |
| (empty) | (empty) | `running`/`queued` (no scheduler data) | `failed` (a `running` job with no scheduler evidence and no live ssh signal is dead — the catch-all is fail-closed) |
| (empty) | (empty) | `unknown`/empty | `unknown` |

So a SLURM job killed by `TIMEOUT`/`OOM`/`NODE_FAIL` is reported
`failed` with a reason in `slurm-<jid>.err` (you can read it with the
slurm log file allow-list in `training.rs:47-74`); `scancel`'d jobs
are `cancelled`.

Evidence: 9 Rust unit tests in `training.rs:716-805` (each row above
is a test), plus `verify:slurm`.

## 10. Orphan / temp-file behaviour (Phase 13)

The trainer intentionally leaves:

- `last.pt.tmp` if `_atomic_save` is interrupted between
  `torch.save(obj, f)` and `os.replace(tmp, path)`
  (`training_template.py:411-430`) — removed in the `finally` block.
  If you see a leftover `.tmp` next to a `last.pt`/`best.pt`, the
  process was killed hard between save and replace; the existing
  `*.pt` is the previous good checkpoint.
- `manifest.json.tmp` if `_manifest_write` is interrupted
  (`:855-872`) — same pattern, also `finally`-removed.
- `events.jsonl` lines that are not valid JSON — the reader
  (`scan_events`, `training.rs:184-211`) skips them silently. The
  integrity gate does the same per line (`:1047-1063`).

The trainer does NOT clean up an old run dir's temp files (no sweep
on startup — it operates on its own dir). Rust-side
`delete_training_run` (`training.rs:636-648`) is a single
`fs::remove_dir_all(&dir)`; it refuses if a live pid is reported.

The sidecar processes clean up their tracked children on
SIGTERM/SIGINT (`process-lifecycle` suite, 75 rows including a
descendant walk over `/proc/<pid>/task/*/children`).

## 11. Symptom → cause → action

| # | Symptom | Likely cause | First action | Evidence |
| - | ------- | ------------ | ------------ | -------- |
| 1 | Run shows `running` but `alive=false` | python crashed; SIGKILL; power loss | Re-open the run — `reconcile_status` reports `failed` (`training.rs:101-105`). Inspect the last event + `stderr.log`. | `verify:states`, `verify:recover**` |
| 2 | Run is `failed` with `stage="numeric"` | NaN/Inf loss, gradient or val metric | Read the failing parameter name in the `run.failed` event; lower lr / enable `GradientClipping` / switch amp dtype / fix the data pipeline. Resume is supported. | `verify:failures` |
| 3 | Run is `failed` with `stage="integrity"` | a required artifact (checkpoint, metrics.json, events.jsonl, manifest.json, stdout/stderr.log) is missing/invalid/empty | DO NOT trust the run — the trainer refused to report `done` precisely because of this. Re-launch with a clean run dir. | `verify:integrity` |
| 4 | Run is `failed` with `stage="resume"` | `resume_from` path is unreadable or the checkpoint doesn't match this model | pick the right `last.pt`; verify the run dir's `manifest.json` matches the checkpoint's `config.snapshot.*`. | `verify:checkpoint` |
| 5 | Run is `failed` with `stage="snapshot"` | the on-disk `model.py` or `model.spinoml` drifted from the launch-time hashes | re-launch — the running experiment must use the launch-time bytes; an external edit is a security/trust violation. | `verify:traingen` |
| 6 | Run is `failed` with `stage="config"` | `run.json` corrupt or unreadable status file | restore `run.json` from autosave / another run; an unreadable status file can never be made `done`. | `verify:failures` |
| 7 | Run is `failed` with `stage="dataset"` | dataset load raised (corrupt file, missing column, NaN/Inf input) | inspect the dataset in the Datasets tab; fix the missing/inf columns; re-launch. | `verify:failures` |
| 8 | Run is `failed` with `stage="split"` | unimplemented `split_strategy`, not enough rows, train/val overlap, empty training loader | set `training.split_strategy = "random"` (the only one implemented, Phase 19) or pre-split the dataset. | `verify:traingen` |
| 9 | Run is `failed` with `stage="train"` | uncaught exception in the epoch loop (DataLoader worker, OOM-killed children, etc.) | inspect `stderr.log`; if OOM, lower `batch_size`/`num_workers`; if dataloader, inspect the input pipeline. | `verify:failures` |
| 10 | Run is `failed` with `stage="fatal"` | an exception escaped `main()` itself — should be unreachable | report with the traceback in `stderr.log`; this is a trainer bug. | `verify:failures` |
| 11 | Run is `cancelled` | you (or the SLURM scheduler) pressed Stop / `scancel` | to resume: start a new run with `resume_from = experiments/runs/<id>/checkpoints/last.pt`. | `verify:cancel` |
| 12 | `metrics.json.resumable: false` with reason `"checkpoint belongs to a different model"` | you pointed `resume_from` at a checkpoint from a different run | copy the right `last.pt`; the hash mismatch is recorded in `manifest.json`. | `verify:checkpoint` |
| 13 | `run.resumed.rng_restore.torch_cuda = "absent in checkpoint"` | resumed on a CPU box, or the original checkpoint was saved on a CPU box | the resume still works; the run is not bitwise reproducible — accept or re-record. | `verify:checkpoint` |
| 14 | `LLM: offline` / `torch: offline` badge | sidecar process died or never started | restart the app (Tauri spawns them automatically); in browser dev run `npm run sidecar:torch` + `npm run sidecar:llm`. | `test:process-lifecycle` |
| 15 | `/infer` request never returns | a slow `exec()` in the user's Custom code is blocking the worker thread | the 30s socket timeout (`REQUEST_TIMEOUT`) will free the socket; fix the code or set `SPINOML_TORCH_TIMEOUT`. | `test:robustness` |
| 16 | Remote run appears as `unknown` | ssh connection lost | reconnect SSH; the next `ssh_training_run_status` re-queries live pid/squeue/sacct. The run on the cluster was NOT affected. | `verify:recovery` |
| 17 | SLURM run shows `failed` after a `scancel` | the scheduler `scancel`ed it (maybe you, maybe the cluster) | check `slurm-<jid>.err` — `scancel` produces a state of `CANCELLED` but the trainer never wrote `cancelled` (it was killed). Our reconciler maps `CANCELLED` to app-status `cancelled` (`training.rs:460`). | `verify:slurm` |
| 18 | SLURM run shows `failed` with `OUT_OF_MEMORY` in `slurm-<jid>.err` | OOM on the compute node | reduce batch size / hidden dims; the run's `last.pt` is at the last completed epoch (resumable). | `verify:slurm` |
| 19 | `local server timeout / curl: (7) couldn't connect` to a remote sidecar | the HPC-side sidecar crashed OR the tunnel didn't come up | open the SSH workspace again to retry the bootstrap; if it persists, ssh in and `fuser -k 7421/tcp` + restart. | `test:process-lifecycle` |
| 20 | Submit of a new run returns `MLF_EXISTS_INCOMPLETE` | a previous attempt left a run dir with no `pid` (the launch died before writing it) | delete the orphan dir with the UI's delete action (only allowed when no live process), or ssh in and `rm -rf` the dir. | `verify:submission` |
| 21 | `stop_training_run` returned OK but the trainer keeps running | the trainer was so stuck that even SIGTERM was eaten | `kill -KILL <pid>` (`<run_id>/pid`); the next `training_run_status` reports it as `failed`. | `verify:cancel` (cancel via SIGKILL test) |
| 22 | A run is `done` but the UI shows `best_val_loss` empty | `metrics.json` is missing OR a hand-written `run.json` lacks the field | the UI falls back to the minimum `val_loss` over `epoch.end` events (`RunSummary::from_parts:159-169`). If THAT is empty too, the run was externally launched with no metric writes — re-attach a real trainer. | `verify:events`, `verify:metrics` |
| 23 | `last.pt.tmp` is sitting next to `last.pt` | the save was interrupted (power loss / SIGKILL) between `torch.save` and `os.replace` | safe to ignore; the existing `last.pt` is the previous good checkpoint. `_atomic_save`'s `finally` would clean it up on a normal exit; here there was no normal exit. | `verify:checkpoint` (atomic-write crash sim) |
| 24 | App crashed mid-train — re-open shows the run still in progress | expected: `setsid` detached the trainer | just wait; the run is alive on the OS. The UI catches up on the next poll. | `verify:recovery` |
| 25 | An aborted submission put the run into `experiments/runs/<id>` with files but no `pid` | the ssh channel died between `write_remote_run_file("pid")` and the launcher's `$! > pid` | the next `ssh_start_training_run` for the same `<id>` returns `MLF_EXISTS_INCOMPLETE` and refuses — delete the dir manually, then re-submit. | `verify:submission` |
| 26 | `run.failed` event has `stage="fatal"` | an exception escaped `main()` — should be unreachable | report with the traceback; this is a trainer bug. | `verify:failures` |
| 27 | A run ends `done` but the integrity gate flagged a `note` like "checkpoint load verification skipped (size)" | `last.pt` > 256 MB | OK: the gate accepted the run (header + size check passed), only the deep load+keys check was skipped. The run is NOT resumable (`run.resumable.resumable = false`, reason `"checkpoint load verification skipped (size)"`). | `verify:integrity` |
| 28 | `manifest.json` is missing on a `done` run | the manifest writer raised and emitted `manifest.error`; the run continued | the run's loss/metrics are still valid; `manifest.json` absence does NOT cause `integrity` to fail (the gate checks `schema != …` — a missing file IS a fail). Verify the trainer emitted `manifest.error` in `events.jsonl`. | `verify:manifest` |
| 29 | `events.jsonl` ends mid-line (no trailing `\n`) | process killed mid-write | reader skips the truncated line; no harm. The line is lost. | `verify:events` |
| 30 | I want to run `kill -KILL <pid>` on a live trainer | the trainer is unresponsive | fine — the run reconciles to `failed`, last completed epoch's checkpoint is on disk. The pid file remains, the next `is_alive` returns false, the next `training_run_status` reflects `failed`. | `verify:cancel` |
| 31 | I want to delete a run dir but the UI says "run is still alive" | there's a process whose `kill -0` succeeds | stop it first (`stop_training_run`, or `kill -KILL` if it's hung), then delete. `delete_training_run` checks `pid_of(dir).map(is_alive)` (`training.rs:644`). | `verify:states` (run dir lifecycle) |
| 32 | I want to know WHICH SLURM partition the run actually used | the cluster scheduler chose one | read `slurm-<jid>.out` (the SLURM stdout; available via `read_training_run_file` — the static `READABLE` list in `training.rs:47-74` allows `slurm-<digits>.out|err`). The `#SBATCH` directives in the run dir's `train.sbatch` say what was REQUESTED; the cluster may have routed elsewhere. | `verify:slurm` |

## 12. Markers for "exercised here" vs "covered only by mocks"

The Phase 80 verification matrix in `TEST_MATRIX.md` is the authoritative
list. In summary:

- **Exercised against real processes** (PASS at baseline
  2026-10-06): `verify:checkpoint`, `verify:cancel`, `verify:failures`,
  `verify:integrity`, `verify:states`, `verify:events`, `verify:recovery`,
  `verify:submission`, `verify:slurm`, `verify:ssh`, `test:process-lifecycle`,
  `test:robustness`, `test:sidecar-auth-torch`, `test:sidecar-auth-llm`,
  `test:opencode-lifecycle`, `cargo test` (slurm 13 + alias 7 = 20+).
- **Covered by unit/integration only — NOT exercised against real hardware**:
  - SLURM end-to-end: only the 9-case `reconcile_slurm_status` table and
    the in-band probes (`squeue -j <jid>` / `sacct -j <jid>` via the ssh
    round-trip) are exercised. **No real sbatch submission was sent**;
    the `remote-live` suite is BLOCKED (`scripts/suites.ts:565-570`).
  - GPU / CUDA branches of `verify:reference`, `verify:reference-train`,
    `_eval_only` weight loading, RNG restoration with `torch_cuda`,
    `atomicAdd`-class nondeterminism: all `SKIPPED CUDA` — `hardware-cuda`
    is BLOCKED (`scripts/suites.ts:573-579`); `torch 2.12.0+cpu` is the
    installed build (BASELINE.md).
  - The Tauri/WebKitGTK window itself (no WebKitGTK here); the CSP is
    verified in a real headless Chromium (`test:webview-csp`, 14 checks).
  - LLM subscription / Anthropic / opencode: end-to-end only via the
    openai-compat fake-server harness (`test:llm-safety`, 124 cases);
    the live CLI is checked once by hand, not by a suite
    (`verify:opencode` BLOCKED unless `SPINOML_LIVE_LLM=1`).

When the row's evidence cites a real suite that exists, the behaviour
is verified end-to-end against a real process (or real Python on a
real torch). When the evidence cites a Rust unit test, the behaviour
is verified against the algorithm — the surrounding wiring
(`is_readable`, `safe_read_text`, the ssh round-trip itself) is
separately exercised by `verify:ssh` / `verify:recovery`.
