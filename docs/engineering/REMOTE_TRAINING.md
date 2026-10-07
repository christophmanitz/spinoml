# SpinoML — Remote training (HPC) — what actually happens

> Phase 80 deliverable. Read this BEFORE pointing SpinoML at a remote
> cluster: the connection model, the path-safety rules, the sidecar
> bootstrap, the submission claim, the SLURM reconciliation, the
> troubleshooting keys (real exit codes / markers — not paraphrased),
> and a hard "what is NOT verified against a real cluster" call-out at
> the end. Sources: `src-tauri/src/ssh.rs`, `src-tauri/src/remote_sidecar.rs`,
> `src-tauri/src/training.rs`, `sidecar-torch/training_template.py`,
> `sidecar-llm/main.mjs`, `src-tauri/src/scope_file.rs`. Every claim
> cites a file/function or a test.

---

## 1. The connection model

A remote workspace is a `{ alias, root }` pair stored in localStorage
(`spinoml.connections.v1`, see `src/connections/store.ts`). The app
never stores credentials — auth flows through the system `ssh` binary
plus the user's `~/.ssh/config`, agent, GSSAPI, ProxyJump, etc.

### Validation (every ssh_* command, every time)

`src-tauri/src/ssh.rs`:

| function                    | what it rejects                                                                                                            |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `validate_alias(s)` (`ssh.rs:42-65`) | empty or >128 chars; leading `-` (would be parsed by ssh as an option, e.g. `-oProxyCommand=x`); any char outside `[A-Za-z0-9._@:-]`. So both `leipzig-hpc` and `zw93onug@login01.sc.uni-leipzig.de` are accepted; `-J`, `-F`, `--` are rejected. |
| `validate_remote_root(s)` (`ssh.rs:67-85`) | empty; not absolute (must start with `/` or `~/`); contains `\0`/`\n`/`\r`; contains a `..` component. |
| `validate_relpath(s)` (`ssh.rs:87-99`) | contains `\0`/`\n`/`\r`; has any component other than `Normal(_)` or `CurDir` — i.e. no `..`, no leading `/`. |
| `validate_plain_filename(s, label)` (`ssh.rs:101-112`) | empty; contains `/`, `\`, `\0`, `\n`; is `.` or `..`. |
| `validate_python(s)` (called by `ssh_start_training_run:944`) | empty after trim → default `python`; otherwise a non-empty string passed through `shell_quote_path`. |

Every ssh invocation passes these checks BEFORE spawning a subprocess;
a violation is a 4xx-equivalent JS error string, not a spawned-but-
broken command.

### ssh options (every ssh call in this codebase)

`src-tauri/src/ssh.rs:158-171`:

```
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3"]
```

`BatchMode=yes` means ssh NEVER prompts for a password or
passphrase / 2FA. A misconfigured alias (no `id_ed25519`, no agent)
fails immediately with exit 255 — `verify:ssh` confirms the
classification into a user-visible error
(`"SSH authentication or host-key verification failed"`).

The interactive terminal uses `SSH_OPTS_INTERACTIVE` (`ssh.rs:167-171`,
NO `BatchMode`) so it CAN prompt for 2FA / host-key confirmation —
this is the one place ssh is allowed to interact.

Connection multiplexing is enabled by `control_args()` (`ssh.rs:198-…`):
`ControlMaster=auto` + `ControlPersist=120s` so the burst of polling
ssh calls (list/status/tail/gpu-stats every few seconds) shares one
TCP connection and doesn't trip the server's `MaxStartups` /
fail2ban. Socket namespace: `/tmp/spinoml-ssh/%C` (auto-created).

### Path quoting

Every shell-quoted value goes through `shell_quote(s)` (`ssh.rs:116-128`,
single-quote with embedded `'\''`) or `shell_quote_path(s)`
(`ssh.rs:130-142`) which converts a leading `~/` or bare `~` to
`"$HOME"` (the only way tilde expansion survives single quotes). So a
`root = "~/spinoml"` becomes literally `"$HOME"'/spinoml'` in the
emitted shell — bash concatenates adjacent quoted strings and expands
`$HOME` on the remote. Absolute roots stay single-quoted. NO user
input is ever pasted into a remote command unquoted.

---

## 2. What runs where

| Component                                              | Where it runs                | How                                                                            |
| ------------------------------------------------------ | ---------------------------- | ------------------------------------------------------------------------------ |
| React webview (UI, GraphStore, training graph, etc.)   | laptop                       | Tauri webview (`tauri://localhost`)                                            |
| Rust commands (`fs::*`, `ssh::*`, `pty::*`)           | laptop                       | compiled into the SpinoML `.deb`                                               |
| Torch sidecar (HTTP/SSE, `127.0.0.1:7421`)            | laptop OR remote             | local: spawned by Rust on launch; remote: deployed to `<root>/.spinoml/venv/` and tunneled back |
| LLM sidecar (`127.0.0.1:7422`)                        | laptop only                  | the LLM sidecar is NEVER remote; it drives the laptop's chat + filesystem      |
| `train.py` (the detached trainer)                      | laptop OR remote             | setsid/nohup locally; `nohup setsid` on remote direct; `sbatch train.sbatch` on remote SLURM |
| File operations                                        | always on the chosen workspace | local: `tauri-fs`; remote: `ssh_exec` (one ssh round-trip per op)             |
| Terminal                                               | laptop shell, remote `ssh -tt <alias>` | `portable-pty` in `src-tauri/src/pty.rs`                          |
| GPU                                                    | wherever the trainer runs    | CPU on laptop; CUDA on HPC GPU node (via SLURM `gres`)                         |

The dispatch lives in `src/connections/backend.ts`: `getCurrentConnection()`
selects the backend once, every consumer routes through it. **Never**
import `tauri-fs` or `tauri-ssh` from outside that file.

### Remote-only paths

For a remote workspace:

- All file reads/writes go through `ssh_exec` (one ssh per op). A
  persistent `sftp -b` session pool is NOT used; the API here would
  stay the same when added.
- The torch sidecar is bootstrapped ON the cluster
  (`src-tauri/src/remote_sidecar.rs`) — see §3.
- Training runs are launched on the cluster, NOT on the laptop — both
  in "direct" (setsid) and "slurm" (sbatch) modes
  (`src-tauri/src/ssh.rs:927-1048`).

---

## 3. Remote sidecar bootstrap (probe → install → deploy → tunnel → spawn)

Implemented in `src-tauri/src/remote_sidecar.rs`. Called from
`ensure_remote_sidecar` (`remote_sidecar.rs:422-476`), guarded by a
`bootstrap_lock` so two concurrent calls (React StrictMode dev double-
invoke) serialize and the second short-circuits on the live `current`.

### Pipeline

```
ensure_remote_sidecar(app, alias, root)
  ├─ validate_alias(alias)                              ssh.rs:42
  ├─ bootstrap_lock.lock().await                        remote_sidecar.rs:437
  ├─ if current has a live child for same target        reuse (no-op); dead → reap + clear token
  ├─ else stop_remote_sidecar_internal (kills old, clears token)
  └─ spawn_blocking → run_bootstrap
       ├─ emit Preparing { phase: "probe" }            remote_sidecar.rs:479
       ├─ probe(alias, root) → {venv_present, deps_ok, python_version}
       │    runs: set -e; mkdir .spinoml; source env.sh if present;
       │    command -v python3 || command -v python; SPINOML_NO_PYTHON exit 10 if neither
       ├─ if !venv_present || !deps_ok:
       │    emit Preparing { phase: "install" }
       │    install(alias, root):
       │      venv at $ROOT/.spinoml/venv; pip install --quiet torch (cpu),
       │      numpy pandas pillow python-dateutil; echo SPINOML_INSTALL_DONE
       │    (the install uses download.pytorch.org/whl/cpu — remote GPU compute happens
       │     via training, NOT the sidecar; sidecar only needs shape inference + smoke)
       ├─ emit Preparing { phase: "deploy" }
       ├─ deploy(alias, root, sidecar_dir):
       │    mkdir -p every subdir a SIDECAR_FILES entry lives under,
       │    then cat > <dst> per file with stdin bytes
       │    (refuses to start if a listed file is missing locally)
       ├─ emit Preparing { phase: "cleanup" }
       ├─ cleanup_stale_remote(alias, root):
       │    fuser -k 7421/tcp || true; pkill -9 -f '$MLDIR/venv/bin/python.*sidecar-torch'
       │    loop up to 10 × 0.3s waiting for the port to go free (RHEL systemd-logind
       │    doesn't reap user processes by default); PORT_FREE marker, else exit 1
       │    "remote port 7421 still held after cleanup attempt"
       ├─ generate_token()  (256-bit, OS randomness)
       ├─ emit Starting
       ├─ free_local_tunnel_port()  (fuser -k 7424/tcp; pkill -f "127.0.0.1:7424:127.0.0.1:7421")
       ├─ spawn ssh -T -L 127.0.0.1:7424:127.0.0.1:7421 -- <alias> <remote-script>
       │    remote-script (build_remote_script, remote_sidecar.rs:331-345):
       │      IFS= read -r SPINOML_TOK
       │      [ -n "$SPINOML_TOK" ] || { echo SPINOML_NO_TOKEN >&2; exit 11; }
       │      ROOT="$ROOT"; MLDIR="$ROOT/.spinoml"
       │      if [ -f $env_sh ]; then . $env_sh; fi
       │      export SPINOML_ALLOWED_ROOTS="$ROOT${SPINOML_ALLOWED_ROOTS:+:$SPINOML_ALLOWED_ROOTS}"
       │      cd "$MLDIR"
       │      export SPINOML_SIDECAR_TOKEN="$SPINOML_TOK" SPINOML_REQUIRE_TOKEN=1
       │      unset SPINOML_TOK
       │      SPINOML_TORCH_PORT=7421 exec "$MLDIR/venv/bin/python" -u sidecar-torch/main.py
       ├─ write token + '\n' to child stdin (kept open for the session)
       ├─ watch stdout for "listening on"  → 60s timeout
       ├─ watch stderr for "Could not request local forwarding" / "cannot listen to port" /
       │   "bind ... Address already in use" → "local port 7424 busy"
       ├─ TCP probe 127.0.0.1:7424 with 5 × 2s timeout
       ├─ register the token in SidecarTokens.remote so the webview can fetch it via
       │   sidecar_token("torch-remote")
       └─ store { alias, root, child, stdin } in state.current
```

### Env vars the remote sidecar sees (vs the local one)

The remote sidecar receives **the same contract** as the local one —
both run `sidecar-torch/main.py`. The differences are:

| env                        | local (Rust-spawned)                                   | remote (deployed)                                                         |
| -------------------------- | ------------------------------------------------------ | ------------------------------------------------------------------------- |
| `SPINOML_SIDECAR_TOKEN`    | per-launch token via spawn env, immediately read then `os.environ.pop`'d | per-session token over ssh stdin, stored in a NON-exported shell variable, only exported to python in the same line as `exec` and immediately `unset` |
| `SPINOML_REQUIRE_TOKEN`    | `1`                                                    | `1`                                                                       |
| `SPINOML_TORCH_PORT`       | `7421` (default)                                       | `7421` (the local sidecar's port; the LOCAL forward binds `7424` → `7421`) |
| `SPINOML_ALLOWED_ROOTS`    | appended to scope.json (the local workspace root)      | appended with `$ROOT` (the remote root) by the bootstrap script           |

So the wire protocol (token, Host/Origin, error codes) is identical —
verified by `test:sidecar-auth-llm` for the in-process path and by
static + behavioural tests in `remote_sidecar.rs` for the token
delivery.

### `env.sh` — the user's escape hatch

If the cluster's python / pip are not on a bare login shell (e.g.
they require `module load python` or a `conda activate`), the user
creates `<root>/.spinoml/env.sh` and the bootstrap script sources it
BEFORE reading `SPINOML_TOK` (so `env.sh` cannot steal or clobber the
token) and BEFORE `export SPINOML_ALLOWED_ROOTS` and BEFORE
`exec python`. A hostile `env.sh` that tries `export
SPINOML_SIDECAR_TOKEN=evil` after the bootstrap line succeeds — but
the bootstrap OVERWRITES `SPINOML_SIDECAR_TOKEN` with the real value
in the same `export …; exec …` sequence, so the `evil` value never
reaches python. This is unit-tested in `remote_sidecar.rs` (the
`setup_stubbed_remote` test at `:845`).

### `SPINOML_SYMLINK_TARGETS` (Phase 46 follow-up)

If a remote workspace has `datasets → /work2/...` (an out-of-tree
symlink, common on HPC), `scope.py` would refuse the symlink target
unless either:

- The remote sidecar was started with `SPINOML_SYMLINK_TARGETS=/work2`
  (export from `env.sh`), OR
- `<root>/.spinoml/scope.json` lists the target under `symlink_targets`.

The error string is `PATH_SYMLINK_OUTSIDE` with the exact fix
(`scope.py` → `scope_file.rs:418-421`). See `verify:paths` and
`test:scope` for the matrix.

---

## 4. Run submission (direct + SLURM)

A submission goes through `ssh_start_training_run`
(`src-tauri/src/ssh.rs:927-1048`). It is idempotent under retries
(Phase 33) — see FAILURE_RECOVERY §8 for the claim protocol.

### Direct (no SLURM)

`backend_kind` defaults to `"local"` (`ssh.rs:1003-1004`). The launch
is a single `ssh_exec` of:

```
cd "<dir>" && { nohup setsid "<python>" -u train.py > stdout.log 2> stderr.log < /dev/null & echo $! > pid; }
```

The brace group is load-bearing: without it, `$!` is the transient
subshell pid (in the wrong dir), not python's, and the run list
reports the run as `failed` on re-open. Documented inline at
`ssh.rs:1034-1040`. Verified by `verify:submission`.

### SLURM (`backend.kind == "slurm"`)

`build_sbatch` (`ssh.rs:1054-1115`) emits the script content. The
fields it writes into `#SBATCH` directives (verbatim):

| config key       | `#SBATCH` flag            | default                | emitted when                |
| ---------------- | ------------------------- | ---------------------- | --------------------------- |
| `partition`      | `--partition=<value>`     | omitted (cluster default) | non-empty                |
| `time`           | `--time=<value>`          | `04:00:00`             | always                      |
| `mem`            | `--mem=<value>`           | omitted                | non-empty                   |
| `cpus_per_task`  | `--cpus-per-task=<n>`     | `8`                    | always (`unwrap_or(8)`)     |
| `gres`           | `--gres=<value>`          | omitted                | non-empty                   |
| `account`        | `--account=<value>`       | omitted                | non-empty                   |
| `qos`            | `--qos=<value>`           | omitted                | non-empty                   |
| `modules[]`      | `module load <name>`      | omitted                | each non-empty string in the array |
| `pre_run_script` | (free-form, written after the directives and before `cd`) | omitted | non-empty                   |

Job-name slug: `spinoml-<job>` where `<job>` is `run_id` with any
non `[A-Za-z0-9_-]` replaced by `-`, capped at 64 chars
(`ssh.rs:1059-1063`). Tested for non-alphanumeric inputs
(`ssh.rs:1551-1556`).

Always emitted:

```
#!/bin/bash
#SBATCH --job-name=spinoml-<job>
[conditional partition/time/mem/cpus/gres/account/qos lines]
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

[conditional module load lines]
[conditional pre_run_script]

cd "$SLURM_SUBMIT_DIR"
"<python>" -u train.py
```

`-u` so SLURM's stdout buffering doesn't hide progress; `cd
"$SLURM_SUBMIT_DIR"` so the script survives SLURM's `cwd` quirk.
`cpus-per-task` defaults to 8 when the user omits it (documented at
`ssh.rs:1077`).

### Submission + claim

```
ssh_start_training_run runs (one ssh round-trip per step):
  1. validate_alias / validate_remote_root / validate_run_id
  2. mkdir -p <root>/experiments/runs/         (the SUBMISSION CLAIM — see below)
     then mkdir <root>/experiments/runs/<run_id>/
     the script emits one of:
       MLF_CLAIM_FAILED       cannot create the runs dir
       MLF_CREATED            fresh run, proceed
       MLF_ALREADY_LAUNCHED   dir exists + has a pid — assume prior submission succeeded, no relaunch
       MLF_EXISTS_INCOMPLETE  dir exists, no pid — refuse duplicate launch (user must clean up)
  3. mkdir <root>/experiments/runs/<run_id>/checkpoints/
  4. write_remote_run_file: run.json, model.spinoml, model.py, train.py, status="queued"
  5. if slurm: write_remote_run_file(train.sbatch), then sbatch train.sbatch,
     parse "Submitted batch job <id>" → freeze pid as "slurm:<jid>"
     else: nohup setsid <python> -u train.py > stdout.log 2> stderr.log < /dev/null & echo $! > pid
```

The submission claim is `mkdir <run_dir>` — `mkdir` is atomic on a
single filesystem (POSIX guarantee). Two clients see "no such dir"
both, but only one creates it.

### What is written into the run dir

Identical to local — `run.json`, `model.spinoml`, `model.py`,
`train.py`, `status` (`queued\n`), `checkpoints/` (empty). After the
launch:

| backend | additional files                                              |
| ------- | -------------------------------------------------------------- |
| direct  | `pid` (numeric), `stdout.log`, `stderr.log`                    |
| SLURM   | `pid` (`slurm:<jid>`), `train.sbatch`, `slurm-<jid>.out`/`slurm-<jid>.err` written by the scheduler |

### The submission claim is the only durable idempotency

If the SSH channel drops after step 4 (files written) but before step 5
(launch), the next submission for the same `<run_id>` sees
`MLF_EXISTS_INCOMPLETE` (`ssh.rs:960-976`) and refuses — the user
must `ssh_delete_training_run` first. This is the documented
behaviour; it prevents two concurrent launches from racing on the same
work directory.

---

## 5. Monitoring + recovery

### Live status

| source                                                | cadence                                                |
| ----------------------------------------------------- | ------------------------------------------------------ |
| `ssh_list_training_runs` (`ssh.rs:1128-1231`)         | every time the UI opens the Runs tab — one round-trip per run dir |
| `ssh_training_run_status` (`ssh.rs:1234-1299`)        | on Run-Detail open + every poll tick                   |
| `ssh_gpu_stats` (`ssh.rs:1468-1501`)                   | on the Run-Detail hardware strip; for SLURM jobs uses `srun --overlap --jobid=<jid>` so nvidia-smi runs on the COMPUTE node, not the login node (`timeout 15` cap) |
| `ssh_remote_training_capabilities` (`ssh.rs:1381-1430`) | once per remote connection — probes `sbatch`/`sinfo`/`nvidia-smi` for the SLURM + partition dropdown |

### Recovery after SSH loss / app restart

- **Direct (setsid)**: the trainer was `nohup setsid`'d — it survives
  the ssh channel's death. `kill -0 <pid>` on the remote confirms
  liveness on reconnect.
- **SLURM**: the job is still in the scheduler. `squeue -j <jid> -h
  -o '%T'` returns the live state (`PENDING`/`RUNNING`/…). Once the
  job leaves the queue, `sacct -j <jid> -n -X -o State%30` provides
  the terminal state (`COMPLETED`/`CANCELLED`/`FAILED`/`TIMEOUT`/
  `OUT_OF_MEMORY`/`NODE_FAIL`/`BOOT_FAIL`/`DEADLINE`/`PREEMPTED`).
- **App restart**: the Rust shell does NOT cache remote run state in
  localStorage (verified by `verify:recovery`). On re-open it
  re-queries the on-host files; the on-host `events.jsonl`/`status`/
  `pid` are the single source of truth.

### Stop

`ssh_stop_training_run` (`ssh.rs:1319-1343`) writes `cancelled` into
the status file FIRST (so the trainer unwinds at the next epoch
boundary), then SIGTERMs the pid (or `scancel`s the SLURM job). A
terminal status (`done`/`failed`/`cancelled`) is protected from
overwrite — see FAILURE_RECOVERY §3.

### Delete

`ssh_delete_training_run` (`ssh.rs:1346-1367`) refuses if the run is
alive (`squeue`/`kill -0` checks per backend), else `rm -rf -- "$d"`
on the remote.

### SLURM state → app status

See FAILURE_RECOVERY §9 for the full table. The single function is
`reconcile_slurm_status(status_raw, squeue_state, sacct_state)`
(`training.rs:430-465`) and it's exhaustively tested (9 Rust unit
tests at `training.rs:716-805`).

---

## 6. Security properties

1. **No secrets stored.** The token is generated per-launch (Rust)
   or per-session (remote) and held in `SidecarTokens`
   (`src-tauri/src/sidecar_auth.rs`). User SSH credentials live in
   `~/.ssh/config` + agent — SpinoML never reads them. The Token is
   the only secret it owns.
2. **Token never in argv or a log line.** The remote script
   (`build_remote_script`, `remote_sidecar.rs:331-345`) takes the
   token from ssh stdin via `read -r SPINOML_TOK` and exports it to
   python in the SAME line as `exec python`. The token never appears
   in the command string, in `argv`, or in any log
   (`cargo test` `script_does_not_embed_token` at
   `remote_sidecar.rs:812-820`, `build_run_command_debug_does_not_contain_token`
   at `:822-835`).
3. **Path scoping on the cluster.** The remote sidecar reads
   `SPINOML_ALLOWED_ROOTS=$ROOT` (the bootstrap script appends it
   at `remote_sidecar.rs:338`), so every path the sidecar opens is
   contained. Same `scope.py` (`test:scope` covers 88 cases
   including hostile manifests).
4. **Same Host/Origin/auth gate** as the local sidecar — both run
   `sidecar-torch/main.py` with `SPINOML_REQUIRE_TOKEN=1`. A
   different user on the HPC login node cannot reach the sidecar
   (lacks the token); the sidecar's loopback is reachable but
   `X-SpinoML-Token` blocks them. See
   `docs/engineering/SIDECAR_AUTH.md`.
5. **Symlink containment.** `SPINOML_SYMLINK_TARGETS` (env or scope
   file) is the escape hatch for `datasets → /work2/...`.
   Without it, the request is refused with `PATH_SYMLINK_OUTSIDE`
   and a one-line fix.
6. **Path injection still open (Phase 46 limits)** — the Node
   ssh path (`sidecar-llm/path-scope.mjs`) is symlink-aware; the
   Rust half is unix-only. Not an active exploit path but documented
   in LIMITATIONS §2.

---

## 7. Setup checklist for a new HPC account

The minimum a user must arrange before "Remote" workspaces work:

1. **ssh key + config**
   - `id_ed25519` (or similar) loaded into `ssh-agent`.
   - `~/.ssh/config` with a `Host <alias>` block whose `HostName`
     points at the login node, `User <uid>`, `IdentityFile` the
     right key. **Test first**: `ssh <alias> echo ok` from a bare
     shell. A `BatchMode` failure → check the agent.
2. **Workspace directory on cluster scratch**
   - The remote `root` may be absolute (`/scratch/<uid>/spinoml`)
     or tilde-prefixed (`~/spinoml`). `/home/<uid>/...` is OK for
     small workspaces but fills fast.
3. **(optional) `env.sh` for non-default python**
   - `<root>/.spinoml/env.sh` (the bootstrap creates the dir, the
     user writes the file). Example: `module load python/3.12 &&
     source /sw/conda/etc/profile.d/conda.sh && conda activate
     myenv`. Must be plain shell; it is sourced every probe/install/
     spawn call. See §3 "env.sh".
4. **(optional) `SPINOML_SYMLINK_TARGETS`** in `env.sh` if the
   workspace symlinks data out of the project root.
5. **SLURM defaults** (only for SLURM-mode training):
   - `partition`, `time`, `cpus_per_task`, `mem`, `gres`, `account`,
     `qos` per the cluster's accounting. The New Run dialog populates
     these from `ssh_remote_training_capabilities`
     (`ssh.rs:1381-1430`).
6. **No port-forwarding to set up** — SpinoML brings its own
   `127.0.0.1:7424 → 127.0.0.1:7421` tunnel via `ssh -L` and tears
   it down on session end (`free_local_tunnel_port`,
   `remote_sidecar.rs:703-725`).

If `ssh <alias> echo ok` fails with a permission/host-key error, the
UI's "Verbindung testen" reports the same `ssh_failure()` classification
(`ssh.rs` `ssh_failure_tests` mod at `:1576-1617`) so the user
sees a one-line German/English explanation instead of a raw `ssh:
connect to host … Connection timed out`.

---

## 8. Troubleshooting (real exit codes / strings from the code)

| Symptom (in the UI or stderr)                                              | Real source                                                                                     | Fix |
| -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | --- |
| `ssh: Could not resolve hostname <alias>: Name or service not known`        | exit 255, `ssh_failure()` classifies as `"SSH host could not be resolved"`                       | Fix `~/.ssh/config` `HostName`; verify with `ssh <alias> echo ok`. |
| `ssh: connect to host <h> port 22: Connection timed out`                   | exit 255, classified as `"SSH connection timed out"`                                            | Check VPN / firewall; `ConnectTimeout=10` is generous — if it times out, the host is unreachable. |
| `Permission denied (publickey)` / `kex_exchange_identification: Connection closed by remote host` | exit 255, classified as `"SSH authentication or host-key verification failed"` | Load your key into the agent (`ssh-add ~/.ssh/id_ed25519`); accept the host key once via `ssh <alias>` (interactive — uses `SSH_OPTS_INTERACTIVE`, allows prompts). |
| `ssh exit 255: …` in any tool / train log                                   | `run_remote` (`remote_sidecar.rs:119-139`) catches ANY ssh failure with a single error format   | First action: `ssh <alias> echo ok` from a bare shell. |
| `local port 7424 busy — the tunnel could not be opened`                     | stderr watcher (`remote_sidecar.rs:598-612`) caught "Could not request local forwarding" / "bind ... Address already in use" | Kill any orphaned tunnel: `fuser -k 7424/tcp`. The Rust shell also calls `free_local_tunnel_port()` on each ensure. |
| `tunnel announced remotely but local port 7424 is unreachable (forward failed)` | explicit TCP probe (`remote_sidecar.rs:629-649`) after the watcher's success                  | Same — tunnel is dead, no listening socket. |
| `install did not complete: …`                                              | bootstrap install didn't print `SPINOML_INSTALL_DONE` (`remote_sidecar.rs:210-213`)             | Check cluster firewall for `download.pytorch.org`; or pre-create `<root>/.spinoml/venv` and pre-install deps to skip the bootstrap. |
| `install failed: ssh exit 255: …`                                            | The ssh carrying the pip install failed mid-install (timeout / network drop). The remote venv may exist but `pip` partially failed. | Re-run `ensure_remote_sidecar`; the probe will detect venv present + deps missing and re-install only the deps. |
| `SPINOML_NO_PYTHON` (stderr)                                                | `probe` script (`remote_sidecar.rs:159-161`) — neither `python3` nor `python` was on PATH        | Add `module load python` or `conda activate` to `<root>/.spinoml/env.sh`. |
| `SPINOML_NO_TOKEN` (stderr)                                                | `build_remote_script` (`remote_sidecar.rs:334-335`) — empty stdin reached the remote            | Should not happen — the Rust side writes the token immediately after spawn (`remote_sidecar.rs:552-561`). If it does, the stdin pipe leaked closed; report. |
| `could not generate remote token: …`                                       | `generate_token()` failed (very rare; OS entropy exhausted)                                     | Retry; if persistent, check `getrandom(2)` on the host kernel. |
| `remote port 7421 still held after cleanup attempt`                        | `cleanup_stale_remote` (`remote_sidecar.rs:222-247`) — 10×0.3s loop couldn't free 7421          | Manually `ssh <alias> 'fuser -k 7421/tcp'` and retry. |
| `sidecar-torch/<file> missing on laptop — SpinoML bundle may be incomplete` | `deploy` (`remote_sidecar.rs:295-301`) — a `SIDECAR_FILES` entry has no source on the laptop     | Reinstall SpinoML; if persistent, the `SIDECAR_FILES` constant drifted from the sidecar's actual imports — `verify:remote-deploy-files` will go red. |
| `path ... not inside any allowed root` / `PATH_SYMLINK_OUTSIDE`             | scope check refused (Python `scope.py`, surfaced via `/dataset/inspect|stats|smoke` 400)        | Add the target to `SPINOML_SYMLINK_TARGETS` in `<root>/.spinoml/env.sh`, or to `~/.cache/spinoml/scope.json`. |
| `sbatch failed: sbatch: error: Batch job submission failed: Invalid partition` | parsed from the output of `sbatch train.sbatch` (`ssh.rs:1011-1026`); MLF_JOBID marker missing | The `partition` in your run config doesn't exist on the cluster; pick one from the Run dialog dropdown (sourced from `sinfo -h -o '%P'`). |
| `run <id> already exists on <alias>, but has no recorded submission; refusing duplicate launch` | `MLF_EXISTS_INCOMPLETE` (`ssh.rs:973-975`) | Delete the orphan dir: ssh in and `rm -rf <root>/experiments/runs/<id>`, or click Delete in the UI after stopping the (dead) process. |
| `could not prepare the remote runs directory on <alias>`                    | `MLF_CLAIM_FAILED` (`ssh.rs:969-971`) — `experiments/runs/` could not be created                | Check permissions on the remote root; check disk space. |
| Run stays `queued` after `sbatch` returns success                            | `squeue` still showing `PENDING`/`CONFIGURING` (reconciler maps both to `queued`)                | Wait for the scheduler; check `squeue -j <jid>` and `sacct -j <jid>` on the cluster. |
| Run shows `failed` immediately after submit                                | `sacct` returned `FAILED` (rare) or `sbatch` itself errored                                       | Read `slurm-<jid>.err` via the Run-Detail file list (`is_readable` allows `slurm-<digits>.err`, `training.rs:47-74`). |
| GPU stats show nothing on a SLURM job                                       | `ssh_gpu_stats` (`ssh.rs:1468-1501`) uses `srun --overlap --jobid=<jid>` with a 15s `timeout`; a non-GPU partition or an uncooperative step returns nothing. | Confirm `gres gpu:<n>` is set; try `srun --overlap --jobid=<jid> nvidia-smi` on the cluster manually. |
| `ssh to the workspace host failed (255)` in `slurm_status` tool             | `slurm_status` (`sidecar-llm/main.mjs:812-827`) — the `squeue`/`sacct` probe round-trip ssh-failed. | Reconnect SSH; the LLM-side tool surfaces the tail of the ssh stderr so the user sees the real cause. |
| Remote smoke test says "Sidecar braucht HPC"                                | `verify:smoke` / the UI's dataset smoke — the LOCAL torch sidecar can't reach the remote path. | This is a documented Phase 12a limit; Phase 12b deploys a sidecar on the HPC side (now done — see §3). |
| Token auth 401 on `/respond` or `/chat` after a restart                     | The remote token was cleared on tunnel death (`stop_remote_sidecar_internal` clears `SidecarTokens.remote`, `remote_sidecar.rs:736-738`); the webview's `sidecar_token("torch-remote")` returns the new token on next call. | Should self-heal on the next request; if not, reconnect SSH and retry. |
| `KILLED: shell run exceeded the login-node 2-minute cap. …`                  | `sidecar-llm/main.mjs:1769-1772` — `run_script` mode `shell` on a remote workspace hit the 2-minute cap. | Switch to `mode:"slurm"` and re-run; SLURM jobs are NOT capped. |
| `Shell exited 255` mid-`run_script`                                          | ssh transport failure (exit 255)                                                               | Reconnect; the `ask`/`respond` channel may have lost its answer — restart the turn. |
| `Permission denied (publickey)` mid-training                                | ssh connection lost (e.g. agent timed out)                                                     | Reconnect; a live run is unaffected (setsid/nohup), only the status polling fails. |

Evidence base: `verify:ssh` (17 checks: transport options, error
classification), `verify:slurm` (33 checks: squeue/sacct mapping,
sbatch parsing), `verify:credentials` (8 checks: secret sanitization),
`verify:recovery` (38 checks: live re-query, detached survival),
`verify:submission` (7 checks: idempotency under retries), `verify:remote-deploy-files`
(SIDECAR_FILES vs the import closure), `cargo test` (ssh alias
validation + slurm state machine + token-delivery scripts).

---

## 9. What is NOT verified against a real cluster in this repo

> The `remote-live` suite is BLOCKED (`scripts/suites.ts:565-570`).
> This section is the honesty contract for every claim above. When
> the user is debugging on a real cluster and the docs say something
> that doesn't match their observation, THIS section is the one to
> re-read first.

### Verified against real processes / unit tests

| Area                                                                              | Coverage                                                                          |
| --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| SSH alias / path / root validation                                                 | `cargo test` `alias_validation_tests` (22 tests) — every reject case listed above  |
| SSH options (BatchMode + ControlMaster) + remote-command failure classification      | `verify:ssh` (17), `cargo test ssh_failure_tests`                                  |
| ssh_quote_path for `~/`-prefixed paths                                             | unit test inside `cargo test` (the test reads `PATH_SYMLINK_OUTSIDE` is checked)  |
| Sbatch script generation                                                           | 4 unit tests in `cargo test slurm_tests` (defaults / all-fields / job-name sanitize / submit parsing) |
| SLURM state → app status reconciliation                                            | 9 Rust unit tests in `training.rs` (one per row of the table)                      |
| Submission claim + idempotency                                                     | `verify:submission` (7 checks)                                                     |
| Recovery after SSH loss / app restart (live re-query)                              | `verify:recovery` (38 checks) — but the ssh target is `localhost` (a local ssh server), not a real cluster |
| Credential sanitization in error messages                                          | `verify:credentials` (8 checks), `cargo test ssh_failure_tests`                    |
| `SIDECAR_FILES` covers the actual import closure of `sidecar-torch/*.py`          | `verify:remote-deploy-files` (computed via static analysis, no actual deploy)      |
| Remote token delivery script (env.sh can't replace it, `unset SPINOML_TOK` order)  | 5 unit tests in `cargo test` (`script_starts_with_token_read`, `script_exits_eleven_on_empty_token`, `script_export_after_envsh_before_exec`, `script_does_not_embed_token`, `build_run_command_debug_does_not_contain_token`) + 1 behavioural test (`setup_stubbed_remote`) |
| SLURM `pid` file format `slurm:<jid>` + `scancel` flow                             | unit-tested by `verify:slurm`/`verify:recovery` against a local sshd                |
| `SPINOML_ALLOWED_ROOTS` export to the remote sidecar                               | static check + behavioural env.sh test                                            |

### NOT verified against a real HPC login node

| Area                                                                              | Why                                                                                | What's missing                                                                                  |
| --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| The full `ensure_remote_sidecar` pipeline against an actual HPC cluster            | the test environment has no HPC login node; only a local sshd is available          | real pip install times on a cold cluster; real `download.pytorch.org` connectivity; cluster quirks in `/proc/<pid>/environ` ownership; systemd-logind's `KillUserProcesses=no` (Phase 12b R053 mentions this in a comment but the cleanup logic was tested against a `fuser -k`/`ss` loop, not a real RHEL login node) |
| `ssh -L 7424:127.0.0.1:7421` tunnel across an actual bastion / jump host            | no bastion in the test environment; `ProxyJump` is configured via `~/.ssh/config` and SSH handles it transparently | end-to-end timing; whether `ExitOnForwardFailure=yes` correctly refuses a refused forward on a real bastion |
| SLURM `sbatch` against a real cluster scheduler                                    | no real SLURM; the `verify:slurm` harness uses a local sshd that fakes the cluster with `#!/bin/bash` scripts          | partition routing, accounting policies, QoS enforcement, scheduler preemption hooks, dependency-chains |
| `srun --overlap --jobid=<jid>` for remote GPU stats                                | no real SLURM job to query                                                        | the gpu-name read on a real compute node                                                        |
| `cleanup_stale_remote` against RHEL/CentOS where user processes outlive ssh        | the comment at `remote_sidecar.rs:218-222` calls this out explicitly; the loop is unit-tested but not against a real RHEL box | the precise `fuser`/`ss` semantics on different distros                                          |
| `env.sh` loading order across different login shells (bash / zsh / fish)            | only bash is exercised (the bootstrap script is bash)                              | tcsh/zsh users; we never read `$SHELL`                                                          |
| Network drops during `deploy` (large ESPF payload)                                 | the unit test deploys to a local fake remote                                      | a deploy that takes >1 minute over WAN; the deploy's per-file `cat > dst` over ssh has no resume |
| SLURM array jobs (`--array=…`), dependencies (`--dependency=…`), preemption policy | build_sbatch only emits the single-job directives documented above                  | user has to add the directives via `pre_run_script`                                             |
| Real HPC GPU topology (multi-GPU node, NVLink, MIG)                                | no GPU here                                                                        | the GPU-strip is best-effort; `verify:reference`/`verify:reference-train` SKIPPED CUDA branches    |
| Concurrent submissions from two app instances to the same `<root>`                 | the `MLF_ALREADY_LAUNCHED` / `MLF_EXISTS_INCOMPLETE` markers were tested with sequential submissions only | true concurrent `mkdir` race (POSIX says it can't lose; we don't have a real cluster to confirm) |
| Long-running SLURM jobs (multiple days) and SLURM rescheduling                     | only squeue/sacct probe behavior is exercised; cluster-side policy unknown         | the day-3 OOM kill, the day-5 node failure, the day-7 quota limit                               |
| Remote Dataset smoke / train end-to-end                                            | blocked at Phase 12a (no HPC-side sidecar); Phase 12b deployed the sidecar but the smoke path's coverage is unit-test only | a real BindingDB on a real HPC with a real `squeue -j <jid>` outcome                            |

When you see a symptom not in this list, the FIRST step is
`ssh <alias> echo ok` from a bare shell — that reproduces what the
app sees, minus the GUI error classification. The second step is to
read the relevant `.err` file (`slurm-<jid>.err` or the app's stderr
under the run dir) — the Rust shell surfaces the relevant tail in the
error message it returns to the UI.
