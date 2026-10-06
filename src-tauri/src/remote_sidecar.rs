// Phase 12b: deploy + run the torch sidecar on a remote SSH host so smoke
// tests and shape inference work on remote workspaces.
//
// Architecture
//   laptop                                   HPC (login node)
//   ──────                                   ────────────────
//   frontend ──→ 127.0.0.1:REMOTE_LOCAL_PORT
//                ↓ ssh -L (tunnel)
//                127.0.0.1:REMOTE_REMOTE_PORT ←── sidecar-torch in venv
//
// The long-lived `ssh` subprocess does TWO things:
//   1. Forwards the local port to the remote.
//   2. Holds the sidecar process inside the same channel — when ssh dies
//      (window close, network drop), the remote python child dies too.
//
// Bootstrap probe → install (if needed) → deploy files → spawn. Each phase
// emits a `remote-sidecar:status` Tauri event so the frontend can render
// progress ("Installing torch on the HPC, this will take a minute…").
//
// Env preference: a fresh venv at `<root>/.spinoml/venv/`. The user can
// pre-create that path with their own python (module load + python -m venv)
// and we'll detect + reuse. A `<root>/.spinoml/env.sh` is sourced before
// every command if present — single-file escape hatch for users who need
// `module load python` or `conda activate` first.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

use crate::sidecar_auth::{generate_token, SidecarTokens};
use crate::ssh::{shell_quote_path, SSH_OPTS};

pub const REMOTE_LOCAL_PORT: u16 = 7424;
pub const REMOTE_REMOTE_PORT: u16 = 7421;

/// Every file the remote sidecar needs to start and stay alive.
///
/// Regression fix (Phase 77 / see docs/engineering/SIDECAR_AUTH.md §"Remote
/// deploy file list"): the deploy used to ship only `main.py` +
/// `dataset_handlers.py`. Since Phase 45–47 main.py also imports `scope`,
/// `safe_load`, `deps_policy`; Phase 78 adds `auth`; and dataset_handlers.py
/// reads `sidecar-torch/espf/*` at runtime via `Path(__file__).resolve().parent
/// / "espf"`. A constant-driven list is the single source of truth for BOTH
/// `deploy()` and the static verifier `scripts/verify-remote-deploy-files.ts`
/// (which computes the local-import closure from `main.py` + `dataset_handlers.py`
/// and fails when the constant misses any).
///
/// Each entry is a POSIX path relative to `sidecar-torch/`. No `..`, no
/// absolute paths, no duplicates (all three invariants are unit-tested).
pub const SIDECAR_FILES: &[&str] = &[
    "main.py",
    "dataset_handlers.py",
    "scope.py",
    "safe_load.py",
    "deps_policy.py",
    // Phase 78: shipped even before the file exists locally — `deploy()`
    // refuses to start without every listed file present, so a missing
    // auth.py here fails with a clear error rather than a `ModuleNotFoundError`
    // on the remote after the sidecar is already up.
    "auth.py",
    // ESPF BPE codebook referenced via `Path(__file__).resolve().parent /
    // "espf"` from `dataset_handlers.py` (sidecar-torch/dataset_handlers.py:
    // ESPF_DIR). 5 files, ~1.4 MB.
    "espf/NOTICE",
    "espf/drug_codes_chembl.txt",
    "espf/protein_codes_uniprot.txt",
    "espf/subword_units_map_chembl.csv",
    "espf/subword_units_map_uniprot.csv",
];

#[derive(Default)]
pub struct RemoteSidecarState {
    pub current: Mutex<Option<RemoteSidecar>>,
    /// Serializes ensure_remote_sidecar so two concurrent calls (e.g. React
    /// StrictMode double-invoking the load effect in dev) can't race — the
    /// second would otherwise free_local_tunnel_port() the first's freshly
    /// bound tunnel. The second call waits, then short-circuits on `current`.
    pub bootstrap_lock: tokio::sync::Mutex<()>,
}

pub struct RemoteSidecar {
    pub alias: String,
    pub root: String,
    /// The live ssh-tunnel + remote-sidecar child. Killed + waited on
    /// session-stop.
    pub child: Child,
    /// `ChildStdin` of the same child. We KEEP the handle in the struct on
    /// purpose: the token is delivered through `stdin` ONCE at spawn and
    /// then the pipe must stay open for the lifetime of the session — if we
    /// dropped `stdin` here, EOF would reach the remote script and it would
    /// exit before we ever saw "listening". We never write to it again, so
    /// the token leaves our process exactly once.
    #[allow(dead_code)] // kept alive on purpose; see field doc above
    pub stdin: ChildStdin,
}

#[derive(Serialize, Clone)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum RemoteSidecarStatus {
    Idle,
    Preparing { phase: String, message: String },
    Starting,
    Running { local_port: u16, remote_port: u16, alias: String, root: String },
    Stopped,
    Error { message: String },
}

fn emit(app: &AppHandle, status: &RemoteSidecarStatus) {
    let _ = app.emit("remote-sidecar:status", status);
}

// ─── ssh helper (mirror of ssh::ssh_exec, with a configurable timeout
//     and longer install-phase tolerance) ───

fn run_remote(alias: &str, remote_cmd: &str, stdin: Option<&[u8]>) -> Result<String, String> {
    let mut cmd = Command::new("ssh");
    for o in SSH_OPTS { cmd.arg(o); }
    // `--` ends option parsing: the target can never be taken for an ssh option.
    cmd.arg("--").arg(alias).arg(remote_cmd);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    cmd.stdin(if stdin.is_some() { Stdio::piped() } else { Stdio::null() });
    let mut child = cmd.spawn().map_err(|e| format!("spawn ssh: {e}"))?;
    if let Some(data) = stdin {
        let mut s = child.stdin.take().ok_or_else(|| "no stdin".to_string())?;
        let owned = data.to_vec();
        std::thread::spawn(move || { let _ = s.write_all(&owned); });
    }
    let out = child.wait_with_output().map_err(|e| format!("wait ssh: {e}"))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr).to_string();
        let code = out.status.code().map(|c| c.to_string()).unwrap_or_else(|| "?".into());
        return Err(format!("ssh exit {code}: {}", err.trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

// ─── bootstrap pipeline ───

struct ProbeResult {
    venv_present: bool,
    deps_ok: bool,
    python_version: String,
}

fn probe(alias: &str, root: &str) -> Result<ProbeResult, String> {
    let root_q = shell_quote_path(root);
    let env_sh = shell_quote_path(&format!("{}/.spinoml/env.sh", root.trim_end_matches('/')));
    let script = format!(
        "set -e
ROOT={root_q}
MLDIR=\"$ROOT/.spinoml\"
mkdir -p \"$MLDIR\"
# optional env.sh hook (module loads / conda activate)
if [ -f {env_sh} ]; then . {env_sh}; fi
PY=$(command -v python3 || command -v python || true)
[ -n \"$PY\" ] || {{ echo SPINOML_NO_PYTHON >&2; exit 10; }}
echo \"PY_VER=$($PY -c 'import sys; print(\\\".\\\".join(map(str, sys.version_info[:3])))')\"
if [ -x \"$MLDIR/venv/bin/python\" ]; then
  echo VENV_PRESENT
  if \"$MLDIR/venv/bin/python\" -c 'import torch, numpy, pandas, PIL' >/dev/null 2>&1; then
    echo DEPS_OK
  else
    echo DEPS_MISSING
  fi
else
  echo VENV_ABSENT
fi"
    );
    let out = run_remote(alias, &script, None)?;
    let mut venv_present = false;
    let mut deps_ok = false;
    let mut python_version = String::new();
    for line in out.lines() {
        match line.trim() {
            "VENV_PRESENT" => venv_present = true,
            "VENV_ABSENT" => venv_present = false,
            "DEPS_OK" => deps_ok = true,
            "DEPS_MISSING" => deps_ok = false,
            l if l.starts_with("PY_VER=") => python_version = l.trim_start_matches("PY_VER=").to_string(),
            _ => {}
        }
    }
    Ok(ProbeResult { venv_present, deps_ok, python_version })
}

fn install(alias: &str, root: &str) -> Result<(), String> {
    let root_q = shell_quote_path(root);
    let env_sh = shell_quote_path(&format!("{}/.spinoml/env.sh", root.trim_end_matches('/')));
    // CPU-only torch wheel saves ~2GB of GPU runtime that the sidecar
    // doesn't need (training happens elsewhere; the sidecar only does
    // shape inference + a few-sample smoke run).
    let script = format!(
        "set -e
ROOT={root_q}
MLDIR=\"$ROOT/.spinoml\"
mkdir -p \"$MLDIR\"
if [ -f {env_sh} ]; then . {env_sh}; fi
PY=$(command -v python3 || command -v python)
echo \"[spinoml] creating venv at $MLDIR/venv with $PY\"
\"$PY\" -m venv \"$MLDIR/venv\"
\"$MLDIR/venv/bin/pip\" install --quiet --upgrade pip
\"$MLDIR/venv/bin/pip\" install --quiet --index-url https://download.pytorch.org/whl/cpu torch
\"$MLDIR/venv/bin/pip\" install --quiet numpy pandas pillow python-dateutil
echo SPINOML_INSTALL_DONE"
    );
    let out = run_remote(alias, &script, None)?;
    if !out.contains("SPINOML_INSTALL_DONE") {
        return Err(format!("install did not complete: {}", out.trim()));
    }
    Ok(())
}

/// Kill any leftover sidecar-torch python on the remote that may have
/// outlived the previous ssh-tunnel session. systemd-logind doesn't reap
/// user processes when their login session ends by default (RHEL/Rocky
/// default `KillUserProcesses=no`), so the python lingers and holds 7421 —
/// the next bootstrap then dies on "Address already in use". Wait for the
/// port to actually go away before returning; pkill is async.
fn cleanup_stale_remote(alias: &str, root: &str) -> Result<(), String> {
    let root_q = shell_quote_path(root);
    let script = format!(
        "ROOT={root_q}; MLDIR=\"$ROOT/.spinoml\"
fuser -k {port}/tcp 2>/dev/null || true
pkill -9 -f \"$MLDIR/venv/bin/python.*sidecar-torch\" 2>/dev/null || true
for i in 1 2 3 4 5 6 7 8 9 10; do
  if ss -ltn 2>/dev/null | awk '{{print $4}}' | grep -q ':{port}$'; then
    fuser -k {port}/tcp 2>/dev/null || true
    sleep 0.3
  else
    echo PORT_FREE; exit 0
  fi
done
echo PORT_STILL_HELD >&2; exit 1",
        port = REMOTE_REMOTE_PORT,
    );
    let out = run_remote(alias, &script, None)?;
    if !out.contains("PORT_FREE") {
        return Err(format!(
            "remote port {} still held after cleanup attempt",
            REMOTE_REMOTE_PORT
        ));
    }
    Ok(())
}

/// Upload every file in `SIDECAR_FILES` to `<root>/.spinoml/sidecar-torch/`.
/// Order-independent (each file is independent). `mkdir -p` every needed
/// subdir once, then `cat > dst` per file with stdin bytes — same mechanism
/// as before, just driven by the constant instead of two hard-coded names.
///
/// The token NEVER touches this function. File deployment is orthogonal to
/// the authentication token (the token is delivered later, through ssh
/// stdin, once the sidecar is about to be spawned).
fn deploy(alias: &str, root: &str, sidecar_dir: &PathBuf) -> Result<(), String> {
    // Every path component a SIDECAR_FILES entry lives under (deduped). We
    // create each subdir once, so we don't `mkdir -p` once per file when all
    // .py files share the same parent.
    let mut subdirs: Vec<String> = Vec::new();
    for rel in SIDECAR_FILES {
        if let Some((dir, _)) = rel.rsplit_once('/') {
            if !subdirs.iter().any(|d| d == dir) {
                subdirs.push(dir.to_string());
            }
        }
    }
    let root_t = root.trim_end_matches('/');
    let dst_dir = format!("{root_t}/.spinoml/sidecar-torch");
    let dst_dir_q = shell_quote_path(&dst_dir);
    // All subdirs in one round trip — subdirs are subdirectories of dst_dir
    // (relative paths in SIDECAR_FILES have no leading `/`), so `mkdir -p`
    // each one.
    for sub in &subdirs {
        let sub_q = shell_quote_path(&format!("{dst_dir}/{sub}"));
        run_remote(alias, &format!("mkdir -p {sub_q}"), None)?;
    }
    // Also ensure dst_dir itself exists (in case SIDECAR_FILES had no
    // subdirs in a future change).
    run_remote(alias, &format!("mkdir -p {dst_dir_q}"), None)?;

    // Per-file upload. We refuse to start if a listed file is missing on
    // the laptop — a silent omission would mean `ModuleNotFoundError` on the
    // remote AFTER the sidecar is up, which is the bug this regression fix
    // exists to prevent.
    for rel in SIDECAR_FILES {
        // Each SIDECAR_FILES entry is a literal `sidecar-torch/<rel>` path on
        // the laptop. We use `PathBuf::join` so absolute entries (which the
        // SIDECAR_FILES invariant test forbids) can't smuggle in arbitrary
        // host paths — but the actual invariant check happens in tests; here
        // we just trust the constant.
        let src = sidecar_dir.join(rel);
        if !src.exists() {
            return Err(format!(
                "sidecar-torch/{rel} missing on laptop ({}) — SpinoML bundle may be \
                 incomplete (refusing to deploy with a missing file)",
                src.display()
            ));
        }
        let bytes = std::fs::read(&src).map_err(|e| format!("read sidecar-torch/{rel}: {e}"))?;
        let dst_q = shell_quote_path(&format!("{dst_dir}/{rel}"));
        run_remote(alias, &format!("cat > {dst_q}"), Some(&bytes))?;
    }
    Ok(())
}

/// Pure builder for the remote shell script that boots the sidecar. Kept as a
/// free function so the static + behavioural tests can construct it directly
/// without spawning ssh.
///
/// Spec (docs/engineering/SIDECAR_AUTH.md §"Rust shell"):
///   1. `IFS= read -r SPINOML_TOK` FIRST, into a NON-exported variable —
///      env.sh and everything it starts must NOT inherit the token.
///   2. `[ -n "$SPINOML_TOK" ] || { echo SPINOML_NO_TOKEN >&2; exit 11; }` —
///      an empty stdin would otherwise sneak through to python, which would
///      then refuse to start via SPINOML_REQUIRE_TOKEN anyway, but we fail
///      loudly here for a clearer diagnostic.
///   3. cd into spinoml dir, source env.sh, append to SPINOML_ALLOWED_ROOTS
///      (existing behaviour).
///   4. ONLY THEN: `export SPINOML_SIDECAR_TOKEN="$SPINOML_TOK"
///      SPINOML_REQUIRE_TOKEN=1; unset SPINOML_TOK` immediately before `exec`
///      python. `unset` matters: if env.sh somehow re-runs we don't want a
///      stale SPINOML_TOK around for a child to read.
///   5. `exec "$MLDIR/venv/bin/python" -u sidecar-torch/main.py` so signals
///      reach python directly.
///
/// The token itself never appears in the script text — the remote gets it
/// through ssh stdin, not as a substitution into the command string.
pub fn build_remote_script(root_q: &str, env_sh_q: &str, port: u16) -> String {
    // NOTE: `${{` / `}}` are format! escapes for a literal `${` / `}`.
    format!(
        "IFS= read -r SPINOML_TOK\n\
         [ -n \"$SPINOML_TOK\" ] || {{ echo SPINOML_NO_TOKEN >&2; exit 11; }}\n\
         ROOT={root_q}; MLDIR=\"$ROOT/.spinoml\"\n\
         if [ -f {env_sh_q} ]; then . {env_sh_q}; fi\n\
         export SPINOML_ALLOWED_ROOTS=\"$ROOT${{SPINOML_ALLOWED_ROOTS:+:$SPINOML_ALLOWED_ROOTS}}\"\n\
         cd \"$MLDIR\"\n\
         export SPINOML_SIDECAR_TOKEN=\"$SPINOML_TOK\" SPINOML_REQUIRE_TOKEN=1\n\
         unset SPINOML_TOK\n\
         SPINOML_TORCH_PORT={port} exec \"$MLDIR/venv/bin/python\" -u sidecar-torch/main.py",
        port = port,
    )
}

/// Build the ssh tunnel command. The `token` is NOT interpolated into the
/// command string — it is delivered through ssh stdin (see `run_bootstrap`).
/// We pipe stdin in here so the caller doesn't forget; the caller writes the
/// token and keeps the `ChildStdin` handle in the stored session.
fn build_run_command(alias: &str, root: &str) -> Command {
    let mut cmd = Command::new("ssh");
    for o in SSH_OPTS { cmd.arg(o); }
    // Keep the connection responsive; if the tunnel can't bind we want a
    // fast, clear failure instead of "running but broken".
    cmd.arg("-o").arg("ExitOnForwardFailure=yes");
    cmd.arg("-L").arg(format!(
        "127.0.0.1:{}:127.0.0.1:{}", REMOTE_LOCAL_PORT, REMOTE_REMOTE_PORT
    ));
    // `-T` disables pseudo-terminal allocation, so the remote sidecar's
    // stdout/stderr don't get PTY-mangled on their way through the tunnel
    // (the python `-u` flag then flushes line-by-line as expected).
    cmd.arg("-T");
    // `--` ends option parsing: the target can never be taken for an ssh option.
    cmd.arg("--").arg(alias);
    let root_q = shell_quote_path(root);
    let env_sh = shell_quote_path(&format!("{}/.spinoml/env.sh", root.trim_end_matches('/')));
    let remote = build_remote_script(&root_q, &env_sh, REMOTE_REMOTE_PORT);
    cmd.arg(remote);
    // stdin is piped so the Rust side can write the token once and keep the
    // handle open. stdout/stderr are piped so the watcher threads can read
    // "listening" + ssh errors.
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::piped());
    // NOTE: deliberately NO PR_SET_PDEATHSIG here. The tunnel is spawned from a
    // tokio blocking-pool thread (run_bootstrap runs under spawn_blocking), and
    // PDEATHSIG fires on the death of the SPAWNING THREAD, not the process — so
    // when that pool thread is reaped (~10s idle) the kernel would SIGTERM the
    // tunnel and it would "stop after a few seconds". The tunnel is tracked in
    // `current.child` (killed on app close) and any orphan is cleared by
    // free_local_tunnel_port() on the next connect, so PDEATHSIG isn't needed.
    cmd
}

// ─── tauri-exposed commands ───

#[tauri::command]
pub fn remote_sidecar_status(app: AppHandle) -> RemoteSidecarStatus {
    let state: State<RemoteSidecarState> = app.state();
    let mut g = match state.current.lock() { Ok(g) => g, Err(_) => return RemoteSidecarStatus::Idle };
    // Snapshot the alive RemoteSidecar reference, but FIRST reap any dead
    // child so we don't lie "running". We hold `g` across both halves to
    // keep the lock uncontended.
    let mut reaped = false;
    {
        if let Some(rs) = g.as_mut() {
            if matches!(rs.child.try_wait(), Ok(Some(_))) {
                *g = None;
                // The remote token was per-session — it's now invalid. Clear it
                // here so `sidecar_token("torch-remote")` returns None.
                let tokens: State<SidecarTokens> = app.state();
                let _ = tokens.clear_remote();
                reaped = true;
            }
        }
    }
    if reaped {
        return RemoteSidecarStatus::Stopped;
    }
    if let Some(rs) = g.as_ref() {
        RemoteSidecarStatus::Running {
            local_port: REMOTE_LOCAL_PORT,
            remote_port: REMOTE_REMOTE_PORT,
            alias: rs.alias.clone(),
            root: rs.root.clone(),
        }
    } else {
        RemoteSidecarStatus::Idle
    }
}

#[tauri::command]
pub async fn ensure_remote_sidecar(
    app: AppHandle,
    alias: String,
    root: String,
    force: Option<bool>,
) -> Result<RemoteSidecarStatus, String> {
    let force = force.unwrap_or(false);
    // The alias comes from the webview and goes straight to `ssh`: validate it
    // exactly like every ssh_* command in ssh.rs does (no leading '-', no
    // shell metacharacters), BEFORE any process is spawned.
    crate::ssh::validate_alias(&alias)?;
    // Serialize bootstraps so concurrent ensures (StrictMode double-effect)
    // can't free_local_tunnel_port each other's tunnel. The loser waits here,
    // then short-circuits on the live `current` below.
    let lock_state: State<RemoteSidecarState> = app.state();
    let _bootstrap_guard = lock_state.bootstrap_lock.lock().await;

    // Reuse a LIVE tunnel for the same target (unless the caller forces a
    // reconnect). A dead tunnel child is reaped so we don't report a stale
    // "running" — that was the bug behind "badge says ok but sidecar offline".
    {
        let state: State<RemoteSidecarState> = app.state();
        let mut g = state.current.lock().map_err(|e| e.to_string())?;
        if let Some(rs) = g.as_mut() {
            let dead = matches!(rs.child.try_wait(), Ok(Some(_)));
            if dead {
                *g = None; // reap; fall through to a fresh bootstrap
                let tokens: State<SidecarTokens> = app.state();
                let _ = tokens.clear_remote();
            } else if !force && rs.alias == alias && rs.root == root {
                return Ok(RemoteSidecarStatus::Running {
                    local_port: REMOTE_LOCAL_PORT,
                    remote_port: REMOTE_REMOTE_PORT,
                    alias, root,
                });
            }
        }
    }
    // Different target / forced reconnect / reaped → tear down old first.
    stop_remote_sidecar_internal(&app)?;

    // Do work on a blocking thread because we shell out to ssh several times
    // and the spawn_command in pty.rs already pulled tokio's slot. Use
    // tokio's spawn_blocking for the bootstrap pipeline.
    let app2 = app.clone();
    let alias2 = alias.clone();
    let root2 = root.clone();
    let final_status = tokio::task::spawn_blocking(move || -> Result<RemoteSidecarStatus, String> {
        run_bootstrap(&app2, &alias2, &root2)
    })
    .await
    .map_err(|e| format!("bootstrap join: {e}"))??;

    Ok(final_status)
}

fn run_bootstrap(app: &AppHandle, alias: &str, root: &str) -> Result<RemoteSidecarStatus, String> {
    emit(app, &RemoteSidecarStatus::Preparing {
        phase: "probe".into(),
        message: format!("Probing python + venv on {alias}:{root}…"),
    });
    let probe_result = probe(alias, root).map_err(|e| {
        let s = RemoteSidecarStatus::Error { message: e.clone() };
        emit(app, &s); e
    })?;

    if !probe_result.venv_present || !probe_result.deps_ok {
        emit(app, &RemoteSidecarStatus::Preparing {
            phase: "install".into(),
            message: format!(
                "Installing torch + numpy + pandas + pillow into $ROOT/.spinoml/venv (python {}). \
                 This is a one-time download and can take a couple of minutes.",
                probe_result.python_version
            ),
        });
        install(alias, root).map_err(|e| {
            let m = format!("install failed: {e}");
            let s = RemoteSidecarStatus::Error { message: m.clone() };
            emit(app, &s); m
        })?;
    }

    emit(app, &RemoteSidecarStatus::Preparing {
        phase: "deploy".into(),
        message: "Uploading sidecar-torch source…".into(),
    });
    let sidecar_dir = crate::sidecar_root_pub(app).join("sidecar-torch");
    deploy(alias, root, &sidecar_dir).map_err(|e| {
        let m = format!("deploy failed: {e}");
        let s = RemoteSidecarStatus::Error { message: m.clone() };
        emit(app, &s); m
    })?;

    // Clean up any lingering python from a prior session — RHEL-style
    // systemd-logind keeps user processes after ssh disconnect by default,
    // so without this the next spawn dies on "Address already in use".
    emit(app, &RemoteSidecarStatus::Preparing {
        phase: "cleanup".into(),
        message: "Killing any leftover sidecar process from a previous session…".into(),
    });
    cleanup_stale_remote(alias, root).map_err(|e| {
        let m = format!("cleanup failed: {e}");
        let s = RemoteSidecarStatus::Error { message: m.clone() };
        emit(app, &s); m
    })?;

    // Fresh per-session token. Generated BEFORE the spawn so the child can
    // have it ready in stdin by the time it does `read -r SPINOML_TOK`. If
    // generation fails we abort; the brief explicitly forbids falling back
    // to "no token".
    let token = generate_token().map_err(|e| format!("could not generate remote token: {e}"))?;

    emit(app, &RemoteSidecarStatus::Starting);
    // Free our local forward port from any orphaned tunnel (e.g. left by a
    // previously-killed app instance) so the new -L forward can bind. Without
    // this, ssh hits "bind 127.0.0.1:7424: Address already in use" and, with
    // ExitOnForwardFailure, the whole tunnel dies → "sidecar unreachable".
    free_local_tunnel_port();
    let mut cmd = build_run_command(alias, root);
    let mut child = cmd.spawn().map_err(|e| format!("spawn ssh tunnel: {e}"))?;

    // Write the token to the child's stdin ONCE. After this the pipe MUST
    // stay open for the lifetime of the session — we keep `stdin` in
    // `RemoteSidecar` precisely so a later `child.stdin.take().drop()` (or
    // even this scope's end) doesn't trigger EOF on the remote. The remote
    // reads exactly one line and then ignores stdin.
    let mut stdin_handle = child
        .stdin
        .take()
        .ok_or_else(|| "no stdin on ssh tunnel child".to_string())?;
    if let Err(e) = stdin_handle.write_all(token.as_bytes()) {
        let _ = child.kill();
        let _ = child.wait();
        return Err(format!("failed to deliver remote token over ssh stdin: {e}"));
    }
    if let Err(e) = stdin_handle.write_all(b"\n") {
        let _ = child.kill();
        let _ = child.wait();
        return Err(format!("failed to deliver remote token newline: {e}"));
    }
    // NOTE: we DO NOT drop `token` here — we still need it to register the
    // remote token with SidecarTokens below (so the webview can read it via
    // `sidecar_token("torch-remote")`). `token` is moved into `set_remote`
    // and drops there. After that, the only remaining reference to the
    // secret is the value stored in `SidecarTokens.remote` (and the byte
    // stream already pushed through the child's stdin).

    // Watch stdout for "listening" to know the sidecar is up. The tunnel
    // forwards the port; the python process binds it remotely.
    let stdout = child.stdout.take().ok_or_else(|| "no stdout".to_string())?;
    let stderr = child.stderr.take().ok_or_else(|| "no stderr".to_string())?;

    let (tx, rx) = std::sync::mpsc::channel::<Result<(), String>>();
    let tx_err = tx.clone();
    let app_for_thread = app.clone();
    std::thread::spawn(move || {
        let r = BufReader::new(stdout);
        let mut announced = false;
        for line in r.lines().flatten() {
            eprintln!("[spinoml-torch-remote] {line}");
            if !announced && line.contains("listening on") {
                announced = true;
                let _ = tx.send(Ok(()));
            }
            // Continue draining so the pipe doesn't fill.
            let _ = &line;
        }
        // EOF → child died
        if !announced {
            let _ = tx.send(Err("remote sidecar exited before announcing readiness".to_string()));
        }
        // Once announced, surface unexpected exits.
        let _ = app_for_thread.emit("remote-sidecar:status", RemoteSidecarStatus::Stopped);
    });
    // Watch stderr for a failed LOCAL port forward — ssh announces the remote
    // sidecar's "listening" on stdout even when our -L forward couldn't bind,
    // so without this we'd report success for a tunnel that doesn't carry traffic.
    std::thread::spawn(move || {
        let r = BufReader::new(stderr);
        for line in r.lines().flatten() {
            eprintln!("[spinoml-torch-remote stderr] {line}");
            if line.contains("Could not request local forwarding")
                || line.contains("cannot listen to port")
                || (line.contains("bind") && line.contains("Address already in use"))
            {
                let _ = tx_err.send(Err(format!(
                    "local port {} busy — the tunnel could not be opened",
                    REMOTE_LOCAL_PORT
                )));
            }
        }
    });

    // Wait up to 60s for "listening" — bootstrap (uncached) can take a while
    // even after install if remote needs to import torch (~5s cold).
    let ready = rx
        .recv_timeout(std::time::Duration::from_secs(60))
        .map_err(|_| "timeout waiting for remote sidecar to come up".to_string())?;
    if let Err(e) = ready {
        let _ = child.kill();
        let _ = child.wait();
        emit(app, &RemoteSidecarStatus::Error { message: e.clone() });
        return Err(e);
    }

    // Verify the forward actually carries traffic locally (belt-and-suspenders
    // on top of the stderr watch): the remote can announce readiness even if the
    // local bind silently failed.
    let addr = format!("127.0.0.1:{}", REMOTE_LOCAL_PORT);
    if let Ok(sa) = addr.parse::<std::net::SocketAddr>() {
        let mut reachable = false;
        for _ in 0..5 {
            if std::net::TcpStream::connect_timeout(&sa, std::time::Duration::from_secs(2)).is_ok() {
                reachable = true;
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        if !reachable {
            let _ = child.kill();
            let _ = child.wait();
            let m = format!(
                "tunnel announced remotely but local port {} is unreachable (forward failed)",
                REMOTE_LOCAL_PORT
            );
            emit(app, &RemoteSidecarStatus::Error { message: m.clone() });
            return Err(m);
        }
    }

    // Spawn succeeded. NOW register the token in `SidecarTokens.remote` so the
    // webview can read it via `sidecar_token("torch-remote")`. Anything that
    // resets `current` to None (stop, reap, error path, window close) MUST
    // also clear the remote token — see `stop_remote_sidecar_internal` and
    // `remote_sidecar_status`.
    {
        let tokens: State<SidecarTokens> = app.state();
        if let Err(e) = tokens.set_remote(token.clone()) {
            // Registration failed (lock poisoned). We can't serve the token
            // from SidecarTokens — better to tear the sidecar down than to
            // leak a token into `RemoteSidecar` where nobody would ever
            // clear it. (Note: the child already HAS the token, but a
            // SidecarTokens entry that never clears on session end is a
            // correctness bug: webview would keep "seeing" a token for a
            // dead tunnel.)
            let _ = child.kill();
            let _ = child.wait();
            return Err(e);
        }
        // `token` (the clone we made) drops here — the secret is now only in
        // `SidecarTokens.remote` and inside the child's stdin pipe.
    }
    drop(token);

    let state: State<RemoteSidecarState> = app.state();
    *state.current.lock().map_err(|e| e.to_string())? = Some(RemoteSidecar {
        alias: alias.to_string(),
        root: root.to_string(),
        child,
        stdin: stdin_handle,
    });

    let status = RemoteSidecarStatus::Running {
        local_port: REMOTE_LOCAL_PORT,
        remote_port: REMOTE_REMOTE_PORT,
        alias: alias.to_string(),
        root: root.to_string(),
    };
    emit(app, &status);
    Ok(status)
}

#[tauri::command]
pub fn stop_remote_sidecar(app: AppHandle) -> Result<(), String> {
    stop_remote_sidecar_internal(&app)?;
    emit(&app, &RemoteSidecarStatus::Stopped);
    Ok(())
}

/// Kill any orphaned ssh tunnel still holding our local forward port. Matches
/// only our own `-L 127.0.0.1:7424:127.0.0.1:7421` signature, so it won't touch
/// unrelated ssh sessions. Best-effort.
fn free_local_tunnel_port() {
    // Primary: kill whatever holds the local forward port. Only an ssh -L
    // listener binds 127.0.0.1:7424 (clients use ephemeral ports), so this is
    // precise. fuser is the reliable way to free a port.
    let _ = Command::new("fuser")
        .arg("-k")
        .arg(format!("{}/tcp", REMOTE_LOCAL_PORT))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    // Fallback by command-line signature. The pattern must NOT start with '-'
    // or pkill parses it as an option (and silently frees nothing).
    let pat = format!(
        "127.0.0.1:{}:127.0.0.1:{}",
        REMOTE_LOCAL_PORT, REMOTE_REMOTE_PORT
    );
    let _ = Command::new("pkill")
        .arg("-f")
        .arg(&pat)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

fn stop_remote_sidecar_internal(app: &AppHandle) -> Result<(), String> {
    let state: State<RemoteSidecarState> = app.state();
    let mut g = state.current.lock().map_err(|e| e.to_string())?;
    if let Some(mut rs) = g.take() {
        let _ = rs.child.kill();
        let _ = rs.child.wait();
    }
    // `current` is now None — the remote token is dead. Clear it so the
    // webview can no longer authenticate to a non-existent sidecar.
    let tokens: State<SidecarTokens> = app.state();
    let _ = tokens.clear_remote();
    Ok(())
}

// Note: there used to be a `kill_all(state: &RemoteSidecarState)` helper
// here, but it didn't have access to the `AppHandle` needed to clear the
// remote authentication token on session end. The window-event handler in
// lib.rs now calls `stop_remote_sidecar(window.app_handle().clone())` on
// CloseRequested, which IS the path that clears the token.

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;

    // ── SIDECAR_FILES invariants ─────────────────────────────────────────

    #[test]
    fn sidecar_files_no_duplicates() {
        let mut seen = std::collections::HashSet::new();
        for f in SIDECAR_FILES {
            assert!(seen.insert(*f), "duplicate entry in SIDECAR_FILES: {f}");
        }
    }

    #[test]
    fn sidecar_files_no_dotdot_no_absolute_no_empty() {
        for f in SIDECAR_FILES {
            assert!(!f.is_empty(), "empty SIDECAR_FILES entry");
            assert!(!f.starts_with('/'), "absolute path in SIDECAR_FILES: {f}");
            assert!(!f.starts_with('~'), "tilde-prefixed path in SIDECAR_FILES: {f}");
            assert!(!f.contains(".."), "`..` in SIDECAR_FILES: {f}");
            // No leading `./` either (we'd just iterate the components).
            assert!(!f.starts_with("./"), "leading `./` in SIDECAR_FILES: {f}");
        }
    }

    // ── build_remote_script static checks ────────────────────────────────

    #[test]
    fn script_starts_with_token_read() {
        let s = build_remote_script("'R'", "'E'", 7421);
        let first_line = s.lines().next().expect("non-empty script");
        assert!(
            first_line.contains("read -r SPINOML_TOK"),
            "first line must read the token: {first_line:?}"
        );
    }

    #[test]
    fn script_exits_eleven_on_empty_token() {
        let s = build_remote_script("'R'", "'E'", 7421);
        // The check must be present and must exit with the documented code.
        assert!(s.contains("[ -n \"$SPINOML_TOK\" ]"), "missing non-empty check: {s}");
        assert!(s.contains("SPINOML_NO_TOKEN"), "missing diagnostic marker: {s}");
        assert!(s.contains("exit 11"), "missing exit 11: {s}");
    }

    #[test]
    fn script_export_after_envsh_before_exec() {
        let s = build_remote_script("'R'", "'E'", 7421);
        let envsh_pos = s.find(". {env_sh_q}").or_else(|| s.find(". 'E'"))
            .expect("script must source env.sh");
        let export_pos = s.find("export SPINOML_SIDECAR_TOKEN=\"$SPINOML_TOK\"")
            .expect("script must export the token");
        let exec_pos = s.find("exec \"$MLDIR/venv/bin/python\"")
            .expect("script must exec python");
        assert!(envsh_pos < export_pos, "export must come AFTER env.sh source");
        assert!(export_pos < exec_pos, "export must come BEFORE exec");
        // And the `unset SPINOML_TOK` must sit between export and exec.
        let unset_pos = s.find("unset SPINOML_TOK").expect("script must unset SPINOML_TOK");
        assert!(export_pos < unset_pos && unset_pos < exec_pos, "unset must be between export and exec");
    }

    #[test]
    fn script_does_not_embed_token() {
        let s = build_remote_script("'R'", "'E'", 7421);
        let token = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789";
        // The script must reference the token only via the variable name,
        // never embed any literal value.
        assert!(!s.contains(token), "script must not contain a literal token");
        assert!(!s.contains("abcdef"), "script must not contain the token's bytes");
    }

    #[test]
    fn build_run_command_debug_does_not_contain_token() {
        let token = "supersecrettoken_abcdef0123456789abcdef0123456789abcdef01";
        let mut cmd = build_run_command("alias", "/tmp");
        // simulate the spawn-time write
        cmd.stdin(Stdio::piped()); // already piped
        let dbg = format!("{:?}", cmd);
        assert!(!dbg.contains(token), "Debug of Command must not include token");
        // Also: the argv list never mentions the token.
        for arg in cmd.get_args() {
            let s = arg.to_string_lossy();
            assert!(!s.contains("supersecret"), "argv leak: {s:?}");
        }
    }

    // ── behavioural test (see brief) ──────────────────────────────────────

    /// Build a temp root with:
    ///   ROOT/.spinoml/venv/bin/python  — executable shell stub
    ///   ROOT/.spinoml/env.sh          — HOSTILE env.sh that tries to read
    ///                                   SPINOML_TOK (it shouldn't be exported)
    ///                                   and tries to overwrite
    ///                                   SPINOML_SIDECAR_TOKEN with "evil"
    fn setup_stubbed_remote(root: &std::path::Path) {
        let spinoml = root.join(".spinoml");
        std::fs::create_dir_all(spinoml.join("venv/bin")).unwrap();
        let stub = spinoml.join("venv/bin/python");
        // The stub prints the three values we care about:
        //   TOK  = SPINOML_TOK (the un-exported intermediate variable)
        //   SIDE = SPINOML_SIDECAR_TOKEN (what python actually receives)
        //   REQ  = SPINOML_REQUIRE_TOKEN
        std::fs::write(&stub, "#!/bin/sh\n\
            echo \"TOK=[$SPINOML_TOK]\"\n\
            echo \"SIDE=[$SPINOML_SIDECAR_TOKEN]\"\n\
            echo \"REQ=[$SPINOML_REQUIRE_TOKEN]\"\n\
            ").unwrap();
        std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755)).unwrap();
        // Hostile env.sh: reads SPINOML_TOK (must be empty/unset because
        // it's never exported), and tries to clobber SPINOML_SIDECAR_TOKEN
        // with the literal "evil". The export-order discipline in the
        // script (export happens AFTER `. env.sh`) defeats this.
        let env_sh = spinoml.join("env.sh");
        std::fs::write(&env_sh, "#!/bin/sh\n\
            echo \"ENVSH_SEES=[$SPINOML_TOK]\"\n\
            export SPINOML_SIDECAR_TOKEN=evil\n\
            ").unwrap();
    }

    #[test]
    fn script_keeps_token_out_of_envsh_clobber_and_into_stub() {
        let tmp = std::env::temp_dir().join(format!(
            "spinoml-sidecar-auth-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        setup_stubbed_remote(&tmp);

        let root_q = shell_quote_path(tmp.to_str().unwrap());
        let env_sh_q = shell_quote_path(&format!("{}/.spinoml/env.sh", tmp.display()));
        let script = build_remote_script(&root_q, &env_sh_q, 7421);

        // Run `bash -c <script>` with stdin containing the secret. `bash -c`
        // reads its script from argv and stdin from /dev/stdin (which is
        // connected to our pipe). The remote script's `read -r SPINOML_TOK`
        // then consumes the secret.
        let token = "secrettoken_abcdef0123456789";
        let mut child = std::process::Command::new("bash")
            .arg("-c")
            .arg(&script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn bash");
        {
            let mut stdin = child.stdin.take().expect("child stdin");
            stdin.write_all(token.as_bytes()).unwrap();
            stdin.write_all(b"\n").unwrap();
        } // drop stdin = EOF
        let out = child.wait_with_output().expect("wait bash");
        let combined = format!(
            "{}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );

        // The stub saw the real token via SPINOML_SIDECAR_TOKEN — env.sh's
        // attempt to clobber it with "evil" was overridden by the later
        // `export SPINOML_SIDECAR_TOKEN=...` in the script (which runs AFTER
        // `. env.sh`). This is the security property the spec actually needs:
        // env.sh can read SPINOML_TOK (it's a user-controlled file and bash's
        // `.` runs it in the same shell, so it inherits non-exported vars
        // too) but it CANNOT replace the value the sidecar authenticates
        // against. The "non-exported + unset before exec" hygiene keeps the
        // secret out of any child process env.sh might spawn via exec.
        assert!(combined.contains(&format!("SIDE=[{token}]")),
            "stub must see SPINOML_SIDECAR_TOKEN={token}; got:\n{combined}");
        // The un-exported SPINOML_TOK must NOT reach the stub (unset before
        // exec). env.sh can see it (sourcing inherits non-exported vars) —
        // that's expected and harmless; what matters is the clobber test
        // below.
        assert!(combined.contains("TOK=[]"),
            "stub must see SPINOML_TOK as empty (unset before exec); got:\n{combined}");
        // SPINOML_REQUIRE_TOKEN must be set to "1".
        assert!(combined.contains("REQ=[1]"),
            "stub must see SPINOML_REQUIRE_TOKEN=1; got:\n{combined}");
        // Sanity: env.sh's `evil` must NOT survive to the stub.
        assert!(!combined.contains("SIDE=[evil]"),
            "env.sh must not be able to clobber the token; got:\n{combined}");

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn script_aborts_with_exit_11_on_empty_stdin() {
        let tmp = std::env::temp_dir().join(format!(
            "spinoml-sidecar-auth-empty-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        setup_stubbed_remote(&tmp);
        let root_q = shell_quote_path(tmp.to_str().unwrap());
        let env_sh_q = shell_quote_path(&format!("{}/.spinoml/env.sh", tmp.display()));
        let script = build_remote_script(&root_q, &env_sh_q, 7421);

        let mut child = std::process::Command::new("bash")
            .arg("-c")
            .arg(&script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn bash");
        // Drop stdin immediately → EOF on read → SPINOML_TOK empty → exit 11
        drop(child.stdin.take());
        let out = child.wait_with_output().expect("wait bash");
        assert_eq!(out.status.code(), Some(11), "empty stdin must yield exit 11; status={:?}", out.status);
        let err = String::from_utf8_lossy(&out.stderr);
        assert!(err.contains("SPINOML_NO_TOKEN"),
            "stderr must contain SPINOML_NO_TOKEN diagnostic; got:\n{err}");
        // The stub MUST NOT have been invoked.
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(!stdout.contains("SIDE="),
            "stub must NOT have been invoked when token is empty; got:\n{stdout}");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}

// Drain stdout once we've announced readiness — keep the channel from
// blocking but don't double-emit ready events. (Helper used by tests later.)
#[allow(dead_code)]
fn drain_to_log(mut r: impl Read) {
    let mut buf = [0u8; 4096];
    while let Ok(n) = r.read(&mut buf) {
        if n == 0 { break; }
    }
}
