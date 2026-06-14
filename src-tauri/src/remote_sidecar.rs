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
// Env preference: a fresh venv at `<root>/.mlforge/venv/`. The user can
// pre-create that path with their own python (module load + python -m venv)
// and we'll detect + reuse. A `<root>/.mlforge/env.sh` is sourced before
// every command if present — single-file escape hatch for users who need
// `module load python` or `conda activate` first.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};

use crate::ssh::{shell_quote_path, SSH_OPTS};

pub const REMOTE_LOCAL_PORT: u16 = 7424;
pub const REMOTE_REMOTE_PORT: u16 = 7421;

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
    pub child: Child,
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
    cmd.arg(alias).arg(remote_cmd);
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
    let env_sh = shell_quote_path(&format!("{}/.mlforge/env.sh", root.trim_end_matches('/')));
    let script = format!(
        "set -e
ROOT={root_q}
MLDIR=\"$ROOT/.mlforge\"
mkdir -p \"$MLDIR\"
# optional env.sh hook (module loads / conda activate)
if [ -f {env_sh} ]; then . {env_sh}; fi
PY=$(command -v python3 || command -v python || true)
[ -n \"$PY\" ] || {{ echo MLFORGE_NO_PYTHON >&2; exit 10; }}
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
    let env_sh = shell_quote_path(&format!("{}/.mlforge/env.sh", root.trim_end_matches('/')));
    // CPU-only torch wheel saves ~2GB of GPU runtime that the sidecar
    // doesn't need (training happens elsewhere; the sidecar only does
    // shape inference + a few-sample smoke run).
    let script = format!(
        "set -e
ROOT={root_q}
MLDIR=\"$ROOT/.mlforge\"
mkdir -p \"$MLDIR\"
if [ -f {env_sh} ]; then . {env_sh}; fi
PY=$(command -v python3 || command -v python)
echo \"[mlforge] creating venv at $MLDIR/venv with $PY\"
\"$PY\" -m venv \"$MLDIR/venv\"
\"$MLDIR/venv/bin/pip\" install --quiet --upgrade pip
\"$MLDIR/venv/bin/pip\" install --quiet --index-url https://download.pytorch.org/whl/cpu torch
\"$MLDIR/venv/bin/pip\" install --quiet numpy pandas pillow python-dateutil
echo MLFORGE_INSTALL_DONE"
    );
    let out = run_remote(alias, &script, None)?;
    if !out.contains("MLFORGE_INSTALL_DONE") {
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
        "ROOT={root_q}; MLDIR=\"$ROOT/.mlforge\"
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

fn deploy(alias: &str, root: &str, sidecar_dir: &PathBuf) -> Result<(), String> {
    // Upload main.py and dataset_handlers.py. We could tar but two files
    // via stdin is simpler and doesn't depend on local `tar`.
    let main_py = sidecar_dir.join("main.py");
    let ds_py = sidecar_dir.join("dataset_handlers.py");
    if !main_py.exists() || !ds_py.exists() {
        return Err(format!(
            "sidecar-torch source missing on laptop ({}). MLForge bundle may be incomplete.",
            sidecar_dir.display()
        ));
    }
    let main_bytes = std::fs::read(&main_py).map_err(|e| format!("read main.py: {e}"))?;
    let ds_bytes = std::fs::read(&ds_py).map_err(|e| format!("read dataset_handlers.py: {e}"))?;

    let root_t = root.trim_end_matches('/');
    let dst_dir = format!("{root_t}/.mlforge/sidecar-torch");
    let dst_dir_q = shell_quote_path(&dst_dir);
    let main_q = shell_quote_path(&format!("{dst_dir}/main.py"));
    let ds_q = shell_quote_path(&format!("{dst_dir}/dataset_handlers.py"));

    run_remote(alias, &format!("mkdir -p {dst_dir_q}"), None)?;
    run_remote(alias, &format!("cat > {main_q}"), Some(&main_bytes))?;
    run_remote(alias, &format!("cat > {ds_q}"), Some(&ds_bytes))?;
    Ok(())
}

fn build_run_command(alias: &str, root: &str) -> Command {
    let mut cmd = Command::new("ssh");
    for o in SSH_OPTS { cmd.arg(o); }
    // Keep the connection responsive; if the tunnel can't bind we want a
    // fast, clear failure instead of "running but broken".
    cmd.arg("-o").arg("ExitOnForwardFailure=yes");
    cmd.arg("-L").arg(format!(
        "127.0.0.1:{}:127.0.0.1:{}", REMOTE_LOCAL_PORT, REMOTE_REMOTE_PORT
    ));
    cmd.arg(alias);
    let root_q = shell_quote_path(root);
    let env_sh = shell_quote_path(&format!("{}/.mlforge/env.sh", root.trim_end_matches('/')));
    // The remote command:
    //   1. cd into mlforge dir
    //   2. source env.sh if present (lets users `module load` first)
    //   3. exec the sidecar with the chosen port
    let remote = format!(
        "ROOT={root_q}; MLDIR=\"$ROOT/.mlforge\"; \
         if [ -f {env_sh} ]; then . {env_sh}; fi; \
         cd \"$MLDIR\" && MLFORGE_TORCH_PORT={port} exec \"$MLDIR/venv/bin/python\" -u sidecar-torch/main.py",
        port = REMOTE_REMOTE_PORT
    );
    cmd.arg(remote);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).stdin(Stdio::null());
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
pub fn remote_sidecar_status(state: State<RemoteSidecarState>) -> RemoteSidecarStatus {
    let mut g = match state.current.lock() { Ok(g) => g, Err(_) => return RemoteSidecarStatus::Idle };
    if let Some(rs) = g.as_mut() {
        // Reap a tunnel whose process has exited so we don't lie "running".
        if matches!(rs.child.try_wait(), Ok(Some(_))) {
            *g = None;
            return RemoteSidecarStatus::Stopped;
        }
        let rs = g.as_ref().unwrap();
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
                "Installing torch + numpy + pandas + pillow into $ROOT/.mlforge/venv (python {}). \
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

    emit(app, &RemoteSidecarStatus::Starting);
    // Free our local forward port from any orphaned tunnel (e.g. left by a
    // previously-killed app instance) so the new -L forward can bind. Without
    // this, ssh hits "bind 127.0.0.1:7424: Address already in use" and, with
    // ExitOnForwardFailure, the whole tunnel dies → "sidecar unreachable".
    free_local_tunnel_port();
    let mut cmd = build_run_command(alias, root);
    let mut child = cmd.spawn().map_err(|e| format!("spawn ssh tunnel: {e}"))?;

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
            eprintln!("[mlforge-torch-remote] {line}");
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
            eprintln!("[mlforge-torch-remote stderr] {line}");
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
            let m = format!(
                "tunnel announced remotely but local port {} is unreachable (forward failed)",
                REMOTE_LOCAL_PORT
            );
            emit(app, &RemoteSidecarStatus::Error { message: m.clone() });
            return Err(m);
        }
    }

    let state: State<RemoteSidecarState> = app.state();
    *state.current.lock().map_err(|e| e.to_string())? = Some(RemoteSidecar {
        alias: alias.to_string(),
        root: root.to_string(),
        child,
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
    Ok(())
}

pub fn kill_all(state: &RemoteSidecarState) {
    if let Ok(mut g) = state.current.lock() {
        if let Some(mut rs) = g.take() {
            let _ = rs.child.kill();
            let _ = rs.child.wait();
        }
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
