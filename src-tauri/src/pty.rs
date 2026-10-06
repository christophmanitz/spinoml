// PTY-backed terminal sessions. Each session attaches an OS PTY pair to
// either a local shell or to an `ssh -tt <alias>` subprocess. The PTY master
// is owned by Rust; xterm.js on the frontend writes keystrokes via the
// `pty_write` command and consumes stdout via Tauri events.
//
// One PtyState lives in the Tauri Builder. Sessions are indexed by a
// short ID generated at spawn time. Closing the app or the terminal tab
// kills the child process and drops the session.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use portable_pty::{native_pty_system, CommandBuilder, MasterPty, PtySize};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, State};

#[derive(Default)]
pub struct PtyState {
    pub sessions: Mutex<HashMap<String, Arc<PtySession>>>,
}

pub struct PtySession {
    pub master: Mutex<Box<dyn MasterPty + Send>>,
    pub writer: Mutex<Box<dyn Write + Send>>,
    pub child: Mutex<Box<dyn portable_pty::Child + Send + Sync>>,
}

fn random_id() -> String {
    // A per-process counter makes ids unique even when the clock reads before the
    // epoch (the 0 fallback below) or two sessions start in the same nanosecond —
    // a colliding id would silently replace the first session in the sessions map.
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0); // clock-before-epoch fallback; the process-wide counter keeps ids unique
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    format!("pty-{:x}-{:x}", now, seq)
}

#[derive(Deserialize)]
pub struct PtySpawnArgs {
    pub kind: String,
    pub local_cwd: Option<String>,
    pub alias: Option<String>,
    pub remote_root: Option<String>,
    pub cols: Option<u16>,
    pub rows: Option<u16>,
}

#[derive(Serialize)]
pub struct PtySpawnResult {
    pub id: String,
}

#[tauri::command]
pub fn pty_spawn(
    app: AppHandle,
    state: State<PtyState>,
    args: PtySpawnArgs,
) -> Result<PtySpawnResult, String> {
    let pty_system = native_pty_system();
    let pair = pty_system
        .openpty(PtySize {
            rows: args.rows.unwrap_or(30), // UI socket-size fallback; the terminal resizes again on first layout
            cols: args.cols.unwrap_or(100), // UI socket-size fallback; the terminal resizes again on first layout
            pixel_width: 0,
            pixel_height: 0,
        })
        .map_err(|e| format!("openpty: {e}"))?;

    let cmd_builder = build_command(&args)?;
    let child = pair
        .slave
        .spawn_command(cmd_builder)
        .map_err(|e| format!("spawn: {e}"))?;
    // We need slave alive only during spawn; dropping it lets the master
    // see EOF when the child exits.
    drop(pair.slave);

    let id = random_id();
    let mut reader = pair
        .master
        .try_clone_reader()
        .map_err(|e| format!("clone reader: {e}"))?;
    let writer = pair
        .master
        .take_writer()
        .map_err(|e| format!("take writer: {e}"))?;

    let session = Arc::new(PtySession {
        master: Mutex::new(pair.master),
        writer: Mutex::new(writer),
        child: Mutex::new(child),
    });

    state
        .sessions
        .lock()
        .map_err(|e| e.to_string())?
        .insert(id.clone(), session.clone());

    let app_handle = app.clone();
    let id_for_thread = id.clone();
    let data_event = format!("pty:{id}:data");
    let exit_event = format!("pty:{id}:exit");
    std::thread::spawn(move || {
        let mut buf = [0u8; 4096];
        loop {
            match reader.read(&mut buf) {
                Ok(0) => {
                    let _ = app_handle.emit(&exit_event, ()); // best-effort UI/channel send; a dropped receiver has nothing left to mislead
                    // Best-effort: drop the session from state so the next
                    // spawn doesn't think it's still alive.
                    if let Some(s) = app_handle
                        .try_state::<PtyState>()
                        .and_then(|s| s.sessions.lock().ok().and_then(|mut m| m.remove(&id_for_thread))) // a poisoned lock reads as the explicit unknown state, never a false value
                    {
                        // child already exited; nothing else to do.
                        drop(s);
                    }
                    break;
                }
                Ok(n) => {
                    let s = String::from_utf8_lossy(&buf[..n]).to_string();
                    let _ = app_handle.emit(&data_event, s); // best-effort UI/channel send; a dropped receiver has nothing left to mislead
                }
                Err(e) => {
                    let _ = app_handle.emit(&exit_event, format!("read error: {e}")); // best-effort UI/channel send; a dropped receiver has nothing left to mislead
                    break;
                }
            }
        }
    });

    Ok(PtySpawnResult { id })
}

fn build_command(args: &PtySpawnArgs) -> Result<CommandBuilder, String> {
    use std::env;
    match args.kind.as_str() {
        "local" => {
            let shell = env::var("SHELL").unwrap_or_else(|_| "bash".into()); // documented executable fallback; a missing binary surfaces as a spawn error
            let mut cmd = CommandBuilder::new(&shell);
            cmd.arg("-l");
            if let Some(cwd) = args.local_cwd.as_ref() {
                if !cwd.is_empty() {
                    cmd.cwd(cwd);
                }
            }
            cmd.env("TERM", "xterm-256color");
            cmd.env("SPINOML_TERMINAL", "local");
            Ok(cmd)
        }
        "remote-ssh" => {
            let alias = args
                .alias
                .as_ref()
                .ok_or_else(|| "remote-ssh requires alias".to_string())?;
            // Single source of truth for what a valid ssh target is (also
            // rejects a leading '-', which ssh would parse as an option).
            crate::ssh::validate_alias(alias)?;
            let mut cmd = CommandBuilder::new("ssh");
            for o in crate::ssh::SSH_OPTS_INTERACTIVE {
                cmd.arg(o);
            }
            cmd.arg("-tt");
            // `--` ends option parsing: the target can never be taken for an ssh option.
            cmd.arg("--");
            cmd.arg(alias);
            let remote_cmd = match args.remote_root.as_deref() {
                Some(r) if !r.is_empty() => {
                    if r.contains('\0') || r.contains('\n') || r.contains('\r') {
                        return Err("remote root has illegal chars".into());
                    }
                    format!(
                        "cd {} 2>/dev/null; exec ${{SHELL:-bash}} -l",
                        crate::ssh::shell_quote_path(r)
                    )
                }
                _ => "exec ${SHELL:-bash} -l".to_string(),
            };
            cmd.arg(remote_cmd);
            cmd.env("TERM", "xterm-256color");
            if let Some(cwd) = args.local_cwd.as_ref() {
                if !cwd.is_empty() {
                    cmd.cwd(cwd);
                }
            }
            Ok(cmd)
        }
        other => Err(format!("unknown pty kind: {other}")),
    }
}

#[tauri::command]
pub fn pty_write(state: State<PtyState>, id: String, data: String) -> Result<(), String> {
    let session = state
        .sessions
        .lock()
        .map_err(|e| e.to_string())?
        .get(&id)
        .cloned();
    let session = session.ok_or_else(|| format!("no pty session {id}"))?;
    let mut writer = session.writer.lock().map_err(|e| e.to_string())?;
    writer
        .write_all(data.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    Ok(())
}

#[tauri::command]
pub fn pty_resize(
    state: State<PtyState>,
    id: String,
    cols: u16,
    rows: u16,
) -> Result<(), String> {
    let session = state
        .sessions
        .lock()
        .map_err(|e| e.to_string())?
        .get(&id)
        .cloned();
    let session = session.ok_or_else(|| format!("no pty session {id}"))?;
    let master = session.master.lock().map_err(|e| e.to_string())?;
    master
        .resize(PtySize {
            rows,
            cols,
            pixel_width: 0,
            pixel_height: 0,
        })
        .map_err(|e| format!("resize: {e}"))?;
    Ok(())
}

#[tauri::command]
pub fn pty_kill(state: State<PtyState>, id: String) -> Result<(), String> {
    let session = state
        .sessions
        .lock()
        .map_err(|e| e.to_string())?
        .remove(&id);
    if let Some(session) = session {
        if let Ok(mut child) = session.child.lock() {
            let _ = child.kill(); // best-effort cleanup: the child may already have exited
        }
    }
    Ok(())
}

pub fn kill_all(state: &PtyState) {
    if let Ok(mut sessions) = state.sessions.lock() {
        for (_, session) in sessions.drain() {
            if let Ok(mut child) = session.child.lock() {
                let _ = child.kill(); // best-effort cleanup: the child may already have exited
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_id_is_unique_even_when_called_in_a_tight_loop() {
        // Old behaviour: nanosecond timestamp only — the clock-before-epoch fallback
        // made EVERY id "pty-0" (a second session replaced the first in the map).
        let mut seen = std::collections::HashSet::new();
        for _ in 0..5000 {
            assert!(seen.insert(random_id()), "duplicate pty id");
        }
    }
}
