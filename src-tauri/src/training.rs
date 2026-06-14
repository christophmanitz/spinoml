//! Phase 13 — local training executor.
//!
//! A "run" is a self-contained directory under `experiments/runs/<run_id>/`
//! (layout documented in TODO.md). The frontend assembles the frozen config
//! (run.json), the architecture snapshot (model.mlforge) and the generated
//! module (model.py); this module drops the shared trainer in as `train.py`
//! and launches it **detached** so it outlives the app.
//!
//! Detach mechanism (Linux): `setsid` puts the python process in its own
//! session/process-group with no controlling terminal, stdio is redirected
//! into the run dir, and stdin is `/dev/null`. When the app exits the python
//! process is reparented to init and keeps running. The run dir's
//! `events.jsonl` + `status` files are the single source of truth — on
//! re-open the UI re-reads them, no live handle required.
//!
//! Remote (ssh-direct / SLURM) lands in Phase 16/17; this module is local-only
//! and goes through the same `crate::WorkspaceState` as the other FS commands.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde::Serialize;
use serde_json::Value;
use tauri::{AppHandle, State};

use crate::{current_root, WorkspaceState};

/// run_id is used as a path segment and embedded in a shell command, so it must
/// be a tame slug. Our generated ids look like `2026-06-15T12-30-00_iris_a8f3`.
fn validate_run_id(run_id: &str) -> Result<(), String> {
    if run_id.is_empty() || run_id.len() > 200 {
        return Err("run id must be 1..200 chars".into());
    }
    if !run_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    {
        return Err("run id may only contain [A-Za-z0-9._-]".into());
    }
    Ok(())
}

fn runs_dir(root: &Path) -> PathBuf {
    root.join("experiments").join("runs")
}

fn run_dir(root: &Path, run_id: &str) -> PathBuf {
    runs_dir(root).join(run_id)
}

/// Files the UI is allowed to read back from a run dir. Keeps the read command
/// from turning into an arbitrary-file-read primitive.
const READABLE: &[&str] = &[
    "run.json",
    "events.jsonl",
    "metrics.json",
    "status",
    "stdout.log",
    "stderr.log",
    "model.py",
    "model.mlforge",
    "train.py",
    "pid",
];

#[derive(Serialize)]
pub struct RunSummary {
    run_id: String,
    run_label: String,
    model_path: String,
    dataset_path: String,
    created_at: String,
    status: String,
    epochs: u32,
    best_val_loss: Option<f64>,
    alive: bool,
}

fn pid_of(dir: &Path) -> Option<i32> {
    fs::read_to_string(dir.join("pid"))
        .ok()
        .and_then(|s| s.trim().parse::<i32>().ok())
}

/// `kill -0 <pid>` — true if the process exists and we may signal it.
fn is_alive(pid: i32) -> bool {
    Command::new("kill")
        .arg("-0")
        .arg(pid.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn read_status(dir: &Path) -> String {
    fs::read_to_string(dir.join("status"))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

fn summarize(dir: &Path, run_id: &str) -> RunSummary {
    let cfg: Value = fs::read_to_string(dir.join("run.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(Value::Null);
    let metrics: Value = fs::read_to_string(dir.join("metrics.json"))
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or(Value::Null);

    let s = |v: &Value, k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();

    let mut status = read_status(dir);
    let alive = pid_of(dir).map(is_alive).unwrap_or(false);
    // Reconcile a stale "running" status: if the process is gone but the file
    // never got updated (e.g. SIGKILL), treat it as failed so the UI isn't lying.
    if status == "running" && !alive {
        status = "failed".into();
    }

    RunSummary {
        run_id: run_id.to_string(),
        run_label: s(&cfg, "run_label"),
        model_path: s(&cfg, "model_path"),
        dataset_path: cfg
            .get("dataset")
            .and_then(|d| d.get("path"))
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string(),
        created_at: s(&cfg, "created_at"),
        status,
        epochs: cfg
            .get("training")
            .and_then(|t| t.get("epochs"))
            .and_then(|x| x.as_u64())
            .unwrap_or(0) as u32,
        best_val_loss: metrics.get("best_val_loss").and_then(|x| x.as_f64()),
        alive,
    }
}

#[tauri::command]
pub fn list_training_runs(state: State<WorkspaceState>) -> Result<Vec<RunSummary>, String> {
    let root = current_root(&state)?;
    let dir = runs_dir(&root);
    if !dir.exists() {
        return Ok(vec![]);
    }
    let mut out: Vec<RunSummary> = Vec::new();
    for entry in fs::read_dir(&dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?.flatten() {
        let p = entry.path();
        if !p.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        out.push(summarize(&p, &name));
    }
    // run_id is timestamp-prefixed, so lexical-desc == chronological-newest-first.
    out.sort_by(|a, b| b.run_id.cmp(&a.run_id));
    Ok(out)
}

#[derive(Serialize)]
pub struct RunStatus {
    status: String,
    alive: bool,
    pid: Option<i32>,
}

#[tauri::command]
pub fn training_run_status(
    state: State<WorkspaceState>,
    run_id: String,
) -> Result<RunStatus, String> {
    validate_run_id(&run_id)?;
    let root = current_root(&state)?;
    let dir = run_dir(&root, &run_id);
    let pid = pid_of(&dir);
    let alive = pid.map(is_alive).unwrap_or(false);
    let mut status = read_status(&dir);
    if status == "running" && !alive {
        status = "failed".into();
    }
    Ok(RunStatus { status, alive, pid })
}

#[tauri::command]
pub fn read_training_run_file(
    state: State<WorkspaceState>,
    run_id: String,
    name: String,
) -> Result<String, String> {
    validate_run_id(&run_id)?;
    if !READABLE.contains(&name.as_str()) {
        return Err(format!("file {name:?} is not readable from a run dir"));
    }
    let root = current_root(&state)?;
    let p = run_dir(&root, &run_id).join(&name);
    if !p.exists() {
        return Ok(String::new());
    }
    fs::read_to_string(&p).map_err(|e| format!("read {}: {e}", p.display()))
}

/// Write the run dir and launch the detached trainer.
#[tauri::command]
pub fn start_training_run(
    app: AppHandle,
    state: State<WorkspaceState>,
    run_id: String,
    run_json: String,
    model_mlforge: String,
    model_py: String,
) -> Result<(), String> {
    validate_run_id(&run_id)?;
    let root = current_root(&state)?;
    let dir = run_dir(&root, &run_id);
    if dir.exists() {
        return Err(format!("run {run_id} already exists"));
    }
    fs::create_dir_all(dir.join("checkpoints"))
        .map_err(|e| format!("mkdir {}: {e}", dir.display()))?;

    // Frozen snapshots.
    fs::write(dir.join("run.json"), run_json).map_err(|e| format!("write run.json: {e}"))?;
    fs::write(dir.join("model.mlforge"), model_mlforge)
        .map_err(|e| format!("write model.mlforge: {e}"))?;
    fs::write(dir.join("model.py"), model_py).map_err(|e| format!("write model.py: {e}"))?;

    // Drop the shared trainer in as train.py (snapshot — the run stays runnable
    // even if MLForge updates the template later).
    let template = crate::sidecar_root_pub(&app)
        .join("sidecar-torch")
        .join("training_template.py");
    let trainer = fs::read_to_string(&template).map_err(|e| {
        format!(
            "training template missing at {} ({e}). MLForge bundle may be incomplete.",
            template.display()
        )
    })?;
    fs::write(dir.join("train.py"), trainer).map_err(|e| format!("write train.py: {e}"))?;

    fs::write(dir.join("status"), "queued\n").map_err(|e| format!("write status: {e}"))?;

    // Detached launch. setsid → own session (survives app close); stdio to
    // files; stdin /dev/null. $! is the python pid (setsid exec's into it).
    let python = std::env::var("MLFORGE_PYTHON").unwrap_or_else(|_| "python".into());
    let script = format!(
        "setsid {python} -u train.py > stdout.log 2> stderr.log < /dev/null & echo $! > pid",
    );
    let status = Command::new("sh")
        .arg("-c")
        .arg(&script)
        .current_dir(&dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| format!("spawn trainer: {e}"))?;
    if !status.success() {
        return Err("failed to launch training process".into());
    }
    eprintln!("[mlforge] training run {run_id} launched ({python})");
    Ok(())
}

#[tauri::command]
pub fn stop_training_run(state: State<WorkspaceState>, run_id: String) -> Result<(), String> {
    validate_run_id(&run_id)?;
    let root = current_root(&state)?;
    let dir = run_dir(&root, &run_id);
    if !dir.exists() {
        return Err(format!("run {run_id} not found"));
    }
    // Cooperative: the trainer checks `status` at each epoch boundary.
    let _ = fs::write(dir.join("status"), "cancelled\n");
    // Forceful: SIGTERM the whole process group (negative pid). setsid made the
    // python pid the group leader, so this also takes child dataloader workers.
    if let Some(pid) = pid_of(&dir) {
        let _ = Command::new("kill")
            .arg("-TERM")
            .arg(format!("-{pid}"))
            .status();
        let _ = Command::new("kill").arg("-TERM").arg(pid.to_string()).status();
    }
    Ok(())
}

#[tauri::command]
pub fn delete_training_run(state: State<WorkspaceState>, run_id: String) -> Result<(), String> {
    validate_run_id(&run_id)?;
    let root = current_root(&state)?;
    let dir = run_dir(&root, &run_id);
    if !dir.exists() {
        return Ok(());
    }
    if pid_of(&dir).map(is_alive).unwrap_or(false) {
        return Err("run is still alive — stop it before deleting".into());
    }
    fs::remove_dir_all(&dir).map_err(|e| format!("rmdir {}: {e}", dir.display()))
}
