//! Phase 13 — local training executor.
//!
//! A "run" is a self-contained directory under `experiments/runs/<run_id>/`
//! (layout documented in TODO.md). The frontend assembles the frozen config
//! (run.json), the architecture snapshot (model.spinoml) and the generated
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
pub(crate) fn validate_run_id(run_id: &str) -> Result<(), String> {
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
/// from turning into an arbitrary-file-read primitive. Shared with the ssh
/// mirror (ssh.rs) so local and remote expose exactly the same surface.
pub(crate) const READABLE: &[&str] = &[
    "run.json",
    "events.jsonl",
    "metrics.json",
    "status",
    "stdout.log",
    "stderr.log",
    "model.py",
    "model.spinoml",
    "train.py",
    "train.sbatch",
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
    has_checkpoint: bool,
}

/// Reconcile a raw status string against process liveness. A run that claims to
/// be "running" OR "queued" but has no live process/job will never make progress
/// — the launch died, a SLURM job was cancelled while pending, or it was
/// orphaned by an old bug — so it's reported as "failed". That keeps the UI from
/// showing a dead run as live/pending forever and makes it terminal (deletable).
/// A genuinely starting/pending run still has a live process (local) or sits in
/// the queue (SLURM, alive via squeue), so it stays queued.
pub(crate) fn reconcile_status(status_raw: &str, alive: bool) -> String {
    let s = status_raw.trim();
    let s = if s.is_empty() { "unknown" } else { s };
    if (s == "running" || s == "queued") && !alive {
        "failed".to_string()
    } else {
        s.to_string()
    }
}

impl RunSummary {
    /// Build a summary from the raw file contents — used by the local executor
    /// (read from disk) and the ssh mirror (read over one ssh round-trip).
    pub(crate) fn from_parts(
        run_id: &str,
        run_json: &str,
        metrics_json: &str,
        status_raw: &str,
        alive: bool,
        has_checkpoint: bool,
    ) -> RunSummary {
        let cfg: Value = serde_json::from_str(run_json).unwrap_or(Value::Null);
        let metrics: Value = serde_json::from_str(metrics_json).unwrap_or(Value::Null);
        let s = |v: &Value, k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").to_string();
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
            status: reconcile_status(status_raw, alive),
            epochs: cfg
                .get("training")
                .and_then(|t| t.get("epochs"))
                .and_then(|x| x.as_u64())
                .unwrap_or(0) as u32,
            best_val_loss: metrics.get("best_val_loss").and_then(|x| x.as_f64()),
            alive,
            has_checkpoint,
        }
    }
}

#[derive(Serialize)]
pub struct GpuStat {
    index: u32,
    name: String,
    util_pct: f64,
    mem_used_mb: f64,
    mem_total_mb: f64,
    temp_c: f64,
}

/// The nvidia-smi query line used by both the local and ssh hardware probes.
pub(crate) const NVIDIA_SMI_QUERY: &str =
    "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits";

/// Parse the CSV rows from NVIDIA_SMI_QUERY into GpuStat. Tolerant of the odd
/// "[Not Supported]" cell (→ 0). Shared by local + ssh so one parser covers both.
pub(crate) fn parse_gpu_stats(out: &str) -> Vec<GpuStat> {
    let num = |s: &str| s.trim().parse::<f64>().unwrap_or(0.0);
    out.lines()
        .filter_map(|line| {
            let cols: Vec<&str> = line.split(',').map(|c| c.trim()).collect();
            if cols.len() < 6 {
                return None;
            }
            Some(GpuStat {
                index: cols[0].parse::<u32>().unwrap_or(0),
                name: cols[1].to_string(),
                util_pct: num(cols[2]),
                mem_used_mb: num(cols[3]),
                mem_total_mb: num(cols[4]),
                temp_c: num(cols[5]),
            })
        })
        .collect()
}

/// Local GPU snapshot (empty if no nvidia-smi). The Run-Detail hardware strip
/// polls this every few seconds while its tab is open. Async + spawn_blocking so
/// the nvidia-smi subprocess never runs on the GTK main thread (a sync command
/// would, and a slow nvidia-smi would then freeze the GUI — the Phase-12/16
/// lesson).
#[tauri::command]
pub async fn gpu_stats() -> Result<Vec<GpuStat>, String> {
    let out = tokio::task::spawn_blocking(|| {
        Command::new("sh")
            .arg("-c")
            .arg(format!("{NVIDIA_SMI_QUERY} 2>/dev/null || true"))
            .output()
    })
    .await
    .map_err(|e| format!("join: {e}"))?
    .map_err(|e| format!("nvidia-smi: {e}"))?;
    Ok(parse_gpu_stats(&String::from_utf8_lossy(&out.stdout)))
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
    let run_json = fs::read_to_string(dir.join("run.json")).unwrap_or_default();
    let metrics_json = fs::read_to_string(dir.join("metrics.json")).unwrap_or_default();
    let status_raw = fs::read_to_string(dir.join("status")).unwrap_or_default();
    let alive = pid_of(dir).map(is_alive).unwrap_or(false);
    let has_checkpoint = dir.join("checkpoints").join("best.pt").exists();
    RunSummary::from_parts(run_id, &run_json, &metrics_json, &status_raw, alive, has_checkpoint)
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

impl RunStatus {
    pub(crate) fn new(status_raw: &str, alive: bool, pid: Option<i32>) -> RunStatus {
        RunStatus {
            status: reconcile_status(status_raw, alive),
            alive,
            pid,
        }
    }

    /// Build from an already-reconciled status (e.g. the SLURM-aware path).
    pub(crate) fn new_with_status(status: String, alive: bool, pid: Option<i32>) -> RunStatus {
        RunStatus { status, alive, pid }
    }
}

/// Refine a SLURM run's status using the scheduler's own accounting. The trainer
/// writes the status file cooperatively, but a job the scheduler kills (TIMEOUT,
/// OUT_OF_MEMORY, node failure, `scancel`) never gets to write a terminal word —
/// it's SIGKILLed. `sacct` then tells us *why* it ended. `squeue_state` is the
/// live `%T` while the job is still in the queue (PENDING/RUNNING/…); when it has
/// left the queue we fall back to `sacct_state`. Either may be empty.
pub(crate) fn reconcile_slurm_status(
    status_raw: &str,
    squeue_state: &str,
    sacct_state: &str,
) -> String {
    let file = status_raw.trim();
    // Still queued/running on the cluster → trust the live scheduler state.
    let q = squeue_state.trim().to_ascii_uppercase();
    if !q.is_empty() {
        return match q.as_str() {
            "PENDING" | "CONFIGURING" => "queued".to_string(),
            // Running on a node: prefer the trainer's own word if it has already
            // moved past "queued" (e.g. it self-reported "running"), else running.
            _ => {
                if file == "done" || file == "failed" || file == "cancelled" {
                    file.to_string()
                } else {
                    "running".to_string()
                }
            }
        };
    }
    // Out of the queue: the trainer's terminal word wins if it wrote one.
    if file == "done" || file == "failed" || file == "cancelled" {
        return file.to_string();
    }
    // Otherwise map sacct's terminal state (strip "CANCELLED by 123" suffix).
    let a = sacct_state.trim().split_whitespace().next().unwrap_or("").to_ascii_uppercase();
    match a.as_str() {
        "COMPLETED" => "done".to_string(),
        "CANCELLED" => "cancelled".to_string(),
        "" => reconcile_status(status_raw, false),
        // FAILED, TIMEOUT, OUT_OF_MEMORY, NODE_FAIL, BOOT_FAIL, DEADLINE, …
        _ => "failed".to_string(),
    }
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
    let status = reconcile_status(&read_status(&dir), alive);
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
    model_spinoml: String,
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
    fs::write(dir.join("model.spinoml"), model_spinoml)
        .map_err(|e| format!("write model.spinoml: {e}"))?;
    fs::write(dir.join("model.py"), model_py).map_err(|e| format!("write model.py: {e}"))?;

    // Drop the shared trainer in as train.py (snapshot — the run stays runnable
    // even if SpinoML updates the template later).
    let template = crate::sidecar_root_pub(&app)
        .join("sidecar-torch")
        .join("training_template.py");
    let trainer = fs::read_to_string(&template).map_err(|e| {
        format!(
            "training template missing at {} ({e}). SpinoML bundle may be incomplete.",
            template.display()
        )
    })?;
    fs::write(dir.join("train.py"), trainer).map_err(|e| format!("write train.py: {e}"))?;

    fs::write(dir.join("status"), "queued\n").map_err(|e| format!("write status: {e}"))?;

    // Detached launch. setsid → own session (survives app close); stdio to
    // files; stdin /dev/null. $! is the python pid (setsid exec's into it).
    let python = std::env::var("SPINOML_PYTHON").unwrap_or_else(|_| "python".into());
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
    eprintln!("[spinoml] training run {run_id} launched ({python})");
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

/// A destination model name for `models/best/<name>.pt`. Same tameness rules as
/// run ids (it becomes a path segment) but a single filename, no slashes.
pub(crate) fn sanitize_model_name(name: &str) -> Result<String, String> {
    let stem = name.trim().trim_end_matches(".pt");
    if stem.is_empty() || stem.len() > 200 {
        return Err("model name must be 1..200 chars".into());
    }
    if !stem
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    {
        return Err("model name may only contain [A-Za-z0-9._-]".into());
    }
    Ok(format!("{stem}.pt"))
}

/// Promote a finished run's best checkpoint to `models/best/<name>.pt` so it can
/// be reused as a pretrained weight. Returns the workspace-relative dest path.
#[tauri::command]
pub fn promote_run_checkpoint(
    state: State<WorkspaceState>,
    run_id: String,
    dest_name: String,
) -> Result<String, String> {
    validate_run_id(&run_id)?;
    let file = sanitize_model_name(&dest_name)?;
    let root = current_root(&state)?;
    let src = run_dir(&root, &run_id).join("checkpoints").join("best.pt");
    if !src.exists() {
        return Err("this run has no checkpoints/best.pt to promote".into());
    }
    let dest_dir = root.join("models").join("best");
    fs::create_dir_all(&dest_dir).map_err(|e| format!("mkdir {}: {e}", dest_dir.display()))?;
    let dest = dest_dir.join(&file);
    fs::copy(&src, &dest).map_err(|e| format!("copy checkpoint: {e}"))?;
    Ok(format!("models/best/{file}"))
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
