// Live remote tests for SpinoML's ssh_* / remote-sidecar code paths.
//
// These tests drive the REAL production code against a real ssh host and
// (when enabled) a real SLURM cluster:
//   - the launch goes through `ssh::start_training_run_with_template` — the
//     whole former body of `ssh_start_training_run`, which needs no AppHandle
//     and is therefore callable from a test crate. This is the code that
//     issues the mkdir submission claim, writes run.json/model.spinoml/
//     model.py/train.py, generates train.sbatch via `build_sbatch`, calls
//     `sbatch`, and parses the `MLF_JOBID` marker. The tests do NOT
//     re-implement any of that.
//   - status / list / stop / delete / read go through the public
//     `ssh_training_run_status`, `ssh_list_training_runs`,
//     `ssh_stop_training_run`, `ssh_delete_training_run`,
//     `ssh_read_training_run_file`, plus `ssh_test_connection`,
//     `ssh_remote_training_capabilities`, `ssh_gpu_stats`.
// Raw `ssh` round-trips (ssh_run_raw / ssh_put_file) are used ONLY for:
// mkdir of the test root, writing the synthetic dataset, cleanup,
// the leftover check, and `squeue`/`sacct` cross-checks.
//
// Gating:
//   export SPINOML_REMOTE_TESTS=1
//   export SPINOML_REMOTE_ALIAS=<ssh-config alias>        # REQUIRED
//   export SPINOML_REMOTE_ROOT_BASE=~/spinoml-live-test   # default
//   export SPINOML_REMOTE_PYTHON=python3                  # default
//   export SPINOML_REMOTE_SLURM_ACCOUNT=<account>         # optional
//   export SPINOML_REMOTE_SLURM_PARTITION=cpu             # default
//   export SPINOML_REMOTE_GPU_PARTITION=<gpu-partition>   # enables live_gpu
//   export SPINOML_REMOTE_CUDA_PYTHON=<abs path>          # required for live_gpu
//   cargo test --manifest-path src-tauri/Cargo.toml live_ -- --ignored --test-threads=1 --nocapture
//
// Safety:
//   - Every test asserts the test root contains `spinoml-live-test`.
//   - Cleanup `rm -rf` ONLY `<base>/<unix-timestamp>`; the path is asserted
//     to contain the marker before any destructive shell command runs.
//   - SLURM jobs: `--time <= 00:10:00`, `--cpus-per-task <= 2`, `--mem <= 4G`;
//     asserted on the run_json right before the launch.
//   - GPU jobs run ONLY when SPINOML_REMOTE_GPU_PARTITION AND
//     SPINOML_REMOTE_CUDA_PYTHON are both set; `--gres=gpu:1`.
//   - TestGuard::drop deletes the whole timestamped root, even on panic.
//   - At most 3 SLURM jobs per suite invocation (live_slurm_run × 2 + live_gpu).
#![cfg(test)]

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::json;

// ─── env + safety gate ──────────────────────────────────────────────────────

const MARKER: &str = "spinoml-live-test";

struct TestGuard {
    alias: String,
    base: String,
    stamp: String,
    name: String,
    root: String,
}

impl Drop for TestGuard {
    fn drop(&mut self) {
        // SAFETY: the path MUST contain the marker.
        let stamp_dir = format!("{}/{}", self.base.trim_end_matches('/'), self.stamp);
        let dir = format!("{stamp_dir}/{}", self.name);
        if !dir.contains(MARKER) {
            eprintln!("[remote-live] REFUSED to rm -rf {} (missing marker)", dir);
            return;
        }
        // Cancel THIS run's SLURM jobs first (job names are `spinoml-<run_id>` and every
        // run id of this suite ends with the shared stamp): a failed test must not
        // leave a job running into a deleted directory.
        let stamp = &self.stamp;
        if stamp.chars().all(|c| c.is_ascii_digit()) {
            let _ = ssh_run_raw(
                &self.alias,
                &format!(
                    "squeue -u \"$USER\" -h -o '%i %j' | awk '$2 ~ /^spinoml-live-.*-{stamp}$/ {{print $1}}' | xargs -r scancel"
                ),
            );
        }
        // Keep the evidence of a FAILED test (events.jsonl, slurm-*.err, status) for
        // inspection; the wrapper reports the leftover directory as a failure.
        if std::thread::panicking() {
            eprintln!("[remote-live] test failed - keeping {dir} for inspection (remove it by hand afterwards)");
            return;
        }
        // best-effort; do not panic in Drop (we may be on the unwind path).
        let _ = ssh_run_raw(&self.alias, &format!("rm -rf -- {}", shell_q_path(&dir)));
        // remove the shared stamp directory too once the last test dir is gone (rmdir fails,
        // harmlessly, while another test's kept evidence is still inside)
        let _ = ssh_run_raw(&self.alias, &format!("rmdir {} 2>/dev/null; true", shell_q_path(&stamp_dir)));
    }
}

impl TestGuard {
    /// `test_name` is embedded in every SKIPPED line as
    /// `SKIPPED: <test_name>: <reason>` so `scripts/remote-live.ts` can count a
    /// skipped test as skipped even though cargo's harness reports it `ok`.
    fn new(test_name: &str) -> Option<TestGuard> {
        let enabled = std::env::var("SPINOML_REMOTE_TESTS").ok().as_deref() == Some("1");
        let alias = std::env::var("SPINOML_REMOTE_ALIAS").unwrap_or_default();
        if !enabled || alias.is_empty() {
            eprintln!(
                "SKIPPED: {test_name}: set SPINOML_REMOTE_TESTS=1 and SPINOML_REMOTE_ALIAS=<alias> to enable live tests"
            );
            return None;
        }
        let base = std::env::var("SPINOML_REMOTE_ROOT_BASE")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| format!("~/{}", MARKER));
        if !base.contains(MARKER) {
            eprintln!("SKIPPED: {test_name}: SPINOML_REMOTE_ROOT_BASE must contain the literal {}", MARKER);
            return None;
        }
        if !(base.starts_with('/') || base.starts_with('~')) {
            eprintln!("SKIPPED: {test_name}: SPINOML_REMOTE_ROOT_BASE must be absolute (/... or ~/...)");
            return None;
        }
        // `scripts/remote-live.ts` exports SPINOML_REMOTE_RUN_STAMP so every
        // test in one suite invocation writes under the SAME
        // `<base>/<stamp>` dir — that is the dir the wrapper's leftover check
        // probes. Only digits are accepted (it becomes a path segment).
        let stamp = std::env::var("SPINOML_REMOTE_RUN_STAMP")
            .ok()
            .filter(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()))
            .unwrap_or_else(|| {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0)
                    .to_string()
            });
        Some(TestGuard { alias, base, stamp, name: test_name.to_string(), root: String::new() })
    }

    fn init_root(&mut self) -> String {
        // one subdirectory per test: a passing test must not delete the evidence of a failed one
        let dir = format!("{}/{}/{}", self.base.trim_end_matches('/'), self.stamp, self.name);
        let cmd = format!(
            "mkdir -p {}/experiments/runs {}/datasets",
            shell_q_path(&dir),
            shell_q_path(&dir)
        );
        ssh_run_raw(&self.alias, &cmd).expect("mkdir -p on remote");
        self.root = dir.clone();
        dir
    }
}

fn python_default() -> String {
    // The HPC login node has no bare `python`; python3 is the default there.
    std::env::var("SPINOML_REMOTE_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

fn slurm_partition() -> String {
    std::env::var("SPINOML_REMOTE_SLURM_PARTITION").unwrap_or_else(|_| "cpu".to_string())
}

fn slurm_account() -> Option<String> {
    std::env::var("SPINOML_REMOTE_SLURM_ACCOUNT").ok().filter(|s| !s.is_empty())
}

// ─── minimal ssh spawner + file writer (mkdir/cleanup/dataset) ──────────────
// We cannot reach the private `ssh_exec` from this test crate; these mirror
// `ssh_exec_blocking` exactly for the few non-launch uses listed in the header.

fn ssh_run_raw(alias: &str, remote_cmd: &str) -> Result<String, String> {
    let out = std::process::Command::new("ssh")
        .arg("-o").arg("BatchMode=yes")
        .arg("-o").arg("ConnectTimeout=10")
        .arg("--")
        .arg(alias)
        .arg(remote_cmd)
        .output()
        .map_err(|e| format!("spawn ssh: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let code = out.status.code().map(|c| c.to_string()).unwrap_or_else(|| "?".into());
        return Err(format!("ssh exit {code}: {}", stderr.trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

/// `ssh <alias> cat > <file>` — write the synthetic dataset via ssh stdin.
fn ssh_put_file(alias: &str, path: &str, content: &[u8]) -> Result<(), String> {
    let mut child = std::process::Command::new("ssh")
        .arg("-o").arg("BatchMode=yes")
        .arg("-o").arg("ConnectTimeout=10")
        .arg("--")
        .arg(alias)
        .arg(format!("cat > {}", shell_q_path(path)))
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("spawn ssh: {e}"))?;
    {
        let mut stdin = child.stdin.take().ok_or_else(|| "no stdin".to_string())?;
        use std::io::Write;
        stdin.write_all(content).map_err(|e| format!("write: {e}"))?;
    }
    let out = child.wait_with_output().map_err(|e| format!("wait: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        return Err(format!("ssh put exit: {}", stderr.trim()));
    }
    Ok(())
}

fn shell_q_path(s: &str) -> String {
    if let Some(rest) = s.strip_prefix("~/") {
        format!("\"$HOME\"'/{rest}'")
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

// ─── private-field accessors (Serialize is derived, fields are not pub) ─────

fn status_str(st: &crate::training::RunStatus) -> String {
    serde_json::to_value(st)
        .ok()
        .and_then(|v| v.get("status").and_then(|s| s.as_str().map(String::from)))
        .unwrap_or_default()
}

fn caps_has_slurm(c: &crate::ssh::RemoteTrainingCapabilities) -> bool {
    serde_json::to_value(c)
        .ok()
        .and_then(|v| v.get("has_slurm").and_then(|b| b.as_bool()))
        .unwrap_or(false)
}

fn caps_partitions(c: &crate::ssh::RemoteTrainingCapabilities) -> Vec<String> {
    serde_json::to_value(c)
        .ok()
        .and_then(|v| v.get("partitions").and_then(|p| p.as_array().cloned()))
        .map(|a| a.into_iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default()
}

fn summary_run_id(s: &crate::training::RunSummary) -> String {
    serde_json::to_value(s)
        .ok()
        .and_then(|v| v.get("run_id").and_then(|s| s.as_str().map(String::from)))
        .unwrap_or_default()
}

// ─── tiny deterministic xorshift64* PRNG for the synthetic CSV ─────────────

fn xorshift(mut s: u64) -> u64 {
    s ^= s >> 12;
    s ^= s << 25;
    s ^= s >> 27;
    s.wrapping_mul(0x2545F4914F6CDD1D)
}

fn make_csv(n: usize, seed: u64) -> String {
    let mut s = seed.max(1);
    let mut next = || {
        s = xorshift(s);
        let u1 = (s as f64 / u64::MAX as f64).clamp(1e-9, 1.0);
        let u2 = ((s >> 17) as f64 / u64::MAX as f64).clamp(1e-9, 1.0);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let mut out = String::from("f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,y\n");
    for i in 0..n {
        let label = (i % 2) as i32;
        let mu = if label == 1 { 0.6 } else { -0.6 };
        let mut row = String::with_capacity(128);
        for j in 0..10 {
            let m = if j < 5 { mu } else { -mu };
            row.push_str(&format!("{:.6}", m + 0.9 * next()));
            if j < 9 { row.push(','); }
        }
        out.push_str(&row);
        out.push(',');
        out.push_str(&label.to_string());
        out.push('\n');
    }
    out
}

// ─── fixtures: mlp reference (examples/reference-experiments/mlp/) ──────────

const MLP_MODEL_SPINOML: &str = include_str!("../../examples/reference-experiments/mlp/model.spinoml");

// Exact bytes emitted by `generateFromSnapshot(parseFile(mlp/model.spinoml))`
// at the time this file was written.
const MLP_MODEL_PY: &str = "import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=10, out_features=16, bias=True)
        self.re_lu = nn.ReLU(inplace=False)
        self.linear_2 = nn.Linear(in_features=16, out_features=2, bias=True)

    def forward(self, x):
        linear = self.linear(x)
        re_lu = self.re_lu(linear)
        linear_2 = self.linear_2(re_lu)
        return linear_2


if __name__ == \"__main__\":
    model = Model()
    x = torch.zeros((1, 10))
    _ = model(x)
    print(\"Parameters:\", sum(p.numel() for p in model.parameters()))
";

/// The bundled trainer the real launch ships as `train.py`.
fn training_template_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("CARGO_MANIFEST_DIR has a parent (the repo root)")
        .join("sidecar-torch")
        .join("training_template.py")
}

// ─── run.json (mirrors src/training/types.ts: RunConfig + SlurmConfig) ──────

/// `backend` is built the way the frontend builds it:
/// `{ kind: "local" }` or `{ kind: "slurm", slurm: { partition, time, mem,
/// cpus_per_task, gres, account?, qos, modules, pre_run_script } }`.
fn build_run_json(
    run_id: &str,
    label: &str,
    backend: serde_json::Value,
    dataset_path: &str,
    dataset_relpath: &str,
) -> serde_json::Value {
    json!({
        "run_id": run_id,
        "run_label": label,
        "created_at": "2026-01-01T00:00:00Z",
        "status": "queued",
        "model_path": "mlp.spinoml",
        "backend": backend,
        "dataset": {
            "path": dataset_path,
            "relpath": dataset_relpath,
            "kind": "tabular",
            "feature_columns": null,
            "target_column": "y",
        },
        "training": {
            "epochs": 2,
            "batch_size": 8,
            "val_split": 0.25,
            "seed": 42,
            "log_every_n_steps": 1,
            "optimizer": { "kind": "Adam", "lr": 0.01, "weight_decay": 0.0 },
            "loss": { "kind": "CrossEntropyLoss", "label_smoothing": 0.0 },
            "scheduler": { "kind": "none" },
            "metrics": [],
            "callbacks": [],
        },
    })
}

fn slurm_backend(partition: &str, gres: &str) -> serde_json::Value {
    let mut slurm = json!({
        "partition": partition,
        "time": "00:10:00",
        "mem": "4G",
        "cpus_per_task": 2,
        "gres": gres,
        "qos": "",
        "modules": [],
        "pre_run_script": "",
    });
    if let Some(acc) = slurm_account() {
        slurm["account"] = json!(acc);
    }
    json!({ "kind": "slurm", "slurm": slurm })
}

// ─── safety-cap assertions (before every SLURM/GPU launch) ──────────────────

fn hms_seconds(t: &str) -> Option<u64> {
    let parts: Vec<&str> = t.trim().split(':').collect();
    let (h, m, s): (u64, u64, u64) = match parts.as_slice() {
        [h, m, s] => (h.parse().ok()?, m.parse().ok()?, s.parse().ok()?),
        [m, s] => (0, m.parse().ok()?, s.parse().ok()?),
        _ => return None,
    };
    Some(h * 3600 + m * 60 + s)
}

fn mem_mib(m: &str) -> Option<u64> {
    let m = m.trim();
    let split = m.find(|c: char| !c.is_ascii_digit()).unwrap_or(m.len());
    let (num, unit) = m.split_at(split);
    let n: u64 = num.parse().ok()?;
    match unit.trim().to_ascii_uppercase().as_str() {
        "" | "M" | "MB" => Some(n),
        "G" | "GB" => Some(n * 1024),
        _ => None,
    }
}

/// Refuse to launch a SLURM/GPU job that exceeds the suite's safety caps.
fn assert_slurm_caps(run_json: &serde_json::Value, expect_gpu: bool) {
    let slurm = run_json["backend"]["slurm"]
        .as_object()
        .expect("slurm backend object");
    let time = slurm["time"].as_str().unwrap_or("");
    let secs = hms_seconds(time).unwrap_or_else(|| panic!("unparseable --time {time:?}"));
    assert!(secs <= 600, "SLURM --time {time} exceeds the 00:10:00 cap");
    let cpus = slurm["cpus_per_task"].as_u64().unwrap_or(0);
    assert!(cpus >= 1 && cpus <= 2, "SLURM cpus_per_task {cpus} must be in 1..=2");
    let mem = slurm["mem"].as_str().unwrap_or("");
    let mib = mem_mib(mem).unwrap_or_else(|| panic!("unparseable --mem {mem:?}"));
    assert!(mib <= 4096, "SLURM --mem {mem} exceeds the 4G cap");
    let gres = slurm["gres"].as_str().unwrap_or("");
    if expect_gpu {
        assert_eq!(gres, "gpu:1", "GPU jobs must request exactly --gres=gpu:1");
    } else {
        assert!(gres.is_empty(), "CPU jobs must not request gres, got {gres:?}");
    }
}

// ─── polling helpers ────────────────────────────────────────────────────────

async fn wait_terminal(
    alias: String,
    root: String,
    run_id: &str,
    timeout: Duration,
    expected: &[&str],
) -> Result<crate::training::RunStatus, String> {
    let start = std::time::Instant::now();
    loop {
        let st = crate::ssh::ssh_training_run_status(alias.clone(), root.clone(), run_id.to_string()).await?;
        let s = status_str(&st);
        if expected.iter().any(|e| *e == s) {
            return Ok(st);
        }
        // A terminal state that is NOT the expected one can never change again:
        // fail now with the evidence instead of waiting out the timeout.
        if matches!(s.as_str(), "done" | "failed" | "cancelled") {
            return Err(format!("run ended `{s}` but one of {expected:?} was expected"));
        }
        if std::time::Instant::now().duration_since(start) > timeout {
            return Err(format!("timed out after {timeout:?} waiting for terminal status; last={s:?}"));
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

/// Cross-check a finished SLURM job through `sacct` (the scheduler's own
/// accounting). Returns the raw state; empty means accounting hasn't caught up.
fn sacct_state(alias: &str, jid: &str) -> String {
    let q = format!("sacct -j '{jid}' -n -X -o State%30 2>/dev/null | head -1");
    ssh_run_raw(alias, &q).unwrap_or_default().trim().to_string()
}

// ─── tests ──────────────────────────────────────────────────────────────────

#[tokio::test]
#[ignore = "set SPINOML_REMOTE_TESTS=1 + SPINOML_REMOTE_ALIAS=<alias> to enable; runs against a real ssh host"]
async fn live_connection() {
    let Some(mut g) = TestGuard::new("live_connection") else { return };
    let root = g.init_root();

    let test = crate::ssh::ssh_test_connection(g.alias.clone())
        .await
        .expect("ssh_test_connection ok");
    assert!(test.ok, "ssh_test_connection.ok");
    assert!(!test.uname.is_empty(), "uname must not be empty");
    assert!(test.home.starts_with('/'), "home must be absolute: {}", test.home);

    let caps = crate::ssh::ssh_remote_training_capabilities(g.alias.clone(), root.clone())
        .await
        .expect("ssh_remote_training_capabilities");
    assert!(caps_has_slurm(&caps), "sbatch not found on {}", g.alias);
    let want = slurm_partition();
    assert!(
        caps_partitions(&caps).contains(&want),
        "partition {want} not in {:?}",
        caps_partitions(&caps)
    );
    if let Some(acc) = slurm_account() {
        eprintln!("[remote-live] using SLURM account {acc} (informational)");
    }
}

/// Full remote-sidecar bootstrap: probe → install → deploy → tunnel → spawn,
/// then `GET /health` through the tunnel WITH the token (expect 200) and
/// WITHOUT it (expect 401/403), then stop and confirm no process is left.
///
/// DISABLED (#[ignore]) because it cannot be driven from a Rust test crate.
///
/// WHY: `remote_sidecar::run_bootstrap` / `ensure_remote_sidecar` take a
/// concrete `tauri::AppHandle<Wry>`. `tauri::test::mock_app()` (tauri
/// v2.11.2, `test` feature) yields `App<MockRuntime>` → `AppHandle<MockRuntime>`,
/// a distinct type Rust will not coerce to `AppHandle<Wry>` (E0308). Getting a
/// real `AppHandle<Wry>` needs `tauri::Builder::<Wry>::default()
/// .build(generate_context!())`, whose `tao::EventLoop::new()` requires
/// X11/Wayland — unavailable in CI. So this test stays documentation.
///
/// MANUAL PROCEDURE (run on a machine with a display, against `leipzig-hpc`):
///   1. `export SPINOML_REMOTE_TESTS=1 SPINOML_REMOTE_ALIAS=leipzig-hpc` and
///      `cargo test --manifest-path src-tauri/Cargo.toml live_connection --
///      --ignored --nocapture` (proves `ssh_test_connection` +
///      `ssh_remote_training_capabilities` before any bootstrap).
///   2. `npm run tauri dev` (or the installed `spinoml`), open the
///      Connections sidebar, add/select `leipzig-hpc`, root
///      `~/spinoml-live-test/<ts>` for a FRESH `<ts>` (never `~/spinoml`).
///   3. Click "Verbindung testen" → expect success.
///   4. Click the remote-sidecar badge to bootstrap; watch the badge move
///      `offline` → `preparing` → `running`.
///   5. CHECK: `ssh leipzig-hpc 'ls ~/spinoml-live-test/<ts>/.spinoml/venv/bin/python'`
///      exists; `ssh leipzig-hpc 'pgrep -u $USER -f "spinoml-live-test/<ts>"'`
///      lists exactly the remote sidecar python; and from the laptop
///      `curl -s -o /dev/null -w '%{http_code}' 127.0.0.1:7424/health`
///      with `X-SpinoML-Token: <from DevTools sidecar_token("torch-remote")>`
///      → 200, and WITHOUT the header → 401/403.
///   6. Stop the sidecar (badge), then `ssh leipzig-hpc 'pgrep -u $USER -f
///      "spinoml-live-test/<ts>"'` → no output. `rm -rf ~/spinoml-live-test/<ts>`.
#[tokio::test]
#[ignore = "runs the full bootstrap pipeline against a real HPC login node — cannot drive AppHandle<Wry> from cargo test; see doc comment for the manual procedure"]
async fn live_bootstrap() {
    let Some(mut g) = TestGuard::new("live_bootstrap") else { return };
    let _root = g.init_root();
    eprintln!(
        "SKIPPED: live_bootstrap: cannot drive run_bootstrap from cargo test \
         (AppHandle<Wry> vs mock AppHandle<MockRuntime>); follow the MANUAL \
         PROCEDURE in this test's doc comment instead."
    );
}

#[tokio::test]
#[ignore = "set SPINOML_REMOTE_TESTS=1 + SPINOML_REMOTE_ALIAS=<alias> to enable"]
async fn live_direct_run() {
    let Some(mut g) = TestGuard::new("live_direct_run") else { return };
    let root = g.init_root();

    let ds_rel = "datasets/live.csv";
    let ds_abs = format!("{root}/{ds_rel}");
    ssh_put_file(&g.alias, &ds_abs, make_csv(40, 42).as_bytes()).expect("write dataset");

    let run_id = format!("live-direct-{}", g.stamp);
    let run_json = build_run_json(&run_id, "live direct", json!({ "kind": "local" }), &ds_abs, ds_rel);

    // REAL launch path (no AppHandle needed).
    crate::ssh::start_training_run_with_template(
        training_template_path(),
        g.alias.clone(),
        root.clone(),
        run_id.clone(),
        python_default(),
        run_json.to_string(),
        MLP_MODEL_SPINOML.to_string(),
        MLP_MODEL_PY.to_string(),
    )
    .await
    .expect("start_training_run_with_template (direct)");

    let st = wait_terminal(g.alias.clone(), root.clone(), &run_id, Duration::from_secs(8 * 60), &["done"])
        .await
        .expect("wait_terminal done");
    assert_eq!(status_str(&st), "done", "final status");

    let metrics = crate::ssh::ssh_read_training_run_file(
        g.alias.clone(), root.clone(), run_id.clone(), "metrics.json".to_string(),
    )
    .await
    .expect("read metrics.json");
    let m: serde_json::Value = serde_json::from_str(&metrics).expect("metrics.json parse");
    assert_eq!(m["status"].as_str(), Some("done"), "metrics.status");
    let best = m["best_val_loss"].as_f64().unwrap_or(f64::NAN);
    assert!(best.is_finite() && best < 10.0, "best_val_loss implausible: {best}");

    let events = crate::ssh::ssh_read_training_run_file(
        g.alias.clone(), root.clone(), run_id.clone(), "events.jsonl".to_string(),
    )
    .await
    .expect("read events.jsonl");
    assert!(events.contains("\"run.integrity\""), "missing run.integrity event: {events}");
    assert!(events.contains("\"ok\": true"), "run.integrity must be ok=true");
    assert!(events.contains("\"run.done\""), "missing run.done event");

    let manifest = crate::ssh::ssh_read_training_run_file(
        g.alias.clone(), root.clone(), run_id.clone(), "manifest.json".to_string(),
    )
    .await
    .expect("read manifest.json");
    let mf: serde_json::Value = serde_json::from_str(&manifest).expect("manifest.json parse");
    assert_eq!(mf["schema"].as_str(), Some("spinoml.run-manifest/1"), "manifest.schema");

    // Stopping a finished run is a no-op (terminal-state shielding).
    crate::ssh::ssh_stop_training_run(g.alias.clone(), root.clone(), run_id.clone())
        .await
        .expect("stop terminal run");
    let st = crate::ssh::ssh_training_run_status(g.alias.clone(), root.clone(), run_id.clone())
        .await
        .expect("status after stop");
    assert_eq!(status_str(&st), "done", "stopped a done run must stay done");

    // Delete removes the dir; a fresh status then reads unknown.
    crate::ssh::ssh_delete_training_run(g.alias.clone(), root.clone(), run_id.clone())
        .await
        .expect("delete");
    let st = crate::ssh::ssh_training_run_status(g.alias.clone(), root.clone(), run_id.clone())
        .await
        .expect("status after delete");
    assert_eq!(status_str(&st), "unknown", "deleted run dir: status=unknown");
}

#[tokio::test]
#[ignore = "set SPINOML_REMOTE_TESTS=1 + SPINOML_REMOTE_ALIAS=<alias> to enable"]
async fn live_slurm_run() {
    let Some(mut g) = TestGuard::new("live_slurm_run") else { return };
    let root = g.init_root();

    let ds_rel = "datasets/live.csv";
    let ds_abs = format!("{root}/{ds_rel}");
    ssh_put_file(&g.alias, &ds_abs, make_csv(40, 11).as_bytes()).expect("write dataset");

    // Run A: completes via the partition.
    let run_id_a = format!("live-slurm-a-{}", g.stamp);
    let run_json_a = build_run_json(&run_id_a, "live slurm A", slurm_backend(&slurm_partition(), ""), &ds_abs, ds_rel);
    assert_slurm_caps(&run_json_a, false);
    crate::ssh::start_training_run_with_template(
        training_template_path(),
        g.alias.clone(),
        root.clone(),
        run_id_a.clone(),
        python_default(),
        run_json_a.to_string(),
        MLP_MODEL_SPINOML.to_string(),
        MLP_MODEL_PY.to_string(),
    )
    .await
    .expect("start_training_run_with_template (slurm A)");

    let st_a = wait_terminal(g.alias.clone(), root.clone(), &run_id_a, Duration::from_secs(8 * 60), &["done", "failed"])
        .await
        .expect("wait run A");
    assert_eq!(status_str(&st_a), "done", "run A must end done (not failed)");

    // The frozen pid is `slurm:<jid>` (read through the real reader).
    let pid = crate::ssh::ssh_read_training_run_file(g.alias.clone(), root.clone(), run_id_a.clone(), "pid".to_string())
        .await
        .expect("read pid");
    let pid = pid.trim();
    assert!(pid.starts_with("slurm:"), "run A pid must start with slurm:, got {pid:?}");
    let jid: String = pid.trim_start_matches("slurm:").chars().filter(|c| c.is_ascii_digit()).collect();

    // sacct cross-check: the scheduler itself should report COMPLETED. Empty
    // (accounting lag) is tolerated with a printed note, a contradictory
    // terminal state is a failure.
    let mut sa = String::new();
    for _ in 0..30 {
        sa = sacct_state(&g.alias, &jid);
        if !sa.is_empty() { break; }
        std::thread::sleep(Duration::from_secs(2));
    }
    let upper = sa.to_ascii_uppercase();
    if upper.is_empty() {
        eprintln!("[remote-live] sacct for {jid}: empty (accounting lag) — skipped cross-check");
    } else {
        assert!(upper.starts_with("COMPLETED"), "sacct for {jid} = {sa:?} (expected COMPLETED)");
    }

    // Run B: started, then cancelled via the real ssh_stop_training_run.
    let run_id_b = format!("live-slurm-b-{}", g.stamp);
    let run_json_b = build_run_json(&run_id_b, "live slurm B", slurm_backend(&slurm_partition(), ""), &ds_abs, ds_rel);
    assert_slurm_caps(&run_json_b, false);
    crate::ssh::start_training_run_with_template(
        training_template_path(),
        g.alias.clone(),
        root.clone(),
        run_id_b.clone(),
        python_default(),
        run_json_b.to_string(),
        MLP_MODEL_SPINOML.to_string(),
        MLP_MODEL_PY.to_string(),
    )
    .await
    .expect("start_training_run_with_template (slurm B)");
    tokio::time::sleep(Duration::from_secs(3)).await;
    crate::ssh::ssh_stop_training_run(g.alias.clone(), root.clone(), run_id_b.clone())
        .await
        .expect("stop run B");
    let st_b = wait_terminal(g.alias.clone(), root.clone(), &run_id_b, Duration::from_secs(2 * 60), &["cancelled"])
        .await
        .expect("wait run B → cancelled");
    assert_eq!(status_str(&st_b), "cancelled", "scancel must yield `cancelled`");
}

#[tokio::test]
#[ignore = "set SPINOML_REMOTE_TESTS=1 + SPINOML_REMOTE_ALIAS=<alias> to enable"]
async fn live_recovery() {
    let Some(mut g) = TestGuard::new("live_recovery") else { return };
    let root = g.init_root();

    let ds_rel = "datasets/live.csv";
    let ds_abs = format!("{root}/{ds_rel}");
    ssh_put_file(&g.alias, &ds_abs, make_csv(40, 21).as_bytes()).expect("write dataset");

    let run_id = format!("live-recovery-{}", g.stamp);
    let run_json = build_run_json(&run_id, "live recovery", json!({ "kind": "local" }), &ds_abs, ds_rel);
    crate::ssh::start_training_run_with_template(
        training_template_path(),
        g.alias.clone(),
        root.clone(),
        run_id.clone(),
        python_default(),
        run_json.to_string(),
        MLP_MODEL_SPINOML.to_string(),
        MLP_MODEL_PY.to_string(),
    )
    .await
    .expect("start_training_run_with_template (recovery)");

    // Simulate an app restart: the in-memory RemoteWorkspaceState is per-process
    // (not persisted), so a fresh ssh_list_training_runs / status call is the
    // exact re-query path FAILURE_RECOVERY.md §8 describes.
    let _ = wait_terminal(g.alias.clone(), root.clone(), &run_id, Duration::from_secs(8 * 60), &["done"])
        .await
        .expect("wait_terminal done");

    let runs = crate::ssh::ssh_list_training_runs(g.alias.clone(), root.clone())
        .await
        .expect("list runs after restart");
    let ids: Vec<String> = runs.iter().map(|r| summary_run_id(r)).collect();
    assert!(
        runs.iter().any(|r| summary_run_id(r) == run_id),
        "fresh ssh_list_training_runs after restart must find {run_id}; got {ids:?}",
    );

    let st = crate::ssh::ssh_training_run_status(g.alias.clone(), root.clone(), run_id.clone())
        .await
        .expect("status after restart");
    assert_eq!(status_str(&st), "done", "post-restart status reconciles to done");
}

#[tokio::test]
#[ignore = "set SPINOML_REMOTE_TESTS=1 + SPINOML_REMOTE_ALIAS=<alias> + SPINOML_REMOTE_GPU_PARTITION + SPINOML_REMOTE_CUDA_PYTHON to enable"]
async fn live_gpu() {
    let gpu_partition = match std::env::var("SPINOML_REMOTE_GPU_PARTITION").ok().filter(|s| !s.is_empty()) {
        Some(p) => p,
        None => {
            eprintln!(
                "SKIPPED: live_gpu: set SPINOML_REMOTE_GPU_PARTITION=<gpu-partition> + \
                 SPINOML_REMOTE_CUDA_PYTHON=<abs-path> to enable live_gpu"
            );
            return;
        }
    };
    let cuda_python = match std::env::var("SPINOML_REMOTE_CUDA_PYTHON").ok().filter(|s| !s.is_empty()) {
        Some(p) => p,
        None => {
            eprintln!("SKIPPED: live_gpu: set SPINOML_REMOTE_CUDA_PYTHON=<abs-path-on-cluster> to enable live_gpu");
            return;
        }
    };
    if !(cuda_python.starts_with('/') || cuda_python.starts_with('~')) {
        eprintln!("SKIPPED: live_gpu: SPINOML_REMOTE_CUDA_PYTHON must be absolute");
        return;
    }

    let Some(mut g) = TestGuard::new("live_gpu") else { return };
    let root = g.init_root();

    let ds_rel = "datasets/live.csv";
    let ds_abs = format!("{root}/{ds_rel}");
    ssh_put_file(&g.alias, &ds_abs, make_csv(40, 31).as_bytes()).expect("write dataset");

    let run_id = format!("live-gpu-{}", g.stamp);
    let run_json = build_run_json(&run_id, "live gpu", slurm_backend(&gpu_partition, "gpu:1"), &ds_abs, ds_rel);
    assert_slurm_caps(&run_json, true);
    crate::ssh::start_training_run_with_template(
        training_template_path(),
        g.alias.clone(),
        root.clone(),
        run_id.clone(),
        cuda_python,
        run_json.to_string(),
        MLP_MODEL_SPINOML.to_string(),
        MLP_MODEL_PY.to_string(),
    )
    .await
    .expect("start_training_run_with_template (gpu)");

    let st = wait_terminal(g.alias.clone(), root.clone(), &run_id, Duration::from_secs(8 * 60), &["done", "failed"])
        .await
        .expect("wait_terminal done|failed");
    assert_eq!(status_str(&st), "done", "GPU run must end done");

    let manifest = crate::ssh::ssh_read_training_run_file(g.alias.clone(), root.clone(), run_id.clone(), "manifest.json".to_string())
        .await
        .expect("read manifest.json");
    let mf: serde_json::Value = serde_json::from_str(&manifest).expect("manifest parse");
    assert_eq!(mf["status"].as_str(), Some("done"), "manifest.status");
    // manifest.json records `device` at the top level (metrics.json nests it
    // under `env`) — see training_template.py _manifest_init.
    assert_eq!(mf["device"].as_str(), Some("cuda"), "GPU run manifest.device must be cuda, got {}", mf["device"]);

    let _gpu_stats = crate::ssh::ssh_gpu_stats(g.alias.clone(), root.clone(), Some(run_id.clone()))
        .await
        .expect("ssh_gpu_stats");
}
