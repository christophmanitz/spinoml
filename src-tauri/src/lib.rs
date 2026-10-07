use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tauri::{Manager, State, WindowEvent};
use tauri_plugin_dialog::DialogExt;

mod scope_file;
mod ssh;
mod pty;
mod remote_sidecar;
mod sidecar_auth;
mod training;

#[cfg(test)]
mod live_tests;

pub(crate) const PROJECT_FILE: &str = "spinoml.project.json";
pub(crate) const SUBDIRS: &[&str] = &["models", "datasets", "notes", "experiments"];

#[derive(Default)]
pub(crate) struct WorkspaceState {
    root: Mutex<Option<PathBuf>>,
}

#[derive(Default)]
struct Sidecars {
    torch: Mutex<Option<Child>>,
    llm: Mutex<Option<Child>>,
}

fn sidecar_root(app: &tauri::App) -> PathBuf {
    // In a release bundle the sidecars live under the resource directory
    // (mapped from ../sidecar-* by tauri.conf.json bundle.resources). During
    // dev there is no resource dir, so fall back to the project root which
    // is one level above CARGO_MANIFEST_DIR.
    if !cfg!(debug_assertions) {
        if let Ok(dir) = app.path().resource_dir() {
            return dir;
        }
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or(manifest) // a missing parent falls back to the manifest dir; never a false path claim
}

// AppHandle variant — needed by remote_sidecar.rs which runs after setup.
pub(crate) fn sidecar_root_pub(app: &tauri::AppHandle) -> PathBuf {
    if !cfg!(debug_assertions) {
        if let Ok(dir) = app.path().resource_dir() {
            return dir;
        }
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or(manifest) // a missing parent falls back to the manifest dir; never a false path claim
}

/// Ask the kernel to SIGTERM this child when its parent (spinoml) dies, no
/// matter how the parent dies — graceful quit, crash, or `kill -9` from a dev
/// restart. Without this, sidecars (and their ports) orphan and the next launch
/// hits "Address already in use". Must run from a long-lived thread (we spawn
/// from setup() on the main thread, which lives for the whole process).
#[cfg(target_os = "linux")]
pub(crate) fn set_pdeathsig(cmd: &mut Command) {
    use std::os::unix::process::CommandExt;
    unsafe {
        cmd.pre_exec(|| {
            libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM as libc::c_ulong, 0, 0, 0);
            Ok(())
        });
    }
}
#[cfg(not(target_os = "linux"))]
pub(crate) fn set_pdeathsig(_cmd: &mut Command) {}

fn spawn_managed(
    label: &str,
    prog: &str,
    arg: PathBuf,
    cwd: &Path,
    tokens: &sidecar_auth::SidecarTokens,
) -> Option<Child> {
    if !arg.exists() {
        eprintln!("[spinoml] {label}: sidecar script not found at {}", arg.display());
        return None;
    }
    // Read the local token once. If reading fails here the whole app is broken
    // (we wouldn't have been able to `.manage()` the state at startup either),
    // so bail out without spawning — never spawn a sidecar without a token.
    let token = match tokens.endpoint_token("torch-local") {
        Ok(Some(t)) => t,
        Ok(None) => {
            eprintln!(
                "[spinoml] {label}: no local sidecar token configured — refusing to spawn"
            );
            return None;
        }
        Err(e) => {
            eprintln!("[spinoml] {label}: token lookup failed: {e}");
            return None;
        }
    };
    let mut cmd = Command::new(prog);
    cmd.arg(&arg)
        .current_dir(cwd)
        // Per-launch authentication token (Phase 77, see docs/engineering/SIDECAR_AUTH.md).
        // We set it on the spawned child ONLY — `Command::env`, never `std::env::set_var`,
        // which would leak the token to every other child of the GUI process. Both
        // sidecars require the token (`SPINOML_REQUIRE_TOKEN=1`); without it they
        // refuse to start. The eprintln! lines below deliberately print the label
        // + program path, never env.
        .env("SPINOML_SIDECAR_TOKEN", &token)
        .env("SPINOML_REQUIRE_TOKEN", "1")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    set_pdeathsig(&mut cmd);
    // Drop our copy of the token before spawn — the child has its own copy in
    // its environment block. We don't store it anywhere; `SidecarTokens.local`
    // is the only remaining reference.
    drop(token);
    match cmd.spawn() {
        Ok(child) => {
            eprintln!("[spinoml] {label}: spawned pid={} ({prog} {})", child.id(), arg.display());
            Some(child)
        }
        Err(e) => {
            eprintln!(
                "[spinoml] {label}: failed to spawn ({prog} {}): {e} \
                 — make sure your shell PATH has the conda env activated",
                arg.display()
            );
            None
        }
    }
}

fn shutdown_sidecars(sc: &Sidecars) {
    if let Ok(mut t) = sc.torch.lock() {
        if let Some(mut c) = t.take() {
            eprintln!("[spinoml] killing torch sidecar pid={}", c.id());
            let _ = c.kill(); // best-effort cleanup: the child may already have exited
            let _ = c.wait(); // best-effort cleanup: the child may already have exited
        }
    }
    if let Ok(mut t) = sc.llm.lock() {
        if let Some(mut c) = t.take() {
            eprintln!("[spinoml] killing llm sidecar pid={}", c.id());
            let _ = c.kill(); // best-effort cleanup: the child may already have exited
            let _ = c.wait(); // best-effort cleanup: the child may already have exited
        }
    }
}

#[derive(Serialize)]
struct SidecarStatus {
    torch: bool,
    llm: bool,
}

#[tauri::command]
fn sidecar_managed_status(sc: State<Sidecars>) -> SidecarStatus {
    SidecarStatus {
        torch: sc.torch.lock().map(|g| g.is_some()).unwrap_or(false), // a poisoned lock reads as not-managed; the holder never panics on these paths
        llm: sc.llm.lock().map(|g| g.is_some()).unwrap_or(false), // a poisoned lock reads as not-managed; the holder never panics on these paths
    }
}

#[derive(Serialize)]
struct FsEntry {
    name: String,
    relpath: String,
    is_dir: bool,
}

fn resolve(root: &Path, relpath: &str) -> Result<PathBuf, String> {
    let rel = PathBuf::from(relpath);
    for comp in rel.components() {
        match comp {
            Component::Normal(_) => {}
            Component::CurDir => {}
            _ => return Err(format!("rejected path segment in {relpath}")),
        }
    }
    // Validate that the JOINED path, after following symlinks, stays inside
    // the workspace root or an allowed symlink target (R016). But return the
    // LEXICAL `root.join(rel)` UNCHANGED: callers must act on the link itself,
    // not its target. Returning the canonical path broke delete/rename of an
    // allowed symlink (e.g. `datasets -> /work2/...`): deleting would remove
    // the user's REAL dataset directory instead of the link.
    let joined = root.join(rel);
    scope_file::check_resolved(root, &joined)?;
    Ok(joined)
}

pub(crate) fn current_root(state: &State<WorkspaceState>) -> Result<PathBuf, String> {
    state
        .root
        .lock()
        .map_err(|e| e.to_string())?
        .clone()
        .ok_or_else(|| "no workspace folder is open".to_string())
}

#[tauri::command]
async fn pick_workspace_dir(app: tauri::AppHandle) -> Result<Option<String>, String> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog()
        .file()
        .pick_folder(move |result| {
            let _ = tx.send(result); // best-effort UI/channel send; a dropped receiver has nothing left to mislead
        });
    let result = rx.await.map_err(|e| e.to_string())?;
    let path = match result {
        Some(p) => p,
        None => return Ok(None),
    };
    let buf = path
        .into_path()
        .map_err(|e| format!("path conversion: {e}"))?;
    // Persist the new workspace root to the sidecar scope file BEFORE we
    // accept it. A failure to write the scope file means the sidecars will
    // run in unconfigured-open mode against this workspace; we refuse rather
    // than silently leaving the user with a workspace the sidecar can't see.
    scope_file::write_roots(&[buf.clone()])?;
    let state: State<WorkspaceState> = app.state();
    *state.root.lock().map_err(|e| e.to_string())? = Some(buf.clone());
    Ok(Some(buf.to_string_lossy().to_string()))
}

#[tauri::command]
fn current_workspace_dir(state: State<WorkspaceState>) -> Option<String> {
    state
        .root
        .lock()
        .ok() // an optional/fallible read yields None, the documented unknown rather than a false value
        .and_then(|g| g.clone())
        .map(|p| p.to_string_lossy().to_string())
}

#[tauri::command]
fn close_workspace_dir(state: State<WorkspaceState>) -> Result<(), String> {
    *state.root.lock().map_err(|e| e.to_string())? = None;
    // Drop the local root from the scope file. A failure here is non-fatal
    // (we are closing anyway; the next set_workspace_dir will overwrite the
    // file), but we log so an operator notices a stale scope.
    if let Err(e) = scope_file::clear_roots() {
        eprintln!("[spinoml] failed to clear scope file on workspace close: {e}");
    }
    Ok(())
}

/// Open a previously-known local workspace folder by path — no dialog. Used by
/// the "recent workspaces" quick-select on the Welcome screen.
#[tauri::command]
fn set_workspace_dir(state: State<WorkspaceState>, path: String) -> Result<String, String> {
    let buf = PathBuf::from(&path);
    if !buf.is_dir() {
        return Err(format!("folder no longer exists: {path}"));
    }
    // Same rationale as pick_workspace_dir: refuse if the sidecar can't see
    // the workspace.
    scope_file::write_roots(&[buf.clone()])?;
    *state.root.lock().map_err(|e| e.to_string())? = Some(buf.clone());
    Ok(buf.to_string_lossy().to_string())
}

#[tauri::command]
fn list_workspace(state: State<WorkspaceState>) -> Result<Vec<FsEntry>, String> {
    let root = current_root(&state)?;
    list_workspace_impl(&root)
}

/// Pure listing helper (no Tauri `State`) so tests can exercise it directly.
/// Walks the LEXICAL workspace tree; canonicalisation is used ONLY for the
/// containment decision. A symlinked directory whose real target is an allowed
/// symlink target (the common `datasets -> /work2/...` layout) is followed and
/// listed under its lexical name. `rel` is always computed lexically from
/// `root` — deriving it from the canonical target would fail `strip_prefix`
/// for an allowed out-of-tree link and abort the whole listing. A visited set
/// of canonical directory paths stops symlink cycles (`a -> .`, `a/b -> ..`).
fn list_workspace_impl(root: &Path) -> Result<Vec<FsEntry>, String> {
    let canonical_root = fs::canonicalize(root)
        .map_err(|e| format!("workspace root is not accessible: {e}"))?;
    let mut out: Vec<FsEntry> = Vec::new();
    let mut visited: std::collections::HashSet<PathBuf> = std::collections::HashSet::new();
    visited.insert(canonical_root.clone());
    walk(root, root, &canonical_root, &mut visited, &mut out)?;
    out.sort_by(|a, b| {
        a.is_dir
            .cmp(&b.is_dir)
            .reverse()
            .then_with(|| a.relpath.to_lowercase().cmp(&b.relpath.to_lowercase()))
    });
    Ok(out)
}

fn walk(
    root: &Path,
    dir: &Path,
    canonical_root: &Path,
    visited: &mut std::collections::HashSet<PathBuf>,
    out: &mut Vec<FsEntry>,
) -> Result<(), String> {
    let read = fs::read_dir(dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
    for entry in read.flatten() {
        let p = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        // R016 — refuse to walk INTO anything whose canonical target is
        // outside the workspace (or not an allowed symlink target). Without
        // this, an LLM-run script could plant `datasets → /etc` and the
        // explorer would expose /etc's contents.
        let canonical_p = match fs::canonicalize(&p) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if !canonical_p.starts_with(canonical_root) && !is_in_symlink_targets(&canonical_p) {
            continue;
        }
        // rel is LEXICAL (root-relative): never derived from the canonical
        // target.
        let rel = p
            .strip_prefix(root)
            .map_err(|e| e.to_string())?
            .to_string_lossy()
            .replace('\\', "/");
        // Decide "is a directory" from the resolved target (a symlinked dir
        // has symlink_metadata().is_dir() == false).
        let is_dir = fs::metadata(&canonical_p)
            .map(|m| m.is_dir())
            .unwrap_or(false); // the absence reads as not-true (the conservative direction)
        if is_dir {
            out.push(FsEntry { name, relpath: rel.clone(), is_dir: true });
            // Cycle protection: recurse only into a canonical directory we
            // have not already visited, so `a -> .` is listed once and never
            // re-walked.
            if visited.insert(canonical_p.clone()) {
                walk(root, &p, canonical_root, visited, out)?;
            }
        } else {
            // Everything visible. The frontend decides what's clickable.
            out.push(FsEntry { name, relpath: rel, is_dir: false });
        }
    }
    Ok(())
}

/// True if `p` is under any configured symlink target (env + scope file).
fn is_in_symlink_targets(p: &Path) -> bool {
    scope_file::allowed_roots_for_check()
        .iter()
        .any(|t| p.starts_with(t))
}

#[tauri::command]
fn read_workspace_file(state: State<WorkspaceState>, relpath: String) -> Result<String, String> {
    let root = current_root(&state)?;
    let full = resolve(&root, &relpath)?;
    fs::read_to_string(&full).map_err(|e| format!("read {}: {e}", full.display()))
}

#[tauri::command]
fn write_workspace_file(
    state: State<WorkspaceState>,
    relpath: String,
    content: String,
) -> Result<(), String> {
    let root = current_root(&state)?;
    let full = resolve(&root, &relpath)?;
    if let Some(parent) = full.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    fs::write(&full, content).map_err(|e| format!("write {}: {e}", full.display()))
}

/// Delete a workspace path. Acts on the LEXICAL path so a symlink is removed
/// as a link and never followed: `symlink_metadata` classifies the entry, and
/// a symlink is always removed with `remove_file`, never `remove_dir_all`
/// (which would delete the link TARGET's contents). `resolve()` validates
/// containment first, so a link whose target escapes the workspace (and is not
/// an allowed symlink target) is refused before anything is touched.
fn delete_path_impl(root: &Path, relpath: &str) -> Result<(), String> {
    let full = resolve(root, relpath)?;
    let meta = fs::symlink_metadata(&full)
        .map_err(|e| format!("stat {}: {e}", full.display()))?;
    if meta.file_type().is_symlink() {
        fs::remove_file(&full).map_err(|e| format!("rm {}: {e}", full.display()))
    } else if meta.is_dir() {
        fs::remove_dir_all(&full).map_err(|e| format!("rmdir {}: {e}", full.display()))
    } else {
        fs::remove_file(&full).map_err(|e| format!("rm {}: {e}", full.display()))
    }
}

#[tauri::command]
fn delete_workspace_path(state: State<WorkspaceState>, relpath: String) -> Result<(), String> {
    let root = current_root(&state)?;
    delete_path_impl(&root, &relpath)
}

#[tauri::command]
fn mkdir_workspace(state: State<WorkspaceState>, relpath: String) -> Result<(), String> {
    let root = current_root(&state)?;
    let full = resolve(&root, &relpath)?;
    fs::create_dir_all(&full).map_err(|e| format!("mkdir {}: {e}", full.display()))
}

#[derive(Serialize, Deserialize, Clone, Default)]
struct ProjectMeta {
    name: String,
    description: String,
    goal: String,
    #[serde(default)]
    active_model: Option<String>,
    #[serde(default)]
    active_dataset: Option<String>,
    #[serde(default)]
    created_at: String,
    #[serde(default)]
    updated_at: String,
    #[serde(default = "default_schema_version")]
    schema_version: u32,
}

fn default_schema_version() -> u32 { 1 }

#[derive(Serialize)]
struct ProjectLoad {
    root: String,
    meta: Option<ProjectMeta>,
    has_legacy_files: bool,
    legacy_spinoml_count: usize,
}

pub(crate) fn now_iso() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0); // clock-before-epoch fallback; pid and a counter keep generated ids unique
    // Crude ISO8601 — good enough for human reading; we don't need timezone math.
    format!("{}Z", secs_to_iso(secs))
}

fn secs_to_iso(secs: u64) -> String {
    let days = secs / 86400;
    let secs_of_day = secs % 86400;
    let h = secs_of_day / 3600;
    let m = (secs_of_day % 3600) / 60;
    let s = secs_of_day % 60;
    let (y, mo, d) = days_to_ymd(days as i64);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}")
}

fn days_to_ymd(mut days: i64) -> (i32, u32, u32) {
    let mut year: i32 = 1970;
    loop {
        let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
        let d = if leap { 366 } else { 365 };
        if days < d { break; }
        days -= d;
        year += 1;
    }
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let months = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut month = 1u32;
    for &dd in &months {
        if days < dd { break; }
        days -= dd;
        month += 1;
    }
    (year, month, (days + 1) as u32)
}

fn project_path(root: &Path) -> PathBuf { root.join(PROJECT_FILE) }

fn ensure_subdirs(root: &Path) -> Result<(), String> {
    for sub in SUBDIRS {
        let p = root.join(sub);
        if !p.exists() {
            fs::create_dir_all(&p).map_err(|e| format!("mkdir {}: {e}", p.display()))?;
        }
    }
    Ok(())
}

fn write_project_meta(root: &Path, meta: &ProjectMeta) -> Result<(), String> {
    let s = serde_json::to_string_pretty(meta).map_err(|e| e.to_string())?;
    fs::write(project_path(root), s).map_err(|e| format!("write project.json: {e}"))
}

fn read_project_meta(root: &Path) -> Result<Option<ProjectMeta>, String> {
    let p = project_path(root);
    if !p.exists() { return Ok(None); }
    let s = fs::read_to_string(&p).map_err(|e| format!("read {}: {e}", p.display()))?;
    let meta: ProjectMeta = serde_json::from_str(&s).map_err(|e| format!("parse project.json: {e}"))?;
    Ok(Some(meta))
}

fn scan_legacy(root: &Path) -> (bool, usize) {
    let mut count = 0usize;
    if let Ok(rd) = fs::read_dir(root) {
        for entry in rd.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.to_lowercase().ends_with(".spinoml") {
                count += 1;
            }
        }
    }
    (count > 0, count)
}

#[tauri::command]
fn load_project(state: State<WorkspaceState>) -> Result<ProjectLoad, String> {
    let root = current_root(&state)?;
    let meta = read_project_meta(&root)?;
    let (has_legacy, count) = scan_legacy(&root);
    Ok(ProjectLoad {
        root: root.to_string_lossy().to_string(),
        meta,
        has_legacy_files: has_legacy,
        legacy_spinoml_count: count,
    })
}

#[tauri::command]
fn init_project(
    state: State<WorkspaceState>,
    name: String,
    description: String,
    goal: String,
) -> Result<ProjectMeta, String> {
    let root = current_root(&state)?;
    if read_project_meta(&root)?.is_some() {
        return Err("project already initialized in this folder".into());
    }
    let now = now_iso();
    let meta = ProjectMeta {
        name,
        description,
        goal,
        active_model: None,
        active_dataset: None,
        created_at: now.clone(),
        updated_at: now,
        schema_version: 1,
    };
    ensure_subdirs(&root)?;
    write_project_meta(&root, &meta)?;
    Ok(meta)
}

#[derive(Deserialize, Default)]
struct ProjectMetaPatch {
    name: Option<String>,
    description: Option<String>,
    goal: Option<String>,
    active_model: Option<Option<String>>,
    active_dataset: Option<Option<String>>,
}

#[tauri::command]
fn update_project_meta(
    state: State<WorkspaceState>,
    patch: ProjectMetaPatch,
) -> Result<ProjectMeta, String> {
    let root = current_root(&state)?;
    let mut meta = read_project_meta(&root)?.ok_or_else(|| "no project in this folder".to_string())?;
    if let Some(n) = patch.name { meta.name = n; }
    if let Some(d) = patch.description { meta.description = d; }
    if let Some(g) = patch.goal { meta.goal = g; }
    if let Some(am) = patch.active_model { meta.active_model = am; }
    if let Some(ad) = patch.active_dataset { meta.active_dataset = ad; }
    meta.updated_at = now_iso();
    write_project_meta(&root, &meta)?;
    Ok(meta)
}

#[tauri::command]
fn migrate_legacy_project(
    state: State<WorkspaceState>,
    name: String,
    description: String,
    goal: String,
) -> Result<ProjectMeta, String> {
    let root = current_root(&state)?;
    if read_project_meta(&root)?.is_some() {
        return Err("project already initialized; nothing to migrate".into());
    }
    ensure_subdirs(&root)?;
    // Move root-level .spinoml and matching .py twins into models/. Use
    // symlink_metadata + canonicalize so a planted symlink at the workspace
    // root (e.g. `evil.py → /etc/passwd`) cannot trick us into renaming the
    // target into the workspace (R016).
    let models = root.join("models");
    let canonical_root = fs::canonicalize(&root).unwrap_or_else(|_| root.clone()); // root was already validated; the lexical fallback is defensive only
    if let Ok(rd) = fs::read_dir(&root) {
        for entry in rd.flatten() {
            let p = entry.path();
            // Refuse symlinks entirely — a one-shot migration must not be
            // a vector for moving arbitrary files into the workspace.
            let meta = match fs::symlink_metadata(&p) {
                Ok(m) => m,
                Err(_) => continue,
            };
            if meta.file_type().is_symlink() {
                continue;
            }
            // And refuse anything whose canonical path is outside the root
            // (covers hardlinks pointing outside — fs::symlink_metadata
            // returns is_symlink()=false for those).
            let canonical_p = match fs::canonicalize(&p) {
                Ok(c) => c,
                Err(_) => continue,
            };
            if !canonical_p.starts_with(&canonical_root) {
                continue;
            }
            let name_os = entry.file_name();
            let fname = name_os.to_string_lossy().to_string();
            let lower = fname.to_lowercase();
            if lower.ends_with(".spinoml") || lower.ends_with(".py") {
                let dest = models.join(&fname);
                fs::rename(&canonical_p, &dest).map_err(|e| format!("move {fname}: {e}"))?;
            }
        }
    }
    let now = now_iso();
    let meta = ProjectMeta {
        name, description, goal,
        active_model: None, active_dataset: None,
        created_at: now.clone(), updated_at: now, schema_version: 1,
    };
    write_project_meta(&root, &meta)?;
    Ok(meta)
}

// ─── Notes ────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct NoteEntry {
    name: String,
    relpath: String,
    size_bytes: u64,
    modified_at: String,
}

#[tauri::command]
fn list_notes(state: State<WorkspaceState>) -> Result<Vec<NoteEntry>, String> {
    let root = current_root(&state)?;
    let dir = root.join("notes");
    if !dir.exists() { fs::create_dir_all(&dir).map_err(|e| format!("mkdir notes/: {e}"))?; }
    let mut out: Vec<NoteEntry> = Vec::new();
    for entry in fs::read_dir(&dir).map_err(|e| e.to_string())?.flatten() {
        let p = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') || !p.is_file() { continue; }
        let lower = name.to_lowercase();
        if !lower.ends_with(".md") && !lower.ends_with(".txt") { continue; }
        let meta = entry.metadata().map_err(|e| e.to_string())?;
        let modified = meta.modified()
            .ok() // an optional/fallible read yields None, the documented unknown rather than a false value
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok()) // an optional/fallible read yields None, the documented unknown rather than a false value
            .map(|d| secs_to_iso(d.as_secs()))
            .unwrap_or_default(); // documented empty default; the surrounding status carries the real state
        out.push(NoteEntry {
            name: name.clone(),
            relpath: format!("notes/{name}"),
            size_bytes: meta.len(),
            modified_at: modified,
        });
    }
    out.sort_by(|a, b| b.modified_at.cmp(&a.modified_at));
    Ok(out)
}

#[tauri::command]
fn read_note(state: State<WorkspaceState>, name: String) -> Result<String, String> {
    let root = current_root(&state)?;
    if name.contains('/') || name.contains('\\') {
        return Err("note name must be a plain filename".into());
    }
    let p = resolve(&root, &format!("notes/{name}"))?;
    fs::read_to_string(&p).map_err(|e| format!("read {}: {e}", p.display()))
}

#[tauri::command]
fn write_note(state: State<WorkspaceState>, name: String, content: String) -> Result<(), String> {
    let root = current_root(&state)?;
    if name.contains('/') || name.contains('\\') {
        return Err("note name must be a plain filename".into());
    }
    let p = resolve(&root, &format!("notes/{name}"))?;
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    fs::write(&p, content).map_err(|e| format!("write {}: {e}", p.display()))
}

#[tauri::command]
fn append_note(state: State<WorkspaceState>, name: String, content: String) -> Result<(), String> {
    use std::io::Write;
    let root = current_root(&state)?;
    if name.contains('/') || name.contains('\\') {
        return Err("note name must be a plain filename".into());
    }
    let p = resolve(&root, &format!("notes/{name}"))?;
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&p)
        .map_err(|e| format!("open {}: {e}", p.display()))?;
    f.write_all(content.as_bytes()).map_err(|e| e.to_string())?;
    Ok(())
}

// ─── Experiments ─────────────────────────────────────────────────────────

#[tauri::command]
fn append_experiment(
    state: State<WorkspaceState>,
    filename: String,
    line: String,
) -> Result<(), String> {
    use std::io::Write;
    let root = current_root(&state)?;
    if filename.contains('/') || filename.contains('\\') {
        return Err("experiment filename must be a plain name".into());
    }
    let p = resolve(&root, &format!("experiments/{filename}"))?;
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&p)
        .map_err(|e| format!("open {}: {e}", p.display()))?;
    let mut bytes = line.into_bytes();
    if !bytes.ends_with(b"\n") { bytes.push(b'\n'); }
    f.write_all(&bytes).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
fn read_experiment(state: State<WorkspaceState>, filename: String) -> Result<String, String> {
    let root = current_root(&state)?;
    if filename.contains('/') || filename.contains('\\') {
        return Err("experiment filename must be a plain name".into());
    }
    let p = resolve(&root, &format!("experiments/{filename}"))?;
    if !p.exists() { return Ok(String::new()); }
    fs::read_to_string(&p).map_err(|e| format!("read {}: {e}", p.display()))
}

#[derive(Serialize)]
struct DatasetEntry {
    name: String,
    relpath: String,
    abspath: String,
    is_dir: bool,
    size_bytes: u64,
}

#[tauri::command]
fn list_datasets(state: State<WorkspaceState>) -> Result<Vec<DatasetEntry>, String> {
    let root = current_root(&state)?;
    // R016 — list under the canonical datasets dir so a symlinked datasets/
    // (e.g. an attacker who got the LLM to symlink it elsewhere) cannot
    // redirect the listing outside the workspace.
    let dsdir = resolve(&root, "datasets")?;
    if !dsdir.exists() {
        fs::create_dir_all(&dsdir).map_err(|e| format!("mkdir datasets/: {e}"))?;
    }
    let canonical_dsdir = fs::canonicalize(&dsdir).unwrap_or_else(|_| dsdir.clone()); // datasets dir was already validated by resolve(); the fallback is defensive only
    let mut out: Vec<DatasetEntry> = Vec::new();
    let read = fs::read_dir(&dsdir).map_err(|e| format!("read_dir {}: {e}", dsdir.display()))?;
    for entry in read.flatten() {
        let p = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        // Skip entries that escape the canonical datasets dir via symlink.
        let entry_canonical = match fs::canonicalize(&p) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if !entry_canonical.starts_with(&canonical_dsdir) {
            continue;
        }
        let meta = entry.metadata().map_err(|e| e.to_string())?;
        let size = if meta.is_file() { meta.len() } else { dir_size(&entry_canonical).unwrap_or(0) }; // display-only size; a failed stat shows 0 bytes, not a false listing
        let rel = format!("datasets/{}", name);
        out.push(DatasetEntry {
            name,
            relpath: rel,
            abspath: entry_canonical.to_string_lossy().to_string(),
            is_dir: meta.is_dir(),
            size_bytes: size,
        });
    }
    out.sort_by(|a, b| {
        a.is_dir
            .cmp(&b.is_dir)
            .reverse()
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });
    Ok(out)
}

fn dir_size(p: &Path) -> Option<u64> {
    let mut total: u64 = 0;
    let read = fs::read_dir(p).ok()?; // a failed read yields None up the Option chain, the documented skip state
    for entry in read.flatten() {
        let meta = entry.metadata().ok()?; // a failed read yields None up the Option chain, the documented skip state
        if meta.is_file() {
            total += meta.len();
        } else if meta.is_dir() {
            total += dir_size(&entry.path()).unwrap_or(0); // display-only size; a failed stat shows 0 bytes, not a false listing
        }
    }
    Some(total)
}

#[tauri::command]
fn dataset_abspath(state: State<WorkspaceState>, relpath: String) -> Result<String, String> {
    let root = current_root(&state)?;
    let full = resolve(&root, &relpath)?;
    Ok(full.to_string_lossy().to_string())
}

/// Rename a workspace path. `fs::rename` on a symlink moves the LINK itself,
/// not its target; `resolve()` validates containment first and returns the
/// lexical paths, so an allowed symlink (e.g. `datasets -> /work2/...`) is
/// renamed without touching the real target directory.
fn rename_path_impl(root: &Path, from_rel: &str, to_rel: &str) -> Result<(), String> {
    let from = resolve(root, from_rel)?;
    let to = resolve(root, to_rel)?;
    if let Some(parent) = to.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    fs::rename(&from, &to).map_err(|e| format!("rename: {e}"))
}

#[tauri::command]
fn rename_workspace_path(
    state: State<WorkspaceState>,
    from_rel: String,
    to_rel: String,
) -> Result<(), String> {
    let root = current_root(&state)?;
    rename_path_impl(&root, &from_rel, &to_rel)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // webkit2gtk often produces blurry CSS-transformed canvas content unless
    // the DMA-BUF renderer is disabled. Setting these before WebContext init.
    #[cfg(target_os = "linux")]
    {
        if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
            std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
        }
        if std::env::var_os("WEBKIT_DISABLE_COMPOSITING_MODE").is_none() {
            std::env::set_var("WEBKIT_DISABLE_COMPOSITING_MODE", "0");
        }
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(WorkspaceState::default())
        .manage(Sidecars::default())
        .manage(ssh::RemoteWorkspaceState::default())
        .manage(pty::PtyState::default())
        .manage(remote_sidecar::RemoteSidecarState::default())
        // Per-launch sidecar authentication token (Phase 77). Generated once at
        // startup from the OS RNG. If the RNG fails the app refuses to start —
        // silently downgrading to "no token" would defeat the whole point.
        .manage({
            match sidecar_auth::SidecarTokens::new() {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("[spinoml] failed to initialize sidecar authentication: {e}");
                    eprintln!("[spinoml] refusing to start — see docs/engineering/SIDECAR_AUTH.md");
                    std::process::exit(2);
                }
            }
        })
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            let root = sidecar_root(app);
            let sc: State<Sidecars> = app.state();
            let tokens: State<sidecar_auth::SidecarTokens> = app.state();
            if let Ok(mut t) = sc.torch.lock() {
                *t = spawn_managed(
                    "sidecar-torch",
                    "python",
                    root.join("sidecar-torch").join("main.py"),
                    &root,
                    &tokens,
                );
            }
            if let Ok(mut l) = sc.llm.lock() {
                *l = spawn_managed(
                    "sidecar-llm",
                    "node",
                    root.join("sidecar-llm").join("main.mjs"),
                    &root,
                    &tokens,
                );
            }
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { .. } = event {
                let sc: State<Sidecars> = window.state();
                shutdown_sidecars(&sc);
                let pty_state: State<pty::PtyState> = window.state();
                pty::kill_all(&pty_state);
                // Use `stop_remote_sidecar` (AppHandle in scope) so the
                // remote sidecar token is cleared on window close — kill_all
                // can't reach SidecarTokens without an AppHandle, and we
                // don't want a stale token outliving the tunnel.
                let _ = remote_sidecar::stop_remote_sidecar(window.app_handle().clone());
            }
        })
        .invoke_handler(tauri::generate_handler![
            pick_workspace_dir,
            current_workspace_dir,
            close_workspace_dir,
            set_workspace_dir,
            list_workspace,
            read_workspace_file,
            write_workspace_file,
            delete_workspace_path,
            mkdir_workspace,
            rename_workspace_path,
            sidecar_managed_status,
            list_datasets,
            dataset_abspath,
            load_project,
            init_project,
            update_project_meta,
            migrate_legacy_project,
            list_notes,
            read_note,
            write_note,
            append_note,
            append_experiment,
            read_experiment,
            ssh::ssh_test_connection,
            ssh::ssh_load_project,
            ssh::ssh_init_project,
            ssh::ssh_update_project_meta,
            ssh::ssh_close,
            ssh::ssh_current,
            ssh::ssh_walk,
            ssh::ssh_read_file,
            ssh::ssh_write_file,
            ssh::ssh_delete_path,
            ssh::ssh_mkdir,
            ssh::ssh_rename,
            ssh::ssh_list_notes,
            ssh::ssh_read_note,
            ssh::ssh_write_note,
            ssh::ssh_append_note,
            ssh::ssh_append_experiment,
            ssh::ssh_read_experiment,
            ssh::ssh_list_datasets,
            ssh::ssh_start_training_run,
            ssh::ssh_list_training_runs,
            ssh::ssh_training_run_status,
            ssh::ssh_read_training_run_file,
            ssh::ssh_stop_training_run,
            ssh::ssh_delete_training_run,
            ssh::ssh_remote_training_capabilities,
            ssh::ssh_promote_checkpoint,
            ssh::ssh_gpu_stats,
            pty::pty_spawn,
            pty::pty_write,
            pty::pty_resize,
            pty::pty_kill,
            remote_sidecar::ensure_remote_sidecar,
            remote_sidecar::stop_remote_sidecar,
            remote_sidecar::remote_sidecar_status,
            sidecar_auth::sidecar_token,
            training::list_training_runs,
            training::training_run_status,
            training::read_training_run_file,
            training::start_training_run,
            training::stop_training_run,
            training::delete_training_run,
            training::promote_run_checkpoint,
            training::gpu_stats,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application"); // startup failure of the whole app has no caller to return to (Phase 48)
}

#[cfg(test)]
mod resolve_tests {
    use super::resolve;

    /// Build a fresh, isolated temp directory under std::env::temp_dir().
    /// Each call returns a unique path (pid + nanos + counter), so parallel
    /// test runners cannot collide on disk. Caller takes ownership; the
    /// directory is not auto-cleaned (drop is explicit in each test).
    fn unique_tempdir(label: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let pid = std::process::id();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!(
            "spinoml-resolve-test-{label}-{pid:x}-{nanos:x}-{n:x}"
        ));
        std::fs::create_dir_all(&p).expect("create temp dir");
        p
    }

    #[test]
    fn resolve_accepts_a_normal_relative_path() {
        let root = unique_tempdir("normal");
        let full = resolve(&root, "datasets/data.csv").unwrap();
        assert!(full.starts_with(root.canonicalize().unwrap()));
    }

    #[test]
    fn resolve_rejects_dot_dot_lexically() {
        let root = unique_tempdir("dotdot");
        assert!(resolve(&root, "../etc/passwd").is_err());
    }

    #[test]
    fn resolve_rejects_absolute_lexically() {
        let root = unique_tempdir("abs");
        assert!(resolve(&root, "/etc/passwd").is_err());
    }

    #[test]
    fn resolve_accepts_new_file_in_existing_subdir() {
        let root = unique_tempdir("newfile");
        let full = resolve(&root, "models/best/foo.pt").unwrap();
        // Canonical root + missing tail → canonical path under the root.
        let real_root = root.canonicalize().unwrap();
        assert!(full.starts_with(&real_root));
        assert!(full.ends_with("models/best/foo.pt"));
    }

    #[test]
    fn resolve_rejects_symlink_to_outside_when_no_symlink_target_configured() {
        let root = unique_tempdir("symlink-out");
        let outside = unique_tempdir("symlink-out-side");
        std::fs::write(outside.join("secret"), "x").unwrap();
        std::os::unix::fs::symlink(outside.join("secret"), root.join("link")).unwrap();
        let res = resolve(&root, "link");
        assert!(res.is_err(), "symlink escape must be rejected: {:?}", res);
    }
}

/// R016 regression tests for the pure path helpers (`delete_path_impl`,
/// `rename_path_impl`, `list_workspace_impl`). These run against real temp
/// dirs and drive the actual scope checks. All env mutation happens under the
/// shared `scope_file::ENV_LOCK` so these tests never stomp on the scope-file
/// tests (or each other) when cargo runs them in parallel.
#[cfg(test)]
mod path_impl_tests {
    use super::{delete_path_impl, list_workspace_impl, rename_path_impl, resolve};
    use crate::scope_file::ENV_LOCK;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::MutexGuard;

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Private HOME + env override for the duration of one test, serialised on
    /// the shared ENV_LOCK. Nothing touches the developer's real HOME.
    struct TestEnv {
        dir: PathBuf,
        prev_home: Option<std::ffi::OsString>,
        prev_xdg: Option<std::ffi::OsString>,
        prev_targets: Option<std::ffi::OsString>,
        _guard: MutexGuard<'static, ()>,
    }

    impl TestEnv {
        fn new() -> Self {
            let guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let pid = std::process::id();
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "spinoml-path-impl-test-{pid:x}-{nanos:x}-{n:x}"
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let prev_home = std::env::var_os("HOME");
            let prev_xdg = std::env::var_os("XDG_RUNTIME_DIR");
            let prev_targets = std::env::var_os("SPINOML_SYMLINK_TARGETS");
            std::env::set_var("HOME", &dir);
            std::env::remove_var("XDG_RUNTIME_DIR");
            std::env::remove_var("SPINOML_SYMLINK_TARGETS");
            Self { dir, prev_home, prev_xdg, prev_targets, _guard: guard }
        }

        fn work(&self) -> PathBuf {
            self.dir.join("ws")
        }

        fn target(&self) -> PathBuf {
            self.dir.join("target")
        }

        /// Configure `t` as an allowed symlink target via the env var.
        fn allow(&self, t: &Path) {
            std::env::set_var("SPINOML_SYMLINK_TARGETS", t.to_string_lossy().to_string());
        }
    }

    impl Drop for TestEnv {
        fn drop(&mut self) {
            match &self.prev_home {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
            match &self.prev_xdg {
                Some(v) => std::env::set_var("XDG_RUNTIME_DIR", v),
                None => std::env::remove_var("XDG_RUNTIME_DIR"),
            }
            match &self.prev_targets {
                Some(v) => std::env::set_var("SPINOML_SYMLINK_TARGETS", v),
                None => std::env::remove_var("SPINOML_SYMLINK_TARGETS"),
            }
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn mkdir(p: &Path) {
        std::fs::create_dir_all(p).unwrap();
    }
    fn write(p: &Path, s: &str) {
        std::fs::write(p, s).unwrap();
    }
    fn symlink(src: &Path, dst: &Path) {
        std::os::unix::fs::symlink(src, dst).unwrap();
    }
    fn link_exists(p: &Path) -> bool {
        std::fs::symlink_metadata(p).is_ok()
    }

    // ── Defect 2: delete/rename act on the LINK, not its target ────────────

    #[test]
    fn delete_symlink_to_allowed_target_removes_link_not_target() {
        let env = TestEnv::new();
        let ws = env.work();
        mkdir(&ws);
        let target = env.target();
        mkdir(&target);
        write(&target.join("data.csv"), "a,b\n1,2\n");
        symlink(&target, &ws.join("datasets"));
        env.allow(&target);

        delete_path_impl(&ws, "datasets").unwrap();

        assert!(!link_exists(&ws.join("datasets")), "the link must be gone");
        assert!(target.is_dir(), "the real target dir must survive");
        assert!(target.join("data.csv").is_file(), "the target files must survive");
    }

    #[test]
    fn delete_regular_dir_and_file_still_works() {
        let env = TestEnv::new();
        let ws = env.work();
        mkdir(&ws.join("sub"));
        write(&ws.join("sub/inner.txt"), "x");
        write(&ws.join("f.txt"), "y");

        delete_path_impl(&ws, "sub").unwrap();
        assert!(!link_exists(&ws.join("sub")), "regular dir must be deleted");
        delete_path_impl(&ws, "f.txt").unwrap();
        assert!(!link_exists(&ws.join("f.txt")), "regular file must be deleted");
    }

    #[test]
    fn delete_symlink_to_escaping_target_rejected_nothing_deleted() {
        let env = TestEnv::new();
        let ws = env.work();
        mkdir(&ws);
        let outside = env.dir.join("outside");
        mkdir(&outside);
        write(&outside.join("secret.txt"), "x");
        symlink(&outside, &ws.join("evil"));
        // No symlink target configured → the escape must be rejected.
        let r = delete_path_impl(&ws, "evil");
        assert!(r.is_err(), "escaping link must be rejected: {r:?}");
        assert!(link_exists(&ws.join("evil")), "the link must still be there");
        assert!(outside.join("secret.txt").is_file(), "outside data must be intact");
    }

    #[test]
    fn rename_symlink_moves_link_not_target() {
        let env = TestEnv::new();
        let ws = env.work();
        mkdir(&ws);
        let target = env.target();
        mkdir(&target);
        write(&target.join("data.csv"), "a\n1\n");
        symlink(&target, &ws.join("datasets"));
        env.allow(&target);

        rename_path_impl(&ws, "datasets", "datasets2").unwrap();

        assert!(!link_exists(&ws.join("datasets")), "old link gone");
        let meta = std::fs::symlink_metadata(ws.join("datasets2")).unwrap();
        assert!(meta.file_type().is_symlink(), "new path must be the symlink");
        assert!(target.is_dir(), "target dir survives");
        assert!(target.join("data.csv").is_file(), "target file survives");
    }

    #[test]
    fn resolve_through_allowed_symlink_returns_lexical_path_read_write() {
        let env = TestEnv::new();
        let ws = env.work();
        mkdir(&ws);
        let target = env.target();
        mkdir(&target);
        symlink(&target, &ws.join("datasets"));
        env.allow(&target);

        let p = resolve(&ws, "datasets/new.csv").unwrap();
        assert_eq!(p, ws.join("datasets/new.csv"), "resolve must return the LEXICAL path");
        // Reading/writing the lexical path goes through the link to the target.
        write(&p, "hello");
        assert_eq!(std::fs::read_to_string(target.join("new.csv")).unwrap(), "hello");
    }

    // ── Defect 3: listing follows allowed links, lexical relpaths, cycles ──

    #[test]
    fn list_workspace_lists_allowed_symlink_target_with_lexical_relpaths() {
        let env = TestEnv::new();
        let ws = env.work();
        mkdir(&ws);
        write(&ws.join("readme.txt"), "x");
        let target = env.target();
        mkdir(&target);
        write(&target.join("data.csv"), "a\n1\n");
        symlink(&target, &ws.join("datasets"));
        env.allow(&target);

        let entries = list_workspace_impl(&ws).unwrap();
        let rels: Vec<&str> = entries.iter().map(|e| e.relpath.as_str()).collect();
        assert!(rels.contains(&"datasets"), "missing datasets dir: {rels:?}");
        assert!(rels.contains(&"datasets/data.csv"), "missing linked file: {rels:?}");
        assert!(
            rels.iter().all(|r| !r.contains("target")),
            "relpaths must be lexical, not canonical: {rels:?}"
        );
    }

    #[test]
    fn list_workspace_omits_escaping_non_allowed_link() {
        let env = TestEnv::new();
        let ws = env.work();
        mkdir(&ws);
        write(&ws.join("ok.txt"), "x");
        let outside = env.dir.join("outside");
        mkdir(&outside);
        write(&outside.join("secret.csv"), "x");
        symlink(&outside, &ws.join("evil"));

        let entries = list_workspace_impl(&ws).unwrap();
        let rels: Vec<&str> = entries.iter().map(|e| e.relpath.as_str()).collect();
        assert!(rels.contains(&"ok.txt"));
        assert!(!rels.contains(&"evil"), "escaping link must be omitted: {rels:?}");
        assert!(!rels.iter().any(|r| r.starts_with("evil/")), "{rels:?}");
    }

    #[test]
    fn list_workspace_terminates_on_symlink_cycle_lists_each_file_once() {
        let env = TestEnv::new();
        let ws = env.work();
        mkdir(&ws);
        write(&ws.join("real.txt"), "x");
        // self -> . (workspace root) and self2 -> self: a 2-cycle.
        symlink(&ws, &ws.join("self"));
        symlink(&ws.join("self"), &ws.join("self2"));

        let entries = list_workspace_impl(&ws).unwrap();
        let real_count = entries.iter().filter(|e| e.relpath == "real.txt").count();
        assert_eq!(real_count, 1, "each real file must be listed once (found {real_count})");
        let rels: Vec<&str> = entries.iter().map(|e| e.relpath.as_str()).collect();
        assert!(rels.contains(&"self"), "{rels:?}");
        assert!(rels.contains(&"self2"), "{rels:?}");
    }

    #[test]
    fn list_workspace_skips_hidden_entries() {
        let env = TestEnv::new();
        let ws = env.work();
        mkdir(&ws);
        write(&ws.join("visible.txt"), "x");
        write(&ws.join(".hidden"), "x");
        mkdir(&ws.join(".hiddendir"));
        write(&ws.join(".hiddendir/deep.txt"), "x");

        let entries = list_workspace_impl(&ws).unwrap();
        let rels: Vec<&str> = entries.iter().map(|e| e.relpath.as_str()).collect();
        assert!(rels.contains(&"visible.txt"), "{rels:?}");
        assert!(
            rels.iter().all(|r| !r.starts_with('.')),
            "hidden entries must be skipped: {rels:?}"
        );
    }

    #[test]
    fn list_workspace_preserves_sort_order() {
        let env = TestEnv::new();
        let ws = env.work();
        mkdir(&ws.join("adir"));
        mkdir(&ws.join("zdir"));
        write(&ws.join("bfile.txt"), "x");
        write(&ws.join("afile.txt"), "x");

        let entries = list_workspace_impl(&ws).unwrap();
        let rels: Vec<&str> = entries.iter().map(|e| e.relpath.as_str()).collect();
        assert_eq!(rels, vec!["adir", "zdir", "afile.txt", "bfile.txt"], "{rels:?}");
    }
}
