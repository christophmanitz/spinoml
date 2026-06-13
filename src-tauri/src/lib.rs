use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tauri::{Manager, State, WindowEvent};
use tauri_plugin_dialog::DialogExt;

mod ssh;
mod pty;

pub(crate) const PROJECT_FILE: &str = "mlforge.project.json";
pub(crate) const SUBDIRS: &[&str] = &["models", "datasets", "notes", "experiments"];

#[derive(Default)]
struct WorkspaceState {
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
        .unwrap_or(manifest)
}

fn spawn_managed(label: &str, prog: &str, arg: PathBuf, cwd: &Path) -> Option<Child> {
    if !arg.exists() {
        eprintln!("[mlforge] {label}: sidecar script not found at {}", arg.display());
        return None;
    }
    match Command::new(prog)
        .arg(&arg)
        .current_dir(cwd)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
    {
        Ok(child) => {
            eprintln!("[mlforge] {label}: spawned pid={} ({prog} {})", child.id(), arg.display());
            Some(child)
        }
        Err(e) => {
            eprintln!(
                "[mlforge] {label}: failed to spawn ({prog} {}): {e} \
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
            eprintln!("[mlforge] killing torch sidecar pid={}", c.id());
            let _ = c.kill();
            let _ = c.wait();
        }
    }
    if let Ok(mut t) = sc.llm.lock() {
        if let Some(mut c) = t.take() {
            eprintln!("[mlforge] killing llm sidecar pid={}", c.id());
            let _ = c.kill();
            let _ = c.wait();
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
        torch: sc.torch.lock().map(|g| g.is_some()).unwrap_or(false),
        llm: sc.llm.lock().map(|g| g.is_some()).unwrap_or(false),
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
    Ok(root.join(rel))
}

fn current_root(state: &State<WorkspaceState>) -> Result<PathBuf, String> {
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
            let _ = tx.send(result);
        });
    let result = rx.await.map_err(|e| e.to_string())?;
    let path = match result {
        Some(p) => p,
        None => return Ok(None),
    };
    let buf = path
        .into_path()
        .map_err(|e| format!("path conversion: {e}"))?;
    let state: State<WorkspaceState> = app.state();
    *state.root.lock().map_err(|e| e.to_string())? = Some(buf.clone());
    Ok(Some(buf.to_string_lossy().to_string()))
}

#[tauri::command]
fn current_workspace_dir(state: State<WorkspaceState>) -> Option<String> {
    state
        .root
        .lock()
        .ok()
        .and_then(|g| g.clone())
        .map(|p| p.to_string_lossy().to_string())
}

#[tauri::command]
fn close_workspace_dir(state: State<WorkspaceState>) -> Result<(), String> {
    *state.root.lock().map_err(|e| e.to_string())? = None;
    Ok(())
}

#[tauri::command]
fn list_workspace(state: State<WorkspaceState>) -> Result<Vec<FsEntry>, String> {
    let root = current_root(&state)?;
    let mut out: Vec<FsEntry> = Vec::new();
    walk(&root, &root, &mut out)?;
    out.sort_by(|a, b| {
        a.is_dir
            .cmp(&b.is_dir)
            .reverse()
            .then_with(|| a.relpath.to_lowercase().cmp(&b.relpath.to_lowercase()))
    });
    Ok(out)
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<FsEntry>) -> Result<(), String> {
    let read = fs::read_dir(dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
    for entry in read.flatten() {
        let p = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        let rel = p
            .strip_prefix(root)
            .map_err(|e| e.to_string())?
            .to_string_lossy()
            .replace('\\', "/");
        let meta = entry.metadata().map_err(|e| e.to_string())?;
        if meta.is_dir() {
            out.push(FsEntry { name, relpath: rel.clone(), is_dir: true });
            walk(root, &p, out)?;
        } else {
            // Everything visible. The frontend decides what's clickable.
            out.push(FsEntry { name, relpath: rel, is_dir: false });
        }
    }
    Ok(())
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

#[tauri::command]
fn delete_workspace_path(state: State<WorkspaceState>, relpath: String) -> Result<(), String> {
    let root = current_root(&state)?;
    let full = resolve(&root, &relpath)?;
    if !full.starts_with(&root) {
        return Err("refusing to delete outside workspace root".into());
    }
    let meta = fs::metadata(&full).map_err(|e| format!("stat {}: {e}", full.display()))?;
    if meta.is_dir() {
        fs::remove_dir_all(&full).map_err(|e| format!("rmdir {}: {e}", full.display()))
    } else {
        fs::remove_file(&full).map_err(|e| format!("rm {}: {e}", full.display()))
    }
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
    legacy_mlforge_count: usize,
}

pub(crate) fn now_iso() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
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
            if name.to_lowercase().ends_with(".mlforge") {
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
        legacy_mlforge_count: count,
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
    // Move root-level .mlforge and matching .py twins into models/.
    let models = root.join("models");
    if let Ok(rd) = fs::read_dir(&root) {
        for entry in rd.flatten() {
            let p = entry.path();
            if !p.is_file() { continue; }
            let name_os = entry.file_name();
            let fname = name_os.to_string_lossy().to_string();
            let lower = fname.to_lowercase();
            if lower.ends_with(".mlforge") || lower.ends_with(".py") {
                let dest = models.join(&fname);
                fs::rename(&p, &dest).map_err(|e| format!("move {fname}: {e}"))?;
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
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| secs_to_iso(d.as_secs()))
            .unwrap_or_default();
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
    let p = root.join("notes").join(&name);
    fs::read_to_string(&p).map_err(|e| format!("read {}: {e}", p.display()))
}

#[tauri::command]
fn write_note(state: State<WorkspaceState>, name: String, content: String) -> Result<(), String> {
    let root = current_root(&state)?;
    if name.contains('/') || name.contains('\\') {
        return Err("note name must be a plain filename".into());
    }
    let dir = root.join("notes");
    fs::create_dir_all(&dir).map_err(|e| format!("mkdir notes/: {e}"))?;
    let p = dir.join(&name);
    fs::write(&p, content).map_err(|e| format!("write {}: {e}", p.display()))
}

#[tauri::command]
fn append_note(state: State<WorkspaceState>, name: String, content: String) -> Result<(), String> {
    use std::io::Write;
    let root = current_root(&state)?;
    if name.contains('/') || name.contains('\\') {
        return Err("note name must be a plain filename".into());
    }
    let dir = root.join("notes");
    fs::create_dir_all(&dir).map_err(|e| format!("mkdir notes/: {e}"))?;
    let p = dir.join(&name);
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
    let dir = root.join("experiments");
    fs::create_dir_all(&dir).map_err(|e| format!("mkdir experiments/: {e}"))?;
    let p = dir.join(&filename);
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
    let p = root.join("experiments").join(&filename);
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
    let dsdir = root.join("datasets");
    if !dsdir.exists() {
        fs::create_dir_all(&dsdir).map_err(|e| format!("mkdir datasets/: {e}"))?;
    }
    let mut out: Vec<DatasetEntry> = Vec::new();
    let read = fs::read_dir(&dsdir).map_err(|e| format!("read_dir {}: {e}", dsdir.display()))?;
    for entry in read.flatten() {
        let p = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        let meta = entry.metadata().map_err(|e| e.to_string())?;
        let size = if meta.is_file() { meta.len() } else { dir_size(&p).unwrap_or(0) };
        let rel = format!("datasets/{}", name);
        out.push(DatasetEntry {
            name,
            relpath: rel,
            abspath: p.to_string_lossy().to_string(),
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
    let read = fs::read_dir(p).ok()?;
    for entry in read.flatten() {
        let meta = entry.metadata().ok()?;
        if meta.is_file() {
            total += meta.len();
        } else if meta.is_dir() {
            total += dir_size(&entry.path()).unwrap_or(0);
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

#[tauri::command]
fn rename_workspace_path(
    state: State<WorkspaceState>,
    from_rel: String,
    to_rel: String,
) -> Result<(), String> {
    let root = current_root(&state)?;
    let from = resolve(&root, &from_rel)?;
    let to = resolve(&root, &to_rel)?;
    if let Some(parent) = to.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
    }
    fs::rename(&from, &to).map_err(|e| format!("rename: {e}"))
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
            if let Ok(mut t) = sc.torch.lock() {
                *t = spawn_managed(
                    "sidecar-torch",
                    "python",
                    root.join("sidecar-torch").join("main.py"),
                    &root,
                );
            }
            if let Ok(mut l) = sc.llm.lock() {
                *l = spawn_managed(
                    "sidecar-llm",
                    "node",
                    root.join("sidecar-llm").join("main.mjs"),
                    &root,
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
            }
        })
        .invoke_handler(tauri::generate_handler![
            pick_workspace_dir,
            current_workspace_dir,
            close_workspace_dir,
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
            pty::pty_spawn,
            pty::pty_write,
            pty::pty_resize,
            pty::pty_kill,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
