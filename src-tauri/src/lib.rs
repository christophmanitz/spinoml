use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;

use serde::Serialize;
use tauri::{Manager, State};
use tauri_plugin_dialog::DialogExt;

#[derive(Default)]
struct WorkspaceState {
    root: Mutex<Option<PathBuf>>,
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
            let lower = name.to_lowercase();
            if lower.ends_with(".mlforge") || lower.ends_with(".py") {
                out.push(FsEntry { name, relpath: rel, is_dir: false });
            }
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
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
