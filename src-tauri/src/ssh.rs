// SSH-backed remote workspace. Mirrors every local FS command as an `ssh_*`
// variant, transported via system `ssh` (so it uses ~/.ssh/config, agent,
// ProxyJump, GSSAPI, etc. — no secrets stored in MLForge itself).
//
// Each call forks an ssh subprocess. That's ~50-200ms per op against most
// hops; acceptable for interactive editing of small files but NOT for hot
// loops. If we ever need sustained throughput we'll add a persistent
// `sftp -b` session pool — but the API here stays the same.
//
// Path safety: remote_root can be either absolute (`/scratch/...`) or
// tilde-prefixed (`~/projects/mlforge`). Relpaths reuse the same component
// rules as the local resolver (no `..`, no absolute paths). Everything
// emitted into a shell command goes through `shell_quote_path()`, which
// handles `~/` by substituting `"$HOME"` (which IS expanded outside single
// quotes), so we never paste user input into bash unquoted.

use std::io::Write;
use std::path::{Component, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;

use serde::Serialize;
use serde_json::Value;
use tauri::{AppHandle, State};

use crate::training;
use crate::{now_iso, PROJECT_FILE, SUBDIRS};

#[derive(Default)]
pub struct RemoteWorkspaceState {
    pub current: Mutex<Option<RemoteWorkspace>>,
}

#[derive(Clone, Debug)]
pub struct RemoteWorkspace {
    pub alias: String,
    pub root: String,
}

// ─── validation ───────────────────────────────────────────────────────────

fn validate_alias(alias: &str) -> Result<(), String> {
    if alias.is_empty() || alias.len() > 128 {
        return Err("ssh target must be 1..128 chars".into());
    }
    for ch in alias.chars() {
        // SSH target = either a `Host` alias from ~/.ssh/config (e.g.
        // "leipzig-hpc") or a bare user@host pair when no config entry
        // exists ("zw93onug@login01.sc.uni-leipzig.de"). Allowed chars
        // cover both, and reject every shell metacharacter.
        if !(ch.is_ascii_alphanumeric()
            || ch == '.' || ch == '_' || ch == '-'
            || ch == '@' || ch == ':')
        {
            return Err(format!("ssh target contains illegal char: {ch:?}"));
        }
    }
    Ok(())
}

fn validate_remote_root(root: &str) -> Result<(), String> {
    if root.is_empty() {
        return Err("remote root is empty".into());
    }
    let absolute = root.starts_with('/') || root.starts_with("~/") || root == "~";
    if !absolute {
        return Err("remote root must be absolute (e.g. /scratch/me/proj) or start with ~/".into());
    }
    if root.contains('\0') || root.contains('\n') || root.contains('\r') {
        return Err("remote root contains illegal characters".into());
    }
    let p = PathBuf::from(root);
    for comp in p.components() {
        if matches!(comp, Component::ParentDir) {
            return Err("remote root must not contain '..'".into());
        }
    }
    Ok(())
}

fn validate_relpath(rel: &str) -> Result<(), String> {
    if rel.contains('\0') || rel.contains('\n') || rel.contains('\r') {
        return Err("relpath contains illegal characters".into());
    }
    let p = PathBuf::from(rel);
    for comp in p.components() {
        match comp {
            Component::Normal(_) | Component::CurDir => {}
            _ => return Err(format!("rejected path segment in {rel}")),
        }
    }
    Ok(())
}

fn validate_plain_filename(name: &str, label: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err(format!("{label} name is empty"));
    }
    if name.contains('/') || name.contains('\\') || name.contains('\0') || name.contains('\n') {
        return Err(format!("{label} name must be a plain filename"));
    }
    if name == "." || name == ".." {
        return Err(format!("{label} name is invalid"));
    }
    Ok(())
}

// ─── shell quoting + path joining ─────────────────────────────────────────

pub(crate) fn shell_quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for ch in s.chars() {
        if ch == '\'' {
            out.push_str("'\\''");
        } else {
            out.push(ch);
        }
    }
    out.push('\'');
    out
}

/// Like `shell_quote` but turns a leading `~/` or bare `~` into `"$HOME"`
/// so the remote shell expands it. Single-quoted segments never expand `~`,
/// so we have to break out of single quotes for that prefix specifically.
pub(crate) fn shell_quote_path(s: &str) -> String {
    if let Some(rest) = s.strip_prefix("~/") {
        // "$HOME" '/rest' — bash concatenates adjacent quoted strings.
        format!("\"$HOME\"{}", shell_quote(&format!("/{}", rest)))
    } else if s == "~" {
        "\"$HOME\"".to_string()
    } else {
        shell_quote(s)
    }
}

fn join_remote(root: &str, rel: &str) -> String {
    if rel.is_empty() {
        return root.to_string();
    }
    let trimmed = rel.trim_start_matches('/');
    if root.ends_with('/') {
        format!("{root}{trimmed}")
    } else {
        format!("{root}/{trimmed}")
    }
}

// ─── ssh subprocess wrapper ───────────────────────────────────────────────

pub(crate) const SSH_OPTS: &[&str] = &[
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=20",
    "-o", "ServerAliveCountMax=3",
];

/// Same as SSH_OPTS but without BatchMode — for the interactive terminal
/// where ssh may need to prompt for a 2FA token, host-key confirmation, etc.
pub(crate) const SSH_OPTS_INTERACTIVE: &[&str] = &[
    "-o", "ConnectTimeout=10",
    "-o", "ServerAliveInterval=20",
    "-o", "ServerAliveCountMax=3",
];

/// Async wrapper: the ssh subprocess blocks (connect + remote exec can take
/// seconds on an HPC login node), so it runs on tokio's blocking pool instead
/// of the caller's thread. Combined with `async` commands this keeps the GTK
/// main thread — and therefore the whole GUI — responsive while ssh is in
/// flight. Without this, every ssh_* command froze the webview ("not
/// responding") for the duration of the call.
async fn ssh_exec(alias: &str, remote_cmd: &str, stdin_data: Option<&[u8]>) -> Result<String, String> {
    let alias = alias.to_string();
    let remote_cmd = remote_cmd.to_string();
    let stdin_data = stdin_data.map(|d| d.to_vec());
    tauri::async_runtime::spawn_blocking(move || ssh_exec_blocking(&alias, &remote_cmd, stdin_data.as_deref()))
        .await
        .map_err(|e| format!("ssh task join: {e}"))?
}

fn ssh_exec_blocking(alias: &str, remote_cmd: &str, stdin_data: Option<&[u8]>) -> Result<String, String> {
    let mut cmd = Command::new("ssh");
    for o in SSH_OPTS {
        cmd.arg(o);
    }
    cmd.arg(alias).arg(remote_cmd);
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
    cmd.stdin(if stdin_data.is_some() { Stdio::piped() } else { Stdio::null() });
    let mut child = cmd
        .spawn()
        .map_err(|e| format!("spawn ssh: {e} — is the `ssh` binary on PATH?"))?;
    if let Some(data) = stdin_data {
        let mut stdin = child.stdin.take().ok_or_else(|| "no stdin handle".to_string())?;
        let owned = data.to_vec();
        std::thread::spawn(move || {
            let _ = stdin.write_all(&owned);
            // Drop closes the pipe → remote `cat` sees EOF and exits.
        });
    }
    let out = child
        .wait_with_output()
        .map_err(|e| format!("wait ssh: {e}"))?;
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr).to_string();
        let code = out
            .status
            .code()
            .map(|c| c.to_string())
            .unwrap_or_else(|| "?".into());
        let hint = if code == "255" {
            "\n(SSH exit 255: connection/auth failed. Check ~/.ssh/config, ssh-agent, network reachability.)"
        } else {
            ""
        };
        return Err(format!("ssh exit {code}: {}{hint}", stderr.trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

// ─── commands ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct SshTestResult {
    pub ok: bool,
    pub uname: String,
    pub home: String,
}

#[tauri::command]
pub async fn ssh_test_connection(alias: String) -> Result<SshTestResult, String> {
    validate_alias(&alias)?;
    let out = ssh_exec(
        &alias,
        "echo MLFORGE_OK && uname -srm && echo \"HOME=$HOME\"",
        None,
    ).await?;
    if !out.contains("MLFORGE_OK") {
        return Err(format!("unexpected reply (no MLFORGE_OK marker): {}", out.trim()));
    }
    let mut uname = String::new();
    let mut home = String::new();
    for line in out.lines() {
        if line == "MLFORGE_OK" {
            continue;
        }
        if let Some(rest) = line.strip_prefix("HOME=") {
            home = rest.to_string();
        } else if uname.is_empty() {
            uname = line.to_string();
        }
    }
    Ok(SshTestResult { ok: true, uname, home })
}

#[derive(Serialize)]
pub struct RemoteProjectLoad {
    pub root: String,
    pub meta: Option<serde_json::Value>,
    pub root_exists: bool,
    pub has_legacy_files: bool,
    pub legacy_mlforge_count: usize,
}

#[tauri::command]
pub async fn ssh_load_project(
    state: State<'_, RemoteWorkspaceState>,
    alias: String,
    root: String,
) -> Result<RemoteProjectLoad, String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    let proj_path = join_remote(&root, PROJECT_FILE);
    let root_q = shell_quote_path(&root);
    let proj_q = shell_quote_path(&proj_path);
    // One round-trip: check root, list .mlforge files, dump project.json if present.
    let cmd = format!(
        "if [ -d {root_q} ]; then \
            echo ROOT_EXISTS; \
            ls -1 {root_q} 2>/dev/null | grep -i '\\.mlforge$' | wc -l; \
            if [ -f {proj_q} ]; then echo PROJECT_BEGIN; cat {proj_q}; echo; echo PROJECT_END; fi; \
         else echo ROOT_MISSING; fi"
    );
    let out = ssh_exec(&alias, &cmd, None).await?;
    let mut root_exists = false;
    let mut legacy_count: usize = 0;
    let mut meta_lines: Vec<&str> = Vec::new();
    let mut in_meta = false;
    let mut saw_count = false;
    for line in out.lines() {
        if line == "ROOT_EXISTS" {
            root_exists = true;
        } else if line == "ROOT_MISSING" {
            root_exists = false;
        } else if line == "PROJECT_BEGIN" {
            in_meta = true;
        } else if line == "PROJECT_END" {
            in_meta = false;
        } else if in_meta {
            meta_lines.push(line);
        } else if root_exists && !saw_count {
            if let Ok(n) = line.trim().parse::<usize>() {
                legacy_count = n;
                saw_count = true;
            }
        }
    }
    let meta = if meta_lines.is_empty() {
        None
    } else {
        let body = meta_lines.join("\n");
        let parsed: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| format!("parse remote project.json: {e}"))?;
        Some(parsed)
    };
    if root_exists {
        *state.current.lock().map_err(|e| e.to_string())? =
            Some(RemoteWorkspace { alias: alias.clone(), root: root.clone() });
    }
    Ok(RemoteProjectLoad {
        root,
        meta,
        root_exists,
        has_legacy_files: legacy_count > 0,
        legacy_mlforge_count: legacy_count,
    })
}

#[tauri::command]
pub async fn ssh_init_project(
    state: State<'_, RemoteWorkspaceState>,
    alias: String,
    root: String,
    name: String,
    description: String,
    goal: String,
) -> Result<serde_json::Value, String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    let now = now_iso();
    let meta = serde_json::json!({
        "name": name,
        "description": description,
        "goal": goal,
        "active_model": null,
        "active_dataset": null,
        "created_at": now,
        "updated_at": now,
        "schema_version": 1,
    });
    let pretty = serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?;
    let root_q = shell_quote_path(&root);
    let proj_q = shell_quote_path(&join_remote(&root, PROJECT_FILE));
    let subdir_qs: Vec<String> = SUBDIRS
        .iter()
        .map(|s| shell_quote_path(&join_remote(&root, s)))
        .collect();
    let cmd = format!(
        "mkdir -p {root_q} {subs} && if [ -f {proj_q} ]; then echo PROJECT_EXISTS >&2; exit 2; else cat > {proj_q}; fi",
        subs = subdir_qs.join(" "),
    );
    ssh_exec(&alias, &cmd, Some(pretty.as_bytes())).await?;
    *state.current.lock().map_err(|e| e.to_string())? =
        Some(RemoteWorkspace { alias: alias.clone(), root: root.clone() });
    Ok(meta)
}

#[tauri::command]
pub async fn ssh_update_project_meta(
    alias: String,
    root: String,
    patch: serde_json::Value,
) -> Result<serde_json::Value, String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    let proj_q = shell_quote_path(&join_remote(&root, PROJECT_FILE));
    let body = ssh_exec(&alias, &format!("cat {proj_q}"), None).await?;
    let mut meta: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| format!("parse remote project.json: {e}"))?;
    let map = meta
        .as_object_mut()
        .ok_or_else(|| "project.json is not a JSON object".to_string())?;
    if let Some(obj) = patch.as_object() {
        for (k, v) in obj {
            map.insert(k.clone(), v.clone());
        }
    }
    map.insert("updated_at".into(), serde_json::Value::String(now_iso()));
    let pretty = serde_json::to_string_pretty(&meta).map_err(|e| e.to_string())?;
    ssh_exec(&alias, &format!("cat > {proj_q}"), Some(pretty.as_bytes())).await?;
    Ok(meta)
}

// Sync: no ssh, just a mutex flip — stays off the async runtime.
#[tauri::command]
pub fn ssh_close(state: State<RemoteWorkspaceState>) -> Result<(), String> {
    *state.current.lock().map_err(|e| e.to_string())? = None;
    Ok(())
}

#[derive(Serialize)]
pub struct CurrentRemote {
    pub alias: String,
    pub root: String,
}

// Sync: no ssh, just reads the mutex.
#[tauri::command]
pub fn ssh_current(state: State<RemoteWorkspaceState>) -> Option<CurrentRemote> {
    let g = state.current.lock().ok()?;
    let r = g.as_ref()?;
    Some(CurrentRemote { alias: r.alias.clone(), root: r.root.clone() })
}

// ─── filesystem ───────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct RemoteFsEntry {
    pub name: String,
    pub relpath: String,
    pub is_dir: bool,
}

#[tauri::command]
pub async fn ssh_walk(alias: String, root: String) -> Result<Vec<RemoteFsEntry>, String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    let root_q = shell_quote_path(&root);
    // GNU find: %y = type letter (f/d/l/…), %P = path relative to start.
    // Skip dotfiles anywhere in the path.
    let cmd = format!(
        "if [ -d {root_q} ]; then \
            find {root_q} -mindepth 1 -maxdepth 16 \\( -type d -o -type f \\) \
              ! -path '*/.*' -printf '%y\\t%P\\n' 2>/dev/null; \
         fi"
    );
    let out = ssh_exec(&alias, &cmd, None).await?;
    let mut entries: Vec<RemoteFsEntry> = Vec::new();
    for line in out.lines() {
        let mut it = line.splitn(2, '\t');
        let kind = it.next().unwrap_or("");
        let rel = it.next().unwrap_or("").to_string();
        if rel.is_empty() {
            continue;
        }
        let is_dir = kind == "d";
        let name = rel.rsplit('/').next().unwrap_or(&rel).to_string();
        entries.push(RemoteFsEntry { name, relpath: rel, is_dir });
    }
    entries.sort_by(|a, b| {
        a.is_dir
            .cmp(&b.is_dir)
            .reverse()
            .then_with(|| a.relpath.to_lowercase().cmp(&b.relpath.to_lowercase()))
    });
    Ok(entries)
}

#[tauri::command]
pub async fn ssh_read_file(alias: String, root: String, relpath: String) -> Result<String, String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    validate_relpath(&relpath)?;
    let p_q = shell_quote_path(&join_remote(&root, &relpath));
    ssh_exec(&alias, &format!("cat {p_q}"), None).await
}

#[tauri::command]
pub async fn ssh_write_file(
    alias: String,
    root: String,
    relpath: String,
    content: String,
) -> Result<(), String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    validate_relpath(&relpath)?;
    let abs = join_remote(&root, &relpath);
    let parent = abs
        .rsplit_once('/')
        .map(|(a, _)| a.to_string())
        .unwrap_or_else(|| root.clone());
    let p_q = shell_quote_path(&abs);
    let parent_q = shell_quote_path(&parent);
    ssh_exec(
        &alias,
        &format!("mkdir -p {parent_q} && cat > {p_q}"),
        Some(content.as_bytes()),
    )
    .await
    .map(|_| ())
}

#[tauri::command]
pub async fn ssh_delete_path(alias: String, root: String, relpath: String) -> Result<(), String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    validate_relpath(&relpath)?;
    if relpath.is_empty() || relpath == "." {
        return Err("refusing to delete workspace root".into());
    }
    let p_q = shell_quote_path(&join_remote(&root, &relpath));
    ssh_exec(&alias, &format!("rm -rf -- {p_q}"), None).await.map(|_| ())
}

#[tauri::command]
pub async fn ssh_mkdir(alias: String, root: String, relpath: String) -> Result<(), String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    validate_relpath(&relpath)?;
    let p_q = shell_quote_path(&join_remote(&root, &relpath));
    ssh_exec(&alias, &format!("mkdir -p {p_q}"), None).await.map(|_| ())
}

#[tauri::command]
pub async fn ssh_rename(
    alias: String,
    root: String,
    from_rel: String,
    to_rel: String,
) -> Result<(), String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    validate_relpath(&from_rel)?;
    validate_relpath(&to_rel)?;
    let from_q = shell_quote_path(&join_remote(&root, &from_rel));
    let to_abs = join_remote(&root, &to_rel);
    let to_parent = to_abs
        .rsplit_once('/')
        .map(|(a, _)| a.to_string())
        .unwrap_or_else(|| root.clone());
    let to_parent_q = shell_quote_path(&to_parent);
    let to_q = shell_quote_path(&to_abs);
    ssh_exec(
        &alias,
        &format!("mkdir -p {to_parent_q} && mv -- {from_q} {to_q}"),
        None,
    )
    .await
    .map(|_| ())
}

// ─── notes ────────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct RemoteNoteEntry {
    pub name: String,
    pub relpath: String,
    pub size_bytes: u64,
    pub modified_at: String,
}

fn strip_iso_nanos(s: &str) -> String {
    if let Some(dot) = s.find('.') {
        // After the dot, find the first non-digit/dot character (e.g. 'Z' or '+').
        let tail = &s[dot..];
        let suffix_idx = tail
            .char_indices()
            .find(|(_, c)| !c.is_ascii_digit() && *c != '.')
            .map(|(i, _)| i)
            .unwrap_or(tail.len());
        let mut out = String::with_capacity(s.len());
        out.push_str(&s[..dot]);
        out.push_str(&tail[suffix_idx..]);
        out
    } else {
        s.to_string()
    }
}

#[tauri::command]
pub async fn ssh_list_notes(alias: String, root: String) -> Result<Vec<RemoteNoteEntry>, String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    let dir_q = shell_quote_path(&join_remote(&root, "notes"));
    let cmd = format!(
        "mkdir -p {dir_q} && find {dir_q} -mindepth 1 -maxdepth 1 -type f \
           \\( -name '*.md' -o -name '*.txt' \\) \
           -printf '%f\\t%s\\t%TY-%Tm-%TdT%TH:%TM:%TSZ\\n' 2>/dev/null"
    );
    let out = ssh_exec(&alias, &cmd, None).await?;
    let mut entries: Vec<RemoteNoteEntry> = Vec::new();
    for line in out.lines() {
        let parts: Vec<&str> = line.splitn(3, '\t').collect();
        if parts.len() < 3 {
            continue;
        }
        let name = parts[0].to_string();
        if name.starts_with('.') {
            continue;
        }
        let size = parts[1].parse::<u64>().unwrap_or(0);
        let modified = strip_iso_nanos(parts[2]);
        entries.push(RemoteNoteEntry {
            relpath: format!("notes/{name}"),
            name,
            size_bytes: size,
            modified_at: modified,
        });
    }
    entries.sort_by(|a, b| b.modified_at.cmp(&a.modified_at));
    Ok(entries)
}

#[tauri::command]
pub async fn ssh_read_note(alias: String, root: String, name: String) -> Result<String, String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    validate_plain_filename(&name, "note")?;
    let p_q = shell_quote_path(&format!("{}/notes/{}", root.trim_end_matches('/'), name));
    ssh_exec(&alias, &format!("cat {p_q}"), None).await
}

#[tauri::command]
pub async fn ssh_write_note(
    alias: String,
    root: String,
    name: String,
    content: String,
) -> Result<(), String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    validate_plain_filename(&name, "note")?;
    let dir_q = shell_quote_path(&join_remote(&root, "notes"));
    let p_q = shell_quote_path(&format!("{}/notes/{}", root.trim_end_matches('/'), name));
    ssh_exec(
        &alias,
        &format!("mkdir -p {dir_q} && cat > {p_q}"),
        Some(content.as_bytes()),
    )
    .await
    .map(|_| ())
}

#[tauri::command]
pub async fn ssh_append_note(
    alias: String,
    root: String,
    name: String,
    content: String,
) -> Result<(), String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    validate_plain_filename(&name, "note")?;
    let dir_q = shell_quote_path(&join_remote(&root, "notes"));
    let p_q = shell_quote_path(&format!("{}/notes/{}", root.trim_end_matches('/'), name));
    ssh_exec(
        &alias,
        &format!("mkdir -p {dir_q} && cat >> {p_q}"),
        Some(content.as_bytes()),
    )
    .await
    .map(|_| ())
}

// ─── experiments ──────────────────────────────────────────────────────────

#[tauri::command]
pub async fn ssh_append_experiment(
    alias: String,
    root: String,
    filename: String,
    line: String,
) -> Result<(), String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    validate_plain_filename(&filename, "experiment")?;
    let dir_q = shell_quote_path(&join_remote(&root, "experiments"));
    let p_q = shell_quote_path(&format!(
        "{}/experiments/{}",
        root.trim_end_matches('/'),
        filename
    ));
    let mut payload = line.into_bytes();
    if !payload.ends_with(b"\n") {
        payload.push(b'\n');
    }
    ssh_exec(
        &alias,
        &format!("mkdir -p {dir_q} && cat >> {p_q}"),
        Some(&payload),
    )
    .await
    .map(|_| ())
}

#[tauri::command]
pub async fn ssh_read_experiment(
    alias: String,
    root: String,
    filename: String,
) -> Result<String, String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    validate_plain_filename(&filename, "experiment")?;
    let p_q = shell_quote_path(&format!(
        "{}/experiments/{}",
        root.trim_end_matches('/'),
        filename
    ));
    // Tolerate missing file: emit empty.
    ssh_exec(&alias, &format!("if [ -f {p_q} ]; then cat {p_q}; fi"), None).await
}

// ─── datasets ─────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct RemoteDatasetEntry {
    pub name: String,
    pub relpath: String,
    pub abspath: String,
    pub is_dir: bool,
    pub size_bytes: u64,
}

#[tauri::command]
pub async fn ssh_list_datasets(
    alias: String,
    root: String,
) -> Result<Vec<RemoteDatasetEntry>, String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    let dsdir = join_remote(&root, "datasets");
    let dir_q = shell_quote_path(&dsdir);
    // Files: stat-based size via find -printf. Dirs: du -sb. Done in one ssh round-trip.
    let cmd = format!(
        "mkdir -p {dir_q} && \
         find {dir_q} -mindepth 1 -maxdepth 1 -type f ! -name '.*' \
           -printf 'F\\t%f\\t%s\\n' 2>/dev/null; \
         for d in {dir_q}/*/; do \
           [ -d \"$d\" ] || continue; \
           bn=$(basename \"$d\"); \
           case \"$bn\" in .*) continue ;; esac; \
           sz=$(du -sb \"$d\" 2>/dev/null | cut -f1); \
           printf 'D\\t%s\\t%s\\n' \"$bn\" \"$sz\"; \
         done"
    );
    let out = ssh_exec(&alias, &cmd, None).await?;
    let mut entries: Vec<RemoteDatasetEntry> = Vec::new();
    // For abspath we need the *expanded* root if it was ~/-prefixed. The
    // shell did the expansion; we don't know HOME locally. Easiest: emit
    // the literal user-typed root in abspath (sidecar runs locally and
    // can't use it anyway; this field is for display/sidecar handoff and
    // will be ignored in Phase 12a — Phase 12b's remote sidecar resolves
    // tilde itself in the sidecar's own shell).
    for line in out.lines() {
        let parts: Vec<&str> = line.splitn(3, '\t').collect();
        if parts.len() < 3 {
            continue;
        }
        let kind = parts[0];
        let name = parts[1].to_string();
        let size = parts[2].parse::<u64>().unwrap_or(0);
        if name.starts_with('.') {
            continue;
        }
        let is_dir = kind == "D";
        let relpath = format!("datasets/{name}");
        let abspath = format!("{}/datasets/{name}", root.trim_end_matches('/'));
        entries.push(RemoteDatasetEntry {
            name,
            relpath,
            abspath,
            is_dir,
            size_bytes: size,
        });
    }
    entries.sort_by(|a, b| {
        a.is_dir
            .cmp(&b.is_dir)
            .reverse()
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });
    Ok(entries)
}

// ─── remote training executor (Phase 16: ssh-direct) ──────────────────────
//
// Mirrors src/training.rs over ssh. A run is the SAME self-contained directory
// (experiments/runs/<run_id>/) on the remote host; we ship the frozen files in,
// launch the trainer detached (`nohup setsid` → survives both the ssh session
// AND the MLForge app), and afterwards only read files + `kill -0` over ssh.
// No SLURM here — that's Phase 17.

/// The python interpreter on the remote is user-configured per connection. Keep
/// it a tame token (path or bare name) — it's pasted into a shell command.
fn validate_python(p: &str) -> Result<(), String> {
    if p.is_empty() || p.len() > 512 {
        return Err("python path must be 1..512 chars".into());
    }
    if p.chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '/' | '~'))
    {
        Ok(())
    } else {
        Err("python path may only contain [A-Za-z0-9._/~-] (no spaces/metachars)".into())
    }
}

fn remote_runs_dir(root: &str) -> String {
    join_remote(root, "experiments/runs")
}
fn remote_run_dir(root: &str, run_id: &str) -> String {
    join_remote(root, &format!("experiments/runs/{run_id}"))
}

async fn write_remote_run_file(alias: &str, dir: &str, name: &str, content: &[u8]) -> Result<(), String> {
    let p_q = shell_quote_path(&format!("{dir}/{name}"));
    ssh_exec(alias, &format!("cat > {p_q}"), Some(content)).await.map(|_| ())
}

#[tauri::command]
pub async fn ssh_start_training_run(
    app: AppHandle,
    alias: String,
    root: String,
    run_id: String,
    python: String,
    run_json: String,
    model_mlforge: String,
    model_py: String,
) -> Result<(), String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    training::validate_run_id(&run_id)?;
    let python = {
        let t = python.trim();
        if t.is_empty() { "python".to_string() } else { t.to_string() }
    };
    validate_python(&python)?;

    let dir = remote_run_dir(&root, &run_id);
    let dir_q = shell_quote_path(&dir);
    let ckpt_q = shell_quote_path(&format!("{dir}/checkpoints"));
    let python_q = shell_quote_path(&python);

    // refuse to clobber an existing run
    let exists = ssh_exec(&alias, &format!("if [ -d {dir_q} ]; then echo EXISTS; fi"), None).await?;
    if exists.contains("EXISTS") {
        return Err(format!("run {run_id} already exists on {alias}"));
    }
    ssh_exec(&alias, &format!("mkdir -p {ckpt_q}"), None).await?;

    // frozen snapshots
    write_remote_run_file(&alias, &dir, "run.json", run_json.as_bytes()).await?;
    write_remote_run_file(&alias, &dir, "model.mlforge", model_mlforge.as_bytes()).await?;
    write_remote_run_file(&alias, &dir, "model.py", model_py.as_bytes()).await?;

    // ship the shared trainer in as train.py (read from the local bundle)
    let template = crate::sidecar_root_pub(&app)
        .join("sidecar-torch")
        .join("training_template.py");
    let trainer = std::fs::read_to_string(&template).map_err(|e| {
        format!(
            "training template missing at {} ({e}). MLForge bundle may be incomplete.",
            template.display()
        )
    })?;
    write_remote_run_file(&alias, &dir, "train.py", trainer.as_bytes()).await?;
    write_remote_run_file(&alias, &dir, "status", b"queued\n").await?;

    // Pick the launch path from the frozen backend in run.json: SLURM → sbatch,
    // anything else → direct detached process (Phase 16).
    let cfg: Value = serde_json::from_str(&run_json).unwrap_or(Value::Null);
    let backend_kind = cfg
        .get("backend")
        .and_then(|b| b.get("kind"))
        .and_then(|k| k.as_str())
        .unwrap_or("local");

    if backend_kind == "slurm" {
        let slurm = cfg.get("backend").and_then(|b| b.get("slurm"));
        let sbatch = build_sbatch(&run_id, &python_q, slurm);
        write_remote_run_file(&alias, &dir, "train.sbatch", sbatch.as_bytes()).await?;
        // Submit; parse "Submitted batch job <id>"; freeze pid as slurm:<id>.
        let submit = format!(
            "cd {dir_q} && out=$(sbatch train.sbatch 2>&1); echo \"$out\"; \
             jid=$(printf '%s' \"$out\" | grep -oE 'job [0-9]+' | grep -oE '[0-9]+' | tail -1); \
             if [ -n \"$jid\" ]; then printf 'slurm:%s\\n' \"$jid\" > pid; echo \"MLF_JOBID $jid\"; \
             else echo MLF_SUBMIT_FAILED; fi"
        );
        let out = ssh_exec(&alias, &submit, None).await?;
        if !out.contains("MLF_JOBID") {
            // surface sbatch's own error text (everything before our markers)
            let msg: String = out
                .lines()
                .filter(|l| !l.starts_with("MLF_"))
                .collect::<Vec<_>>()
                .join("\n");
            return Err(format!("sbatch failed: {}", msg.trim()));
        }
        eprintln!("[mlforge] slurm run {run_id} submitted on {alias}");
    } else {
        // Detached launch. setsid → own session (immune to the ssh-channel HUP
        // and app close); nohup → belt-and-suspenders; stdio to files; stdin
        // /dev/null. The remote command shell has job control off, so setsid
        // execs in place and $! is the python pid.
        let launch = format!(
            "cd {dir_q} && nohup setsid {python_q} -u train.py > stdout.log 2> stderr.log < /dev/null & echo $! > pid"
        );
        ssh_exec(&alias, &launch, None).await?;
        eprintln!("[mlforge] remote training run {run_id} launched on {alias} ({python})");
    }
    Ok(())
}

/// Emit a train.sbatch from the SLURM config (a serde_json object). Values are
/// written into a file (not pasted into our shell command), and run on the
/// user's own cluster — so simple fields are lightly sanitised and the module
/// list / pre_run_script are free-form by design (the plan calls for it).
fn build_sbatch(run_id: &str, python_q: &str, slurm: Option<&Value>) -> String {
    let s = |k: &str| slurm.and_then(|v| v.get(k)).and_then(|x| x.as_str()).unwrap_or("").trim().to_string();
    let n = |k: &str| slurm.and_then(|v| v.get(k)).and_then(|x| x.as_u64());

    // job name: a short, tame slug from the run id
    let job: String = run_id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '-' })
        .take(64)
        .collect();

    let mut out = String::from("#!/bin/bash\n");
    out.push_str(&format!("#SBATCH --job-name=mlforge-{job}\n"));
    let partition = s("partition");
    if !partition.is_empty() {
        out.push_str(&format!("#SBATCH --partition={partition}\n"));
    }
    let time = s("time");
    out.push_str(&format!("#SBATCH --time={}\n", if time.is_empty() { "04:00:00".into() } else { time }));
    let mem = s("mem");
    if !mem.is_empty() {
        out.push_str(&format!("#SBATCH --mem={mem}\n"));
    }
    let cpus = n("cpus_per_task").unwrap_or(8);
    out.push_str(&format!("#SBATCH --cpus-per-task={cpus}\n"));
    let gres = s("gres");
    if !gres.is_empty() {
        out.push_str(&format!("#SBATCH --gres={gres}\n"));
    }
    let account = s("account");
    if !account.is_empty() {
        out.push_str(&format!("#SBATCH --account={account}\n"));
    }
    let qos = s("qos");
    if !qos.is_empty() {
        out.push_str(&format!("#SBATCH --qos={qos}\n"));
    }
    out.push_str("#SBATCH --output=slurm-%j.out\n");
    out.push_str("#SBATCH --error=slurm-%j.err\n\n");

    if let Some(mods) = slurm.and_then(|v| v.get("modules")).and_then(|x| x.as_array()) {
        for m in mods {
            if let Some(name) = m.as_str() {
                let name = name.trim();
                if !name.is_empty() {
                    out.push_str(&format!("module load {name}\n"));
                }
            }
        }
    }
    let pre = s("pre_run_script");
    if !pre.is_empty() {
        out.push_str(&pre);
        out.push('\n');
    }
    out.push('\n');
    // train.py writes events.jsonl/status itself; -u so SLURM's buffered stdout
    // isn't the only signal. We still cd via SLURM_SUBMIT_DIR for safety.
    out.push_str("cd \"$SLURM_SUBMIT_DIR\"\n");
    out.push_str(&format!("{python_q} -u train.py\n"));
    out
}

struct ParsedRun {
    id: String,
    alive: bool,
    status: String,
    run_json: String,
    metrics: String,
}

#[tauri::command]
pub async fn ssh_list_training_runs(
    alias: String,
    root: String,
) -> Result<Vec<training::RunSummary>, String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    let runs_q = shell_quote_path(&remote_runs_dir(&root));
    // One round-trip: for each run dir emit liveness + status + run.json +
    // metrics.json between line-delimited markers.
    let cmd = format!(
        "RUNS={runs_q}; \
         if [ -d \"$RUNS\" ]; then \
           for d in \"$RUNS\"/*/; do \
             [ -d \"$d\" ] || continue; \
             name=$(basename \"$d\"); \
             case \"$name\" in .*) continue;; esac; \
             echo \"MLF_RUN $name\"; \
             pid=$(cat \"$d/pid\" 2>/dev/null); \
             case \"$pid\" in \
               slurm:*) jid=${{pid#slurm:}}; if squeue -j \"$jid\" -h -o '%T' 2>/dev/null | grep -q .; then echo 'MLF_ALIVE 1'; else echo 'MLF_ALIVE 0'; fi ;; \
               '') echo 'MLF_ALIVE 0' ;; \
               *) if kill -0 \"$pid\" 2>/dev/null; then echo 'MLF_ALIVE 1'; else echo 'MLF_ALIVE 0'; fi ;; \
             esac; \
             echo MLF_STATUS_BEGIN; cat \"$d/status\" 2>/dev/null; echo; echo MLF_STATUS_END; \
             echo MLF_RUNJSON_BEGIN; cat \"$d/run.json\" 2>/dev/null; echo; echo MLF_RUNJSON_END; \
             echo MLF_METRICS_BEGIN; cat \"$d/metrics.json\" 2>/dev/null; echo; echo MLF_METRICS_END; \
           done; \
         fi"
    );
    let out = ssh_exec(&alias, &cmd, None).await?;

    let mut parsed: Vec<ParsedRun> = Vec::new();
    let mut cur: Option<ParsedRun> = None;
    let mut section = "";
    for line in out.lines() {
        if let Some(name) = line.strip_prefix("MLF_RUN ") {
            if let Some(r) = cur.take() {
                parsed.push(r);
            }
            cur = Some(ParsedRun {
                id: name.to_string(),
                alive: false,
                status: String::new(),
                run_json: String::new(),
                metrics: String::new(),
            });
            section = "";
            continue;
        }
        if let Some(a) = line.strip_prefix("MLF_ALIVE ") {
            if let Some(r) = cur.as_mut() {
                r.alive = a.trim() == "1";
            }
            continue;
        }
        match line {
            "MLF_STATUS_BEGIN" => { section = "status"; continue; }
            "MLF_RUNJSON_BEGIN" => { section = "runjson"; continue; }
            "MLF_METRICS_BEGIN" => { section = "metrics"; continue; }
            "MLF_STATUS_END" | "MLF_RUNJSON_END" | "MLF_METRICS_END" => { section = ""; continue; }
            _ => {}
        }
        if let Some(r) = cur.as_mut() {
            match section {
                "status" => { r.status.push_str(line); r.status.push('\n'); }
                "runjson" => { r.run_json.push_str(line); r.run_json.push('\n'); }
                "metrics" => { r.metrics.push_str(line); r.metrics.push('\n'); }
                _ => {}
            }
        }
    }
    if let Some(r) = cur.take() {
        parsed.push(r);
    }

    // run_id is timestamp-prefixed → lexical-desc == newest-first.
    parsed.sort_by(|a, b| b.id.cmp(&a.id));
    Ok(parsed
        .into_iter()
        .map(|r| {
            training::RunSummary::from_parts(
                &r.id,
                r.run_json.trim(),
                r.metrics.trim(),
                r.status.trim(),
                r.alive,
            )
        })
        .collect())
}

#[tauri::command]
pub async fn ssh_training_run_status(
    alias: String,
    root: String,
    run_id: String,
) -> Result<training::RunStatus, String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    training::validate_run_id(&run_id)?;
    let dir_q = shell_quote_path(&remote_run_dir(&root, &run_id));
    let cmd = format!(
        "d={dir_q}; pid=$(cat \"$d/pid\" 2>/dev/null); \
         case \"$pid\" in \
           slurm:*) jid=${{pid#slurm:}}; if squeue -j \"$jid\" -h -o '%T' 2>/dev/null | grep -q .; then echo 'MLF_ALIVE 1'; else echo 'MLF_ALIVE 0'; fi ;; \
           '') echo 'MLF_ALIVE 0' ;; \
           *) if kill -0 \"$pid\" 2>/dev/null; then echo 'MLF_ALIVE 1'; else echo 'MLF_ALIVE 0'; fi ;; \
         esac; \
         echo \"MLF_PID $pid\"; \
         echo MLF_STATUS_BEGIN; cat \"$d/status\" 2>/dev/null; echo; echo MLF_STATUS_END"
    );
    let out = ssh_exec(&alias, &cmd, None).await?;
    let mut alive = false;
    let mut pid: Option<i32> = None;
    let mut status = String::new();
    let mut in_status = false;
    for line in out.lines() {
        if let Some(a) = line.strip_prefix("MLF_ALIVE ") {
            alive = a.trim() == "1";
        } else if let Some(p) = line.strip_prefix("MLF_PID ") {
            pid = p.trim().parse::<i32>().ok();
        } else if line == "MLF_STATUS_BEGIN" {
            in_status = true;
        } else if line == "MLF_STATUS_END" {
            in_status = false;
        } else if in_status {
            status.push_str(line);
            status.push('\n');
        }
    }
    Ok(training::RunStatus::new(status.trim(), alive, pid))
}

#[tauri::command]
pub async fn ssh_read_training_run_file(
    alias: String,
    root: String,
    run_id: String,
    name: String,
) -> Result<String, String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    training::validate_run_id(&run_id)?;
    if !training::READABLE.contains(&name.as_str()) {
        return Err(format!("file {name:?} is not readable from a run dir"));
    }
    let p_q = shell_quote_path(&format!("{}/{}", remote_run_dir(&root, &run_id), name));
    ssh_exec(&alias, &format!("if [ -f {p_q} ]; then cat {p_q}; fi"), None).await
}

#[tauri::command]
pub async fn ssh_stop_training_run(alias: String, root: String, run_id: String) -> Result<(), String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    training::validate_run_id(&run_id)?;
    let dir_q = shell_quote_path(&remote_run_dir(&root, &run_id));
    // Cooperative (status file, checked each epoch) + forceful (SIGTERM the whole
    // process group via negative pid — setsid made python the group leader).
    let cmd = format!(
        "d={dir_q}; if [ -d \"$d\" ]; then \
           printf 'cancelled\\n' > \"$d/status\"; \
           pid=$(cat \"$d/pid\" 2>/dev/null); \
           case \"$pid\" in \
             slurm:*) scancel \"${{pid#slurm:}}\" 2>/dev/null ;; \
             '') : ;; \
             *) kill -TERM -\"$pid\" 2>/dev/null; kill -TERM \"$pid\" 2>/dev/null ;; \
           esac; \
         fi"
    );
    ssh_exec(&alias, &cmd, None).await.map(|_| ())
}

#[tauri::command]
pub async fn ssh_delete_training_run(alias: String, root: String, run_id: String) -> Result<(), String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    training::validate_run_id(&run_id)?;
    let dir_q = shell_quote_path(&remote_run_dir(&root, &run_id));
    let cmd = format!(
        "d={dir_q}; if [ -d \"$d\" ]; then \
           pid=$(cat \"$d/pid\" 2>/dev/null); alive=0; \
           case \"$pid\" in \
             slurm:*) jid=${{pid#slurm:}}; if squeue -j \"$jid\" -h -o '%T' 2>/dev/null | grep -q .; then alive=1; fi ;; \
             '') : ;; \
             *) if kill -0 \"$pid\" 2>/dev/null; then alive=1; fi ;; \
           esac; \
           if [ \"$alive\" = 1 ]; then echo MLF_ALIVE; else rm -rf -- \"$d\"; echo MLF_DELETED; fi; \
         else echo MLF_DELETED; fi"
    );
    let out = ssh_exec(&alias, &cmd, None).await?;
    if out.contains("MLF_ALIVE") {
        return Err("run is still alive — stop it before deleting".into());
    }
    Ok(())
}

#[derive(Serialize)]
pub struct RemoteTrainingCapabilities {
    has_slurm: bool,
    has_gpu: bool,
    partitions: Vec<String>,
    gpu_names: Vec<String>,
}

/// Probe what the remote host offers for training (Phase 17). Cheap, one
/// round-trip; the UI calls it once per remote connection to decide whether to
/// offer the SLURM backend and to populate the partition dropdown.
#[tauri::command]
pub async fn ssh_remote_training_capabilities(
    alias: String,
    _root: String,
) -> Result<RemoteTrainingCapabilities, String> {
    validate_alias(&alias)?;
    let cmd = "if command -v sbatch >/dev/null 2>&1; then echo MLF_HAS_SLURM; fi; \
               if command -v nvidia-smi >/dev/null 2>&1; then echo MLF_HAS_GPU; \
                 echo MLF_GPU_BEGIN; nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null; echo MLF_GPU_END; fi; \
               if command -v sinfo >/dev/null 2>&1; then \
                 echo MLF_PART_BEGIN; sinfo -h -o '%P' 2>/dev/null | sort -u; echo MLF_PART_END; fi";
    let out = ssh_exec(&alias, cmd, None).await?;

    let mut has_slurm = false;
    let mut has_gpu = false;
    let mut partitions: Vec<String> = Vec::new();
    let mut gpu_names: Vec<String> = Vec::new();
    let mut section = "";
    for line in out.lines() {
        match line {
            "MLF_HAS_SLURM" => has_slurm = true,
            "MLF_HAS_GPU" => has_gpu = true,
            "MLF_PART_BEGIN" => section = "part",
            "MLF_GPU_BEGIN" => section = "gpu",
            "MLF_PART_END" | "MLF_GPU_END" => section = "",
            _ => {
                let v = line.trim();
                if v.is_empty() {
                    continue;
                }
                match section {
                    // sinfo marks the default partition with a trailing '*'
                    "part" => partitions.push(v.trim_end_matches('*').to_string()),
                    "gpu" => gpu_names.push(v.to_string()),
                    _ => {}
                }
            }
        }
    }
    partitions.dedup();
    Ok(RemoteTrainingCapabilities {
        has_slurm,
        has_gpu,
        partitions,
        gpu_names,
    })
}
