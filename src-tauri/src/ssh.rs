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
use tauri::State;

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

fn ssh_exec(alias: &str, remote_cmd: &str, stdin_data: Option<&[u8]>) -> Result<String, String> {
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
pub fn ssh_test_connection(alias: String) -> Result<SshTestResult, String> {
    validate_alias(&alias)?;
    let out = ssh_exec(
        &alias,
        "echo MLFORGE_OK && uname -srm && echo \"HOME=$HOME\"",
        None,
    )?;
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
pub fn ssh_load_project(
    state: State<RemoteWorkspaceState>,
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
    let out = ssh_exec(&alias, &cmd, None)?;
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
pub fn ssh_init_project(
    state: State<RemoteWorkspaceState>,
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
    ssh_exec(&alias, &cmd, Some(pretty.as_bytes()))?;
    *state.current.lock().map_err(|e| e.to_string())? =
        Some(RemoteWorkspace { alias: alias.clone(), root: root.clone() });
    Ok(meta)
}

#[tauri::command]
pub fn ssh_update_project_meta(
    alias: String,
    root: String,
    patch: serde_json::Value,
) -> Result<serde_json::Value, String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    let proj_q = shell_quote_path(&join_remote(&root, PROJECT_FILE));
    let body = ssh_exec(&alias, &format!("cat {proj_q}"), None)?;
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
    ssh_exec(&alias, &format!("cat > {proj_q}"), Some(pretty.as_bytes()))?;
    Ok(meta)
}

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
pub fn ssh_walk(alias: String, root: String) -> Result<Vec<RemoteFsEntry>, String> {
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
    let out = ssh_exec(&alias, &cmd, None)?;
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
pub fn ssh_read_file(alias: String, root: String, relpath: String) -> Result<String, String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    validate_relpath(&relpath)?;
    let p_q = shell_quote_path(&join_remote(&root, &relpath));
    ssh_exec(&alias, &format!("cat {p_q}"), None)
}

#[tauri::command]
pub fn ssh_write_file(
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
    .map(|_| ())
}

#[tauri::command]
pub fn ssh_delete_path(alias: String, root: String, relpath: String) -> Result<(), String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    validate_relpath(&relpath)?;
    if relpath.is_empty() || relpath == "." {
        return Err("refusing to delete workspace root".into());
    }
    let p_q = shell_quote_path(&join_remote(&root, &relpath));
    ssh_exec(&alias, &format!("rm -rf -- {p_q}"), None).map(|_| ())
}

#[tauri::command]
pub fn ssh_mkdir(alias: String, root: String, relpath: String) -> Result<(), String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    validate_relpath(&relpath)?;
    let p_q = shell_quote_path(&join_remote(&root, &relpath));
    ssh_exec(&alias, &format!("mkdir -p {p_q}"), None).map(|_| ())
}

#[tauri::command]
pub fn ssh_rename(
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
pub fn ssh_list_notes(alias: String, root: String) -> Result<Vec<RemoteNoteEntry>, String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    let dir_q = shell_quote_path(&join_remote(&root, "notes"));
    let cmd = format!(
        "mkdir -p {dir_q} && find {dir_q} -mindepth 1 -maxdepth 1 -type f \
           \\( -name '*.md' -o -name '*.txt' \\) \
           -printf '%f\\t%s\\t%TY-%Tm-%TdT%TH:%TM:%TSZ\\n' 2>/dev/null"
    );
    let out = ssh_exec(&alias, &cmd, None)?;
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
pub fn ssh_read_note(alias: String, root: String, name: String) -> Result<String, String> {
    validate_alias(&alias)?;
    validate_remote_root(&root)?;
    validate_plain_filename(&name, "note")?;
    let p_q = shell_quote_path(&format!("{}/notes/{}", root.trim_end_matches('/'), name));
    ssh_exec(&alias, &format!("cat {p_q}"), None)
}

#[tauri::command]
pub fn ssh_write_note(
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
    .map(|_| ())
}

#[tauri::command]
pub fn ssh_append_note(
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
    .map(|_| ())
}

// ─── experiments ──────────────────────────────────────────────────────────

#[tauri::command]
pub fn ssh_append_experiment(
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
    .map(|_| ())
}

#[tauri::command]
pub fn ssh_read_experiment(
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
    ssh_exec(&alias, &format!("if [ -f {p_q} ]; then cat {p_q}; fi"), None)
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
pub fn ssh_list_datasets(
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
    let out = ssh_exec(&alias, &cmd, None)?;
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
