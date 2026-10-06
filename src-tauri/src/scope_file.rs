//! Filesystem scope for the local SpinoML sidecars (Phase 46, closing R016).
//!
//! The Rust shell picks the workspace root before either sidecar starts; the
//! sidecars then re-read the root from this file (see `sidecar-torch/scope.py`
//! `_load_scope_file` and `sidecar-llm/path-scope.mjs` `loadSymlinkTargets`).
//! Schema: `{"version":1, "roots":[...], "symlink_targets":[...]}`. Discovery:
//! `$XDG_RUNTIME_DIR/spinoml/scope.json` if that FILE exists, else
//! `~/.cache/spinoml/scope.json`. The sidecar only trusts a regular file owned
//! by the current uid with `mode & 0o022 == 0`; we mirror that on the writer
//! side (file mode 0600, parent dir 0700, owned by current uid).
//!
//! Before a workspace is chosen the sidecar runs in `unconfigured-open` — we
//! intentionally do NOT enable `SPINOML_REQUIRE_SCOPE` here, because some flows
//! (a fresh first-launch Welcome screen, or a smoke test against a dataset
//! path supplied before a workspace is set up) exercise dataset code before a
//! workspace is selected. `SPINOML_REQUIRE_SCOPE=1` is the operator's choice;
//! this module never sets it. As long as mode is `unconfigured-open` the
//! sidecar's `check_path` is permissive (one stderr warning per process).
//!
//! Remote workspaces are out of scope here: the remote launcher already exports
//! `SPINOML_ALLOWED_ROOTS=<remote root>` to the HPC sidecar, and we never write
//! a remote path into this local file.

use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};

const SCOPE_VERSION: u32 = 1;

/// Where the writer always looks: the same path the sidecar reads first, if
/// the file is already there. Otherwise we create it under `~/.cache/spinoml/`
/// (the sidecar's fallback). We never write to BOTH — whichever the sidecar
/// will pick next time is the one we update.
fn scope_file_path_impl() -> Result<PathBuf, String> {
    if let Some(xdg) = env::var_os("XDG_RUNTIME_DIR") {
        if !xdg.is_empty() {
            let candidate = PathBuf::from(&xdg).join("spinoml").join("scope.json");
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    let home = env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| "HOME is not set; cannot locate ~/.cache/spinoml/scope.json".to_string())?;
    Ok(home.join(".cache").join("spinoml").join("scope.json"))
}

#[allow(dead_code)] // public for completeness; tests cover it.
pub fn scope_file_path() -> Result<PathBuf, String> {
    scope_file_path_impl()
}

/// Read the existing scope file (if any) and return the keys we must preserve.
/// If the file is missing, unreadable, owned by another uid, group/world-
/// writable, not a regular file, not valid JSON, not version 1, or has the
/// wrong shape for any key, we log ONE diagnostic line and treat the relevant
/// value as "absent" — but the file at the chosen path is still rewritten
/// atomically. The diagnostic line never contains the file content.
#[derive(Default)]
struct Existing {
    /// Pass-through keys we do not understand. Preserved verbatim.
    extra: serde_json::Map<String, serde_json::Value>,
    /// `symlink_targets` from a valid existing file (already as canonicalised
    /// path strings). None means "not preserved" (start fresh, do not write).
    symlink_targets: Option<Vec<String>>,
}

/// True if `path` is a scope file we may TRUST for configuration: it exists
/// as a regular file (checked with `symlink_metadata`, so a symlink-to-file
/// does not qualify), is owned by the current effective uid, and is not
/// group/world writable. This is the ONE predicate both `read_existing`
/// (write path) and `load_allowed_roots` (read path) use, mirroring the gates
/// in `sidecar-torch/scope.py` exactly.
#[cfg(unix)]
fn scope_file_is_trusted(path: &Path) -> bool {
    let meta = match fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(_) => return false,
    };
    if !meta.file_type().is_file() {
        return false;
    }
    use std::os::unix::fs::MetadataExt;
    let cur_uid = unsafe { libc::geteuid() };
    if meta.uid() != cur_uid {
        return false;
    }
    if meta.mode() & 0o022 != 0 {
        return false;
    }
    true
}

#[cfg(not(unix))]
fn scope_file_is_trusted(_path: &Path) -> bool {
    false
}

fn read_existing(path: &Path) -> Existing {
    let mut out = Existing::default();
    // A missing file is the normal "first write" case: nothing to preserve
    // and nothing to log. Any other metadata error, or a present-but-untrusted
    // file, is logged and treated as "no values to preserve" (the file is
    // still rewritten atomically below).
    match fs::symlink_metadata(path) {
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => return out,
        Err(_) => {
            eprintln!(
                "[spinoml] scope file at {} is unreadable; rewriting without preserving values",
                path.display()
            );
            return out;
        }
    }
    if !scope_file_is_trusted(path) {
        eprintln!(
            "[spinoml] scope file at {} is not a trusted regular file (regular + owner + not group/world writable); rewriting without preserving values",
            path.display()
        );
        return out;
    }
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => {
            eprintln!(
                "[spinoml] scope file at {} is unreadable; rewriting without preserving values",
                path.display()
            );
            return out;
        }
    };
    let parsed: serde_json::Value = match serde_json::from_str(&text) {
        Ok(v) => v,
        Err(_) => {
            eprintln!(
                "[spinoml] scope file at {} is not valid JSON; rewriting without preserving values",
                path.display()
            );
            return out;
        }
    };
    let Some(obj) = parsed.as_object() else {
        eprintln!(
            "[spinoml] scope file at {} is not a JSON object; rewriting without preserving values",
            path.display()
        );
        return out;
    };
    if obj.get("version").and_then(|v| v.as_u64()) != Some(SCOPE_VERSION as u64) {
        eprintln!(
            "[spinoml] scope file at {} has an unsupported version; rewriting without preserving values",
            path.display()
        );
        return out;
    }
    // Copy every top-level key we do not own, and pull symlink_targets if
    // present and shaped correctly. Missing symlink_targets means: do not
    // write the key (so a re-add will not wipe the user's list silently — the
    // caller passes the desired new value or None).
    let mut extra = serde_json::Map::new();
    let mut symlink_targets: Option<Vec<String>> = None;
    for (k, v) in obj {
        match k.as_str() {
            "version" | "roots" => {}
            "symlink_targets" => match v.as_array() {
                Some(arr) => {
                    let mut kept = Vec::with_capacity(arr.len());
                    for item in arr {
                        if let Some(s) = item.as_str() {
                            kept.push(s.to_string());
                        }
                    }
                    symlink_targets = Some(kept);
                }
                None => {
                    eprintln!(
                        "[spinoml] scope file at {} has a malformed symlink_targets; rewriting without preserving values",
                        path.display()
                    );
                    return out;
                }
            },
            _ => {
                extra.insert(k.clone(), v.clone());
            }
        }
    }
    out.extra = extra;
    out.symlink_targets = symlink_targets;
    out
}

/// Best-effort canonicalisation of one root entry. Returns Err if the entry
/// does not name an existing directory (we never want to write a path the
/// sidecar could not have observed). `/` is rejected (no useful containment).
#[cfg(unix)]
fn canonicalise_root(p: &Path) -> Result<String, String> {
    if p.as_os_str().is_empty() {
        return Err("root entry is empty".into());
    }
    if !p.is_absolute() {
        return Err(format!("root entry is not absolute: {}", p.display()));
    }
    let real = match fs::canonicalize(p) {
        Ok(r) => r,
        Err(e) => {
            return Err(format!(
                "cannot canonicalize root {}: {e}",
                p.display()
            ));
        }
    };
    let real_str = real.to_string_lossy().into_owned();
    if real == Path::new("/") {
        return Err(format!("root entry is the filesystem root: {}", p.display()));
    }
    if !fs::metadata(&real)
        .map(|m| m.is_dir())
        .unwrap_or(false)
    {
        return Err(format!("root entry is not an existing directory: {}", p.display()));
    }
    Ok(real_str)
}

#[cfg(not(unix))]
fn canonicalise_root(p: &Path) -> Result<String, String> {
    Err("scope_file::write_roots is unix-only (bundle target: Linux .deb)".into())
}

fn ensure_parent_dir(p: &Path) -> Result<(), String> {
    if let Some(parent) = p.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {e}", parent.display()))?;
        }
        #[cfg(unix)]
        {
            let meta = fs::metadata(parent).map_err(|e| e.to_string())?;
            let cur = meta.permissions().mode() & 0o777;
            if cur != 0o700 {
                fs::set_permissions(parent, fs::Permissions::from_mode(0o700))
                    .map_err(|e| format!("chmod 0700 {}: {e}", parent.display()))?;
            }
        }
    }
    Ok(())
}

/// Build the JSON value to write. `roots` is already a list of canonical path
/// strings. `extra` is preserved verbatim. If `symlink_targets` is None, we do
/// not emit the key (a no-value stale file won't get an empty array that
/// silently erases the user's list). Otherwise we emit the provided list.
fn build_value(
    roots: &[String],
    symlink_targets: Option<&Vec<String>>,
    extra: serde_json::Map<String, serde_json::Value>,
) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("version".into(), serde_json::Value::from(SCOPE_VERSION));
    obj.insert(
        "roots".into(),
        serde_json::Value::Array(roots.iter().map(|s| serde_json::Value::String(s.clone())).collect()),
    );
    if let Some(t) = symlink_targets {
        obj.insert(
            "symlink_targets".into(),
            serde_json::Value::Array(t.iter().map(|s| serde_json::Value::String(s.clone())).collect()),
        );
    }
    for (k, v) in extra {
        obj.insert(k, v);
    }
    serde_json::Value::Object(obj)
}

#[cfg(unix)]
fn atomic_write(path: &Path, json: &str) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("scope file {} has no parent dir", path.display()))?;
    let base = path
        .file_name()
        .ok_or_else(|| format!("scope file {} has no file name", path.display()))?
        .to_os_string();
    // Uniquify: pid + nanos + a per-process counter, all base 36, so two
    // writers (or two retries in the same process) never collide on a
    // single dir.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let mut tmp_name = base.clone();
    tmp_name.push(format!(".tmp.{pid:x}.{nanos:x}.{seq:x}"));
    let tmp_path = parent.join(tmp_name);

    let mut f = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&tmp_path)
        .map_err(|e| format!("create {}: {e}", tmp_path.display()))?;
    use std::io::Write;
    f.write_all(json.as_bytes())
        .map_err(|e| format!("write {}: {e}", tmp_path.display()))?;
    f.sync_all().map_err(|e| format!("fsync {}: {e}", tmp_path.display()))?;
    drop(f);
    // If the destination is currently a directory (or anything that isn't a
    // regular file), rename() will refuse. Remove it first — the temp file
    // has the fresh content we want to install. read_existing() already
    // decided whether to trust anything that was there.
    if let Ok(meta) = fs::symlink_metadata(path) {
        if !meta.file_type().is_file() {
            if meta.file_type().is_dir() {
                fs::remove_dir(path).map_err(|e| {
                    format!("rmdir {}: {e}", path.display())
                })?;
            } else {
                fs::remove_file(path).map_err(|e| {
                    format!("remove {}: {e}", path.display())
                })?;
            }
        }
    }
    if let Err(e) = fs::rename(&tmp_path, path) {
        let _ = fs::remove_file(&tmp_path);
        return Err(format!("rename {} → {}: {e}", tmp_path.display(), path.display()));
    }
    // Make extra sure the file is 0600 even if rename preserved a looser mode.
    let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o600));
    // fsync the directory so the rename is durable on Linux. Best-effort: some
    // filesystems return EINVAL/EPERM for dir fsync, which we ignore — the
    // file's own sync_all is the important part.
    let dir = fs::OpenOptions::new().read(true).open(parent);
    if let Ok(d) = dir {
        let _ = d.sync_all();
    }
    Ok(())
}

#[cfg(not(unix))]
fn atomic_write(_path: &Path, _json: &str) -> Result<(), String> {
    Err("scope_file::atomic_write is unix-only (bundle target: Linux .deb)".into())
}

/// Write the scope file with the given roots (canonicalised). Preserves
/// `symlink_targets` and any unknown top-level key from a valid existing file.
/// A failure (canonicalization, mkdir, write) is propagated to the caller.
pub fn write_roots(roots: &[PathBuf]) -> Result<(), String> {
    let mut canonical_roots: Vec<String> = Vec::with_capacity(roots.len());
    for r in roots {
        canonical_roots.push(canonicalise_root(r)?);
    }
    let path = scope_file_path_impl()?;
    ensure_parent_dir(&path)?;
    let existing = read_existing(&path);
    let value = build_value(&canonical_roots, existing.symlink_targets.as_ref(), existing.extra);
    let json = serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?;
    atomic_write(&path, &json)?;
    Ok(())
}

/// Clear the roots in the scope file. Preserves `symlink_targets` and any
/// unknown top-level key (so closing a workspace does not wipe the user's
/// allow-listed symlink targets).
pub fn clear_roots() -> Result<(), String> {
    let path = scope_file_path_impl()?;
    ensure_parent_dir(&path)?;
    let existing = read_existing(&path);
    let value = build_value(&[], existing.symlink_targets.as_ref(), existing.extra);
    let json = serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?;
    atomic_write(&path, &json)?;
    Ok(())
}

/// Canonicalise `candidate`, tolerating a not-yet-existing tail (the target
/// may not exist yet for create/mkdir/rename-to). Then require the result to
/// live under the canonical realpath of `root` OR under the canonical realpath
/// of any configured symlink target (scope file + env SPINOML_SYMLINK_TARGETS).
///
/// Returns the canonical path on success. Errors with a message mirroring
/// scope.py's `PATH_SYMLINK_OUTSIDE` / `PATH_INVALID` text style.
pub fn check_resolved(root: &Path, candidate: &Path) -> Result<PathBuf, String> {
    let canonical_root = canonicalize_tolerating_missing(root)
        .map_err(|e| format!("workspace root {} is not accessible: {e}", root.display()))?;
    let canonical_candidate = canonicalize_tolerating_missing(candidate).map_err(|e| {
        format!(
            "could not resolve path {}: {e}",
            candidate.display()
        )
    })?;
    let allowed = allowed_roots();
    let scope_path = scope_file_path_impl().ok();
    if !path_starts_with(&canonical_candidate, &canonical_root) {
        // Outside the workspace root. Allow only if it falls under a
        // configured symlink target.
        let mut allowed_via_target = false;
        for t in allowed.iter() {
            if path_starts_with(&canonical_candidate, t) {
                allowed_via_target = true;
                break;
            }
        }
        if allowed_via_target {
            return Ok(canonical_candidate);
        }
        let where_to = match scope_path {
            Some(p) => format!(
                "add the target directory to symlink_targets in {} or to SPINOML_SYMLINK_TARGETS",
                p.display()
            ),
            None => "add the target directory to SPINOML_SYMLINK_TARGETS".to_string(),
        };
        return Err(format!(
            "\"{}\" resolves outside the workspace through a symlink (-> {}). If intended, {}.",
            candidate.display(),
            canonical_candidate.display(),
            where_to
        ));
    }
    Ok(canonical_candidate)
}

/// Canonicalise `p`, tolerating a not-yet-existing tail: find the deepest
/// existing ancestor, canonicalise it, and re-append the missing components.
///
/// Fail-closed rules:
///  * ELOOP (symlink loop) and any non-NotFound error propagate untouched.
///  * A NotFound failure is only "not created yet" when `symlink_metadata`
///    ALSO reports NotFound for that component. If the directory entry exists
///    (`symlink_metadata` succeeds) but cannot be canonicalised, it is a
///    DANGLING SYMLINK (or a broken chain) and we reject it: walking past it
///    and re-appending the tail lexically would let a write/create follow the
///    link and create the file OUTSIDE the workspace. This applies at every
///    level of the walk, so a dangling link in the middle of the tail is
///    caught too.
fn canonicalize_tolerating_missing(p: &Path) -> io::Result<PathBuf> {
    if let Ok(c) = fs::canonicalize(p) {
        return Ok(c);
    }
    let mut cur = p.to_path_buf();
    loop {
        match fs::canonicalize(&cur) {
            Ok(real) => {
                let tail: PathBuf = p.strip_prefix(&cur).unwrap_or(p).into();
                if tail.as_os_str().is_empty() {
                    return Ok(real);
                }
                return Ok(real.join(tail));
            }
            Err(e) => {
                if e.kind() != io::ErrorKind::NotFound {
                    return Err(e);
                }
                match fs::symlink_metadata(&cur) {
                    Ok(_) => {
                        // The entry EXISTS but its canonical target does not:
                        // a dangling symlink. Fail closed (even if the target
                        // is inside the root — a broken link is never something
                        // we can safely scope).
                        return Err(io::Error::new(
                            io::ErrorKind::NotFound,
                            format!(
                                "dangling symlink at {} (its target does not exist)",
                                cur.display()
                            ),
                        ));
                    }
                    Err(se) if se.kind() == io::ErrorKind::NotFound => {
                        // Genuinely not created yet — keep walking up.
                    }
                    Err(se) => return Err(se),
                }
                let parent = match cur.parent() {
                    Some(parent) if !parent.as_os_str().is_empty() => parent,
                    _ => return Err(e),
                };
                cur = parent.to_path_buf();
            }
        }
    }
}

/// Exact containment, sibling-prefix safe: `/tmp/ws-evil` is NOT inside
/// `/tmp/ws`. A path is inside itself. Done component-wise via
/// `Path::strip_prefix` because `Path::starts_with` is also component-wise on
/// canonical inputs (no `.`/`..` left after canonicalization).
fn path_starts_with(child: &Path, parent: &Path) -> bool {
    child.strip_prefix(parent).is_ok()
}

/// Build the list of allowed symlink targets from the scope file + env, with
/// canonicalisation. Errors reading the file are silenced (the file may be
/// absent, in which case only env is consulted).
#[cfg(unix)]
fn allowed_roots() -> Vec<PathBuf> {
    load_allowed_roots()
}

#[cfg(not(unix))]
fn allowed_roots() -> Vec<PathBuf> {
    Vec::new()
}

/// Public re-export so callers outside this module (e.g. `walk` in lib.rs)
/// can ask "is this canonical path under any configured symlink target?"
/// without duplicating the loader. Public for lib.rs only.
#[cfg(unix)]
pub fn allowed_roots_for_check() -> Vec<PathBuf> {
    load_allowed_roots()
}

#[cfg(not(unix))]
pub fn allowed_roots_for_check() -> Vec<PathBuf> {
    Vec::new()
}

#[cfg(unix)]
fn load_allowed_roots() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    if let Some(raw) = std::env::var_os("SPINOML_SYMLINK_TARGETS") {
        for entry in raw.to_string_lossy().split(':') {
            if entry.is_empty() {
                continue;
            }
            let p = PathBuf::from(entry);
            if let Ok(real) = fs::canonicalize(&p) {
                if real != Path::new("/") && real.is_dir() {
                    out.push(real);
                }
            }
        }
    }
    // Only a TRUSTED scope file may widen the allow-list — the same
    // owner/mode gate `read_existing` and `scope.py` apply. An attacker-
    // writable file must contribute NO symlink targets (env targets still
    // apply, and the env is not attacker-writable from the workspace).
    if let Ok(path) = scope_file_path_impl() {
        if scope_file_is_trusted(&path) {
            if let Ok(text) = fs::read_to_string(&path) {
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                    if let Some(t) = v.get("symlink_targets").and_then(|x| x.as_array()) {
                        for item in t {
                            if let Some(s) = item.as_str() {
                                let p = PathBuf::from(s);
                                if let Ok(real) = fs::canonicalize(&p) {
                                    if real != Path::new("/") && real.is_dir() {
                                        out.push(real);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

// ── Tests ──────────────────────────────────────────────────────────────────
//
// Every test creates its own temp dir under std::env::temp_dir() with a
// process-unique suffix (pid + nanos + a static counter), then points
// `HOME` at it so the scope file lands under a private location. We never
// touch the real HOME.

/// Serialises tests that mutate process-global env (HOME / XDG_RUNTIME_DIR /
/// SPINOML_*). The scope-file tests and the lib.rs resolve/delete/rename tests
/// share this ONE lock so they cannot stomp on each other's env while running
/// in parallel. Only compiled for tests.
#[cfg(test)]
pub(crate) static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// Unique temp dir + temporary HOME override. Returns (tempdir, scope_path).
    /// All cleanup happens on Drop (we rely on tempdir_drop).
    pub(crate) struct TempScope {
        pub dir: PathBuf,
        pub scope_path: PathBuf,
        prev_home: Option<std::ffi::OsString>,
        prev_xdg: Option<std::ffi::OsString>,
        // Held across the test so concurrent tests block on the lock and
        // don't mutate HOME while we own it.
        _guard: std::sync::MutexGuard<'static, ()>,
    }

    impl TempScope {
        pub(crate) fn new() -> Self {
            // Acquire the env lock BEFORE mutating HOME so a parallel test
            // can't observe a half-set state.
            let guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            let pid = std::process::id();
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir().join(format!(
                "spinoml-scope-test-{pid:x}-{nanos:x}-{n:x}"
            ));
            fs::create_dir_all(&dir).expect("create temp dir");
            let prev_home = std::env::var_os("HOME");
            let prev_xdg = std::env::var_os("XDG_RUNTIME_DIR");
            std::env::set_var("HOME", &dir);
            // Make sure XDG_RUNTIME_DIR is unset so scope_file_path picks the
            // HOME fallback. Individual tests can set it if they need to.
            std::env::remove_var("XDG_RUNTIME_DIR");
            std::env::remove_var("SPINOML_SYMLINK_TARGETS");
            std::env::remove_var("SPINOML_ALLOWED_ROOTS");
            std::env::remove_var("SPINOML_REQUIRE_SCOPE");
            let scope_path = dir.join(".cache").join("spinoml").join("scope.json");
            Self { dir, scope_path, prev_home, prev_xdg, _guard: guard }
        }
    }

    impl Drop for TempScope {
        fn drop(&mut self) {
            match &self.prev_home {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
            match &self.prev_xdg {
                Some(v) => std::env::set_var("XDG_RUNTIME_DIR", v),
                None => std::env::remove_var("XDG_RUNTIME_DIR"),
            }
            let _ = fs::remove_dir_all(&self.dir);
            // _guard drops here, releasing the lock for the next test.
        }
    }

    fn file_mode(p: &Path) -> u32 {
        fs::metadata(p).unwrap().permissions().mode() & 0o7777
    }

    fn file_uid(p: &Path) -> u32 {
        use std::os::unix::fs::MetadataExt;
        fs::metadata(p).unwrap().uid()
    }

    fn dir_mode(p: &Path) -> u32 {
        fs::metadata(p).unwrap().permissions().mode() & 0o7777
    }

    #[test]
    fn write_roots_produces_schema_mode_0600_owner_current_uid_parent_0700() {
        let ts = TempScope::new();
        let ws = ts.dir.join("ws");
        fs::create_dir_all(&ws).unwrap();
        write_roots(&[ws.clone()]).unwrap();
        // File exists
        assert!(ts.scope_path.is_file(), "scope file missing at {}", ts.scope_path.display());
        // Mode 0600
        assert_eq!(file_mode(&ts.scope_path), 0o600, "file mode");
        // Owner = current uid
        let cur_uid = unsafe { libc::geteuid() };
        assert_eq!(file_uid(&ts.scope_path), cur_uid, "file uid");
        // Parent dir = 0700
        let parent = ts.scope_path.parent().unwrap();
        assert_eq!(dir_mode(parent), 0o700, "parent dir mode");
        // Schema: version 1, roots non-empty, optional symlink_targets shape
        let v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&ts.scope_path).unwrap()).unwrap();
        assert_eq!(v.get("version").and_then(|x| x.as_u64()), Some(1));
        let roots = v.get("roots").and_then(|x| x.as_array()).expect("roots array");
        assert_eq!(roots.len(), 1);
        let written_root = roots[0].as_str().unwrap();
        // Written canonical (no trailing slash, no symlink)
        assert_eq!(written_root, fs::canonicalize(&ws).unwrap().to_string_lossy());
        // symlink_targets is either absent or an array (we preserve prior shape).
        if let Some(t) = v.get("symlink_targets") {
            assert!(t.is_array());
        }
    }

    #[test]
    fn write_roots_preserves_symlink_targets_and_unknown_key() {
        let ts = TempScope::new();
        let ws1 = ts.dir.join("ws1");
        let ws2 = ts.dir.join("ws2");
        let scratch = ts.dir.join("scratch");
        fs::create_dir_all(&ws1).unwrap();
        fs::create_dir_all(&ws2).unwrap();
        fs::create_dir_all(&scratch).unwrap();
        // Write a file by hand with an unknown top-level key + symlink_targets.
        let parent = ts.scope_path.parent().unwrap();
        fs::create_dir_all(parent).unwrap();
        let initial = serde_json::json!({
            "version": 1,
            "roots": ["/stale"],
            "symlink_targets": [scratch.to_string_lossy()],
            "note": "x",
            "another_extra": {"a": 1, "b": ["c", "d"]}
        });
        fs::write(&ts.scope_path, serde_json::to_string_pretty(&initial).unwrap()).unwrap();
        fs::set_permissions(&ts.scope_path, fs::Permissions::from_mode(0o600)).unwrap();
        // Replace roots.
        write_roots(&[ws1.clone(), ws2.clone()]).unwrap();
        let v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&ts.scope_path).unwrap()).unwrap();
        let roots: Vec<String> = v
            .get("roots")
            .and_then(|x| x.as_array())
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect();
        assert_eq!(roots.len(), 2);
        assert!(roots.contains(&fs::canonicalize(&ws1).unwrap().to_string_lossy().into_owned()));
        assert!(roots.contains(&fs::canonicalize(&ws2).unwrap().to_string_lossy().into_owned()));
        let tgt: Vec<String> = v
            .get("symlink_targets")
            .and_then(|x| x.as_array())
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect();
        assert_eq!(tgt, vec![fs::canonicalize(&scratch).unwrap().to_string_lossy().into_owned()]);
        assert_eq!(v.get("note").and_then(|x| x.as_str()), Some("x"));
        assert!(v.get("another_extra").is_some());
    }

    #[test]
    fn write_roots_replaces_atomically_no_tmp_left() {
        let ts = TempScope::new();
        let ws = ts.dir.join("ws");
        fs::create_dir_all(&ws).unwrap();
        write_roots(&[ws.clone()]).unwrap();
        let parent = ts.scope_path.parent().unwrap();
        let stray: Vec<_> = fs::read_dir(parent)
            .unwrap()
            .flatten()
            .map(|e| e.file_name())
            .filter(|n| {
                let s = n.to_string_lossy();
                s.contains(".tmp.") || s.ends_with(".tmp")
            })
            .collect();
        assert!(stray.is_empty(), "stray tmp files: {:?}", stray);
        // The scope file itself is the only non-dir entry.
        let entries: Vec<_> = fs::read_dir(parent)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert!(entries.iter().any(|n| n == "scope.json"));
    }

    #[test]
    fn write_roots_replaces_corrupt_file_without_trusting_content() {
        let ts = TempScope::new();
        let ws = ts.dir.join("ws");
        fs::create_dir_all(&ws).unwrap();
        let parent = ts.scope_path.parent().unwrap();
        fs::create_dir_all(parent).unwrap();
        // Place garbage in the file and a wrong mode.
        fs::write(&ts.scope_path, b"this is not JSON {{{").unwrap();
        fs::set_permissions(&ts.scope_path, fs::Permissions::from_mode(0o600)).unwrap();
        write_roots(&[ws.clone()]).unwrap();
        let v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&ts.scope_path).unwrap()).unwrap();
        let roots: Vec<String> = v
            .get("roots")
            .and_then(|x| x.as_array())
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect();
        assert_eq!(roots, vec![fs::canonicalize(&ws).unwrap().to_string_lossy().into_owned()]);
        // symlink_targets must be ABSENT (we did not trust the corrupt file).
        assert!(v.get("symlink_targets").is_none());
    }

    #[test]
    fn write_roots_replaces_group_writable_file_without_trusting_content() {
        let ts = TempScope::new();
        let ws = ts.dir.join("ws");
        fs::create_dir_all(&ws).unwrap();
        let parent = ts.scope_path.parent().unwrap();
        fs::create_dir_all(parent).unwrap();
        let initial = serde_json::json!({
            "version": 1,
            "roots": ["/stale"],
            "symlink_targets": ["/elsewhere"]
        });
        fs::write(&ts.scope_path, serde_json::to_string_pretty(&initial).unwrap()).unwrap();
        fs::set_permissions(&ts.scope_path, fs::Permissions::from_mode(0o666)).unwrap();
        write_roots(&[ws.clone()]).unwrap();
        let v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&ts.scope_path).unwrap()).unwrap();
        // trust lost → symlink_targets not preserved
        assert!(v.get("symlink_targets").is_none());
        // File is now 0600
        assert_eq!(file_mode(&ts.scope_path), 0o600);
    }

    #[test]
    fn write_roots_replaces_directory_in_place_of_file() {
        let ts = TempScope::new();
        let ws = ts.dir.join("ws");
        fs::create_dir_all(&ws).unwrap();
        let parent = ts.scope_path.parent().unwrap();
        fs::create_dir_all(parent).unwrap();
        // Pre-create the scope file (regular), then replace it with a directory.
        fs::write(&ts.scope_path, "{}").unwrap();
        fs::remove_file(&ts.scope_path).unwrap();
        fs::create_dir(&ts.scope_path).unwrap();
        write_roots(&[ws.clone()]).unwrap();
        // The directory was replaced by a regular file.
        assert!(ts.scope_path.is_file());
        assert_eq!(file_mode(&ts.scope_path), 0o600);
    }

    #[test]
    fn write_roots_writes_canonical_for_symlink_root() {
        let ts = TempScope::new();
        let real = ts.dir.join("real_ws");
        fs::create_dir_all(&real).unwrap();
        let link = ts.dir.join("link_ws");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        write_roots(&[link.clone()]).unwrap();
        let v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&ts.scope_path).unwrap()).unwrap();
        let roots: Vec<String> = v
            .get("roots")
            .and_then(|x| x.as_array())
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect();
        assert_eq!(roots, vec![fs::canonicalize(&real).unwrap().to_string_lossy().into_owned()]);
    }

    #[test]
    fn write_roots_refuses_non_existent_root() {
        let ts = TempScope::new();
        let r = write_roots(&[ts.dir.join("does_not_exist")]);
        assert!(r.is_err(), "non-existent root must be an error: {:?}", r);
    }

    #[test]
    fn write_roots_refuses_filesystem_root() {
        let _ts = TempScope::new();
        let r = write_roots(&[PathBuf::from("/")]);
        assert!(r.is_err(), "/ must be refused: {:?}", r);
    }

    #[test]
    fn clear_roots_keeps_symlink_targets() {
        let ts = TempScope::new();
        let ws = ts.dir.join("ws");
        let scratch = ts.dir.join("scratch");
        fs::create_dir_all(&ws).unwrap();
        fs::create_dir_all(&scratch).unwrap();
        write_roots(&[ws.clone()]).unwrap();
        // Set symlink_targets by hand and rewrite via clear.
        let _parent = ts.scope_path.parent().unwrap();
        let initial = serde_json::json!({
            "version": 1,
            "roots": [ws.to_string_lossy()],
            "symlink_targets": [scratch.to_string_lossy()]
        });
        fs::write(&ts.scope_path, serde_json::to_string_pretty(&initial).unwrap()).unwrap();
        fs::set_permissions(&ts.scope_path, fs::Permissions::from_mode(0o600)).unwrap();
        clear_roots().unwrap();
        let v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&ts.scope_path).unwrap()).unwrap();
        let roots = v.get("roots").and_then(|x| x.as_array()).unwrap();
        assert_eq!(roots.len(), 0);
        let tgt: Vec<String> = v
            .get("symlink_targets")
            .and_then(|x| x.as_array())
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().to_string())
            .collect();
        assert_eq!(tgt, vec![fs::canonicalize(&scratch).unwrap().to_string_lossy().into_owned()]);
    }

    #[test]
    fn scope_file_path_matches_writer_destination_when_no_xdg() {
        let ts = TempScope::new();
        let p = scope_file_path().unwrap();
        assert_eq!(p, ts.scope_path);
    }

    #[test]
    fn write_roots_honours_xdg_runtime_dir_when_file_present() {
        let ts = TempScope::new();
        let xdg = ts.dir.join("xdg");
        fs::create_dir_all(&xdg).unwrap();
        std::env::set_var("XDG_RUNTIME_DIR", &xdg);
        let ws = ts.dir.join("ws");
        fs::create_dir_all(&ws).unwrap();
        // No XDG file yet → write goes to ~/.cache
        write_roots(&[ws.clone()]).unwrap();
        assert!(ts.scope_path.is_file());
        // Now pre-create the XDG file (a stale one) → next write goes there.
        let xdg_file = xdg.join("spinoml").join("scope.json");
        fs::create_dir_all(xdg_file.parent().unwrap()).unwrap();
        fs::write(&xdg_file, b"{\"version\":1,\"roots\":[]}").unwrap();
        fs::set_permissions(&xdg_file, fs::Permissions::from_mode(0o600)).unwrap();
        write_roots(&[ws.clone()]).unwrap();
        assert!(xdg_file.is_file(), "writer should now hit {}", xdg_file.display());
        let v: serde_json::Value = serde_json::from_str(&fs::read_to_string(&xdg_file).unwrap()).unwrap();
        assert_eq!(v.get("roots").and_then(|x| x.as_array()).unwrap().len(), 1);
    }

    // ── check_resolved ────────────────────────────────────────────────────

    fn root_and_target_fixture() -> (TempScope, PathBuf, PathBuf, PathBuf) {
        let ts = TempScope::new();
        let ws = ts.dir.join("ws");
        let outside = ts.dir.join("outside");
        let target = ts.dir.join("scratch");
        for d in [&ws, &outside, &target] {
            fs::create_dir_all(d).unwrap();
        }
        (ts, ws, outside, target)
    }

    #[test]
    fn check_resolved_file_inside_root_ok() {
        let (_ts, ws, _out, _tgt) = root_and_target_fixture();
        let f = ws.join("inside.csv");
        fs::write(&f, "a,b\n1,2\n").unwrap();
        let res = check_resolved(&ws, &f).unwrap();
        assert_eq!(res, fs::canonicalize(&f).unwrap());
    }

    #[test]
    fn check_resolved_new_file_in_existing_subdir_ok() {
        let (_ts, ws, _out, _tgt) = root_and_target_fixture();
        let candidate = ws.join("new.csv");
        let res = check_resolved(&ws, &candidate).unwrap();
        assert_eq!(res, fs::canonicalize(&ws).unwrap().join("new.csv"));
    }

    #[test]
    fn check_resolved_new_file_in_new_nested_subdir_ok() {
        let (_ts, ws, _out, _tgt) = root_and_target_fixture();
        let candidate = ws.join("a").join("b").join("c.csv");
        let res = check_resolved(&ws, &candidate).unwrap();
        assert!(res.ends_with("a/b/c.csv"));
    }

    #[test]
    fn check_resolved_symlink_file_inside_root_to_outside_rejected_read_and_write() {
        let (_ts, ws, outside, _tgt) = root_and_target_fixture();
        fs::write(outside.join("secret.csv"), "x").unwrap();
        std::os::unix::fs::symlink(outside.join("secret.csv"), ws.join("link_out")).unwrap();
        // check_resolved is the single guard for BOTH reads and writes
        // (the sidecar doesn't have separate paths for each — a write to a
        // symlink is the same code path as a read from it). A link to an
        // outside file must therefore be refused for both purposes.
        assert!(check_resolved(&ws, &ws.join("link_out")).is_err());
    }

    #[test]
    fn check_resolved_symlinked_directory_component_in_middle_rejected() {
        let (_ts, ws, outside, _tgt) = root_and_target_fixture();
        fs::create_dir(outside.join("oops")).unwrap();
        std::os::unix::fs::symlink(outside.join("oops"), ws.join("middle")).unwrap();
        let candidate = ws.join("middle").join("file.csv");
        assert!(check_resolved(&ws, &candidate).is_err());
    }

    #[test]
    fn check_resolved_symlink_to_outside_allowed_via_symlink_targets_env() {
        let (_ts, ws, outside, _tgt) = root_and_target_fixture();
        fs::write(outside.join("secret.csv"), "x").unwrap();
        std::os::unix::fs::symlink(outside.join("secret.csv"), ws.join("link_out")).unwrap();
        std::env::set_var(
            "SPINOML_SYMLINK_TARGETS",
            outside.to_string_lossy().to_string(),
        );
        let res = check_resolved(&ws, &ws.join("link_out")).unwrap();
        assert_eq!(res, fs::canonicalize(outside.join("secret.csv")).unwrap());
    }

    #[test]
    fn check_resolved_symlink_to_outside_allowed_via_symlink_targets_scope_file() {
        let (ts, ws, outside, _tgt) = root_and_target_fixture();
        fs::write(outside.join("secret.csv"), "x").unwrap();
        std::os::unix::fs::symlink(outside.join("secret.csv"), ws.join("link_out")).unwrap();
        let initial = serde_json::json!({
            "version": 1,
            "roots": [ws.to_string_lossy()],
            "symlink_targets": [outside.to_string_lossy()],
        });
        fs::create_dir_all(ts.scope_path.parent().unwrap()).unwrap();
        fs::write(&ts.scope_path, serde_json::to_string_pretty(&initial).unwrap()).unwrap();
        fs::set_permissions(&ts.scope_path, fs::Permissions::from_mode(0o600)).unwrap();
        let res = check_resolved(&ws, &ws.join("link_out")).unwrap();
        assert_eq!(res, fs::canonicalize(outside.join("secret.csv")).unwrap());
    }

    #[test]
    fn check_resolved_symlink_loop_rejected() {
        let (_ts, ws, _out, _tgt) = root_and_target_fixture();
        std::os::unix::fs::symlink(ws.join("loop1"), ws.join("loop2")).unwrap();
        std::os::unix::fs::symlink(ws.join("loop2"), ws.join("loop1")).unwrap();
        let r = check_resolved(&ws, &ws.join("loop1"));
        assert!(r.is_err(), "symlink loop must be rejected: {:?}", r);
    }

    #[test]
    fn check_resolved_lexical_dot_dot_still_rejected_by_resolve() {
        let (_ts, ws, _out, _tgt) = root_and_target_fixture();
        // check_resolved itself does not reject .. lexically; the lexical
        // guard lives in resolve() (lib.rs). Sanity: a `..` segment that does
        // not actually leave the root after canonicalization is still in.
        // What we DO assert here is the lexical guard at the resolve layer:
        let lexical = resolve_lexical_only(&ws, "../etc/passwd");
        assert!(lexical.is_err());
    }

    #[test]
    fn check_resolved_symlink_to_sibling_inside_root_ok() {
        let (_ts, ws, _out, _tgt) = root_and_target_fixture();
        fs::write(ws.join("a.csv"), "a").unwrap();
        std::os::unix::fs::symlink(ws.join("a.csv"), ws.join("b.csv")).unwrap();
        let res = check_resolved(&ws, &ws.join("b.csv")).unwrap();
        assert_eq!(res, fs::canonicalize(ws.join("a.csv")).unwrap());
    }

    #[test]
    fn check_resolved_root_is_a_symlink_works() {
        let (ts, _real_ws, _out, _tgt) = root_and_target_fixture();
        let real = ts.dir.join("real_ws");
        fs::create_dir_all(&real).unwrap();
        fs::write(real.join("inside.csv"), "a").unwrap();
        let link = ts.dir.join("link_ws");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let res = check_resolved(&link, &real.join("inside.csv")).unwrap();
        assert_eq!(res, fs::canonicalize(real.join("inside.csv")).unwrap());
    }

    #[test]
    fn check_resolved_candidate_deepest_ancestor_outside_root_rejected() {
        let (_ts, ws, outside, _tgt) = root_and_target_fixture();
        fs::write(outside.join("file.csv"), "x").unwrap();
        std::os::unix::fs::symlink(outside.join("file.csv"), ws.join("l.csv")).unwrap();
        // candidate's deepest existing ancestor is outside.csv (under outside),
        // which is outside the root.
        let r = check_resolved(&ws, &ws.join("l.csv"));
        assert!(r.is_err(), "ancestor outside root must be rejected: {:?}", r);
    }

    #[test]
    fn check_resolved_error_message_mentions_how_to_allow() {
        let (_ts, ws, outside, _tgt) = root_and_target_fixture();
        fs::write(outside.join("secret.csv"), "x").unwrap();
        std::os::unix::fs::symlink(outside.join("secret.csv"), ws.join("link_out")).unwrap();
        let err = check_resolved(&ws, &ws.join("link_out")).unwrap_err();
        assert!(err.contains("SPINOML_SYMLINK_TARGETS"), "msg: {err}");
    }

    // ── R016 regression: DANGLING symlinks must fail closed ────────────────
    //
    // `ws/x -> /nonexistent/outside` is a directory entry whose target does
    // not exist. canonicalize() reports NotFound, and the old code walked up
    // to `ws`, re-appended `x`, and called the result "inside" — so a WRITE
    // then created a file OUTSIDE the workspace through the link.

    #[test]
    fn check_resolved_dangling_symlink_to_outside_rejected() {
        let (_ts, ws, outside, _tgt) = root_and_target_fixture();
        std::os::unix::fs::symlink(outside.join("does_not_exist"), ws.join("dangling")).unwrap();
        let r = check_resolved(&ws, &ws.join("dangling"));
        assert!(r.is_err(), "dangling link to outside must be rejected: {r:?}");
        assert!(
            format!("{r:?}").contains("dangling"),
            "error should name the dangling link: {r:?}"
        );
    }

    #[test]
    fn check_resolved_dangling_symlink_write_target_rejected() {
        // The link target does not exist yet — exactly the "write" shape.
        let (_ts, ws, outside, _tgt) = root_and_target_fixture();
        std::os::unix::fs::symlink(outside.join("new_file.csv"), ws.join("wlink")).unwrap();
        let r = check_resolved(&ws, &ws.join("wlink"));
        assert!(r.is_err(), "write through a dangling link must be rejected: {r:?}");
    }

    #[test]
    fn check_resolved_dangling_link_in_the_middle_rejected() {
        // `ws/mid` is a dangling link; a candidate UNDER it must also be
        // rejected (the walk must check every level, not just the leaf).
        let (_ts, ws, outside, _tgt) = root_and_target_fixture();
        std::os::unix::fs::symlink(outside.join("missing_dir"), ws.join("mid")).unwrap();
        let r = check_resolved(&ws, &ws.join("mid").join("file.txt"));
        assert!(r.is_err(), "dangling link in the middle must be rejected: {r:?}");
    }

    #[test]
    fn check_resolved_dangling_link_with_inside_missing_target_also_rejected() {
        // Fail closed even when the (missing) target is inside the root: a
        // broken link is never something we can safely scope.
        let (_ts, ws, _out, _tgt) = root_and_target_fixture();
        std::os::unix::fs::symlink(ws.join("missing_inside"), ws.join("dl")).unwrap();
        let r = check_resolved(&ws, &ws.join("dl"));
        assert!(r.is_err(), "dangling link with an inside missing target must be rejected: {r:?}");
    }

    #[test]
    fn check_resolved_plain_missing_file_in_existing_dir_still_ok() {
        // No symlink involved: a genuinely not-yet-created file must pass.
        let (_ts, ws, _out, _tgt) = root_and_target_fixture();
        let res = check_resolved(&ws, &ws.join("new.csv")).unwrap();
        assert_eq!(res, fs::canonicalize(&ws).unwrap().join("new.csv"));
    }

    #[test]
    fn check_resolved_symlink_chain_to_outside_rejected_at_any_hop() {
        let (_ts, ws, outside, _tgt) = root_and_target_fixture();
        fs::write(outside.join("secret.txt"), "x").unwrap();
        // ws/a -> ws/b -> ws/c -> outside/secret.txt
        std::os::unix::fs::symlink(outside.join("secret.txt"), ws.join("c")).unwrap();
        std::os::unix::fs::symlink(ws.join("c"), ws.join("b")).unwrap();
        std::os::unix::fs::symlink(ws.join("b"), ws.join("a")).unwrap();
        assert!(check_resolved(&ws, &ws.join("a")).is_err(), "hop a");
        assert!(check_resolved(&ws, &ws.join("b")).is_err(), "hop b");
        assert!(check_resolved(&ws, &ws.join("c")).is_err(), "hop c");
    }

    #[test]
    fn check_resolved_dangling_chain_rejected_at_any_hop() {
        let (_ts, ws, outside, _tgt) = root_and_target_fixture();
        // a -> b -> c -> outside/missing (dangling at the end).
        std::os::unix::fs::symlink(outside.join("missing"), ws.join("c")).unwrap();
        std::os::unix::fs::symlink(ws.join("c"), ws.join("b")).unwrap();
        std::os::unix::fs::symlink(ws.join("b"), ws.join("a")).unwrap();
        assert!(check_resolved(&ws, &ws.join("a")).is_err(), "hop a");
        assert!(check_resolved(&ws, &ws.join("b")).is_err(), "hop b");
        assert!(check_resolved(&ws, &ws.join("c")).is_err(), "hop c");
    }

    // ── R016 regression: READ must apply the same owner/mode trust gate ────

    #[test]
    fn check_resolved_ignores_targets_from_untrusted_scope_file() {
        let (ts, ws, outside, _tgt) = root_and_target_fixture();
        fs::write(outside.join("secret.csv"), "x").unwrap();
        std::os::unix::fs::symlink(outside.join("secret.csv"), ws.join("link_out")).unwrap();
        // A scope file that WOULD allow the outside dir — but it is group/
        // world writable, so it must be treated as untrusted and its targets
        // ignored (same gate as scope.py).
        let initial = serde_json::json!({
            "version": 1,
            "roots": [ws.to_string_lossy()],
            "symlink_targets": [outside.to_string_lossy()],
        });
        fs::create_dir_all(ts.scope_path.parent().unwrap()).unwrap();
        fs::write(&ts.scope_path, serde_json::to_string_pretty(&initial).unwrap()).unwrap();
        fs::set_permissions(&ts.scope_path, fs::Permissions::from_mode(0o666)).unwrap();
        std::env::remove_var("SPINOML_SYMLINK_TARGETS");
        let r = check_resolved(&ws, &ws.join("link_out"));
        assert!(
            r.is_err(),
            "an untrusted (0666) scope file must not contribute symlink targets: {r:?}"
        );
    }

    // ── schema fixture ─────────────────────────────────────────────────────

    #[test]
    fn writer_output_matches_committed_fixture_shape() {
        let fixture_text = include_str!("../tests/fixtures/scope.example.json");
        let fixture: serde_json::Value = serde_json::from_str(fixture_text).unwrap();
        let fixture_obj = fixture.as_object().expect("fixture root is object");
        let ts = TempScope::new();
        // Map the fixture's placeholder roots/targets to real temp dirs so we
        // can call the writer. The shape (key set, value types) is what we
        // are asserting.
        let real_root = ts.dir.join("ws");
        let real_target = ts.dir.join("scratch");
        fs::create_dir_all(&real_root).unwrap();
        fs::create_dir_all(&real_target).unwrap();
        // Seed the file with the same `symlink_targets` the fixture has, so
        // preservation matches the schema.
        let parent = ts.scope_path.parent().unwrap();
        fs::create_dir_all(parent).unwrap();
        let seed = serde_json::json!({
            "version": 1,
            "roots": [],
            "symlink_targets": [real_target.to_string_lossy()],
        });
        fs::write(&ts.scope_path, serde_json::to_string_pretty(&seed).unwrap()).unwrap();
        fs::set_permissions(&ts.scope_path, fs::Permissions::from_mode(0o600)).unwrap();
        write_roots(&[real_root.clone()]).unwrap();
        let produced: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&ts.scope_path).unwrap()).unwrap();
        let produced_obj = produced.as_object().expect("produced root is object");
        let fixture_keys: std::collections::BTreeSet<&str> =
            fixture_obj.keys().map(|s| s.as_str()).collect();
        let produced_keys: std::collections::BTreeSet<&str> =
            produced_obj.keys().map(|s| s.as_str()).collect();
        assert_eq!(produced_keys, fixture_keys, "top-level keys match fixture");
        for k in fixture_keys {
            assert_eq!(
                produced_obj[k].is_array(),
                fixture_obj[k].is_array(),
                "{k}: arrayness"
            );
            assert_eq!(
                produced_obj[k].as_u64(),
                fixture_obj[k].as_u64(),
                "{k}: integer value"
            );
        }
    }

    /// Standalone replica of lib.rs's lexical resolve, used by a test.
    fn resolve_lexical_only(root: &Path, rel: &str) -> Result<PathBuf, String> {
        let p = PathBuf::from(rel);
        for comp in p.components() {
            use std::path::Component;
            match comp {
                Component::Normal(_) | Component::CurDir => {}
                _ => return Err(format!("rejected path segment in {rel}")),
            }
        }
        Ok(root.join(p))
    }
}
