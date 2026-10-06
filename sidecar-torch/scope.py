"""Filesystem scope for the SpinoML torch sidecar (Phase 46).

The sidecar is started by Rust BEFORE a workspace is chosen and has no channel
back to Rust, so the allowed roots come from configuration the sidecar reads
itself, per request:

  * env ``SPINOML_ALLOWED_ROOTS``   (entries separated by ``os.pathsep``)
  * env ``SPINOML_SYMLINK_TARGETS`` (same format)
  * the JSON file ``$XDG_RUNTIME_DIR/spinoml/scope.json`` if set and present,
    else ``~/.cache/spinoml/scope.json``; schema
    ``{"version": 1, "roots": [...], "symlink_targets": [...]}``.  Cached by
    ``(mtime_ns, size)`` and re-read when it changes.  IGNORED with a recorded
    ``load_error`` unless it is a regular file owned by ``os.getuid()`` with
    ``mode & 0o022 == 0``; a wrong version/shape is ignored too.  A later Rust
    change will write this file; remote HPC sidecars already pass
    ``SPINOML_ALLOWED_ROOTS=<workspace root>`` from the launcher.
  * env ``SPINOML_REQUIRE_SCOPE=1`` = strict (deny everything when unconfigured).

Legitimate symlinks OUT of the workspace exist (datasets symlinked to cluster
scratch).  The rule is therefore NOT "no symlink may leave the root": the fully
resolved path must lie under the realpath of an allowed root OR under the
realpath of a configured symlink target.

HONEST LIMIT: path scoping does NOT protect endpoints that exec model ``code``
(``/dataset/smoke``, ``/activations``, ``/infer``): arbitrary code can open
anything the process can.  That needs the sidecar token (a later phase).  This
module protects the file-reading/writing paths only.

Stdlib only; thread-safe (the server is a ThreadingHTTPServer).
"""

from __future__ import annotations

import json
import os
import stat
import sys
import threading
from typing import Any

MAX_PATH_LEN = 4096
MAX_LINK_DEPTH = 64

# Re-export the error codes so callers/tests can name them without magic strings.
SCOPE_DENIED = "SCOPE_DENIED"
PATH_SYMLINK_OUTSIDE = "PATH_SYMLINK_OUTSIDE"
SCOPE_UNCONFIGURED = "SCOPE_UNCONFIGURED"
PATH_INVALID = "PATH_INVALID"

_LOCK = threading.RLock()
_TEST_OVERRIDE: dict[str, Any] | None = None
_FILE_CACHE: dict[str, dict[str, Any]] = {}
_WARNED_UNCONFIGURED = False


class ScopeError(Exception):
    """A path was rejected by the filesystem scope.  Carries ``code`` and ``message``."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _getuid() -> int | None:
    """Current uid, or None where the platform has no ``os.getuid``."""
    try:
        return os.getuid()
    except AttributeError:  # platform has no os.getuid (Windows) → uid unknown, not invented
        return None


def set_scope_for_tests(env: dict | None = None, home: str | None = None,
                        xdg: str | None = None, uid: int | None = None) -> None:
    """Inject scope configuration for tests; ``set_scope_for_tests()`` resets it."""
    global _TEST_OVERRIDE, _WARNED_UNCONFIGURED
    with _LOCK:
        if env is None and home is None and xdg is None and uid is None:
            _TEST_OVERRIDE = None
        else:
            _TEST_OVERRIDE = {"env": env, "home": home, "xdg": xdg, "uid": uid}
        _FILE_CACHE.clear()
        _WARNED_UNCONFIGURED = False


def _env_sources() -> tuple[dict, str, str | None, int | None]:
    """Return (env, home, xdg, uid) — the test override or the real process env."""
    with _LOCK:
        ov = _TEST_OVERRIDE
    if ov is None:
        env = os.environ
        xdg = env.get("XDG_RUNTIME_DIR")
        return env, os.path.expanduser("~"), xdg, _getuid()
    env = ov["env"] if ov["env"] is not None else {}
    home = ov["home"] if ov["home"] is not None else os.path.expanduser("~")
    xdg = ov["xdg"] if ov["xdg"] is not None else env.get("XDG_RUNTIME_DIR")
    uid = ov["uid"] if ov["uid"] is not None else _getuid()
    return env, home, xdg, uid


def _split_entries(raw: Any) -> list:
    """Split a pathsep-separated env value into non-empty entries."""
    if raw is None:
        return []
    return [e for e in str(raw).split(os.pathsep) if e != ""]


def _normalize_root(entry: Any, label: str, reasons: list[str]) -> str | None:
    """Validate one root/target entry → realpath, or None (path-free reason)."""
    if not isinstance(entry, str) or entry == "":
        reasons.append(f"{label}: empty or non-string entry")
        return None
    if not os.path.isabs(entry):
        reasons.append(f"{label}: entry is not absolute")
        return None
    real = os.path.realpath(entry)
    if real == os.sep:
        reasons.append(f"{label}: filesystem root refused")
        return None
    if not os.path.isdir(real):
        reasons.append(f"{label}: entry is not an existing directory")
        return None
    return real


def _store_file_cache(key: str, st: os.stat_result, roots: list, targets: list,
                      error: str | None) -> None:
    with _LOCK:
        _FILE_CACHE[key] = {"mtime_ns": st.st_mtime_ns, "size": st.st_size,
                            "roots": list(roots), "targets": list(targets),
                            "error": error}


def _load_scope_file(env: dict, home: str, xdg: str | None,
                     uid: int | None) -> tuple[list, list, str | None]:
    """Read the scope.json (XDG first, then ~/.cache), with an (mtime_ns,size) cache.

    Returns (roots_raw, symlink_targets_raw, error).  The whole file is ignored
    (empty lists) with an error if it fails any ownership/permission/shape gate.
    Error strings never embed configured paths.
    """
    candidates: list[str] = []
    if xdg:
        candidates.append(os.path.join(str(xdg), "spinoml", "scope.json"))
    candidates.append(os.path.join(str(home), ".cache", "spinoml", "scope.json"))
    chosen = None
    for c in candidates:
        if os.path.exists(c):
            chosen = c
            break
    if chosen is None:
        return [], [], None
    try:
        st = os.stat(chosen)
    except OSError:
        return [], [], "scope file is unreadable"
    if not stat.S_ISREG(st.st_mode):
        return [], [], "scope file is not a regular file"
    if uid is not None and st.st_uid != uid:
        return [], [], "scope file is not owned by the current uid"
    if st.st_mode & 0o022 != 0:
        return [], [], "scope file is group/world writable"
    with _LOCK:
        cached = _FILE_CACHE.get(chosen)
    if cached and cached["mtime_ns"] == st.st_mtime_ns and cached["size"] == st.st_size:
        return list(cached["roots"]), list(cached["targets"]), cached["error"]
    try:
        with open(chosen, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        _store_file_cache(chosen, st, [], [], "scope file is unreadable")
        return [], [], "scope file is unreadable"
    try:
        parsed = json.loads(text)
    except Exception:
        _store_file_cache(chosen, st, [], [], "scope file is not valid JSON")
        return [], [], "scope file is not valid JSON"
    reasons: list[str] = []
    roots_raw: list = []
    targets_raw: list = []
    if not isinstance(parsed, dict):
        reasons.append("scope file root is not an object")
    elif parsed.get("version") != 1:
        reasons.append("unsupported scope file version")
    else:
        r = parsed.get("roots", [])
        t = parsed.get("symlink_targets", [])
        if not isinstance(r, list):
            reasons.append("scope file roots is not an array")
        else:
            roots_raw = list(r)
        if not isinstance(t, list):
            reasons.append("scope file symlink_targets is not an array")
        else:
            targets_raw = list(t)
    if reasons:
        roots_raw, targets_raw = [], []
    err = "; ".join(reasons) if reasons else None
    _store_file_cache(chosen, st, roots_raw, targets_raw, err)
    return roots_raw, targets_raw, err


def _get_config() -> dict[str, Any]:
    """Resolve the effective config (env + scope.json) into one dict."""
    env, home, xdg, uid = _env_sources()
    reasons: list[str] = []
    env_roots_raw = _split_entries(env.get("SPINOML_ALLOWED_ROOTS"))
    env_targets_raw = _split_entries(env.get("SPINOML_SYMLINK_TARGETS"))
    file_roots_raw, file_targets_raw, file_error = _load_scope_file(env, home, xdg, uid)
    if file_error:
        reasons.append(file_error)
    roots: list[str] = []
    targets: list[str] = []
    for e in env_roots_raw:
        r = _normalize_root(e, "env root", reasons)
        if r:
            roots.append(r)
    for e in file_roots_raw:
        r = _normalize_root(e, "scope file root", reasons)
        if r:
            roots.append(r)
    for e in env_targets_raw:
        t = _normalize_root(e, "env symlink target", reasons)
        if t:
            targets.append(t)
    for e in file_targets_raw:
        t = _normalize_root(e, "scope file symlink target", reasons)
        if t:
            targets.append(t)
    roots = list(dict.fromkeys(roots))
    targets = list(dict.fromkeys(targets))
    configured = (len(env_roots_raw) + len(file_roots_raw)) > 0
    strict = env.get("SPINOML_REQUIRE_SCOPE") == "1"
    if configured:
        mode = "enforced"
    elif strict:
        mode = "unconfigured-closed"
    else:
        mode = "unconfigured-open"
    env_used = bool(env_roots_raw or env_targets_raw)
    file_used = bool(file_roots_raw or file_targets_raw)
    source = "both" if (env_used and file_used) else "env" if env_used else "file" if file_used else "none"
    return {
        "mode": mode,
        "source": source,
        "roots": roots,
        "symlink_targets": targets,
        "root_count": len(roots),
        "symlink_target_count": len(targets),
        "load_error": "; ".join(reasons) if reasons else None,
        "strict": strict,
    }


def scope_status() -> dict[str, Any]:
    """Health-summary of the scope: mode/source + counts + a path-free load_error."""
    cfg = _get_config()
    return {
        "mode": cfg["mode"],
        "source": cfg["source"],
        "root_count": cfg["root_count"],
        "symlink_target_count": cfg["symlink_target_count"],
        "load_error": cfg["load_error"],
    }


def _is_inside(child: str, parent: str) -> bool:
    """Exact containment: ``/tmp/ws-evil`` is NOT inside ``/tmp/ws``."""
    try:
        return os.path.commonpath([child, parent]) == parent
    except ValueError:  # mixed absolute/relative or different drives → not inside (fail closed)
        return False


def _resolve_symlinks(path: str) -> str:
    """Realpath with a tolerated non-existing tail and explicit symlink-loop detection.

    Walks components left to right, following links; the first non-existing
    component is re-appended lexically (so writes to a not-yet-created file
    work).  A repeated link or > MAX_LINK_DEPTH links raises PATH_INVALID
    instead of hanging or raising an unhandled OSError.
    """
    resolved = os.sep
    stack = [c for c in path.split(os.sep) if c not in ("", ".")]
    link_depth = 0
    seen: set[str] = set()
    while stack:
        comp = stack.pop(0)
        if comp == "..":
            resolved = os.path.dirname(resolved) or os.sep
            continue
        candidate = os.path.join(resolved, comp)
        try:
            st = os.lstat(candidate)
        except OSError:
            tail = [comp] + [c for c in stack if c not in ("", ".")]
            return os.path.normpath(os.path.join(resolved, *tail))
        if stat.S_ISLNK(st.st_mode):
            if candidate in seen:
                raise ScopeError(PATH_INVALID, f"symlink loop while resolving {path!r}")
            seen.add(candidate)
            link_depth += 1
            if link_depth > MAX_LINK_DEPTH:
                raise ScopeError(PATH_INVALID, f"too many levels of symbolic links resolving {path!r}")
            try:
                target = os.readlink(candidate)
            except OSError as e:
                raise ScopeError(PATH_INVALID, f"could not read symlink {candidate!r}: {e}") from e
            if not os.path.isabs(target):
                target = os.path.join(resolved, target)
            stack = [c for c in target.split(os.sep) if c not in ("", ".")] + stack
            resolved = os.sep
            continue
        resolved = candidate
    return os.path.normpath(resolved)


def _warn_unconfigured() -> None:
    """Emit the one-time stderr warning for the unscoped (unconfigured-open) mode."""
    global _WARNED_UNCONFIGURED
    with _LOCK:
        if _WARNED_UNCONFIGURED:
            return
        _WARNED_UNCONFIGURED = True
    sys.stderr.write(
        "[spinoml-scope] WARNING: no allowed roots configured — filesystem scope is "
        "DISABLED (unconfigured-open). Set SPINOML_ALLOWED_ROOTS=<workspace root> or "
        "write scope.json.\n")


def check_path(path: str, *, write: bool = False) -> str:
    """Scope-check ``path`` and return the RESOLVED absolute path to open.

    Callers MUST use the returned path for every later operation.  Raises
    ScopeError on rejection.  In unconfigured-open mode there is no restriction
    (one stderr warning is emitted); in unconfigured-closed every call raises
    SCOPE_UNCONFIGURED.
    """
    if not isinstance(path, str):
        raise ScopeError(PATH_INVALID, "path must be a string")
    if path == "":
        raise ScopeError(PATH_INVALID, "path must not be empty")
    if "\x00" in path:
        raise ScopeError(PATH_INVALID, "path may not contain NUL")
    if len(path) > MAX_PATH_LEN:
        raise ScopeError(PATH_INVALID, f"path too long (max {MAX_PATH_LEN} characters)")
    if not os.path.isabs(path):
        raise ScopeError(PATH_INVALID, f"path must be absolute: {path!r}")

    cfg = _get_config()
    if cfg["mode"] == "unconfigured-closed":
        raise ScopeError(SCOPE_UNCONFIGURED,
                         "set SPINOML_ALLOWED_ROOTS=<workspace root> or write scope.json")
    if cfg["mode"] == "unconfigured-open":
        _warn_unconfigured()
        return _resolve_symlinks(path)

    resolved = _resolve_symlinks(path)
    allowed = cfg["roots"] + cfg["symlink_targets"]
    if any(_is_inside(resolved, root) for root in allowed):
        return resolved
    lexical = os.path.normpath(path)
    if any(_is_inside(lexical, root) for root in cfg["roots"]):
        raise ScopeError(
            PATH_SYMLINK_OUTSIDE,
            f'"{path}" resolves outside the workspace through a symlink (-> {resolved}). '
            "If intended, add the TARGET directory to SPINOML_SYMLINK_TARGETS "
            "(or scope.json symlink_targets).")
    raise ScopeError(
        SCOPE_DENIED,
        f'"{path}" is outside the allowed workspace roots. '
        "Open the project in SpinoML or set SPINOML_ALLOWED_ROOTS.")
