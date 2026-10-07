#!/usr/bin/env python3
"""SpinoML training runner (Phase 13 foundation).

This script is dropped into a run directory as ``train.py`` and executed
detached (``setsid`` + redirected stdio). It is intentionally a *pure*
Python + torch program with no SpinoML runtime dependency: everything it
needs lives next to it on disk, so the run survives the app being closed
and can later be rsynced to a remote host unchanged.

Contract (see TODO.md "Phase 13"):

  cwd is the run directory, which contains:
    run.json        frozen config (this script reads it)
    model.py        generated nn.Module (defines ``class Model``)
    train.py        this file
  and into which this script writes:
    events.jsonl    append-only, one JSON event per line (flush'd)
    metrics.json    final summary
    status          single word: queued|running|done|failed|cancelled
    checkpoints/    best.pt + last.pt

Supported dataset kinds: 'tabular' (csv/tsv/parquet → feature/target columns)
and 'manifest' (paired branches, e.g. ligand+protein → a model with one input per
branch; each branch is a PyG graph, an inline molecule graph, or a tokenized
sequence — graphs batch via PyG Batch, sequences via padded [B, Lmax] LongTensors).
Other kinds fail loudly.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
EVENTS = RUN_DIR / "events.jsonl"
STATUS = RUN_DIR / "status"
METRICS = RUN_DIR / "metrics.json"
CKPT_DIR = RUN_DIR / "checkpoints"
# run dir is <workspace>/experiments/runs/<run_id> → workspace root is 3 up.
# Used to resolve workspace-relative resume_from paths.
WORKSPACE_ROOT = RUN_DIR.parents[2] if len(RUN_DIR.parents) >= 3 else RUN_DIR


# >>> safe_load (synced block: keep byte-identical with the copy in training_template.py; checked by scripts/test-safe-load.py)
class UnsafePickleError(Exception):
    """Raised when a .pt file requires arbitrary unpickling that weights_only=True refuses."""

    def __init__(self, path: str, blocked: list[str]) -> None:
        import os
        self.path = path
        self.blocked = list(blocked)
        names = ", ".join(self.blocked) if self.blocked else "unknown globals"
        super().__init__(
            f"{os.path.basename(str(path))}: refusing to unpickle arbitrary Python objects "
            f"(blocked: {names}). Re-save the data as tensors/dicts/PyG Data objects, or "
            "start the sidecar/trainer with SPINOML_ALLOW_UNSAFE_PICKLE=1 only if you trust "
            "where the file came from."
        )


def unsafe_pickle_allowed() -> bool:
    """True iff SPINOML_ALLOW_UNSAFE_PICKLE=1 (the explicit trust escape hatch)."""
    import os
    return os.environ.get("SPINOML_ALLOW_UNSAFE_PICKLE") == "1"


_SAFE_GLOBALS_CACHE: list[str] | None = None


def register_safe_globals() -> list[str]:
    """Idempotently allowlist PyG/numpy globals weights_only=True rejects; returns their names."""
    global _SAFE_GLOBALS_CACHE
    if _SAFE_GLOBALS_CACHE is not None:
        return list(_SAFE_GLOBALS_CACHE)
    import torch
    registered: list[str] = []
    allow: list = []

    def _add(qual: str, obj: object) -> None:
        if obj is None:
            return
        allow.append(obj)
        registered.append(qual)

    try:
        import torch_geometric.data.data as _pg_data
        _add("torch_geometric.data.Data", getattr(_pg_data, "Data", None))
        _add("torch_geometric.data.data.DataEdgeAttr", getattr(_pg_data, "DataEdgeAttr", None))
        _add("torch_geometric.data.data.DataTensorAttr", getattr(_pg_data, "DataTensorAttr", None))
    except ImportError:  # optional dependency: PyG absent → these globals are not registered
        pass
    try:
        import torch_geometric.data as _pg
        _add("torch_geometric.data.HeteroData", getattr(_pg, "HeteroData", None))
    except ImportError:  # optional dependency: PyG absent → HeteroData global not registered
        pass
    try:
        import torch_geometric.data.storage as _pg_store
        _add("torch_geometric.data.storage.GlobalStorage", getattr(_pg_store, "GlobalStorage", None))
        _add("torch_geometric.data.storage.NodeStorage", getattr(_pg_store, "NodeStorage", None))
        _add("torch_geometric.data.storage.EdgeStorage", getattr(_pg_store, "EdgeStorage", None))
        _add("torch_geometric.data.storage.BaseStorage", getattr(_pg_store, "BaseStorage", None))
    except ImportError:  # optional dependency: PyG absent → storage globals not registered
        pass
    try:
        import numpy as _np
    except ImportError:
        _np = None
    if _np is not None:
        _add("numpy.ndarray", _np.ndarray)
        _add("numpy.dtype", _np.dtype)
        try:
            import numpy._core.multiarray as _np_ma
        except ImportError:
            try:
                import numpy.core.multiarray as _np_ma  # type: ignore[no-redef]
            except ImportError:
                _np_ma = None
        if _np_ma is not None:
            _add("numpy.multiarray._reconstruct", getattr(_np_ma, "_reconstruct", None))
            _add("numpy.multiarray.scalar", getattr(_np_ma, "scalar", None))
        try:
            import numpy.dtypes as _np_dtypes
            for _name in dir(_np_dtypes):
                if _name.endswith("DType"):
                    _cls = getattr(_np_dtypes, _name, None)
                    if isinstance(_cls, type):
                        _add("numpy.dtypes." + _name, _cls)
        except ImportError:  # optional: numpy.dtypes unavailable on this build → skip
            pass
    try:
        torch.serialization.add_safe_globals(allow)
    except AttributeError:
        # torch < 2.4 has no safe-globals API: plain tensors/dicts still load;
        # anything else is refused by weights_only=True rather than crashing.
        pass
    _SAFE_GLOBALS_CACHE = registered
    return list(registered)


def _is_weights_only_error(exc: BaseException) -> bool:
    """True iff `exc` is torch's weights-only refusal (not a corrupt/missing file)."""
    text = str(exc)
    return ("Weights only load failed" in text
            or "WeightsUnpickler error" in text
            or "was not an allowed global" in text
            or ("whose module" in text and "is blocked" in text))


def _blocked_globals(exc: BaseException) -> list[str]:
    """Extract the qualified names of the globals torch's weights-only unpickler refused."""
    import re
    names: list[str] = []
    for match in re.finditer(r"GLOBAL\s+([A-Za-z_][\w.]*)", str(exc)):
        name = match.group(1)
        if name not in names:
            names.append(name)
    return names


def safe_torch_load(path, map_location="cpu"):
    """Load a .pt/.pth with weights_only=True; refuse arbitrary unpickling unless trusted."""
    import os
    import sys
    import torch
    register_safe_globals()
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        if not _is_weights_only_error(exc):
            raise
        blocked = _blocked_globals(exc)
        if unsafe_pickle_allowed():
            sys.stderr.write(
                "[spinoml] WARNING: loading %s with weights_only=False because "
                "SPINOML_ALLOW_UNSAFE_PICKLE=1 — arbitrary code in the file would run\n"
                % os.path.basename(str(path))
            )
            return torch.load(path, map_location=map_location, weights_only=False)
        raise UnsafePickleError(str(path), blocked) from exc
# <<< safe_load


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def emit(kind: str, **fields) -> None:
    """Append one event line. Atomic per-line (POSIX, <4KB) + flush so the
    UI tail sees it immediately even through SLURM stdout buffering later."""
    rec = {"t": _now(), "kind": kind, **fields}
    with EVENTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def set_status(s: str) -> bool:
    STATUS.write_text(s + "\n", encoding="utf-8")
    return True


# ─── Local job state machine (Phase 30) ─────────────────────────────────────
# queued → running → done|failed|cancelled. Terminal states are FINAL: a late
# write (a 'done' racing a user cancellation, a 'running' after a crash) is
# REJECTED so CANCELLED→SUCCEEDED / FAILED→RUNNING / SUCCEEDED→RUNNING can
# never happen. Returns True when the transition was applied.

_TERMINAL_STATUSES = ("done", "failed", "cancelled")
# A read failure that is NOT an absent file (permissions, IO, …) is surfaced as
# this sentinel rather than as "" — "" is the "not started yet" state, and
# silently treating an unreadable status as not-started would let a transition
# overwrite a real terminal state (Phase 30 CANCELLED → SUCCEEDED must be
# impossible). transition_status rejects every transition while the status is
# the sentinel; main() converts it into an explicit failed run.
_STATUS_UNREADABLE = "<unreadable>"


def _read_status() -> str:
    try:
        return STATUS.read_text(encoding="utf-8").strip()
    except FileNotFoundError:  # genuinely absent = not started yet (fail-safe default)
        return ""
    except Exception:  # noqa: BLE001  # any other read failure: explicit unknown, not ""
        return _STATUS_UNREADABLE


def transition_status(next_status: str) -> bool:
    """Apply a status transition through the Phase-30 state machine.
    Allowed: queued→running, queued→cancelled, running→terminal,
    idempotent re-write of the SAME value. Everything else is rejected.
    An unreadable status is never a valid current state — every transition
    is refused until the file is readable again."""
    cur = _read_status()
    if cur == _STATUS_UNREADABLE:
        return False  # unknown state: refuse to invent a transition (Phase 30 invariant)
    if cur == next_status:
        STATUS.write_text(next_status + "\n", encoding="utf-8")
        return True
    if cur in _TERMINAL_STATUSES:
        return False  # final states are protected from stale asynchronous updates
    if cur in ("", "queued") and next_status in ("running", "cancelled", "done", "failed"):
        STATUS.write_text(next_status + "\n", encoding="utf-8")
        return True
    if cur == "running" and next_status in _TERMINAL_STATUSES:
        STATUS.write_text(next_status + "\n", encoding="utf-8")
        return True
    return False


class _Cancelled(BaseException):
    """Raised by the SIGTERM/SIGINT handler to unwind to the single graceful
    cancellation point. A cancellation is NOT a failure: the run ends
    'cancelled', never 'failed' (Phase 32). Deliberately derives from
    BaseException (like KeyboardInterrupt/SystemExit) so no `except Exception`
    guard anywhere in the trainer — the dataset/model/validate/resume failure
    paths included — can swallow it and misclassify a cancellation as a run
    failure (startup-window cancel → 'failed' would be a lie)."""


_CANCEL_RECORDED = {"v": False}  # Phase 32 — sentinel: the run.cancelled event
#                                 must be emitted exactly once even when a
#                                 duplicate signal/write races in.
_SHUTDOWN = {"v": False}  # Phase 32 — sentinel: the FIRST signal wins; duplicate
#                          signals landing during unwinding/exit are ignored so
#                          a second SIGTERM can't interrupt the shutdown (the
#                          Rust/ssh stop path sends SIGTERM to the group AND the
#                          pid — a double signal is the production pattern).


def _finish_cancel(epoch: int, note: str = "") -> None:
    """Phase 32 — the single, terminal-shielded cancellation record, used by the
    cooperative status-file path and the SIGTERM/SIGINT path alike. The Phase-30
    state machine decides: if the run already reached a FINAL state (done/failed)
    a late cancel is REJECTED — a post-SUCCESS stop must never resurrect the run.
    Otherwise the run is marked cancelled, exactly ONE run.cancelled event is
    emitted (idempotent even on a double signal), and metrics are written."""
    if not transition_status("cancelled"):
        return  # already terminal (done/failed) — the final state wins; emit nothing
    if not _CANCEL_RECORDED["v"]:
        _CANCEL_RECORDED["v"] = True
        emit("run.cancelled", epoch=epoch, applied=True, note=note)
    METRICS.write_text(json.dumps({"status": "cancelled", "epoch": epoch}, indent=2))
    _manifest_write("cancelled", status="cancelled", finished_at=_now())
    # Phase 74 — a cancelled run can be resumed if its last.pt is valid and
    # belongs to this model; the flag is recorded so the UI banner can show it.
    _record_terminal_resumable("cancelled")


def cancel_run(epoch: int, note: str = "") -> None:
    """The cooperative stop path (historic name kept): the status file already
    says 'cancelled', so the state-machine transition is idempotent; record the
    cancellation, emit the event and the metrics each exactly once."""
    _finish_cancel(epoch, note)


def fail(stage: str, msg: str, tb: str | None = None) -> None:
    emit("run.failed", stage=stage, error=msg, traceback=tb or "")
    # Phase 30 — only move running→failed; if the user already cancelled (or the
    # run already ended), the terminal state wins and the failure is recorded
    # in events but must NOT resurrect the run.
    transition_status("failed")
    METRICS.write_text(json.dumps({"status": "failed", "stage": stage, "error": msg}, indent=2))
    _manifest_write("failed", status="failed", finished_at=_now(),
                    failure={"stage": stage, "message": _strip_abs_paths(str(msg))[:500]})
    # Phase 74 — a failed run can be resumed if its last.pt is valid and
    # belongs to this model; the flag is recorded so the UI banner can show it.
    _record_terminal_resumable("failed")
    sys.stderr.write(f"[spinoml-train] FAILED in {stage}: {msg}\n")
    if tb:
        sys.stderr.write(tb)
    sys.exit(1)


def _is_finite(v) -> bool:
    """Scalar/tensor finiteness (NaN or ±Inf → False)."""
    import math
    if hasattr(v, "isfinite") and hasattr(v, "all"):
        return bool(v.isfinite().all())
    return math.isfinite(float(v))


def require_finite(name: str, value, where: str) -> None:
    """Phase 25 — numerical failure detection. If a monitored training value
    (loss, metric, or gradient) is NaN/inf the result is unusable: the run
    must NOT be reported as successful. Fails loudly with an explicit reason."""
    try:
        finite = _is_finite(value)
    except (TypeError, ValueError):
        finite = True  # non-numeric (None, dict, …) isn't a numerical failure
    if not finite:
        fail("numeric", f"non-finite {name} at {where}: {value!r} — "
                        "result is unusable; not reporting success.")


def _rng_state() -> dict:
    """Phase 26 — capture the RNG states (torch CPU + all CUDA devices, NumPy,
    Python stdlib) so a resumed run continues from the same random streams.
    'Where supported' — each backend is best-effort and skipped on absence."""
    import torch
    state: dict = {"torch": torch.get_rng_state()}
    try:
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    except Exception:  # noqa: BLE001  # no CUDA backend here → nothing to capture (best-effort)
        pass
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except Exception:  # noqa: BLE001  # NumPy absent → no NumPy stream to capture (best-effort)
        pass
    try:
        import random
        state["python"] = random.getstate()
    except Exception:  # noqa: BLE001  # stdlib random always present; capture is best-effort
        pass
    return state


def _restore_rng(state: dict) -> dict:
    """Restore RNG states captured by _rng_state. Never fatal (resuming on a
    machine without CUDA must still work), but never silent either: returns
    {stream: "restored" | "absent in checkpoint" | "failed: <reason>"} and the
    caller records it in the `run.resumed` event — a resume whose random
    streams could not be restored is NOT a bitwise continuation, and the run
    record has to say so."""
    import torch
    status: dict = {}

    def attempt(name: str, present: bool, restore) -> None:
        if not present:
            status[name] = "absent in checkpoint"
            return
        try:
            restore()
            status[name] = "restored"
        except Exception as e:  # noqa: BLE001  # recorded in `status` below and in run.resumed
            status[name] = f"failed: {type(e).__name__}: {e}"

    def _numpy() -> None:
        import numpy as np
        np.random.set_state(state["numpy"])

    def _stdlib() -> None:
        import random
        random.setstate(state["python"])

    attempt("torch", "torch" in state, lambda: torch.set_rng_state(state["torch"]))
    attempt("torch_cuda", bool(state.get("torch_cuda")), lambda: torch.cuda.set_rng_state_all(state["torch_cuda"]))
    attempt("numpy", "numpy" in state, _numpy)
    attempt("python", "python" in state, _stdlib)
    return status


def _atomic_save(obj, path: Path) -> None:
    """Phase 27 — atomic checkpoint writes: serialize to a temp file in the
    SAME directory, fsync it, then os.replace() onto the final name. A crash
    mid-write can never leave a truncated best.pt/last.pt — the previous
    valid checkpoint survives until the replace is complete."""
    import torch
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        try:  # best-effort: fsync the directory so the rename itself survives a crash
            dfd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except Exception:  # noqa: BLE001  # dir fsync is durability polish; the rename happened
            pass
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:  # noqa: BLE001  # best-effort temp cleanup; final file in place
                pass


# ─── Dataset fingerprint verification (Phase 18) ──────────────────────────
# The fingerprint is computed by the SIDECAR at inspect time
# (dataset_handlers._fingerprint_for) and frozen into run.json. We recompute
# here so a run catches a dataset that was replaced/moved between the UI
# freeze and the actual execution (e.g. detached remote launch days later).
# This mirrors dataset_handlers' hashing on purpose: the template is
# self-contained by contract, so the ~15-line helper is duplicated rather than
# importing the sidecar.

_FP_HEADER = b"spinoml-dataset-fp-v1\x00"


def _content_sha256(path: Path) -> tuple[str, int]:
    """Stream SHA-256 → (hex hash, size). Mirrors dataset_handlers._sha256_file."""
    h = hashlib.sha256(_FP_HEADER)
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            h.update(chunk)
    return h.hexdigest(), size


def _resolve_primary(path: Path) -> Path | None:
    """Pick the main data file inside a prepared-dataset directory (mirrors
    dataset_handlers._table_path), returning None for non-tabular dir kinds."""
    if not path.is_dir():
        return path
    prefer = ("pairs.csv", "data.csv", "table.csv", "dataset.csv", "train.csv", "test.csv")
    tables = [c for c in path.iterdir() if c.is_file() and c.suffix.lower() in (".csv", ".tsv", ".parquet")]
    if not tables:
        return None
    by_name = {c.name.lower(): c for c in tables}
    return next((by_name[n] for n in prefer if n in by_name),
                max(tables, key=lambda c: c.stat().st_size))


def _verify_fingerprint(cfg: dict) -> dict:
    """Recompute the primary file hash and compare against the frozen record.
    Returns a result dict suitable for emit('dataset.fingerprint', ...)."""
    fp = cfg.get("fingerprint")
    if not isinstance(fp, dict) or not fp.get("hash"):
        return {"ok": "not_frozen", "note": "run.json records no fingerprint — verify unavailable"}
    mode = fp.get("mode")
    path = Path(os.path.expanduser(str(cfg.get("path", ""))))
    if not path.exists():
        return {"ok": False, "error": "dataset path missing at run time"}
    if mode == "structure":
        return {"ok": "skipped", "note": "structure-mode fingerprint (folder) not rehashed at train time"}
    primary = _resolve_primary(path)
    if primary is None or not primary.is_file():
        return {"ok": False, "error": "no table file found in prepared dataset dir"}
    actual, size = _content_sha256(primary)
    expected = str(fp["hash"])
    return {
        "ok": actual == expected,
        "mode": mode,
        "expected": expected,
        "actual": actual,
        "size_bytes": size,
    }


def _plain_sha256(path: Path) -> str:
    """Plain SHA-256 (no header) — mirrors the frontend's crypto.subtle hash of
    the model.spinoml / model.py strings frozen into the run dir at launch."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _verify_snapshot(cfg: dict) -> dict:
    """Phase 20 — prove the run-dir copies of the graph + generated model are
    byte-identical to what was frozen at launch. Returns a result dict suitable
    for emit('run.snapshot', ...)."""
    snap = cfg.get("snapshot")
    if not isinstance(snap, dict) or not snap.get("graph_sha256") or not snap.get("model_py_sha256"):
        return {"ok": "not_frozen", "note": "run.json records no snapshot — verify unavailable"}
    graph_path = RUN_DIR / "model.spinoml"
    model_path = RUN_DIR / "model.py"
    actual_graph = _plain_sha256(graph_path) if graph_path.exists() else None
    actual_model = _plain_sha256(model_path) if model_path.exists() else None
    matches = (
        actual_graph == snap.get("graph_sha256")
        and actual_model == snap.get("model_py_sha256")
    )
    return {
        "ok": matches,
        "graph": {"expected": snap.get("graph_sha256"), "actual": actual_graph, "matches": actual_graph == snap.get("graph_sha256")},
        "model_py": {"expected": snap.get("model_py_sha256"), "actual": actual_model, "matches": actual_model == snap.get("model_py_sha256")},
        "preprocessing": snap.get("preprocessing", []),
    }


def _env_info(torch_mod, device, amp: str | None, git_root: Path | None = None) -> dict:
    """Phase 21 — record the RUNTIME environment the experiment actually ran on:
    software versions, device/dtype, hardware. Fired once right after the model
    is built so a bad run can be blamed on (or exonerated from) the environment.
    Best-effort: git commit omitted when the workspace isn't a git repo."""
    info: dict = {
        "python": platform.python_version(),
        "python_exe": sys.executable,
    }
    try:
        info["torch"] = torch_mod.__version__
        info["cuda"] = torch_mod.version.cuda or None
        try:
            info["cudnn"] = torch_mod.backends.cudnn.version()
        except Exception:  # noqa: BLE001  # build without cuDNN → version unknown, field omitted
            pass
    except Exception:  # noqa: BLE001  # torch metadata unavailable → field omitted, unknown
        pass
    try:
        import numpy as np
        info["numpy"] = np.__version__
    except Exception:  # noqa: BLE001  # NumPy absent → version field omitted (unknown, not faked)
        pass
    # PyG decides whether graph models are reproducible across machines; read the
    # installed version from package metadata (no slow import). None = not installed.
    import importlib.metadata as _md
    try:
        info["torch_geometric"] = _md.version("torch_geometric")
    except _md.PackageNotFoundError:
        info["torch_geometric"] = None
    info["os"] = platform.platform()  # e.g. Linux-6.8-x86_64-with-glibc2.39: no hostname/user
    info["device"] = device.type
    if device.type == "cuda":
        try:
            info["gpu"] = torch_mod.cuda.get_device_name(0)
            info["gpu_mem_mb"] = round(torch_mod.cuda.get_device_properties(0).total_memory / 1e6)
        except Exception:  # noqa: BLE001  # CUDA query failed → GPU fields omitted (unknown)
            pass
    info["dtype"] = "bf16" if amp == "bf16" else ("fp16" if amp else "fp32")
    info["cpus"] = os.cpu_count()
    try:
        page = os.sysconf("SC_PAGESIZE")
        info["ram_bytes"] = os.sysconf("SC_PHYS_PAGES") * page
    except Exception:  # noqa: BLE001  # sysconf unsupported → RAM field omitted (unknown)
        pass
    if git_root is not None:
        try:
            out = subprocess.check_output(
                ["git", "-C", str(git_root), "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL, timeout=2,
            ).decode().strip()
            if out:
                info["git_commit"] = out
        except Exception:  # noqa: BLE001  # not a git repo / git absent → commit omitted
            pass
    info["unsafe_pickle"] = unsafe_pickle_allowed()
    return info


# ─── Run manifest (Phase 54) ────────────────────────────────────────────────
# A machine-readable record of exactly what a run was: git state, content hashes,
# a canonical "config identity" hash (equal for equal inputs, different when any
# input differs), seed/software/hardware, and an explicit, HONEST statement of
# whether the run is reproducible from a git commit. It never claims
# reproducibility when the working tree was dirty. Written atomically at launch,
# after config.env, and at every terminal state; a write failure is recorded as a
# `manifest.error` event and NEVER changes the run's outcome.

MANIFEST = RUN_DIR / "manifest.json"
MANIFEST_SCHEMA = "spinoml.run-manifest/1"
_MANIFEST_STATE: dict = {}

# Training keys that feed the config identity. Excluded on purpose because they
# don't change WHAT is run, only the bookkeeping around it: run_id/run_label/
# created_at/status (top level), backend + slurm submission settings, dataset
# path/relpath, model_path, resume_from, and eval_only/validate execution mode.
# Paths and names must never influence identity — only content hashes and
# hyperparameters do (split/seed are also recorded separately below).
_IDENTITY_TRAINING_KEYS = (
    "epochs", "batch_size", "val_split", "split_strategy", "seed",
    "log_every_n_steps", "shuffle", "num_workers", "drop_last",
    "val_every_n_epochs", "gradient_accumulation_steps",
    "optimizer", "loss", "heads", "scheduler", "metrics", "callbacks",
)


def _sha256_file(path: Path) -> str | None:
    """Plain sha256 of a file's bytes, or None when it cannot be read."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception as exc:  # noqa: BLE001  # recorded via the None return
        sys.stderr.write(f"[spinoml-train] manifest: cannot hash {path.name}: {exc}\n")
        return None


def _strip_abs_paths(text: str) -> str:
    """Replace absolute-path-like tokens with <path> so the manifest never leaks paths."""
    import re
    return re.sub(r"(?<![\w.])/[^\s:'\"]+", "<path>", text)


def _first_stderr_line(text: str) -> str:
    """First non-empty stderr line with absolute paths stripped (≤200 chars)."""
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        return _strip_abs_paths(line)[:200]
    return "git command failed"


def _classify_git_error(stderr: str) -> str:
    """Map a git failure to a manifest `reason` string."""
    low = (stderr or "").lower()
    if "not a git repository" in low or "not a git repo" in low:
        return "not a git repository"
    return _first_stderr_line(stderr)


def _parse_tracked_status(out: str) -> list[str]:
    """Parse porcelain v1 (untracked=no) into workspace-relative changed paths."""
    files: list[str] = []
    for line in (out or "").splitlines():
        if len(line) < 4:
            continue
        rest = line[3:].strip()
        parts = rest.split(" -> ", 1) if " -> " in rest else [rest]  # renames: both sides
        for p in parts:
            p = p.strip().strip('"')
            if p and p not in files:
                files.append(p)
    return files


def _git_state(git_root: Path) -> dict:
    """Collect best-effort git provenance for `git_root`; never raises."""
    state = {
        "available": False, "commit": None, "branch": None,
        "dirty_tracked": None, "dirty_files": [], "untracked_count": None,
        "reason": None,
    }

    def _run(args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(git_root), *args],
            capture_output=True, text=True, timeout=5, check=False,
        )

    try:
        head = _run(["rev-parse", "HEAD"])
    except FileNotFoundError:
        state["reason"] = "git executable not found"
        return state
    except subprocess.TimeoutExpired:
        state["reason"] = "git timed out"
        return state
    except Exception as exc:  # noqa: BLE001
        state["reason"] = _first_stderr_line(str(exc))
        return state
    if head.returncode != 0:
        state["reason"] = _classify_git_error(head.stderr)
        return state
    commit = head.stdout.strip().lower()
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        state["reason"] = "git returned an unexpected HEAD value"
        return state
    state["available"] = True
    state["commit"] = commit
    try:
        br = _run(["rev-parse", "--abbrev-ref", "HEAD"])
        if br.returncode == 0:
            b = br.stdout.strip()
            state["branch"] = None if b in ("", "HEAD") else b
        else:
            state["reason"] = _first_stderr_line(br.stderr)
        st = _run(["status", "--porcelain=v1", "--untracked-files=no"])
        if st.returncode == 0:
            files = _parse_tracked_status(st.stdout)
            state["dirty_files"] = files[:50]
            state["dirty_tracked"] = bool(files)
        else:
            state["reason"] = state["reason"] or _first_stderr_line(st.stderr)
        run_rel = f"experiments/runs/{RUN_DIR.name}"
        ut = _run(["status", "--porcelain=v1", "--untracked-files=normal",
                   "--", ".", f":(exclude){run_rel}"])
        if ut.returncode == 0:
            state["untracked_count"] = sum(1 for ln in ut.stdout.splitlines() if ln.startswith("??"))
        else:
            state["reason"] = state["reason"] or _first_stderr_line(ut.stderr)
    except subprocess.TimeoutExpired:
        state["reason"] = "git timed out"
    except Exception as exc:  # noqa: BLE001
        state["reason"] = state["reason"] or _first_stderr_line(str(exc))
    if state["reason"] is not None:
        state["dirty_tracked"] = None  # a failed git call must never look "clean"
    return state


def _config_identity(cfg: dict) -> tuple[str, list[str]]:
    """Canonical config-identity sha256 over content hashes + hyperparameters."""
    snap = cfg.get("snapshot") or {}
    ds = cfg.get("dataset") or {}
    train = cfg.get("training") or {}
    fp = ds.get("fingerprint") if isinstance(ds.get("fingerprint"), dict) else None
    pre_hashes = sorted(
        hashlib.sha256(str(step.get("script", "")).encode("utf-8")).hexdigest()
        for step in (snap.get("preprocessing") or []) if isinstance(step, dict)
    )
    trust_hashes = sorted(
        str(t.get("sha256", "")) for t in (snap.get("code_trust") or []) if isinstance(t, dict)
    )
    training = {k: train[k] for k in _IDENTITY_TRAINING_KEYS if k in train}
    identity = {
        "graph_sha256": snap.get("graph_sha256"),
        "model_py_sha256": snap.get("model_py_sha256"),
        "preprocessing_sha256": pre_hashes,
        "code_trust_sha256": trust_hashes,
        "training": training,
        "dataset": {
            "fingerprint": (f"{fp.get('alg')}:{fp.get('hash')}" if fp and fp.get("hash") else None),
            "kind": ds.get("kind"),
            "feature_columns": ds.get("feature_columns"),
            "target_column": ds.get("target_column"),
            "heads": train.get("heads"),
            "adapter": (cfg.get("validate") or {}).get("adapter"),
        },
        "split": {
            "strategy": train.get("split_strategy"),
            "val_split": train.get("val_split"),
            "seed": train.get("seed"),
        },
        "seed": train.get("seed"),
    }
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    fields = [
        "graph_sha256", "model_py_sha256", "preprocessing_sha256", "code_trust_sha256",
    ] + [f"training.{k}" for k in sorted(training)] + [
        "dataset.fingerprint", "dataset.kind", "dataset.feature_columns",
        "dataset.target_column", "dataset.heads", "dataset.adapter",
        "split.strategy", "split.val_split", "split.seed", "seed",
    ]
    return hashlib.sha256(blob).hexdigest(), fields


def _collect_hashes(cfg: dict) -> tuple[dict, list[str]]:
    """Content hashes for the manifest: graph/model/train bytes + config identity."""
    snap = cfg.get("snapshot") or {}
    ds = cfg.get("dataset") or {}
    fp = ds.get("fingerprint") if isinstance(ds.get("fingerprint"), dict) else None
    identity_sha, fields = _config_identity(cfg)
    return {
        "graph_sha256": snap.get("graph_sha256") or _sha256_file(RUN_DIR / "model.spinoml"),
        "model_py_sha256": snap.get("model_py_sha256") or _sha256_file(RUN_DIR / "model.py"),
        "train_py_sha256": _sha256_file(RUN_DIR / "train.py"),
        "dataset_fingerprint": (f"{fp.get('alg')}:{fp.get('hash')}" if fp and fp.get("hash") else None),
        "config_identity_sha256": identity_sha,
    }, fields


def _manifest_env_split(env: dict) -> tuple[dict, dict]:
    """_env_info → (software, hardware) subsets; drops python_exe (absolute path)."""
    software_keys = ("python", "torch", "torch_geometric", "numpy", "cuda", "cudnn", "os")
    hardware_keys = ("gpu", "gpu_mem_mb", "cpus", "ram_bytes")
    return ({k: env[k] for k in software_keys if k in env},
            {k: env[k] for k in hardware_keys if k in env})


def _manifest_notes(git: dict, fp_mode: str | None, device: str | None) -> list[str]:
    """Explicit, human-readable limitations that actually apply to THIS run."""
    notes: list[str] = []
    if git.get("dirty_tracked") is True:
        notes.append("workspace has uncommitted changes to tracked files")
    if git.get("commit") is None and git.get("reason") == "not a git repository":
        notes.append("workspace is not a git repository: reproducibility from git is not claimed")
    uc = git.get("untracked_count")
    if isinstance(uc, int) and uc > 0:
        notes.append(f"{uc} untracked files in the workspace are not part of the commit")
    if fp_mode == "reference":
        notes.append("dataset fingerprint mode 'reference' pins only the reference file, not the downloaded data")
    if device == "cuda":
        notes.append("device cuda: bit-level reproducibility is not claimed (nondeterministic CUDA reductions)")
    if unsafe_pickle_allowed():
        notes.append("unsafe pickle loading was enabled for this run")
    return notes


def _manifest_summary(best_val_loss: float | None, epochs_done: int | None,
                      n_params: int | None) -> dict | None:
    """Terminal run summary (numbers only), or None when not applicable."""
    if best_val_loss is None and epochs_done is None and n_params is None:
        return None
    return {
        "best_val_loss": None if best_val_loss is None else round(float(best_val_loss), 6),
        "epochs_done": epochs_done,
        "n_params": n_params,
    }


def _manifest_cleanup_tmp(tmp: Path, stage: str) -> None:
    """Remove a leftover temp manifest; a cleanup failure is recorded, not swallowed."""
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception as exc:  # noqa: BLE001
        emit("manifest.error", stage=stage, error=f"tmp cleanup: {exc}")


def _manifest_write(stage: str, **updates) -> None:
    """Atomically (re)write manifest.json; a failure is emitted, never fatal."""
    if not _MANIFEST_STATE:
        return
    _MANIFEST_STATE.update(updates)
    fp_mode = (_MANIFEST_STATE.get("dataset") or {}).get("fingerprint_mode")
    _MANIFEST_STATE["notes"] = _manifest_notes(
        _MANIFEST_STATE.get("git") or {}, fp_mode, _MANIFEST_STATE.get("device"))
    tmp = MANIFEST.with_name(MANIFEST.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_MANIFEST_STATE, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, MANIFEST)
        try:
            dfd = os.open(str(MANIFEST.parent), os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except Exception as exc:  # noqa: BLE001
            emit("manifest.error", stage=stage, error=f"dir fsync: {exc}")
    except Exception as exc:  # noqa: BLE001
        emit("manifest.error", stage=stage, error=str(exc))
    finally:
        _manifest_cleanup_tmp(tmp, stage)


def _manifest_init(cfg: dict) -> None:
    """Prime the running manifest right after run.provenance (pre-dataset-load)."""
    global _MANIFEST_STATE
    git = _git_state(WORKSPACE_ROOT)
    hashes, fields = _collect_hashes(cfg)
    ds = cfg.get("dataset") or {}
    train = cfg.get("training") or {}
    fp = ds.get("fingerprint") if isinstance(ds.get("fingerprint"), dict) else None
    _MANIFEST_STATE = {
        "schema": MANIFEST_SCHEMA,
        "experiment_id": RUN_DIR.name,
        "created_at": _now(),
        "finished_at": None,
        "status": "running",
        "failure": None,
        "git": git,
        "reproducible_from_git": bool(git.get("commit")
                                      and git.get("dirty_tracked") is False
                                      and git.get("reason") is None),
        "hashes": hashes,
        "seed": int(train.get("seed", 42)),
        "dtype": None,
        "device": None,
        "software": {},
        "hardware": {},
        "dataset": {
            "kind": ds.get("kind"),
            "fingerprint_mode": (fp.get("mode") if fp else None),
        },
        "split": {
            "strategy": train.get("split_strategy"),
            "val_split": train.get("val_split"),
            "seed": train.get("seed"),
        },
        "code_trust_count": len((cfg.get("snapshot") or {}).get("code_trust") or []),
        "unsafe_pickle": unsafe_pickle_allowed(),
        "summary": None,
        "identity_fields": fields,
        "notes": [],
    }
    _manifest_write("running")


# ─── Integrity gate (Phase 73) + Resumable (Phase 74) ───────────────────────
# A run may only be declared `done` when its required artifacts really exist
# and are valid; otherwise it ends `failed` with stage `integrity`, never `done`.
# Independently, every terminal write (done/failed/cancelled) computes whether
# the run is resumable from `checkpoints/last.pt` and persists the flag plus a
# `run.resumable` event so the UI can show a banner — without ever auto-resuming.

# Checkpoints larger than this skip the deep load+keys check in both the gate
# and _compute_resumable (zip+size header check is still performed). The size
# is recorded as a note in the gate output and as the resumable reason.
_CKPT_LOAD_SKIP_BYTES = 256 * 1024 * 1024  # 256 MB

# Events that MUST appear in events.jsonl for a run to be considered honest.
_REQUIRED_EVENT_KINDS = ("run.provenance", "config.env", "run.snapshot")


def _is_finite_loss(v) -> bool:
    """Same finiteness contract as _is_finite, but tolerant of strings/None —
    Phase 25/73: a non-finite best_val_loss is unusable and must fail integrity."""
    if v is None:
        return False
    try:
        import math
        return math.isfinite(float(v))
    except (TypeError, ValueError):  # non-numeric loss → unusable → integrity must fail closed
        return False


def _checkpoint_ok(path: Path) -> tuple[bool, str]:
    """Verify a single .pt: exists, non-empty, valid zip, and (when ≤ 256 MB)
    loadable via safe_torch_load with the required keys. Returns (ok, reason).
    `reason` is "" on success, otherwise a one-line explanation of the failure."""
    if not path.exists():
        return False, "missing"
    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, f"stat failed: {type(exc).__name__}"
    if size == 0:
        return False, "empty (0 bytes)"
    try:
        if not zipfile.is_zipfile(str(path)):
            return False, "not a valid zip"
    except (OSError, zipfile.BadZipFile) as exc:
        return False, f"zip check failed: {type(exc).__name__}"
    if size > _CKPT_LOAD_SKIP_BYTES:
        # Header+size check only — refuse to load multi-hundred-MB checkpoints
        # in the gate; record the skip reason for the caller.
        return True, "checkpoint load verification skipped (size)"
    try:
        ckpt = safe_torch_load(path)
    except UnsafePickleError as exc:
        return False, f"unsafe pickle: {exc}"[:200]
    except Exception as exc:
        return False, f"load failed: {type(exc).__name__}"
    if not isinstance(ckpt, dict):
        return False, "not a dict"
    needed = {"model_state", "optim_state", "epoch", "global_step", "config"}
    missing_keys = sorted(needed - set(ckpt.keys()))
    if missing_keys:
        return False, f"missing keys: {missing_keys}"
    return True, ""


def _verify_run_integrity(eval_only: bool) -> dict:
    """Phase 73 — verify every required artifact is present and valid BEFORE
    declaring the run done. Runs AFTER metrics.json + the final checkpoints are
    written and BEFORE the terminal `done` status / `run.done` event.

    Returns {"ok", "missing", "invalid", "notes"}. The caller decides whether
    to proceed (emit run.integrity with ok=true and continue) or call
    fail("integrity", msg) when ok is False. Never raises — any exception
    inside the gate is reported as an `invalid` entry and counts as failure
    (fail closed)."""
    try:
        missing: list[str] = []
        invalid: list[str] = []
        notes: list[str] = []

        # (a) metrics.json — required in normal form (full payload) or eval form.
        if not METRICS.exists():
            missing.append("metrics.json (not written)")
        else:
            try:
                m = json.loads(METRICS.read_text(encoding="utf-8"))
            except Exception as exc:
                invalid.append(f"metrics.json: JSON parse failed ({type(exc).__name__})")
                m = None
            if isinstance(m, dict):
                if "status" not in m:
                    invalid.append("metrics.json: missing 'status' field")
                if eval_only:
                    if m.get("eval_only") is not True:
                        invalid.append("metrics.json: eval_only form expected (eval_only != true)")
                    if not _is_finite_loss(m.get("best_val_loss")):
                        invalid.append("metrics.json: best_val_loss is not finite")
                else:
                    if not _is_finite_loss(m.get("best_val_loss")):
                        invalid.append("metrics.json: best_val_loss is not finite")
                    ep = m.get("epochs")
                    if not isinstance(ep, int) or ep < 1:
                        invalid.append("metrics.json: epochs < 1 or not an int")
                    np_ = m.get("n_params")
                    if not isinstance(np_, int) or np_ < 1:
                        invalid.append("metrics.json: n_params < 1 or not an int")
            else:
                invalid.append("metrics.json: not a JSON object")

        # (b) checkpoints — required only for non-eval runs.
        if not eval_only:
            for name in ("best.pt", "last.pt"):
                ok, reason = _checkpoint_ok(CKPT_DIR / name)
                if not ok:
                    (missing if reason == "missing" else invalid).append(
                        f"checkpoints/{name}: {reason}"
                    )
                elif reason == "checkpoint load verification skipped (size)":
                    notes.append(reason)

        # (c) events.jsonl — required events + at least one epoch.end.
        if not EVENTS.exists():
            missing.append("events.jsonl (missing)")
        else:
            try:
                seen_kinds: set[str] = set()
                saw_epoch_end = False
                for ln in EVENTS.read_text(encoding="utf-8").splitlines():
                    if not ln.strip():
                        continue
                    try:
                        ev = json.loads(ln)
                    except Exception:  # a torn/corrupt event line is skipped; required kinds still checked
                        continue
                    if isinstance(ev, dict):
                        k = ev.get("kind")
                        if isinstance(k, str):
                            seen_kinds.add(k)
                            if k == "epoch.end":
                                saw_epoch_end = True
                for needed in _REQUIRED_EVENT_KINDS:
                    if needed not in seen_kinds:
                        invalid.append(f"events.jsonl: missing required event '{needed}'")
                if not saw_epoch_end:
                    invalid.append("events.jsonl: no epoch.end event")
            except Exception as exc:
                invalid.append(f"events.jsonl: read/parse failed ({type(exc).__name__})")

        # (d) manifest.json — schema version pinned.
        if not MANIFEST.exists():
            missing.append("manifest.json (missing)")
        else:
            try:
                mf = json.loads(MANIFEST.read_text(encoding="utf-8"))
            except Exception as exc:
                invalid.append(f"manifest.json: JSON parse failed ({type(exc).__name__})")
                mf = None
            if isinstance(mf, dict):
                if mf.get("schema") != MANIFEST_SCHEMA:
                    invalid.append(f"manifest.json: schema != {MANIFEST_SCHEMA!r}")
            else:
                invalid.append("manifest.json: not a JSON object")

        # (e) stdout.log / stderr.log — they must EXIST when the executor launched
        # the run (the run dir carries a `pid` file), because their absence means the
        # launch redirection was lost. They may be EMPTY: the trainer writes its
        # events to events.jsonl, so a healthy run leaves both logs at 0 bytes (an
        # empty-log rule turned every executor-launched run into `failed`, found by
        # the live cluster run). Direct `python train.py` launches (e.g. the
        # verify-* harnesses) have no pid file — record a note.
        pid_file = RUN_DIR / "pid"
        if pid_file.exists():
            for name in ("stdout.log", "stderr.log"):
                p = RUN_DIR / name
                if not p.exists():
                    missing.append(f"{name} (missing; required by executor launch)")
                else:
                    try:
                        p.stat()
                    except OSError:
                        invalid.append(f"{name}: stat failed")
        else:
            notes.append("no pid file present; stdout.log/stderr.log not required")

        return {
            "ok": not missing and not invalid,
            "missing": missing,
            "invalid": invalid,
            "notes": notes,
        }
    except Exception as exc:
        # Fail closed: the gate itself MUST NEVER crash a run. An unexpected
        # exception becomes an explicit invalid entry → caller calls fail().
        return {
            "ok": False,
            "missing": [],
            "invalid": [f"integrity check error: {type(exc).__name__}"],
            "notes": [],
        }


def _compute_resumable(final_status: str) -> dict:
    """Phase 74 — terminal-time resumable flag. The status stays failed/cancelled;
    this is an additional flag the UI banner reads, NOT a new status value.
    True only for failed/cancelled AND last.pt exists, is valid, and belongs to
    this run (config.snapshot hashes match the manifest)."""
    rel_path = f"experiments/runs/{RUN_DIR.name}/checkpoints/last.pt"
    if final_status == "done":
        return {"resumable": False, "resume_from": None, "epoch": None,
                "reason": "run completed"}
    last = CKPT_DIR / "last.pt"
    if not last.exists():
        return {"resumable": False, "resume_from": None, "epoch": None,
                "reason": "no checkpoint"}
    try:
        size = last.stat().st_size
    except OSError:
        return {"resumable": False, "resume_from": None, "epoch": None,
                "reason": "checkpoint corrupt"}
    if size == 0:
        return {"resumable": False, "resume_from": None, "epoch": None,
                "reason": "no checkpoint"}
    if not zipfile.is_zipfile(str(last)):
        return {"resumable": False, "resume_from": None, "epoch": None,
                "reason": "checkpoint corrupt"}
    if size > _CKPT_LOAD_SKIP_BYTES:
        return {"resumable": False, "resume_from": None, "epoch": None,
                "reason": "checkpoint load verification skipped (size)"}
    try:
        ckpt = safe_torch_load(last)
    except (UnsafePickleError, Exception):
        return {"resumable": False, "resume_from": None, "epoch": None,
                "reason": "checkpoint corrupt"}
    if not isinstance(ckpt, dict):
        return {"resumable": False, "resume_from": None, "epoch": None,
                "reason": "checkpoint corrupt"}
    needed = {"model_state", "optim_state", "epoch", "global_step", "config"}
    if not needed.issubset(set(ckpt.keys())):
        return {"resumable": False, "resume_from": None, "epoch": None,
                "reason": "checkpoint corrupt"}
    # config.snapshot hash match — must belong to THIS run. We read the live
    # manifest.json first (it's the persistent truth) and fall back to the
    # in-memory _MANIFEST_STATE which is only populated during a fresh training
    # invocation. This matters for the "restart on a pre-cancelled run dir"
    # path: the trainer doesn't re-init the manifest, so _MANIFEST_STATE is
    # empty — but the snapshot is still on disk in manifest.json.
    cfg = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
    snap = cfg.get("snapshot") if isinstance(cfg.get("snapshot"), dict) else {}
    ckpt_graph = snap.get("graph_sha256")
    ckpt_model = snap.get("model_py_sha256")
    run_graph = None
    run_model = None
    try:
        if MANIFEST.exists():
            mf = json.loads(MANIFEST.read_text(encoding="utf-8"))
            if isinstance(mf, dict) and isinstance(mf.get("hashes"), dict):
                run_graph = mf["hashes"].get("graph_sha256")
                run_model = mf["hashes"].get("model_py_sha256")
    except Exception:  # noqa: BLE001  # recorded via the unknown-hashes check below
        run_graph = None
        run_model = None
    if (run_graph is None or run_model is None) \
            and _MANIFEST_STATE and isinstance(_MANIFEST_STATE.get("hashes"), dict):
        run_graph = run_graph or _MANIFEST_STATE["hashes"].get("graph_sha256")
        run_model = run_model or _MANIFEST_STATE["hashes"].get("model_py_sha256")
    if run_graph is None or run_model is None:
        # The run's OWN hashes could not be read (missing/corrupt manifest).
        # Treating that as a match would claim "resumable" for a checkpoint that
        # may belong to a different model — an invented value. Report UNKNOWN.
        return {"resumable": False, "resume_from": None, "epoch": None,
                "reason": "run hashes unavailable (manifest unreadable)"}
    same_graph = (not ckpt_graph) or ckpt_graph == run_graph
    same_model = (not ckpt_model) or ckpt_model == run_model
    if not (same_graph and same_model):
        return {"resumable": False, "resume_from": None, "epoch": None,
                "reason": "checkpoint belongs to a different model"}
    return {
        "resumable": True,
        "resume_from": rel_path,
        "epoch": int(ckpt.get("epoch", 0)),
        "reason": "last.pt valid and matches this model",
    }


def _record_terminal_resumable(final_status: str, base_metrics: dict | None = None) -> dict:
    """Phase 74 — write the resumable state into metrics.json + manifest.json
    and emit the `run.resumable` event. Called from every terminal write
    (done, failed, cancelled) so the UI banner has the data without needing
    a second read. Never raises: a manifest/metrics write failure is recorded
    via `manifest.error` so the run's outcome is NEVER changed here."""
    try:
        r = _compute_resumable(final_status)
    except Exception as exc:
        r = {"resumable": False, "resume_from": None, "epoch": None,
             "reason": f"resumable computation failed: {type(exc).__name__}"}
    # metrics.json: preserve existing fields, add `resumable`.
    try:
        cur: dict
        if METRICS.exists():
            try:
                loaded = json.loads(METRICS.read_text(encoding="utf-8"))
                cur = loaded if isinstance(loaded, dict) else {}
            except Exception:
                cur = {}
        else:
            cur = {}
        if base_metrics:
            for k, v in base_metrics.items():
                cur.setdefault(k, v)
        cur["resumable"] = r
        METRICS.write_text(json.dumps(cur, indent=2))
    except Exception as exc:
        emit("manifest.error", stage="resumable.metrics", error=str(exc))
    # manifest.json: add `resumable`.
    try:
        if MANIFEST.exists():
            _manifest_write("resumable", resumable=r)
    except Exception as exc:
        emit("manifest.error", stage="resumable.manifest", error=str(exc))
    emit("run.resumable", **r)
    return r


# ─── Multitask plumbing ─────────────────────────────────────────────────────
# The trainer is uniformly multi-head: a list of "heads", each binding one model
# output → a target column + loss + weight. A single-task run is just one head
# (output '' = the model's sole output). The combined objective is the weighted
# sum of the per-head losses; metrics/eval are reported per head.

LOSS_TASK = {
    "CrossEntropyLoss": "classification",
    "BCEWithLogitsLoss": "binary",
    "MSELoss": "regression",
    "L1Loss": "regression",
}


def make_loss_fn(kind: str, label_smoothing: float = 0.0):
    import torch

    ls = float(label_smoothing or 0)
    if kind == "CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss(label_smoothing=ls)
    if kind == "BCEWithLogitsLoss":
        return torch.nn.BCEWithLogitsLoss()
    if kind == "MSELoss":
        return torch.nn.MSELoss()
    if kind == "L1Loss":
        return torch.nn.L1Loss()
    raise ValueError(f"unknown loss kind: {kind}")


def resolve_heads(train_cfg: dict, ds_cfg: dict):
    """Returns (heads, multitask). Each head dict has output/target/loss_kind/
    task/weight/label_smoothing/loss_fn. Single-task synthesises one head from
    the legacy `loss` + dataset.target_column."""
    raw = train_cfg.get("heads")
    multitask = bool(raw)
    if not raw:
        lcfg = train_cfg.get("loss", {}) or {}
        raw = [{
            "output": "",
            "target": ds_cfg.get("target_column"),
            "loss": lcfg.get("kind", "CrossEntropyLoss"),
            "label_smoothing": lcfg.get("label_smoothing", 0),
        }]
    heads = []
    for h in raw:
        kind = h.get("loss", "CrossEntropyLoss")
        heads.append({
            "output": str(h.get("output", "") or ""),
            "target": h.get("target"),
            "loss_kind": kind,
            "task": LOSS_TASK.get(kind, "classification"),
            "weight": float(h.get("weight", 1.0)),
            "label_smoothing": float(h.get("label_smoothing", 0) or 0),
            "loss_fn": make_loss_fn(kind, h.get("label_smoothing", 0)),
        })
    return heads, multitask


def encode_target(series, task: str, known_classes: list | None = None):
    """pandas Series → (y_tensor[N], classes|None, n_classes|None) for a head.

    MISSING values become NaN (regression/binary) or the -1 code (classification)
    so the trainer can MASK them out per head — this is what lets a head be
    defined for only some rows (e.g. an affinity-regression head valid only for
    real binders; decoy rows have an empty target and simply don't contribute).

    `known_classes` (external validation): encode against the TRAINED class order
    so codes match the model the checkpoint came from; labels unseen in training
    get the -1 code (masked, not silently misclassified to a wrong index)."""
    import numpy as np
    import pandas as pd
    import torch

    if task == "regression":
        # coerce → NaN for empty/non-numeric cells (masked out in the loss)
        vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float32")
        return torch.from_numpy(np.ascontiguousarray(vals)), None, None
    # classification + binary: map labels → integer codes; missing → -1 code.
    if known_classes:
        classes = [str(c) for c in known_classes]
        codes = pd.Categorical(series.astype(str), categories=classes).codes  # -1 = unseen
    else:
        # default (training): sorted categories from the data itself.
        cat = series.astype("category")
        classes = [str(c) for c in cat.cat.categories.tolist()]
        codes = cat.cat.codes.to_numpy()
    if task == "binary":
        # BCEWithLogits wants float {0,1}; >2 categories is a config error.
        if len(classes) > 2:
            raise ValueError(f"binary head got {len(classes)} categories {classes[:5]}… — use CrossEntropyLoss")
        vals = codes.astype("float32")
        vals[codes < 0] = np.nan  # missing → NaN so it's masked, not trained as 0
        return torch.from_numpy(np.ascontiguousarray(vals)), classes, len(classes)
    return torch.from_numpy(np.ascontiguousarray(codes.astype("int64"))), classes, len(classes)


# ─── Dataset loading ────────────────────────────────────────────────────────

def load_tabular(cfg: dict, heads: list[dict], known_classes_by_head: dict | None = None):
    """Returns (X: FloatTensor [N, F], targets, feature_cols) where targets maps
    each head's output name → {y, classes, n_classes}. For external validation,
    `known_classes_by_head` pins each head's class order to the trained model."""
    import numpy as np
    import pandas as pd
    import torch

    path = Path(os.path.expanduser(cfg["path"]))
    # A prepared dataset DIRECTORY (e.g. a TDC BindingDB export: pairs.csv +
    # sequences.csv + embeddings) → read its inner table file.
    if path.is_dir():
        prefer = ("pairs.csv", "data.csv", "table.csv", "dataset.csv", "train.csv", "test.csv")
        tables = [c for c in path.iterdir() if c.is_file() and c.suffix.lower() in (".csv", ".tsv", ".parquet")]
        by_name = {c.name.lower(): c for c in tables}
        path = next((by_name[n] for n in prefer if n in by_name),
                    max(tables, key=lambda c: c.stat().st_size) if tables else path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    target_cols = []
    for h in heads:
        col = h["target"]
        if not col:
            raise ValueError("tabular training needs a target column for every head")
        if col not in df.columns:
            raise ValueError(f"target column {col!r} not in dataset columns {list(df.columns)}")
        target_cols.append(col)

    feature_cols = cfg.get("feature_columns")
    if feature_cols:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature columns not in dataset: {missing}")
    else:
        # default: all numeric columns except any target
        feature_cols = [
            c for c in df.select_dtypes(include="number").columns if c not in target_cols
        ]
    if not feature_cols:
        raise ValueError("no usable feature columns")

    X_np_raw = df[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    # Phase 24 — invalid (NaN/inf) input must FAIL loudly, never be silently
    # imputed or trivially passed through. garbage_in_failed lets the user fix
    # the dataset instead of training on zeros and believing the metrics.
    if not np.isfinite(X_np_raw).all():
        bad = int(np.count_nonzero(~np.isfinite(X_np_raw)))
        cols = [c for c in feature_cols
                if not np.isfinite(df[c].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")).all()]
        raise ValueError(
            f"non-finite values found in the feature matrix ({bad} NaN/inf cells; "
            f"columns affected: {cols[:8]}{'…' if len(cols) > 8 else ''}). "
            "A dataset with NaN/inf inputs cannot be trained on — fix or drop the "
            "affected samples/columns and re-inspect the dataset.")
    X_np = X_np_raw.astype(np.float32)
    X = torch.from_numpy(np.ascontiguousarray(X_np))

    targets = {}
    for h in heads:
        y, classes, n_classes = encode_target(
            df[h["target"]], h["task"], (known_classes_by_head or {}).get(h["output"]))
        targets[h["output"]] = {"y": y, "classes": classes, "n_classes": n_classes}
    return X, targets, feature_cols


# ── manifest (paired graph) datasets — self-contained, no dataset_handlers ──
# The run is a standalone snapshot (train.py + model.py), so the manifest logic
# is inlined here. Builds one item per branch per row: a PyG Data (file branches
# from .pt on disk, molecule branches from SMILES via RDKit cached as .pt) or a
# token-id LongTensor (sequence branches tokenized from a string column).

def _manifest_mol_graph(smi: str, cache_dir):
    import hashlib
    import torch
    from torch_geometric.data import Data
    fp = None
    if cache_dir is not None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            fp = cache_dir / f"mol_{hashlib.sha1(smi.encode('utf-8')).hexdigest()[:16]}.pt"
            if fp.exists():
                d = safe_torch_load(fp)
                if hasattr(d, "edge_index"):
                    return d
        except UnsafePickleError:
            raise
        except Exception:
            fp = None
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES {smi!r}")
    feats = [[float(a.GetAtomicNum()), float(a.GetDegree()), float(a.GetFormalCharge()),
              float(int(a.GetIsAromatic())), float(a.GetTotalNumHs())] for a in mol.GetAtoms()]
    src, dst = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src += [i, j]; dst += [j, i]
    x = torch.tensor(feats, dtype=torch.float32) if feats else torch.zeros((1, 5))
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    d = Data(x=x, edge_index=ei)
    if fp is not None:
        try:
            torch.save(d, fp)
        except Exception:  # mol-graph disk cache is best-effort; the graph is still in memory
            pass
    return d


# Sequence branches (manifest kind="sequence"): tokenize a string cell → a 1-D
# LongTensor of token ids, NO graph. Token 0 = PAD, 1 = UNK, real chars start at 2
# (fixed vocab → deterministic). Kept in sync with dataset_handlers.tokenize_sequence.
_SEQ_VOCABS = {
    "protein": "ACDEFGHIKLMNPQRSTVWYXBZUO",
    "smiles": "#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVZ[\\]abcdefgilmnoprstuy",
}


def _manifest_tokenize(value, spec):
    import torch
    s = str(value).strip()
    max_len = spec.get("max_len")
    if isinstance(max_len, int) and max_len > 0:
        s = s[:max_len]
    v = spec.get("vocab")
    chars = _SEQ_VOCABS.get(v, v) if isinstance(v, str) else None
    if not chars:  # byte-level fallback: id = min(ord, 255) + 1 (0 = PAD)
        ids = [min(ord(c), 255) + 1 for c in s]
    else:
        vmap = {c: i + 2 for i, c in enumerate(chars)}
        ids = [vmap.get(c, 1) for c in s]  # 1 = UNK
    return torch.tensor(ids or [0], dtype=torch.long)


# ESPF branches (manifest kind="espf"): ESPF substructure subword tokens (MolTrans).
# The run is a standalone snapshot, so it can't read sidecar-torch/espf/; instead it
# reads the compact codebook the sidecar cached next to the manifest in <base>/.espf/
# (primed when the dataset is inspected). Token 0 = PAD, 1 = UNK, subwords start at 2.
_ESPF_CACHE = {}  # name → {ranks, sub2id}


def _load_espf_codebook(base, name="drug"):
    import gzip
    import json
    name = str(name or "drug")
    if name in _ESPF_CACHE:
        return _ESPF_CACHE[name]
    fp = Path(base) / ".espf" / f"{name}.json.gz"
    if not fp.exists():
        raise FileNotFoundError(
            f"ESPF codebook cache {fp} not found — open this dataset once in the "
            f"Datasets tab (inspect/smoke-test) so the sidecar materializes .espf/, "
            f"then re-run training.")
    with gzip.open(fp, "rt", encoding="utf-8") as f:
        cb = json.load(f)
    ranks = {}
    for line in cb["merges"]:
        parts = line.split()
        if len(parts) == 2:
            ranks[(parts[0], parts[1])] = len(ranks)
    sub2id = {s: i for i, s in enumerate(cb["subwords"])}
    out = {"ranks": ranks, "sub2id": sub2id}
    _ESPF_CACHE[name] = out
    return out


def _espf_encode(orig, ranks):
    if len(orig) < 2:
        return [orig] if orig else []
    word = list(orig[:-1]) + [orig[-1] + "</w>"]
    while len(word) > 1:
        pairs = set(zip(word[:-1], word[1:]))
        cand = [(ranks[p], p) for p in pairs if p in ranks]
        if not cand:
            break
        _, (a, b) = min(cand, key=lambda x: x[0])
        merged, i = [], 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(word[i])
                i += 1
        word = merged
    if word and word[-1] == "</w>":
        word = word[:-1]
    elif word and word[-1].endswith("</w>"):
        word[-1] = word[-1][:-4]
    return word


def _manifest_espf_tokenize(base, value, spec, cache_dir=None):
    import hashlib
    import torch
    name = str(spec.get("codebook", "drug")).strip().lower() or "drug"
    max_len = spec.get("max_len")
    s = str(value).strip()
    # Disk cache (cross-run): ESPF BPE on long protein sequences is pure-Python
    # and slow; the token ids are deterministic for (codebook, max_len, value), so
    # cache them next to the mol-graph cache. Mirrors _manifest_mol_graph.
    fp = None
    if cache_dir is not None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            ml = max_len if (isinstance(max_len, int) and max_len > 0) else 0
            h = hashlib.sha1(f"{name}|{ml}|{s}".encode("utf-8")).hexdigest()[:16]
            fp = cache_dir / f"espf_{name}_{h}.pt"
            if fp.exists():
                t = safe_torch_load(fp)
                if hasattr(t, "dtype"):
                    return t
        except UnsafePickleError:
            raise
        except Exception:
            fp = None
    cb = _load_espf_codebook(base, name)
    toks = _espf_encode(s, cb["ranks"])
    ids = [(cb["sub2id"][t] + 2) if t in cb["sub2id"] else 1 for t in toks]  # 1 = UNK
    if isinstance(max_len, int) and max_len > 0:
        ids = ids[:max_len]
    t = torch.tensor(ids or [0], dtype=torch.long)
    if fp is not None:
        try:
            torch.save(t, fp)
        except Exception:  # ESPF token disk cache is best-effort; the ids are still returned
            pass
    return t


_MANIFEST_LOOKUP_CACHE = {}


def _manifest_lookup(base, spec, value):
    """JOIN a side table by key to resolve a branch's real cell value (e.g. a
    prot_seq branch keyed by uniprot pulls the sequence from sequences.csv).
    Cached. Kept in sync with dataset_handlers._lookup_value."""
    import pandas as pd
    lk = spec.get("lookup")
    if not lk:
        return value
    key_col = str(spec.get("lookup_key", "")) or "id"
    val_col = str(spec.get("lookup_value", "")) or "value"
    lp = Path(os.path.expanduser(str(lk)))
    if not lp.is_absolute():
        lp = base / str(lk)
    ck = (str(lp), key_col, val_col)
    table = _MANIFEST_LOOKUP_CACHE.get(ck)
    if table is None:
        if lp.suffix.lower() == ".parquet":
            df = pd.read_parquet(lp)
        elif lp.suffix.lower() == ".tsv":
            df = pd.read_csv(lp, sep="\t")
        else:
            df = pd.read_csv(lp)
        table = {str(k): str(v) for k, v in zip(df[key_col], df[val_col])}
        _MANIFEST_LOOKUP_CACHE[ck] = table
    return table.get(str(value).strip(), "")


def _manifest_resolve_file(base, spec, value):
    value = str(value).strip()
    if "dir" not in spec:  # the column holds a path
        p = Path(value)
        return p if p.is_absolute() else (base / value)
    d = base / str(spec["dir"])
    ext = str(spec.get("ext", "") or "")
    if str(spec.get("match", "exact")) == "contains":
        try:
            cands = sorted(c for c in d.iterdir() if c.is_file() and value in c.name)
        except OSError:
            cands = []
        if ext:
            cands = [c for c in cands if c.suffix.lower() == ext.lower()] or cands
        return cands[0] if cands else None
    cand = d / (value if (not ext or value.endswith(ext)) else value + ext)
    return cand if cand.exists() else (d / value if (d / value).exists() else cand)


def load_manifest_graphs(ds_cfg: dict, heads: list[dict], known_classes_by_head: dict | None = None):
    """Returns (graphs_list, targets, branches, skipped). graphs_list[i] is the
    list of per-branch items for kept row i (a PyG Data for graph/molecule
    branches, a token-id LongTensor for sequence/ESPF branches); targets maps head →
    {y, classes, n_classes} aligned to graphs_list. A head with no target falls
    back to the manifest's own target column (single-task manifest). For external
    validation, `known_classes_by_head` pins each head's class order to the model."""
    import json
    import pandas as pd
    import torch
    man = Path(os.path.expanduser(ds_cfg["path"]))
    base = man.resolve().parent
    cfg = json.loads(man.read_text(encoding="utf-8"))
    tp = base / str(cfg["table"])
    suffix = tp.suffix.lower()
    df = (pd.read_parquet(tp) if suffix == ".parquet"
          else pd.read_csv(tp, sep="\t") if suffix == ".tsv" else pd.read_csv(tp))
    branches = list(cfg["pairs"].keys())
    cache_dir = (base / ".graphcache") if cfg.get("cache", True) else None
    man_tcol = (cfg.get("target") or {}).get("column")

    # Pre-load ESPF codebooks so a missing .espf cache fails loudly with an
    # actionable message instead of silently skipping every row in the loop below.
    for b in branches:
        spec = cfg["pairs"][b]
        if str(spec.get("kind", "")) == "espf":
            _load_espf_codebook(base, str(spec.get("codebook", "drug")).strip().lower() or "drug")

    # Resolve each head's column (head target wins; manifest target is the fallback).
    head_cols = []
    for h in heads:
        col = h["target"] or man_tcol
        if not col:
            raise ValueError("manifest training needs a target column (set one on each Head, or a `target` in the manifest)")
        if col not in df.columns:
            raise ValueError(f"target column {col!r} not in manifest table columns {list(df.columns)}")
        head_cols.append(col)

    # Per-branch in-memory dedup: the same value (e.g. a uniprot or canonical
    # SMILES) recurs across many rows, so memoize each branch's result keyed by the
    # raw cell value. Collapses the expensive ESPF tokenization / torch.load /
    # lookup from #rows down to #unique-values (often 10-50x on dual-encoder
    # datasets like BindingDB). Results are shared by reference — safe because the
    # collate/batch path reads them read-only (it never mutates an item in place).
    _MISS, _FAIL = object(), object()
    memo = {b: {} for b in branches}

    def resolve_branch(b, spec, raw):
        if spec.get("lookup"):  # JOIN a side table by key → real value
            raw = _manifest_lookup(base, spec, raw)
        kind = str(spec.get("kind", ""))
        if kind == "espf":
            return _manifest_espf_tokenize(base, raw, spec, cache_dir)  # ESPF ids
        if kind == "sequence":
            return _manifest_tokenize(raw, spec)  # 1-D LongTensor, not a graph
        if kind == "molecule":
            return _manifest_mol_graph(str(raw), cache_dir)
        fp = _manifest_resolve_file(base, spec, raw)
        if fp is None or not Path(fp).exists():
            raise FileNotFoundError(f"no graph file for {raw!r}")
        return safe_torch_load(fp)

    graphs_list = []
    kept_idx = []
    skipped = 0
    n_rows = len(df)
    emit("dataset.preprocess", done=0, total=n_rows, kept=0, skipped=0)
    for i in range(n_rows):
        r = df.iloc[i]
        graphs = []
        ok = True
        for b in branches:
            spec = cfg["pairs"][b]
            key = str(r[spec["column"]])
            hit = memo[b].get(key, _MISS)
            if hit is _FAIL:
                ok = False
                break
            if hit is not _MISS:
                graphs.append(hit)
                continue
            try:
                g = resolve_branch(b, spec, r[spec["column"]])
                memo[b][key] = g
                graphs.append(g)
            except UnsafePickleError:
                raise
            except Exception:
                memo[b][key] = _FAIL
                ok = False
                break
        if not ok:
            skipped += 1
        else:
            graphs_list.append(graphs)
            kept_idx.append(i)
        # Periodic progress so the run doesn't look frozen during a long single-
        # threaded preprocess (the UI tails events.jsonl; no event ≈ "stuck").
        if (i + 1) % 2000 == 0 or (i + 1) == n_rows:
            emit("dataset.preprocess", done=i + 1, total=n_rows,
                 kept=len(kept_idx), skipped=skipped)
    if not graphs_list:
        raise ValueError("manifest produced no usable paired rows (check branch columns / file paths)")

    kept_df = df.iloc[kept_idx].reset_index(drop=True)
    targets = {}
    for h, col in zip(heads, head_cols):
        y, classes, n_classes = encode_target(
            kept_df[col], h["task"], (known_classes_by_head or {}).get(h["output"]))
        targets[h["output"]] = {"y": y, "classes": classes, "n_classes": n_classes}
    return graphs_list, targets, branches, skipped


class MultiTaskDataset:
    """Yields (x, ydict): a feature row (tabular) or a per-branch Data list
    (graph), paired with {head_output → target scalar}. Plain Dataset protocol so
    random_split / DataLoader work unchanged."""

    def __init__(self, xs, targets: dict):
        self.xs = xs  # FloatTensor [N,F] (tabular) or list[list[Data]] (graph)
        self.ys = {k: v["y"] for k, v in targets.items()}
        self.n = xs.shape[0] if hasattr(xs, "shape") else len(xs)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.xs[i], {k: v[i] for k, v in self.ys.items()}


def _pad_stack(seqs):
    """A list of variable-length tensors → a padded batch, right-padded with 0 along
    dim 0. 1-D token sequences → [B, Lmax] (PAD=0); 2-D per-residue/atom features or
    embeddings ([L, D]) → [B, Lmax, D]. Equal-length items just stack."""
    import torch
    lmax = max(int(s.shape[0]) for s in seqs)
    rest = tuple(seqs[0].shape[1:])
    out = torch.zeros((len(seqs), lmax, *rest), dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        out[i, : s.shape[0]] = s
    return out


def make_collate(is_graph: bool, head_names: list[str]):
    """Collate [(x, ydict), …] → (xb, {out → stacked y}). For a manifest, xb is a
    tuple with one element PER BRANCH: a PyG Batch for graph branches, a padded
    [B, Lmax] LongTensor for sequence branches (each branch handled by its type)."""
    import torch

    def collate(batch):
        ys = {k: torch.stack([item[1][k] for item in batch]) for k in head_names}
        if is_graph:
            from torch_geometric.data import Batch
            n_branches = len(batch[0][0])
            cols = []
            for k in range(n_branches):
                items = [item[0][k] for item in batch]
                if isinstance(items[0], torch.Tensor):  # sequence branch
                    cols.append(_pad_stack(items))
                else:                                    # graph branch (PyG Data)
                    cols.append(Batch.from_data_list(items))
            xb = tuple(cols)
        else:
            xb = torch.stack([item[0] for item in batch])
        return xb, ys

    return collate


def resolve_outputs(out, heads: list[dict]) -> dict:
    """Normalise a model forward() return → {head_output → tensor}. The generated
    model returns a dict (named Output nodes), a tuple (multiple terminal nodes),
    or a single tensor (one output). Single-head is forgiving (takes the first);
    multi-head must supply one value per head (by name for a dict, by order for a
    tuple) — a single tensor with >1 head is a model/config mismatch."""
    names = [h["output"] for h in heads]
    if isinstance(out, dict):
        if len(heads) == 1:
            n = names[0]
            return {n: out[n] if n in out else next(iter(out.values()))}
        missing = [n for n in names if n not in out]
        if missing:
            raise ValueError(f"model output dict has keys {list(out.keys())} but heads need {missing} — "
                             f"name an Output node for each head")
        return {n: out[n] for n in names}
    if isinstance(out, (tuple, list)):
        if len(heads) == 1:
            return {names[0]: out[0]}
        if len(out) < len(heads):
            raise ValueError(f"model returns {len(out)} outputs but {len(heads)} heads are configured")
        return {names[i]: out[i] for i in range(len(heads))}
    # single tensor
    if len(heads) > 1:
        raise ValueError(f"model returns a single tensor but {len(heads)} heads are configured — "
                         "give each head its own Output node")
    return {names[0]: out}


def build_optimizer(cfg: dict, params):
    import torch

    kind = cfg.get("kind", "Adam")
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("weight_decay", 0.0))
    if kind == "Adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if kind == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if kind == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=float(cfg.get("momentum", 0.9)), weight_decay=wd)
    if kind == "RMSprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
    raise ValueError(f"unknown optimizer kind: {kind}")


def build_scheduler(cfg: dict, optimizer, epochs: int):
    import torch

    kind = (cfg or {}).get("kind", "none")
    if kind in (None, "none", "None"):
        return None
    sched = torch.optim.lr_scheduler
    if kind == "StepLR":
        return sched.StepLR(optimizer, step_size=int(cfg.get("step_size", 30)), gamma=float(cfg.get("gamma", 0.1)))
    if kind == "CosineAnnealingLR":
        return sched.CosineAnnealingLR(optimizer, T_max=int(cfg.get("t_max", epochs)))
    if kind == "ReduceLROnPlateau":
        return sched.ReduceLROnPlateau(optimizer, mode="min", patience=int(cfg.get("patience", 10)))
    raise ValueError(f"unknown scheduler kind: {kind}")


def compute_metrics(task: str, out, y, kinds: list[str]) -> dict:
    """Extra metrics over a full val pass. Pure torch, no sklearn dep."""
    import torch

    res: dict[str, float] = {}
    if not kinds:
        return res
    with torch.no_grad():
        if task == "regression":
            pred = out.float().view(-1)
            tgt = y.float().view(-1)
            for k in kinds:
                if k == "mse":
                    res["mse"] = float(((pred - tgt) ** 2).mean())
                elif k == "mae":
                    res["mae"] = float((pred - tgt).abs().mean())
                elif k == "r2":
                    ss_res = float(((tgt - pred) ** 2).sum())
                    ss_tot = float(((tgt - tgt.mean()) ** 2).sum()) or 1e-12
                    res["r2"] = 1.0 - ss_res / ss_tot
            return res
        # classification / binary
        if task == "binary":
            pred = (out.view(-1) > 0).long()
            n_classes = 2
        else:
            pred = out.argmax(dim=-1)
            n_classes = int(out.shape[-1])
        tgt = y.long().view(-1)
        for k in kinds:
            if k == "accuracy":
                res["accuracy"] = float((pred == tgt).float().mean())
            elif k in ("precision", "recall", "f1"):
                precs, recs, f1s = [], [], []
                for c in range(n_classes):
                    tp = float(((pred == c) & (tgt == c)).sum())
                    fp = float(((pred == c) & (tgt != c)).sum())
                    fn = float(((pred != c) & (tgt == c)).sum())
                    p = tp / (tp + fp) if (tp + fp) else 0.0
                    r = tp / (tp + fn) if (tp + fn) else 0.0
                    precs.append(p); recs.append(r)
                    f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
                if k == "precision":
                    res["precision"] = sum(precs) / len(precs)
                elif k == "recall":
                    res["recall"] = sum(recs) / len(recs)
                else:
                    res["f1"] = sum(f1s) / len(f1s)
    return res


def parse_callbacks(callbacks: list[dict]) -> dict:
    """Flatten the graph's callback list into the knobs the loop needs."""
    out = {"early_stop": None, "grad_clip": None, "amp": None}
    for cb in callbacks or []:
        kind = cb.get("kind")
        if kind == "EarlyStopping":
            out["early_stop"] = {
                "monitor": cb.get("monitor", "val_loss"),
                "patience": int(cb.get("patience", 20)),
                "mode": cb.get("mode", "min"),
            }
        elif kind == "GradientClipping":
            out["grad_clip"] = float(cb.get("max_norm", 1.0))
        elif kind == "MixedPrecision":
            out["amp"] = cb.get("dtype", "bf16")
    return out


def sample_predictions(task: str, out, y, classes, k: int = 12) -> list[dict]:
    """A handful of evenly-spaced val predictions vs. ground truth, for the
    Run-Detail "Vorhersagen" tab. Pure torch, bounded size."""
    import torch

    n = int(out.shape[0])
    if n == 0:
        return []
    k = min(k, n)
    idx = torch.linspace(0, n - 1, k).round().long()
    out_s = out[idx]
    y_s = y[idx]
    rows: list[dict] = []
    if task == "regression":
        pred = out_s.float().view(-1)
        tgt = y_s.float().view(-1)
        for i in range(k):
            rows.append({"pred": round(float(pred[i]), 4), "truth": round(float(tgt[i]), 4)})
        return rows
    if task == "binary":
        prob = torch.sigmoid(out_s.view(-1))
        pred = (prob > 0.5).long()
        conf = torch.where(pred.bool(), prob, 1.0 - prob)
    else:
        probs = torch.softmax(out_s.float(), dim=-1)
        conf, pred = probs.max(dim=-1)
    tgt = y_s.long().view(-1)
    for i in range(k):
        pi = int(pred[i]); ti = int(tgt[i])
        rows.append({
            "pred": classes[pi] if classes and pi < len(classes) else pi,
            "truth": classes[ti] if classes and ti < len(classes) else ti,
            "conf": round(float(conf[i]), 4),
            "correct": pi == ti,
        })
    return rows


def eval_summary(task: str, out, y, classes, max_points: int = 2000) -> dict:
    """Full-val-set evaluation payload that drives the Run-Detail diagrams. The
    SHAPE is fixed by the task (decided by the loss): classification/binary →
    a confusion matrix (rows = truth, cols = pred) + class labels; regression →
    (pred, truth) scatter points (bounded) + the total count. Pure torch, no
    sklearn. Returns just {"task": …} when there's nothing meaningful to plot."""
    import torch

    n = int(out.shape[0])
    if n == 0:
        return {"task": task}
    with torch.no_grad():
        if task == "regression":
            pred = out.float().view(-1)
            tgt = y.float().view(-1)
            if n > max_points:  # evenly subsample so the scatter stays light
                idx = torch.linspace(0, n - 1, max_points).round().long()
                pred, tgt = pred[idx], tgt[idx]
            pts = [[round(float(pred[i]), 5), round(float(tgt[i]), 5)] for i in range(pred.shape[0])]
            return {"task": "regression",
                    "scatter": {"points": pts, "n_total": n,
                                "pred_label": "Vorhersage", "truth_label": "Wahrheit"}}
        # classification / binary → confusion matrix
        if task == "binary":
            pred = (out.view(-1) > 0).long()
            n_classes = 2
        else:
            pred = out.argmax(dim=-1)
            n_classes = int(out.shape[-1])
        tgt = y.long().view(-1)
        n_classes = max(n_classes, (int(tgt.max().item()) + 1) if tgt.numel() else n_classes, 2)
        if n_classes > 100:  # too many classes for a readable matrix
            return {"task": task}
        # Vectorized confusion matrix via bincount over flattened (truth, pred).
        flat = tgt.clamp(0, n_classes - 1) * n_classes + pred.clamp(0, n_classes - 1)
        cm = torch.bincount(flat, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
        labels = [str(classes[c]) if classes and c < len(classes) else str(c)
                  for c in range(n_classes)]
        return {"task": task, "confusion": {"labels": labels, "matrix": cm.tolist()}}


def to_device(obj, device):
    """Move a batch to `device`, recursing into the shapes our loaders yield:
    a bare tensor (non-graph xb), a tuple of branches (graph xb — each a PyG
    Batch or padded tensor), or a dict (yb, name → target tensor). PyG Batch and
    plain tensors both expose ``.to``. No-op on CPU so we never pay a copy."""
    if device.type == "cpu":
        return obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(o, device) for o in obj)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if hasattr(obj, "to"):
        return obj.to(device)
    return obj


def evaluate(model, loader, heads, head_names, multitask, forward_loss, batch_len, metric_kinds, device):
    """Run `model` over `loader` once → (val_loss, val_acc, extra, val_cat). Pure:
    no events, no checkpoints. val_cat[output] = (cat_out, cat_y) over that head's
    LABELED rows, feeding eval_summary / sample_predictions. Shared by the training
    per-epoch validation AND eval-only external validation."""
    import torch

    model.eval()
    vrun, vseen = 0.0, 0
    hloss = {n: 0.0 for n in head_names}
    hcorrect = {n: 0 for n in head_names}
    hcount = {n: 0 for n in head_names}   # labeled rows per head
    houts = {n: [] for n in head_names}
    hys = {n: [] for n in head_names}
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = to_device(xb, device), to_device(yb, device)
            per, loss = forward_loss(xb, yb)
            bs = batch_len(xb)
            vrun += float(loss.item()) * bs
            vseen += bs
            for h in heads:
                n = h["output"]
                o, l, valid = per[n]
                nv = int(valid.sum().item())
                hloss[n] += float(l.item()) * max(1, nv)
                hcount[n] += nv
                if nv == 0:
                    continue
                ov, yv = o[valid], yb[n][valid]
                houts[n].append(ov.float())
                hys[n].append(yv)
                if h["task"] == "classification":
                    hcorrect[n] += int((ov.argmax(dim=-1) == yv.long()).sum().item())
                elif h["task"] == "binary":
                    hcorrect[n] += int(((ov > 0).long() == yv.long()).sum().item())
    val_loss = vrun / max(1, vseen)
    val_acc = None
    extra: dict = {}
    val_cat: dict = {}
    for h in heads:
        n = h["output"]
        # Back to CPU: the metric/summary helpers index with CPU linspace tensors
        # and the eval payloads serialize to JSON — keep them off the GPU.
        cat_out = torch.cat(houts[n]).cpu() if houts[n] else None
        cat_y = torch.cat(hys[n]).cpu() if hys[n] else None
        if cat_out is not None:
            val_cat[n] = (cat_out, cat_y)
        denom = max(1, hcount[n])  # over LABELED rows for this head
        acc = hcorrect[n] / denom if h["task"] in ("classification", "binary") else None
        hl = hloss[n] / denom
        hm = compute_metrics(h["task"], cat_out, cat_y, metric_kinds) if (cat_out is not None and metric_kinds) else {}
        if multitask:
            extra[f"{n}/loss"] = round(hl, 6)
            if acc is not None:
                extra[f"{n}/acc"] = round(acc, 6)
            for k, v in hm.items():
                extra[f"{n}/{k}"] = round(v, 6)
        else:
            val_acc = acc
            extra = {k: round(v, 6) for k, v in hm.items()}
    return val_loss, val_acc, extra, val_cat


def emit_eval(epoch, heads, head_names, multitask, head_classes, val_cat):
    """Emit sample.preds + eval.summary for the per-head (cat_out, cat_y) from
    evaluate(). The Run-Detail UI renders these (confusion/scatter + predictions)."""
    if not val_cat:
        return
    task_of = {h["output"]: h["task"] for h in heads}
    if multitask:
        emit("sample.preds", epoch=epoch, heads=[
            {"output": n, "task": task_of[n],
             "rows": sample_predictions(task_of[n], val_cat[n][0], val_cat[n][1], head_classes.get(n))}
            for n in head_names if n in val_cat])
        emit("eval.summary", epoch=epoch, heads=[
            {"output": n, **eval_summary(task_of[n], val_cat[n][0], val_cat[n][1], head_classes.get(n))}
            for n in head_names if n in val_cat])
    else:
        n = head_names[0]
        t = heads[0]["task"]
        emit("sample.preds", epoch=epoch,
             rows=sample_predictions(t, val_cat[n][0], val_cat[n][1], head_classes.get(n)))
        emit("eval.summary", epoch=epoch,
             **eval_summary(t, val_cat[n][0], val_cat[n][1], head_classes.get(n)))


def main() -> None:
    # Phase 32 — graceful cancellation on SIGTERM/SIGINT. The UI stop path
    # (stop_training_run / ssh_stop_training_run) writes status=cancelled FIRST,
    # then signals us; default signal handling would kill the process mid-epoch,
    # losing the resumable last.pt and the run.cancelled event (breaking the
    # Phase-26 stop → load → resume promise). We instead unwind to a single
    # cancellation point — `_finish_cancel` is terminal-shielded, so a late
    # signal after done/failed can never resurrect the run (the Phase-30
    # CANCELLED→SUCCEEDED / SUCCEEDED→CANCELLED prohibition holds for signals too).
    def _on_cancel(signum, frame):
        if _SHUTDOWN["v"]:
            return  # duplicate signal during unwinding/exiting — ignore
        _SHUTDOWN["v"] = True
        raise _Cancelled(signum)

    signal.signal(signal.SIGTERM, _on_cancel)
    signal.signal(signal.SIGINT, _on_cancel)

    # Phase 30 — queued→running via the state machine; if the launch wrote
    # 'cancelled' first (cancel-before-start race) the transition is rejected.
    # Phase 32 — record that pre-termination cancellation and exit cleanly
    # instead of paying the startup cost to discover it at the first epoch check.
    if not transition_status("running"):
        if _read_status() == "cancelled":
            _finish_cancel(0, "cancelled before start")
            return
        if _read_status() == _STATUS_UNREADABLE:
            # Do not run with an unknown run state: the terminal-state machine
            # cannot protect a file it cannot read. Fail explicitly instead.
            fail("config", "run status file is unreadable — refusing to start with an unknown state")
    t0 = time.time()
    emit("run.start", pid=os.getpid())

    # ── config ──
    try:
        cfg = json.loads((RUN_DIR / "run.json").read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        fail("config", f"cannot read run.json: {e}", traceback.format_exc())

    train_cfg = cfg.get("training", {})
    ds_cfg = cfg.get("dataset", {})
    epochs = int(train_cfg.get("epochs", 10))
    batch_size = int(train_cfg.get("batch_size", 32))
    val_split = float(train_cfg.get("val_split", 0.2))
    seed = int(train_cfg.get("seed", 42))
    split_strategy = str(train_cfg.get("split_strategy", "random")).lower()

    # Phase 19 — we NEVER silently change a user's split strategy. The trainer
    # only implements 'random' today; any other strategy frozen into run.json is
    # an explicit, loud failure instead of a silent random fallback.
    IMPLEMENTED_SPLIT_STRATEGIES = ("random",)
    if split_strategy not in IMPLEMENTED_SPLIT_STRATEGIES:
        fail("split",
             f"split_strategy={split_strategy!r} is not implemented by the trainer yet "
             f"(implemented: {', '.join(IMPLEMENTED_SPLIT_STRATEGIES)}). The run's strategy "
             f"was frozen into run.json at launch and is honored verbatim — it will NOT be "
             f"silently changed to 'random'.")
    log_every = int(train_cfg.get("log_every_n_steps", 10))
    shuffle = bool(train_cfg.get("shuffle", True))
    num_workers = int(train_cfg.get("num_workers", 0))
    drop_last = bool(train_cfg.get("drop_last", False))
    val_every = max(1, int(train_cfg.get("val_every_n_epochs", 1)))
    accum_steps = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))

    # ── imports (heavy) ──
    try:
        import torch
        from torch.utils.data import DataLoader, random_split
    except Exception as e:  # noqa: BLE001
        fail("import-torch", f"torch import failed: {e}", traceback.format_exc())

    # ── seeding (Phase 22) — cover every known random source so a second run
    #    with the same seed + same stack produces the same weights/metrics.
    #    Full determinism is NOT possible with CUDA (atomicAdd nondeterminism)
    #    and some third-party ops; we document this via the run.determinism
    #    event instead of claiming "fully reproducible". ──
    import random as _random
    try:
        import numpy as _np
        _np.random.seed(seed)
    except ImportError:
        _np = None  # type: ignore[assignment]
    _random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # flag nondeterministic CUDA ops as warnings (don't crash — document only)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (TypeError, AttributeError):
        pass  # older torch without warn_only

    def _seed_worker(worker_id: int) -> None:
        """Seed Python stdlib + NumPy in each DataLoader worker so that
        multi-worker shuffling is reproducible across runs with the same seed."""
        _worker_seed = (seed + worker_id) % (2**32)
        _random.seed(_worker_seed)
        if _np is not None:
            _np.random.seed(_worker_seed)
        torch.manual_seed(_worker_seed)

    # Determinism event: document what was set + caveats for the record.
    _determinism: dict = {
        "seed": seed,
        "python_random": True,
        "numpy_random": _np is not None,
        "torch_cpu": True,
        "torch_cuda_all": torch.cuda.is_available(),
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cudnn_enabled": torch.backends.cudnn.enabled,
    }
    if torch.cuda.is_available():
        _determinism["cuda_nondeterministic_ops_possible"] = True
        _determinism["cuda_nondeterministic_ops_note"] = (
            "atomicAdd in float reductions and scatter_add are nondeterministic "
            "on CUDA even with deterministic_algorithms=True; full GPU "
            "determinism requires CPU execution or torch>=2.0 with use_deterministic_algorithms(True)."
        )
    emit("run.determinism", **_determinism)

    # ── snapshot (Phase 20) — fail-closed: the frozen model.spinoml + model.py
    #    in the RUN DIR must be byte-identical to the launch-time freeze, else
    #    the running experiment would depend on post-launch mutation. Checked
    #    before heavy imports/training so nothing executes on drifted bytes. ──
    snap_result = _verify_snapshot(cfg)
    emit("run.snapshot", **snap_result)
    if snap_result.get("ok") is False:
        fail("snapshot",
             f"run dir artifacts mutated after launch — model.spinoml and/or model.py "
             f"no longer match the frozen snapshot. Refusing to train on drifted code. "
             f"(graph match={snap_result['graph']['matches']}, "
             f"model.py match={snap_result['model_py']['matches']})")

    # ── external validation (eval-only): load the SOURCE checkpoint up-front so the
    #    external target is encoded against the model's TRAINED class order. ──
    eval_only = bool(cfg.get("eval_only"))
    eval_ckpt = None
    known_classes_by_head = None
    if eval_only:
        try:
            ck = (cfg.get("validate") or {}).get("checkpoint_from")
            if not ck:
                fail("validate", "eval_only run has no validate.checkpoint_from")
            cp = Path(os.path.expanduser(str(ck)))
            if not cp.is_absolute():
                cp = WORKSPACE_ROOT / str(ck)
            eval_ckpt = safe_torch_load(cp)
            if isinstance(eval_ckpt, dict):
                known_classes_by_head = eval_ckpt.get("head_classes") or None
        except Exception as e:  # noqa: BLE001
            fail("validate", f"cannot read checkpoint for validation: {e}", traceback.format_exc())

    # ── heads (multitask) — one head per model output, single-task = one head ──
    heads, multitask = resolve_heads(train_cfg, ds_cfg)
    head_names = [h["output"] for h in heads]
    # classes per head, filled by dataset loading (None for regression heads).
    head_classes: dict = {h["output"]: None for h in heads}

    # ── provenance (Phase 18) — record WHICHT dataset + split this run pinned,
    #    independent of human-readable names, before anything can fail. ──
    fp = ds_cfg.get("fingerprint")
    emit("run.provenance",
         dataset=ds_cfg.get("relpath") or ds_cfg.get("path"),
         model=cfg.get("model_path"),
         ds_kind=ds_cfg.get("kind"),
         fingerprint_id=(f"{fp['alg']}:{fp['hash']}" if isinstance(fp, dict) and fp.get("hash") else None),
         unsafe_pickle=unsafe_pickle_allowed(),
         split={"val_split": val_split, "seed": seed, "strategy": split_strategy})
    try:
        _manifest_init(cfg)
    except Exception as e:  # noqa: BLE001
        emit("manifest.error", stage="init", error=str(e))
    try:
        emit("dataset.fingerprint", **_verify_fingerprint(ds_cfg))
    except Exception as e:  # noqa: BLE001
        emit("dataset.fingerprint", ok=False, error=type(e).__name__)

    # ── dataset ──
    is_graph = False
    try:
        kind = ds_cfg.get("kind", "tabular")
        if kind == "manifest":
            # Paired graph dataset (e.g. ligand + protein) → a model with one
            # graph input per branch (forward(self, branch0, branch1, …)).
            is_graph = True
            graphs_list, targets, branches, skipped = load_manifest_graphs(ds_cfg, heads, known_classes_by_head)
            full = MultiTaskDataset(graphs_list, targets)
            emit("dataset.loaded", n_rows=len(graphs_list), branches=branches,
                 fingerprint_id=(f"{fp['alg']}:{fp['hash']}" if isinstance(fp, dict) and fp.get("hash") else None),
                 heads=[{"output": h["output"], "task": h["task"]} for h in heads], skipped=skipped)
        elif kind == "tabular":
            X, targets, feature_cols = load_tabular(ds_cfg, heads, known_classes_by_head)
            full = MultiTaskDataset(X, targets)
            emit("dataset.loaded", n_rows=int(X.shape[0]), n_features=int(X.shape[1]),
                 fingerprint_id=(f"{fp['alg']}:{fp['hash']}" if isinstance(fp, dict) and fp.get("hash") else None),
                 heads=[{"output": h["output"], "task": h["task"],
                         "n_classes": targets[h["output"]]["n_classes"]} for h in heads],
                 features=feature_cols)
        else:
            raise ValueError(f"unsupported dataset kind {kind!r} — use 'tabular' or 'manifest'.")
        for h in heads:
            head_classes[h["output"]] = targets[h["output"]]["classes"]
    except Exception as e:  # noqa: BLE001
        fail("dataset", str(e), traceback.format_exc())

    # ── split ──
    n_val = max(1, int(len(full) * val_split)) if val_split > 0 else 0
    n_train = len(full) - n_val
    if n_train <= 0:
        fail("split", f"not enough rows ({len(full)}) for the chosen val_split={val_split}")
    gen = torch.Generator().manual_seed(seed)
    if n_val > 0:
        train_ds, val_ds = random_split(full, [n_train, n_val], generator=gen)
    else:
        train_ds, val_ds = full, None

    # ── split integrity (Phase 19) — prove no overlap between partition classes ──
    # random_split is structurally a permutation partition (disjoint), so
    # overlap is provably 0 today; the check is a fail-closed assertion that
    # stays live when grouped/stratified/predefined strategies are added later.
    train_indices = list(getattr(train_ds, "indices", range(n_train)))
    val_indices = list(getattr(val_ds, "indices", [])) if val_ds is not None else []
    overlaps = sorted(set(train_indices) & set(val_indices))
    emit("split.integrity",
         train_size=len(train_indices), val_size=len(val_indices),
         overlap=len(overlaps), overlaps=overlaps[:20],
         strategy=split_strategy, seed=seed, val_split=val_split)
    if overlaps:
        fail("split",
             f"train/val leakage — {len(overlaps)} sample(s) appear in BOTH partitions "
             f"(strategy={strategy}, seed={seed}); refusing to train on a leaking split")

    collate = make_collate(is_graph, head_names)
    # DataLoader generator: separate from the model-seed torch.Generator so that
    # shuffle order is independently reproducible.  worker_init_fn seeds Python
    # stdlib + NumPy inside each forked worker (torch itself is seeded via the
    # forked state, but random/numpy are not).
    _dl_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, drop_last=drop_last,
                             collate_fn=collate, generator=_dl_generator,
                             worker_init_fn=_seed_worker if num_workers > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers,
                            collate_fn=collate, generator=_dl_generator,
                            worker_init_fn=_seed_worker if num_workers > 0 else None) if val_ds is not None else None

    # ── model + optimizer ──
    try:
        sys.path.insert(0, str(RUN_DIR))
        from model import Model  # type: ignore

        model = Model()
        # Pick the compute device once and move the model onto it BEFORE the dummy
        # forward and the optimizer — so lazy layers materialize their params on
        # the GPU and the optimizer binds the on-device params. Without this the
        # whole run silently stays on CPU even with gpu:1 + a CUDA module loaded.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # Lazy layers (in_channels=-1) only materialize their params on the first
        # forward — run one dummy batch BEFORE building the optimizer (else those
        # params are never registered with it) and before counting params.
        try:
            xb0, _ = next(iter(train_loader))
            xb0 = to_device(xb0, device)
            with torch.no_grad():
                model(*xb0) if is_graph else model(xb0)
            model.to(device)  # re-pin lazily-materialized params onto the device
        except StopIteration:
            # An empty training loader (e.g. drop_last=True with batch_size >
            # training rows) means there is NOTHING to train on. Silently
            # skipping the dummy forward would let every epoch report
            # train_loss=0.0 and the run end "done" — a false SUCCESS. Fail loud.
            fail("split", "training loader is empty (drop_last=True with batch_size larger "
                          "than the training split?) — refusing to report an untrained run as done")
        optimizer = build_optimizer(train_cfg.get("optimizer", {}), model.parameters())
        scheduler = build_scheduler(train_cfg.get("scheduler", {}), optimizer, epochs)
        n_params = sum(p.numel() for p in model.parameters())
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
        emit("model.built", n_params=int(n_params), device=device.type, gpu=gpu_name)
    except Exception as e:  # noqa: BLE001
        fail("model", str(e), traceback.format_exc())

    # ── metrics + callbacks (Phase 14 training-graph) ──
    metric_kinds = train_cfg.get("metrics") or []
    cb = parse_callbacks(train_cfg.get("callbacks") or [])
    grad_clip = cb["grad_clip"]
    early = cb["early_stop"]
    amp_dtype = None
    if cb["amp"]:
        amp_dtype = torch.bfloat16 if cb["amp"] == "bf16" else torch.float16
    use_amp = amp_dtype is not None
    device_type = device.type
    if metric_kinds or grad_clip or early or use_amp:
        emit("config.extras", metrics=metric_kinds, grad_clip=grad_clip,
             early_stopping=early, amp=cb["amp"])
    # Phase 21 — record the RUNTIME environment once (software versions, device/
    # dtype, hardware, optional workspace git commit) so a run's outcome can be
    # attributed to the exact stack it actually executed on.
    try:
        env_info = _env_info(torch, device, cb["amp"], WORKSPACE_ROOT)
        emit("config.env", **env_info)
    except Exception as e:  # noqa: BLE001
        env_info = {}
        emit("config.env", error=str(e))
    try:
        _software, _hardware = _manifest_env_split(env_info)
        _manifest_write("config.env", device=device.type, dtype=env_info.get("dtype"),
                        software=_software, hardware=_hardware)
    except Exception as e:  # noqa: BLE001
        emit("manifest.error", stage="config.env", error=str(e))
    es_best = float("inf") if (early and early["mode"] == "min") else float("-inf")
    es_wait = 0

    def prep_target(t, task):
        return t.long() if task == "classification" else t.float()

    def shape_out(o, task):
        # squeeze a trailing singleton for regression/binary so [N,1] matches [N].
        if task in ("regression", "binary"):
            return o.squeeze(-1) if o.dim() > 1 and o.shape[-1] == 1 else o
        return o

    def batch_len(xb):
        # manifest mode: xb is a tuple with one element PER BRANCH — a PyG Batch
        # (graph) OR a padded tensor (sequence/embedding). Read the batch size from
        # whichever the first branch is (all branches share it).
        if not is_graph:
            return xb.shape[0]
        first = xb[0]
        return first.num_graphs if hasattr(first, "num_graphs") else int(first.shape[0])

    def forward_loss(xb, yb):
        """Returns (per, total) where per[output] = (shaped_out, head_loss,
        valid_mask) and total is the weighted sum the optimizer steps on. Each
        head's loss is computed ONLY over rows that have a target for it
        (masked / partial-label multitask), so e.g. an affinity head trains on
        binders while decoys (empty target) are skipped."""
        out = model(*xb) if is_graph else model(xb)
        outs = resolve_outputs(out, heads)
        total = None
        per = {}
        for h in heads:
            name, task = h["output"], h["task"]
            o = shape_out(outs[name], task)
            t = prep_target(yb[name], task)
            # rows with a missing target for this head don't contribute to it
            valid = (t >= 0) if task == "classification" else ~torch.isnan(t)
            if bool(valid.all()):
                l = h["loss_fn"](o, t)
            elif bool(valid.any()):
                l = h["loss_fn"](o[valid], t[valid])
            else:
                l = o.sum() * 0.0  # nothing labeled this batch → no contribution
            contrib = l * h["weight"]
            total = contrib if total is None else total + contrib
            per[name] = (o, l, valid)
        return per, total

    CKPT_DIR.mkdir(exist_ok=True)

    # ── eval-only (external validation) ──────────────────────────────────────
    # Load the source checkpoint into the model and evaluate the WHOLE external
    # dataset once — emit the same eval.summary/sample.preds the Run-Detail UI
    # renders, write metrics.json, and return. No optimizer/scheduler/training.
    if eval_only:
        try:
            state = eval_ckpt.get("model_state", eval_ckpt) if isinstance(eval_ckpt, dict) else eval_ckpt
            msd = model.state_dict()
            compat = {k: v for k, v in state.items()
                      if k in msd and hasattr(v, "shape") and tuple(v.shape) == tuple(msd[k].shape)}
            model.load_state_dict(compat, strict=False)
            n_missing = len(msd) - len(compat)
            emit("checkpoint.loaded", source=str((cfg.get("validate") or {}).get("checkpoint_from")),
                 loaded=len(compat), missing=n_missing)
            if n_missing:
                emit("validation.warning",
                     message=f"{n_missing} model tensors had no matching checkpoint weight "
                             "(architecture mismatch) — they stayed at their init values.")
            eval_loader = DataLoader(full, batch_size=batch_size, num_workers=num_workers, collate_fn=collate,
                                   generator=_dl_generator,
                                   worker_init_fn=_seed_worker if num_workers > 0 else None)
            val_loss, val_acc, extra, val_cat = evaluate(
                model, eval_loader, heads, head_names, multitask, forward_loss, batch_len, metric_kinds, device)
            # Phase 25 — external validation on a NaN/inf signal must not be
            # reported as success either.
            require_finite("val loss", val_loss, "eval-only")
            if val_acc is not None:
                require_finite("val accuracy", val_acc, "eval-only")
            for mn, mv in (extra or {}).items():
                require_finite(f"val metric {mn}", mv, "eval-only")
            emit("epoch.end", epoch=0, train_loss=0.0,
                 val_loss=round(val_loss, 6), val_acc=None if val_acc is None else round(val_acc, 6),
                 metrics=extra or None,
                 **({"heads": [{"output": h["output"], "task": h["task"]} for h in heads]} if multitask else {}),
                 lr=0.0)
            emit_eval(0, heads, head_names, multitask, head_classes, val_cat)
            emit("validation.summary", n_rows=len(full), val_loss=round(val_loss, 6), metrics=extra or None)
            total = time.time() - t0
            # Write metrics.json BEFORE the integrity gate — the gate reads it.
            METRICS.write_text(json.dumps({
                "status": "done", "eval_only": True, "total_seconds": round(total, 2),
                "best_val_loss": round(val_loss, 6), "n_rows": len(full),
                "n_params": int(n_params), "metrics": extra or None,
            }, indent=2))
            # Phase 73 — integrity gate (eval-only form: no checkpoint required).
            integ = _verify_run_integrity(eval_only=True)
            emit("run.integrity", ok=integ["ok"], missing=integ["missing"],
                 invalid=integ["invalid"], notes=integ["notes"])
            if not integ["ok"]:
                problems = []
                if integ["missing"]:
                    problems.append("missing: " + "; ".join(integ["missing"]))
                if integ["invalid"]:
                    problems.append("invalid: " + "; ".join(integ["invalid"]))
                fail("integrity", " | ".join(problems) or "integrity check failed")
                return
            emit("run.done", total_seconds=round(total, 2), best_val_loss=round(val_loss, 6))
            set_status("done")
            _manifest_write("done", status="done", finished_at=_now(),
                            summary=_manifest_summary(val_loss, 0, int(n_params)))
            # Phase 74 — eval-only runs are never resumable (they read a foreign
            # checkpoint, they don't produce one); record it so the UI is honest.
            _record_terminal_resumable("done")
            return
        except Exception as e:  # noqa: BLE001
            fail("validate", str(e), traceback.format_exc())

    best_val = float("inf")
    start_epoch = 0

    # ── resume from a prior checkpoint (Phase 17; RNG/global_step Phase 26) ──
    resume_from = cfg.get("resume_from") or train_cfg.get("resume_from")
    global_step = 0
    if resume_from:
        try:
            rp = Path(os.path.expanduser(str(resume_from)))
            if not rp.is_absolute():
                rp = WORKSPACE_ROOT / str(resume_from)
            ckpt = safe_torch_load(rp)
            model.load_state_dict(ckpt["model_state"])
            if "optim_state" in ckpt:
                optimizer.load_state_dict(ckpt["optim_state"])
            if scheduler is not None and ckpt.get("sched_state") is not None:
                scheduler.load_state_dict(ckpt["sched_state"])
            best_val = float(ckpt.get("best_val", best_val))
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            global_step = int(ckpt.get("global_step", 0))
            # Phase 26 — continue from the saved random streams where available.
            rng_restore = _restore_rng(ckpt["rng"]) if isinstance(ckpt.get("rng"), dict) else {"all": "absent in checkpoint"}
            emit("run.resumed", source=str(resume_from), start_epoch=start_epoch,
                 global_step=global_step, prev_val_loss=ckpt.get("val_loss"),
                 rng_restore=rng_restore)
        except Exception as e:  # noqa: BLE001
            fail("resume", f"cannot resume from {resume_from!r}: {e}", traceback.format_exc())

    end_epoch = start_epoch + epochs

    def build_ckpt(epoch: int, val_loss_v=None) -> dict:
        """Phase 26 — a checkpoint preserves everything a resume needs:
        model/optimizer/scheduler state, epoch, global step, RNG streams
        (where supported), and the frozen experiment configuration."""
        return {
            "epoch": epoch,
            "global_step": int(global_step),
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict() if scheduler is not None else None,
            "best_val": best_val,
            "val_loss": val_loss_v,
            # single-task keeps the flat `classes` (Explain viz reads it);
            # multitask records per-head class lists too.
            "classes": (None if multitask else head_classes.get(head_names[0])),
            "head_classes": head_classes,
            "rng": _rng_state(),
            "config": cfg,
        }

    current_epoch: list[int | None] = [None]  # Phase 32 — signal-safe epoch tracker
    try:
        for epoch in range(start_epoch, end_epoch):
            current_epoch[0] = epoch
            if _read_status() == "cancelled":
                # Phase 26 — a stopped run must leave a resumable checkpoint
                # (last completed epoch) so stop → load → resume works.
                last_done = epoch - 1
                try:
                    _atomic_save(build_ckpt(last_done), CKPT_DIR / "last.pt")
                except Exception:  # noqa: BLE001  # save is best-effort; resumable check reports honestly
                    pass
                # Phase 30 — cancel through the state machine (idempotent;
                # double cancellation is a no-op after the first terminal write).
                _finish_cancel(epoch, "cancelled at epoch boundary")
                return
            emit("epoch.start", epoch=epoch)
            model.train()
            running = 0.0
            n_seen = 0
            step = 0
            optimizer.zero_grad()
            for xb, yb in train_loader:
                global_step += 1
                xb, yb = to_device(xb, device), to_device(yb, device)
                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                    _, loss = forward_loss(xb, yb)
                # Phase 25 — a NaN/inf loss poisons everything that follows:
                # stop immediately with an explicit reason, never report success.
                require_finite("train loss", loss, f"epoch={epoch} step={step}")
                # Accumulate over accum_steps batches → larger effective batch on
                # limited memory; step (and zero) only at the window boundary.
                (loss / accum_steps).backward()
                if (step + 1) % accum_steps == 0:
                    # Phase 25 — NaN/inf gradients (e.g. exploding/vanishing)
                    # make the update garbage; fail loudly rather than stepping.
                    bad_grads = [(n, float(p.grad.abs().max())) for n, p in model.named_parameters()
                                 if p.grad is not None and not _is_finite(p.grad)]
                    if bad_grads:
                        n0, _ = bad_grads[0]
                        fail("numeric", f"non-finite gradient after backward for '{n0}' "
                                        f"at epoch={epoch} step={step}. Non-finite params: "
                                        f"{[n for n, _ in bad_grads[:8]]}. "
                                        "Result is unusable; not reporting success.")
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                bs = batch_len(xb)
                running += float(loss.item()) * bs
                n_seen += bs
                step += 1
                if step % log_every == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    emit("batch", epoch=epoch, step=step, loss=round(float(loss.item()), 6), lr=lr)
            # Flush a trailing partial accumulation window.
            if step % accum_steps != 0:
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            train_loss = running / max(1, n_seen)

            # ── validation ── (shared evaluate(); also used by eval-only runs)
            val_loss = None   # combined (weighted) val loss — the monitored metric
            val_acc = None    # single-task convenience (top-level); None in multitask
            extra: dict = {}  # per-metric floats (multitask keys are "<output>/<metric>")
            val_cat = None     # per-head {output → (cat_out, cat_y)} for eval payloads
            # Validate every val_every epochs (always on the final epoch). On a
            # skipped epoch val_loss stays None — handled like the no-val-split case.
            do_val = (epoch + 1) % val_every == 0 or epoch == end_epoch - 1
            if val_loader is not None and do_val:
                val_loss, val_acc, extra, val_cat = evaluate(
                    model, val_loader, heads, head_names, multitask, forward_loss, batch_len, metric_kinds, device)
                # Phase 25 — a NaN/inf VAL metric makes the monitored curve (and
                # anything derived from it: early stop, best-val, checkpointing)
                # garbage. Fail loudly instead of reporting success on a broken
                # validation signal.
                if do_val:
                    require_finite("val loss", val_loss, f"epoch={epoch}")
                    if val_acc is not None:
                        require_finite("val accuracy", val_acc, f"epoch={epoch}")
                    for mn, mv in (extra or {}).items():
                        require_finite(f"val metric {mn}", mv, f"epoch={epoch}")

            monitor = val_loss if val_loss is not None else train_loss
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(monitor)
                else:
                    scheduler.step()

            emit("epoch.end", epoch=epoch, train_loss=round(train_loss, 6),
                 val_loss=None if val_loss is None else round(val_loss, 6),
                 val_acc=None if val_acc is None else round(val_acc, 6),
                 metrics=extra or None,
                 **({"heads": [{"output": h["output"], "task": h["task"]} for h in heads]} if multitask else {}),
                 lr=optimizer.param_groups[0]["lr"])

            # ── checkpoint best ──
            if monitor < best_val:
                best_val = monitor
                _atomic_save(build_ckpt(epoch, val_loss_v=val_loss), CKPT_DIR / "best.pt")
                emit("checkpoint", epoch=epoch, path="checkpoints/best.pt",
                     val_loss=None if val_loss is None else round(val_loss, 6), is_best=True)
                emit_eval(epoch, heads, head_names, multitask, head_classes, val_cat)

            # ── early stopping ──
            if early is not None:
                cur = {"val_loss": val_loss, "val_acc": val_acc, "train_loss": train_loss}.get(early["monitor"])
                if cur is not None:
                    improved = cur < es_best - 1e-9 if early["mode"] == "min" else cur > es_best + 1e-9
                    if improved:
                        es_best = cur
                        es_wait = 0
                    else:
                        es_wait += 1
                        if es_wait >= early["patience"]:
                            emit("run.earlystop", epoch=epoch, monitor=early["monitor"], best=round(es_best, 6))
                            break

        _atomic_save(build_ckpt(end_epoch - 1, val_loss_v=val_loss), CKPT_DIR / "last.pt")
    except _Cancelled as c:  # Phase 32 — graceful signal cancellation, any point in the epoch
        # The signal unwound us mid-epoch/mid-validation. Leave a resumable
        # checkpoint at the last COMPLETED epoch and record the cancellation —
        # unless the run already reached a final state, in which case the
        # terminal state wins and nothing is overwritten.
        canc_epoch = current_epoch[0]
        last_done = (canc_epoch if canc_epoch is not None else start_epoch) - 1
        try:
            _atomic_save(build_ckpt(last_done), CKPT_DIR / "last.pt")
        except Exception:  # noqa: BLE001  # save is best-effort; resumable check reports honestly
            pass
        _finish_cancel(canc_epoch or 0, f"cancelled (signal {c})")
        return
    except Exception as e:  # noqa: BLE001
        # Phase 74 — save last.pt at the last COMPLETED epoch (if any) so the
        # failed run is resumable. Refuse to save when no epoch finished
        # (last_done < start_epoch): a negative-epoch checkpoint would mislead
        # _compute_resumable into a false "checkpoint load verification skipped
        # (size)" or similar reason.
        canc_epoch = current_epoch[0]
        last_done = (canc_epoch if canc_epoch is not None else start_epoch) - 1
        if last_done >= start_epoch and last_done >= 0:
            try:
                _atomic_save(build_ckpt(last_done), CKPT_DIR / "last.pt")
            except Exception:  # noqa: BLE001  # save is best-effort; resumable check reports honestly
                pass
        fail("train", str(e), traceback.format_exc())

    total = time.time() - t0
    try:
        env_summary = _env_info(torch, device, cb["amp"], WORKSPACE_ROOT)
    except Exception:  # noqa: BLE001
        env_summary = {}
    # Phase 30 — running→done via the state machine. A cancellation racing in
    # after the last epoch check (user cancelled while we were finalising) makes
    # the terminal 'cancelled' win: run.done is NOT emitted, a run.cancelled
    # event is, and the run is never reported as succeeded after a cancel.
    # CANCELLED → SUCCEEDED is impossible.
    # Phase 73 — write metrics.json BEFORE the integrity gate so the gate has
    # the (status=done, full payload) to read. The gate decides whether the
    # terminal 'done' transition + run.done event are actually emitted; on
    # failure it calls fail("integrity", ...) which overwrites metrics.json
    # with status=failed and emits run.failed.
    METRICS.write_text(json.dumps({
        "status": "done",
        "total_seconds": round(total, 2),
        "best_val_loss": round(best_val, 6),
        "epochs": epochs,
        "n_params": int(n_params),
        "device": device.type,
        "gpu": gpu_name,
        "env": env_summary,
    }, indent=2))
    integ = _verify_run_integrity(eval_only=False)
    emit("run.integrity", ok=integ["ok"], missing=integ["missing"],
         invalid=integ["invalid"], notes=integ["notes"])
    if not integ["ok"]:
        problems = []
        if integ["missing"]:
            problems.append("missing: " + "; ".join(integ["missing"]))
        if integ["invalid"]:
            problems.append("invalid: " + "; ".join(integ["invalid"]))
        fail("integrity", " | ".join(problems) or "integrity check failed")
        return
    final_status = "done"
    if transition_status("done"):
        emit("run.done", total_seconds=round(total, 2), best_val_loss=round(best_val, 6))
    else:
        final_status = _read_status() or "cancelled"
        if final_status == "cancelled":
            # The loop never saw the cancel (it landed while we were finalising),
            # so no run.cancelled exists yet — emit one and let the cancelled
            # metrics below be the final word.
            _finish_cancel(max(start_epoch, end_epoch - 1), "cancelled during finalisation")
            return
        # Race: status changed to failed between integrity check and transition
        # (e.g. an outside observer wrote 'failed'). Don't emit run.done; the
        # existing terminal state wins.
        if final_status == "failed":
            return
        if final_status == _STATUS_UNREADABLE:
            # The status file became unreadable during finalisation. Never
            # report done on an unknown state — fail explicitly.
            fail("config", "run status file became unreadable before finalisation")
            return
    if final_status == "done":
        _manifest_write("done", status="done", finished_at=_now(),
                        summary=_manifest_summary(best_val, epochs, int(n_params)))
    # Phase 74 — terminal-time resumable flag (false for done runs; the gate
    # already proved the artifacts are present, so this is a noop in the happy
    # path but the data + event are emitted so the UI is uniform).
    _record_terminal_resumable(final_status, base_metrics={
        "epochs": epochs, "n_params": int(n_params), "device": device.type,
    })


if __name__ == "__main__":
    try:
        main()
    except _Cancelled as c:
        # Last resort: a signal that landed OUTSIDE the epoch loop (during
        # startup, e.g. imports/model build/dataset load — no epoch completed,
        # so there is nothing to make resumable yet). Record the cancellation
        # through the terminal-shielded path and exit cleanly.
        _finish_cancel(0, f"cancelled during startup (signal {c})")
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        fail("fatal", str(e), traceback.format_exc())
