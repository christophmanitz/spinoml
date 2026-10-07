#!/usr/bin/env python3
"""Phase 73 — sabotage launcher for verify-integrity.ts.

A small helper that pre-loads the trainer via importlib (so all module-level
constants — RUN_DIR, the safe_load block, the helper functions — are
available), applies a SABOTAGE hook to one of its persistence functions,
then calls trainer.main() directly.

Usage (from TypeScript):
  python -u scripts/lib/integrity_wrap.py <run_dir> <sabotage_id>

where <sabotage_id> is one of:
  - none                       : no sabotage, run a healthy training
  - missing-best               : best.pt is never written
  - zero-best                  : best.pt is written then truncated to 0 bytes
  - garbage-best               : best.pt is overwritten with garbage after writing
  - missing-last               : last.pt is never written
  - missing-model-state        : best.pt is written then model_state key is stripped
  - delete-metrics             : metrics.json is removed before the gate runs
  - nan-metric                 : metrics.json best_val_loss is set to NaN before the gate
  - corrupt-manifest           : manifest.json is corrupted before the gate runs
  - missing-config-env         : the config.env event is stripped from events.jsonl
  - inject-pid-stderr-missing  : create a pid file + delete stderr.log after save
  - inject-pid-empty-logs      : executor-style launch: pid file + EMPTY stdout.log/stderr.log
  - raise-in-gate              : patch _checkpoint_ok to raise (gate fails closed)
  - crash-after-epoch-0        : writes a model.py whose forward raises on call 9
  - crash-epoch-0-first        : writes a model.py whose forward raises on call 1
  - corrupt-last               : last.pt is overwritten with garbage after writing
  - different-model-last       : last.pt.config.snapshot.graph_sha256 is mutated
  - eval-only                  : set cfg.eval_only=true with a real source checkpoint

This script is a launcher, not a test: it executes ONE run in <run_dir> with
the requested sabotage applied, prints stdout/stderr, and exits with the
trainer's exit code. The TypeScript harness reads the run dir afterwards.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import runpy
import sys
import traceback
from pathlib import Path


def _sabotage_none(trainer, run_dir):
    return


def _wrap_atomic_save(trainer, name, hook):
    """Helper: wrap _atomic_save to install `hook(obj, path)` ONLY for files
    matching `name` ("best.pt" or "last.pt"). Other files use the original."""
    orig = trainer._atomic_save

    def wrapped(obj, path):
        if path.name == name:
            hook(obj, path)
            return
        return orig(obj, path)

    trainer._atomic_save = wrapped


def _sabotage_missing_best(trainer, run_dir):
    def drop(obj, path): return None
    _wrap_atomic_save(trainer, "best.pt", drop)


def _sabotage_zero_best(trainer, run_dir):
    def truncate(obj, path):
        trainer._atomic_save.__wrapped__ if hasattr(trainer._atomic_save, "__wrapped__") else None
        # call the real one explicitly to write, then truncate
        import torch
        with open(path, "wb") as f:
            pass
    # We need the un-wrapped original; simpler: just write garbage (zero bytes).
    def zero(obj, path):
        with open(path, "wb") as f:
            pass
    _wrap_atomic_save(trainer, "best.pt", zero)


def _sabotage_garbage_best(trainer, run_dir):
    def garble(obj, path):
        with open(path, "wb") as f:
            f.write(b"\x00\x01\x02\x03not-a-torch-file-garbage")
    _wrap_atomic_save(trainer, "best.pt", garble)


def _sabotage_missing_last(trainer, run_dir):
    def drop(obj, path): return None
    _wrap_atomic_save(trainer, "last.pt", drop)


def _sabotage_missing_model_state(trainer, run_dir):
    def strip(obj, path):
        obj2 = dict(obj)
        obj2.pop("model_state", None)
        # write the stripped object using the ORIGINAL save path:
        # call torch.save directly (the trainer uses torch.save under the hood).
        import torch
        tmp = path.with_name(path.name + ".tmp")
        try:
            with open(tmp, "wb") as f:
                torch.save(obj2, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                try: tmp.unlink()
                except Exception: pass
    _wrap_atomic_save(trainer, "best.pt", strip)


class _SabotagePath(type(Path())):
    """Path subclass with a custom write_text that runs a hook after writing."""
    def __init__(self, real_path, hook):
        self._real = real_path
        self._hook = hook

    def write_text(self, data, *a, **k):
        self._real.write_text(data, *a, **k)
        try:
            self._hook(self._real)
        except Exception:
            pass


def _sabotage_delete_metrics(trainer, run_dir):
    """Delete metrics.json ONCE — right after the trainer writes it the first
    time. The gate then sees the file missing, fails integrity, calls
    fail() which writes a SECOND metrics.json (status=failed) — that one
    survives because we only delete on the first write."""
    real = trainer.METRICS
    counter = {"n": 0}

    class _Wrapped(type(real)):
        def write_text(self, data, *a, **k):
            super().write_text(data, *a, **k)
            counter["n"] += 1
            if counter["n"] == 1:
                try:
                    os.remove(self)
                except FileNotFoundError:
                    pass

    trainer.METRICS = _Wrapped(str(real))


def _sabotage_nan_metric(trainer, run_dir):
    """Corrupt the first metrics.json write with NaN best_val_loss. The gate
    catches it and calls fail(); fail()'s second write is left alone so the
    final metrics.json carries status=failed."""
    real = trainer.METRICS
    counter = {"n": 0}

    class _Wrapped(type(real)):
        def write_text(self, data, *a, **k):
            counter["n"] += 1
            if counter["n"] == 1:
                try:
                    m = json.loads(data)
                    if isinstance(m, dict):
                        m["best_val_loss"] = float("nan")
                    data = json.dumps(m, indent=2)
                except Exception:
                    pass
            super().write_text(data, *a, **k)

    trainer.METRICS = _Wrapped(str(real))


def _sabotage_corrupt_manifest(trainer, run_dir):
    orig = trainer._manifest_write

    def corrupt_after(stage, **updates):
        orig(stage, **updates)
        try:
            with open(trainer.MANIFEST, "w", encoding="utf-8") as f:
                f.write('{ "schema": "spinoml.run-manifest/1", "status": "running"')
        except Exception:
            pass

    trainer._manifest_write = corrupt_after


def _sabotage_missing_config_env(trainer, run_dir):
    """Wrap `emit` to drop every config.env event before it lands on disk.
    Path.open can't be instance-patched (read-only attribute), so we patch
    the higher-level `emit` function — same end effect: the gate never sees
    the event."""
    orig_emit = trainer.emit

    def stripped_emit(kind, **fields):
        if kind == "config.env":
            return None
        return orig_emit(kind, **fields)

    trainer.emit = stripped_emit


def _sabotage_pid_stderr_missing(trainer, run_dir):
    pid_path = pathlib.Path(run_dir) / "pid"
    pid_path.write_text("12345\n", encoding="utf-8")
    stderr_log = pathlib.Path(run_dir) / "stderr.log"
    if stderr_log.exists():
        stderr_log.unlink()


def _sabotage_pid_empty_logs(trainer, run_dir):
    # What the real executors produce (training.rs / ssh.rs): `python -u train.py
    # > stdout.log 2> stderr.log` + a pid file. The trainer writes its events to
    # events.jsonl, so both logs are legitimately EMPTY in a healthy run.
    d = pathlib.Path(run_dir)
    (d / "pid").write_text("12345\n", encoding="utf-8")
    (d / "stdout.log").write_text("", encoding="utf-8")
    (d / "stderr.log").write_text("", encoding="utf-8")


def _sabotage_raise_in_gate(trainer, run_dir):
    def boom(path):
        raise RuntimeError("simulated integrity-gate failure")
    trainer._checkpoint_ok = boom


_CRASH_MODEL_PY = '''import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)
        self.calls = 0
    def forward(self, x):
        self.calls += 1
        if self.calls >= {THRESHOLD}:
            raise RuntimeError("{MSG}")
        return self.fc2(self.act(self.fc1(x)))
'''


def _sabotage_crash_after_epoch_0(trainer, run_dir):
    """Overwrite model.py with one whose forward raises on the 9th call.
    With batch_size=16 and 80 rows (60 train / 20 val) the 9th forward call
    lands inside epoch 1's first batch — after epoch 0 finished (best.pt
    saved at end of epoch 0) and crashed before last.pt was written by the
    training loop. The trainer's exception handler then saves last.pt at the
    last COMPLETED epoch (epoch 0)."""
    new_model = _CRASH_MODEL_PY.format(THRESHOLD=9, MSG="simulated crash")
    pathlib.Path(run_dir, "model.py").write_text(new_model, encoding="utf-8")
    _update_model_py_snapshot(run_dir, new_model)


def _sabotage_crash_epoch_0_first(trainer, run_dir):
    """Overwrite model.py so the very first forward call (inside model build,
    on the dummy batch) raises — no checkpoint is ever saved."""
    new_model = _CRASH_MODEL_PY.format(THRESHOLD=1, MSG="simulated early crash")
    pathlib.Path(run_dir, "model.py").write_text(new_model, encoding="utf-8")
    _update_model_py_snapshot(run_dir, new_model)


def _update_model_py_snapshot(run_dir: str, new_model_py: str) -> None:
    """Recompute model_py_sha256 in run.json's snapshot so the trainer's
    Phase-20 snapshot check still passes after we've replaced model.py."""
    import hashlib
    rj = pathlib.Path(run_dir) / "run.json"
    cfg = json.loads(rj.read_text(encoding="utf-8"))
    snap = cfg.get("snapshot") if isinstance(cfg.get("snapshot"), dict) else None
    if snap is None:
        return
    snap["model_py_sha256"] = hashlib.sha256(new_model_py.encode("utf-8")).hexdigest()
    rj.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _sabotage_corrupt_last(trainer, run_dir):
    def garble(obj, path):
        with open(path, "wb") as f:
            f.write(b"\x00\x01\x02not-a-torch-file-garbage")
    _wrap_atomic_save(trainer, "last.pt", garble)


def _sabotage_different_model_last(trainer, run_dir):
    """After writing last.pt, mutate config.snapshot.graph_sha256 + model_py
    hashes so the resumable check sees a foreign model."""
    import torch

    def mutate(obj, path):
        # Use the original write path, then mutate on disk.
        import torch
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        try:
            ck = torch.load(str(path), map_location="cpu", weights_only=False)
            if isinstance(ck, dict) and isinstance(ck.get("config"), dict):
                snap = ck["config"].get("snapshot")
                if isinstance(snap, dict):
                    snap["graph_sha256"] = "f" * 64
                    snap["model_py_sha256"] = "e" * 64
            tmp2 = path.with_name(path.name + ".tmp")
            with open(tmp2, "wb") as f:
                torch.save(ck, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp2, path)
        except Exception:
            pass

    _wrap_atomic_save(trainer, "last.pt", mutate)


def _sabotage_eval_only(trainer, run_dir):
    rj = pathlib.Path(run_dir) / "run.json"
    cfg = json.loads(rj.read_text(encoding="utf-8"))
    src = os.environ.get("SPINOML_EVAL_SOURCE")
    if not src:
        src = os.path.join(run_dir, "checkpoints", "best.pt")
    cfg["eval_only"] = True
    cfg["validate"] = {"checkpoint_from": src}
    cfg.setdefault("training", {})
    cfg["training"]["epochs"] = 0
    rj.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


SABOTAGES = {
    "none": _sabotage_none,
    "missing-best": _sabotage_missing_best,
    "zero-best": _sabotage_zero_best,
    "garbage-best": _sabotage_garbage_best,
    "missing-last": _sabotage_missing_last,
    "missing-model-state": _sabotage_missing_model_state,
    "delete-metrics": _sabotage_delete_metrics,
    "nan-metric": _sabotage_nan_metric,
    "corrupt-manifest": _sabotage_corrupt_manifest,
    "missing-config-env": _sabotage_missing_config_env,
    "inject-pid-stderr-missing": _sabotage_pid_stderr_missing,
    "inject-pid-empty-logs": _sabotage_pid_empty_logs,
    "raise-in-gate": _sabotage_raise_in_gate,
    "crash-after-epoch-0": _sabotage_crash_after_epoch_0,
    "crash-epoch-0-first": _sabotage_crash_epoch_0_first,
    "corrupt-last": _sabotage_corrupt_last,
    "different-model-last": _sabotage_different_model_last,
    "eval-only": _sabotage_eval_only,
}


def main():
    if len(sys.argv) != 3:
        print("usage: integrity_wrap.py <run_dir> <sabotage_id>", file=sys.stderr)
        sys.exit(2)
    run_dir = sys.argv[1]
    sabotage_id = sys.argv[2]
    fn = SABOTAGES.get(sabotage_id)
    if fn is None:
        print(f"unknown sabotage: {sabotage_id}", file=sys.stderr)
        sys.exit(2)
    spec = importlib.util.spec_from_file_location("trainer", os.path.join(run_dir, "train.py"))
    trainer = importlib.util.module_from_spec(spec)
    sys.modules["trainer"] = trainer
    spec.loader.exec_module(trainer)
    try:
        fn(trainer, run_dir)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
    try:
        trainer.main()
    except SystemExit as e:
        sys.exit(e.code if isinstance(e.code, int) else 1)


if __name__ == "__main__":
    main()
