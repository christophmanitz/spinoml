#!/usr/bin/env python3
"""MLForge training runner (Phase 13 foundation).

This script is dropped into a run directory as ``train.py`` and executed
detached (``setsid`` + redirected stdio). It is intentionally a *pure*
Python + torch program with no MLForge runtime dependency: everything it
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

Only tabular datasets (csv/tsv/parquet) are supported in Phase 13 — that
is the iris-mlp happy path. Other kinds fail loudly with a clear message;
Phase 14 widens this via the visual training graph + dataset config.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
EVENTS = RUN_DIR / "events.jsonl"
STATUS = RUN_DIR / "status"
METRICS = RUN_DIR / "metrics.json"
CKPT_DIR = RUN_DIR / "checkpoints"


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


def set_status(s: str) -> None:
    STATUS.write_text(s + "\n", encoding="utf-8")


def fail(stage: str, msg: str, tb: str | None = None) -> None:
    emit("run.failed", stage=stage, error=msg, traceback=tb or "")
    METRICS.write_text(json.dumps({"status": "failed", "stage": stage, "error": msg}, indent=2))
    set_status("failed")
    sys.stderr.write(f"[mlforge-train] FAILED in {stage}: {msg}\n")
    if tb:
        sys.stderr.write(tb)
    sys.exit(1)


# ─── Dataset loading (tabular only in Phase 13) ────────────────────────────

def load_tabular(cfg: dict):
    """Returns (X: FloatTensor [N, F], y: Tensor [N], n_classes|None, classes|None)."""
    import numpy as np
    import pandas as pd
    import torch

    path = os.path.expanduser(cfg["path"])
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in (".tsv",):
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    target_col = cfg.get("target_column")
    if not target_col:
        raise ValueError("tabular training needs a target_column")
    if target_col not in df.columns:
        raise ValueError(f"target column {target_col!r} not in dataset columns {list(df.columns)}")

    feature_cols = cfg.get("feature_columns")
    if feature_cols:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature columns not in dataset: {missing}")
    else:
        # default: all numeric columns except the target
        feature_cols = [
            c for c in df.select_dtypes(include="number").columns if c != target_col
        ]
    if not feature_cols:
        raise ValueError("no usable feature columns")

    X_np = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype="float32")
    X = torch.from_numpy(np.ascontiguousarray(X_np))

    tgt = df[target_col]
    classes = None
    n_classes = None
    if cfg.get("task") == "regression":
        y = torch.from_numpy(np.ascontiguousarray(tgt.apply(lambda v: float(v)).to_numpy(dtype="float32")))
    else:
        # classification: map labels → integer codes
        cat = tgt.astype("category")
        classes = [str(c) for c in cat.cat.categories.tolist()]
        n_classes = len(classes)
        y = torch.from_numpy(np.ascontiguousarray(cat.cat.codes.to_numpy(dtype="int64")))
    return X, y, n_classes, classes, feature_cols


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


def build_loss(cfg: dict):
    import torch

    kind = cfg.get("kind", "CrossEntropyLoss")
    if kind == "CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss(), "classification"
    if kind == "BCEWithLogitsLoss":
        return torch.nn.BCEWithLogitsLoss(), "binary"
    if kind == "MSELoss":
        return torch.nn.MSELoss(), "regression"
    if kind == "L1Loss":
        return torch.nn.L1Loss(), "regression"
    raise ValueError(f"unknown loss kind: {kind}")


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


def main() -> None:
    set_status("running")
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
    log_every = int(train_cfg.get("log_every_n_steps", 10))

    # ── imports (heavy) ──
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset, random_split
    except Exception as e:  # noqa: BLE001
        fail("import-torch", f"torch import failed: {e}", traceback.format_exc())

    torch.manual_seed(seed)

    # ── dataset ──
    try:
        loss_fn, task = build_loss(train_cfg.get("loss", {}))
        ds_cfg = {**ds_cfg, "task": "regression" if task == "regression" else "classification"}
        kind = ds_cfg.get("kind", "tabular")
        if kind != "tabular":
            raise ValueError(
                f"Phase 13 trains tabular datasets only (got {kind!r}). "
                "Image/tensor/graph training arrives with the visual training graph (Phase 14)."
            )
        X, y, n_classes, classes, feature_cols = load_tabular(ds_cfg)
        emit("dataset.loaded", n_rows=int(X.shape[0]), n_features=int(X.shape[1]),
             task=task, n_classes=n_classes, features=feature_cols)
    except Exception as e:  # noqa: BLE001
        fail("dataset", str(e), traceback.format_exc())

    # ── split ──
    full = TensorDataset(X, y)
    n_val = max(1, int(len(full) * val_split)) if val_split > 0 else 0
    n_train = len(full) - n_val
    if n_train <= 0:
        fail("split", f"not enough rows ({len(full)}) for the chosen val_split={val_split}")
    gen = torch.Generator().manual_seed(seed)
    if n_val > 0:
        train_ds, val_ds = random_split(full, [n_train, n_val], generator=gen)
    else:
        train_ds, val_ds = full, None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size) if val_ds is not None else None

    # ── model + optimizer ──
    try:
        sys.path.insert(0, str(RUN_DIR))
        from model import Model  # type: ignore

        model = Model()
        optimizer = build_optimizer(train_cfg.get("optimizer", {}), model.parameters())
        scheduler = build_scheduler(train_cfg.get("scheduler", {}), optimizer, epochs)
        n_params = sum(p.numel() for p in model.parameters())
        emit("model.built", n_params=int(n_params))
    except Exception as e:  # noqa: BLE001
        fail("model", str(e), traceback.format_exc())

    def prep_target(t):
        if task == "regression":
            return t.float()
        if task == "binary":
            return t.float()
        return t.long()

    def forward_loss(xb, yb):
        out = model(xb)
        if isinstance(out, tuple):
            out = out[0]
        if task == "regression":
            out = out.squeeze(-1) if out.dim() > 1 and out.shape[-1] == 1 else out
        elif task == "binary":
            out = out.squeeze(-1) if out.dim() > 1 and out.shape[-1] == 1 else out
        return out, loss_fn(out, prep_target(yb))

    CKPT_DIR.mkdir(exist_ok=True)
    best_val = float("inf")

    try:
        for epoch in range(epochs):
            if STATUS.read_text().strip() == "cancelled":
                emit("run.cancelled", epoch=epoch)
                METRICS.write_text(json.dumps({"status": "cancelled", "epoch": epoch}, indent=2))
                return
            emit("epoch.start", epoch=epoch)
            model.train()
            running = 0.0
            n_seen = 0
            step = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                _, loss = forward_loss(xb, yb)
                loss.backward()
                optimizer.step()
                bs = xb.shape[0]
                running += float(loss.item()) * bs
                n_seen += bs
                step += 1
                if step % log_every == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    emit("batch", epoch=epoch, step=step, loss=round(float(loss.item()), 6), lr=lr)
            train_loss = running / max(1, n_seen)

            # ── validation ──
            val_loss = None
            val_acc = None
            if val_loader is not None:
                model.eval()
                vrun = 0.0
                vseen = 0
                correct = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        out, loss = forward_loss(xb, yb)
                        bs = xb.shape[0]
                        vrun += float(loss.item()) * bs
                        vseen += bs
                        if task == "classification":
                            correct += int((out.argmax(dim=-1) == yb.long()).sum().item())
                        elif task == "binary":
                            correct += int(((out > 0).long() == yb.long()).sum().item())
                val_loss = vrun / max(1, vseen)
                if task in ("classification", "binary"):
                    val_acc = correct / max(1, vseen)

            monitor = val_loss if val_loss is not None else train_loss
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(monitor)
                else:
                    scheduler.step()

            emit("epoch.end", epoch=epoch, train_loss=round(train_loss, 6),
                 val_loss=None if val_loss is None else round(val_loss, 6),
                 val_acc=None if val_acc is None else round(val_acc, 6),
                 lr=optimizer.param_groups[0]["lr"])

            # ── checkpoint best ──
            if monitor < best_val:
                best_val = monitor
                torch.save({"epoch": epoch, "model_state": model.state_dict(),
                            "val_loss": val_loss, "classes": classes},
                           CKPT_DIR / "best.pt")
                emit("checkpoint", epoch=epoch, path="checkpoints/best.pt",
                     val_loss=None if val_loss is None else round(val_loss, 6), is_best=True)

        torch.save({"epoch": epochs - 1, "model_state": model.state_dict(), "classes": classes},
                   CKPT_DIR / "last.pt")
    except Exception as e:  # noqa: BLE001
        fail("train", str(e), traceback.format_exc())

    total = time.time() - t0
    emit("run.done", total_seconds=round(total, 2), best_val_loss=round(best_val, 6))
    METRICS.write_text(json.dumps({
        "status": "done",
        "total_seconds": round(total, 2),
        "best_val_loss": round(best_val, 6),
        "epochs": epochs,
        "n_params": int(n_params),
    }, indent=2))
    set_status("done")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        fail("fatal", str(e), traceback.format_exc())
