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
and 'manifest' (paired PyG graphs, e.g. ligand+protein → a model with one graph
input per branch, trained with a PyG-Batch collate). Other kinds fail loudly.
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
# run dir is <workspace>/experiments/runs/<run_id> → workspace root is 3 up.
# Used to resolve workspace-relative resume_from paths.
WORKSPACE_ROOT = RUN_DIR.parents[2] if len(RUN_DIR.parents) >= 3 else RUN_DIR


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
    sys.stderr.write(f"[spinoml-train] FAILED in {stage}: {msg}\n")
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


# ── manifest (paired graph) datasets — self-contained, no dataset_handlers ──
# The run is a standalone snapshot (train.py + model.py), so the manifest logic
# is inlined here. Builds one PyG Data per branch per row; molecule branches
# come from SMILES (RDKit, cached as .pt), file branches from .pt on disk.

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
                d = torch.load(fp, map_location="cpu", weights_only=False)
                if hasattr(d, "edge_index"):
                    return d
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
        except Exception:
            pass
    return d


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


def load_manifest_graphs(ds_cfg: dict, task: str):
    """Returns (rows, branches) where rows = [([Data per branch], y_scalar), …]."""
    import json
    import torch
    import pandas as pd
    man = Path(os.path.expanduser(ds_cfg["path"]))
    base = man.resolve().parent
    cfg = json.loads(man.read_text(encoding="utf-8"))
    tp = base / str(cfg["table"])
    suffix = tp.suffix.lower()
    df = (pd.read_parquet(tp) if suffix == ".parquet"
          else pd.read_csv(tp, sep="\t") if suffix == ".tsv" else pd.read_csv(tp))
    branches = list(cfg["pairs"].keys())
    cache_dir = (base / ".graphcache") if cfg.get("cache", True) else None
    tgt = cfg.get("target") or {}
    tcol = tgt.get("column")
    rows = []
    skipped = 0
    for i in range(len(df)):
        r = df.iloc[i]
        graphs = []
        ok = True
        for b in branches:
            spec = cfg["pairs"][b]
            val = r[spec["column"]]
            try:
                if str(spec.get("kind", "")) == "molecule":
                    g = _manifest_mol_graph(str(val), cache_dir)
                else:
                    fp = _manifest_resolve_file(base, spec, val)
                    if fp is None or not Path(fp).exists():
                        raise FileNotFoundError(f"no graph file for {val!r}")
                    g = torch.load(fp, map_location="cpu", weights_only=False)
                graphs.append(g)
            except Exception:
                ok = False
                break
        if not ok:
            skipped += 1
            continue
        if tcol is None:
            y = torch.zeros((), dtype=torch.float32)
        elif task in ("classification",):
            y = torch.tensor(int(r[tcol]), dtype=torch.long)
        else:
            y = torch.tensor(float(r[tcol]), dtype=torch.float32)
        rows.append((graphs, y))
    if not rows:
        raise ValueError("manifest produced no usable paired rows (check branch columns / file paths)")
    return rows, branches, skipped


def graph_collate(batch):
    """Collate [( [Data,…], y ), …] → ( (Batch per branch…), y_stacked )."""
    import torch
    from torch_geometric.data import Batch
    n_branches = len(batch[0][0])
    branch_batches = tuple(
        Batch.from_data_list([item[0][k] for item in batch]) for k in range(n_branches)
    )
    ys = torch.stack([item[1] for item in batch])
    return branch_batches, ys


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
    is_graph = False
    classes = None
    try:
        loss_fn, task = build_loss(train_cfg.get("loss", {}))
        ds_cfg = {**ds_cfg, "task": "regression" if task == "regression" else "classification"}
        kind = ds_cfg.get("kind", "tabular")
        if kind == "manifest":
            # Paired graph dataset (e.g. ligand + protein) → a model with one
            # graph input per branch (forward(self, branch0, branch1, …)).
            is_graph = True
            rows, branches, skipped = load_manifest_graphs(ds_cfg, task)
            full = rows
            emit("dataset.loaded", n_rows=len(rows), branches=branches, task=task, skipped=skipped)
        elif kind == "tabular":
            X, y, n_classes, classes, feature_cols = load_tabular(ds_cfg)
            full = TensorDataset(X, y)
            emit("dataset.loaded", n_rows=int(X.shape[0]), n_features=int(X.shape[1]),
                 task=task, n_classes=n_classes, features=feature_cols)
        else:
            raise ValueError(f"unsupported dataset kind {kind!r} — use 'tabular' or 'manifest'.")
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
    collate = graph_collate if is_graph else None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, collate_fn=collate) if val_ds is not None else None

    # ── model + optimizer ──
    try:
        sys.path.insert(0, str(RUN_DIR))
        from model import Model  # type: ignore

        model = Model()
        # Lazy layers (in_channels=-1) only materialize their params on the first
        # forward — run one dummy batch BEFORE building the optimizer (else those
        # params are never registered with it) and before counting params.
        try:
            xb0, _ = next(iter(train_loader))
            with torch.no_grad():
                model(*xb0) if is_graph else model(xb0)
        except StopIteration:
            pass
        optimizer = build_optimizer(train_cfg.get("optimizer", {}), model.parameters())
        scheduler = build_scheduler(train_cfg.get("scheduler", {}), optimizer, epochs)
        n_params = sum(p.numel() for p in model.parameters())
        emit("model.built", n_params=int(n_params))
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
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    if metric_kinds or grad_clip or early or use_amp:
        emit("config.extras", metrics=metric_kinds, grad_clip=grad_clip,
             early_stopping=early, amp=cb["amp"])
    es_best = float("inf") if (early and early["mode"] == "min") else float("-inf")
    es_wait = 0

    def prep_target(t):
        if task == "regression":
            return t.float()
        if task == "binary":
            return t.float()
        return t.long()

    def batch_len(xb):
        # graph mode: xb is a tuple of PyG Batches → count graphs, not tensor rows.
        return xb[0].num_graphs if is_graph else xb.shape[0]

    def forward_loss(xb, yb):
        out = model(*xb) if is_graph else model(xb)
        if isinstance(out, tuple):
            out = out[0]
        if task == "regression":
            out = out.squeeze(-1) if out.dim() > 1 and out.shape[-1] == 1 else out
        elif task == "binary":
            out = out.squeeze(-1) if out.dim() > 1 and out.shape[-1] == 1 else out
        return out, loss_fn(out, prep_target(yb))

    CKPT_DIR.mkdir(exist_ok=True)
    best_val = float("inf")
    start_epoch = 0

    # ── resume from a prior checkpoint (Phase 17) ──
    resume_from = cfg.get("resume_from") or train_cfg.get("resume_from")
    if resume_from:
        try:
            rp = Path(os.path.expanduser(str(resume_from)))
            if not rp.is_absolute():
                rp = WORKSPACE_ROOT / str(resume_from)
            ckpt = torch.load(rp, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            if "optim_state" in ckpt:
                optimizer.load_state_dict(ckpt["optim_state"])
            if scheduler is not None and ckpt.get("sched_state") is not None:
                scheduler.load_state_dict(ckpt["sched_state"])
            best_val = float(ckpt.get("best_val", best_val))
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            emit("run.resumed", source=str(resume_from), start_epoch=start_epoch,
                 prev_val_loss=ckpt.get("val_loss"))
        except Exception as e:  # noqa: BLE001
            fail("resume", f"cannot resume from {resume_from!r}: {e}", traceback.format_exc())

    end_epoch = start_epoch + epochs

    try:
        for epoch in range(start_epoch, end_epoch):
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
                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                    _, loss = forward_loss(xb, yb)
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                bs = batch_len(xb)
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
            extra: dict = {}
            val_cat = None
            if val_loader is not None:
                model.eval()
                vrun = 0.0
                vseen = 0
                correct = 0
                outs = []
                ys = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=use_amp):
                            out, loss = forward_loss(xb, yb)
                        bs = batch_len(xb)
                        vrun += float(loss.item()) * bs
                        vseen += bs
                        outs.append(out.float())
                        ys.append(yb)
                        if task == "classification":
                            correct += int((out.argmax(dim=-1) == yb.long()).sum().item())
                        elif task == "binary":
                            correct += int(((out > 0).long() == yb.long()).sum().item())
                val_loss = vrun / max(1, vseen)
                if task in ("classification", "binary"):
                    val_acc = correct / max(1, vseen)
                if outs:
                    val_cat = (torch.cat(outs), torch.cat(ys))
                    if metric_kinds:
                        extra = {k: round(v, 6) for k, v in
                                 compute_metrics(task, val_cat[0], val_cat[1], metric_kinds).items()}

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
                 lr=optimizer.param_groups[0]["lr"])

            # ── checkpoint best ──
            if monitor < best_val:
                best_val = monitor
                torch.save({"epoch": epoch, "model_state": model.state_dict(),
                            "optim_state": optimizer.state_dict(),
                            "sched_state": scheduler.state_dict() if scheduler is not None else None,
                            "best_val": best_val,
                            "val_loss": val_loss, "classes": classes},
                           CKPT_DIR / "best.pt")
                emit("checkpoint", epoch=epoch, path="checkpoints/best.pt",
                     val_loss=None if val_loss is None else round(val_loss, 6), is_best=True)
                if val_cat is not None:
                    emit("sample.preds", epoch=epoch,
                         rows=sample_predictions(task, val_cat[0], val_cat[1], classes))

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

        torch.save({"epoch": end_epoch - 1, "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "sched_state": scheduler.state_dict() if scheduler is not None else None,
                    "best_val": best_val, "classes": classes},
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
