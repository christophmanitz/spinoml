"""Checkpoint <-> generated-model consistency for ``verify-reference-train``.

Given a REAL run directory produced by ``training_template.py`` (dropped in as
``train.py``), this helper:

  * loads ``checkpoints/best.pt`` and ``checkpoints/last.pt`` through the
    project's single safe loader (``sidecar-torch/safe_load.py`` — never
    ``torch.load`` directly),
  * asserts the required checkpoint keys are present,
  * imports the run's generated ``model.py`` and loads ``best.pt``'s
    ``model_state`` into a fresh instance with ``strict=True``,
  * re-evaluates the generated model on the SAME validation split the trainer
    used (the split is the ``torch.randperm(N, generator=manual_seed(seed))``
    permutation ``random_split`` performs), and reports whether the re-evaluated
    val loss matches the trainer's recorded best val loss,
  * checks the dtype of the checkpointed floating-point parameters against the
    run's recorded ``config.env`` dtype.

Usage::

    python reference_train_check.py <run_dir> <tabular|manifest>

Prints ONE JSON object on stdout; hard failures are reported as
``{"ok": false, "error": ...}`` (exit 0) so the TypeScript harness owns the
pass/fail verdict and can print a per-experiment message.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[2]


def _load_safe_load():
    path = _REPO_ROOT / "sidecar-torch" / "safe_load.py"
    spec = importlib.util.spec_from_file_location("spinoml_safe_load", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import safe loader from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


safe_load = _load_safe_load()

_DTYPE_NAMES = {
    "fp32": "torch.float32",
    "fp16": "torch.float16",
    "bf16": "torch.bfloat16",
}


def _dtype_name(dtype) -> str:
    return str(dtype).replace("torch.", "")


def _load_model_class(model_py: Path):
    import torch.nn as nn

    ns: dict[str, object] = {"__name__": "<spinoml-run-model>"}
    with open(model_py, "r", encoding="utf-8") as fh:
        code = fh.read()
    exec(compile(code, "<spinoml-run-model>", "exec"), ns)
    cls = ns.get("Model")
    if isinstance(cls, type) and issubclass(cls, nn.Module) and cls is not nn.Module:
        return cls
    for value in ns.values():
        if isinstance(value, type) and issubclass(value, nn.Module) and value is not nn.Module:
            return value
    raise RuntimeError("no nn.Module subclass found in generated model.py")


def _read_table(path: Path):
    import pandas as pd

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _resolve_primary(path: Path) -> Path:
    """Mirror training_template._resolve_primary for a prepared-dataset dir."""
    if not path.is_dir():
        return path
    prefer = ("pairs.csv", "data.csv", "table.csv", "dataset.csv", "train.csv", "test.csv")
    tables = [c for c in path.iterdir() if c.is_file() and c.suffix.lower() in (".csv", ".tsv", ".parquet")]
    by_name = {c.name.lower(): c for c in tables}
    return next((by_name[n] for n in prefer if n in by_name),
                max(tables, key=lambda c: c.stat().st_size))


def _load_tabular(cfg: dict, target: str):
    import numpy as np
    import pandas as pd
    import torch

    path = _resolve_primary(Path(os.path.expanduser(str(cfg["path"]))))
    df = _read_table(path)
    fcols = cfg.get("feature_columns")
    if not fcols:
        fcols = [c for c in df.select_dtypes(include="number").columns if c != target]
    x_np = df[fcols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64").astype(np.float32)
    x = torch.from_numpy(np.ascontiguousarray(x_np))
    cat = df[target].astype("category")
    codes = cat.cat.codes.to_numpy().astype("int64")
    y = torch.from_numpy(np.ascontiguousarray(codes))
    return x, y, int(x.shape[0])


def _manifest_branch(base: Path, spec: dict, value: str):
    value = str(value).strip()
    if "dir" in spec:
        d = base / str(spec["dir"])
        ext = str(spec.get("ext", "") or "")
        if str(spec.get("match", "exact")) == "contains":
            cands = sorted(c for c in d.iterdir() if c.is_file() and value in c.name)
            if ext:
                cands = [c for c in cands if c.suffix.lower() == ext.lower()] or cands
            return cands[0] if cands else None
        cand = d / (value if (not ext or value.endswith(ext)) else value + ext)
        return cand if cand.exists() else None
    p = Path(value)
    return p if p.is_absolute() else (base / value)


def _load_manifest(cfg: dict, target: str):
    import numpy as np
    import torch

    import pandas as pd

    man = Path(os.path.expanduser(str(cfg["path"])))
    base = man.resolve().parent
    mcfg = json.loads(man.read_text(encoding="utf-8"))
    df = _read_table(base / str(mcfg["table"]))
    pairs = mcfg["pairs"]
    branches = list(pairs.keys())
    rows = []
    skipped = 0
    for i in range(len(df)):
        items = []
        ok = True
        for b in branches:
            spec = pairs[b]
            fp = _manifest_branch(base, spec, df.iloc[i][spec["column"]])
            if fp is None or not Path(fp).exists():
                ok = False
                break
            items.append(safe_load.safe_torch_load(fp))
        if ok:
            rows.append(items)
        else:
            skipped += 1
    if skipped:
        raise RuntimeError(f"manifest produced {skipped} skipped row(s); "
                           "the reference harness expects every row to resolve")
    cat = df[target].astype("category")
    codes = cat.cat.codes.to_numpy().astype("int64")
    y = torch.from_numpy(np.ascontiguousarray(codes))
    return rows, y, len(rows), skipped


def _checkpoint_keys(ck: dict) -> bool:
    required = ("model_state", "optim_state", "epoch", "global_step", "rng", "config")
    return all(k in ck for k in required)


def _param_dtype_summary(state: dict):
    import torch

    dtypes = set()
    for v in state.values():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            dtypes.add(_dtype_name(v.dtype))
    return sorted(dtypes)


def _run(run_dir: Path, kind: str) -> dict:
    import torch
    import torch.nn as nn

    cfg = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    training = cfg.get("training", {})
    ds_cfg = cfg.get("dataset", {})
    seed = int(training.get("seed", 42))
    val_split = float(training.get("val_split", 0.2))

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    env = metrics.get("env") or {}
    env_dtype = str(env.get("dtype", ""))
    best_val_loss = float(metrics["best_val_loss"])

    result: dict[str, object] = {
        "ok": True,
        "best_val_loss": best_val_loss,
        "env_device": env.get("device"),
        "env_dtype": env_dtype,
    }

    best = safe_load.safe_torch_load(run_dir / "checkpoints" / "best.pt")
    last = safe_load.safe_torch_load(run_dir / "checkpoints" / "last.pt")
    result["keys_best_ok"] = _checkpoint_keys(best)
    result["keys_last_ok"] = _checkpoint_keys(last)
    result["ckpt_val_loss"] = best.get("val_loss")

    dtypes = _param_dtype_summary(best["model_state"])
    result["param_dtypes"] = dtypes
    expected = _DTYPE_NAMES.get(env_dtype)
    result["dtype_match"] = bool(dtypes) and all(d == _dtype_name(expected) for d in dtypes) if expected else False

    model_cls = _load_model_class(run_dir / "model.py")
    model = model_cls()
    try:
        model.load_state_dict(best["model_state"], strict=True)
        result["strict_load_ok"] = True
    except Exception as exc:  # noqa: BLE001
        result["strict_load_ok"] = False
        result["strict_load_error"] = f"{type(exc).__name__}: {exc}"
        return result

    n_params = sum(p.numel() for p in model.parameters())
    result["n_params"] = int(n_params)

    # ── reproduce the trainer's validation split exactly ──
    if kind == "tabular":
        head_target = ds_cfg.get("target_column") or ""
        x, y, n = _load_tabular(ds_cfg, head_target)
        n_val = max(1, int(n * val_split)) if val_split > 0 else 0
        n_train = n - n_val
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=gen)
        val_idx = perm[n_train:]
        model.eval()
        with torch.no_grad():
            logits = model(x[val_idx])
            loss = nn.CrossEntropyLoss()(logits, y[val_idx])
        result["n"] = n
    elif kind == "manifest":
        mcfg = json.loads(Path(os.path.expanduser(str(ds_cfg["path"]))).read_text(encoding="utf-8"))
        target = (mcfg.get("target") or {}).get("column") or ds_cfg.get("target_column") or ""
        rows, y, n, skipped = _load_manifest(ds_cfg, target)
        result["n"] = n
        result["skipped"] = skipped
        n_val = max(1, int(n * val_split)) if val_split > 0 else 0
        n_train = n - n_val
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=gen)
        val_idx = perm[n_train:].tolist()
        a = torch.stack([rows[i][0] for i in val_idx])
        b = torch.stack([rows[i][1] for i in val_idx])
        yv = y[torch.tensor(val_idx)]
        model.eval()
        with torch.no_grad():
            logits = model(a, b)
            loss = nn.CrossEntropyLoss()(logits, yv)
    else:
        raise ValueError(f"unknown kind {kind!r}")

    re_val_loss = float(loss.item())
    result["re_val_loss"] = re_val_loss
    result["outputs_finite"] = bool(torch.isfinite(logits).all())
    result["val_loss_match"] = abs(re_val_loss - best_val_loss) <= 1e-5
    ckv = best.get("val_loss")
    result["ckpt_vs_metrics_match"] = (ckv is None) or (abs(float(ckv) - best_val_loss) <= 1e-6)
    return result


def main() -> int:
    if len(sys.argv) < 3:
        print(json.dumps({"ok": False, "error": "usage: reference_train_check.py <run_dir> <tabular|manifest>"}))
        return 0
    run_dir = Path(sys.argv[1])
    kind = sys.argv[2]
    try:
        print(json.dumps(_run(run_dir, kind)))
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc(limit=6)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
