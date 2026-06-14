"""Dataset inspection & stats helpers for the MLForge sidecar.

Six dataset kinds — tabular (csv/parquet), image-folder, tensor (.pt/.npy),
huggingface (by name), protein (.pdb), molecule (SMILES file).

Each kind has three operations: detect/inspect (cheap metadata), stats
(more expensive — distributions, correlations), sample (returns a torch
tensor suitable as model input, used by smoke_test).

Heavy deps (pandas, PIL, rdkit, biopython, datasets) are imported lazily
inside the handler functions so an environment missing one library can
still serve the others.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any

import torch


# ─── Detection ────────────────────────────────────────────────────────────

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
TABULAR_EXTS = {".csv", ".tsv", ".parquet"}
TENSOR_EXTS = {".pt", ".pth", ".npy", ".npz"}
SMILES_EXTS = {".smi", ".smiles"}
PROTEIN_EXTS = {".pdb"}


def detect_kind(abspath: str) -> str:
    """Return 'tabular' | 'image_folder' | 'tensor' | 'protein' | 'molecule'
    | 'huggingface' | 'unknown'."""
    p = Path(abspath)
    if not p.exists():
        return "unknown"
    if p.is_file():
        ext = p.suffix.lower()
        if ext in TABULAR_EXTS:
            return "tabular"
        if ext in TENSOR_EXTS:
            return "tensor"
        if ext in PROTEIN_EXTS:
            return "protein"
        if ext in SMILES_EXTS:
            return "molecule"
        # .txt could be SMILES — sniff first line
        if ext == ".txt":
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    first = f.readline().strip()
                if first and _looks_like_smiles(first.split()[0]):
                    return "molecule"
            except OSError:
                pass
        # HuggingFace reference file: contains 'hf:<name>'
        if ext == ".hf":
            return "huggingface"
        return "unknown"
    # Directory: image-folder if it has class subdirs with images
    if _looks_like_image_folder(p):
        return "image_folder"
    return "unknown"


def _looks_like_smiles(token: str) -> bool:
    # Very rough: SMILES strings tend to have these chars, and no whitespace.
    if not token or " " in token:
        return False
    smiles_chars = set("CcNnOoSsPpFIBrClHcnos()[]=#@+-./\\1234567890")
    hits = sum(1 for c in token if c in smiles_chars)
    return hits / max(1, len(token)) > 0.85


def _looks_like_image_folder(path: Path) -> bool:
    try:
        subdirs = [c for c in path.iterdir() if c.is_dir()]
    except OSError:
        return False
    if not subdirs:
        return False
    for sub in subdirs[:6]:
        try:
            for f in sub.iterdir():
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                    return True
        except OSError:
            continue
    return False


# ─── Inspect (cheap metadata) ─────────────────────────────────────────────


def inspect(abspath: str) -> dict[str, Any]:
    # Phase 12b: remote workspaces send tilde-prefixed paths ('~/mlforge/...');
    # Python's os.path doesn't expand those, so we do it once at the entry point.
    abspath = os.path.expanduser(abspath)
    kind = detect_kind(abspath)
    if kind == "tabular":
        return _inspect_tabular(abspath)
    if kind == "image_folder":
        return _inspect_image_folder(abspath)
    if kind == "tensor":
        return _inspect_tensor(abspath)
    if kind == "protein":
        return _inspect_protein(abspath)
    if kind == "molecule":
        return _inspect_molecule(abspath)
    if kind == "huggingface":
        return _inspect_huggingface(abspath)
    return {"kind": "unknown", "ok": False, "error": "could not detect dataset kind"}


def _inspect_tabular(abspath: str) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        return _missing_dep("tabular", "pandas")
    p = Path(abspath)
    try:
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        elif p.suffix.lower() == ".tsv":
            df = pd.read_csv(p, sep="\t")
        else:
            df = pd.read_csv(p)
    except Exception as e:
        return {"kind": "tabular", "ok": False, "error": f"{type(e).__name__}: {e}"}
    head = df.head(10).fillna("").astype(str).values.tolist()
    return {
        "kind": "tabular",
        "ok": True,
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "dtypes": [str(t) for t in df.dtypes],
        "head": head,
        "size_bytes": p.stat().st_size,
    }


def _inspect_image_folder(abspath: str) -> dict[str, Any]:
    p = Path(abspath)
    classes: list[dict[str, Any]] = []
    total = 0
    sample_paths: list[Path] = []
    for sub in sorted(p.iterdir()):
        if not sub.is_dir():
            continue
        imgs = [f for f in sub.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        if not imgs:
            continue
        classes.append({"name": sub.name, "count": len(imgs)})
        total += len(imgs)
        if len(sample_paths) < 8 and imgs:
            sample_paths.append(imgs[0])
    sample_size = None
    thumbnails: list[dict[str, Any]] = []
    try:
        from PIL import Image
        for sp in sample_paths[:6]:
            try:
                img = Image.open(sp)
                if sample_size is None:
                    sample_size = list(img.size)  # (w, h)
                thumb = img.copy()
                thumb.thumbnail((96, 96))
                buf = io.BytesIO()
                thumb.convert("RGB").save(buf, format="JPEG", quality=72)
                thumbnails.append({
                    "name": sp.parent.name + "/" + sp.name,
                    "b64": base64.b64encode(buf.getvalue()).decode(),
                    "w": img.size[0], "h": img.size[1],
                })
            except Exception:
                continue
    except ImportError:
        pass
    return {
        "kind": "image_folder",
        "ok": True,
        "classes": classes,
        "n_classes": len(classes),
        "n_images": total,
        "sample_size": sample_size,
        "thumbnails": thumbnails,
    }


def _inspect_tensor(abspath: str) -> dict[str, Any]:
    p = Path(abspath)
    ext = p.suffix.lower()
    try:
        if ext in (".pt", ".pth"):
            t = torch.load(p, map_location="cpu", weights_only=False)
        elif ext == ".npy":
            import numpy as np
            t = torch.from_numpy(np.load(p, allow_pickle=False))
        elif ext == ".npz":
            import numpy as np
            data = np.load(p, allow_pickle=False)
            arrays = {k: list(data[k].shape) for k in data.files}
            return {
                "kind": "tensor", "ok": True, "container": "npz",
                "arrays": arrays, "size_bytes": p.stat().st_size,
            }
        else:
            return {"kind": "tensor", "ok": False, "error": f"unsupported tensor ext {ext}"}
    except Exception as e:
        return {"kind": "tensor", "ok": False, "error": f"{type(e).__name__}: {e}"}
    info: dict[str, Any] = {"kind": "tensor", "ok": True, "size_bytes": p.stat().st_size}
    if isinstance(t, torch.Tensor):
        info["shape"] = list(t.shape)
        info["dtype"] = str(t.dtype)
        try:
            ft = t.float()
            info["min"] = float(ft.min())
            info["max"] = float(ft.max())
            info["mean"] = float(ft.mean())
        except Exception:
            pass
    elif isinstance(t, dict):
        info["container"] = "dict"
        info["keys"] = [
            {"key": str(k), "shape": list(v.shape) if isinstance(v, torch.Tensor) else None,
             "dtype": str(v.dtype) if isinstance(v, torch.Tensor) else type(v).__name__}
            for k, v in list(t.items())[:32]
        ]
    else:
        info["container"] = type(t).__name__
    return info


def _inspect_protein(abspath: str) -> dict[str, Any]:
    p = Path(abspath)
    # Try Biopython first; fall back to a simple line-parser.
    try:
        from Bio.PDB import PDBParser
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("s", str(p))
        chains, residues, atoms = 0, 0, 0
        chain_info: list[dict[str, Any]] = []
        for model in structure:
            for chain in model:
                chains += 1
                rs = list(chain.get_residues())
                ats = sum(1 for _ in chain.get_atoms())
                residues += len(rs)
                atoms += ats
                chain_info.append({"id": chain.id, "residues": len(rs), "atoms": ats})
        return {
            "kind": "protein", "ok": True, "parser": "biopython",
            "chains": chains, "residues": residues, "atoms": atoms,
            "chain_info": chain_info[:16],
            "size_bytes": p.stat().st_size,
        }
    except ImportError:
        pass
    except Exception as e:
        return {"kind": "protein", "ok": False, "error": f"biopython parse failed: {e}"}
    chains_seen: set[str] = set()
    residues_seen: set[tuple] = set()
    atoms = 0
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    atoms += 1
                    if len(line) >= 26:
                        cid = line[21]
                        rseq = line[22:26].strip()
                        chains_seen.add(cid)
                        residues_seen.add((cid, rseq))
    except OSError as e:
        return {"kind": "protein", "ok": False, "error": str(e)}
    return {
        "kind": "protein", "ok": True, "parser": "fallback",
        "chains": len(chains_seen), "residues": len(residues_seen), "atoms": atoms,
        "size_bytes": p.stat().st_size,
    }


def _inspect_molecule(abspath: str) -> dict[str, Any]:
    p = Path(abspath)
    smiles: list[str] = []
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                tok = line.strip().split()
                if tok and _looks_like_smiles(tok[0]):
                    smiles.append(tok[0])
    except OSError as e:
        return {"kind": "molecule", "ok": False, "error": str(e)}
    info: dict[str, Any] = {
        "kind": "molecule", "ok": True,
        "n_molecules": len(smiles),
        "head": smiles[:10],
        "size_bytes": p.stat().st_size,
    }
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        canonicals: list[dict[str, Any]] = []
        for s in smiles[:8]:
            m = Chem.MolFromSmiles(s)
            if m is None:
                canonicals.append({"smiles": s, "valid": False})
                continue
            canonicals.append({
                "smiles": s,
                "canonical": Chem.MolToSmiles(m),
                "atoms": m.GetNumAtoms(),
                "bonds": m.GetNumBonds(),
                "mw": round(Descriptors.MolWt(m), 3),
                "valid": True,
            })
        info["parser"] = "rdkit"
        info["sample_info"] = canonicals
    except ImportError:
        info["parser"] = "fallback"
    return info


def _inspect_huggingface(abspath: str) -> dict[str, Any]:
    p = Path(abspath)
    try:
        text = p.read_text(encoding="utf-8").strip()
    except OSError as e:
        return {"kind": "huggingface", "ok": False, "error": str(e)}
    name = text
    if name.lower().startswith("hf:"):
        name = name[3:].strip()
    if not name:
        return {"kind": "huggingface", "ok": False, "error": "empty hf: reference"}
    try:
        from datasets import load_dataset_builder
    except ImportError:
        return _missing_dep("huggingface", "datasets")
    try:
        builder = load_dataset_builder(name)
        info = builder.info
        splits = {}
        if info.splits:
            for k, sp in info.splits.items():
                splits[str(k)] = {"num_examples": getattr(sp, "num_examples", None)}
        features = {}
        if info.features:
            for k, v in info.features.items():
                features[str(k)] = str(v)
        return {
            "kind": "huggingface", "ok": True,
            "name": name,
            "description": (info.description or "")[:600],
            "splits": splits,
            "features": features,
        }
    except Exception as e:
        return {"kind": "huggingface", "ok": False, "error": f"{type(e).__name__}: {e}", "name": name}


def _missing_dep(kind: str, dep: str) -> dict[str, Any]:
    return {
        "kind": kind, "ok": False,
        "error": f"required dependency '{dep}' not installed in the sidecar's Python env",
        "missing_dep": dep,
    }


# ─── Stats (more expensive) ───────────────────────────────────────────────


def stats(abspath: str) -> dict[str, Any]:
    abspath = os.path.expanduser(abspath)
    kind = detect_kind(abspath)
    if kind == "tabular":
        return _stats_tabular(abspath)
    if kind == "image_folder":
        return _stats_image_folder(abspath)
    if kind == "tensor":
        return _stats_tensor(abspath)
    if kind == "molecule":
        return _stats_molecule(abspath)
    if kind == "protein":
        return {"kind": "protein", "ok": True, "note": "stats limited to inspect for protein structures"}
    if kind == "huggingface":
        return {"kind": "huggingface", "ok": True, "note": "stats limited to inspect for HuggingFace refs"}
    return {"kind": kind, "ok": False, "error": "unsupported"}


def _stats_tabular(abspath: str) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        return _missing_dep("tabular", "pandas")
    p = Path(abspath)
    try:
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        elif p.suffix.lower() == ".tsv":
            df = pd.read_csv(p, sep="\t")
        else:
            df = pd.read_csv(p)
    except Exception as e:
        return {"kind": "tabular", "ok": False, "error": str(e)}
    numeric = df.select_dtypes(include="number")
    desc = numeric.describe().fillna(0).round(4)
    summary: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        item: dict[str, Any] = {
            "col": str(col),
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "unique": int(s.nunique(dropna=True)),
        }
        if col in numeric.columns:
            item["mean"] = float(desc.loc["mean", col])
            item["std"] = float(desc.loc["std", col])
            item["min"] = float(desc.loc["min", col])
            item["max"] = float(desc.loc["max", col])
            # Build a small histogram for the UI.
            try:
                import numpy as np
                vals = s.dropna().values
                if len(vals) > 0:
                    hist, edges = np.histogram(vals, bins=20)
                    item["hist"] = {
                        "counts": [int(c) for c in hist],
                        "edges": [float(e) for e in edges],
                    }
            except Exception:
                pass
        summary.append(item)
    corr: list[list[float]] | None = None
    corr_cols: list[str] = []
    if numeric.shape[1] >= 2 and numeric.shape[1] <= 32:
        c = numeric.corr().fillna(0).round(3)
        corr = [[float(v) for v in row] for row in c.values]
        corr_cols = [str(x) for x in c.columns]
    return {
        "kind": "tabular", "ok": True,
        "rows": int(len(df)), "cols": int(df.shape[1]),
        "summary": summary,
        "corr": corr, "corr_cols": corr_cols,
    }


def _stats_image_folder(abspath: str) -> dict[str, Any]:
    p = Path(abspath)
    classes: list[dict[str, Any]] = []
    sizes: list[tuple[int, int]] = []
    try:
        from PIL import Image
    except ImportError:
        Image = None  # type: ignore
    for sub in sorted(p.iterdir()):
        if not sub.is_dir():
            continue
        imgs = [f for f in sub.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        if not imgs:
            continue
        classes.append({"name": sub.name, "count": len(imgs)})
        if Image is not None:
            for f in imgs[:3]:
                try:
                    with Image.open(f) as im:
                        sizes.append(im.size)
                except Exception:
                    continue
    size_hist: dict[str, int] = {}
    for w, h in sizes:
        key = f"{w}x{h}"
        size_hist[key] = size_hist.get(key, 0) + 1
    return {
        "kind": "image_folder", "ok": True,
        "classes": classes,
        "size_hist": [{"size": k, "count": v} for k, v in sorted(size_hist.items(), key=lambda kv: -kv[1])[:16]],
        "n_samples_for_size": len(sizes),
    }


def _stats_tensor(abspath: str) -> dict[str, Any]:
    info = _inspect_tensor(abspath)
    if not info.get("ok"):
        return info
    p = Path(abspath)
    ext = p.suffix.lower()
    try:
        if ext in (".pt", ".pth"):
            t = torch.load(p, map_location="cpu", weights_only=False)
        elif ext == ".npy":
            import numpy as np
            t = torch.from_numpy(np.load(p, allow_pickle=False))
        else:
            return info
    except Exception as e:
        return {"kind": "tensor", "ok": False, "error": str(e)}
    if not isinstance(t, torch.Tensor):
        return info
    ft = t.flatten().float()
    if ft.numel() == 0:
        return info
    sample = ft if ft.numel() <= 10_000 else ft[torch.randperm(ft.numel())[:10_000]]
    try:
        import numpy as np
        hist, edges = np.histogram(sample.numpy(), bins=40)
        info["hist"] = {"counts": [int(c) for c in hist], "edges": [float(e) for e in edges]}
    except Exception:
        pass
    info["std"] = float(ft.std())
    info["zeros_frac"] = float((ft == 0).float().mean())
    return info


def _stats_molecule(abspath: str) -> dict[str, Any]:
    base = _inspect_molecule(abspath)
    if not base.get("ok"):
        return base
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:
        base["note"] = "stats requires rdkit"
        return base
    p = Path(abspath)
    mws: list[float] = []
    atoms: list[int] = []
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= 5000:
                break
            tok = line.strip().split()
            if not tok:
                continue
            m = Chem.MolFromSmiles(tok[0])
            if m is None:
                continue
            mws.append(Descriptors.MolWt(m))
            atoms.append(m.GetNumAtoms())
    if mws:
        import numpy as np
        h_mw, e_mw = np.histogram(mws, bins=30)
        h_a, e_a = np.histogram(atoms, bins=range(min(atoms), max(atoms) + 2))
        base["mw_hist"] = {"counts": [int(c) for c in h_mw], "edges": [float(e) for e in e_mw]}
        base["atom_hist"] = {"counts": [int(c) for c in h_a], "edges": [int(e) for e in e_a]}
        base["mw_mean"] = round(float(np.mean(mws)), 3)
        base["atom_mean"] = round(float(np.mean(atoms)), 3)
    return base


# ─── Sample → torch.Tensor (for smoke test) ───────────────────────────────


def sample_tensor(
    abspath: str,
    target_shape: list[int] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one input tensor with batch dim from the dataset.

    target_shape: shape hint (trim/pad/resize). If None, the natural shape is used.
    options: per-input bag — currently {features: list[str]} for tabular.
    """
    abspath = os.path.expanduser(abspath)
    kind = detect_kind(abspath)
    if kind == "tabular":
        feats = (options or {}).get("features") if options else None
        return _sample_tabular(abspath, target_shape, feats if isinstance(feats, list) else None)
    if kind == "image_folder":
        return _sample_image_folder(abspath, target_shape)
    if kind == "tensor":
        return _sample_tensor_file(abspath, target_shape)
    if kind == "molecule":
        return _sample_molecule(abspath, target_shape)
    if kind == "protein":
        return _sample_protein(abspath, target_shape)
    return {"ok": False, "error": f"sampling not supported for kind {kind}"}


def _sample_tabular(
    abspath: str,
    target: list[int] | None,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        return _missing_dep("tabular", "pandas")
    p = Path(abspath)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix.lower() == ".tsv":
        df = pd.read_csv(p, sep="\t")
    else:
        df = pd.read_csv(p)
    if feature_cols:
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            return {"ok": False, "error": f"feature columns not in dataset: {missing}"}
        feats = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        note_src = f"{len(feature_cols)} chosen cols ({', '.join(feature_cols[:4])}{'…' if len(feature_cols) > 4 else ''})"
    else:
        feats = df.select_dtypes(include="number").fillna(0)
        note_src = f"{feats.shape[1]} numeric cols"
    if feats.shape[1] == 0:
        return {"ok": False, "error": "no usable feature columns in tabular dataset"}
    batch = 1
    n_feat = feats.shape[1]
    if target and len(target) == 2:
        batch = max(1, target[0])
        n_feat = target[1]
    elif target and len(target) == 1:
        n_feat = target[0]
    rows = feats.iloc[:batch].values
    if rows.shape[1] < n_feat:
        import numpy as np
        pad = np.zeros((rows.shape[0], n_feat - rows.shape[1]))
        rows = np.concatenate([rows, pad], axis=1)
    elif rows.shape[1] > n_feat:
        rows = rows[:, :n_feat]
    while rows.shape[0] < batch:
        rows = rows.repeat(2, axis=0)[:batch]
    return {"ok": True, "tensor": torch.tensor(rows, dtype=torch.float32),
            "natural_shape": [batch, n_feat], "note": f"used first {batch} rows × {note_src}"}


def _sample_image_folder(abspath: str, target: list[int] | None) -> dict[str, Any]:
    try:
        from PIL import Image
    except ImportError:
        return _missing_dep("image_folder", "Pillow")
    p = Path(abspath)
    # Find one image.
    img_path: Path | None = None
    for sub in sorted(p.iterdir()):
        if sub.is_dir():
            for f in sub.iterdir():
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                    img_path = f
                    break
        if img_path:
            break
    if img_path is None:
        return {"ok": False, "error": "no images found in image folder"}
    # target shape: [N, C, H, W] or [C, H, W]
    n, c, h, w = 1, 3, 64, 64
    if target and len(target) == 4:
        n, c, h, w = target
    elif target and len(target) == 3:
        c, h, w = target
    img = Image.open(img_path).convert("RGB" if c == 3 else "L")
    img = img.resize((w, h))
    import numpy as np
    arr = np.asarray(img, dtype=np.float32) / 255.0  # H,W,C or H,W
    if c == 1 and arr.ndim == 2:
        arr = arr[None, :, :]
    else:
        arr = arr.transpose(2, 0, 1)
    t = torch.tensor(arr).unsqueeze(0)
    if n > 1:
        t = t.expand(n, -1, -1, -1).contiguous()
    return {"ok": True, "tensor": t, "natural_shape": list(t.shape),
            "note": f"resized 1 image from {img_path.parent.name}/ to {c}×{h}×{w}, batched ×{n}"}


def _sample_tensor_file(abspath: str, target: list[int] | None) -> dict[str, Any]:
    p = Path(abspath)
    ext = p.suffix.lower()
    if ext in (".pt", ".pth"):
        t = torch.load(p, map_location="cpu", weights_only=False)
    elif ext == ".npy":
        import numpy as np
        t = torch.from_numpy(np.load(p, allow_pickle=False))
    else:
        return {"ok": False, "error": f"sampling not supported for tensor ext {ext}"}
    if isinstance(t, dict):
        t = next(iter(t.values()))
    if not isinstance(t, torch.Tensor):
        return {"ok": False, "error": f"loaded object is {type(t).__name__}, not a tensor"}
    t = t.float()
    # If target_shape is given and matches in length, use it; otherwise return as-is with batch dim.
    if target:
        try:
            t = t.reshape(*target)
        except Exception:
            return {"ok": False, "error": f"cannot reshape tensor of shape {list(t.shape)} to target {target}"}
    elif t.ndim == len(t.shape) and t.shape[0] != 1:
        t = t.unsqueeze(0) if t.ndim < 4 else t
    return {"ok": True, "tensor": t, "natural_shape": list(t.shape)}


def _sample_molecule(abspath: str, target: list[int] | None) -> dict[str, Any]:
    # Without a learned tokenizer we fall back to a fixed-length integer encoding
    # of the first SMILES string. Useful only to confirm the model accepts the shape.
    smiles: list[str] = []
    with open(abspath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tok = line.strip().split()
            if tok and _looks_like_smiles(tok[0]):
                smiles.append(tok[0])
            if len(smiles) >= 4:
                break
    if not smiles:
        return {"ok": False, "error": "no parseable SMILES in file"}
    max_len = 64
    if target and len(target) >= 2:
        max_len = target[-1]
    rows = []
    for s in smiles[: (target[0] if target and len(target) >= 1 else 1)]:
        ids = [ord(c) % 128 for c in s[:max_len]]
        ids += [0] * (max_len - len(ids))
        rows.append(ids)
    t = torch.tensor(rows, dtype=torch.long)
    return {"ok": True, "tensor": t, "natural_shape": list(t.shape),
            "note": "byte-encoded SMILES, no real tokenizer — smoke-test only"}


def _sample_protein(abspath: str, target: list[int] | None) -> dict[str, Any]:
    # Encode first chain's residue sequence as integer IDs.
    seq: list[int] = []
    AA = "ACDEFGHIKLMNPQRSTVWY"
    aa_idx = {a: i + 1 for i, a in enumerate(AA)}
    try:
        with open(abspath, "r", encoding="utf-8", errors="ignore") as f:
            seen: set[tuple] = set()
            for line in f:
                if line.startswith("ATOM") and len(line) >= 26:
                    cid = line[21]
                    rseq = line[22:26].strip()
                    key = (cid, rseq)
                    if key in seen:
                        continue
                    seen.add(key)
                    rname = line[17:20].strip()
                    one = _three_to_one(rname)
                    seq.append(aa_idx.get(one, 0))
    except OSError as e:
        return {"ok": False, "error": str(e)}
    if not seq:
        return {"ok": False, "error": "could not extract residue sequence"}
    max_len = 256
    batch = 1
    if target and len(target) >= 2:
        batch, max_len = target[0], target[-1]
    elif target and len(target) == 1:
        max_len = target[0]
    seq = (seq[:max_len] + [0] * max_len)[:max_len]
    t = torch.tensor([seq] * batch, dtype=torch.long)
    return {"ok": True, "tensor": t, "natural_shape": list(t.shape),
            "note": "residue-index encoding (A=1..Y=20)"}


_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _three_to_one(three: str) -> str:
    return _THREE_TO_ONE.get(three.upper(), "X")
