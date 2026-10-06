"""Dataset inspection & stats helpers for the SpinoML sidecar.

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
import hashlib
import io
import os
from pathlib import Path
from typing import Any

import torch

import scope
from safe_load import UnsafePickleError, safe_torch_load
from scope import ScopeError

# Phase 46 — audited open sites (every one goes through scope.check_path and uses
# the returned resolved path):
#   * entry points inspect() / stats() / sample_tensor()                  (request abspath)
#   * _table_path() inner table of a prepared dataset dir                 (dir content)
#   * _read_manifest() / _manifest_table_df() table                       (file content)
#   * _resolve_branch_file() dir listing + exact/contains/absolute cell   (file content)
#   * _lookup_value() side-table join                                     (file content)
#   * _cached_mol_data() .graphcache read/write and ensure_espf_cache()   (cache dirs)
#   * image folders: each image actually opened/thumbnailed               (dir content)
#   * graph folders: each .pt actually loaded                             (dir content)
#   * _describe_dir_bundle() prep_card.json                               (dir content)
#   * _sha256_file() / _fingerprint_dir() fingerprint reads               (request path)


def _scope_error(exc: "ScopeError", kind: str = "unknown") -> dict[str, Any]:
    """Turn a scope denial into the module's normal data-level error dict."""
    return {"kind": kind, "ok": False,
            "error": getattr(exc, "message", str(exc)) or str(exc),
            "error_code": getattr(exc, "code", "SCOPE_DENIED")}


def _checked(path: Any, write: bool = False) -> str:
    """Scope-check ``path`` and return the resolved absolute path to open."""
    return scope.check_path(str(path), write=write)


def _mark_unsafe_pickle(result: dict[str, Any], exc: BaseException) -> None:
    """Tag a handler error dict when the failure is a refused unsafe-pickle load."""
    if isinstance(exc, UnsafePickleError):
        result["error_code"] = "UNSAFE_PICKLE"
        result["error"] = str(exc)


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
            except OSError:  # sniffing is heuristic: an unreadable .txt just stays "unknown"
                pass
        # HuggingFace reference file: contains 'hf:<name>'
        if ext == ".hf":
            return "huggingface"
        # PyTorch Geometric dataset reference: contains 'pyg:<Class[/Name]>'
        if ext == ".pyg":
            return "pyg"
        # Paired/manifest dataset: a JSON descriptor gluing a table to per-branch
        # graph sources (ligand + protein, etc.) + a target column.
        if ext == ".manifest":
            return "manifest"
        return "unknown"
    # Directory of .pt/.pth graph files → a graph dataset (one Data per file).
    if _looks_like_graph_folder(p):
        return "graph_folder"
    # Directory: image-folder if it has class subdirs with images
    if _looks_like_image_folder(p):
        return "image_folder"
    # Prepared dataset directory: a folder whose primary content is a table
    # (pairs.csv / data.csv / the main CSV), optionally with side files
    # (sequences.csv, per-id embeddings, a prep_card.json) — e.g. a TDC BindingDB
    # export. Read it as tabular on its inner table; the bundle is surfaced in
    # inspect so the structure stays visible.
    if _dir_table(p) is not None:
        return "tabular"
    return "unknown"


# Preferred table filenames inside a prepared dataset directory, in priority order.
_DIR_TABLE_NAMES = ("pairs.csv", "data.csv", "table.csv", "dataset.csv", "train.csv", "test.csv")


def _dir_table(p: Path) -> Path | None:
    """The primary table file inside a dataset directory (or None). Prefers a
    conventional name, else the largest tabular file directly under the dir."""
    try:
        if not p.is_dir():
            return None
        files = [c for c in p.iterdir() if c.is_file() and c.suffix.lower() in TABULAR_EXTS]
    except OSError:  # unreadable dir → no primary table here (kind stays unknown)
        return None
    if not files:
        return None
    by_name = {c.name.lower(): c for c in files}
    for name in _DIR_TABLE_NAMES:
        if name in by_name:
            return by_name[name]
    # fall back to the biggest table file
    try:
        return max(files, key=lambda c: c.stat().st_size)
    except OSError:
        return files[0]


def _table_path(abspath: str) -> Path:
    """Resolve a tabular source to its actual file: a directory → its inner table
    (prepared dataset dir), a file → itself."""
    p = Path(abspath)
    if p.is_dir():
        t = _dir_table(p)
        if t is not None:
            return t
    return p


def _looks_like_smiles(token: str) -> bool:
    # Very rough: SMILES strings tend to have these chars, and no whitespace.
    if not token or " " in token:
        return False
    smiles_chars = set("CcNnOoSsPpFIBrClHcnos()[]=#@+-./\\1234567890")
    hits = sum(1 for c in token if c in smiles_chars)
    return hits / max(1, len(token)) > 0.85


def _list_pt_files(path: Path, cap: int = 100000) -> list[Path]:
    """`.pt`/`.pth` files directly under `path` (sorted), capped for speed."""
    try:
        files = sorted(c for c in path.iterdir() if c.is_file() and c.suffix.lower() in (".pt", ".pth"))
    except OSError:  # unreadable dir → no .pt files discovered (kind stays unknown)
        return []
    return files[:cap]


def _looks_like_graph_folder(path: Path) -> bool:
    return len(_list_pt_files(path, cap=1)) > 0


def _looks_like_image_folder(path: Path) -> bool:
    try:
        subdirs = [c for c in path.iterdir() if c.is_dir()]
    except OSError:  # unreadable dir → not recognized as an image folder
        return False
    if not subdirs:
        return False
    for sub in subdirs[:6]:
        try:
            for f in sub.iterdir():
                if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                    return True
        except OSError:  # unreadable subdir → skip it when probing for images
            continue
    return False


# ─── Inspect (cheap metadata) ─────────────────────────────────────────────


def _missing_path(abspath: str) -> dict[str, Any]:
    """One explicit shape for 'the path does not exist' — a missing file or
    directory is a dataset error, never silently 'unknown' or empty."""
    return {"kind": "unknown", "ok": False, "error": f"file not found: {abspath}"}


# ─── Dataset fingerprinting (Phase 18) ─────────────────────────────────────

_FP_HEADER = b"spinoml-dataset-fp-v1\x00"


def _sha256_file(path: Path) -> tuple[str, int]:
    """Stream a SHA-256 over the file bytes → (hex hash, size). Copy-stable:
    the hash depends only on content, so a dataset copied to a new host or
    renamed keeps the same identifier."""
    h = hashlib.sha256(_FP_HEADER)
    size = 0
    path = Path(_checked(path))
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            size += len(chunk)
            h.update(chunk)
    return h.hexdigest(), size


def _fingerprint_dir(path: Path, exts: set[str] | None) -> dict[str, Any] | None:
    """Structural fingerprint for a dataset DIRECTORY: the canonical sorted
    (relpath, size) listing, hashed. Deterministic across renames/copies —
    does NOT depend on mtimes or read order. Returns None for an empty dir."""
    entries: list[tuple[str, int]] = []
    total = 0
    for p in path.rglob("*"):
        if not p.is_file():
            continue
        if exts and p.suffix.lower() not in exts:
            continue
        try:
            rp = Path(_checked(p))
        except ScopeError:
            continue  # never hash a file that resolves outside the scope
        entries.append((rp.relative_to(path).as_posix(), rp.stat().st_size))
        total += rp.stat().st_size
    if not entries:
        return None
    h = hashlib.sha256(_FP_HEADER)
    h.update(f"dir {len(entries)}\n".encode())
    for rel, size in sorted(entries):
        h.update(f"{rel}:{size}\n".encode())
    return {"alg": "sha256", "mode": "structure", "hash": h.hexdigest(),
            "size_bytes": total, "n_files": len(entries)}


def _fingerprint_manifest(abspath: str) -> dict[str, Any] | None:
    """A .manifest glues a JSON descriptor to its table. Fingerprint both so
    changing the TABLE (the actual data) changes the id. Derived cache
    artifacts (.graphcache) are excluded on purpose."""
    try:
        cfg = _read_manifest(abspath)
        main_hash, main_size = _sha256_file(Path(abspath))
        parts: list[tuple[str, str, int]] = [("manifest", main_hash, main_size)]
        tbl = cfg.get("table")
        if isinstance(tbl, str):
            tp = Path(abspath).resolve().parent / tbl
            if tp.is_file():
                th, ts = _sha256_file(tp)
                parts.append(("table", th, ts))
        h = hashlib.sha256(_FP_HEADER)
        for name, hx, size in parts:
            h.update(f"{name}:{size}:{hx}\n".encode())
        return {"alg": "sha256", "mode": "config+content", "hash": h.hexdigest(),
                "size_bytes": sum(p[2] for p in parts), "n_files": len(parts)}
    except Exception:  # fingerprint is optional provenance; None = unknown, never a false id
        return None


def _fingerprint_for(abspath: str, kind: str) -> dict[str, Any] | None:
    """Kind-aware stable dataset identifier, attached to ok inspect results.
    Copy-stable, deterministic, and content-derived — never a human-readable
    name. Returns None (key omitted) when the source cannot be fingerprinted."""
    try:
        if kind == "tabular":
            p = _table_path(abspath)  # prepared-dataset dir → its inner table
            h, size = _sha256_file(p)
            return {"alg": "sha256", "mode": "content", "hash": h, "size_bytes": size}
        if kind == "image_folder":
            return _fingerprint_dir(Path(abspath), IMAGE_EXTS)
        if kind == "graph_folder":
            return _fingerprint_dir(Path(abspath), set(".pt .pth".split()))
        if kind == "manifest":
            return _fingerprint_manifest(abspath)
        if kind == "tensor":
            h, size = _sha256_file(Path(abspath))
            return {"alg": "sha256", "mode": "content", "hash": h, "size_bytes": size}
        if kind in ("molecule", "protein"):
            h, size = _sha256_file(Path(abspath))
            return {"alg": "sha256", "mode": "content", "hash": h, "size_bytes": size}
        if kind in ("pyg", "huggingface"):
            # The actual data lives elsewhere (a torch_geometric/repo dataset);
            # the hash pins the reference file, NOT the remote content.
            h, size = _sha256_file(Path(abspath))
            return {"alg": "sha256", "mode": "reference", "hash": h, "size_bytes": size}
    except (OSError, ScopeError):  # fingerprint is optional provenance; None = no id claimed
        return None
    return None


def inspect(abspath: str) -> dict[str, Any]:
    # Phase 12b: remote workspaces send tilde-prefixed paths ('~/spinoml/...');
    # Python's os.path doesn't expand those, so we do it once at the entry point.
    abspath = os.path.expanduser(abspath)
    try:
        abspath = _checked(abspath)
    except ScopeError as e:
        return _scope_error(e)
    if not os.path.exists(abspath):
        return _missing_path(abspath)
    kind = detect_kind(abspath)
    if kind == "tabular":
        info = _inspect_tabular(abspath)
    elif kind == "image_folder":
        info = _inspect_image_folder(abspath)
    elif kind == "graph_folder":
        info = _inspect_graph_folder(abspath)
    elif kind == "tensor":
        info = _inspect_tensor(abspath)
    elif kind == "protein":
        info = _inspect_protein(abspath)
    elif kind == "molecule":
        info = _inspect_molecule(abspath)
    elif kind == "huggingface":
        info = _inspect_huggingface(abspath)
    elif kind == "pyg":
        info = _inspect_pyg(abspath)
    elif kind == "manifest":
        info = _inspect_manifest(abspath)
    else:
        return {"kind": "unknown", "ok": False, "error": "could not detect dataset kind"}
    # Attach the stable content fingerprint to every ok result — one change
    # point so all kinds stay in sync (Phase 18).
    if isinstance(info, dict) and info.get("ok"):
        fp = _fingerprint_for(abspath, str(info.get("kind")))
        if fp:
            info["fingerprint"] = fp
    return info


def _inspect_tabular(abspath: str) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        return _missing_dep("tabular", "pandas")
    src = Path(abspath)
    p = _table_path(abspath)  # a prepared-dataset dir → its inner table
    try:
        p = Path(_checked(p))
    except ScopeError as e:
        return _scope_error(e, "tabular")
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
    out: dict[str, Any] = {
        "kind": "tabular",
        "ok": True,
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns],
        "dtypes": [str(t) for t in df.dtypes],
        "head": head,
        "size_bytes": p.stat().st_size,
    }
    if src.is_dir():
        out["table"] = p.name  # the table inside the prepared dir
        out["bundle"] = _describe_dir_bundle(src, p)
    return out


def _describe_dir_bundle(d: Path, table: Path) -> dict[str, Any]:
    """Summarize a prepared dataset directory's side files so the UI can show that
    the structure is fully readable (sequences, per-id embeddings, prep card …)."""
    import json
    files: list[str] = []
    subdirs: list[dict[str, Any]] = []
    card: dict[str, Any] | None = None
    try:
        for c in sorted(d.iterdir()):
            if c.name.startswith("."):
                continue
            if c.is_dir():
                try:
                    n = sum(1 for _ in c.iterdir())
                except OSError:
                    n = 0
                subdirs.append({"name": c.name, "entries": n})
            elif c != table:
                files.append(c.name)
            if c.name == "prep_card.json":
                try:
                    card = json.loads(Path(_checked(c)).read_text(encoding="utf-8"))
                except ScopeError:
                    card = None
                except Exception:
                    card = None
    except OSError:  # unreadable dir → empty side-file summary; the table itself is valid
        pass
    return {"files": files, "subdirs": subdirs, "prep_card": card}


def _inspect_image_folder(abspath: str) -> dict[str, Any]:
    p = Path(abspath)
    classes: list[dict[str, Any]] = []
    total = 0
    sample_paths: list[Path] = []
    try:
        for sub in sorted(p.iterdir()):
            if not sub.is_dir():
                continue
            try:
                imgs = [f for f in sub.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
            except OSError:
                imgs = []
            if not imgs:
                continue
            classes.append({"name": sub.name, "count": len(imgs)})
            total += len(imgs)
            if len(sample_paths) < 8 and imgs:
                sample_paths.append(imgs[0])
    except OSError as e:
        # Unreadable dir (permissions, IO) is an EXPLICIT error, not an empty set.
        return {"kind": "image_folder", "ok": False, "error": f"could not read image folder: {e}"}
    sample_size = None
    thumbnails: list[dict[str, Any]] = []
    try:
        from PIL import Image
        for sp in sample_paths[:6]:
            try:
                sp = Path(_checked(sp))
            except ScopeError:
                continue  # a symlinked-out image is never opened/thumbnailed
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
            except Exception:  # a failed thumbnail is skipped; the others stay valid
                continue
    except ImportError:  # optional dependency: Pillow absent → no thumbnails, structure still shown
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


def _as_pyg_data(obj):
    """Return a PyG Data-like object from a loaded .pt payload, or None.
    Accepts a Data, a list/tuple of Data (takes [0]), or an InMemoryDataset
    (data, slices) tuple (uses the concatenated `data`)."""
    if obj is None:
        return None
    if hasattr(obj, "edge_index") and hasattr(obj, "num_nodes"):
        return obj
    if isinstance(obj, (list, tuple)) and obj:
        first = obj[0]
        if hasattr(first, "edge_index"):
            return first
    return None


def _graph_fields(d) -> list[dict[str, Any]]:
    """Every tensor attribute actually present on the Data, with shapes — so the
    UI shows what's really there (e.g. 3D coords baked into x), not a guess."""
    try:
        items = list(d.to_dict().items())
    except Exception:
        items = [(k, getattr(d, k, None)) for k in ("x", "edge_index", "edge_attr", "pos", "y")]
    out = []
    for k, v in items:
        if hasattr(v, "shape"):
            out.append({"name": str(k), "shape": list(v.shape), "dtype": str(v.dtype)})
    return out


def _graph_info(d) -> dict[str, Any]:
    n_nodes = int(getattr(d, "num_nodes", 0) or (d.x.shape[0] if getattr(d, "x", None) is not None else 0))
    ei = getattr(d, "edge_index", None)
    n_edges = int(ei.shape[1]) if ei is not None else 0
    x = getattr(d, "x", None)
    n_feat = int(x.shape[1]) if x is not None and x.dim() == 2 else 0
    ea = getattr(d, "edge_attr", None)
    edge_dim = int(ea.shape[1]) if ea is not None and ea.dim() == 2 else (1 if ea is not None else 0)
    x_preview = None
    if x is not None and x.dim() == 2 and x.numel():
        # Show enough columns that continuous features (coords/physico-chemical)
        # are visible, not just leading one-hot columns. Table scrolls.
        sub = x[:16, :64].float()
        x_preview = {
            "rows": int(x.shape[0]), "cols": int(x.shape[1]),
            "grid": [[round(float(v), 4) for v in row] for row in sub.tolist()],
        }
    return {
        "is_graph": True, "num_nodes": n_nodes, "num_edges": n_edges,
        "num_node_features": n_feat, "edge_dim": edge_dim,
        "fields": _graph_fields(d),
        "x_preview": x_preview,
    }


def _sample_graph_field(d, field: str) -> dict[str, Any]:
    """Pull one field (x/edge_index/edge_attr/pos/batch/y) from a PyG Data."""
    note = f"PyG graph: {_graph_info(d)['num_nodes']} nodes, {_graph_info(d)['num_edges']} edges"
    if field == "edge_index":
        return {"ok": True, "tensor": d.edge_index.long(), "natural_shape": list(d.edge_index.shape), "note": note}
    if field == "batch":
        b = getattr(d, "batch", None)
        if b is None:
            n = int(getattr(d, "num_nodes", d.x.shape[0]))
            b = torch.zeros(n, dtype=torch.long)
        return {"ok": True, "tensor": b.long(), "natural_shape": list(b.shape), "note": note}
    val = getattr(d, field if field != "x" else "x", None)
    if val is None:
        return {"ok": False, "error": f"graph has no field '{field}'"}
    t = val.float() if field in ("x", "edge_attr", "pos") else val
    return {"ok": True, "tensor": t, "natural_shape": list(t.shape), "note": note}


def _inspect_graph_folder(abspath: str) -> dict[str, Any]:
    """A directory of .pt graph files = a graph dataset (one PyG Data per file)."""
    p = Path(abspath)
    files = _list_pt_files(p)
    if not files:
        return {"kind": "graph_folder", "ok": False, "error": "no .pt files in folder"}
    try:
        first = Path(_checked(files[0]))
    except ScopeError as e:
        return _scope_error(e, "graph_folder")
    try:
        d = _as_pyg_data(safe_torch_load(first))
    except Exception as e:
        result = {"kind": "graph_folder", "ok": False, "error": f"{type(e).__name__}: {e}"}
        _mark_unsafe_pickle(result, e)
        return result
    if d is None:
        return {"kind": "graph_folder", "ok": False, "error": f"{files[0].name} is not a PyG graph"}
    info = _graph_info(d)
    ei = getattr(d, "edge_index", None)
    preview_edges = [[int(a), int(b)] for a, b in ei.t().tolist()[:64]] if ei is not None else []
    return {
        "kind": "graph_folder", "ok": True,
        "n_graphs": len(files),
        "example": files[0].name,
        "num_node_features": info["num_node_features"],
        "edge_dim": info["edge_dim"],
        "num_nodes": info["num_nodes"],
        "num_edges": info["num_edges"],
        "fields": info["fields"],
        "x_preview": info["x_preview"],
        "preview": {"n_nodes": info["num_nodes"], "edges": preview_edges},
    }


def _sample_graph_folder(abspath: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
    p = Path(abspath)
    files = _list_pt_files(p)
    if not files:
        return {"ok": False, "error": "no .pt files in folder"}
    idx = 0
    if options and isinstance(options.get("index"), int):
        idx = max(0, min(int(options["index"]), len(files) - 1))
    try:
        target = Path(_checked(files[idx]))
    except ScopeError as e:
        return _scope_error(e)
    try:
        d = _as_pyg_data(safe_torch_load(target))
    except Exception as e:
        result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        _mark_unsafe_pickle(result, e)
        return result
    if d is None:
        return {"ok": False, "error": f"{files[idx].name} is not a PyG graph"}
    field = (options or {}).get("field") if options else None
    return _sample_graph_field(d, field if isinstance(field, str) else "x")


def _inspect_tensor(abspath: str) -> dict[str, Any]:
    p = Path(abspath)
    ext = p.suffix.lower()
    try:
        if ext in (".pt", ".pth"):
            t = safe_torch_load(p)
            d = _as_pyg_data(t)
            if d is not None:
                return {"kind": "tensor", "ok": True, "size_bytes": p.stat().st_size, **_graph_info(d)}
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
        result = {"kind": "tensor", "ok": False, "error": f"{type(e).__name__}: {e}"}
        _mark_unsafe_pickle(result, e)
        return result
    info: dict[str, Any] = {"kind": "tensor", "ok": True, "size_bytes": p.stat().st_size}
    if isinstance(t, torch.Tensor):
        info["shape"] = list(t.shape)
        info["dtype"] = str(t.dtype)
        try:
            ft = t.float()
            info["min"] = float(ft.min())
            info["max"] = float(ft.max())
            info["mean"] = float(ft.mean())
        except Exception:  # tensor stats are informational; the tensor itself loaded fine
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
    except ImportError:  # optional dependency: Biopython absent → use the line-based fallback
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
        # First valid molecule as a node-link graph for the Overview preview.
        if smiles:
            g = _mol_to_graph(smiles[0])
            if g.get("ok"):
                edges = g["edge_index"].t().tolist()[:64]
                info["graph0"] = {"smiles": smiles[0], "n_nodes": int(g["n_atoms"]),
                                  "edges": [[int(a), int(b)] for a, b in edges]}
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


def _load_pyg(abspath: str):
    """Resolve a 'pyg:<Class[/Name]>' reference to a torch_geometric dataset.
    e.g. 'pyg:KarateClub', 'pyg:Planetoid/Cora', 'pyg:TUDataset/MUTAG'."""
    text = Path(abspath).read_text(encoding="utf-8").strip()
    ref = text[4:].strip() if text.lower().startswith("pyg:") else text
    if not ref:
        raise ValueError("empty pyg: reference (expected e.g. 'pyg:Planetoid/Cora')")
    import torch_geometric.datasets as D
    parts = ref.split("/")
    cls = getattr(D, parts[0], None)
    if cls is None:
        raise ValueError(f"unknown torch_geometric dataset class {parts[0]!r}")
    root = os.path.expanduser(f"~/.cache/spinoml-pyg/{parts[0]}")
    if len(parts) > 1:
        ds = cls(root=root, name=parts[1])
    else:
        try:
            ds = cls(root=root)
        except TypeError:
            ds = cls()  # e.g. KarateClub takes no args
    return ref, ds


def _inspect_pyg(abspath: str) -> dict[str, Any]:
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        return _missing_dep("pyg", "torch_geometric")
    try:
        ref, ds = _load_pyg(abspath)
        data = ds[0]
        n_nodes = int(getattr(data, "num_nodes", 0) or (data.x.shape[0] if data.x is not None else 0))
        n_edges = int(data.edge_index.shape[1]) if getattr(data, "edge_index", None) is not None else 0
        n_feat = int(data.x.shape[1]) if getattr(data, "x", None) is not None and data.x.dim() == 2 else 0
        n_classes = int(getattr(ds, "num_classes", 0) or 0)
        return {"kind": "pyg", "ok": True, "name": ref, "num_graphs": len(ds),
                "num_nodes": n_nodes, "num_edges": n_edges,
                "num_node_features": n_feat, "num_classes": n_classes}
    except Exception as e:
        return {"kind": "pyg", "ok": False, "error": f"{type(e).__name__}: {e}"}


def _sample_pyg(abspath: str, target: list[int] | None, options: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        return _missing_dep("pyg", "torch_geometric")
    field = (options or {}).get("field") if options else None
    try:
        ref, ds = _load_pyg(abspath)
        data = ds[0]
        note = f"{ref}: graph[0]"
        if field == "edge_index":
            return {"ok": True, "tensor": data.edge_index.long(),
                    "natural_shape": list(data.edge_index.shape), "note": note}
        if field == "batch":
            n = int(getattr(data, "num_nodes", data.x.shape[0]))
            return {"ok": True, "tensor": torch.zeros(n, dtype=torch.long), "natural_shape": [n], "note": note}
        if field == "y":
            return {"ok": True, "tensor": data.y, "natural_shape": list(data.y.shape), "note": note}
        # default / 'x' → node features
        return {"ok": True, "tensor": data.x.float(), "natural_shape": list(data.x.shape), "note": note}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ─── Manifest (paired multi-source) dataset ───────────────────────────────
# A small JSON descriptor that GLUES separate sources together row-by-row: each
# row of a table references a graph per "branch" (e.g. ligand + protein) living
# in different dirs/formats, plus a target column. This is what lets a dual-
# encoder know which ligand pairs with which protein — the pairing comes from
# the table row, not from independent file ordering.
#
#   {
#     "table": "reactions.csv",
#     "row": 0,                                  // optional preview row
#     "pairs": {
#       "ligand":  {"column": "smiles",  "kind": "molecule"},
#       "protein": {"column": "uniprot", "dir": "graphs/proteins",
#                   "match": "contains", "ext": ".pt"}
#     },
#     "target": {"column": "affinity", "type": "regression"}   // or classification
#   }
#
# Reference resolution per branch (all three the user asked for + filename match):
#   kind=="molecule"  → build a graph from the SMILES cell inline (RDKit, no dir)
#   kind=="sequence"  → tokenize the string cell → token-id LongTensor (char/byte
#                       vocab via `vocab`: "protein"/"smiles"/explicit/omit; `max_len`)
#   kind=="espf"      → ESPF substructure subword tokens from the SMILES/seq cell →
#                       LongTensor (BPE codebook via `codebook`: "drug"/"protein";
#                       `max_len`). Interpretable: token id ↔ named substructure.
#   dir + match=exact → <dir>/<cell><ext>
#   dir + match=contains → first file in <dir> whose name CONTAINS the cell value
#                          (e.g. UniProt "P12345" matches "AF-P12345-F1-model_v4.pt")
#   no dir            → the cell value IS a path (relative to the manifest, or absolute)


def _read_manifest(abspath: str) -> dict[str, Any]:
    import json
    with open(_checked(abspath), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict) or "table" not in cfg or "pairs" not in cfg:
        raise ValueError("manifest must be a JSON object with 'table' and 'pairs'")
    if not isinstance(cfg["pairs"], dict) or not cfg["pairs"]:
        raise ValueError("manifest 'pairs' must be a non-empty object of branches")
    return cfg


def _manifest_table_df(base: Path, cfg: dict[str, Any]):
    import pandas as pd
    tp = Path(_checked((base / str(cfg["table"])).expanduser()))
    if tp.suffix.lower() == ".parquet":
        return pd.read_parquet(tp)
    if tp.suffix.lower() == ".tsv":
        return pd.read_csv(tp, sep="\t")
    return pd.read_csv(tp)


def _resolve_branch_file(base: Path, spec: dict[str, Any], value: str) -> Path | None:
    value = str(value).strip()
    if not value:
        return None
    if "dir" in spec:
        d = Path(_checked((base / str(spec["dir"])).expanduser()))  # may be outside → ScopeError
        ext = str(spec.get("ext", "") or "")
        match = str(spec.get("match", "exact"))
        if match == "contains":
            try:
                cands = sorted(c for c in d.iterdir() if c.is_file() and value in c.name)
            except OSError:
                cands = []
            if ext:
                narrowed = [c for c in cands if c.suffix.lower() == ext.lower()]
                cands = narrowed or cands
            if not cands:
                return None
            return Path(_checked(cands[0]))
        # exact: <dir>/<value><ext>
        cand = d / (value if (not ext or value.endswith(ext)) else value + ext)
        if cand.exists():
            return Path(_checked(cand))
        alt = d / value  # tolerate a value that already carries its extension
        if alt.exists():
            return Path(_checked(alt))
        return Path(_checked(cand))  # missing but scoped → caller raises FileNotFoundError
    # no dir → the cell holds a path (relative to the manifest dir, or absolute)
    p = Path(value)
    return Path(_checked(p if p.is_absolute() else (base / value)))


def _mol_data(smi: str):
    """SMILES → PyG Data(x, edge_index) via RDKit (atoms=nodes, bonds=edges)."""
    from torch_geometric.data import Data
    g = _mol_to_graph(smi)
    if not g.get("ok"):
        raise ValueError(g.get("error", "RDKit failed to parse SMILES"))
    return Data(x=g["x"], edge_index=g["edge_index"])


def _cached_mol_data(smi: str, cache_dir: Path):
    """Build (or load) the molecule graph and persist it as a real PyG .pt, so
    RDKit work is done once and the molecule graphs exist as .pt like the rest."""
    cache_dir = Path(_checked(cache_dir, write=True))
    fp = cache_dir / f"mol_{hashlib.sha1(smi.encode('utf-8')).hexdigest()[:16]}.pt"
    if fp.exists():
        try:
            d = _as_pyg_data(safe_torch_load(Path(_checked(fp))))
            if d is not None:
                return d
        except ScopeError:
            raise
        except UnsafePickleError:
            raise
        except Exception:  # corrupt mol-cache entry → rebuild it from the SMILES
            pass
    d = _mol_data(smi)
    try:
        fp = Path(_checked(fp, write=True))
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(d, fp)
    except ScopeError:
        raise
    except Exception:
        pass  # caching is best-effort; sampling still works without it
    return d


# Built-in vocabularies for sequence branches (manifest kind="sequence").
# Token 0 = PAD, 1 = UNK, real characters start at 2 — a fixed, deterministic
# index map so the same string always tokenizes the same way (reproducibility).
_SEQ_VOCABS = {
    # 20 standard amino acids + the usual ambiguity/extra codes.
    "protein": "ACDEFGHIKLMNPQRSTVWYXBZUO",
    # A broad SMILES character set (case matters: lowercase = aromatic atoms).
    "smiles": "#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVZ[\\]abcdefgilmnoprstuy",
}


def _seq_vocab_map(spec: dict[str, Any]) -> dict[str, int] | None:
    """Resolve a branch's vocab → {char: token_id} (ids start at 2). A preset name
    ('protein'/'smiles') or an explicit character string both work; None means the
    byte-level fallback (no fixed vocab, id = clamped ord + 1)."""
    v = spec.get("vocab")
    chars = _SEQ_VOCABS.get(v, v) if isinstance(v, str) else None
    if not chars:
        return None
    return {c: i + 2 for i, c in enumerate(chars)}


def seq_vocab_size(spec: dict[str, Any]) -> int:
    """num_embeddings the model's Embedding needs for this branch's vocab."""
    vmap = _seq_vocab_map(spec)
    return 257 if vmap is None else len(vmap) + 2  # byte-level: 0=PAD, 1..256=bytes


def tokenize_sequence(value: Any, spec: dict[str, Any]) -> "torch.Tensor":
    """A string cell → 1-D LongTensor [L] of token ids (no graph). Char-level via a
    fixed vocab (UNK=1) or byte-level fallback; optional `max_len` truncates."""
    s = str(value).strip()
    max_len = spec.get("max_len")
    if isinstance(max_len, int) and max_len > 0:
        s = s[:max_len]
    vmap = _seq_vocab_map(spec)
    if vmap is None:  # byte-level: id = min(ord, 255) + 1, reserving 0 for PAD
        ids = [min(ord(c), 255) + 1 for c in s]
    else:
        ids = [vmap.get(c, 1) for c in s]  # 1 = UNK
    return torch.tensor(ids or [0], dtype=torch.long)


# ─── ESPF tokenization (interpretable substructures) ──────────────────────
# ESPF (Explainable Substructure Partition Fingerprint, Huang et al. 2019; the
# tokenizer used in MolTrans) is a chemically-aware *subword* tokenizer: a SMILES
# (or protein) string is split into frequent SUBSTRUCTURE tokens via a BPE-style
# codebook, and every token id maps back to a named substructure — that mapping
# is what makes it "interpretable". The codebooks live (vendored, BSD-3) under
# sidecar-torch/espf/ (see NOTICE). Token scheme matches the sequence tokenizer:
# 0 = PAD, 1 = UNK, real substructures start at id 2.

ESPF_DIR = Path(__file__).resolve().parent / "espf"
_ESPF_FILES = {  # codebook name → (bpe merge rules, subword→index map)
    "drug": ("drug_codes_chembl.txt", "subword_units_map_chembl.csv"),
    "protein": ("protein_codes_uniprot.txt", "subword_units_map_uniprot.csv"),
}
_ESPF_CACHE: dict[str, dict[str, Any]] = {}  # name → {ranks, subwords, sub2id}


def espf_codebook_name(spec: dict[str, Any]) -> str:
    name = str(spec.get("codebook", "drug")).strip().lower()
    return name if name in _ESPF_FILES else "drug"


def _load_espf_codebook(name: str = "drug") -> dict[str, Any]:
    """Load (and module-cache) an ESPF codebook from the vendored files: the
    ordered BPE merge rules and the subword→id map. Raises FileNotFoundError if
    the codebook isn't vendored (callers degrade to a clear error, never crash)."""
    name = name if name in _ESPF_FILES else "drug"
    if name in _ESPF_CACHE:
        return _ESPF_CACHE[name]
    codes_file, map_file = _ESPF_FILES[name]
    codes_path, map_path = ESPF_DIR / codes_file, ESPF_DIR / map_file
    if not codes_path.exists() or not map_path.exists():
        raise FileNotFoundError(
            f"ESPF codebook '{name}' missing under {ESPF_DIR} "
            f"({codes_file} / {map_file}) — see sidecar-torch/espf/NOTICE")
    ranks: dict[tuple[str, str], int] = {}
    with open(codes_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0 and line.startswith("#version"):
                continue
            parts = line.split()
            if len(parts) == 2:
                ranks[(parts[0], parts[1])] = len(ranks)
    import csv
    subwords: list[str] = []
    with open(map_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subwords.append(str(row.get("index", "")))
    sub2id = {s: i for i, s in enumerate(subwords)}  # vocab index (0-based)
    cb = {"ranks": ranks, "subwords": subwords, "sub2id": sub2id}
    _ESPF_CACHE[name] = cb
    return cb


def _espf_encode(orig: str, ranks: dict[tuple[str, str], int]) -> list[str]:
    """Pure-Python subword-nmt apply: greedily merge the highest-priority adjacent
    pair until none remain. Bit-for-bit parity with subword_nmt.apply_bpe (verified
    against the reference), so no runtime dependency on subword-nmt is needed."""
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


def espf_vocab_size(spec: dict[str, Any]) -> int:
    """num_embeddings the model's Embedding needs (subwords + PAD + UNK)."""
    try:
        cb = _load_espf_codebook(espf_codebook_name(spec))
    except FileNotFoundError:
        # No codebook → tokenize_espf silently degrades to the char-level
        # sequence tokenizer. Report THAT vocabulary size so the inspect note
        # is not an invented (and unusably small) 2.
        return seq_vocab_size({**spec, "vocab": "smiles"})
    return len(cb["subwords"]) + 2


def espf_substructures(spec: dict[str, Any]) -> list[str]:
    """id → substructure string (index 0/1 = PAD/UNK), for interpretable labels."""
    try:
        cb = _load_espf_codebook(espf_codebook_name(spec))
    except FileNotFoundError:  # no codebook → no labels; the token preview still shows
        return []
    return ["<pad>", "<unk>"] + list(cb["subwords"])


def tokenize_espf(value: Any, spec: dict[str, Any]) -> "torch.Tensor":
    """A SMILES/sequence cell → 1-D LongTensor [L] of ESPF substructure token ids
    (offset +2 over the codebook index; 0=PAD, 1=UNK). Honors `max_len` (subword
    count). Falls back to the char/byte sequence tokenizer if the codebook is
    missing (graceful — a manifest still samples/trains, just not as ESPF)."""
    try:
        cb = _load_espf_codebook(espf_codebook_name(spec))
    except FileNotFoundError:
        return tokenize_sequence(value, {**spec, "vocab": "smiles"})
    sub2id = cb["sub2id"]
    toks = _espf_encode(str(value).strip(), cb["ranks"])
    ids = [(sub2id[t] + 2) if t in sub2id else 1 for t in toks]  # 1 = UNK
    max_len = spec.get("max_len")
    if isinstance(max_len, int) and max_len > 0:
        ids = ids[:max_len]
    return torch.tensor(ids or [0], dtype=torch.long)


def ensure_espf_cache(base: Path, name: str = "drug") -> Path | None:
    """Materialize a compact, self-contained ESPF codebook cache next to the
    manifest (<base>/.espf/<name>.json.gz: {merges, subwords}). A training run is a
    standalone snapshot that can't see sidecar-torch/espf/, so train.py reads THIS
    cache instead — analogous to .graphcache. Best-effort; returns the path or None."""
    try:
        cb = _load_espf_codebook(name)
    except FileNotFoundError:  # no codebook → nothing to cache; explicit absent file
        return None
    import gzip
    import json
    out_dir = Path(_checked(Path(base) / ".espf", write=True))
    fp = Path(_checked(out_dir / f"{name}.json.gz", write=True))
    if fp.exists():
        return fp
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        merges = [f"{a} {b}" for (a, b) in sorted(cb["ranks"], key=lambda p: cb["ranks"][p])]
        blob = json.dumps({"merges": merges, "subwords": cb["subwords"]}, separators=(",", ":"))
        with gzip.open(fp, "wt", encoding="utf-8") as f:
            f.write(blob)
    except Exception:
        return None  # caching is best-effort; the sidecar still tokenizes in-memory
    return fp


# Cache for manifest `lookup` side-tables: (abs path, key, value col) → {key: value}.
_LOOKUP_CACHE: dict[tuple[str, str, str], dict[str, str]] = {}


def _lookup_value(base: Path, spec: dict[str, Any], value: Any) -> Any:
    """A branch may JOIN a side table by key to obtain its real cell value — e.g.
    a `prot_seq` branch keyed by uniprot pulls the sequence from sequences.csv. The
    side table is read once and cached. Spec: {lookup: <csv relpath/abs>, lookup_key,
    lookup_value}. Returns the joined value (or '' if the key isn't found)."""
    lk = spec.get("lookup")
    if not lk:
        return value
    import pandas as pd
    key_col = str(spec.get("lookup_key", "")) or "id"
    val_col = str(spec.get("lookup_value", "")) or "value"
    lp = Path(os.path.expanduser(str(lk)))
    if not lp.is_absolute():
        lp = base / str(lk)
    lp = Path(_checked(lp))
    ck = (str(lp), key_col, val_col)
    table = _LOOKUP_CACHE.get(ck)
    if table is None:
        if lp.suffix.lower() == ".parquet":
            df = pd.read_parquet(lp)
        elif lp.suffix.lower() == ".tsv":
            df = pd.read_csv(lp, sep="\t")
        else:
            df = pd.read_csv(lp)
        if key_col not in df.columns or val_col not in df.columns:
            raise ValueError(f"lookup {lp.name}: needs columns {key_col!r} and {val_col!r}, has {list(df.columns)}")
        table = {str(k): str(v) for k, v in zip(df[key_col], df[val_col])}
        _LOOKUP_CACHE[ck] = table
    return table.get(str(value).strip(), "")


def _load_branch_graph(base: Path, spec: dict[str, Any], value: Any, cache_dir: Path | None = None):
    """Return ('data', PyG Data) for one branch+row. Molecule branches build the
    graph from SMILES (RDKit) and, when cache_dir is given, save/reuse it as .pt.
    Sequence/ESPF branches tokenize the string cell → ('tensor', ids). A `lookup`
    spec first JOINs a side table by key to resolve the real cell value."""
    if spec.get("lookup"):
        value = _lookup_value(base, spec, value)
    if str(spec.get("kind", "")) == "espf":
        return ("tensor", tokenize_espf(value, spec))
    if str(spec.get("kind", "")) == "sequence":
        return ("tensor", tokenize_sequence(value, spec))
    if str(spec.get("kind", "")) == "molecule":
        d = _cached_mol_data(str(value), cache_dir) if cache_dir is not None else _mol_data(str(value))
        return ("data", d)
    fp = _resolve_branch_file(base, spec, value)
    if fp is None or not fp.exists():
        raise FileNotFoundError(
            f"no graph file for value {value!r} (dir={spec.get('dir')}, "
            f"match={spec.get('match', 'exact')}, ext={spec.get('ext', '')})")
    loaded = safe_torch_load(fp)
    d = _as_pyg_data(loaded)
    if d is not None:
        return ("data", d)
    # Not a PyG graph — a branch can be ANY type (e.g. protein_seq token tensors).
    t = _as_branch_tensor(loaded)
    if t is not None:
        return ("tensor", t)
    raise ValueError(f"{fp.name} is neither a PyG graph nor a loadable tensor")


def _as_branch_tensor(loaded):
    """Extract a tensor from a NON-graph branch .pt — a bare tensor, a common
    token-dict key, or the first tensor in a dict/list. Used for sequence/feature
    branches (e.g. protein_seq token ids) that aren't PyG graphs."""
    if isinstance(loaded, torch.Tensor):
        return loaded
    if isinstance(loaded, dict):
        for k in ("input_ids", "tokens", "ids", "x", "seq", "sequence"):
            v = loaded.get(k)
            if isinstance(v, torch.Tensor):
                return v
        for v in loaded.values():
            if isinstance(v, torch.Tensor):
                return v
    if isinstance(loaded, (list, tuple)):
        for v in loaded:
            if isinstance(v, torch.Tensor):
                return v
    return None


def _branch_field(obj_kind: str, obj, field: str) -> dict[str, Any]:
    if obj_kind == "data":
        return _sample_graph_field(obj, field)
    if obj_kind == "tensor":
        # A non-graph branch (token ids / feature tensor): one field, the tensor.
        return {"ok": True, "tensor": obj, "natural_shape": list(obj.shape)}
    # molecule-graph dict {x, edge_index, n_atoms, ...}
    if field == "edge_index":
        return {"ok": True, "tensor": obj["edge_index"], "natural_shape": list(obj["edge_index"].shape)}
    if field == "batch":
        n = int(obj["n_atoms"])
        return {"ok": True, "tensor": torch.zeros(n, dtype=torch.long), "natural_shape": [n]}
    return {"ok": True, "tensor": obj["x"], "natural_shape": list(obj["x"].shape)}


def _branch_slots(obj_kind: str, obj, branch: str) -> list[dict[str, Any]]:
    """Bindable slots for one branch, prefixed '<branch>.' — what the Inspector
    lists so the user picks which Input gets x / edge_index / batch."""
    slots: list[dict[str, Any]] = []
    if obj_kind == "data":
        for fld in _graph_fields(obj):
            dt = "int64" if any(s in fld["dtype"] for s in ("int", "long", "bool")) else "float32"
            slots.append({"field": f"{branch}.{fld['name']}", "shape": fld["shape"], "dtype": dt})
        if not any(s["field"] == f"{branch}.batch" for s in slots):
            n = int(getattr(obj, "num_nodes", obj.x.shape[0]))
            slots.append({"field": f"{branch}.batch", "shape": [n], "dtype": "int64"})
    elif obj_kind == "tensor":
        # Non-graph branch (e.g. protein_seq token ids): one slot, '<branch>.x'.
        dt = "int64" if any(s in str(obj.dtype) for s in ("int", "long", "bool")) else "float32"
        slots.append({"field": f"{branch}.x", "shape": list(obj.shape), "dtype": dt})
    else:
        slots.append({"field": f"{branch}.x", "shape": list(obj["x"].shape), "dtype": "float32"})
        slots.append({"field": f"{branch}.edge_index", "shape": list(obj["edge_index"].shape), "dtype": "int64"})
        slots.append({"field": f"{branch}.batch", "shape": [int(obj["n_atoms"])], "dtype": "int64"})
    return slots


def _manifest_row(df, idx: int) -> dict[str, Any]:
    idx = max(0, min(int(idx), len(df) - 1))
    return {str(k): df.iloc[idx][k] for k in df.columns}


def _manifest_target_tensor(cfg: dict[str, Any], row: dict[str, Any]) -> dict[str, Any] | None:
    t = cfg.get("target")
    if not t:
        return None
    col = str(t["column"])
    typ = str(t.get("type", "regression"))
    val = row.get(col)
    if typ == "classification":
        try:
            return {"ok": True, "tensor": torch.tensor([int(val)], dtype=torch.long), "natural_shape": [1]}
        except Exception:
            return {"ok": False, "error": f"target '{col}'={val!r} is not an int class index"}
    try:
        return {"ok": True, "tensor": torch.tensor([[float(val)]], dtype=torch.float32), "natural_shape": [1, 1]}
    except Exception:
        return {"ok": False, "error": f"target '{col}'={val!r} is not a float"}


def _inspect_manifest(abspath: str) -> dict[str, Any]:
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        return _missing_dep("manifest", "pandas")
    base = Path(abspath).resolve().parent
    try:
        cfg = _read_manifest(abspath)
        df = _manifest_table_df(base, cfg)
    except ScopeError as e:
        return _scope_error(e, "manifest")
    except Exception as e:
        return {"kind": "manifest", "ok": False, "error": f"{type(e).__name__}: {e}"}
    if len(df) == 0:
        return {"kind": "manifest", "ok": False, "error": "manifest table is empty"}
    row = _manifest_row(df, int(cfg.get("row", 0)))
    cache_dir = (base / ".graphcache") if bool(cfg.get("cache", True)) else None
    slots: list[dict[str, Any]] = []
    notes: list[str] = []
    for branch, spec in cfg["pairs"].items():
        col = str(spec.get("column", ""))
        if col not in df.columns:
            notes.append(f"branch '{branch}': column '{col}' not in table")
            continue
        try:
            kind, obj = _load_branch_graph(base, spec, row.get(col), cache_dir)
            slots.extend(_branch_slots(kind, obj, str(branch)))
            if str(spec.get("kind", "")) == "sequence":
                notes.append(f"branch '{branch}': sequence (vocab='{spec.get('vocab', 'bytes')}', "
                             f"num_embeddings={seq_vocab_size(spec)})")
            elif str(spec.get("kind", "")) == "espf":
                # Prime the .espf cache so a (snapshot) training run can read the
                # codebook without the vendored files — opening the dataset is enough.
                ensure_espf_cache(base, espf_codebook_name(spec))
                notes.append(f"branch '{branch}': ESPF substructures "
                             f"(codebook='{espf_codebook_name(spec)}', "
                             f"num_embeddings={espf_vocab_size(spec)})")
        except UnsafePickleError as e:
            return {"kind": "manifest", "ok": False, "error": str(e), "error_code": "UNSAFE_PICKLE"}
        except ScopeError as e:
            return _scope_error(e, "manifest")
        except ImportError:
            return _missing_dep("molecule", "rdkit")
        except Exception as e:
            notes.append(f"branch '{branch}' (row, {col}={row.get(col)!r}): {type(e).__name__}: {e}")
    tgt = cfg.get("target")
    if tgt:
        typ = str(tgt.get("type", "regression"))
        slots.append({"field": "target", "shape": [1, 1] if typ == "regression" else [1],
                      "dtype": "float32" if typ == "regression" else "int64"})
    return {
        "kind": "manifest", "ok": True,
        "n_rows": int(len(df)),
        "table": str(cfg["table"]),
        "columns": [str(c) for c in df.columns],
        "branches": list(cfg["pairs"].keys()),
        "target": ({"column": str(tgt["column"]), "type": str(tgt.get("type", "regression"))} if tgt else None),
        "slots": slots,
        "notes": notes,
    }


def _stats_manifest(abspath: str) -> dict[str, Any]:
    """Stats for a manifest = stats of its TABLE (target distribution, column
    histograms, correlations). Rendered as tabular stats in the UI."""
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        return _missing_dep("manifest", "pandas")
    base = Path(abspath).resolve().parent
    try:
        cfg = _read_manifest(abspath)
        table_abs = str((base / str(cfg["table"])).expanduser())
    except ScopeError as e:
        return _scope_error(e, "manifest")
    except Exception as e:
        return {"kind": "manifest", "ok": False, "error": f"{type(e).__name__}: {e}"}
    return _stats_tabular(table_abs)


def _sample_manifest(abspath: str, target_shape, options: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        return _missing_dep("manifest", "pandas")
    base = Path(abspath).resolve().parent
    try:
        cfg = _read_manifest(abspath)
        df = _manifest_table_df(base, cfg)
    except ScopeError as e:
        return _scope_error(e, "manifest")
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if len(df) == 0:
        return {"ok": False, "error": "manifest table is empty"}
    idx = int(options["index"]) if (options and isinstance(options.get("index"), int)) else int(cfg.get("row", 0))
    row = _manifest_row(df, idx)
    field = (options or {}).get("field") if options else None
    field = field if isinstance(field, str) else "target"
    tgt = cfg.get("target")
    if field == "target" or (tgt and field == str(tgt.get("column"))):
        res = _manifest_target_tensor(cfg, row)
        return res if res else {"ok": False, "error": "manifest has no target"}
    if "." not in field:
        return {"ok": False, "error": f"manifest field must be '<branch>.<x|edge_index|batch>' or 'target', got {field!r}"}
    branch, sub = field.split(".", 1)
    spec = cfg["pairs"].get(branch)
    if spec is None:
        return {"ok": False, "error": f"manifest has no branch '{branch}'"}
    col = str(spec.get("column", ""))
    cache_dir = (base / ".graphcache") if bool(cfg.get("cache", True)) else None
    try:
        kind, obj = _load_branch_graph(base, spec, row.get(col), cache_dir)
    except ScopeError as e:
        return _scope_error(e, "manifest")
    except ImportError:
        return _missing_dep("molecule", "rdkit")
    except Exception as e:
        result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        _mark_unsafe_pickle(result, e)
        return result
    res = _branch_field(kind, obj, sub)
    if res.get("ok"):
        res["note"] = f"manifest row {idx}: {branch}.{sub} ({col}={row.get(col)})"
    return res


def _missing_dep(kind: str, dep: str) -> dict[str, Any]:
    return {
        "kind": kind, "ok": False,
        "error": f"required dependency '{dep}' not installed in the sidecar's Python env",
        "missing_dep": dep,
    }


# ─── Stats (more expensive) ───────────────────────────────────────────────


def stats(abspath: str) -> dict[str, Any]:
    abspath = os.path.expanduser(abspath)
    try:
        abspath = _checked(abspath)
    except ScopeError as e:
        return _scope_error(e)
    if not os.path.exists(abspath):
        return _missing_path(abspath)
    kind = detect_kind(abspath)
    if kind == "tabular":
        return _stats_tabular(abspath)
    if kind == "image_folder":
        return _stats_image_folder(abspath)
    if kind == "graph_folder":
        return _inspect_graph_folder(abspath)
    if kind == "tensor":
        return _stats_tensor(abspath)
    if kind == "molecule":
        return _stats_molecule(abspath)
    if kind == "protein":
        return {"kind": "protein", "ok": True, "note": "stats limited to inspect for protein structures"}
    if kind == "huggingface":
        return {"kind": "huggingface", "ok": True, "note": "stats limited to inspect for HuggingFace refs"}
    if kind == "pyg":
        return _inspect_pyg(abspath)
    if kind == "manifest":
        return _stats_manifest(abspath)
    return {"kind": kind, "ok": False, "error": "unsupported"}


def _stats_tabular(abspath: str) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError:
        return _missing_dep("tabular", "pandas")
    p = _table_path(abspath)  # prepared-dataset dir → its inner table
    try:
        p = Path(_checked(p))
    except ScopeError as e:
        return _scope_error(e, "tabular")
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
    if numeric.shape[1] == 0:
        # Header-only table (0 rows) or all-object columns: describe() would
        # raise on an empty frame — surface the honest shape instead.
        _numeric_desc = None
    else:
        _numeric_desc = numeric.describe().fillna(0).round(4)
    summary: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        item: dict[str, Any] = {
            "col": str(col),
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "unique": int(s.nunique(dropna=True)),
        }
        if _numeric_desc is not None and col in numeric.columns:
            desc = _numeric_desc
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
            except Exception:  # histogram is informational; the column summary is still emitted
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
    try:
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
                        f = Path(_checked(f))
                    except ScopeError:
                        continue  # skip images that resolve outside the scope
                    try:
                        with Image.open(f) as im:
                            sizes.append(im.size)
                    except Exception:  # a failed image open is skipped; other sizes still counted
                        continue
    except OSError as e:
        return {"kind": "image_folder", "ok": False, "error": f"could not read image folder: {e}"}
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
            t = safe_torch_load(p)
        elif ext == ".npy":
            import numpy as np
            t = torch.from_numpy(np.load(p, allow_pickle=False))
        else:
            return info
    except Exception as e:
        result = {"kind": "tensor", "ok": False, "error": str(e)}
        _mark_unsafe_pickle(result, e)
        return result
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
    except Exception:  # histogram is informational; the tensor stats are still emitted
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
    try:
        abspath = _checked(abspath)
    except ScopeError as e:
        return _scope_error(e)
    if not os.path.exists(abspath):
        return _missing_path(abspath)
    kind = detect_kind(abspath)
    if kind == "tabular":
        field = (options or {}).get("field") if options else None
        if field in ("x", "edge_index", "batch"):
            smiles_col = (options or {}).get("target") if options else None
            return _sample_tabular_graph(abspath, field, smiles_col if isinstance(smiles_col, str) else None)
        feats = (options or {}).get("features") if options else None
        return _sample_tabular(abspath, target_shape, feats if isinstance(feats, list) else None)
    if kind == "image_folder":
        return _sample_image_folder(abspath, target_shape)
    if kind == "graph_folder":
        return _sample_graph_folder(abspath, options)
    if kind == "tensor":
        return _sample_tensor_file(abspath, target_shape, options)
    if kind == "molecule":
        return _sample_molecule(abspath, target_shape, options)
    if kind == "protein":
        return _sample_protein(abspath, target_shape)
    if kind == "pyg":
        return _sample_pyg(abspath, target_shape, options)
    if kind == "manifest":
        return _sample_manifest(abspath, target_shape, options)
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
    p = _table_path(abspath)  # prepared-dataset dir → its inner table
    try:
        p = Path(_checked(p))
    except ScopeError as e:
        return _scope_error(e, "tabular")
    try:
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        elif p.suffix.lower() == ".tsv":
            df = pd.read_csv(p, sep="\t")
        else:
            df = pd.read_csv(p)
    except Exception as e:
        return {"ok": False, "error": f"could not read table: {type(e).__name__}: {e}"}
    # An empty-but-valid table (header-only CSV, 0-row numeric parquet) must be
    # an EXPLICIT error, never a silent empty tensor. The old repeat-pad loop
    # below turns a 0-row frame into an infinite loop (rows.repeat stays 0-row)
    # that permanently wedges a sidecar worker thread.
    if len(df) == 0:
        return {"ok": False, "error": "tabular dataset has no rows"}
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
    if rows.shape[1] == 0:
        return {"ok": False, "error": "no usable feature columns in tabular dataset"}
    # fillna(0) above already turned NaN into zeros, but Inf survives to the
    # model — a scientifically silent corruption. Surface it explicitly.
    import numpy as np
    if rows.size and not np.isfinite(rows).all():
        return {"ok": False, "error": "tabular features contain NaN/Inf in the first batch — clean the data before training"}
    if rows.shape[0] == 0:
        return {"ok": False, "error": "tabular dataset has no rows"}
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
    try:
        for sub in sorted(p.iterdir()):
            if sub.is_dir():
                for f in sub.iterdir():
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                        img_path = f
                        break
            if img_path:
                break
    except OSError as e:
        return {"ok": False, "error": f"could not read image folder: {e}"}
    if img_path is None:
        return {"ok": False, "error": "no images found in image folder"}
    try:
        img_path = Path(_checked(img_path))
    except ScopeError as e:
        return _scope_error(e, "image_folder")
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


def _sample_tensor_file(abspath: str, target: list[int] | None, options: dict[str, Any] | None = None) -> dict[str, Any]:
    p = Path(abspath)
    ext = p.suffix.lower()
    # A corrupt/truncated/wrong-format file must be an explicit error, not an
    # uncaught exception that surfaces as a generic HTTP 500 (while inspect of
    # the SAME file already returns a clean ok:false). Never a silent empty.
    try:
        if ext in (".pt", ".pth"):
            t = safe_torch_load(p)
        elif ext == ".npy":
            import numpy as np
            t = torch.from_numpy(np.load(p, allow_pickle=False))
        else:
            return {"ok": False, "error": f"sampling not supported for tensor ext {ext}"}
    except Exception as e:
        result = {"ok": False, "error": f"could not load tensor file: {type(e).__name__}: {e}"}
        _mark_unsafe_pickle(result, e)
        return result
    # A saved PyG graph → expose the requested field (x/edge_index/edge_attr/…).
    d = _as_pyg_data(t)
    if d is not None:
        field = (options or {}).get("field") if options else None
        return _sample_graph_field(d, field if isinstance(field, str) else "x")
    if isinstance(t, dict):
        t = next(iter(t.values()))
    if not isinstance(t, torch.Tensor):
        return {"ok": False, "error": f"loaded object is {type(t).__name__}, not a tensor"}
    # An archived 0-element tensor is data that exists but carries nothing —
    # surface it as an explicit error, never an 'ok' empty tensor.
    if t.numel() == 0:
        return {"ok": False, "error": "tensor has 0 elements"}
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


def _read_first_smiles(abspath: str, n: int = 4) -> list[str]:
    smiles: list[str] = []
    with open(abspath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tok = line.strip().split()
            if tok and _looks_like_smiles(tok[0]):
                smiles.append(tok[0])
            if len(smiles) >= n:
                break
    return smiles


def _mol_to_graph(smi: str) -> dict[str, Any]:
    """SMILES → (atom features x [N, F], bond edge_index [2, 2E]) via RDKit.
    Atoms = nodes, bonds = (bidirectional) edges."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {"ok": False, "error": f"RDKit could not parse SMILES {smi!r}"}
    feats = []
    for a in mol.GetAtoms():
        feats.append([
            float(a.GetAtomicNum()),
            float(a.GetDegree()),
            float(a.GetFormalCharge()),
            float(int(a.GetIsAromatic())),
            float(a.GetTotalNumHs()),
        ])
    src, dst = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]
    x = torch.tensor(feats, dtype=torch.float32) if feats else torch.zeros((1, 5))
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
    return {"ok": True, "x": x, "edge_index": edge_index, "n_atoms": x.shape[0], "n_bonds": len(src) // 2}


def _sample_tabular_graph(abspath: str, field: str, smiles_col: str | None) -> dict[str, Any]:
    """Build a molecular graph from a SMILES column of a tabular dataset.
    smiles_col picks the column (the Input's 'target'); auto-detected if None."""
    try:
        import pandas as pd
    except ImportError:
        return _missing_dep("tabular", "pandas")
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        return _missing_dep("molecule", "rdkit")
    p = _table_path(abspath)  # prepared-dataset dir → its inner table
    try:
        p = Path(_checked(p))
    except ScopeError as e:
        return _scope_error(e, "tabular")
    try:
        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        elif p.suffix.lower() == ".tsv":
            df = pd.read_csv(p, sep="\t")
        else:
            df = pd.read_csv(p)
    except Exception as e:
        return {"ok": False, "error": f"could not read table: {e}"}
    if df.empty:
        return {"ok": False, "error": "empty table"}
    col = smiles_col if (smiles_col and smiles_col in df.columns) else None
    if col is None:
        for c in df.columns:
            v = df[c].dropna()
            if len(v) and isinstance(v.iloc[0], str) and _looks_like_smiles(str(v.iloc[0])):
                col = c
                break
    if col is None:
        return {"ok": False, "error": "no SMILES column found — set the Input's 'target' to the SMILES column"}
    smi = str(df[col].dropna().iloc[0])
    g = _mol_to_graph(smi)
    if not g.get("ok"):
        return {"ok": False, "error": g.get("error")}
    note = f"graph from column '{col}': {g['n_atoms']} atoms, {g['n_bonds']} bonds ({smi})"
    if field == "x":
        return {"ok": True, "tensor": g["x"], "natural_shape": list(g["x"].shape), "note": note}
    if field == "edge_index":
        return {"ok": True, "tensor": g["edge_index"], "natural_shape": list(g["edge_index"].shape), "note": note}
    return {"ok": True, "tensor": torch.zeros(g["n_atoms"], dtype=torch.long), "natural_shape": [g["n_atoms"]], "note": note}


def _sample_molecule(abspath: str, target: list[int] | None, options: dict[str, Any] | None = None) -> dict[str, Any]:
    field = (options or {}).get("field") if options else None

    # Graph fields (x / edge_index / batch): build a real atom-bond graph from the
    # first SMILES via RDKit, so GNN models get a meaningful molecular graph.
    if field in ("x", "edge_index", "batch"):
        try:
            from rdkit import Chem  # noqa: F401
        except ImportError:
            return _missing_dep("molecule", "rdkit")
        smiles = _read_first_smiles(abspath, 1)
        if not smiles:
            return {"ok": False, "error": "no parseable SMILES in file"}
        g = _mol_to_graph(smiles[0])
        if not g.get("ok"):
            return {"ok": False, "error": g.get("error")}
        note = f"molecule graph: {g['n_atoms']} atoms, {g['n_bonds']} bonds ({smiles[0]})"
        if field == "x":
            return {"ok": True, "tensor": g["x"], "natural_shape": list(g["x"].shape), "note": note}
        if field == "edge_index":
            return {"ok": True, "tensor": g["edge_index"], "natural_shape": list(g["edge_index"].shape), "note": note}
        return {"ok": True, "tensor": torch.zeros(g["n_atoms"], dtype=torch.long),
                "natural_shape": [g["n_atoms"]], "note": note}

    # Default (no field): fixed-length byte encoding of the first SMILES strings.
    smiles = _read_first_smiles(abspath, 4)
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
