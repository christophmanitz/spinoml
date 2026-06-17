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
    return "unknown"


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
    except OSError:
        return []
    return files[:cap]


def _looks_like_graph_folder(path: Path) -> bool:
    return len(_list_pt_files(path, cap=1)) > 0


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
    # Phase 12b: remote workspaces send tilde-prefixed paths ('~/spinoml/...');
    # Python's os.path doesn't expand those, so we do it once at the entry point.
    abspath = os.path.expanduser(abspath)
    kind = detect_kind(abspath)
    if kind == "tabular":
        return _inspect_tabular(abspath)
    if kind == "image_folder":
        return _inspect_image_folder(abspath)
    if kind == "graph_folder":
        return _inspect_graph_folder(abspath)
    if kind == "tensor":
        return _inspect_tensor(abspath)
    if kind == "protein":
        return _inspect_protein(abspath)
    if kind == "molecule":
        return _inspect_molecule(abspath)
    if kind == "huggingface":
        return _inspect_huggingface(abspath)
    if kind == "pyg":
        return _inspect_pyg(abspath)
    if kind == "manifest":
        return _inspect_manifest(abspath)
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
        d = _as_pyg_data(torch.load(files[0], map_location="cpu", weights_only=False))
    except Exception as e:
        return {"kind": "graph_folder", "ok": False, "error": f"{type(e).__name__}: {e}"}
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
        d = _as_pyg_data(torch.load(files[idx], map_location="cpu", weights_only=False))
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if d is None:
        return {"ok": False, "error": f"{files[idx].name} is not a PyG graph"}
    field = (options or {}).get("field") if options else None
    return _sample_graph_field(d, field if isinstance(field, str) else "x")


def _inspect_tensor(abspath: str) -> dict[str, Any]:
    p = Path(abspath)
    ext = p.suffix.lower()
    try:
        if ext in (".pt", ".pth"):
            t = torch.load(p, map_location="cpu", weights_only=False)
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
#   dir + match=exact → <dir>/<cell><ext>
#   dir + match=contains → first file in <dir> whose name CONTAINS the cell value
#                          (e.g. UniProt "P12345" matches "AF-P12345-F1-model_v4.pt")
#   no dir            → the cell value IS a path (relative to the manifest, or absolute)


def _read_manifest(abspath: str) -> dict[str, Any]:
    import json
    with open(abspath, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict) or "table" not in cfg or "pairs" not in cfg:
        raise ValueError("manifest must be a JSON object with 'table' and 'pairs'")
    if not isinstance(cfg["pairs"], dict) or not cfg["pairs"]:
        raise ValueError("manifest 'pairs' must be a non-empty object of branches")
    return cfg


def _manifest_table_df(base: Path, cfg: dict[str, Any]):
    import pandas as pd
    tp = (base / str(cfg["table"])).expanduser()
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
        d = (base / str(spec["dir"])).expanduser()
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
            return cands[0] if cands else None
        # exact: <dir>/<value><ext>
        cand = d / (value if (not ext or value.endswith(ext)) else value + ext)
        if cand.exists():
            return cand
        alt = d / value  # tolerate a value that already carries its extension
        return alt if alt.exists() else cand
    # no dir → the cell holds a path (relative to the manifest dir, or absolute)
    p = Path(value)
    return p if p.is_absolute() else (base / value)


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
    cache_dir = Path(cache_dir)
    fp = cache_dir / f"mol_{hashlib.sha1(smi.encode('utf-8')).hexdigest()[:16]}.pt"
    if fp.exists():
        try:
            d = _as_pyg_data(torch.load(fp, map_location="cpu", weights_only=False))
            if d is not None:
                return d
        except Exception:
            pass
    d = _mol_data(smi)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(d, fp)
    except Exception:
        pass  # caching is best-effort; sampling still works without it
    return d


def _load_branch_graph(base: Path, spec: dict[str, Any], value: Any, cache_dir: Path | None = None):
    """Return ('data', PyG Data) for one branch+row. Molecule branches build the
    graph from SMILES (RDKit) and, when cache_dir is given, save/reuse it as .pt."""
    if str(spec.get("kind", "")) == "molecule":
        d = _cached_mol_data(str(value), cache_dir) if cache_dir is not None else _mol_data(str(value))
        return ("data", d)
    fp = _resolve_branch_file(base, spec, value)
    if fp is None or not fp.exists():
        raise FileNotFoundError(
            f"no graph file for value {value!r} (dir={spec.get('dir')}, "
            f"match={spec.get('match', 'exact')}, ext={spec.get('ext', '')})")
    d = _as_pyg_data(torch.load(fp, map_location="cpu", weights_only=False))
    if d is None:
        raise ValueError(f"{fp.name} is not a PyG graph")
    return ("data", d)


def _branch_field(obj_kind: str, obj, field: str) -> dict[str, Any]:
    if obj_kind == "data":
        return _sample_graph_field(obj, field)
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
    except ImportError:
        return _missing_dep("molecule", "rdkit")
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
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


def _sample_tensor_file(abspath: str, target: list[int] | None, options: dict[str, Any] | None = None) -> dict[str, Any]:
    p = Path(abspath)
    ext = p.suffix.lower()
    if ext in (".pt", ".pth"):
        t = torch.load(p, map_location="cpu", weights_only=False)
    elif ext == ".npy":
        import numpy as np
        t = torch.from_numpy(np.load(p, allow_pickle=False))
    else:
        return {"ok": False, "error": f"sampling not supported for tensor ext {ext}"}
    # A saved PyG graph → expose the requested field (x/edge_index/edge_attr/…).
    d = _as_pyg_data(t)
    if d is not None:
        field = (options or {}).get("field") if options else None
        return _sample_graph_field(d, field if isinstance(field, str) else "x")
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
    p = Path(abspath)
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
