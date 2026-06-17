"""SpinoML shape-inference + dataset sidecar.

Runs a tiny HTTP server on 127.0.0.1:7421.

Endpoints:
  GET  /health
  POST /infer            { code, input_shape }            → per-layer output shapes
  POST /dataset/inspect  { abspath }                       → kind + cheap metadata
  POST /dataset/stats    { abspath }                       → stats/histograms (heavier)
  POST /dataset/smoke    { code, abspath, input_shape? }   → run sample through generated model
  POST /activations      { code, input_shapes, abspaths? } → per-layer activations + weights (downsampled)
  POST /deps/check       { specs: str[] }                   → pip dry-run resolve (compat smoke test)
  POST /deps/install     { specs: str[] }                   → pip install into the sidecar env

Safety: this exec's code from the local frontend only. CORS is permissive
because the dev server (Vite, port 5173) and the Tauri webview both need
to call it; the bind address is 127.0.0.1 so no external host can reach it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn as nn
import torch.nn.functional as F

import dataset_handlers as ds_mod

# Default 7421 keeps local-mode behaviour unchanged. Override via env so a
# remote deploy (Phase 12b) can pick a free port on the HPC login node
# without colliding with another user's sidecar.
PORT = int(os.environ.get("SPINOML_TORCH_PORT", "7421"))


# ── Graph (PyG Data) inputs ───────────────────────────────────────────────
# A `Graph` model input is a whole PyG Data object, not a bare tensor. These
# helpers assemble one so the model — whose forward does `x, edge_index, batch =
# data.x, …` — can run, both for synthetic shape inference and real dataset
# samples. Mirrors the standalone __main__ harness the codegen emits.

def _synth_graph(shape: list[int] | None, n_edges: int = 0, edge_dim: int = 0):
    """A deterministic stand-in graph for shape inference / preview: x ramp in
    [-2,2], a random edge_index over the nodes, single-graph batch."""
    from torch_geometric.data import Data
    n = int(shape[0]) if shape else 1
    fdim = int(shape[1]) if shape and len(shape) > 1 else 1
    e = int(n_edges) if n_edges else max(1, n * 2)
    x = torch.linspace(-2.0, 2.0, max(1, n * fdim)).reshape(n, fdim) if n * fdim > 0 else torch.zeros((n, fdim))
    edge_index = torch.randint(0, max(1, n), (2, e), dtype=torch.long)
    d = Data(x=x, edge_index=edge_index, batch=torch.zeros((n,), dtype=torch.long))
    if int(edge_dim) > 0:
        d.edge_attr = torch.zeros((e, int(edge_dim)))
    return d


def _sample_graph_data(path: str, options: dict | None):
    """Assemble a PyG Data from ONE dataset source by pulling each field. For a
    manifest, options.branch selects the branch ('ligand' → 'ligand.x', …); for
    a single-graph dataset (graph_folder/pyg/molecule) the bare field is used.
    Returns (Data | None, sub) where sub is the failing field result on error."""
    from torch_geometric.data import Data
    branch = (options or {}).get("branch") or ""

    def field(fname: str):
        opt = dict(options or {})
        opt["field"] = f"{branch}.{fname}" if branch else fname
        return ds_mod.sample_tensor(path, None, opt)

    xr = field("x")
    if not xr.get("ok"):
        return None, xr
    x = xr["tensor"].float()
    n = int(x.shape[0])
    eir = field("edge_index")
    edge_index = eir["tensor"].long() if eir.get("ok") else torch.zeros((2, 0), dtype=torch.long)
    br = field("batch")
    batch = br["tensor"].long() if br.get("ok") else torch.zeros((n,), dtype=torch.long)
    d = Data(x=x, edge_index=edge_index, batch=batch)
    ear = field("edge_attr")
    if ear.get("ok"):
        d.edge_attr = ear["tensor"].float()
    return d, xr


def infer(
    code: str,
    input_shapes: list[list[int]],
    input_dtypes: list[str] | None = None,
) -> dict:
    shapes: dict[str, list[int]] = {}
    ns: dict = {"__name__": "<spinoml-model>"}
    try:
        exec(compile(code, "<spinoml-model>", "exec"), ns)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        # Graceful hint for the optional GNN dependency.
        if isinstance(e, ModuleNotFoundError) and "torch_geometric" in str(e):
            msg += " — GNN layers need PyTorch Geometric. Install it with: pip install torch_geometric"
        return {
            "ok": False,
            "stage": "compile",
            "error": msg,
            "trace": traceback.format_exc(limit=4),
            "shapes": shapes,
        }

    Model = ns.get("Model")
    if Model is None:
        return {"ok": False, "stage": "compile", "error": "no Model class in generated code", "shapes": shapes}

    try:
        model = Model()
        model.eval()
    except Exception as e:
        return {
            "ok": False,
            "stage": "construct",
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=6),
            "shapes": shapes,
        }

    for name, mod in model.named_modules():
        if name == "":
            continue

        def make_hook(attr_name: str):
            def hook(_m, _inp, out):
                if isinstance(out, torch.Tensor):
                    shapes[attr_name] = list(out.shape)
                elif isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
                    shapes[attr_name] = list(out[0].shape)
            return hook

        mod.register_forward_hook(make_hook(name))

    try:
        n_params = int(sum(p.numel() for p in model.parameters()))
    except Exception:
        n_params = 0

    try:
        dtypes = input_dtypes or []
        xs = []
        for i, s in enumerate(input_shapes):
            dt = dtypes[i] if i < len(dtypes) else "float32"
            if dt == "graph":
                xs.append(_synth_graph(s))
            elif dt in ("int64", "long"):
                xs.append(torch.zeros(s, dtype=torch.long))
            else:
                xs.append(torch.zeros(s))
    except Exception as e:
        return {
            "ok": False,
            "stage": "input",
            "error": f"could not build zero tensor of shape {input_shapes}: {e}",
            "shapes": shapes,
            "n_params": n_params,
        }

    try:
        with torch.no_grad():
            out = model(*xs)
    except Exception as e:
        return {
            "ok": False,
            "stage": "forward",
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=6),
            "shapes": shapes,
            "n_params": n_params,
        }

    if isinstance(out, torch.Tensor):
        shapes["__output__"] = list(out.shape)
    elif isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
        shapes["__output__"] = [list(o.shape) if isinstance(o, torch.Tensor) else None for o in out]
    elif isinstance(out, dict):
        shapes["__output__"] = {k: list(v.shape) if isinstance(v, torch.Tensor) else None for k, v in out.items()}

    # Recount AFTER forward so lazy (in_channels=-1) params are materialized.
    try:
        n_params = int(sum(p.numel() for p in model.parameters()))
    except Exception:
        pass

    return {"ok": True, "shapes": shapes, "n_params": n_params}


def smoke_test(
    code: str,
    abspaths: list[str],
    input_shapes: list[list[int]] | None,
    input_options: list[dict] | None = None,
) -> dict:
    """Build sample tensors from one dataset per input, then run them through the model.

    abspaths is a list — one dataset path per model input.
    input_options is an optional per-input dict bag (e.g. {features: [...]} for
    tabular column selection); aligned to abspaths/input_shapes by index.
    """
    t0 = time.perf_counter()
    if not abspaths:
        return {"ok": False, "stage": "sample", "error": "no dataset paths provided"}

    xs: list[torch.Tensor] = []
    notes: list[str] = []

    def opts_for(i: int) -> dict | None:
        if input_options and i < len(input_options):
            v = input_options[i]
            if isinstance(v, dict):
                return v
        return None

    def sample_one(path: str, sh, opt):
        """One model input: a whole PyG Data for a graph input (opt.graph), else
        a bare tensor. Returns (value, note) or raises via the returned error."""
        if opt and opt.get("graph"):
            d, sub = _sample_graph_data(path, opt)
            return d, (sub.get("note") if sub else None), (None if d is not None else sub)
        sub = ds_mod.sample_tensor(path, sh, opt)
        if not sub.get("ok"):
            return None, None, sub
        return sub["tensor"], sub.get("note"), None

    if len(abspaths) == 1 and input_shapes and len(input_shapes) > 1:
        path = abspaths[0]
        for i, sh in enumerate(input_shapes):
            val, note, err = sample_one(path, sh, opts_for(i))
            if err is not None or val is None:
                return {"ok": False, "stage": "sample", "error": (err or {}).get("error"), "details": err, "dataset": path}
            xs.append(val)
            if note: notes.append(f"{path.split('/')[-1]}: {note}")
    else:
        n = max(len(abspaths), len(input_shapes) if input_shapes else 0)
        for i in range(n):
            path = abspaths[i] if i < len(abspaths) else abspaths[-1]
            sh = input_shapes[i] if input_shapes and i < len(input_shapes) else None
            val, note, err = sample_one(path, sh, opts_for(i))
            if err is not None or val is None:
                return {"ok": False, "stage": "sample", "error": (err or {}).get("error"), "details": err, "dataset": path}
            xs.append(val)
            if note: notes.append(f"{path.split('/')[-1]}: {note}")
    t_sample = time.perf_counter() - t0

    ns: dict = {"__name__": "<spinoml-model>"}
    try:
        exec(compile(code, "<spinoml-model>", "exec"), ns)
    except Exception as e:
        return {
            "ok": False, "stage": "compile",
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=4),
        }
    Model = ns.get("Model")
    if Model is None:
        return {"ok": False, "stage": "compile", "error": "no Model class in generated code"}

    try:
        model = Model()
        model.eval()
    except Exception as e:
        return {
            "ok": False, "stage": "construct",
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=4),
        }

    try:
        n_params = int(sum(p.numel() for p in model.parameters()))
    except Exception:
        n_params = 0

    # A graph input reports its node-feature shape [N, F] (the Data itself has
    # no .shape). Keeps the report meaningful without leaking the Data object.
    def _shape_of(v):
        if isinstance(v, torch.Tensor):
            return list(v.shape)
        x = getattr(v, "x", None)
        return list(x.shape) if x is not None else []

    t1 = time.perf_counter()
    try:
        with torch.no_grad():
            out = model(*xs)
    except Exception as e:
        return {
            "ok": False, "stage": "forward",
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=6),
            "input_shape": [_shape_of(t) for t in xs],
            "n_params": n_params,
        }
    t_forward = time.perf_counter() - t1
    # Recount AFTER forward so lazy (in_channels=-1) params are materialized.
    try:
        n_params = int(sum(p.numel() for p in model.parameters()))
    except Exception:
        pass

    if isinstance(out, torch.Tensor):
        out_shape: list[int] | list[list[int]] | None = list(out.shape)
    elif isinstance(out, tuple) and out and all(isinstance(o, torch.Tensor) for o in out):
        out_shape = [list(o.shape) for o in out]
    elif isinstance(out, dict):
        out_shape = [list(v.shape) for v in out.values() if isinstance(v, torch.Tensor)]
    else:
        out_shape = None

    input_shape_report: list[int] | list[list[int]] = (
        _shape_of(xs[0]) if len(xs) == 1 else [_shape_of(t) for t in xs]
    )

    return {
        "ok": True,
        "input_shape": input_shape_report,
        "output_shape": out_shape,
        "n_params": n_params,
        "sample_note": " · ".join(notes) if notes else None,
        "timings_ms": {
            "sample": round(t_sample * 1000, 2),
            "forward": round(t_forward * 1000, 2),
        },
    }


# ── Activation capture ──────────────────────────────────────────────────────
# Runs a REAL forward pass and captures per-module activations, but downsamples
# everything aggressively so the JSON payload stays small (~tens of KB). Feeds
# the 3Blue1Brown-style "what flows through the model" visualisations.

def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def _act_stats(t: torch.Tensor) -> dict:
    tf = t.detach().float().reshape(-1)
    if tf.numel() == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0, "frac_zero": 0.0}
    return {
        "min": _r(tf.min()), "max": _r(tf.max()),
        "mean": _r(tf.mean()), "std": _r(tf.std(unbiased=False)),
        "frac_zero": _r((tf == 0).float().mean()),
    }


def _vec_preview(t: torch.Tensor, cap: int = 256) -> list[float]:
    tf = t.detach().float().reshape(-1)
    n = tf.numel()
    if n > cap:
        idx = torch.linspace(0, n - 1, cap).round().long()
        tf = tf[idx]
    return [_r(v) for v in tf.tolist()]


def _grid_preview(t: torch.Tensor, cap: int = 32) -> list[list[float]]:
    """Any >=1D tensor → a small 2D float grid (extra leading dims mean-reduced)."""
    t = t.detach().float()
    while t.dim() > 2:
        t = t.mean(dim=0)
    if t.dim() <= 1:
        t = t.reshape(1, -1)
    h, w = t.shape
    oh, ow = max(1, min(h, cap)), max(1, min(w, cap))
    g = F.adaptive_avg_pool2d(t.reshape(1, 1, h, w), (oh, ow))[0, 0]
    return [[_r(v) for v in row] for row in g.tolist()]


def _preview_tensor(t) -> dict | None:
    """t has the batch dim already removed (a single sample)."""
    if not isinstance(t, torch.Tensor):
        return None
    nd = t.dim()
    if t.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8, torch.bool):
        return {"kind": "tokens", "values": [int(v) for v in t.detach().reshape(-1)[:64].tolist()]}
    if nd == 0:
        return {"kind": "scalar", "value": _r(t)}
    if nd == 1:
        return {"kind": "vector", "values": _vec_preview(t)}
    if nd == 2:
        return {"kind": "matrix", "grid": _grid_preview(t)}
    # nd >= 3 → channels-first feature maps.
    c = int(t.shape[0])
    shown = min(c, 16)
    maps = [_grid_preview(t[i], cap=8) for i in range(shown)]
    return {"kind": "maps", "channels": c, "shown": shown, "maps": maps}


def _preview_input(t: torch.Tensor) -> dict | None:
    """Preview for a model INPUT (whole tensor, batch dim kept). Detects a GNN
    edge_index (int [2, E]) and returns it as an edge list so the frontend can
    draw the actual graph instead of a meaningless matrix."""
    if t.dtype in (torch.int64, torch.int32, torch.long) and t.dim() == 2 and int(t.shape[0]) == 2:
        E = int(t.shape[1])
        shown = min(E, 64)
        edges = [[int(t[0, j]), int(t[1, j])] for j in range(shown)]
        n_nodes = (int(t.max().item()) + 1) if E > 0 else 0
        return {"kind": "edges", "n_edges": E, "n_nodes": n_nodes, "edges": edges}
    return _preview_tensor(t[0] if t.dim() >= 1 else t)


def _weights_for(mod) -> dict | None:
    if isinstance(mod, nn.Linear):
        return {"kind": "matrix", "shape": list(mod.weight.shape),
                "grid": _grid_preview(mod.weight, cap=48)}
    if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        w = mod.weight  # [out, in, *k]
        outc = int(w.shape[0])
        shown = min(outc, 16)
        kernels = [_grid_preview(w[i].mean(dim=0) if w[i].dim() > 2 else w[i], cap=7) for i in range(shown)]
        return {"kind": "kernels", "out_channels": outc, "in_channels": int(w.shape[1]),
                "shown": shown, "kernels": kernels}
    return None


def activations(
    code: str,
    input_shapes: list[list[int]],
    input_dtypes: list[str] | None = None,
    abspaths: list[str] | None = None,
    input_options: list[dict] | None = None,
    checkpoint: str | None = None,
    graph_attrs: list[str] | None = None,
) -> dict:
    ns: dict = {"__name__": "<spinoml-model>"}
    try:
        exec(compile(code, "<spinoml-model>", "exec"), ns)
    except Exception as e:
        return {"ok": False, "stage": "compile", "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=4)}
    Model = ns.get("Model")
    if Model is None:
        return {"ok": False, "stage": "compile", "error": "no Model class in generated code"}
    try:
        model = Model()
    except Exception as e:
        return {"ok": False, "stage": "construct", "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=6)}

    # Trained weights from a checkpoint are loaded LATER (after inputs are built
    # and lazy params materialized) — see the load block below.
    weights_source = "random"
    weights_note: str | None = None

    # Build inputs: real dataset sample if given, else a deterministic ramp in
    # [-2, 2] (spans negative+positive so ReLU/sigmoid/tanh visibly respond).
    notes: list[str] = []
    try:
        xs: list = []
        if abspaths:
            for i, sh in enumerate(input_shapes):
                path = abspaths[i] if i < len(abspaths) else abspaths[-1]
                opt = (input_options[i] if input_options and i < len(input_options)
                       and isinstance(input_options[i], dict) else None)
                if opt and opt.get("graph"):
                    d, sub = _sample_graph_data(path, opt)
                    if d is None:
                        return {"ok": False, "stage": "sample", "error": (sub or {}).get("error"), "details": sub}
                    xs.append(d)
                    if sub and sub.get("note"):
                        notes.append(sub["note"])
                    continue
                sub = ds_mod.sample_tensor(path, sh, opt)
                if not sub.get("ok"):
                    return {"ok": False, "stage": "sample", "error": sub.get("error"), "details": sub}
                xs.append(sub["tensor"])
                if sub.get("note"):
                    notes.append(sub["note"])
        else:
            dtypes = input_dtypes or []
            # Node-count hint for a synthetic edge_index: the first 2D float
            # input's leading dim (GNN node features are [N, F]).
            n_hint = 0
            for j, s in enumerate(input_shapes):
                dtj = dtypes[j] if j < len(dtypes) else "float32"
                if dtj not in ("int64", "long") and len(s) == 2:
                    n_hint = int(s[0])
                    break
            if n_hint <= 0:
                n_hint = 10
            for i, s in enumerate(input_shapes):
                dt = dtypes[i] if i < len(dtypes) else "float32"
                if dt == "graph":
                    xs.append(_synth_graph(s))
                elif dt in ("int64", "long"):
                    if len(s) == 2 and int(s[0]) == 2:
                        # edge_index → a random graph so the GNN viz is meaningful
                        # (synthetic, illustrative; mechanics not learned structure).
                        E = int(s[1])
                        xs.append(torch.randint(0, max(2, n_hint), (2, E), dtype=torch.long))
                    else:
                        xs.append(torch.zeros(s, dtype=torch.long))
                else:
                    n = 1
                    for d in s:
                        n *= int(d)
                    xs.append(torch.linspace(-2.0, 2.0, n).reshape(s) if n > 0 else torch.zeros(s))
    except Exception as e:
        return {"ok": False, "stage": "input", "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=4)}

    # Materialize lazy (in_channels=-1) params from the ACTUAL inputs, THEN load
    # the checkpoint — only shape-COMPATIBLE tensors, so a checkpoint whose dims
    # don't match the current inputs just leaves those layers random (with an
    # honest note) instead of crashing the forward. Dimension mismatches never break it.
    try:
        with torch.no_grad():
            model(*xs)
    except Exception:
        pass  # best-effort lazy init; a real shape error surfaces at the hooked forward
    if checkpoint:
        try:
            ckpt = torch.load(os.path.expanduser(checkpoint), map_location="cpu", weights_only=False)
            state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
            model_sd = model.state_dict()
            compat = {k: v for k, v in state.items()
                      if k in model_sd and hasattr(v, "shape") and tuple(v.shape) == tuple(model_sd[k].shape)}
            model.load_state_dict(compat, strict=False)
            total = len(model_sd)
            matched = len(compat)
            if matched == 0:
                weights_note = "Checkpoint passt nicht zum aktuellen Graphen — zeige zufällige Gewichte."
            else:
                weights_source = "trained"
                ep = ckpt.get("epoch") if isinstance(ckpt, dict) else None
                weights_note = "trainierte Gewichte" + (f" (Epoche {ep + 1})" if isinstance(ep, int) else "")
                if matched < total:
                    weights_note += f" · {matched}/{total} Schichten geladen (Rest zufällig)"
        except Exception as e:
            weights_note = f"Checkpoint nicht ladbar: {type(e).__name__}: {e}"
    model.eval()

    acts: dict[str, dict] = {}
    gset = set(graph_attrs or [])
    handles = []
    for name, mod in model.named_modules():
        if name == "":
            continue

        def make_hook(attr_name: str):
            def hook(_m, _inp, out):
                t = out
                if isinstance(t, tuple) and t and isinstance(t[0], torch.Tensor):
                    t = t[0]
                if not isinstance(t, torch.Tensor):
                    return
                # Graph layers: dim 0 is NODES (not batch) — keep the full tensor
                # so the viz sees node×feature, not just node 0.
                sample0 = t if attr_name in gset else (t[0] if t.dim() >= 1 else t)
                acts[attr_name] = {"stats": _act_stats(t), "preview": _preview_tensor(sample0)}
            return hook

        handles.append(mod.register_forward_hook(make_hook(name)))

    try:
        n_params = int(sum(p.numel() for p in model.parameters()))
    except Exception:
        n_params = 0

    try:
        with torch.no_grad():
            out = model(*xs)
    except Exception as e:
        for h in handles:
            h.remove()
        return {"ok": False, "stage": "forward", "error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=6), "n_params": n_params}
    for h in handles:
        h.remove()

    # Post-processing (input/output previews + weight snapshots) is best-effort:
    # the forward already succeeded, so a glitch building a preview must NEVER
    # turn into a crash — degrade to whatever we captured.
    weights: dict[str, dict] = {}
    try:
        for i, xi in enumerate(xs):
            if isinstance(xi, torch.Tensor):
                acts[f"__input_{i}__"] = {"stats": _act_stats(xi), "preview": _preview_input(xi)}
            else:
                # PyG Data/Batch graph input: stats from node features, preview = edges.
                x = getattr(xi, "x", None)
                ei = getattr(xi, "edge_index", None)
                acts[f"__input_{i}__"] = {
                    "stats": _act_stats(x) if isinstance(x, torch.Tensor) else None,
                    "preview": _preview_input(ei) if isinstance(ei, torch.Tensor) else None,
                }
        if "__input_0__" in acts:
            acts["__input__"] = acts["__input_0__"]  # back-compat alias
        out_t = out[0] if isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor) else out
        if isinstance(out_t, torch.Tensor):
            acts["__output__"] = {"stats": _act_stats(out_t),
                                  "preview": _preview_tensor(out_t[0] if out_t.dim() >= 1 else out_t)}
        for name, mod in model.named_modules():
            if name == "":
                continue
            w = _weights_for(mod)
            if w is not None:
                weights[name] = w
    except Exception:
        pass  # keep the activations we already captured; previews are non-essential

    return {"ok": True, "activations": acts, "weights": weights, "n_params": n_params,
            "sample_note": " · ".join(notes) if notes else None,
            "weights_source": weights_source, "weights_note": weights_note}


def _dist_name(spec: str) -> str:
    """Pull the bare distribution name out of a requirement spec.
    'torch_geometric==2.8.0' -> 'torch_geometric'; 'rdkit[extra]>=1' -> 'rdkit'."""
    return re.split(r"[<>=!~;\[\( ]", spec.strip(), 1)[0].strip()


def deps_check(specs: list[str]) -> dict:
    """Compatibility smoke test: resolve `specs` against THIS interpreter's env
    via `pip install --dry-run` WITHOUT installing anything. Returns whether the
    requested versions resolve, what pip would add/upgrade, and currently
    installed versions. Needs network access (pip queries the index)."""
    import importlib.metadata as im

    specs = [s.strip() for s in specs if isinstance(s, str) and s.strip()]
    requested = []
    for s in specs:
        base = _dist_name(s)
        try:
            ver = im.version(base)
        except Exception:
            ver = None
        requested.append({"spec": s, "name": base, "installed": ver})

    if not specs:
        return {"ok": True, "compatible": True, "python": sys.version.split()[0],
                "requested": [], "would_install": [], "log": "no dependencies specified"}

    cmd = [sys.executable, "-m", "pip", "install", "--dry-run", "--quiet",
           "--disable-pip-version-check", "--no-input", "--report", "-"] + specs
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "pip resolution timed out (240s) — check network / index"}
    except Exception as e:
        return {"ok": False, "error": f"could not run pip: {type(e).__name__}: {e}"}

    if r.returncode != 0:
        # ResolutionImpossible / not-found / etc. — incompatible.
        return {"ok": True, "compatible": False, "python": sys.version.split()[0],
                "requested": requested, "would_install": [],
                "error": (r.stderr or r.stdout or "pip failed").strip()[-4000:]}

    would: list[str] = []
    try:
        rep = json.loads(r.stdout or "{}")
        for it in rep.get("install", []):
            m = it.get("metadata", {})
            if m.get("name"):
                would.append(f"{m['name']}=={m.get('version', '?')}")
    except Exception:
        would = []
    would.sort()
    return {"ok": True, "compatible": True, "python": sys.version.split()[0],
            "requested": requested, "would_install": would,
            "log": "already satisfied" if not would else f"{len(would)} package(s) would be installed/updated"}


def deps_install(specs: list[str]) -> dict:
    """Actually install `specs` into this interpreter's env. Long-running."""
    specs = [s.strip() for s in specs if isinstance(s, str) and s.strip()]
    if not specs:
        return {"ok": False, "error": "no dependencies specified"}
    cmd = [sys.executable, "-m", "pip", "install",
           "--disable-pip-version-check", "--no-input"] + specs
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "pip install timed out (1800s)"}
    except Exception as e:
        return {"ok": False, "error": f"could not run pip: {type(e).__name__}: {e}"}
    return {"ok": r.returncode == 0, "returncode": r.returncode,
            "log": ((r.stdout or "") + (r.stderr or "")).strip()[-8000:]}


class Handler(BaseHTTPRequestHandler):
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._json(200, {"ok": True, "torch": torch.__version__})
            return
        self.send_response(404)
        self._cors()
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or "0")
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as e:
            self._json(400, {"ok": False, "error": f"invalid json: {e}"})
            return

        try:
            if self.path == "/infer":
                self._handle_infer(payload)
                return
            if self.path == "/dataset/inspect":
                abspath = payload.get("abspath")
                if not isinstance(abspath, str):
                    self._json(400, {"ok": False, "error": "expected {abspath: str}"})
                    return
                self._json(200, ds_mod.inspect(abspath))
                return
            if self.path == "/dataset/stats":
                abspath = payload.get("abspath")
                if not isinstance(abspath, str):
                    self._json(400, {"ok": False, "error": "expected {abspath: str}"})
                    return
                self._json(200, ds_mod.stats(abspath))
                return
            if self.path == "/dataset/smoke":
                code = payload.get("code")
                # Accept either {abspath: str} (single dataset, broadcast) or
                # {abspaths: str[]} (one per input, multi-binding).
                abspaths_in = payload.get("abspaths")
                if abspaths_in is None:
                    single = payload.get("abspath")
                    if isinstance(single, str):
                        abspaths_in = [single]
                shapes_in = payload.get("input_shapes")
                if shapes_in is None:
                    single_shape = payload.get("input_shape")
                    if isinstance(single_shape, list):
                        shapes_in = [single_shape]
                if not isinstance(code, str) or not isinstance(abspaths_in, list) or not abspaths_in:
                    self._json(400, {"ok": False, "error": "expected {code, abspaths: str[] | abspath: str, input_shapes?: int[][]}"})
                    return
                abspaths = [str(p) for p in abspaths_in if isinstance(p, str)]
                shapes: list[list[int]] | None = None
                if isinstance(shapes_in, list):
                    shapes = [[int(v) for v in s] for s in shapes_in if isinstance(s, list)]
                    if not shapes:
                        shapes = None
                opts_in = payload.get("input_options")
                opts = opts_in if isinstance(opts_in, list) else None
                self._json(200, smoke_test(code, abspaths, shapes, opts))
                return
            if self.path == "/activations":
                code = payload.get("code")
                shapes_in = payload.get("input_shapes")
                if shapes_in is None:
                    single_shape = payload.get("input_shape")
                    if isinstance(single_shape, list):
                        shapes_in = [single_shape]
                if not isinstance(code, str) or not isinstance(shapes_in, list) or not all(isinstance(s, list) for s in shapes_in):
                    self._json(400, {"ok": False, "error": "expected {code: str, input_shapes: int[][]}"})
                    return
                normalized = [[int(v) for v in s] for s in shapes_in]
                dtypes_in = payload.get("input_dtypes")
                dtypes = [str(d) for d in dtypes_in] if isinstance(dtypes_in, list) else None
                abspaths_in = payload.get("abspaths")
                if abspaths_in is None:
                    single = payload.get("abspath")
                    if isinstance(single, str):
                        abspaths_in = [single]
                abspaths = [str(p) for p in abspaths_in if isinstance(p, str)] if isinstance(abspaths_in, list) else None
                opts_in = payload.get("input_options")
                opts = opts_in if isinstance(opts_in, list) else None
                ckpt = payload.get("checkpoint")
                ckpt = ckpt if isinstance(ckpt, str) and ckpt else None
                ga_in = payload.get("graph_attrs")
                ga = [str(a) for a in ga_in if isinstance(a, str)] if isinstance(ga_in, list) else None
                self._json(200, activations(code, normalized, dtypes, abspaths or None, opts, ckpt, ga))
                return
            if self.path == "/deps/check":
                specs = payload.get("specs")
                if not isinstance(specs, list):
                    self._json(400, {"ok": False, "error": "expected {specs: str[]}"})
                    return
                self._json(200, deps_check([str(s) for s in specs]))
                return
            if self.path == "/deps/install":
                specs = payload.get("specs")
                if not isinstance(specs, list):
                    self._json(400, {"ok": False, "error": "expected {specs: str[]}"})
                    return
                self._json(200, deps_install([str(s) for s in specs]))
                return
        except Exception as e:
            self._json(500, {
                "ok": False, "stage": "sidecar",
                "error": f"sidecar crash: {type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=4),
            })
            return

        self.send_response(404)
        self._cors()
        self.end_headers()

    def _handle_infer(self, payload: dict) -> None:
        code = payload.get("code")
        # Accept either input_shapes (multi-input list[list[int]]) or legacy input_shape (list[int]).
        shapes_in = payload.get("input_shapes")
        if shapes_in is None:
            single = payload.get("input_shape")
            if isinstance(single, list):
                shapes_in = [single]
        if not isinstance(code, str) or not isinstance(shapes_in, list) or not all(isinstance(s, list) for s in shapes_in):
            self._json(400, {"ok": False, "error": "expected {code: str, input_shapes: int[][]} or {input_shape: int[]}"})
            return
        dtypes_in = payload.get("input_dtypes")
        dtypes = [str(d) for d in dtypes_in] if isinstance(dtypes_in, list) else None
        try:
            normalized = [[int(v) for v in s] for s in shapes_in]
            result = infer(code, normalized, dtypes)
        except Exception as e:
            result = {
                "ok": False, "stage": "sidecar",
                "error": f"sidecar crash: {type(e).__name__}: {e}",
                "trace": traceback.format_exc(limit=4),
                "shapes": {},
            }
        self._json(200, result)

    def _json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_a) -> None:
        return


def main() -> None:
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[spinoml-torch] listening on http://127.0.0.1:{PORT} (torch {torch.__version__})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("[spinoml-torch] shutting down", file=sys.stderr)
        srv.server_close()


if __name__ == "__main__":
    main()
