"""MLForge shape-inference + dataset sidecar.

Runs a tiny HTTP server on 127.0.0.1:7421.

Endpoints:
  GET  /health
  POST /infer            { code, input_shape }            → per-layer output shapes
  POST /dataset/inspect  { abspath }                       → kind + cheap metadata
  POST /dataset/stats    { abspath }                       → stats/histograms (heavier)
  POST /dataset/smoke    { code, abspath, input_shape? }   → run sample through generated model

Safety: this exec's code from the local frontend only. CORS is permissive
because the dev server (Vite, port 5173) and the Tauri webview both need
to call it; the bind address is 127.0.0.1 so no external host can reach it.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

import dataset_handlers as ds_mod

# Default 7421 keeps local-mode behaviour unchanged. Override via env so a
# remote deploy (Phase 12b) can pick a free port on the HPC login node
# without colliding with another user's sidecar.
PORT = int(os.environ.get("MLFORGE_TORCH_PORT", "7421"))


def infer(
    code: str,
    input_shapes: list[list[int]],
    input_dtypes: list[str] | None = None,
) -> dict:
    shapes: dict[str, list[int]] = {}
    ns: dict = {"__name__": "<mlforge-model>"}
    try:
        exec(compile(code, "<mlforge-model>", "exec"), ns)
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
            if dt in ("int64", "long"):
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

    if len(abspaths) == 1 and input_shapes and len(input_shapes) > 1:
        path = abspaths[0]
        for i, sh in enumerate(input_shapes):
            sub = ds_mod.sample_tensor(path, sh, opts_for(i))
            if not sub.get("ok"):
                return {"ok": False, "stage": "sample", "error": sub.get("error"), "details": sub, "dataset": path}
            xs.append(sub["tensor"])
            if sub.get("note"): notes.append(f"{path.split('/')[-1]}: {sub['note']}")
    else:
        n = max(len(abspaths), len(input_shapes) if input_shapes else 0)
        for i in range(n):
            path = abspaths[i] if i < len(abspaths) else abspaths[-1]
            sh = input_shapes[i] if input_shapes and i < len(input_shapes) else None
            sub = ds_mod.sample_tensor(path, sh, opts_for(i))
            if not sub.get("ok"):
                return {"ok": False, "stage": "sample", "error": sub.get("error"), "details": sub, "dataset": path}
            xs.append(sub["tensor"])
            if sub.get("note"): notes.append(f"{path.split('/')[-1]}: {sub['note']}")
    t_sample = time.perf_counter() - t0

    ns: dict = {"__name__": "<mlforge-model>"}
    try:
        exec(compile(code, "<mlforge-model>", "exec"), ns)
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

    t1 = time.perf_counter()
    try:
        with torch.no_grad():
            out = model(*xs)
    except Exception as e:
        return {
            "ok": False, "stage": "forward",
            "error": f"{type(e).__name__}: {e}",
            "trace": traceback.format_exc(limit=6),
            "input_shape": [list(t.shape) for t in xs],
            "n_params": n_params,
        }
    t_forward = time.perf_counter() - t1

    if isinstance(out, torch.Tensor):
        out_shape: list[int] | list[list[int]] | None = list(out.shape)
    elif isinstance(out, tuple) and out and all(isinstance(o, torch.Tensor) for o in out):
        out_shape = [list(o.shape) for o in out]
    elif isinstance(out, dict):
        out_shape = [list(v.shape) for v in out.values() if isinstance(v, torch.Tensor)]
    else:
        out_shape = None

    input_shape_report: list[int] | list[list[int]] = (
        list(xs[0].shape) if len(xs) == 1 else [list(t.shape) for t in xs]
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
    print(f"[mlforge-torch] listening on http://127.0.0.1:{PORT} (torch {torch.__version__})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("[mlforge-torch] shutting down", file=sys.stderr)
        srv.server_close()


if __name__ == "__main__":
    main()
