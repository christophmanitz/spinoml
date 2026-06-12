"""MLForge shape-inference sidecar.

Runs a tiny HTTP server on 127.0.0.1:7421. Frontend POSTs the generated
PyTorch code plus an input shape; we exec it in a fresh namespace, register
forward hooks on every named submodule, run a zero-tensor forward pass, and
return the per-attr output shapes.

The frontend then maps those attr names back to node ids via the
attrMap returned alongside the code by the codegen.

Safety: this exec's code from the local frontend only. CORS is permissive
because the dev server (Vite, port 5173) and the Tauri webview both need
to call it; the bind address is 127.0.0.1 so no external host can reach it.
"""

from __future__ import annotations

import json
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

PORT = 7421


def infer(code: str, input_shape: list[int]) -> dict:
    shapes: dict[str, list[int]] = {}
    ns: dict = {"__name__": "<mlforge-model>"}
    try:
        exec(compile(code, "<mlforge-model>", "exec"), ns)
    except Exception as e:
        return {
            "ok": False,
            "stage": "compile",
            "error": f"{type(e).__name__}: {e}",
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
        x = torch.zeros(input_shape)
    except Exception as e:
        return {
            "ok": False,
            "stage": "input",
            "error": f"could not build zero tensor of shape {input_shape}: {e}",
            "shapes": shapes,
            "n_params": n_params,
        }

    try:
        with torch.no_grad():
            out = model(x)
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
        shapes["__output__"] = list(out[0].shape)

    return {"ok": True, "shapes": shapes, "n_params": n_params}


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
            body = json.dumps({"ok": True, "torch": torch.__version__}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self._cors()
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/infer":
            self.send_response(404)
            self._cors()
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0") or "0")
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as e:
            self._json(400, {"ok": False, "error": f"invalid json: {e}"})
            return
        code = payload.get("code")
        input_shape = payload.get("input_shape")
        if not isinstance(code, str) or not isinstance(input_shape, list):
            self._json(400, {"ok": False, "error": "expected {code: str, input_shape: int[]}"})
            return
        try:
            result = infer(code, [int(v) for v in input_shape])
        except Exception as e:
            result = {
                "ok": False,
                "stage": "sidecar",
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
