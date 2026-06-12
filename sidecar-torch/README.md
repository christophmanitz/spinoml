# sidecar-torch

Python process for PyTorch shape inference.

**Status**: Phase 3 implemented.

Tiny stdlib HTTP server on `127.0.0.1:7421`. The frontend POSTs the generated
`nn.Module` source plus an input shape; we exec the code in a fresh namespace,
register forward hooks on every named submodule, run a zero-tensor forward
pass, and return per-attr output shapes plus parameter count. Errors
(compile / construct / forward) come back with stage + traceback so the
frontend can flag the offending stage.

```
POST /infer
in : {"code": "<generated nn.Module source>", "input_shape": [1, 3, 224, 224]}
out: {"ok": true,  "shapes": {"conv2d_1": [1, 64, 112, 112], ..., "__output__": [...]}, "n_params": 12345}
     {"ok": false, "stage": "forward", "error": "RuntimeError: …", "trace": "…", "shapes": {…}}

GET  /health
out: {"ok": true, "torch": "2.12.0+cpu"}
```

Start:

```bash
conda activate mlforge-dev
python sidecar-torch/main.py     # or: npm run sidecar:torch
```
