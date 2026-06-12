# sidecar-torch

Python process for PyTorch shape inference.

**Status**: stub, populated in Phase 3.

Reads graph snapshots on stdin (JSON), builds a transient `nn.Module`, runs a
dummy forward pass with a zero tensor of the user-specified input shape, and
returns per-node output shapes plus parameter count.

```
in : {"op": "infer", "graph": {...}, "input_shape": [1, 3, 224, 224]}
out: {"shapes": {"node_id": [1, 64, 112, 112], ...}, "params": 12345, "error": null}
```
