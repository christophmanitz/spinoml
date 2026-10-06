"""Safe torch.load wrapper — the single place SpinoML unpickles .pt data.

A `.pt`/`.pth` file may come from a colleague, a download, or a dataset
mirror, so it is NOT trusted: loading one with `weights_only=False` runs
arbitrary Python from the file (pickle = remote code execution). This module
loads with `weights_only=True` plus a curated allow-list of the real SpinoML
artifact types (PyG `Data`, numpy RNG-state arrays), and refuses — loudly —
anything else, unless the operator explicitly sets
`SPINOML_ALLOW_UNSAFE_PICKLE=1`.

The block between the `# >>> safe_load` / `# <<< safe_load` markers is kept
byte-identical inside `training_template.py` (which is copied standalone into
every run directory and therefore cannot import this module); `scripts/test-safe-load.py`
enforces the equality.
"""

from __future__ import annotations

# >>> safe_load (synced block: keep byte-identical with the copy in training_template.py; checked by scripts/test-safe-load.py)
class UnsafePickleError(Exception):
    """Raised when a .pt file requires arbitrary unpickling that weights_only=True refuses."""

    def __init__(self, path: str, blocked: list[str]) -> None:
        import os
        self.path = path
        self.blocked = list(blocked)
        names = ", ".join(self.blocked) if self.blocked else "unknown globals"
        super().__init__(
            f"{os.path.basename(str(path))}: refusing to unpickle arbitrary Python objects "
            f"(blocked: {names}). Re-save the data as tensors/dicts/PyG Data objects, or "
            "start the sidecar/trainer with SPINOML_ALLOW_UNSAFE_PICKLE=1 only if you trust "
            "where the file came from."
        )


def unsafe_pickle_allowed() -> bool:
    """True iff SPINOML_ALLOW_UNSAFE_PICKLE=1 (the explicit trust escape hatch)."""
    import os
    return os.environ.get("SPINOML_ALLOW_UNSAFE_PICKLE") == "1"


_SAFE_GLOBALS_CACHE: list[str] | None = None


def register_safe_globals() -> list[str]:
    """Idempotently allowlist PyG/numpy globals weights_only=True rejects; returns their names."""
    global _SAFE_GLOBALS_CACHE
    if _SAFE_GLOBALS_CACHE is not None:
        return list(_SAFE_GLOBALS_CACHE)
    import torch
    registered: list[str] = []
    allow: list = []

    def _add(qual: str, obj: object) -> None:
        if obj is None:
            return
        allow.append(obj)
        registered.append(qual)

    try:
        import torch_geometric.data.data as _pg_data
        _add("torch_geometric.data.Data", getattr(_pg_data, "Data", None))
        _add("torch_geometric.data.data.DataEdgeAttr", getattr(_pg_data, "DataEdgeAttr", None))
        _add("torch_geometric.data.data.DataTensorAttr", getattr(_pg_data, "DataTensorAttr", None))
    except ImportError:  # optional dependency: PyG absent → these globals are not registered
        pass
    try:
        import torch_geometric.data as _pg
        _add("torch_geometric.data.HeteroData", getattr(_pg, "HeteroData", None))
    except ImportError:  # optional dependency: PyG absent → HeteroData global not registered
        pass
    try:
        import torch_geometric.data.storage as _pg_store
        _add("torch_geometric.data.storage.GlobalStorage", getattr(_pg_store, "GlobalStorage", None))
        _add("torch_geometric.data.storage.NodeStorage", getattr(_pg_store, "NodeStorage", None))
        _add("torch_geometric.data.storage.EdgeStorage", getattr(_pg_store, "EdgeStorage", None))
        _add("torch_geometric.data.storage.BaseStorage", getattr(_pg_store, "BaseStorage", None))
    except ImportError:  # optional dependency: PyG absent → storage globals not registered
        pass
    try:
        import numpy as _np
    except ImportError:
        _np = None
    if _np is not None:
        _add("numpy.ndarray", _np.ndarray)
        _add("numpy.dtype", _np.dtype)
        try:
            import numpy._core.multiarray as _np_ma
        except ImportError:
            try:
                import numpy.core.multiarray as _np_ma  # type: ignore[no-redef]
            except ImportError:
                _np_ma = None
        if _np_ma is not None:
            _add("numpy.multiarray._reconstruct", getattr(_np_ma, "_reconstruct", None))
            _add("numpy.multiarray.scalar", getattr(_np_ma, "scalar", None))
        try:
            import numpy.dtypes as _np_dtypes
            for _name in dir(_np_dtypes):
                if _name.endswith("DType"):
                    _cls = getattr(_np_dtypes, _name, None)
                    if isinstance(_cls, type):
                        _add("numpy.dtypes." + _name, _cls)
        except ImportError:  # optional: numpy.dtypes unavailable on this build → skip
            pass
    try:
        torch.serialization.add_safe_globals(allow)
    except AttributeError:
        # torch < 2.4 has no safe-globals API: plain tensors/dicts still load;
        # anything else is refused by weights_only=True rather than crashing.
        pass
    _SAFE_GLOBALS_CACHE = registered
    return list(registered)


def _is_weights_only_error(exc: BaseException) -> bool:
    """True iff `exc` is torch's weights-only refusal (not a corrupt/missing file)."""
    text = str(exc)
    return ("Weights only load failed" in text
            or "WeightsUnpickler error" in text
            or "was not an allowed global" in text
            or ("whose module" in text and "is blocked" in text))


def _blocked_globals(exc: BaseException) -> list[str]:
    """Extract the qualified names of the globals torch's weights-only unpickler refused."""
    import re
    names: list[str] = []
    for match in re.finditer(r"GLOBAL\s+([A-Za-z_][\w.]*)", str(exc)):
        name = match.group(1)
        if name not in names:
            names.append(name)
    return names


def safe_torch_load(path, map_location="cpu"):
    """Load a .pt/.pth with weights_only=True; refuse arbitrary unpickling unless trusted."""
    import os
    import sys
    import torch
    register_safe_globals()
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as exc:
        if not _is_weights_only_error(exc):
            raise
        blocked = _blocked_globals(exc)
        if unsafe_pickle_allowed():
            sys.stderr.write(
                "[spinoml] WARNING: loading %s with weights_only=False because "
                "SPINOML_ALLOW_UNSAFE_PICKLE=1 — arbitrary code in the file would run\n"
                % os.path.basename(str(path))
            )
            return torch.load(path, map_location=map_location, weights_only=False)
        raise UnsafePickleError(str(path), blocked) from exc
# <<< safe_load
