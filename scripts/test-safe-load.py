#!/usr/bin/env python3
"""Phase 47 — safe torch.load.

Proves that sidecar-torch/safe_load.py refuses arbitrary pickle payloads
(remote code execution on an untrusted .pt) while every real SpinoML artifact
still loads, that the escape hatch works and is loud, and that the synced
loader block inside training_template.py is byte-identical.

Run:  python scripts/test-safe-load.py
"""
from __future__ import annotations

import ast
import contextlib
import glob
import io
import os
import pickle
import py_compile
import random
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sidecar-torch"))

import torch  # noqa: E402
import numpy as np  # noqa: E402

import safe_load  # noqa: E402
from safe_load import UnsafePickleError, safe_torch_load, register_safe_globals, unsafe_pickle_allowed  # noqa: E402
import dataset_handlers as dh  # noqa: E402

PASS = 0
FAIL = 0
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name} {detail}".strip())
        print(f"  ✗ {name} {detail}".rstrip())


TMP = Path(tempfile.mkdtemp(prefix="spinoml-safe-load-"))


class Evil:
    """Pickle payload whose __reduce__ runs a shell command when unpickled."""

    def __init__(self, sentinel: str) -> None:
        self.sentinel = sentinel

    def __reduce__(self):
        import os
        return (os.system, (f"touch {self.sentinel}",))


def marker_block(text: str) -> str:
    """Return the text strictly between the two safe_load marker lines."""
    lines = text.splitlines()
    start = end = None
    for i, line in enumerate(lines):
        if line.startswith("# >>> safe_load"):
            start = i
        elif line.startswith("# <<< safe_load"):
            end = i
    if start is None or end is None:
        return ""
    return "\n".join(lines[start + 1:end])


warnings.filterwarnings("ignore")

print("phase 47: safe torch.load")

# ── 1. REAL RCE ATTEMPT: an untrusted .pt must never execute its payload ──────
print("  [RCE attempt blocked]")
sentinel = str(TMP / "PWNED")
evil_dict = str(TMP / "evil_dict.pt")
evil_obj = str(TMP / "evil_obj.pt")
evil_raw = str(TMP / "evil_raw.pt")
torch.save({"w": torch.zeros(2), "x": Evil(sentinel)}, evil_dict)
torch.save(Evil(sentinel), evil_obj)
with open(evil_raw, "wb") as f:
    pickle.dump(Evil(sentinel), f)

for label, path in (("dict", evil_dict), ("obj", evil_obj), ("raw", evil_raw)):
    if os.path.exists(sentinel):
        os.unlink(sentinel)
    raised: BaseException | None = None
    try:
        safe_torch_load(path)
    except BaseException as e:  # noqa: BLE001
        raised = e
    check(f"evil {label}: not loaded (raised)", raised is not None, "loaded without error!")
    check(f"evil {label}: never executed payload", not os.path.exists(sentinel), "sentinel created!")
    if label in ("dict", "obj"):
        check(f"evil {label}: UnsafePickleError", isinstance(raised, UnsafePickleError),
              f"got {type(raised).__name__}")
    check(f"evil {label}: no blocked-name leak", raised is None or isinstance(raised, Exception))
check("sentinel absent at very end", not os.path.exists(sentinel), "sentinel exists!")

# blocked-name extraction from a PyG-less payload message
try:
    raise UnsafePickleError(evil_dict, ["posix.system"])
except UnsafePickleError as e:
    check("UnsafePickleError attrs path/blocked", e.path == evil_dict and e.blocked == ["posix.system"])
    check("UnsafePickleError message names blocked global", "posix.system" in str(e))
    check("UnsafePickleError message is user-facing", "SPINOML_ALLOW_UNSAFE_PICKLE" in str(e))

# ── 2. Escape hatch ───────────────────────────────────────────────────────────
print("  [escape hatch]")
_env_keys = ("SPINOML_ALLOW_UNSAFE_PICKLE",)
_saved_env = {k: os.environ.get(k) for k in _env_keys}
try:
    if os.path.exists(sentinel):
        os.unlink(sentinel)
    os.environ["SPINOML_ALLOW_UNSAFE_PICKLE"] = "1"
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        safe_torch_load(evil_obj)
    check("escape hatch: untrusted payload executed when opted in", os.path.exists(sentinel))
    check("escape hatch: warning written to stderr", "WARNING" in err.getvalue() and "SPINOML_ALLOW_UNSAFE_PICKLE=1" in err.getvalue())
finally:
    if os.path.exists(sentinel):
        os.unlink(sentinel)
    for k in _env_keys:
        if _saved_env[k] is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = _saved_env[k]
    if os.path.exists(sentinel):
        os.unlink(sentinel)
try:
    with contextlib.redirect_stderr(io.StringIO()):
        safe_torch_load(evil_obj)
    check("default: still refuses without env var", False, "loaded without opt-in!")
except UnsafePickleError:
    check("default: still refuses without env var", True)
check("default: sentinel not created", not os.path.exists(sentinel))

# unsafe_pickle_allowed truth table
def _with_env(value: str | None) -> bool:
    if value is None:
        os.environ.pop("SPINOML_ALLOW_UNSAFE_PICKLE", None)
    else:
        os.environ["SPINOML_ALLOW_UNSAFE_PICKLE"] = value
    return unsafe_pickle_allowed()

check('truth: unset -> False', _with_env(None) is False)
check('truth: "0" -> False', _with_env("0") is False)
check('truth: "1" -> True', _with_env("1") is True)
check('truth: "true" -> False', _with_env("true") is False)
check('truth: " 1" -> False', _with_env(" 1") is False)
os.environ.pop("SPINOML_ALLOW_UNSAFE_PICKLE", None)

# ── 3. Benign loads ───────────────────────────────────────────────────────────
print("  [benign loads]")
p_tensor = str(TMP / "tensor.pt")
t_val = torch.arange(6, dtype=torch.float32).reshape(2, 3)
torch.save(t_val, p_tensor)
check("benign: plain tensor", torch.equal(safe_torch_load(p_tensor), t_val))

p_dt = str(TMP / "dict_tensors.pt")
dt_val = {"a": torch.ones(2), "b": torch.zeros(3, dtype=torch.long)}
torch.save(dt_val, p_dt)
dt_out = safe_torch_load(p_dt)
check("benign: dict of tensors", torch.equal(dt_out["a"], dt_val["a"]) and torch.equal(dt_out["b"], dt_val["b"]))

p_nested = str(TMP / "nested.pt")
nested = {"l": [1, 2.5, True, None, "s"], "t": (1, "x"), "d": {"k": [3, 4]}}
torch.save(nested, p_nested)
check("benign: nested dict/list/tuple/scalars", safe_torch_load(p_nested) == nested)

from collections import OrderedDict  # noqa: E402
p_od = str(TMP / "ordered.pt")
od = OrderedDict([("a", 1), ("b", 2)])
torch.save(od, p_od)
od_out = safe_torch_load(p_od)
check("benign: OrderedDict", isinstance(od_out, OrderedDict) and list(od_out.items()) == [("a", 1), ("b", 2)])

p_size = str(TMP / "size.pt")
torch.save(torch.Size([2, 3]), p_size)
check("benign: torch.Size", tuple(safe_torch_load(p_size)) == (2, 3))

# trainer-style checkpoint (mirrors _rng_state() + build_ckpt())
p_ck = str(TMP / "trainer_ckpt.pt")
ckpt = {
    "epoch": 5,
    "global_step": 123,
    "model_state": {"fc.weight": torch.randn(3, 2), "fc.bias": torch.randn(3)},
    "optim_state": {"state": {}, "param_groups": [{"lr": 0.01, "weight_decay": 0.0}]},
    "sched_state": {"last_epoch": 5},
    "best_val": 0.5,
    "val_loss": 0.6,
    "classes": ["a", "b"],
    "head_classes": {"out": ["a", "b"]},
    "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate()},
    "config": {"training": {"epochs": 6}, "flag": True},
}
torch.save(ckpt, p_ck)
ck_out = safe_torch_load(p_ck)
check("benign: trainer checkpoint keys", sorted(ck_out.keys()) == sorted(ckpt.keys()))
check("benign: model_state tensors", torch.equal(ck_out["model_state"]["fc.weight"], ckpt["model_state"]["fc.weight"]))
check("benign: epoch/global_step scalars", ck_out["epoch"] == 5 and ck_out["global_step"] == 123)
check("benign: classes/head_classes", ck_out["classes"] == ["a", "b"] and ck_out["head_classes"] == {"out": ["a", "b"]})
check("benign: rng torch stream is a Tensor", torch.is_tensor(ck_out["rng"]["torch"]))
check("benign: rng numpy state present", "numpy" in ck_out["rng"])
check("benign: rng python state present", "python" in ck_out["rng"])

from torch_geometric.data import Data  # noqa: E402
p_data = str(TMP / "data.pt")
g_val = Data(
    x=torch.randn(4, 3),
    edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
    edge_attr=torch.randn(3, 2),
    pos=torch.randn(4, 3),
    y=torch.tensor([1.0]),
)
g_val.smi = "CCO"
torch.save(g_val, p_data)
g_out = safe_torch_load(p_data)
check("benign: PyG Data x", torch.equal(g_out.x, g_val.x))
check("benign: PyG Data edge_index", torch.equal(g_out.edge_index, g_val.edge_index))
check("benign: PyG Data edge_attr", torch.equal(g_out.edge_attr, g_val.edge_attr))
check("benign: PyG Data pos", torch.equal(g_out.pos, g_val.pos))
check("benign: PyG Data y", torch.equal(g_out.y, g_val.y))
check("benign: PyG Data extra string attr", g_out.smi == "CCO")

p_list = str(TMP / "list_data.pt")
torch.save([g_val, Data(x=torch.zeros(2, 3), edge_index=torch.zeros(2, 0, dtype=torch.long))], p_list)
l_out = safe_torch_load(p_list)
check("benign: list of Data", isinstance(l_out, list) and len(l_out) == 2 and torch.equal(l_out[0].x, g_val.x))

try:
    from torch_geometric.data import HeteroData
    p_het = str(TMP / "hetero.pt")
    h_val = HeteroData()
    h_val["a"].x = torch.randn(2, 3)
    h_val["a", "to", "b"].edge_index = torch.tensor([[0], [1]])
    torch.save(h_val, p_het)
    h_out = safe_torch_load(p_het)
    check("benign: HeteroData", list(h_out.node_types) == ["a"] and torch.equal(h_out["a"].x, h_val["a"].x))
except ImportError:
    pass

# ── 4. Real SpinoML artifacts ─────────────────────────────────────────────────
print("  [real artifacts]")
art_root = ROOT / "examples" / "reaction-workspace"
if art_root.is_dir():
    art_files = sorted(set(
        glob.glob(str(art_root / "**" / "*.pt"), recursive=True)
        + glob.glob(str(art_root / "datasets" / ".graphcache" / "*.pt"))
    ))
    check("real artifacts: found >= 1 .pt", len(art_files) >= 1, f"found {len(art_files)}")
    n_ok = 0
    for f in art_files:
        try:
            safe_torch_load(f)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            check(f"real artifact loads: {Path(f).name}", False, f"{type(e).__name__}: {e}")
    check(f"real artifacts: all {len(art_files)} load", n_ok == len(art_files), f"{n_ok} ok")
else:
    print("  SKIPPED real artifacts (examples/reaction-workspace not present)")

# ── 5. Non-pickle errors propagate unchanged ──────────────────────────────────
print("  [non-pickle errors propagate]")
missing = str(TMP / "does_not_exist.pt")
try:
    safe_torch_load(missing)
    check("missing file -> FileNotFoundError", False, "no error raised")
except FileNotFoundError:
    check("missing file -> FileNotFoundError", True)
except Exception as e:  # noqa: BLE001
    check("missing file -> FileNotFoundError", False, f"got {type(e).__name__}")

valid_pt = str(TMP / "valid_for_truncate.pt")
torch.save({"a": torch.arange(100)}, valid_pt)
truncated = str(TMP / "truncated.pt")
with open(valid_pt, "rb") as f:
    raw_bytes = f.read()
with open(truncated, "wb") as f:
    f.write(raw_bytes[: max(1, len(raw_bytes) // 2)])
try:
    safe_torch_load(truncated)
    check("truncated file raises", False, "no error raised")
except UnsafePickleError:
    check("truncated file is NOT UnsafePickleError", False, "misclassified as pickle problem")
except Exception:
    check("truncated file raises a non-pickle error", True)

# ── 6. register_safe_globals idempotent ───────────────────────────────────────
print("  [safe-globals registry]")
try:
    g1 = register_safe_globals()
    g2 = register_safe_globals()
    check("register_safe_globals: no exception, idempotent", g1 == g2, "lists differ")
    check("register_safe_globals: returns a list", isinstance(g1, list) and len(g1) > 0)
    try:
        import torch_geometric  # noqa: F401
        check("register_safe_globals: includes PyG Data", "torch_geometric.data.Data" in g1)
    except ImportError:
        pass
    try:
        import numpy  # noqa: F401
        check("register_safe_globals: includes numpy.ndarray", "numpy.ndarray" in g1)
        check("register_safe_globals: includes numpy dtype", "numpy.dtype" in g1)
    except ImportError:
        pass
    check("register_safe_globals: no dangerous module registered",
          not any(x.startswith(("os.", "builtins.", "subprocess.", "posix.")) for x in g1))
except Exception as e:  # noqa: BLE001
    check("register_safe_globals: no exception", False, f"{type(e).__name__}: {e}")

# ── 7. PARITY of the synced block ─────────────────────────────────────────────
print("  [synced block parity]")
safe_src = (ROOT / "sidecar-torch" / "safe_load.py").read_text(encoding="utf-8")
tmpl_src = (ROOT / "sidecar-torch" / "training_template.py").read_text(encoding="utf-8")
b_safe = marker_block(safe_src)
b_tmpl = marker_block(tmpl_src)
check("parity: safe_load.py has the marker block", bool(b_safe.strip()))
check("parity: training_template.py has the marker block", bool(b_tmpl.strip()))
check("parity: blocks byte-identical", b_safe == b_tmpl,
      "" if b_safe == b_tmpl else "\n" + "\n".join(
          __import__("difflib").unified_diff(b_safe.splitlines(), b_tmpl.splitlines(), "safe_load.py", "training_template.py", lineterm="")))

# ── 8. STATIC: no bare torch.load, one escape-hatch weights_only=False ────────
print("  [static call-site audit]")
sidecar = ROOT / "sidecar-torch"
for py in sorted(sidecar.glob("*.py")):
    text = py.read_text(encoding="utf-8")
    has_block = "# >>> safe_load" in text and "# <<< safe_load" in text
    try:
        tree = ast.parse(text)
    except SyntaxError as e:  # noqa: BLE001
        check(f"static: {py.name} parses", False, str(e))
        continue
    # count weights_only=False keyword args anywhere
    false_kw = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "weights_only" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                    false_kw += 1
    check(f"static: {py.name} weights_only=False count", false_kw == (1 if has_block else 0),
          f"got {false_kw}, expected {1 if has_block else 0}")
    # every torch.load call must live inside the synced block
    if has_block:
        lines = text.splitlines()
        start = next(i for i, l in enumerate(lines, 1) if l.startswith("# >>> safe_load"))
        end = next(i for i, l in enumerate(lines, 1) if l.startswith("# <<< safe_load"))
    else:
        start = end = -1
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "load":
            v = node.func.value
            if isinstance(v, ast.Name) and v.id == "torch":
                check(f"static: {py.name}:{node.lineno} torch.load inside synced block",
                      has_block and start < node.lineno < end,
                      "bare torch.load outside the synced block")

# ── 9. dataset_handlers integration ───────────────────────────────────────────
print("  [dataset_handlers integration]")
evil_ds = str(TMP / "ds_evil.pt")
torch.save({"w": torch.zeros(2), "x": Evil(sentinel)}, evil_ds)
benign_ds = str(TMP / "ds_benign.pt")
torch.save(torch.arange(6, dtype=torch.float32), benign_ds)

for fn_name in ("inspect", "stats", "sample_tensor"):
    fn = getattr(dh, fn_name)
    if os.path.exists(sentinel):
        os.unlink(sentinel)
    res = fn(evil_ds)
    check(f"dataset_handlers.{fn_name}: ok is False", isinstance(res, dict) and res.get("ok") is False,
          f"got {res}")
    check(f"dataset_handlers.{fn_name}: error_code UNSAFE_PICKLE",
          isinstance(res, dict) and res.get("error_code") == "UNSAFE_PICKLE", f"got {res}")
    check(f"dataset_handlers.{fn_name}: error message present", bool(isinstance(res, dict) and res.get("error")))
    check(f"dataset_handlers.{fn_name}: payload never executed", not os.path.exists(sentinel))

check("dataset_handlers.inspect: benign ok", dh.inspect(benign_ds).get("ok") is True)
check("dataset_handlers.stats: benign ok", dh.stats(benign_ds).get("ok") is True)
_benign_sample = dh.sample_tensor(benign_ds)
check("dataset_handlers.sample_tensor: benign ok", _benign_sample.get("ok") is True)
check("dataset_handlers.sample_tensor: benign tensor non-empty",
      torch.is_tensor(_benign_sample.get("tensor")) and _benign_sample["tensor"].numel() > 0)

# ── 10. training_template still compiles ──────────────────────────────────────
print("  [training_template compiles]")
tmpl_path = str(ROOT / "sidecar-torch" / "training_template.py")
try:
    ast.parse(tmpl_src)
    check("training_template: ast.parse succeeds", True)
except SyntaxError as e:  # noqa: BLE001
    check("training_template: ast.parse succeeds", False, str(e))
try:
    py_compile.compile(tmpl_path, cfile=str(TMP / "train.pyc"), doraise=True)
    check("training_template: py_compile succeeds", True)
except py_compile.PyCompileError as e:  # noqa: BLE001
    check("training_template: py_compile succeeds", False, str(e))

# ── summary ───────────────────────────────────────────────────────────────────
print()
if FAIL:
    print(f"✗ {FAIL} check(s) failed, {PASS} passed")
    for f in FAILURES:
        print(f"    - {f}")
    sys.exit(1)
print(f"✓ all {PASS} checks passed")
