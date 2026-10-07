#!/usr/bin/env python3
"""SpinoML — CUDA hardware suite (standalone, no node/npm required).

Checks: (1) env, (2) device↔CPU fp32 forward parity, (3) grad parity,
(4) REAL trainer on device AND CPU (same seed) comparing best_val_loss,
(5) repeatability across two DISTINCT run dirs with a real weight diff,
(6) fp32 lookup verified then bf16/fp16 amp, (7) in-process CUDA peak memory.

Without --allow-cpu-as-device AND no CUDA: prints exactly
"SKIPPED  CUDA (no CUDA device)" and exits 0.

The committed examples/reference-experiments/*/ has only model.spinoml (no model.py) —
codegen needs the TS toolchain, so we import the hand-written reference models from
scripts/lib/reference_models.py. Trainer = training_template.py copied into each run dir.

Verified artifact locations: metrics["env"]["device"|"dtype"], manifest["device"|"dtype"]
(no manifest.env).
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util as _ilu
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
import torch
import torch.nn as nn

# Tolerances (one-line justifications)
ATOL_FWD = 1e-4            # fp32 forward CUDA↔CPU: float rounding between devices
RTOL_FWD = 1e-4
ATOL_GRAD = 1e-3           # backward accumulates more ops → looser
TOL_REPEAT_LOSS_REL = 1e-2     # same-seed same-device × 2 may diverge from atomicAdd
TOL_REPEAT_WEIGHTS_REL = 1e-2  # ditto, ||w1-w2||_2 / ||w2||_2
LOSS_REL_TOL = 0.05        # cuda↔cpu best-loss: relative fudge for a small loss
LOSS_ABS_TOL = 0.02        # ...but an absolute floor so the ~3e-4 cnn loss isn't meaningless
AMP_LOSS_ABS_TOL = 0.5     # amp noise is larger than fp32, but a diverged run must fail
SEED = 11
EPOCHS = {"mlp": 4, "cnn": 6, "multi-input": 6}
PERTURB_ENV = "SPINOML_CUDA_TEST_PERTURB"  # documented test hook for mutation testing

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINER_SRC = REPO_ROOT / "sidecar-torch" / "training_template.py"
SAFE_LOAD_SRC = REPO_ROOT / "sidecar-torch" / "safe_load.py"
REF_MODELS_SRC = REPO_ROOT / "scripts" / "lib" / "reference_models.py"
EXPERIMENTS = ("mlp", "cnn", "multi-input")

def _check(name: str, ok: bool, detail: str = "", skipped: bool = False) -> bool:
    mark = "⊘" if skipped else ("✓" if ok else "✗")
    print(f"  {mark} {name}" + (f": {detail}" if detail else ""))
    return ok

def _fmt(x: object) -> str:
    """Safe number formatting — a missing/non-finite loss must never raise."""
    return f"{x:.4f}" if isinstance(x, (int, float)) and math.isfinite(x) else str(x)


# Reference models: load scripts/lib/reference_models.py (single source of truth)
_sp = _ilu.spec_from_file_location("_ref_models", REF_MODELS_SRC)
_REF = _ilu.module_from_spec(_sp); _sp.loader.exec_module(_REF)  # type: ignore[union-attr]
MODEL_CLASSES = _REF.REFERENCE_MODELS

def _model_py_source(name: str) -> str:
    """Emit a model.py that re-exports the hand-written reference class as `Model`.
    Single source of truth: scripts/lib/reference_models.py (re-imported at runtime)."""
    return (
        f"import sys\nsys.path.insert(0, {str(REF_MODELS_SRC.parent)!r})\n"
        f"import reference_models as _rm\nModel = _rm.{MODEL_CLASSES[name].__name__}\n"
    )

def _build_inputs(name: str, batch: int, dtype: torch.dtype, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    if name == "mlp":
        return {"x": torch.randn(batch, 10, generator=g, dtype=dtype)}
    if name == "cnn":
        return {"x": torch.randn(batch, 64, generator=g, dtype=dtype)}
    return {"a": torch.randn(batch, 6, generator=g, dtype=dtype), "b": torch.randn(batch, 4, generator=g, dtype=dtype)}

def _forward(model: nn.Module, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    params = [p for p in inspect.signature(model.forward).parameters if p != "self"]
    vals = list(inputs.values())
    return model(*[inputs[p] if p in inputs else vals[i] for i, p in enumerate(params)])

def _make_rng(seed: int):
    state = [seed & 0xFFFFFFFF]
    def rand() -> float:
        s = state[0]; s ^= (s << 13) & 0xFFFFFFFF; s ^= (s >> 17) & 0xFFFFFFFF; s ^= (s << 5) & 0xFFFFFFFF
        state[0] = s & 0xFFFFFFFF; return s / 0x100000000
    def randn() -> float:
        return math.sqrt(-2 * math.log(rand() + 1e-12)) * math.cos(2 * math.pi * rand())
    return rand, randn

def _synth_csv(name: str, seed: int, n: int = 200) -> tuple[str, list[str], str]:
    _, randn = _make_rng(seed)
    if name == "mlp":
        cols = [f"f{i}" for i in range(10)]
        rows = [",".join(cols + ["y"])]
        for i in range(n):
            mu = 0.6 if i % 2 else -0.6
            rows.append(",".join(f"{(mu if j < 5 else -mu) + randn() * 0.9:.6f}" for j in range(10)) + f",{i%2}")
        return "\n".join(rows) + "\n", cols, "y"
    if name == "cnn":
        cols = [f"p{i}" for i in range(64)]
        rows = [",".join(cols + ["y"])]
        for i in range(n):
            label = i % 2
            vals = []
            for r in range(8):
                for c in range(8):
                    v = randn() * 0.3
                    if (label == 1 and r < 4 and c < 4) or (label == 0 and r >= 4 and c >= 4):
                        v += 1.5
                    vals.append(v)
            rows.append(",".join(f"{v:.6f}" for v in vals) + f",{label}")
        return "\n".join(rows) + "\n", cols, "y"
    cols = [f"a{i}" for i in range(6)] + [f"b{i}" for i in range(4)]
    rows = [",".join(["id", "apath", "bpath"] + cols + ["y"])]
    for i in range(n):
        mu = 0.5 if i % 2 else -0.5
        a = [mu + randn() * 0.5 for _ in range(6)]
        b = [randn() * 1.0 for _ in range(4)]
        rows.append(",".join([str(i), f"a/{i}.pt", f"b/{i}.pt"] + [f"{v:.6f}" for v in a + b] + [str(i % 2)]))
    return "\n".join(rows) + "\n", cols, "y"

def _write_branch_tensors(csv_text: str, base: Path) -> None:
    (base / "a").mkdir(exist_ok=True); (base / "b").mkdir(exist_ok=True)
    for line in csv_text.splitlines()[1:]:
        if not line: continue
        c = line.split(",")
        torch.save(torch.tensor([float(x) for x in c[3:9]], dtype=torch.float32), base / "a" / f"{c[0]}.pt")
        torch.save(torch.tensor([float(x) for x in c[9:13]], dtype=torch.float32), base / "b" / f"{c[0]}.pt")

def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def _build_run_dir(name: str, parent: Path, seed: int, amp: str | None, tag: str = "") -> tuple[Path, str]:
    """Build a complete run dir mirroring verify-reference-train.ts layout.
    `tag` makes each trial a DISTINCT directory so no run inherits another's
    checkpoints/metrics (the trainer writes last.pt/best.pt into the run dir)."""
    run_dir = parent / f"run-{name}-seed{seed}-amp{amp or 'none'}{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TRAINER_SRC, run_dir / "train.py")
    shutil.copy2(SAFE_LOAD_SRC, run_dir / "safe_load.py")
    (run_dir / "model.py").write_text(_model_py_source(name), encoding="utf-8")
    (run_dir / "model.spinoml").write_text('{"version":1,"kind":"hw-cuda"}\n', encoding="utf-8")
    csv_text, feature_cols, target = _synth_csv(name, 7000 + EXPERIMENTS.index(name) * 100)
    (run_dir / "data.csv").write_text(csv_text, encoding="utf-8")
    fp = {"alg": "sha256", "mode": "content", "hash": _sha256_file(run_dir / "data.csv"),
          "size_bytes": (run_dir / "data.csv").stat().st_size}
    if name == "multi-input":
        _write_branch_tensors(csv_text, run_dir)
        (run_dir / "data.manifest").write_text(json.dumps(
            {"table": "data.csv", "pairs": {"a": {"column": "apath"}, "b": {"column": "bpath"}},
             "target": {"column": target, "type": "classification"}, "cache": False}), encoding="utf-8")
        ds = {"path": "data.manifest", "relpath": "data.manifest", "kind": "manifest",
              "feature_columns": None, "target_column": target, "fingerprint": fp}
    else:
        ds = {"path": "data.csv", "relpath": "data.csv", "kind": "tabular",
              "feature_columns": feature_cols, "target_column": target, "fingerprint": fp}
    training = {"epochs": EPOCHS[name], "batch_size": 32, "val_split": 0.2, "split_strategy": "random",
                "seed": seed, "log_every_n_steps": 1,
                "optimizer": {"kind": "Adam", "lr": 0.01, "weight_decay": 0},
                "loss": {"kind": "CrossEntropyLoss"}, "scheduler": {"kind": "none"},
                "metrics": ["accuracy"],
                "callbacks": [{"kind": "MixedPrecision", "dtype": amp}] if amp else []}
    snap = {"version": 1, "graph_sha256": _sha256_file(run_dir / "model.spinoml"),
            "model_py_sha256": _sha256_file(run_dir / "model.py"), "preprocessing": [], "code_trust": []}
    cfg = {"run_id": f"hw-{name}", "run_label": f"hw-{name}", "created_at": "2026-01-01T00:00:00Z",
           "status": "queued", "model_path": "model.spinoml", "backend": {"kind": "local"},
           "dataset": ds, "training": training, "snapshot": snap}
    (run_dir / "run.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return run_dir, "manifest" if name == "multi-input" else "tabular"

def _run_trainer(run_dir: Path, device: str) -> tuple[bool, str]:
    """Spawn train.py with THIS interpreter (the cluster venv python may not be
    first on PATH under srun). `device='cpu'` forces CUDA_VISIBLE_DEVICES=''."""
    env = os.environ.copy()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""  # force CPU even on a GPU host
    try:
        proc = subprocess.run([sys.executable, "-u", "train.py"], cwd=run_dir, env=env,
                              capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired:
        return False, "trainer timed out after 180s"
    return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")

def _load_mod(path: Path):
    sp = _ilu.spec_from_file_location("_sl", path)
    mod = _ilu.module_from_spec(sp); sp.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod

def _read_artifacts(run_dir: Path):
    """Return (metrics, manifest, ckpt_ok) or None when artifacts are missing/bad."""
    try:
        m = json.loads((run_dir / "metrics.json").read_text())
        man = json.loads((run_dir / "manifest.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    ckpt_ok = False
    try:
        mod = _load_mod(run_dir / "safe_load.py")
        best = mod.safe_torch_load(run_dir / "checkpoints" / "best.pt")
        ckpt_ok = all(k in best for k in ("model_state", "optim_state", "epoch", "global_step", "rng", "config"))
    except Exception as exc:
        print(f"    ckpt load: {type(exc).__name__}: {exc}", file=sys.stderr)
    return m, man, ckpt_ok

def _weight_diff(w1: dict, w2: dict) -> tuple[float, float]:
    """(max_abs, rel_l2) over all floating-point tensors present in both."""
    keys = [k for k in w1 if torch.is_tensor(w1[k]) and w1[k].is_floating_point() and k in w2]
    if not keys:
        return 0.0, 0.0
    d = torch.cat([(w1[k].float() - w2[k].float()).flatten() for k in keys])
    b = torch.cat([w2[k].float().flatten() for k in keys])
    return float(d.abs().max()), float(d.norm() / (b.norm() + 1e-12))

def _set_tf32(enabled: bool) -> None:
    torch.backends.cuda.matmul.allow_tf32 = enabled; torch.backends.cudnn.allow_tf32 = enabled

def check_env(device: str, cpu_standin: bool) -> bool:
    label = "(cpu stand-in — NOT a CUDA verification)" if cpu_standin else "(real CUDA)"
    print(f"\n[1/7] env  {label}")
    info = [f"torch={torch.__version__}", f"cuda_compiled={torch.version.cuda or 'none'}",
            f"is_available={torch.cuda.is_available()}"]
    if device == "cuda":
        info += [f"device={torch.cuda.get_device_name(0)}",
                 f"capability={torch.cuda.get_device_capability(0)}",
                 f"memory_mb={round(torch.cuda.get_device_properties(0).total_memory / 1e6)}"]
        try: info.append(f"cudnn={torch.backends.cudnn.version()}")
        except Exception as exc: info.append(f"cudnn=unknown({type(exc).__name__})")
    else:
        info.append("device=cpu")
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        if smi.returncode == 0 and smi.stdout.strip():
            info.append(f"driver={smi.stdout.strip().splitlines()[0]}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info.append("driver=n/a")
    line = " ".join(info)
    print("  " + line)
    return _check("env", True, line)
def check_forward(device: str, cpu_standin: bool) -> bool:
    print(f"\n[2/7] forward parity  {'(cpu standin)' if cpu_standin else '(cuda↔cpu, tf32 off)'}")
    _set_tf32(False)
    dev = "cpu" if cpu_standin else "cuda"  # the ONLY difference between stand-in and CUDA
    ok_all = True
    for name in EXPERIMENTS:
        ref = MODEL_CLASSES[name]().eval()
        inputs_cpu = _build_inputs(name, 8, torch.float32, SEED)
        with torch.no_grad():
            cpu_out = _forward(ref, inputs_cpu)
        ref_dev = copy.deepcopy(ref).to(dev)
        inputs_dev = {k: v.to(dev) for k, v in inputs_cpu.items()}
        with torch.no_grad():
            dev_out = _forward(ref_dev, inputs_dev)
        if PERTURB_ENV in os.environ:
            dev_out = dev_out + float(os.environ[PERTURB_ENV])  # mutation hook
        _set_tf32(True)  # also report TF32-enabled deviation (info only)
        with torch.no_grad():
            dev_tf32 = _forward(copy.deepcopy(ref_dev), inputs_dev)
        _set_tf32(False)
        maxd = (dev_out.cpu() - cpu_out).abs().max().item()
        tf32_dev = (dev_tf32.cpu() - cpu_out).abs().max().item()
        ok_all &= _check(name, torch.allclose(dev_out.cpu(), cpu_out, rtol=RTOL_FWD, atol=ATOL_FWD),
                         f"max_abs={maxd:.3e} (atol={ATOL_FWD}) tf32_dev={tf32_dev:.3e}")
    return ok_all
def check_grads(device: str, cpu_standin: bool) -> bool:
    print(f"\n[3/7] grad parity  {'(cpu standin)' if cpu_standin else '(cuda↔cpu)'}")
    _set_tf32(False)
    dev = "cpu" if cpu_standin else "cuda"
    target = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    ok_all = True
    for name in EXPERIMENTS:
        cpu_m = MODEL_CLASSES[name]()          # CPU copy
        dev_m = copy.deepcopy(cpu_m).to(dev)   # INDEPENDENT copy on the device (never .cuda() in place)
        for p in cpu_m.parameters(): p.requires_grad_(True)
        for p in dev_m.parameters(): p.requires_grad_(True)
        inputs_cpu = _build_inputs(name, 8, torch.float32, SEED)
        inputs_dev = {k: v.to(dev) for k, v in inputs_cpu.items()}
        nn.CrossEntropyLoss()(_forward(cpu_m, inputs_cpu), target).backward()
        nn.CrossEntropyLoss()(_forward(dev_m, inputs_dev), target.to(dev)).backward()
        worst = max(float((pc.grad - pg.grad.cpu()).abs().max().item())
                    for pc, pg in zip(cpu_m.parameters(), dev_m.parameters()))
        if PERTURB_ENV in os.environ:
            worst += float(os.environ[PERTURB_ENV])  # mutation hook
        ok_all &= _check(name, worst <= ATOL_GRAD, f"max_grad_diff={worst:.3e} (atol={ATOL_GRAD})")
    return ok_all
def check_trainer(device: str, cpu_standin: bool) -> bool:
    print(f"\n[4/7] trainer run  {'(cpu standin)' if cpu_standin else '(cuda vs cpu)'}")
    parent = Path(tempfile.mkdtemp(prefix="spinoml-hw-"))
    ok_all = True
    dev = "cpu" if cpu_standin else "cuda"
    for name in EXPERIMENTS:
        cpu_dir, _ = _build_run_dir(name, parent, SEED, None, "-cpu")
        dev_dir, _ = _build_run_dir(name, parent, SEED, None, "-dev")
        cpu_ok, cpu_log = _run_trainer(cpu_dir, "cpu")
        dev_ok, dev_log = _run_trainer(dev_dir, dev)
        if not cpu_ok:
            ok_all &= _check(f"{name}/cpu", False, f"exit nonzero: {cpu_log[-200:]}"); continue
        if not dev_ok:
            ok_all &= _check(f"{name}/{dev}", False, f"exit nonzero: {dev_log[-200:]}"); continue
        ca, da = _read_artifacts(cpu_dir), _read_artifacts(dev_dir)
        if ca is None or da is None:
            ok_all &= _check(name, False, "missing/bad artifacts"); continue
        cm, cman, cck = ca; dm, dman, dck = da
        cpu_loss = cm.get("best_val_loss"); dev_loss = dm.get("best_val_loss")
        cpu_dev = cm.get("env", {}).get("device"); dev_devv = dm.get("env", {}).get("device")
        nums = isinstance(cpu_loss, (int, float)) and isinstance(dev_loss, (int, float)) and math.isfinite(cpu_loss) and math.isfinite(dev_loss)
        delta = abs(dev_loss - cpu_loss) if nums else None
        tol = max(LOSS_REL_TOL * abs(cpu_loss), LOSS_ABS_TOL) if nums else None
        loss_ok = delta is not None and tol is not None and delta <= tol
        ok_sub = (cm.get("status") == "done" and dm.get("status") == "done" and cck and dck and loss_ok
                  and cpu_dev == "cpu" and cman.get("device") == "cpu"
                  and dev_devv == dev and dman.get("device") == dev)
        ok_all &= _check(name, ok_sub,
                         f"{dev}_loss={_fmt(dev_loss)} cpu_loss={_fmt(cpu_loss)} |Δ|={_fmt(delta)} tol={_fmt(tol)} "
                         f"dev={dev_devv}/cpu={cpu_dev} ckpt={dck}")
    shutil.rmtree(parent, ignore_errors=True)
    return ok_all
def check_repeatability(device: str, cpu_standin: bool) -> bool:
    print(f"\n[5/7] repeatability  {'(cpu standin)' if cpu_standin else '(cuda × 2)'}")
    dev = "cpu" if cpu_standin else "cuda"
    parent = Path(tempfile.mkdtemp(prefix="spinoml-hw-rep-"))
    safe_load = _load_mod(SAFE_LOAD_SRC).safe_torch_load
    ok_all = True
    results: list[tuple[float, float, float]] = []
    for name in EXPERIMENTS:
        runs = []
        for tag in ("-t1", "-t2"):
            run_dir, _ = _build_run_dir(name, parent, SEED, None, tag)  # distinct dir per trial
            ok, log = _run_trainer(run_dir, dev)
            if not ok:
                ok_all &= _check(name, False, f"{tag} failed: {log[-200:]}"); runs = None; break
            try:
                m = json.loads((run_dir / "metrics.json").read_text())
                best = safe_load(run_dir / "checkpoints" / "best.pt")
            except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
                ok_all &= _check(name, False, f"{tag} missing/bad: {exc}"); runs = None; break
            runs.append((m.get("best_val_loss"), best["model_state"]))
        if not runs:
            continue
        (l1, w1), (l2, w2) = runs
        nums = isinstance(l1, (int, float)) and isinstance(l2, (int, float)) and math.isfinite(l1) and math.isfinite(l2)
        loss_rel = abs(l1 - l2) / max(abs(l1), abs(l2), 1e-12) if nums else float("inf")
        wmax, wl2 = _weight_diff(w1, w2)
        results.append((loss_rel, wmax, wl2))
        ok_all &= _check(name, loss_rel <= TOL_REPEAT_LOSS_REL and wl2 <= TOL_REPEAT_WEIGHTS_REL,
                         f"loss {_fmt(l1)}/{_fmt(l2)} loss_rel={loss_rel:.3e} "
                         f"weights_max_abs={wmax:.3e} weights_rel_l2={wl2:.3e}")
    if results:
        mx = lambda i: max(r[i] for r in results)
        print(f"  MEASURED repeatability loss_rel={mx(0):.6e} weights_max_abs={mx(1):.6e} weights_rel_l2={mx(2):.6e}")
    shutil.rmtree(parent, ignore_errors=True)
    return ok_all
def check_amp(device: str, cpu_standin: bool) -> bool:
    print(f"\n[6/7] mixed precision  {'(cpu standin)' if cpu_standin else '(cuda amp)'}")
    parent = Path(tempfile.mkdtemp(prefix="spinoml-hw-amp-"))
    ok_all = True
    expect_dev = "cpu" if cpu_standin else "cuda"
    fp32_loss: dict[str, object] = {}
    # fp32 baseline: also verifies the real metrics.env.dtype / manifest.dtype lookup path
    for name in EXPERIMENTS:
        run_dir, _ = _build_run_dir(name, parent, SEED, None, "-fp32")
        ok, log = _run_trainer(run_dir, device)
        art = _read_artifacts(run_dir) if ok else None
        if art is None:
            ok_all &= _check(f"{name}/fp32", False, f"trainer failed: {log[-200:]}"); continue
        m, man, _ = art
        dt = m.get("env", {}).get("dtype"); dv = m.get("env", {}).get("device")
        fp32_loss[name] = m.get("best_val_loss")
        ok_all &= _check(f"{name}/fp32",
                         m.get("status") == "done" and dt == "fp32" and man.get("dtype") == "fp32"
                         and dv == expect_dev and man.get("device") == expect_dev,
                         f"dtype={dt}/{man.get('dtype')} dev={dv}/{man.get('device')} loss={_fmt(m.get('best_val_loss'))}")
    if device != "cuda":
        ok_all &= _check("bf16", True, "CPU: amp backward unsupported by DNNL — fp32 lookup path exercised above", skipped=True)
        ok_all &= _check("fp16", True, "CPU: amp backward unsupported by DNNL — fp32 lookup path exercised above", skipped=True)
        shutil.rmtree(parent, ignore_errors=True); return ok_all
    for amp in ("bf16", "fp16"):
        if amp == "bf16" and not torch.cuda.is_bf16_supported():
            ok_all &= _check("bf16", True, "torch.cuda.is_bf16_supported() == False", skipped=True); continue
        for name in EXPERIMENTS:
            run_dir, _ = _build_run_dir(name, parent, SEED, amp, f"-{amp}")
            ok, log = _run_trainer(run_dir, device)
            art = _read_artifacts(run_dir) if ok else None
            if art is None:
                ok_all &= _check(f"{name}/{amp}", False, f"trainer failed: {log[-200:]}"); continue
            m, man, _ = art
            dt = m.get("env", {}).get("dtype"); loss = m.get("best_val_loss"); base = fp32_loss.get(name)
            fin = isinstance(loss, (int, float)) and math.isfinite(loss)
            within = (fin and isinstance(base, (int, float)) and math.isfinite(base)
                      and abs(loss - base) <= AMP_LOSS_ABS_TOL)
            ok_all &= _check(f"{name}/{amp}",
                             m.get("status") == "done" and dt == amp and man.get("dtype") == amp and within,
                             f"dtype={dt}/{man.get('dtype')} loss={_fmt(loss)} fp32={_fmt(base)} "
                             f"|Δ|={_fmt(abs(loss - base) if fin and isinstance(base, (int, float)) else None)}")
    shutil.rmtree(parent, ignore_errors=True)
    return ok_all
def check_memory(device: str, cpu_standin: bool) -> bool:
    print(f"\n[7/7] memory  {'(cpu standin)' if cpu_standin else '(cuda)'}")
    if device != "cuda":
        return _check("memory", True, "not measured — no CUDA device (cpu stand-in)", skipped=True)
    # The trainer runs as a SUBPROCESS, so its peak is invisible here. Measure a
    # real in-process mlp training peak instead of reporting an un-measurable number.
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model = MODEL_CLASSES["mlp"]().cuda()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        inputs = {k: v.cuda() for k, v in _build_inputs("mlp", 32, torch.float32, SEED).items()}
        tgt = torch.randint(0, 2, (32,), generator=torch.Generator().manual_seed(SEED)).cuda()
        for _ in range(4):
            opt.zero_grad(); nn.CrossEntropyLoss()(_forward(model, inputs), tgt).backward(); opt.step()
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"  INFO peak_mb={peak:.1f}")
        return _check("memory", peak > 0, f"peak_mb={peak:.1f} (in-process mlp, 4 steps)")
    except RuntimeError as exc:
        return _check("memory", False, f"{type(exc).__name__}: {exc}")
def main() -> int:
    cpu_standin = "--allow-cpu-as-device" in sys.argv
    if cpu_standin:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not cpu_standin:
        print("SKIPPED  CUDA (no CUDA device)"); return 0
    label = "cpu stand-in — NOT a CUDA verification" if cpu_standin else "real CUDA"
    print(f"SpinoML CUDA hardware suite — device={device} ({label})")
    checks = [check_env(device, cpu_standin), check_forward(device, cpu_standin),
              check_grads(device, cpu_standin), check_trainer(device, cpu_standin),
              check_repeatability(device, cpu_standin), check_amp(device, cpu_standin),
              check_memory(device, cpu_standin)]
    failed = sum(1 for ok in checks if not ok)
    if failed:
        print(f"\n✗ {failed} check(s) failed ({label})"); return 1
    print(f"\nAll checks passed ({label})")
    return 0
if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # noqa: BLE001  — last-resort: print traceback, exit 2
        traceback.print_exc()
        raise SystemExit(2)
