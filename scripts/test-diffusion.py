"""Diffusion (score head) through the REAL trainer: corruption, per-sigma metrics, sampling run,
determinism and loud failures. Run inside the conda env: `npm run test:diffusion`.

Toy task: the clean point-set branch ("wat") is a noisy subset of the conditioning atoms ("prot"),
so a denoiser that points from the nearest atom to the particle can learn eps."""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "sidecar-torch" / "training_template.py"

MODEL = '''import torch, torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.f = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 1))
    def forward(self, prot, wat):
        assert wat.sigma.shape[0] == wat.pos.shape[0] == wat.batch.shape[0]
        out = torch.zeros_like(wat.pos)
        for g in range(int(prot.num_graphs)):
            pm, wm = prot.batch == g, wat.batch == g
            a, x = prot.pos[pm], wat.pos[wm]
            d = torch.cdist(x, a); j = d.argmin(1); vec = x - a[j]; dist = d.min(1).values
            w = self.f(torch.stack([torch.log(wat.sigma[wm]), dist], -1))
            out[wm] = vec / (dist[:, None] + 1e-6) * w
        return {"eps_hat": out}
'''

fails = 0


def check(name, cond, detail=""):
    global fails
    print(("  ✓ " if cond else "  ✗ ") + name + (f"  ({detail})" if detail and not cond else ""))
    fails += 0 if cond else 1


def build_data(base: Path):
    torch.manual_seed(0)
    (base / "prot").mkdir()
    (base / "water").mkdir()
    rows = []
    for i in range(12):
        n, m = 8 + i % 4, 3 + i % 3
        pp = torch.randn(n, 3) * 3
        w = pp[torch.randperm(n)[:m]] + 0.05 * torch.randn(m, 3)
        torch.save(Data(x=torch.randn(n, 4), edge_index=torch.randint(0, n, (2, 10)), pos=pp), base / f"prot/p{i}.pt")
        torch.save(Data(x=torch.zeros(m, 1), edge_index=torch.zeros(2, 0, dtype=torch.long), pos=w), base / f"water/w{i}.pt")
        rows.append(f"p{i},w{i},0.0")
    (base / "t.csv").write_text("prot,wat,y\n" + "\n".join(rows) + "\n")
    (base / "m.manifest").write_text(json.dumps({
        "table": "t.csv", "target": {"column": "y", "type": "regression"},
        "pairs": {"prot": {"column": "prot", "dir": str(base / "prot"), "ext": ".pt"},
                  "wat": {"column": "wat", "dir": str(base / "water"), "ext": ".pt"}}}))


def run(base: Path, name: str, **extra) -> tuple[Path, list[dict]]:
    d = base / name
    d.mkdir()
    shutil.copy(TEMPLATE, d / "train.py")
    (d / "model.py").write_text(MODEL)
    heads = [{"output": "eps_hat", "target_kind": "score", "task": "regression", "loss": "MSELoss", "weight": 1.0, "target": ""}]
    cfg = {"run_id": name, "run_label": name, "created_at": "2026-01-01T00:00:00Z", "status": "queued", "model_path": "model.py",
           "backend": {"kind": "local"}, "dataset": {"path": str(base / "m.manifest"), "relpath": "datasets/m.manifest", "kind": "manifest"},
           "training": {"epochs": 40, "batch_size": 4, "val_split": 0.25, "split_strategy": "random", "seed": 1,
                        "optimizer": {"kind": "Adam", "lr": 0.01}, "heads": heads,
                        "diffusion": {"branch": "wat", "sigma_min": 0.05, "sigma_max": 6.0, "n_rep": 4}}}
    for k, v in extra.items():
        if k == "heads":
            cfg["training"]["heads"] = v
        elif k == "diffusion":
            cfg["training"]["diffusion"] = v
        else:
            cfg[k] = v
    (d / "run.json").write_text(json.dumps(cfg))
    subprocess.run([sys.executable, "-u", "train.py"], cwd=d, capture_output=True)
    ev = [json.loads(line) for line in (d / "events.jsonl").read_text().splitlines()] if (d / "events.jsonl").exists() else []
    return d, ev


def status(d: Path) -> str:
    return (d / "status").read_text().strip() if (d / "status").exists() else "?"


with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp)
    build_data(base)

    print("training with a score head")
    d, ev = run(base, "train")
    epochs = [e for e in ev if e["kind"] == "epoch.end"]
    check("run done", status(d) == "done", status(d))
    check("val loss decreases", len(epochs) == 40 and epochs[-1]["val_loss"] < epochs[0]["val_loss"],
          f"{epochs[0]['val_loss'] if epochs else None} -> {epochs[-1]['val_loss'] if epochs else None}")
    bins = [k for k in (epochs[-1]["metrics"] or {}) if k.startswith("dsm_err/")] if epochs else []
    check("4 per-sigma DSM metrics", len(bins) == 4, str(bins))
    check("best checkpoint written", (d / "checkpoints" / "best.pt").exists())

    print("sampling run")
    scfg = {"checkpoint_from": str(d / "checkpoints" / "best.pt"), "n_rows": 3, "n_steps": 20, "n_particles": 16,
            "anchor_branch": "prot", "traj_sigmas": [6.0, 0.05], "seed": 0}
    s1, ev1 = run(base, "s1", sample=scfg)
    check("sampling run done", status(s1) == "done", status(s1))
    check("sample.done event", any(e["kind"] == "sample.done" and e["n_points"] == 48 for e in ev1))
    if (s1 / "samples.pt").exists():
        smp = torch.load(s1 / "samples.pt")
        check("samples.pt shapes", tuple(smp["pos"].shape) == (48, 3) and tuple(smp["batch"].shape) == (48,)
              and smp["row_idx"].tolist() == [0, 1, 2], str({k: tuple(v.shape) for k, v in smp.items()}))
        check("samples finite", bool(torch.isfinite(smp["pos"]).all()))
    else:
        check("samples.pt exists", False)
    check("trajectory files", (s1 / "trajectory_6.pt").exists() and (s1 / "trajectory_0.05.pt").exists())
    s2, _ = run(base, "s2", sample=scfg)
    same = (s1 / "samples.pt").exists() and (s2 / "samples.pt").exists() and torch.equal(
        torch.load(s1 / "samples.pt")["pos"], torch.load(s2 / "samples.pt")["pos"])
    check("same seed -> identical samples", same)
    sd, _ = run(base, "s3", sample={**scfg, "seed": 1})
    check("other seed -> different samples", (sd / "samples.pt").exists() and not torch.equal(
        torch.load(s1 / "samples.pt")["pos"], torch.load(sd / "samples.pt")["pos"]))

    print("loud failures")
    two = [{"output": "eps_hat", "target_kind": "score", "task": "regression", "loss": "MSELoss", "weight": 1.0, "target": ""},
           {"output": "aux", "target_kind": "column", "task": "regression", "loss": "MSELoss", "weight": 1.0, "target": "y"}]
    f1, e1 = run(base, "f1", heads=two)
    check("score + column head fails", status(f1) == "failed" and any("ONLY head" in str(e.get("error")) for e in e1))
    f2, e2 = run(base, "f2", diffusion={"branch": "nope", "sigma_min": 0.05, "sigma_max": 6.0, "n_rep": 4})
    check("unknown branch fails", status(f2) == "failed" and any("not a manifest branch" in str(e.get("error")) for e in e2))
    f3, e3 = run(base, "f3", diffusion={"branch": "wat", "sigma_min": 6.0, "sigma_max": 0.05, "n_rep": 4})
    check("sigma_min >= sigma_max fails", status(f3) == "failed")

print("\n✓ all diffusion checks passed" if not fails else f"\n✗ {fails} check(s) failed")
sys.exit(1 if fails else 0)
