#!/usr/bin/env python3
"""Phase 17 — dataset handler reliability matrix.

Every dataset kind (except the pyarrow-gated parquet path) is exercised
over the TODO §18 failure scenarios: valid / empty / missing / corrupt /
wrong-dtype / NaN / Inf / single-sample / large / unicode-path / spaces-path /
relative-path / absolute-path.

Invariant under test: results are EXPLICIT — either ok:true with a real,
non-empty tensor, or ok:false with a human error string. Never a silent
empty tensor, never a thrown exception (the HTTP layer would 500), and
never a hang.

Run:  python scripts/test-datasets.py   (locally, no sidecar needed)
"""
from __future__ import annotations

import os
import signal
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sidecar-torch"))

import dataset_handlers as dh  # noqa: E402

import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

PASS = 0
FAIL = 0
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        FAILURES.append(f"{name} {detail}".strip())


def ok_result(r, name: str, require_tensor: bool = True) -> None:
    """r is a handler response. If ok and require_tensor, it must carry a
    real non-empty tensor."""
    global PASS, FAIL
    if not isinstance(r, dict):
        check(f"{name}: map result", False, f"got {type(r).__name__}")
        return
    if r.get("ok"):
        if not require_tensor:
            check(f"{name}: ok", True)
            return
        t = r.get("tensor")
        if t is None:
            check(f"{name}: ok must carry tensor", False, "no tensor key")
            return
        if not isinstance(t, torch.Tensor):
            check(f"{name}: tensor must be torch.Tensor", False, f"got {type(t).__name__}")
            return
        if t.numel() == 0:
            check(f"{name}: never silently empty", False, "ok:true but 0 elements")
            return
        check(f"{name}: ok", True)
        return
    err = r.get("error")
    check(f"{name}: explicit error", isinstance(err, str) and bool(err), "ok:false but no error string")


def either_counted_or_err(r, name: str) -> None:
    """Empty-but-valid datasets may report honest zero counts instead of an error."""
    check(f"{name}: explicit", (r.get("ok") is True) or bool(r.get("error")), str(r))


def err_result(r, name: str, hint: str = "") -> None:
    """r must be an explicit ok:false with a message."""
    check(f"{name}: explicit error", isinstance(r, dict) and r.get("ok") is False
          and isinstance(r.get("error"), str) and bool(r.get("error")),
          f"{hint} got {r}")

RES = {}   # accumulated results dumped at exit


def make_tabular(root: Path, name: str, body: str, subdir: str = ".") -> str:
    d = root / subdir
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(body, encoding="utf-8")
    return str(p)


def builds() -> dict[str, str]:
    root = Path(tempfile.mkdtemp(prefix="spinoml-dset-"))
    f = {}

    # ── tabular ──────────────────────────────────────────────────────────
    valid_rows = "\n".join(f"a{i},{b},CCO,{t}" for i, b, t in
                           [(i, 2 * i, 99.5) for i in range(1, 51)])
    f["tabular_valid"] = make_tabular(root, "valid.csv",
                                      f"a,b,mol,label\n{valid_rows}\n", subdir="t/tab")
    f["tabular_empty"] = make_tabular(root, "empty.csv", "a,b\n", subdir="t/tab")
    f["tabular_single"] = make_tabular(root, "single.csv", "a,b\n1,2\n", subdir="t/tab")
    f["tabular_large"] = make_tabular(root, "large.csv",
                                      "a,b\n" + "\n".join(f"{i},{i * 3}" for i in range(10_000)) + "\n",
                                      subdir="t/tab")
    f["tabular_nan"] = make_tabular(root, "nan.csv", "a,b\nnan,2\n,3\n", subdir="t/tab")
    f["tabular_inf"] = make_tabular(root, "inf.csv", "a,b\ninf,2\n3,4\n", subdir="t/tab")
    f["tabular_text"] = make_tabular(root, "text.csv", "x,y\nfoo,bar\nbaz,qux\n", subdir="t/tab")
    f["tabular_corrupt"] = make_tabular(root, "corrupt.csv", "\x00\x01\x02GARBAGE\xff", subdir="t/tab")
    f["tabular_unicode"] = make_tabular(root, "mölecules ünïcode.csv", "a,b\n1,2\n3,4\n", subdir="t/ünï")
    f["tabular_missing"] = str(root / "t/tab/does-not-exist.csv")

    # ── image folder ─────────────────────────────────────────────────────
    from PIL import Image
    import numpy as np
    img_dir = root / "t/img" / "spaces and ünïcode"
    img_dir.mkdir(parents=True)
    for cls, hue in (("class0", 200), ("class1", 60)):
        cdir = img_dir / cls
        cdir.mkdir()
        arr = np.full((8, 8, 3), hue, dtype=np.uint8)
        Image.fromarray(arr.astype("uint8")).save(cdir / "a b.png")
        Image.fromarray((arr * 2).clip(0, 255).astype("uint8")).save(cdir / "b.jpg")
    f["image_folder"] = str(img_dir)
    f["image_folder_empty"] = str(root / "t/imgempty")
    (root / "t/imgempty").mkdir()
    f["image_folder_missing"] = str(root / "t/no-such-dir")

    # ── tensor ───────────────────────────────────────────────────────────
    tdir = root / "t/ten"
    tdir.mkdir()
    torch.save(torch.randn(8, 4), tdir / "w w.pt")
    torch.save({"lr": 0.001, "dict": torch.randn(3, 3)}, tdir / "dict.pt")
    torch.save(Data(x=torch.randn(5, 3), edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)),
               tdir / "graph.pt")
    (tdir / "corrupt.pt").write_bytes(b"\x00\x01\x02not a torch archive\xff")
    import numpy as np
    np.save(tdir / "arr.npy", np.random.rand(6, 6))
    (tdir / "corrupt.npy").write_bytes(b"junk")
    f["tensor_pt"] = str(tdir / "w w.pt")
    f["tensor_dict"] = str(tdir / "dict.pt")
    f["tensor_graph"] = str(tdir / "graph.pt")
    f["tensor_corrupt"] = str(tdir / "corrupt.pt")
    f["tensor_npy"] = str(tdir / "arr.npy")
    f["tensor_npy_corrupt"] = str(tdir / "corrupt.npy")
    f["tensor_empty"] = str(tdir / "empty.pt")
    torch.save(torch.zeros(0), tdir / "empty.pt")

    # ── molecule ─────────────────────────────────────────────────────────
    mdir = root / "t/mol"
    mdir.mkdir()
    (mdir / "valid.smi").write_text("Cc1ccccc1\nCCO\nCCOC\n", encoding="utf-8")
    (mdir / "single.smi").write_text("CCO\n", encoding="utf-8")
    (mdir / "invalid.smi").write_text("not_a_molecule_string_x\n", encoding="utf-8")
    (mdir / "empty.smi").write_text("", encoding="utf-8")
    (mdir / "spaces.smi").write_text("C C O\n", encoding="utf-8")
    f["molecule"] = str(mdir / "valid.smi")
    f["molecule_single"] = str(mdir / "single.smi")
    f["molecule_invalid"] = str(mdir / "invalid.smi")
    f["molecule_empty"] = str(mdir / "empty.smi")
    f["molecule_spaces"] = str(mdir / "spaces.smi")

    # ── protein ──────────────────────────────────────────────────────────
    pdir = root / "t/pro"
    pdir.mkdir()
    atom = ("ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00  1.00           N\n"
            "ATOM      2  CA  ALA A   1       1.000   1.000   1.000  1.00  1.00           C\n"
            "ATOM      3  C   ALA A   1       2.000   1.000   1.000  1.00  1.00           C\n"
            "ATOM      4  O   ALA A   1       2.000   1.000   1.000  1.00  1.00           O\n")
    (pdir / "one.pdb").write_text(atom + "END\n", encoding="utf-8")
    (pdir / "corrupt.pdb").write_text("NOT A PDB AT ALL\njust, text!\n", encoding="utf-8")
    (pdir / "empty.pdb").write_text("", encoding="utf-8")
    f["protein"] = str(pdir / "one.pdb")
    f["protein_corrupt"] = str(pdir / "corrupt.pdb")
    f["protein_empty"] = str(pdir / "empty.pdb")

    # ── graph_folder ─────────────────────────────────────────────────────
    gdir = root / "t/graph"
    gdir.mkdir()
    torch.save(Data(x=torch.randn(4, 2), edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long)),
               gdir / "g0.pt")
    gdir_bad = root / "t/graphbad"
    gdir_bad.mkdir()
    (gdir_bad / "corrupt.pt").write_bytes(b"garbage")
    f["graph_folder"] = str(gdir)
    f["graph_folder_bad"] = str(gdir_bad)

    # ── pyg / hf / manifest ──────────────────────────────────────────────
    refdir = root / "t/ref"
    refdir.mkdir()
    (refdir / "karate.pyg").write_text("pyg:KarateClub\n", encoding="utf-8")
    (refdir / "imdb.hf").write_text("hf:imdb\n", encoding="utf-8")
    f["pyg"] = str(refdir / "karate.pyg")
    f["hf"] = str(refdir / "imdb.hf")

    mdir2 = root / "t/man"
    mdir2.mkdir()
    (mdir2 / "manifest.manifest").write_text(
        '{"table":"pairs.tsv","pairs":{"ligand":{"column":"smiles","kind":"molecule"}},'
        '"target":{"column":"affinity","type":"regression"}}', encoding="utf-8")
    (mdir2 / "pairs.tsv").write_text("smiles\taffinity\nCc1ccccc1\t1.5\nCCO\t2.5\n", encoding="utf-8")
    (mdir2 / "manifest_empty.manifest").write_text(
        '{"table":"empty.tsv","pairs":{"ligand":{"column":"smiles","kind":"molecule"}},'
        '"target":{"column":"affinity","type":"regression"}}', encoding="utf-8")
    (mdir2 / "empty.tsv").write_text("smiles\taffinity\n", encoding="utf-8")
    f["manifest"] = str(mdir2 / "manifest.manifest")
    f["manifest_empty"] = str(mdir2 / "manifest_empty.manifest")

    f["_root"] = str(root)
    return f


def main() -> int:
    F = builds()
    t0 = time.time()

    # ── detect_kind ──────────────────────────────────────────────────────
    expect = {
        "tabular_valid": "tabular", "tabular_corrupt": "tabular", "tensor_pt": "tensor",
        "tensor_npy": "tensor", "molecule": "molecule", "protein": "protein",
        "image_folder": "image_folder", "graph_folder": "graph_folder", "pyg": "pyg",
        "hf": "huggingface", "manifest": "manifest",
    }
    for key, kind in expect.items():
        check(f"detect_kind({key})", dh.detect_kind(F[key]) == kind,
              f"got {dh.detect_kind(F[key])} want {kind}")

    # ── tabular ──────────────────────────────────────────────────────────
    ok_result(dh.inspect(F["tabular_valid"]), "tab.inspect.valid", require_tensor=False)
    ok_result(dh.inspect(F["tabular_empty"]), "tab.inspect.empty", require_tensor=False)   # header-only: known shape, 0 rows
    err_result(dh.inspect(F["tabular_missing"]), "tab.inspect.missing")
    err_result(dh.sample_tensor(F["tabular_missing"]), "tab.sample.missing")
    err_result(dh.stats(F["tabular_missing"]), "tab.stats.missing")
    ok_result(dh.sample_tensor(F["tabular_valid"]), "tab.sample.valid")
    ok_result(dh.sample_tensor(F["tabular_valid"], [4, 2]), "tab.sample.pad")
    ok_result(dh.sample_tensor(F["tabular_nan"]), "tab.sample.nan")     # NaN → 0 fill, present
    err_result(dh.sample_tensor(F["tabular_inf"]), "tab.sample.inf")    # Inf must NOT pass through
    ok_result(dh.sample_tensor(F["tabular_single"], [4, 2]), "tab.sample.single")
    ok_result(dh.sample_tensor(F["tabular_large"]), "tab.sample.large")
    err_result(dh.sample_tensor(F["tabular_text"]), "tab.sample.textcol")  # all-object → explicit error
    err_result(dh.sample_tensor(F["tabular_corrupt"]), "tab.sample.corrupt")
    ok_result(dh.sample_tensor(F["tabular_unicode"]), "tab.sample.unicode")
    ok_result(dh.sample_tensor(F["tabular_valid"], options={"features": ["a", "b"]}),
              "tab.sample.features.ok")
    err_result(dh.sample_tensor(F["tabular_valid"], options={"features": ["a", "nope"]}),
              "tab.sample.features.missing")
    err_result(dh.sample_tensor(F["tabular_empty"], options={"features": ["a", "b"]}),
              "tab.sample.empty.features")   # REGRESSION: used to hang forever

    # ── stats (tabular) ──────────────────────────────────────────────────
    s = dh.stats(F["tabular_valid"])
    check("tab.stats.valid.rows", s.get("rows") == 50, f"got {s.get('rows')}")
    s = dh.stats(F["tabular_empty"])
    check("tab.stats.empty.explicit", s.get("ok") and s.get("rows") == 0, f"got {s}")

    # ── image folder ─────────────────────────────────────────────────────
    ok_result(dh.inspect(F["image_folder"]), "img.inspect.valid", require_tensor=False)
    ok_result(dh.stats(F["image_folder"]), "img.stats.valid", require_tensor=False)
    ok_result(dh.sample_tensor(F["image_folder"]), "img.sample.valid")
    err_result(dh.sample_tensor(F["image_folder_empty"]), "img.sample.empty")
    either_counted_or_err(dh.inspect(F["image_folder_empty"]), "img.inspect.empty")
    either_counted_or_err(dh.stats(F["image_folder_empty"]), "img.stats.empty")
    err_result(dh.inspect(F["image_folder_missing"]), "img.inspect.missing")
    ok_result(dh.sample_tensor(F["image_folder"], [2, 3, 16, 16]), "img.sample.shape")

    # ── tensor ───────────────────────────────────────────────────────────
    ok_result(dh.inspect(F["tensor_pt"]), "ten.inspect.pt", require_tensor=False)
    ok_result(dh.sample_tensor(F["tensor_pt"]), "ten.sample.pt")
    ok_result(dh.stats(F["tensor_pt"]), "ten.stats.pt", require_tensor=False)
    ok_result(dh.inspect(F["tensor_dict"]), "ten.inspect.dict", require_tensor=False)   # dict archive → expose value tensor
    ok_result(dh.inspect(F["tensor_graph"]), "ten.inspect.graph", require_tensor=False)
    ok_result(dh.sample_tensor(F["tensor_npy"]), "ten.sample.npy")
    err_result(dh.inspect(F["tensor_corrupt"]), "ten.inspect.corrupt")
    err_result(dh.sample_tensor(F["tensor_corrupt"]), "ten.sample.corrupt")
    err_result(dh.sample_tensor(F["tensor_npy_corrupt"]), "ten.sample.npy.corrupt")
    either_counted_or_err(dh.inspect(F["tensor_empty"]), "ten.inspect.empty")
    err_result(dh.sample_tensor(F["tensor_empty"]), "ten.sample.empty.graphless")

    # ── graph_folder ─────────────────────────────────────────────────────
    ok_result(dh.inspect(F["graph_folder"]), "graph.inspect.valid", require_tensor=False)
    ok_result(dh.sample_tensor(F["graph_folder"], options={"field": "x"}), "graph.sample.x")
    err_result(dh.sample_tensor(F["graph_folder_bad"], options={"field": "x"}), "graph.sample.corrupt")

    # ── molecule ─────────────────────────────────────────────────────────
    ok_result(dh.inspect(F["molecule"]), "mol.inspect.valid", require_tensor=False)
    ok_result(dh.sample_tensor(F["molecule"]), "mol.sample.valid")
    ok_result(dh.stats(F["molecule"]), "mol.stats.valid", require_tensor=False)
    ok_result(dh.sample_tensor(F["molecule_single"]), "mol.sample.single")
    err_result(dh.sample_tensor(F["molecule_empty"]), "mol.sample.empty")
    err_result(dh.sample_tensor(F["molecule_invalid"]), "mol.sample.invalid")
    ok_result(dh.sample_tensor(F["molecule_spaces"]), "mol.sample.spaces")  # token 'C' parses → ok

    # ── protein ──────────────────────────────────────────────────────────
    ok_result(dh.inspect(F["protein"]), "pro.inspect.valid", require_tensor=False)
    r = dh.sample_tensor(F["protein"], [1, 25])
    check("pro.sample.valid.either", r.get("ok") or (r.get("ok") is False and r.get("error")),
          str(r))
    either_counted_or_err(dh.inspect(F["protein_corrupt"]), "pro.inspect.corrupt")
    err_result(dh.sample_tensor(F["protein_empty"]), "pro.sample.empty")
    either_counted_or_err(dh.stats(F["protein_corrupt"]), "pro.stats.corrupt")   # stats is a 'limited' ok note

    # ── pyg ──────────────────────────────────────────────────────────────
    r = dh.inspect(F["pyg"])
    check("pyg.inspect.explicit", r.get("ok") is True or bool(r.get("error")), str(r))
    if r.get("ok"):
        ok_result(dh.sample_tensor(F["pyg"], options={"field": "x"}), "pyg.sample.x")
    else:
        check("pyg.sample.skips-on-missing-net", True, "pyg offline; inspect already explicit")

    # ── huggingface ──────────────────────────────────────────────────────
    r = dh.inspect(F["hf"])
    check("hf.inspect.explicit", r.get("ok") is False and bool(r.get("error")), str(r))   # datasets not installed → explicit missing-dep
    err_result(dh.sample_tensor(F["hf"]), "hf.sample")

    # ── manifest ─────────────────────────────────────────────────────────
    ok_result(dh.inspect(F["manifest"]), "man.inspect.valid", require_tensor=False)
    ok_result(dh.sample_tensor(F["manifest"], options={"field": "ligand.x"}), "man.sample.ligand")
    ok_result(dh.sample_tensor(F["manifest"], options={"field": "target"}), "man.sample.target")
    err_result(dh.sample_tensor(F["manifest"], options={"field": "nope"}), "man.sample.badfield")
    err_result(dh.sample_tensor(F["manifest_empty"]), "man.sample.empty")

    # ── relative path (chdir into fixture root, then expect explicit) ────
    base = Path(F["_root"]) / "t" / "tab"
    prev_cwd = os.getcwd()
    try:
        os.chdir(base)
        r = dh.inspect("does-not-exist.csv")
        err_result(r, "rel.missing")
        os.chdir(prev_cwd)
    finally:
        os.chdir(prev_cwd)

    # ── Phase 18: dataset fingerprinting ───────────────────────────────────
    # Every ok inspect result carries a SHA-256 content fingerprint; it is
    # deterministic, copy-stable, and changes when the data changes.
    import re as _re
    sha_hex = _re.compile(r"^[0-9a-f]{64}$")
    for key in ("tabular_valid", "tensor_pt", "molecule", "protein", "image_folder",
                "graph_folder", "pyg", "hf", "manifest"):
        r = dh.inspect(F[key])
        fp = r.get("fingerprint") if r.get("ok") else None
        if key in ("pyg", "hf"):
            # reference-mode datasets still get a file-based id when offline.
            either_counted_or_err(r, f"fp.{key}.inspect")
        if r.get("ok") and fp:
            check(f"fp.{key}.alg", fp.get("alg") == "sha256", f"got {fp.get('alg')}")
            check(f"fp.{key}.hash", isinstance(fp.get("hash"), str) and bool(sha_hex.match(fp["hash"])), str(fp)[:80])
            check(f"fp.{key}.size", isinstance(fp.get("size_bytes"), int) and fp["size_bytes"] > 0,
                  f"got {fp.get('size_bytes')}")
            check(f"fp.{key}.id", dh.inspect(F[key])["fingerprint"]["hash"] == fp["hash"], "nondeterministic")
    # deterministic across repeated inspect
    check("fp.deterministic", dh.inspect(F["tabular_valid"])["fingerprint"]["hash"]
          == dh.inspect(F["tabular_valid"])["fingerprint"]["hash"])
    # copy-stability: same bytes at a new path/name → same id
    import shutil
    tdir = F["tabular_valid"].rsplit("/", 1)[0]
    copy = os.path.join(str(F["_root"]), "t/tabcopy.csv")
    shutil.copy(F["tabular_valid"], copy)
    check("fp.copy-stable", dh.inspect(copy)["fingerprint"]["hash"]
          == dh.inspect(F["tabular_valid"])["fingerprint"]["hash"])
    # content change → id change
    import io as _io
    with open(copy, "a", encoding="utf-8") as fh:
        fh.write("a9,9,CCO,9.5\n")
    check("fp.content-change", dh.inspect(copy)["fingerprint"]["hash"]
          != dh.inspect(F["tabular_valid"])["fingerprint"]["hash"])
    # structure-mode folders carry n_files > 0
    img_fp = dh.inspect(F["image_folder"])["fingerprint"]
    check("fp.image.structure", img_fp.get("mode") == "structure" and (img_fp.get("n_files") or 0) > 0, str(img_fp)[:80])
    g_fp = dh.inspect(F["graph_folder"])["fingerprint"]
    check("fp.graph.structure", g_fp.get("mode") == "structure" and g_fp.get("n_files") == 1, str(g_fp)[:80])
    # manifest = descriptor + table (n_files 2)
    m_fp = dh.inspect(F["manifest"])["fingerprint"]
    check("fp.manifest.n_files", m_fp.get("mode") == "config+content" and m_fp.get("n_files") == 2, str(m_fp)[:80])

# ── structural invariants on ok results ──────────────────────────────
    for key in ("tabular_valid", "tensor_pt", "molecule", "image_folder"):
        r = dh.sample_tensor(F[key])
        if r.get("ok") and isinstance(r.get("tensor"), torch.Tensor):
            t = r["tensor"]
            check(f"inv.{key}.batch1", t.shape[0] == 1, f"shape {tuple(t.shape)}")
            check(f"inv.{key}.finite", bool(torch.isfinite(t.float()).all()), "non-finite leak!")
    r = dh.sample_tensor(F["graph_folder"], options={"field": "x"})
    if r.get("ok") and isinstance(r.get("tensor"), torch.Tensor):
        t = r["tensor"]
        check("inv.graph_folder.2d", t.ndim == 2, f"shape {tuple(t.shape)}")   # [nodes, feat], no batch
        check("inv.graph_folder.finite", bool(torch.isfinite(t.float()).all()), "non-finite leak!")

    dt = time.time() - t0
    print(f"dataset-reliability: {PASS} pass, {FAIL} fail, {dt:.1f}s")
    for f_ in FAILURES:
        print("  FAIL", f_)
    return 1 if FAIL else 0


if __name__ == "__main__":
    def _alarm(*_):
        print("dataset-reliability: TIMEOUT after 90s — a handler hung (wedged worker!)")
        sys.exit(2)
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(90)
    sys.exit(main())