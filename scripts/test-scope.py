#!/usr/bin/env python3
"""Phase 46 — torch-sidecar filesystem scope test matrix + real-HTTP test.

A. check_path matrix (injected config via set_scope_for_tests)
B. modes / configuration (env + scope.json)
C. dataset_handlers integration (enforced scope on a temp workspace)
D. real HTTP: the sidecar subprocess enforces 403 on out-of-scope paths

Run:  conda run --no-capture-output -n mlforge-dev npm run test:scope
Exit code 1 on any failure. Fresh temp dirs; every spawned sidecar is
terminated in a finally and waited on.
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "sidecar-torch"))

import dataset_handlers as dh  # noqa: E402
import scope  # noqa: E402
from scope import ScopeError  # noqa: E402

PASS = 0
FAIL = 0
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  \u2713 {name}")
    else:
        FAIL += 1
        FAILURES.append(f"{name} {detail}".strip())
        print(f"  \u2717 {name}{(' \u2014 ' + detail) if detail else ''}")


def expect_scope(name: str, code, fn) -> None:
    """`fn()` must raise ScopeError with the given code (str or tuple of str)."""
    want = code if isinstance(code, tuple) else (code,)
    try:
        fn()
    except ScopeError as e:
        check(name, e.code in want, f"code={e.code} want {want}")
    except Exception as e:  # noqa: BLE001
        check(name, False, f"unexpected {type(e).__name__}: {e}")
    else:
        check(name, False, "did not raise")


def _png(path: Path, color=(200, 40, 40)) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


# ─────────────────────────────────────────────────────────────────────────
def section_a() -> Path:
    print("\n\u2014 A. check_path matrix \u2014")
    tmp = Path(tempfile.mkdtemp(prefix="scope-A-"))
    ws = tmp / "ws"
    outside = tmp / "outside"
    other = tmp / "other"
    for d in (ws, outside, other):
        d.mkdir()
    (ws / "inside.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (ws / "with space.csv").write_text("a\n1\n", encoding="utf-8")
    (ws / "Pr\u00fcfung \u2713").mkdir()
    (ws / "Pr\u00fcfung \u2713" / "\u30c7\u30fc\u30bf.csv").write_text("a\n1\n", encoding="utf-8")
    (ws / "existing_dir").mkdir()
    (outside / "secret.csv").write_text("x,y\n1,2\n", encoding="utf-8")

    os.symlink(ws / "inside.csv", ws / "link_in")
    os.symlink(outside / "secret.csv", ws / "link_out")
    os.symlink(outside, ws / "linkdir_out")
    os.symlink(str(outside / "nope.csv"), ws / "dangling")
    os.symlink(ws / "inside.csv", ws / "chain_b")
    os.symlink(ws / "chain_b", ws / "chain_a")
    os.symlink(ws / "loop2", ws / "loop1")
    os.symlink(ws / "loop1", ws / "loop2")
    os.symlink(ws, tmp / "ws_as_link")

    # sibling-prefix fixture
    sib = tmp / "sib"
    sib.mkdir()
    (sib / "ws").mkdir()
    (sib / "ws-evil").mkdir()

    scope.set_scope_for_tests(env={"SPINOML_ALLOWED_ROOTS": str(ws)})
    wsr = str(ws.resolve())

    r = scope.check_path(str(ws / "inside.csv"))
    check("A inside root ok", r == os.path.realpath(ws / "inside.csv"), r)
    check("A spaces path ok",
          scope.check_path(str(ws / "with space.csv")) == os.path.realpath(ws / "with space.csv"))
    check("A unicode path ok",
          scope.check_path(str(ws / "Pr\u00fcfung \u2713" / "\u30c7\u30fc\u30bf.csv"))
          == os.path.realpath(ws / "Pr\u00fcfung \u2713" / "\u30c7\u30fc\u30bf.csv"))
    expect_scope("A relative -> PATH_INVALID", "PATH_INVALID", lambda: scope.check_path("inside.csv"))
    expect_scope("A /root/a/../../etc/passwd -> SCOPE_DENIED", "SCOPE_DENIED",
                 lambda: scope.check_path("/root/a/../../etc/passwd"))
    expect_scope("A /etc/passwd -> SCOPE_DENIED", "SCOPE_DENIED", lambda: scope.check_path("/etc/passwd"))
    expect_scope("A /proc/self/cwd/... -> denied", "SCOPE_DENIED",
                 lambda: scope.check_path("/proc/self/cwd/etc/passwd"))
    expect_scope("A /proc/self/root/etc/passwd -> denied", "SCOPE_DENIED",
                 lambda: scope.check_path("/proc/self/root/etc/passwd"))
    expect_scope("A NUL -> PATH_INVALID", "PATH_INVALID", lambda: scope.check_path(str(ws / "a\x00b.csv")))
    expect_scope("A len>4096 -> PATH_INVALID", "PATH_INVALID",
                 lambda: scope.check_path("/" + "a" * 4096))
    expect_scope("A non-str -> PATH_INVALID", "PATH_INVALID", lambda: scope.check_path(1234))  # type: ignore[arg-type]

    check("A missing file under existing dir write ok",
          scope.check_path(str(ws / "existing_dir" / "new.csv"), write=True)
          == os.path.realpath(ws / "existing_dir" / "new.csv"))
    check("A missing parent chain ok",
          scope.check_path(str(ws / "n1" / "n2" / "new.csv"), write=True)
          == os.path.realpath(ws / "n1" / "n2" / "new.csv"))
    check("A symlink in-root->in-root ok",
          scope.check_path(str(ws / "link_in")) == os.path.realpath(ws / "inside.csv"))
    check("A link chain ok",
          scope.check_path(str(ws / "chain_a")) == os.path.realpath(ws / "inside.csv"))
    expect_scope("A symlink out (read) -> PATH_SYMLINK_OUTSIDE", "PATH_SYMLINK_OUTSIDE",
                 lambda: scope.check_path(str(ws / "link_out")))
    expect_scope("A symlink out (write) -> PATH_SYMLINK_OUTSIDE", "PATH_SYMLINK_OUTSIDE",
                 lambda: scope.check_path(str(ws / "link_out"), write=True))
    expect_scope("A dangling link out (write) -> PATH_SYMLINK_OUTSIDE", "PATH_SYMLINK_OUTSIDE",
                 lambda: scope.check_path(str(ws / "dangling"), write=True))
    expect_scope("A symlink loop -> PATH_INVALID", "PATH_INVALID",
                 lambda: scope.check_path(str(ws / "loop1")))

    # sibling prefix: root /sib/ws must NOT contain /sib/ws-evil
    scope.set_scope_for_tests(env={"SPINOML_ALLOWED_ROOTS": str(sib / "ws")})
    expect_scope("A sibling prefix /ws-evil -> SCOPE_DENIED", "SCOPE_DENIED",
                 lambda: scope.check_path(str(sib / "ws-evil" / "x.csv")))

    # allowed via a configured symlink target
    scope.set_scope_for_tests(env={"SPINOML_ALLOWED_ROOTS": str(ws),
                                   "SPINOML_SYMLINK_TARGETS": str(outside)})
    check("A symlink out allowed via symlink target",
          scope.check_path(str(ws / "link_out")) == os.path.realpath(outside / "secret.csv"))
    # a different outside dir is still rejected even with a target configured
    scope.set_scope_for_tests(env={"SPINOML_ALLOWED_ROOTS": str(ws),
                                   "SPINOML_SYMLINK_TARGETS": str(other)})
    expect_scope("A different outside dir rejected", "PATH_SYMLINK_OUTSIDE",
                 lambda: scope.check_path(str(ws / "link_out")))

    # root passed as a symlink works
    scope.set_scope_for_tests(env={"SPINOML_ALLOWED_ROOTS": str(tmp / "ws_as_link")})
    check("A root passed as symlink works",
          scope.check_path(str(ws / "inside.csv")) == os.path.realpath(ws / "inside.csv"))
    check("A root-as-symlink resolved to its target", wsr == os.path.realpath(ws))
    return tmp


# ─────────────────────────────────────────────────────────────────────────
def section_b() -> Path:
    print("\n\u2014 B. modes / configuration \u2014")
    tmp = Path(tempfile.mkdtemp(prefix="scope-B-"))
    ws1 = tmp / "ws1"
    ws2 = tmp / "ws2"
    home = tmp / "home"
    for d in (ws1, ws2, home):
        d.mkdir()
    (ws1 / "a.csv").write_text("a\n1\n", encoding="utf-8")
    (ws2 / "b.csv").write_text("b\n2\n", encoding="utf-8")
    arbitrary = tmp / "arbitrary.csv"
    arbitrary.write_text("z\n9\n", encoding="utf-8")

    # unconfigured-open: no restriction, warning emitted once
    scope.set_scope_for_tests(env={}, home=str(home))
    st = scope.scope_status()
    check("B unconfigured-open mode", st["mode"] == "unconfigured-open", str(st))
    check("B unconfigured-open root_count 0", st["root_count"] == 0)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        scope.check_path(str(arbitrary))
        scope.check_path(str(arbitrary))
    warnings = [ln for ln in err.getvalue().splitlines() if "WARNING" in ln]
    check("B unconfigured-open allows paths outside any root", True)
    check("B unconfigured-open warns exactly once", len(warnings) == 1, f"{len(warnings)} lines")

    # strict + nothing -> unconfigured-closed
    scope.set_scope_for_tests(env={"SPINOML_REQUIRE_SCOPE": "1"}, home=str(home))
    check("B strict+nothing mode", scope.scope_status()["mode"] == "unconfigured-closed")
    expect_scope("B strict+nothing -> SCOPE_UNCONFIGURED", "SCOPE_UNCONFIGURED",
                 lambda: scope.check_path(str(arbitrary)))

    # env roots with two entries
    scope.set_scope_for_tests(env={"SPINOML_ALLOWED_ROOTS": str(ws1) + os.pathsep + str(ws2)})
    st = scope.scope_status()
    check("B env two roots mode enforced", st["mode"] == "enforced", str(st))
    check("B env two roots count 2", st["root_count"] == 2, str(st))
    check("B env root1 path ok", scope.check_path(str(ws1 / "a.csv")) == os.path.realpath(ws1 / "a.csv"))
    check("B env root2 path ok", scope.check_path(str(ws2 / "b.csv")) == os.path.realpath(ws2 / "b.csv"))

    # all-invalid entries -> enforced with 0 valid roots -> everything denied
    scope.set_scope_for_tests(env={"SPINOML_ALLOWED_ROOTS":
                                   "rel" + os.pathsep + "/does-not-exist-xyz" + os.pathsep + "/"})
    st = scope.scope_status()
    check("B all-invalid mode still enforced", st["mode"] == "enforced", str(st))
    check("B all-invalid root_count 0", st["root_count"] == 0, str(st))
    check("B all-invalid recorded load_error", bool(st["load_error"]), str(st))
    expect_scope("B all-invalid denies everything", "SCOPE_DENIED",
                 lambda: scope.check_path(str(ws1 / "a.csv")))

    # env symlink targets
    scope.set_scope_for_tests(env={"SPINOML_ALLOWED_ROOTS": str(ws1),
                                   "SPINOML_SYMLINK_TARGETS": str(ws2)})
    check("B env symlink_target count 1", scope.scope_status()["symlink_target_count"] == 1)

    # scope.json valid
    scope_dir = home / ".cache" / "spinoml"
    scope_dir.mkdir(parents=True)
    sf = scope_dir / "scope.json"
    sf.write_text(json.dumps({"version": 1, "roots": [str(ws1)], "symlink_targets": []}), encoding="utf-8")
    os.chmod(sf, 0o600)
    scope.set_scope_for_tests(env={}, home=str(home), uid=os.getuid())
    st = scope.scope_status()
    check("B scope.json valid mode enforced", st["mode"] == "enforced", str(st))
    check("B scope.json source file", st["source"] == "file", str(st))
    check("B scope.json root usable", scope.check_path(str(ws1 / "a.csv")) == os.path.realpath(ws1 / "a.csv"))

    # scope.json group/world writable -> ignored + load_error
    os.chmod(sf, 0o666)
    scope.set_scope_for_tests(env={}, home=str(home), uid=os.getuid())
    st = scope.scope_status()
    check("B scope.json 0666 ignored (unconfigured-open)", st["mode"] == "unconfigured-open", str(st))
    check("B scope.json 0666 recorded load_error", bool(st["load_error"]), str(st))

    # scope.json wrong version -> ignored
    os.chmod(sf, 0o600)
    sf.write_text(json.dumps({"version": 2, "roots": [str(ws1)]}), encoding="utf-8")
    scope.set_scope_for_tests(env={}, home=str(home), uid=os.getuid())
    st = scope.scope_status()
    check("B scope.json wrong version ignored", st["mode"] == "unconfigured-open", str(st))
    check("B scope.json wrong version load_error", bool(st["load_error"]), str(st))

    # scope.json change picked up without restart (mtime_ns)
    sf.write_text(json.dumps({"version": 1, "roots": [str(ws1)]}), encoding="utf-8")
    scope.set_scope_for_tests(env={}, home=str(home), uid=os.getuid())
    check("B scope.json change before count 1", scope.scope_status()["root_count"] == 1)
    st0 = os.stat(sf)
    sf.write_text(json.dumps({"version": 1, "roots": [str(ws1), str(ws2)]}), encoding="utf-8")
    os.utime(sf, ns=(st0.st_mtime_ns + 10 ** 9, st0.st_mtime_ns + 10 ** 9))
    check("B scope.json change after count 2 (no restart)", scope.scope_status()["root_count"] == 2)

    # scope_status never contains a path
    scope.set_scope_for_tests(env={"SPINOML_ALLOWED_ROOTS": str(ws1),
                                   "SPINOML_SYMLINK_TARGETS": str(ws2)})
    blob = json.dumps(scope.scope_status())
    check("B scope_status hides root path", str(ws1) not in blob, blob)
    check("B scope_status hides symlink-target path", str(ws2) not in blob, blob)
    check("B scope_status keys exact",
          set(scope.scope_status().keys()) == {"mode", "source", "root_count",
                                               "symlink_target_count", "load_error"})
    return tmp


# ─────────────────────────────────────────────────────────────────────────
def section_c() -> Path:
    print("\n\u2014 C. dataset_handlers integration \u2014")
    tmp = Path(tempfile.mkdtemp(prefix="scope-C-"))
    ws = tmp / "ws"
    ws.mkdir()
    marker = "PHASE46_MARKER_ZZTOP_UNIQUE"
    outside = tmp / "outside.csv"
    outside.write_text(f"{marker},7,8\n", encoding="utf-8")
    inside = ws / "data.csv"
    inside.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    scope.set_scope_for_tests(env={"SPINOML_ALLOWED_ROOTS": str(ws)})

    check("C inspect inside ok", dh.inspect(str(inside)).get("ok") is True)
    check("C stats inside ok", dh.stats(str(inside)).get("ok") is True)
    check("C sample inside ok", dh.sample_tensor(str(inside)).get("ok") is True)

    r = dh.inspect(str(outside))
    check("C inspect outside -> SCOPE_DENIED", r.get("error_code") == "SCOPE_DENIED", str(r))
    check("C inspect outside leaks no content", marker not in json.dumps(r))
    r = dh.stats(str(outside))
    check("C stats outside -> SCOPE_DENIED", r.get("error_code") == "SCOPE_DENIED", str(r))
    check("C stats outside leaks no content", marker not in json.dumps(r))
    r = dh.sample_tensor(str(outside))
    check("C sample outside -> SCOPE_DENIED", r.get("error_code") == "SCOPE_DENIED", str(r))
    check("C sample outside leaks no content", marker not in json.dumps(r))

    os.symlink(outside, ws / "link.csv")
    r = dh.inspect(str(ws / "link.csv"))
    check("C symlink escape -> PATH_SYMLINK_OUTSIDE", r.get("error_code") == "PATH_SYMLINK_OUTSIDE", str(r))
    check("C symlink escape leaks no content", marker not in json.dumps(r))

    # manifest branch sources that escape
    def write_manifest(name: str, cell: str) -> Path:
        mdir = ws / name
        mdir.mkdir()
        (mdir / "pairs.csv").write_text(f"path,y\n{cell},1.0\n", encoding="utf-8")
        (mdir / "m.manifest").write_text(json.dumps(
            {"table": "pairs.csv", "pairs": {"lig": {"column": "path"}},
             "target": {"column": "y", "type": "regression"}}), encoding="utf-8")
        return mdir / "m.manifest"

    r = dh.inspect(str(write_manifest("abs", "/etc/passwd")))
    check("C manifest /etc/passwd -> SCOPE_*", r.get("error_code", "").startswith("SCOPE_"), str(r))
    r = dh.inspect(str(write_manifest("rel", "../../outside.csv")))
    check("C manifest ../outside.csv -> SCOPE_*", r.get("error_code", "").startswith("SCOPE_"), str(r))
    r = dh.inspect(str(write_manifest("aabs", str(outside))))
    check("C manifest absolute outside -> SCOPE_*", r.get("error_code", "").startswith("SCOPE_"), str(r))
    check("C manifest escape leaks no content", marker not in json.dumps(r))

    # image folder with a symlink to an outside image: never thumbnailed
    outside_img = tmp / "outside_img.png"
    _png(outside_img, (10, 240, 10))
    img_dir = ws / "img"
    (img_dir / "class1").mkdir(parents=True)
    _png(img_dir / "class1" / "good.png", (240, 10, 10))
    (img_dir / "class0").mkdir()
    os.symlink(outside_img, img_dir / "class0" / "evil.png")
    r = dh.inspect(str(img_dir))
    thumb_names = [t.get("name") for t in r.get("thumbnails", [])]
    check("C image folder inspect ok", r.get("ok") is True, str(r)[:120])
    check("C outside image not thumbnailed", "class0/evil.png" not in thumb_names, str(thumb_names))
    check("C inside image still thumbnailed", "class1/good.png" in thumb_names, str(thumb_names))

    # cache dir write outside denied
    scope.set_scope_for_tests(env={"SPINOML_ALLOWED_ROOTS": str(ws)})
    cache_out = tmp / "cache_out"
    expect_scope("C ensure_espf_cache outside -> ScopeError", "SCOPE_DENIED",
                 lambda: dh.ensure_espf_cache(cache_out))
    check("C outside cache dir not created", not (cache_out / ".espf").exists())
    expect_scope("C _cached_mol_data outside cache -> ScopeError", "SCOPE_DENIED",
                 lambda: dh._cached_mol_data("CCO", tmp / "cache_out2"))
    return tmp


# ─────────────────────────────────────────────────────────────────────────
VALID_LINEAR = "import torch\nimport torch.nn as nn\nclass Model(nn.Module):\n" \
               "    def __init__(self):\n        super().__init__()\n        self.fc = nn.Linear(8, 4)\n" \
               "    def forward(self, x):\n        return self.fc(x)\n"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_sidecar(env_extra: dict, port: int, home: Path) -> subprocess.Popen:
    home.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SPINOML_")}
    env.pop("XDG_RUNTIME_DIR", None)
    env["HOME"] = str(home)
    # Keep the real user-site importable even though HOME points at a temp dir
    # (in this dev env pandas/PIL live in ~/.local/lib/python3.x/site-packages).
    env.setdefault("PYTHONUSERBASE", os.path.expanduser("~/.local"))
    env["SPINOML_TORCH_PORT"] = str(port)
    env.update(env_extra)
    proc = subprocess.Popen(
        [sys.executable, "main.py"], cwd=str(ROOT / "sidecar-torch"), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 40
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("sidecar exited during startup")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.5)
            return proc
        except Exception:  # noqa: BLE001
            time.sleep(0.15)
    proc.terminate()
    raise RuntimeError("sidecar did not become healthy")


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _post(port: int, path: str, body: dict):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=20)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(port: int, path: str):
    try:
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10)
        return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def section_d() -> None:
    print("\n\u2014 D. real HTTP \u2014")
    tmp = Path(tempfile.mkdtemp(prefix="scope-D-"))
    ws = tmp / "ws"
    ws.mkdir()
    marker = "PHASE46_HTTP_MARKER_UNIQUE"
    inside = ws / "data.csv"
    inside.write_text("a,b\n1,2\n", encoding="utf-8")
    outside = tmp / "outside.csv"
    outside.write_text(f"{marker},7,8\n", encoding="utf-8")
    outside_dir = tmp / "outside_dir"
    outside_dir.mkdir()

    # (1) enforced sidecar
    port = _free_port()
    proc = _start_sidecar({"SPINOML_ALLOWED_ROOTS": str(ws)}, port, tmp / "home1")
    try:
        status, body = _get(port, "/health")
        sc = body.get("scope", {})
        check("D /health scope.mode enforced", sc.get("mode") == "enforced", str(sc))
        check("D /health has no path string", str(ws) not in json.dumps(body), json.dumps(sc))

        status, body = _post(port, "/dataset/inspect", {"abspath": str(inside)})
        check("D inspect inside -> 200 ok", status == 200 and body.get("ok") is True,
              f"status={status} body={json.dumps(body)[:300]}")
        check("D inspect inside has content", "a" in body.get("columns", []), json.dumps(body)[:200])

        status, body = _post(port, "/dataset/inspect", {"abspath": str(outside)})
        check("D inspect outside -> 403", status == 403, f"status={status}")
        check("D inspect outside -> SCOPE_* code",
              str(body.get("error_code", "")).startswith("SCOPE_"), str(body))
        check("D inspect outside leaks no content", marker not in json.dumps(body))

        status, body = _post(port, "/run_script",
                             {"root": str(outside_dir), "relpath": "x.py",
                              "code": "print('pwn')", "mode": "shell"})
        check("D run_script outside root -> 403", status == 403, f"status={status}")
        check("D run_script outside writes nothing", not (outside_dir / "x.py").exists())

        status, body = _post(port, "/run_script",
                             {"root": str(ws), "relpath": "a/../b.py",
                              "code": "print('pwn')", "mode": "shell"})
        check("D run_script hostile relpath -> INVALID_RELPATH",
              status == 200 and body.get("error_code") == "INVALID_RELPATH",
              f"status={status} body={body}")
        check("D run_script hostile writes nothing", not (ws / "b.py").exists())

        status, body = _post(port, "/activations",
                             {"code": VALID_LINEAR, "input_shapes": [[1, 8]],
                              "checkpoint": str(tmp / "outside.pt")})
        check("D activations outside checkpoint -> 403", status == 403, f"status={status}")
        check("D activations checkpoint SCOPE_* code",
              str(body.get("error_code", "")).startswith("SCOPE_"), str(body))
    finally:
        _stop(proc)

    # (2) no scope configured -> unconfigured-open
    port = _free_port()
    proc = _start_sidecar({}, port, tmp / "home2")
    try:
        status, body = _get(port, "/health")
        check("D no-scope /health unconfigured-open",
              body.get("scope", {}).get("mode") == "unconfigured-open", str(body.get("scope")))
    finally:
        _stop(proc)

    # (3) strict with no scope -> unconfigured-closed, inspect 403
    port = _free_port()
    proc = _start_sidecar({"SPINOML_REQUIRE_SCOPE": "1"}, port, tmp / "home3")
    try:
        status, body = _get(port, "/health")
        check("D strict /health unconfigured-closed",
              body.get("scope", {}).get("mode") == "unconfigured-closed", str(body.get("scope")))
        status, body = _post(port, "/dataset/inspect", {"abspath": str(inside)})
        check("D strict inspect -> 403 SCOPE_UNCONFIGURED",
              status == 403 and body.get("error_code") == "SCOPE_UNCONFIGURED",
              f"status={status} body={body}")
    finally:
        _stop(proc)


# ─────────────────────────────────────────────────────────────────────────
def main() -> int:
    try:
        section_a()
        section_b()
        section_c()
        section_d()
    finally:
        scope.set_scope_for_tests()
    print(f"\nscope-matrix: {PASS} pass, {FAIL} fail")
    for f in FAILURES:
        print("  FAIL", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
