"""Tests for run_workspace_script's relpath validation / command construction."""

import os
import shutil
import sys
import tempfile

# Add sidecar-torch to sys.path so we can import main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../sidecar-torch")))

import main  # noqa: E402


class FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


CALLS = []


def fake_run(*args, **kwargs):
    CALLS.append((args, kwargs))
    return FakeCompleted()


main.subprocess.run = fake_run


def files_under(root):
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            found.append(os.path.join(dirpath, fn))
    return found


def main_test():
    failed = 0
    total = 0

    def check(cond, label):
        nonlocal failed, total
        total += 1
        if cond:
            print(f"✓ {label}")
        else:
            print(f"✗ {label}")
            failed += 1

    # --- Accepted cases: (relpath, mode, expected argv) ---
    accepted = [
        ("agent/a.py", "shell", [sys.executable, "-u", "./agent/a.py"]),
        ("agent/a.sh", "shell", ["bash", "./agent/a.sh"]),
        ("agent/j.sbatch", "slurm", ["sbatch", "./agent/j.sbatch"]),
        ("./agent/b.py", "shell", [sys.executable, "-u", "./agent/b.py"]),
        ("agent/Prüfung 1.py", "shell", [sys.executable, "-u", "./agent/Prüfung 1.py"]),
    ]
    for rel, mode, expected in accepted:
        root = tempfile.mkdtemp()
        try:
            CALLS.clear()
            res = main.run_workspace_script({"root": root, "relpath": rel, "code": "print('hi')", "mode": mode})
            check(res.get("ok") is True, f"accepted {rel!r} -> ok=True")
            check(len(CALLS) == 1, f"accepted {rel!r} -> subprocess.run called once")
            if CALLS:
                argv = CALLS[0][0][0]
                check(list(argv) == list(expected), f"accepted {rel!r} -> argv {expected}")
            else:
                check(False, f"accepted {rel!r} -> argv (no call recorded)")
            norm = rel[2:] if rel.startswith("./") else rel
            check(os.path.isfile(os.path.join(root, norm)), f"accepted {rel!r} -> file written")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    # --- Rejected cases ---
    rejected = [
        "x; touch PWN",
        "--wrap=touch PWN",
        "-c.sh",
        "-x/y.sh",
        "a/../b.py",
        "../b.py",
        "a\nb.py",
        "a$(id).sh",
        "a`id`.sh",
        "noext",
        "agent/x.txt",
        "",
        "a|b.sh",
        "a&b.sh",
        "a'b.sh",
        'a"b.sh',
        "a>b.sh",
        "a*.sh",
        "agent//x.py",
        "a.PY",
        "a" * 296 + ".sh",
    ]
    for rel in rejected:
        root = tempfile.mkdtemp()
        try:
            CALLS.clear()
            res = main.run_workspace_script({"root": root, "relpath": rel, "code": "print('hi')", "mode": "shell"})
            check(res.get("ok") is False, f"rejected {rel!r} -> ok=False")
            check(res.get("error_code") == "INVALID_RELPATH", f"rejected {rel!r} -> INVALID_RELPATH")
            check(len(CALLS) == 0, f"rejected {rel!r} -> subprocess.run not called")
            created = files_under(root)
            check(created == [], f"rejected {rel!r} -> no file created")
            check(not any(os.path.basename(p) == "PWN" for p in created), f"rejected {rel!r} -> no PWN sentinel")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    if failed == 0:
        print(f"\n✓ {total}/{total} checks passed")
        sys.exit(0)
    else:
        print(f"\n✗ {failed}/{total} checks failed")
        sys.exit(1)


if __name__ == "__main__":
    main_test()
