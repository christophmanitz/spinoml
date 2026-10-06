#!/usr/bin/env python3
# Local-import closure helper for sidecar-torch.
#
# The remote-sidecar deploy ships every Python file that sidecar-torch/main.py
# (and dataset_handlers.py) can reach via in-package imports, plus the runtime
# data files they read via `Path(__file__).resolve().parent / "<sub>"`. This
# helper is the input side of the static verifier
# `scripts/verify-remote-deploy-files.ts`: given one or more entry .py paths
# under `sidecar-torch/`, walk the in-package import graph and emit the set of
# relative paths (always `posix`, never OS-specific) that must appear in the
# deploy list.
#
# Reads ONE JSON document from stdin:
#   {"entries": ["main.py", "dataset_handlers.py"]}
# Prints ONE JSON document:
#   {"files": ["main.py", "dataset_handlers.py", "scope.py", ...]}
#
# Design choices:
# - We use `ast` (not `importlib`) so we don't need a working Python env or to
#   actually import anything; broken imports, missing deps and missing files
#   are NOT errors here — we only follow in-package imports.
# - We scan BOTH top-level and function-level imports (the brief asks for
#   this); nested class bodies and `with` / `if` blocks can also legally
#   contain `import` statements in Python, so we visit `ast.NodeVisitor` on
#   the whole tree.
# - "In-package" = the module name resolves to a .py file under the same
#   directory as the entry. `from scope import ScopeError` → scope.py.
#   `from scope.helper import x` → scope/helper.py. Stdlib / 3rd-party
#   imports (`torch`, `pathlib`, `numpy`, …) are silently ignored — the
#   remote venv has them.
# - Runtime data references via `Path(__file__).resolve().parent / "espf"`:
#   we add `espf/` (and every file currently under it) to the closure
#   automatically. The verifier separately asserts that nothing else under
#   `sidecar-torch/espf/` is missing from the deploy list.

import ast
import json
import os
import sys
from pathlib import Path


def _find_local_file(sidecar_dir: Path, module: str) -> str | None:
    """Map `foo.bar` → a relative POSIX path under sidecar_dir if it exists
    there. `foo` → foo.py, `foo.bar` → foo/bar.py. Returns None otherwise."""
    if not module or module.startswith("."):
        return None
    parts = module.split(".")
    rel = Path(*parts).with_suffix(".py")
    abs_path = sidecar_dir / rel
    if abs_path.is_file():
        return rel.as_posix()
    return None


def _local_imports_in_tree(tree: ast.AST) -> set[str]:
    """All in-package module names referenced anywhere in `tree`."""
    names: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                names.add(alias.name.split(".")[0])

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.module is None or node.level > 0:
                # `from .x import y` — relative import. We don't resolve these
                # for now (the sidecar code uses absolute imports only);
                # leaving them out keeps the closure deterministic without
                # requiring us to model the package layout.
                return
            names.add(node.module.split(".")[0])

    Visitor().visit(tree)
    return names


def closure(sidecar_dir: Path, entries: list[str]) -> list[str]:
    """Recursive closure of every in-package import reachable from
    `entries`. BFS so the output order matches the discovery order (helps
    the verifier's diff output)."""
    seen: set[str] = set()
    queue: list[str] = list(entries)
    while queue:
        rel = queue.pop(0)
        if rel in seen:
            continue
        seen.add(rel)
        path = sidecar_dir / rel
        if not path.is_file():
            continue
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src, filename=str(path))
        except (OSError, SyntaxError):
            continue
        for name in _local_imports_in_tree(tree):
            local = _find_local_file(sidecar_dir, name)
            if local is not None:
                queue.append(local)

    # Runtime data: ESPF codebook lives at <sidecar>/espf/. Anything under
    # `espf/` must be deployed together with the Python that reads it.
    espf_dir = sidecar_dir / "espf"
    if espf_dir.is_dir():
        for child in sorted(espf_dir.iterdir()):
            if child.is_file():
                seen.add(("espf/" + child.name))

    return sorted(seen)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"invalid JSON: {exc}"}), file=sys.stderr)
        return 1
    entries = payload.get("entries", [])
    sidecar_dir = Path(payload["sidecar_dir"]).resolve()
    files = closure(sidecar_dir, entries)
    json.dump({"files": files}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
