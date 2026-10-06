#!/usr/bin/env python3
"""Phase 50 — silent-exception guard for the Python sidecar + trainer.

Stops new "swallowing" error handlers from appearing without a written
justification. A handler is a swallow when its body has no statement other than
``pass`` / ``...`` / a docstring / ``continue`` / ``break`` / ``return`` /
``return <literal>``, or a ``contextlib.suppress(...)`` context manager.

Every such site must carry a comment of >= 15 characters of prose that is not
merely "ignore"/"noop"/... AND must be listed in
docs/engineering/SILENT_EXCEPTIONS.md (the ``## Python allow-list`` section).
Matching is by file + normalised pattern text, so line-number drift is
tolerated; a pattern occurring N times in a file must be listed N times.

Usage:
    python scripts/verify-silent-except-py.py            # guard (exit 1 on violation)
    python scripts/verify-silent-except-py.py --all      # list every detected site
    python scripts/verify-silent-except-py.py --tsv      # machine-readable rows
    python scripts/verify-silent-except-py.py --hidden   # list single-default assignments
    python scripts/verify-silent-except-py.py --self-test

Stdlib ``ast`` + ``tokenize`` only.
"""

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "engineering" / "SILENT_EXCEPTIONS.md"

# Files in scope for the audit. Paths are relative to the repo root.
SCOPE_FILES = [
    "sidecar-torch/main.py",
    "sidecar-torch/dataset_handlers.py",
    "sidecar-torch/safe_load.py",
    "sidecar-torch/scope.py",
    "sidecar-torch/deps_policy.py",
    "sidecar-torch/auth.py",
    "sidecar-torch/training_template.py",
]

IGNORE_RE = re.compile(r"^\s*(ignore|ignored|noop|todo|none)\s*$", re.IGNORECASE)

_SHOW_ALL = "--all" in sys.argv
_SHOW_TSV = "--tsv" in sys.argv
_SHOW_HIDDEN = "--hidden" in sys.argv
_SELF_TEST = "--self-test" in sys.argv


# ── swallow classification ────────────────────────────────────────────────

def _is_literal(node: ast.AST) -> bool:
    """Mirror the TS guard's isLiteral: constants + empty collections.

    A NAME, call, attribute, f-string or non-empty container is NOT a literal."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return len(node.elts) == 0
    if isinstance(node, ast.Dict):
        return len(node.keys) == 0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _is_literal(node.operand)
    return False


def _literal_src(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - defensive
        return "<expr>"


def _stmt_swallow(stmt: ast.stmt) -> str | None:
    """Return a short token when `stmt` is a swallow statement, else None."""
    if isinstance(stmt, ast.Pass):
        return "pass"
    if isinstance(stmt, ast.Continue):
        return "continue"
    if isinstance(stmt, ast.Break):
        return "break"
    if isinstance(stmt, ast.Return):
        if stmt.value is None:
            return "return"
        if _is_literal(stmt.value):
            return f"return {_literal_src(stmt.value)}"
        return None
    if isinstance(stmt, ast.Expr):
        v = stmt.value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            return "docstring"
        if isinstance(v, ast.Constant) and v.value is Ellipsis:
            return "..."
        return None
    return None


def analyze_handler_body(body: list[ast.stmt]) -> str | None:
    """Return the swallow pattern text, or None when the body does real work."""
    if not body:
        return "pass"
    parts: list[str] = []
    for stmt in body:
        token = _stmt_swallow(stmt)
        if token is None:
            return None
        parts.append(token)
    return " ".join(parts)


def _is_default_assignment(stmt: ast.stmt) -> bool:
    """`x = None` / `x = []` / `x = {}` / `x = 0` / `x = False` — a defaulted
    value hidden behind a swallow. NOT in the strict definition (listed only)."""
    if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
        value = stmt.value
        return value is not None and _is_literal(value)
    return False


def looks_hidden_swallow(body: list[ast.stmt]) -> bool:
    """Body made only of swallow statements plus literal assignments."""
    if not body:
        return False
    saw_default = False
    for stmt in body:
        if _stmt_swallow(stmt) is not None:
            continue
        if _is_default_assignment(stmt):
            saw_default = True
            continue
        return False
    return saw_default


# ── comment gathering ─────────────────────────────────────────────────────

def comments_by_line(src: str) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                out.setdefault(tok.start[0], []).append(tok.string)
    except (tokenize.TokenError, IndentationError):
        pass
    return out


def gather_comments(node: ast.AST, lines: list[str], comments: dict[int, list[str]]) -> list[str]:
    """Comments on the handler's own lines plus its immediately preceding
    comment/blank block (so a comment above the `try`/`except` still counts)."""
    start = node.lineno
    end = getattr(node, "end_lineno", start) or start
    found: list[str] = []
    for ln in range(start, end + 1):
        found.extend(comments.get(ln, []))
    ln = start - 1
    while ln >= 1:
        stripped = lines[ln - 1].strip()
        if stripped == "" or stripped.startswith("#"):
            found.extend(comments.get(ln, []))
            ln -= 1
        else:
            break
    return found


def clean_comment(c: str) -> str:
    return c.lstrip("#").strip()


def has_justified_comment(comments: list[str], doc_prose: list[str]) -> bool:
    for c in comments:
        text = clean_comment(c)
        if len(text) >= 15 and not IGNORE_RE.match(text):
            return True
    for text in doc_prose:
        if len(text.strip()) >= 15 and not IGNORE_RE.match(text):
            return True
    return False


# ── site model ────────────────────────────────────────────────────────────

class Site:
    def __init__(self, file: str, line: int, pattern: str, comments: list[str],
                 snippet: str, hidden: bool = False, doc_prose: list[str] | None = None):
        self.file = file
        self.line = line
        self.pattern = pattern
        self.comments = comments
        self.snippet = snippet
        self.hidden = hidden
        self.doc_prose = doc_prose or []


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _except_type_src(h: ast.ExceptHandler) -> str:
    if h.type is None:
        return ""
    return _norm_ws(ast.unparse(h.type))


def find_sites(src: str, rel: str) -> list[Site]:
    tree = ast.parse(src)
    lines = src.splitlines()
    comments = comments_by_line(src)
    sites: list[Site] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            body_pattern = analyze_handler_body(node.body)
            if body_pattern is not None:
                type_src = _except_type_src(node)
                header = f"except {type_src}:" if type_src else "except:"
                pattern = f"{header} {body_pattern}"
                doc_prose = [c for c in _docstring_prose(node.body)]
                sites.append(Site(rel, node.lineno, pattern,
                                  gather_comments(node, lines, comments),
                                  _norm_ws(ast.get_source_segment(src, node) or "")[:90],
                                  doc_prose=doc_prose))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                call = item.context_expr
                if isinstance(call, ast.Call):
                    fn = call.func
                    name = None
                    if isinstance(fn, ast.Name):
                        name = fn.id
                    elif isinstance(fn, ast.Attribute):
                        name = fn.attr
                    if name == "suppress":
                        pattern = "with contextlib.suppress(...):"
                        sites.append(Site(rel, node.lineno, pattern,
                                          gather_comments(node, lines, comments),
                                          _norm_ws(ast.get_source_segment(src, node) or "")[:90]))
    # hidden (non-strict) handlers, for the report only
    if _SHOW_HIDDEN or _SHOW_ALL:
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and looks_hidden_swallow(node.body):
                type_src = _except_type_src(node)
                header = f"except {type_src}:" if type_src else "except:"
                sites.append(Site(rel, node.lineno, f"{header} <default-assign>",
                                  gather_comments(node, lines, comments),
                                  _norm_ws(ast.get_source_segment(src, node) or "")[:90],
                                  hidden=True))
    return sites


def _docstring_prose(body: list[ast.stmt]) -> list[str]:
    out = []
    for stmt in body:
        if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)):
            out.append(stmt.value.value)
    return out


# ── allow-list parsing ────────────────────────────────────────────────────

class DocRow:
    def __init__(self, file: str, pattern: str, classification: str):
        self.file = file
        self.pattern = pattern
        self.classification = classification


def parse_doc() -> list[DocRow]:
    """Parse the ``## Python allow-list`` section's table rows only."""
    if not DOC.exists():
        return []
    text = DOC.read_text(encoding="utf-8")
    lines = text.splitlines()
    rows: list[DocRow] = []
    in_section = False
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("## "):
            in_section = stripped.startswith("## Python allow-list")
            continue
        if not in_section:
            continue
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.split("|")]
        # ['', Location, Pattern, Class, Reason, Action, '']
        if len(cells) < 6:
            continue
        loc = cells[1].replace("`", "")
        m = re.match(r"^(.+\.py):(\d+)$", loc)
        if not m:
            continue
        cls = cells[3]
        if cls not in ("EXPECTED", "FIXED", "REVIEW"):
            continue
        rows.append(DocRow(m.group(1), _norm_ws(cells[2].replace("`", "")), cls))
    return rows


# ── self-test ─────────────────────────────────────────────────────────────

_SELF_TEST_CASES: list[tuple[str, str, bool]] = [
    ("bare except pass", "try:\n    f()\nexcept:\n    pass\n", True),
    ("except Exception pass", "try:\n    f()\nexcept Exception:\n    pass\n", True),
    ("except tuple continue",
     "for x in y:\n    try:\n        f()\n    except (A, B):\n        continue\n", True),
    ("except return None", "try:\n    f()\nexcept X:\n    return None\n", True),
    ("except return empty dict", "try:\n    f()\nexcept X:\n    return {}\n", True),
    ("except return empty list", "try:\n    f()\nexcept X:\n    return []\n", True),
    ("except return False", "try:\n    f()\nexcept X:\n    return False\n", True),
    ("except ellipsis", "try:\n    f()\nexcept X:\n    ...\n", True),
    ("docstring-only body", "try:\n    f()\nexcept X:\n    '''documented but silent, long enough'''\n", True),
    ("contextlib.suppress", "import contextlib\nwith contextlib.suppress(X):\n    f()\n", True),
    ("from-import suppress", "from contextlib import suppress\nwith suppress(X):\n    f()\n", True),
    ("nested handler in loop",
     "for i in range(3):\n    try:\n        f(i)\n    except X:\n        pass\n", True),
    ("except bare return", "try:\n    f()\nexcept X:\n    return\n", True),
    ("except break", "for x in y:\n    try:\n        f()\n    except X:\n        break\n", True),
    ("except return 0", "try:\n    f()\nexcept X:\n    return 0\n", True),
    ("except multiple swallow stmts", "try:\n    f()\nexcept X:\n    x = 1\n    pass\n", False),
    ("rereaise", "try:\n    f()\nexcept X:\n    raise\n", False),
    ("raise from", "try:\n    f()\nexcept X as e:\n    raise Foo() from e\n", False),
    ("return error dict from e",
     "try:\n    f()\nexcept Exception as e:\n    return {'ok': False, 'error': str(e)}\n", False),
    ("log and return error struct",
     "import sys\ntry:\n    f()\nexcept Exception as e:\n    print(e, file=sys.stderr)\n    return {'ok': False, 'error': str(e)}\n", False),
    ("assign err and use",
     "try:\n    f()\nexcept Exception as e:\n    err = str(e)\n    raise ValueError(err)\n", False),
    ("KeyboardInterrupt raise", "try:\n    f()\nexcept KeyboardInterrupt:\n    raise\n", False),
    ("return non-empty dict", "try:\n    f()\nexcept X:\n    return {'ok': True}\n", False),
]


def self_test() -> int:
    failures = 0
    for name, source, expected in _SELF_TEST_CASES:
        sites = [s for s in find_sites(source, "<selftest>") if not s.hidden]
        got = len(sites) > 0
        if got != expected:
            failures += 1
            print(f"  X self-test '{name}': expected swallow={expected}, got={got}")
    total = len(_SELF_TEST_CASES)
    if failures:
        print(f"\n{failures} self-test case(s) failed of {total}")
        return 1
    print(f"OK self-test: {total} cases passed")
    return 0


# ── main ──────────────────────────────────────────────────────────────────

def main() -> int:
    if _SELF_TEST:
        return self_test()

    all_sites: list[Site] = []
    scanned = 0
    for rel in SCOPE_FILES:
        path = ROOT / rel
        if not path.exists():
            print(f"  ! missing scope file: {rel}")
            continue
        scanned += 1
        all_sites.extend(find_sites(path.read_text(encoding="utf-8"), rel))

    strict = [s for s in all_sites if not s.hidden]
    hidden = [s for s in all_sites if s.hidden]
    strict.sort(key=lambda s: (s.file, s.line))

    print("phase 50: silent-exception guard (python)")
    print(f"  scanned {scanned} python files, {len(strict)} swallowing site(s)"
          f"{f', {len(hidden)} hidden default-assign' if hidden else ''}")

    if _SHOW_TSV:
        for s in strict:
            justified = "ok" if has_justified_comment(s.comments, s.doc_prose) else "NO"
            print(f"{s.file}\t{s.line}\t{s.pattern}\t{justified}")
        return 0

    if _SHOW_ALL or _SHOW_HIDDEN:
        for s in strict:
            justified = "commented" if has_justified_comment(s.comments, s.doc_prose) else "NO-COMMENT"
            print(f"  · {s.file}:{s.line}  [{s.pattern}]  {justified}  || {s.snippet}")
        for s in hidden:
            print(f"  ~ {s.file}:{s.line}  [{s.pattern}]  HIDDEN-DEFAULT  || {s.snippet}")
        print(f"\n{len(strict)} swallowing site(s), {len(hidden)} hidden default-assign(s)")
        return 0

    failures = 0

    def fail(msg: str) -> None:
        nonlocal failures
        failures += 1
        print(f"  X {msg}")

    # 1. every swallow needs a justifying comment
    for s in strict:
        if not has_justified_comment(s.comments, s.doc_prose):
            fail(f"{s.file}:{s.line} — {s.pattern} has no justifying comment "
                 "(>=15 chars prose, not 'ignore')")

    # 2. every swallow documented; 3. no stale rows
    doc_rows = parse_doc()
    doc_count: dict[str, int] = {}
    for r in doc_rows:
        if r.classification == "REVIEW":
            fail(f"allow-list contains a REVIEW row: {r.file} — {r.pattern}")
        key = f"{r.file}|{r.pattern}"
        doc_count[key] = doc_count.get(key, 0) + 1
    det_count: dict[str, int] = {}
    for s in strict:
        key = f"{s.file}|{s.pattern}"
        det_count[key] = det_count.get(key, 0) + 1
    for key, count in det_count.items():
        documented = doc_count.get(key, 0)
        if documented < count:
            fail(f"{key.replace('|', ' — ')}: {count} detected, {documented} documented")
    for key, count in doc_count.items():
        detected = det_count.get(key, 0)
        if detected < count:
            fail(f"stale allow-list row: {key.replace('|', ' — ')}: "
                 f"{count} documented, {detected} detected")

    # table for humans
    for s in strict:
        key = f"{s.file}|{s.pattern}"
        cls = "EXPECTED" if doc_count.get(key) else "-"
        justified = "yes" if has_justified_comment(s.comments, s.doc_prose) else "NO"
        print(f"  {s.file}:{s.line} | {s.pattern} | {cls} | {justified}")

    if failures:
        print(f"\n{failures} silent-except-py violation(s)")
        return 1
    print("\nOK all swallowing sites are commented and documented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
