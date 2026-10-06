#!/usr/bin/env python3
# Adversarial regression helper for SpinoML codegen security.
#
# Reads ONE JSON document from stdin:
#   {"cases":    [{"id": str, "code": str, "sentinel": str}],
#    "literals": [{"id": str, "literal": str, "expected": str}]}
# and prints ONE JSON document:
#   {"cases":    [{"id": str, "parses": bool, "inert": bool, "error": str|null}],
#    "literals": [{"id": str, "ok": bool, "error": str|null}]}
#
# `parses` = ast.parse(code) succeeds.
# `inert`  = parses AND no tokenize NAME token equals the sentinel AND no NAME
#            token is one of the JS leaks {NaN, Infinity, undefined, null}.
# A payload that broke out of a string/comment shows up as a NAME token; a
# payload kept inside a string literal or comment is a STRING/COMMENT token and
# therefore inert.
#
# stdlib only, no prompts. Exit 0 unless the stdin JSON is malformed.

import ast
import io
import json
import sys
import tokenize

LEAK_NAMES = {"NaN", "Infinity", "undefined", "null"}


def check_case(case):
    cid = case.get("id", "")
    code = case.get("code", "")
    sentinel = case.get("sentinel", "")
    parses = False
    error = None
    try:
        ast.parse(code)
        parses = True
    except (SyntaxError, ValueError) as exc:
        if isinstance(exc, SyntaxError):
            error = "SyntaxError: %s" % (exc.msg,)
        else:
            error = "%s: %s" % (type(exc).__name__, exc)

    inert = False
    if parses:
        leaked = False
        try:
            for tok in tokenize.generate_tokens(io.StringIO(code).readline):
                if tok.type == tokenize.NAME:
                    if sentinel and tok.string == sentinel:
                        leaked = True
                        break
                    if tok.string in LEAK_NAMES:
                        leaked = True
                        break
        except Exception as exc:  # TokenError, IndentationError, ...
            leaked = True
            error = "%s: %s" % (type(exc).__name__, exc)
        inert = not leaked

    return {"id": cid, "parses": parses, "inert": inert, "error": error}


def check_literal(item):
    lid = item.get("id", "")
    literal = item.get("literal", "")
    expected = item.get("expected")
    try:
        got = ast.literal_eval(literal)
    except (SyntaxError, ValueError) as exc:
        return {"id": lid, "ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    if got == expected:
        return {"id": lid, "ok": True, "error": None}
    return {"id": lid, "ok": False, "error": "literal_eval=%r != expected=%r" % (got, expected)}


def main():
    raw = sys.stdin.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        sys.stderr.write("py_inert_check: malformed JSON: %s\n" % (exc,))
        return 2
    out = {
        "cases": [check_case(c) for c in data.get("cases", [])],
        "literals": [check_literal(item) for item in data.get("literals", [])],
    }
    sys.stdout.write(json.dumps(out))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
