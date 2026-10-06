"""Unit tests for sidecar-torch/auth.py (Phase 77/78). Pure stdlib, no torch.

Run standalone:  python sidecar-torch/test_auth.py     (exit 1 on any failure)
or via pytest if available. Also invoked by scripts/test-sidecar-auth-torch.ts.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import auth  # noqa: E402

_PASS = 0
_FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  \u2713 {name}")
    else:
        _FAIL += 1
        print(f"  \u2717 {name}{(' \u2014 ' + detail) if detail else ''}")


TOKEN = "a" * 64


def _cfg(token: str | None = TOKEN, require: bool | None = None, extra: str = ""):
    env: dict[str, str] = {}
    if token is not None:
        env["SPINOML_SIDECAR_TOKEN"] = token
    if require is None:
        require = token is not None
    if require:
        env["SPINOML_REQUIRE_TOKEN"] = "1"
    if extra:
        env["SPINOML_ALLOWED_ORIGINS"] = extra
    return auth.load_config(env)


def _getter(mapping: dict):
    def get(name, default=None):
        return mapping.get(name, default)
    return get


# ── be host:matrix ───────────────────────────────────────────────────────
def test_check_host() -> None:
    print("check_host")
    good = ["127.0.0.1", "localhost", "LOCALHOST", "LocalHost:1", "127.0.0.1:7421",
            "[::1]", "[::1]:80", "[::1]:1"]
    bad = ["127.0.0.1.evil.com", "evil.com#@127.0.0.1", "127.0.0.1@evil.com",
           "localhost.", "", "   ", "evil.example", "0.0.0.0", "::1", "[::2]",
           "127.0.0.1:abc", "127.0.0.1:", "[::1]evil"]
    for h in good:
        check(f"accept {h!r}", auth.check_host(h) is True)
    for h in bad:
        check(f"reject {h!r}", auth.check_host(h) is False)
    check("reject None", auth.check_host(None) is False)


# ── be origin:exact ──────────────────────────────────────────────────────
def test_check_origin() -> None:
    print("check_origin")
    cfg = _cfg()
    check("None allowed", auth.check_origin(None, cfg) is True)
    for o in sorted(auth.DEFAULT_ORIGINS):
        check(f"member {o}", auth.check_origin(o, cfg) is True)
    bad = ["null", "", "http://localhost:5173/", "HTTP://LOCALHOST:5173",
           "https://localhost:5173", "http://localhost:51730",
           "https://evil.example", "http://127.0.0.1:5173/"]
    for o in bad:
        check(f"reject {o!r}", auth.check_origin(o, cfg) is False)
    check("duplicate sentinel rejected", auth.check_origin(auth.DUPLICATE_HEADER, cfg) is False)

    cfg2 = _cfg(extra="http://my.tool:9, http://other:1")
    check("extra origin allowed", auth.check_origin("http://my.tool:9", cfg2) is True)
    check("extra near miss rejected", auth.check_origin("http://my.tool:99", cfg2) is False)


# ── be token:compare ─────────────────────────────────────────────────────
def test_token_matches() -> None:
    print("token_matches")
    cfg = _cfg()
    check("exact matches", auth.token_matches(TOKEN, cfg) is True)
    check("wrong same length", auth.token_matches("b" * 64, cfg) is False)
    check("shorter", auth.token_matches("a" * 31, cfg) is False)
    check("longer", auth.token_matches("a" * 65, cfg) is False)
    check("prefix", auth.token_matches(TOKEN[:63], cfg) is False)
    check("empty", auth.token_matches("", cfg) is False)
    check("None", auth.token_matches(None, cfg) is False)
    check("duplicate", auth.token_matches(auth.DUPLICATE_HEADER, cfg) is False)
    check("unauthenticated cfg never matches", auth.token_matches(TOKEN, _cfg(token=None)) is False)


# ── be config:load ───────────────────────────────────────────────────────
def test_load_config() -> None:
    print("load_config")
    cfg = _cfg()
    check("token stored", cfg.token == TOKEN)
    check("require_token true when token set", cfg.require_token is True)

    dev = auth.load_config({})
    check("no token -> None", dev.token is None)
    check("no token -> require false", dev.require_token is False)
    check("no token -> default origins", dev.origins == auth.DEFAULT_ORIGINS)

    empty = auth.load_config({"SPINOML_SIDECAR_TOKEN": ""})
    check("empty token treated as unset", empty.token is None)

    bad_tokens = [
        ("short", "a" * 31),
        ("space", "a" * 31 + " "),
        ("newline", "a" * 31 + "\n"),
        ("unicode", "a" * 31 + "\u00e9"),
        ("dot", "a" * 31 + "."),
    ]
    for label, tok in bad_tokens:
        try:
            auth.load_config({"SPINOML_SIDECAR_TOKEN": tok})
        except auth.AuthConfigError as e:
            leaked = tok in str(e)
            check(f"token {label} rejected", True)
            check(f"token {label} error hides value", leaked is False, str(e))
        else:
            check(f"token {label} rejected", False, "no AuthConfigError")

    try:
        auth.load_config({"SPINOML_REQUIRE_TOKEN": "1"})
    except auth.AuthConfigError as e:
        check("require without token rejected", "SPINOML_REQUIRE_TOKEN" in str(e))
    else:
        check("require without token rejected", False, "no AuthConfigError")

    check("require=1 with token ok", auth.load_config(
        {"SPINOML_SIDECAR_TOKEN": TOKEN, "SPINOML_REQUIRE_TOKEN": "1"}).require_token is True)

    cfg3 = _cfg(extra=" http://x:1 ,, http://y:2 ")
    check("extra origins parsed", {"http://x:1", "http://y:2"} <= cfg3.origins)


# ── be config:scrub ──────────────────────────────────────────────────────
def test_scrub_environ() -> None:
    print("scrub_environ")
    env = {"SPINOML_SIDECAR_TOKEN": TOKEN, "PATH": "/x"}
    auth.scrub_environ(env)
    check("token removed", "SPINOML_SIDECAR_TOKEN" not in env)
    check("other keys kept", env.get("PATH") == "/x")
    auth.scrub_environ({})  # no crash on missing key


# ── be decide:order ──────────────────────────────────────────────────────
def test_decide() -> None:
    print("decide")
    cfg = _cfg(extra="http://my.tool:9")
    host = {"Host": "127.0.0.1"}

    def dec(method, path, extra_headers, c=cfg):
        h = dict(host)
        h.update(extra_headers)
        return auth.decide(method, path, _getter(h), c)

    d = dec("POST", "/infer", {})
    check("host ok origin absent token missing -> 401 missing",
          d.kind == "reject" and d.status == 401 and d.code == auth.CODE_UNAUTHORIZED
          and d.reason == "missing")
    d = dec("POST", "/infer", {"X-SpinoML-Token": "b" * 64})
    check("wrong token -> 401 invalid", d.status == 401 and d.reason == "invalid")
    d = dec("POST", "/infer", {"X-SpinoML-Token": TOKEN})
    check("right token -> ok", d.kind == "ok" and d.token_ok is True)

    d = dec("POST", "/infer", {"Host": "evil.example", "X-SpinoML-Token": TOKEN})
    check("bad host checked first", d.code == auth.CODE_BAD_HOST and d.status == 403)
    d = dec("POST", "/infer", {"Origin": "https://evil.example", "X-SpinoML-Token": TOKEN})
    check("bad origin -> 403", d.code == auth.CODE_BAD_ORIGIN)
    check("bad origin carries no echo origin", d.origin is None)
    d = dec("POST", "/infer", {"Origin": "http://my.tool:9", "X-SpinoML-Token": TOKEN})
    check("allowed origin echoed", d.origin == "http://my.tool:9")

    d = dec("OPTIONS", "/infer", {})
    check("OPTIONS no token -> ok", d.kind == "ok")
    d = dec("OPTIONS", "/infer", {"Origin": "http://my.tool:9"})
    check("OPTIONS echoes origin", d.origin == "http://my.tool:9")

    d = dec("GET", "/health", {})
    check("health no token -> limited", d.kind == "ok_health_limited" and d.token_ok is False)
    d = dec("GET", "/health", {"X-SpinoML-Token": TOKEN})
    check("health right token -> full", d.kind == "ok" and d.token_ok is True)
    d = dec("GET", "/health", {"X-SpinoML-Token": "b" * 64})
    check("health wrong token -> limited", d.kind == "ok_health_limited")

    d = dec("POST", "/infer", {"X-SpinoML-Token": auth.DUPLICATE_HEADER})
    check("duplicate token sentinel -> 401 invalid", d.status == 401 and d.reason == "invalid")

    dev = _cfg(token=None)
    d = auth.decide("POST", "/infer", _getter(host), dev)
    check("dev mode no token -> ok", d.kind == "ok" and d.token_ok is True)
    d = auth.decide("GET", "/health", _getter(host), dev)
    check("dev health -> ok full", d.kind == "ok")
    d = auth.decide("POST", "/infer",
                    _getter({"Host": "evil.example", "X-SpinoML-Token": TOKEN}), dev)
    check("dev still enforces host", d.code == auth.CODE_BAD_HOST)
    d = auth.decide("POST", "/infer",
                    _getter({"Host": "127.0.0.1", "Origin": "https://evil.example"}), dev)
    check("dev still enforces origin", d.code == auth.CODE_BAD_ORIGIN)


def main() -> int:
    for fn in (test_check_host, test_check_origin, test_token_matches,
               test_load_config, test_scrub_environ, test_decide):
        fn()
    print(f"\nauth unit tests: {_PASS} passed, {_FAIL} failed")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
