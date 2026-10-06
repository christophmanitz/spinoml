"""Sidecar authentication / Host / Origin enforcement (Phase 77/78).

Pure stdlib, importable alone. Owns:

  * ``load_config`` / ``scrub_environ`` — read ``SPINOML_*`` once at startup,
    validate, then delete the token from ``os.environ`` so subprocess children
    (pip, ``/run_script``, sbatch) never inherit it.
  * ``check_host`` / ``check_origin`` / ``token_matches`` — the three primitives
    whose combination is the per-request gate (see ``docs/engineering/SIDECAR_AUTH.md``).
  * ``decide`` — the single decision function the HTTP handler calls BEFORE
    reading any body, encoding the normative enforcement order in ONE place
    (Host → Origin → OPTIONS → GET /health → token → otherwise).

The module never logs, prints or echoes the token value; error messages must
not contain it. ``hmac.compare_digest`` is used for constant-time comparison.

Stdlib only. No torch, no Flask, no third-party.
"""

from __future__ import annotations

import hmac
import re
from dataclasses import dataclass
from typing import Callable, Mapping

TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+\Z")

DEFAULT_ORIGINS: frozenset[str] = frozenset({
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
})

CODE_UNAUTHORIZED = "UNAUTHORIZED"
CODE_BAD_ORIGIN = "BAD_ORIGIN"
CODE_BAD_HOST = "BAD_HOST"


class AuthConfigError(Exception):
    """Configuration was rejected at startup. Message MUST NOT contain the token."""


class _Duplicate:
    """Sentinel returned by ``headers_get`` when a header appeared more than once."""

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return "<DUPLICATE_HEADER>"


DUPLICATE_HEADER = _Duplicate()


@dataclass(frozen=True)
class Decision:
    """Result of ``decide``. ``origin`` is the validated Origin to echo in CORS, or
    ``None`` (no CORS — never the literal string ``"*"``)."""

    kind: str  # 'ok' | 'ok_health_limited' | 'reject'
    status: int = 0
    code: str = ""
    reason: str = ""
    origin: str | None = None
    token_ok: bool = False


@dataclass(frozen=True)
class AuthConfig:
    token: str | None
    require_token: bool
    origins: frozenset[str]


def _allow(origin: str | None, token_ok: bool = False) -> Decision:
    return Decision("ok", origin=origin, token_ok=token_ok)


def _health_limited(origin: str | None) -> Decision:
    return Decision("ok_health_limited", origin=origin, token_ok=False)


def _reject(status: int, code: str, reason: str = "", origin: str | None = None) -> Decision:
    return Decision("reject", status=status, code=code, reason=reason, origin=origin)


def load_config(environ: Mapping[str, str]) -> AuthConfig:
    """Read the ``SPINOML_SIDECAR_TOKEN`` / ``SPINOML_REQUIRE_TOKEN`` /
    ``SPINOML_ALLOWED_ORIGINS`` triple from ``environ``. Validates and raises
    ``AuthConfigError`` with a message that never contains the token value."""
    raw_token = environ.get("SPINOML_SIDECAR_TOKEN")
    token: str | None = None
    if raw_token is not None:
        # Empty string is treated as "unset" — common in env configs that blank
        # a token rather than removing the key.
        if raw_token != "":
            if len(raw_token) < 32:
                raise AuthConfigError(
                    f"SPINOML_SIDECAR_TOKEN must be at least 32 characters "
                    f"(got {len(raw_token)})"
                )
            if not TOKEN_RE.fullmatch(raw_token):
                raise AuthConfigError(
                    "SPINOML_SIDECAR_TOKEN may only contain characters [A-Za-z0-9_-]"
                )
            token = raw_token
    require_raw = str(environ.get("SPINOML_REQUIRE_TOKEN") or "") == "1"
    if require_raw and token is None:
        raise AuthConfigError(
            "SPINOML_REQUIRE_TOKEN=1 requires SPINOML_SIDECAR_TOKEN to be set"
        )
    require_token = require_raw or token is not None
    origins: set[str] = set(DEFAULT_ORIGINS)
    extra = environ.get("SPINOML_ALLOWED_ORIGINS")
    if extra:
        for entry in str(extra).split(","):
            entry = entry.strip()
            if entry:
                origins.add(entry)
    return AuthConfig(token=token, require_token=require_token, origins=frozenset(origins))


def scrub_environ(environ: Mapping[str, str]) -> None:
    """Delete the token from the given environment mapping. Mutates in place.

    ``os.environ`` is a mutable mapping and supports ``pop``. A pure read-only
    Mapping silently no-ops via ``get`` (and never deletes anything), so this
    is safe for tests."""
    try:
        environ.pop("SPINOML_SIDECAR_TOKEN", None)  # type: ignore[attr-defined]
    except Exception:
        # Truly read-only: nothing more we can do — but we never crash here.
        pass


def check_host(host_header: str | None) -> bool:
    """Loopback host allow-list. Port stripped, case-insensitive, no DNS
    re-binding. Returns True ONLY for ``127.0.0.1``, ``localhost`` and
    IPv6 ``[::1]`` (with any port)."""
    if not isinstance(host_header, str):
        return False
    h = host_header.strip()
    if h == "":
        return False
    if h.startswith("["):
        # IPv6 literal: [<host>]:<port> or [<host>]
        end = h.find("]")
        if end == -1:
            return False
        hostpart = h[1:end].strip().lower()
        rest = h[end + 1:]
        if rest != "":
            if not rest.startswith(":"):
                return False
            port = rest[1:]
            if not port.isdigit():
                return False
        return hostpart == "::1"
    # Non-bracketed: strip a trailing :port (only one colon allowed — IPv4/hostname)
    if ":" in h:
        head, _, tail = h.rpartition(":")
        if not tail.isdigit():
            return False
        hostpart = head
    else:
        hostpart = h
    return hostpart.lower() in ("127.0.0.1", "localhost")


def check_origin(origin_header: str | None, cfg: AuthConfig) -> bool:
    """Exact-member allow-list. ``None`` (no header) is allowed; present
    must match an entry byte-for-byte. ``null``, trailing slash, case
    differences and different scheme/port all fail."""
    if origin_header is None:
        return True
    if not isinstance(origin_header, str):
        return False
    return origin_header in cfg.origins


def token_matches(supplied: str | None, cfg: AuthConfig) -> bool:
    """Constant-time comparison via ``hmac.compare_digest``. The size difference
    is folded into the C implementation so we never short-circuit on length.

    ``DUPLICATE_HEADER``, missing, empty and non-string all fail. A sidecar in
    mode ``unauthenticated-dev`` (``cfg.token is None``) fails — the caller
    shouldn't be calling us in that mode, but defending in depth is cheap."""
    if cfg.token is None:
        return False
    if not isinstance(supplied, str):
        return False
    try:
        a = supplied.encode("utf-8")
        b = cfg.token.encode("utf-8")
    except (UnicodeEncodeError, AttributeError):
        return False
    return hmac.compare_digest(a, b)


# Type of the callable handed to ``decide`` for header lookup. Returns the
# header value, ``None`` when absent, or ``auth.DUPLICATE_HEADER`` when the
# header was present more than once.
HeaderGet = Callable[[str], str | None | _Duplicate]


def decide(method: str, path: str, headers_get: HeaderGet, cfg: AuthConfig) -> Decision:
    """Normative enforcement order (see ``docs/engineering/SIDECAR_AUTH.md`` §
    'Per-request enforcement order'): Host → Origin → OPTIONS → GET /health →
    token → otherwise. The HTTP handler MUST call this BEFORE ``rfile.read``
    and act on the returned ``Decision``."""
    # 1. Host (both modes).
    if not check_host(headers_get("Host")):
        return _reject(403, CODE_BAD_HOST)

    # 2. Origin (both modes).
    raw_origin = headers_get("Origin")
    if not check_origin(raw_origin, cfg):
        return _reject(403, CODE_BAD_ORIGIN)
    echo_origin: str | None = raw_origin if isinstance(raw_origin, str) else None

    # 3. OPTIONS.
    if method == "OPTIONS":
        return _allow(echo_origin, token_ok=not cfg.require_token)

    # 4. GET /health.
    if method == "GET" and path == "/health":
        if not cfg.require_token:
            return _allow(echo_origin, token_ok=True)
        if token_matches(headers_get("X-SpinoML-Token"), cfg):
            return _allow(echo_origin, token_ok=True)
        return _health_limited(echo_origin)

    # 5. Everything else.
    if not cfg.require_token:
        return _allow(echo_origin, token_ok=True)
    supplied = headers_get("X-SpinoML-Token")
    if supplied is DUPLICATE_HEADER:
        return _reject(401, CODE_UNAUTHORIZED, reason="invalid", origin=echo_origin)
    if not isinstance(supplied, str) or supplied == "":
        return _reject(401, CODE_UNAUTHORIZED, reason="missing", origin=echo_origin)
    if not token_matches(supplied, cfg):
        return _reject(401, CODE_UNAUTHORIZED, reason="invalid", origin=echo_origin)
    return _allow(echo_origin, token_ok=True)
