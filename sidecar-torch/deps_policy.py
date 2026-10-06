"""Dependency specification validation policy.

Provides `validate_specs` that ensures a list of pip requirement strings
contains only plain package specifications (PEP 508) and rejects any
options, URLs, VCS references, local paths, environment markers, or newline
characters. Protected packages (torch stack, pip, setuptools, wheel) cannot
be installed/updated at runtime.
"""

import re
from typing import List, Tuple, Union

# Normalised protected package names per PEP 503 (lowercase, replace runs of -_. with -)
_PROTECTED_NORMALISED = {"torch", "torchvision", "torchaudio", "triton", "pip", "setuptools", "wheel"}

# Regex for a plain requirement spec (PEP 508 name with optional extras and version specifiers)
# This is a simplified version matching the description in the task.
_SPEC_REGEX = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
    r"(?:\[[A-Za-z0-9._,-]+\])?"
    r"(?:\s*(?:===|==|~=|!=|<=|>=|<|>)\s*[A-Za-z0-9._*+!-]+"
    r"(?:\s*,\s*(?:===|==|~=|!=|<=|>=|<|>)\s*[A-Za-z0-9._*+!-]+)*)?$"
)


def _normalize_name(name: str) -> str:
    """PEP 503 normalisation: lowercase and collapse runs of -_. into a single '-'"""
    name = name.lower()
    return re.sub(r"[-_.]+", "-", name)


def validate_specs(specs: Union[List[str], Tuple[str, ...]]) -> Tuple[bool, Union[List[str], str]]:
    """Validate a list/tuple of requirement strings.

    Returns (True, cleaned_specs) on success where cleaned_specs is a list of
    unique specs preserving order, or (False, error_message) on failure.
    """
    if not isinstance(specs, (list, tuple)):
        return False, "specs must be a list or tuple"
    if not (1 <= len(specs) <= 50):
        return False, "specs length must be between 1 and 50"

    cleaned: List[str] = []
    seen: set = set()
    for raw in specs:
        if not isinstance(raw, str):
            return False, "each spec must be a string"
        # Detect disallowed control characters in the original raw string before stripping
        if "\n" in raw or "\r" in raw or "\0" in raw:
            return False, "spec contains disallowed control characters"
        s = raw.strip()
        if not s:
            return False, "empty spec string"
        if len(s) > 200:
            return False, "spec exceeds maximum length of 200"
        # Disallow newline, carriage return, null byte early
        if "\n" in s or "\r" in s or "\0" in s:
            return False, "spec contains disallowed control characters"
        # Quick reject obvious unsafe prefixes / characters
        if s.startswith("-") or "://" in s or "@" in s or ";" in s or "/" in s or "\\" in s:
            return False, f"spec '{s}' is not a plain requirement"
        # Full regex validation
        if not _SPEC_REGEX.fullmatch(s):
            return False, f"spec '{s}' does not match allowed pattern"
        # Protected package check after normalisation of the distribution name
        name_part = re.split(r"[<>=!~;\[\( \t]", s, 1)[0].strip()
        norm_name = _normalize_name(name_part)
        if norm_name in _PROTECTED_NORMALISED:
            return False, f"{name_part} cannot be changed at runtime; recreate the environment instead"
        if s not in seen:
            seen.add(s)
            cleaned.append(s)
    return True, cleaned
