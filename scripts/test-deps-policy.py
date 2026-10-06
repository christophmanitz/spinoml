"""Unit tests for sidecar-torch/deps_policy.py."""

import sys
import os

# Add sidecar-torch to sys.path so we can import deps_policy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../sidecar-torch")))

from deps_policy import validate_specs

def run_tests():
    failed = 0
    total = 0

    def assert_case(name, specs, expected_ok, expected_msg=None):
        nonlocal failed, total
        total += 1
        ok, result = validate_specs(specs)
        if ok != expected_ok:
            print(f"✗ {name}: expected ok={expected_ok}, got ok={ok}")
            failed += 1
            return
        if expected_msg and (isinstance(result, str) and expected_msg not in result):
            print(f"✗ {name}: expected error containing {expected_msg!r}, got {result!r}")
            failed += 1
            return
        if not expected_msg and not ok:
            print(f"✗ {name}: expected ok, but got error {result!r}")
            failed += 1
            return
        if expected_ok and isinstance(result, str):
            print(f"✗ {name}: expected cleaned list, got {result!r}")
            failed += 1
            return
        print(f"✓ {name}")

    # --- Accepted cases ---
    assert_case("numpy", ["numpy"], True)
    assert_case("numpy versioned", ["numpy>=1.24,<2"], True)
    assert_case("torch_geometric versioned", ["torch_geometric==2.5.0"], True)
    assert_case("rdkit-pypi", ["rdkit-pypi"], True)
    assert_case("scikit-learn extras", ["scikit-learn[extra]"], True)
    assert_case("Pillow", ["Pillow"], True)
    assert_case("biopython tilde", ["biopython~=1.83"], True)
    assert_case("pandas whitespace version", ["pandas ==2.2.*"], True)
    assert_case("cleaning and de-duplicating", ["numpy", " numpy", "numpy"], True) # Result should be [numpy, numpy] but it's cleaned. Wait, deduplication happens in validate_specs.

    # --- Rejected cases ---
    assert_case("pip option --index-url", ["--index-url=http://evil/simple"], False, "not a plain requirement")
    assert_case("pip option -r", ["-r", "/etc/passwd"], False, "not a plain requirement")
    assert_case("pip option --target", ["--target=/tmp/x"], False, "not a plain requirement")
    assert_case("pip option -e", ["-e", "."], False, "not a plain requirement")
    assert_case("vcs git+", ["git+https://x/y.git"], False, "not a plain requirement")
    assert_case("vcs @", ["pkg @ https://x/y.whl"], False, "not a plain requirement")
    assert_case("local path", ["./local"], False, "not a plain requirement")
    assert_case("parent path", ["../../x"], False, "not a plain requirement")
    assert_case("absolute path", ["/abs/path"], False, "not a plain requirement")
    assert_case("protocol file://", ["file:///x"], False, "not a plain requirement")
    assert_case("env marker", ["numpy; python_version<'4'"], False, "not a plain requirement")
    assert_case("newline spec", ["numpy\n--target=/x"], False, "contains disallowed control characters")
    assert_case("carriage return", ["numpy\r"], False, "contains disallowed control characters")
    assert_case("null byte", ["numpy\0"], False, "contains disallowed control characters")
    assert_case("empty spec", [""], False, "empty spec string")
    assert_case("whitespace spec", ["   "], False, "empty spec string")
    assert_case("too long name", ["a" * 201], False, "exceeds maximum length")
    assert_case("too many specs", ["numpy"] * 51, False, "length must be between 1 and 50")
    assert_case("non-str item", [123], False, "each spec must be a string")
    assert_case("None instead of list", None, False, "must be a list or tuple")
    assert_case("string instead of list", "numpy", False, "must be a list or tuple")

    # --- Protected packages ---
    assert_case("protected torch", ["torch"], False, "cannot be changed at runtime")
    assert_case("protected Torch (case)", ["Torch==2.0"], False, "cannot be changed at runtime")
    assert_case("protected torchvision", ["torchvision"], False, "cannot be changed at runtime")
    assert_case("protected torch-vision (not protected)", ["torch-vision"], True)
    # Note: torch-vision normalises to torch-vision, and torch-vision != torch.
    # Wait, if torch-vision normalises to torch-vision, then it is NOT protected.
    # Only torch is protected.

    if failed == 0:
        print(f"\n✓ {total}/{total} tests passed")
        sys.exit(0)
    else:
        print(f"\n✗ {failed}/{total} tests failed")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
