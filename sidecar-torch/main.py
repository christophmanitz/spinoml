"""Phase 3 will implement stdio JSON-RPC for shape inference. Stub for now."""

import json
import sys


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"invalid json: {e}"}), flush=True)
            continue
        print(json.dumps({"echo": msg, "todo": "phase 3"}), flush=True)


if __name__ == "__main__":
    main()
