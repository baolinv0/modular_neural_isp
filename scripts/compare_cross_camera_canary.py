#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    first = json.loads(args.first.read_text(encoding="utf-8"))
    second = json.loads(args.second.read_text(encoding="utf-8"))
    if first != second:
        print("SYNTHETIC_CANARY_DETERMINISM=FAIL")
        return 1
    if not first.get("synthetic") or first.get("real_model"):
        print("SYNTHETIC_CANARY_LABELS=FAIL")
        return 1
    print("SYNTHETIC_CANARY_DETERMINISM=PASS")
    print(f"SYNTHETIC_CANARY_ROUTE={first.get('route')}")
    print(f"SYNTHETIC_CANARY_CERTIFIED={str(first.get('certification_accepted')).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
