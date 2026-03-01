#!/usr/bin/env bash
set -euo pipefail

echo "[smoke] validating shell scripts..."
bash -n tools/ip-scan.sh
bash -n tools/smoke.sh

echo "[smoke] validating JSON schemas..."
python3 - <<'PY'
import json
from pathlib import Path

for path in Path("specs/jsonschema").glob("*.json"):
    with open(path, "r", encoding="utf-8") as f:
        json.load(f)
print("ok")
PY

echo "[smoke] complete"
