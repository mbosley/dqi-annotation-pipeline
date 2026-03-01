#!/usr/bin/env bash
set -euo pipefail

echo "[ip-scan] scanning for likely secrets..."
rg -n 'AKIA|BEGIN( RSA| OPENSSH) PRIVATE KEY|xox[baprs]-|ghp_|sk-[A-Za-z0-9]{20,}' -S . \
  --glob '!tools/ip-scan.sh' || true

echo "[ip-scan] scanning for likely private-source provenance markers..."
DEFAULT_PATTERNS='Desktop/projects/dqi-annotation-pipeline|complete_speech_data\\.csv|sampled_speeches\\.txt|dqi_reasoning_examples|speech_analysis_report\\.txt|theory_comparison_report\\.txt|mitchellbosley@gmail\\.com|data\\.stanford\\.edu/congress_text|SERVIU|CAF|IDB'
PATTERNS="${IP_SCAN_PATTERNS:-$DEFAULT_PATTERNS}"
rg -n "$PATTERNS" -S . \
  --glob '!tools/ip-scan.sh' \
  --glob '!docs/harvest-plan.md' || true

echo "[ip-scan] complete"
