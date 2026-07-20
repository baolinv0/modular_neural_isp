#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TORCH_PATH="${CROSS_CAMERA_TORCH_PATH:-/tmp/cross-camera-torch}"
export PYTHONPATH="${TORCH_PATH}:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
RUN_ROOT="$(mktemp -d /tmp/cross-camera-verification.XXXXXX)"

echo "VERIFICATION_ROOT=${RUN_ROOT}"
python -m compileall -q "${ROOT}/cross_camera_tm" "${ROOT}/main" "${ROOT}/tests" "${ROOT}/scripts"
echo "COMPILEALL=PASS"
python -m unittest discover -s "${ROOT}/tests" -v
echo "UNITTESTS=PASS"
python "${ROOT}/main/run_cross_camera_adaptation.py" validate-config --config "${ROOT}/configs/cross_camera_tm_v2.yaml"
python "${ROOT}/main/run_cross_camera_adaptation.py" synthetic-canary --config "${ROOT}/configs/cross_camera_tm_v2.yaml" --output-dir "${RUN_ROOT}/run-a"
python "${ROOT}/main/run_cross_camera_adaptation.py" synthetic-canary --config "${ROOT}/configs/cross_camera_tm_v2.yaml" --output-dir "${RUN_ROOT}/run-b"
python "${ROOT}/scripts/compare_cross_camera_canary.py" "${RUN_ROOT}/run-a/canary_report.json" "${RUN_ROOT}/run-b/canary_report.json"
python "${ROOT}/scripts/check_real_samsung_checkpoint.py" --checkpoint "${ROOT}/photofinishing/models/photofinishing_s24-style-0.pth"
echo "REAL_CHECKPOINT_INTERFACE_CANARY=PASS"
test -s "${RUN_ROOT}/run-a/manifest.jsonl"
test -s "${RUN_ROOT}/run-a/canary_report.json"
echo "OUTPUT_VALIDATION=PASS"
echo "REAL_DATA_EFFECTIVENESS=UNVERIFIED"
echo "VERIFICATION=PASS"
