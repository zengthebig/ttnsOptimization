#!/usr/bin/env bash
# R5 autoresearch：校验双图 → git add → commit
# 用法: ./simple_ttns_l2/autoresearch/r5_fix/finish_attempt.sh 001
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "用法: $0 <attempt_id> [extra git paths...]" >&2
  exit 1
fi

ATTEMPT_ID="$1"
shift

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
AR_DIR="${REPO_ROOT}/simple_ttns_l2/autoresearch/r5_fix"
ART_DIR="${AR_DIR}/artifacts/${ATTEMPT_ID}"

MARGINAL="${ART_DIR}/marginal_slices.png"
REFINED="${ART_DIR}/slice_refined.png"
METRICS="${ART_DIR}/metrics.json"
MANIFEST="${ART_DIR}/manifest.json"

missing=()
[[ -f "$MARGINAL" ]] || missing+=("marginal_slices.png")
[[ -f "$REFINED" ]] || missing+=("slice_refined.png")
[[ -f "$METRICS" ]] || missing+=("metrics.json")

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "错误: attempt ${ATTEMPT_ID} 缺少必产物: ${missing[*]}" >&2
  echo "请先运行: python -m simple_ttns_l2.autoresearch.r5_fix.run_attempt --attempt-id ${ATTEMPT_ID} ..." >&2
  exit 1
fi

NAME="$(python3 -c "import json; print(json.load(open('${METRICS}')).get('name',''))")"
VARIANT="$(python3 -c "import json; print(json.load(open('${METRICS}')).get('variant',''))")"
L4_LL="$(python3 -c "import json; d=json.load(open('${METRICS}')); print(f\"{d['per_layer'][-1]['joint_ll']:.2f}\")")"
L4_NP="$(python3 -c "import json; d=json.load(open('${METRICS}')); print(f\"{d['per_layer'][-1]['nonpos_rate']:.2f}\")")"
L4_SR="$(python3 -c "import json; d=json.load(open('${METRICS}')); print(f\"{d['per_layer'][-1]['std_ratio_mean']:.2f}\")")"

cd "$REPO_ROOT"

git add \
  "${ART_DIR}/" \
  "${AR_DIR}/PROGRESS.md" \
  "$@"

MSG="autoresearch(r5): ${ATTEMPT_ID} ${NAME}

variant=${VARIANT}  L4 LL=${L4_LL}  nonpos=${L4_NP}  std_ratio=${L4_SR}

artifacts:
  simple_ttns_l2/autoresearch/r5_fix/artifacts/${ATTEMPT_ID}/marginal_slices.png
  simple_ttns_l2/autoresearch/r5_fix/artifacts/${ATTEMPT_ID}/slice_refined.png"

git commit -m "$MSG"
echo "committed: ${ATTEMPT_ID}"
