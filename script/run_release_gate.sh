#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
gate="${1:-quick}"

if (( $# > 1 )) || [[ "${gate}" != "quick" ]]; then
  echo "Usage: bash script/run_release_gate.sh quick" >&2
  exit 2
fi
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python executable is unavailable: ${python_bin}" >&2
  exit 2
fi

cd "${project_root}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ada_grasp_ctrl_mpl}"
if [[ -n "${ADA_GRASP_CTRL_RELEASE_GATE_ROOT:-}" ]]; then
  release_root="${ADA_GRASP_CTRL_RELEASE_GATE_ROOT}"
  if [[ -e "${release_root}" ]]; then
    if [[ ! -d "${release_root}" ]]; then
      echo "Release gate root is not a directory: ${release_root}" >&2
      exit 2
    fi
    if [[ -n "$(ls -A -- "${release_root}")" ]]; then
      echo "Release gate root is not empty; choose a new path: ${release_root}" >&2
      exit 2
    fi
  fi
  mkdir -p "${release_root}"
else
  release_root="$(mktemp -d /tmp/ada-grasp-release-gate.XXXXXX)"
fi

fixture_manifest="${project_root}/examples/quick_manifest.json"
expected_status="${project_root}/examples/quick_expected_status.json"
"${python_bin}" script/audit_example_fixtures.py --manifest "${fixture_manifest}"

for hand in shadow allegro leap_tac3d; do
  run_status=0
  ADA_GRASP_CTRL_EXAMPLE_BASE="${release_root}" \
    ADA_GRASP_CTRL_RUN_ID="release-${hand}" \
    PYTHON_BIN="${python_bin}" \
    bash script/run_example.sh "${hand}" quick || run_status=$?
  if (( run_status > 1 )); then
    echo "Quick example for ${hand} failed preflight with exit code ${run_status}." >&2
    exit "${run_status}"
  fi
  "${python_bin}" script/validate_quick_results.py verify \
    --manifest "${fixture_manifest}" \
    --expected "${expected_status}" \
    --hand "${hand}" \
    --output-root "${release_root}/${hand}/release-${hand}"
done

echo "[ada-grasp-ctrl] release gate 'quick' passed"
echo "[ada-grasp-ctrl] release artifacts: ${release_root}"
