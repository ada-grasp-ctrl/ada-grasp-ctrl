#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
gate="${1:-portable}"

case "${gate}" in
  quick|fixed|wheel|portable|release300|all) ;;
  *) echo "Unsupported release gate '${gate}'." >&2; exit 2 ;;
esac
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

# Run a batch command while allowing the documented exit code 1. The strict
# golden/audit step decides whether a degraded result matches the release set.
run_batch_gate() {
  local status=0
  "$@" || status=$?
  if (( status > 1 )); then
    echo "Release batch failed preflight with exit code ${status}: $*" >&2
    exit "${status}"
  fi
}

# Run one configured source-checkout CLI command with explicit external roots.
run_cli() {
  "${python_bin}" src/main.py \
    asset_root="${project_root}/assets" \
    data_root="${project_root}" \
    "$@"
}

run_quick_gate() {
  local hand
  local quick_root="${release_root}/quick"
  for hand in shadow allegro leap_tac3d; do
    ADA_GRASP_CTRL_EXAMPLE_BASE="${quick_root}" \
      ADA_GRASP_CTRL_RUN_ID="release-${hand}" \
      PYTHON_BIN="${python_bin}" \
      bash script/run_example.sh "${hand}" quick
  done
}

run_fixed_gate() {
  local hand method hand_root
  local fixed_root="${release_root}/fixed_matrix"
  for hand in shadow allegro leap_tac3d; do
    hand_root="${fixed_root}/${hand}"
    for method in ours op bs1 bs2 bs3; do
      run_batch_gate run_cli \
        setting=tabletop hand="dummy_arm_${hand}" task=control_eval \
        exp_name="release_fixed_${hand}_${method}" n_worker=1 \
        output_root="${hand_root}" save_dir="${hand_root}" \
        grasp_dir="${project_root}/examples/data/${hand}/dummy_arm" \
        control_dir="${hand_root}/control" log_dir="${hand_root}/log/${method}" \
        task.method="${method}" task.input_data=grasp_dir task.max_num=-1 \
        task.debug_viewer=false task.debug_render=false
    done
  done
  "${python_bin}" script/compare_golden.py \
    release/golden/fixed_matrix "${fixed_root}" \
    --json-report "${release_root}/fixed_matrix_comparison.json"
}

run_wheel_gate() {
  local wheel_root="${release_root}/wheel"
  local wheel_dir="${wheel_root}/dist"
  local wheel_venv="${wheel_root}/venv"
  local wheel_output="${wheel_root}/output"
  local wheel_cwd="${wheel_root}/arbitrary-cwd"
  local wheel_main="${wheel_root}/main.py"
  local runtime_site
  local wheel_site
  mkdir -p "${wheel_dir}" "${wheel_cwd}"
  cp "${project_root}/src/main.py" "${wheel_main}"

  "${python_bin}" -m pip wheel --no-deps --no-build-isolation \
    third_party/pytorch_kinematics third_party/utils_python . -w "${wheel_dir}"
  "${python_bin}" -m venv --system-site-packages "${wheel_venv}"
  env -u PYTHONPATH "${wheel_venv}/bin/python" -m pip install --no-deps --force-reinstall \
    "${wheel_dir}"/pytorch_kinematics-*.whl \
    "${wheel_dir}"/mingrui_utils_python-*.whl \
    "${wheel_dir}"/ada_grasp_ctrl-*.whl

  # Reuse the maintained environment's compiled scientific dependencies while
  # keeping the three newly built wheels ahead of any parent-environment copy.
  # A .pth entry is appended while inherited PYTHONPATH is removed from every
  # wheel command, preventing a source checkout or older wheel from winning.
  runtime_site="$("${python_bin}" -c 'import site; print(site.getsitepackages()[0])')"
  wheel_site="$(env -u PYTHONPATH "${wheel_venv}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
  printf '%s\n' "${runtime_site}" > "${wheel_site}/ada_grasp_ctrl_runtime.pth"
  env -u PYTHONPATH "${wheel_venv}/bin/python" -m pip check
  env -u PYTHONPATH "${wheel_venv}/bin/python" -c '
from pathlib import Path
import sys
import ada_grasp_ctrl
import mr_utils
import pytorch_kinematics

wheel_site = Path(sys.argv[1]).resolve()
for module in (ada_grasp_ctrl, mr_utils, pytorch_kinematics):
    origin = Path(module.__file__).resolve()
    assert origin.is_relative_to(wheel_site), f"{module.__name__} loaded from {origin}, not {wheel_site}"
' "${wheel_site}"
  if [[ -e "${wheel_venv}/bin/ada-grasp-ctrl" ]]; then
    echo "Wheel unexpectedly installed the removed ada-grasp-ctrl console command." >&2
    exit 1
  fi
  env -u PYTHONPATH "${wheel_venv}/bin/python" "${wheel_main}" --help >/dev/null

  local missing_root_status=0
  env -u ADA_GRASP_CTRL_ASSET_ROOT \
      -u ADA_GRASP_CTRL_DATA_ROOT \
      -u ADA_GRASP_CTRL_OUTPUT_ROOT \
      -u PYTHONPATH \
      "${wheel_venv}/bin/python" "${wheel_main}" \
      task=control_stat hand=dummy_arm_shadow n_worker=1 || missing_root_status=$?
  if (( missing_root_status != 2 )); then
    echo "Wheel without external roots returned ${missing_root_status}; expected 2." >&2
    exit 1
  fi

  (
    cd "${wheel_cwd}"
    env -u PYTHONPATH "${wheel_venv}/bin/python" "${wheel_main}" \
      setting=tabletop hand=dummy_arm_shadow task=control_eval \
      exp_name=release_wheel n_worker=1 \
      asset_root="${project_root}/assets" data_root="${project_root}" \
      output_root="${wheel_output}" \
      grasp_dir="${project_root}/examples/data/shadow/dummy_arm" \
      control_dir="${wheel_output}/control" log_dir="${wheel_output}/log/control_eval" \
      task.method=ours task.input_data=grasp_dir task.max_num=-1 \
      task.debug_viewer=false task.debug_render=false
  )
  "${python_bin}" script/compare_golden.py \
    release/golden/fixed_matrix/shadow/control/ours_default \
    "${wheel_output}/control/ours_default" \
    --json-report "${wheel_root}/comparison.json"
}

run_release300_gate() {
  local input_root="${ADA_GRASP_CTRL_RELEASE_INPUT_ROOT:-}"
  local result_root="${release_root}/release300"
  local hand hand_root grasp_root
  if [[ -z "${input_root}" ]]; then
    echo "ADA_GRASP_CTRL_RELEASE_INPUT_ROOT is required for release300." >&2
    exit 2
  fi
  if [[ ! -d "${input_root}" ]]; then
    echo "Release input root is missing: ${input_root}" >&2
    exit 2
  fi
  input_root="$(cd "${input_root}" && pwd)"
  for hand in shadow allegro leap_tac3d; do
    hand_root="${result_root}/${hand}"
    grasp_root="${input_root}/learn_dummy_arm_${hand}/graspdata"
    if [[ ! -d "${grasp_root}" ]]; then
      echo "Release input directory is missing: ${grasp_root}" >&2
      exit 2
    fi
    run_batch_gate run_cli \
      setting=tabletop hand="dummy_arm_${hand}" task=control_eval \
      exp_name=release300 n_worker=8 output_root="${hand_root}" \
      save_dir="${hand_root}" grasp_dir="${grasp_root}" \
      control_dir="${hand_root}/control" log_dir="${hand_root}/log/control_eval" \
      task.method=ours task.input_data=grasp_dir task.max_num=-1 \
      task.control.solver_failure_policy=hold_current \
      task.debug_viewer=false task.debug_render=false
    run_batch_gate run_cli \
      setting=tabletop hand="dummy_arm_${hand}" task=control_stat \
      exp_name=release300 n_worker=8 output_root="${hand_root}" \
      save_dir="${hand_root}" control_dir="${hand_root}/control" \
      log_dir="${hand_root}/log/control_stat" task.method=ours task.setting_name=dist_0 \
      task.input_report="${hand_root}/log/control_eval/run_report.json"
  done
  "${python_bin}" script/audit_golden.py verify release/golden/artifact.json \
    --release-root "${result_root}" \
    --release-input-root "${input_root}"
}

case "${gate}" in
  quick) run_quick_gate ;;
  fixed) run_fixed_gate ;;
  wheel) run_wheel_gate ;;
  portable) run_quick_gate; run_fixed_gate; run_wheel_gate ;;
  release300) run_release300_gate ;;
  all) run_quick_gate; run_fixed_gate; run_wheel_gate; run_release300_gate ;;
esac

echo "[ada-grasp-ctrl] release gate '${gate}' passed"
echo "[ada-grasp-ctrl] release artifacts: ${release_root}"
