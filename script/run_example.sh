#!/usr/bin/env bash
set -u -o pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
hand="${1:-shadow}"
mode="${2:-quick}"
python_bin="${PYTHON_BIN:-python}"

case "${hand}" in
  shadow|allegro|leap_tac3d) ;;
  *) echo "Unsupported hand '${hand}'. Use shadow, allegro, or leap_tac3d." >&2; exit 2 ;;
esac
case "${mode}" in
  quick|full) ;;
  *) echo "Unsupported mode '${mode}'. Use quick or full." >&2; exit 2 ;;
esac

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python executable is unavailable: ${python_bin}" >&2
  exit 2
fi

cd "${project_root}"
export PYTHONPATH="${project_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/ada_grasp_ctrl_mpl}"

run_id="${ADA_GRASP_CTRL_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
case "${run_id}" in
  *[!A-Za-z0-9._-]*) echo "Invalid ADA_GRASP_CTRL_RUN_ID '${run_id}'." >&2; exit 2 ;;
  .|..) echo "Invalid ADA_GRASP_CTRL_RUN_ID '${run_id}'." >&2; exit 2 ;;
esac
example_base="${ADA_GRASP_CTRL_EXAMPLE_BASE:-${project_root}/output/examples}"
example_root="${example_base}/${hand}/${run_id}"
if [[ -e "${example_root}" || -L "${example_root}" ]]; then
  echo "Example run directory already exists; choose a new ADA_GRASP_CTRL_RUN_ID: ${example_root}" >&2
  exit 2
fi
if ! mkdir -p "${example_root}"; then
  echo "Cannot create example run directory: ${example_root}" >&2
  exit 2
fi
fixture_root="${project_root}/examples/data/${hand}"
object_info="${project_root}/examples/assets/object/core_bottle_15787789482f045d8add95bf56d3d2fa/info/simplified.json"
if [[ ! -f "${object_info}" ]]; then
  echo "Bundled example object is missing: ${object_info}" >&2
  exit 2
fi

overall_status=0
run_stage() {
  local stage_name="$1"
  shift
  echo "[ada-grasp-ctrl] ${stage_name}"
  "$@"
  local stage_status=$?
  if (( stage_status > overall_status )); then
    overall_status=${stage_status}
  fi
  return 0
}

if [[ "${mode}" == "full" ]]; then
  run_stage format "${python_bin}" src/main.py \
    setting=tabletop hand="${hand}" task=format exp_name=example \
    n_worker=1 output_root="${example_root}" save_dir="${example_root}" grasp_dir="${example_root}/formatted" \
    log_dir="${example_root}/log/format" task.data_name=Learning task.max_num=-1 \
    task.data_path="${fixture_root}/raw"
  if (( overall_status == 2 )); then exit 2; fi

  run_stage dummy_arm_qpos "${python_bin}" src/main.py \
    setting=tabletop hand="${hand}" task=dummy_arm_qpos exp_name=example \
    n_worker=1 output_root="${example_root}" save_dir="${example_root}" grasp_dir="${example_root}/formatted" \
    dummy_arm_grasp_dir="${example_root}/dummy_arm" log_dir="${example_root}/log/dummy_arm_qpos" \
    task.max_num=-1
  if (( overall_status == 2 )); then exit 2; fi
  control_input="${example_root}/dummy_arm"
else
  control_input="${fixture_root}/dummy_arm"
fi

run_stage control_eval "${python_bin}" src/main.py \
  setting=tabletop hand="dummy_arm_${hand}" task=control_eval exp_name=example \
  n_worker=1 output_root="${example_root}" save_dir="${example_root}" grasp_dir="${control_input}" \
  control_dir="${example_root}/control" log_dir="${example_root}/log/control_eval" \
  task.method=ours task.input_data=grasp_dir task.max_num=-1 \
  task.debug_viewer=false task.debug_render=false
if (( overall_status == 2 )); then exit 2; fi

run_stage control_stat "${python_bin}" src/main.py \
  setting=tabletop hand="dummy_arm_${hand}" task=control_stat exp_name=example \
  n_worker=1 output_root="${example_root}" save_dir="${example_root}" control_dir="${example_root}/control" \
  log_dir="${example_root}/log/control_stat" task.method=ours task.setting_name=dist_0 \
  task.input_report="${example_root}/log/control_eval/run_report.json"

run_stage report "${python_bin}" script/report_example.py "${example_root}"
echo "[ada-grasp-ctrl] run_id: ${run_id}"
echo "[ada-grasp-ctrl] outputs: ${example_root}"
exit "${overall_status}"
