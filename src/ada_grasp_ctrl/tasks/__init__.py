"""Explicit registry for the supported public pipeline."""

from .convert_format import task_format
from .dummy_arm_qpos import task_dummy_arm_qpos
from .control_eval import task_control_eval
from .control_stat import task_control_stat

TASK_REGISTRY = {
    "format": task_format,
    "dummy_arm_qpos": task_dummy_arm_qpos,
    "control_eval": task_control_eval,
    "control_stat": task_control_stat,
}
