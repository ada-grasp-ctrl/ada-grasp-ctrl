# Architecture Knowledge

Load this note when tracing ownership or extending the application. Current code and configuration remain authoritative.

## Execution flow

```text
ada-grasp-ctrl / src/main.py
  -> Hydra composes config/base.yaml + task/<task>.yaml + hand/<hand>.yaml
  -> cli.configure_runtime() resolves roots, workers, seeds, and log paths
  -> TASK_REGISTRY selects one task
  -> task writes manifests/reports and processes deterministic inputs
```

The maintained data pipeline is:

```text
raw BODex/Learning/Batched record
  -> tasks/convert_format.py
  -> common grasp record
  -> tasks/dummy_arm_qpos.py
  -> dummy-arm + hand qpos record
  -> tasks/control_eval.py
  -> per-perturbation control trajectories and diagnostics
  -> tasks/control_stat.py
  -> aggregate YAML statistics
```

## Ownership map

| Concern | Primary location |
| --- | --- |
| CLI and public error mapping | `src/ada_grasp_ctrl/cli.py`, `errors.py` |
| Root resolution and run metadata | `paths.py`, `runtime.py` |
| Deterministic batches and reports | `batch.py` |
| External record validation | `schema.py` |
| Task implementations | `tasks/` |
| Shared control lifecycle | `tasks/control_eval_func/episode_runner.py` |
| Method-specific policies | `tasks/control_eval_func/tabletop_dummy_arm_*.py` |
| Optimization validation/fallback | `optimization.py`, `utils/grasp_controller.py` |
| MuJoCo hand/object/contact behavior | `utils/hand_util.py` |
| Robot metadata and joint mappings | `utils/robots/`, `utils/robot_adaptor.py` |
| Hydra public defaults | `config/base.yaml`, `config/task/`, `config/hand/` |

## Extension points

- A converter is registered in `CONVERTER_REGISTRY`, validates one exact raw schema, and writes the common v1 grasp schema.
- A controller is registered in `METHOD_REGISTRY` and expresses policy differences through the shared episode lifecycle.
- A hand adds configuration plus robot/MJCF metadata and exact joint-order mappings. Public support also requires fixtures, integration tests, provenance, and release evidence.
- A task is registered in `TASK_REGISTRY` and must participate in runtime setup, manifest generation, stable error mapping, and applicable batch reporting.

The registries are deliberate trust boundaries. They make public support auditable and prevent configuration strings from executing arbitrary code.

