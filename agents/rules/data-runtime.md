# Data and Runtime Rules

Apply this rule to converters, NPY/YAML/JSON schemas, runtime roots, packaging, batch execution, reports, and statistics.

## Runtime paths

- Resolve roots in this order: explicit Hydra/CLI value, environment variable, then source-checkout default.
- Use `asset_root` for hand/robot assets, `data_root` for inputs and paths stored in records, and `output_root` for every generated artifact. `save_root` is a compatibility alias only; new code uses `output_root`.
- Outside a source checkout, external roots must be explicit and absolute. Do not infer a wheel's assets from package-parent layout.
- Resolve relative paths with `ada_grasp_ctrl.paths`; never use caller CWD, hand-name string replacement, or unrelated directory scans to derive inputs or outputs.
- Never reuse an existing example/release run directory or summarize an output tree that may contain data from an older run.

## Data contracts

- Validate external records before MuJoCo, IK, optimization, or statistics consume them. Error messages must name the sample and failing field/path.
- New common grasp and control records include `schema_version: 1`; legacy v0 records without a version remain readable.
- Preserve qpos dimensions, configured joint order, WXYZ quaternion convention, contact-frame handedness, trajectory alignment, and finite-value checks.
- New converters must validate their exact raw format and emit the common grasp schema. Do not weaken one converter's shape rules to accommodate another.
- Do not silently repair malformed scientific input unless the repair is part of an explicitly documented compatibility rule.

## Batch and diagnostic contracts

- Keep deterministic discovery/selection and derive each sample seed from the global seed plus stable sample index. Results must not depend on multiprocessing completion order.
- Every task run keeps a resolved `run_manifest.yaml`. Batch tasks keep `run_report.json`; execution errors and solver degradation also appear in `failures.jsonl`.
- Preserve process exit semantics: `0` means application execution succeeded, `1` means all possible samples finished but an execution error or solver degradation occurred, and `2` means preflight/configuration/input/environment failure.
- Keep `success`, `failure`, `invalid_initialization`, `solver_degraded`, and `execution_error` mutually exclusive. The primary success-rate denominator is `success + failure`; undefined rates/metrics are YAML `null`, never NaN.
- `skip=true` may be a successful no-op only when all expected outputs exist, and it must still produce reports for the current invocation.

## Load on demand

- Read [../knowledge/runtime-data-contracts.md](../knowledge/runtime-data-contracts.md) for root mappings, record layouts, report meaning, and compatibility details.
- Read [../knowledge/architecture.md](../knowledge/architecture.md) for converter/task ownership.

