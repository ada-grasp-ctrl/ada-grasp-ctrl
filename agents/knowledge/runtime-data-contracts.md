# Runtime and Data Contract Knowledge

Load this note for path, schema, report, or statistics details. Validate against current tests before changing a contract.

## External roots

| Root | Environment variable | Owns | Source-checkout default |
| --- | --- | --- | --- |
| `asset_root` | `ADA_GRASP_CTRL_ASSET_ROOT` | Hand MJCF, meshes, robot files | `<checkout>/assets` |
| `data_root` | `ADA_GRASP_CTRL_DATA_ROOT` | Raw inputs and relative object/scene paths | `<checkout>` |
| `output_root` | `ADA_GRASP_CTRL_OUTPUT_ROOT` | Grasp, control, log, report, and statistics output | `<checkout>/output` |

Precedence is explicit config, environment, then checkout default. A code-only wheel has no checkout fallback, so all needed roots must be supplied. Relative configured roots are accepted only when a checkout supplies an unambiguous anchor.

## Core record shapes

A common grasp record contains `obj_path`, finite scalar `obj_scale`, seven-value `obj_pose` with a WXYZ quaternion, and equally sized one-dimensional `pregrasp_qpos`, `grasp_qpos`, and `squeeze_qpos`. `joint_names` is optional for legacy/pose-prefixed inputs but, when required by a stage, must match the configured dimension and order.

A control record retains trajectories such as `obj_pose`, `dof`, `doa`, `contacts`, planned values, and optimization diagnostics. Contact entries carry position, a six-value local wrench, and a right-handed frame. Per-step arrays must align with the contact/control timeline.

New records use `schema_version: 1`. Absence of the field identifies readable legacy v0 data; it is not permission to skip validation.

## Status and exit semantics

| Episode/sample status | Meaning | Primary success denominator |
| --- | --- | --- |
| `success` | Completed and passed the lift/scientific criterion | Included |
| `failure` | Completed but failed the scientific criterion | Included |
| `invalid_initialization` | Initial geometry/state is scientifically invalid | Excluded |
| `solver_degraded` | A control solve was rejected and its configured apply/hold/abort policy was used | Excluded |
| `execution_error` | The sample could not complete because of an exception | Excluded |

Exit `0` can coexist with invalid initializations because the application ran correctly. Exit `1` signals sample execution errors or solver degradation after best-effort completion. Exit `2` signals task-level preflight/configuration/environment failure.

Each log directory records the resolved config and roots, git state, dependency versions and import origins, hardware, and sorted inputs in `run_manifest.yaml`. Batch reports are the authoritative list of inputs and outputs for that invocation; statistics and example reporting should consume an explicit current report rather than scan a shared output tree.
