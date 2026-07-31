# Coordinated Contact Control for Adaptive Dexterous Grasping Under Uncertainty

[Project website](https://ada-grasp-ctrl.github.io/)

Ada Grasp Ctrl is the reference implementation of coordinated contact control for dexterous grasp execution. The maintained application is a reproducible four-stage pipeline:

```text
raw grasp → format → dummy_arm_qpos → control_eval → control_stat
```

The supported matrix is Shadow, Allegro, and LEAP Tac3D hands (plus their dummy-arm models), the `ours`, `op`, `bs1`, `bs2`, and `bs3` control methods, and BODex, Learning, and Batched input converters. Underactuated hands and the former `eval`, `stat`, `vobj`, `vusd`, and `collect` prototype tasks are not supported.

![Diverse grasps](docs/sim_diverse_grasps.jpg)

## Installation and preflight

The maintained environment is Linux x86-64, Python 3.10, an NVIDIA GPU, and a driver compatible with CUDA 12.1. The headless control simulation itself uses MuJoCo on CPU; the full `dummy_arm_qpos` stage uses CUDA IK.

```bash
git clone --recurse-submodules https://github.com/ada-grasp-ctrl/ada-grasp-ctrl.git
cd ada-grasp-ctrl
conda env create -f environment.yml
conda activate ada-grasp-ctrl

python -c "import torch, mujoco, pinocchio, ada_grasp_ctrl; print(torch.__version__, torch.version.cuda)"
ada-grasp-ctrl --help
```

For an existing clone, synchronize the pinned submodules first:

```bash
git submodule sync --recursive
git submodule update --init --recursive --progress
```

`python src/main.py ...` remains a compatible wrapper around the installed `ada-grasp-ctrl` command. Paths are resolved from the repository root, so invocation does not depend on the shell's current directory.

## 60-second quick start

The repository contains an authorized minimal DGN bottle object and one precomputed dummy-arm grasp for each supported hand. Quick mode runs `control_eval → control_stat` headlessly and writes only below `output/example_<hand>`:

```bash
bash script/run_example.sh shadow quick
bash script/run_example.sh allegro quick
bash script/run_example.sh leap_tac3d quick
```

The historical wrapper names invoke the same quick mode:

```bash
bash script/test_learning_dummy_arm_shadow.sh
bash script/test_learning_dummy_arm_allegro.sh
bash script/test_learning_dummy_arm_leap_tac3d.sh
```

Full mode starts from the included raw Learning fixture, formats it, runs CUDA IK, then evaluates and summarizes it:

```bash
bash script/run_example.sh shadow full
```

At completion the script prints the output directory, episode status, lift result, statistics file, and success-rate denominator. Exit code `1` means all possible samples were processed but at least one execution error or solver degradation occurred; inspect the reports described below.

The bundled data is not relicensed by the source MIT License. Its source, authorization scope, and checksums are recorded in [the object attribution](examples/assets/object/core_bottle_15787789482f045d8add95bf56d3d2fa/ATTRIBUTION.md).

## Complete four-stage pipeline

### 1. Convert raw grasps

```bash
ada-grasp-ctrl setting=tabletop hand=shadow task=format exp_name=learn \
  task.data_name=Learning task.max_num=100 task.data_path=<RAW_GRASP_DIRECTORY>
```

`task.data_name` is one of `BODex`, `Learning`, or `Batched`. `task.max_num<=0` processes all inputs. An empty raw input is a preflight error. New outputs include `schema_version: 1`; legacy files without a version continue to load as v0.

### 2. Calculate dummy-arm qpos

```bash
ada-grasp-ctrl setting=tabletop hand=shadow task=dummy_arm_qpos exp_name=learn \
  task.max_num=-1 task.device=cuda:0
```

The six-DoF dummy arm maps a sampled hand base pose to simulation joints. Outputs are generated below the explicit `dummy_arm_grasp_dir`; the implementation never derives paths by replacing hand-name substrings.

### 3. Evaluate a controller

```bash
ada-grasp-ctrl setting=tabletop hand=dummy_arm_shadow task=control_eval exp_name=learn \
  task.method=ours task.input_data=grasp_dir task.offsets='[0.0]' \
  task.debug_viewer=false
```

`task.method` accepts `ours`, `op`, `bs1`, `bs2`, or `bs3`. `ours` and `bs2` share one implementation; BS2's only policy difference is disabling dummy-arm motion in Stage 1. Paper hyperparameters, friction coefficients, controller objectives, and lift threshold are unchanged. The corrected wrench-balance gradient and rejected-solver fallback can intentionally change trajectories that depended on the former invalid derivative or an infeasible solution.

For native interactive visualization:

```bash
ada-grasp-ctrl setting=tabletop hand=dummy_arm_shadow task=control_eval exp_name=learn \
  task.method=ours task.debug_viewer=true task.debug_viewer_backend=mujoco
```

For a persistent browser viewer on a headless server:

```bash
ada-grasp-ctrl setting=tabletop hand=dummy_arm_shadow task=control_eval exp_name=learn \
  task.method=ours task.debug_viewer=true task.debug_viewer_backend=mjviser \
  task.mjviser.host=127.0.0.1 task.mjviser.port=8080
ssh -L 8080:127.0.0.1:8080 <server>
```

Use the actual URL printed by mjviser if the preferred port was occupied.

### 4. Compute statistics

```bash
ada-grasp-ctrl setting=tabletop hand=dummy_arm_shadow task=control_stat exp_name=learn \
  task.method=ours task.setting_name=dist_0
```

The historical YAML keys remain available. New counts distinguish `success`, `failure`, `invalid_initialization`, `solver_degraded`, and `execution_error`. The primary success-rate denominator is `success + failure`; invalid initialization and solver degradation are excluded. Empty or all-invalid result sets use YAML `null` for undefined rates and continuous metrics, never NaN.

## Data schemas and diagnostics

A formatted grasp record requires:

- `obj_path`, `obj_scale`, and a seven-value WXYZ `obj_pose`;
- one-dimensional, equally sized `pregrasp_qpos`, `grasp_qpos`, and `squeeze_qpos`;
- optional `joint_names`, whose length must equal qpos length;
- `schema_version: 1` for new files.

A control record keeps the historical trajectories (`obj_pose`, `dof`, `doa`, `contacts`, and planned/optimization fields) and adds `schema_version`, `episode_status`, and `solver_diagnostics`. A rejected SLSQP result is never applied: that control step holds the previous qpos, zeros delta/history, continues the episode, marks it `solver_degraded`, and causes the completed batch to exit `1`.

Every task log directory contains:

- `run_manifest.yaml`: resolved config, roots, seed, workers, git state, dependencies, hardware, and sorted inputs;
- `run_report.json`: totals and one structured result per input;
- `failures.jsonl`: execution-error and solver-degraded records with messages and tracebacks.

Process exit codes are stable:

- `0`: program execution succeeded; scientifically invalid initializations may exist;
- `1`: all possible samples finished, with an execution error or solver degradation;
- `2`: configuration, input, assets, environment, or viewer preflight failed.

With `skip=true`, a batch in which every expected output already exists is a successful no-op and still writes reports.

## Full datasets and release benchmark

For BODex/DGN evaluation, download `DGN_2k_processed.zip` from the [BODex dataset](https://huggingface.co/datasets/JiayiChenPKU/BODex) and arrange it as documented by that dataset. The maintained release gate uses three hands × five methods on one fixed sample plus 100 `ours` episodes per hand. Historical release classifications are:

| Hand | Success | Failure | Invalid initialization |
|---|---:|---:|---:|
| Shadow | 75 | 4 | 21 |
| Allegro | 80 | 6 | 14 |
| LEAP Tac3D | 88 | 5 | 7 |

After correcting the normal derivative at `fx=0` and rejecting infeasible SLSQP results, the intermediate corrected baseline was:

| Hand | Success | Failure | Invalid initialization | Solver degraded |
|---|---:|---:|---:|---:|
| Shadow | 68 | 5 | 21 | 6 |
| Allegro | 80 | 5 | 14 | 1 |
| LEAP Tac3D | 88 | 5 | 7 | 0 |

The current promoted baseline also sorts collision meshes before MuJoCo geom declaration and uses diagnosed direct linear solves instead of `inv(A) @ B`, making trajectories independent of filesystem directory iteration order while avoiding explicit matrix inversion:

| Hand | Success | Failure | Invalid initialization | Solver degraded | Execution error |
|---|---:|---:|---:|---:|---:|
| Shadow | 69 | 4 | 21 | 6 | 0 |
| Allegro | 80 | 5 | 14 | 1 | 0 |
| LEAP Tac3D | 88 | 5 | 7 | 0 | 0 |

All invalid-initialization and solver-degraded classifications are unchanged. The seven degraded episodes previously consumed infeasible/nonconverged solutions; two borderline episodes changed scientific outcome under the corrected derivative (one Shadow success became failure, while one Allegro failure became success). Stable mesh declaration then changed one Shadow sample from failure to success. Direct solve changed 248/300 closed-loop trajectories because last-bit linear-algebra differences are amplified by contact dynamics, but changed no classifications. Two independent 300-episode runs reproduced every promoted trajectory and classification within the strict tolerances below.

The three-hand × five-method fixed matrix uses a strict golden comparison: timing and approved additive metadata are ignored, while keys, shapes, stages, contact order, classifications, and floating trajectories must match (`rtol=1e-5`, `atol=1e-6`). The checked-in [golden evidence](release/golden/README.md) contains the 15 raw trajectories plus a machine-readable audit of both 15-case and 300-case repeat runs, input/output checksums, run manifests, and old-to-new differences. Verify the checked-in evidence with:

```bash
PYTHONPATH=src python script/audit_golden.py verify release/golden/artifact.json
```

One single-process Shadow `ours` episode on the pinned environment gives the following implementation benchmark. These one-run values are a release sanity check, not a hardware-independent performance guarantee.

| Metric | Pre-refactor | Refactored |
|---|---:|---:|
| Wall time | 5.41 s | 4.60 s |
| Recorded solver time | 1.154 s | 1.175 s |
| Peak RSS | 637 MiB | 483 MiB |

The incremental phase-4 direct-solve benchmark compared the immediately preceding optimization refactor with the final implementation on the same fixed Shadow episode: wall time changed from 5.71 s to 5.60 s, peak RSS from 483,972 KiB to 482,808 KiB, and aggregate fixed-matrix optimization time from 6.712 s to 6.804 s (+1.37%).

## Static grasp visualization

The visualizer accepts paths on the command line and can open a window or export a scene:

```bash
PYTHONPATH=src python script/quick_grasp_vis/vis_dexlearn_grasp.py \
  --hand shadow --grasp examples/data/shadow/formatted/grasp.npy \
  --object-root examples/assets/object/core_bottle_15787789482f045d8add95bf56d3d2fa \
  --export /tmp/shadow_grasp.glb
```

## Extending the application

- A new converter must validate its raw schema, write the common v1 grasp schema, and be registered in `CONVERTER_REGISTRY`.
- A new controller implements or configures an episode policy and is registered in `METHOD_REGISTRY`; dynamic `eval(...)` dispatch is intentionally prohibited.
- A new hand must define robot/MJCF metadata, qpos/joint-order tests, converter fixtures, headless integration coverage, and a release golden before it becomes public.

## Development

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl python -m unittest discover -s tests -v
python -m compileall -q src tests
ruff check src tests script
ruff format --check src tests script
```

CI runs Python 3.10 lint/format checks, unit/schema/CLI tests, and a precomputed Shadow headless smoke test. GPU full examples and release golden suites remain release gates.

## License and acknowledgements

Ada Grasp Ctrl source is licensed under the [MIT License](LICENSE), copyright 2026 Ada Grasp Ctrl Authors. Third-party submodules and example data retain their own licenses and attribution. The evaluation codebase builds upon [DexGraspBench](https://github.com/JYChen18/DexGraspBench); the example object comes from DGN/BODex as attributed above.
