# Coordinated Contact Control for Adaptive Dexterous Grasping Under Uncertainty

[Project website](https://ada-grasp-ctrl.github.io/)

```text
raw grasp -> format -> dummy_arm_qpos -> control_eval -> control_stat
```

Supported hands: Shadow, Allegro, and LEAP Tac3D. Supported controllers: `ours`, `op`, `bs1`, `bs2`, and `bs3`. Supported input converters: BODex, Learning, and Batched.

## Installation

The maintained environment is Linux x86-64 with Python 3.10. Quick control simulation runs on CPU; the complete pipeline requires an NVIDIA GPU compatible with CUDA 12.1 for `dummy_arm_qpos`.

```bash
git clone --recurse-submodules https://github.com/ada-grasp-ctrl/ada-grasp-ctrl.git
cd ada-grasp-ctrl
conda env create -f environment.yml
conda activate ada-grasp-ctrl
ada-grasp-ctrl --help
```

For an existing clone, initialize the pinned dependencies before creating or updating the environment:

```bash
git submodule sync --recursive
git submodule update --init --recursive
```

## Quick start

Run the bundled Shadow example headlessly:

```bash
bash script/run_example.sh shadow quick
```

Quick mode uses a precomputed dummy-arm grasp and runs `control_eval -> control_stat`. The other supported hands use the same command:

```bash
bash script/run_example.sh allegro quick
bash script/run_example.sh leap_tac3d quick
```

To watch the Shadow example in a browser:

```bash
bash script/run_example.sh shadow quick --viewer mjviser
```

Open the URL printed by mjviser. On a remote server, use the SSH forwarding command printed beside it.

To run all four stages from the bundled raw Learning fixture:

```bash
bash script/run_example.sh shadow full
```

Each example uses a new directory under `output/examples/<hand>/` and prints the final output path and result summary.

## Complete pipeline

Run the following commands from the repository root. Use the same `exp_name` and `output_root` across all stages.

### 1. Format raw grasps

```bash
ada-grasp-ctrl setting=tabletop hand=shadow task=format exp_name=learn \
  task.data_name=Learning task.data_path=/path/to/raw_grasps task.max_num=-1
```

Set `task.data_name` to `BODex`, `Learning`, or `Batched`. A negative `task.max_num` processes all inputs.

### 2. Calculate dummy-arm qpos

```bash
ada-grasp-ctrl setting=tabletop hand=shadow task=dummy_arm_qpos exp_name=learn \
  task.device=cuda:0 task.max_num=-1
```

### 3. Evaluate a controller

```bash
ada-grasp-ctrl setting=tabletop hand=dummy_arm_shadow task=control_eval exp_name=learn \
  task.method=ours task.input_data=grasp_dir task.offsets='[0.0]' \
  task.debug_viewer=false
```

Set `task.method` to `ours`, `op`, `bs1`, `bs2`, or `bs3`. For browser visualization, set `task.debug_viewer=true task.debug_viewer_backend=mjviser`.

### 4. Compute statistics

```bash
ada-grasp-ctrl setting=tabletop hand=dummy_arm_shadow task=control_stat exp_name=learn \
  task.method=ours task.setting_name=dist_0
```

Use the matching hand name at each stage:

| Format and IK | Control and statistics |
|---|---|
| `shadow` | `dummy_arm_shadow` |
| `allegro` | `dummy_arm_allegro` |
| `leap_tac3d` | `dummy_arm_leap_tac3d` |

## Outputs and status

Outputs default to `<checkout>/output`; override them with `output_root=/path/to/output`.

Each task log directory contains:

- `run_manifest.yaml`: resolved configuration and runtime information.
- `run_report.json`: structured per-input results.
- `failures.jsonl`: execution errors and solver-degraded records, when present.

Exit codes are:

- `0`: execution completed without execution errors or solver degradation.
- `1`: processing completed, but at least one execution error or solver-degraded episode occurred.
- `2`: configuration, input, asset, environment, or viewer preflight failed.

## References and license

- [Implementation notes](docs/practical-modifications-in-implementation.md)
- [Golden release evidence](release/golden/README.md)
- [Example object attribution](examples/assets/object/core_bottle_15787789482f045d8add95bf56d3d2fa/ATTRIBUTION.md)
- [Hand asset audit](assets/hand/README.md)

Project source is licensed under the [MIT License](LICENSE). Third-party submodules, hand assets, and bundled data retain their own licenses and attribution. The example-object authorization record and local LEAP Tac3D mesh provenance remain unresolved for public redistribution; see the linked audits.
