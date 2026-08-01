# Ada Grasp Ctrl Agent Guide

This file is the thin project entry point. Load only the rule files required by the current task; load knowledge notes only when a routed rule points to them and the detail is relevant.

## Project overview

Ada Grasp Ctrl is the reference implementation of coordinated contact control for dexterous grasp execution under object-pose uncertainty. The maintained application is this four-stage pipeline:

```text
raw grasp -> format -> dummy_arm_qpos -> control_eval -> control_stat
```

The public support matrix is Shadow, Allegro, and LEAP Tac3D hands; `ours`, `op`, `bs1`, `bs2`, and `bs3` control methods; and BODex, Learning, and Batched input converters. Treat [README.md](README.md) as the public product contract and the code, configuration, and tests as the executable contract.

## Repository map

- `src/ada_grasp_ctrl/`: installed Python package, CLI, runtime, schemas, tasks, controllers, and robot utilities.
- `src/ada_grasp_ctrl/config/`: Hydra defaults for tasks and hands.
- `tests/`: unit, schema, CLI, simulation, runtime-path, and release-flow contracts.
- `script/`: examples, golden comparison/audit tools, visualization, and release gates.
- `examples/`: redistributable quick-start fixtures and their attribution records.
- `assets/hand/`: project-maintained hand MJCF and local meshes; provenance is not uniformly complete.
- `release/golden/`: promoted fixed-matrix trajectories and the release audit artifact.
- `third_party/`: commit-pinned submodules. Do not edit them as ordinary project source.
- `output/`, `build/`, caches, and task logs: generated artifacts, not source.

## Context loading

Read every rule whose task column matches. Do not bulk-read `agents/knowledge/`.

| Task domain | Required rule | Load deeper knowledge when needed |
| --- | --- | --- |
| General Python, CLI, Hydra config, refactoring, new features | [agents/rules/development.md](agents/rules/development.md) | Architecture or extension-point questions |
| Converters, schemas, paths, packaging, batch reports | [agents/rules/data-runtime.md](agents/rules/data-runtime.md) | Record layouts, root precedence, status semantics |
| MuJoCo, contacts, controllers, optimization, hands | [agents/rules/control-simulation.md](agents/rules/control-simulation.md) | Numerical history or controller lifecycle |
| Tests, benchmark/eval, golden data, CI, release gates | [agents/rules/testing-release.md](agents/rules/testing-release.md) | Gate selection, comparison semantics, baseline history |
| README, public support claims, assets, licenses, attribution | [agents/rules/documentation-assets.md](agents/rules/documentation-assets.md) | Known provenance gaps and release blockers |

When a task spans domains, load all matching rules. Rules are mandatory constraints; knowledge notes are explanatory and may become stale. If a note conflicts with current code, tests, `README.md`, or `release/golden/artifact.json`, verify the current behavior and update the note rather than following it blindly.

## Development entry points

Use the maintained Python 3.10 conda environment from `environment.yml`. Run the application from a source checkout with `python src/main.py`; the package does not install a console command.

Start validation at the smallest relevant scope, then widen in proportion to risk. The canonical repository checks are:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl python -m unittest discover -s tests -v
python -m compileall -q src tests
ruff check src tests script
ruff format --check src tests script
```

Use `bash script/run_example.sh <shadow|allegro|leap_tac3d> quick` for a headless smoke test. Use `script/run_release_gate.sh` only under the release rule.

## Behavior boundaries

### Always do

- Inspect the current worktree and preserve unrelated user changes.
- Read the rules routed for the task before editing code or project documentation.
- Preserve deterministic input ordering, per-sample seeding, structured reports, and documented exit codes.
- Add or update tests for behavior changes and report the exact checks run.
- Keep code comments, docstrings, configuration comments, and project instruction files in English.
- Treat generated output as untrusted until its manifest/report proves it came from the current invocation.

### Ask first

- Changing paper hyperparameters, friction, controller objectives, lift thresholds, stage semantics, or success-rate definitions.
- Breaking public CLI/config fields, record schemas, joint order, output layout, or legacy-read compatibility.
- Promoting or replacing golden trajectories, accepting a classification change, or weakening comparison tolerances.
- Adding/removing dependencies, changing pinned submodules, or changing the supported hand/method/converter matrix.
- Deleting assets, changing redistribution claims, or declaring an unresolved provenance item legally cleared.

### Never do

- Update a golden baseline merely to make a failing comparison pass.
- Swallow worker/solver failures, apply a malformed or nonfinite solver result, apply a rejected result outside its explicit configured policy, or report stale outputs as success.
- Reintroduce dynamic `eval(...)` task/method dispatch or depend on the caller's current working directory.
- Apply module-wide warning suppression to hide numerical problems.
- Modify third-party submodules, generated outputs, or attribution facts unless the task explicitly places them in scope.
