# TODO: Add mjviser Visualization to the 60-Second Quick Start

## Status

Open. This task may be closed only after every acceptance criterion below has been verified.

## Goal

Add a first-class, one-command mjviser path to the 60-second quick start so new users can watch the grasp execution in a browser and understand the controller behavior instead of seeing only terminal output and final statistics.

## Current Behavior

- `README.md` documents `bash script/run_example.sh <hand> quick` as a headless workflow.
- `script/run_example.sh` hard-codes `task.debug_viewer=false`.
- The mjviser backend already exists, but users must leave the quick-start flow and compose a separate `control_eval` command to use it.
- The existing headless quick path is used by CI and release automation and must remain non-interactive.

## Required Work

1. Extend `script/run_example.sh` with an explicit optional viewer interface. The public command must be:

   ```bash
   bash script/run_example.sh <shadow|allegro|leap_tac3d> quick --viewer mjviser
   ```

2. Keep the existing two-argument command headless by default. `--viewer none` may be supported as an explicit alias for the default.
3. When mjviser is selected, pass the existing Hydra viewer settings to `control_eval`:

   ```text
   task.debug_viewer=true
   task.debug_viewer_backend=mjviser
   ```

4. Preserve controller parameters, stage semantics, seeds, deterministic input ordering, output layout, reports, statistics, and exit-code behavior.
5. Validate viewer arguments before starting any pipeline stage. Unsupported viewer names or malformed arguments must print actionable usage text and exit with code `2`.
6. Update the README 60-second quick-start section so browser visualization is immediately discoverable. Document:
   - the exact local command;
   - that simulation waits for the first browser client;
   - where the actual viewer URL is printed;
   - how to use the printed SSH port-forwarding command on a remote or headless server;
   - what the user should observe during grasp execution;
   - how to run the unchanged headless form for automation.
7. Add regression tests for argument validation, default headless behavior, and the exact Hydra viewer overrides passed by the script. Automated tests must not require a human browser client.

## Non-Goals

- Do not change paper hyperparameters, friction, controller objectives, lift thresholds, stage semantics, success definitions, or golden trajectories.
- Do not enable an interactive viewer in CI, release gates, or the default headless command.
- Do not add or change dependencies; the pinned `mjviser` and `viser` packages already provide this capability.
- Do not change the supported hand, method, or converter matrix.

## Acceptance Criteria

- [ ] `bash script/run_example.sh shadow quick --viewer mjviser` starts one mjviser server and prints `http://<host>:<actual-port>` before simulation.
- [ ] The visual quick start waits for the first browser connection and then shows the live grasp execution through completion.
- [ ] The same `--viewer mjviser` option works for the Shadow, Allegro, and LEAP Tac3D quick fixtures.
- [ ] If port `8080` is occupied, the command prints and uses the actual fallback port. The displayed URL and SSH forwarding hint use the same actual port.
- [ ] `bash script/run_example.sh <hand> quick` remains headless and non-interactive for all three supported hands.
- [ ] The existing unique-run-directory, current-invocation report/statistics, and documented exit-code contracts remain unchanged.
- [ ] Invalid or malformed viewer arguments fail before `control_eval` with exit code `2` and actionable usage text.
- [ ] Automated tests prove that the default command passes `task.debug_viewer=false`.
- [ ] Automated tests prove that the mjviser command passes `task.debug_viewer=true` and `task.debug_viewer_backend=mjviser`.
- [ ] The README 60-second quick start contains a copy-pasteable local mjviser command, remote SSH guidance, expected viewer behavior, and the headless alternative.
- [ ] The implementing pull request includes visual evidence, such as a short recording or screenshots, showing the Shadow browser viewer at the initial grasp scene and at a later control or lift state.
- [ ] The focused viewer and quick-script tests pass.
- [ ] The canonical repository checks below pass without weakening tests or tolerances:

  ```bash
  PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl python -m unittest tests.test_debug_viewer tests.test_release_flow -v
  PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl python -m unittest discover -s tests -v
  python -m compileall -q src tests
  ruff check src tests script
  ruff format --check src tests script
  ```

## Required Verification Evidence

The implementation or pull request must record:

- the exact commands executed;
- the focused and full test results;
- headless quick-run results for Shadow, Allegro, and LEAP Tac3D;
- an mjviser visual smoke-test result for each supported hand;
- visual evidence for at least the Shadow quick example;
- confirmation that CI and release-gate quick runs remain headless.

## Closure Rule

Do not close this TODO based only on README changes, the existing low-level mjviser unit tests, or a successful server startup. Close it only after the end-to-end quick-start entry point, automated regression coverage, documentation, visual evidence, three-hand smoke verification, and all acceptance criteria above are complete and verified.
