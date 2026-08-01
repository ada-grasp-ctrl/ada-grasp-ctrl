# TODO: Add Hydra-Selectable Failed Control-Solve Policies

## Status

Open. This task may be closed only after every acceptance criterion below has been verified.

## Goal

Make the control-command SLSQP failure behavior selectable through Hydra. The controller must support:

1. applying the solver candidate even when the solve is rejected;
2. holding the current actuator position and continuing; and
3. aborting the current episode immediately.

After the three-policy, three-hand evaluation, `apply_candidate` is the authorized default. Unchanged configurations may therefore change degraded trajectories, while rejected solves remain auditable and classified as `solver_degraded`.

## Pre-Implementation Behavior

`diagnose_slsqp_result(...)` currently accepts an SLSQP result only when all of the following conditions hold:

- `result.success` is true;
- the candidate variables, objective, and evaluated constraints are finite;
- equality and inequality constraints satisfy their configured tolerances;
- scalar bounds satisfy their configured tolerance; and
- the joint-limit constraint satisfies its configured tolerance.

`select_control_solution(...)` applies an accepted candidate. For every rejected candidate, it currently:

- uses the current actuator qpos as the next target;
- returns a zero applied delta;
- preserves the currently measured contact forces for solver history;
- continues the episode and lift sequence; and
- latches `solver_degraded`, which makes the output status `solver_degraded` and the completed batch exit with code `1`.

This is precisely a hold-current-qpos behavior. It does not replay a separately cached previous target action.

## Public Configuration Contract

Add the following field below `task.control` in `src/ada_grasp_ctrl/config/task/control_eval.yaml`:

```yaml
control:
  solver_failure_policy: apply_candidate
```

The supported values are:

```text
apply_candidate
hold_current
fail_episode
```

Users must be able to select them through Hydra:

```bash
task.control.solver_failure_policy=apply_candidate
task.control.solver_failure_policy=hold_current
task.control.solver_failure_policy=fail_episode
```

`apply_candidate` is the default. The field remains additive and stored records remain readable, but commands that omit it now intentionally apply runtime-applicable rejected candidates.

An unsupported value is a task-level configuration error. It must fail during control preflight, before worker or simulation startup, print an actionable message containing the supported values, and produce process exit code `2`.

## Definitions

### Accepted control solve

A control solve is accepted when the existing SLSQP validator reports `diagnostics["accepted"] == true`.

### Rejected control solve

A control solve is rejected when `diagnostics["accepted"] == false`. Rejection includes any of the following:

- `result.success == false`;
- a missing or malformed solver candidate;
- a nonfinite candidate, objective, or constraint evaluation;
- equality-constraint residual above tolerance;
- inequality-constraint slack below tolerance;
- scalar-bound violation above tolerance; or
- joint-limit violation above tolerance.

### Runtime-applicable candidate

A candidate is runtime-applicable when `result.x`:

- can be converted to the expected numeric vector;
- has exactly the expected total variable dimension; and
- contains only finite values.

Runtime applicability is intentionally weaker than solver acceptance. In `apply_candidate` mode, a runtime-applicable candidate may be applied even if the solver reported failure or feasibility, bound, or joint-limit checks failed.

A missing, malformed, incorrectly sized, NaN, or infinite candidate cannot be sent to MuJoCo under any policy because it does not define an executable actuator action.

## Required Policy Semantics

The policy is evaluated only after the result has been diagnosed and the original diagnostic values have been recorded.

| Solver outcome | `apply_candidate` | `hold_current` | `fail_episode` |
| --- | --- | --- | --- |
| Accepted result | Apply the candidate normally | Apply the candidate normally | Apply the candidate normally |
| Rejected, runtime-applicable candidate | Apply the candidate | Hold the current qpos | Abort the current episode |
| Rejected, non-applicable candidate | Abort the current episode | Hold the current qpos | Abort the current episode |

### `apply_candidate`

For a rejected but runtime-applicable result:

- apply `result.x[:num_dof]` as the actuator delta;
- set the next actuator target to `current_qpos_a + delta_qpos_a`;
- return the remaining candidate values as the solver contact-force vector;
- use the applied delta as the next control step's delta/history contribution;
- continue subsequent control steps;
- execute the lift phase; and
- keep the episode classified as `solver_degraded` with batch exit code `1`.

This policy is explicitly authorized to apply a runtime-applicable candidate when any of the following are true:

- `result.success == false`;
- equality or inequality constraints fail validation;
- scalar bounds fail validation; or
- joint limits fail validation.

The validator's `accepted` field must remain false. Applying the candidate is a policy decision and must not rewrite the original solver assessment.

If the candidate is not runtime-applicable, `apply_candidate` must abort the current episode using the same control flow as `fail_episode`.

### `hold_current`

For every rejected result:

- do not apply the solver candidate;
- set `target_qpos_a` to a copy of `current_qpos_a`;
- return a zero actuator delta of the expected command dimension;
- preserve the currently measured contact-force vector;
- use the zero delta as the next control step's delta/history contribution;
- continue subsequent control steps;
- execute the lift phase; and
- classify the episode as `solver_degraded` with batch exit code `1`.

This policy remains available as the explicit conservative alternative.

### `fail_episode`

For every rejected result:

- do not apply an actuator action for the failing control step;
- stop the current object-offset episode immediately;
- do not execute subsequent control steps;
- do not execute the lift phase;
- save the partial trajectory and complete solver diagnostics collected up to the failure;
- classify that output as `solver_degraded`; and
- continue with other configured offsets and other batch samples when they can run safely.

`fail_episode` must not be converted into a generic `execution_error`. It must also not be classified as the scientific `failure` outcome because the episode did not complete the lift criterion. Using scientific `failure` would incorrectly add the aborted episode to the `success + failure` success-rate denominator.

## Scope

This task applies only to the SLSQP calls that produce actuator commands:

- `solver="control"`, used by `ours` and `bs2`; and
- `solver="control_bs3"`, used by `bs3`.

The configured policy must not change the behavior of:

- the `wrench_balance` diagnostic SLSQP solve;
- `linear:*` diagnosed linear-system fallbacks;
- the `op` controller;
- the `bs1` controller;
- control objectives, constraints, tolerances, or paper hyperparameters;
- stage-switch criteria or lift criteria;
- deterministic input ordering or per-sample seeding;
- the supported hand, method, or converter matrix; or
- the existing episode-status and process-exit-code vocabulary.

`op` and `bs1` may carry the resolved configuration field, but it has no behavioral effect because those methods do not use the command-producing SLSQP path.

## Status and Reporting Contract

Every rejected control solve must latch solver degradation regardless of the selected policy. Therefore:

- an episode that applies a rejected candidate remains `solver_degraded`;
- an episode that holds the current qpos remains `solver_degraded`;
- an episode that aborts immediately remains `solver_degraded`;
- a batch containing any such episode exits with code `1` after all possible samples finish;
- the degraded sample appears in `run_report.json`; and
- the degraded sample appears in `failures.jsonl`.

`control_stat` must continue excluding `solver_degraded` outputs from the primary `success + failure` denominator. Changing that statistical contract is outside this task.

The existing control-record schema version and episode-status set remain valid. Additive solver-diagnostic fields do not require a schema-version increment.

## Solver Diagnostic Contract

Every command-producing solver diagnostic must preserve the existing solver fields and additionally make the policy decision auditable. At minimum, each diagnostic must contain:

```text
solver
stage
success
accepted
finite
status
message
max_equality_residual
min_inequality_slack
bound_violation
joint_limit_violation
failure_policy
candidate_applicable
decision
action_applied
episode_aborted
```

`decision` must use one of these stable values:

```text
apply_accepted
apply_candidate
hold_current
abort_episode
```

Required relationships include:

- an accepted result records `decision=apply_accepted`, `action_applied=true`, and `episode_aborted=false` under every configured policy;
- a rejected candidate applied by policy records `accepted=false`, `decision=apply_candidate`, `action_applied=true`, and `episode_aborted=false`;
- a held result records `accepted=false`, `decision=hold_current`, `action_applied=true`, and `episode_aborted=false`; and
- an aborted result records `accepted=false`, `decision=abort_episode`, `action_applied=false`, and `episode_aborted=true`.

Holding the current actuator position counts as applying a deliberate safe command, so `action_applied` is true for `hold_current`.

## Required Implementation Behavior

1. Keep solver diagnosis separate from the policy decision. Do not redefine `accepted` based on the configured policy.
2. Separate candidate runtime applicability from feasibility and solver-success checks.
3. Avoid unconditional access to `result.x` after diagnosis; malformed results must remain attributable and must not become accidental `AttributeError` execution failures.
4. Route `apply_candidate`, `hold_current`, and `fail_episode` through one shared decision implementation used by both coordinated control and BS3 equal-contact control.
5. Preserve the alignment of recorded `obj_pose`, `dof`, `doa`, `contacts`, `planned_dof`, stages, and optimization fields.
6. Use an explicit domain-level abort result or exception for `fail_episode`. Catch it inside the per-offset evaluation lifecycle so partial output can be saved as `solver_degraded` without being converted to `execution_error` by `safe_eval_one(...)`.
7. Abort only the current offset episode. Do not prevent later offsets or independent batch samples from running.
8. Do not run the lift sequence after an episode-level solver abort.
9. Preserve current warning handling and the recorded SLSQP bound-clipping warning count.
10. Keep `ours` and `bs2` aligned except for BS2's existing Stage-1 arm-motion difference.

## Non-Goals

- Do not change SLSQP objectives, gradients, constraints, bounds, tolerances, initial points, or maximum iterations.
- Do not change contact frames, wrench signs, stiffness models, or force ordering.
- Do not add a new scientific `failure` category or a new batch status.
- Do not change success-rate definitions or denominators.
- Do not update or replace golden trajectories merely because a non-default policy intentionally produces different trajectories.
- Do not promote golden data for non-default policies as part of this task.
- Do not change dependencies or pinned third-party submodules.
- Do not refactor unrelated controller or episode-runner behavior.

## Acceptance Criteria

- [ ] `task.control.solver_failure_policy` exists and defaults to `apply_candidate`.
- [ ] Existing commands that do not specify the field continue to compose and run.
- [ ] `apply_candidate`, `hold_current`, and `fail_episode` are all accepted Hydra values.
- [ ] An unsupported value fails during preflight, before simulation or worker startup, with exit code `2` and an actionable list of supported values.
- [ ] An accepted solver result produces the same actuator target, delta, contact-force result, and degradation state under all three policies.
- [ ] `apply_candidate` applies a finite, correctly sized candidate when `result.success == false`.
- [ ] `apply_candidate` applies a finite, correctly sized candidate when constraint validation fails.
- [ ] `apply_candidate` applies a finite, correctly sized candidate when bound validation fails.
- [ ] `apply_candidate` applies a finite, correctly sized candidate when joint-limit validation fails.
- [ ] Applying a rejected candidate does not change the diagnostic `accepted` value to true.
- [ ] No policy sends a missing, malformed, incorrectly sized, NaN, or infinite candidate to MuJoCo.
- [ ] A non-applicable candidate in `apply_candidate` mode aborts the current episode rather than applying or silently repairing it.
- [ ] `hold_current` returns the current actuator qpos, a zero delta, and the measured contact forces for every rejected result.
- [ ] `hold_current` continues subsequent control steps and executes the lift phase.
- [ ] `fail_episode` applies no action for the failing step.
- [ ] `fail_episode` stops subsequent control steps and skips the lift phase.
- [ ] `fail_episode` saves a structurally valid partial control record with `episode_status=solver_degraded`.
- [ ] An aborted offset does not prevent later offsets or other samples from running.
- [ ] All rejected command solves remain visible in `solver_diagnostics` with the configured policy and actual decision.
- [ ] All three rejected-solve behaviors produce `solver_degraded`, appear in `failures.jsonl`, and cause batch exit code `1`.
- [ ] No rejected-solve policy is reported as scientific `failure` or added to the success-rate denominator.
- [ ] `wrench_balance`, `linear:*`, `op`, and `bs1` behavior is unchanged.
- [ ] The default `apply_candidate` configuration preserves accepted-solve fixed-golden keys, shapes, stages, contacts, classifications, and numeric trajectories within `rtol=1e-5` and `atol=1e-6`.
- [ ] No golden baseline is modified to make the implementation pass.
- [ ] Focused unit and lifecycle tests pass.
- [ ] The complete unit suite, compile checks, Ruff checks, three-hand quick smoke tests, and portable release gate pass.

## Acceptance Tests

### Solver diagnosis and decision unit tests

Extend `tests/test_optimization.py` with deterministic SciPy-like results covering the following matrix.

#### Accepted result

For each configured policy:

- apply the candidate;
- return the candidate delta and contact-force portion;
- record `decision=apply_accepted`;
- do not latch solver degradation; and
- produce identical numeric outputs across policies.

#### Rejected but feasible finite result

Use `success=false` with a correctly sized, finite candidate that otherwise satisfies constraints and bounds:

- `apply_candidate` applies the candidate and records `decision=apply_candidate`;
- `hold_current` returns current qpos, zero delta, and measured forces;
- `fail_episode` returns or raises the explicit episode-abort signal; and
- all three paths latch solver degradation.

#### Rejected infeasible result

Independently inject:

- equality-constraint violation;
- inequality-constraint violation;
- scalar-bound violation; and
- joint-limit violation.

For each case, prove that:

- `apply_candidate` applies the finite, correctly sized candidate;
- the original diagnostic remains `accepted=false`;
- the exact rejection metrics remain recorded; and
- the policy decision fields are correct.

#### Non-applicable result

Independently inject:

- a result without `x`;
- a non-numeric `x`;
- a vector with the wrong length;
- a vector containing NaN; and
- a vector containing positive or negative infinity.

Prove that:

- `apply_candidate` aborts the episode;
- `hold_current` returns only finite held values;
- `fail_episode` aborts the episode;
- no invalid candidate is passed to the simulated actuator interface; and
- malformed results do not become unstructured `AttributeError` execution failures.

#### Warning preservation

Retain the existing test proving that only SciPy's documented bound-clipping `RuntimeWarning` is handled locally, its count is recorded, and unrelated warnings remain visible.

### Episode-lifecycle tests

Extend `tests/test_episode_runner.py` or add a focused evaluator lifecycle fixture that uses a deterministic fake simulator and injected failed solver result.

Prove that:

- `apply_candidate` sends the expected candidate-derived target to `ctrl_qpos_a_with_interp(...)` and continues;
- `hold_current` sends the current qpos, continues later steps, and reaches lift;
- `fail_episode` does not call `ctrl_qpos_a_with_interp(...)` for the failing step;
- `fail_episode` makes no later policy calls;
- `fail_episode` does not enter the lift command sequence;
- partial trajectory arrays remain aligned; and
- the saved episode status is `solver_degraded`.

Add a multiple-offset fixture proving that an abort terminates only the current offset and the next offset is still evaluated.

### Hydra and preflight tests

Extend `tests/test_pipeline_contracts.py` or an equivalent focused configuration test to prove that:

- the default composed value is `apply_candidate`;
- all three supported overrides compose successfully;
- an unsupported value fails before evaluator construction and worker startup;
- the error names the invalid value and all supported values; and
- the CLI process exit code is `2`.

### Batch and statistics tests

Use deterministic degraded sample results to prove that every rejected-solve policy:

- produces a `solver_degraded` output status;
- increments `num_solver_degraded`;
- writes the sample to `failures.jsonl`;
- causes `raise_for_batch_failures(...)` and CLI exit code `1`; and
- remains excluded from the `success + failure` denominator.

Prove separately that `fail_episode` is not converted to `execution_error`.

### Regression and integration validation

Run the focused tests first:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl \
python -m unittest tests.test_optimization tests.test_episode_runner tests.test_pipeline_contracts -v
```

Then run the canonical repository checks:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/ada_grasp_ctrl_mpl python -m unittest discover -s tests -v
python -m compileall -q src tests
ruff check src tests script
ruff format --check src tests script
```

Run the default-policy headless quick smoke for every supported hand, using a unique empty output root for each invocation:

```bash
bash script/run_example.sh shadow quick
bash script/run_example.sh allegro quick
bash script/run_example.sh leap_tac3d quick
```

Because the implementation touches a trajectory-producing control path, run the portable release gate:

```bash
PYTHON_BIN=python bash script/run_release_gate.sh portable
```

The default `apply_candidate` gate must match promoted fixed trajectories whose command solves are accepted at `rtol=1e-5` and `atol=1e-6`. Degraded release trajectories may intentionally differ and must not be promoted without separate approval.

Non-default failure branches must be tested with deterministic injected solver results. Acceptance must not depend on a real optimizer happening to fail during a smoke run.

Do not run `release300` unless the external release input tree is available and the maintainer separately requests that gate.

## Implementation Plan

1. Add `solver_failure_policy: apply_candidate` to the control-eval Hydra configuration with an English comment listing the supported values.
2. Define a single stable set of supported policy names in project-owned Python code.
3. Validate the resolved policy in `_validate_control_preflight(...)` before worker or simulator startup.
4. Extend SLSQP diagnosis so the normalized candidate and `candidate_applicable` state are available without unsafe unconditional access to `result.x`.
5. Keep `accepted` as the policy-independent solver-validation result.
6. Replace the binary accepted/hold selector with one shared decision function that implements all three policies and returns explicit action/history/abort metadata.
7. Route both `ctrl_opt(...)` and `ctrl_opt_bs3(...)` through the same decision function.
8. Record the configured policy and actual decision in every command-solver diagnostic.
9. Introduce an explicit domain-level episode-abort signal for `fail_episode` and non-applicable `apply_candidate` results.
10. Catch that signal inside the per-offset evaluator lifecycle, before lift, save the partial record as `solver_degraded`, and return normally to the surrounding offset/sample loop.
11. Preserve the existing batch aggregation, `failures.jsonl`, statistics exclusion, and exit-code behavior.
12. Add the solver, lifecycle, Hydra, batch, and statistics tests specified above.
13. Update `README.md` to document the new field, all three values, the default behavior, the intentional safety-contract difference, and the unchanged degraded-status semantics.
14. Run focused tests, canonical checks, three-hand quick smoke tests, and the portable release gate.
15. Inspect `git diff --check`, the final diff, generated-output cleanliness, and the worktree to ensure unrelated user changes were preserved.

## Required Verification Evidence

The implementing pull request or task record must include:

- the exact commands executed;
- focused-test results;
- full unit, compile, Ruff check, and Ruff format results;
- the resolved Hydra configuration for each supported policy;
- deterministic evidence for every row of the policy decision table;
- evidence that invalid candidates never reach the actuator interface;
- evidence that `fail_episode` skips lift and preserves partial records;
- `run_report.json` and `failures.jsonl` evidence for degraded behavior;
- default quick-smoke results for Shadow, Allegro, and LEAP Tac3D;
- the portable gate comparison result; and
- confirmation that no golden baseline, third-party submodule, or generated output was modified.

## Closure Rule

Do not close this task based only on adding the Hydra field or unit-testing the selection helper. Close it only after all three policies are wired through the real shared command-solver path, failure and abort behavior is recorded end to end, invalid candidates are prevented from reaching MuJoCo, default behavior is proven golden-compatible, all required tests and gates pass, documentation is updated, and every acceptance criterion is verified.
