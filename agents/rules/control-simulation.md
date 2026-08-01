# Control and Simulation Rules

Apply this rule to MuJoCo models, contact extraction, robot/hand mappings, controller policies, optimization, IK, and scientific metrics.

## Scientific contract

- Treat configured paper hyperparameters, friction coefficients, object mass, control objectives, stage transitions, perturbation layout, lift threshold, and success definitions as frozen unless the maintainer approves a scientific change.
- Keep the maintained control scope to the tabletop dummy-arm Shadow, Allegro, and LEAP Tac3D hands and the five registered methods.
- Preserve configured joint names/order and the mapping between stored data DOFs, simulator DOFs, and actuator DOAs. Do not infer order from incidental dictionary, XML, or filesystem iteration.
- Keep collision-mesh declaration and all discovered simulation inputs deterministically sorted.

## Contacts and numerics

- Canonicalize hand-object contacts so geom declaration order cannot change contact semantics. Contact frames must be finite, orthonormal, and right-handed; recorded contact wrenches remain six-dimensional and aligned with control steps.
- Do not change frames, signs, or force/torque ordering without tests covering both hand/object geom orders and world-wrench equivalence.
- Diagnose every optimizer result for solver success, finite values, and constraint feasibility before making a policy decision.
- A rejected command candidate may be applied only under the resolved `apply_candidate` Hydra policy, which is the public default, and only when its dimension is correct and all values are finite. Missing, malformed, incorrectly sized, NaN, or infinite candidates never reach MuJoCo.
- On every rejected command solve, follow the configured apply/hold/abort policy, preserve the rejected diagnostic, mark the episode `solver_degraded`, and make the batch exit nonzero. Hold mode clears delta/history; abort mode saves a partial record and skips lift for that offset.
- Prefer diagnosed linear solves over explicit matrix inversion. Singular or ill-conditioned systems require a finite, recorded fallback; never hide them with broad warning filters.
- Limit warning suppression to a known library boundary and retain a diagnostic count or equivalent evidence.

## Structural changes

- Keep common episode initialization, sampling, stepping, stage progression, and recording in the shared runner. Method classes should express policy differences rather than copy the lifecycle.
- Keep `ours` and `bs2` aligned except for BS2's documented Stage-1 dummy-arm-motion difference.
- Separate numerical changes from refactors. A refactor must first demonstrate unchanged keys, shapes, stages, contact ordering, classifications, and trajectories within approved tolerances.
- A new hand requires robot/MJCF metadata, exact joint-order coverage, converter fixtures, headless integration coverage, provenance review, and a release golden before public support is claimed.

## Load on demand

- Read [../knowledge/control-numerics.md](../knowledge/control-numerics.md) before changing contact logic, solver handling, the shared episode runner, or linear algebra.
- Read [../knowledge/testing-release.md](../knowledge/testing-release.md) before judging any trajectory or classification difference.
