# Control and Numerical Knowledge

Load this note before modifying contact semantics, controller lifecycle, optimization validation, or numerical linear algebra.

## Controller organization

`control_eval` supports the tabletop dummy-arm variants of Shadow, Allegro, and LEAP Tac3D. It discovers grasp records deterministically, preflights shared object assets, seeds each stable sample index, and dispatches through `METHOD_REGISTRY`.

The shared episode runner owns approach/squeeze interpolation, state/contact sampling, actuator stepping, stage progression, and trajectory recording. Method classes should contain policy choices and hooks. `ours` and `bs2` intentionally share the coordinated-control implementation; BS2 disables dummy-arm motion in Stage 1 and should not drift in unrelated behavior.

## Contact invariants

MuJoCo may report the hand as either contact geom. Project code canonicalizes hand-object contacts so this incidental ordering does not change the local frame, normal-force sign, or object world wrench. Regression coverage should include both geom orders, tangential and torsional components, finite/right-handed frames, and agreement with MuJoCo constraint forces.

Mesh enumeration is sorted before geom declaration. This is scientifically important: filesystem iteration order once changed geom/contact order and changed one release classification.

## Solver rejection policy

Command-producing SLSQP results are always diagnosed independently of the configured failure policy. `apply_candidate` is the default: it applies a rejected vector when its dimension is correct and every value is finite; solver acceptance remains false and the episode remains degraded. `hold_current` holds the sampled actuator qpos, clears delta/history, and continues. `fail_episode` applies no failing-step action, saves the partial trajectory, and skips lift for that offset. Missing, malformed, incorrectly sized, NaN, or infinite candidates never reach the simulator. The enclosing batch exits `1` for every rejected command solve.

The `wrench_balance` diagnostic solve and diagnosed linear-system fallbacks keep their dedicated behavior and are not controlled by the command-solver failure policy.

## Linear algebra experience

Contact stiffness uses diagnosed `solve(A, B)` rather than explicit `inv(A) @ B`. Singular or poorly conditioned systems need a finite least-squares/zero fallback with diagnostics. Last-bit differences can amplify through closed-loop contact dynamics, so a mathematically sound local rewrite may change many trajectories even when classifications are unchanged. Separate such a change from refactoring, locate the first numerical divergence, run strict fixed/release comparisons, and record the rationale before promotion.

Do not suppress `RuntimeWarning` for an entire controller module. If SciPy emits a known trial-step warning at the solver boundary, scope handling to that call and retain evidence in diagnostics.
