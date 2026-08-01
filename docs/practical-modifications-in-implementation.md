# Practical Modifications in the Implementation

## Purpose

The paper [*CoorGrasp: Coordinated Contact Control for Adaptive Dexterous Grasping Under
Uncertainty*](https://arxiv.org/pdf/2607.03557) presents a compact, idealized formulation of the controller so that
the central ideas are clear: coordination-aware phase separation, arm-hand coordination during approach, and
online force allocation based on wrench balance.

The implementation preserves these core ideas while adding several practical modifications for noisy contact
measurements, bounded robot motion, numerical conditioning, and finite-duration batch evaluation. These additions
make the controller more robust in simulation and deployment without changing the intended coordinated-control
mechanism. This document records the behavior of the current default implementation and clarifies details that are
intentionally abstracted in the paper.

The principal implementation locations are:

- [controller defaults](../src/ada_grasp_ctrl/config/task/control_eval.yaml);
- [coordinated episode policy](../src/ada_grasp_ctrl/tasks/control_eval_func/optimized_runner.py);
- [MPC objectives and constraints](../src/ada_grasp_ctrl/utils/grasp_controller.py);
- [solver diagnostics and fallback policies](../src/ada_grasp_ctrl/optimization.py);
- [MuJoCo contact handling](../src/ada_grasp_ctrl/utils/hand_util.py).

## Relaxed tangential motion-contact consistency

The ideal formulation writes the motion-contact prediction as a full three-dimensional equality:

$$
\Delta \boldsymbol f = \boldsymbol K_s \boldsymbol J \boldsymbol u.
$$

In practice, measured tangential forces may temporarily lie outside the estimated friction cone because of contact
transients, solver/model mismatch, discretization, or sensing noise. Requiring the measured tangential force, the
linearized motion-contact prediction, the desired force, the friction cone, and all joint-motion bounds to agree in
one control step can make the optimization infeasible.

The implementation applies the following stage-specific treatment for the coordinated controller:

- Stage 1 uses the paper's full three-dimensional motion-contact equality by default.
- During Stage 2, the normal component of the motion-contact model remains a hard equality.
- During Stage 2, tangential consistency is represented by a quadratic penalty between the optimized force and the
  force predicted from the measured force and commanded joint motion.
- The optimized force still satisfies the friction cone and the desired total-normal-force constraint.
- During Stage 1, tangential contact-point motion remains strongly penalized by the paper's tangential-motion cost.
- The Stage-1 and Stage-2 contact-model choices are independently configurable, which supports controlled ablations
  without coupling the two phases.

Conceptually, the implemented Stage-2 model is

$$
\left(\boldsymbol f_{t+1}-\boldsymbol f_t-\boldsymbol K_s\boldsymbol J\boldsymbol u_t\right)_n = 0,
$$

with the additional cost

$$
\mathcal J_{\mathrm{tan-force}} =
\lambda_t\left\|
\left(\boldsymbol f_{t+1}-\boldsymbol f_t-\boldsymbol K_s\boldsymbol J\boldsymbol u_t\right)_t
\right\|_2^2.
$$

This keeps normal-force regulation precise while allowing the optimizer to recover smoothly from tangential states
that are not immediately compatible with the idealized model. The default controller therefore retains the full
three-dimensional equality in Stage 1 and uses the normal-only hard equality with the explicit tangential
force-prediction penalty in Stage 2.

## Practical phase progression

Wrench balance remains the primary criterion for entering Stage 2. The implementation adds two mechanisms for
finite-duration execution:

1. **Path-completion fallback.** If the full guiding path has been traversed without satisfying the balance
   threshold, the controller enters Stage 2 so that grasp execution can continue instead of remaining indefinitely
   in the approaching phase. This is useful when sparse, noisy, or changing contacts prevent the ideal transition
   optimization from crossing the threshold.
2. **Online phase re-evaluation.** Before the path-completion fallback becomes active, the default controller may
   return to Stage 1 if the contact set changes and the current contacts no longer satisfy the balance criterion.
   This allows the hand to re-establish a suitable contact configuration after contact creation or loss.

The balance-triggered transition and the path-completion fallback therefore represent two complementary transition
paths: one based on the preferred coordinated-contact condition and one that guarantees progress under imperfect
contact observations.

The controller also has a finite horizon of twice the interpolated path length. Under the default 10 Hz and
two-second-plus-two-second path, this gives at most 80 actions, or eight seconds, before lifting proceeds. This bound
prevents a non-convergent episode from running indefinitely. Normally, the controller stops earlier after the full
path has been reached and the measured total normal force exceeds the hand-specific final-force target.

## Additional Stage-2 regularization

The paper emphasizes the principal Stage-2 terms: guiding non-contact fingers, minimizing the net wrench, and
maintaining smooth motion. The implementation adds the following regularizers to improve closed-loop behavior.

### Normal-force-induced joint-torque penalty

The implementation penalizes

$$
\mathcal J_{\mathrm{joint-load}} =
\lambda_\tau\left\|\boldsymbol J_{n,h}^{\mathsf T}\boldsymbol f_n\right\|_2^2,
$$

where `J_{n,h}` is the normal contact Jacobian with respect to the hand joints. This discourages force allocations
that create excessive or highly concentrated hand-joint loading and favors mechanically better-distributed load
paths. The term is a load regularizer rather than a hard requirement that every joint carry exactly equal torque.
Its default weight is `0.01`.

### Contact-aware velocity damping

For joints that currently contribute to an active contact Jacobian, the Stage-2 joint-velocity penalty is multiplied
by `100`. The larger local damping reduces aggressive command changes after contact, making the controller less
prone to overshoot or oscillation while contact forces are increasing. Non-contact joints retain the nominal
velocity weight so they can continue following the guiding path.

These terms preserve wrench balance as the main force-allocation objective while regularizing solutions that are
difficult for a position-controlled dexterous hand to execute reliably.

## Stiffness-model regularization

The paper derives the motion-contact model using the joint stiffness `K_p`. The controller uses the actuator
stiffness specified in the XML as the physical reference, with numerical clipping applied in the predictive contact
model for improved conditioning and robustness. The MuJoCo plant retains the original XML gains. The resulting
effective stiffness used by the predictive controller is

$$
\boldsymbol K_{p,\mathrm{eff}} = \operatorname{clip}(\boldsymbol K_p, 0, 10^3).
$$

For the dummy arm, this clips the three translational stiffness values from `10000` to `1000`; rotational and hand
stiffness values remain below the clipping threshold. The clipping improves conditioning of the compliance and
contact-stiffness calculations and prevents very large stiffness ratios from dominating the linearized model.

During Stage 2, commanded arm motion is fixed and the default stiffness prediction is built from the hand-joint
subsystem. This focuses the force-control model on the actively commanded hand joints and avoids introducing the
very stiff dummy-arm directions into the Stage-2 contact model.

The implementation additionally diagnoses every linear system. Well-conditioned systems use a direct solve;
singular or ill-conditioned systems use a recorded finite least-squares fallback. Thus, stiffness clipping and
diagnosed linear solves jointly provide numerical regularization without allowing nonfinite commands to reach the
simulator.

## Force scheduling and bounded optimization

Several small execution safeguards supplement the paper's force schedule.

- **Stage-1 force recovery.** The nominal Stage-1 upper force is `F_appr = 0.2 N`. If the measured force is already
  higher, for example after a phase reversal or a transient, the implementation reduces the permitted force by one
  force increment per action instead of requiring an infeasible instantaneous drop to `0.2 N`.
- **Stage-2 target cap.** The desired total force follows the measured/previous target plus
  `delta_F = F_ub / L_g`, and is capped at `F_ub + 0.2 N`. The small margin prevents unbounded growth while allowing
  the measured force to cross the termination target.
- **Hand-specific final force.** The current simulation targets are `15 N` for Shadow, `10 N` for Allegro, and `8 N`
  for LEAP Tac3D. These values implement the paper's policy of matching the desired total force to representative
  open-loop final forces for each hand.
- **Command bounds.** Per-action joint changes are bounded using the configured actuator velocity limits. Contact
  force variables also have broad finite bounds for stable SLSQP operation. These bounds are normally safeguards,
  but they make the one-step optimization consistent with the finite motion available on the physical or simulated
  robot.

## Solver-result handling

The paper assumes that each MPC optimization produces a usable solution. The implementation separates solver
acceptance from the runtime action selected after a rejected solve and records both decisions explicitly.

The available policies are:

- `apply_candidate`: apply a rejected candidate only when it has the exact expected dimension and all values are
  finite, while retaining the rejected diagnostic and marking the episode as solver-degraded;
- `hold_current`: keep the current actuator target, clear the command increment/history, and continue;
- `fail_episode`: save the partial record, skip lifting for the affected offset, and continue the remaining batch.

Malformed, incorrectly sized, NaN, or infinite candidates are never applied. The public default is
`apply_candidate`.
