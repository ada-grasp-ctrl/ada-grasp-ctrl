# Golden release evidence

This directory contains the promoted fixed-matrix golden and a compact audit artifact for the larger release suite.

## Promoted baseline

The current baseline includes the direct linear-system solve and diagnosed fallback repair from commit `8defb93`, on top of the deterministic collision-mesh declaration fix from commit `4277c4c`. Contact stiffness now uses `solve(A, B)` instead of forming `inv(A) @ B`; singular or ill-conditioned systems are explicitly diagnosed and use a finite least-squares/zero fallback. BS1 and BS3 no longer install module-wide `RuntimeWarning` filters. The one expected SciPy SLSQP trial-step clipping warning is handled only at the solver boundary and its count is retained in solver diagnostics.

The fixed matrix contains three hands × five methods (15 raw trajectories). Two independent runs matched for keys, shapes, stage/contact sequences, classifications, and numeric arrays with `rtol=1e-5` and `atol=1e-6`.

The 300-case `ours` release suite was run twice from clean commit `8defb93`, with seed 12 and eight workers:

| Hand | Success | Failure | Invalid initialization | Solver degraded | Execution error |
|---|---:|---:|---:|---:|---:|
| Shadow | 69 | 4 | 21 | 6 | 0 |
| Allegro | 80 | 5 | 14 | 1 | 0 |
| LEAP Tac3D | 88 | 5 | 7 | 0 | 0 |

Both 300-case runs matched strictly. Relative to the previously promoted solver-corrected baseline, mesh ordering changed scientific trajectories in 251 files and changed one classification: Shadow `core_mug_ef24c302911bcde6ea6ff2182dd34668/.../partial_pc_00_4` changed from failure to success. Invalid and degraded sets did not change.

Relative to the immediately preceding mesh-ordered baseline, direct solve changed 9/15 optimized fixed trajectories and 248/300 release trajectories, with zero classification changes. The first observed stiffness-system difference was approximately `1.14e-13`; closed-loop contact dynamics amplified that last-bit change, while the direct solve reduced the sampled linear residual from approximately `1.47e-11` to `5.55e-13`. The phase-4 numerical-stability repair therefore promotes the reproducible direct-solve result instead of retaining inverse-compatible rounding. The old-to-new file list, mismatch categories, scientific digests, and unchanged classifications are recorded in `artifact.json`.

The phase-4 performance gate showed no significant regression on the fixed Shadow `ours` case: wall time changed from 5.71 s to 5.60 s and peak RSS from 483,972 KiB to 482,808 KiB. Aggregate fixed-matrix optimization time changed from 6.712 s to 6.804 s (+1.37%).

## Files

- `fixed_matrix/`: the 15 promoted raw control records required for tolerance-based comparison in a clean clone.
- `artifact.json`: input/output SHA-256 values, metadata-independent scientific digests, classifications, complete run manifests, release statistics, strict repeat reports, and per-file historical difference summaries.

The two 300-case raw output trees are intentionally not committed. They are approximately 107 MB each. `artifact.json` retains all 300 relative paths, raw checksums, scientific digests, and classifications so a regenerated suite can be verified without adding the large trajectories to Git.

## Verification

Verify the checked-in fixed golden and bundled fixed inputs:

```bash
PYTHONPATH=src python script/audit_golden.py verify release/golden/artifact.json
```

After regenerating the fixed matrix and full release suite, also verify their scientific content and the external release inputs:

```bash
PYTHONPATH=src python script/audit_golden.py verify release/golden/artifact.json \
  --fixed-current /path/to/fixed-matrix-run \
  --release-root /path/to/release-run \
  --release-input-root output
```

`--release-root` must contain `shadow/control`, `allegro/control`, and `leap_tac3d/control`. `--release-input-root` must contain the repository-style `learn_dummy_arm_<hand>/graspdata` directories.
