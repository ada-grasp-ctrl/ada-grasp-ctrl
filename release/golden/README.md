# Golden release evidence

This directory contains the promoted fixed-matrix golden and a compact audit artifact for the larger release suite.

## Promoted baseline

The current baseline includes the deterministic collision-mesh declaration fix from commit `4277c4c`. Collision meshes are now declared in filename order, so copying an object directory cannot silently change MuJoCo geom IDs, contact order, or control trajectories.

The fixed matrix contains three hands × five methods (15 raw trajectories). Two independent runs matched for keys, shapes, stage/contact sequences, classifications, and numeric arrays with `rtol=1e-5` and `atol=1e-6`.

The 300-case `ours` release suite was run twice from clean commit `d6163ca`, with seed 12 and eight workers:

| Hand | Success | Failure | Invalid initialization | Solver degraded | Execution error |
|---|---:|---:|---:|---:|---:|
| Shadow | 69 | 4 | 21 | 6 | 0 |
| Allegro | 80 | 5 | 14 | 1 | 0 |
| LEAP Tac3D | 88 | 5 | 7 | 0 | 0 |

Both 300-case runs matched strictly. Relative to the previously promoted solver-corrected baseline, mesh ordering changed scientific trajectories in 251 files and changed one classification: Shadow `core_mug_ef24c302911bcde6ea6ff2182dd34668/.../partial_pc_00_4` changed from failure to success. Invalid and degraded sets did not change.

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
