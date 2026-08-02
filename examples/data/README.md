# Bundled fixture sources

The `dummy_arm/` tree for each maintained hand contains the canonical 100-record quick input set. Shadow, Allegro,
and LEAP Tac3D retain the same relative scene/sample identities while preserving their hand-specific qpos and joint
orders. The single `raw/` and `formatted/` fixtures remain separate inputs for the full four-stage example.

The accepted quick sources are:

```text
output/learn_dummy_arm_shadow/graspdata
output/learn_dummy_arm_allegro/graspdata
output/learn_dummy_arm_leap_tac3d/graspdata
```

`script/build_example_fixtures.py` validates all 300 records before replacing tracked destinations, converts them to
schema v1 without changing scientific arrays, copies the exact referenced DGN subset, retargets the one-sample full
fixtures to the deduplicated bottle, and writes `examples/quick_manifest.json`.

```bash
PYTHONPATH=src python script/build_example_fixtures.py \
  --dummy-arm-root shadow=/absolute/path/to/output/learn_dummy_arm_shadow/graspdata \
  --dummy-arm-root allegro=/absolute/path/to/output/learn_dummy_arm_allegro/graspdata \
  --dummy-arm-root leap_tac3d=/absolute/path/to/output/learn_dummy_arm_leap_tac3d/graspdata \
  --dgn-root /absolute/path/to/assets/object/DGN_2k
```

The builder follows the external DGN source and copies real files; it never writes the workstation symlink into the
checkout. Verify a generated or checked-in tree with:

```bash
PYTHONPATH=src python script/audit_example_fixtures.py --manifest examples/quick_manifest.json
```

The manifest records every bundled grasp, scene configuration, processed-object dependency, file size, SHA-256 value,
object ID, count, and aggregate digest. Runtime output under `output/` is generated state and is never a fixture source.
