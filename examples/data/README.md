# Bundled fixture sources

The records in this directory are the canonical checked-in quick-start inputs for a clean checkout. Runtime outputs
under `output/` are generated state and are not fixture sources. This fixture designation does not change the
repository's licensing or provenance status described in the public README.

`script/build_example_fixtures.py` remains available for a deliberate regeneration from an externally archived copy
of the fixed formatted and dummy-arm sample. The archive is not bundled. Pass the six source roots explicitly; each
root must be absolute and must directly contain the following fixed relative path:

```text
core_bottle_15787789482f045d8add95bf56d3d2fa/tabletop_ur10e/scale006_pose004_0/partial_pc_00_6.npy
```

Example invocation:

```bash
python script/build_example_fixtures.py \
  --formatted-root shadow=/archive/formatted/learn_shadow/graspdata \
  --formatted-root allegro=/archive/formatted/learn_allegro/graspdata \
  --formatted-root leap_tac3d=/archive/formatted/learn_leap_tac3d/graspdata \
  --dummy-arm-root shadow=/archive/dummy-arm/learn_dummy_arm_shadow/graspdata \
  --dummy-arm-root allegro=/archive/dummy-arm/learn_dummy_arm_allegro/graspdata \
  --dummy-arm-root leap_tac3d=/archive/dummy-arm/learn_dummy_arm_leap_tac3d/graspdata \
  --destination-root /path/to/ada-grasp-ctrl/examples/data
```

The builder validates all six roots and records before writing any destination. Regeneration is an explicit
provenance-sensitive maintenance action: inspect the fixture diff and run all three full examples before committing
changed `.npy` records.
