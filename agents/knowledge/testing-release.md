# Testing and Release Knowledge

Load this note when selecting validation or interpreting a regression. Current workflow files, gate scripts, and the golden artifact are authoritative.

## Validation layers

| Layer | Purpose | Typical entry point |
| --- | --- | --- |
| Focused tests | Local math, schema, path, status, or policy contract | `python -m unittest tests.test_<area> -v` |
| Full unit suite | Cross-module Python contracts and subprocess behavior | `python -m unittest discover -s tests -v` |
| Quick example | Current CLI/config/assets/MuJoCo/report smoke | `bash script/run_example.sh <hand> quick` |
| Fixed matrix | Three hands x five methods, strict trajectory comparison | `run_release_gate.sh fixed` |
| Wheel gate | Code-only wheel with explicit external roots and isolated imports | `run_release_gate.sh wheel` |
| Portable gate | Three quick examples + fixed matrix + wheel | `run_release_gate.sh portable` |
| Release 300 | Three hands x 100 `ours` cases using external inputs | `run_release_gate.sh release300` |

CI runs Python 3.10 lint/format, compile/unit/schema/CLI tests, and a precomputed Shadow quick smoke. Full GPU IK and large release suites are release responsibilities, not ordinary CI assumptions.

## Strict comparison meaning

The fixed matrix contains 15 raw trajectories. Strict validation compares structure, stage/contact sequences, classifications, and numeric content at `rtol=1e-5`, `atol=1e-6`, excluding timing and only explicitly approved additive metadata. `script/audit_golden.py` verifies checksums and recorded scientific evidence; `script/compare_golden.py` performs trajectory comparisons.

The promoted 300-case counts recorded in the current artifact are:

| Hand | Success | Failure | Invalid | Degraded | Error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shadow | 69 | 4 | 21 | 6 | 0 |
| Allegro | 80 | 5 | 14 | 1 | 0 |
| LEAP Tac3D | 88 | 5 | 7 | 0 | 0 |

The artifact records the historical `hold_current` degraded trajectories. The release300 driver pins that policy explicitly for audit reproducibility, while ordinary commands default to `apply_candidate`. Classification counts are unchanged by this distinction, but degraded trajectory contents are intentionally policy-dependent.

These counts are a release baseline, not a substitute for verifying the artifact and regenerated raw trajectories. The large 300-case output trees are intentionally not committed.

## Regression triage

Classify mismatches as timing, approved additive metadata, tolerance-only floating error, structural/contact/stage change, classification change, or expected correctness repair. Find the first divergent step and correlate it with config, inputs, dependency origins, contact order, solver diagnostics, and linear algebra. Never infer equivalence from an aggregate success rate alone.

Every example/release invocation uses a new empty output root and passes its exact `run_report.json` into downstream statistics/reporting. Existing-run refusal and empty-golden refusal are protections against false success, not inconveniences to bypass.
