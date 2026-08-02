# Testing and Release Knowledge

Load this note when selecting validation or interpreting a regression. Current workflow files and gate scripts are
authoritative. Golden trajectories and their audit artifact are not bundled in the repository.

## Validation layers

| Layer | Purpose | Typical entry point |
| --- | --- | --- |
| Focused tests | Local math, schema, path, status, or policy contract | `python -m unittest tests.test_<area> -v` |
| Full unit suite | Cross-module Python contracts and subprocess behavior | `python -m unittest discover -s tests -v` |
| Quick example | Current CLI/config/assets/MuJoCo/report smoke | `bash script/run_example.sh <hand> quick` |
| Fixed matrix | Three hands x five methods, strict trajectory comparison against an approved external baseline | `run_release_gate.sh fixed` |
| Wheel gate | Code-only wheel with explicit external roots and isolated imports | `run_release_gate.sh wheel` |
| Portable gate | Three quick examples + fixed matrix + wheel | `run_release_gate.sh portable` |
| Release 300 | Three hands x 100 `ours` cases using external inputs | `run_release_gate.sh release300` |

CI runs Python 3.10 lint/format, compile/unit/schema/CLI tests, and a precomputed Shadow quick smoke. Full GPU IK and large release suites are release responsibilities, not ordinary CI assumptions.

## Strict comparison meaning

The fixed-matrix workflow compares 15 raw trajectories. Strict validation compares structure, stage/contact sequences,
classifications, and numeric content at `rtol=1e-5`, `atol=1e-6`, excluding timing and only explicitly approved additive
metadata. `script/audit_golden.py` verifies checksums and recorded scientific evidence; `script/compare_golden.py`
performs trajectory comparisons. Do not report promoted counts or trajectory equivalence without an approved external
artifact and its regenerated raw trajectories. Large release output trees remain intentionally uncommitted.

## Regression triage

Classify mismatches as timing, approved additive metadata, tolerance-only floating error, structural/contact/stage change, classification change, or expected correctness repair. Find the first divergent step and correlate it with config, inputs, dependency origins, contact order, solver diagnostics, and linear algebra. Never infer equivalence from an aggregate success rate alone.

Every example/release invocation uses a new empty output root and passes its exact `run_report.json` into downstream statistics/reporting. Existing-run refusal and empty-golden refusal are protections against false success, not inconveniences to bypass.
