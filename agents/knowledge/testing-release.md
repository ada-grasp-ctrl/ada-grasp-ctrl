# Testing and Release Knowledge

Load this note when selecting validation or interpreting a quick regression. Current workflow files and gate scripts
are authoritative.

## Validation layers

| Layer | Purpose | Typical entry point |
| --- | --- | --- |
| Focused tests | Local schema, path, status, policy, fixture, or report contract | `python -m unittest tests.test_<area> -v` |
| Full unit suite | Cross-module Python contracts and subprocess behavior | `python -m unittest discover -s tests -v` |
| Hand quick | One bundled 100-record current-run evaluation | `bash script/run_example.sh <hand> quick` |
| Quick gate | Fixture audit plus all three expected-status comparisons | `bash script/run_release_gate.sh quick` |

CI runs Python 3.10 lint/format, compile/unit/schema/CLI tests, then a three-hand quick matrix. Each hand is internally
serial (`n_worker=1`) and uses a unique output root.

## Quick acceptance meaning

`examples/quick_manifest.json` is the authoritative input/asset inventory. It proves 100 records per hand, shared
sample identities, 89 selected objects, exact scene and processed-data files, and portable SHA-256 inventories.

`examples/quick_expected_status.json` is tied to the fixture-manifest file digest and records every sample's final
classification. The quick validator requires exactly 100 control-eval results, 100 control-stat results, one current
statistics file, and exact input/output identity agreement before comparing classifications. This makes a known batch
exit code `1` distinguishable from a new execution error or solver-degradation change.

## Regression triage

Classify mismatches as input/asset drift, missing current-run evidence, execution error, solver degradation, scientific
classification change, or aggregate-count mismatch. Find the first differing sample and correlate it with the resolved
config, seed, dependency origins, contact order, and solver diagnostics. Never infer acceptance from aggregate success
rate or process exit code alone.
