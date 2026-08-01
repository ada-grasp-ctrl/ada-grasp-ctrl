# Testing and Release Rules

Apply this rule to tests, CI, quick examples, benchmarks, golden comparisons, release artifacts, and performance claims.

## Validation ladder

- Run the smallest relevant unit/schema test first. Then run the full unit suite for shared runtime, schema, controller, or batch changes.
- Run Ruff and compile checks for Python changes. Use the exact canonical commands from `AGENTS.md`/`README.md` rather than inventing a parallel test workflow.
- Use a quick example when the change reaches MuJoCo, assets, CLI composition, runtime roots, or end-to-end reports.
- Use a release gate when the change can affect trajectories, contacts, classifications, wheel isolation, the supported matrix, or release automation.
- Use unique empty output roots for integration/release work. Never accept a gate that consumed stale or untracked outputs.

## Golden comparison

- Compare keys, shapes, stage sequences, contact count/order, episode classifications, and floating trajectories. Timing fields and explicitly approved additive metadata may be excluded.
- The promoted numeric tolerances are `rtol=1e-5` and `atol=1e-6`; do not loosen them without approval.
- Process exit code alone is not scientific validation. Inspect batch reports, statistics, comparison reports, and manifests.
- Never replace promoted trajectories to conceal an unexplained mismatch. Classify the difference, find its first cause, and fix unintended regressions.
- Golden promotion or any accepted classification change requires maintainer approval plus updated machine-readable audit evidence and human-readable rationale.
- Reproducibility claims require two clean runs with the same commit, dependencies, inputs, seed, worker count, and resolved config.

## Maintained gates

```bash
# Audit checked-in golden evidence without running simulation.
PYTHONPATH=src python script/audit_golden.py verify release/golden/artifact.json

# Portable gate: three quick examples, 15-case fixed matrix, and isolated wheel mode.
PYTHON_BIN=python bash script/run_release_gate.sh portable

# External 300-case release gate.
ADA_GRASP_CTRL_RELEASE_INPUT_ROOT=/absolute/path/to/release-inputs \
  PYTHON_BIN=python bash script/run_release_gate.sh release300
```

Do not run the expensive `release300` gate unless the task needs it and the recorded external input tree is available. Do not claim it passed from the checked-in audit alone.

## Load on demand

- Read [../knowledge/testing-release.md](../knowledge/testing-release.md) for CI/release coverage, baseline meaning, and failure triage.
- Read [../knowledge/runtime-data-contracts.md](../knowledge/runtime-data-contracts.md) when interpreting manifests, reports, statuses, or exit codes.

