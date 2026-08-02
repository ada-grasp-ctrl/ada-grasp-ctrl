# Testing and Release Rules

Apply this rule to tests, CI, quick examples, benchmarks, trajectory comparisons, release artifacts, and performance
claims.

## Validation ladder

- Run the smallest relevant unit/schema test first. Then run the full unit suite for shared runtime, schema,
  controller, or batch changes.
- Run Ruff and compile checks for Python changes. Use the exact canonical commands from `AGENTS.md`/`README.md` rather
  than inventing a parallel test workflow.
- Use a quick example when the change reaches MuJoCo, assets, CLI composition, runtime roots, or end-to-end reports.
- Use the maintained quick gate when the change can affect trajectories, contacts, classifications, the supported
  matrix, bundled fixtures, or release automation.
- Use unique empty output roots for integration/release work. Never accept a gate that consumed stale or untracked
  outputs.

## Trajectory comparison

- Generic comparison utilities may compare keys, shapes, stage sequences, contact count/order, episode
  classifications, and floating trajectories. Timing fields and explicitly approved additive metadata may be
  excluded.
- The retained numeric tolerances are `rtol=1e-5` and `atol=1e-6`; do not loosen them without approval.
- Process exit code alone is not scientific validation. Inspect batch reports, statistics, comparison reports, and
  manifests.
- Never replace trajectory evidence to conceal an unexplained mismatch. Classify the difference, find its first cause,
  and fix unintended regressions.
- Reproducibility claims require two clean runs with the same commit, dependencies, inputs, seed, worker count, and
  resolved config.

## Maintained gate

The only maintained release gate is the self-contained bundled quick set:

```bash
PYTHON_BIN=python bash script/run_release_gate.sh quick
```

It audits the 3x100 fixtures and exact 89-object subset, runs every hand serially with `ours`, and verifies each fresh
classification against `examples/quick_expected_status.json`. A documented batch exit code `1` is acceptable only when
the complete current-run reports and classifications match that inventory. Missing outputs, execution errors,
solver-degradation drift, stale reports, count mismatches, or input/asset drift fail the gate.

## Load on demand

- Read [../knowledge/testing-release.md](../knowledge/testing-release.md) for CI coverage and quick-result triage.
- Read [../knowledge/runtime-data-contracts.md](../knowledge/runtime-data-contracts.md) when interpreting manifests,
  reports, statuses, or exit codes.
