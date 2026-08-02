# Repository hygiene and local-data retention

The repository treats `output/` as generated-only state. Removing historical `output/` paths from the Git index does
not remove a maintainer's local files, and the project does not provide an automatic cleanup command that deletes
experimental results.

The following local artifacts are normally reproducible and can be removed manually after confirming that no process
is using them:

- Python, test, coverage, and analysis caches such as `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, and `.coverage`;
- packaging products such as `build/`, `dist/`, wheels, and `*.egg-info/`;
- root-level MuJoCo/debug artifacts `MUJOCO_LOG.TXT` and `debug.xml`;
- disposable example runs under `output/examples/` after their current-run reports are no longer needed.

Do not treat every ignored path as disposable. Archive research evidence externally before any manual cleanup,
including `output/experiments/`, historical `output/learn_*` and `output/retest_*` trees, or any custom output root
whose manifests, reports, statistics, or trajectories may be needed for a paper or comparison. The externally
restored `release/golden/` tree is approved evidence rather than a cache. `assets/object/` is the local external BODex
object-data location and may be expensive to restore.

The checked-in quick-start fixtures under `examples/data/` are source artifacts, not generated runtime output. Their
external regeneration workflow is documented in [the fixture source record](../examples/data/README.md).

Historical records under `agents/tasks/` are retained as maintained audit documentation. They are non-normative and
may describe obsolete behavior; current rules, code, tests, and the public README take precedence. New transient tool
state belongs in ignored `.agents/` or `.codex/` directories rather than `agents/tasks/`.
