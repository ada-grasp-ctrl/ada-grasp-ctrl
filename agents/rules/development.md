# Development Rules

Apply this rule to general source, CLI, Hydra configuration, refactoring, and feature work.

## Mandatory constraints

- Preserve the public four-stage workflow and supported matrix unless the maintainer explicitly approves a scope change.
- Keep task dispatch in `TASK_REGISTRY`, control-method dispatch in `METHOD_REGISTRY`, and converter dispatch in `CONVERTER_REGISTRY`. Do not use string-based `eval(...)` dispatch.
- Keep `python src/main.py` independent of the shell working directory. Resolve runtime files through the root/path helpers, not fixed `Path(__file__).parents[...]` assumptions.
- Separate task-level preflight failures from sample-level execution failures. Configuration, environment, viewer startup, and shared-asset failures must fail before batch work; one malformed sample must remain attributable to that sample when the rest of the batch can continue safely.
- Preserve backward-compatible reads for legacy records and configuration aliases unless a breaking change is explicitly approved.
- Use Python 3.10-compatible syntax. Follow the repository's 120-column Ruff/Black configuration.
- Add type annotations to new or materially rewritten public/internal interfaces. Write English docstrings with `Args`, `Returns`, and `Raises` where applicable; comments should explain non-obvious intent, not restate code.
- Do not edit vendored/submodule code to avoid fixing an integration problem in project-owned code.
- Do not commit `output/`, `build/`, caches, Hydra logs, ad-hoc benchmarks, or unpromoted golden data.

## Change procedure

1. Identify the executable contract in current config, registry, schema, and tests.
2. Make the smallest coherent change without mixing scientific changes, structural refactors, and golden promotion.
3. Add a focused regression test for the changed contract.
4. Run the focused test, then the broader checks required by the affected domain.
5. Inspect `git diff --check` and the final diff for accidental generated or unrelated changes.

## Load on demand

- Read [../knowledge/architecture.md](../knowledge/architecture.md) when locating ownership, tracing the pipeline, or adding a task, converter, controller, or hand.
- Also read `data-runtime.md`, `control-simulation.md`, or `testing-release.md` when the change crosses those domains.
