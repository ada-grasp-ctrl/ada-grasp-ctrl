# Documentation and Asset Rules

Apply this rule to `README.md`, public examples, support claims, licenses, attribution, hand/object assets, and release notes.

## Public documentation

- Keep `README.md` aligned with executable config, registries, scripts, tests, and release gates. Verify commands before documenting them.
- Describe only the maintained four-stage pipeline and supported hand/method/converter matrix. Do not revive removed prototype tasks or imply underactuated-hand support.
- Distinguish technical reproducibility from scientific interpretation and legal permission.
- State historical, intermediate, and promoted baselines precisely; do not present one-run performance numbers as hardware-independent guarantees.
- Keep mutable implementation detail out of `AGENTS.md`; route it to rules or knowledge and link to its primary source.

## Assets and provenance

- The repository MIT license covers project source, not automatically third-party submodules, hand descriptions/meshes, or bundled data.
- Preserve upstream license and attribution files. Do not invent an authorizing party, date, scope, copyright holder, import commit, or permission statement.
- Treat the DGN/BODex example object's redistribution evidence and the local LEAP Tac3D mesh provenance as unresolved until primary records are added.
- Reachability is not permission: an unused asset may still have legal constraints, and an upstream license for a similar asset does not prove coverage of a local derivative.
- Ask before deleting retained asset candidates or changing public redistribution status.
- When adding or replacing an asset, record source, exact version/commit, modifications, license, redistribution basis, and checksums where practical.

## Load on demand

- Read [../knowledge/asset-provenance.md](../knowledge/asset-provenance.md) before editing bundled object/hand assets, attribution, release-readiness language, or deletion candidates.
- Read [../knowledge/testing-release.md](../knowledge/testing-release.md) before changing benchmark or reproducibility claims.

