# Documentation and Asset Rules

Apply this rule to `README.md`, public examples, support claims, licenses, attribution, hand/object assets, and release notes.

## Public documentation

- Keep `README.md` aligned with executable config, registries, scripts, tests, and release gates. Verify commands before documenting them.
- Describe only the maintained four-stage pipeline and supported hand/method/converter matrix. Do not revive removed prototype tasks or imply underactuated-hand support.
- Distinguish technical reproducibility from scientific interpretation and legal permission.
- State historical, intermediate, and promoted baselines precisely; do not present one-run performance numbers as hardware-independent guarantees.
- Keep mutable implementation detail out of `AGENTS.md`; route it to rules or knowledge and link to its primary source.

## Assets and provenance

- The repository currently provides no project-wide license. Third-party submodules, hand descriptions/meshes, and bundled data remain subject to their own licenses and permission records.
- Preserve upstream license and attribution files. Do not invent an authorizing party, date, scope, copyright holder, import commit, or permission statement.
- For the bundled DGN quick subset, record the original DGN 2k source and manifest checksums without adding unsupported
  authorization claims. Local LEAP Tac3D mesh provenance remains unresolved until primary records are added.
- Reachability is not permission: an unused asset may still have legal constraints, and an upstream license for a similar asset does not prove coverage of a local derivative.
- Ask before deleting retained asset candidates or changing public redistribution status.
- When adding or replacing an asset, record the available source facts and checksums without inventing missing metadata.
  The accepted DGN quick-subset contract requires its original DGN 2k source plus the machine-readable fixture manifest.

## Load on demand

- Read [../knowledge/asset-provenance.md](../knowledge/asset-provenance.md) before editing bundled object/hand assets, attribution, release-readiness language, or deletion candidates.
- Read [../knowledge/testing-release.md](../knowledge/testing-release.md) before changing benchmark or reproducibility claims.
