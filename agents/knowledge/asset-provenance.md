# Asset Provenance Knowledge

Load this note for asset changes, attribution, deletion, or public-release wording. The primary records are `assets/hand/README.md` and the attribution file below `examples/assets/object/`.

## Current supported assets

- Allegro uses project-maintained XML plus meshes from the pinned MuJoCo Menagerie Allegro asset with BSD-2-Clause evidence.
- Shadow uses project-maintained XML plus meshes from the pinned MuJoCo Menagerie Shadow asset with Apache-2.0 evidence.
- LEAP Tac3D uses local XML and STL files. The repository does not currently record the provider, import commit, copyright holder, license, or Tac3D modification/redistribution permission for those local meshes. The Menagerie LEAP Hand MIT license is context, not proof that it covers this asset set.
- The bundled quick data uses an exact 89-object subset from the original DGN 2k collection. Its source and complete
  file checksums are recorded by the DGN attribution and `examples/quick_manifest.json`.

The retained LEAP Tac3D provenance gap is separate from the technical quick gate.

## Removed legacy assets

The hand asset audit records legacy LEAP/UR composites, superseded Shadow XML, a one-off attachment script, and five
unused LEAP Tac3D meshes removed on 2026-08-03 after explicit maintainer approval. Git history remains the recovery
path for historical reproduction. Static unreachability and deletion do not answer provenance or redistribution
questions for the retained assets.

## Evidence expected for new or repaired provenance

Record available upstream facts and stable checksums without turning missing metadata into certainty. For the bundled
DGN quick subset, the accepted record is the original DGN 2k source, the subset operation, and the machine-readable
manifest. Preserve more detailed primary records when they are available.
