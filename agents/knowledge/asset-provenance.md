# Asset Provenance Knowledge

Load this note for asset changes, attribution, deletion, or public-release wording. The primary records are `assets/hand/README.md` and the attribution file below `examples/assets/object/`.

## Current supported assets

- Allegro uses project-maintained XML plus meshes from the pinned MuJoCo Menagerie Allegro asset with BSD-2-Clause evidence.
- Shadow uses project-maintained XML plus meshes from the pinned MuJoCo Menagerie Shadow asset with Apache-2.0 evidence.
- LEAP Tac3D uses local XML and STL files. The repository does not currently record the provider, import commit, copyright holder, license, or Tac3D modification/redistribution permission for those local meshes. The Menagerie LEAP Hand MIT license is context, not proof that it covers this asset set.
- The bundled DGN/BODex bottle fixture has source links and checksums, but the authorizing party, authorization date, redistribution scope/conditions, and primary message/ticket/reference are not recorded.

These gaps block a claim of fully audited public redistribution even though the technical release gates can pass.

## Retained deletion candidates

The hand asset audit lists currently unreachable directories/files, including legacy LEAP/UR composites, superseded Shadow XML, a one-off attachment script, and unused LEAP Tac3D meshes. They remain tracked pending explicit maintainer confirmation. Static unreachability supports a deletion proposal but does not answer provenance, redistribution, or historical-reproduction questions.

## Evidence expected for new or repaired provenance

Record the upstream project and exact version/commit, original filename/path, copyright holder, license text/reference, project modifications, redistribution basis, import date if known, and stable checksums. For private permission, preserve the actual authorizing communication or a stable internal reference; do not paraphrase missing evidence into certainty.

