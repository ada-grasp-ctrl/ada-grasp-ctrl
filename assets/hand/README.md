# Hand asset audit

This directory contains project-maintained MJCF files plus local meshes used by the three supported hands. The repository currently provides no project-wide license for project-owned content, and it does not alter or replace the licenses that apply to upstream robot descriptions or meshes.

## Runtime reachability (audited 2026-08-03)

The active paths below are selected by `src/ada_grasp_ctrl/config/hand/*.yaml` and `RobotFactory`, and are exercised by the quick and fixed release gates.

| Hand | Active local files | External mesh source | License evidence in the repository |
|---|---|---|---|
| Allegro | `allegro/right_hand.xml`, `dummy_arm_allegro/right.xml` | `third_party/mujoco_menagerie/wonik_allegro/assets` | Pinned submodule `wonik_allegro/LICENSE` (BSD-2-Clause, SimLab) and `wonik_allegro/README.md` |
| Shadow | `shadow/right_hand.xml`, `dummy_arm_shadow/right_no_tendon.xml` | `third_party/mujoco_menagerie/shadow_hand/assets` | Pinned submodule `shadow_hand/LICENSE` (Apache-2.0) and `shadow_hand/README.md` |
| LEAP Tac3D | `leap_tac3d/leap_tac3d.xml`, nine referenced STL files below it, and `dummy_arm_leap_tac3d/leap_tac3d.xml` | Local `leap_tac3d/leap_hand/meshes` | **Incomplete:** the repository does not identify the provider, import commit, copyright holder, license, or Tac3D modification permission for these local STL files. The separately pinned Menagerie `leap_hand` MIT license is useful upstream context but is not evidence that it covers this Tac3D asset set. |

The project-modified Allegro and Shadow XML files load licensed Menagerie meshes, but the repository history also lacks a primary import/derivation record for the XML text itself. Preserve the upstream license files with every distribution and complete the derivation record before treating this audit as final legal clearance.

## Removed static legacy assets

The following paths were not referenced by maintained configuration, robot registration, release scripts, tests, or
active MJCF includes and were removed after explicit maintainer approval:

| Removed path | Approximate size | Audit note |
|---|---:|---|
| `leap/` | 1.7 MB | No registered `leap` hand; maintained hand name is `leap_tac3d`. |
| `ur5_leap_tac3d/` | 25 MB | No maintained task or configuration loads the UR5 composite; it duplicates the 20 MB Tac3D mesh tree. |
| `ur10e_shadow/` | 36 KB | No maintained reference; its XML references mesh directories that are not present below that path. |
| `attach_hand_to_arm.py` | 4 KB | One-off generator with hard-coded working-directory paths; not imported or called. |
| `dummy_arm_shadow/right.xml` | 28 KB | Superseded by the registered `right_no_tendon.xml`. |
| `shadow/right_with_forearm.xml` | 20 KB | Superseded by the maintained forearm-free `right_hand.xml`. |
| Five unreferenced STL files under `leap_tac3d/leap_hand/meshes` | 6.9 MB | `fingertip_base.stl`, `fingertip_custom.stl`, `palm_lower_left.stl`, `thumb_fingertip_base.stl`, and `thumb_left_temp_base.stl` are not referenced by either active LEAP Tac3D MJCF. |

These files remain recoverable from Git history for historical reproduction. Their removal is a technical reachability
decision only; it does not establish or change the provenance, licensing, or redistribution status of the retained
assets.

## Modification clarification

### Shadow

1. Remove the forearm and wrist, because they are more like parts of the arm.
2. Exclude the contact between some neighboring links, such as `palm` and `rh_ffproximal`, which are not excluded by default because of `rh_ffknuckle`.
3. Unify the forcerange and kp of different joints. kp is set to 5 because the object is heavy (object mass=100g).

### Allegro

1. Use kp=5.

### Leap

1. Use simplified visual mesh to speedup loading.
