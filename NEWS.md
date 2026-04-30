# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (fork: sequential SIMP→level-set pipeline)
- Adaptive `α_min` decay in `HilbertianProjection` via new `α_min_decay_tol`
  keyword. Effective lower bound is scaled by
  `min(1, max|C| / α_min_decay_tol)`, smoothly relaxing `α_min` as constraints
  saturate (`|C| → 0`). Suppresses oscillation around the feasible set when
  warm-starting from near-optimal geometries; default `0.0` preserves original
  behavior.
- `lsf_from_array(arr, domain; eps_zero)` in `src/Utilities.jl`: builds an
  `x -> arr[idx]` callable sampling a Cartesian array (e.g. SDF from SIMP
  results) at the nearest node. Optional `eps_zero` clamps near-zero entries
  with sign preservation to avoid CutFEM ill-conditioning at exact zeros.
- `combine_lsfs(fs, weights)` in `src/Utilities.jl`: weighted sum of level-set
  callables for blending analytic and array-sampled fields.
- 3D benchmark scripts under repository root:
  - `3D_2x1x1_MBB-porous.jl` / `3D_2x1x1_MBB-custom_sdf_seq.jl`
  - `3D_2x1x1_4Legs-porous.jl` / `3D_2x1x1_4Legs-custom_sdf_seq.jl`
  - Shared helpers in `script_utils.jl` (`visualize_boundary_conditions`,
    `write_history_energy`).
  - CLI-dispatched custom-SDF variants select from a predefined list of
    JLD2 datasets via a single integer argument.
- Reference SDF datasets under `data/3D_2x1x1_MBB/` and `data/3D_2x1x1_4Legs/`
  (JLD2, `raw_sdf` key) for reproducing the warm-start runs.

### Changed
- Adjusted README.md to discuss AD and link to docs.
- Now export `StateParamMap` and `val_and_gradient`.

## [0.4.1] - 2025-8-19

### Added
- Added options to optimise the state map update for `AffineFEStateMap` and `NonlinearFEStateMap`.
- Added transient tests.

### Fixed
- Bug fix in `StateParamMaps` to correctly use analytic gradient.

## [0.4.0] - 2025-7-15

### Added
- Added `Evolver` and `Reinitialiser` as part of full `LevelSetEvolution` refactor.
- Added `HeatReinitialiser` based on Feng and Crane (2024) [doi: 10.1145/3658220]. As of PR[#81](https://github.com/zjwegert/GridapTopOpt.jl/pull/81), similarly below.
- Added `IdentityReinitialiser`, does nothing.
- Added `get_element_diameters` methods of QUAD and HEX.

### Changed
- Refactored caching in StateMaps to remove constructor dependence on primal variable.
- Refactored `LevelSetEvolution` to split evolution and reinitialisation method.
- Deprecated `γ_reinit` from optimiser options.
- Deprecated `StateParamIntegrandWithMeasure` with an error.
- Warning when passing `U_reg` to state maps has been replaced with an error to fully deprecate methods.
- Disabled out-of-date methods in Benchmarks.
- Overhauled Breaking Changes section of Docs

## [0.3.0] - 2025-7-4

### Added
- Backwards AD via Zygote is now supported in serial and parallel. As of PR[#81](https://github.com/zjwegert/GridapTopOpt.jl/pull/80).

### Changed
- StateMaps now always differentiate into a consistent space. As of PR[#81](https://github.com/zjwegert/GridapTopOpt.jl/pull/80).
- Removed `U_reg` space from StateMaps. As of PR[#81](https://github.com/zjwegert/GridapTopOpt.jl/pull/80).
- Refactored allocation of vectors in distributed. As of PR[#81](https://github.com/zjwegert/GridapTopOpt.jl/pull/80).

### Fixed
- Resolved Issue[#46](https://github.com/zjwegert/GridapTopOpt.jl/issues/46)

## [0.2.2] - 2025-6-19

### Fixed
- Minor fixes and doc updates. As of PR[#77](https://github.com/zjwegert/GridapTopOpt.jl/pull/77).

## [0.2.1] - 2025-6-19

### Fixed
- Minor fixes and doc updates.

## [0.2.0] - 2025-6-18

### Added
- Added compatibility with GridapEmbedded for unfitted level-set topology optimisation. As of PR[#75](https://github.com/zjwegert/GridapTopOpt.jl/pull/75) and similarly below.
- Added isolated volume detection via polytopal cutting.
- Added embedded collections for updating embedded triangulations.
- Added unfitted evolution and reinitialisation methods.
- Added StaggeredFEStateMap for computing adjoints for problems involving [staggered FE problems](https://github.com/gridap/GridapSolvers.jl/blob/main/src/BlockSolvers/StaggeredFEOperators.jl).
- Added tests with finite differences for all StateMaps

### Changed
- Split ChainRules.jl into StateMaps/...
- Removed `IntegrandWithMeasure` and requirement to pass measures as arguments
