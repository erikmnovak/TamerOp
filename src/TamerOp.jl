module TamerOp
# =============================================================================
# TamerOp.jl
#
# Umbrella entrypoint for the library.
#
# This file has three jobs:
#   1) Load internal submodules in a stable include order.
#   2) Bind/export the curated public API contracts from APISurface.jl.
#   3) Define the opt-in broad surface for power users (TamerOp.Advanced).
#
# Notes:
# - `CoreModules.jl` is reserved for low-level runtime helpers and caches.
# - Higher-level option/data/result/encoding-map contracts live in sibling
#   modules (`Options`, `DataTypes`, `EncodingCore`, `Results`, `Stats`).
# - Workflow.jl defines orchestration wrappers (encode/coarsen/resolve/ext/tor/...).
# - DataFileIO.jl, DataIngestion.jl, and Featurizers.jl are standalone sibling
#   modules (not nested in Workflow) and are loaded before APISurface binding.
# =============================================================================

# 1) Low-level runtime, shared contracts, and light public data/result wrappers.
include("CoreModules.jl")
include("Stats.jl")
include("Options.jl")
include("DataTypes.jl")
include("EncodingCore.jl")
include("Results.jl")
include("RegionGeometry.jl")

# 2) Linear algebra engines (field-generic)
include("FieldLinAlg.jl")

# 3) Finite poset + indicator sets + fringe presentations
include("FiniteFringe.jl")

# 4) Shared record types and finite encodings built on top of fringe modules
include("IndicatorTypes.jl")
include("Encoding.jl")

# 5) Module and abelian-category layers
include("Modules.jl")
include("AbelianCategories.jl")

# 6) Indicator resolutions (resolution builders)
include("IndicatorResolutions.jl")

# 7) Zn flange data structure
include("FlangeZn.jl")

# 8) ZnEncoding (needs FlangeZn + FiniteFringe)
include("ZnEncoding.jl")

# 9) PL backends for R^n (general polyhedra + axis-aligned boxes)
include("PLPolyhedra.jl")
include("PLBackend.jl")

# 10) Derived-functor and complexes layer
include("ChainComplexes.jl")
include("DerivedFunctors.jl")
include("ModuleComplexes.jl")
include("ChangeOfPosets.jl")

# 11) JSON IO layer (internal formats + external adapters)
include("Serialization.jl")

# 12) Shared invariant plumbing + invariants and summaries
include("InvariantCore.jl")
include("SignedMeasures.jl")
include("SliceInvariants.jl")
include("Fibered2D.jl")
include("MultiparameterImages.jl")
include("Invariants.jl")

# 13) High-level workflow orchestration wrappers
include("Workflow.jl")
@doc raw"""
    Workflow

High-level orchestration owner for the canonical task-oriented workflow surface
in `TamerOp`.

- start here for `encode`, `resolve`, `invariant`, `hom_dimension`, `hom`,
  `ext`, `tor`, `slice_barcodes`, and `mp_landscape`,
- prefer the unified `?TamerOp.encode` help page for the two `encode`
  families (presentation encoding vs data-ingestion execution),
- use `cache=:auto` for one-shot calls and `cache=sc::SessionCache` for
  explicit cross-call reuse,
- prefer typed wrappers such as `EncodingResult`, `ResolutionResult`, and
  `InvariantResult` before opting into raw tuples or intermediate pipeline
  stages.

Workflow output policy
- `encode`, `resolve`, and `invariant` return typed wrappers by default,
- `hom`, `ext`, `tor`, `rhom`, `hyperext`, and `hypertor` return algebraic
  objects directly,
- convenience wrappers such as `rank_invariant(enc)` return bare values.
""" Workflow

# 14) File-oriented ingestion adapters
include("DataFileIO.jl")

# 15) Data ingestion subsystem
include("DataIngestion.jl")

# 16) Synthetic data generators
include("SyntheticData.jl")

# 17) Visualization engine
include("Visualization.jl")

# 18) Featurization/experiment subsystem
include("Featurizers.jl")

# -----------------------------------------------------------------------------
# Cross-subsystem semantic accessors
#
# `DataTypes` owns container-style counting/accessor names used across the
# project. Extend those generics onto finite-poset objects here, after both
# subsystems are loaded, so user/test surfaces can ask for `nvertices(...)`
# without needing to know which owner module provides the implementation.
# -----------------------------------------------------------------------------

DataTypes.nvertices(P::FiniteFringe.AbstractPoset) = FiniteFringe.nvertices(P)

# 18) Curated API surface contracts (simple + advanced)
include("APISurface.jl")

# -----------------------------------------------------------------------------
# Source-include optional extension loader
#
# In script/include mode (`include("src/TamerOp.jl")`) Julia package
# extensions are not auto-activated. Load selected extension modules explicitly
# when their dependency package is available.
# -----------------------------------------------------------------------------

@inline function _try_load_source_extension!(dep::Symbol, extmod::Symbol, extfile::String)
    isdefined(@__MODULE__, extmod) && return true
    try
        Base.require(Main, dep)
    catch
        return false
    end
    isdefined(@__MODULE__, extmod) && return true
    path = normpath(joinpath(@__DIR__, "..", "ext", extfile))
    isfile(path) || return false
    try
        include(path)
        return true
    catch
        return false
    end
end

const _SOURCE_INCLUDE_MODE = pathof(@__MODULE__) === nothing

const _SOURCE_EXT_NEARESTNEIGHBORS = _SOURCE_INCLUDE_MODE && _try_load_source_extension!(
    :NearestNeighbors,
    :TamerOpNearestNeighborsExt,
    "TamerOpNearestNeighborsExt.jl",
)

const _SOURCE_EXT_DELAUNAY = _SOURCE_INCLUDE_MODE && _try_load_source_extension!(
    :DelaunayTriangulation,
    :TamerOpDelaunayTriangulationExt,
    "TamerOpDelaunayTriangulationExt.jl",
)

# =============================================================================
# Curated Public API Surface
#
# APISurface.jl is the single source of truth for TamerOp' curated public
# symbol contracts. This file binds/exports those symbols.
#
# Everything not listed there remains accessible by qualification,
# e.g. 
#     TamerOp.Invariants.rank_invariant
#     TamerOp.DerivedFunctors.Ext
#
# Guiding principle:
#   - Most users should be productive with the workflow layer (encode/resolve/...).
#   - Power users should have a small set of composable "primitives" they can
#     use to build custom examples, do sanity checks, and pipeline/cache results.
# =============================================================================

# Root public bindings are explicitly contract-driven from APISurface.jl.
_bind_api_bindings!(@__MODULE__, SIMPLE_API_BINDINGS)

# -----------------------------------------------------------------------------
# Stable public exports
# -----------------------------------------------------------------------------
# Root exports are intentionally strict: the default public surface is exactly
# SIMPLE_API, plus the API contract constants for inspection/documentation.
_assert_api_list_defined!(@__MODULE__, SIMPLE_API; label="SIMPLE_API")
_export_api_list!(@__MODULE__, SIMPLE_API)
export SIMPLE_API, ADVANCED_API

# =============================================================================
# Advanced Namespace
#
# Opt-in broad surface for power users.
#
# The default stable public surface is bound from APISurface contracts and
# exported by top-level `TamerOp`.
#
# This section defines the nested module `TamerOp.Advanced`, intended for
# interactive and research use where convenience matters more than a small,
# curated namespace.
#
# Design goals:
# - `using TamerOp` stays clean and narrative for most users.
# - `using TamerOp.Advanced` is the curated power-user superset.
#
# Implementation strategy:
# - Always export major submodules (qualified access is never hidden).
# - Bind/export the explicit symbol contracts from APISurface.jl:
#     SIMPLE_API
#     ADVANCED_API
#     ADVANCED_ONLY_API_BINDINGS
# - No dynamic auto-lifting based on module export lists.
# =============================================================================

module Advanced

# Root package module (TamerOp)
const _ROOT = parentmodule(@__MODULE__)
import ..SIMPLE_API, ..ADVANCED_API, ..ADVANCED_ONLY_API_BINDINGS
import .._bind_api_list!, .._bind_api_bindings!, .._assert_api_list_defined!

# -----------------------------------------------------------------------------
# Always expose the major submodules (users can always qualify).
# -----------------------------------------------------------------------------

import ..CoreModules
import ..Stats
import ..Options
import ..DataTypes
import ..EncodingCore
import ..Results
import ..RegionGeometry
import ..FiniteFringe
import ..IndicatorTypes
import ..Encoding
import ..Modules
import ..AbelianCategories
import ..IndicatorResolutions
import ..FlangeZn
import ..Serialization
import ..SyntheticData
import ..Visualization
import ..PLPolyhedra
import ..PLBackend
import ..InvariantCore
import ..SignedMeasures
import ..SliceInvariants
import ..Fibered2D
import ..MultiparameterImages
import ..ChainComplexes
import ..ZnEncoding
import ..DerivedFunctors
import ..DerivedFunctors.Resolutions
import ..DerivedFunctors.ExtTorSpaces
import ..DerivedFunctors.Functoriality
import ..DerivedFunctors.Algebras
import ..DerivedFunctors.SpectralSequences
import ..DerivedFunctors.Backends
import ..DerivedFunctors.HomExtEngine
import ..DerivedFunctors.Utils
import ..ModuleComplexes
import ..ChangeOfPosets
import ..Invariants
import ..Workflow
import ..DataFileIO
import ..DataIngestion
import ..Featurizers

export CoreModules, Stats, Options, DataTypes, EncodingCore, Results,
       RegionGeometry, FiniteFringe, IndicatorTypes, Encoding,
       Modules, AbelianCategories, IndicatorResolutions, FlangeZn, Serialization, Visualization,
       PLPolyhedra, PLBackend, ChainComplexes, ZnEncoding, DerivedFunctors,
       Resolutions, ExtTorSpaces, Functoriality, Algebras, SpectralSequences, Backends,
       HomExtEngine, Utils,
       ModuleComplexes, ChangeOfPosets, Invariants, Workflow, DataFileIO, DataIngestion, Featurizers

# -----------------------------------------------------------------------------
# Static curated API binding (no dynamic lifting).
# -----------------------------------------------------------------------------

# Advanced re-exports the curated simple surface directly from root.
_bind_api_list!(@__MODULE__, _ROOT, SIMPLE_API)

# Advanced-only symbols come from explicit module bindings.
_bind_api_bindings!(@__MODULE__, ADVANCED_ONLY_API_BINDINGS)
_assert_api_list_defined!(@__MODULE__, ADVANCED_API; label="ADVANCED_API")

end # module Advanced

export Advanced

end # module
