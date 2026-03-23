# =============================================================================
# Featurizers.jl
#
# Featurizer specs, batch APIs, feature IO, and experiment runner for Workflow.
# This module is a sibling of Workflow.
# =============================================================================

module Featurizers

using LinearAlgebra
using SparseArrays
using JSON3
using Dates

using ..CoreModules: QQ, QQField, AbstractCoeffField, coeff_type, coerce,
                     ResolutionCache, SessionCache, EncodingCache,
                     _encoding_cache!, _session_resolution_cache, _session_hom_cache, _set_session_hom_cache!,
                     _session_slice_plan_cache, _set_session_slice_plan_cache!,
                     _resolve_workflow_session_cache, _resolve_workflow_specialized_cache,
                     _workflow_encoding_cache,
                     _resolution_cache_from_session,
                     _slot_cache_from_session,
                     _session_encoding_values, _session_module_values,
                     _session_encoding_bucket_count, _session_module_bucket_count,
                     _session_zn_encoding_artifact_count, _session_zn_pushforward_plan_count,
                     _session_zn_pushforward_fringe_count, _session_zn_pushforward_module_count
using ..Options: EncodingOptions, ResolutionOptions, DerivedFunctorOptions, InvariantOptions,
                 FiltrationSpec, ConstructionBudget, ConstructionOptions, PipelineOptions
using ..DataTypes: PointCloud, ImageNd, GraphData, EmbeddedPlanarGraph2D,
                   GradedComplex, MultiCriticalGradedComplex,
                   SimplexTreeMulti, simplex_count, max_simplex_dim, simplex_vertices, simplex_grades
using ..EncodingCore: AbstractPLikeEncodingMap, CompiledEncoding, compile_encoding, GridEncodingMap,
                      _compile_encoding_cached
using ..Results: EncodingResult, ResolutionResult, InvariantResult,
                 _encoding_with_session_cache, materialize_module
import ..EncodingCore: locate, dimension, representatives, axes_from_encoding, _grid_strides

import ..Serialization
import ..Serialization: TAMER_FEATURE_SCHEMA_VERSION,
                        feature_schema_header,
                        validate_feature_metadata_schema,
                        _write_feature_csv_wide,
                        _write_feature_csv_long

import ..IndicatorResolutions
import ..ZnEncoding
using ..IndicatorResolutions: pmodule_from_fringe
using ..PLPolyhedra
using ..PLBackend: BoxUpset, BoxDownset, encode_fringe_boxes
using ..Encoding: build_uptight_encoding_from_fringe,
                  pushforward_fringe_along_encoding,
                  PostcomposedEncodingMap
using ..Modules: PModule, PMorphism, cover_edges, dim_at
using ..FiniteFringe: AbstractPoset, FinitePoset, GridPoset, ProductOfChainsPoset, FringeModule,
                      Upset, Downset, principal_upset, principal_downset, leq, nvertices,
                      poset_equal_opposite
using ..DerivedFunctors
using ..Invariants
using ..InvariantCore: RankQueryCache
using ..SliceInvariants: CompiledSlicePlan, SlicePlanCache, SliceBarcodesTask,
                         PersistenceImage1D
using ..Fibered2D: ProjectedArrangement, ProjectedBarcodeCache, FiberedBarcodeCache2D,
                   PackedBarcodeGrid, PackedBarcode, PackedIndexBarcode, PackedFloatBarcode
using ..MultiparameterImages: MPPDecomposition, MPLandscape, MPPImage
using ..SignedMeasures: PointSignedMeasure, RectSignedBarcode, Rect
import ..SignedMeasures
import ..Modules
import ..ModuleComplexes
import ..ChainComplexes: describe
using ..ModuleComplexes: ModuleCochainComplex

using ..FlangeZn: Face, IndFlat, IndInj, Flange
import ..Workflow: _slice_plan_cache_from_session

# Featurizer specs and dataset featurization
# -----------------------------------------------------------------------------

abstract type AbstractFeaturizerSpec end

function feature_set_summary end
function experiment_summary end
function experiment_artifact_summary end
function featurizer_summary end
function sample_ids end
function feature_matrix end
function feature_values end
function sample_labels end
function metadata end
function experiment_name end
function artifacts end
function nartifacts end
function artifact_keys end
function artifact end
function artifact_spec end
function artifact_features end
function artifact_elapsed_seconds end
function has_features end
function check_feature_set end
function check_experiment_spec end
function check_loaded_experiment end
function check_featurizer_spec end

"""
    AbstractInvariantCache

Typed cache protocol used by featurizers for `build_cache` / `transform(spec, cache)`.
"""
abstract type AbstractInvariantCache end

"""
    BatchOptions

Execution contract for dataset-level batch APIs.
- `threaded`: enable parallel execution.
- `backend`: `:serial`, `:threads`, or `:folds` (optional extension backend).
- `progress`: reserved hook for optional progress extensions.
- `deterministic`: use static partitioning with deterministic slot writes.
- `chunk_size`: chunk size for static chunking (`0` means auto/static partition).
"""
struct BatchOptions
    threaded::Bool
    backend::Symbol
    progress::Bool
    deterministic::Bool
    chunk_size::Int
end

function BatchOptions(; threaded::Bool=true,
                        backend::Symbol = (threaded ? :threads : :serial),
                        progress::Bool=false,
                        deterministic::Bool=true,
                        chunk_size::Int=0)
    backend in (:serial, :threads, :folds) ||
        throw(ArgumentError("BatchOptions.backend must be :serial, :threads, or :folds"))
    chunk_size >= 0 || throw(ArgumentError("BatchOptions.chunk_size must be >= 0"))
    return BatchOptions(threaded, backend, progress, deterministic, chunk_size)
end

"""
    ModuleInvariantCache

Cache wrapper for module-only featurizers (`RankGridSpec`).
"""
struct ModuleInvariantCache{K,F<:AbstractCoeffField,MatT<:AbstractMatrix{K}} <: AbstractInvariantCache
    M::PModule{K,F,MatT}
    opts::InvariantOptions
    threads::Bool
    level::Symbol
end

"""
    RestrictedHilbertInvariantCache

Dedicated cache object for restricted-Hilbert-driven computations on `(M, pi)`.
This is currently used by `EulerSurfaceSpec` transforms.
"""
struct RestrictedHilbertInvariantCache{K,F<:AbstractCoeffField,MatT<:AbstractMatrix{K},PiT} <: AbstractInvariantCache
    M::PModule{K,F,MatT}
    pi::PiT
    opts::InvariantOptions
    threads::Bool
    level::Symbol
    hilbert::Vector{Int}
end

"""
    ResolutionInvariantCache

Cache wrapper for featurizers that consume a fixed projective or injective
resolution.
"""
struct ResolutionInvariantCache{ResT} <: AbstractInvariantCache
    res::ResT
    threads::Bool
    level::Symbol
    kind::Symbol
end

"""
    ResolutionMeasureInvariantCache

Cache wrapper for resolution-based featurizers that also need an encoding map
`pi` (for region weights / support measures).
"""
struct ResolutionMeasureInvariantCache{ResT,PiT,W} <: AbstractInvariantCache
    res::ResT
    pi::PiT
    opts::InvariantOptions
    threads::Bool
    level::Symbol
    kind::Symbol
    weights::W
end

"""
    EncodingInvariantCache

Cache wrapper for featurizers that require both `(M, pi)`.
It owns per-sample typed memo tables for slice plans and projected caches.
"""
mutable struct EncodingInvariantCache{K,F<:AbstractCoeffField,MatT<:AbstractMatrix{K},PiT} <: AbstractInvariantCache
    M::PModule{K,F,MatT}
    pi::PiT
    opts::InvariantOptions
    threads::Bool
    level::Symbol
    session_cache::Union{Nothing,SessionCache}
    slice_plan_cache::SlicePlanCache
    slice_plans::Dict{UInt,CompiledSlicePlan}
    projected_arrangements::Dict{UInt,ProjectedArrangement}
    projected_module::Dict{UInt,ProjectedBarcodeCache{K}}
    projected_refs::Dict{Tuple{UInt,UInt},ProjectedBarcodeCache}
    fibered::Dict{UInt,FiberedBarcodeCache2D{K}}
    mpp_decompositions::Dict{UInt,MPPDecomposition}
    rank_query::Union{Nothing,RankQueryCache}
end

"""
    PointMeasureInvariantCache

Direct cache wrapper for sparse signed point measures.
"""
struct PointMeasureInvariantCache{PmT} <: AbstractInvariantCache
    pm::PmT
    opts::InvariantOptions
    threads::Bool
    level::Symbol
end

"""
    SignedBarcodeInvariantCache

Direct cache wrapper for sparse rectangle signed barcodes.
"""
struct SignedBarcodeInvariantCache{SbT} <: AbstractInvariantCache
    sb::SbT
    opts::InvariantOptions
    threads::Bool
    level::Symbol
end

const _InvariantCacheHandle = Union{
    ModuleInvariantCache,
    RestrictedHilbertInvariantCache,
    ResolutionInvariantCache,
    ResolutionMeasureInvariantCache,
    EncodingInvariantCache,
    PointMeasureInvariantCache,
    SignedBarcodeInvariantCache,
}

"""
    LandscapeSpec

Typed spec for slice-based landscape vectorization.
"""
struct LandscapeSpec <: AbstractFeaturizerSpec
    directions::Vector{Vector{Float64}}
    offsets::Vector{Vector{Float64}}
    offset_weights::Union{Nothing,Vector{Float64},Matrix{Float64}}
    kmax::Int
    tgrid::Vector{Float64}
    aggregate::Symbol
    normalize_weights::Bool
    tmin::Union{Nothing,Float64}
    tmax::Union{Nothing,Float64}
    nsteps::Int
    strict::Bool
    drop_unknown::Bool
    dedup::Bool
    normalize_dirs::Symbol
    direction_weight::Symbol
    threads::Union{Nothing,Bool}
end

"""
    PersistenceImageSpec

Typed spec for slice-based persistence-image vectorization.
"""
struct PersistenceImageSpec <: AbstractFeaturizerSpec
    directions::Vector{Vector{Float64}}
    offsets::Vector{Vector{Float64}}
    offset_weights::Union{Nothing,Vector{Float64},Matrix{Float64}}
    xgrid::Vector{Float64}
    ygrid::Vector{Float64}
    sigma::Float64
    coords::Symbol
    weighting::Symbol
    p::Float64
    normalize::Symbol
    aggregate::Symbol
    normalize_weights::Bool
    tmin::Union{Nothing,Float64}
    tmax::Union{Nothing,Float64}
    nsteps::Int
    strict::Bool
    drop_unknown::Bool
    dedup::Bool
    normalize_dirs::Symbol
    direction_weight::Symbol
    threads::Union{Nothing,Bool}
end

"""
    MPLandscapeSpec

Typed spec for flattening a sampled multiparameter persistence landscape.
"""
struct MPLandscapeSpec <: AbstractFeaturizerSpec
    directions::Vector{Vector{Float64}}
    offsets::Vector{Vector{Float64}}
    offset_weights::Union{Nothing,Vector{Float64},Matrix{Float64}}
    kmax::Int
    tgrid::Vector{Float64}
    normalize_weights::Bool
    tmin::Union{Nothing,Float64}
    tmax::Union{Nothing,Float64}
    nsteps::Int
    strict::Bool
    drop_unknown::Bool
    dedup::Bool
    normalize_dirs::Symbol
    direction_weight::Symbol
    threads::Union{Nothing,Bool}
end

"""
    EulerSurfaceSpec

Typed spec for Euler characteristic surface vectorization.
"""
struct EulerSurfaceSpec{A<:Union{Nothing,NTuple}} <: AbstractFeaturizerSpec
    axes::A
    axes_policy::Symbol
    max_axis_len::Int
    strict::Union{Nothing,Bool}
    threads::Union{Nothing,Bool}
end

"""
    RankGridSpec

Typed spec for flattening rank invariant on an n x n grid of region pairs.
"""
struct RankGridSpec <: AbstractFeaturizerSpec
    nvertices::Int
    store_zeros::Bool
    threads::Union{Nothing,Bool}
end

"""
    RestrictedHilbertSpec

Typed spec for flattening the restricted Hilbert function to a fixed-length
feature vector indexed by poset vertex.
"""
struct RestrictedHilbertSpec <: AbstractFeaturizerSpec
    nvertices::Int
end

"""
    BarcodeTopKSpec

Canonical raw-barcode featurizer for slice/fibered families.

Policy:
- per slice, sort intervals by persistence descending, then birth ascending, then death ascending,
- keep the top `k`,
- encode `[count, present_1, birth_1, persistence_1, ..., present_k, birth_k, persistence_k]`,
- missing slots are padded with zeros,
- `source` selects only the barcode backend (`:slice` or `:fibered`); feature semantics stay the same.
"""
struct BarcodeTopKSpec <: AbstractFeaturizerSpec
    directions::Vector{Vector{Float64}}
    offsets::Vector{Vector{Float64}}
    offset_weights::Union{Nothing,Vector{Float64},Matrix{Float64}}
    k::Int
    aggregate::Symbol
    source::Symbol
    infinite_policy::Symbol
    window::Union{Nothing,Tuple{Float64,Float64}}
    normalize_weights::Bool
    tmin::Union{Nothing,Float64}
    tmax::Union{Nothing,Float64}
    nsteps::Int
    strict::Bool
    drop_unknown::Bool
    dedup::Bool
    normalize_dirs::Symbol
    direction_weight::Symbol
    threads::Union{Nothing,Bool}
end

"""
    SlicedBarcodeSpec

Typed spec for slice-barcode derived features (summary/entropy).
"""
struct SlicedBarcodeSpec{S<:Tuple} <: AbstractFeaturizerSpec
    directions::Vector{Vector{Float64}}
    offsets::Vector{Vector{Float64}}
    offset_weights::Union{Nothing,Vector{Float64},Matrix{Float64}}
    featurizer::Symbol
    summary_fields::S
    summary_normalize_entropy::Bool
    entropy_normalize::Bool
    entropy_weighting::Symbol
    entropy_p::Float64
    aggregate::Symbol
    normalize_weights::Bool
    tmin::Union{Nothing,Float64}
    tmax::Union{Nothing,Float64}
    nsteps::Int
    strict::Bool
    drop_unknown::Bool
    dedup::Bool
    normalize_dirs::Symbol
    direction_weight::Symbol
    threads::Union{Nothing,Bool}
end

"""
    BarcodeSummarySpec

Canonical fixed-field barcode-summary featurizer for slice/fibered families.

Policy:
- default summary fields are `(:count, :sum_persistence, :mean_persistence, :max_persistence, :entropy)`,
- aggregation across many barcodes uses the same `:mean` / `:sum` / `:stack` contract as slice featurizers,
- `source` switches only the barcode backend (`:slice` or `:fibered`).
"""
struct BarcodeSummarySpec{S<:Tuple} <: AbstractFeaturizerSpec
    directions::Vector{Vector{Float64}}
    offsets::Vector{Vector{Float64}}
    offset_weights::Union{Nothing,Vector{Float64},Matrix{Float64}}
    fields::S
    normalize_entropy::Bool
    aggregate::Symbol
    source::Symbol
    infinite_policy::Symbol
    window::Union{Nothing,Tuple{Float64,Float64}}
    normalize_weights::Bool
    tmin::Union{Nothing,Float64}
    tmax::Union{Nothing,Float64}
    nsteps::Int
    strict::Bool
    drop_unknown::Bool
    dedup::Bool
    normalize_dirs::Symbol
    direction_weight::Symbol
    threads::Union{Nothing,Bool}
end

"""
    SignedBarcodeImageSpec

Typed spec for rectangle signed-barcode image vectorization.
"""
struct SignedBarcodeImageSpec{A<:Union{Nothing,NTuple}} <: AbstractFeaturizerSpec
    xs::Vector{Float64}
    ys::Vector{Float64}
    sigma::Float64
    mode::Symbol
    method::Symbol
    axes::A
    axes_policy::Symbol
    max_axis_len::Int
    strict::Union{Nothing,Bool}
    threads::Union{Nothing,Bool}
end

"""
    PointSignedMeasureSpec

Canonical sparse featurizer for a `PointSignedMeasure`.

Policy:
- sort support points by `abs(weight)` descending, then lexicographic support index,
- keep the top `k`,
- encode `[count, present_1, weight_1, x_1^(1), ..., x_n^(1), ..., present_k, weight_k, ...]`,
- missing slots are padded with zeros,
- `coords` chooses whether slot coordinates use axis values (`:values`) or grid indices (`:indices`).
"""
struct PointSignedMeasureSpec <: AbstractFeaturizerSpec
    ndims::Int
    k::Int
    coords::Symbol
end

"""
    EulerSignedMeasureSpec

Canonical sparse featurizer for the Euler signed measure of an encoded sample.

Policy:
- compute `euler_signed_measure` on the chosen grid,
- then apply the same sparse top-k contract as `PointSignedMeasureSpec`,
- ordering is by `abs(weight)` descending, then lexicographic support index,
- slots encode `(present, weight, coordinates...)` and pad with zeros.
"""
struct EulerSignedMeasureSpec{A<:Union{Nothing,NTuple}} <: AbstractFeaturizerSpec
    ndims::Int
    k::Int
    coords::Symbol
    axes::A
    axes_policy::Symbol
    max_axis_len::Int
    strict::Union{Nothing,Bool}
    drop_zeros::Bool
    max_terms::Int
    min_abs_weight::Float64
    threads::Union{Nothing,Bool}
end

"""
    RectangleSignedBarcodeTopKSpec

Sparse top-k featurizer for rectangle signed barcodes.

Policy:
- sort rectangles by `abs(weight)` descending, then lexicographic `(lo, hi)`,
- keep the top `k`,
- encode `[count, present_1, weight_1, lo_1^(1), ..., hi_n^(1), ..., present_k, ...]`,
- missing slots are padded with zeros,
- `coords` chooses axis values (`:values`) or grid indices (`:indices`) for rectangle corners.
"""
struct RectangleSignedBarcodeTopKSpec{A<:Union{Nothing,NTuple},S<:Union{Nothing,Int,Tuple}} <: AbstractFeaturizerSpec
    ndims::Int
    k::Int
    coords::Symbol
    axes::A
    axes_policy::Symbol
    max_axis_len::Int
    strict::Union{Nothing,Bool}
    drop_zeros::Bool
    tol::Int
    max_span::S
    keep_endpoints::Bool
    method::Symbol
    bulk_max_elems::Int
    threads::Union{Nothing,Bool}
end

"""
    ProjectedDistancesSpec

Typed spec for projected-distance features against a fixed reference bank.
"""
struct ProjectedDistancesSpec{R<:AbstractVector,D<:Union{Nothing,Vector{Vector{Float64}}}} <: AbstractFeaturizerSpec
    references::R
    reference_names::Vector{Symbol}
    dist::Symbol
    p::Float64
    q::Float64
    agg::Symbol
    directions::D
    n_dirs::Int
    normalize::Symbol
    enforce_monotone::Symbol
    precompute::Bool
    threads::Union{Nothing,Bool}
end

"""
    MatchingDistanceBankSpec

Reference-bank featurizer using multiparameter matching distance.

Each feature is the matching distance from the sample to one reference module
in a common encoding family. This keeps matching-distance use inside the same
batch featurizer surface as the other bank-based specs.
"""
struct MatchingDistanceBankSpec{R<:AbstractVector,D<:Union{Nothing,Vector{Vector{Float64}}},O<:Union{Nothing,Vector{Vector{Float64}}}} <: AbstractFeaturizerSpec
    references::R
    reference_names::Vector{Symbol}
    method::Symbol
    directions::D
    offsets::O
    n_dirs::Int
    n_offsets::Int
    max_den::Int
    include_axes::Bool
    normalize_dirs::Symbol
    weight::Symbol
    threads::Union{Nothing,Bool}
end

"""
    MPPImageSpec

Typed spec for flattening a 2D multiparameter persistence image.
"""
struct MPPImageSpec <: AbstractFeaturizerSpec
    resolution::Int
    xgrid::Union{Nothing,Vector{Float64}}
    ygrid::Union{Nothing,Vector{Float64}}
    sigma::Float64
    N::Int
    delta::Union{Symbol,Float64}
    q::Float64
    tie_break::Symbol
    cutoff_radius::Union{Nothing,Float64}
    cutoff_tol::Union{Nothing,Float64}
    segment_prune::Bool
    threads::Union{Nothing,Bool}
end

"""
    MPPDecompositionHistogramSpec

Stable histogram featurizer for a raw `MPPDecomposition`.

Policy:
- histogram segments by orientation and segment-length scale,
- normalize segment lengths by the decomposition-box diagonal before binning,
- `weight=:count` counts segments,
- `weight=:mass` distributes each summand weight equally across its segments so total mass is preserved,
- append global scalars `(total_mass, n_summands, n_segments, max_scale, entropy)`.
"""
struct MPPDecompositionHistogramSpec <: AbstractFeaturizerSpec
    orientation_bins::Int
    scale_bins::Int
    scale_range::Union{Nothing,Tuple{Float64,Float64}}
    weight::Symbol
    normalize::Symbol
    N::Int
    delta::Union{Symbol,Float64}
    q::Float64
    tie_break::Symbol
    threads::Union{Nothing,Bool}
end

"""
    BettiTableSpec

Typed spec for flattening a projective Betti table to a fixed-size feature
vector.
"""
struct BettiTableSpec <: AbstractFeaturizerSpec
    nvertices::Int
    pad_to::Int
    resolution_maxlen::Int
    minimal::Bool
    check::Bool
end

"""
    BassTableSpec

Typed spec for flattening an injective Bass table to a fixed-size feature
vector.
"""
struct BassTableSpec <: AbstractFeaturizerSpec
    nvertices::Int
    pad_to::Int
    resolution_maxlen::Int
    minimal::Bool
    check::Bool
end

"""
    BettiSupportMeasuresSpec

Typed spec for flattening support/mass summaries of multigraded Betti numbers.
"""
struct BettiSupportMeasuresSpec <: AbstractFeaturizerSpec
    pad_to::Int
    resolution_maxlen::Int
    minimal::Bool
    check::Bool
end

"""
    BassSupportMeasuresSpec

Typed spec for flattening support/mass summaries of multigraded Bass numbers.
"""
struct BassSupportMeasuresSpec <: AbstractFeaturizerSpec
    pad_to::Int
    resolution_maxlen::Int
    minimal::Bool
    check::Bool
end

"""
    CompositeSpec

Concatenate multiple featurizer specs in a deterministic order.
"""
struct CompositeSpec{S<:Tuple} <: AbstractFeaturizerSpec
    specs::S
    namespacing::Bool
end

"""
    FeatureSet

Canonical output for batch featurization.
`X` uses row-major sample layout: size(X) == (nsamples, nfeatures).
"""
struct FeatureSet{T,MT}
    X::Matrix{T}
    names::Vector{Symbol}
    ids::Vector{String}
    meta::MT
end

Base.size(F::FeatureSet) = size(F.X)
@inline nsamples(F::FeatureSet) = size(F.X, 1)
@inline nfeatures(F::FeatureSet) = size(F.X, 2)
@inline asrowmajor(F::FeatureSet) = F.X
@inline ascolmajor(F::FeatureSet) = permutedims(F.X)
@inline feature_axes(F::FeatureSet) = _meta_get(F.meta, :feature_axes, nothing)

"""
    FeatureSetWideTable(fs)
    FeatureSetLongTable(fs)

Table wrappers used by optional `Tables.jl` extension.
"""
struct FeatureSetWideTable{T}
    fs::T
end

struct FeatureSetLongTable{T}
    fs::T
end

const TAMER_EXPERIMENT_SCHEMA_VERSION = v"0.1.0"

"""
    ExperimentIOConfig

Output policy for `run_experiment`.

- `format` controls table mode (`:wide` / `:long`) for table-based writers.
- `formats` controls which artifact files are emitted per featurizer:
  `:arrow`, `:parquet`, `:npz`, `:csv_wide`, `:csv_long`.
"""
struct ExperimentIOConfig
    outdir::Union{Nothing,String}
    prefix::String
    format::Symbol
    formats::Vector{Symbol}
    write_metadata::Bool
    overwrite::Bool
end

function ExperimentIOConfig(; outdir::Union{Nothing,AbstractString}=nothing,
                              prefix::AbstractString="experiment",
                              format::Symbol=:wide,
                              formats::Union{Symbol,AbstractVector,Tuple}=(:arrow,),
                              write_metadata::Bool=true,
                              overwrite::Bool=true)
    format in (:wide, :long) || throw(ArgumentError("ExperimentIOConfig.format must be :wide or :long"))
    formats0 = Symbol[]
    raw = formats isa Symbol ? (formats,) : formats
    raw isa AbstractVector || raw isa Tuple ||
        throw(ArgumentError("ExperimentIOConfig.formats must be a Symbol, tuple, or vector"))
    for f in raw
        fs = Symbol(f)
        fs in (:arrow, :parquet, :npz, :csv_wide, :csv_long) ||
            throw(ArgumentError("ExperimentIOConfig.formats contains unsupported format $(fs)"))
        push!(formats0, fs)
    end
    # deterministic order + dedup for manifest reproducibility
    pref = Dict(:arrow=>1, :parquet=>2, :npz=>3, :csv_wide=>4, :csv_long=>5)
    formats1 = unique(formats0)
    sort!(formats1; by=f->pref[f])
    return ExperimentIOConfig(outdir === nothing ? nothing : String(outdir),
                              String(prefix),
                              format,
                              formats1,
                              write_metadata,
                              overwrite)
end

"""
    ExperimentSpec

Typed experiment plan: list of featurizers + shared execution/output settings.
"""
struct ExperimentSpec{S<:Tuple,F1,F2,C,MT}
    name::String
    featurizers::S
    opts::InvariantOptions
    batch::BatchOptions
    cache::C
    on_unsupported::Symbol
    idfun::F1
    labelfun::F2
    io::ExperimentIOConfig
    metadata::MT
end

@inline function _normalize_featurizers(specs)
    if specs isa AbstractFeaturizerSpec
        return (specs,)
    elseif specs isa Tuple
        all(s -> s isa AbstractFeaturizerSpec, specs) ||
            throw(ArgumentError("ExperimentSpec.featurizers tuple must contain only AbstractFeaturizerSpec items"))
        return specs
    elseif specs isa AbstractVector
        all(s -> s isa AbstractFeaturizerSpec, specs) ||
            throw(ArgumentError("ExperimentSpec.featurizers vector must contain only AbstractFeaturizerSpec items"))
        return tuple(specs...)
    end
    throw(ArgumentError("ExperimentSpec.featurizers must be a spec, tuple of specs, or vector of specs"))
end

function ExperimentSpec(featurizers;
                        name::AbstractString="experiment",
                        opts::InvariantOptions=InvariantOptions(),
                        batch::Union{BatchOptions,Nothing}=nothing,
                        cache=:auto,
                        on_unsupported::Symbol=:error,
                        idfun=nothing,
                        labelfun=nothing,
                        io::ExperimentIOConfig=ExperimentIOConfig(),
                        metadata=NamedTuple())
    specs = _normalize_featurizers(featurizers)
    isempty(specs) && throw(ArgumentError("ExperimentSpec requires at least one featurizer"))
    (on_unsupported in (:error, :skip, :missing)) ||
        throw(ArgumentError("ExperimentSpec.on_unsupported must be :error, :skip, or :missing"))
    (cache === :auto || cache === nothing || cache isa SessionCache) ||
        throw(ArgumentError("ExperimentSpec.cache must be :auto, nothing, or SessionCache"))
    opts0 = opts
    batch0 = batch === nothing ? BatchOptions() : batch
    return ExperimentSpec{typeof(specs),typeof(idfun),typeof(labelfun),typeof(cache),typeof(metadata)}(
        String(name),
        specs,
        opts0,
        batch0,
        cache,
        on_unsupported,
        idfun,
        labelfun,
        io,
        metadata,
    )
end

"""
    ExperimentArtifact

One featurizer run in an experiment.
"""
struct ExperimentArtifact{F<:AbstractFeaturizerSpec,FS}
    key::Symbol
    spec::F
    features::FS
    elapsed_seconds::Float64
    cache_stats::Dict{String,Any}
    feature_paths::Dict{Symbol,String}
    metadata_path::Union{Nothing,String}
end

"""
    ExperimentResult

Output of `run_experiment`.
"""
struct ExperimentResult{ES,AV,MT}
    spec::ES
    artifacts::AV
    total_elapsed_seconds::Float64
    run_dir::Union{Nothing,String}
    manifest_path::Union{Nothing,String}
    metadata::MT
end

"""
    LoadedExperimentArtifact

Artifact record returned by `load_experiment`.
"""
struct LoadedExperimentArtifact
    key::Symbol
    spec::Union{Nothing,AbstractFeaturizerSpec}
    opts::Union{Nothing,InvariantOptions}
    features::Union{Nothing,FeatureSet}
    metadata::Any
    elapsed_seconds::Float64
    cache_stats::Dict{String,Any}
    feature_paths::Dict{Symbol,String}
    metadata_path::Union{Nothing,String}
end

"""
    LoadedExperimentResult

Return type for `load_experiment`.
"""
struct LoadedExperimentResult
    manifest::Dict{String,Any}
    artifacts::Vector{LoadedExperimentArtifact}
    run_dir::Union{Nothing,String}
    manifest_path::String
    total_elapsed_seconds::Float64
end

"""
    FeaturizerSpecValidationSummary

Typed validation summary returned by [`check_featurizer_spec`](@ref).
"""
struct FeaturizerSpecValidationSummary
    kind::Symbol
    valid::Bool
    spec_type::Symbol
    nfeatures::Int
    issues::Vector{String}
end

"""
    FeatureSetValidationSummary

Typed validation summary returned by [`check_feature_set`](@ref).
"""
struct FeatureSetValidationSummary
    kind::Symbol
    valid::Bool
    nsamples::Int
    nfeatures::Int
    issues::Vector{String}
end

"""
    ExperimentSpecValidationSummary

Typed validation summary returned by [`check_experiment_spec`](@ref).
"""
struct ExperimentSpecValidationSummary
    kind::Symbol
    valid::Bool
    experiment_name::String
    nfeaturizers::Int
    invalid_specs::Vector{Symbol}
    issues::Vector{String}
end

"""
    LoadedExperimentValidationSummary

Typed validation summary returned by [`check_loaded_experiment`](@ref).
"""
struct LoadedExperimentValidationSummary
    kind::Symbol
    valid::Bool
    nartifacts::Int
    artifacts_with_features::Int
    issues::Vector{String}
end

@inline feature_names(fs::FeatureSet) = fs.names
@inline sample_ids(fs::FeatureSet) = fs.ids
@inline feature_matrix(fs::FeatureSet) = fs.X
@inline feature_values(fs::FeatureSet) = fs.X
@inline sample_labels(fs::FeatureSet) = _meta_get(fs.meta, :labels, nothing)
@inline metadata(fs::FeatureSet) = fs.meta

@inline experiment_name(exp::ExperimentSpec) = exp.name
@inline experiment_name(res::ExperimentResult) = experiment_name(res.spec)
@inline function experiment_name(res::LoadedExperimentResult)
    return String(get(res.manifest, "experiment_name", "experiment"))
end

@inline artifacts(res::Union{ExperimentResult,LoadedExperimentResult}) = res.artifacts
@inline nartifacts(res::Union{ExperimentResult,LoadedExperimentResult}) = length(artifacts(res))
@inline artifact_keys(res::Union{ExperimentResult,LoadedExperimentResult}) = [a.key for a in artifacts(res)]

function artifact(res::Union{ExperimentResult,LoadedExperimentResult}, key::Symbol)
    idx = findfirst(==(key), artifact_keys(res))
    idx === nothing && throw(KeyError(key))
    return artifacts(res)[idx]
end

@inline artifact_spec(a::Union{ExperimentArtifact,LoadedExperimentArtifact}) = a.spec
@inline artifact_features(a::ExperimentArtifact) = a.features
@inline artifact_features(a::LoadedExperimentArtifact) = a.features
@inline artifact_elapsed_seconds(a::Union{ExperimentArtifact,LoadedExperimentArtifact}) = a.elapsed_seconds
@inline feature_paths(a::Union{ExperimentArtifact,LoadedExperimentArtifact}) = a.feature_paths
@inline manifest_path(res::Union{ExperimentResult,LoadedExperimentResult}) = res.manifest_path
@inline run_dir(res::Union{ExperimentResult,LoadedExperimentResult}) = res.run_dir
@inline total_elapsed_seconds(res::Union{ExperimentResult,LoadedExperimentResult}) = res.total_elapsed_seconds
@inline has_features(::ExperimentArtifact) = true
@inline has_features(a::LoadedExperimentArtifact) = a.features !== nothing
@inline has_features(res::Union{ExperimentResult,LoadedExperimentResult}) = any(has_features, artifacts(res))

@inline function _meta_get(meta::NamedTuple, key::Symbol, default=nothing)
    return haskey(meta, key) ? getfield(meta, key) : default
end
@inline function _meta_get(meta::AbstractDict, key::Symbol, default=nothing)
    return haskey(meta, key) ? meta[key] : default
end
@inline function _meta_get(meta, key::Symbol, default=nothing)
    return hasproperty(meta, key) ? getproperty(meta, key) : default
end

@inline function _meta_keys(meta)
    if meta isa NamedTuple
        return collect(keys(meta))
    elseif meta isa AbstractDict
        return Symbol[Symbol(k) for k in keys(meta)]
    end
    return Symbol[Symbol(k) for k in propertynames(meta)]
end

@inline function _spec_field_namedtuple(spec::AbstractFeaturizerSpec)
    names = fieldnames(typeof(spec))
    vals = ntuple(i -> getfield(spec, names[i]), length(names))
    return NamedTuple{names}(vals)
end

@inline function _throw_invalid_featurizer(kind_name::AbstractString, issues::Vector{String})
    msg = isempty(issues) ? "invalid $(kind_name)" : join(issues, "; ")
    throw(ArgumentError("$(kind_name): $(msg)"))
end

@inline function _push_issue!(issues::Vector{String}, msg::AbstractString)
    push!(issues, String(msg))
    return issues
end

@inline function _strictly_increasing(v)
    @inbounds for i in 2:length(v)
        v[i] > v[i - 1] || return false
    end
    return true
end

function _check_axis_payload!(issues::Vector{String}, axis, label::AbstractString)
    isempty(axis) && _push_issue!(issues, "$(label) must be nonempty.")
    all(isfinite, axis) || _push_issue!(issues, "$(label) must contain only finite entries.")
    length(axis) <= 1 || _strictly_increasing(axis) ||
        _push_issue!(issues, "$(label) must be strictly increasing with no duplicates.")
    return issues
end

function _check_axes_payload!(issues::Vector{String}, axes, label::AbstractString="axes")
    axes === nothing && return issues
    for i in eachindex(axes)
        _check_axis_payload!(issues, axes[i], "$(label)[$i]")
    end
    return issues
end

function _check_directions_offsets!(issues::Vector{String},
                                    directions,
                                    offsets,
                                    offset_weights,
                                    label::AbstractString)
    nd = length(directions)
    no = length(offsets)
    nd > 0 || _push_issue!(issues, "$(label).directions must be nonempty.")
    no > 0 || _push_issue!(issues, "$(label).offsets must be nonempty.")

    dirdim = nothing
    for i in eachindex(directions)
        dir = directions[i]
        isempty(dir) && _push_issue!(issues, "$(label).directions[$i] must be nonempty.")
        all(isfinite, dir) || _push_issue!(issues, "$(label).directions[$i] must contain only finite entries.")
        any(x -> x < 0, dir) && _push_issue!(issues, "$(label).directions[$i] must have nonnegative entries.")
        any(x -> x > 0, dir) || _push_issue!(issues, "$(label).directions[$i] must contain a positive entry.")
        if dirdim === nothing
            dirdim = length(dir)
        elseif length(dir) != dirdim
            _push_issue!(issues, "$(label).directions must all have the same ambient dimension.")
            break
        end
    end
    dirdim = something(dirdim, 0)

    offdim = nothing
    for j in eachindex(offsets)
        off = offsets[j]
        isempty(off) && _push_issue!(issues, "$(label).offsets[$j] must be nonempty.")
        all(isfinite, off) || _push_issue!(issues, "$(label).offsets[$j] must contain only finite entries.")
        if offdim === nothing
            offdim = length(off)
        elseif length(off) != offdim
            _push_issue!(issues, "$(label).offsets must all have the same ambient dimension.")
            break
        end
    end
    offdim = something(offdim, 0)

    dirdim == 0 || offdim == 0 || dirdim == offdim ||
        _push_issue!(issues, "$(label).directions and $(label).offsets must have the same ambient dimension.")

    if offset_weights isa AbstractVector
        length(offset_weights) == no ||
            _push_issue!(issues, "$(label).offset_weights vector must have length equal to the number of offsets.")
        all(isfinite, offset_weights) ||
            _push_issue!(issues, "$(label).offset_weights vector must contain only finite entries.")
    elseif offset_weights isa AbstractMatrix
        size(offset_weights) == (nd, no) ||
            _push_issue!(issues, "$(label).offset_weights matrix must have size (ndirections, noffsets).")
        all(isfinite, offset_weights) ||
            _push_issue!(issues, "$(label).offset_weights matrix must contain only finite entries.")
    elseif offset_weights !== nothing
        _push_issue!(issues, "$(label).offset_weights must be a vector, a matrix, or nothing.")
    end

    return issues
end

function _check_common_slice_spec!(issues::Vector{String}, spec; label::AbstractString)
    _check_directions_offsets!(issues, spec.directions, spec.offsets, spec.offset_weights, label)
    spec.nsteps > 0 || _push_issue!(issues, "$(label).nsteps must be > 0.")
    spec.normalize_dirs in (:none, :L1, :Linf) ||
        _push_issue!(issues, "$(label).normalize_dirs must be :none, :L1, or :Linf.")
    spec.direction_weight in (:none, :lesnick_l1, :lesnick_linf) ||
        _push_issue!(issues, "$(label).direction_weight must be :none, :lesnick_l1, or :lesnick_linf.")
    if spec.tmin !== nothing
        isfinite(spec.tmin) || _push_issue!(issues, "$(label).tmin must be finite when provided.")
    end
    if spec.tmax !== nothing
        isfinite(spec.tmax) || _push_issue!(issues, "$(label).tmax must be finite when provided.")
    end
    if spec.tmin !== nothing && spec.tmax !== nothing
        spec.tmin <= spec.tmax ||
            _push_issue!(issues, "$(label).tmin must be <= $(label).tmax.")
    end
    return issues
end

@inline function _check_feature_grid!(issues::Vector{String}, grid, label::AbstractString)
    _check_axis_payload!(issues, grid, label)
    return issues
end

function _check_axes_policy!(issues::Vector{String}, axes_policy::Symbol, label::AbstractString)
    axes_policy in (:encoding, :coarsen, :as_given) ||
        _push_issue!(issues, "$(label).axes_policy must be :encoding, :coarsen, or :as_given.")
    return issues
end

function _check_reference_names!(issues::Vector{String}, refs, names, label::AbstractString)
    isempty(refs) && _push_issue!(issues, "$(label).references must be nonempty.")
    length(names) == length(refs) ||
        _push_issue!(issues, "$(label).reference_names length must match the number of references.")
    length(unique(names)) == length(names) ||
        _push_issue!(issues, "$(label).reference_names must be unique.")
    return issues
end

function _check_spec_issues(spec::LandscapeSpec)
    issues = String[]
    _check_common_slice_spec!(issues, spec; label="LandscapeSpec")
    spec.kmax > 0 || _push_issue!(issues, "LandscapeSpec.kmax must be > 0.")
    spec.aggregate in (:mean, :sum, :stack) ||
        _push_issue!(issues, "LandscapeSpec.aggregate must be :mean, :sum, or :stack.")
    _check_feature_grid!(issues, spec.tgrid, "LandscapeSpec.tgrid")
    return issues
end

function _check_spec_issues(spec::PersistenceImageSpec)
    issues = String[]
    _check_common_slice_spec!(issues, spec; label="PersistenceImageSpec")
    _check_feature_grid!(issues, spec.xgrid, "PersistenceImageSpec.xgrid")
    _check_feature_grid!(issues, spec.ygrid, "PersistenceImageSpec.ygrid")
    spec.sigma > 0 || _push_issue!(issues, "PersistenceImageSpec.sigma must be > 0.")
    spec.p > 0 || _push_issue!(issues, "PersistenceImageSpec.p must be > 0.")
    spec.coords in (:birth_persistence, :birth_death, :midlife_persistence) ||
        _push_issue!(issues, "PersistenceImageSpec.coords must be :birth_persistence, :birth_death, or :midlife_persistence.")
    spec.weighting in (:none, :persistence) ||
        _push_issue!(issues, "PersistenceImageSpec.weighting must be :none or :persistence.")
    spec.normalize in (:none, :l1, :l2, :max) ||
        _push_issue!(issues, "PersistenceImageSpec.normalize must be :none, :l1, :l2, or :max.")
    spec.aggregate in (:mean, :sum, :stack) ||
        _push_issue!(issues, "PersistenceImageSpec.aggregate must be :mean, :sum, or :stack.")
    return issues
end

function _check_spec_issues(spec::MPLandscapeSpec)
    issues = String[]
    _check_common_slice_spec!(issues, spec; label="MPLandscapeSpec")
    spec.kmax > 0 || _push_issue!(issues, "MPLandscapeSpec.kmax must be > 0.")
    _check_feature_grid!(issues, spec.tgrid, "MPLandscapeSpec.tgrid")
    return issues
end

function _check_spec_issues(spec::EulerSurfaceSpec)
    issues = String[]
    _check_axes_policy!(issues, spec.axes_policy, "EulerSurfaceSpec")
    spec.max_axis_len > 0 || _push_issue!(issues, "EulerSurfaceSpec.max_axis_len must be > 0.")
    _check_axes_payload!(issues, spec.axes, "EulerSurfaceSpec.axes")
    return issues
end

function _check_spec_issues(spec::RankGridSpec)
    issues = String[]
    spec.nvertices >= 0 || _push_issue!(issues, "RankGridSpec.nvertices must be >= 0.")
    return issues
end

function _check_spec_issues(spec::RestrictedHilbertSpec)
    issues = String[]
    spec.nvertices >= 0 || _push_issue!(issues, "RestrictedHilbertSpec.nvertices must be >= 0.")
    return issues
end

function _check_spec_issues(spec::BarcodeTopKSpec)
    issues = String[]
    _check_common_slice_spec!(issues, spec; label="BarcodeTopKSpec")
    spec.k >= 0 || _push_issue!(issues, "BarcodeTopKSpec.k must be >= 0.")
    spec.aggregate in (:mean, :sum, :stack) ||
        _push_issue!(issues, "BarcodeTopKSpec.aggregate must be :mean, :sum, or :stack.")
    spec.source in (:slice, :fibered) ||
        _push_issue!(issues, "BarcodeTopKSpec.source must be :slice or :fibered.")
    spec.infinite_policy in (:clip_to_window, :error) ||
        _push_issue!(issues, "BarcodeTopKSpec.infinite_policy must be :clip_to_window or :error.")
    if spec.window !== nothing
        all(isfinite, spec.window) || _push_issue!(issues, "BarcodeTopKSpec.window must contain finite bounds.")
        spec.window[1] <= spec.window[2] ||
            _push_issue!(issues, "BarcodeTopKSpec.window must satisfy lo <= hi.")
    end
    return issues
end

function _check_spec_issues(spec::SlicedBarcodeSpec)
    issues = String[]
    _check_common_slice_spec!(issues, spec; label="SlicedBarcodeSpec")
    spec.aggregate in (:mean, :sum, :stack) ||
        _push_issue!(issues, "SlicedBarcodeSpec.aggregate must be :mean, :sum, or :stack.")
    spec.featurizer in (:summary, :entropy) ||
        _push_issue!(issues, "SlicedBarcodeSpec.featurizer must be :summary or :entropy.")
    spec.entropy_weighting in (:none, :persistence) ||
        _push_issue!(issues, "SlicedBarcodeSpec.entropy_weighting must be :none or :persistence.")
    spec.entropy_p > 0 || _push_issue!(issues, "SlicedBarcodeSpec.entropy_p must be > 0.")
    if spec.featurizer == :summary
        isempty(spec.summary_fields) &&
            _push_issue!(issues, "SlicedBarcodeSpec.summary_fields must be nonempty for featurizer=:summary.")
        allowed = Set((:count, :sum_persistence, :mean_persistence, :max_persistence, :entropy))
        all(f -> Symbol(f) in allowed, spec.summary_fields) ||
            _push_issue!(issues, "SlicedBarcodeSpec.summary_fields must be chosen from (:count, :sum_persistence, :mean_persistence, :max_persistence, :entropy).")
    end
    return issues
end

function _check_spec_issues(spec::BarcodeSummarySpec)
    issues = String[]
    _check_common_slice_spec!(issues, spec; label="BarcodeSummarySpec")
    spec.aggregate in (:mean, :sum, :stack) ||
        _push_issue!(issues, "BarcodeSummarySpec.aggregate must be :mean, :sum, or :stack.")
    spec.source in (:slice, :fibered) ||
        _push_issue!(issues, "BarcodeSummarySpec.source must be :slice or :fibered.")
    spec.infinite_policy in (:clip_to_window, :error) ||
        _push_issue!(issues, "BarcodeSummarySpec.infinite_policy must be :clip_to_window or :error.")
    isempty(spec.fields) && _push_issue!(issues, "BarcodeSummarySpec.fields must be nonempty.")
    allowed = Set((:count, :sum_persistence, :mean_persistence, :max_persistence, :entropy))
    all(f -> Symbol(f) in allowed, spec.fields) ||
        _push_issue!(issues, "BarcodeSummarySpec.fields must be chosen from (:count, :sum_persistence, :mean_persistence, :max_persistence, :entropy).")
    if spec.window !== nothing
        all(isfinite, spec.window) || _push_issue!(issues, "BarcodeSummarySpec.window must contain finite bounds.")
        spec.window[1] <= spec.window[2] ||
            _push_issue!(issues, "BarcodeSummarySpec.window must satisfy lo <= hi.")
    end
    return issues
end

function _check_spec_issues(spec::SignedBarcodeImageSpec)
    issues = String[]
    _check_feature_grid!(issues, spec.xs, "SignedBarcodeImageSpec.xs")
    _check_feature_grid!(issues, spec.ys, "SignedBarcodeImageSpec.ys")
    spec.sigma > 0 || _push_issue!(issues, "SignedBarcodeImageSpec.sigma must be > 0.")
    spec.mode in (:center, :lo, :hi) ||
        _push_issue!(issues, "SignedBarcodeImageSpec.mode must be :center, :lo, or :hi.")
    spec.method in (:auto, :bulk, :local) ||
        _push_issue!(issues, "SignedBarcodeImageSpec.method must be :auto, :bulk, or :local.")
    _check_axes_policy!(issues, spec.axes_policy, "SignedBarcodeImageSpec")
    spec.max_axis_len > 0 || _push_issue!(issues, "SignedBarcodeImageSpec.max_axis_len must be > 0.")
    _check_axes_payload!(issues, spec.axes, "SignedBarcodeImageSpec.axes")
    spec.axes === nothing || length(spec.axes) == 2 ||
        _push_issue!(issues, "SignedBarcodeImageSpec.axes must have length 2 when provided.")
    return issues
end

function _check_spec_issues(spec::PointSignedMeasureSpec)
    issues = String[]
    spec.ndims > 0 || _push_issue!(issues, "PointSignedMeasureSpec.ndims must be > 0.")
    spec.k >= 0 || _push_issue!(issues, "PointSignedMeasureSpec.k must be >= 0.")
    spec.coords in (:values, :indices) ||
        _push_issue!(issues, "PointSignedMeasureSpec.coords must be :values or :indices.")
    return issues
end

function _check_spec_issues(spec::EulerSignedMeasureSpec)
    issues = String[]
    spec.ndims > 0 || _push_issue!(issues, "EulerSignedMeasureSpec.ndims must be > 0.")
    spec.k >= 0 || _push_issue!(issues, "EulerSignedMeasureSpec.k must be >= 0.")
    spec.coords in (:values, :indices) ||
        _push_issue!(issues, "EulerSignedMeasureSpec.coords must be :values or :indices.")
    _check_axes_policy!(issues, spec.axes_policy, "EulerSignedMeasureSpec")
    spec.max_axis_len > 0 || _push_issue!(issues, "EulerSignedMeasureSpec.max_axis_len must be > 0.")
    _check_axes_payload!(issues, spec.axes, "EulerSignedMeasureSpec.axes")
    spec.axes === nothing || length(spec.axes) == spec.ndims ||
        _push_issue!(issues, "EulerSignedMeasureSpec.axes must match EulerSignedMeasureSpec.ndims when provided.")
    spec.max_terms >= 0 || _push_issue!(issues, "EulerSignedMeasureSpec.max_terms must be >= 0.")
    spec.min_abs_weight >= 0 || _push_issue!(issues, "EulerSignedMeasureSpec.min_abs_weight must be >= 0.")
    return issues
end

function _check_spec_issues(spec::RectangleSignedBarcodeTopKSpec)
    issues = String[]
    spec.ndims > 0 || _push_issue!(issues, "RectangleSignedBarcodeTopKSpec.ndims must be > 0.")
    spec.k >= 0 || _push_issue!(issues, "RectangleSignedBarcodeTopKSpec.k must be >= 0.")
    spec.coords in (:values, :indices) ||
        _push_issue!(issues, "RectangleSignedBarcodeTopKSpec.coords must be :values or :indices.")
    _check_axes_policy!(issues, spec.axes_policy, "RectangleSignedBarcodeTopKSpec")
    spec.max_axis_len > 0 || _push_issue!(issues, "RectangleSignedBarcodeTopKSpec.max_axis_len must be > 0.")
    _check_axes_payload!(issues, spec.axes, "RectangleSignedBarcodeTopKSpec.axes")
    spec.axes === nothing || length(spec.axes) == spec.ndims ||
        _push_issue!(issues, "RectangleSignedBarcodeTopKSpec.axes must match RectangleSignedBarcodeTopKSpec.ndims when provided.")
    spec.tol >= 0 || _push_issue!(issues, "RectangleSignedBarcodeTopKSpec.tol must be >= 0.")
    if spec.max_span isa Integer
        spec.max_span >= 0 ||
            _push_issue!(issues, "RectangleSignedBarcodeTopKSpec.max_span must be >= 0 when given as an integer.")
    elseif spec.max_span isa Tuple
        length(spec.max_span) == spec.ndims ||
            _push_issue!(issues, "RectangleSignedBarcodeTopKSpec.max_span tuple must match RectangleSignedBarcodeTopKSpec.ndims.")
        all(x -> x >= 0, spec.max_span) ||
            _push_issue!(issues, "RectangleSignedBarcodeTopKSpec.max_span tuple entries must be >= 0.")
    end
    spec.method in (:auto, :bulk, :local) ||
        _push_issue!(issues, "RectangleSignedBarcodeTopKSpec.method must be :auto, :bulk, or :local.")
    spec.bulk_max_elems > 0 ||
        _push_issue!(issues, "RectangleSignedBarcodeTopKSpec.bulk_max_elems must be > 0.")
    return issues
end

function _check_spec_issues(spec::ProjectedDistancesSpec)
    issues = String[]
    _check_reference_names!(issues, spec.references, spec.reference_names, "ProjectedDistancesSpec")
    spec.dist in (:bottleneck, :wasserstein) ||
        _push_issue!(issues, "ProjectedDistancesSpec.dist must be :bottleneck or :wasserstein.")
    spec.agg in (:mean, :sum, :maximum) ||
        _push_issue!(issues, "ProjectedDistancesSpec.agg must be :mean, :sum, or :maximum.")
    spec.p > 0 || _push_issue!(issues, "ProjectedDistancesSpec.p must be > 0.")
    spec.q > 0 || _push_issue!(issues, "ProjectedDistancesSpec.q must be > 0.")
    spec.n_dirs > 0 || _push_issue!(issues, "ProjectedDistancesSpec.n_dirs must be > 0.")
    spec.normalize in (:none, :L1, :Linf) ||
        _push_issue!(issues, "ProjectedDistancesSpec.normalize must be :none, :L1, or :Linf.")
    spec.enforce_monotone in (:upper, :none) ||
        _push_issue!(issues, "ProjectedDistancesSpec.enforce_monotone must be :upper or :none.")
    if spec.directions !== nothing
        _check_directions_offsets!(issues, spec.directions, spec.directions, nothing, "ProjectedDistancesSpec")
    end
    return issues
end

function _check_spec_issues(spec::MatchingDistanceBankSpec)
    issues = String[]
    _check_reference_names!(issues, spec.references, spec.reference_names, "MatchingDistanceBankSpec")
    spec.method in (:auto, :approx, :exact_2d) ||
        _push_issue!(issues, "MatchingDistanceBankSpec.method must be :auto, :approx, or :exact_2d.")
    spec.n_dirs > 0 || _push_issue!(issues, "MatchingDistanceBankSpec.n_dirs must be > 0.")
    spec.n_offsets > 0 || _push_issue!(issues, "MatchingDistanceBankSpec.n_offsets must be > 0.")
    spec.max_den > 0 || _push_issue!(issues, "MatchingDistanceBankSpec.max_den must be > 0.")
    spec.normalize_dirs in (:none, :L1, :Linf) ||
        _push_issue!(issues, "MatchingDistanceBankSpec.normalize_dirs must be :none, :L1, or :Linf.")
    spec.weight in (:none, :lesnick_l1, :lesnick_linf) ||
        _push_issue!(issues, "MatchingDistanceBankSpec.weight must be :none, :lesnick_l1, or :lesnick_linf.")
    if spec.directions !== nothing && spec.offsets !== nothing
        _check_directions_offsets!(issues, spec.directions, spec.offsets, nothing, "MatchingDistanceBankSpec")
    elseif spec.directions !== nothing
        _check_directions_offsets!(issues, spec.directions, spec.directions, nothing, "MatchingDistanceBankSpec")
    elseif spec.offsets !== nothing
        _check_directions_offsets!(issues, spec.offsets, spec.offsets, nothing, "MatchingDistanceBankSpec")
    end
    return issues
end

function _check_spec_issues(spec::MPPImageSpec)
    issues = String[]
    spec.resolution > 1 || _push_issue!(issues, "MPPImageSpec.resolution must be >= 2.")
    spec.xgrid === nothing || _check_feature_grid!(issues, spec.xgrid, "MPPImageSpec.xgrid")
    spec.ygrid === nothing || _check_feature_grid!(issues, spec.ygrid, "MPPImageSpec.ygrid")
    spec.sigma > 0 || _push_issue!(issues, "MPPImageSpec.sigma must be > 0.")
    spec.N > 0 || _push_issue!(issues, "MPPImageSpec.N must be > 0.")
    if spec.delta isa Symbol
        spec.delta == :auto || _push_issue!(issues, "MPPImageSpec.delta must be :auto or a positive real.")
    else
        spec.delta > 0 || _push_issue!(issues, "MPPImageSpec.delta must be > 0 when given numerically.")
    end
    spec.q > 0 || _push_issue!(issues, "MPPImageSpec.q must be > 0.")
    spec.tie_break in (:center, :up, :down) ||
        _push_issue!(issues, "MPPImageSpec.tie_break must be :center, :up, or :down.")
    spec.cutoff_radius === nothing || spec.cutoff_radius >= 0 ||
        _push_issue!(issues, "MPPImageSpec.cutoff_radius must be >= 0 when provided.")
    spec.cutoff_tol === nothing || (0 <= spec.cutoff_tol < 1) ||
        _push_issue!(issues, "MPPImageSpec.cutoff_tol must lie in [0, 1) when provided.")
    return issues
end

function _check_spec_issues(spec::MPPDecompositionHistogramSpec)
    issues = String[]
    spec.orientation_bins > 0 ||
        _push_issue!(issues, "MPPDecompositionHistogramSpec.orientation_bins must be > 0.")
    spec.scale_bins > 0 ||
        _push_issue!(issues, "MPPDecompositionHistogramSpec.scale_bins must be > 0.")
    if spec.scale_range !== nothing
        all(isfinite, spec.scale_range) ||
            _push_issue!(issues, "MPPDecompositionHistogramSpec.scale_range must contain finite bounds.")
        spec.scale_range[1] <= spec.scale_range[2] ||
            _push_issue!(issues, "MPPDecompositionHistogramSpec.scale_range must satisfy lo <= hi.")
    end
    spec.weight in (:count, :mass) ||
        _push_issue!(issues, "MPPDecompositionHistogramSpec.weight must be :count or :mass.")
    spec.normalize in (:none, :l1, :l2, :max) ||
        _push_issue!(issues, "MPPDecompositionHistogramSpec.normalize must be :none, :l1, :l2, or :max.")
    spec.N > 0 || _push_issue!(issues, "MPPDecompositionHistogramSpec.N must be > 0.")
    if spec.delta isa Symbol
        spec.delta == :auto || _push_issue!(issues, "MPPDecompositionHistogramSpec.delta must be :auto or a positive real.")
    else
        spec.delta > 0 || _push_issue!(issues, "MPPDecompositionHistogramSpec.delta must be > 0 when given numerically.")
    end
    spec.q > 0 || _push_issue!(issues, "MPPDecompositionHistogramSpec.q must be > 0.")
    spec.tie_break in (:center, :up, :down) ||
        _push_issue!(issues, "MPPDecompositionHistogramSpec.tie_break must be :center, :up, or :down.")
    return issues
end

function _check_spec_issues(spec::BettiTableSpec)
    issues = String[]
    spec.nvertices >= 0 || _push_issue!(issues, "BettiTableSpec.nvertices must be >= 0.")
    spec.pad_to >= 0 || _push_issue!(issues, "BettiTableSpec.pad_to must be >= 0.")
    spec.resolution_maxlen >= 0 || _push_issue!(issues, "BettiTableSpec.resolution_maxlen must be >= 0.")
    return issues
end

function _check_spec_issues(spec::BassTableSpec)
    issues = String[]
    spec.nvertices >= 0 || _push_issue!(issues, "BassTableSpec.nvertices must be >= 0.")
    spec.pad_to >= 0 || _push_issue!(issues, "BassTableSpec.pad_to must be >= 0.")
    spec.resolution_maxlen >= 0 || _push_issue!(issues, "BassTableSpec.resolution_maxlen must be >= 0.")
    return issues
end

function _check_spec_issues(spec::BettiSupportMeasuresSpec)
    issues = String[]
    spec.pad_to >= 0 || _push_issue!(issues, "BettiSupportMeasuresSpec.pad_to must be >= 0.")
    spec.resolution_maxlen >= 0 || _push_issue!(issues, "BettiSupportMeasuresSpec.resolution_maxlen must be >= 0.")
    return issues
end

function _check_spec_issues(spec::BassSupportMeasuresSpec)
    issues = String[]
    spec.pad_to >= 0 || _push_issue!(issues, "BassSupportMeasuresSpec.pad_to must be >= 0.")
    spec.resolution_maxlen >= 0 || _push_issue!(issues, "BassSupportMeasuresSpec.resolution_maxlen must be >= 0.")
    return issues
end

function _check_spec_issues(spec::CompositeSpec)
    issues = String[]
    isempty(spec.specs) && _push_issue!(issues, "CompositeSpec.specs must be nonempty.")
    for (i, sub) in pairs(spec.specs)
        sub_summary = check_featurizer_spec(sub; throw=false)
        sub_summary.valid || append!(issues, ["CompositeSpec.specs[$i]: " * msg for msg in sub_summary.issues])
    end
    return issues
end

_check_spec_issues(::AbstractFeaturizerSpec) = String[]

function check_featurizer_spec(spec::AbstractFeaturizerSpec; throw::Bool=false)
    issues = _check_spec_issues(spec)
    nf = try
        nfeatures(spec)
    catch
        0
    end
    valid = isempty(issues)
    summary = FeaturizerSpecValidationSummary(
        :featurizer_spec_validation,
        valid,
        Symbol(nameof(typeof(spec))),
        nf,
        issues,
    )
    throw && !valid && _throw_invalid_featurizer(string(nameof(typeof(spec))), issues)
    return summary
end

function check_feature_set(fs::FeatureSet; throw::Bool=false)
    issues = String[]
    size(fs.X, 1) == length(fs.ids) ||
        _push_issue!(issues, "FeatureSet.ids length must match the number of samples.")
    size(fs.X, 2) == length(fs.names) ||
        _push_issue!(issues, "FeatureSet.names length must match the number of feature columns.")
    length(unique(fs.names)) == length(fs.names) ||
        _push_issue!(issues, "FeatureSet.names must be unique.")
    all(!isempty, fs.ids) || _push_issue!(issues, "FeatureSet.ids must be nonempty strings.")
    labels = sample_labels(fs)
    if labels !== nothing
        length(labels) == nsamples(fs) ||
            _push_issue!(issues, "FeatureSet meta labels must match the number of samples.")
    end
    valid = isempty(issues)
    summary = FeatureSetValidationSummary(
        :feature_set_validation,
        valid,
        nsamples(fs),
        nfeatures(fs),
        issues,
    )
    throw && !valid && _throw_invalid_featurizer("FeatureSet", issues)
    return summary
end

function check_experiment_spec(exp::ExperimentSpec; throw::Bool=false)
    issues = String[]
    isempty(exp.name) && _push_issue!(issues, "ExperimentSpec.name must be nonempty.")
    isempty(exp.featurizers) && _push_issue!(issues, "ExperimentSpec.featurizers must be nonempty.")
    exp.on_unsupported in (:error, :skip, :missing) ||
        _push_issue!(issues, "ExperimentSpec.on_unsupported must be :error, :skip, or :missing.")
    exp.io.format in (:wide, :long) ||
        _push_issue!(issues, "ExperimentIOConfig.format must be :wide or :long.")
    all(f -> f in (:arrow, :parquet, :npz, :csv_wide, :csv_long), exp.io.formats) ||
        _push_issue!(issues, "ExperimentIOConfig.formats contains an unsupported format.")
    length(unique(exp.io.formats)) == length(exp.io.formats) ||
        _push_issue!(issues, "ExperimentIOConfig.formats must be unique.")

    invalid_specs = Symbol[]
    for spec in exp.featurizers
        sub = check_featurizer_spec(spec; throw=false)
        if !sub.valid
            push!(invalid_specs, sub.spec_type)
            append!(issues, ["$(sub.spec_type): " * msg for msg in sub.issues])
        end
    end

    valid = isempty(issues)
    summary = ExperimentSpecValidationSummary(
        :experiment_spec_validation,
        valid,
        exp.name,
        length(exp.featurizers),
        invalid_specs,
        issues,
    )
    throw && !valid && _throw_invalid_featurizer("ExperimentSpec", issues)
    return summary
end

function check_loaded_experiment(res::LoadedExperimentResult; throw::Bool=false)
    issues = String[]
    kind = String(get(res.manifest, "kind", ""))
    kind == "experiment_manifest" ||
        _push_issue!(issues, "LoadedExperimentResult.manifest must have kind=\"experiment_manifest\".")
    keys = artifact_keys(res)
    length(unique(keys)) == length(keys) ||
        _push_issue!(issues, "LoadedExperimentResult artifact keys must be unique.")

    with_features = 0
    for art in artifacts(res)
        if art.spec !== nothing
            sub = check_featurizer_spec(art.spec; throw=false)
            sub.valid || append!(issues, ["artifact $(art.key) spec: " * msg for msg in sub.issues])
        end
        if art.features !== nothing
            with_features += 1
            sub = check_feature_set(art.features; throw=false)
            sub.valid || append!(issues, ["artifact $(art.key) features: " * msg for msg in sub.issues])
        end
        if art.metadata !== nothing && _obj_haskey(art.metadata, "kind")
            md_kind = String(art.metadata["kind"])
            (md_kind == "features" || md_kind == "experiment_feature") ||
                _push_issue!(issues, "artifact $(art.key) metadata kind must be \"features\" or \"experiment_feature\".")
        end
    end

    valid = isempty(issues)
    summary = LoadedExperimentValidationSummary(
        :loaded_experiment_validation,
        valid,
        nartifacts(res),
        with_features,
        issues,
    )
    throw && !valid && _throw_invalid_featurizer("LoadedExperimentResult", issues)
    return summary
end

function describe(fs::FeatureSet)
    return (
        kind = :feature_set,
        nsamples = nsamples(fs),
        nfeatures = nfeatures(fs),
        feature_names = feature_names(fs),
        feature_axes = feature_axes(fs),
        sample_ids = sample_ids(fs),
        has_labels = sample_labels(fs) !== nothing,
        metadata_keys = _meta_keys(metadata(fs)),
        eltype = eltype(fs.X),
    )
end

@inline feature_set_summary(fs::FeatureSet) = describe(fs)

function describe(cfg::ExperimentIOConfig)
    return (
        kind = :experiment_io_config,
        outdir = cfg.outdir,
        prefix = cfg.prefix,
        format = cfg.format,
        formats = cfg.formats,
        write_metadata = cfg.write_metadata,
        overwrite = cfg.overwrite,
    )
end

function describe(exp::ExperimentSpec)
    return (
        kind = :experiment_spec,
        name = exp.name,
        nfeaturizers = length(exp.featurizers),
        featurizer_types = [Symbol(nameof(typeof(s))) for s in exp.featurizers],
        on_unsupported = exp.on_unsupported,
        cache = exp.cache,
        batch = exp.batch,
        io = describe(exp.io),
        metadata_keys = _meta_keys(exp.metadata),
    )
end

@inline experiment_summary(x::Union{ExperimentIOConfig,ExperimentSpec,ExperimentResult,LoadedExperimentResult}) = describe(x)

function describe(a::ExperimentArtifact)
    return (
        kind = :experiment_artifact,
        key = a.key,
        spec_type = Symbol(nameof(typeof(a.spec))),
        nsamples = nsamples(a.features),
        nfeatures = nfeatures(a.features),
        elapsed_seconds = a.elapsed_seconds,
        feature_paths = a.feature_paths,
        metadata_path = a.metadata_path,
        has_features = true,
    )
end

function describe(a::LoadedExperimentArtifact)
    return (
        kind = :loaded_experiment_artifact,
        key = a.key,
        spec_type = a.spec === nothing ? nothing : Symbol(nameof(typeof(a.spec))),
        has_features = has_features(a),
        nfeatures = a.features === nothing ? nothing : nfeatures(a.features),
        nsamples = a.features === nothing ? nothing : nsamples(a.features),
        elapsed_seconds = a.elapsed_seconds,
        feature_paths = a.feature_paths,
        metadata_path = a.metadata_path,
    )
end

@inline experiment_artifact_summary(x::Union{ExperimentArtifact,LoadedExperimentArtifact}) = describe(x)

function describe(res::ExperimentResult)
    return (
        kind = :experiment_result,
        experiment_name = experiment_name(res),
        nartifacts = nartifacts(res),
        artifact_keys = artifact_keys(res),
        total_elapsed_seconds = total_elapsed_seconds(res),
        run_dir = run_dir(res),
        manifest_path = manifest_path(res),
        has_features = has_features(res),
    )
end

function describe(res::LoadedExperimentResult)
    return (
        kind = :loaded_experiment_result,
        experiment_name = experiment_name(res),
        nartifacts = nartifacts(res),
        artifact_keys = artifact_keys(res),
        total_elapsed_seconds = total_elapsed_seconds(res),
        run_dir = run_dir(res),
        manifest_path = manifest_path(res),
        has_features = has_features(res),
    )
end

function describe(spec::AbstractFeaturizerSpec)
    return (
        kind = :featurizer_spec,
        spec_type = Symbol(nameof(typeof(spec))),
        nfeatures = nfeatures(spec),
        feature_axes = feature_axes(spec),
        fields = _spec_field_namedtuple(spec),
    )
end

@inline featurizer_summary(spec::AbstractFeaturizerSpec) = describe(spec)

function Base.show(io::IO, fs::FeatureSet)
    print(io, "FeatureSet(nsamples=", nsamples(fs),
          ", nfeatures=", nfeatures(fs), ")")
end

function Base.show(io::IO, ::MIME"text/plain", fs::FeatureSet)
    d = describe(fs)
    print(io, "FeatureSet",
          "\n  nsamples: ", d.nsamples,
          "\n  nfeatures: ", d.nfeatures,
          "\n  feature_names: ", repr(d.feature_names),
          "\n  sample_ids: ", repr(d.sample_ids),
          "\n  has_labels: ", d.has_labels,
          "\n  metadata_keys: ", repr(d.metadata_keys))
end

function Base.show(io::IO, cfg::ExperimentIOConfig)
    print(io, "ExperimentIOConfig(format=", repr(cfg.format),
          ", formats=", repr(cfg.formats), ")")
end

function Base.show(io::IO, ::MIME"text/plain", cfg::ExperimentIOConfig)
    d = describe(cfg)
    print(io, "ExperimentIOConfig",
          "\n  outdir: ", repr(d.outdir),
          "\n  prefix: ", repr(d.prefix),
          "\n  format: ", repr(d.format),
          "\n  formats: ", repr(d.formats),
          "\n  write_metadata: ", d.write_metadata,
          "\n  overwrite: ", d.overwrite)
end

function Base.show(io::IO, exp::ExperimentSpec)
    print(io, "ExperimentSpec(name=", repr(exp.name),
          ", nfeaturizers=", length(exp.featurizers), ")")
end

function Base.show(io::IO, ::MIME"text/plain", exp::ExperimentSpec)
    d = describe(exp)
    print(io, "ExperimentSpec",
          "\n  name: ", repr(d.name),
          "\n  nfeaturizers: ", d.nfeaturizers,
          "\n  featurizer_types: ", repr(d.featurizer_types),
          "\n  on_unsupported: ", repr(d.on_unsupported),
          "\n  cache: ", repr(d.cache))
end

function Base.show(io::IO, a::ExperimentArtifact)
    print(io, "ExperimentArtifact(key=", repr(a.key),
          ", nfeatures=", nfeatures(a.features),
          ", elapsed_seconds=", round(a.elapsed_seconds; digits=6), ")")
end

function Base.show(io::IO, ::MIME"text/plain", a::ExperimentArtifact)
    d = describe(a)
    print(io, "ExperimentArtifact",
          "\n  key: ", repr(d.key),
          "\n  spec_type: ", repr(d.spec_type),
          "\n  nsamples: ", d.nsamples,
          "\n  nfeatures: ", d.nfeatures,
          "\n  elapsed_seconds: ", d.elapsed_seconds,
          "\n  feature_paths: ", repr(d.feature_paths),
          "\n  metadata_path: ", repr(d.metadata_path))
end

function Base.show(io::IO, a::LoadedExperimentArtifact)
    print(io, "LoadedExperimentArtifact(key=", repr(a.key),
          ", has_features=", has_features(a),
          ", elapsed_seconds=", round(a.elapsed_seconds; digits=6), ")")
end

function Base.show(io::IO, ::MIME"text/plain", a::LoadedExperimentArtifact)
    d = describe(a)
    print(io, "LoadedExperimentArtifact",
          "\n  key: ", repr(d.key),
          "\n  spec_type: ", repr(d.spec_type),
          "\n  has_features: ", d.has_features,
          "\n  nsamples: ", repr(d.nsamples),
          "\n  nfeatures: ", repr(d.nfeatures),
          "\n  elapsed_seconds: ", d.elapsed_seconds,
          "\n  feature_paths: ", repr(d.feature_paths),
          "\n  metadata_path: ", repr(d.metadata_path))
end

function Base.show(io::IO, res::ExperimentResult)
    print(io, "ExperimentResult(name=", repr(experiment_name(res)),
          ", nartifacts=", nartifacts(res),
          ", total_elapsed_seconds=", round(total_elapsed_seconds(res); digits=6), ")")
end

function Base.show(io::IO, ::MIME"text/plain", res::ExperimentResult)
    d = describe(res)
    print(io, "ExperimentResult",
          "\n  experiment_name: ", repr(d.experiment_name),
          "\n  nartifacts: ", d.nartifacts,
          "\n  artifact_keys: ", repr(d.artifact_keys),
          "\n  total_elapsed_seconds: ", d.total_elapsed_seconds,
          "\n  run_dir: ", repr(d.run_dir),
          "\n  manifest_path: ", repr(d.manifest_path))
end

function Base.show(io::IO, res::LoadedExperimentResult)
    print(io, "LoadedExperimentResult(name=", repr(experiment_name(res)),
          ", nartifacts=", nartifacts(res),
          ", has_features=", has_features(res), ")")
end

function Base.show(io::IO, ::MIME"text/plain", res::LoadedExperimentResult)
    d = describe(res)
    print(io, "LoadedExperimentResult",
          "\n  experiment_name: ", repr(d.experiment_name),
          "\n  nartifacts: ", d.nartifacts,
          "\n  artifact_keys: ", repr(d.artifact_keys),
          "\n  total_elapsed_seconds: ", d.total_elapsed_seconds,
          "\n  run_dir: ", repr(d.run_dir),
          "\n  manifest_path: ", repr(d.manifest_path),
          "\n  has_features: ", d.has_features)
end

function Base.show(io::IO, spec::AbstractFeaturizerSpec)
    print(io, string(nameof(typeof(spec))), "(nfeatures=", nfeatures(spec), ")")
end

function Base.show(io::IO, ::MIME"text/plain", spec::AbstractFeaturizerSpec)
    d = describe(spec)
    print(io, string(nameof(typeof(spec))),
          "\n  nfeatures: ", d.nfeatures,
          "\n  feature_axes: ", repr(d.feature_axes))
end

function Base.show(io::IO, summary::FeaturizerSpecValidationSummary)
    print(io, "FeaturizerSpecValidationSummary(spec_type=", repr(summary.spec_type),
          ", valid=", summary.valid,
          ", issues=", length(summary.issues), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::FeaturizerSpecValidationSummary)
    print(io, "FeaturizerSpecValidationSummary",
          "\n  kind: ", repr(summary.kind),
          "\n  spec_type: ", repr(summary.spec_type),
          "\n  valid: ", summary.valid,
          "\n  nfeatures: ", summary.nfeatures)
    isempty(summary.issues) || print(io, "\n  issues: ", join(summary.issues, "\n          "))
end

function Base.show(io::IO, summary::FeatureSetValidationSummary)
    print(io, "FeatureSetValidationSummary(valid=", summary.valid,
          ", issues=", length(summary.issues), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::FeatureSetValidationSummary)
    print(io, "FeatureSetValidationSummary",
          "\n  kind: ", repr(summary.kind),
          "\n  valid: ", summary.valid,
          "\n  nsamples: ", summary.nsamples,
          "\n  nfeatures: ", summary.nfeatures)
    isempty(summary.issues) || print(io, "\n  issues: ", join(summary.issues, "\n          "))
end

function Base.show(io::IO, summary::ExperimentSpecValidationSummary)
    print(io, "ExperimentSpecValidationSummary(valid=", summary.valid,
          ", issues=", length(summary.issues), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::ExperimentSpecValidationSummary)
    print(io, "ExperimentSpecValidationSummary",
          "\n  kind: ", repr(summary.kind),
          "\n  valid: ", summary.valid,
          "\n  experiment_name: ", repr(summary.experiment_name),
          "\n  nfeaturizers: ", summary.nfeaturizers,
          "\n  invalid_specs: ", repr(summary.invalid_specs))
    isempty(summary.issues) || print(io, "\n  issues: ", join(summary.issues, "\n          "))
end

function Base.show(io::IO, summary::LoadedExperimentValidationSummary)
    print(io, "LoadedExperimentValidationSummary(valid=", summary.valid,
          ", issues=", length(summary.issues), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::LoadedExperimentValidationSummary)
    print(io, "LoadedExperimentValidationSummary",
          "\n  kind: ", repr(summary.kind),
          "\n  valid: ", summary.valid,
          "\n  nartifacts: ", summary.nartifacts,
          "\n  artifacts_with_features: ", summary.artifacts_with_features)
    isempty(summary.issues) || print(io, "\n  issues: ", join(summary.issues, "\n          "))
end

@inline describe(summary::FeaturizerSpecValidationSummary) =
    (; kind=summary.kind, valid=summary.valid, spec_type=summary.spec_type,
       nfeatures=summary.nfeatures, issues=summary.issues)
@inline describe(summary::FeatureSetValidationSummary) =
    (; kind=summary.kind, valid=summary.valid, nsamples=summary.nsamples,
       nfeatures=summary.nfeatures, issues=summary.issues)
@inline describe(summary::ExperimentSpecValidationSummary) =
    (; kind=summary.kind, valid=summary.valid, experiment_name=summary.experiment_name,
       nfeaturizers=summary.nfeaturizers, invalid_specs=summary.invalid_specs, issues=summary.issues)
@inline describe(summary::LoadedExperimentValidationSummary) =
    (; kind=summary.kind, valid=summary.valid, nartifacts=summary.nartifacts,
       artifacts_with_features=summary.artifacts_with_features, issues=summary.issues)

@inline _jsonable(x::Nothing) = nothing
@inline _jsonable(x::Bool) = x
@inline _jsonable(x::Integer) = x
@inline _jsonable(x::AbstractFloat) = x
@inline _jsonable(x::AbstractString) = String(x)
@inline _jsonable(x::Symbol) = String(x)
@inline _jsonable(x::VersionNumber) = string(x)
@inline _jsonable(x::DataType) = string(x)
@inline _jsonable(x::AbstractCoeffField) = string(x)

function _jsonable(x::NamedTuple)
    out = Dict{String,Any}()
    for (k, v) in pairs(x)
        out[String(k)] = _jsonable(v)
    end
    return out
end

function _jsonable(x::AbstractVector)
    return [_jsonable(v) for v in x]
end

function _jsonable(x::Tuple)
    return [_jsonable(v) for v in x]
end

function _jsonable(x::AbstractDict)
    out = Dict{String,Any}()
    for (k, v) in pairs(x)
        out[String(k)] = _jsonable(v)
    end
    return out
end

function _jsonable(x)
    T = typeof(x)
    if isstructtype(T)
        fields = Dict{String,Any}()
        for f in fieldnames(T)
            fields[String(f)] = _jsonable(getfield(x, f))
        end
        return Dict(
            "type" => string(nameof(T)),
            "module" => string(parentmodule(T)),
            "fields" => fields,
        )
    end
    return string(x)
end

@inline _is_simple_json_scalar(x) =
    x === nothing || x isa Bool || x isa Integer || x isa AbstractFloat || x isa AbstractString || x isa Symbol

function _reference_stub_json(x)
    if _is_simple_json_scalar(x)
        return _jsonable(x)
    end
    T = typeof(x)
    return Dict(
        "kind" => "reference_stub",
        "type" => string(nameof(T)),
        "module" => string(parentmodule(T)),
        "objectid" => string(UInt(objectid(x))),
    )
end

function _spec_jsonable(spec::ProjectedDistancesSpec)
    ref_ids = String[string(nm) for nm in spec.reference_names]
    return Dict(
        "type" => "ProjectedDistancesSpec",
        "module" => string(parentmodule(typeof(spec))),
        "fields" => Dict(
            "reference_ids" => ref_ids,
            "references" => [_reference_stub_json(r) for r in spec.references],
            "reference_names" => [String(nm) for nm in spec.reference_names],
            "dist" => String(spec.dist),
            "p" => spec.p,
            "q" => spec.q,
            "agg" => String(spec.agg),
            "directions" => spec.directions === nothing ? nothing : _jsonable(spec.directions),
            "n_dirs" => spec.n_dirs,
            "normalize" => String(spec.normalize),
            "enforce_monotone" => String(spec.enforce_monotone),
            "precompute" => spec.precompute,
            "threads" => spec.threads,
        ),
    )
end

function _spec_jsonable(spec::MatchingDistanceBankSpec)
    ref_ids = String[string(nm) for nm in spec.reference_names]
    return Dict(
        "type" => "MatchingDistanceBankSpec",
        "module" => string(parentmodule(typeof(spec))),
        "fields" => Dict(
            "reference_ids" => ref_ids,
            "references" => [_reference_stub_json(r) for r in spec.references],
            "reference_names" => [String(nm) for nm in spec.reference_names],
            "method" => String(spec.method),
            "directions" => spec.directions === nothing ? nothing : _jsonable(spec.directions),
            "offsets" => spec.offsets === nothing ? nothing : _jsonable(spec.offsets),
            "n_dirs" => spec.n_dirs,
            "n_offsets" => spec.n_offsets,
            "max_den" => spec.max_den,
            "include_axes" => spec.include_axes,
            "normalize_dirs" => String(spec.normalize_dirs),
            "weight" => String(spec.weight),
            "threads" => spec.threads,
        ),
    )
end

function _spec_jsonable(spec::CompositeSpec)
    return Dict(
        "type" => "CompositeSpec",
        "module" => string(parentmodule(typeof(spec))),
        "fields" => Dict(
            "specs" => [_spec_jsonable(s) for s in spec.specs],
            "namespacing" => spec.namespacing,
        ),
    )
end

_spec_jsonable(spec::AbstractFeaturizerSpec) = _jsonable(spec)

function _pkg_version_or_unknown()
    try
        v = Base.pkgversion(parentmodule(@__MODULE__))
        return v === nothing ? "unknown" : string(v)
    catch
        return "unknown"
    end
end

function _git_commit_or_unknown()
    root = normpath(joinpath(@__DIR__, ".."))
    try
        return readchomp(`git -C $root rev-parse --short=12 HEAD`)
    catch
        return "unknown"
    end
end

@inline default_feature_metadata_path(path::AbstractString) = String(path) * ".meta.json"

function feature_metadata(fs::FeatureSet;
                          format::Symbol=:wide,
                          git_commit::Union{Nothing,AbstractString}=nothing)
    spec = _meta_get(fs.meta, :spec, nothing)
    opts = _meta_get(fs.meta, :opts, nothing)
    labels = _meta_get(fs.meta, :labels, nothing)
    unsupported_policy = _meta_get(fs.meta, :unsupported_policy, nothing)
    skipped_indices = _meta_get(fs.meta, :skipped_indices, Int[])
    cache_mode = _meta_get(fs.meta, :cache_mode, nothing)
    threaded = _meta_get(fs.meta, :threaded, nothing)
    gc = git_commit === nothing ? _git_commit_or_unknown() : String(git_commit)
    out = feature_schema_header(format=format)
    out["layout"] = "rows=samples, cols=features"
    out["n_samples"] = nsamples(fs)
    out["n_features"] = nfeatures(fs)
    out["numeric_type"] = string(eltype(fs.X))
    out["feature_names"] = [String(nm) for nm in fs.names]
    spec_axes = spec isa AbstractFeaturizerSpec ? feature_axes(spec) : _meta_get(fs.meta, :feature_axes, nothing)
    out["feature_axes"] = _jsonable(spec_axes)
    out["ids"] = copy(fs.ids)
    out["spec"] = spec isa AbstractFeaturizerSpec ? _spec_jsonable(spec) : _jsonable(spec)
    out["opts"] = _jsonable(opts)
    out["labels"] = _jsonable(labels)
    out["unsupported_policy"] = _jsonable(unsupported_policy)
    out["skipped_indices"] = _jsonable(skipped_indices)
    out["cache_mode"] = _jsonable(cache_mode)
    out["threaded"] = _jsonable(threaded)
    out["package_version"] = _pkg_version_or_unknown()
    out["git_commit"] = gc
    return out
end

function save_metadata_json(path::AbstractString, meta)
    open(path, "w") do io
        JSON3.write(io, _jsonable(meta); allow_inf=true, indent=2)
    end
    return path
end

@inline _obj_haskey(obj::AbstractDict, key::AbstractString) = haskey(obj, key)
@inline _obj_haskey(obj, key::AbstractString) = try
    haskey(obj, key)
catch
    false
end

@inline _obj_get(obj, key::AbstractString, default=nothing) =
    _obj_haskey(obj, key) ? obj[key] : default

@inline _as_symbol(x) = Symbol(String(x))
@inline _as_bool_or_nothing(x) = x === nothing ? nothing : Bool(x)
@inline _as_float_or_nothing(x) = x === nothing ? nothing : Float64(x)
@inline _as_int_or_nothing(x) = x === nothing ? nothing : Int(x)
@inline function _tuple2_float_or_nothing(x)
    x === nothing && return nothing
    length(x) == 2 || throw(ArgumentError("expected a 2-tuple/2-vector numeric payload"))
    return (Float64(x[1]), Float64(x[2]))
end
@inline function _int_tuple_or_scalar_or_nothing(x)
    x === nothing && return nothing
    if x isa AbstractVector || x isa Tuple
        return Tuple(Int(v) for v in x)
    end
    return Int(x)
end

@inline _float_vec(v) = Float64[Float64(x) for x in v]
@inline _int_vec(v) = Int[Int(x) for x in v]
@inline _vecvec_float(vv) = [Float64[Float64(x) for x in v] for v in vv]

function _axes_from_json(x)
    x === nothing && return nothing
    parts = [Float64[Float64(v) for v in ax] for ax in x]
    return tuple(parts...)
end

function _box_from_json(x)
    x === nothing && return nothing
    if x isa AbstractVector && length(x) == 2 &&
       x[1] isa AbstractVector && x[2] isa AbstractVector
        return (Float64[Float64(v) for v in x[1]], Float64[Float64(v) for v in x[2]])
    end
    return x
end

function _offset_weights_from_json(x)
    x === nothing && return nothing
    x isa AbstractVector || throw(ArgumentError("offset_weights metadata must be a vector/matrix-like JSON array or null"))
    if isempty(x)
        return Float64[]
    end
    if x[1] isa AbstractVector
        nr = length(x)
        nc = length(x[1])
        M = Matrix{Float64}(undef, nr, nc)
        @inbounds for i in 1:nr
            length(x[i]) == nc || throw(ArgumentError("offset_weights rows must have consistent lengths"))
            for j in 1:nc
                M[i, j] = Float64(x[i][j])
            end
        end
        return M
    end
    return Float64[Float64(v) for v in x]
end

function _to_plain(x)
    if x isa AbstractVector
        return Any[_to_plain(v) for v in x]
    elseif x isa AbstractDict
        out = Dict{String,Any}()
        for (k, v) in pairs(x)
            out[String(k)] = _to_plain(v)
        end
        return out
    else
        return x
    end
end

@inline function _spec_fields_dict(meta)
    if _obj_haskey(meta, "fields")
        return meta["fields"]
    end
    return meta
end

"""
    spec_from_metadata(meta_or_spec; resolve_ref=nothing, require_resolved_refs=false) -> AbstractFeaturizerSpec

Reconstruct a typed featurizer spec from metadata JSON payloads produced by
`feature_metadata`/`save_metadata_json`. Accepts either a full metadata object
containing `"spec"` or the spec payload itself.

For `ProjectedDistancesSpec`, pass `resolve_ref=id -> obj` to rehydrate
reference objects from persisted `reference_ids`.
"""
function spec_from_metadata(meta_or_spec;
                            resolve_ref::Union{Nothing,Function}=nothing,
                            require_resolved_refs::Bool=false)::AbstractFeaturizerSpec
    meta_or_spec isa AbstractFeaturizerSpec && return meta_or_spec
    node = _obj_haskey(meta_or_spec, "spec") ? meta_or_spec["spec"] : meta_or_spec
    (node === nothing || !_obj_haskey(node, "type")) &&
        throw(ArgumentError("spec_from_metadata expects a serialized spec payload with a `type` field"))
    Tname = String(node["type"])
    fields = _spec_fields_dict(node)

    if Tname == "LandscapeSpec"
        return LandscapeSpec(
            directions=_vecvec_float(fields["directions"]),
            offsets=_vecvec_float(fields["offsets"]),
            offset_weights=_offset_weights_from_json(_obj_get(fields, "offset_weights", nothing)),
            kmax=Int(fields["kmax"]),
            tgrid=_float_vec(fields["tgrid"]),
            aggregate=_as_symbol(fields["aggregate"]),
            normalize_weights=Bool(fields["normalize_weights"]),
            tmin=_as_float_or_nothing(_obj_get(fields, "tmin", nothing)),
            tmax=_as_float_or_nothing(_obj_get(fields, "tmax", nothing)),
            nsteps=Int(fields["nsteps"]),
            strict=Bool(fields["strict"]),
            drop_unknown=Bool(fields["drop_unknown"]),
            dedup=Bool(fields["dedup"]),
            normalize_dirs=_as_symbol(fields["normalize_dirs"]),
            direction_weight=_as_symbol(fields["direction_weight"]),
            threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        )
    elseif Tname == "PersistenceImageSpec"
        return PersistenceImageSpec(
            directions=_vecvec_float(fields["directions"]),
            offsets=_vecvec_float(fields["offsets"]),
            offset_weights=_offset_weights_from_json(_obj_get(fields, "offset_weights", nothing)),
            xgrid=_float_vec(fields["xgrid"]),
            ygrid=_float_vec(fields["ygrid"]),
            sigma=Float64(fields["sigma"]),
            coords=_as_symbol(fields["coords"]),
            weighting=_as_symbol(fields["weighting"]),
            p=Float64(fields["p"]),
            normalize=_as_symbol(fields["normalize"]),
            aggregate=_as_symbol(fields["aggregate"]),
            normalize_weights=Bool(fields["normalize_weights"]),
            tmin=_as_float_or_nothing(_obj_get(fields, "tmin", nothing)),
            tmax=_as_float_or_nothing(_obj_get(fields, "tmax", nothing)),
            nsteps=Int(fields["nsteps"]),
            strict=Bool(fields["strict"]),
            drop_unknown=Bool(fields["drop_unknown"]),
            dedup=Bool(fields["dedup"]),
            normalize_dirs=_as_symbol(fields["normalize_dirs"]),
            direction_weight=_as_symbol(fields["direction_weight"]),
            threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        )
    elseif Tname == "MPLandscapeSpec"
        return MPLandscapeSpec(
            directions=_vecvec_float(fields["directions"]),
            offsets=_vecvec_float(fields["offsets"]),
            offset_weights=_offset_weights_from_json(_obj_get(fields, "offset_weights", nothing)),
            kmax=Int(fields["kmax"]),
            tgrid=_float_vec(fields["tgrid"]),
            normalize_weights=Bool(fields["normalize_weights"]),
            tmin=_as_float_or_nothing(_obj_get(fields, "tmin", nothing)),
            tmax=_as_float_or_nothing(_obj_get(fields, "tmax", nothing)),
            nsteps=Int(fields["nsteps"]),
            strict=Bool(fields["strict"]),
            drop_unknown=Bool(fields["drop_unknown"]),
            dedup=Bool(fields["dedup"]),
            normalize_dirs=_as_symbol(fields["normalize_dirs"]),
            direction_weight=_as_symbol(fields["direction_weight"]),
            threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        )
    elseif Tname == "EulerSurfaceSpec"
        return EulerSurfaceSpec(
            axes=_axes_from_json(_obj_get(fields, "axes", nothing)),
            axes_policy=_as_symbol(fields["axes_policy"]),
            max_axis_len=Int(fields["max_axis_len"]),
            strict=_as_bool_or_nothing(_obj_get(fields, "strict", nothing)),
            threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        )
    elseif Tname == "RankGridSpec"
        return RankGridSpec(
            nvertices=Int(fields["nvertices"]),
            store_zeros=Bool(fields["store_zeros"]),
            threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        )
    elseif Tname == "RestrictedHilbertSpec"
        return RestrictedHilbertSpec(
            nvertices=Int(fields["nvertices"]),
        )
    elseif Tname == "BarcodeTopKSpec"
        return BarcodeTopKSpec(
            directions=_vecvec_float(fields["directions"]),
            offsets=_vecvec_float(fields["offsets"]),
            offset_weights=_offset_weights_from_json(_obj_get(fields, "offset_weights", nothing)),
            k=Int(fields["k"]),
            aggregate=_as_symbol(fields["aggregate"]),
            source=_as_symbol(fields["source"]),
            infinite_policy=_as_symbol(fields["infinite_policy"]),
            window=_tuple2_float_or_nothing(_obj_get(fields, "window", nothing)),
            normalize_weights=Bool(fields["normalize_weights"]),
            tmin=_as_float_or_nothing(_obj_get(fields, "tmin", nothing)),
            tmax=_as_float_or_nothing(_obj_get(fields, "tmax", nothing)),
            nsteps=Int(fields["nsteps"]),
            strict=Bool(fields["strict"]),
            drop_unknown=Bool(fields["drop_unknown"]),
            dedup=Bool(fields["dedup"]),
            normalize_dirs=_as_symbol(fields["normalize_dirs"]),
            direction_weight=_as_symbol(fields["direction_weight"]),
            threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        )
    elseif Tname == "SlicedBarcodeSpec"
        summary_fields = Tuple(_as_symbol(v) for v in fields["summary_fields"])
        return SlicedBarcodeSpec(
            directions=_vecvec_float(fields["directions"]),
            offsets=_vecvec_float(fields["offsets"]),
            offset_weights=_offset_weights_from_json(_obj_get(fields, "offset_weights", nothing)),
            featurizer=_as_symbol(fields["featurizer"]),
            summary_fields=summary_fields,
            summary_normalize_entropy=Bool(fields["summary_normalize_entropy"]),
            entropy_normalize=Bool(fields["entropy_normalize"]),
            entropy_weighting=_as_symbol(fields["entropy_weighting"]),
            entropy_p=Float64(fields["entropy_p"]),
            aggregate=_as_symbol(fields["aggregate"]),
            normalize_weights=Bool(fields["normalize_weights"]),
            tmin=_as_float_or_nothing(_obj_get(fields, "tmin", nothing)),
            tmax=_as_float_or_nothing(_obj_get(fields, "tmax", nothing)),
            nsteps=Int(fields["nsteps"]),
            strict=Bool(fields["strict"]),
            drop_unknown=Bool(fields["drop_unknown"]),
            dedup=Bool(fields["dedup"]),
            normalize_dirs=_as_symbol(fields["normalize_dirs"]),
            direction_weight=_as_symbol(fields["direction_weight"]),
            threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        )
    elseif Tname == "BarcodeSummarySpec"
        return BarcodeSummarySpec(
            directions=_vecvec_float(fields["directions"]),
            offsets=_vecvec_float(fields["offsets"]),
            offset_weights=_offset_weights_from_json(_obj_get(fields, "offset_weights", nothing)),
            fields=Tuple(_as_symbol(v) for v in fields["fields"]),
            normalize_entropy=Bool(fields["normalize_entropy"]),
            aggregate=_as_symbol(fields["aggregate"]),
            source=_as_symbol(fields["source"]),
            infinite_policy=_as_symbol(fields["infinite_policy"]),
            window=_tuple2_float_or_nothing(_obj_get(fields, "window", nothing)),
            normalize_weights=Bool(fields["normalize_weights"]),
            tmin=_as_float_or_nothing(_obj_get(fields, "tmin", nothing)),
            tmax=_as_float_or_nothing(_obj_get(fields, "tmax", nothing)),
            nsteps=Int(fields["nsteps"]),
            strict=Bool(fields["strict"]),
            drop_unknown=Bool(fields["drop_unknown"]),
            dedup=Bool(fields["dedup"]),
            normalize_dirs=_as_symbol(fields["normalize_dirs"]),
            direction_weight=_as_symbol(fields["direction_weight"]),
            threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        )
    elseif Tname == "SignedBarcodeImageSpec"
        return SignedBarcodeImageSpec(
            xs=_float_vec(fields["xs"]),
            ys=_float_vec(fields["ys"]),
            sigma=Float64(fields["sigma"]),
            mode=_as_symbol(fields["mode"]),
            method=_as_symbol(fields["method"]),
            axes=_axes_from_json(_obj_get(fields, "axes", nothing)),
            axes_policy=_as_symbol(fields["axes_policy"]),
            max_axis_len=Int(fields["max_axis_len"]),
            strict=_as_bool_or_nothing(_obj_get(fields, "strict", nothing)),
            threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        )
    elseif Tname == "PointSignedMeasureSpec"
        return PointSignedMeasureSpec(
            ndims=Int(fields["ndims"]),
            k=Int(fields["k"]),
            coords=_as_symbol(fields["coords"]),
        )
    elseif Tname == "EulerSignedMeasureSpec"
        return EulerSignedMeasureSpec(
            ndims=Int(fields["ndims"]),
            k=Int(fields["k"]),
            coords=_as_symbol(fields["coords"]),
            axes=_axes_from_json(_obj_get(fields, "axes", nothing)),
            axes_policy=_as_symbol(fields["axes_policy"]),
            max_axis_len=Int(fields["max_axis_len"]),
            strict=_as_bool_or_nothing(_obj_get(fields, "strict", nothing)),
            drop_zeros=Bool(fields["drop_zeros"]),
            max_terms=Int(fields["max_terms"]),
            min_abs_weight=Float64(fields["min_abs_weight"]),
            threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        )
    elseif Tname == "RectangleSignedBarcodeTopKSpec"
        return RectangleSignedBarcodeTopKSpec(
            ndims=Int(fields["ndims"]),
            k=Int(fields["k"]),
            coords=_as_symbol(fields["coords"]),
            axes=let ax = _obj_get(fields, "axes", nothing)
                ax === nothing ? nothing : ntuple(i -> _int_vec(ax[i]), length(ax))
            end,
            axes_policy=_as_symbol(fields["axes_policy"]),
            max_axis_len=Int(fields["max_axis_len"]),
            strict=_as_bool_or_nothing(_obj_get(fields, "strict", nothing)),
            drop_zeros=Bool(fields["drop_zeros"]),
            tol=Int(fields["tol"]),
            max_span=_int_tuple_or_scalar_or_nothing(_obj_get(fields, "max_span", nothing)),
            keep_endpoints=Bool(fields["keep_endpoints"]),
            method=_as_symbol(fields["method"]),
            bulk_max_elems=Int(fields["bulk_max_elems"]),
            threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        )
    elseif Tname == "ProjectedDistancesSpec"
        refs_raw = _obj_get(fields, "references", Any[])
        names = Symbol[_as_symbol(nm) for nm in fields["reference_names"]]
        ref_ids = String[string(x) for x in _obj_get(fields, "reference_ids", String[string(nm) for nm in names])]
        refs = if resolve_ref === nothing
            Any[_to_plain(r) for r in refs_raw]
        else
            out = Vector{Any}(undef, length(ref_ids))
            @inbounds for i in eachindex(ref_ids)
                resolved = resolve_ref(ref_ids[i])
                if resolved === nothing
                    require_resolved_refs &&
                        throw(ArgumentError("spec_from_metadata: unresolved projected reference id $(ref_ids[i])"))
                    if i <= length(refs_raw)
                        out[i] = _to_plain(refs_raw[i])
                    else
                        out[i] = Dict("kind" => "missing_reference", "id" => ref_ids[i])
                    end
                else
                    out[i] = resolved
                end
            end
            out
        end
        dirs = _obj_get(fields, "directions", nothing)
        return ProjectedDistancesSpec(
            refs;
            reference_names=names,
            dist=_as_symbol(fields["dist"]),
            p=Float64(fields["p"]),
            q=Float64(fields["q"]),
            agg=_as_symbol(fields["agg"]),
            directions=dirs === nothing ? nothing : _vecvec_float(dirs),
            n_dirs=Int(fields["n_dirs"]),
            normalize=_as_symbol(fields["normalize"]),
            enforce_monotone=_as_symbol(fields["enforce_monotone"]),
            precompute=Bool(fields["precompute"]),
            threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        )
    elseif Tname == "MatchingDistanceBankSpec"
        refs_raw = _obj_get(fields, "references", Any[])
        names = Symbol[_as_symbol(nm) for nm in fields["reference_names"]]
        ref_ids = String[string(x) for x in _obj_get(fields, "reference_ids", String[string(nm) for nm in names])]
        refs = if resolve_ref === nothing
            Any[_to_plain(r) for r in refs_raw]
        else
            out = Vector{Any}(undef, length(ref_ids))
            @inbounds for i in eachindex(ref_ids)
                resolved = resolve_ref(ref_ids[i])
                if resolved === nothing
                    require_resolved_refs &&
                        throw(ArgumentError("spec_from_metadata: unresolved matching-distance reference id $(ref_ids[i])"))
                    if i <= length(refs_raw)
                        out[i] = _to_plain(refs_raw[i])
                    else
                        out[i] = Dict("kind" => "missing_reference", "id" => ref_ids[i])
                    end
                else
                    out[i] = resolved
                end
            end
            out
        end
        dirs = _obj_get(fields, "directions", nothing)
        offs = _obj_get(fields, "offsets", nothing)
        return MatchingDistanceBankSpec(
            refs;
            reference_names=names,
            method=_as_symbol(fields["method"]),
            directions=dirs === nothing ? nothing : _vecvec_float(dirs),
            offsets=offs === nothing ? nothing : _vecvec_float(offs),
            n_dirs=Int(fields["n_dirs"]),
            n_offsets=Int(fields["n_offsets"]),
            max_den=Int(fields["max_den"]),
            include_axes=Bool(fields["include_axes"]),
            normalize_dirs=_as_symbol(fields["normalize_dirs"]),
            weight=_as_symbol(fields["weight"]),
            threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        )
    elseif Tname == "MPPImageSpec"
        delta_raw = _obj_get(fields, "delta", :auto)
        delta_val = if delta_raw isa AbstractString
            Symbol(String(delta_raw))
        elseif delta_raw isa Symbol
            delta_raw
        else
            Float64(delta_raw)
        end
        return MPPImageSpec(
            resolution=Int(fields["resolution"]),
            xgrid=_obj_get(fields, "xgrid", nothing) === nothing ? nothing : _float_vec(fields["xgrid"]),
            ygrid=_obj_get(fields, "ygrid", nothing) === nothing ? nothing : _float_vec(fields["ygrid"]),
            sigma=Float64(fields["sigma"]),
            N=Int(fields["N"]),
            delta=delta_val,
            q=Float64(fields["q"]),
            tie_break=_as_symbol(fields["tie_break"]),
            cutoff_radius=_as_float_or_nothing(_obj_get(fields, "cutoff_radius", nothing)),
            cutoff_tol=_as_float_or_nothing(_obj_get(fields, "cutoff_tol", nothing)),
            segment_prune=Bool(fields["segment_prune"]),
            threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        )
    elseif Tname == "MPPDecompositionHistogramSpec"
        delta_raw = _obj_get(fields, "delta", :auto)
        delta_val = if delta_raw isa AbstractString
            Symbol(String(delta_raw))
        elseif delta_raw isa Symbol
            delta_raw
        else
            Float64(delta_raw)
        end
        return MPPDecompositionHistogramSpec(
            orientation_bins=Int(fields["orientation_bins"]),
            scale_bins=Int(fields["scale_bins"]),
            scale_range=_tuple2_float_or_nothing(_obj_get(fields, "scale_range", nothing)),
            weight=_as_symbol(fields["weight"]),
            normalize=_as_symbol(fields["normalize"]),
            N=Int(fields["N"]),
            delta=delta_val,
            q=Float64(fields["q"]),
            tie_break=_as_symbol(fields["tie_break"]),
            threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        )
    elseif Tname == "BettiTableSpec"
        return BettiTableSpec(
            nvertices=Int(fields["nvertices"]),
            pad_to=Int(fields["pad_to"]),
            resolution_maxlen=Int(fields["resolution_maxlen"]),
            minimal=Bool(fields["minimal"]),
            check=Bool(fields["check"]),
        )
    elseif Tname == "BassTableSpec"
        return BassTableSpec(
            nvertices=Int(fields["nvertices"]),
            pad_to=Int(fields["pad_to"]),
            resolution_maxlen=Int(fields["resolution_maxlen"]),
            minimal=Bool(fields["minimal"]),
            check=Bool(fields["check"]),
        )
    elseif Tname == "BettiSupportMeasuresSpec"
        return BettiSupportMeasuresSpec(
            pad_to=Int(fields["pad_to"]),
            resolution_maxlen=Int(fields["resolution_maxlen"]),
            minimal=Bool(fields["minimal"]),
            check=Bool(fields["check"]),
        )
    elseif Tname == "BassSupportMeasuresSpec"
        return BassSupportMeasuresSpec(
            pad_to=Int(fields["pad_to"]),
            resolution_maxlen=Int(fields["resolution_maxlen"]),
            minimal=Bool(fields["minimal"]),
            check=Bool(fields["check"]),
        )
    elseif Tname == "CompositeSpec"
        specs_raw = _obj_get(fields, "specs", Any[])
        sub_specs = map(s -> spec_from_metadata(s;
                                                resolve_ref=resolve_ref,
                                                require_resolved_refs=require_resolved_refs),
                        specs_raw)
        return CompositeSpec(tuple(sub_specs...); namespacing=Bool(_obj_get(fields, "namespacing", true)))
    else
        throw(ArgumentError("Unsupported featurizer spec type in metadata: $(Tname)"))
    end
end

"""
    load_spec_with_resolver(meta_or_spec, resolve_ref; require_all=true) -> AbstractFeaturizerSpec

Typed spec round-trip with explicit projected-reference rehydration callback.
`resolve_ref` is called as `resolve_ref(reference_id::String)` for each projected
reference id encoded in metadata.
"""
function load_spec_with_resolver(meta_or_spec,
                                 resolve_ref::Function;
                                 require_all::Bool=true)::AbstractFeaturizerSpec
    return spec_from_metadata(meta_or_spec;
                              resolve_ref=resolve_ref,
                              require_resolved_refs=require_all)
end

"""
    invariant_options_from_metadata(meta_or_opts) -> InvariantOptions

Reconstruct a typed `InvariantOptions` object from metadata payloads. Accepts
either a full metadata object containing `"opts"` or the serialized options.
"""
function invariant_options_from_metadata(meta_or_opts)::InvariantOptions
    meta_or_opts isa InvariantOptions && return meta_or_opts
    node = _obj_haskey(meta_or_opts, "opts") ? meta_or_opts["opts"] : meta_or_opts
    node === nothing && return InvariantOptions()
    fields = _spec_fields_dict(node)
    return InvariantOptions(
        axes=_axes_from_json(_obj_get(fields, "axes", nothing)),
        axes_policy=_as_symbol(_obj_get(fields, "axes_policy", :encoding)),
        max_axis_len=Int(_obj_get(fields, "max_axis_len", 256)),
        box=_box_from_json(_obj_get(fields, "box", nothing)),
        threads=_as_bool_or_nothing(_obj_get(fields, "threads", nothing)),
        strict=_as_bool_or_nothing(_obj_get(fields, "strict", nothing)),
        pl_mode=_as_symbol(_obj_get(fields, "pl_mode", :fast)),
    )
end

function load_metadata_json(path::AbstractString;
                            validate_feature_schema::Bool=false,
                            typed::Bool=false,
                            resolve_ref::Union{Nothing,Function}=nothing,
                            require_resolved_refs::Bool=false)
    obj = open(JSON3.read, path)
    if validate_feature_schema
        validate_feature_metadata_schema(obj)
    end
    if !typed
        return obj
    end
    spec = _obj_haskey(obj, "spec") ? spec_from_metadata(obj["spec"];
                                                          resolve_ref=resolve_ref,
                                                          require_resolved_refs=require_resolved_refs) : nothing
    opts = _obj_haskey(obj, "opts") ? invariant_options_from_metadata(obj["opts"]) : InvariantOptions()
    return (raw=obj, spec=spec, opts=opts)
end

function _featureset_from_wide_columntable(cols; ids_col::Symbol=:id, meta=NamedTuple())
    names_nt = propertynames(cols)
    ids_col in names_nt || throw(ArgumentError("wide feature table must include id column :$(ids_col)"))
    ids_raw = getproperty(cols, ids_col)
    ids = String[string(x) for x in ids_raw]
    ns = length(ids)
    feat_syms = [nm for nm in names_nt if nm != ids_col]
    nf = length(feat_syms)
    nf == 0 && return FeatureSet(zeros(Float64, ns, 0), Symbol[], ids, meta)

    colvecs = Vector{AbstractVector}(undef, nf)
    T = Union{}
    @inbounds for j in 1:nf
        vj = collect(getproperty(cols, feat_syms[j]))
        length(vj) == ns || throw(DimensionMismatch("column $(feat_syms[j]) has length $(length(vj)); expected $ns"))
        colvecs[j] = vj
        T = j == 1 ? eltype(vj) : promote_type(T, eltype(vj))
    end

    X = Matrix{T}(undef, ns, nf)
    @inbounds for j in 1:nf
        copyto!(view(X, :, j), colvecs[j])
    end
    return FeatureSet(X, Symbol.(feat_syms), ids, meta)
end

function _featureset_from_long_columntable(cols; meta=NamedTuple())
    names_nt = propertynames(cols)
    (:id in names_nt && :feature in names_nt && :value in names_nt) ||
        throw(ArgumentError("long feature table must include columns :id, :feature, :value"))
    id_col = collect(getproperty(cols, :id))
    feat_col = collect(getproperty(cols, :feature))
    val_col = collect(getproperty(cols, :value))
    n = length(id_col)
    (length(feat_col) == n && length(val_col) == n) ||
        throw(DimensionMismatch("long feature table columns must have equal lengths"))

    names = Symbol[]
    feat_pos = Dict{Symbol,Int}()
    if :sample_index in names_nt
        sidx_col = collect(getproperty(cols, :sample_index))
        length(sidx_col) == n || throw(DimensionMismatch("sample_index column must match long table length"))

        ns = 0
        @inbounds for s in sidx_col
            si = Int(s)
            si >= 1 || throw(ArgumentError("sample_index values must be >= 1"))
            ns = max(ns, si)
        end
        ids = ["" for _ in 1:ns]

        @inbounds for k in 1:n
            fk = Symbol(feat_col[k])
            if !haskey(feat_pos, fk)
                push!(names, fk)
                feat_pos[fk] = length(names)
            end
            i = Int(sidx_col[k])
            if isempty(ids[i])
                ids[i] = string(id_col[k])
            end
        end
        @inbounds for i in eachindex(ids)
            isempty(ids[i]) && (ids[i] = string(i))
        end

        nf = length(names)
        T = eltype(val_col)
        has_missing = Missing <: T
        X = has_missing ? Matrix{Union{Missing,Base.nonmissingtype(T)}}(missing, ns, nf) :
                          zeros(T, ns, nf)
        @inbounds for k in 1:n
            i = Int(sidx_col[k])
            j = feat_pos[Symbol(feat_col[k])]
            X[i, j] = val_col[k]
        end
        return FeatureSet(X, names, ids, meta)
    else
        ids = String[]
        id_pos = Dict{String,Int}()
        @inbounds for k in 1:n
            idk = string(id_col[k])
            if !haskey(id_pos, idk)
                push!(ids, idk)
                id_pos[idk] = length(ids)
            end
            fk = Symbol(feat_col[k])
            if !haskey(feat_pos, fk)
                push!(names, fk)
                feat_pos[fk] = length(names)
            end
        end

        ns = length(ids)
        nf = length(names)
        T = eltype(val_col)
        has_missing = Missing <: T
        X = has_missing ? Matrix{Union{Missing,Base.nonmissingtype(T)}}(missing, ns, nf) :
                          zeros(T, ns, nf)
        @inbounds for k in 1:n
            i = id_pos[string(id_col[k])]
            j = feat_pos[Symbol(feat_col[k])]
            X[i, j] = val_col[k]
        end
        return FeatureSet(X, names, ids, meta)
    end
end

function _featureset_from_columntable(cols;
                                      format::Symbol=:wide,
                                      ids_col::Symbol=:id,
                                      meta=NamedTuple())
    if format == :wide
        return _featureset_from_wide_columntable(cols; ids_col=ids_col, meta=meta)
    elseif format == :long
        return _featureset_from_long_columntable(cols; meta=meta)
    else
        throw(ArgumentError("_featureset_from_columntable: format must be :wide or :long"))
    end
end

function save_features_arrow(path, fs; kwargs...)
    throw(ArgumentError("save_features_arrow requires Arrow.jl extension (load Arrow and ensure TamerOpArrowExt is available)."))
end

function load_features_arrow(path; kwargs...)
    throw(ArgumentError("load_features_arrow requires Arrow.jl extension (load Arrow and ensure TamerOpArrowExt is available)."))
end

function save_features_parquet(path, fs; kwargs...)
    throw(ArgumentError("save_features_parquet requires Parquet2.jl extension (load Parquet2 and ensure TamerOpParquet2Ext is available)."))
end

function load_features_parquet(path; kwargs...)
    throw(ArgumentError("load_features_parquet requires Parquet2.jl extension (load Parquet2 and ensure TamerOpParquet2Ext is available)."))
end

function save_features_npz(path, fs; kwargs...)
    throw(ArgumentError("save_features_npz requires NPZ.jl extension (load NPZ and ensure TamerOpNPZExt is available)."))
end

function load_features_npz(path; kwargs...)
    throw(ArgumentError("load_features_npz requires NPZ.jl extension (load NPZ and ensure TamerOpNPZExt is available)."))
end

function save_features_csv(path, fs; kwargs...)
    mode = get(kwargs, :format, :wide)
    layout = get(kwargs, :layout, :samples_by_features)
    include_metadata = get(kwargs, :include_metadata, true)
    metadata_path = get(kwargs, :metadata_path, nothing)
    ids_col = get(kwargs, :ids_col, :id)
    include_ids = get(kwargs, :include_ids, true)
    include_sample_index = get(kwargs, :include_sample_index, true)

    layout in (:samples_by_features, :features_by_samples) ||
        throw(ArgumentError("save_features_csv: layout must be :samples_by_features or :features_by_samples"))
    mode in (:wide, :long) ||
        throw(ArgumentError("save_features_csv: format must be :wide or :long"))

    # If extra CSV.jl-specific kwargs are provided, require the CSV extension path.
    allowed = Set([
        :format, :layout, :include_metadata, :metadata_path, :ids_col, :include_ids, :include_sample_index,
    ])
    extra = Symbol[]
    for (k, _) in kwargs
        kk = Symbol(k)
        kk in allowed || push!(extra, kk)
    end
    isempty(extra) || throw(ArgumentError("save_features_csv fallback does not support CSV-specific kwargs ($(join(string.(extra), ", "))). Load CSV.jl to enable extension-backed writer."))

    fs_out = if layout == :samples_by_features
        fs
    else
        XF = permutedims(fs.X)
        feature_ids = String[string(nm) for nm in fs.names]
        sample_names = Symbol["sample_$(i)" for i in eachindex(fs.ids)]
        FeatureSet(XF, sample_names, feature_ids, fs.meta)
    end

    if mode == :wide
        _write_feature_csv_wide(path, fs_out.X, fs_out.names, fs_out.ids; ids_col=ids_col, include_ids=include_ids)
    else
        _write_feature_csv_long(path, fs_out.X, fs_out.names, fs_out.ids; include_sample_index=include_sample_index)
    end

    if include_metadata
        md = feature_metadata(fs; format=mode)
        md["layout"] = String(layout)
        md["csv_ids_col"] = String(ids_col)
        md["csv_include_ids"] = include_ids
        md["csv_include_sample_index"] = include_sample_index
        mpath = metadata_path === nothing ? default_feature_metadata_path(path) : String(metadata_path)
        save_metadata_json(mpath, md)
    end
    return path
end

function load_features_csv(path; kwargs...)
    throw(ArgumentError("load_features_csv requires CSV.jl extension (load CSV and ensure TamerOpCSVExt is available)."))
end

@inline function _interop_format_from_path(path::AbstractString)
    ext = lowercase(splitext(String(path))[2])
    if ext == ".arrow"
        return :arrow
    elseif ext == ".parquet" || ext == ".pq"
        return :parquet
    elseif ext == ".npz"
        return :npz
    elseif ext == ".csv"
        return :csv
    end
    return nothing
end

@inline function _resolve_interop_format(path::AbstractString, format::Symbol)
    if format === :auto
        fmt = _interop_format_from_path(path)
        fmt === nothing && throw(ArgumentError("save/load_features with format=:auto requires path extension in (.arrow, .parquet, .npz, .csv)."))
        return fmt
    end
    return format
end

@inline function _normalize_feature_mode(mode::Symbol)
    (mode === :wide || mode === :long) && return mode
    throw(ArgumentError("feature mode must be :wide or :long (got $(mode))"))
end

@inline function _normalize_feature_layout(layout::Symbol)
    (layout === :samples_by_features || layout === :features_by_samples) && return layout
    throw(ArgumentError("feature layout must be :samples_by_features or :features_by_samples (got $(layout))"))
end

"""
    save_features(path, fs::FeatureSet; format=:auto, layout=:samples_by_features,
                  mode=:wide, metadata=true, kwargs...)

Unified feature artifact writer. Dispatches to Arrow/Parquet/NPZ/CSV backend writers.
Actual backend implementations live in package extensions.
"""
function save_features(path::AbstractString,
                       fs::FeatureSet;
                       format::Symbol=:auto,
                       layout::Symbol=:samples_by_features,
                       mode::Symbol=:wide,
                       metadata::Bool=true,
                       kwargs...)
    fmt = _resolve_interop_format(path, format)
    mode0 = _normalize_feature_mode(mode)
    layout0 = _normalize_feature_layout(layout)

    if (fmt === :arrow || fmt === :parquet) && layout0 !== :samples_by_features
        throw(ArgumentError("layout=$(layout0) is not supported for $(fmt); use :samples_by_features"))
    end

    if fmt === :arrow
        return save_features_arrow(path, fs; format=mode0, include_metadata=metadata, kwargs...)
    elseif fmt === :parquet
        return save_features_parquet(path, fs; format=mode0, include_metadata=metadata, kwargs...)
    elseif fmt === :npz
        return save_features_npz(path, fs; format=mode0, layout=layout0, include_metadata=metadata, kwargs...)
    elseif fmt === :csv
        return save_features_csv(path, fs; format=mode0, layout=layout0, include_metadata=metadata, kwargs...)
    end
    throw(ArgumentError("Unsupported feature format: $(fmt). Supported: :arrow, :parquet, :npz, :csv"))
end

"""
    load_features(path; format=:auto, layout=:auto, mode=:wide, kwargs...)

Unified feature artifact loader. Dispatches to Arrow/Parquet/NPZ/CSV backend loaders.
Actual backend implementations live in package extensions.
"""
function load_features(path::AbstractString;
                       format::Symbol=:auto,
                       layout::Symbol=:auto,
                       mode::Symbol=:wide,
                       kwargs...)
    fmt = _resolve_interop_format(path, format)
    mode0 = _normalize_feature_mode(mode)
    layout0 = layout === :auto ? nothing : _normalize_feature_layout(layout)

    if (fmt === :arrow || fmt === :parquet) &&
       !(layout0 === nothing || layout0 === :samples_by_features)
        throw(ArgumentError("layout=$(layout0) is not supported for $(fmt); use :samples_by_features"))
    end

    if fmt === :arrow
        return load_features_arrow(path; format=mode0, kwargs...)
    elseif fmt === :parquet
        return load_features_parquet(path; format=mode0, kwargs...)
    elseif fmt === :npz
        return load_features_npz(path; format=mode0, layout=layout0, kwargs...)
    elseif fmt === :csv
        return load_features_csv(path; format=mode0, layout=layout0, kwargs...)
    end
    throw(ArgumentError("Unsupported feature format: $(fmt). Supported: :arrow, :parquet, :npz, :csv"))
end

"""
    EulerSurfaceLongTable(values; axes=nothing, id="sample")

Long-form table wrapper for 2D Euler surfaces.
"""
struct EulerSurfaceLongTable{T<:Real}
    id::String
    x::Vector{Float64}
    y::Vector{Float64}
    values::Matrix{T}
end

"""
    PersistenceImageLongTable(pi; id="sample")

Long-form table wrapper for `PersistenceImage1D`.
"""
struct PersistenceImageLongTable
    id::String
    image::PersistenceImage1D
end

"""
    MPLandscapeLongTable(L; id="sample")

Long-form table wrapper for `MPLandscape`.
"""
struct MPLandscapeLongTable{D,O}
    id::String
    landscape::MPLandscape{D,O}
end

"""
    PointSignedMeasureLongTable(pm; id="sample")

Long-form table wrapper for signed point measures.
"""
struct PointSignedMeasureLongTable{N,T,W}
    id::String
    measure::PointSignedMeasure{N,T,W}
end

@inline feature_table(fs::FeatureSet; format::Symbol=:wide) =
    format == :wide ? FeatureSetWideTable(fs) :
    format == :long ? FeatureSetLongTable(fs) :
    throw(ArgumentError("feature_table: format must be :wide or :long"))

function euler_surface_table(values::AbstractMatrix{T};
                             axes=nothing,
                             id::AbstractString="sample") where {T<:Real}
    if axes === nothing
        x = Float64[i for i in 1:size(values, 1)]
        y = Float64[j for j in 1:size(values, 2)]
    else
        length(axes) == 2 || throw(ArgumentError("euler_surface_table: axes must be a 2-tuple"))
        x = Float64[float(v) for v in axes[1]]
        y = Float64[float(v) for v in axes[2]]
        length(x) == size(values, 1) || throw(DimensionMismatch("x-axis length does not match first dimension"))
        length(y) == size(values, 2) || throw(DimensionMismatch("y-axis length does not match second dimension"))
    end
    return EulerSurfaceLongTable{T}(String(id), x, y, Matrix(values))
end

@inline persistence_image_table(pi::PersistenceImage1D; id::AbstractString="sample") =
    PersistenceImageLongTable(String(id), pi)

@inline mp_landscape_table(L::MPLandscape{D,O}; id::AbstractString="sample") where {D,O} =
    MPLandscapeLongTable{D,O}(String(id), L)

@inline point_signed_measure_table(pm::PointSignedMeasure{N,T,W}; id::AbstractString="sample") where {N,T,W} =
    PointSignedMeasureLongTable{N,T,W}(String(id), pm)

@inline _default_threads_flag(flag::Union{Nothing,Bool}) =
    flag === nothing ? (Threads.nthreads() > 1) : flag

@inline function _resolve_spec_threads(spec_threads::Union{Nothing,Bool},
                                       opts::InvariantOptions,
                                       threaded_default::Bool)
    if spec_threads !== nothing
        return spec_threads
    end
    if opts.threads !== nothing
        return opts.threads
    end
    return threaded_default
end

@inline _float_vector(v::Real) = Float64[float(v)]
@inline _float_vector(v::AbstractVector) = Float64[float(x) for x in v]
@inline _float_vector(v::AbstractArray) = Float64[float(x) for x in vec(v)]

function _to_vecvec_float(xs)
    out = Vector{Vector{Float64}}(undef, length(xs))
    @inbounds for i in eachindex(xs)
        out[i] = Float64[float(x) for x in xs[i]]
    end
    return out
end

function _to_offset_weights(weights)
    weights === nothing && return nothing
    if weights isa AbstractVector
        return Float64[float(x) for x in weights]
    elseif weights isa AbstractMatrix
        return Float64[float(x) for x in weights]
    else
        throw(ArgumentError("offset_weights must be a vector, matrix, or nothing"))
    end
end

function LandscapeSpec(; directions,
                        offsets,
                        offset_weights=nothing,
                        kmax::Int=5,
                        tgrid,
                        aggregate::Symbol=:mean,
                        normalize_weights::Bool=true,
                        tmin=nothing,
                        tmax=nothing,
                        nsteps::Int=401,
                        strict::Bool=true,
                        drop_unknown::Bool=true,
                        dedup::Bool=true,
                        normalize_dirs::Symbol=:none,
                        direction_weight::Symbol=:none,
                        threads=nothing)
    return LandscapeSpec(
        _to_vecvec_float(directions),
        _to_vecvec_float(offsets),
        _to_offset_weights(offset_weights),
        kmax,
        Float64[float(x) for x in tgrid],
        aggregate,
        normalize_weights,
        tmin === nothing ? nothing : float(tmin),
        tmax === nothing ? nothing : float(tmax),
        nsteps,
        strict,
        drop_unknown,
        dedup,
        normalize_dirs,
        direction_weight,
        threads === nothing ? nothing : Bool(threads),
    )
end

function PersistenceImageSpec(; directions,
                               offsets,
                               offset_weights=nothing,
                               xgrid,
                               ygrid,
                               sigma::Real=1.0,
                               coords::Symbol=:birth_persistence,
                               weighting::Symbol=:persistence,
                               p::Real=1.0,
                               normalize::Symbol=:none,
                               aggregate::Symbol=:mean,
                               normalize_weights::Bool=true,
                               tmin=nothing,
                               tmax=nothing,
                               nsteps::Int=401,
                               strict::Bool=true,
                               drop_unknown::Bool=true,
                               dedup::Bool=true,
                               normalize_dirs::Symbol=:none,
                               direction_weight::Symbol=:none,
                               threads=nothing)
    return PersistenceImageSpec(
        _to_vecvec_float(directions),
        _to_vecvec_float(offsets),
        _to_offset_weights(offset_weights),
        Float64[float(x) for x in xgrid],
        Float64[float(y) for y in ygrid],
        float(sigma),
        coords,
        weighting,
        float(p),
        normalize,
        aggregate,
        normalize_weights,
        tmin === nothing ? nothing : float(tmin),
        tmax === nothing ? nothing : float(tmax),
        nsteps,
        strict,
        drop_unknown,
        dedup,
        normalize_dirs,
        direction_weight,
        threads === nothing ? nothing : Bool(threads),
    )
end

function MPLandscapeSpec(; directions,
                          offsets,
                          offset_weights=nothing,
                          kmax::Int=5,
                          tgrid,
                          normalize_weights::Bool=true,
                          tmin=nothing,
                          tmax=nothing,
                          nsteps::Int=401,
                          strict::Bool=true,
                          drop_unknown::Bool=true,
                          dedup::Bool=true,
                          normalize_dirs::Symbol=:none,
                          direction_weight::Symbol=:none,
                          threads=nothing)
    return MPLandscapeSpec(
        _to_vecvec_float(directions),
        _to_vecvec_float(offsets),
        _to_offset_weights(offset_weights),
        kmax,
        Float64[float(x) for x in tgrid],
        normalize_weights,
        tmin === nothing ? nothing : float(tmin),
        tmax === nothing ? nothing : float(tmax),
        nsteps,
        strict,
        drop_unknown,
        dedup,
        normalize_dirs,
        direction_weight,
        threads === nothing ? nothing : Bool(threads),
    )
end

function EulerSurfaceSpec(; axes=nothing,
                           axes_policy::Symbol=:encoding,
                           max_axis_len::Int=256,
                           strict=nothing,
                           threads=nothing)
    axes2 = axes === nothing ? nothing :
        ntuple(i -> Float64[float(x) for x in axes[i]], length(axes))
    strict2 = strict === nothing ? nothing : Bool(strict)
    threads2 = threads === nothing ? nothing : Bool(threads)
    return EulerSurfaceSpec{typeof(axes2)}(axes2, axes_policy, max_axis_len, strict2, threads2)
end

RankGridSpec(; nvertices::Int, store_zeros::Bool=true, threads=nothing) =
    RankGridSpec(nvertices, store_zeros, threads === nothing ? nothing : Bool(threads))

function BarcodeTopKSpec(; directions,
                           offsets,
                           offset_weights=nothing,
                           k::Int=5,
                           aggregate::Symbol=:stack,
                           source::Symbol=:slice,
                           infinite_policy::Symbol=:clip_to_window,
                           window=nothing,
                           normalize_weights::Bool=true,
                           tmin=nothing,
                           tmax=nothing,
                           nsteps::Int=401,
                           strict::Bool=true,
                           drop_unknown::Bool=true,
                           dedup::Bool=true,
                           normalize_dirs::Symbol=:none,
                           direction_weight::Symbol=:none,
                           threads=nothing)
    k >= 0 || throw(ArgumentError("BarcodeTopKSpec.k must be >= 0"))
    aggregate in (:mean, :sum, :stack) ||
        throw(ArgumentError("BarcodeTopKSpec.aggregate must be :mean, :sum, or :stack"))
    source in (:slice, :fibered) ||
        throw(ArgumentError("BarcodeTopKSpec.source must be :slice or :fibered"))
    infinite_policy in (:clip_to_window, :error) ||
        throw(ArgumentError("BarcodeTopKSpec.infinite_policy must be :clip_to_window or :error"))
    window2 = window === nothing ? nothing : (float(window[1]), float(window[2]))
    window2 === nothing || window2[1] <= window2[2] ||
        throw(ArgumentError("BarcodeTopKSpec.window must satisfy lo <= hi"))
    return BarcodeTopKSpec(
        _to_vecvec_float(directions),
        _to_vecvec_float(offsets),
        _to_offset_weights(offset_weights),
        k,
        aggregate,
        source,
        infinite_policy,
        window2,
        normalize_weights,
        tmin === nothing ? nothing : float(tmin),
        tmax === nothing ? nothing : float(tmax),
        nsteps,
        strict,
        drop_unknown,
        dedup,
        normalize_dirs,
        direction_weight,
        threads === nothing ? nothing : Bool(threads),
    )
end

function SlicedBarcodeSpec(; directions,
                            offsets,
                            offset_weights=nothing,
                            featurizer::Symbol=:summary,
                            summary_fields::Tuple=_DEFAULT_CANONICAL_BARCODE_SUMMARY_FIELDS,
                            summary_normalize_entropy::Bool=true,
                            entropy_normalize::Bool=true,
                            entropy_weighting::Symbol=:persistence,
                            entropy_p::Real=1.0,
                            aggregate::Symbol=:mean,
                            normalize_weights::Bool=true,
                            tmin=nothing,
                            tmax=nothing,
                            nsteps::Int=401,
                            strict::Bool=true,
                            drop_unknown::Bool=true,
                            dedup::Bool=true,
                            normalize_dirs::Symbol=:none,
                            direction_weight::Symbol=:none,
                            threads=nothing)
    return SlicedBarcodeSpec(
        _to_vecvec_float(directions),
        _to_vecvec_float(offsets),
        _to_offset_weights(offset_weights),
        featurizer,
        Tuple(summary_fields),
        summary_normalize_entropy,
        entropy_normalize,
        entropy_weighting,
        float(entropy_p),
        aggregate,
        normalize_weights,
        tmin === nothing ? nothing : float(tmin),
        tmax === nothing ? nothing : float(tmax),
        nsteps,
        strict,
        drop_unknown,
        dedup,
        normalize_dirs,
        direction_weight,
        threads === nothing ? nothing : Bool(threads),
    )
end

const _DEFAULT_CANONICAL_BARCODE_SUMMARY_FIELDS =
    (:count, :sum_persistence, :mean_persistence, :max_persistence, :entropy)

@inline function _invariant_barcode_summary_field(field::Symbol)
    field === :count && return :n_intervals
    field === :sum_persistence && return :total_persistence
    field === :mean_persistence && return :mean_persistence
    field === :max_persistence && return :max_persistence
    field === :entropy && return :entropy
    throw(ArgumentError("unsupported barcode summary field $(field)"))
end

@inline _invariant_barcode_summary_fields(fields::Tuple) =
    Tuple(_invariant_barcode_summary_field(Symbol(field)) for field in fields)

function BarcodeSummarySpec(; directions,
                              offsets,
                              offset_weights=nothing,
                              fields::Tuple=_DEFAULT_CANONICAL_BARCODE_SUMMARY_FIELDS,
                              normalize_entropy::Bool=true,
                              aggregate::Symbol=:stack,
                              source::Symbol=:slice,
                              infinite_policy::Symbol=:clip_to_window,
                              window=nothing,
                              normalize_weights::Bool=true,
                              tmin=nothing,
                              tmax=nothing,
                              nsteps::Int=401,
                              strict::Bool=true,
                              drop_unknown::Bool=true,
                              dedup::Bool=true,
                              normalize_dirs::Symbol=:none,
                              direction_weight::Symbol=:none,
                              threads=nothing)
    aggregate in (:mean, :sum, :stack) ||
        throw(ArgumentError("BarcodeSummarySpec.aggregate must be :mean, :sum, or :stack"))
    source in (:slice, :fibered) ||
        throw(ArgumentError("BarcodeSummarySpec.source must be :slice or :fibered"))
    infinite_policy in (:clip_to_window, :error) ||
        throw(ArgumentError("BarcodeSummarySpec.infinite_policy must be :clip_to_window or :error"))
    allowed = Set((:count, :sum_persistence, :mean_persistence, :max_persistence, :entropy))
    all(f -> Symbol(f) in allowed, fields) ||
        throw(ArgumentError("BarcodeSummarySpec.fields must be chosen from $(collect(allowed))"))
    window2 = window === nothing ? nothing : (float(window[1]), float(window[2]))
    window2 === nothing || window2[1] <= window2[2] ||
        throw(ArgumentError("BarcodeSummarySpec.window must satisfy lo <= hi"))
    return BarcodeSummarySpec(
        _to_vecvec_float(directions),
        _to_vecvec_float(offsets),
        _to_offset_weights(offset_weights),
        Tuple(Symbol(f) for f in fields),
        normalize_entropy,
        aggregate,
        source,
        infinite_policy,
        window2,
        normalize_weights,
        tmin === nothing ? nothing : float(tmin),
        tmax === nothing ? nothing : float(tmax),
        nsteps,
        strict,
        drop_unknown,
        dedup,
        normalize_dirs,
        direction_weight,
        threads === nothing ? nothing : Bool(threads),
    )
end

function SignedBarcodeImageSpec(; xs,
                                 ys,
                                 sigma::Real=1.0,
                                 mode::Symbol=:center,
                                 method::Symbol=:bulk,
                                 axes=nothing,
                                 axes_policy::Symbol=:encoding,
                                 max_axis_len::Int=256,
                                 strict=nothing,
                                 threads=nothing)
    axes2 = axes === nothing ? nothing :
        ntuple(i -> Float64[float(x) for x in axes[i]], length(axes))
    strict2 = strict === nothing ? nothing : Bool(strict)
    threads2 = threads === nothing ? nothing : Bool(threads)
    return SignedBarcodeImageSpec{typeof(axes2)}(
        Float64[float(x) for x in xs],
        Float64[float(y) for y in ys],
        float(sigma),
        mode,
        method,
        axes2,
        axes_policy,
        max_axis_len,
        strict2,
        threads2,
    )
end

function PointSignedMeasureSpec(; ndims::Int,
                                 k::Int=16,
                                 coords::Symbol=:values)
    ndims > 0 || throw(ArgumentError("PointSignedMeasureSpec.ndims must be > 0"))
    k >= 0 || throw(ArgumentError("PointSignedMeasureSpec.k must be >= 0"))
    coords in (:values, :indices) ||
        throw(ArgumentError("PointSignedMeasureSpec.coords must be :values or :indices"))
    return PointSignedMeasureSpec(ndims, k, coords)
end

function EulerSignedMeasureSpec(; ndims::Int,
                                 k::Int=16,
                                 coords::Symbol=:values,
                                 axes=nothing,
                                 axes_policy::Symbol=:encoding,
                                 max_axis_len::Int=256,
                                 strict=nothing,
                                 drop_zeros::Bool=true,
                                 max_terms::Int=0,
                                 min_abs_weight::Real=0,
                                 threads=nothing)
    ndims > 0 || throw(ArgumentError("EulerSignedMeasureSpec.ndims must be > 0"))
    k >= 0 || throw(ArgumentError("EulerSignedMeasureSpec.k must be >= 0"))
    max_terms >= 0 || throw(ArgumentError("EulerSignedMeasureSpec.max_terms must be >= 0"))
    coords in (:values, :indices) ||
        throw(ArgumentError("EulerSignedMeasureSpec.coords must be :values or :indices"))
    axes2 = axes === nothing ? nothing :
        ntuple(i -> Float64[float(x) for x in axes[i]], length(axes))
    strict2 = strict === nothing ? nothing : Bool(strict)
    return EulerSignedMeasureSpec{typeof(axes2)}(
        ndims,
        k,
        coords,
        axes2,
        axes_policy,
        max_axis_len,
        strict2,
        drop_zeros,
        max_terms,
        float(min_abs_weight),
        threads === nothing ? nothing : Bool(threads),
    )
end

function RectangleSignedBarcodeTopKSpec(; ndims::Int,
                                         k::Int=16,
                                         coords::Symbol=:values,
                                         axes=nothing,
                                         axes_policy::Symbol=:encoding,
                                         max_axis_len::Int=256,
                                         strict=nothing,
                                         drop_zeros::Bool=true,
                                         tol::Int=0,
                                         max_span=nothing,
                                         keep_endpoints::Bool=true,
                                         method::Symbol=:auto,
                                         bulk_max_elems::Int=20_000_000,
                                         threads=nothing)
    ndims > 0 || throw(ArgumentError("RectangleSignedBarcodeTopKSpec.ndims must be > 0"))
    k >= 0 || throw(ArgumentError("RectangleSignedBarcodeTopKSpec.k must be >= 0"))
    coords in (:values, :indices) ||
        throw(ArgumentError("RectangleSignedBarcodeTopKSpec.coords must be :values or :indices"))
    method in (:auto, :bulk, :local) ||
        throw(ArgumentError("RectangleSignedBarcodeTopKSpec.method must be :auto, :bulk, or :local"))
    tol >= 0 || throw(ArgumentError("RectangleSignedBarcodeTopKSpec.tol must be >= 0"))
    bulk_max_elems > 0 || throw(ArgumentError("RectangleSignedBarcodeTopKSpec.bulk_max_elems must be > 0"))
    axes2 = axes === nothing ? nothing :
        ntuple(i -> Int[Int(x) for x in axes[i]], length(axes))
    strict2 = strict === nothing ? nothing : Bool(strict)
    max_span2 = if max_span === nothing
        nothing
    elseif max_span isa Tuple
        Tuple(Int(x) for x in max_span)
    else
        Int(max_span)
    end
    return RectangleSignedBarcodeTopKSpec{typeof(axes2),typeof(max_span2)}(
        ndims,
        k,
        coords,
        axes2,
        axes_policy,
        max_axis_len,
        strict2,
        drop_zeros,
        tol,
        max_span2,
        keep_endpoints,
        method,
        bulk_max_elems,
        threads === nothing ? nothing : Bool(threads),
    )
end

function ProjectedDistancesSpec(references;
                                reference_names=nothing,
                                dist::Symbol=:bottleneck,
                                p::Real=1.0,
                                q::Real=1.0,
                                agg::Symbol=:mean,
                                directions=nothing,
                                n_dirs::Int=32,
                                normalize::Symbol=:L1,
                                enforce_monotone::Symbol=:upper,
                                precompute::Bool=true,
                                threads=nothing)
    refs = collect(references)
    names = if reference_names === nothing
        [Symbol("ref_$(i)") for i in 1:length(refs)]
    else
        Symbol[Symbol(x) for x in reference_names]
    end
    length(names) == length(refs) || throw(ArgumentError("reference_names length must match references length"))
    dirs2 = directions === nothing ? nothing : _to_vecvec_float(directions)
    return ProjectedDistancesSpec{typeof(refs),typeof(dirs2)}(
        refs,
        names,
        dist,
        float(p),
        float(q),
        agg,
        dirs2,
        n_dirs,
        normalize,
        enforce_monotone,
        precompute,
        threads === nothing ? nothing : Bool(threads),
    )
end

function MatchingDistanceBankSpec(references;
                                  reference_names=nothing,
                                  method::Symbol=:auto,
                                  directions=nothing,
                                  offsets=nothing,
                                  n_dirs::Int=100,
                                  n_offsets::Int=50,
                                  max_den::Int=8,
                                  include_axes::Bool=false,
                                  normalize_dirs::Symbol=:L1,
                                  weight::Symbol=:lesnick_l1,
                                  threads=nothing)
    method in (:auto, :approx, :exact_2d) ||
        throw(ArgumentError("MatchingDistanceBankSpec.method must be :auto, :approx, or :exact_2d"))
    n_dirs > 0 || throw(ArgumentError("MatchingDistanceBankSpec.n_dirs must be > 0"))
    n_offsets > 0 || throw(ArgumentError("MatchingDistanceBankSpec.n_offsets must be > 0"))
    max_den > 0 || throw(ArgumentError("MatchingDistanceBankSpec.max_den must be > 0"))
    refs = collect(references)
    names = if reference_names === nothing
        [Symbol("ref_$(i)") for i in 1:length(refs)]
    else
        Symbol[Symbol(x) for x in reference_names]
    end
    length(names) == length(refs) || throw(ArgumentError("reference_names length must match references length"))
    dirs2 = directions === nothing ? nothing : _to_vecvec_float(directions)
    offs2 = offsets === nothing ? nothing : _to_vecvec_float(offsets)
    return MatchingDistanceBankSpec{typeof(refs),typeof(dirs2),typeof(offs2)}(
        refs,
        names,
        method,
        dirs2,
        offs2,
        n_dirs,
        n_offsets,
        max_den,
        include_axes,
        normalize_dirs,
        weight,
        threads === nothing ? nothing : Bool(threads),
    )
end

function MPPImageSpec(; resolution::Int=32,
                       xgrid=nothing,
                       ygrid=nothing,
                       sigma::Real=0.05,
                       N::Int=16,
                       delta::Union{Real,Symbol}=:auto,
                       q::Real=1.0,
                       tie_break::Symbol=:center,
                       cutoff_radius=nothing,
                       cutoff_tol=nothing,
                       segment_prune::Bool=true,
                       threads=nothing)
    resolution > 1 || throw(ArgumentError("MPPImageSpec.resolution must be >= 2"))
    xgrid2 = xgrid === nothing ? nothing : Float64[float(x) for x in xgrid]
    ygrid2 = ygrid === nothing ? nothing : Float64[float(y) for y in ygrid]
    xgrid2 === nothing || length(xgrid2) > 1 || throw(ArgumentError("MPPImageSpec.xgrid must have length >= 2"))
    ygrid2 === nothing || length(ygrid2) > 1 || throw(ArgumentError("MPPImageSpec.ygrid must have length >= 2"))
    delta2 = delta isa Symbol ? delta : float(delta)
    cutoff_radius2 = cutoff_radius === nothing ? nothing : float(cutoff_radius)
    cutoff_tol2 = cutoff_tol === nothing ? nothing : float(cutoff_tol)
    return MPPImageSpec(
        resolution,
        xgrid2,
        ygrid2,
        float(sigma),
        N,
        delta2,
        float(q),
        tie_break,
        cutoff_radius2,
        cutoff_tol2,
        segment_prune,
        threads === nothing ? nothing : Bool(threads),
    )
end

function MPPDecompositionHistogramSpec(; orientation_bins::Int=12,
                                         scale_bins::Int=12,
                                         scale_range=nothing,
                                         weight::Symbol=:mass,
                                         normalize::Symbol=:l1,
                                         N::Int=16,
                                         delta::Union{Real,Symbol}=:auto,
                                         q::Real=1.0,
                                         tie_break::Symbol=:center,
                                         threads=nothing)
    orientation_bins > 0 || throw(ArgumentError("MPPDecompositionHistogramSpec.orientation_bins must be > 0"))
    scale_bins > 0 || throw(ArgumentError("MPPDecompositionHistogramSpec.scale_bins must be > 0"))
    weight in (:count, :mass) ||
        throw(ArgumentError("MPPDecompositionHistogramSpec.weight must be :count or :mass"))
    normalize in (:none, :l1, :l2, :max) ||
        throw(ArgumentError("MPPDecompositionHistogramSpec.normalize must be :none, :l1, :l2, or :max"))
    scale_range2 = scale_range === nothing ? nothing : (float(scale_range[1]), float(scale_range[2]))
    scale_range2 === nothing || scale_range2[1] <= scale_range2[2] ||
        throw(ArgumentError("MPPDecompositionHistogramSpec.scale_range must satisfy lo <= hi"))
    delta2 = delta isa Symbol ? delta : float(delta)
    return MPPDecompositionHistogramSpec(
        orientation_bins,
        scale_bins,
        scale_range2,
        weight,
        normalize,
        N,
        delta2,
        float(q),
        tie_break,
        threads === nothing ? nothing : Bool(threads),
    )
end

BettiTableSpec(; nvertices::Int,
                 pad_to::Int,
                 resolution_maxlen::Int=pad_to,
                 minimal::Bool=false,
                 check::Bool=true) = begin
    nvertices >= 0 || throw(ArgumentError("BettiTableSpec.nvertices must be >= 0"))
    pad_to >= 0 || throw(ArgumentError("BettiTableSpec.pad_to must be >= 0"))
    resolution_maxlen >= 0 || throw(ArgumentError("BettiTableSpec.resolution_maxlen must be >= 0"))
    BettiTableSpec(nvertices, pad_to, resolution_maxlen, minimal, check)
end

BassTableSpec(; nvertices::Int,
                pad_to::Int,
                resolution_maxlen::Int=pad_to,
                minimal::Bool=false,
                check::Bool=true) = begin
    nvertices >= 0 || throw(ArgumentError("BassTableSpec.nvertices must be >= 0"))
    pad_to >= 0 || throw(ArgumentError("BassTableSpec.pad_to must be >= 0"))
    resolution_maxlen >= 0 || throw(ArgumentError("BassTableSpec.resolution_maxlen must be >= 0"))
    BassTableSpec(nvertices, pad_to, resolution_maxlen, minimal, check)
end

RestrictedHilbertSpec(; nvertices::Int) = begin
    nvertices >= 0 || throw(ArgumentError("RestrictedHilbertSpec.nvertices must be >= 0"))
    RestrictedHilbertSpec(nvertices)
end

BettiSupportMeasuresSpec(; pad_to::Int,
                           resolution_maxlen::Int=pad_to,
                           minimal::Bool=false,
                           check::Bool=true) = begin
    pad_to >= 0 || throw(ArgumentError("BettiSupportMeasuresSpec.pad_to must be >= 0"))
    resolution_maxlen >= 0 || throw(ArgumentError("BettiSupportMeasuresSpec.resolution_maxlen must be >= 0"))
    BettiSupportMeasuresSpec(pad_to, resolution_maxlen, minimal, check)
end

BassSupportMeasuresSpec(; pad_to::Int,
                          resolution_maxlen::Int=pad_to,
                          minimal::Bool=false,
                          check::Bool=true) = begin
    pad_to >= 0 || throw(ArgumentError("BassSupportMeasuresSpec.pad_to must be >= 0"))
    resolution_maxlen >= 0 || throw(ArgumentError("BassSupportMeasuresSpec.resolution_maxlen must be >= 0"))
    BassSupportMeasuresSpec(pad_to, resolution_maxlen, minimal, check)
end

CompositeSpec(specs::Tuple; namespacing::Bool=true) = CompositeSpec{typeof(specs)}(specs, namespacing)

function _slice_feature_names(prefix::String, per_slice::Int, nd::Int, no::Int, aggregate::Symbol)
    if aggregate == :stack
        out = Vector{Symbol}(undef, per_slice * nd * no)
        idx = 1
        for i in 1:nd, j in 1:no, k in 1:per_slice
            out[idx] = Symbol(prefix * "__d$(i)_o$(j)_f$(k)")
            idx += 1
        end
        return out
    else
        return [Symbol(prefix * "__f$(k)") for k in 1:per_slice]
    end
end

feature_axes(::AbstractFeaturizerSpec) = NamedTuple()

@inline _copy_vecvec(xs::Vector{Vector{Float64}}) = [copy(v) for v in xs]

function _axis_namedtuple(axes::NTuple{N,Vector{Float64}}, prefix::String="axis") where {N}
    names = ntuple(i -> Symbol(prefix * "_" * string(i)), N)
    vals = ntuple(i -> copy(axes[i]), N)
    return NamedTuple{names}(vals)
end

function feature_axes(spec::LandscapeSpec)
    base = (k=collect(1:spec.kmax), t=copy(spec.tgrid))
    if spec.aggregate == :stack
        return (direction=collect(1:length(spec.directions)),
                offset=collect(1:length(spec.offsets)),
                base...)
    end
    return (aggregate=spec.aggregate,
            base...,
            directions=_copy_vecvec(spec.directions),
            offsets=_copy_vecvec(spec.offsets))
end

feature_names(spec::LandscapeSpec) =
    _slice_feature_names("landscape", spec.kmax * length(spec.tgrid),
                         length(spec.directions), length(spec.offsets), spec.aggregate)

nfeatures(spec::LandscapeSpec) = length(feature_names(spec))

function feature_axes(spec::PersistenceImageSpec)
    base = (x=copy(spec.xgrid), y=copy(spec.ygrid))
    if spec.aggregate == :stack
        return (direction=collect(1:length(spec.directions)),
                offset=collect(1:length(spec.offsets)),
                base...)
    end
    return (aggregate=spec.aggregate,
            base...,
            directions=_copy_vecvec(spec.directions),
            offsets=_copy_vecvec(spec.offsets))
end

feature_names(spec::PersistenceImageSpec) =
    _slice_feature_names("persistence_image", length(spec.xgrid) * length(spec.ygrid),
                         length(spec.directions), length(spec.offsets), spec.aggregate)

nfeatures(spec::PersistenceImageSpec) = length(feature_names(spec))

function feature_axes(spec::MPLandscapeSpec)
    return (
        direction=collect(1:length(spec.directions)),
        offset=collect(1:length(spec.offsets)),
        k=collect(1:spec.kmax),
        t=copy(spec.tgrid),
        directions=_copy_vecvec(spec.directions),
        offsets=_copy_vecvec(spec.offsets),
    )
end

function feature_names(spec::MPLandscapeSpec)
    nd = length(spec.directions)
    no = length(spec.offsets)
    nt = length(spec.tgrid)
    out = Vector{Symbol}(undef, nd * no * spec.kmax * nt)
    idx = 1
    for it in 1:nt, k in 1:spec.kmax, j in 1:no, i in 1:nd
        out[idx] = Symbol("mp_landscape__d$(i)_o$(j)_k$(k)_t$(it)")
        idx += 1
    end
    return out
end

nfeatures(spec::MPLandscapeSpec) = length(feature_names(spec))

function feature_axes(spec::EulerSurfaceSpec)
    spec.axes === nothing && return (axes_policy=spec.axes_policy, axes=nothing)
    return (_axis_namedtuple(spec.axes)..., axes_policy=spec.axes_policy)
end

function feature_names(spec::EulerSurfaceSpec)
    spec.axes === nothing && return Symbol[]
    nf = prod(length(ax) for ax in spec.axes)
    return [Symbol("euler__f$(i)") for i in 1:nf]
end

nfeatures(spec::EulerSurfaceSpec) = length(feature_names(spec))

feature_axes(spec::RankGridSpec) =
    (a=collect(1:spec.nvertices), b=collect(1:spec.nvertices), store_zeros=spec.store_zeros)

feature_names(spec::RankGridSpec) = [Symbol("rank__a$(a)_b$(b)") for a in 1:spec.nvertices for b in 1:spec.nvertices]
nfeatures(spec::RankGridSpec) = length(feature_names(spec))

feature_axes(spec::RestrictedHilbertSpec) = (vertex=collect(1:spec.nvertices),)

feature_names(spec::RestrictedHilbertSpec) =
    [Symbol("restricted_hilbert__p$(p)") for p in 1:spec.nvertices]

nfeatures(spec::RestrictedHilbertSpec) = length(feature_names(spec))

@inline function _barcode_topk_field_labels(k::Int)
    out = Symbol[:count]
    for i in 1:k
        push!(out, Symbol("present_$(i)"))
        push!(out, Symbol("birth_$(i)"))
        push!(out, Symbol("persistence_$(i)"))
    end
    return out
end

function feature_axes(spec::BarcodeTopKSpec)
    fields = _barcode_topk_field_labels(spec.k)
    if spec.aggregate == :stack
        return (direction=collect(1:length(spec.directions)),
                offset=collect(1:length(spec.offsets)),
                field=fields,
                source=spec.source)
    end
    return (aggregate=spec.aggregate,
            field=fields,
            source=spec.source,
            directions=_copy_vecvec(spec.directions),
            offsets=_copy_vecvec(spec.offsets))
end

function feature_names(spec::BarcodeTopKSpec)
    base = _barcode_topk_field_labels(spec.k)
    if spec.aggregate == :stack
        out = Vector{Symbol}(undef, length(spec.directions) * length(spec.offsets) * length(base))
        idx = 1
        for i in 1:length(spec.directions), j in 1:length(spec.offsets), fld in base
            out[idx] = Symbol("barcode_topk__d$(i)_o$(j)__" * String(fld))
            idx += 1
        end
        return out
    end
    return [Symbol("barcode_topk__" * String(fld)) for fld in base]
end

nfeatures(spec::BarcodeTopKSpec) = length(feature_names(spec))

@inline function _sliced_stat_axis(spec::SlicedBarcodeSpec)
    if spec.featurizer == :summary
        return Tuple(Symbol(s) for s in spec.summary_fields)
    elseif spec.featurizer == :entropy
        return (:entropy,)
    end
    return ()
end

function feature_axes(spec::SlicedBarcodeSpec)
    stats = _sliced_stat_axis(spec)
    if spec.aggregate == :stack
        return (direction=collect(1:length(spec.directions)),
                offset=collect(1:length(spec.offsets)),
                stat=stats)
    end
    return (aggregate=spec.aggregate,
            stat=stats,
            directions=_copy_vecvec(spec.directions),
            offsets=_copy_vecvec(spec.offsets))
end

function feature_names(spec::SlicedBarcodeSpec)
    base = if spec.featurizer == :summary
        length(spec.summary_fields)
    elseif spec.featurizer == :entropy
        1
    else
        throw(ArgumentError("SlicedBarcodeSpec supports featurizer=:summary or :entropy"))
    end
    return _slice_feature_names("sliced_barcode", base,
                                length(spec.directions), length(spec.offsets), spec.aggregate)
end

nfeatures(spec::SlicedBarcodeSpec) = length(feature_names(spec))

function feature_axes(spec::BarcodeSummarySpec)
    if spec.aggregate == :stack
        return (direction=collect(1:length(spec.directions)),
                offset=collect(1:length(spec.offsets)),
                field=collect(spec.fields),
                source=spec.source)
    end
    return (aggregate=spec.aggregate,
            field=collect(spec.fields),
            source=spec.source,
            directions=_copy_vecvec(spec.directions),
            offsets=_copy_vecvec(spec.offsets))
end

function feature_names(spec::BarcodeSummarySpec)
    base = [Symbol("barcode_summary__" * String(fld)) for fld in spec.fields]
    if spec.aggregate == :stack
        out = Vector{Symbol}(undef, length(spec.directions) * length(spec.offsets) * length(base))
        idx = 1
        for i in 1:length(spec.directions), j in 1:length(spec.offsets), fld in base
            out[idx] = Symbol(String(fld) * "__d$(i)_o$(j)")
            idx += 1
        end
        return out
    end
    return base
end

nfeatures(spec::BarcodeSummarySpec) = length(feature_names(spec))

@inline function _point_measure_slot_fields(ndims::Int)
    out = Symbol[]
    for i in 1:ndims
        push!(out, Symbol("coord_$(i)"))
    end
    return out
end

function feature_axes(spec::PointSignedMeasureSpec)
    return (term=collect(1:spec.k), field=(:present, :weight, _point_measure_slot_fields(spec.ndims)...), coords=spec.coords)
end

function feature_names(spec::PointSignedMeasureSpec)
    out = Symbol[:point_signed_measure__count]
    for t in 1:spec.k
        push!(out, Symbol("point_signed_measure__term$(t)__present"))
        push!(out, Symbol("point_signed_measure__term$(t)__weight"))
        for i in 1:spec.ndims
            push!(out, Symbol("point_signed_measure__term$(t)__coord$(i)"))
        end
    end
    return out
end

nfeatures(spec::PointSignedMeasureSpec) = length(feature_names(spec))

function feature_axes(spec::EulerSignedMeasureSpec)
    out = (term=collect(1:spec.k), field=(:present, :weight, _point_measure_slot_fields(spec.ndims)...), coords=spec.coords)
    spec.axes === nothing && return (out..., axes_policy=spec.axes_policy, max_axis_len=spec.max_axis_len)
    return (out..., _axis_namedtuple(spec.axes, "axis")..., axes_policy=spec.axes_policy, max_axis_len=spec.max_axis_len)
end

function feature_names(spec::EulerSignedMeasureSpec)
    out = Symbol[:euler_signed_measure__count]
    for t in 1:spec.k
        push!(out, Symbol("euler_signed_measure__term$(t)__present"))
        push!(out, Symbol("euler_signed_measure__term$(t)__weight"))
        for i in 1:spec.ndims
            push!(out, Symbol("euler_signed_measure__term$(t)__coord$(i)"))
        end
    end
    return out
end

nfeatures(spec::EulerSignedMeasureSpec) = length(feature_names(spec))

function feature_axes(spec::RectangleSignedBarcodeTopKSpec)
    out = (term=collect(1:spec.k), field=(:present, :weight, :lo, :hi), coords=spec.coords, ndims=spec.ndims)
    spec.axes === nothing && return (out..., axes_policy=spec.axes_policy, max_axis_len=spec.max_axis_len)
    return (out..., _axis_namedtuple(spec.axes, "axis")..., axes_policy=spec.axes_policy, max_axis_len=spec.max_axis_len)
end

function feature_names(spec::RectangleSignedBarcodeTopKSpec)
    out = Symbol[:rectangle_signed_barcode__count]
    for t in 1:spec.k
        push!(out, Symbol("rectangle_signed_barcode__term$(t)__present"))
        push!(out, Symbol("rectangle_signed_barcode__term$(t)__weight"))
        for i in 1:spec.ndims
            push!(out, Symbol("rectangle_signed_barcode__term$(t)__lo$(i)"))
        end
        for i in 1:spec.ndims
            push!(out, Symbol("rectangle_signed_barcode__term$(t)__hi$(i)"))
        end
    end
    return out
end

nfeatures(spec::RectangleSignedBarcodeTopKSpec) = length(feature_names(spec))

function feature_axes(spec::SignedBarcodeImageSpec)
    out = (x=copy(spec.xs), y=copy(spec.ys), mode=spec.mode, method=spec.method)
    spec.axes === nothing && return out
    return (out..., _axis_namedtuple(spec.axes, "source_axis")..., axes_policy=spec.axes_policy)
end

feature_names(spec::SignedBarcodeImageSpec) =
    [Symbol("signed_barcode_image__x$(i)_y$(j)") for i in 1:length(spec.xs) for j in 1:length(spec.ys)]

nfeatures(spec::SignedBarcodeImageSpec) = length(feature_names(spec))

function feature_axes(spec::ProjectedDistancesSpec)
    out = (reference=copy(spec.reference_names), dist=spec.dist, agg=spec.agg)
    if spec.directions === nothing
        return out
    end
    return (out..., direction=collect(1:length(spec.directions)))
end

feature_names(spec::ProjectedDistancesSpec) = copy(spec.reference_names)
nfeatures(spec::ProjectedDistancesSpec) = length(spec.reference_names)

function feature_axes(spec::MatchingDistanceBankSpec)
    out = (reference=copy(spec.reference_names), method=spec.method, normalize_dirs=spec.normalize_dirs, weight=spec.weight)
    if spec.directions === nothing
        return (out..., n_dirs=spec.n_dirs, n_offsets=spec.n_offsets, max_den=spec.max_den, include_axes=spec.include_axes)
    end
    return (out..., directions=_copy_vecvec(spec.directions), offsets=spec.offsets === nothing ? nothing : _copy_vecvec(spec.offsets))
end

feature_names(spec::MatchingDistanceBankSpec) = copy(spec.reference_names)
nfeatures(spec::MatchingDistanceBankSpec) = length(spec.reference_names)

@inline _grid_feature_length(grid::Union{Nothing,Vector{Float64}}, resolution::Int) =
    grid === nothing ? resolution : length(grid)

function feature_axes(spec::MPPImageSpec)
    xax = spec.xgrid === nothing ? collect(1:spec.resolution) : copy(spec.xgrid)
    yax = spec.ygrid === nothing ? collect(1:spec.resolution) : copy(spec.ygrid)
    return (
        x=xax,
        y=yax,
        sigma=spec.sigma,
        resolution=spec.resolution,
        N=spec.N,
        delta=spec.delta,
        q=spec.q,
        tie_break=spec.tie_break,
    )
end

function feature_names(spec::MPPImageSpec)
    nx = _grid_feature_length(spec.xgrid, spec.resolution)
    ny = _grid_feature_length(spec.ygrid, spec.resolution)
    return [Symbol("mpp_image__x$(ix)_y$(iy)") for ix in 1:nx for iy in 1:ny]
end

nfeatures(spec::MPPImageSpec) = length(feature_names(spec))

function feature_axes(spec::MPPDecompositionHistogramSpec)
    return (
        orientation_bin=collect(1:spec.orientation_bins),
        scale_bin=collect(1:spec.scale_bins),
        weight=spec.weight,
        normalize=spec.normalize,
        scale_range=spec.scale_range,
        scale_normalization=:box_diagonal,
        scalars=(:total_mass, :n_summands, :n_segments, :max_scale, :entropy),
        N=spec.N,
        delta=spec.delta,
        q=spec.q,
        tie_break=spec.tie_break,
    )
end

function feature_names(spec::MPPDecompositionHistogramSpec)
    out = Vector{Symbol}(undef, spec.orientation_bins * spec.scale_bins + 5)
    idx = 1
    for si in 1:spec.scale_bins, oi in 1:spec.orientation_bins
        out[idx] = Symbol("mpp_decomposition_hist__theta$(oi)_scale$(si)")
        idx += 1
    end
    out[idx] = :mpp_decomposition_hist__total_mass
    out[idx + 1] = :mpp_decomposition_hist__n_summands
    out[idx + 2] = :mpp_decomposition_hist__n_segments
    out[idx + 3] = :mpp_decomposition_hist__max_scale
    out[idx + 4] = :mpp_decomposition_hist__entropy
    return out
end

nfeatures(spec::MPPDecompositionHistogramSpec) = length(feature_names(spec))

feature_axes(spec::BettiTableSpec) =
    (degree=collect(0:spec.pad_to), vertex=collect(1:spec.nvertices), resolution_maxlen=spec.resolution_maxlen)

feature_names(spec::BettiTableSpec) =
    [Symbol("betti__a$(a)_p$(p)") for a in 0:spec.pad_to for p in 1:spec.nvertices]

nfeatures(spec::BettiTableSpec) = length(feature_names(spec))

feature_axes(spec::BassTableSpec) =
    (degree=collect(0:spec.pad_to), vertex=collect(1:spec.nvertices), resolution_maxlen=spec.resolution_maxlen)

feature_names(spec::BassTableSpec) =
    [Symbol("bass__b$(b)_p$(p)") for b in 0:spec.pad_to for p in 1:spec.nvertices]

nfeatures(spec::BassTableSpec) = length(feature_names(spec))

feature_axes(spec::BettiSupportMeasuresSpec) =
    (degree=collect(0:spec.pad_to),
     quantity=(:support, :mass),
     summary=(:support_union, :support_total, :mass_total),
     resolution_maxlen=spec.resolution_maxlen)

function feature_names(spec::BettiSupportMeasuresSpec)
    out = Symbol[]
    append!(out, Symbol("betti_support__support_a$(a)") for a in 0:spec.pad_to)
    append!(out, Symbol("betti_support__mass_a$(a)") for a in 0:spec.pad_to)
    append!(out, (:betti_support__support_union, :betti_support__support_total, :betti_support__mass_total))
    return out
end

nfeatures(spec::BettiSupportMeasuresSpec) = length(feature_names(spec))

feature_axes(spec::BassSupportMeasuresSpec) =
    (degree=collect(0:spec.pad_to),
     quantity=(:support, :mass),
     summary=(:support_union, :support_total, :mass_total),
     resolution_maxlen=spec.resolution_maxlen)

function feature_names(spec::BassSupportMeasuresSpec)
    out = Symbol[]
    append!(out, Symbol("bass_support__support_b$(b)") for b in 0:spec.pad_to)
    append!(out, Symbol("bass_support__mass_b$(b)") for b in 0:spec.pad_to)
    append!(out, (:bass_support__support_union, :bass_support__support_total, :bass_support__mass_total))
    return out
end

nfeatures(spec::BassSupportMeasuresSpec) = length(feature_names(spec))

function feature_axes(spec::CompositeSpec)
    comps = Vector{Any}(undef, length(spec.specs))
    start_idx = 1
    @inbounds for i in eachindex(spec.specs)
        sub = spec.specs[i]
        nsub = nfeatures(sub)
        stop_idx = start_idx + nsub - 1
        comps[i] = (
            name = spec.namespacing ? string(nameof(typeof(sub))) : string("component_", i),
            spec_type = string(nameof(typeof(sub))),
            start = start_idx,
            stop = stop_idx,
            axes = feature_axes(sub),
        )
        start_idx = stop_idx + 1
    end
    return (namespacing=spec.namespacing, components=comps)
end

function feature_names(spec::CompositeSpec)
    out = Symbol[]
    for (i, sub) in enumerate(spec.specs)
        names = feature_names(sub)
        if spec.namespacing
            prefix = string(nameof(typeof(sub)))
            append!(out, Symbol(prefix * "__" * String(n)) for n in names)
        else
            append!(out, names)
        end
    end
    return out
end

nfeatures(spec::CompositeSpec) = sum(nfeatures(s) for s in spec.specs)

@inline function _sample_module(obj)
    if obj isa EncodingResult
        return materialize_module(obj.M)
    elseif obj isa PModule
        return obj
    elseif obj isa Tuple && length(obj) >= 1 && obj[1] isa PModule
        return obj[1]
    elseif obj isa NamedTuple && hasproperty(obj, :M) && getproperty(obj, :M) isa PModule
        return getproperty(obj, :M)
    end
    throw(ArgumentError("sample must provide a PModule (EncodingResult, PModule, or (M,pi))"))
end

@inline function _sample_module_pi(obj)
    if obj isa EncodingResult
        return materialize_module(obj.M), obj.pi
    elseif obj isa Tuple && length(obj) >= 2 && obj[1] isa PModule
        return obj[1], obj[2]
    elseif obj isa NamedTuple && hasproperty(obj, :M) && hasproperty(obj, :pi)
        M = getproperty(obj, :M)
        M isa PModule || throw(ArgumentError("NamedTuple sample field :M must be PModule"))
        return M, getproperty(obj, :pi)
    end
    throw(ArgumentError("sample must provide both module and encoding map (EncodingResult or (M,pi))"))
end

@inline _supports_projective_resolution(obj) =
    (obj isa ResolutionResult && obj.res isa DerivedFunctors.ProjectiveResolution) ||
    (obj isa DerivedFunctors.ProjectiveResolution)

@inline _supports_injective_resolution(obj) =
    (obj isa ResolutionResult && obj.res isa DerivedFunctors.InjectiveResolution) ||
    (obj isa DerivedFunctors.InjectiveResolution)

@inline function _sample_projective_resolution(obj)
    if obj isa ResolutionResult
        obj.res isa DerivedFunctors.ProjectiveResolution ||
            throw(ArgumentError("sample ResolutionResult does not store a projective resolution"))
        return obj.res
    elseif obj isa DerivedFunctors.ProjectiveResolution
        return obj
    end
    throw(ArgumentError("sample must provide a projective resolution or ResolutionResult"))
end

@inline function _sample_injective_resolution(obj)
    if obj isa ResolutionResult
        obj.res isa DerivedFunctors.InjectiveResolution ||
            throw(ArgumentError("sample ResolutionResult does not store an injective resolution"))
        return obj.res
    elseif obj isa DerivedFunctors.InjectiveResolution
        return obj
    end
    throw(ArgumentError("sample must provide an injective resolution or ResolutionResult"))
end

@inline _supports_projective_resolution_pi(obj) =
    (obj isa ResolutionResult &&
     obj.res isa DerivedFunctors.ProjectiveResolution &&
     obj.enc !== nothing &&
     hasproperty(obj.enc, :pi)) ||
    (obj isa Tuple && length(obj) >= 2 && obj[1] isa DerivedFunctors.ProjectiveResolution) ||
    (obj isa NamedTuple && hasproperty(obj, :res) && hasproperty(obj, :pi) &&
     getproperty(obj, :res) isa DerivedFunctors.ProjectiveResolution)

@inline _supports_injective_resolution_pi(obj) =
    (obj isa ResolutionResult &&
     obj.res isa DerivedFunctors.InjectiveResolution &&
     obj.enc !== nothing &&
     hasproperty(obj.enc, :pi)) ||
    (obj isa Tuple && length(obj) >= 2 && obj[1] isa DerivedFunctors.InjectiveResolution) ||
    (obj isa NamedTuple && hasproperty(obj, :res) && hasproperty(obj, :pi) &&
     getproperty(obj, :res) isa DerivedFunctors.InjectiveResolution)

@inline function _sample_projective_resolution_pi(obj)
    if obj isa ResolutionResult
        obj.res isa DerivedFunctors.ProjectiveResolution ||
            throw(ArgumentError("sample ResolutionResult does not store a projective resolution"))
        obj.enc !== nothing && hasproperty(obj.enc, :pi) ||
            throw(ArgumentError("sample ResolutionResult must carry provenance with a pi field"))
        return obj.res, getproperty(obj.enc, :pi)
    elseif obj isa Tuple && length(obj) >= 2 && obj[1] isa DerivedFunctors.ProjectiveResolution
        return obj[1], obj[2]
    elseif obj isa NamedTuple && hasproperty(obj, :res) && hasproperty(obj, :pi)
        res = getproperty(obj, :res)
        res isa DerivedFunctors.ProjectiveResolution ||
            throw(ArgumentError("sample NamedTuple field :res must be a projective resolution"))
        return res, getproperty(obj, :pi)
    end
    throw(ArgumentError("sample must provide a projective resolution together with pi"))
end

@inline function _sample_injective_resolution_pi(obj)
    if obj isa ResolutionResult
        obj.res isa DerivedFunctors.InjectiveResolution ||
            throw(ArgumentError("sample ResolutionResult does not store an injective resolution"))
        obj.enc !== nothing && hasproperty(obj.enc, :pi) ||
            throw(ArgumentError("sample ResolutionResult must carry provenance with a pi field"))
        return obj.res, getproperty(obj.enc, :pi)
    elseif obj isa Tuple && length(obj) >= 2 && obj[1] isa DerivedFunctors.InjectiveResolution
        return obj[1], obj[2]
    elseif obj isa NamedTuple && hasproperty(obj, :res) && hasproperty(obj, :pi)
        res = getproperty(obj, :res)
        res isa DerivedFunctors.InjectiveResolution ||
            throw(ArgumentError("sample NamedTuple field :res must be an injective resolution"))
        return res, getproperty(obj, :pi)
    end
    throw(ArgumentError("sample must provide an injective resolution together with pi"))
end

@inline _supports_module(obj) = (obj isa EncodingResult) || (obj isa PModule) ||
    (obj isa Tuple && length(obj) >= 1 && obj[1] isa PModule) ||
    (obj isa NamedTuple && hasproperty(obj, :M) && getproperty(obj, :M) isa PModule)

@inline _supports_module_pi(obj) = (obj isa EncodingResult) ||
    (obj isa Tuple && length(obj) >= 2 && obj[1] isa PModule) ||
    (obj isa NamedTuple && hasproperty(obj, :M) && hasproperty(obj, :pi) && getproperty(obj, :M) isa PModule)

"""
    supports(spec, sample) -> Bool

Return whether a featurizer spec supports a sample without throwing.
"""
supports(::LandscapeSpec, obj) = _supports_module_pi(obj)
supports(::PersistenceImageSpec, obj) = _supports_module_pi(obj)
supports(::MPLandscapeSpec, obj) = _supports_module_pi(obj)
supports(::EulerSurfaceSpec, obj) = _supports_module_pi(obj)
supports(::RankGridSpec, obj) = _supports_module(obj)
supports(::RestrictedHilbertSpec, obj) = _supports_module(obj)
supports(spec::PointSignedMeasureSpec, obj) =
    obj isa PointSignedMeasure && length(obj.axes) == spec.ndims
supports(spec::EulerSignedMeasureSpec, obj) =
    _supports_module_pi(obj) && begin
        _, pi = _sample_module_pi(obj)
        pi0 = _unwrap_cache_pi(pi)
        !hasproperty(pi0, :n) || Int(getproperty(pi0, :n)) == spec.ndims
    end
supports(spec::BarcodeTopKSpec, obj) =
    spec.source == :fibered ? (_supports_module_pi(obj) && _supports_fibered_cache(last(_sample_module_pi(obj)))) :
                              _supports_module_pi(obj)
supports(::SlicedBarcodeSpec, obj) = _supports_module_pi(obj)
supports(spec::BarcodeSummarySpec, obj) =
    spec.source == :fibered ? (_supports_module_pi(obj) && _supports_fibered_cache(last(_sample_module_pi(obj)))) :
                              _supports_module_pi(obj)
supports(spec::RectangleSignedBarcodeTopKSpec, obj) =
    (obj isa RectSignedBarcode && length(obj.axes) == spec.ndims) ||
    (_supports_module_pi(obj) && begin
        _, pi = _sample_module_pi(obj)
        pi0 = _unwrap_cache_pi(pi)
        pi0 isa ZnEncoding.ZnEncodingMap && pi0.n == spec.ndims
    end)
supports(::BettiTableSpec, obj) = _supports_projective_resolution(obj) || _supports_module(obj)
supports(::BassTableSpec, obj) = _supports_injective_resolution(obj) || _supports_module(obj)
supports(::BettiSupportMeasuresSpec, obj) = _supports_projective_resolution_pi(obj) || _supports_module_pi(obj)
supports(::BassSupportMeasuresSpec, obj) = _supports_injective_resolution_pi(obj) || _supports_module_pi(obj)

function supports(::SignedBarcodeImageSpec, obj)
    _supports_module_pi(obj) || return false
    _, pi = _sample_module_pi(obj)
    pi0 = pi isa CompiledEncoding ? pi.pi : pi
    return pi0 isa ZnEncoding.ZnEncodingMap
end

function supports(spec::ProjectedDistancesSpec, obj)
    _supports_module_pi(obj) || return false
    @inbounds for r in spec.references
        _supports_module_pi(r) || return false
    end
    return true
end

function supports(spec::MatchingDistanceBankSpec, obj)
    _supports_module_pi(obj) || return false
    if spec.method == :exact_2d
        _, pi = _sample_module_pi(obj)
        _supports_fibered_cache(pi) || return false
    end
    @inbounds for r in spec.references
        _supports_module_pi(r) || return false
        if spec.method == :exact_2d
            _, pi = _sample_module_pi(r)
            _supports_fibered_cache(pi) || return false
        end
    end
    return true
end

function supports(::MPPImageSpec, obj)
    _supports_module_pi(obj) || return false
    _, pi = _sample_module_pi(obj)
    return _supports_fibered_cache(pi)
end

function supports(::MPPDecompositionHistogramSpec, obj)
    _supports_module_pi(obj) || return false
    _, pi = _sample_module_pi(obj)
    return _supports_fibered_cache(pi)
end

@inline _is_projective_resolution_spec(spec) =
    spec isa Union{BettiTableSpec,BettiSupportMeasuresSpec}

@inline _is_injective_resolution_spec(spec) =
    spec isa Union{BassTableSpec,BassSupportMeasuresSpec}

@inline _is_resolution_spec(spec) =
    _is_projective_resolution_spec(spec) || _is_injective_resolution_spec(spec)

@inline _needs_resolution_pi(spec) =
    spec isa Union{BettiSupportMeasuresSpec,BassSupportMeasuresSpec}

@inline function _composite_resolution_family(spec::CompositeSpec)
    has_proj = any(_is_projective_resolution_spec, spec.specs)
    has_inj = any(_is_injective_resolution_spec, spec.specs)
    if has_proj && has_inj
        return :mixed
    elseif has_proj
        return :projective
    elseif has_inj
        return :injective
    end
    return :none
end

@inline function _composite_pure_resolution_family(spec::CompositeSpec)
    fam = _composite_resolution_family(spec)
    fam == :mixed && return :mixed
    fam == :none && return :none
    all(_is_resolution_spec, spec.specs) || return :none
    return fam
end

function supports(spec::CompositeSpec, obj)
    fam = _composite_resolution_family(spec)
    if fam == :mixed
        return false
    end
    @inbounds for s in spec.specs
        supports(s, obj) || return false
    end
    return true
end

@inline _cache_sample(cache::ModuleInvariantCache) = cache.M
@inline _cache_sample(cache::RestrictedHilbertInvariantCache) = (cache.M, cache.pi)
@inline _cache_sample(cache::ResolutionInvariantCache) = cache.res
@inline _cache_sample(cache::ResolutionMeasureInvariantCache) = (cache.res, cache.pi)
@inline _cache_sample(cache::EncodingInvariantCache) = (cache.M, cache.pi)
@inline _cache_sample(cache::PointMeasureInvariantCache) = cache.pm
@inline _cache_sample(cache::SignedBarcodeInvariantCache) = cache.sb

@inline function _default_invariant_opts(opts::InvariantOptions)
    return opts.axes === nothing &&
           opts.axes_policy === :encoding &&
           opts.max_axis_len == 256 &&
           opts.box === nothing &&
           opts.threads === nothing &&
           opts.strict === nothing &&
           opts.pl_mode === :fast
end

@inline _cache_opts(cache::AbstractInvariantCache, opts::InvariantOptions) =
    (_default_invariant_opts(opts) ? cache.opts : opts)
@inline _cache_opts(cache::AbstractInvariantCache, ::Nothing) = cache.opts
@inline _cache_opts(cache::ResolutionInvariantCache, opts::InvariantOptions) = opts
@inline _cache_opts(cache::ResolutionInvariantCache, ::Nothing) = InvariantOptions()
@inline _cache_opts(cache::ResolutionMeasureInvariantCache, opts::InvariantOptions) =
    (_default_invariant_opts(opts) ? cache.opts : opts)
@inline _cache_opts(cache::ResolutionMeasureInvariantCache, ::Nothing) = cache.opts

@inline function _cache_threaded(cache::AbstractInvariantCache, threaded::Bool)
    return cache.threads ? threaded : false
end

@inline function _validate_cache_level(level::Symbol)
    level in (:auto, :none, :slice, :projected, :fibered, :all) ||
        throw(ArgumentError("cache level must be one of :auto, :none, :slice, :projected, :fibered, :all"))
    return level
end

@inline _unwrap_cache_pi(pi) = pi isa CompiledEncoding ? pi.pi : pi

@inline function _supports_fibered_cache(pi)
    pi0 = _unwrap_cache_pi(pi)
    return pi0 isa PLikeEncodingMap &&
           hasproperty(pi0, :n) &&
           Int(getproperty(pi0, :n)) == 2
end

@inline function _effective_cache_level(level::Symbol, spec::AbstractFeaturizerSpec, pi=nothing)
    level == :auto || return level
    if spec isa ProjectedDistancesSpec
        return :projected
    elseif spec isa MPPImageSpec
        return :fibered
    elseif spec isa SignedBarcodeImageSpec
        return :all
    elseif spec isa Union{LandscapeSpec,PersistenceImageSpec,MPLandscapeSpec,SlicedBarcodeSpec}
        # Keep semantics stable by default; fibered path is opt-in via level=:fibered/:all.
        return :slice
    elseif spec isa CompositeSpec
        return :all
    end
    return :none
end

@inline function _effective_cache_level(level::Symbol,
                                        spec::Union{BarcodeTopKSpec,BarcodeSummarySpec},
                                        pi=nothing)
    if spec.source == :fibered
        return level == :all ? :all : :fibered
    end
    return level == :none ? :none : :slice
end

@inline function _effective_cache_level(level::Symbol,
                                        ::MPPDecompositionHistogramSpec,
                                        pi=nothing)
    return level == :auto ? :fibered : level
end

@inline function _effective_cache_level(level::Symbol,
                                        spec::MatchingDistanceBankSpec,
                                        pi=nothing)
    if level == :auto
        return spec.method == :exact_2d ? :fibered : :slice
    end
    return level
end

@inline function _effective_cache_level(level::Symbol,
                                        ::RectangleSignedBarcodeTopKSpec,
                                        pi=nothing)
    return level == :auto ? :all : level
end

function build_cache(M::PModule{K,F,MatT};
                     opts::InvariantOptions=InvariantOptions(),
                     level::Symbol=:auto,
                     threaded::Bool=true) where {K,F<:AbstractCoeffField,MatT<:AbstractMatrix{K}}
    _validate_cache_level(level)
    opts0 = opts
    return ModuleInvariantCache{K,F,MatT}(M, opts0, threaded, level)
end

function build_cache(M::PModule{K,F,MatT}, pi;
                     opts::InvariantOptions=InvariantOptions(),
                     level::Symbol=:auto,
                     threaded::Bool=true,
                     cache=:auto) where {K,F<:AbstractCoeffField,MatT<:AbstractMatrix{K}}
    _validate_cache_level(level)
    opts0 = opts
    session_cache = _resolve_workflow_session_cache(cache)
    spcache = _slice_plan_cache_from_session(nothing, session_cache)
    if spcache === nothing
        spcache = SlicePlanCache()
    end
    return EncodingInvariantCache{K,F,MatT,typeof(pi)}(
        M,
        pi,
        opts0,
        threaded,
        level,
        session_cache,
        spcache,
        Dict{UInt,CompiledSlicePlan}(),
        Dict{UInt,ProjectedArrangement}(),
        Dict{UInt,ProjectedBarcodeCache{K}}(),
        Dict{Tuple{UInt,UInt},ProjectedBarcodeCache}(),
        Dict{UInt,FiberedBarcodeCache2D{K}}(),
        Dict{UInt,MPPDecomposition}(),
        nothing,
    )
end

function build_restricted_hilbert_cache(
    M::PModule{K,F,MatT},
    pi;
    opts::InvariantOptions=InvariantOptions(),
    level::Symbol=:auto,
    threaded::Bool=true,
) where {K,F<:AbstractCoeffField,MatT<:AbstractMatrix{K}}
    _validate_cache_level(level)
    opts0 = opts
    h = Invariants.restricted_hilbert(M)
    return RestrictedHilbertInvariantCache{K,F,MatT,typeof(pi)}(
        M, pi, opts0, threaded, level, h
    )
end

build_cache(enc::EncodingResult; kwargs...) = build_cache(materialize_module(enc.M), enc.pi; kwargs...)

function build_cache(obj;
                     opts::InvariantOptions=InvariantOptions(),
                     level::Symbol=:auto,
                     threaded::Bool=true,
                     cache=:auto)
    _validate_cache_level(level)
    if _supports_module_pi(obj)
        M, pi = _sample_module_pi(obj)
        return build_cache(M, pi; opts=opts, level=level, threaded=threaded, cache=cache)
    elseif _supports_module(obj)
        M = _sample_module(obj)
        return build_cache(M; opts=opts, level=level, threaded=threaded)
    end
    throw(ArgumentError("build_cache: sample must provide a PModule or (PModule, pi)"))
end

build_cache(M::PModule, pi, opts::InvariantOptions; kwargs...) =
    build_cache(M, pi; opts=opts, kwargs...)

build_cache(obj, opts::InvariantOptions; kwargs...) =
    build_cache(obj; opts=opts, kwargs...)

@inline _resolution_options(spec::Union{BettiTableSpec,BettiSupportMeasuresSpec}) =
    ResolutionOptions(maxlen=spec.resolution_maxlen, minimal=spec.minimal, check=spec.check)

@inline _resolution_options(spec::Union{BassTableSpec,BassSupportMeasuresSpec}) =
    ResolutionOptions(maxlen=spec.resolution_maxlen, minimal=spec.minimal, check=spec.check)

function _build_resolution_cache(obj,
                                 spec::BettiTableSpec;
                                 level::Symbol=:auto,
                                 threaded::Bool=true,
                                 cache=:auto)
    _validate_cache_level(level)
    res = if _supports_projective_resolution(obj)
        _sample_projective_resolution(obj)
    else
        M = _sample_module(obj)
        session_cache = _resolve_workflow_session_cache(cache)
        rcache = _resolution_cache_from_session(nothing, session_cache, M)
        DerivedFunctors.projective_resolution(M, _resolution_options(spec); cache=rcache)
    end
    return ResolutionInvariantCache{typeof(res)}(res, threaded, level, :projective)
end

function _build_resolution_cache(obj,
                                 spec::BassTableSpec;
                                 level::Symbol=:auto,
                                 threaded::Bool=true,
                                 cache=:auto)
    _validate_cache_level(level)
    res = if _supports_injective_resolution(obj)
        _sample_injective_resolution(obj)
    else
        M = _sample_module(obj)
        session_cache = _resolve_workflow_session_cache(cache)
        rcache = _resolution_cache_from_session(nothing, session_cache, M)
        DerivedFunctors.injective_resolution(M, _resolution_options(spec); cache=rcache)
    end
    return ResolutionInvariantCache{typeof(res)}(res, threaded, level, :injective)
end

function _build_resolution_cache(obj,
                                 spec::CompositeSpec;
                                 level::Symbol=:auto,
                                 threaded::Bool=true,
                                 cache=:auto)
    fam = _composite_pure_resolution_family(spec)
    if fam == :projective
        maxlen = maximum(s.resolution_maxlen for s in spec.specs)
        minimal = any(s.minimal for s in spec.specs)
        check = any(s.check for s in spec.specs)
        nverts = maximum(s.nvertices for s in spec.specs)
        agg = BettiTableSpec(nverts, 0, maxlen, minimal, check)
        return _build_resolution_cache(obj, agg; level=level, threaded=threaded, cache=cache)
    elseif fam == :injective
        maxlen = maximum(s.resolution_maxlen for s in spec.specs)
        minimal = any(s.minimal for s in spec.specs)
        check = any(s.check for s in spec.specs)
        nverts = maximum(s.nvertices for s in spec.specs)
        agg = BassTableSpec(nverts, 0, maxlen, minimal, check)
        return _build_resolution_cache(obj, agg; level=level, threaded=threaded, cache=cache)
    end
    throw(ArgumentError("resolution cache requires a homogeneous BettiTableSpec or BassTableSpec composite"))
end

@inline function _support_measure_weights(pi, opts0::InvariantOptions)
    opts0.box === nothing &&
        throw(ArgumentError("support-measure featurizers require opts.box (or opts.box=:auto)"))
    pi0 = _unwrap_cache_pi(pi)
    if opts0.strict === nothing
        return Invariants.region_weights(pi0; box=opts0.box)
    end
    return Invariants.region_weights(pi0; box=opts0.box, strict=opts0.strict)
end

function _build_resolution_measure_cache(obj,
                                         spec::BettiSupportMeasuresSpec;
                                         opts::InvariantOptions=InvariantOptions(),
                                         level::Symbol=:auto,
                                         threaded::Bool=true,
                                         cache=:auto)
    _validate_cache_level(level)
    opts0 = opts
    if _supports_projective_resolution_pi(obj)
        res, pi = _sample_projective_resolution_pi(obj)
        weights = _support_measure_weights(pi, opts0)
        return ResolutionMeasureInvariantCache{typeof(res),typeof(pi),typeof(weights)}(
            res, pi, opts0, threaded, level, :projective, weights
        )
    end
    M, pi = _sample_module_pi(obj)
    session_cache = _resolve_workflow_session_cache(cache)
    rcache = _resolution_cache_from_session(nothing, session_cache, M)
    res = DerivedFunctors.projective_resolution(M, _resolution_options(spec); cache=rcache)
    weights = _support_measure_weights(pi, opts0)
    return ResolutionMeasureInvariantCache{typeof(res),typeof(pi),typeof(weights)}(
        res, pi, opts0, threaded, level, :projective, weights
    )
end

function _build_resolution_measure_cache(obj,
                                         spec::BassSupportMeasuresSpec;
                                         opts::InvariantOptions=InvariantOptions(),
                                         level::Symbol=:auto,
                                         threaded::Bool=true,
                                         cache=:auto)
    _validate_cache_level(level)
    opts0 = opts
    if _supports_injective_resolution_pi(obj)
        res, pi = _sample_injective_resolution_pi(obj)
        weights = _support_measure_weights(pi, opts0)
        return ResolutionMeasureInvariantCache{typeof(res),typeof(pi),typeof(weights)}(
            res, pi, opts0, threaded, level, :injective, weights
        )
    end
    M, pi = _sample_module_pi(obj)
    session_cache = _resolve_workflow_session_cache(cache)
    rcache = _resolution_cache_from_session(nothing, session_cache, M)
    res = DerivedFunctors.injective_resolution(M, _resolution_options(spec); cache=rcache)
    weights = _support_measure_weights(pi, opts0)
    return ResolutionMeasureInvariantCache{typeof(res),typeof(pi),typeof(weights)}(
        res, pi, opts0, threaded, level, :injective, weights
    )
end

function _build_resolution_measure_cache(obj,
                                         spec::CompositeSpec;
                                         opts::InvariantOptions=InvariantOptions(),
                                         level::Symbol=:auto,
                                         threaded::Bool=true,
                                         cache=:auto)
    fam = _composite_pure_resolution_family(spec)
    fam == :projective || fam == :injective ||
        throw(ArgumentError("resolution-measure cache requires a homogeneous projective or injective composite"))
    if fam == :projective
        maxlen = maximum(s.resolution_maxlen for s in spec.specs)
        minimal = any(s.minimal for s in spec.specs)
        check = any(s.check for s in spec.specs)
        agg = BettiSupportMeasuresSpec(0, maxlen, minimal, check)
        return _build_resolution_measure_cache(obj, agg; opts=opts, level=level, threaded=threaded, cache=cache)
    end
    maxlen = maximum(s.resolution_maxlen for s in spec.specs)
    minimal = any(s.minimal for s in spec.specs)
    check = any(s.check for s in spec.specs)
    agg = BassSupportMeasuresSpec(0, maxlen, minimal, check)
    return _build_resolution_measure_cache(obj, agg; opts=opts, level=level, threaded=threaded, cache=cache)
end

function build_cache(M::PModule{K,F,MatT},
                     spec::AbstractFeaturizerSpec;
                     opts::InvariantOptions=InvariantOptions(),
                     level::Symbol=:auto,
                     threaded::Bool=true,
                     cache=:auto) where {K,F<:AbstractCoeffField,MatT<:AbstractMatrix{K}}
    supports(spec, M) || throw(ArgumentError("build_cache: sample is unsupported for $(typeof(spec))"))
    if spec isa BettiTableSpec || spec isa BassTableSpec
        return _build_resolution_cache(M, spec; level=level, threaded=threaded, cache=cache)
    end
    c = build_cache(M; opts=opts, level=level, threaded=threaded)
    _prime_cache!(c, spec; opts=opts, threaded=threaded)
    return c
end

@inline function _slice_compile_kwargs(spec::Union{LandscapeSpec,MPLandscapeSpec,BarcodeTopKSpec,BarcodeSummarySpec},
                                       opts0::InvariantOptions,
                                       threads0::Bool)
    return (
        directions=spec.directions,
        offsets=spec.offsets,
        offset_weights=spec.offset_weights,
        normalize_weights=spec.normalize_weights,
        tmin=spec.tmin,
        tmax=spec.tmax,
        nsteps=spec.nsteps,
        drop_unknown=spec.drop_unknown,
        dedup=spec.dedup,
        normalize_dirs=spec.normalize_dirs,
        direction_weight=spec.direction_weight,
        threads=threads0,
    )
end

@inline function _slice_compile_kwargs(spec::PersistenceImageSpec, opts0::InvariantOptions, threads0::Bool)
    return (
        directions=spec.directions,
        offsets=spec.offsets,
        offset_weights=spec.offset_weights,
        normalize_weights=spec.normalize_weights,
        tmin=spec.tmin,
        tmax=spec.tmax,
        nsteps=spec.nsteps,
        drop_unknown=spec.drop_unknown,
        dedup=spec.dedup,
        normalize_dirs=spec.normalize_dirs,
        direction_weight=spec.direction_weight,
        threads=threads0,
    )
end

@inline function _slice_compile_kwargs(spec::SlicedBarcodeSpec, opts0::InvariantOptions, threads0::Bool)
    return (
        directions=spec.directions,
        offsets=spec.offsets,
        offset_weights=spec.offset_weights,
        normalize_weights=spec.normalize_weights,
        tmin=spec.tmin,
        tmax=spec.tmax,
        nsteps=spec.nsteps,
        drop_unknown=spec.drop_unknown,
        dedup=spec.dedup,
        normalize_dirs=spec.normalize_dirs,
        direction_weight=spec.direction_weight,
        threads=threads0,
    )
end

@inline function _slice_compile_opts(spec::Union{LandscapeSpec,PersistenceImageSpec,MPLandscapeSpec,SlicedBarcodeSpec,BarcodeTopKSpec,BarcodeSummarySpec},
                                     opts0::InvariantOptions)
    strict0 = opts0.strict === nothing ? spec.strict : opts0.strict
    box0 = opts0.box
    if box0 === nothing && spec.tmin === nothing && spec.tmax === nothing
        box0 = :auto
    end
    return InvariantOptions(
        axes=opts0.axes,
        axes_policy=opts0.axes_policy,
        max_axis_len=opts0.max_axis_len,
        box=box0,
        threads=opts0.threads,
        strict=strict0,
        pl_mode=opts0.pl_mode,
    )
end

function _slice_plan_for!(cache::EncodingInvariantCache,
                          spec::Union{LandscapeSpec,PersistenceImageSpec,MPLandscapeSpec,SlicedBarcodeSpec,BarcodeTopKSpec,BarcodeSummarySpec},
                          opts0::InvariantOptions,
                          threads0::Bool)
    opts_compile = _slice_compile_opts(spec, opts0)
    kws = _slice_compile_kwargs(spec, opts0, threads0)
    # Keep geometry-identical slice specs on the same compiled plan.
    key = UInt(hash((opts_compile.box, opts_compile.strict, kws)))
    return get!(cache.slice_plans, key) do
        Invariants.compile_slices(cache.pi, opts_compile; cache=cache.slice_plan_cache, kws...)
    end
end

function _fibered_cache_for!(cache::EncodingInvariantCache,
                             spec::Union{LandscapeSpec,PersistenceImageSpec,SlicedBarcodeSpec,BarcodeTopKSpec,BarcodeSummarySpec},
                             opts0::InvariantOptions,
                             threads0::Bool,
                             level::Symbol)
    pi0 = _unwrap_cache_pi(cache.pi)
    _supports_fibered_cache(pi0) ||
        throw(ArgumentError("fibered cache requested for non-2D/non-PL encoding"))
    precompute = level == :all ? :cells_barcodes : :none
    key = UInt(hash((opts0.box, opts0.strict, spec.normalize_dirs, precompute)))
    return get!(cache.fibered, key) do
        Invariants.fibered_barcode_cache_2d(
            cache.M,
            pi0,
            opts0;
            include_axes=true,
            precompute=precompute,
            normalize_dirs=spec.normalize_dirs,
            threads=threads0,
        )
    end
end

function _fibered_cache_for!(cache::EncodingInvariantCache,
                             spec::MPPImageSpec,
                             opts0::InvariantOptions,
                             threads0::Bool,
                             level::Symbol)
    pi0 = _unwrap_cache_pi(cache.pi)
    _supports_fibered_cache(pi0) ||
        throw(ArgumentError("fibered cache requested for non-2D/non-PL encoding"))
    precompute = level == :all ? :full : :none
    key = UInt(hash((:mpp_image, opts0.box, opts0.strict, precompute)))
    return get!(cache.fibered, key) do
        Invariants.fibered_barcode_cache_2d(
            cache.M,
            pi0,
            opts0;
            include_axes=true,
            precompute=precompute,
            threads=threads0,
        )
    end
end

function _fibered_cache_for!(cache::EncodingInvariantCache,
                             spec::MPPDecompositionHistogramSpec,
                             opts0::InvariantOptions,
                             threads0::Bool,
                             level::Symbol)
    pi0 = _unwrap_cache_pi(cache.pi)
    _supports_fibered_cache(pi0) ||
        throw(ArgumentError("fibered cache requested for non-2D/non-PL encoding"))
    precompute = level == :all ? :full : :none
    key = UInt(hash((:mpp_hist, opts0.box, opts0.strict, precompute)))
    return get!(cache.fibered, key) do
        Invariants.fibered_barcode_cache_2d(
            cache.M,
            pi0,
            opts0;
            include_axes=true,
            precompute=precompute,
            threads=threads0,
        )
    end
end

@inline function _mpp_decomposition_key(spec::MPPImageSpec, opts0::InvariantOptions)
    return UInt(hash((opts0.box, opts0.strict, spec.N, spec.delta, spec.q, spec.tie_break)))
end

@inline function _mpp_decomposition_key(spec::MPPDecompositionHistogramSpec, opts0::InvariantOptions)
    return UInt(hash((opts0.box, opts0.strict, spec.N, spec.delta, spec.q, spec.tie_break)))
end

function _mpp_decomposition_for!(cache::EncodingInvariantCache,
                                 spec::MPPImageSpec,
                                 opts0::InvariantOptions,
                                 threads0::Bool,
                                 level::Symbol)
    fcache = _fibered_cache_for!(cache, spec, opts0, threads0, level)
    key = _mpp_decomposition_key(spec, opts0)
    decomp = get!(cache.mpp_decompositions, key) do
        Invariants.mpp_decomposition(
            fcache;
            N=spec.N,
            delta=spec.delta,
            q=spec.q,
            tie_break=spec.tie_break,
        )
    end
    return fcache, decomp
end

function _mpp_decomposition_for!(cache::EncodingInvariantCache,
                                 spec::MPPDecompositionHistogramSpec,
                                 opts0::InvariantOptions,
                                 threads0::Bool,
                                 level::Symbol)
    fcache = _fibered_cache_for!(cache, spec, opts0, threads0, level)
    key = _mpp_decomposition_key(spec, opts0)
    decomp = get!(cache.mpp_decompositions, key) do
        Invariants.mpp_decomposition(
            fcache;
            N=spec.N,
            delta=spec.delta,
            q=spec.q,
            tie_break=spec.tie_break,
        )
    end
    return fcache, decomp
end

@inline function _rank_query_cache_for!(cache::EncodingInvariantCache)
    pi0 = _unwrap_cache_pi(cache.pi)
    pi0 isa ZnEncoding.ZnEncodingMap || return nothing
    if cache.rank_query === nothing
        cache.rank_query = RankQueryCache(pi0)
    end
    return cache.rank_query
end

function _projected_arrangement_for!(cache::EncodingInvariantCache,
                                     spec::ProjectedDistancesSpec,
                                     threads0::Bool)
    key = UInt(hash((spec.directions, spec.n_dirs, spec.normalize, spec.enforce_monotone)))
    return get!(cache.projected_arrangements, key) do
        enc_cache = _workflow_encoding_cache(cache.session_cache)
        if spec.directions === nothing
            Invariants.projected_arrangement(cache.pi;
                n_dirs=spec.n_dirs,
                normalize=spec.normalize,
                enforce_monotone=spec.enforce_monotone,
                cache=enc_cache,
                threads=threads0)
        else
            Invariants.projected_arrangement(cache.pi;
                dirs=spec.directions,
                normalize=spec.normalize,
                enforce_monotone=spec.enforce_monotone,
                cache=enc_cache,
                threads=threads0)
        end
    end
end

function _projected_cache_for!(cache::EncodingInvariantCache,
                               spec::ProjectedDistancesSpec,
                               threads0::Bool)
    arr_key = UInt(hash((spec.directions, spec.n_dirs, spec.normalize, spec.enforce_monotone)))
    arr = _projected_arrangement_for!(cache, spec, threads0)
    cM = get!(cache.projected_module, arr_key) do
        Invariants.projected_barcode_cache(cache.M, arr; precompute=spec.precompute)
    end
    if spec.precompute
        Invariants.projected_barcodes(cM; threads=threads0)
    end
    refs = Vector{ProjectedBarcodeCache}(undef, length(spec.references))
    @inbounds for i in eachindex(spec.references)
        Mi, _ = _sample_module_pi(spec.references[i])
        rid = UInt(objectid(Mi))
        refs[i] = get!(cache.projected_refs, (arr_key, rid)) do
            Invariants.projected_barcode_cache(Mi, arr; precompute=spec.precompute)
        end
        if spec.precompute
            Invariants.projected_barcodes(refs[i]; threads=threads0)
        end
    end
    return cM, refs
end

@inline function _prime_cache!(cache::AbstractInvariantCache, spec::AbstractFeaturizerSpec;
                               opts::InvariantOptions=InvariantOptions(),
                               threaded::Bool=true)
    return cache
end

function _prime_cache!(cache::EncodingInvariantCache,
                       spec::Union{LandscapeSpec,PersistenceImageSpec,MPLandscapeSpec,SlicedBarcodeSpec};
                       opts::InvariantOptions=InvariantOptions(),
                       threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    if spec isa MPLandscapeSpec
        _slice_plan_for!(cache, spec, opts0, threads0)
    elseif lvl == :fibered || lvl == :all
        _fibered_cache_for!(cache, spec, opts0, threads0, lvl)
    else
        _slice_plan_for!(cache, spec, opts0, threads0)
    end
    return cache
end

function _prime_cache!(cache::EncodingInvariantCache,
                       spec::Union{BarcodeTopKSpec,BarcodeSummarySpec};
                       opts::InvariantOptions=InvariantOptions(),
                       threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    if spec.source == :fibered
        _fibered_cache_for!(cache, spec, opts0, threads0, lvl)
    elseif lvl != :none
        _slice_plan_for!(cache, spec, opts0, threads0)
    end
    return cache
end

function _prime_cache!(cache::EncodingInvariantCache,
                       spec::ProjectedDistancesSpec;
                       opts::InvariantOptions=InvariantOptions(),
                       threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    if lvl == :projected || lvl == :all
        _projected_cache_for!(cache, spec, threads0)
    end
    return cache
end

function _prime_cache!(cache::EncodingInvariantCache,
                       spec::SignedBarcodeImageSpec;
                       opts::InvariantOptions=InvariantOptions(),
                       threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    _ = threads0
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    if lvl == :all
        _rank_query_cache_for!(cache)
    end
    return cache
end

function _prime_cache!(cache::EncodingInvariantCache,
                       spec::RectangleSignedBarcodeTopKSpec;
                       opts::InvariantOptions=InvariantOptions(),
                       threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    _ = threads0
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    if lvl == :all
        _rank_query_cache_for!(cache)
    end
    return cache
end

function _prime_cache!(cache::EncodingInvariantCache,
                       spec::MPPImageSpec;
                       opts::InvariantOptions=InvariantOptions(),
                       threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    if lvl == :fibered || lvl == :all
        _fibered_cache_for!(cache, spec, opts0, threads0, lvl)
    end
    return cache
end

function _prime_cache!(cache::EncodingInvariantCache,
                       spec::MPPDecompositionHistogramSpec;
                       opts::InvariantOptions=InvariantOptions(),
                       threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    if lvl == :fibered || lvl == :all
        _mpp_decomposition_for!(cache, spec, opts0, threads0, lvl)
    end
    return cache
end

function _prime_cache!(cache::AbstractInvariantCache,
                       spec::CompositeSpec;
                       opts::InvariantOptions=InvariantOptions(),
                       threaded::Bool=true)
    @inbounds for sub in spec.specs
        _prime_cache!(cache, sub; opts=opts, threaded=threaded)
    end
    return cache
end

function build_cache(obj,
                     spec::AbstractFeaturizerSpec;
                     opts::InvariantOptions=InvariantOptions(),
                     level::Symbol=:auto,
                     threaded::Bool=true,
                     cache=:auto)
    supports(spec, obj) || throw(ArgumentError("build_cache: sample is unsupported for $(typeof(spec))"))
    if spec isa PointSignedMeasureSpec
        obj isa PointSignedMeasure ||
            throw(ArgumentError("PointSignedMeasureSpec requires a PointSignedMeasure sample"))
        return PointMeasureInvariantCache{typeof(obj)}(obj, opts, threaded, level)
    elseif spec isa RectangleSignedBarcodeTopKSpec && obj isa RectSignedBarcode
        return SignedBarcodeInvariantCache{typeof(obj)}(obj, opts, threaded, level)
    elseif spec isa RankGridSpec
        return build_cache(_sample_module(obj);
            opts=opts,
            level=level,
            threaded=threaded)
    elseif spec isa CompositeSpec
        fam = _composite_pure_resolution_family(spec)
        if fam == :projective || fam == :injective
            if any(_needs_resolution_pi, spec.specs)
                return _build_resolution_measure_cache(obj, spec; opts=opts, level=level, threaded=threaded, cache=cache)
            end
            return _build_resolution_cache(obj, spec; level=level, threaded=threaded, cache=cache)
        end
    elseif spec isa BettiTableSpec || spec isa BassTableSpec
        return _build_resolution_cache(obj, spec; level=level, threaded=threaded, cache=cache)
    elseif spec isa BettiSupportMeasuresSpec || spec isa BassSupportMeasuresSpec
        return _build_resolution_measure_cache(obj, spec; opts=opts, level=level, threaded=threaded, cache=cache)
    elseif spec isa RestrictedHilbertSpec
        if _supports_module_pi(obj)
            M, pi = _sample_module_pi(obj)
            return build_restricted_hilbert_cache(M, pi; opts=opts, level=level, threaded=threaded)
        end
        return build_cache(_sample_module(obj); opts=opts, level=level, threaded=threaded)
    elseif spec isa EulerSurfaceSpec
        M, pi = _sample_module_pi(obj)
        return build_restricted_hilbert_cache(M, pi;
            opts=opts,
            level=level,
            threaded=threaded)
    end
    cache_obj = build_cache(obj; opts=opts, level=level, threaded=threaded, cache=cache)
    _prime_cache!(cache_obj, spec; opts=opts, threaded=threaded)
    return cache_obj
end

function cache_stats(cache::ModuleInvariantCache)
    return (kind=:module, threads=cache.threads, level=cache.level)
end

function cache_stats(cache::RestrictedHilbertInvariantCache)
    return (kind=:restricted_hilbert, threads=cache.threads, level=cache.level, n=length(cache.hilbert))
end

function cache_stats(cache::ResolutionInvariantCache)
    return (kind=:resolution, threads=cache.threads, level=cache.level, resolution_kind=cache.kind)
end

function cache_stats(cache::ResolutionMeasureInvariantCache)
    return (kind=:resolution_measure,
            threads=cache.threads,
            level=cache.level,
            resolution_kind=cache.kind,
            n_weights=length(cache.weights))
end

function cache_stats(cache::EncodingInvariantCache)
    return (
        kind=:encoding,
        threads=cache.threads,
        level=cache.level,
        n_slice_plans=length(cache.slice_plans),
        n_projected_arrangements=length(cache.projected_arrangements),
        n_projected_module=length(cache.projected_module),
        n_projected_refs=length(cache.projected_refs),
        n_fibered=length(cache.fibered),
        n_mpp_decompositions=length(cache.mpp_decompositions),
        has_rank_query=(cache.rank_query !== nothing),
    )
end

function cache_stats(cache::PointMeasureInvariantCache)
    return (kind=:point_measure, threads=cache.threads, level=cache.level, n=length(cache.pm))
end

function cache_stats(cache::SignedBarcodeInvariantCache)
    return (kind=:signed_barcode, threads=cache.threads, level=cache.level, n=length(cache.sb))
end

function transform(spec::AbstractFeaturizerSpec,
                   cache::AbstractInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    return transform(spec, _cache_sample(cache); opts=opts, threaded=_cache_threaded(cache, threaded))
end

function _slice_feature_from_barcode(
    bc,
    featurizer;
    kmax::Int=5,
    tgrid=nothing,
    sil_weighting=:persistence,
    sil_p::Real=1,
    sil_normalize::Bool=true,
    xgrid=nothing,
    ygrid=nothing,
    img_sigma::Real=1.0,
    img_coords::Symbol=:birth_persistence,
    img_weighting=:persistence,
    img_p::Real=1,
    img_normalize::Symbol=:none,
    entropy_normalize::Bool=true,
    entropy_weighting=:persistence,
    entropy_p::Real=1,
    summary_fields=_DEFAULT_CANONICAL_BARCODE_SUMMARY_FIELDS,
    summary_normalize_entropy::Bool=true,
)
    if featurizer == :landscape
        pl = Invariants.persistence_landscape(bc; kmax=kmax, tgrid=tgrid)
        return Invariants._landscape_feature_vector(pl)
    elseif featurizer == :silhouette
        s = Invariants.persistence_silhouette(
            bc;
            tgrid=tgrid,
            weighting=sil_weighting,
            p=sil_p,
            normalize=sil_normalize,
        )
        return Float64[s...]
    elseif featurizer == :image
        PI = Invariants.persistence_image(
            bc;
            xgrid=xgrid,
            ygrid=ygrid,
            sigma=img_sigma,
            coords=img_coords,
            weighting=img_weighting,
            p=img_p,
            normalize=img_normalize,
        )
        return Invariants._image_feature_vector(PI)
    elseif featurizer == :entropy
        e = Invariants.barcode_entropy(
            bc;
            normalize=entropy_normalize,
            weighting=entropy_weighting,
            p=entropy_p,
        )
        return Float64[float(e)]
    elseif featurizer == :summary
        return _canonical_barcode_summary_vector(
            bc;
            fields=summary_fields,
            normalize_entropy=summary_normalize_entropy,
        )
    elseif featurizer isa Function
        return Invariants._as_feature_vector(featurizer(bc))
    end
    throw(ArgumentError("unsupported slice featurizer $(featurizer)"))
end

function _canonical_barcode_summary_vector(
    bar;
    fields=_DEFAULT_CANONICAL_BARCODE_SUMMARY_FIELDS,
    normalize_entropy::Bool=true,
)
    nt = Invariants.barcode_summary(bar; normalize_entropy=normalize_entropy)
    values = (
        count=Float64(nt.n_intervals),
        sum_persistence=Float64(nt.total_persistence),
        mean_persistence=Float64(nt.mean_persistence),
        max_persistence=Float64(nt.max_persistence),
        entropy=Float64(nt.entropy),
    )
    out = Vector{Float64}(undef, length(fields))
    @inbounds for i in eachindex(fields)
        out[i] = getproperty(values, fields[i])
    end
    return out
end

function _slice_features_from_packed_grid(
    bars::PackedBarcodeGrid,
    W::AbstractMatrix{Float64};
    featurizer,
    aggregate::Symbol,
    threads::Bool,
    kmax::Int=5,
    tgrid=nothing,
    sil_weighting=:persistence,
    sil_p::Real=1,
    sil_normalize::Bool=true,
    xgrid=nothing,
    ygrid=nothing,
    img_sigma::Real=1.0,
    img_coords::Symbol=:birth_persistence,
    img_weighting=:persistence,
    img_p::Real=1,
    img_normalize::Symbol=:none,
    entropy_normalize::Bool=true,
    entropy_weighting=:persistence,
    entropy_p::Real=1,
    summary_fields=_DEFAULT_CANONICAL_BARCODE_SUMMARY_FIELDS,
    summary_normalize_entropy::Bool=true,
)
    tg = tgrid
    if (featurizer == :landscape || featurizer == :silhouette) && tg === nothing
        tg = Invariants._default_tgrid_from_barcodes(bars; nsteps=401)
    end
    if tg !== nothing
        tg = Invariants._clean_tgrid(tg)
    end

    xg = xgrid
    yg = ygrid
    if featurizer == :image && (xg === nothing || yg === nothing)
        xg2, yg2 = Invariants._default_image_grids_from_barcodes(
            bars;
            img_xgrid=xg,
            img_ygrid=yg,
            img_birth_range=nothing,
            img_pers_range=nothing,
            img_nbirth=20,
            img_npers=20,
        )
        xg = xg2
        yg = yg2
    end

    nd, no = size(bars)
    feats = Array{Vector{Float64}}(undef, nd, no)
    if threads && Threads.nthreads() > 1
        Threads.@threads for idx in 1:(nd * no)
            i = div(idx - 1, no) + 1
            j = (idx - 1) % no + 1
            feats[i, j] = _slice_feature_from_barcode(
                bars[i, j],
                featurizer;
                kmax=kmax,
                tgrid=tg,
                sil_weighting=sil_weighting,
                sil_p=sil_p,
                sil_normalize=sil_normalize,
                xgrid=xg,
                ygrid=yg,
                img_sigma=img_sigma,
                img_coords=img_coords,
                img_weighting=img_weighting,
                img_p=img_p,
                img_normalize=img_normalize,
                entropy_normalize=entropy_normalize,
                entropy_weighting=entropy_weighting,
                entropy_p=entropy_p,
                summary_fields=summary_fields,
                summary_normalize_entropy=summary_normalize_entropy,
            )
        end
    else
        @inbounds for i in 1:nd, j in 1:no
            feats[i, j] = _slice_feature_from_barcode(
                bars[i, j],
                featurizer;
                kmax=kmax,
                tgrid=tg,
                sil_weighting=sil_weighting,
                sil_p=sil_p,
                sil_normalize=sil_normalize,
                xgrid=xg,
                ygrid=yg,
                img_sigma=img_sigma,
                img_coords=img_coords,
                img_weighting=img_weighting,
                img_p=img_p,
                img_normalize=img_normalize,
                entropy_normalize=entropy_normalize,
                entropy_weighting=entropy_weighting,
                entropy_p=entropy_p,
                summary_fields=summary_fields,
                summary_normalize_entropy=summary_normalize_entropy,
            )
        end
    end
    return Invariants._aggregate_feature_vectors(feats, W; aggregate=aggregate, unwrap_scalar=true)
end

@inline function _barcode_working_window(pb::PackedBarcode,
                                         window::Union{Nothing,Tuple{Float64,Float64}})
    window !== nothing && return window
    lo = Inf
    hi = -Inf
    seen = false
    @inbounds for p in pb.pairs
        b = Float64(p.b)
        d = Float64(p.d)
        if isfinite(b)
            lo = min(lo, b)
            hi = max(hi, b)
            seen = true
        end
        if isfinite(d)
            lo = min(lo, d)
            hi = max(hi, d)
            seen = true
        end
    end
    seen || return nothing
    return (lo, hi)
end

function _barcode_grid_window(bars::PackedBarcodeGrid,
                              window::Union{Nothing,Tuple{Float64,Float64}})
    window !== nothing && return window
    lo = Inf
    hi = -Inf
    seen = false
    @inbounds for pb in bars.flat
        w = _barcode_working_window(pb, nothing)
        w === nothing && continue
        lo = min(lo, w[1])
        hi = max(hi, w[2])
        seen = true
    end
    return seen ? (lo, hi) : nothing
end

@inline function _clip_barcode_endpoint(x::Float64,
                                        lo::Float64,
                                        hi::Float64,
                                        policy::Symbol)
    isfinite(x) && return x
    policy == :clip_to_window || throw(ArgumentError("barcode featurizer requires a finite working window when intervals are unbounded"))
    return x > 0 ? hi : lo
end

function _canonical_barcode_entries(pb::PackedBarcode,
                                    window::Union{Nothing,Tuple{Float64,Float64}},
                                    infinite_policy::Symbol)
    work = _barcode_working_window(pb, window)
    if infinite_policy == :clip_to_window && work === nothing
        throw(ArgumentError("barcode featurizer requires a finite working window to clip unbounded intervals"))
    end
    lo = work === nothing ? 0.0 : work[1]
    hi = work === nothing ? 0.0 : work[2]
    entries = Vector{NTuple{4,Float64}}()
    count = 0
    total_persistence = 0.0
    max_persistence = 0.0
    weighted_log = 0.0
    weight_total = 0.0

    # Canonical policy: clip or reject unbounded endpoints first, then rank
    # intervals by persistence desc / birth asc / death asc on the normalized
    # finite representatives.
    @inbounds for idx in eachindex(pb.pairs)
        mult = pb.mults[idx]
        mult <= 0 && continue
        p = pb.pairs[idx]
        b = Float64(p.b)
        d = Float64(p.d)
        if !isfinite(b) || !isfinite(d)
            if infinite_policy == :error
                throw(ArgumentError("barcode featurizer encountered an unbounded interval; use infinite_policy=:clip_to_window with a finite working window"))
            end
            b = _clip_barcode_endpoint(b, lo, hi, infinite_policy)
            d = _clip_barcode_endpoint(d, lo, hi, infinite_policy)
        end
        d > b || continue
        pers = d - b
        count += mult
        total_persistence += mult * pers
        max_persistence = max(max_persistence, pers)
        if pers > 0.0
            w = mult * pers
            weight_total += w
            weighted_log += w * log(pers)
        end
        push!(entries, (pers, b, d, Float64(mult)))
    end
    sort!(entries; by=x -> (-x[1], x[2], x[3]))
    entropy = 0.0
    if weight_total > 0.0
        entropy = log(weight_total) - weighted_log / weight_total
        if count > 1
            entropy /= log(Float64(count))
        end
    end
    return entries, count, total_persistence, max_persistence, entropy
end

function _barcode_topk_vector(pb::PackedBarcode,
                              spec::BarcodeTopKSpec,
                              window::Union{Nothing,Tuple{Float64,Float64}}=spec.window)
    entries, count, _, _, _ = _canonical_barcode_entries(pb, window, spec.infinite_policy)
    out = zeros(Float64, 1 + 3 * spec.k)
    out[1] = Float64(count)
    idx = 2
    remaining = spec.k
    for (pers, birth, _, multf) in entries
        remaining == 0 && break
        copies = min(Int(multf), remaining)
        @inbounds for _ in 1:copies
            out[idx] = 1.0
            out[idx + 1] = birth
            out[idx + 2] = pers
            idx += 3
        end
        remaining -= copies
    end
    return out
end

function _barcode_summary_vector(pb::PackedBarcode,
                                 spec::BarcodeSummarySpec,
                                 window::Union{Nothing,Tuple{Float64,Float64}}=spec.window)
    _, count, total_persistence, max_persistence, entropy =
        _canonical_barcode_entries(pb, window, spec.infinite_policy)
    mean_persistence = count == 0 ? 0.0 : total_persistence / Float64(count)
    values = (
        count=Float64(count),
        sum_persistence=total_persistence,
        mean_persistence=mean_persistence,
        max_persistence=max_persistence,
        entropy=spec.normalize_entropy ? entropy : entropy * (count > 1 ? log(Float64(count)) : 1.0),
    )
    out = Vector{Float64}(undef, length(spec.fields))
    @inbounds for i in eachindex(spec.fields)
        out[i] = getproperty(values, spec.fields[i])
    end
    return out
end

function _aggregate_barcode_feature_grid(
    bars::PackedBarcodeGrid,
    W::AbstractMatrix{Float64},
    builder::F;
    aggregate::Symbol,
    threads::Bool,
) where {F<:Function}
    nd, no = size(bars)
    feats = Matrix{Vector{Float64}}(undef, nd, no)
    if threads && Threads.nthreads() > 1
        Threads.@threads for idx in 1:(nd * no)
            i = div(idx - 1, no) + 1
            j = (idx - 1) % no + 1
            feats[i, j] = builder(bars[i, j])
        end
    else
        @inbounds for i in 1:nd, j in 1:no
            feats[i, j] = builder(bars[i, j])
        end
    end
    d = length(feats[1, 1])
    if aggregate == :stack
        out = Vector{Float64}(undef, nd * no * d)
        idx = 1
        @inbounds for i in 1:nd, j in 1:no
            v = feats[i, j]
            for k in 1:d
                out[idx] = v[k]
                idx += 1
            end
        end
        return out
    elseif aggregate == :sum || aggregate == :mean
        acc = zeros(Float64, d)
        sumw = sum(W)
        @inbounds for i in 1:nd, j in 1:no
            w = W[i, j]
            w == 0.0 && continue
            v = feats[i, j]
            for k in 1:d
                acc[k] += w * v[k]
            end
        end
        if aggregate == :mean
            sumw > 0.0 || error("barcode featurizer: total weight is zero")
            acc ./= sumw
        end
        return acc
    end
    throw(ArgumentError("barcode featurizer aggregate must be :mean, :sum, or :stack"))
end

@inline function _slice_backend(spec::Union{BarcodeTopKSpec,BarcodeSummarySpec}, lvl::Symbol)
    _ = lvl
    return spec.source
end

@inline function _slice_backend(spec::AbstractFeaturizerSpec, lvl::Symbol)
    return (lvl == :fibered || lvl == :all) ? :fibered : :slice
end

function _barcode_grid_for_spec(
    cache::EncodingInvariantCache,
    spec::Union{BarcodeTopKSpec,BarcodeSummarySpec},
    opts0::InvariantOptions,
    threads0::Bool,
)
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    if _slice_backend(spec, lvl) == :fibered
        fcache = _fibered_cache_for!(cache, spec, opts0, threads0, lvl)
        data = Invariants.slice_barcodes(
            fcache;
            dirs=spec.directions,
            offsets=spec.offsets,
            values=:t,
            packed=true,
            direction_weight=spec.direction_weight,
            offset_weights=spec.offset_weights,
            normalize_weights=spec.normalize_weights,
            threads=threads0,
        )
        return data.barcodes, data.weights
    end
    spcache = lvl == :none ? nothing : cache.slice_plan_cache
    if spcache !== nothing
        _slice_plan_for!(cache, spec, opts0, threads0)
    end
    opts_compile = _slice_compile_opts(spec, opts0)
    kws = _slice_compile_kwargs(spec, opts0, threads0)
    plan = Invariants.compile_slices(cache.pi, opts_compile; cache=spcache, kws...)
    task = Invariants.SliceBarcodesTask(; packed=true, threads=threads0)
    data = Invariants.run_invariants(plan, cache.M, task)
    return data.barcodes, data.weights
end

function _mpp_histogram_vector_from_decomp(spec::MPPDecompositionHistogramSpec,
                                           decomp::MPPDecomposition)
    hist = zeros(Float64, spec.orientation_bins, spec.scale_bins)
    total_mass = 0.0
    total_segments = 0
    max_scale = 0.0
    lo_box, hi_box = decomp.box
    diag = 0.0
    @inbounds for i in eachindex(lo_box, hi_box)
        diag += (Float64(hi_box[i]) - Float64(lo_box[i]))^2
    end
    diag = diag > 0.0 ? sqrt(diag) : 1.0
    scale_lo, scale_hi = spec.scale_range === nothing ? (0.0, 1.0) : spec.scale_range
    scale_hi > scale_lo || throw(ArgumentError("MPPDecompositionHistogramSpec.scale_range must satisfy lo < hi"))
    positive_weights = Float64[]

    # Canonical policy: histogram segments in orientation/normalized-scale space,
    # then append global mass/size scalars from the same decomposition.
    @inbounds for si in eachindex(decomp.summands)
        segs = decomp.summands[si]
        mass = si <= length(decomp.weights) ? max(Float64(decomp.weights[si]), 0.0) : 0.0
        total_mass += mass
        mass > 0.0 && push!(positive_weights, mass)
        isempty(segs) && continue
        per_seg = spec.weight == :mass ? (mass / length(segs)) : 1.0
        for seg in segs
            p, q, _ = seg
            dx = Float64(q[1]) - Float64(p[1])
            dy = Float64(q[2]) - Float64(p[2])
            len = hypot(dx, dy)
            len > 0.0 || continue
            normlen = len / diag
            max_scale = max(max_scale, normlen)
            theta = atan(dy, dx)
            theta < 0.0 && (theta += pi)
            theta >= pi && (theta -= pi)
            oi = clamp(fld(Int(floor(theta / pi * spec.orientation_bins)), 1) + 1, 1, spec.orientation_bins)
            sfrac = clamp((normlen - scale_lo) / (scale_hi - scale_lo), 0.0, 1.0)
            si_bin = clamp(fld(Int(floor(sfrac * spec.scale_bins)), 1) + 1, 1, spec.scale_bins)
            hist[oi, si_bin] += per_seg
            total_segments += 1
        end
    end

    flat = _float_vector(hist)
    if spec.normalize == :l1
        s = sum(flat)
        s > 0.0 && (flat ./= s)
    elseif spec.normalize == :l2
        s = sqrt(sum(abs2, flat))
        s > 0.0 && (flat ./= s)
    elseif spec.normalize == :max
        s = maximum(abs, flat)
        s > 0.0 && (flat ./= s)
    end

    entropy = 0.0
    if !isempty(positive_weights)
        ws = sum(positive_weights)
        if ws > 0.0
            @inbounds for w in positive_weights
                p = w / ws
                entropy -= p * log(p)
            end
            length(positive_weights) > 1 && (entropy /= log(Float64(length(positive_weights))))
        end
    end
    append!(flat, (total_mass, Float64(length(decomp.summands)), Float64(total_segments), max_scale, entropy))
    return flat
end

function transform(spec::LandscapeSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    vals = if lvl == :fibered || lvl == :all
        fcache = _fibered_cache_for!(cache, spec, opts0, threads0, lvl)
        data = Invariants.slice_barcodes(
            fcache;
            dirs=spec.directions,
            offsets=spec.offsets,
            values=:t,
            packed=true,
            direction_weight=spec.direction_weight,
            offset_weights=spec.offset_weights,
            normalize_weights=spec.normalize_weights,
            threads=threads0,
        )
        _slice_features_from_packed_grid(
            data.barcodes,
            data.weights;
            featurizer=:landscape,
            aggregate=spec.aggregate,
            threads=threads0,
            kmax=spec.kmax,
            tgrid=spec.tgrid,
        )
    else
        spcache = lvl == :none ? nothing : cache.slice_plan_cache
        if spcache !== nothing
            _slice_plan_for!(cache, spec, opts0, threads0)
        end
        opts_slice = _slice_compile_opts(spec, opts0)
        featres = Invariants.slice_features(
            cache.M, cache.pi;
            opts=opts_slice,
            directions=spec.directions,
            offsets=spec.offsets,
            offset_weights=spec.offset_weights,
            tmin=spec.tmin,
            tmax=spec.tmax,
            nsteps=spec.nsteps,
            drop_unknown=spec.drop_unknown,
            dedup=spec.dedup,
            normalize_dirs=spec.normalize_dirs,
            direction_weight=spec.direction_weight,
            featurizer=:landscape,
            aggregate=spec.aggregate,
            normalize_weights=spec.normalize_weights,
            kmax=spec.kmax,
            tgrid=spec.tgrid,
            threads=threads0,
            cache=spcache,
        )
        Invariants.slice_features(featres)
    end
    return _float_vector(vals)
end

function transform(spec::PersistenceImageSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    vals = if lvl == :fibered || lvl == :all
        fcache = _fibered_cache_for!(cache, spec, opts0, threads0, lvl)
        data = Invariants.slice_barcodes(
            fcache;
            dirs=spec.directions,
            offsets=spec.offsets,
            values=:t,
            packed=true,
            direction_weight=spec.direction_weight,
            offset_weights=spec.offset_weights,
            normalize_weights=spec.normalize_weights,
            threads=threads0,
        )
        _slice_features_from_packed_grid(
            data.barcodes,
            data.weights;
            featurizer=:image,
            aggregate=spec.aggregate,
            threads=threads0,
            xgrid=spec.xgrid,
            ygrid=spec.ygrid,
            img_sigma=spec.sigma,
            img_coords=spec.coords,
            img_weighting=spec.weighting,
            img_p=spec.p,
            img_normalize=spec.normalize,
        )
    else
        spcache = lvl == :none ? nothing : cache.slice_plan_cache
        if spcache !== nothing
            _slice_plan_for!(cache, spec, opts0, threads0)
        end
        opts_slice = _slice_compile_opts(spec, opts0)
        featres = Invariants.slice_features(
            cache.M, cache.pi;
            opts=opts_slice,
            directions=spec.directions,
            offsets=spec.offsets,
            offset_weights=spec.offset_weights,
            tmin=spec.tmin,
            tmax=spec.tmax,
            nsteps=spec.nsteps,
            drop_unknown=spec.drop_unknown,
            dedup=spec.dedup,
            normalize_dirs=spec.normalize_dirs,
            direction_weight=spec.direction_weight,
            featurizer=:image,
            aggregate=spec.aggregate,
            normalize_weights=spec.normalize_weights,
            img_xgrid=spec.xgrid,
            img_ygrid=spec.ygrid,
            img_sigma=spec.sigma,
            img_coords=spec.coords,
            img_weighting=spec.weighting,
            img_p=spec.p,
            img_normalize=spec.normalize,
            threads=threads0,
            cache=spcache,
        )
        Invariants.slice_features(featres)
    end
    return _float_vector(vals)
end

function transform(spec::SlicedBarcodeSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    vals = if lvl == :fibered || lvl == :all
        fcache = _fibered_cache_for!(cache, spec, opts0, threads0, lvl)
        data = Invariants.slice_barcodes(
            fcache;
            dirs=spec.directions,
            offsets=spec.offsets,
            values=:t,
            packed=true,
            direction_weight=spec.direction_weight,
            offset_weights=spec.offset_weights,
            normalize_weights=spec.normalize_weights,
            threads=threads0,
        )
        _slice_features_from_packed_grid(
            data.barcodes,
            data.weights;
            featurizer=spec.featurizer,
            aggregate=spec.aggregate,
            threads=threads0,
            summary_fields=spec.summary_fields,
            summary_normalize_entropy=spec.summary_normalize_entropy,
            entropy_normalize=spec.entropy_normalize,
            entropy_weighting=spec.entropy_weighting,
            entropy_p=spec.entropy_p,
        )
    else
        spcache = lvl == :none ? nothing : cache.slice_plan_cache
        if spcache !== nothing
            _slice_plan_for!(cache, spec, opts0, threads0)
        end
        opts_slice = _slice_compile_opts(spec, opts0)
        featres = Invariants.slice_features(
            cache.M, cache.pi;
            opts=opts_slice,
            directions=spec.directions,
            offsets=spec.offsets,
            offset_weights=spec.offset_weights,
            tmin=spec.tmin,
            tmax=spec.tmax,
            nsteps=spec.nsteps,
            drop_unknown=spec.drop_unknown,
            dedup=spec.dedup,
            normalize_dirs=spec.normalize_dirs,
            direction_weight=spec.direction_weight,
            featurizer=spec.featurizer,
            aggregate=spec.aggregate,
            normalize_weights=spec.normalize_weights,
            summary_fields=_invariant_barcode_summary_fields(spec.summary_fields),
            summary_normalize_entropy=spec.summary_normalize_entropy,
            entropy_normalize=spec.entropy_normalize,
            entropy_weighting=spec.entropy_weighting,
            entropy_p=spec.entropy_p,
            threads=threads0,
            cache=spcache,
        )
        Invariants.slice_features(featres)
    end
    return _float_vector(vals)
end

function transform(spec::BarcodeTopKSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    bars, W = _barcode_grid_for_spec(cache, spec, opts0, threads0)
    window0 = _barcode_grid_window(bars, spec.window)
    return _aggregate_barcode_feature_grid(
        bars,
        W,
        pb -> _barcode_topk_vector(pb, spec, window0);
        aggregate=spec.aggregate,
        threads=threads0,
    )
end

function transform(spec::BarcodeSummarySpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    bars, W = _barcode_grid_for_spec(cache, spec, opts0, threads0)
    window0 = _barcode_grid_window(bars, spec.window)
    return _aggregate_barcode_feature_grid(
        bars,
        W,
        pb -> _barcode_summary_vector(pb, spec, window0);
        aggregate=spec.aggregate,
        threads=threads0,
    )
end

function transform(spec::ProjectedDistancesSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    cM, refs = _projected_cache_for!(cache, spec, threads0)
    out = Vector{Float64}(undef, length(refs))
    @inbounds for i in eachindex(refs)
        out[i] = Invariants.projected_distance(cM, refs[i];
            dist=spec.dist,
            p=spec.p,
            q=spec.q,
            agg=spec.agg,
            threads=threads0)
    end
    return out
end

function transform(spec::MatchingDistanceBankSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    sample_pi = _unwrap_cache_pi(cache.pi)
    out = Vector{Float64}(undef, length(spec.references))
    cache_slice = _slice_plan_cache_from_session(cache.slice_plan_cache, cache.session_cache)
    @inbounds for i in eachindex(spec.references)
        Mref_raw, piref = _sample_module_pi(spec.references[i])
        piref0 = _unwrap_cache_pi(piref)
        sample_pi === piref0 || throw(ArgumentError("MatchingDistanceBankSpec requires all references to share the sample encoding map"))
        Mref = materialize_module(Mref_raw)
        cache.M.Q === Mref.Q || throw(ArgumentError("MatchingDistanceBankSpec requires all references to share the sample poset"))
        if spec.method == :exact_2d
            out[i] = Invariants.matching_distance_exact_2d(
                cache.M,
                Mref,
                sample_pi,
                opts0;
                weight=spec.weight,
                normalize_dirs=spec.normalize_dirs,
                include_axes=spec.include_axes,
            )
        else
            md_kwargs = if spec.directions === nothing && spec.offsets === nothing
                (;
                    n_dirs=spec.n_dirs,
                    n_offsets=spec.n_offsets,
                    max_den=spec.max_den,
                    include_axes=spec.include_axes,
                    normalize_dirs=spec.normalize_dirs,
                    weight=spec.weight,
                    cache=cache_slice,
                )
            else
                (;
                    directions=spec.directions,
                    offsets=spec.offsets,
                    include_axes=spec.include_axes,
                    normalize_dirs=spec.normalize_dirs,
                    weight=spec.weight,
                    cache=cache_slice,
                )
            end
            out[i] = Invariants.matching_distance_approx(cache.M, Mref, sample_pi, opts0; md_kwargs...)
        end
    end
    return out
end

function transform(spec::MatchingDistanceBankSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

@inline function _eulersurface_vector(obj, pi, spec::EulerSurfaceSpec, opts0::InvariantOptions, threads0::Bool)
    axes0 = spec.axes === nothing ? opts0.axes : spec.axes
    strict0 = spec.strict === nothing ? opts0.strict : spec.strict
    opts2 = InvariantOptions(
        axes=axes0,
        axes_policy=spec.axes_policy,
        max_axis_len=spec.max_axis_len,
        box=opts0.box,
        threads=threads0,
        strict=strict0,
        pl_mode=opts0.pl_mode,
    )
    surf = SignedMeasures.euler_surface(obj, pi, opts2)
    return _float_vector(surf)
end

function transform(spec::PointSignedMeasureSpec,
                   cache::PointMeasureInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    _ = opts
    _ = threaded
    return _point_measure_topk_vector(cache.pm, spec.ndims, spec.k, spec.coords)
end

function transform(spec::PointSignedMeasureSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::EulerSignedMeasureSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    pi0 = _unwrap_cache_pi(cache.pi)
    hasproperty(pi0, :n) && Int(getproperty(pi0, :n)) == spec.ndims ||
        throw(ArgumentError("EulerSignedMeasureSpec.ndims=$(spec.ndims) does not match encoding dimension"))
    opts2 = InvariantOptions(
        axes=spec.axes === nothing ? opts0.axes : spec.axes,
        axes_policy=spec.axes_policy,
        max_axis_len=spec.max_axis_len,
        box=opts0.box,
        threads=threads0,
        strict=spec.strict === nothing ? opts0.strict : spec.strict,
        pl_mode=opts0.pl_mode,
    )
    max_terms0 = spec.max_terms > 0 ? max(spec.max_terms, spec.k) : 0
    pm = SignedMeasures.euler_signed_measure(
        cache.M,
        pi0,
        opts2;
        drop_zeros=spec.drop_zeros,
        max_terms=max_terms0,
        min_abs_weight=spec.min_abs_weight,
    )
    return _point_measure_topk_vector(pm, spec.ndims, spec.k, spec.coords)
end

function transform(spec::EulerSignedMeasureSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::RectangleSignedBarcodeTopKSpec,
                   cache::SignedBarcodeInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    _ = opts
    _ = threaded
    return _rect_signed_barcode_topk_vector(cache.sb, spec.ndims, spec.k, spec.coords)
end

function transform(spec::RectangleSignedBarcodeTopKSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    pi0 = _unwrap_cache_pi(cache.pi)
    pi0 isa ZnEncoding.ZnEncodingMap ||
        throw(ArgumentError("RectangleSignedBarcodeTopKSpec requires a ZnEncodingMap-compatible encoding"))
    pi0.n == spec.ndims ||
        throw(ArgumentError("RectangleSignedBarcodeTopKSpec.ndims=$(spec.ndims) does not match encoding dimension $(pi0.n)"))
    opts2 = InvariantOptions(
        axes=spec.axes === nothing ? opts0.axes : spec.axes,
        axes_policy=spec.axes_policy,
        max_axis_len=spec.max_axis_len,
        box=opts0.box,
        threads=threads0,
        strict=spec.strict === nothing ? opts0.strict : spec.strict,
        pl_mode=opts0.pl_mode,
    )
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    rq_cache = lvl == :all ? _rank_query_cache_for!(cache) : nothing
    sb = Invariants.rectangle_signed_barcode(
        cache.M,
        pi0,
        opts2;
        drop_zeros=spec.drop_zeros,
        tol=spec.tol,
        max_span=spec.max_span,
        rq_cache=rq_cache,
        keep_endpoints=spec.keep_endpoints,
        method=spec.method,
        bulk_max_elems=spec.bulk_max_elems,
        threads=threads0,
    )
    return _rect_signed_barcode_topk_vector(sb, spec.ndims, spec.k, spec.coords)
end

function transform(spec::RectangleSignedBarcodeTopKSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

@inline function _rankgrid_vector(M::PModule, spec::RankGridSpec, opts0::InvariantOptions)
    nP = nvertices(M.Q)
    spec.nvertices == nP || throw(ArgumentError("RankGridSpec.nvertices=$(spec.nvertices) does not match module poset size $nP"))
    tbl = Invariants.rank_invariant(M, opts0; store_zeros=spec.store_zeros)
    out = zeros(Float64, spec.nvertices * spec.nvertices)
    @inbounds for a in 1:spec.nvertices, b in 1:spec.nvertices
        out[(a - 1) * spec.nvertices + b] = Float64(get(tbl, (a, b), 0))
    end
    return out
end

@inline function _restricted_hilbert_vector(h::AbstractVector{<:Integer}, spec::RestrictedHilbertSpec)
    spec.nvertices == length(h) ||
        throw(ArgumentError("RestrictedHilbertSpec.nvertices=$(spec.nvertices) does not match restricted_hilbert length $(length(h))"))
    out = Vector{Float64}(undef, spec.nvertices)
    @inbounds for i in eachindex(h)
        out[i] = Float64(h[i])
    end
    return out
end

@inline function _mplandscape_vector(L::MPLandscape)
    nd, no, kk, nt = size(L.values)
    out = Vector{Float64}(undef, nd * no * kk * nt)
    idx = 1
    @inbounds for it in 1:nt, k in 1:kk, j in 1:no, i in 1:nd
        out[idx] = L.values[i, j, k, it]
        idx += 1
    end
    return out
end

@inline function _mpp_image_vector(img::MPPImage)
    nx = length(img.xgrid)
    ny = length(img.ygrid)
    out = Vector{Float64}(undef, nx * ny)
    idx = 1
    @inbounds for ix in 1:nx, iy in 1:ny
        out[idx] = img.img[iy, ix]
        idx += 1
    end
    return out
end

@inline function _betti_table_vector(res::DerivedFunctors.ProjectiveResolution, spec::BettiTableSpec)
    nP = nvertices(res.M.Q)
    spec.nvertices == nP ||
        throw(ArgumentError("BettiTableSpec.nvertices=$(spec.nvertices) does not match module poset size $nP"))
    tbl = DerivedFunctors.betti_table(res; pad_to=spec.pad_to)
    out = Vector{Float64}(undef, (spec.pad_to + 1) * spec.nvertices)
    idx = 1
    @inbounds for a in 0:spec.pad_to, p in 1:spec.nvertices
        out[idx] = Float64(tbl[a + 1, p])
        idx += 1
    end
    return out
end

@inline function _bass_table_vector(res::DerivedFunctors.InjectiveResolution, spec::BassTableSpec)
    nP = nvertices(res.N.Q)
    spec.nvertices == nP ||
        throw(ArgumentError("BassTableSpec.nvertices=$(spec.nvertices) does not match module poset size $nP"))
    tbl = DerivedFunctors.bass_table(res; pad_to=spec.pad_to)
    out = Vector{Float64}(undef, (spec.pad_to + 1) * spec.nvertices)
    idx = 1
    @inbounds for b in 0:spec.pad_to, p in 1:spec.nvertices
        out[idx] = Float64(tbl[b + 1, p])
        idx += 1
    end
    return out
end

@inline function _support_measures_vector(meas, pad_to::Int)
    support_by = meas.support_by_degree
    mass_by = meas.mass_by_degree
    out = Vector{Float64}(undef, 2 * (pad_to + 1) + 3)
    idx = 1
    @inbounds for a in 0:pad_to
        out[idx] = a + 1 <= length(support_by) ? Float64(support_by[a + 1]) : 0.0
        idx += 1
    end
    @inbounds for a in 0:pad_to
        out[idx] = a + 1 <= length(mass_by) ? Float64(mass_by[a + 1]) : 0.0
        idx += 1
    end
    out[idx] = Float64(meas.support_union)
    out[idx + 1] = Float64(meas.support_total)
   out[idx + 2] = Float64(meas.mass_total)
    return out
end

@inline function _point_measure_coord(pm::PointSignedMeasure{N},
                                      ind::NTuple{N,Int},
                                      coords::Symbol,
                                      d::Int) where {N}
    return coords == :values ? Float64(pm.axes[d][ind[d]]) : Float64(ind[d])
end

function _point_measure_topk_vector(pm::PointSignedMeasure{N},
                                    ndims::Int,
                                    k::Int,
                                    coords::Symbol) where {N}
    ndims == N || throw(ArgumentError("signed-point-measure feature ndim mismatch: spec ndims=$(ndims), measure ndims=$(N)"))
    keep = collect(eachindex(pm.wts))
    sort!(keep; by=i -> (-abs(pm.wts[i]), pm.inds[i]))
    out = zeros(Float64, 1 + k * (2 + N))
    out[1] = Float64(length(pm))
    idx = 2
    @inbounds for slot in 1:min(k, length(keep))
        i = keep[slot]
        out[idx] = 1.0
        out[idx + 1] = Float64(pm.wts[i])
        idx += 2
        ind = pm.inds[i]
        for d in 1:N
            out[idx] = _point_measure_coord(pm, ind, coords, d)
            idx += 1
        end
    end
    return out
end

@inline function _rect_signed_barcode_coord(sb::RectSignedBarcode{N},
                                            rect::Rect{N},
                                            coords::Symbol,
                                            d::Int,
                                            side::Symbol) where {N}
    idx = side === :lo ? rect.lo[d] : rect.hi[d]
    return coords == :values ? Float64(sb.axes[d][idx]) : Float64(idx)
end

function _rect_signed_barcode_topk_vector(sb::RectSignedBarcode{N},
                                          ndims::Int,
                                          k::Int,
                                          coords::Symbol) where {N}
    ndims == N || throw(ArgumentError("rectangle-signed-barcode feature ndim mismatch: spec ndims=$(ndims), barcode ndims=$(N)"))
    keep = collect(eachindex(sb.weights))
    sort!(keep; by=i -> (-abs(sb.weights[i]), sb.rects[i].lo, sb.rects[i].hi))
    out = zeros(Float64, 1 + k * (2 + 2 * N))
    out[1] = Float64(length(sb))
    idx = 2
    @inbounds for slot in 1:min(k, length(keep))
        i = keep[slot]
        rect = sb.rects[i]
        out[idx] = 1.0
        out[idx + 1] = Float64(sb.weights[i])
        idx += 2
        for d in 1:N
            out[idx] = _rect_signed_barcode_coord(sb, rect, coords, d, :lo)
            idx += 1
        end
        for d in 1:N
            out[idx] = _rect_signed_barcode_coord(sb, rect, coords, d, :hi)
            idx += 1
        end
    end
    return out
end

@inline function _betti_support_vector(res::DerivedFunctors.ProjectiveResolution,
                                       pi,
                                       spec::BettiSupportMeasuresSpec,
                                       opts0::InvariantOptions;
                                       weights=nothing)
    tbl = DerivedFunctors.betti_table(res; pad_to=spec.pad_to)
    meas = Invariants.betti_support_measures(tbl, _unwrap_cache_pi(pi), opts0; weights=weights)
    return _support_measures_vector(meas, spec.pad_to)
end

@inline function _bass_support_vector(res::DerivedFunctors.InjectiveResolution,
                                      pi,
                                      spec::BassSupportMeasuresSpec,
                                      opts0::InvariantOptions;
                                      weights=nothing)
    tbl = DerivedFunctors.bass_table(res; pad_to=spec.pad_to)
    meas = Invariants.bass_support_measures(tbl, _unwrap_cache_pi(pi), opts0; weights=weights)
    return _support_measures_vector(meas, spec.pad_to)
end

@inline function _projective_resolution_for_cache(cache::EncodingInvariantCache,
                                                  spec::Union{BettiTableSpec,BettiSupportMeasuresSpec})
    rcache = _resolution_cache_from_session(nothing, cache.session_cache, cache.M)
    return DerivedFunctors.projective_resolution(cache.M, _resolution_options(spec); cache=rcache)
end

@inline function _injective_resolution_for_cache(cache::EncodingInvariantCache,
                                                 spec::Union{BassTableSpec,BassSupportMeasuresSpec})
    rcache = _resolution_cache_from_session(nothing, cache.session_cache, cache.M)
    return DerivedFunctors.injective_resolution(cache.M, _resolution_options(spec); cache=rcache)
end

@inline function _mplandscape_vector_from_plan(M::PModule,
                                               spec::MPLandscapeSpec,
                                               plan::CompiledSlicePlan,
                                               threads0::Bool)
    L = Invariants.mp_landscape(
        M,
        plan;
        kmax=spec.kmax,
        tgrid=spec.tgrid,
        normalize_weights=spec.normalize_weights,
        threads=threads0,
    )
    return _mplandscape_vector(L)
end

@inline function _mpp_image_vector_from_decomp(spec::MPPImageSpec,
                                               decomp::MPPDecomposition,
                                               threads0::Bool)
    img = Invariants.mpp_image(
        decomp;
        resolution=spec.resolution,
        xgrid=spec.xgrid,
        ygrid=spec.ygrid,
        sigma=spec.sigma,
        cutoff_radius=spec.cutoff_radius,
        cutoff_tol=spec.cutoff_tol,
        segment_prune=spec.segment_prune,
        threads=threads0,
    )
    return _mpp_image_vector(img)
end

function transform(spec::MPLandscapeSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    plan = _slice_plan_for!(cache, spec, opts0, threads0)
    return _mplandscape_vector_from_plan(cache.M, spec, plan, threads0)
end

function transform(spec::MPLandscapeSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::MPPImageSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    img = if lvl == :fibered || lvl == :all
        _, decomp = _mpp_decomposition_for!(cache, spec, opts0, threads0, lvl)
        _mpp_image_vector_from_decomp(spec, decomp, threads0)
    else
        _mpp_image_vector(Invariants.mpp_image(
            cache.M,
            _unwrap_cache_pi(cache.pi),
            opts0;
            resolution=spec.resolution,
            xgrid=spec.xgrid,
            ygrid=spec.ygrid,
            sigma=spec.sigma,
            N=spec.N,
            delta=spec.delta,
            q=spec.q,
            tie_break=spec.tie_break,
            cutoff_radius=spec.cutoff_radius,
            cutoff_tol=spec.cutoff_tol,
            segment_prune=spec.segment_prune,
            threads=threads0,
        ))
    end
    return img
end

function transform(spec::MPPImageSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::MPPDecompositionHistogramSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    _, decomp = _mpp_decomposition_for!(cache, spec, opts0, threads0, lvl)
    return _mpp_histogram_vector_from_decomp(spec, decomp)
end

function transform(spec::MPPDecompositionHistogramSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::BettiTableSpec,
                   cache::ResolutionInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    _ = opts
    _ = threaded
    cache.kind == :projective ||
        throw(ArgumentError("BettiTableSpec requires a projective resolution cache"))
    return _betti_table_vector(cache.res, spec)
end

function transform(spec::BettiTableSpec,
                   cache::ResolutionMeasureInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    _ = opts
    _ = threaded
    cache.kind == :projective ||
        throw(ArgumentError("BettiTableSpec requires a projective resolution cache"))
    return _betti_table_vector(cache.res, spec)
end

function transform(spec::BettiTableSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    _ = threaded
    res = _projective_resolution_for_cache(cache, spec)
    return _betti_table_vector(res, spec)
end

function transform(spec::BettiTableSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::BassTableSpec,
                   cache::ResolutionInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    _ = opts
    _ = threaded
    cache.kind == :injective ||
        throw(ArgumentError("BassTableSpec requires an injective resolution cache"))
    return _bass_table_vector(cache.res, spec)
end

function transform(spec::BassTableSpec,
                   cache::ResolutionMeasureInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    _ = opts
    _ = threaded
    cache.kind == :injective ||
        throw(ArgumentError("BassTableSpec requires an injective resolution cache"))
    return _bass_table_vector(cache.res, spec)
end

function transform(spec::BassTableSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    _ = threaded
    res = _injective_resolution_for_cache(cache, spec)
    return _bass_table_vector(res, spec)
end

function transform(spec::BassTableSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::BettiSupportMeasuresSpec,
                   cache::ResolutionMeasureInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    _ = threaded
    opts0 = _cache_opts(cache, opts)
    cache.kind == :projective ||
        throw(ArgumentError("BettiSupportMeasuresSpec requires a projective resolution cache"))
    weights = (opts0 == cache.opts ? cache.weights : _support_measure_weights(cache.pi, opts0))
    return _betti_support_vector(cache.res, cache.pi, spec, opts0; weights=weights)
end

function transform(spec::BettiSupportMeasuresSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    _ = threaded
    opts0 = _cache_opts(cache, opts)
    res = _projective_resolution_for_cache(cache, spec)
    weights = _support_measure_weights(cache.pi, opts0)
    return _betti_support_vector(res, cache.pi, spec, opts0; weights=weights)
end

function transform(spec::BettiSupportMeasuresSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::BassSupportMeasuresSpec,
                   cache::ResolutionMeasureInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    _ = threaded
    opts0 = _cache_opts(cache, opts)
    cache.kind == :injective ||
        throw(ArgumentError("BassSupportMeasuresSpec requires an injective resolution cache"))
    weights = (opts0 == cache.opts ? cache.weights : _support_measure_weights(cache.pi, opts0))
    return _bass_support_vector(cache.res, cache.pi, spec, opts0; weights=weights)
end

function transform(spec::BassSupportMeasuresSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    _ = threaded
    opts0 = _cache_opts(cache, opts)
    res = _injective_resolution_for_cache(cache, spec)
    weights = _support_measure_weights(cache.pi, opts0)
    return _bass_support_vector(res, cache.pi, spec, opts0; weights=weights)
end

function transform(spec::BassSupportMeasuresSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::EulerSurfaceSpec,
                   cache::RestrictedHilbertInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    return _eulersurface_vector(cache.hilbert, cache.pi, spec, opts0, threads0)
end

function transform(spec::EulerSurfaceSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    return _eulersurface_vector(cache.M, cache.pi, spec, opts0, threads0)
end

function transform(spec::EulerSurfaceSpec,
                   cache::AbstractInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    return transform(spec, _cache_sample(cache); opts=opts, threaded=_cache_threaded(cache, threaded))
end

function transform(spec::RestrictedHilbertSpec,
                   cache::RestrictedHilbertInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    _ = opts
    _ = threaded
    return _restricted_hilbert_vector(cache.hilbert, spec)
end

function transform(spec::RestrictedHilbertSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    _ = opts
    _ = threaded
    return _restricted_hilbert_vector(Invariants.restricted_hilbert(cache.M), spec)
end

function transform(spec::RestrictedHilbertSpec,
                   cache::ModuleInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    _ = opts
    _ = threaded
    return _restricted_hilbert_vector(Invariants.restricted_hilbert(cache.M), spec)
end

function transform(spec::RestrictedHilbertSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::RankGridSpec,
                   cache::AbstractInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    M = _sample_module(_cache_sample(cache))
    return _rankgrid_vector(M, spec, opts0)
end

function transform(spec::SignedBarcodeImageSpec,
                   cache::EncodingInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    pi0 = cache.pi isa CompiledEncoding ? cache.pi.pi : cache.pi
    pi0 isa ZnEncoding.ZnEncodingMap ||
        throw(ArgumentError("SignedBarcodeImageSpec currently requires a ZnEncodingMap-compatible encoding"))
    axes0 = spec.axes === nothing ? opts0.axes : spec.axes
    strict0 = spec.strict === nothing ? opts0.strict : spec.strict
    threads0 = _resolve_spec_threads(spec.threads, opts0, _cache_threaded(cache, threaded))
    opts2 = InvariantOptions(
        axes=axes0,
        axes_policy=spec.axes_policy,
        max_axis_len=spec.max_axis_len,
        box=opts0.box,
        threads=threads0,
        strict=strict0,
        pl_mode=opts0.pl_mode,
    )
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    rq_cache = lvl == :all ? _rank_query_cache_for!(cache) : nothing
    sb = Invariants.rectangle_signed_barcode(
        cache.M,
        pi0,
        opts2;
        method=spec.method,
        threads=threads0,
        rq_cache=rq_cache,
    )
    img = Invariants.rectangle_signed_barcode_image(sb;
        xs=spec.xs,
        ys=spec.ys,
        sigma=spec.sigma,
        mode=spec.mode,
        threads=threads0)
    return _float_vector(img)
end

function transform(spec::LandscapeSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::PersistenceImageSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::EulerSurfaceSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::RankGridSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    opts0 = opts
    M = _sample_module(obj)
    return _rankgrid_vector(M, spec, opts0)
end

function transform(spec::SlicedBarcodeSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::BarcodeTopKSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::BarcodeSummarySpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::SignedBarcodeImageSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::ProjectedDistancesSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

@inline _is_slice_family_spec(spec) =
    spec isa Union{LandscapeSpec,PersistenceImageSpec,SlicedBarcodeSpec,BarcodeTopKSpec,BarcodeSummarySpec}
@inline _is_mplandscape_family_spec(spec) = spec isa MPLandscapeSpec

function _slice_group_data_for_spec(
    cache::EncodingInvariantCache,
    spec::Union{LandscapeSpec,PersistenceImageSpec,SlicedBarcodeSpec,BarcodeTopKSpec,BarcodeSummarySpec},
    opts0::InvariantOptions,
    threads0::Bool,
)
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    if _slice_backend(spec, lvl) == :fibered
        fcache = _fibered_cache_for!(cache, spec, opts0, threads0, lvl)
        data = Invariants.slice_barcodes(
            fcache;
            dirs=spec.directions,
            offsets=spec.offsets,
            values=:t,
            packed=true,
            direction_weight=spec.direction_weight,
            offset_weights=spec.offset_weights,
            normalize_weights=spec.normalize_weights,
            threads=threads0,
        )
        return data.barcodes, data.weights
    end
    spcache = lvl == :none ? nothing : cache.slice_plan_cache
    if spcache !== nothing
        _slice_plan_for!(cache, spec, opts0, threads0)
    end
    opts_compile = _slice_compile_opts(spec, opts0)
    kws = _slice_compile_kwargs(spec, opts0, threads0)
    plan = Invariants.compile_slices(cache.pi, opts_compile; cache=spcache, kws...)
    task = Invariants.SliceBarcodesTask(; packed=true, threads=threads0)
    data = Invariants.run_invariants(plan, cache.M, task)
    return data.barcodes, data.weights
end

function _composite_slice_vector(
    spec::LandscapeSpec,
    bars::PackedBarcodeGrid,
    W::AbstractMatrix{Float64},
    threads0::Bool,
)
    return _float_vector(_slice_features_from_packed_grid(
        bars, W;
        featurizer=:landscape,
        aggregate=spec.aggregate,
        threads=threads0,
        kmax=spec.kmax,
        tgrid=spec.tgrid,
    ))
end

function _composite_slice_vector(
    spec::PersistenceImageSpec,
    bars::PackedBarcodeGrid,
    W::AbstractMatrix{Float64},
    threads0::Bool,
)
    return _float_vector(_slice_features_from_packed_grid(
        bars, W;
        featurizer=:image,
        aggregate=spec.aggregate,
        threads=threads0,
        xgrid=spec.xgrid,
        ygrid=spec.ygrid,
        img_sigma=spec.sigma,
        img_coords=spec.coords,
        img_weighting=spec.weighting,
        img_p=spec.p,
        img_normalize=spec.normalize,
    ))
end

function _composite_slice_vector(
    spec::SlicedBarcodeSpec,
    bars::PackedBarcodeGrid,
    W::AbstractMatrix{Float64},
    threads0::Bool,
)
    return _float_vector(_slice_features_from_packed_grid(
        bars, W;
        featurizer=spec.featurizer,
        aggregate=spec.aggregate,
        threads=threads0,
        summary_fields=spec.summary_fields,
        summary_normalize_entropy=spec.summary_normalize_entropy,
        entropy_normalize=spec.entropy_normalize,
        entropy_weighting=spec.entropy_weighting,
        entropy_p=spec.entropy_p,
    ))
end

function _composite_slice_vector(
    spec::BarcodeTopKSpec,
    bars::PackedBarcodeGrid,
    W::AbstractMatrix{Float64},
    threads0::Bool,
)
    window0 = _barcode_grid_window(bars, spec.window)
    return _aggregate_barcode_feature_grid(
        bars,
        W,
        pb -> _barcode_topk_vector(pb, spec, window0);
        aggregate=spec.aggregate,
        threads=threads0,
    )
end

function _composite_slice_vector(
    spec::BarcodeSummarySpec,
    bars::PackedBarcodeGrid,
    W::AbstractMatrix{Float64},
    threads0::Bool,
)
    window0 = _barcode_grid_window(bars, spec.window)
    return _aggregate_barcode_feature_grid(
        bars,
        W,
        pb -> _barcode_summary_vector(pb, spec, window0);
        aggregate=spec.aggregate,
        threads=threads0,
    )
end

function _fill_composite_slice_parts!(
    parts::Vector{Vector{Float64}},
    filled::BitVector,
    spec::CompositeSpec,
    cache::EncodingInvariantCache,
    opts0::InvariantOptions,
    threaded0::Bool,
)
    groups = Dict{UInt,Vector{Int}}()
    for i in eachindex(spec.specs)
        sub = spec.specs[i]
        _is_slice_family_spec(sub) || continue
        threads_i = _resolve_spec_threads(getproperty(sub, :threads), opts0, threaded0)
        key = UInt(hash((_slice_compile_kwargs(sub, opts0, threads_i), _effective_cache_level(cache.level, sub, cache.pi))))
        push!(get!(groups, key) do
            Int[]
        end, i)
    end

    for idxs in values(groups)
        first_spec = spec.specs[first(idxs)]::Union{LandscapeSpec,PersistenceImageSpec,SlicedBarcodeSpec,BarcodeTopKSpec,BarcodeSummarySpec}
        threads_g = _resolve_spec_threads(getproperty(first_spec, :threads), opts0, threaded0)
        bars, W = _slice_group_data_for_spec(cache, first_spec, opts0, threads_g)
        for idx in idxs
            sub = spec.specs[idx]
            parts[idx] = _composite_slice_vector(sub, bars, W, threads_g)
            filled[idx] = true
        end
    end
    return nothing
end

@inline function _mplandscape_feature_key(spec::MPLandscapeSpec)
    return UInt(hash((spec.kmax, spec.tgrid, spec.normalize_weights)))
end

function _fill_composite_mplandscape_parts!(
    parts::Vector{Vector{Float64}},
    filled::BitVector,
    spec::CompositeSpec,
    cache::EncodingInvariantCache,
    opts0::InvariantOptions,
    threaded0::Bool,
)
    groups = Dict{UInt,Vector{Int}}()
    for i in eachindex(spec.specs)
        sub = spec.specs[i]
        _is_mplandscape_family_spec(sub) || continue
        threads_i = _resolve_spec_threads(sub.threads, opts0, threaded0)
        key = UInt(hash((_slice_compile_kwargs(sub, opts0, threads_i), _effective_cache_level(cache.level, sub, cache.pi), threads_i)))
        push!(get!(groups, key) do
            Int[]
        end, i)
    end

    for idxs in values(groups)
        first_spec = spec.specs[first(idxs)]::MPLandscapeSpec
        threads_g = _resolve_spec_threads(first_spec.threads, opts0, threaded0)
        plan = _slice_plan_for!(cache, first_spec, opts0, threads_g)
        feature_groups = Dict{UInt,Vector{Int}}()
        for idx in idxs
            sub = spec.specs[idx]::MPLandscapeSpec
            push!(get!(feature_groups, _mplandscape_feature_key(sub)) do
                Int[]
            end, idx)
        end
        for fidxs in values(feature_groups)
            sub = spec.specs[first(fidxs)]::MPLandscapeSpec
            vals = _mplandscape_vector_from_plan(cache.M, sub, plan, threads_g)
            for idx in fidxs
                parts[idx] = vals
                filled[idx] = true
            end
        end
    end
    return nothing
end

@inline _is_projected_family_spec(spec) = spec isa ProjectedDistancesSpec

function _fill_composite_projected_parts!(
    parts::Vector{Vector{Float64}},
    filled::BitVector,
    spec::CompositeSpec,
    cache::EncodingInvariantCache,
    opts0::InvariantOptions,
    threaded0::Bool,
)
    groups = Dict{UInt,Vector{Int}}()
    for i in eachindex(spec.specs)
        sub = spec.specs[i]
        _is_projected_family_spec(sub) || continue
        threads_i = _resolve_spec_threads(sub.threads, opts0, threaded0)
        key = UInt(hash((sub.references, sub.directions, sub.n_dirs, sub.normalize, sub.enforce_monotone, sub.precompute, threads_i)))
        push!(get!(groups, key) do
            Int[]
        end, i)
    end

    for idxs in values(groups)
        first_spec = spec.specs[first(idxs)]::ProjectedDistancesSpec
        threads_g = _resolve_spec_threads(first_spec.threads, opts0, threaded0)
        cM, refs = _projected_cache_for!(cache, first_spec, threads_g)
        for idx in idxs
            sub = spec.specs[idx]::ProjectedDistancesSpec
            out = Vector{Float64}(undef, length(refs))
            @inbounds for i in eachindex(refs)
                out[i] = Invariants.projected_distance(cM, refs[i];
                    dist=sub.dist,
                    p=sub.p,
                    q=sub.q,
                    agg=sub.agg,
                    threads=threads_g)
            end
            parts[idx] = out
            filled[idx] = true
        end
    end
    return nothing
end

@inline _is_signed_barcode_family_spec(spec) = spec isa SignedBarcodeImageSpec

@inline _is_mpp_image_family_spec(spec) = spec isa MPPImageSpec
@inline _is_mpp_hist_family_spec(spec) = spec isa MPPDecompositionHistogramSpec

@inline function _mpp_image_feature_key(spec::MPPImageSpec)
    return UInt(hash((spec.resolution, spec.xgrid, spec.ygrid, spec.sigma, spec.cutoff_radius, spec.cutoff_tol, spec.segment_prune)))
end

function _fill_composite_mpp_image_parts!(
    parts::Vector{Vector{Float64}},
    filled::BitVector,
    spec::CompositeSpec,
    cache::EncodingInvariantCache,
    opts0::InvariantOptions,
    threaded0::Bool,
)
    groups = Dict{UInt,Vector{Int}}()
    for i in eachindex(spec.specs)
        sub = spec.specs[i]
        _is_mpp_image_family_spec(sub) || continue
        threads_i = _resolve_spec_threads(sub.threads, opts0, threaded0)
        key = UInt(hash((_mpp_decomposition_key(sub, opts0), _effective_cache_level(cache.level, sub, cache.pi), threads_i)))
        push!(get!(groups, key) do
            Int[]
        end, i)
    end

    for idxs in values(groups)
        first_spec = spec.specs[first(idxs)]::MPPImageSpec
        threads_g = _resolve_spec_threads(first_spec.threads, opts0, threaded0)
        lvl = _effective_cache_level(cache.level, first_spec, cache.pi)
        _, decomp = _mpp_decomposition_for!(cache, first_spec, opts0, threads_g, lvl)
        feature_groups = Dict{UInt,Vector{Int}}()
        for idx in idxs
            sub = spec.specs[idx]::MPPImageSpec
            push!(get!(feature_groups, _mpp_image_feature_key(sub)) do
                Int[]
            end, idx)
        end
        for fidxs in values(feature_groups)
            sub = spec.specs[first(fidxs)]::MPPImageSpec
            vals = _mpp_image_vector_from_decomp(sub, decomp, threads_g)
            for idx in fidxs
                parts[idx] = vals
                filled[idx] = true
            end
        end
    end
    return nothing
end

function _fill_composite_mpp_hist_parts!(
    parts::Vector{Vector{Float64}},
    filled::BitVector,
    spec::CompositeSpec,
    cache::EncodingInvariantCache,
    opts0::InvariantOptions,
    threaded0::Bool,
)
    groups = Dict{UInt,Vector{Int}}()
    for i in eachindex(spec.specs)
        sub = spec.specs[i]
        _is_mpp_hist_family_spec(sub) || continue
        threads_i = _resolve_spec_threads(sub.threads, opts0, threaded0)
        key = UInt(hash((_mpp_decomposition_key(sub, opts0), _effective_cache_level(cache.level, sub, cache.pi), threads_i)))
        push!(get!(groups, key) do
            Int[]
        end, i)
    end

    for idxs in values(groups)
        first_spec = spec.specs[first(idxs)]::MPPDecompositionHistogramSpec
        threads_g = _resolve_spec_threads(first_spec.threads, opts0, threaded0)
        lvl = _effective_cache_level(cache.level, first_spec, cache.pi)
        _, decomp = _mpp_decomposition_for!(cache, first_spec, opts0, threads_g, lvl)
        feature_groups = Dict{UInt,Vector{Int}}()
        for idx in idxs
            sub = spec.specs[idx]::MPPDecompositionHistogramSpec
            push!(get!(feature_groups, UInt(hash((sub.orientation_bins, sub.scale_bins, sub.scale_range, sub.weight, sub.normalize)))) do
                Int[]
            end, idx)
        end
        for fidxs in values(feature_groups)
            sub = spec.specs[first(fidxs)]::MPPDecompositionHistogramSpec
            vals = _mpp_histogram_vector_from_decomp(sub, decomp)
            for idx in fidxs
                parts[idx] = vals
                filled[idx] = true
            end
        end
    end
    return nothing
end

function _signed_barcode_for_spec(
    cache::EncodingInvariantCache,
    spec::SignedBarcodeImageSpec,
    opts0::InvariantOptions,
    threads0::Bool,
)
    pi0 = _unwrap_cache_pi(cache.pi)
    pi0 isa ZnEncoding.ZnEncodingMap ||
        throw(ArgumentError("SignedBarcodeImageSpec currently requires a ZnEncodingMap-compatible encoding"))
    axes0 = spec.axes === nothing ? opts0.axes : spec.axes
    strict0 = spec.strict === nothing ? opts0.strict : spec.strict
    opts2 = InvariantOptions(
        axes=axes0,
        axes_policy=spec.axes_policy,
        max_axis_len=spec.max_axis_len,
        box=opts0.box,
        threads=threads0,
        strict=strict0,
        pl_mode=opts0.pl_mode,
    )
    lvl = _effective_cache_level(cache.level, spec, cache.pi)
    rq_cache = lvl == :all ? _rank_query_cache_for!(cache) : nothing
    return Invariants.rectangle_signed_barcode(
        cache.M,
        pi0,
        opts2;
        method=spec.method,
        threads=threads0,
        rq_cache=rq_cache,
    )
end

function _fill_composite_signed_barcode_parts!(
    parts::Vector{Vector{Float64}},
    filled::BitVector,
    spec::CompositeSpec,
    cache::EncodingInvariantCache,
    opts0::InvariantOptions,
    threaded0::Bool,
)
    groups = Dict{UInt,Vector{Int}}()
    for i in eachindex(spec.specs)
        sub = spec.specs[i]
        _is_signed_barcode_family_spec(sub) || continue
        threads_i = _resolve_spec_threads(sub.threads, opts0, threaded0)
        axes_i = sub.axes === nothing ? opts0.axes : sub.axes
        strict_i = sub.strict === nothing ? opts0.strict : sub.strict
        key = UInt(hash((sub.method, axes_i, sub.axes_policy, sub.max_axis_len, strict_i, threads_i)))
        push!(get!(groups, key) do
            Int[]
        end, i)
    end

    for idxs in values(groups)
        first_spec = spec.specs[first(idxs)]::SignedBarcodeImageSpec
        threads_g = _resolve_spec_threads(first_spec.threads, opts0, threaded0)
        sb = _signed_barcode_for_spec(cache, first_spec, opts0, threads_g)
        for idx in idxs
            sub = spec.specs[idx]::SignedBarcodeImageSpec
            img = Invariants.rectangle_signed_barcode_image(sb;
                xs=sub.xs,
                ys=sub.ys,
                sigma=sub.sigma,
                mode=sub.mode,
                threads=threads_g)
            parts[idx] = _float_vector(img)
            filled[idx] = true
        end
    end
    return nothing
end

@inline _is_euler_family_spec(spec) = spec isa EulerSurfaceSpec

function _fill_composite_euler_parts!(
    parts::Vector{Vector{Float64}},
    filled::BitVector,
    spec::CompositeSpec,
    cache::EncodingInvariantCache,
    opts0::InvariantOptions,
    threaded0::Bool,
)
    groups = Dict{UInt,Vector{Int}}()
    for i in eachindex(spec.specs)
        sub = spec.specs[i]
        _is_euler_family_spec(sub) || continue
        threads_i = _resolve_spec_threads(sub.threads, opts0, threaded0)
        axes_i = sub.axes === nothing ? opts0.axes : sub.axes
        strict_i = sub.strict === nothing ? opts0.strict : sub.strict
        key = UInt(hash((axes_i, sub.axes_policy, sub.max_axis_len, opts0.box, strict_i, threads_i)))
        push!(get!(groups, key) do
            Int[]
        end, i)
    end

    isempty(groups) && return nothing
    h = Invariants.restricted_hilbert(cache.M)
    for idxs in values(groups)
        first_spec = spec.specs[first(idxs)]::EulerSurfaceSpec
        threads_g = _resolve_spec_threads(first_spec.threads, opts0, threaded0)
        vals = _eulersurface_vector(h, cache.pi, first_spec, opts0, threads_g)
        for idx in idxs
            parts[idx] = vals
            filled[idx] = true
        end
    end
    return nothing
end

function _fill_composite_euler_parts!(
    parts::Vector{Vector{Float64}},
    filled::BitVector,
    spec::CompositeSpec,
    cache::RestrictedHilbertInvariantCache,
    opts0::InvariantOptions,
    threaded0::Bool,
)
    groups = Dict{UInt,Vector{Int}}()
    for i in eachindex(spec.specs)
        sub = spec.specs[i]
        _is_euler_family_spec(sub) || continue
        threads_i = _resolve_spec_threads(sub.threads, opts0, threaded0)
        axes_i = sub.axes === nothing ? opts0.axes : sub.axes
        strict_i = sub.strict === nothing ? opts0.strict : sub.strict
        key = UInt(hash((axes_i, sub.axes_policy, sub.max_axis_len, opts0.box, strict_i, threads_i)))
        push!(get!(groups, key) do
            Int[]
        end, i)
    end

    for idxs in values(groups)
        first_spec = spec.specs[first(idxs)]::EulerSurfaceSpec
        threads_g = _resolve_spec_threads(first_spec.threads, opts0, threaded0)
        vals = _eulersurface_vector(cache.hilbert, cache.pi, first_spec, opts0, threads_g)
        for idx in idxs
            parts[idx] = vals
            filled[idx] = true
        end
    end
    return nothing
end

@inline _is_rank_family_spec(spec) = spec isa RankGridSpec

function _fill_composite_rank_parts!(
    parts::Vector{Vector{Float64}},
    filled::BitVector,
    spec::CompositeSpec,
    cache::AbstractInvariantCache,
    opts0::InvariantOptions,
)
    any(_is_rank_family_spec, spec.specs) || return nothing
    M = _sample_module(_cache_sample(cache))
    groups = Dict{UInt,Vector{Int}}()
    for i in eachindex(spec.specs)
        sub = spec.specs[i]
        _is_rank_family_spec(sub) || continue
        key = UInt(hash((sub.nvertices, sub.store_zeros)))
        push!(get!(groups, key) do
            Int[]
        end, i)
    end

    for idxs in values(groups)
        first_spec = spec.specs[first(idxs)]::RankGridSpec
        vals = _rankgrid_vector(M, first_spec, opts0)
        for idx in idxs
            parts[idx] = vals
            filled[idx] = true
        end
    end
    return nothing
end

function transform(spec::CompositeSpec, obj; opts::InvariantOptions=InvariantOptions(), threaded::Bool=true)
    cache = build_cache(obj, spec; opts=opts, threaded=threaded)
    return transform(spec, cache; opts=opts, threaded=threaded)
end

function transform(spec::CompositeSpec,
                   cache::AbstractInvariantCache;
                   opts::InvariantOptions=InvariantOptions(),
                   threaded::Bool=true)
    opts0 = _cache_opts(cache, opts)
    threaded0 = _cache_threaded(cache, threaded)
    _prime_cache!(cache, spec; opts=opts0, threaded=threaded0)
    parts = Vector{Vector{Float64}}(undef, length(spec.specs))
    filled = falses(length(spec.specs))

    if cache isa EncodingInvariantCache
        _fill_composite_slice_parts!(parts, filled, spec, cache, opts0, threaded0)
        _fill_composite_mplandscape_parts!(parts, filled, spec, cache, opts0, threaded0)
        _fill_composite_projected_parts!(parts, filled, spec, cache, opts0, threaded0)
        _fill_composite_mpp_image_parts!(parts, filled, spec, cache, opts0, threaded0)
        _fill_composite_mpp_hist_parts!(parts, filled, spec, cache, opts0, threaded0)
        _fill_composite_signed_barcode_parts!(parts, filled, spec, cache, opts0, threaded0)
        _fill_composite_euler_parts!(parts, filled, spec, cache, opts0, threaded0)
    elseif cache isa RestrictedHilbertInvariantCache
        _fill_composite_euler_parts!(parts, filled, spec, cache, opts0, threaded0)
    end
    _fill_composite_rank_parts!(parts, filled, spec, cache, opts0)

    total = 0
    @inbounds for i in eachindex(spec.specs)
        if !filled[i]
            parts[i] = transform(spec.specs[i], cache; opts=opts0, threaded=threaded0)
        end
        v = parts[i]
        total += length(v)
    end
    out = Vector{Float64}(undef, total)
    pos = 1
    @inbounds for v in parts
        copyto!(out, pos, v, 1, length(v))
        pos += length(v)
    end
    return out
end

function transform!(dest::AbstractVector, spec::AbstractFeaturizerSpec, obj;
                    opts::InvariantOptions=InvariantOptions(),
                    threaded::Bool=true)
    vals = transform(spec, obj; opts=opts, threaded=threaded)
    length(dest) == length(vals) || throw(DimensionMismatch("destination length $(length(dest)) != feature length $(length(vals))"))
    copyto!(dest, vals)
    return dest
end

function _feature_names_or_default(spec::AbstractFeaturizerSpec, n::Int)
    names = feature_names(spec)
    isempty(names) && return [Symbol("f$(i)") for i in 1:n]
    length(names) == n || throw(DimensionMismatch("feature_names(spec) has length $(length(names)) but transform output has length $n"))
    return names
end

function _resolve_feature_session_cache(cache)
    if cache === :auto
        return SessionCache(), :session_auto
    elseif cache === nothing
        return nothing, :none
    elseif cache isa SessionCache
        return cache, :session_user
    else
        throw(ArgumentError("cache must be :auto, nothing, or SessionCache"))
    end
end

@inline function _batch_foreach_serial(n::Int, f)
    @inbounds for i in 1:n
        f(i)
    end
    return nothing
end

function _batch_foreach_threads(n::Int, f; chunk_size::Int=0, deterministic::Bool=true)
    n <= 1 && return _batch_foreach_serial(n, f)
    nt = Threads.nthreads()
    nt <= 1 && return _batch_foreach_serial(n, f)

    if deterministic
        if chunk_size > 0
            nchunks = cld(n, chunk_size)
            Threads.@threads for c in 1:nchunks
                lo = (c - 1) * chunk_size + 1
                hi = min(n, c * chunk_size)
                @inbounds for i in lo:hi
                    f(i)
                end
            end
            return nothing
        end

        nt_eff = min(nt, n)
        base = fld(n, nt_eff)
        remn = n - base * nt_eff
        Threads.@threads for t in 1:nt_eff
            extra = t <= remn ? 1 : 0
            start = (t - 1) * base + min(t - 1, remn) + 1
            stop = start + base + extra - 1
            @inbounds for i in start:stop
                f(i)
            end
        end
        return nothing
    end

    Threads.@threads for i in 1:n
        f(i)
    end
    return nothing
end

@inline function _batch_foreach(n::Int, opts::BatchOptions, f)
    if !opts.threaded || n <= 1 || opts.backend == :serial
        return _batch_foreach_serial(n, f)
    elseif opts.backend == :threads
        return _batch_foreach_threads(n, f; chunk_size=opts.chunk_size, deterministic=opts.deterministic)
    elseif opts.backend == :folds
        impl = _BATCH_IMPL[]
        if impl !== nothing
            return impl.foreach_indexed(n, f; chunk_size=opts.chunk_size, deterministic=opts.deterministic)
        end
        # Fallback keeps behavior usable when Folds extension is not loaded.
        return _batch_foreach_threads(n, f; chunk_size=opts.chunk_size, deterministic=opts.deterministic)
    end
    throw(ArgumentError("unsupported batch backend: $(opts.backend)"))
end

@inline _batch_foreach(f, n::Int, opts::BatchOptions) = _batch_foreach(n, opts, f)

const _BATCH_IMPL = Ref{Any}(nothing)
@inline _set_batch_impl!(impl) = (_BATCH_IMPL[] = impl; nothing)

const _PROGRESS_IMPL = Ref{Any}(nothing)
@inline _set_progress_impl!(impl) = (_PROGRESS_IMPL[] = impl; nothing)

const _KERNELFUNCTIONS_IMPL = Ref{Any}(nothing)
@inline _set_kernelfunctions_impl!(impl) = (_KERNELFUNCTIONS_IMPL[] = impl; nothing)

const _DISTANCES_IMPL = Ref{Any}(nothing)
@inline _set_distances_impl!(impl) = (_DISTANCES_IMPL[] = impl; nothing)

"""
    mp_landscape_kernel_object(; kwargs...)
    projected_kernel_object(; kwargs...)
    mpp_image_kernel_object(; kwargs...)
    point_signed_measure_kernel_object(; kwargs...)
    rectangle_signed_barcode_kernel_object(; kwargs...)

Construct KernelFunctions-compatible kernel wrapper objects (provided by the
KernelFunctions extension). These wrappers can be passed to
`KernelFunctions.kernelmatrix` for pairwise kernel evaluation on vectors of
compatible objects.
"""
@inline function mp_landscape_kernel_object(; kwargs...)
    impl = _KERNELFUNCTIONS_IMPL[]
    impl === nothing && throw(ArgumentError("mp_landscape_kernel_object requires KernelFunctions.jl extension (load KernelFunctions and ensure TamerOpKernelFunctionsExt is available)."))
    return impl.mp_landscape(; kwargs...)
end

@inline function projected_kernel_object(; kwargs...)
    impl = _KERNELFUNCTIONS_IMPL[]
    impl === nothing && throw(ArgumentError("projected_kernel_object requires KernelFunctions.jl extension (load KernelFunctions and ensure TamerOpKernelFunctionsExt is available)."))
    return impl.projected(; kwargs...)
end

@inline function mpp_image_kernel_object(; kwargs...)
    impl = _KERNELFUNCTIONS_IMPL[]
    impl === nothing && throw(ArgumentError("mpp_image_kernel_object requires KernelFunctions.jl extension (load KernelFunctions and ensure TamerOpKernelFunctionsExt is available)."))
    return impl.mpp_image(; kwargs...)
end

@inline function point_signed_measure_kernel_object(; kwargs...)
    impl = _KERNELFUNCTIONS_IMPL[]
    impl === nothing && throw(ArgumentError("point_signed_measure_kernel_object requires KernelFunctions.jl extension (load KernelFunctions and ensure TamerOpKernelFunctionsExt is available)."))
    return impl.point_signed_measure(; kwargs...)
end

@inline function rectangle_signed_barcode_kernel_object(; kwargs...)
    impl = _KERNELFUNCTIONS_IMPL[]
    impl === nothing && throw(ArgumentError("rectangle_signed_barcode_kernel_object requires KernelFunctions.jl extension (load KernelFunctions and ensure TamerOpKernelFunctionsExt is available)."))
    return impl.rectangle_signed_barcode(; kwargs...)
end

"""
    matching_distance_metric(; kwargs...)
    mp_landscape_distance_metric(; kwargs...)
    projected_distance_metric(; kwargs...)
    bottleneck_distance_metric(; kwargs...)
    wasserstein_distance_metric(; kwargs...)
    mpp_image_distance_metric(; kwargs...)

Construct Distances.jl `PreMetric` wrappers around TamerOp distances.
These can be used with `Distances.evaluate` and `Distances.pairwise`.
"""
@inline function matching_distance_metric(; kwargs...)
    impl = _DISTANCES_IMPL[]
    impl === nothing && throw(ArgumentError("matching_distance_metric requires Distances.jl extension (load Distances and ensure TamerOpDistancesExt is available)."))
    return impl.matching(; kwargs...)
end

@inline function mp_landscape_distance_metric(; kwargs...)
    impl = _DISTANCES_IMPL[]
    impl === nothing && throw(ArgumentError("mp_landscape_distance_metric requires Distances.jl extension (load Distances and ensure TamerOpDistancesExt is available)."))
    return impl.mp_landscape(; kwargs...)
end

@inline function projected_distance_metric(; kwargs...)
    impl = _DISTANCES_IMPL[]
    impl === nothing && throw(ArgumentError("projected_distance_metric requires Distances.jl extension (load Distances and ensure TamerOpDistancesExt is available)."))
    return impl.projected(; kwargs...)
end

@inline function bottleneck_distance_metric(; kwargs...)
    impl = _DISTANCES_IMPL[]
    impl === nothing && throw(ArgumentError("bottleneck_distance_metric requires Distances.jl extension (load Distances and ensure TamerOpDistancesExt is available)."))
    return impl.bottleneck(; kwargs...)
end

@inline function wasserstein_distance_metric(; kwargs...)
    impl = _DISTANCES_IMPL[]
    impl === nothing && throw(ArgumentError("wasserstein_distance_metric requires Distances.jl extension (load Distances and ensure TamerOpDistancesExt is available)."))
    return impl.wasserstein(; kwargs...)
end

@inline function mpp_image_distance_metric(; kwargs...)
    impl = _DISTANCES_IMPL[]
    impl === nothing && throw(ArgumentError("mpp_image_distance_metric requires Distances.jl extension (load Distances and ensure TamerOpDistancesExt is available)."))
    return impl.mpp_image(; kwargs...)
end

@inline function _progress_init(total::Int, label::AbstractString, enabled::Bool)
    enabled || return nothing
    impl = _PROGRESS_IMPL[]
    impl === nothing && return nothing
    return impl.init(total, label)
end

@inline function _progress_step!(state, delta::Int=1)
    impl = _PROGRESS_IMPL[]
    (state === nothing || impl === nothing) && return nothing
    return impl.step!(state, delta)
end

@inline function _progress_finish!(state)
    impl = _PROGRESS_IMPL[]
    (state === nothing || impl === nothing) && return nothing
    return impl.finish!(state)
end

struct _CompiledFeaturizePlan{FS<:AbstractFeaturizerSpec,AX,LAB}
    spec::FS
    opts::InvariantOptions
    on_unsupported::Symbol
    cache_mode::Symbol
    batch::BatchOptions
    feature_names::Vector{Symbol}
    feature_axes::AX
    ids::Vector{String}
    labels::LAB
    skipped_indices::Vector{Int}
    session_cache::Union{Nothing,SessionCache}
    sample_indices::Vector{Int}
    output_rows::Vector{Int}
    caches::Vector{_InvariantCacheHandle}
    first_slot::Int
    first_values::Vector{Float64}
    nrows::Int
    nfeatures::Int
    allow_missing::Bool
end

@inline function _feature_ids(samples::AbstractVector, idfun, idxs::AbstractVector{Int})
    if idfun === nothing
        return [string(i) for i in idxs]
    end
    return [string(idfun(samples[i])) for i in idxs]
end

@inline function _feature_labels(samples::AbstractVector, labelfun, idxs::AbstractVector{Int})
    labelfun === nothing && return nothing
    return [labelfun(samples[i]) for i in idxs]
end

function _compile_featurize(samples::AbstractVector,
                            spec::AbstractFeaturizerSpec,
                            opts0::InvariantOptions,
                            batch_opts::BatchOptions,
                            cache_mode::Symbol,
                            on_unsupported::Symbol,
                            session_cache::Union{Nothing,SessionCache},
                            idfun,
                            labelfun)
    ns = length(samples)
    axes_spec = feature_axes(spec)

    supported = BitVector(undef, ns)
    first_supported = 0
    @inbounds for i in 1:ns
        ok = supports(spec, samples[i])
        supported[i] = ok
        if ok && first_supported == 0
            first_supported = i
        end
    end

    if on_unsupported == :error
        bad = findfirst(x -> !x, supported)
        bad === nothing || throw(ArgumentError("sample at index $(bad) is unsupported for $(typeof(spec))"))
    end

    skipped = findall(x -> !x, supported)
    all_idxs = collect(1:ns)

    if first_supported == 0
        nf = nfeatures(spec)
        names = _feature_names_or_default(spec, nf)
        if on_unsupported == :skip
            labels = labelfun === nothing ? nothing : Any[]
            return _CompiledFeaturizePlan(
                spec, opts0, on_unsupported, cache_mode, batch_opts,
                names, axes_spec, String[], labels, skipped, session_cache,
                Int[], Int[], _InvariantCacheHandle[], 0, Float64[], 0, nf, false,
            )
        end
        return _CompiledFeaturizePlan(
            spec, opts0, on_unsupported, cache_mode, batch_opts,
            names, axes_spec, _feature_ids(samples, idfun, all_idxs),
            _feature_labels(samples, labelfun, all_idxs), skipped, session_cache,
            Int[], Int[], _InvariantCacheHandle[], 0, Float64[], ns, nf, true,
        )
    end

    sample_indices = findall(supported)
    output_rows = if on_unsupported == :skip
        collect(1:length(sample_indices))
    else
        copy(sample_indices)
    end
    ids = on_unsupported == :skip ?
        _feature_ids(samples, idfun, sample_indices) :
        _feature_ids(samples, idfun, all_idxs)
    labels = on_unsupported == :skip ?
        _feature_labels(samples, labelfun, sample_indices) :
        _feature_labels(samples, labelfun, all_idxs)

    caches = Vector{_InvariantCacheHandle}(undef, length(sample_indices))
    first_slot = 0
    @inbounds for slot in eachindex(sample_indices)
        sidx = sample_indices[slot]
        cache = build_cache(samples[sidx], spec;
                            opts=opts0,
                            threaded=batch_opts.threaded,
                            cache=session_cache)
        caches[slot] = cache
        if sidx == first_supported
            first_slot = slot
        end
    end
    first_slot > 0 || error("_compile_featurize: internal error (missing first supported slot)")

    first_values = transform(spec, caches[first_slot]; opts=opts0, threaded=batch_opts.threaded)
    nf = length(first_values)
    names = _feature_names_or_default(spec, nf)
    nrows = on_unsupported == :skip ? length(sample_indices) : ns
    allow_missing = on_unsupported == :missing

    return _CompiledFeaturizePlan(
        spec, opts0, on_unsupported, cache_mode, batch_opts,
        names, axes_spec, ids, labels, skipped, session_cache,
        sample_indices, output_rows, caches, first_slot, first_values, nrows, nf, allow_missing,
    )
end

function _run_featurize!(X, compiled::_CompiledFeaturizePlan, batch_opts::BatchOptions)
    ns = length(compiled.sample_indices)
    ns == 0 && return X

    pstate = _progress_init(ns, "featurize", batch_opts.progress)
    row0 = compiled.output_rows[compiled.first_slot]
    @inbounds copyto!(view(X, row0, :), compiled.first_values)
    _progress_step!(pstate, 1)

    threaded_run = (
        batch_opts.threaded &&
        batch_opts.backend != :serial &&
        Threads.nthreads() > 1 &&
        ns > 1
    )

    _batch_foreach(ns, batch_opts) do slot
        slot == compiled.first_slot && return
        row = compiled.output_rows[slot]
        cache = compiled.caches[slot]
        transform!(view(X, row, :), compiled.spec, cache;
                   opts=compiled.opts,
                   threaded=batch_opts.threaded)
        if !threaded_run
            _progress_step!(pstate, 1)
        end
    end
    if threaded_run && ns > 1
        _progress_step!(pstate, ns - 1)
    end
    _progress_finish!(pstate)
    return X
end

"""
    featurize(samples, spec; opts=InvariantOptions(), idfun=nothing, labelfun=nothing,
              batch=BatchOptions(), cache=:auto, on_unsupported=:error) -> FeatureSet

Apply a typed featurizer spec to a batch of samples and return a matrix with
row-major sample layout (`nsamples x nfeatures`).
"""
function featurize(samples::AbstractVector,
                   spec::AbstractFeaturizerSpec;
                   opts::InvariantOptions=InvariantOptions(),
                   idfun=nothing,
                   labelfun=nothing,
                   batch::BatchOptions=BatchOptions(),
                   cache=:auto,
                   on_unsupported::Symbol=:error)
    opts0 = opts
    batch_opts = batch
    ns = length(samples)
    session_cache, cache_mode = _resolve_feature_session_cache(cache)
    (on_unsupported == :error || on_unsupported == :skip || on_unsupported == :missing) ||
        throw(ArgumentError("on_unsupported must be :error, :skip, or :missing"))

    if ns == 0
        nf = nfeatures(spec)
        X0 = zeros(Float64, 0, nf)
        names0 = _feature_names_or_default(spec, nf)
        meta0 = (spec=spec, opts=opts0, cache_mode=cache_mode, threaded=batch_opts.threaded, batch=batch_opts,
                 feature_axes=feature_axes(spec),
                 labels=nothing, unsupported_policy=on_unsupported,
                 skipped_indices=Int[])
        return FeatureSet(X0, names0, String[], meta0)
    end

    compiled = _compile_featurize(samples, spec, opts0, batch_opts, cache_mode,
                                  on_unsupported, session_cache, idfun, labelfun)
    X = if compiled.allow_missing
        Matrix{Union{Missing,Float64}}(missing, compiled.nrows, compiled.nfeatures)
    else
        Matrix{Float64}(undef, compiled.nrows, compiled.nfeatures)
    end
    _run_featurize!(X, compiled, batch_opts)

    meta = (spec=compiled.spec,
            opts=compiled.opts,
            cache_mode=compiled.cache_mode,
            threaded=batch_opts.threaded,
            batch=batch_opts,
            feature_axes=compiled.feature_axes,
            labels=compiled.labels,
            session_cache=compiled.session_cache,
            unsupported_policy=compiled.on_unsupported,
            skipped_indices=compiled.skipped_indices)
    return FeatureSet(X, compiled.feature_names, compiled.ids, meta)
end

"""
    batch_transform(samples, spec; kwargs...) -> FeatureSet

Explicit dataset-level batch transform entrypoint. Semantics are identical to
`featurize` and it returns the same `FeatureSet` payload.
"""
function batch_transform(samples::AbstractVector,
                         spec::AbstractFeaturizerSpec;
                         opts::InvariantOptions=InvariantOptions(),
                         idfun=nothing,
                         labelfun=nothing,
                         batch::BatchOptions=BatchOptions(),
                         cache=:auto,
                         on_unsupported::Symbol=:error)
    return featurize(samples, spec;
                     opts=opts,
                     idfun=idfun,
                     labelfun=labelfun,
                     batch=batch,
                     cache=cache,
                     on_unsupported=on_unsupported)
end

"""
    batch_transform!(X, samples, spec; kwargs...) -> FeatureSet

In-place dataset-level batch transform. `X` must have shape
`(nrows, nfeatures)` consistent with the chosen unsupported-sample policy.
Returns a `FeatureSet` wrapper around the provided matrix.
"""
function batch_transform!(X::Matrix,
                          samples::AbstractVector,
                          spec::AbstractFeaturizerSpec;
                          opts::InvariantOptions=InvariantOptions(),
                          idfun=nothing,
                          labelfun=nothing,
                          batch::BatchOptions=BatchOptions(),
                          cache=:auto,
                          on_unsupported::Symbol=:error)
    opts0 = opts
    batch_opts = batch
    session_cache, cache_mode = _resolve_feature_session_cache(cache)
    (on_unsupported == :error || on_unsupported == :skip || on_unsupported == :missing) ||
        throw(ArgumentError("on_unsupported must be :error, :skip, or :missing"))

    compiled = _compile_featurize(samples, spec, opts0, batch_opts, cache_mode,
                                  on_unsupported, session_cache, idfun, labelfun)
    size(X, 1) == compiled.nrows ||
        throw(DimensionMismatch("batch_transform!: destination has $(size(X,1)) rows, expected $(compiled.nrows)"))
    size(X, 2) == compiled.nfeatures ||
        throw(DimensionMismatch("batch_transform!: destination has $(size(X,2)) columns, expected $(compiled.nfeatures)"))

    if compiled.allow_missing
        has_missing_support = Missing <: eltype(X)
        if !has_missing_support && compiled.nrows != length(compiled.sample_indices)
            throw(ArgumentError("batch_transform!: destination element type $(eltype(X)) cannot represent missing values required by on_unsupported=:missing"))
        end
        has_missing_support && fill!(X, missing)
    end

    _run_featurize!(X, compiled, batch_opts)
    meta = (spec=compiled.spec,
            opts=compiled.opts,
            cache_mode=compiled.cache_mode,
            threaded=batch_opts.threaded,
            batch=batch_opts,
            feature_axes=compiled.feature_axes,
            labels=compiled.labels,
            session_cache=compiled.session_cache,
            unsupported_policy=compiled.on_unsupported,
            skipped_indices=compiled.skipped_indices)
    return FeatureSet(X, compiled.feature_names, compiled.ids, meta)
end

@inline function _resolution_cache_stats(rc::ResolutionCache)
    return Dict(
        "projective" => length(rc.projective),
        "injective" => length(rc.injective),
        "indicator" => length(rc.indicator),
        "projective_shards" => sum(length, rc.projective_shards),
        "injective_shards" => sum(length, rc.injective_shards),
        "indicator_shards" => sum(length, rc.indicator_shards),
    )
end

function _session_cache_stats(session::SessionCache)
    n_encoding = _session_encoding_bucket_count(session)
    n_modules = _session_module_bucket_count(session)
    enc_posets = 0
    enc_cubical = 0
    enc_region_posets = 0
    mod_payload = 0
    mod_resolution_projective = 0
    mod_resolution_injective = 0
    mod_resolution_indicator = 0
    for ec in _session_encoding_values(session)
        enc_posets += length(ec.posets)
        enc_cubical += length(ec.cubical)
        enc_region_posets += length(ec.region_posets)
    end
    for mc in _session_module_values(session)
        mod_payload += length(mc.payload)
        mod_resolution_projective += length(mc.resolution.projective)
        mod_resolution_injective += length(mc.resolution.injective)
        mod_resolution_indicator += length(mc.resolution.indicator)
    end
    out = Dict{String,Any}(
        "encoding_buckets" => n_encoding,
        "module_buckets" => n_modules,
        "encoding_posets" => enc_posets,
        "encoding_cubical" => enc_cubical,
        "encoding_region_posets" => enc_region_posets,
        "module_payload_entries" => mod_payload,
        "module_resolution_projective" => mod_resolution_projective,
        "module_resolution_injective" => mod_resolution_injective,
        "module_resolution_indicator" => mod_resolution_indicator,
        "zn_encoding_artifacts" => _session_zn_encoding_artifact_count(session),
        "zn_pushforward_plans" => _session_zn_pushforward_plan_count(session),
        "zn_pushforward_fringes" => _session_zn_pushforward_fringe_count(session),
        "zn_pushforward_modules" => _session_zn_pushforward_module_count(session),
        "session_resolution" => _resolution_cache_stats(session.resolution),
        "has_hom_system_cache" => session.hom_system !== nothing,
        "has_slice_plan_cache" => session.slice_plan !== nothing,
        "product_dense_entries" => length(session.product_dense),
        "product_object_entries" => length(session.product_obj),
    )
    return out
end

@inline function _experiment_cache_handle(cache)
    if cache === :auto
        return SessionCache(), :session_auto_shared
    elseif cache === nothing
        return nothing, :none
    elseif cache isa SessionCache
        return cache, :session_user
    end
    throw(ArgumentError("run_experiment: cache must be :auto, nothing, or SessionCache"))
end

@inline function _sanitize_filename(s::AbstractString)
    out = IOBuffer()
    for c in s
        if isletter(c) || isnumeric(c) || c == '_' || c == '-' || c == '.'
            print(out, c)
        else
            print(out, '_')
        end
    end
    str = String(take!(out))
    return isempty(str) ? "item" : str
end

@inline function _featurizer_key(spec::AbstractFeaturizerSpec, i::Int)
    tname = String(nameof(typeof(spec)))
    return Symbol("s", lpad(string(i), 2, '0'), "__", _sanitize_filename(lowercase(tname)))
end

@inline function _resolve_run_dir(io::ExperimentIOConfig, exp_name::String)
    do_write = !isempty(io.formats) || io.write_metadata
    do_write || return nothing
    base = io.outdir === nothing ? joinpath(pwd(), "experiment_outputs") : io.outdir
    run_tag = _sanitize_filename(io.prefix * "__" * exp_name)
    run_dir = joinpath(base, run_tag)
    if isdir(run_dir) && !io.overwrite
        throw(ArgumentError("run_experiment: output directory exists and overwrite=false: $(run_dir)"))
    end
    mkpath(run_dir)
    return run_dir
end

@inline function _run_one_experiment_featurizer(samples::AbstractVector,
                                                spec::AbstractFeaturizerSpec,
                                                opts::InvariantOptions,
                                                batch::BatchOptions,
                                                cache,
                                                on_unsupported::Symbol,
                                                idfun,
                                                labelfun)
    t0 = time_ns()
    fs = featurize(samples, spec;
                   opts=opts,
                   batch=batch,
                   cache=cache,
                   on_unsupported=on_unsupported,
                   idfun=idfun,
                   labelfun=labelfun)
    elapsed = (time_ns() - t0) / 1.0e9
    return fs, elapsed
end

"""
    run_experiment(exp::ExperimentSpec, samples) -> ExperimentResult

Run a featurization experiment over `samples` with one or more featurizers.

Behavior:
- returns in-memory `FeatureSet` outputs + timing/cache metadata,
- optionally writes Arrow/Parquet feature tables and JSON sidecars,
- writes a run manifest JSON when metadata writing is enabled.
"""
function run_experiment(exp::ExperimentSpec, samples::AbstractVector)
    run_dir = _resolve_run_dir(exp.io, exp.name)
    cache, cache_mode = _experiment_cache_handle(exp.cache)

    artifacts = Vector{ExperimentArtifact}(undef, length(exp.featurizers))
    t_total0 = time_ns()

    for i in eachindex(exp.featurizers)
        sp = exp.featurizers[i]
        key = _featurizer_key(sp, i)
        fs, elapsed = _run_one_experiment_featurizer(samples, sp, exp.opts, exp.batch, cache,
                                                     exp.on_unsupported, exp.idfun, exp.labelfun)
        cstats = cache isa SessionCache ? _session_cache_stats(cache) : Dict{String,Any}()
        feature_paths = Dict{Symbol,String}()
        metadata_path = nothing

        if run_dir !== nothing
            stem = _sanitize_filename(String(key))
            for fmt in exp.io.formats
                if fmt == :arrow
                    apath = joinpath(run_dir, stem * ".arrow")
                    save_features_arrow(apath, fs; format=exp.io.format, include_metadata=false)
                    feature_paths[:arrow] = apath
                elseif fmt == :parquet
                    ppath = joinpath(run_dir, stem * ".parquet")
                    save_features_parquet(ppath, fs; format=exp.io.format, include_metadata=false)
                    feature_paths[:parquet] = ppath
                elseif fmt == :npz
                    npath = joinpath(run_dir, stem * ".npz")
                    save_features_npz(npath, fs; format=exp.io.format, layout=:samples_by_features, include_metadata=false)
                    feature_paths[:npz] = npath
                elseif fmt == :csv_wide
                    cpath = joinpath(run_dir, stem * "__wide.csv")
                    save_features_csv(cpath, fs; format=:wide, layout=:samples_by_features, include_metadata=false)
                    feature_paths[:csv_wide] = cpath
                elseif fmt == :csv_long
                    cpath = joinpath(run_dir, stem * "__long.csv")
                    save_features_csv(cpath, fs; format=:long, layout=:samples_by_features, include_metadata=false)
                    feature_paths[:csv_long] = cpath
                else
                    throw(ArgumentError("run_experiment: unsupported output format $(fmt)"))
                end
            end
            if exp.io.write_metadata
                md = feature_metadata(fs; format=exp.io.format)
                md["kind"] = "experiment_feature"
                md["experiment_name"] = exp.name
                md["experiment_schema_version"] = string(TAMER_EXPERIMENT_SCHEMA_VERSION)
                md["featurizer_key"] = String(key)
                md["elapsed_seconds"] = elapsed
                md["cache_mode"] = String(cache_mode)
                md["cache_stats"] = _jsonable(cstats)
                md["batch"] = _jsonable(exp.batch)
                md["on_unsupported"] = String(exp.on_unsupported)
                md["artifact_formats"] = [String(f) for f in exp.io.formats]
                md["artifact_layout"] = "samples_by_features"
                md["feature_paths"] = _jsonable(feature_paths)
                metadata_path = joinpath(run_dir, stem * ".meta.json")
                save_metadata_json(metadata_path, md)
            end
        end

        artifacts[i] = ExperimentArtifact(key, sp, fs, elapsed, cstats, feature_paths, metadata_path)
    end

    total_elapsed = (time_ns() - t_total0) / 1.0e9
    manifest_path = nothing
    result_meta = Dict{String,Any}(
        "experiment_name" => exp.name,
        "experiment_schema_version" => string(TAMER_EXPERIMENT_SCHEMA_VERSION),
        "feature_schema_version" => string(TAMER_FEATURE_SCHEMA_VERSION),
        "n_featurizers" => length(exp.featurizers),
        "n_samples" => length(samples),
        "cache_mode" => String(cache_mode),
        "total_elapsed_seconds" => total_elapsed,
        "started_at_utc" => string(now(UTC)),
        "metadata" => _jsonable(exp.metadata),
    )
    if cache isa SessionCache
        result_meta["final_cache_stats"] = _jsonable(_session_cache_stats(cache))
    end

    if run_dir !== nothing && exp.io.write_metadata
        manifest = Dict{String,Any}(
            "kind" => "experiment_manifest",
            "experiment_name" => exp.name,
            "experiment_schema_version" => string(TAMER_EXPERIMENT_SCHEMA_VERSION),
            "feature_schema_version" => string(TAMER_FEATURE_SCHEMA_VERSION),
            "total_elapsed_seconds" => total_elapsed,
            "batch" => _jsonable(exp.batch),
            "cache_mode" => String(cache_mode),
            "on_unsupported" => String(exp.on_unsupported),
            "table_format" => String(exp.io.format),
            "artifact_formats" => [String(f) for f in exp.io.formats],
            "artifact_layout" => "samples_by_features",
            "run_dir" => run_dir,
            "artifacts" => [
                Dict(
                    "key" => String(a.key),
                    "featurizer_type" => string(nameof(typeof(a.spec))),
                    "elapsed_seconds" => a.elapsed_seconds,
                    "feature_paths" => _jsonable(a.feature_paths),
                    "metadata_path" => a.metadata_path,
                    "cache_stats" => _jsonable(a.cache_stats),
                    "n_features" => nfeatures(a.features),
                    "n_rows" => nsamples(a.features),
                ) for a in artifacts
            ],
            "metadata" => _jsonable(exp.metadata),
        )
        manifest_path = joinpath(run_dir, "manifest.json")
        save_metadata_json(manifest_path, manifest)
    end

    return ExperimentResult(exp, artifacts, total_elapsed, run_dir, manifest_path, result_meta)
end

function run_experiment(featurizers,
                        samples::AbstractVector;
                        name::AbstractString="experiment",
                        opts::InvariantOptions=InvariantOptions(),
                        batch::Union{BatchOptions,Nothing}=nothing,
                        cache=:auto,
                        on_unsupported::Symbol=:error,
                        idfun=nothing,
                        labelfun=nothing,
                        io::ExperimentIOConfig=ExperimentIOConfig(),
                        metadata=NamedTuple())
    exp = ExperimentSpec(featurizers;
                         name=name,
                         opts=opts,
                         batch=batch,
                         cache=cache,
                         on_unsupported=on_unsupported,
                         idfun=idfun,
                         labelfun=labelfun,
                         io=io,
                         metadata=metadata)
    return run_experiment(exp, samples)
end

@inline function _resolve_manifest_path(path::AbstractString)
    if isdir(path)
        mp = joinpath(path, "manifest.json")
        isfile(mp) || throw(ArgumentError("load_experiment: manifest.json not found in directory $(path)"))
        return mp
    end
    isfile(path) || throw(ArgumentError("load_experiment: path does not exist: $(path)"))
    return String(path)
end

@inline function _abspath_or_join(base::Union{Nothing,AbstractString}, path::AbstractString)
    if isabspath(path)
        return String(path)
    end
    base === nothing && return String(path)
    return joinpath(String(base), String(path))
end

@inline function _feature_paths_from_manifest(entry, run_dir::Union{Nothing,String})
    out = Dict{Symbol,String}()
    raw = _obj_get(entry, "feature_paths", nothing)
    raw === nothing && return out
    for (k, v) in pairs(raw)
        ks = Symbol(String(k))
        vs = String(v)
        out[ks] = _abspath_or_join(run_dir, vs)
    end
    return out
end

@inline function _pick_feature_path(paths::Dict{Symbol,String}, prefer::Symbol)
    if isempty(paths)
        return nothing, nothing
    end
    if prefer == :arrow
        return get(paths, :arrow, nothing), :arrow
    elseif prefer == :parquet
        return get(paths, :parquet, nothing), :parquet
    elseif prefer == :npz
        return get(paths, :npz, nothing), :npz
    elseif prefer == :csv_wide
        return get(paths, :csv_wide, nothing), :csv_wide
    elseif prefer == :csv_long
        return get(paths, :csv_long, nothing), :csv_long
    elseif prefer == :csv
        if haskey(paths, :csv_wide)
            return paths[:csv_wide], :csv_wide
        elseif haskey(paths, :csv_long)
            return paths[:csv_long], :csv_long
        end
        return nothing, nothing
    elseif prefer == :auto
        if haskey(paths, :arrow)
            return paths[:arrow], :arrow
        elseif haskey(paths, :parquet)
            return paths[:parquet], :parquet
        elseif haskey(paths, :npz)
            return paths[:npz], :npz
        elseif haskey(paths, :csv_wide)
            return paths[:csv_wide], :csv_wide
        elseif haskey(paths, :csv_long)
            return paths[:csv_long], :csv_long
        end
        return nothing, nothing
    end
    throw(ArgumentError("load_experiment: prefer must be :auto, :arrow, :parquet, :npz, :csv, :csv_wide, :csv_long, or :none"))
end

@inline function _load_feature_file(path::AbstractString, fmt::Symbol;
                                    table_format::Symbol=:wide,
                                    ids_col::Symbol=:id)
    if fmt == :arrow
        return load_features_arrow(path; format=table_format, ids_col=ids_col)
    elseif fmt == :parquet
        return load_features_parquet(path; format=table_format, ids_col=ids_col)
    elseif fmt == :npz
        return load_features_npz(path; format=table_format)
    elseif fmt == :csv_wide
        return load_features_csv(path; format=:wide, ids_col=ids_col)
    elseif fmt == :csv_long
        return load_features_csv(path; format=:long, ids_col=ids_col)
    end
    throw(ArgumentError("_load_feature_file: unsupported feature format $(fmt)"))
end

"""
    load_experiment(path; load_features=true, prefer=:auto, strict=true,
                    resolve_ref=nothing, require_resolved_refs=false) -> LoadedExperimentResult

Load an experiment run from a manifest path (or run directory containing `manifest.json`).

`load_features` controls whether feature tables are loaded from persisted files.
`prefer` chooses file format when multiple exist (`:auto`, `:arrow`, `:parquet`,
`:npz`, `:csv`, `:csv_wide`, `:csv_long`, `:none`).
"""
function load_experiment(path::AbstractString;
                         load_features::Bool=true,
                         prefer::Symbol=:auto,
                         strict::Bool=true,
                         resolve_ref::Union{Nothing,Function}=nothing,
                         require_resolved_refs::Bool=false)
    prefer in (:auto, :arrow, :parquet, :npz, :csv, :csv_wide, :csv_long, :none) ||
        throw(ArgumentError("load_experiment: prefer must be :auto, :arrow, :parquet, :npz, :csv, :csv_wide, :csv_long, or :none"))

    manifest_path = _resolve_manifest_path(path)
    manifest_raw = load_metadata_json(manifest_path; validate_feature_schema=false)
    manifest = _to_plain(manifest_raw)
    kind = String(_obj_get(manifest, "kind", ""))
    kind == "experiment_manifest" ||
        throw(ArgumentError("load_experiment: expected kind=experiment_manifest, got $(kind)"))

    run_dir = _obj_get(manifest, "run_dir", dirname(manifest_path))
    run_dir = run_dir === nothing ? nothing : String(run_dir)
    entries = _obj_get(manifest, "artifacts", Any[])
    artifacts = Vector{LoadedExperimentArtifact}(undef, length(entries))

    for i in eachindex(entries)
        e = entries[i]
        key = Symbol(String(_obj_get(e, "key", "artifact_$(i)")))
        elapsed = Float64(_obj_get(e, "elapsed_seconds", 0.0))
        cstats_raw = _obj_get(e, "cache_stats", Dict{String,Any}())
        cstats_plain = _to_plain(cstats_raw)
        cstats = cstats_plain isa AbstractDict ? Dict{String,Any}(String(k)=>v for (k,v) in pairs(cstats_plain)) : Dict{String,Any}()

        metadata_path = _obj_get(e, "metadata_path", nothing)
        metadata_path = metadata_path === nothing ? nothing : _abspath_or_join(run_dir, String(metadata_path))
        md = nothing
        spec = nothing
        opts = nothing
        if metadata_path !== nothing && isfile(metadata_path)
            md = load_metadata_json(metadata_path; validate_feature_schema=false)
            if _obj_haskey(md, "kind")
                md_kind = String(md["kind"])
                if !(md_kind == "features" || md_kind == "experiment_feature")
                    strict && throw(ArgumentError("load_experiment: unsupported artifact metadata kind $(md_kind) for $(String(key))"))
                end
            end
            try
                spec = _obj_haskey(md, "spec") ? spec_from_metadata(md["spec"];
                                                                    resolve_ref=resolve_ref,
                                                                    require_resolved_refs=require_resolved_refs) : nothing
            catch err
                strict && rethrow(err)
                spec = nothing
            end
            try
                opts = _obj_haskey(md, "opts") ? invariant_options_from_metadata(md["opts"]) : nothing
            catch err
                strict && rethrow(err)
                opts = nothing
            end
        elseif strict
            throw(ArgumentError("load_experiment: missing metadata sidecar for artifact $(String(key))"))
        end

        feature_paths = _feature_paths_from_manifest(e, run_dir)
        feat = nothing
        table_format = if md !== nothing && _obj_haskey(md, "format")
            Symbol(String(md["format"]))
        else
            :wide
        end
        if load_features && prefer != :none
            fpath, ffmt = _pick_feature_path(feature_paths, prefer)
            if fpath === nothing
                strict && throw(ArgumentError("load_experiment: no feature file path found for artifact $(String(key))"))
            elseif isfile(fpath)
                try
                    feat = _load_feature_file(fpath, ffmt; table_format=table_format)
                catch err
                    strict && rethrow(err)
                    feat = nothing
                end
            elseif strict
                throw(ArgumentError("load_experiment: feature file not found for artifact $(String(key)): $(fpath)"))
            end
        end

        artifacts[i] = LoadedExperimentArtifact(
            key,
            spec,
            opts,
            feat,
            md,
            elapsed,
            cstats,
            feature_paths,
            metadata_path,
        )
    end

    total_elapsed = Float64(_obj_get(manifest, "total_elapsed_seconds", 0.0))
    return LoadedExperimentResult(manifest, artifacts, run_dir, manifest_path, total_elapsed)
end


"""
    matching_distance(encA, encB; method=:auto, opts=InvariantOptions(), kwargs...)

Distance between two encoded modules, assuming a common encoding map `pi` (as produced by
`encode([A,B]; ...)` style common-encoding).

`method`:
- `:auto` or `:approx`    uses `Invariants.matching_distance_approx`
- `:exact_2d`             uses `Invariants.matching_distance_exact_2d`
"""
function matching_distance(encA::EncodingResult, encB::EncodingResult;
                           method::Symbol=:auto,
                           opts::InvariantOptions=InvariantOptions(),
                           cache=:auto,
                           kwargs...)
    opts = opts
    (encA.P === encB.P) || error("matching_distance: encodings are on different posets; common-encode first.")
    (encA.pi === encB.pi) || error("matching_distance: encodings do not share a common classifier map pi; common-encode first.")

    pi = encA.pi
    MA = materialize_module(encA.M)
    MB = materialize_module(encB.M)
    cache_slice, session_cache = _resolve_workflow_specialized_cache(cache, SlicePlanCache)
    cache2 = _slice_plan_cache_from_session(cache_slice, session_cache)
    if method == :auto || method == :approx
        return Invariants.matching_distance_approx(MA, MB, pi, opts; cache=cache2, kwargs...)
    elseif method == :exact_2d
        return Invariants.matching_distance_exact_2d(MA, MB, pi, opts; kwargs...)
    else
        error("matching_distance: unknown method=$(method). Supported: :auto, :approx, :exact_2d")
    end
end

end # module Featurizers
