# =============================================================================
# DataIngestion.jl
#
# Data-ingestion and graded-complex builders.
# This module is a sibling of Workflow and extends `Workflow.encode`.
# =============================================================================

module DataIngestion

using LinearAlgebra
using SparseArrays
using JSON3
using Dates

using ..CoreModules: QQ, QQField, PrimeField, RealField, AbstractCoeffField, coeff_type, coerce,
                     ResolutionCache, SessionCache, EncodingCache,
                     _encoding_cache!, _session_resolution_cache, _session_hom_cache, _set_session_hom_cache!,
                     _session_slice_plan_cache, _set_session_slice_plan_cache!,
                     _resolve_workflow_session_cache, _resolve_workflow_specialized_cache,
                     _workflow_encoding_cache,
                     _resolution_cache_from_session, _slot_cache_from_session,
                     PosetCachePayload, CubicalCachePayload, GeometryCachePayload
using ..Options: EncodingOptions, ResolutionOptions, DerivedFunctorOptions, InvariantOptions,
                 FiltrationSpec, ConstructionBudget, ConstructionOptions, PipelineOptions, DataFileOptions
using ..DataTypes: PointCloud, ImageNd, GraphData, EmbeddedPlanarGraph2D,
                   GradedComplex, MultiCriticalGradedComplex,
                   SimplexTreeMulti, MatrixRowsView, point_matrix,
                   simplex_count, max_simplex_dim, simplex_vertices, simplex_grades,
                   _packed_cell_storage, _packed_multigrade_storage,
                   _packed_dim_count, _packed_dim_counts, _packed_total_cells,
                   _rebuild_graded_complex, _rebuild_multicritical_complex
using ..EncodingCore: AbstractPLikeEncodingMap, CompiledEncoding, compile_encoding, GridEncodingMap,
                      _compile_encoding_cached, _compile_encoding_without_reps
using ..Results: EncodingResult, EncodedComplexResult, CohomologyDimsResult, ResolutionResult, InvariantResult,
                 _encoding_with_session_cache
import ..Results: materialize_module, module_dims, _materialize_complex
import ..EncodingCore: locate, dimension, representatives, axes_from_encoding, _grid_strides, GridEncodingMap

import ..Serialization
import ..Serialization: TAMER_FEATURE_SCHEMA_VERSION, feature_schema_header, validate_feature_metadata_schema
import ..DataFileIO
import ..DataTypes

import ..IndicatorResolutions
import ..ZnEncoding
using ..IndicatorResolutions: pmodule_from_fringe
using ..PLPolyhedra
using ..PLBackend: BoxUpset, BoxDownset, encode_fringe_boxes
using ..Encoding: build_uptight_encoding_from_fringe,
                  pushforward_fringe_along_encoding,
                  PostcomposedEncodingMap
using ..Modules: CoverEdgeMapStore, _get_cover_cache,
                 PModule, PMorphism, cover_edges, dim_at
using ..FiniteFringe: AbstractPoset, FinitePoset, GridPoset, ProductOfChainsPoset, FringeModule,
                      Upset, Downset, principal_upset, principal_downset, leq, nvertices,
                      poset_equal_opposite, _succs, _preds, _pred_slots_of_succ
using ..DerivedFunctors
using ..Invariants
using ..SignedMeasures: PointSignedMeasure, truncate_point_signed_measure
import ..SignedMeasures: euler_characteristic_surface, euler_surface, euler_signed_measure
import ..Modules
import ..ModuleComplexes
import ..FieldLinAlg
using ..ModuleComplexes: ModuleCochainComplex, cohomology_module
using ..AbelianCategories: kernel_with_inclusion, image_with_inclusion, _cokernel_module, is_zero_morphism

using ..FlangeZn: Face, IndFlat, IndInj, Flange
import ..Workflow: encode, fringe_presentation, flange_presentation
import ..ChainComplexes: describe, filtration_summary

const _POINTCLOUD_KNN_GRAPH_IMPL = Ref{Any}(nothing)
const _POINTCLOUD_RADIUS_GRAPH_IMPL = Ref{Any}(nothing)
const _POINTCLOUD_KNN_DISTANCES_IMPL = Ref{Any}(nothing)
const _POINTCLOUD_DTM_VALUES_IMPL = Ref{Any}(nothing)
const _POINTCLOUD_KNN_GRAPH_EDGES_IMPL = Ref{Any}(nothing)
const _POINTCLOUD_RADIUS_GRAPH_EDGES_IMPL = Ref{Any}(nothing)
const _POINTCLOUD_DELAUNAY_2D_IMPL = Ref{Any}(nothing)
const _POINTCLOUD_LARGE_N_SPARSE_ONLY = Ref{Int}(5_000)
const _LAZY_DIFF_THREADS_MIN_VERTICES = Ref{Int}(128)
const _H0_UNIONFIND_MIN_POS_VERTICES = Ref{Int}(96)
const _H0_UNIONFIND_MIN_TOTAL_ACTIVE_VERTICES = Ref{Int}(8_000)
const _H0_UNIONFIND_MIN_TOTAL_ACTIVE_EDGES = Ref{Int}(6_000)
const _POINTCLOUD_STREAM_DIST_NONSPARSE = Ref{Bool}(true)
const _POINTCLOUD_LOWDIM_RADIUS_STREAMING = Ref{Bool}(true)
const _POINTCLOUD_DIM2_PACKED_KERNEL = Ref{Bool}(true)
const _GRAPH_CLIQUE_ENUM_MODE = Ref{Symbol}(:auto) # :auto | :intersection | :combinations
const _H0_CHAIN_SWEEP_FASTPATH = Ref{Bool}(true)
const _H1_COKERNEL_FASTPATH = Ref{Bool}(true)
const _H0_ACTIVE_CHAIN_INCREMENTAL = Ref{Bool}(true)
const _H0_ACTIVE_CHAIN_INCREMENTAL_MIN_POS_VERTICES = Ref{Int}(32)
const _H0_ACTIVE_CHAIN_INCREMENTAL_MIN_TOTAL_ACTIVE_VERTICES = Ref{Int}(3_000)
const _H0_ACTIVE_CHAIN_INCREMENTAL_MIN_TOTAL_ACTIVE_EDGES = Ref{Int}(8_000)
const _COHOMOLOGY_DEGREE_LOCAL_FASTPATH = Ref{Bool}(true)
const _COHOMOLOGY_DEGREE_LOCAL_ALL_T = Ref{Bool}(true)
const _COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH = Ref{Bool}(true)
const _COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_RANK = Ref{Bool}(true)
const _COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_MODE = Ref{Symbol}(:auto) # :auto | :on | :off
const _COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_PROBE = Ref{Bool}(true)
const _COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_CACHE_ENABLED = Ref{Bool}(true)
const _COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_CACHE_MAX = Ref{Int}(96)
const _COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_CACHE = Dict{NTuple{5,Int},Bool}()
const _COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_CACHE_ORDER = NTuple{5,Int}[]
const _COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_CACHE_LOCK = ReentrantLock()
const _COHOMOLOGY_DEGREE_LOCAL_T1_MIN_POS_VERTICES = Ref{Int}(32)
const _COHOMOLOGY_DEGREE_LOCAL_T1_MIN_TOTAL_ACTIVE_DIM1 = Ref{Int}(2_500)
const _COHOMOLOGY_DEGREE_LOCAL_T1_MIN_TOTAL_ACTIVE_DIM2 = Ref{Int}(1_200)
const _COHOMOLOGY_DIMS_USE_DIRECT_RESTRICTED_RANK = Ref{Bool}(true)
const _STRUCTURAL_MAP_FAST_KERNELS = Ref{Bool}(true)
const _ACTIVE_LISTS_CHAIN_FASTPATH = Ref{Bool}(true)
const _INGESTION_SKIP_LAZY_ON_MODULE_CACHE_HIT = Ref{Bool}(true)
const _INGESTION_PLAN_NORM_CACHE = Ref{Bool}(true)
const _SIMPLICIAL_BOUNDARY_SPECIALIZED = Ref{Bool}(true)
const _ACTIVE_INDEX_TABLE_MAX_ROWS = Ref{Int}(500_000)
const _GRAPH_PACKED_EDGELIST_BACKEND = Ref{Bool}(true)
const _ENCODING_RESULT_LAZY_MODULE = Ref{Bool}(true)
const _GRAPH_BACKEND_WINNER_CACHE_ENABLED = Ref{Bool}(true)
const _GRAPH_BACKEND_WINNER_CACHE_PROBE = Ref{Bool}(true)
const _GRAPH_BACKEND_WINNER_CACHE_MAX = Ref{Int}(96)
const _GRAPH_BACKEND_WINNER_CACHE = Dict{NTuple{4,Int},Bool}()
const _GRAPH_BACKEND_WINNER_CACHE_ORDER = NTuple{4,Int}[]
const _GRAPH_BACKEND_WINNER_CACHE_LOCK = ReentrantLock()
const _POINTCLOUD_BACKEND_RESOLVE_CACHE_ENABLED = Ref{Bool}(true)
const _POINTCLOUD_BACKEND_RESOLVE_CACHE_MAX = Ref{Int}(96)
const _POINTCLOUD_BACKEND_RESOLVE_CACHE = Dict{NTuple{3,Int},Symbol}()
const _POINTCLOUD_BACKEND_RESOLVE_CACHE_ORDER = NTuple{3,Int}[]
const _POINTCLOUD_BACKEND_RESOLVE_CACHE_LOCK = ReentrantLock()
const _POINTCLOUD_DELAUNAY_CACHE_ENABLED = Ref{Bool}(true)
const _POINTCLOUD_DELAUNAY_CACHE_MAX = Ref{Int}(48)
const _POINTCLOUD_DELAUNAY_CACHE = Dict{NTuple{4,UInt64},Any}()
const _POINTCLOUD_DELAUNAY_CACHE_ORDER = NTuple{4,UInt64}[]
const _POINTCLOUD_DELAUNAY_CACHE_LOCK = ReentrantLock()
const _POINTCLOUD_DELAUNAY_AUTOLOAD_ATTEMPTED = Ref{Bool}(false)
const _CUBICAL_2D_FASTPATH = Ref{Bool}(true)

struct _PackedDelaunay2D
    edges::Vector{NTuple{2,Int}}
    edge_radius::Vector{Float64}
    triangles::Vector{NTuple{3,Int}}
    tri_radius::Vector{Float64}
end

mutable struct _PackedDelaunay2DCacheEntry
    packed::_PackedDelaunay2D
    edge_boundary::Union{Nothing,SparseMatrixCSC{Int,Int}}
    tri_boundary::Union{Nothing,SparseMatrixCSC{Int,Int}}
end

struct _StructuralInclusionMap{K} <: AbstractMatrix{K}
    nrows::Int
    ncols::Int
    row_for_col::Vector{Int}
end

Base.size(A::_StructuralInclusionMap) = (A.nrows, A.ncols)

function Base.convert(::Type{_StructuralInclusionMap{K}}, A::AbstractMatrix{K}) where {K}
    nrows, ncols = size(A)
    row_for_col = Vector{Int}(undef, ncols)
    @inbounds for j in 1:ncols
        r = 0
        for i in 1:nrows
            v = A[i, j]
            iszero(v) && continue
            v == one(K) || throw(ArgumentError("convert(_StructuralInclusionMap): non-unit entry at ($i,$j)."))
            r == 0 || throw(ArgumentError("convert(_StructuralInclusionMap): multiple nonzero entries in column $j."))
            r = i
        end
        row_for_col[j] = r
    end
    return _StructuralInclusionMap{K}(nrows, ncols, row_for_col)
end

@inline function Base.getindex(A::_StructuralInclusionMap{K}, i::Int, j::Int) where {K}
    @boundscheck checkbounds(A, i, j)
    @inbounds return A.row_for_col[j] == i ? one(K) : zero(K)
end

function Base.Matrix(A::_StructuralInclusionMap{K}) where {K}
    M = zeros(K, A.nrows, A.ncols)
    @inbounds for j in 1:A.ncols
        i = A.row_for_col[j]
        i == 0 && continue
        M[i, j] = one(K)
    end
    return M
end

function SparseArrays.sparse(A::_StructuralInclusionMap{K}) where {K}
    I = Int[]
    J = Int[]
    V = K[]
    sizehint!(I, A.ncols)
    sizehint!(J, A.ncols)
    sizehint!(V, A.ncols)
    @inbounds for j in 1:A.ncols
        i = A.row_for_col[j]
        i == 0 && continue
        push!(I, i)
        push!(J, j)
        push!(V, one(K))
    end
    return sparse(I, J, V, A.nrows, A.ncols)
end

function Base.:*(A::_StructuralInclusionMap{K}, X::AbstractMatrix{T}) where {K,T}
    nrows, ncols = size(A)
    ncols == size(X, 1) || throw(DimensionMismatch("size mismatch in inclusion-map multiplication"))
    U = promote_type(K, T)
    Y = zeros(U, nrows, size(X, 2))
    @inbounds for j in 1:ncols
        i = A.row_for_col[j]
        i == 0 && continue
        @views Y[i, :] .+= X[j, :]
    end
    return Y
end

function Base.:*(A::_StructuralInclusionMap{K}, x::AbstractVector{T}) where {K,T}
    nrows, ncols = size(A)
    ncols == length(x) || throw(DimensionMismatch("size mismatch in inclusion-map multiplication"))
    U = promote_type(K, T)
    y = zeros(U, nrows)
    @inbounds for j in 1:ncols
        i = A.row_for_col[j]
        i == 0 && continue
        y[i] += x[j]
    end
    return y
end

function Base.:*(A::_StructuralInclusionMap{K}, B::_StructuralInclusionMap{K}) where {K}
    size(B, 1) == size(A, 2) || throw(DimensionMismatch("size mismatch in structural inclusion composition"))
    row_for_col = Vector{Int}(undef, B.ncols)
    @inbounds for j in 1:B.ncols
        i = B.row_for_col[j]
        row_for_col[j] = i == 0 ? 0 : A.row_for_col[i]
    end
    return _StructuralInclusionMap{K}(A.nrows, B.ncols, row_for_col)
end

@inline function _rank_structural_inclusion(A::_StructuralInclusionMap)
    row_for_col = A.row_for_col
    rank = 0
    last = 0
    monotone = true
    @inbounds for r in row_for_col
        if r < last
            monotone = false
            break
        end
        if r != 0 && r != last
            rank += 1
            last = r
        elseif r != 0
            last = r
        end
    end
    monotone && return rank

    seen = falses(A.nrows)
    rank = 0
    @inbounds for r in row_for_col
        if r > 0 && !seen[r]
            seen[r] = true
            rank += 1
        end
    end
    return rank
end

function FieldLinAlg.rank(field::AbstractCoeffField,
                          A::_StructuralInclusionMap;
                          backend::Symbol=:auto)
    return _rank_structural_inclusion(A)
end

function FieldLinAlg.rank_dim(field::AbstractCoeffField,
                              A::_StructuralInclusionMap;
                              backend::Symbol=:auto,
                              kwargs...)
    return _rank_structural_inclusion(A)
end

function FieldLinAlg.rank_restricted(field::AbstractCoeffField,
                                     A::_StructuralInclusionMap,
                                     rows::AbstractVector{Int},
                                     cols::AbstractVector{Int};
                                     backend::Symbol=:auto,
                                     kwargs...)
    if !_STRUCTURAL_MAP_FAST_KERNELS[]
        return FieldLinAlg.rank_restricted(field, Matrix(A), rows, cols; backend=backend, kwargs...)
    end
    nrows, ncols = size(A)
    row_active = falses(nrows)
    @inbounds for rr in rows
        r = Int(rr)
        (1 <= r <= nrows) && (row_active[r] = true)
    end
    seen = falses(nrows)
    rank = 0
    @inbounds for cc in cols
        c = Int(cc)
        (1 <= c <= ncols) || continue
        r = A.row_for_col[c]
        if r > 0 && row_active[r] && !seen[r]
            seen[r] = true
            rank += 1
        end
    end
    return rank
end

function FieldLinAlg.colspace(field::AbstractCoeffField,
                              A::_StructuralInclusionMap{K};
                              backend::Symbol=:auto) where {K}
    if !_STRUCTURAL_MAP_FAST_KERNELS[]
        return FieldLinAlg.colspace(field, Matrix(A); backend=backend)
    end
    seen = falses(A.nrows)
    rows = Int[]
    sizehint!(rows, min(A.nrows, A.ncols))
    @inbounds for r in A.row_for_col
        if r > 0 && !seen[r]
            seen[r] = true
            push!(rows, r)
        end
    end
    return _StructuralInclusionMap{K}(A.nrows, length(rows), rows)
end

function FieldLinAlg.nullspace(field::AbstractCoeffField,
                               A::_StructuralInclusionMap{K};
                               backend::Symbol=:auto) where {K}
    if !_STRUCTURAL_MAP_FAST_KERNELS[]
        return FieldLinAlg.nullspace(field, Matrix(A); backend=backend)
    end
    nrows, ncols = size(A)
    first_col_for_row = zeros(Int, nrows)
    I = Int[]
    J = Int[]
    V = K[]
    sizehint!(I, ncols)
    sizehint!(J, ncols)
    sizehint!(V, ncols)
    nb = 0
    oneK = one(K)
    neg_oneK = -oneK
    @inbounds for col in 1:ncols
        r = A.row_for_col[col]
        if r == 0
            nb += 1
            push!(I, col); push!(J, nb); push!(V, oneK)
            continue
        end
        leader = first_col_for_row[r]
        if leader == 0
            first_col_for_row[r] = col
            continue
        end
        nb += 1
        push!(I, col); push!(J, nb); push!(V, oneK)
        push!(I, leader); push!(J, nb); push!(V, neg_oneK)
    end
    return sparse(I, J, V, ncols, nb)
end

function FieldLinAlg.solve_fullcolumn(field::AbstractCoeffField,
                                      B::_StructuralInclusionMap{K},
                                      Y::AbstractVecOrMat{K};
                                      check_rhs::Bool=true,
                                      backend::Symbol=:auto,
                                      cache::Bool=true,
                                      factor=nothing) where {K}
    if !_STRUCTURAL_MAP_FAST_KERNELS[]
        return FieldLinAlg.solve_fullcolumn(field, Matrix(B), Y;
                                            check_rhs=check_rhs, backend=backend, cache=cache, factor=factor)
    end
    nrows, ncols = size(B)
    Ymat = Y isa AbstractVector ? reshape(Y, :, 1) : Y
    size(Ymat, 1) == nrows || throw(DimensionMismatch("B and Y must have same row count"))

    _rank_structural_inclusion(B) == ncols ||
        error("solve_fullcolumn: structural inclusion does not have full column rank")

    X = _solve_through_structural_inclusion(B, Ymat)
    if check_rhs
        BX = B * X
        BX == Matrix(Ymat) || error("solve_fullcolumn: RHS check failed")
    end
    return Y isa AbstractVector ? vec(X) : X
end

mutable struct _ActiveIndexLookupScratch
    pos::Vector{Int}
    stamp::Vector{UInt32}
    current::UInt32
end

@inline _ActiveIndexLookupScratch(n::Int) =
    _ActiveIndexLookupScratch(zeros(Int, n), zeros(UInt32, n), UInt32(0))

@inline function _active_lookup_begin!(scratch::_ActiveIndexLookupScratch)
    s = scratch.current + UInt32(1)
    if s == 0
        fill!(scratch.stamp, UInt32(0))
        s = UInt32(1)
    end
    scratch.current = s
    return s
end

@inline function _active_lookup_fill!(scratch::_ActiveIndexLookupScratch, ids::Vector{Int})
    s = _active_lookup_begin!(scratch)
    @inbounds for (i, id) in enumerate(ids)
        scratch.pos[id] = i
        scratch.stamp[id] = s
    end
    return nothing
end

@inline function _active_lookup_get(scratch::_ActiveIndexLookupScratch, id::Int)
    @inbounds return (1 <= id <= length(scratch.stamp) && scratch.stamp[id] == scratch.current) ? scratch.pos[id] : 0
end

function _set_pointcloud_nn_impl!(; knn_graph=nothing, radius_graph=nothing, knn_distances=nothing,
                                  dtm_values=nothing, knn_graph_edges=nothing,
                                  radius_graph_edges=nothing)
    knn_graph !== nothing && (_POINTCLOUD_KNN_GRAPH_IMPL[] = knn_graph)
    radius_graph !== nothing && (_POINTCLOUD_RADIUS_GRAPH_IMPL[] = radius_graph)
    knn_distances !== nothing && (_POINTCLOUD_KNN_DISTANCES_IMPL[] = knn_distances)
    dtm_values !== nothing && (_POINTCLOUD_DTM_VALUES_IMPL[] = dtm_values)
    knn_graph_edges !== nothing && (_POINTCLOUD_KNN_GRAPH_EDGES_IMPL[] = knn_graph_edges)
    radius_graph_edges !== nothing && (_POINTCLOUD_RADIUS_GRAPH_EDGES_IMPL[] = radius_graph_edges)
    Base.lock(_POINTCLOUD_BACKEND_RESOLVE_CACHE_LOCK)
    empty!(_POINTCLOUD_BACKEND_RESOLVE_CACHE)
    empty!(_POINTCLOUD_BACKEND_RESOLVE_CACHE_ORDER)
    Base.unlock(_POINTCLOUD_BACKEND_RESOLVE_CACHE_LOCK)
    return nothing
end

function _clear_pointcloud_delaunay_cache!()
    lock(_POINTCLOUD_DELAUNAY_CACHE_LOCK) do
        empty!(_POINTCLOUD_DELAUNAY_CACHE)
        empty!(_POINTCLOUD_DELAUNAY_CACHE_ORDER)
    end
    return nothing
end

function _set_pointcloud_delaunay_2d_impl!(impl=nothing)
    _POINTCLOUD_DELAUNAY_2D_IMPL[] = impl
    _POINTCLOUD_DELAUNAY_AUTOLOAD_ATTEMPTED[] = false
    _clear_pointcloud_delaunay_cache!()
    return nothing
end

@inline _have_pointcloud_nn_backend() =
    _POINTCLOUD_KNN_GRAPH_IMPL[] !== nothing &&
    _POINTCLOUD_KNN_DISTANCES_IMPL[] !== nothing

@inline _have_pointcloud_delaunay_backend() =
    _POINTCLOUD_DELAUNAY_2D_IMPL[] !== nothing

@inline function _try_activate_pointcloud_delaunay_backend!()
    _have_pointcloud_delaunay_backend() && return true
    _POINTCLOUD_DELAUNAY_AUTOLOAD_ATTEMPTED[] && return false
    _POINTCLOUD_DELAUNAY_AUTOLOAD_ATTEMPTED[] = true
    try
        Core.eval(@__MODULE__, :(import DelaunayTriangulation))
    catch
        # Optional dependency may be unavailable in this environment.
    end
    if !_have_pointcloud_delaunay_backend()
        try
            pm = parentmodule(@__MODULE__)
            if isdefined(pm, :_try_load_source_extension!)
                getfield(pm, :_try_load_source_extension!)(
                    :DelaunayTriangulation,
                    :TamerOpDelaunayTriangulationExt,
                    "TamerOpDelaunayTriangulationExt.jl",
                )
            end
        catch
            # Source-mode extension loader is best-effort only.
        end
    end
    return _have_pointcloud_delaunay_backend()
end

@inline function _pointcloud_nn_backend(spec::FiltrationSpec)::Symbol
    b = Symbol(get(spec.params, :nn_backend, :auto))
    (b == :auto || b == :bruteforce || b == :nearestneighbors || b == :approx) ||
        throw(ArgumentError("PointCloud nn_backend must be :auto, :bruteforce, :nearestneighbors, or :approx (got $(b))."))
    if b == :auto
        return _have_pointcloud_nn_backend() ? :auto : :bruteforce
    end
    if (b == :nearestneighbors || b == :approx) && !_have_pointcloud_nn_backend()
        throw(ArgumentError("PointCloud nn_backend=$(b) requires NearestNeighbors extension. Install NearestNeighbors.jl and load the extension, or use nn_backend=:bruteforce."))
    end
    return b
end

@inline function _pointcloud_delaunay_backend(spec::FiltrationSpec)::Symbol
    b = Symbol(get(spec.params, :delaunay_backend, :auto))
    (b == :auto || b == :naive || b == :fast) ||
        throw(ArgumentError("PointCloud delaunay_backend must be :auto, :naive, or :fast (got $(b))."))
    if b == :auto
        _have_pointcloud_delaunay_backend() || _try_activate_pointcloud_delaunay_backend!()
        return _have_pointcloud_delaunay_backend() ? :fast : :naive
    end
    if b == :fast && !_have_pointcloud_delaunay_backend() && !_try_activate_pointcloud_delaunay_backend!()
        throw(ArgumentError("PointCloud delaunay_backend=:fast requires DelaunayTriangulation extension. Install DelaunayTriangulation.jl and load the extension, or use delaunay_backend=:naive."))
    end
    return b
end

@inline function _pointcloud_bucket_value(n::Int)
    n <= 0 && return 0
    return min(16, max(0, floor(Int, log2(Float64(n)))))
end

@inline function _resolve_pointcloud_auto_backend(n::Int, d::Int, op_code::Int)::Symbol
    _have_pointcloud_nn_backend() || return :bruteforce
    if op_code == 1 && d > 20 && n > 1500
        return :approx
    end
    return :nearestneighbors
end

@inline function _resolve_pointcloud_runtime_backend(backend::Symbol, n::Int, d::Int, op_code::Int)::Symbol
    backend == :auto || return backend
    if !_POINTCLOUD_BACKEND_RESOLVE_CACHE_ENABLED[]
        return _resolve_pointcloud_auto_backend(n, d, op_code)
    end
    key = (_pointcloud_bucket_value(n), _pointcloud_bucket_value(d), op_code)
    if haskey(_POINTCLOUD_BACKEND_RESOLVE_CACHE, key)
        return _POINTCLOUD_BACKEND_RESOLVE_CACHE[key]
    end
    b = _resolve_pointcloud_auto_backend(n, d, op_code)
    Base.lock(_POINTCLOUD_BACKEND_RESOLVE_CACHE_LOCK)
    if haskey(_POINTCLOUD_BACKEND_RESOLVE_CACHE, key)
        b = _POINTCLOUD_BACKEND_RESOLVE_CACHE[key]
    else
        _POINTCLOUD_BACKEND_RESOLVE_CACHE[key] = b
        push!(_POINTCLOUD_BACKEND_RESOLVE_CACHE_ORDER, key)
        max_keep = _POINTCLOUD_BACKEND_RESOLVE_CACHE_MAX[]
        if max_keep > 0 && length(_POINTCLOUD_BACKEND_RESOLVE_CACHE_ORDER) > max_keep
            old = popfirst!(_POINTCLOUD_BACKEND_RESOLVE_CACHE_ORDER)
            delete!(_POINTCLOUD_BACKEND_RESOLVE_CACHE, old)
        end
    end
    Base.unlock(_POINTCLOUD_BACKEND_RESOLVE_CACHE_LOCK)
    return b
end

@inline function _pointcloud_nn_approx_candidates(spec::FiltrationSpec)::Int
    c = Int(get(spec.params, :nn_approx_candidates, 0))
    c >= 0 || throw(ArgumentError("PointCloud nn_approx_candidates must be >= 0 (got $(c))."))
    return c
end

@inline function _default_landmark_count(n::Int)::Int
    n <= 2 && return n
    if n >= 50_000
        return min(n, 2_048)
    elseif n >= 10_000
        return min(n, 1_536)
    elseif n >= 2_000
        return min(n, 1_024)
    end
    return max(2, min(n, ceil(Int, sqrt(max(n, 1)))))
end

# Section 2: Typed filtrations (runtime API)
# -----------------------------------------------------------------------------

abstract type AbstractFiltration end

const _BUILTIN_FILTRATION_KINDS = (
    :graded,
    :rips,
    :rips_density,
    :rips_codensity,
    :rips_lowerstar,
    :function_rips,
    :landmark_rips,
    :graph_lower_star,
    :clique_lower_star,
    :edge_weighted,
    :graph_centrality,
    :graph_geodesic,
    :graph_function_geodesic_bifiltration,
    :graph_weight_threshold,
    :lower_star,
    :image_distance_bifiltration,
    :wing_vein_bifiltration,
    :cubical,
    :delaunay_lower_star,
    :alpha,
    :function_delaunay,
    :core_delaunay,
    :core,
    :degree_rips,
    :rhomboid,
)

const _CUSTOM_FILTRATION_REGISTRY = Dict{Symbol,NamedTuple}()
const _CUSTOM_FILTRATION_REGISTRY_LOCK = ReentrantLock()

@inline function _filtration_registry_get(kind::Symbol)
    Base.lock(_CUSTOM_FILTRATION_REGISTRY_LOCK)
    try
        return get(_CUSTOM_FILTRATION_REGISTRY, kind, nothing)
    finally
        Base.unlock(_CUSTOM_FILTRATION_REGISTRY_LOCK)
    end
end

@inline function _filtration_registry_set!(kind::Symbol, entry::NamedTuple)
    Base.lock(_CUSTOM_FILTRATION_REGISTRY_LOCK)
    try
        _CUSTOM_FILTRATION_REGISTRY[kind] = entry
    finally
        Base.unlock(_CUSTOM_FILTRATION_REGISTRY_LOCK)
    end
    return entry
end

@inline function _normalize_schema_required(raw)
    if raw === nothing
        return ()
    elseif raw isa Symbol
        return (raw,)
    elseif raw isa Tuple || raw isa AbstractVector
        out = Symbol[]
        for x in raw
            x isa Symbol || throw(ArgumentError("filtration schema `required` must contain symbols."))
            push!(out, x)
        end
        return Tuple(out)
    end
    throw(ArgumentError("filtration schema `required` must be Symbol or tuple/vector of Symbol."))
end

@inline function _normalize_schema_namedtuple(raw, name::AbstractString)
    if raw === nothing
        return NamedTuple()
    elseif raw isa NamedTuple
        return raw
    elseif raw isa AbstractDict
        ks = Symbol[]
        vs = Any[]
        for (k, v) in pairs(raw)
            push!(ks, Symbol(k))
            push!(vs, v)
        end
        return NamedTuple{Tuple(ks)}(Tuple(vs))
    end
    throw(ArgumentError("filtration schema `$name` must be NamedTuple or AbstractDict."))
end

@inline function _normalize_filtration_schema(schema)
    if schema === nothing
        return (
            required = (),
            defaults = NamedTuple(),
            types = NamedTuple(),
            checks = NamedTuple(),
        )
    end
    schema isa NamedTuple || throw(ArgumentError("filtration schema must be a NamedTuple."))
    return (
        required = _normalize_schema_required(get(schema, :required, ())),
        defaults = _normalize_schema_namedtuple(get(schema, :defaults, NamedTuple()), "defaults"),
        types = _normalize_schema_namedtuple(get(schema, :types, NamedTuple()), "types"),
        checks = _normalize_schema_namedtuple(get(schema, :checks, NamedTuple()), "checks"),
    )
end

@inline function _schema_type_matches(v, T)
    if T isa Tuple
        for Ti in T
            v isa Ti && return true
        end
        return false
    end
    return v isa T
end

function _schema_check_message(check_spec, key::Symbol, kind::Symbol)
    if check_spec isa Tuple && length(check_spec) >= 2 && check_spec[2] isa AbstractString
        return String(check_spec[2])
    end
    return "Filtration kind=:$kind parameter `$key` failed domain check."
end

function _run_schema_check(check_spec, value, params::NamedTuple)
    if check_spec isa Function
        return check_spec(value)
    elseif check_spec isa Tuple && !isempty(check_spec) && check_spec[1] isa Function
        pred = check_spec[1]
        if hasmethod(pred, Tuple{typeof(value), NamedTuple})
            return pred(value, params)
        end
        return pred(value)
    end
    throw(ArgumentError("filtration schema checks entries must be Function or (Function, message)."))
end

function _normalize_filtration_params(kind::Symbol,
                                      params::NamedTuple,
                                      schema::NamedTuple;
                                      context::AbstractString="filtration")
    merged = merge(schema.defaults, params)
    for key in schema.required
        haskey(merged, key) || throw(ArgumentError("$context kind=:$kind requires parameter `$key`."))
    end
    for (key, T) in pairs(schema.types)
        haskey(merged, key) || continue
        v = merged[key]
        _schema_type_matches(v, T) && continue
        throw(ArgumentError("$context kind=:$kind parameter `$key` has type $(typeof(v)); expected $(T)."))
    end
    for (key, check_spec) in pairs(schema.checks)
        haskey(merged, key) || continue
        ok = _run_schema_check(check_spec, merged[key], merged)
        ok === true || throw(ArgumentError(_schema_check_message(check_spec, key, kind)))
    end
    return merged
end

@inline function _filtration_entry_arity(entry::NamedTuple, f::AbstractFiltration, data)
    a = entry.arity
    if a isa Function
        if hasmethod(a, Tuple{typeof(f), typeof(data)})
            return a(f, data)
        end
        return a(f)
    end
    return a
end

"""
    filtration_kind(::Type{T}) where {T<:AbstractFiltration} -> Symbol

Public contract for filtration-family identity.
Custom filtration families should define this for their filtration type.
"""
function filtration_kind(::Type{T}) where {T<:AbstractFiltration}
    throw(ArgumentError("No filtration kind contract for $(T). Define `filtration_kind(::Type{$(T)})::Symbol`."))
end

@inline filtration_kind(filtration::AbstractFiltration) = filtration_kind(typeof(filtration))
@inline filtration_kind(spec::FiltrationSpec) = spec.kind

"""
    filtration_arity(filtration, data=nothing) -> Int | :variable

Public contract for persistence-parameter count.
"""
function filtration_arity(filtration::AbstractFiltration, data=nothing)
    kind = filtration_kind(typeof(filtration))
    entry = _filtration_registry_get(kind)
    if entry !== nothing
        return _filtration_entry_arity(entry, filtration, data)
    end
    if kind === :rips || kind === :landmark_rips || kind === :edge_weighted ||
       kind === :graph_centrality || kind === :graph_geodesic ||
       kind === :graph_weight_threshold || kind === :delaunay_lower_star ||
       kind === :alpha
        return 1
    elseif kind === :rips_density || kind === :rips_codensity || kind === :rips_lowerstar ||
           kind === :function_rips ||
           kind === :degree_rips || kind === :rhomboid || kind === :core ||
           kind === :function_delaunay || kind === :core_delaunay ||
           kind === :graph_function_geodesic_bifiltration ||
           kind === :image_distance_bifiltration || kind === :wing_vein_bifiltration
        return 2
    elseif kind === :graph_lower_star || kind === :clique_lower_star ||
           kind === :lower_star || kind === :cubical || kind === :graded
        return :variable
    end
    return :variable
end

@inline filtration_arity(spec::FiltrationSpec, data=nothing) = filtration_signature(filtration_kind(spec)).arity

"""
    filtration_parameters(filtration::AbstractFiltration) -> NamedTuple

Public hook for converting typed filtrations to `FiltrationSpec` params.
"""
function filtration_parameters(filtration::AbstractFiltration)
    if hasproperty(filtration, :params)
        p = getproperty(filtration, :params)
        p isa NamedTuple || throw(ArgumentError("filtration `params` must be a NamedTuple."))
        return _params_with_nonnothing(p)
    end
    return NamedTuple()
end

@inline filtration_parameters(spec::FiltrationSpec) = spec.params

"""
    construction_mode(filtration_or_spec) -> ConstructionOptions

Return the resolved construction contract carried by a typed filtration or
serialized [`FiltrationSpec`](@ref).

This is the cheap semantic accessor to use when you want the normalized
construction-stage choices without unpacking raw `params` payloads.
"""
@inline construction_mode(f::AbstractFiltration) = _construction_from_filtration(f)
@inline construction_mode(spec::FiltrationSpec) = _construction_from_params(spec.params)

"""
    register_filtration_family!(; kind, ctor, builder, arity=:variable, schema=nothing)

Register a custom filtration family so that:
- `to_filtration(FiltrationSpec(kind=kind, ...))` works,
- `encode(data, FiltrationSpec(kind=kind, ...))` works,
- custom schema/default validation is enforced.

Contract
- `ctor(spec)` must return the typed runtime filtration object for the canonical
  serialized `FiltrationSpec` boundary;
- `builder(data, filtration; cache=...)` is the internal graded-complex builder
  contract and must return the raw tuple `(G, axes, orientation)`;
- `arity` is the persistence-parameter count exposed to users through
  [`filtration_signature`](@ref) and `describe(...)`.

Best practices
- prefer typed filtration constructors for runtime work and `FiltrationSpec` as
  the serialized/config boundary;
- inspect the family first with `filtration_signature(kind)` before wiring it
  into a larger ingestion workflow;
- keep `builder(...)` focused on the graded-complex stage and let
  `plan_ingestion(...)` / `run_ingestion(...)` handle the rest of the pipeline.
"""
function register_filtration_family!(;
                                     kind::Symbol,
                                     ctor::Function,
                                     builder::Function,
                                     arity::Union{Int,Symbol,Function}=:variable,
                                     schema=nothing)
    kind in _BUILTIN_FILTRATION_KINDS &&
        throw(ArgumentError("Cannot register custom filtration for built-in kind=:$kind."))
    if arity isa Int
        arity > 0 || throw(ArgumentError("filtration arity must be positive (got $arity)."))
    elseif arity isa Symbol
        arity === :variable || throw(ArgumentError("filtration arity symbol must be :variable (got $arity)."))
    elseif !(arity isa Function)
        throw(ArgumentError("filtration arity must be Int, :variable, or Function."))
    end
    sch = _normalize_filtration_schema(schema)
    entry = (kind=kind, ctor=ctor, builder=builder, arity=arity, schema=sch)
    _filtration_registry_set!(kind, entry)
    return kind
end

"""
    available_filtrations() -> Vector{Symbol}

Return all built-in and registered custom filtration kinds.
"""
function available_filtrations()
    Base.lock(_CUSTOM_FILTRATION_REGISTRY_LOCK)
    custom = try
        collect(keys(_CUSTOM_FILTRATION_REGISTRY))
    finally
        Base.unlock(_CUSTOM_FILTRATION_REGISTRY_LOCK)
    end
    return sort!(unique!(vcat(collect(_BUILTIN_FILTRATION_KINDS), custom)))
end

"""
    registered_filtration_families() -> Vector{Symbol}

Return the custom filtration kinds currently installed through
[`register_filtration_family!`](@ref).

Use [`available_filtrations`](@ref) when you want both built-in and custom
families. This helper is specifically for registry-driven extensibility
workflows.
"""
function registered_filtration_families()
    Base.lock(_CUSTOM_FILTRATION_REGISTRY_LOCK)
    custom = try
        collect(keys(_CUSTOM_FILTRATION_REGISTRY))
    finally
        Base.unlock(_CUSTOM_FILTRATION_REGISTRY_LOCK)
    end
    return sort!(custom)
end

"""
    filtration_signature(kind::Symbol) -> NamedTuple

Return the declared contract for a filtration family.

Cheap-first workflow
- use `filtration_signature(kind)` when you want to inspect a family without
  constructing data or a typed filtration object;
- use [`filtration_family_summary`](@ref) when you want the same contract in
  the owner-local summary style used elsewhere in `DataIngestion`;
- use `describe(filtration)` / [`filtration_summary`](@ref) once you already
  have the typed runtime filtration in hand;
- use `register_filtration_family!` when you need to add a new family to this
  contract surface.
"""
function filtration_signature(kind::Symbol)
    _builtin_arity(k::Symbol) = begin
        if k === :rips || k === :landmark_rips || k === :edge_weighted ||
           k === :graph_centrality || k === :graph_geodesic ||
           k === :graph_weight_threshold || k === :delaunay_lower_star ||
           k === :alpha
            return 1
        elseif k === :rips_density || k === :rips_codensity || k === :rips_lowerstar ||
               k === :function_rips ||
               k === :degree_rips || k === :rhomboid || k === :core ||
               k === :function_delaunay || k === :core_delaunay ||
               k === :graph_function_geodesic_bifiltration ||
               k === :image_distance_bifiltration || k === :wing_vein_bifiltration
            return 2
        end
        return :variable
    end
    if kind in _BUILTIN_FILTRATION_KINDS
        return (
            kind = kind,
            registered = false,
            arity = _builtin_arity(kind),
            required = (),
            defaults = NamedTuple(),
            types = NamedTuple(),
            checks = NamedTuple(),
        )
    end
    entry = _filtration_registry_get(kind)
    entry === nothing && throw(ArgumentError("No filtration family registered for kind=:$kind."))
    schema = entry.schema
    return (
        kind = kind,
        registered = true,
        arity = entry.arity,
        required = schema.required,
        defaults = schema.defaults,
        types = schema.types,
        checks = schema.checks,
    )
end

"""
    filtration_parameters(kind::Symbol) -> NamedTuple

Introspection helper for registered parameter schema.
"""
function filtration_parameters(kind::Symbol)
    sig = filtration_signature(kind)
    return (
        required = sig.required,
        defaults = sig.defaults,
        types = sig.types,
        checks = sig.checks,
    )
end

struct GradedFiltration <: AbstractFiltration end

struct RipsFiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
RipsFiltration(; max_dim::Int=1,
               radius::Union{Nothing,Real}=nothing,
               knn::Union{Nothing,Int}=nothing,
               n_landmarks::Union{Nothing,Int}=nothing,
               nn_backend::Symbol=:auto,
               nn_approx_candidates::Int=0,
               construction::ConstructionOptions=ConstructionOptions()) =
    RipsFiltration((;
        max_dim,
        radius = radius === nothing ? nothing : Float64(radius),
        knn,
        n_landmarks,
        nn_backend,
        nn_approx_candidates,
        construction,
    ))

struct RipsDensityFiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
RipsDensityFiltration(; max_dim::Int=1,
                      radius::Union{Nothing,Real}=nothing,
                      knn::Union{Nothing,Int}=nothing,
                      n_landmarks::Union{Nothing,Int}=nothing,
                      nn_backend::Symbol=:auto,
                      nn_approx_candidates::Int=0,
                      density_k::Int=2,
                      construction::ConstructionOptions=ConstructionOptions()) =
    RipsDensityFiltration((;
        max_dim,
        radius = radius === nothing ? nothing : Float64(radius),
        knn,
        n_landmarks,
        nn_backend,
        nn_approx_candidates,
        density_k,
        construction,
    ))

struct RipsCodensityFiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
RipsCodensityFiltration(; max_dim::Int=1,
                        radius::Union{Nothing,Real}=nothing,
                        knn::Union{Nothing,Int}=nothing,
                        n_landmarks::Union{Nothing,Int}=nothing,
                        nn_backend::Symbol=:auto,
                        nn_approx_candidates::Int=0,
                        dtm_mass::Real=0.1,
                        construction::ConstructionOptions=ConstructionOptions()) =
    RipsCodensityFiltration((;
        max_dim,
        radius = radius === nothing ? nothing : Float64(radius),
        knn,
        n_landmarks,
        nn_backend,
        nn_approx_candidates,
        dtm_mass = Float64(dtm_mass),
        construction,
    ))

struct RipsLowerStarFiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
RipsLowerStarFiltration(; max_dim::Int=1,
                        radius::Union{Nothing,Real}=nothing,
                        knn::Union{Nothing,Int}=nothing,
                        n_landmarks::Union{Nothing,Int}=nothing,
                        nn_backend::Symbol=:auto,
                        nn_approx_candidates::Int=0,
                        coord::Int=1,
                        construction::ConstructionOptions=ConstructionOptions()) =
    RipsLowerStarFiltration((;
        max_dim,
        radius = radius === nothing ? nothing : Float64(radius),
        knn,
        n_landmarks,
        nn_backend,
        nn_approx_candidates,
        coord,
        construction,
    ))

struct FunctionRipsFiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
FunctionRipsFiltration(; max_dim::Int=1,
                       radius::Union{Nothing,Real}=nothing,
                       knn::Union{Nothing,Int}=nothing,
                       n_landmarks::Union{Nothing,Int}=nothing,
                       nn_backend::Symbol=:auto,
                       nn_approx_candidates::Int=0,
                       vertex_values=nothing,
                       vertex_function=nothing,
                       simplex_agg::Symbol=:max,
                       construction::ConstructionOptions=ConstructionOptions()) =
    FunctionRipsFiltration((;
        max_dim,
        radius = radius === nothing ? nothing : Float64(radius),
        knn,
        n_landmarks,
        nn_backend,
        nn_approx_candidates,
        vertex_values,
        vertex_function,
        simplex_agg,
        construction,
    ))

struct LandmarkRipsFiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
LandmarkRipsFiltration(; max_dim::Int=1,
                        landmarks,
                        radius::Union{Nothing,Real}=nothing,
                        knn::Union{Nothing,Int}=nothing,
                        nn_backend::Symbol=:auto,
                        nn_approx_candidates::Int=0,
                        construction::ConstructionOptions=ConstructionOptions()) =
    LandmarkRipsFiltration((;
        max_dim,
        landmarks,
        radius = radius === nothing ? nothing : Float64(radius),
        knn,
        nn_backend,
        nn_approx_candidates,
        construction,
    ))

struct GraphLowerStarFiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
GraphLowerStarFiltration(; vertex_grades=nothing,
                         vertex_values=nothing,
                         vertex_function=nothing,
                         simplex_agg::Symbol=:max,
                         construction::ConstructionOptions=ConstructionOptions()) =
    GraphLowerStarFiltration((; vertex_grades, vertex_values, vertex_function, simplex_agg, construction))

struct CliqueLowerStarFiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
CliqueLowerStarFiltration(; max_dim::Int=2,
                          vertex_grades=nothing,
                          vertex_values=nothing,
                          vertex_function=nothing,
                          simplex_agg::Symbol=:max,
                          construction::ConstructionOptions=ConstructionOptions()) =
    CliqueLowerStarFiltration((; max_dim, vertex_grades, vertex_values, vertex_function, simplex_agg, construction))

struct EdgeWeightedFiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
EdgeWeightedFiltration(; edge_weights,
                        construction::ConstructionOptions=ConstructionOptions()) =
    EdgeWeightedFiltration((; edge_weights, construction))

struct GraphCentralityFiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
GraphCentralityFiltration(;
                          centrality::Symbol=:degree,
                          metric::Symbol=:hop,
                          lift::Symbol=:lower_star,
                          max_dim::Int=2,
                          simplex_agg::Symbol=:max,
                          edge_weights=nothing,
                          construction::ConstructionOptions=ConstructionOptions()) =
    GraphCentralityFiltration((; centrality, metric, lift, max_dim, simplex_agg, edge_weights, construction))

struct GraphGeodesicFiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
GraphGeodesicFiltration(;
                        sources,
                        metric::Symbol=:hop,
                        lift::Symbol=:lower_star,
                        max_dim::Int=2,
                        simplex_agg::Symbol=:max,
                        edge_weights=nothing,
                        construction::ConstructionOptions=ConstructionOptions()) =
    GraphGeodesicFiltration((; sources, metric, lift, max_dim, simplex_agg, edge_weights, construction))

struct GraphFunctionGeodesicBifiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
GraphFunctionGeodesicBifiltration(;
                                  sources,
                                  metric::Symbol=:hop,
                                  vertex_values=nothing,
                                  vertex_function=nothing,
                                  lift::Symbol=:lower_star,
                                  max_dim::Int=2,
                                  simplex_agg::Symbol=:max,
                                  edge_weights=nothing,
                                  construction::ConstructionOptions=ConstructionOptions()) =
    GraphFunctionGeodesicBifiltration((; sources, metric, vertex_values, vertex_function, lift, max_dim, simplex_agg, edge_weights, construction))

struct GraphWeightThresholdFiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
GraphWeightThresholdFiltration(;
                               edge_weights=nothing,
                               lift::Symbol=:graph,
                               max_dim::Int=2,
                               construction::ConstructionOptions=ConstructionOptions()) =
    GraphWeightThresholdFiltration((; edge_weights, lift, max_dim, construction))

struct ImageLowerStarFiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
ImageLowerStarFiltration(; channels=nothing,
                          construction::ConstructionOptions=ConstructionOptions()) =
    ImageLowerStarFiltration((; channels, construction))

struct ImageDistanceBifiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
ImageDistanceBifiltration(; mask=nothing, channels=nothing,
                           construction::ConstructionOptions=ConstructionOptions()) =
    ImageDistanceBifiltration((; mask, channels, construction))

struct WingVeinBifiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
WingVeinBifiltration(; grid=(32, 32), bbox=nothing, orientation=(-1, 1),
                      construction::ConstructionOptions=ConstructionOptions()) =
    WingVeinBifiltration((; grid, bbox, orientation, construction))

struct DelaunayLowerStarFiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
DelaunayLowerStarFiltration(; max_dim::Int=2,
                            vertex_values=nothing,
                            vertex_function=nothing,
                            simplex_agg::Symbol=:max,
                            delaunay_backend::Symbol=:auto,
                            highdim_policy::Symbol=:rips,
                            construction::ConstructionOptions=ConstructionOptions()) =
    DelaunayLowerStarFiltration((; max_dim, vertex_values, vertex_function, simplex_agg, delaunay_backend, highdim_policy, construction))

struct AlphaFiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
AlphaFiltration(; max_dim::Int=2,
                delaunay_backend::Symbol=:auto,
                highdim_policy::Symbol=:rips,
                construction::ConstructionOptions=ConstructionOptions()) =
    AlphaFiltration((; max_dim, delaunay_backend, highdim_policy, construction))

struct FunctionDelaunayFiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
FunctionDelaunayFiltration(; max_dim::Int=2,
                           vertex_values=nothing,
                           vertex_function=nothing,
                           simplex_agg::Symbol=:max,
                           delaunay_backend::Symbol=:auto,
                           highdim_policy::Symbol=:rips,
                           construction::ConstructionOptions=ConstructionOptions()) =
    FunctionDelaunayFiltration((; max_dim, vertex_values, vertex_function, simplex_agg, delaunay_backend, highdim_policy, construction))

struct CoreDelaunayFiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
CoreDelaunayFiltration(; max_dim::Int=2,
                       delaunay_backend::Symbol=:auto,
                       highdim_policy::Symbol=:rips,
                       construction::ConstructionOptions=ConstructionOptions()) =
    CoreDelaunayFiltration((; max_dim, delaunay_backend, highdim_policy, construction))

struct CoreFiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
CoreFiltration(; radius=nothing,
               knn::Int=8,
               nn_backend::Symbol=:auto,
               nn_approx_candidates::Int=0,
               vertex_values=nothing,
               vertex_function=nothing,
               construction::ConstructionOptions=ConstructionOptions()) =
    CoreFiltration((; radius, knn, nn_backend, nn_approx_candidates, vertex_values, vertex_function, construction))

struct DegreeRipsFiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
DegreeRipsFiltration(; max_dim::Int=1,
                     radius::Union{Nothing,Real}=nothing,
                     knn::Union{Nothing,Int}=nothing,
                     n_landmarks::Union{Nothing,Int}=nothing,
                     nn_backend::Symbol=:auto,
                     nn_approx_candidates::Int=0,
                     construction::ConstructionOptions=ConstructionOptions()) =
    DegreeRipsFiltration((;
        max_dim,
        radius = radius === nothing ? nothing : Float64(radius),
        knn,
        n_landmarks,
        nn_backend,
        nn_approx_candidates,
        construction,
    ))

struct CubicalFiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
CubicalFiltration(; channels=nothing,
                  construction::ConstructionOptions=ConstructionOptions()) =
    CubicalFiltration((; channels, construction))

struct RhomboidFiltration{P<:NamedTuple} <: AbstractFiltration
    params::P
end
RhomboidFiltration(; max_dim::Int=2,
                   radius=nothing,
                   knn::Union{Nothing,Int}=nothing,
                   n_landmarks::Union{Nothing,Int}=nothing,
                   nn_backend::Symbol=:auto,
                   nn_approx_candidates::Int=0,
                   vertex_values=nothing,
                   vertex_function=nothing,
                   construction::ConstructionOptions=ConstructionOptions()) =
    RhomboidFiltration((; max_dim, radius, knn, n_landmarks, nn_backend, nn_approx_candidates, vertex_values, vertex_function, construction))

filtration_kind(::Type{<:GradedFiltration}) = :graded
filtration_kind(::Type{<:RipsFiltration}) = :rips
filtration_kind(::Type{<:RipsDensityFiltration}) = :rips_density
filtration_kind(::Type{<:RipsCodensityFiltration}) = :rips_codensity
filtration_kind(::Type{<:RipsLowerStarFiltration}) = :rips_lowerstar
filtration_kind(::Type{<:FunctionRipsFiltration}) = :function_rips
filtration_kind(::Type{<:LandmarkRipsFiltration}) = :landmark_rips
filtration_kind(::Type{<:GraphLowerStarFiltration}) = :graph_lower_star
filtration_kind(::Type{<:CliqueLowerStarFiltration}) = :clique_lower_star
filtration_kind(::Type{<:EdgeWeightedFiltration}) = :edge_weighted
filtration_kind(::Type{<:GraphCentralityFiltration}) = :graph_centrality
filtration_kind(::Type{<:GraphGeodesicFiltration}) = :graph_geodesic
filtration_kind(::Type{<:GraphFunctionGeodesicBifiltration}) = :graph_function_geodesic_bifiltration
filtration_kind(::Type{<:GraphWeightThresholdFiltration}) = :graph_weight_threshold
filtration_kind(::Type{<:ImageLowerStarFiltration}) = :lower_star
filtration_kind(::Type{<:ImageDistanceBifiltration}) = :image_distance_bifiltration
filtration_kind(::Type{<:WingVeinBifiltration}) = :wing_vein_bifiltration
filtration_kind(::Type{<:DelaunayLowerStarFiltration}) = :delaunay_lower_star
filtration_kind(::Type{<:AlphaFiltration}) = :alpha
filtration_kind(::Type{<:FunctionDelaunayFiltration}) = :function_delaunay
filtration_kind(::Type{<:CoreDelaunayFiltration}) = :core_delaunay
filtration_kind(::Type{<:CoreFiltration}) = :core
filtration_kind(::Type{<:DegreeRipsFiltration}) = :degree_rips
filtration_kind(::Type{<:CubicalFiltration}) = :cubical
filtration_kind(::Type{<:RhomboidFiltration}) = :rhomboid

@inline function _params_with_nonnothing(params::NamedTuple)
    return (; (k => (k == :construction && v isa ConstructionOptions ? _construction_to_namedtuple(v) : v)
             for (k, v) in pairs(params) if v !== nothing)...)
end

const _CONSTRUCTION_LEGACY_KEYS = (
    :sparse_rips, :approx_rips, :max_edges, :max_degree, :sample_frac, :max_simplices
)

@inline function _construction_to_namedtuple(c::ConstructionOptions)
    return (
        sparsify = c.sparsify,
        collapse = c.collapse,
        output_stage = c.output_stage,
        budget = (
            max_simplices = c.budget.max_simplices,
            max_edges = c.budget.max_edges,
            memory_budget_bytes = c.budget.memory_budget_bytes,
        ),
    )
end

function _construction_from_raw(raw;
                                default_output_stage::Symbol=:encoding_result)
    if raw === nothing
        return ConstructionOptions(; output_stage=default_output_stage)
    elseif raw isa ConstructionOptions
        return raw
    elseif raw isa NamedTuple
        return ConstructionOptions(;
            sparsify = Symbol(get(raw, :sparsify, :none)),
            collapse = Symbol(get(raw, :collapse, :none)),
            output_stage = Symbol(get(raw, :output_stage, default_output_stage)),
            budget = get(raw, :budget, (nothing, nothing, nothing)),
        )
    elseif raw isa AbstractDict
        sparsify = Symbol(get(raw, "sparsify", get(raw, :sparsify, :none)))
        collapse = Symbol(get(raw, "collapse", get(raw, :collapse, :none)))
        output_stage = Symbol(get(raw, "output_stage", get(raw, :output_stage, default_output_stage)))
        budget_raw = get(raw, "budget", get(raw, :budget, (nothing, nothing, nothing)))
        if budget_raw isa AbstractDict
            budget = (
                max_simplices = get(budget_raw, "max_simplices", get(budget_raw, :max_simplices, nothing)),
                max_edges = get(budget_raw, "max_edges", get(budget_raw, :max_edges, nothing)),
                memory_budget_bytes = get(budget_raw, "memory_budget_bytes", get(budget_raw, :memory_budget_bytes, nothing)),
            )
        else
            budget = budget_raw
        end
        return ConstructionOptions(; sparsify=sparsify, collapse=collapse, output_stage=output_stage, budget=budget)
    end
    throw(ArgumentError("Invalid construction payload. Expected ConstructionOptions, NamedTuple, AbstractDict, or nothing."))
end

function _construction_from_params(params::NamedTuple;
                                   default_output_stage::Symbol=:encoding_result)
    for k in _CONSTRUCTION_LEGACY_KEYS
        if haskey(params, k)
            throw(ArgumentError("FiltrationSpec uses deprecated key `$(k)`. Use `construction=ConstructionOptions(...)` with budget/sparsify settings."))
        end
    end
    raw = haskey(params, :construction) ? params[:construction] : nothing
    return _construction_from_raw(raw; default_output_stage=default_output_stage)
end

function _construction_from_filtration(f::AbstractFiltration;
                                       default_output_stage::Symbol=:encoding_result)
    if hasproperty(f, :params)
        p = getproperty(f, :params)
        if p isa NamedTuple && haskey(p, :construction)
            return _construction_from_raw(p[:construction]; default_output_stage=default_output_stage)
        end
    end
    return ConstructionOptions(; output_stage=default_output_stage)
end

"""
    IngestionEstimate

Typed preflight estimate returned by [`estimate_ingestion`](@ref).

Use `describe(...)`, [`ingestion_estimate_summary`](@ref), and the scalar
accessors in `DataIngestion` for cheap-first inspection before running the
heavier ingestion stages.
"""
struct IngestionEstimate{R}
    report::R
end

"""
    IngestionValidationSummary

Readable wrapper around a validation report returned by the `check_*`
ingestion helpers in this module.
"""
struct IngestionValidationSummary{R}
    report::R
end

"""
    GradedComplexBuildResult

Typed owner-level wrapper around the graded-complex builder payload
`(G, axes, orientation)`.

Use [`describe(...)`](@ref) or [`graded_complex_build_summary`](@ref) for the
cheap structural picture, then [`graded_complex`](@ref), [`grade_axes`](@ref),
and [`grade_orientation`](@ref) when you need the full payload.
"""
struct GradedComplexBuildResult{G,A,O}
    complex::G
    axes::A
    orientation::O
end

"""
    PointCodensityResult

Typed owner-level wrapper for pointwise codensity values attached to a
`PointCloud` and a `:rips_codensity` filtration contract.

Use `describe(...)` for the cheap structural picture, then
`codensity_values(...)`, `codensity_mass(...)`, and `neighbor_count(...)` when
you need the underlying scalar data. The result also supports notebook-facing
visualization via `available_visuals(...)` and `visualize(...)`.
"""
struct PointCodensityResult{D,F,V}
    data::D
    filtration::F
    values::V
    k::Int
end

GradedComplexBuildResult(out::Tuple) =
    length(out) == 3 ? GradedComplexBuildResult(out[1], out[2], out[3]) :
    throw(ArgumentError("GradedComplexBuildResult expects a tuple `(G, axes, orientation)`."))

@inline estimated_cells(est::IngestionEstimate) = getfield(est, :report).n_cells_est
@inline cell_counts_by_dim(est::IngestionEstimate) = getfield(est, :report).cell_counts_by_dim
@inline estimated_axis_sizes(est::IngestionEstimate) = getfield(est, :report).axis_sizes
@inline estimated_poset_size(est::IngestionEstimate) = getfield(est, :report).poset_size
@inline estimated_nnz(est::IngestionEstimate) = getfield(est, :report).nnz_est
@inline estimated_dense_bytes(est::IngestionEstimate) = getfield(est, :report).dense_bytes_est
@inline estimate_warnings(est::IngestionEstimate) = getfield(est, :report).warnings

"""
    graded_complex(res::GradedComplexBuildResult)

Return the constructed graded complex carried by a
[`GradedComplexBuildResult`](@ref).
"""
@inline graded_complex(res::GradedComplexBuildResult) = getfield(res, :complex)

"""
    grade_axes(res::GradedComplexBuildResult)

Return the grade axes paired with the constructed complex in a
[`GradedComplexBuildResult`](@ref).
"""
@inline grade_axes(res::GradedComplexBuildResult) = getfield(res, :axes)

"""
    grade_orientation(res::GradedComplexBuildResult)

Return the axis-orientation tuple carried by a
[`GradedComplexBuildResult`](@ref).
"""
@inline grade_orientation(res::GradedComplexBuildResult) = getfield(res, :orientation)

"""
    codensity_values(res::PointCodensityResult)

Return the pointwise codensity values carried by a
[`PointCodensityResult`](@ref).
"""
@inline codensity_values(res::PointCodensityResult) = getfield(res, :values)

"""
    codensity_mass(res::PointCodensityResult) -> Float64

Return the DTM mass parameter used to compute the codensity values carried by
[`PointCodensityResult`](@ref).
"""
@inline function codensity_mass(res::PointCodensityResult)
    params = filtration_parameters(getfield(res, :filtration))
    haskey(params, :dtm_mass) ||
        error("PointCodensityResult filtration is missing canonical `dtm_mass`.")
    return Float64(params.dtm_mass)
end

"""
    neighbor_count(res::PointCodensityResult) -> Int

Return the effective DTM neighbor count `k = floor(mass * n) + 1` used for the
codensity values in [`PointCodensityResult`](@ref).
"""
@inline neighbor_count(res::PointCodensityResult) = getfield(res, :k)

"""
    ingestion_validation_summary(report) -> IngestionValidationSummary

Wrap an ingestion validation report in a readable summary object for notebook
and REPL inspection.
"""
@inline ingestion_validation_summary(report) = IngestionValidationSummary(report)

@inline function _construction_mode_summary(c::ConstructionOptions)
    return (
        sparsify = c.sparsify,
        collapse = c.collapse,
        output_stage = c.output_stage,
    )
end

@inline function _ingestion_value_summary(x)
    if x isa ConstructionOptions
        return _construction_mode_summary(x)
    elseif x isa AbstractVector
        return (type=Symbol(nameof(typeof(x))), length=length(x))
    elseif x isa AbstractMatrix
        return (type=Symbol(nameof(typeof(x))), size=size(x))
    elseif x isa Tuple
        return length(x) <= 6 ? map(_ingestion_value_summary, x) : (length=length(x),)
    elseif x isa NamedTuple
        return NamedTuple{keys(x)}(map(_ingestion_value_summary, values(x)))
    elseif x isa Function
        return Symbol(nameof(typeof(x)))
    elseif x isa AbstractCoeffField
        return Symbol(nameof(typeof(x)))
    end
    return x
end

@inline function _filtration_key_params(f::AbstractFiltration)
    params = filtration_parameters(f)
    pairs_keep = Pair{Symbol,Any}[]
    for (k, v) in pairs(params)
        k == :construction && continue
        push!(pairs_keep, k => _ingestion_value_summary(v))
    end
    isempty(pairs_keep) && return NamedTuple()
    return (; pairs_keep...)
end

@inline function _filtration_spec_key_params(spec::FiltrationSpec)
    params = filtration_parameters(spec)
    pairs_keep = Pair{Symbol,Any}[]
    for (k, v) in pairs(params)
        k == :construction && continue
        push!(pairs_keep, k => _ingestion_value_summary(v))
    end
    isempty(pairs_keep) && return NamedTuple()
    return (; pairs_keep...)
end

@inline function _ingestion_data_kind(data)
    if data isa PointCloud
        return :point_cloud
    elseif data isa ImageNd
        return :image
    elseif data isa GraphData
        return :graph
    elseif data isa EmbeddedPlanarGraph2D
        return :embedded_planar_graph_2d
    elseif data isa GradedComplex
        return :graded_complex
    elseif data isa MultiCriticalGradedComplex
        return :multicritical_graded_complex
    elseif data isa SimplexTreeMulti
        return :simplex_tree_multi
    end
    return Symbol(nameof(typeof(data)))
end

@inline function _point_codensity_describe(res::PointCodensityResult)
    vals = codensity_values(res)
    return (
        kind = :point_codensity_result,
        data_kind = _ingestion_data_kind(getfield(res, :data)),
        npoints = DataTypes.npoints(getfield(res, :data)),
        ambient_dim = DataTypes.ambient_dim(getfield(res, :data)),
        filtration_kind = getfield(res, :filtration) isa FiltrationSpec ?
                          filtration_kind(getfield(res, :filtration)) :
                          filtration_kind(typeof(getfield(res, :filtration))),
        dtm_mass = codensity_mass(res),
        neighbor_count = neighbor_count(res),
        value_range = isempty(vals) ? nothing : extrema(vals),
    )
end

@inline function _filtration_describe(f::AbstractFiltration)
    construction = _construction_from_filtration(f)
    return (
        kind = filtration_kind(typeof(f)),
        filtration_type = Symbol(nameof(typeof(f))),
        arity = filtration_arity(f),
        key_params = _filtration_key_params(f),
        construction_mode = _construction_mode_summary(construction),
    )
end

@inline function _filtration_family_describe(kind::Symbol;
                                             provided_parameters::Union{Nothing,Tuple}=nothing)
    sig = filtration_signature(kind)
    return (
        kind = :filtration_family,
        filtration_kind = sig.kind,
        registered = sig.registered,
        arity = sig.arity,
        required = sig.required,
        default_parameters = Tuple(keys(sig.defaults)),
        typed_parameters = Tuple(keys(sig.types)),
        checked_parameters = Tuple(keys(sig.checks)),
        provided_parameters = provided_parameters,
    )
end

@inline function _filtration_spec_describe(spec::FiltrationSpec)
    sig = filtration_signature(filtration_kind(spec))
    return (
        kind = :filtration_spec,
        filtration_kind = filtration_kind(spec),
        registered = sig.registered,
        arity = sig.arity,
        nparameters = length(filtration_parameters(spec)),
        key_params = _filtration_spec_key_params(spec),
        construction_mode = _construction_mode_summary(construction_mode(spec)),
    )
end

"""
    describe(spec::FiltrationSpec) -> NamedTuple

Return a compact semantic summary of a serialized filtration specification.

Use this when you are inspecting a `FiltrationSpec` before conversion with
[`to_filtration`](@ref) or before planning with [`plan_ingestion`](@ref).
Prefer [`filtration_spec_summary`](@ref) when you want the owner-local
`DataIngestion` entrypoint.
"""
describe(spec::FiltrationSpec) = _filtration_spec_describe(spec)

"""
    describe(filtration::AbstractFiltration) -> NamedTuple

Return a compact semantic summary of a typed ingestion filtration.

This shared inspection surface exposes the filtration family, persistence
parameter arity, the non-`nothing` key parameters, and the construction mode
that would be threaded into the ingestion pipeline. Use
[`filtration_summary`](@ref) when you want the owner-local entrypoint inside
`DataIngestion`.
"""
describe(f::AbstractFiltration) = _filtration_describe(f)

"""
    filtration_summary(filtration::AbstractFiltration) -> NamedTuple

Owner-local summary helper for typed ingestion filtrations.

This is the cheap-first inspection path for `DataIngestion` users working with
typed filtrations directly. Start here before calling `estimate_ingestion(...)`
or `plan_ingestion(...)` when you want to confirm the family, arity, and
construction mode that will drive ingestion. Use [`construction_mode`](@ref)
and [`filtration_parameters`](@ref) when you need the resolved construction
contract or full parameter payload directly.
"""
filtration_summary(f::AbstractFiltration) = _filtration_describe(f)

"""
    filtration_spec_summary(spec::FiltrationSpec) -> NamedTuple

Owner-local summary helper for serialized filtration specifications.

This is the cheap-first inspection path for `FiltrationSpec` objects before
calling [`to_filtration`](@ref), [`plan_ingestion`](@ref), or
[`estimate_ingestion`](@ref).
"""
filtration_spec_summary(spec::FiltrationSpec) = _filtration_spec_describe(spec)

"""
    filtration_family_summary(kind_or_spec) -> NamedTuple

Return the declared family contract for a filtration kind or serialized
[`FiltrationSpec`](@ref).

Use this when you want the family-level schema surface exposed by
[`register_filtration_family!`](@ref) and [`filtration_signature`](@ref)
without manually unpacking raw schema tuples.
"""
filtration_family_summary(kind::Symbol) = _filtration_family_describe(kind)
filtration_family_summary(spec::FiltrationSpec) =
    _filtration_family_describe(filtration_kind(spec);
                                provided_parameters=Tuple(keys(filtration_parameters(spec))))

function Base.show(io::IO, f::AbstractFiltration)
    d = _filtration_describe(f)
    print(io, nameof(typeof(f)),
          "(kind=", d.kind,
          ", arity=", d.arity,
          ", params=", collect(keys(d.key_params)),
          ", output_stage=", d.construction_mode.output_stage, ")")
end

function Base.show(io::IO, ::MIME"text/plain", f::AbstractFiltration)
    d = _filtration_describe(f)
    print(io, nameof(typeof(f)),
          "\n  kind: ", d.kind,
          "\n  arity: ", d.arity,
          "\n  key_params: ", repr(d.key_params),
          "\n  construction_mode: ", repr(d.construction_mode))
end

function _filtration_spec(::GradedFiltration)
    return FiltrationSpec(kind=:graded)
end

function _filtration_spec(f::RipsFiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:rips, p...)
end

function _filtration_spec(f::RipsDensityFiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:rips_density, p...)
end

function _filtration_spec(f::RipsCodensityFiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:rips_codensity, p...)
end

function _filtration_spec(f::RipsLowerStarFiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:rips_lowerstar, p...)
end

function _filtration_spec(f::FunctionRipsFiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:function_rips, p...)
end

function _filtration_spec(f::LandmarkRipsFiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:landmark_rips, p...)
end

function _filtration_spec(f::GraphLowerStarFiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:graph_lower_star, p...)
end

function _filtration_spec(f::CliqueLowerStarFiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:clique_lower_star, p...)
end

function _filtration_spec(f::EdgeWeightedFiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:edge_weighted, p...)
end

function _filtration_spec(f::GraphCentralityFiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:graph_centrality, p...)
end

function _filtration_spec(f::GraphGeodesicFiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:graph_geodesic, p...)
end

function _filtration_spec(f::GraphFunctionGeodesicBifiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:graph_function_geodesic_bifiltration, p...)
end

function _filtration_spec(f::GraphWeightThresholdFiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:graph_weight_threshold, p...)
end

function _filtration_spec(f::ImageLowerStarFiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:lower_star, p...)
end

function _filtration_spec(f::ImageDistanceBifiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:image_distance_bifiltration, p...)
end

function _filtration_spec(f::WingVeinBifiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:wing_vein_bifiltration, p...)
end

function _filtration_spec(f::CubicalFiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:cubical, p...)
end

function _filtration_spec(f::DelaunayLowerStarFiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:delaunay_lower_star, p...)
end

function _filtration_spec(f::AlphaFiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:alpha, p...)
end

function _filtration_spec(f::FunctionDelaunayFiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:function_delaunay, p...)
end

function _filtration_spec(f::CoreDelaunayFiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:core_delaunay, p...)
end

function _filtration_spec(f::CoreFiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:core, p...)
end

function _filtration_spec(f::DegreeRipsFiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:degree_rips, p...)
end

function _filtration_spec(f::RhomboidFiltration)
    p = _params_with_nonnothing(f.params)
    return FiltrationSpec(; kind=:rhomboid, p...)
end

function _filtration_spec(f::AbstractFiltration)
    kind = filtration_kind(typeof(f))
    params = filtration_parameters(f)
    entry = _filtration_registry_get(kind)
    params2 = entry === nothing ? params :
              _normalize_filtration_params(kind, params, entry.schema;
                                           context="_filtration_spec")
    return FiltrationSpec(; kind=kind, params2...)
end

"""
    save_pipeline_json(path, data, filtration::AbstractFiltration; degree=nothing, pipeline_opts=nothing)

Serialize a typed filtration by first converting it to canonical `FiltrationSpec`.
"""
function Serialization.save_pipeline_json(path::AbstractString, data, filtration::AbstractFiltration; degree=nothing, pipeline_opts=nothing)
    return Serialization.save_pipeline_json(path, data, _filtration_spec(filtration); degree=degree, pipeline_opts=pipeline_opts)
end

"""
    to_filtration(spec::FiltrationSpec) -> AbstractFiltration

Convert serialized `FiltrationSpec` into a typed runtime filtration object.
"""
function to_filtration(spec::FiltrationSpec)::AbstractFiltration
    p = spec.params
    k = spec.kind
    construction = _construction_from_params(p)
    if k === :graded
        return GradedFiltration()
    elseif k === :rips
        return RipsFiltration(;
            max_dim = get(p, :max_dim, 1),
            radius = get(p, :radius, nothing),
            knn = get(p, :knn, nothing),
            n_landmarks = get(p, :n_landmarks, nothing),
            nn_backend = Symbol(get(p, :nn_backend, :auto)),
            nn_approx_candidates = Int(get(p, :nn_approx_candidates, 0)),
            construction = construction,
        )
    elseif k === :rips_density
        return RipsDensityFiltration(;
            max_dim = get(p, :max_dim, 1),
            radius = get(p, :radius, nothing),
            knn = get(p, :knn, nothing),
            n_landmarks = get(p, :n_landmarks, nothing),
            nn_backend = Symbol(get(p, :nn_backend, :auto)),
            nn_approx_candidates = Int(get(p, :nn_approx_candidates, 0)),
            density_k = get(p, :density_k, 2),
            construction = construction,
        )
    elseif k === :rips_codensity
        return RipsCodensityFiltration(;
            max_dim = get(p, :max_dim, 1),
            radius = get(p, :radius, nothing),
            knn = get(p, :knn, nothing),
            n_landmarks = get(p, :n_landmarks, nothing),
            nn_backend = Symbol(get(p, :nn_backend, :auto)),
            nn_approx_candidates = Int(get(p, :nn_approx_candidates, 0)),
            dtm_mass = Float64(get(p, :dtm_mass, 0.1)),
            construction = construction,
        )
    elseif k === :rips_lowerstar
        return RipsLowerStarFiltration(;
            max_dim = get(p, :max_dim, 1),
            radius = get(p, :radius, nothing),
            knn = get(p, :knn, nothing),
            n_landmarks = get(p, :n_landmarks, nothing),
            nn_backend = Symbol(get(p, :nn_backend, :auto)),
            nn_approx_candidates = Int(get(p, :nn_approx_candidates, 0)),
            coord = Int(get(p, :coord, 1)),
            construction = construction,
        )
    elseif k === :function_rips
        return FunctionRipsFiltration(;
            max_dim = get(p, :max_dim, 1),
            radius = get(p, :radius, nothing),
            knn = get(p, :knn, nothing),
            n_landmarks = get(p, :n_landmarks, nothing),
            nn_backend = Symbol(get(p, :nn_backend, :auto)),
            nn_approx_candidates = Int(get(p, :nn_approx_candidates, 0)),
            vertex_values = get(p, :vertex_values, nothing),
            vertex_function = get(p, :vertex_function, nothing),
            simplex_agg = get(p, :simplex_agg, :max),
            construction = construction,
        )
    elseif k === :landmark_rips
        landmarks = get(p, :landmarks, nothing)
        landmarks === nothing && error("to_filtration: landmark_rips requires landmarks.")
        return LandmarkRipsFiltration(;
            max_dim = get(p, :max_dim, 1),
            landmarks = landmarks,
            radius = get(p, :radius, nothing),
            knn = get(p, :knn, nothing),
            nn_backend = Symbol(get(p, :nn_backend, :auto)),
            nn_approx_candidates = Int(get(p, :nn_approx_candidates, 0)),
            construction = construction,
        )
    elseif k === :graph_lower_star
        return GraphLowerStarFiltration(;
            vertex_grades = get(p, :vertex_grades, nothing),
            vertex_values = get(p, :vertex_values, nothing),
            vertex_function = get(p, :vertex_function, nothing),
            simplex_agg = get(p, :simplex_agg, :max),
            construction = construction,
        )
    elseif k === :clique_lower_star
        return CliqueLowerStarFiltration(;
            max_dim = get(p, :max_dim, 2),
            vertex_grades = get(p, :vertex_grades, nothing),
            vertex_values = get(p, :vertex_values, nothing),
            vertex_function = get(p, :vertex_function, nothing),
            simplex_agg = get(p, :simplex_agg, :max),
            construction = construction,
        )
    elseif k === :edge_weighted
        edge_weights = get(p, :edge_weights, nothing)
        edge_weights === nothing && error("to_filtration: edge_weighted requires edge_weights.")
        return EdgeWeightedFiltration(; edge_weights=edge_weights, construction=construction)
    elseif k === :graph_centrality
        return GraphCentralityFiltration(;
            centrality = get(p, :centrality, :degree),
            metric = get(p, :metric, :hop),
            lift = get(p, :lift, :lower_star),
            max_dim = get(p, :max_dim, 2),
            simplex_agg = get(p, :simplex_agg, :max),
            edge_weights = get(p, :edge_weights, nothing),
            construction = construction,
        )
    elseif k === :graph_geodesic
        sources = get(p, :sources, nothing)
        sources === nothing && error("to_filtration: graph_geodesic requires sources.")
        return GraphGeodesicFiltration(;
            sources = sources,
            metric = get(p, :metric, :hop),
            lift = get(p, :lift, :lower_star),
            max_dim = get(p, :max_dim, 2),
            simplex_agg = get(p, :simplex_agg, :max),
            edge_weights = get(p, :edge_weights, nothing),
            construction = construction,
        )
    elseif k === :graph_function_geodesic_bifiltration
        sources = get(p, :sources, nothing)
        sources === nothing && error("to_filtration: graph_function_geodesic_bifiltration requires sources.")
        return GraphFunctionGeodesicBifiltration(;
            sources = sources,
            metric = get(p, :metric, :hop),
            vertex_values = get(p, :vertex_values, nothing),
            vertex_function = get(p, :vertex_function, nothing),
            lift = get(p, :lift, :lower_star),
            max_dim = get(p, :max_dim, 2),
            simplex_agg = get(p, :simplex_agg, :max),
            edge_weights = get(p, :edge_weights, nothing),
            construction = construction,
        )
    elseif k === :graph_weight_threshold
        return GraphWeightThresholdFiltration(;
            edge_weights = get(p, :edge_weights, nothing),
            lift = get(p, :lift, :graph),
            max_dim = get(p, :max_dim, 2),
            construction = construction,
        )
    elseif k === :lower_star
        return ImageLowerStarFiltration(; channels=get(p, :channels, nothing), construction=construction)
    elseif k === :image_distance_bifiltration
        return ImageDistanceBifiltration(; mask=get(p, :mask, nothing), channels=get(p, :channels, nothing), construction=construction)
    elseif k === :wing_vein_bifiltration
        return WingVeinBifiltration(;
            grid = get(p, :grid, (32, 32)),
            bbox = get(p, :bbox, nothing),
            orientation = get(p, :orientation, (-1, 1)),
            construction = construction,
        )
    elseif k === :cubical
        return CubicalFiltration(;
            channels = get(p, :channels, nothing),
            construction = construction,
        )
    elseif k === :delaunay_lower_star
        return DelaunayLowerStarFiltration(;
            max_dim = get(p, :max_dim, 2),
            vertex_values = get(p, :vertex_values, nothing),
            vertex_function = get(p, :vertex_function, nothing),
            simplex_agg = get(p, :simplex_agg, :max),
            delaunay_backend = Symbol(get(p, :delaunay_backend, :auto)),
            highdim_policy = get(p, :highdim_policy, :rips),
            construction = construction,
        )
    elseif k === :alpha
        return AlphaFiltration(;
            max_dim = get(p, :max_dim, 2),
            delaunay_backend = Symbol(get(p, :delaunay_backend, :auto)),
            highdim_policy = get(p, :highdim_policy, :rips),
            construction = construction,
        )
    elseif k === :function_delaunay
        return FunctionDelaunayFiltration(;
            max_dim = get(p, :max_dim, 2),
            vertex_values = get(p, :vertex_values, nothing),
            vertex_function = get(p, :vertex_function, nothing),
            simplex_agg = get(p, :simplex_agg, :max),
            delaunay_backend = Symbol(get(p, :delaunay_backend, :auto)),
            highdim_policy = get(p, :highdim_policy, :rips),
            construction = construction,
        )
    elseif k === :core_delaunay
        return CoreDelaunayFiltration(;
            max_dim = get(p, :max_dim, 2),
            delaunay_backend = Symbol(get(p, :delaunay_backend, :auto)),
            highdim_policy = get(p, :highdim_policy, :rips),
            construction = construction,
        )
    elseif k === :core
        return CoreFiltration(;
            radius = get(p, :radius, nothing),
            knn = get(p, :knn, 8),
            nn_backend = Symbol(get(p, :nn_backend, :auto)),
            nn_approx_candidates = Int(get(p, :nn_approx_candidates, 0)),
            vertex_values = get(p, :vertex_values, nothing),
            vertex_function = get(p, :vertex_function, nothing),
            construction = construction,
        )
    elseif k === :degree_rips
        return DegreeRipsFiltration(;
            max_dim = get(p, :max_dim, 1),
            radius = get(p, :radius, nothing),
            knn = get(p, :knn, nothing),
            n_landmarks = get(p, :n_landmarks, nothing),
            nn_backend = Symbol(get(p, :nn_backend, :auto)),
            nn_approx_candidates = Int(get(p, :nn_approx_candidates, 0)),
            construction = construction,
        )
    elseif k === :rhomboid
        return RhomboidFiltration(;
            max_dim = get(p, :max_dim, 2),
            radius = get(p, :radius, nothing),
            knn = get(p, :knn, nothing),
            n_landmarks = get(p, :n_landmarks, nothing),
            nn_backend = Symbol(get(p, :nn_backend, :auto)),
            nn_approx_candidates = Int(get(p, :nn_approx_candidates, 0)),
            vertex_values = get(p, :vertex_values, nothing),
            vertex_function = get(p, :vertex_function, nothing),
            construction = construction,
        )
    end
    entry = _filtration_registry_get(k)
    if entry !== nothing
        params2 = _normalize_filtration_params(k, p, entry.schema; context="to_filtration")
        spec2 = FiltrationSpec(; kind=k, params2...)
        filtration = if hasmethod(entry.ctor, Tuple{FiltrationSpec})
            entry.ctor(spec2)
        elseif hasmethod(entry.ctor, Tuple{NamedTuple})
            entry.ctor(params2)
        else
            try
                entry.ctor(spec2)
            catch err1
                try
                    entry.ctor(params2)
                catch err2
                    throw(ArgumentError("to_filtration kind=:$k: ctor must accept FiltrationSpec or NamedTuple.\nSpec call error: $(sprint(showerror, err1))\nParams call error: $(sprint(showerror, err2))"))
                end
            end
        end
        filtration isa AbstractFiltration ||
            throw(ArgumentError("to_filtration kind=:$k: ctor must return AbstractFiltration, got $(typeof(filtration))."))
        return filtration
    end
    error("to_filtration: unsupported kind=$(k).")
end

@inline function _ingestion_warn!(warnings::Vector{String}, msg::AbstractString, strict::Bool)
    strict && throw(ArgumentError(msg))
    push!(warnings, String(msg))
    return nothing
end

@inline function _prod_bigint(xs::Tuple)
    p = big(1)
    @inbounds for x in xs
        p *= big(x)
    end
    return p
end

@inline function _sum_bigint(xs::Vector{BigInt})
    s = big(0)
    @inbounds for x in xs
        s += x
    end
    return s
end

function _estimate_axis_sizes_from_spec(spec::FiltrationSpec)
    axes = get(spec.params, :axes, nothing)
    axes === nothing && return nothing
    if axes isa Tuple
        return ntuple(i -> length(axes[i]), length(axes))
    elseif axes isa AbstractVector
        return tuple((length(a) for a in axes)...)
    end
    return nothing
end

function _estimate_axis_sizes_from_grades(grades)
    isempty(grades) && return nothing
    N = length(grades[1])
    seen = [Set{Float64}() for _ in 1:N]
    for g in grades
        @inbounds for i in 1:N
            push!(seen[i], Float64(g[i]))
        end
    end
    return ntuple(i -> length(seen[i]), N)
end

function _estimate_axis_sizes_from_grades(grades::Vector{<:AbstractVector{<:Tuple}})
    isempty(grades) && return nothing
    first_cell = findfirst(!isempty, grades)
    first_cell === nothing && return nothing
    N = length(grades[first_cell][1])
    seen = [Set{Float64}() for _ in 1:N]
    for cell in grades
        for g in cell
            @inbounds for i in 1:N
                push!(seen[i], Float64(g[i]))
            end
        end
    end
    return ntuple(i -> length(seen[i]), N)
end

function _estimate_axis_sizes_from_multigrade_storage(grade_data::AbstractVector{<:Tuple})
    isempty(grade_data) && return nothing
    return _estimate_axis_sizes_from_grades(grade_data)
end

@inline function _estimate_nnz_from_cell_counts(cell_counts::Vector{BigInt})
    nnz_est = big(0)
    # idx = simplex_size = (dimension + 1)
    for idx in 2:length(cell_counts)
        nnz_est += cell_counts[idx] * big(idx)
    end
    return nnz_est
end

@inline function _estimate_dense_bytes_from_cell_counts(cell_counts::Vector{BigInt}; elem_bytes::Integer=8)
    peak = big(0)
    for idx in 2:length(cell_counts)
        dense = cell_counts[idx - 1] * cell_counts[idx] * big(elem_bytes)
        dense > peak && (peak = dense)
    end
    return peak
end

function _estimate_rips_like_cell_counts(n::Int,
                                         spec::FiltrationSpec;
                                         exact_pairwise_limit::Int,
                                         warnings::Vector{String},
                                         strict::Bool)
    n < 0 && throw(ArgumentError("estimate_ingestion: invalid point count $n"))
    max_dim = max(Int(get(spec.params, :max_dim, 1)), 0)
    construction = _construction_from_params(spec.params)
    if construction.sparsify == :radius
        radius = get(spec.params, :radius, nothing)
        if radius === nothing
            _ingestion_warn!(warnings, "construction.sparsify=:radius without radius; using complete-graph edge upper bound in estimate.", strict)
            edges = binomial(big(n), big(2))
        elseif n > exact_pairwise_limit
            _ingestion_warn!(warnings, "Point count $n exceeds exact_pairwise_limit=$exact_pairwise_limit; using complete-graph edge upper bound.", strict)
            edges = binomial(big(n), big(2))
        else
            points_ref = get(spec.params, :_points_ref, nothing)
            points_ref === nothing && throw(ArgumentError("estimate_ingestion internal error: missing point reference for sparse radius estimate."))
            edges_i = 0
            r = Float64(radius)
            for i in 1:n, j in (i + 1):n
                if _euclidean_distance(points_ref[i], points_ref[j]) <= r
                    edges_i += 1
                end
            end
            edges = big(edges_i)
        end
        return BigInt[big(n), edges]
    end
    if construction.sparsify == :knn
        k = max(Int(get(spec.params, :knn, 8)), 0)
        max_pairs = binomial(big(n), big(2))
        edges = min(max_pairs, big(cld(n * k, 2)))
        return BigInt[big(n), edges]
    end
    if construction.sparsify == :greedy_perm
        n_landmarks = get(spec.params, :n_landmarks, nothing)
        m = n_landmarks === nothing ? _default_landmark_count(n) : Int(n_landmarks)
        m = max(1, min(n, m))
        _ingestion_warn!(warnings, "construction.sparsify=:greedy_perm estimate uses landmark count m=$m.", strict)
        return BigInt[binomial(big(m), big(k)) for k in 1:(max_dim + 1)]
    end
    if construction.sparsify != :none
        throw(ArgumentError("estimate_ingestion: unsupported construction.sparsify=$(construction.sparsify)."))
    end
    return BigInt[binomial(big(n), big(k)) for k in 1:(max_dim + 1)]
end

@inline function _construction_budget(spec::FiltrationSpec)
    return _construction_from_params(spec.params).budget
end

@inline function _construction_max_simplices(spec::FiltrationSpec)
    return _construction_budget(spec).max_simplices
end

@inline function _construction_max_edges(spec::FiltrationSpec)
    return _construction_budget(spec).max_edges
end

function _construction_merge_for_fallback(spec::FiltrationSpec,
                                          construction::ConstructionOptions)
    p = _filter_params(spec.params, Symbol[:construction])
    return merge(p, (construction=_construction_to_namedtuple(construction),))
end

function _construction_budget_edge_cap(spec::FiltrationSpec)
    budget = _construction_budget(spec)
    return budget.max_edges
end

function _construction_memory_budget(spec::FiltrationSpec)
    return _construction_budget(spec).memory_budget_bytes
end

function _construction_check_memory_budget!(dense_bytes_est::BigInt,
                                            spec::FiltrationSpec)
    mb = _construction_memory_budget(spec)
    if mb !== nothing && dense_bytes_est > big(mb)
        throw(ArgumentError("Ingestion construction budget exceeded: estimated dense boundary footprint $dense_bytes_est bytes > memory_budget_bytes=$mb."))
    end
    return nothing
end

function _construction_check_max_simplices!(total::Integer, dim_idx::Int, spec::FiltrationSpec)
    ms = _construction_max_simplices(spec)
    if ms !== nothing && total > ms
        throw(ArgumentError("Ingestion construction budget exceeded at simplex dimension $(dim_idx): count=$(total) > max_simplices=$(ms)."))
    end
    return nothing
end

function _construction_check_max_edges!(edge_count::Integer, spec::FiltrationSpec)
    cap = _construction_max_edges(spec)
    if cap !== nothing && edge_count > cap
        throw(ArgumentError("Ingestion construction budget exceeded: edge count=$(edge_count) > max_edges=$(cap)."))
    end
    return nothing
end

@inline function _construction_cap_edges!(edges::Vector, spec::FiltrationSpec)
    cap = _construction_budget_edge_cap(spec)
    if cap !== nothing && length(edges) > cap
        throw(ArgumentError("Ingestion construction budget exceeded: edge count=$(length(edges)) > max_edges=$(cap)."))
    end
    return edges
end

@inline _combination_count(n::Int, k::Int) = binomial(big(n), big(k))

function _construction_precheck_combination_enumeration!(n::Int,
                                                         k::Int,
                                                         total_before::BigInt,
                                                         spec::FiltrationSpec)
    count = _combination_count(n, k)
    if count > big(typemax(Int))
        throw(ArgumentError("Ingestion combinatorial explosion at simplex dimension $(k - 1): C($(n),$(k))=$(count) exceeds representable collection size before enumeration."))
    end
    ms = _construction_max_simplices(spec)
    if ms !== nothing
        total_after = total_before + count
        if total_after > big(ms)
            throw(ArgumentError("Ingestion construction budget would be exceeded before enumeration at simplex dimension $(k - 1): count=$(total_after) > max_simplices=$(ms)."))
        end
    end
    return Int(count)
end

function _construction_precheck_combination_candidates!(n::Int,
                                                        k::Int,
                                                        total_before::BigInt,
                                                        spec::FiltrationSpec;
                                                        context::AbstractString="candidate")
    count = _combination_count(n, k)
    if count > big(typemax(Int))
        throw(ArgumentError("Ingestion combinatorial explosion at simplex dimension $(k - 1): C($(n),$(k))=$(count) exceeds representable collection size before $(context) enumeration."))
    end
    ms = _construction_max_simplices(spec)
    if ms !== nothing
        total_upper = total_before + count
        if total_upper > big(ms)
            throw(ArgumentError("Ingestion construction budget upper bound would be exceeded before $(context) enumeration at simplex dimension $(k - 1): upper_bound=$(total_upper) > max_simplices=$(ms)."))
        end
    end
    return Int(count)
end

function _greedy_perm_indices(points::AbstractVector{<:AbstractVector{<:Real}}, m::Int)
    n = length(points)
    n == 0 && return Int[]
    m = max(1, min(n, m))
    chosen = Int[1]
    min_d = fill(Inf, n)
    for i in 1:n
        min_d[i] = _euclidean_distance(points[i], points[1])
    end
    min_d[1] = 0.0
    while length(chosen) < m
        best = 1
        bestd = -Inf
        for i in 1:n
            di = min_d[i]
            if di > bestd
                bestd = di
                best = i
            end
        end
        push!(chosen, best)
        for i in 1:n
            d = _euclidean_distance(points[i], points[best])
            d < min_d[i] && (min_d[i] = d)
        end
        min_d[best] = 0.0
    end
    return sort(unique(chosen))
end

function _collapse_acyclic_edges_with_dists(edges::Vector{NTuple{2,Int}},
                                            dists::Vector{Float64},
                                            n::Int)
    parent = collect(1:n)
    rank = zeros(Int, n)
    function findp(x)
        while parent[x] != x
            parent[x] = parent[parent[x]]
            x = parent[x]
        end
        return x
    end
    function unite(x, y)
        rx, ry = findp(x), findp(y)
        rx == ry && return false
        if rank[rx] < rank[ry]
            parent[rx] = ry
        elseif rank[rx] > rank[ry]
            parent[ry] = rx
        else
            parent[ry] = rx
            rank[rx] += 1
        end
        return true
    end
    idx = sortperm(eachindex(edges); by=i -> dists[i])
    out_edges = NTuple{2,Int}[]
    out_dists = Float64[]
    sizehint!(out_edges, length(edges))
    sizehint!(out_dists, length(edges))
    for i in idx
        u, v = edges[i]
        if unite(u, v)
            e = edges[i]
            push!(out_edges, e)
            push!(out_dists, dists[i])
        end
    end
    return out_edges, out_dists
end

function _collapse_dominated_edges_points(edges::Vector{NTuple{2,Int}},
                                          dists::Vector{Float64},
                                          points::AbstractVector{<:AbstractVector{<:Real}};
                                          tol::Float64=1e-12)
    n = length(points)
    out_edges = NTuple{2,Int}[]
    out_dists = Float64[]
    sizehint!(out_edges, length(edges))
    sizehint!(out_dists, length(edges))
    for idx in eachindex(edges)
        u, v = edges[idx]
        duv = dists[idx]
        dominated = false
        for w in 1:n
            (w == u || w == v) && continue
            if max(_euclidean_distance(points[u], points[w]),
                   _euclidean_distance(points[w], points[v])) <= duv + tol
                dominated = true
                break
            end
        end
        if !dominated
            push!(out_edges, (u, v))
            push!(out_dists, duv)
        end
    end
    return out_edges, out_dists
end

function _collapse_dominated_edges_points(edges::Vector{NTuple{2,Int}},
                                          dists::Vector{Float64},
                                          points::AbstractMatrix{<:Real};
                                          tol::Float64=1e-12)
    n = size(points, 1)
    out_edges = NTuple{2,Int}[]
    out_dists = Float64[]
    sizehint!(out_edges, length(edges))
    sizehint!(out_dists, length(edges))
    @inbounds for idx in eachindex(edges)
        u, v = edges[idx]
        duv = dists[idx]
        dominated = false
        for w in 1:n
            (w == u || w == v) && continue
            if max(_euclidean_distance(points, u, w),
                   _euclidean_distance(points, w, v)) <= duv + tol
                dominated = true
                break
            end
        end
        dominated && continue
        push!(out_edges, (u, v))
        push!(out_dists, duv)
    end
    return out_edges, out_dists
end

function _apply_construction_collapse_edge_driven(edges::Vector{NTuple{2,Int}},
                                                  dists::Vector{Float64},
                                                  points::AbstractVector{<:AbstractVector{<:Real}},
                                                  construction::ConstructionOptions)
    if construction.collapse == :none
        return edges, dists
    elseif construction.collapse == :dominated_edges
        return _collapse_dominated_edges_points(edges, dists, points)
    elseif construction.collapse == :acyclic
        return _collapse_acyclic_edges_with_dists(edges, dists, length(points))
    end
    throw(ArgumentError("Unsupported construction.collapse=$(construction.collapse)."))
end

function _apply_construction_collapse_edge_driven(edges::Vector{NTuple{2,Int}},
                                                  dists::Vector{Float64},
                                                  points::AbstractMatrix{<:Real},
                                                  construction::ConstructionOptions)
    if construction.collapse == :none
        return edges, dists
    elseif construction.collapse == :dominated_edges
        return _collapse_dominated_edges_points(edges, dists, points)
    elseif construction.collapse == :acyclic
        return _collapse_acyclic_edges_with_dists(edges, dists, size(points, 1))
    end
    throw(ArgumentError("Unsupported construction.collapse=$(construction.collapse)."))
end

function _point_cloud_sparsify_edge_driven(points::AbstractVector{<:AbstractVector{<:Real}},
                                           spec::FiltrationSpec,
                                           construction::ConstructionOptions)
    n = length(points)
    n == 0 && return NTuple{2,Int}[], Float64[], Float64[]
    backend = _pointcloud_nn_backend(spec)
    approx_candidates = _pointcloud_nn_approx_candidates(spec)
    if construction.sparsify == :radius
        radius = get(spec.params, :radius, nothing)
        radius === nothing && error("construction.sparsify=:radius requires radius.")
        edges, dists = _point_cloud_radius_graph(points, Float64(radius); backend=backend, approx_candidates=approx_candidates)
        _construction_cap_edges!(edges, spec)
        return edges, dists, Float64[]
    elseif construction.sparsify == :knn
        k = Int(get(spec.params, :knn, 8))
        k > 0 || error("construction.sparsify=:knn requires knn > 0.")
        edges, dists, kdist = _point_cloud_knn_graph(points, k; backend=backend, approx_candidates=approx_candidates)
        _construction_cap_edges!(edges, spec)
        return edges, dists, kdist
    end
    return NTuple{2,Int}[], Float64[], Float64[]
end

function _point_cloud_sparsify_edge_driven(points::AbstractMatrix{<:Real},
                                           spec::FiltrationSpec,
                                           construction::ConstructionOptions)
    n = size(points, 1)
    n == 0 && return NTuple{2,Int}[], Float64[], Float64[]
    backend = _pointcloud_nn_backend(spec)
    approx_candidates = _pointcloud_nn_approx_candidates(spec)
    if construction.sparsify == :radius
        radius = get(spec.params, :radius, nothing)
        radius === nothing && error("construction.sparsify=:radius requires radius.")
        edges, dists = _point_cloud_radius_graph(points, Float64(radius); backend=backend, approx_candidates=approx_candidates)
        _construction_cap_edges!(edges, spec)
        return edges, dists, Float64[]
    elseif construction.sparsify == :knn
        k = Int(get(spec.params, :knn, 8))
        k > 0 || error("construction.sparsify=:knn requires knn > 0.")
        edges, dists, kdist = _point_cloud_knn_graph(points, k; backend=backend, approx_candidates=approx_candidates)
        _construction_cap_edges!(edges, spec)
        return edges, dists, kdist
    end
    return NTuple{2,Int}[], Float64[], Float64[]
end

function _maybe_greedy_perm_reduce(data::PointCloud, spec::FiltrationSpec)
    construction = _construction_from_params(spec.params)
    if construction.sparsify != :greedy_perm
        return data, spec, nothing
    end
    points = data.points
    n = length(points)
    n_landmarks = get(spec.params, :n_landmarks, nothing)
    m = n_landmarks === nothing ? _default_landmark_count(n) : Int(n_landmarks)
    idx = _greedy_perm_indices(points, m)
    p2 = [points[i] for i in idx]
    p = _construction_merge_for_fallback(spec, ConstructionOptions(;
        sparsify=:none,
        collapse=construction.collapse,
        output_stage=construction.output_stage,
        budget=construction.budget,
    ))
    if haskey(spec.params, :vertex_values)
        vals = spec.params[:vertex_values]
        length(vals) == n || error("vertex_values length mismatch: expected $(n), got $(length(vals)).")
        p = merge(p, (vertex_values=[vals[i] for i in idx],))
    end
    p = _filter_params(p, Symbol[:n_landmarks])
    return PointCloud(p2), FiltrationSpec(; kind=spec.kind, p...), idx
end

function _estimate_pointcloud_cell_counts(data::PointCloud,
                                          spec::FiltrationSpec;
                                          exact_pairwise_limit::Int,
                                          warnings::Vector{String},
                                          strict::Bool)
    n = length(data.points)
    kind = spec.kind
    if kind == :landmark_rips
        landmarks = get(spec.params, :landmarks, nothing)
        landmarks === nothing && throw(ArgumentError("estimate_ingestion: landmark_rips requires landmarks."))
        m = length(landmarks)
        m == 0 && _ingestion_warn!(warnings, "landmark_rips with empty landmarks.", strict)
        ps = merge(spec.params, (_points_ref = [data.points[i] for i in landmarks],))
        return _estimate_rips_like_cell_counts(m, FiltrationSpec(; kind=:rips, ps...);
                                               exact_pairwise_limit=exact_pairwise_limit,
                                               warnings=warnings,
                                               strict=strict)
    elseif kind == :delaunay_lower_star || kind == :function_delaunay ||
           kind == :alpha || kind == :core_delaunay
        d = length(data.points[1])
        max_dim = min(max(Int(get(spec.params, :max_dim, 2)), 0), 2)
        if d <= 2
            counts = BigInt[big(n)]
            if max_dim >= 1
                edges = if d == 1
                    big(max(n - 1, 0))
                elseif n <= 1
                    big(0)
                else
                    min(big(3 * n - 6), binomial(big(n), big(2)))
                end
                push!(counts, edges)
            end
            if max_dim >= 2
                tri = if n < 3
                    big(0)
                else
                    min(big(2 * n - 5), binomial(big(n), big(3)))
                end
                push!(counts, tri)
            end
            return counts
        end
        policy = Symbol(get(spec.params, :highdim_policy, :rips))
        if policy == :error
            _ingestion_warn!(warnings, "Delaunay high-dimensional input (dimension=$d) with highdim_policy=:error will fail at ingestion.", strict)
            return BigInt[big(n)]
        elseif policy != :rips
            throw(ArgumentError("estimate_ingestion: unsupported highdim_policy=$(policy) for Delaunay filtrations."))
        end
        _ingestion_warn!(warnings, "Delaunay high-dimensional estimate uses Rips fallback policy (:rips).", strict)
        ps = merge(spec.params, (_points_ref = data.points,))
        fallback_kind = kind == :alpha ? :rips : (kind == :core_delaunay ? :degree_rips : :function_rips)
        return _estimate_rips_like_cell_counts(n, FiltrationSpec(; kind=fallback_kind, ps...);
                                               exact_pairwise_limit=exact_pairwise_limit,
                                               warnings=warnings,
                                               strict=strict)
    elseif kind == :core
        if haskey(spec.params, :radius) && spec.params[:radius] !== nothing
            if n > exact_pairwise_limit
                _ingestion_warn!(warnings, "core(radius=...) estimate used complete-graph edge upper bound (n=$n > exact_pairwise_limit=$exact_pairwise_limit).", strict)
                edges = binomial(big(n), big(2))
            else
                r = Float64(spec.params[:radius])
                e = 0
                for i in 1:n, j in (i + 1):n
                    if _euclidean_distance(data.points[i], data.points[j]) <= r
                        e += 1
                    end
                end
                edges = big(e)
            end
        else
            k = Int(get(spec.params, :knn, 8))
            edges = min(binomial(big(n), big(2)), big(cld(n * max(k, 0), 2)))
        end
        return BigInt[big(n), edges]
    elseif kind == :rhomboid || kind == :rips || kind == :rips_density ||
           kind == :rips_codensity || kind == :rips_lowerstar ||
           kind == :function_rips || kind == :degree_rips
        ps = merge(spec.params, (_points_ref = data.points,))
        return _estimate_rips_like_cell_counts(n, FiltrationSpec(; kind=:rips, ps...);
                                               exact_pairwise_limit=exact_pairwise_limit,
                                               warnings=warnings,
                                               strict=strict)
    end
    throw(ArgumentError("estimate_ingestion: unsupported point-cloud filtration kind=$(kind)."))
end

function _estimate_graph_cell_counts(data::GraphData,
                                     spec::FiltrationSpec;
                                     warnings::Vector{String},
                                     strict::Bool)
    n = data.n
    m = length(data.edges)
    kind = spec.kind
    if kind == :graph_lower_star || kind == :edge_weighted || kind == :core ||
       kind == :graph_centrality || kind == :graph_geodesic || kind == :graph_function_geodesic_bifiltration
        lift = Symbol(get(spec.params, :lift, :lower_star))
        if lift == :clique
            max_dim = max(Int(get(spec.params, :max_dim, 2)), 1)
            _ingestion_warn!(warnings, "Graph clique-based estimate is an upper bound (complete-graph assumption).", strict)
            return BigInt[binomial(big(n), big(k)) for k in 1:(max_dim + 1)]
        elseif lift != :lower_star
            throw(ArgumentError("estimate_ingestion: unsupported graph lift=$(lift)."))
        end
        return BigInt[big(n), big(m)]
    elseif kind == :clique_lower_star || kind == :rhomboid ||
           (kind == :graph_weight_threshold && Symbol(get(spec.params, :lift, :graph)) == :clique)
        max_dim = max(Int(get(spec.params, :max_dim, 2)), 1)
        _ingestion_warn!(warnings, "Graph clique-based estimate is an upper bound (complete-graph assumption).", strict)
        return BigInt[binomial(big(n), big(k)) for k in 1:(max_dim + 1)]
    elseif kind == :graph_weight_threshold
        lift = Symbol(get(spec.params, :lift, :graph))
        lift == :graph || throw(ArgumentError("estimate_ingestion: unsupported graph_weight_threshold lift=$(lift)."))
        return BigInt[big(n), big(m)]
    end
    throw(ArgumentError("estimate_ingestion: unsupported graph filtration kind=$(kind)."))
end

function _estimate_cubical_cell_counts(dims::NTuple{N,Int}) where {N}
    counts = fill(big(0), N + 1)
    for mask in 0:(Int(2^N) - 1)
        k = count_ones(mask)
        c = big(1)
        for i in 1:N
            c *= big(dims[i] - Int((mask >> (i - 1)) & 1))
        end
        counts[k + 1] += c
    end
    return counts
end

function _estimate_image_cell_counts(data::ImageNd,
                                     spec::FiltrationSpec;
                                     warnings::Vector{String},
                                     strict::Bool)
    kind = spec.kind
    if kind == :lower_star || kind == :cubical
        return _estimate_cubical_cell_counts(size(data.data))
    elseif kind == :image_distance_bifiltration
        channels = get(spec.params, :channels, nothing)
        if channels === nothing
            return _estimate_cubical_cell_counts(size(data.data))
        end
        dims = size(channels[1])
        for ch in channels
            size(ch) == dims || throw(ArgumentError("estimate_ingestion: all channels must have the same size."))
        end
        return _estimate_cubical_cell_counts(dims)
    end
    throw(ArgumentError("estimate_ingestion: unsupported image filtration kind=$(kind)."))
end

function _estimate_embedded_planar_cell_counts(data::EmbeddedPlanarGraph2D,
                                               spec::FiltrationSpec;
                                               warnings::Vector{String},
                                               strict::Bool)
    kind = spec.kind
    if kind == :wing_vein_bifiltration
        grid = get(spec.params, :grid, (32, 32))
        return _estimate_cubical_cell_counts(Tuple(grid))
    elseif kind == :image_distance_bifiltration
        grid = get(spec.params, :grid, nothing)
        if grid !== nothing
            return _estimate_cubical_cell_counts(Tuple(grid))
        end
        _ingestion_warn!(warnings, "EmbeddedPlanarGraph2D image_distance_bifiltration estimate used graph surrogate counts.", strict)
        return BigInt[big(length(data.vertices)), big(length(data.edges))]
    elseif kind == :graph_lower_star || kind == :clique_lower_star || kind == :edge_weighted ||
           kind == :core || kind == :rhomboid || kind == :graph_centrality ||
           kind == :graph_geodesic || kind == :graph_function_geodesic_bifiltration ||
           kind == :graph_weight_threshold
        gd = GraphData(length(data.vertices), data.edges)
        return _estimate_graph_cell_counts(gd, spec; warnings=warnings, strict=strict)
    end
    throw(ArgumentError("estimate_ingestion: unsupported embedded planar filtration kind=$(kind)."))
end

function _estimate_cell_counts(data, spec::FiltrationSpec;
                               exact_pairwise_limit::Int,
                               warnings::Vector{String},
                               strict::Bool)
    if data isa GradedComplex
        _, dim_offsets = _packed_cell_storage(data)
        nd = length(dim_offsets) - 1
        counts = Vector{BigInt}(undef, nd)
        @inbounds for slot in 1:nd
            counts[slot] = big(_packed_dim_count(dim_offsets, slot))
        end
        return counts
    elseif data isa MultiCriticalGradedComplex
        _, dim_offsets = _packed_cell_storage(data)
        nd = length(dim_offsets) - 1
        counts = Vector{BigInt}(undef, nd)
        @inbounds for slot in 1:nd
            counts[slot] = big(_packed_dim_count(dim_offsets, slot))
        end
        return counts
    elseif data isa SimplexTreeMulti
        ns = simplex_count(data)
        ns == 0 && return BigInt[]
        maxd = max_simplex_dim(data)
        counts = fill(big(0), maxd + 1)
        @inbounds for d in data.simplex_dims
            counts[d + 1] += 1
        end
        return counts
    elseif data isa PointCloud
        return _estimate_pointcloud_cell_counts(data, spec;
                                               exact_pairwise_limit=exact_pairwise_limit,
                                               warnings=warnings,
                                               strict=strict)
    elseif data isa GraphData
        return _estimate_graph_cell_counts(data, spec; warnings=warnings, strict=strict)
    elseif data isa ImageNd
        return _estimate_image_cell_counts(data, spec; warnings=warnings, strict=strict)
    elseif data isa EmbeddedPlanarGraph2D
        return _estimate_embedded_planar_cell_counts(data, spec; warnings=warnings, strict=strict)
    end
    throw(ArgumentError("estimate_ingestion: unsupported dataset type $(typeof(data))."))
end

function _estimate_axis_sizes(data, spec::FiltrationSpec; warnings::Vector{String}, strict::Bool)
    axis_sizes = _estimate_axis_sizes_from_spec(spec)
    axis_sizes === nothing || return axis_sizes
    if data isa GradedComplex
        return _estimate_axis_sizes_from_grades(data.grades)
    elseif data isa MultiCriticalGradedComplex
        _, grade_data = _packed_multigrade_storage(data)
        return _estimate_axis_sizes_from_multigrade_storage(grade_data)
    elseif data isa SimplexTreeMulti
        n = simplex_count(data)
        n == 0 && return nothing
        N = length(data.grade_data[1])
        seen = [Set{Float64}() for _ in 1:N]
        @inbounds for i in 1:n
            gi = simplex_grades(data, i)
            for g in gi
                for k in 1:N
                    push!(seen[k], Float64(g[k]))
                end
            end
        end
        return ntuple(i -> length(seen[i]), N)
    end
    _ingestion_warn!(warnings, "axis_sizes unavailable without explicit axes for dataset type $(typeof(data)).", strict)
    return nothing
end

"""
    estimate_ingestion(data, spec; kwargs...) -> IngestionEstimate
    estimate_ingestion(data, filtration; kwargs...) -> IngestionEstimate

Return a typed preflight estimate for ingestion complexity and memory risk
before running `plan_ingestion(...)` or `encode(data, filtration_or_spec; ...)`.

Cheap-first workflow
- call `estimate_ingestion(...)` when you want a lightweight estimate of cell
  growth, axis sizes, and dense linear-algebra pressure before paying for full
  ingestion;
- inspect the returned [`IngestionEstimate`](@ref) with `describe(...)` or
  [`ingestion_estimate_summary`](@ref) and the scalar accessors in this module;
- use `strict=true` only when you want warnings such as budget overruns or
  unsupported high-dimensional Delaunay policies to throw immediately.

This function does not build the graded complex or encoding. It only estimates
the likely combinatorial footprint of the requested ingestion route.
"""
function _estimate_ingestion_report(data,
                                    spec::FiltrationSpec;
                                    poset_threshold::Integer=200_000,
                                    dense_memory_budget_bytes::Integer=2_000_000_000,
                                    dense_elem_bytes::Integer=8,
                                    exact_pairwise_limit::Int=5_000,
                                    strict::Bool=false)
    warnings = String[]
    cell_counts = _estimate_cell_counts(data, spec;
                                       exact_pairwise_limit=exact_pairwise_limit,
                                       warnings=warnings,
                                       strict=strict)
    n_cells_est = _sum_bigint(cell_counts)
    nnz_est = _estimate_nnz_from_cell_counts(cell_counts)
    dense_bytes_est = _estimate_dense_bytes_from_cell_counts(cell_counts; elem_bytes=dense_elem_bytes)

    axis_sizes = _estimate_axis_sizes(data, spec; warnings=warnings, strict=strict)
    poset_size = axis_sizes === nothing ? nothing : _prod_bigint(axis_sizes)

    budget = _construction_budget(spec)
    max_simplices = budget.max_simplices
    if max_simplices !== nothing
        ms = big(max_simplices)
        for (d, c) in enumerate(cell_counts)
            if c > ms
                _ingestion_warn!(warnings, "Estimated simplex count at dimension $(d - 1) is $c > max_simplices=$max_simplices.", strict)
            end
        end
        if n_cells_est > ms
            _ingestion_warn!(warnings, "Estimated total cells $n_cells_est > max_simplices=$max_simplices.", strict)
        end
    end
    max_edges = budget.max_edges
    if max_edges !== nothing && length(cell_counts) >= 2
        edge_est = cell_counts[2]
        if edge_est > big(max_edges)
            _ingestion_warn!(warnings, "Estimated edge count $edge_est > max_edges=$max_edges.", strict)
        end
    end
    if poset_size !== nothing && poset_size > big(poset_threshold)
        _ingestion_warn!(warnings, "Estimated encoding poset size |P|=$poset_size exceeds threshold=$poset_threshold.", strict)
    end
    mem_budget = budget.memory_budget_bytes === nothing ? dense_memory_budget_bytes : budget.memory_budget_bytes
    if mem_budget !== nothing && dense_bytes_est > big(mem_budget)
        _ingestion_warn!(warnings, "Estimated dense boundary footprint $dense_bytes_est bytes exceeds budget=$mem_budget bytes.", strict)
    end

    return (
        n_cells_est = n_cells_est,
        cell_counts_by_dim = cell_counts,
        axis_sizes = axis_sizes,
        poset_size = poset_size,
        nnz_est = nnz_est,
        dense_bytes_est = dense_bytes_est,
        warnings = warnings,
    )
end

function estimate_ingestion(data,
                            spec::FiltrationSpec;
                            poset_threshold::Integer=200_000,
                            dense_memory_budget_bytes::Integer=2_000_000_000,
                            dense_elem_bytes::Integer=8,
                            exact_pairwise_limit::Int=5_000,
                            strict::Bool=false)
    return IngestionEstimate(_estimate_ingestion_report(
        data,
        spec;
        poset_threshold=poset_threshold,
        dense_memory_budget_bytes=dense_memory_budget_bytes,
        dense_elem_bytes=dense_elem_bytes,
        exact_pairwise_limit=exact_pairwise_limit,
        strict=strict,
    ))
end

estimate_ingestion(data, filtration::AbstractFiltration; kwargs...) =
    estimate_ingestion(data, _filtration_spec(filtration); kwargs...)

@inline function _ingestion_estimate_describe(est::IngestionEstimate)
    return (
        kind = :ingestion_estimate,
        estimated_cells = estimated_cells(est),
        cell_counts_by_dim = cell_counts_by_dim(est),
        estimated_axis_sizes = estimated_axis_sizes(est),
        estimated_poset_size = estimated_poset_size(est),
        estimated_nnz = estimated_nnz(est),
        estimated_dense_bytes = estimated_dense_bytes(est),
        nwarnings = length(estimate_warnings(est)),
    )
end

"""
    describe(est::IngestionEstimate) -> NamedTuple

Return a compact semantic summary of an ingestion preflight estimate.

Use this when you want the cheap scalar picture of a proposed ingestion route
before planning or executing it. The heavier ingestion stages are still
uncomputed at this point.
"""
describe(est::IngestionEstimate) = _ingestion_estimate_describe(est)

"""
    ingestion_estimate_summary(est::IngestionEstimate) -> NamedTuple

Owner-local summary helper for ingestion preflight estimates.

Use this before `plan_ingestion(...)` or `run_ingestion(...)` when you only
need the scalar combinatorial/memory picture of a proposed route.
"""
ingestion_estimate_summary(est::IngestionEstimate) = _ingestion_estimate_describe(est)

function Base.show(io::IO, est::IngestionEstimate)
    d = _ingestion_estimate_describe(est)
    print(io, "IngestionEstimate(cells=", d.estimated_cells,
          ", poset_size=", d.estimated_poset_size,
          ", warnings=", d.nwarnings, ")")
end

function Base.show(io::IO, ::MIME"text/plain", est::IngestionEstimate)
    d = _ingestion_estimate_describe(est)
    print(io, "IngestionEstimate",
          "\n  estimated_cells: ", d.estimated_cells,
          "\n  cell_counts_by_dim: ", repr(d.cell_counts_by_dim),
          "\n  estimated_axis_sizes: ", repr(d.estimated_axis_sizes),
          "\n  estimated_poset_size: ", repr(d.estimated_poset_size),
          "\n  estimated_nnz: ", d.estimated_nnz,
          "\n  estimated_dense_bytes: ", d.estimated_dense_bytes,
          "\n  warnings: ", repr(estimate_warnings(est)))
end

@inline function _axes_from_complex_grades(G::GradedComplex, orientation)
    N = length(G.grades[1])
    return _axes_from_grades(G.grades, N; orientation=orientation)
end

@inline function _axes_from_complex_grades(G::MultiCriticalGradedComplex, orientation)
    _, grade_data = _packed_multigrade_storage(G)
    isempty(grade_data) && error("MultiCriticalGradedComplex has no grades.")
    N = length(grade_data[1])
    return _axes_from_grades(grade_data, N; orientation=orientation)
end

@inline function _tuple_dom_leq(a::NTuple{N,T}, b::NTuple{N,T}) where {N,T}
    @inbounds for i in 1:N
        a[i] <= b[i] || return false
    end
    return true
end

function _minimal_tuple_set(vals::Vector{NTuple{N,T}}) where {N,T}
    isempty(vals) && return vals
    keep = trues(length(vals))
    @inbounds for i in eachindex(vals)
        keep[i] || continue
        vi = vals[i]
        for j in eachindex(vals)
            i == j && continue
            _tuple_dom_leq(vals[j], vi) || continue
            if vals[j] != vi
                keep[i] = false
                break
            end
        end
    end
    return vals[keep]
end

function _maximal_tuple_set(vals::Vector{NTuple{N,T}}) where {N,T}
    isempty(vals) && return vals
    keep = trues(length(vals))
    @inbounds for i in eachindex(vals)
        keep[i] || continue
        vi = vals[i]
        for j in eachindex(vals)
            i == j && continue
            _tuple_dom_leq(vi, vals[j]) || continue
            if vals[j] != vi
                keep[i] = false
                break
            end
        end
    end
    return vals[keep]
end

"""
    criticality(G)

Return the per-cell criticality upper bound for a graded complex.
- `GradedComplex`: always `1`
- `MultiCriticalGradedComplex`: max number of grades attached to any cell
"""
criticality(::GradedComplex) = 1
function criticality(G::MultiCriticalGradedComplex)
    grade_offsets, _ = _packed_multigrade_storage(G)
    maxcrit = 0
    @inbounds for i in 1:(length(grade_offsets) - 1)
        maxcrit = max(maxcrit, grade_offsets[i + 1] - grade_offsets[i])
    end
    return maxcrit
end

"""
    normalize_multicritical(G::MultiCriticalGradedComplex; keep=:minimal)

Normalize per-cell grade sets by removing redundancy:
- `keep=:minimal` keeps antichain-minimal grades (default; union semantics)
- `keep=:maximal` keeps antichain-maximal grades (intersection semantics)
- `keep=:unique` keeps only duplicates removed
"""
function normalize_multicritical(G::MultiCriticalGradedComplex; keep::Symbol=:minimal)
    keep in (:minimal, :maximal, :unique) ||
        throw(ArgumentError("normalize_multicritical: keep must be :minimal, :maximal, or :unique."))
    keep === :unique && return G
    grade_offsets, grade_data = _packed_multigrade_storage(G)
    isempty(grade_data) && error("normalize_multicritical: complex has no grades.")
    N = length(grade_data[1])
    T = eltype(grade_data[1])
    out_offsets = Vector{Int}(undef, length(grade_offsets))
    out_offsets[1] = 1
    out_data = NTuple{N,T}[]
    sizehint!(out_data, length(grade_data))
    @inbounds for i in 1:(length(grade_offsets) - 1)
        lo = grade_offsets[i]
        hi = grade_offsets[i + 1] - 1
        gi = Vector{NTuple{N,T}}(undef, hi - lo + 1)
        copyto!(gi, 1, grade_data, lo, length(gi))
        if keep === :minimal
            gi = _minimal_tuple_set(gi)
        else
            gi = _maximal_tuple_set(gi)
        end
        append!(out_data, gi)
        out_offsets[i + 1] = length(out_data) + 1
    end
    return _rebuild_multicritical_complex(G, out_offsets, out_data)
end

normalize_multicritical(G::GradedComplex; kwargs...) = G

# Section 3: GridEncodingMap helpers
# -----------------------------------------------------------------------------

"""
    poset_from_axes(axes; orientation=ntuple(_->1, N), kind=:grid) -> AbstractPoset

Build the product-of-chains poset on a grid defined by `axes`.
`axes` is an N-tuple of sorted coordinate vectors. `orientation[i]` is +1 for
sublevel-style order (increasing) and -1 for superlevel-style order (decreasing).

Notes
-----
- If all `orientation[i] == 1` and `kind=:grid`, a structured `ProductOfChainsPoset`
  is returned to avoid materializing the transitive closure.
- Otherwise, we fall back to a dense `FinitePoset`.
"""
function poset_from_axes(axes::NTuple{N,Vector{T}};
                         orientation::NTuple{N,Int}=ntuple(_ -> 1, N),
                         kind::Symbol = :grid) where {N,T}
    sizes = ntuple(i -> length(axes[i]), N)
    total = 1
    for i in 1:N
        total *= sizes[i]
    end

    if kind == :grid
        if all(o -> o == 1, orientation)
            return ProductOfChainsPoset(sizes)
        end
        kind = :dense
    end
    if kind == :dense
        # Enumerate all multi-indices in mixed radix order.
        idxs = Vector{NTuple{N,Int}}(undef, total)
        cur = ones(Int, N)
        for lin in 1:total
            idxs[lin] = ntuple(i -> cur[i], N)
            for i in 1:N
                cur[i] += 1
                if cur[i] <= sizes[i]
                    break
                else
                    cur[i] = 1
                end
            end
        end

        leq = falses(total, total)
        for i in 1:total
            ai = idxs[i]
            for j in 1:total
                bj = idxs[j]
                ok = true
                for k in 1:N
                    if orientation[k] == 1
                        if ai[k] > bj[k]
                            ok = false
                            break
                        end
                    else
                        if ai[k] < bj[k]
                            ok = false
                            break
                        end
                    end
                end
                leq[i, j] = ok
            end
        end

        return FinitePoset(leq; check=false)
    else
        error("poset_from_axes: kind must be :grid or :dense")
    end
end

function GridEncodingMap(P::AbstractPoset, coords::NTuple{N,Vector{T}};
                         orientation::NTuple{N,Int}=ntuple(_ -> 1, N)) where {N,T}
    sizes = ntuple(i -> length(coords[i]), N)
    total = 1
    for i in 1:N
        total *= sizes[i]
    end
    total == nvertices(P) || error("GridEncodingMap: grid size $(total) does not match nvertices(P)=$(nvertices(P)).")
    for i in 1:N
        o = orientation[i]
        (o == 1 || o == -1) || error("GridEncodingMap: orientation[$i] must be +1 or -1.")
    end
    return GridEncodingMap{N,T,typeof(P)}(P, coords, orientation, sizes, _grid_strides(sizes))
end
# Section 4: Internal helpers (data ingestion pipeline)
# -----------------------------------------------------------------------------

@inline function _tuple_leq(a::NTuple{N,Int}, b::NTuple{N,Int},
                            orientation::NTuple{N,Int}) where {N}
    @inbounds for i in 1:N
        if orientation[i] == 1
            if a[i] > b[i]
                return false
            end
        else
            if a[i] < b[i]
                return false
            end
        end
    end
    return true
end

function _grid_tuples(sizes::NTuple{N,Int}) where {N}
    total = 1
    for i in 1:N
        total *= sizes[i]
    end
    out = Vector{NTuple{N,Int}}(undef, total)
    cur = ones(Int, N)
    for lin in 1:total
        out[lin] = ntuple(i -> cur[i], N)
        for i in 1:N
            cur[i] += 1
            if cur[i] <= sizes[i]
                break
            else
                cur[i] = 1
            end
        end
    end
    return out
end

function _axes_from_grades(grades::Vector{<:NTuple{N,T}}, n::Int;
                           orientation::Union{Nothing,NTuple{N,Int}}=nothing) where {N,T}
    n == N || error("grade length $N does not match n=$n")
    orient = orientation === nothing ? ntuple(_ -> 1, N) : orientation
    for i in 1:N
        (orient[i] == 1 || orient[i] == -1) || error("orientation[$i] must be +1 or -1.")
    end

    RT = promote_type(T, Int)
    axes = [RT[] for _ in 1:n]
    for g in grades
        length(g) == n || error("grade has length $(length(g)) but expected $n")
        for i in 1:n
            val = orient[i] == 1 ? RT(g[i]) : -RT(g[i])
            push!(axes[i], val)
        end
    end
    for i in 1:n
        axes[i] = sort!(unique!(axes[i]))
    end
    return ntuple(i -> axes[i], n)
end

function _axes_from_multigrades(grades::Vector{<:AbstractVector{<:NTuple{N,T}}}, n::Int;
                                orientation::Union{Nothing,NTuple{N,Int}}=nothing) where {N,T}
    flat = NTuple{N,T}[]
    sizehint!(flat, sum(length, grades))
    for gs in grades
        append!(flat, gs)
    end
    return _axes_from_grades(flat, n; orientation=orientation)
end

function _validate_axes_sorted(axes)
    for (i, ax) in enumerate(axes)
        if !issorted(ax)
            error("encode(data, filtration): axis $i must be sorted ascending.")
        end
    end
    return nothing
end

function _validate_axes_kind(axes; axis_kind=nothing)
    # axis_kind may be :zn / :rn to force integer vs real axes.
    is_int_typed_axis(ax) = all(x -> x isa Integer, ax)
    is_int_valued_axis(ax) = all(x -> (x isa Integer) || (x isa Real && isinteger(x)), ax)

    if axis_kind === nothing
        return nothing
    end

    axis_kind in (:zn, :rn) || error("encode(data, filtration): axis_kind must be :zn or :rn.")
    if axis_kind == :zn
        for (i, ax) in enumerate(axes)
            if !is_int_typed_axis(ax)
                error("encode(data, filtration): axis $i must be integer-typed for axis_kind=:zn.")
            end
        end
    else
        for (i, ax) in enumerate(axes)
            if is_int_typed_axis(ax)
                error("encode(data, filtration): axis $i is integer-typed but axis_kind=:rn.")
            end
        end
    end
    return nothing
end

function _axes_key(axes)
    return tuple((Tuple(ax) for ax in axes)...)
end

function _coarsen_axis(ax::Vector{T}, max_len::Int) where {T}
    n = length(ax)
    n <= max_len && return ax
    idxs = round.(Int, range(1, n; length=max_len))
    return ax[idxs]
end

function _coarsen_axes(axes::NTuple{N,Vector{T}}, max_len::Int) where {N,T}
    return ntuple(i -> _coarsen_axis(axes[i], max_len), N)
end

@inline function _quantize_eps_vec(eps, ::Val{N}) where {N}
    if eps isa Number
        return ntuple(_ -> Float64(eps), N)
    elseif eps isa Tuple
        length(eps) == N || error("encode(data, filtration): eps length mismatch.")
        return ntuple(i -> Float64(eps[i]), N)
    end
    error("encode(data, filtration): eps must be a number or tuple.")
end

function _quantize_grades(G::GradedComplex, eps)
    N = length(G.grades[1])
    eps_vec = _quantize_eps_vec(eps, Val(N))
    grades = Vector{NTuple{N,Float64}}(undef, length(G.grades))
    for i in eachindex(G.grades)
        g = G.grades[i]
        grades[i] = ntuple(j -> round(Float64(g[j]) / eps_vec[j]) * eps_vec[j], N)
    end
    return _rebuild_graded_complex(G, grades)
end

function _quantize_grades(G::MultiCriticalGradedComplex, eps)
    grade_offsets, grade_data = _packed_multigrade_storage(G)
    isempty(grade_data) && error("MultiCriticalGradedComplex has no grades.")
    N = length(grade_data[1])
    eps_vec = _quantize_eps_vec(eps, Val(N))
    out_offsets = Vector{Int}(undef, length(grade_offsets))
    out_offsets[1] = 1
    out_data = NTuple{N,Float64}[]
    sizehint!(out_data, length(grade_data))
    @inbounds for i in 1:(length(grade_offsets) - 1)
        lo = grade_offsets[i]
        hi = grade_offsets[i + 1] - 1
        qtmp = Vector{NTuple{N,Float64}}(undef, hi - lo + 1)
        p = 1
        for k in lo:hi
            g = grade_data[k]
            qtmp[p] = ntuple(j -> round(Float64(g[j]) / eps_vec[j]) * eps_vec[j], N)
            p += 1
        end
        sort!(qtmp)
        unique!(qtmp)
        append!(out_data, qtmp)
        out_offsets[i + 1] = length(out_data) + 1
    end
    return _rebuild_multicritical_complex(G, out_offsets, out_data)
end

function _quantize_simplex_tree(ST::SimplexTreeMulti{N,T}, eps) where {N,T}
    eps_vec = _quantize_eps_vec(eps, Val(N))
    out_offsets = Vector{Int}(undef, simplex_count(ST) + 1)
    out_offsets[1] = 1
    out_grades = NTuple{N,Float64}[]
    sizehint!(out_grades, length(ST.grade_data))
    @inbounds for sid in 1:simplex_count(ST)
        lo = ST.grade_offsets[sid]
        hi = ST.grade_offsets[sid + 1] - 1
        qtmp = Vector{NTuple{N,Float64}}(undef, hi - lo + 1)
        p = 1
        for k in lo:hi
            g = ST.grade_data[k]
            q = ntuple(i -> round(Float64(g[i]) / eps_vec[i]) * eps_vec[i], N)
            qtmp[p] = q
            p += 1
        end
        sort!(qtmp)
        unique!(qtmp)
        for q in qtmp
            push!(out_grades, q)
        end
        out_offsets[sid + 1] = length(out_grades) + 1
    end
    return SimplexTreeMulti(
        copy(ST.simplex_offsets),
        copy(ST.simplex_vertices),
        copy(ST.simplex_dims),
        copy(ST.dim_offsets),
        out_offsets,
        out_grades,
    )
end

function _poset_from_axes_cached(axes, orientation;
                                 cache::Union{Nothing,EncodingCache}=nothing)
    key = (_axes_key(axes), orientation)
    if cache === nothing
        return poset_from_axes(axes; orientation=orientation, kind=:grid)
    end
    Base.lock(cache.lock)
    try
        entry = get(cache.posets, key, nothing)
        if entry === nothing
            P = poset_from_axes(axes; orientation=orientation, kind=:grid)
            cache.posets[key] = PosetCachePayload(P)
            return P
        end
        return entry.value
    finally
        Base.unlock(cache.lock)
    end
end

@inline function _ingestion_poset_fast_key(data,
                                           filtration::AbstractFiltration,
                                           orientation,
                                           axes_policy::Symbol,
                                           max_axis_len,
                                           axis_kind,
                                           eps,
                                           multicritical::Symbol,
                                           onecritical_selector::Symbol,
                                           onecritical_enforce_boundary::Bool)
    fs_hash = let f = filtration
        try
            fs = _filtration_spec(f)
            UInt(hash((fs.kind, fs.params)))
        catch
            UInt(hash((typeof(f), f)))
        end
    end
    return (
        :ingestion_poset_fast,
        UInt(objectid(data)),
        fs_hash,
        orientation,
        axes_policy,
        max_axis_len,
        axis_kind,
        eps,
        multicritical,
        onecritical_selector,
        onecritical_enforce_boundary,
    )
end

@inline function _get_geometry_cached(cache::Union{Nothing,EncodingCache}, key)
    cache === nothing && return nothing
    Base.lock(cache.lock)
    try
        entry = get(cache.geometry, key, nothing)
        return entry === nothing ? nothing : entry.value
    finally
        Base.unlock(cache.lock)
    end
end

@inline function _set_geometry_cached!(cache::Union{Nothing,EncodingCache}, key, value)
    cache === nothing && return value
    Base.lock(cache.lock)
    try
        cache.geometry[key] = GeometryCachePayload(value)
    finally
        Base.unlock(cache.lock)
    end
    return value
end

@inline function _ingestion_module_fast_key(poset_fast_key, P, degree::Int, field)
    return (
        :ingestion_module_fast,
        poset_fast_key,
        UInt(objectid(P)),
        Int(degree),
        UInt(hash((typeof(field), field))),
    )
end

@inline function _ingestion_plan_norm_key(spec::FiltrationSpec,
                                          stage::Symbol,
                                          field::AbstractCoeffField)
    return (
        :ingestion_plan_norm,
        UInt(hash((spec.kind, spec.params))),
        stage,
        UInt(hash((typeof(field), field))),
    )
end

function _poset_vertex_coords(axes::NTuple{N,Vector{T}}) where {N,T}
    sizes = ntuple(i -> length(axes[i]), N)
    idxs = _grid_tuples(sizes)
    coords = Vector{Vector{Int}}(undef, length(idxs))
    for i in eachindex(idxs)
        coords[i] = [Int(axes[j][idxs[i][j]]) for j in 1:N]
    end
    return coords
end

function _grades_by_dim(G::GradedComplex)
    _, dim_offsets = _packed_cell_storage(G)
    counts = _packed_dim_counts(dim_offsets)
    total = _packed_total_cells(dim_offsets)
    length(G.grades) == total ||
        error("GradedComplex.grades length $(length(G.grades)) does not match total cells $(total).")
    out = Vector{Vector{typeof(G.grades[1])}}(undef, length(counts))
    idx = 1
    for d in 1:length(counts)
        out[d] = Vector{typeof(G.grades[1])}(undef, counts[d])
        for j in 1:counts[d]
            out[d][j] = G.grades[idx]
            idx += 1
        end
    end
    return out
end

function _grades_by_dim(G::MultiCriticalGradedComplex)
    _, dim_offsets = _packed_cell_storage(G)
    grade_offsets, grade_data = _packed_multigrade_storage(G)
    counts = _packed_dim_counts(dim_offsets)
    total = _packed_total_cells(dim_offsets)
    length(grade_offsets) == total + 1 ||
        error("MultiCriticalGradedComplex: grade_offsets length mismatch.")
    CellT = SubArray{eltype(grade_data),1,typeof(grade_data),Tuple{UnitRange{Int}},true}
    out = Vector{Vector{CellT}}(undef, length(counts))
    idx = 1
    @inbounds for d in 1:length(counts)
        out[d] = Vector{CellT}(undef, counts[d])
        for j in 1:counts[d]
            lo = grade_offsets[idx]
            hi = grade_offsets[idx + 1] - 1
            out[d][j] = view(grade_data, lo:hi)
            idx += 1
        end
    end
    return out
end

@inline function _tuple_lex_lt(a::NTuple{N,T}, b::NTuple{N,T}) where {N,T}
    @inbounds for i in 1:N
        ai = a[i]
        bi = b[i]
        ai < bi && return true
        ai > bi && return false
    end
    return false
end

function _select_one_critical_grade(grades::AbstractVector{<:NTuple{N,T}},
                                    selector::Symbol) where {N,T}
    isempty(grades) && error("one_criticalify: encountered empty per-cell grade set.")
    best = grades[1]
    if selector === :first
        return best
    elseif selector === :lexmin
        @inbounds for i in 2:length(grades)
            gi = grades[i]
            if _tuple_lex_lt(gi, best)
                best = gi
            end
        end
        return best
    elseif selector === :lexmax
        @inbounds for i in 2:length(grades)
            gi = grades[i]
            if _tuple_lex_lt(best, gi)
                best = gi
            end
        end
        return best
    end
    throw(ArgumentError("one_criticalify: unsupported selector=$(selector). Supported: :lexmin, :lexmax, :first"))
end

@inline function _tuple_componentwise_max(a::NTuple{N,T}, b::NTuple{N,T}) where {N,T}
    return ntuple(i -> (a[i] >= b[i] ? a[i] : b[i]), N)
end

"""
    one_criticalify(G::MultiCriticalGradedComplex;
                    selector::Symbol=:lexmin,
                    enforce_boundary::Bool=true) -> GradedComplex

Convert a multi-critical graded complex into a one-critical graded complex by
selecting one representative birth grade per cell.

Selection policy:
- `:lexmin` (default): lexicographically smallest grade in each per-cell set.
- `:lexmax`: lexicographically largest grade.
- `:first`: first stored grade.

If `enforce_boundary=true`, selected grades are lifted so each higher-dimensional
cell grade is componentwise at least each of its boundary-face grades.
"""
function one_criticalify(G::MultiCriticalGradedComplex;
                         selector::Symbol=:lexmin,
                         enforce_boundary::Bool=true)
    _, dim_offsets = _packed_cell_storage(G)
    grade_offsets, grade_data = _packed_multigrade_storage(G)
    total = length(grade_offsets) - 1
    total > 0 || error("one_criticalify: complex has no cells.")
    N = length(grade_data[1])
    T = eltype(grade_data[1])
    grades = Vector{NTuple{N,T}}(undef, total)
    @inbounds for i in 1:total
        lo = grade_offsets[i]
        hi = grade_offsets[i + 1] - 1
        grades[i] = _select_one_critical_grade(view(grade_data, lo:hi), selector)
    end

    if enforce_boundary
        counts = _packed_dim_counts(dim_offsets)
        nd = length(counts)
        @inbounds for slot in 2:nd
            B = G.boundaries[slot - 1]
            nrows, ncols = size(B)
            nrows == counts[slot - 1] || error("one_criticalify: boundary $(slot-1) row count mismatch.")
            ncols == counts[slot] || error("one_criticalify: boundary $(slot-1) col count mismatch.")
            prev_off = dim_offsets[slot - 1] - 1
            cur_off = dim_offsets[slot] - 1
            for j in 1:ncols
                g = grades[cur_off + j]
                for p in nzrange(B, j)
                    i = rowvals(B)[p]
                    g = _tuple_componentwise_max(g, grades[prev_off + i])
                end
                grades[cur_off + j] = g
            end
        end
    end

    return _rebuild_graded_complex(G, grades)
end

one_criticalify(G::GradedComplex; kwargs...) = G

function _birth_indices(grades::Vector{<:NTuple{N,<:Real}},
                        axes::NTuple{N,Vector{<:Real}},
                        orientation::NTuple{N,Int}) where {N}
    births = Vector{NTuple{N,Int}}(undef, length(grades))
    for i in eachindex(grades)
        g = grades[i]
        length(g) == N || error("grade length $(length(g)) does not match N=$(N)")
        births[i] = ntuple(k -> begin
            val = orientation[k] == 1 ? g[k] : -g[k]
            idx = searchsortedlast(axes[k], val)
            idx >= 1 || error("grade component $(g[k]) falls below axis minimum for axis $k")
            idx
        end, N)
    end
    return births
end

function _birth_indices(grades::Vector{<:AbstractVector{<:NTuple{N,<:Real}}},
                        axes::NTuple{N,Vector{<:Real}},
                        orientation::NTuple{N,Int}) where {N}
    births = Vector{Vector{NTuple{N,Int}}}(undef, length(grades))
    for i in eachindex(grades)
        gi = grades[i]
        isempty(gi) && error("multi-critical grade set at cell $i is empty.")
        out = Vector{NTuple{N,Int}}(undef, length(gi))
        for j in eachindex(gi)
            g = gi[j]
            out[j] = ntuple(k -> begin
                val = orientation[k] == 1 ? g[k] : -g[k]
                idx = searchsortedlast(axes[k], val)
                idx >= 1 || error("grade component $(g[k]) falls below axis minimum for axis $k")
                idx
            end, N)
        end
        births[i] = _minimal_tuple_set(unique(out))
    end
    return births
end

@inline function _is_chain_vertex_index_1d(vertex_idxs::Vector{NTuple{1,Int}})
    @inbounds for i in eachindex(vertex_idxs)
        vertex_idxs[i][1] == i || return false
    end
    return true
end

function _active_lists_chain_1d(births::Vector{NTuple{1,Int}},
                                vertex_idxs::Vector{NTuple{1,Int}})
    n = length(vertex_idxs)
    buckets = [Int[] for _ in 1:n]
    @inbounds for c in eachindex(births)
        b = births[c][1]
        1 <= b <= n || continue
        push!(buckets[b], c)
    end
    active = Vector{Vector{Int}}(undef, n)
    current = Int[]
    sizehint!(current, length(births))
    @inbounds for i in 1:n
        bi = buckets[i]
        if !isempty(bi)
            merged = Vector{Int}(undef, length(current) + length(bi))
            a = 1
            b = 1
            t = 1
            while a <= length(current) && b <= length(bi)
                if current[a] <= bi[b]
                    merged[t] = current[a]
                    a += 1
                else
                    merged[t] = bi[b]
                    b += 1
                end
                t += 1
            end
            while a <= length(current)
                merged[t] = current[a]
                a += 1
                t += 1
            end
            while b <= length(bi)
                merged[t] = bi[b]
                b += 1
                t += 1
            end
            current = merged
        end
        active[i] = copy(current)
    end
    return active
end

function _active_lists(births::Vector{NTuple{N,Int}},
                       vertex_idxs::Vector{NTuple{N,Int}},
                       orientation::NTuple{N,Int};
                       multicritical::Symbol=:union) where {N}
    multicritical in (:union, :intersection) ||
        throw(ArgumentError("_active_lists: multicritical must be :union or :intersection."))
    if _ACTIVE_LISTS_CHAIN_FASTPATH[] &&
       N == 1 && multicritical === :union &&
       orientation[1] == 1 &&
       _is_chain_vertex_index_1d(vertex_idxs)
        return _active_lists_chain_1d(births, vertex_idxs)
    end
    active = Vector{Vector{Int}}(undef, length(vertex_idxs))
    for i in eachindex(vertex_idxs)
        lst = Int[]
        vi = vertex_idxs[i]
        for c in eachindex(births)
            if _tuple_leq(births[c], vi, orientation)
                push!(lst, c)
            end
        end
        active[i] = lst
    end
    return active
end

function _active_lists(births::Vector{<:AbstractVector{NTuple{N,Int}}},
                       vertex_idxs::Vector{NTuple{N,Int}},
                       orientation::NTuple{N,Int};
                       multicritical::Symbol=:union) where {N}
    multicritical in (:union, :intersection) ||
        throw(ArgumentError("_active_lists: multicritical must be :union or :intersection."))
    if _ACTIVE_LISTS_CHAIN_FASTPATH[] &&
       N == 1 &&
       orientation[1] == 1 &&
       _is_chain_vertex_index_1d(vertex_idxs)
        reduced = Vector{NTuple{1,Int}}(undef, length(births))
        @inbounds for c in eachindex(births)
            bc = births[c]
            isempty(bc) && error("_active_lists: empty multi-critical birth set at cell $c.")
            if multicritical === :union
                bmin = typemax(Int)
                for b in bc
                    b[1] < bmin && (bmin = b[1])
                end
                reduced[c] = (bmin,)
            else
                bmax = typemin(Int)
                for b in bc
                    b[1] > bmax && (bmax = b[1])
                end
                reduced[c] = (bmax,)
            end
        end
        return _active_lists_chain_1d(reduced, vertex_idxs)
    end
    active = Vector{Vector{Int}}(undef, length(vertex_idxs))
    for i in eachindex(vertex_idxs)
        lst = Int[]
        vi = vertex_idxs[i]
        for c in eachindex(births)
            bc = births[c]
            keep = multicritical === :intersection
            @inbounds for b in bc
                hit = _tuple_leq(b, vi, orientation)
                if multicritical === :union
                    if hit
                        keep = true
                        break
                    end
                else
                    if !hit
                        keep = false
                        break
                    end
                end
            end
            keep && push!(lst, c)
        end
        active[i] = lst
    end
    return active
end

function _pmodule_from_active_lists(P::AbstractPoset,
                                    active::Vector{Vector{Int}};
                                    field::AbstractCoeffField=QQField())
    K = coeff_type(field)
    dims = [length(lst) for lst in active]
    cc = _get_cover_cache(P)
    n = nvertices(P)
    preds = [_preds(cc, v) for v in 1:n]
    succs = [_succs(cc, u) for u in 1:n]
    pred_slot_of_succ = u -> _pred_slots_of_succ(cc, u)

    @inline function _row_for_col(Mu::Vector{Int}, Mv::Vector{Int})
        out = Vector{Int}(undef, length(Mu))
        i = 1
        @inbounds for j in eachindex(Mu)
            c = Mu[j]
            while i <= length(Mv) && Mv[i] < c
                i += 1
            end
            out[j] = (i <= length(Mv) && Mv[i] == c) ? i : 0
        end
        return out
    end

    maps_from_pred = [Vector{_StructuralInclusionMap{K}}(undef, length(preds[v])) for v in 1:n]
    maps_to_succ = [Vector{_StructuralInclusionMap{K}}(undef, length(succs[u])) for u in 1:n]
    @inbounds for u in 1:n
        Mu = active[u]
        su = succs[u]
        outu = maps_to_succ[u]
        for j in eachindex(su)
            v = su[j]
            row_for_col = _row_for_col(Mu, active[v])
            A = _StructuralInclusionMap{K}(dims[v], dims[u], row_for_col)
            outu[j] = A
            ip = pred_slot_of_succ(u)[j]
            maps_from_pred[v][ip] = A
        end
    end
    store = CoverEdgeMapStore{K,_StructuralInclusionMap{K}}(preds, succs, maps_from_pred, maps_to_succ, cc.nedges)
    return PModule{K}(P, dims, store; field=field)
end

function _normalize_cochain_boundaries(grades_by_dim::Vector,
                                       boundaries_in::Vector{SparseMatrixCSC{Int,Int}})
    expected = length(grades_by_dim) - 1
    boundaries = copy(boundaries_in)
    if length(boundaries) < expected
        for k in (length(boundaries) + 1):expected
            rows = length(grades_by_dim[k])
            cols = length(grades_by_dim[k + 1])
            push!(boundaries, spzeros(Int, rows, cols))
        end
    elseif length(boundaries) > expected
        error("cochain boundaries length mismatch (expected $(expected)).")
    end
    return boundaries
end

mutable struct LazyModuleCochainComplex
    tmin::Int
    tmax::Int
    P::Any
    axes::Any
    orientation::Any
    field::Any
    multicritical::Symbol
    grades_by_dim::Vector
    boundaries::Vector{SparseMatrixCSC{Int,Int}}
    boundaries_field::Vector
    vertex_idxs::Union{Nothing,Vector}
    births_by_dim::Vector
    active_by_dim::Vector
    pos_by_dim::Vector
    terms::Vector
    diffs::Vector
end

function _lazy_cochain_complex_from_grades_and_boundaries(grades_by_dim::Vector{GT},
                                                          boundaries_in::Vector{SparseMatrixCSC{Int,Int}},
                                                          P::AbstractPoset,
                                                          axes::NTuple{N,Vector{T}};
                                                          orientation::NTuple{N,Int}=ntuple(_ -> 1, N),
                                                          field::AbstractCoeffField=QQField(),
                                                          multicritical::Symbol=:union) where {GT,N,T}
    sizes = ntuple(i -> length(axes[i]), N)
    vertex_idxs = N == 1 ? _grid_tuples(sizes) : nothing
    boundaries = _normalize_cochain_boundaries(grades_by_dim, boundaries_in)

    nd = length(grades_by_dim)
    nd > 0 || error("cochain requires at least one term.")

    births_by_dim = Vector{Any}(undef, nd)
    for d in 1:nd
        births_by_dim[d] = _birth_indices(grades_by_dim[d], axes, orientation)
    end

    active_by_dim = Vector{Any}(undef, nd)
    pos_by_dim = Vector{Any}(undef, nd)
    terms = Vector{Any}(undef, nd)
    for d in 1:nd
        active_by_dim[d] = nothing
        pos_by_dim[d] = nothing
        terms[d] = nothing
    end
    diffs = Vector{Any}(undef, max(0, nd - 1))
    for k in eachindex(diffs)
        diffs[k] = nothing
    end
    boundaries_field = Vector{Any}(undef, length(boundaries))
    fill!(boundaries_field, nothing)

    return LazyModuleCochainComplex(
        0,
        nd - 1,
        P,
        axes,
        orientation,
        field,
        multicritical,
        grades_by_dim,
        boundaries,
        boundaries_field,
        vertex_idxs,
        births_by_dim,
        active_by_dim,
        pos_by_dim,
        terms,
        diffs,
    )
end

@inline function _lazy_vertex_idxs!(L::LazyModuleCochainComplex)
    idxs = L.vertex_idxs
    if idxs === nothing
        sizes = ntuple(i -> length(L.axes[i]), length(L.axes))
        idxs = _grid_tuples(sizes)
        L.vertex_idxs = idxs
    end
    return idxs
end

function _lazy_ensure_active!(L::LazyModuleCochainComplex, idx::Int)
    active = L.active_by_dim[idx]
    if active === nothing
        births = L.births_by_dim[idx]
        active = _active_lists(births, _lazy_vertex_idxs!(L), L.orientation; multicritical=L.multicritical)
        L.active_by_dim[idx] = active
        L.pos_by_dim[idx] = nothing
    end
    return active, L.pos_by_dim[idx]
end

function _boundary_to_field_sparse(B::SparseMatrixCSC{Int,Int},
                                   field::AbstractCoeffField)
    K = coeff_type(field)
    vals = Vector{K}(undef, length(B.nzval))
    @inbounds for i in eachindex(B.nzval)
        vals[i] = coerce(field, B.nzval[i])
    end
    return SparseMatrixCSC(size(B, 1), size(B, 2), copy(B.colptr), copy(B.rowval), vals)
end

function _lazy_boundary_field!(L::LazyModuleCochainComplex, k::Int)
    B = L.boundaries[k]
    cached = L.boundaries_field[k]
    if cached !== nothing
        return cached
    end
    Bf = _boundary_to_field_sparse(B, L.field)
    L.boundaries_field[k] = Bf
    return Bf
end

function _lazy_term_idx!(L::LazyModuleCochainComplex, idx::Int)
    term = L.terms[idx]
    if term === nothing
        active, _ = _lazy_ensure_active!(L, idx)
        term = _pmodule_from_active_lists(L.P, active; field=L.field)
        L.terms[idx] = term
    end
    return term
end

@inline function _sorted_pos_or_zero(sorted::Vector{Int}, x::Int)
    i = searchsortedfirst(sorted, x)
    if i <= length(sorted) && @inbounds(sorted[i] == x)
        return i
    end
    return 0
end

@inline function _restricted_coboundary_component(B::SparseMatrixCSC{Int,Int},
                                                  active_rows::Vector{Int},
                                                  active_cols::Vector{Int},
                                                  field::AbstractCoeffField)
    K = coeff_type(field)
    I = Int[]
    J = Int[]
    V = K[]

    colptr = B.colptr
    nnz_hint = 0
    @inbounds for c_global in active_cols
        nnz_hint += colptr[c_global + 1] - colptr[c_global]
    end
    sizehint!(I, nnz_hint)
    sizehint!(J, nnz_hint)
    sizehint!(V, nnz_hint)

    rows = rowvals(B)
    vals = nonzeros(B)
    @inbounds for (c_local, c_global) in enumerate(active_cols)
        for ptr in colptr[c_global]:(colptr[c_global + 1] - 1)
            r_local = _sorted_pos_or_zero(active_rows, rows[ptr])
            if r_local != 0
                push!(I, c_local)
                push!(J, r_local)
                push!(V, coerce(field, vals[ptr]))
            end
        end
    end
    return sparse(I, J, V, length(active_cols), length(active_rows))
end

@inline function _restricted_coboundary_component(B::SparseMatrixCSC{Int,Int},
                                                  active_rows::Vector{Int},
                                                  active_cols::Vector{Int},
                                                  field::AbstractCoeffField,
                                                  scratch::_ActiveIndexLookupScratch)
    K = coeff_type(field)
    I = Int[]
    J = Int[]
    V = K[]

    _active_lookup_fill!(scratch, active_rows)

    colptr = B.colptr
    nnz_hint = 0
    @inbounds for c_global in active_cols
        nnz_hint += colptr[c_global + 1] - colptr[c_global]
    end
    sizehint!(I, nnz_hint)
    sizehint!(J, nnz_hint)
    sizehint!(V, nnz_hint)

    rows = rowvals(B)
    vals = nonzeros(B)
    @inbounds for (c_local, c_global) in enumerate(active_cols)
        for ptr in colptr[c_global]:(colptr[c_global + 1] - 1)
            r_local = _active_lookup_get(scratch, rows[ptr])
            if r_local != 0
                push!(I, c_local)
                push!(J, r_local)
                push!(V, coerce(field, vals[ptr]))
            end
        end
    end
    return sparse(I, J, V, length(active_cols), length(active_rows))
end

function _lazy_diff_components(L::LazyModuleCochainComplex,
                               k::Int;
                               threaded::Bool=true)
    K = coeff_type(L.field)
    active_k = L.active_by_dim[k]
    active_k1 = L.active_by_dim[k + 1]
    B = L.boundaries[k]
    nP = nvertices(L.P)
    comps = Vector{SparseMatrixCSC{K,Int}}(undef, nP)

    use_lookup = size(B, 1) <= _ACTIVE_INDEX_TABLE_MAX_ROWS[]
    threaded_ok = threaded &&
                  Threads.nthreads() > 1 &&
                  nP >= _LAZY_DIFF_THREADS_MIN_VERTICES[]
    if threaded_ok
        scratch = use_lookup ? [_ActiveIndexLookupScratch(size(B, 1)) for _ in 1:Threads.nthreads()] : nothing
        Threads.@threads :static for i in 1:nP
            Lk = active_k[i]
            Lk1 = active_k1[i]
            if use_lookup
                comps[i] = _restricted_coboundary_component(B, Lk, Lk1, L.field, scratch[Threads.threadid()])
            else
                comps[i] = _restricted_coboundary_component(B, Lk, Lk1, L.field)
            end
        end
    else
        scratch = use_lookup ? _ActiveIndexLookupScratch(size(B, 1)) : nothing
        @inbounds for i in 1:nP
            Lk = active_k[i]
            Lk1 = active_k1[i]
            if use_lookup
                comps[i] = _restricted_coboundary_component(B, Lk, Lk1, L.field, scratch)
            else
                comps[i] = _restricted_coboundary_component(B, Lk, Lk1, L.field)
            end
        end
    end
    return comps
end

function _lazy_diff_idx!(L::LazyModuleCochainComplex, k::Int)
    d = L.diffs[k]
    if d === nothing
        K = coeff_type(L.field)
        dom = _lazy_term_idx!(L, k)
        cod = _lazy_term_idx!(L, k + 1)
        if nnz(L.boundaries[k]) == 0
            d = Modules.zero_morphism(dom, cod)
        else
            comps = _lazy_diff_components(L, k; threaded=true)
            d = PMorphism{K}(dom, cod, comps)
        end
        L.diffs[k] = d
    end
    return d
end

function _lazy_term(L::LazyModuleCochainComplex, t::Int)
    if t < L.tmin || t > L.tmax
        return Modules.zero_pmodule(L.P; field=L.field)
    end
    idx = t - L.tmin + 1
    return _lazy_term_idx!(L, idx)
end

@inline function _lazy_term_dims_idx!(L::LazyModuleCochainComplex, idx::Int)
    active, _ = _lazy_ensure_active!(L, idx)
    dims = Vector{Int}(undef, length(active))
    @inbounds for i in eachindex(active)
        dims[i] = length(active[i])
    end
    return dims
end

@inline function _euler_dims_fastpath_chain_1d(L::LazyModuleCochainComplex)
    length(L.orientation) == 1 || return false
    idxs = L.vertex_idxs
    return idxs === nothing || _is_chain_vertex_index_1d(idxs)
end

function _accumulate_euler_dims_chain_1d!(chi::Vector{Int},
                                          scratch::Vector{Int},
                                          births::Vector{NTuple{1,Int}},
                                          sgn::Int,
                                          orientation::Int)
    n = length(chi)
    fill!(scratch, 0)
    base = 0
    if orientation == 1
        @inbounds for bc in births
            b = bc[1]
            1 <= b <= n || continue
            scratch[b] += sgn
        end
    else
        @inbounds for bc in births
            b = bc[1]
            1 <= b <= n || continue
            base += sgn
            b < n && (scratch[b + 1] -= sgn)
        end
    end
    acc = base
    @inbounds for i in 1:n
        acc += scratch[i]
        chi[i] += acc
    end
    return chi
end

@inline function _chain_1d_birth_threshold(births::AbstractVector{NTuple{1,Int}},
                                           multicritical::Symbol,
                                           orientation::Int)
    isempty(births) && error("_euler_dims_from_lazy: empty multi-critical birth set.")
    if orientation == 1
        if multicritical === :union
            b = typemax(Int)
            @inbounds for gi in births
                gi[1] < b && (b = gi[1])
            end
            return b
        end
        b = typemin(Int)
        @inbounds for gi in births
            gi[1] > b && (b = gi[1])
        end
        return b
    end
    if multicritical === :union
        b = typemin(Int)
        @inbounds for gi in births
            gi[1] > b && (b = gi[1])
        end
        return b
    end
    b = typemax(Int)
    @inbounds for gi in births
        gi[1] < b && (b = gi[1])
    end
    return b
end

function _accumulate_euler_dims_chain_1d!(chi::Vector{Int},
                                          scratch::Vector{Int},
                                          births::Vector{<:AbstractVector{NTuple{1,Int}}},
                                          sgn::Int,
                                          multicritical::Symbol,
                                          orientation::Int)
    n = length(chi)
    fill!(scratch, 0)
    base = 0
    if orientation == 1
        @inbounds for bc in births
            b = _chain_1d_birth_threshold(bc, multicritical, orientation)
            1 <= b <= n || continue
            scratch[b] += sgn
        end
    else
        @inbounds for bc in births
            b = _chain_1d_birth_threshold(bc, multicritical, orientation)
            1 <= b <= n || continue
            base += sgn
            b < n && (scratch[b + 1] -= sgn)
        end
    end
    acc = base
    @inbounds for i in 1:n
        acc += scratch[i]
        chi[i] += acc
    end
    return chi
end

function _euler_dims_from_lazy_chain_1d(L::LazyModuleCochainComplex)
    nd = length(L.terms)
    nd == 0 && return Int[]
    chi = zeros(Int, length(L.axes[1]))
    scratch = zeros(Int, length(chi))
    orientation = L.orientation[1]
    @inbounds for idx in 1:nd
        births = L.births_by_dim[idx]
        t = L.tmin + idx - 1
        sgn = isodd(t) ? -1 : 1
        if births isa Vector{NTuple{1,Int}}
            _accumulate_euler_dims_chain_1d!(chi, scratch, births, sgn, orientation)
        else
            _accumulate_euler_dims_chain_1d!(chi, scratch, births, sgn, L.multicritical, orientation)
        end
    end
    return chi
end

function _euler_dims_from_lazy(L::LazyModuleCochainComplex)
    nd = length(L.terms)
    nd == 0 && return Int[]
    _euler_dims_fastpath_chain_1d(L) && return _euler_dims_from_lazy_chain_1d(L)
    chi = zeros(Int, nvertices(L.P))
    @inbounds for idx in 1:nd
        dims = _lazy_term_dims_idx!(L, idx)
        t = L.tmin + idx - 1
        sgn = isodd(t) ? -1 : 1
        for u in eachindex(chi)
            chi[u] += sgn * dims[u]
        end
    end
    return chi
end

@inline function _can_use_direct_euler_signed_measure_chain_1d(L::LazyModuleCochainComplex,
                                                               pi::GridEncodingMap{1},
                                                               opts::InvariantOptions)
    _euler_dims_fastpath_chain_1d(L) || return false
    L.orientation[1] == 1 || return false
    pi.orientation[1] == 1 || return false
    opts.axes === nothing || return false
    opts.axes_policy === :encoding || return false
    return true
end

@inline function _can_use_direct_euler_signed_measure_grid_2d(L::LazyModuleCochainComplex,
                                                              pi::GridEncodingMap{2},
                                                              opts::InvariantOptions)
    length(L.orientation) == 2 || return false
    L.orientation == (1, 1) || return false
    pi.orientation == (1, 1) || return false
    opts.axes === nothing || return false
    opts.axes_policy === :encoding || return false
    @inbounds for births in L.births_by_dim
        births isa Vector{NTuple{2,Int}} || return false
    end
    return true
end

@inline function _accumulate_euler_measure_chain_1d!(delta::Vector{Int},
                                                     births::Vector{NTuple{1,Int}},
                                                     sgn::Int)
    n = length(delta)
    @inbounds for bc in births
        b = bc[1]
        1 <= b <= n || continue
        delta[b] += sgn
    end
    return delta
end

@inline function _accumulate_euler_measure_chain_1d!(delta::Vector{Int},
                                                     births::Vector{<:AbstractVector{NTuple{1,Int}}},
                                                     sgn::Int,
                                                     multicritical::Symbol)
    n = length(delta)
    @inbounds for bc in births
        b = _chain_1d_birth_threshold(bc, multicritical, 1)
        1 <= b <= n || continue
        delta[b] += sgn
    end
    return delta
end

function _euler_signed_measure_from_lazy_chain_1d(L::LazyModuleCochainComplex,
                                                  ax::AbstractVector;
                                                  drop_zeros::Bool=true,
                                                  max_terms::Int=0,
                                                  min_abs_weight::Real=0)
    ax1 = collect(ax)
    delta = zeros(Int, length(ax1))
    @inbounds for idx in eachindex(L.births_by_dim)
        births = L.births_by_dim[idx]
        t = L.tmin + idx - 1
        sgn = isodd(t) ? -1 : 1
        if births isa Vector{NTuple{1,Int}}
            _accumulate_euler_measure_chain_1d!(delta, births, sgn)
        else
            _accumulate_euler_measure_chain_1d!(delta, births, sgn, L.multicritical)
        end
    end

    inds = Vector{NTuple{1,Int}}()
    wts = Vector{Int}()
    sizehint!(inds, length(delta))
    sizehint!(wts, length(delta))
    @inbounds for i in eachindex(delta)
        wt = delta[i]
        if !drop_zeros || wt != 0
            push!(inds, (i,))
            push!(wts, wt)
        end
    end

    pm = PointSignedMeasure((ax1,), inds, wts)
    if max_terms > 0 || min_abs_weight > 0
        pm = truncate_point_signed_measure(pm;
            max_terms=max_terms,
            min_abs_weight=min_abs_weight)
    end
    return pm
end

function _euler_signed_measure_from_lazy_grid_2d(L::LazyModuleCochainComplex,
                                                 axes::NTuple{2,AbstractVector};
                                                 drop_zeros::Bool=true,
                                                 max_terms::Int=0,
                                                 min_abs_weight::Real=0)
    ax = (collect(axes[1]), collect(axes[2]))
    n1 = length(ax[1])
    n2 = length(ax[2])
    weights = Dict{Int,Int}()

    @inbounds for idx in eachindex(L.births_by_dim)
        births = L.births_by_dim[idx]::Vector{NTuple{2,Int}}
        t = L.tmin + idx - 1
        sgn = isodd(t) ? -1 : 1
        for bc in births
            i, j = bc
            1 <= i <= n1 || continue
            1 <= j <= n2 || continue
            key = i + (j - 1) * n1
            weights[key] = get(weights, key, 0) + sgn
        end
    end

    keys_sorted = sort!(collect(keys(weights)))
    inds = Vector{NTuple{2,Int}}()
    wts = Int[]
    sizehint!(inds, length(keys_sorted))
    sizehint!(wts, length(keys_sorted))
    @inbounds for key in keys_sorted
        wt = weights[key]
        if !drop_zeros || wt != 0
            j = cld(key, n1)
            i = key - (j - 1) * n1
            push!(inds, (i, j))
            push!(wts, wt)
        end
    end

    pm = PointSignedMeasure(ax, inds, wts)
    if max_terms > 0 || min_abs_weight > 0
        pm = truncate_point_signed_measure(pm;
            max_terms=max_terms,
            min_abs_weight=min_abs_weight)
    end
    return pm
end

function _lazy_diff(L::LazyModuleCochainComplex, t::Int)
    if t < L.tmin || t >= L.tmax
        return Modules.zero_morphism(_lazy_term(L, t), _lazy_term(L, t + 1))
    end
    idx = t - L.tmin + 1
    return _lazy_diff_idx!(L, idx)
end

mutable struct _LazyEncodedModule
    lazy::LazyModuleCochainComplex
    degree::Int
    cached_module::Any
    dims::Any
end

@inline function _lazy_encoded_module_from_lazy(lazy::LazyModuleCochainComplex, degree::Int)
    return _LazyEncodedModule(lazy, degree, nothing, nothing)
end

function _materialize_lazy_module!(M::_LazyEncodedModule)
    if M.cached_module === nothing
        M.cached_module = _cohomology_module_from_lazy(M.lazy, M.degree)
    end
    return M.cached_module
end

function _lazy_encoded_module_dims!(M::_LazyEncodedModule)
    if M.dims === nothing
        M.dims = _cohomology_dims_from_lazy(M.lazy, M.degree)
    end
    return M.dims
end

materialize_module(M::_LazyEncodedModule) = _materialize_lazy_module!(M)
module_dims(M::_LazyEncodedModule) = _lazy_encoded_module_dims!(M)

_materialize_complex(L::LazyModuleCochainComplex) = _materialize_cochain(L; check=true)

function change_field(L::LazyModuleCochainComplex, field::AbstractCoeffField)
    return change_field(_materialize_cochain(L; check=true), field)
end

@inline function _uf_find!(parent::Vector{Int}, x::Int)
    @inbounds while parent[x] != x
        parent[x] = parent[parent[x]]
        x = parent[x]
    end
    return x
end

@inline function _uf_union!(parent::Vector{Int}, sz::Vector{Int}, a::Int, b::Int)
    ra = _uf_find!(parent, a)
    rb = _uf_find!(parent, b)
    ra == rb && return ra
    @inbounds if sz[ra] < sz[rb]
        ra, rb = rb, ra
    end
    parent[rb] = ra
    sz[ra] += sz[rb]
    return ra
end

@inline function _uf_union_with_min!(parent::Vector{Int},
                                     sz::Vector{Int},
                                     root_min::Vector{Int},
                                     a::Int,
                                     b::Int)
    ra = _uf_find!(parent, a)
    rb = _uf_find!(parent, b)
    ra == rb && return ra
    @inbounds if sz[ra] < sz[rb]
        ra, rb = rb, ra
    end
    parent[rb] = ra
    sz[ra] += sz[rb]
    if root_min[rb] < root_min[ra]
        root_min[ra] = root_min[rb]
    end
    return ra
end

@inline function _sum_nested_lengths(xs::Vector{Vector{Int}})
    s = 0
    @inbounds for x in xs
        s += length(x)
    end
    return s
end

@inline function _use_h0_active_chain_incremental(nP::Int,
                                                  total_active_vertices::Int,
                                                  total_active_edges::Int)
    nP >= _H0_ACTIVE_CHAIN_INCREMENTAL_MIN_POS_VERTICES[] || return false
    return (total_active_vertices >= _H0_ACTIVE_CHAIN_INCREMENTAL_MIN_TOTAL_ACTIVE_VERTICES[]) ||
           (total_active_edges >= _H0_ACTIVE_CHAIN_INCREMENTAL_MIN_TOTAL_ACTIVE_EDGES[])
end

@inline function _use_degree_local_t1_fastpath(L::LazyModuleCochainComplex)
    _COHOMOLOGY_DEGREE_LOCAL_ALL_T[] || return false
    nP = nvertices(L.P)
    nP >= _COHOMOLOGY_DEGREE_LOCAL_T1_MIN_POS_VERTICES[] || return false

    idx1 = 1 - L.tmin + 1
    (1 <= idx1 <= length(L.active_by_dim)) || return false
    active1, _ = _lazy_ensure_active!(L, idx1)
    total_active1 = _sum_nested_lengths(active1)
    total_active1 >= _COHOMOLOGY_DEGREE_LOCAL_T1_MIN_TOTAL_ACTIVE_DIM1[] || return false

    idx2 = idx1 + 1
    idx2 > length(L.active_by_dim) && return true
    active2, _ = _lazy_ensure_active!(L, idx2)
    total_active2 = _sum_nested_lengths(active2)
    return total_active2 >= _COHOMOLOGY_DEGREE_LOCAL_T1_MIN_TOTAL_ACTIVE_DIM2[]
end

@inline function _use_degree_local_module_fastpath(L::LazyModuleCochainComplex, t::Int)
    _COHOMOLOGY_DEGREE_LOCAL_FASTPATH[] || return false
    t >= 2 && return true
    t == 1 || return false
    _COHOMOLOGY_DEGREE_LOCAL_ALL_T[] && return true
    return _use_degree_local_t1_fastpath(L)
end

function _births_from_prefix_active_sets(active::Vector{Vector{Int}}, nitems::Int)
    nitems >= 0 || return nothing
    births = zeros(Int, nitems)
    prev = Int[]
    @inbounds for p in eachindex(active)
        cur = active[p]
        last = 0
        for id in cur
            (1 <= id <= nitems) || return nothing
            id > last || return nothing
            last = id
            births[id] == 0 && (births[id] = p)
        end
        # Prefix active sets must be monotone on chain sweeps: prev subseteq cur.
        i = 1
        j = 1
        while i <= length(prev) && j <= length(cur)
            a = prev[i]
            b = cur[j]
            if a == b
                i += 1
                j += 1
            elseif a > b
                j += 1
            else
                return nothing
            end
        end
        i <= length(prev) && return nothing
        prev = cur
    end
    @inbounds for b in births
        b == 0 && return nothing
    end
    return births
end

function _edge_endpoints_from_boundary(B::SparseMatrixCSC{Int,Int})
    ne = size(B, 2)
    colptr = B.colptr
    rows = rowvals(B)
    endpoints = Vector{NTuple{2,Int}}(undef, ne)
    @inbounds for e in 1:ne
        lo = colptr[e]
        hi = colptr[e + 1] - 1
        if (hi - lo + 1) != 2
            return nothing
        end
        a = rows[lo]
        b = rows[lo + 1]
        if a == b
            return nothing
        end
        endpoints[e] = a < b ? (a, b) : (b, a)
    end
    return endpoints
end

@inline function _reconcile_component_basis(raw_component_reps::Vector{Int})
    order = sortperm(raw_component_reps)
    canon_of_raw = Vector{Int}(undef, length(raw_component_reps))
    canon_reps = Vector{Int}(undef, length(raw_component_reps))
    @inbounds for (canon, raw) in enumerate(order)
        canon_of_raw[raw] = canon
        canon_reps[canon] = raw_component_reps[raw]
    end
    return canon_of_raw, canon_reps
end

function _h0_components_unionfind_vertex(verts::Vector{Int},
                                         active_edges::Vector{Int},
                                         edge_endpoints::Vector{NTuple{2,Int}};
                                         lookup::Union{Nothing,_ActiveIndexLookupScratch}=nothing)
    nv = length(verts)
    if nv == 0
        return Int[], Int[]
    end

    lookup !== nothing && _active_lookup_fill!(lookup, verts)

    parent = collect(1:nv)
    sz = ones(Int, nv)
    @inbounds for e in active_edges
        (e < 1 || e > length(edge_endpoints)) && continue
        a, b = edge_endpoints[e]
        ia = lookup === nothing ? _sorted_pos_or_zero(verts, a) : _active_lookup_get(lookup, a)
        ib = lookup === nothing ? _sorted_pos_or_zero(verts, b) : _active_lookup_get(lookup, b)
        if ia != 0 && ib != 0 && ia != ib
            _uf_union!(parent, sz, ia, ib)
        end
    end

    root_to_raw = zeros(Int, nv)
    raw_reps = Int[]
    local_raw = Vector{Int}(undef, nv)
    @inbounds for j in 1:nv
        r = _uf_find!(parent, j)
        raw = root_to_raw[r]
        if raw == 0
            raw = length(raw_reps) + 1
            root_to_raw[r] = raw
            push!(raw_reps, verts[j])
        elseif verts[j] < raw_reps[raw]
            raw_reps[raw] = verts[j]
        end
        local_raw[j] = raw
    end

    canon_of_raw, canon_reps = _reconcile_component_basis(raw_reps)
    local_comp = Vector{Int}(undef, nv)
    @inbounds for j in 1:nv
        local_comp[j] = canon_of_raw[local_raw[j]]
    end
    return local_comp, canon_reps
end

function _cohomology_module_h0_unionfind_from_lazy(
    L::LazyModuleCochainComplex,
    active0::Vector{Vector{Int}},
    active1::Vector{Vector{Int}},
)
    nP = nvertices(L.P)
    K = coeff_type(L.field)
    nd = length(L.grades_by_dim)
    edge_endpoints = if nd >= 2
        _edge_endpoints_from_boundary(L.boundaries[1])
    else
        NTuple{2,Int}[]
    end
    edge_endpoints === nothing && return nothing

    if _H0_ACTIVE_CHAIN_INCREMENTAL[] &&
       nP >= _H0_ACTIVE_CHAIN_INCREMENTAL_MIN_POS_VERTICES[] &&
       _is_chain_cover_cache(_get_cover_cache(L.P), nP)
        total_active_vertices = _sum_nested_lengths(active0)
        total_active_edges = _sum_nested_lengths(active1)
        if (total_active_vertices >= _H0_ACTIVE_CHAIN_INCREMENTAL_MIN_TOTAL_ACTIVE_VERTICES[] ||
            total_active_edges >= _H0_ACTIVE_CHAIN_INCREMENTAL_MIN_TOTAL_ACTIVE_EDGES[]) &&
           _use_h0_active_chain_incremental(nP, total_active_vertices, total_active_edges)
            n0 = length(L.grades_by_dim[1])
            ne = nd >= 2 ? length(L.grades_by_dim[2]) : 0
            vb = _births_from_prefix_active_sets(active0, n0)
            eb = _births_from_prefix_active_sets(active1, ne)
            if vb !== nothing && eb !== nothing
                Mchain = _h0_module_chain_sweep(L.P, vb, eb, edge_endpoints; field=L.field)
                Mchain !== nothing && return Mchain
            end
        end
    end

    dims = Vector{Int}(undef, nP)
    comp_of_local = Vector{Vector{Int}}(undef, nP)
    comp_reps = Vector{Vector{Int}}(undef, nP)

    n0_global = isempty(L.boundaries) ? 0 : size(L.boundaries[1], 1)
    use_lookup = n0_global > 0 && n0_global <= _ACTIVE_INDEX_TABLE_MAX_ROWS[]
    threaded_ok = Threads.nthreads() > 1 && nP >= _LAZY_DIFF_THREADS_MIN_VERTICES[]
    if threaded_ok
        lookup = use_lookup ? [_ActiveIndexLookupScratch(n0_global) for _ in 1:Threads.nthreads()] : nothing
        Threads.@threads :static for i in 1:nP
            lk = use_lookup ? lookup[Threads.threadid()] : nothing
            loc, reps = _h0_components_unionfind_vertex(active0[i], active1[i], edge_endpoints; lookup=lk)
            comp_of_local[i] = loc
            comp_reps[i] = reps
            dims[i] = length(reps)
        end
    else
        lookup = use_lookup ? _ActiveIndexLookupScratch(n0_global) : nothing
        @inbounds for i in 1:nP
            loc, reps = _h0_components_unionfind_vertex(active0[i], active1[i], edge_endpoints; lookup=lookup)
            comp_of_local[i] = loc
            comp_reps[i] = reps
            dims[i] = length(reps)
        end
    end

    cc = _get_cover_cache(L.P)
    preds = [_preds(cc, v) for v in 1:nP]
    succs = [_succs(cc, u) for u in 1:nP]
    pred_slot_of_succ = u -> _pred_slots_of_succ(cc, u)
    maps_from_pred = [Vector{_StructuralInclusionMap{K}}(undef, length(preds[v])) for v in 1:nP]
    maps_to_succ = [Vector{_StructuralInclusionMap{K}}(undef, length(succs[u])) for u in 1:nP]
    lookup = use_lookup ? _ActiveIndexLookupScratch(n0_global) : nothing
    @inbounds for u in 1:nP
        su = succs[u]
        outu = maps_to_succ[u]
        reps_u = comp_reps[u]
        cu = length(reps_u)
        for j in eachindex(su)
            v = su[j]
            cv = dims[v]
            comps_v = comp_of_local[v]
            row_for_col = Vector{Int}(undef, cu)
            if lookup === nothing
                for comp_u in 1:cu
                    g = reps_u[comp_u]
                    lv = _sorted_pos_or_zero(active0[v], g)
                    row_for_col[comp_u] = lv == 0 ? 0 : comps_v[lv]
                end
            else
                _active_lookup_fill!(lookup, active0[v])
                for comp_u in 1:cu
                    g = reps_u[comp_u]
                    lv = _active_lookup_get(lookup, g)
                    row_for_col[comp_u] = lv == 0 ? 0 : comps_v[lv]
                end
            end
            A = _StructuralInclusionMap{K}(cv, cu, row_for_col)
            outu[j] = A
            ip = pred_slot_of_succ(u)[j]
            maps_from_pred[v][ip] = A
        end
    end
    store = CoverEdgeMapStore{K,_StructuralInclusionMap{K}}(preds, succs, maps_from_pred, maps_to_succ, cc.nedges)
    return PModule{K}(L.P, dims, store; field=L.field)
end

function _cohomology_dims_h0_unionfind_from_lazy(
    L::LazyModuleCochainComplex,
    active0::Vector{Vector{Int}},
    active1::Vector{Vector{Int}},
)
    nP = nvertices(L.P)
    nd = length(L.grades_by_dim)
    edge_endpoints = if nd >= 2
        _edge_endpoints_from_boundary(L.boundaries[1])
    else
        NTuple{2,Int}[]
    end
    edge_endpoints === nothing && return nothing

    dims = Vector{Int}(undef, nP)
    n0_global = isempty(L.boundaries) ? 0 : size(L.boundaries[1], 1)
    use_lookup = n0_global > 0 && n0_global <= _ACTIVE_INDEX_TABLE_MAX_ROWS[]
    threaded_ok = Threads.nthreads() > 1 && nP >= _LAZY_DIFF_THREADS_MIN_VERTICES[]
    if threaded_ok
        lookup = use_lookup ? [_ActiveIndexLookupScratch(n0_global) for _ in 1:Threads.nthreads()] : nothing
        Threads.@threads :static for i in 1:nP
            lk = use_lookup ? lookup[Threads.threadid()] : nothing
            _, reps = _h0_components_unionfind_vertex(active0[i], active1[i], edge_endpoints; lookup=lk)
            dims[i] = length(reps)
        end
    else
        lookup = use_lookup ? _ActiveIndexLookupScratch(n0_global) : nothing
        @inbounds for i in 1:nP
            _, reps = _h0_components_unionfind_vertex(active0[i], active1[i], edge_endpoints; lookup=lookup)
            dims[i] = length(reps)
        end
    end
    return dims
end

@inline function _is_chain_cover_cache(cc, nP::Int)
    succs = [_succs(cc, u) for u in 1:nP]
    preds = [_preds(cc, v) for v in 1:nP]
    length(succs) == nP || return false
    length(preds) == nP || return false
    @inbounds for u in 1:nP
        su = succs[u]
        pu = preds[u]
        if u == 1
            length(pu) == 0 || return false
        else
            length(pu) == 1 || return false
            pu[1] == (u - 1) || return false
        end
        if u == nP
            length(su) == 0 || return false
        else
            length(su) == 1 || return false
            su[1] == (u + 1) || return false
        end
    end
    return true
end

function _h0_module_chain_sweep(P::AbstractPoset,
                                vertex_births::Vector{Int},
                                edge_births::Vector{Int},
                                edge_endpoints::Vector{NTuple{2,Int}};
                                field::AbstractCoeffField=QQField())
    nP = nvertices(P)
    nv = length(vertex_births)
    length(edge_births) == length(edge_endpoints) || return nothing

    cc = _get_cover_cache(P)
    _is_chain_cover_cache(cc, nP) || return nothing

    # Birth indices are 1-based indices into the axis/chain vertices.
    @inbounds for b in vertex_births
        (1 <= b <= nP) || return nothing
    end
    @inbounds for b in edge_births
        (1 <= b <= nP) || return nothing
    end

    verts_at_birth = [Int[] for _ in 1:nP]
    edges_at_birth = [Int[] for _ in 1:nP]
    @inbounds for v in 1:nv
        push!(verts_at_birth[vertex_births[v]], v)
    end
    @inbounds for e in eachindex(edge_endpoints)
        push!(edges_at_birth[edge_births[e]], e)
    end

    parent = collect(1:nv)
    sz = ones(Int, nv)
    root_min = collect(1:nv)
    active = falses(nv)
    root_candidates = Int[]
    sizehint!(root_candidates, nv)
    root_stamp = zeros(Int, nv)
    stamp = 0

    dims = Vector{Int}(undef, nP)
    reps_by_vertex = Vector{Vector{Int}}(undef, nP)
    chain_row_for_col = Vector{Vector{Int}}(undef, max(0, nP - 1))
    lookup = nv <= _ACTIVE_INDEX_TABLE_MAX_ROWS[] ? _ActiveIndexLookupScratch(nv) : nothing

    prev_reps = Int[]
    @inbounds for p in 1:nP
        for v in verts_at_birth[p]
            active[v] = true
            parent[v] = v
            sz[v] = 1
            root_min[v] = v
            push!(root_candidates, v)
        end
        for e in edges_at_birth[p]
            a, b = edge_endpoints[e]
            # Edge-only fast path assumes edge births are not before endpoint births.
            # Fall back when this invariant does not hold.
            if !(active[a] && active[b])
                return nothing
            end
            a == b && continue
            _uf_union_with_min!(parent, sz, root_min, a, b)
        end

        stamp += 1
        roots = Int[]
        for c in root_candidates
            active[c] || continue
            r = _uf_find!(parent, c)
            if root_stamp[r] != stamp
                root_stamp[r] = stamp
                push!(roots, r)
            end
        end

        reps = Vector{Int}(undef, length(roots))
        for i in eachindex(roots)
            reps[i] = root_min[_uf_find!(parent, roots[i])]
        end
        sort!(reps)
        unique!(reps)
        reps_by_vertex[p] = reps
        dims[p] = length(reps)

        if p > 1
            row_for_col = Vector{Int}(undef, length(prev_reps))
            if lookup === nothing
                for j in eachindex(prev_reps)
                    g = prev_reps[j]
                    rg = _uf_find!(parent, g)
                    gg = root_min[rg]
                    row_for_col[j] = _sorted_pos_or_zero(reps, gg)
                end
            else
                _active_lookup_fill!(lookup, reps)
                for j in eachindex(prev_reps)
                    g = prev_reps[j]
                    rg = _uf_find!(parent, g)
                    gg = root_min[rg]
                    row_for_col[j] = _active_lookup_get(lookup, gg)
                end
            end
            chain_row_for_col[p - 1] = row_for_col
        end
        prev_reps = reps
    end

    K = coeff_type(field)
    preds = [_preds(cc, v) for v in 1:nP]
    succs = [_succs(cc, u) for u in 1:nP]
    pred_slot_of_succ = u -> _pred_slots_of_succ(cc, u)
    maps_from_pred = [Vector{_StructuralInclusionMap{K}}(undef, length(preds[v])) for v in 1:nP]
    maps_to_succ = [Vector{_StructuralInclusionMap{K}}(undef, length(succs[u])) for u in 1:nP]
    @inbounds for u in 1:nP
        su = succs[u]
        outu = maps_to_succ[u]
        for j in eachindex(su)
            v = su[j]
            v == u + 1 || return nothing
            row_for_col = chain_row_for_col[u]
            A = _StructuralInclusionMap{K}(dims[v], dims[u], row_for_col)
            outu[j] = A
            ip = pred_slot_of_succ(u)[j]
            maps_from_pred[v][ip] = A
        end
    end
    store = CoverEdgeMapStore{K,_StructuralInclusionMap{K}}(preds, succs, maps_from_pred, maps_to_succ, cc.nedges)
    return PModule{K}(P, dims, store; field=field)
end

function _simplex_tree_dim01_h0_payload(ST::SimplexTreeMulti{1,T}) where {T}
    _simplex_tree_is_onecritical(ST) || return nothing
    nd = length(ST.dim_offsets) - 1
    nd >= 1 || return nothing
    nd <= 2 || return nothing

    vlo = ST.dim_offsets[1]
    vhi = ST.dim_offsets[2] - 1
    nv = max(0, vhi - vlo + 1)
    vertex_labels = Vector{Int}(undef, nv)
    vertex_grades = Vector{NTuple{1,T}}(undef, nv)
    @inbounds for sid in vlo:vhi
        i = sid - vlo + 1
        lo = ST.simplex_offsets[sid]
        hi = ST.simplex_offsets[sid + 1] - 1
        (hi - lo + 1) == 1 || return nothing
        vertex_labels[i] = ST.simplex_vertices[lo]
        vertex_grades[i] = ST.grade_data[ST.grade_offsets[sid]]
    end
    isempty(vertex_labels) && return (vertex_grades, NTuple{2,Int}[], NTuple{1,T}[])

    max_label = maximum(vertex_labels)
    max_label > 0 || return nothing
    label_to_local = zeros(Int, max_label)
    @inbounds for i in eachindex(vertex_labels)
        lbl = vertex_labels[i]
        (1 <= lbl <= max_label) || return nothing
        label_to_local[lbl] == 0 || return nothing
        label_to_local[lbl] = i
    end

    if nd == 1
        return (vertex_grades, NTuple{2,Int}[], NTuple{1,T}[])
    end

    elo = ST.dim_offsets[2]
    ehi = ST.dim_offsets[3] - 1
    ne = max(0, ehi - elo + 1)
    edge_endpoints = Vector{NTuple{2,Int}}(undef, ne)
    edge_grades = Vector{NTuple{1,T}}(undef, ne)
    @inbounds for sid in elo:ehi
        i = sid - elo + 1
        lo = ST.simplex_offsets[sid]
        hi = ST.simplex_offsets[sid + 1] - 1
        (hi - lo + 1) == 2 || return nothing
        la = ST.simplex_vertices[lo]
        lb = ST.simplex_vertices[lo + 1]
        (1 <= la <= max_label && 1 <= lb <= max_label) || return nothing
        a = label_to_local[la]
        b = label_to_local[lb]
        (a > 0 && b > 0 && a != b) || return nothing
        edge_endpoints[i] = a < b ? (a, b) : (b, a)
        edge_grades[i] = ST.grade_data[ST.grade_offsets[sid]]
    end
    return (vertex_grades, edge_endpoints, edge_grades)
end

function _cohomology_module_h0_chain_sweep_from_simplex_tree(ST::SimplexTreeMulti{1,T},
                                                             P::AbstractPoset,
                                                             axes::NTuple{1,Vector{A}};
                                                             orientation::NTuple{1,Int}=(1,),
                                                             field::AbstractCoeffField=QQField()) where {T,A}
    payload = _simplex_tree_dim01_h0_payload(ST)
    payload === nothing && return nothing
    vertex_grades, edge_endpoints, edge_grades = payload
    vertex_births = _birth_indices(vertex_grades, axes, orientation)
    edge_births = _birth_indices(edge_grades, axes, orientation)
    vertex_births_1 = [b[1] for b in vertex_births]
    edge_births_1 = [b[1] for b in edge_births]
    return _h0_module_chain_sweep(P, vertex_births_1, edge_births_1, edge_endpoints; field=field)
end

function _h0_coboundary_components_from_lazy(L::LazyModuleCochainComplex)
    nP = nvertices(L.P)
    K = coeff_type(L.field)
    active0, _ = _lazy_ensure_active!(L, 1)
    nd = length(L.grades_by_dim)

    if nd >= 2
        _lazy_ensure_active!(L, 2)
        comps = _lazy_diff_components(L, 1; threaded=true)
        return comps
    end

    comps = Vector{SparseMatrixCSC{K,Int}}(undef, nP)
    @inbounds for i in 1:nP
        comps[i] = spzeros(K, 0, length(active0[i]))
    end
    return comps
end

function _kernel_module_from_vertex_maps(M::PModule{K},
                                         comps::Vector{MatT}) where {K,MatT<:AbstractMatrix{K}}
    n = nvertices(M.Q)
    basisK = Vector{Matrix{K}}(undef, n)
    K_dims = zeros(Int, n)
    cc = _get_cover_cache(M.Q)
    succs = [_succs(cc, u) for u in 1:n]
    pred_slot_of_succ = u -> _pred_slots_of_succ(cc, u)
    @inbounds for i in 1:n
        B = FieldLinAlg.nullspace(M.field, comps[i])
        basisK[i] = B
        K_dims[i] = size(B, 2)
    end

    preds = [_preds(cc, v) for v in 1:n]
    maps_from_pred = [Vector{Matrix{K}}(undef, length(preds[v])) for v in 1:n]
    maps_to_succ = [Vector{Matrix{K}}(undef, length(succs[u])) for u in 1:n]
    @inbounds for u in 1:n
        su = succs[u]
        maps_u_M = M.edge_maps.maps_to_succ[u]
        outu = maps_to_succ[u]
        for j in eachindex(su)
            v = su[j]
            X = if K_dims[u] == 0 || K_dims[v] == 0
                zeros(K, K_dims[v], K_dims[u])
            else
                Im = maps_u_M[j] * basisK[u]
                FieldLinAlg.solve_fullcolumn(M.field, basisK[v], Im; check_rhs=false)
            end
            outu[j] = X
            ip = pred_slot_of_succ(u)[j]
            maps_from_pred[v][ip] = X
        end
    end
    storeK = CoverEdgeMapStore{K,Matrix{K}}(preds, succs, maps_from_pred, maps_to_succ, cc.nedges)
    return PModule{K}(M.Q, K_dims, storeK; field=M.field)
end

function _cohomology_module_h0_lowdim_from_lazy(L::LazyModuleCochainComplex)
    dom0 = _lazy_term_idx!(L, 1)
    comps = _h0_coboundary_components_from_lazy(L)
    return _kernel_module_from_vertex_maps(dom0, comps)
end

function _cohomology_module_h1_cokernel_from_lazy(L::LazyModuleCochainComplex)
    M1 = _lazy_term(L, 1)
    d0 = _lazy_diff(L, 0)
    B, iB = image_with_inclusion(d0)
    H, _ = _cokernel_module(iB)
    return H
end

function _identity_structural_morphism(M::PModule{K}) where {K}
    n = nvertices(M.Q)
    comps = Vector{_StructuralInclusionMap{K}}(undef, n)
    @inbounds for i in 1:n
        d = M.dims[i]
        row_for_col = collect(1:d)
        comps[i] = _StructuralInclusionMap{K}(d, d, row_for_col)
    end
    return PMorphism{K}(M, M, comps)
end

@inline function _solve_through_structural_inclusion(A::_StructuralInclusionMap{K},
                                                     Y::AbstractMatrix{K}) where {K}
    nr, nc = size(A)
    size(Y, 1) == nr || throw(DimensionMismatch("size mismatch in structural-inclusion solve"))
    X = zeros(K, nc, size(Y, 2))
    @inbounds for j in 1:nc
        i = A.row_for_col[j]
        i == 0 && continue
        @views X[j, :] .= Y[i, :]
    end
    return X
end

function _cohomology_module_degree_local_from_lazy(L::LazyModuleCochainComplex, t::Int)
    K = coeff_type(L.field)
    M = _lazy_term(L, t)
    d0 = _lazy_diff(L, t - 1)
    d1 = _lazy_diff(L, t)

    Z, iZ = if is_zero_morphism(d1)
        M, _identity_structural_morphism(M)
    else
        kernel_with_inclusion(d1)
    end

    if is_zero_morphism(d0)
        return Z
    end

    B, iB = image_with_inclusion(d0)
    if all(==(0), B.dims)
        return Z
    end

    n = nvertices(M.Q)
    jcomps = Vector{Matrix{K}}(undef, n)
    field = M.field
    @inbounds for u in 1:n
        bd = B.dims[u]
        zd = Z.dims[u]
        if bd == 0
            jcomps[u] = zeros(K, zd, 0)
        elseif zd == 0
            jcomps[u] = zeros(K, 0, bd)
        else
            A = iZ.comps[u]
            Y = iB.comps[u]
            jcomps[u] = if A isa _StructuralInclusionMap{K}
                _solve_through_structural_inclusion(A, Y)
            else
                FieldLinAlg.solve_fullcolumn(field, A, Y; check_rhs=false)
            end
        end
    end
    j = PMorphism{K}(B, Z, jcomps)
    H, _ = _cokernel_module(j)
    return H
end

function _cohomology_module_from_lazy_generic(L::LazyModuleCochainComplex, t::Int)
    K = coeff_type(L.field)
    terms = Vector{PModule{K}}(undef, 3)
    terms[1] = _lazy_term(L, t - 1)
    terms[2] = _lazy_term(L, t)
    terms[3] = _lazy_term(L, t + 1)
    diffs = Vector{PMorphism{K}}(undef, 2)
    diffs[1] = _lazy_diff(L, t - 1)
    diffs[2] = _lazy_diff(L, t)
    C_local = ModuleCochainComplex(terms, diffs; tmin=t - 1, check=false)
    try
        return cohomology_module(C_local, t)
    catch
        # Some exact-field module kernels/cokernels remain sensitive to the
        # reduced local presentation. Fall back to the fully materialized
        # cochain complex so the lazy encoding_result path preserves the eager
        # result contract instead of surfacing an internal solver failure.
        C_full = _materialize_cochain(L; check=false)
        return cohomology_module(C_full, t)
    end
end

function _boundary_rows_in_field(B::SparseMatrixCSC{Int,Int},
                                 field::AbstractCoeffField)
    K = coeff_type(field)
    m = size(B, 1)
    row_cols = [Int[] for _ in 1:m]
    row_vals = [K[] for _ in 1:m]
    @inbounds for col in 1:size(B, 2)
        for ptr in B.colptr[col]:(B.colptr[col + 1] - 1)
            r = B.rowval[ptr]
            push!(row_cols[r], col)
            push!(row_vals[r], coerce(field, B.nzval[ptr]))
        end
    end
    return row_cols, row_vals
end

@inline function _rref_push_restricted_row!(R,
                                            row_cols::Vector{Int},
                                            row_vals,
                                            colpos::Vector{Int},
                                            scratch_idx::Vector{Int},
                                            scratch_val)
    empty!(scratch_idx)
    empty!(scratch_val)
    @inbounds for t in eachindex(row_cols)
        lp = colpos[row_cols[t]]
        lp == 0 && continue
        v = row_vals[t]
        if !isempty(scratch_idx) && scratch_idx[end] == lp
            nv = scratch_val[end] + v
            if iszero(nv)
                pop!(scratch_idx)
                pop!(scratch_val)
            else
                scratch_val[end] = nv
            end
        elseif !iszero(v)
            push!(scratch_idx, lp)
            push!(scratch_val, v)
        end
    end
    isempty(scratch_idx) && return false
    K = eltype(scratch_val)
    return FieldLinAlg._sparse_rref_push_homogeneous!(
        R,
        FieldLinAlg.SparseRow{K}(copy(scratch_idx), copy(scratch_val)),
    )
end

@inline _monotone_inc_bucket_value(n::Int) = n <= 1 ? 1 : (1 << floor(Int, log2(Float64(n))))

@inline function _monotone_inc_field_code(field::AbstractCoeffField)
    field isa QQField && return 1
    field isa PrimeField && field.p == 2 && return 2
    field isa PrimeField && field.p == 3 && return 3
    field isa PrimeField && field.p > 3 && return 4
    field isa RealField && return 5
    return 0
end

@inline function _monotone_inc_cache_key(nP::Int, m::Int, n::Int, field::AbstractCoeffField)
    ratio = Float64(min(m, n)) / Float64(max(1, max(m, n)))
    ratio_bucket = clamp(floor(Int, ratio * 20.0), 0, 20)
    return (_monotone_inc_bucket_value(nP),
            _monotone_inc_bucket_value(m),
            _monotone_inc_bucket_value(n),
            ratio_bucket,
            _monotone_inc_field_code(field))
end

@inline function _monotone_inc_mode()
    mode = _COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_MODE[]
    (mode == :auto || mode == :on || mode == :off) ||
        throw(ArgumentError("monotone incremental rank mode must be :auto, :on, or :off (got $(mode))."))
    return mode
end

function _monotone_inc_cache_lookup(key::NTuple{5,Int})
    lock(_COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_CACHE_LOCK) do
        return get(_COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_CACHE, key, nothing)
    end
end

function _monotone_inc_cache_store!(key::NTuple{5,Int}, prefer_incremental::Bool)
    lock(_COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_CACHE_LOCK) do
        if !haskey(_COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_CACHE, key)
            push!(_COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_CACHE_ORDER, key)
        end
        _COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_CACHE[key] = prefer_incremental
        max_entries = max(1, _COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_CACHE_MAX[])
        while length(_COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_CACHE_ORDER) > max_entries
            old = popfirst!(_COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_CACHE_ORDER)
            delete!(_COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_CACHE, old)
        end
    end
    return prefer_incremental
end

function _rank_dims_from_lazy_diff_monotone_chain_incremental(L::LazyModuleCochainComplex,
                                                              k::Int,
                                                              row_births::Vector{Int},
                                                              col_births::Vector{Int};
                                                              nlimit::Int=nvertices(L.P))
    B = L.boundaries[k]
    nP_full = nvertices(L.P)
    nP = min(max(0, nlimit), nP_full)
    nP == 0 && return Int[]
    m = size(B, 1)
    n = size(B, 2)
    (length(row_births) == m && length(col_births) == n) || return nothing
    (nP_full >= 128 && (m * n) >= 50_000) || return nothing

    K = coeff_type(L.field)
    K <: AbstractFloat && return nothing

    rows_by_birth = [Int[] for _ in 1:nP]
    cols_by_birth = [Int[] for _ in 1:nP]
    @inbounds for r in 1:m
        b = row_births[r]
        if !(1 <= b <= nP_full)
            return nothing
        elseif b <= nP
            push!(rows_by_birth[b], r)
        end
    end
    @inbounds for c in 1:n
        b = col_births[c]
        if !(1 <= b <= nP_full)
            return nothing
        elseif b <= nP
            push!(cols_by_birth[b], c)
        end
    end

    row_cols, row_vals = _boundary_rows_in_field(B, L.field)
    ranks = zeros(Int, nP)
    active_rows = Int[]
    active_cols = Int[]
    sizehint!(active_rows, m)
    sizehint!(active_cols, n)
    colpos = zeros(Int, n)
    touched_cols = Int[]
    sizehint!(touched_cols, n)
    scratch_idx = Int[]
    scratch_val = K[]
    tracker = nothing

    try
        @inbounds for p in 1:nP
            new_rows = rows_by_birth[p]
            new_cols = cols_by_birth[p]
            isempty(new_rows) || append!(active_rows, new_rows)
            cols_changed = !isempty(new_cols)
            if cols_changed
                append!(active_cols, new_cols)
                for c in touched_cols
                    colpos[c] = 0
                end
                empty!(touched_cols)
                for (j, c) in enumerate(active_cols)
                    colpos[c] = j
                    push!(touched_cols, c)
                end
                tracker = FieldLinAlg._SparseRREF{K}(length(active_cols))
                for r in active_rows
                    _rref_push_restricted_row!(tracker, row_cols[r], row_vals[r], colpos, scratch_idx, scratch_val)
                end
            elseif tracker !== nothing && !isempty(new_rows)
                for r in new_rows
                    _rref_push_restricted_row!(tracker, row_cols[r], row_vals[r], colpos, scratch_idx, scratch_val)
                end
            elseif tracker === nothing && !isempty(active_cols)
                for (j, c) in enumerate(active_cols)
                    colpos[c] = j
                    push!(touched_cols, c)
                end
                tracker = FieldLinAlg._SparseRREF{K}(length(active_cols))
                for r in active_rows
                    _rref_push_restricted_row!(tracker, row_cols[r], row_vals[r], colpos, scratch_idx, scratch_val)
                end
            end
            ranks[p] = tracker === nothing ? 0 : FieldLinAlg._rref_rank(tracker)
        end
    catch
        return nothing
    end
    return ranks
end

function _rank_dims_from_lazy_diff_monotone_chain_baseline(L::LazyModuleCochainComplex,
                                                            k::Int,
                                                            row_births::Vector{Int},
                                                            col_births::Vector{Int};
                                                            nlimit::Int=nvertices(L.P))
    B = L.boundaries[k]
    nP_full = nvertices(L.P)
    nP = min(max(0, nlimit), nP_full)
    nP == 0 && return Int[]
    active_rows = L.active_by_dim[k]
    active_cols = L.active_by_dim[k + 1]
    Bf = _COHOMOLOGY_DIMS_USE_DIRECT_RESTRICTED_RANK[] ? _lazy_boundary_field!(L, k) : nothing

    rows_added = zeros(Int, nP)
    cols_added = zeros(Int, nP)
    @inbounds for b in row_births
        if !(1 <= b <= nP_full)
            return nothing
        elseif b <= nP
            rows_added[b] += 1
        end
    end
    @inbounds for b in col_births
        if !(1 <= b <= nP_full)
            return nothing
        elseif b <= nP
            cols_added[b] += 1
        end
    end

    ranks = Vector{Int}(undef, nP)
    prev_rank = 0
    prev_rows = 0
    prev_cols = 0
    use_lookup = size(B, 1) <= _ACTIVE_INDEX_TABLE_MAX_ROWS[]
    lookup = use_lookup ? _ActiveIndexLookupScratch(size(B, 1)) : nothing

    @inbounds for p in 1:nP
        rows = length(active_rows[p])
        cols = length(active_cols[p])

        if rows == 0 || cols == 0
            ranks[p] = 0
            prev_rank = 0
            prev_rows = rows
            prev_cols = cols
            continue
        end

        if p > 1 && rows_added[p] == 0 && cols_added[p] == 0
            ranks[p] = prev_rank
            prev_rows = rows
            prev_cols = cols
            continue
        end

        curr_min = min(rows, cols)
        prev_min = min(prev_rows, prev_cols)
        if p > 1 && prev_rank == prev_min && curr_min == prev_rank
            ranks[p] = prev_rank
            prev_rows = rows
            prev_cols = cols
            continue
        end

        r = if _COHOMOLOGY_DIMS_USE_DIRECT_RESTRICTED_RANK[] && Bf !== nothing
            FieldLinAlg.rank_restricted(L.field, Bf, active_rows[p], active_cols[p]; check=false)
        else
            comp = if use_lookup
                _restricted_coboundary_component(B, active_rows[p], active_cols[p], L.field, lookup)
            else
                _restricted_coboundary_component(B, active_rows[p], active_cols[p], L.field)
            end
            FieldLinAlg.rank_dim(L.field, comp)
        end
        ranks[p] = r
        prev_rank = r
        prev_rows = rows
        prev_cols = cols
    end
    return ranks
end

function _should_use_monotone_incremental_rank(L::LazyModuleCochainComplex,
                                               k::Int,
                                               row_births::Vector{Int},
                                               col_births::Vector{Int})
    _COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_RANK[] || return false
    mode = _monotone_inc_mode()
    mode == :off && return false
    mode == :on && return true

    B = L.boundaries[k]
    nP = nvertices(L.P)
    m = size(B, 1)
    n = size(B, 2)
    K = coeff_type(L.field)
    K <: AbstractFloat && return false

    heuristic = nP >= 384 && m >= 384 && n >= 384 && (m * n) >= 250_000
    key = _monotone_inc_cache_key(nP, m, n, L.field)
    if _COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_CACHE_ENABLED[]
        cached = _monotone_inc_cache_lookup(key)
        cached === nothing || return cached
    end

    if !heuristic || !_COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_PROBE[]
        return _COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_CACHE_ENABLED[] ?
               _monotone_inc_cache_store!(key, heuristic) : heuristic
    end

    nprobe = min(nP, 96)
    inc_ok = false
    tinc = @elapsed begin
        r = _rank_dims_from_lazy_diff_monotone_chain_incremental(L, k, row_births, col_births; nlimit=nprobe)
        inc_ok = (r !== nothing)
    end
    tbase = @elapsed begin
        _rank_dims_from_lazy_diff_monotone_chain_baseline(L, k, row_births, col_births; nlimit=nprobe)
    end
    prefer = inc_ok && (tinc <= 0.92 * tbase)
    return _COHOMOLOGY_DIMS_MONOTONE_INCREMENTAL_CACHE_ENABLED[] ?
           _monotone_inc_cache_store!(key, prefer) : prefer
end

function _rank_dims_from_lazy_diff_monotone_chain(L::LazyModuleCochainComplex, k::Int)
    nP = nvertices(L.P)
    _is_chain_cover_cache(_get_cover_cache(L.P), nP) || return nothing

    active_rows = L.active_by_dim[k]
    active_cols = L.active_by_dim[k + 1]
    B = L.boundaries[k]
    row_births = _births_from_prefix_active_sets(active_rows, size(B, 1))
    col_births = _births_from_prefix_active_sets(active_cols, size(B, 2))
    (row_births === nothing || col_births === nothing) && return nothing

    if _should_use_monotone_incremental_rank(L, k, row_births, col_births)
        ranks_inc = _rank_dims_from_lazy_diff_monotone_chain_incremental(L, k, row_births, col_births; nlimit=nP)
        ranks_inc === nothing || return ranks_inc
    end
    return _rank_dims_from_lazy_diff_monotone_chain_baseline(L, k, row_births, col_births; nlimit=nP)
end

function _rank_dims_from_lazy_diff_direct(L::LazyModuleCochainComplex, k::Int)
    B = _lazy_boundary_field!(L, k)
    B isa SparseMatrixCSC || return nothing
    active_rows = L.active_by_dim[k]
    active_cols = L.active_by_dim[k + 1]
    nP = nvertices(L.P)
    ranks = zeros(Int, nP)
    @inbounds for i in 1:nP
        rows = active_rows[i]
        cols = active_cols[i]
        if isempty(rows) || isempty(cols)
            ranks[i] = 0
        else
            ranks[i] = FieldLinAlg.rank_restricted(L.field, B, rows, cols; check=false)
        end
    end
    return ranks
end

function _cohomology_dims_from_lazy(L::LazyModuleCochainComplex, t::Int)
    nP = nvertices(L.P)
    if t < L.tmin || t > L.tmax
        return zeros(Int, nP)
    end
    if t == 0 && L.tmin == 0
        active0, _ = _lazy_ensure_active!(L, 1)
        active1 = if length(L.grades_by_dim) >= 2
            _lazy_ensure_active!(L, 2)[1]
        else
            [Int[] for _ in 1:nP]
        end
        # For dims-only H0 we can always try union-find first; this avoids
        # expensive per-vertex rank work on common d=1 ingestion paths.
        duf = _cohomology_dims_h0_unionfind_from_lazy(L, active0, active1)
        duf !== nothing && return duf
    end
    idx = t - L.tmin + 1
    active_t, _ = _lazy_ensure_active!(L, idx)
    dims = Vector{Int}(undef, nP)
    @inbounds for i in 1:nP
        dims[i] = length(active_t[i])
    end

    rank_dt = zeros(Int, nP)
    if t < L.tmax
        _lazy_ensure_active!(L, idx + 1)
        ranks_fast = _COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[] ?
            _rank_dims_from_lazy_diff_monotone_chain(L, idx) : nothing
        if ranks_fast === nothing
            ranks_direct = _COHOMOLOGY_DIMS_USE_DIRECT_RESTRICTED_RANK[] ?
                _rank_dims_from_lazy_diff_direct(L, idx) : nothing
            if ranks_direct === nothing
                comps_t = _lazy_diff_components(L, idx; threaded=true)
                @inbounds for i in 1:nP
                    rank_dt[i] = FieldLinAlg.rank_dim(L.field, comps_t[i])
                end
            else
                rank_dt .= ranks_direct
            end
        else
            rank_dt .= ranks_fast
        end
    end

    rank_dprev = zeros(Int, nP)
    if t > L.tmin
        _lazy_ensure_active!(L, idx - 1)
        ranks_fast = _COHOMOLOGY_DIMS_MONOTONE_RANK_FASTPATH[] ?
            _rank_dims_from_lazy_diff_monotone_chain(L, idx - 1) : nothing
        if ranks_fast === nothing
            ranks_direct = _COHOMOLOGY_DIMS_USE_DIRECT_RESTRICTED_RANK[] ?
                _rank_dims_from_lazy_diff_direct(L, idx - 1) : nothing
            if ranks_direct === nothing
                comps_prev = _lazy_diff_components(L, idx - 1; threaded=true)
                @inbounds for i in 1:nP
                    rank_dprev[i] = FieldLinAlg.rank_dim(L.field, comps_prev[i])
                end
            else
                rank_dprev .= ranks_direct
            end
        else
            rank_dprev .= ranks_fast
        end
    end

    out = Vector{Int}(undef, nP)
    @inbounds for i in 1:nP
        out[i] = dims[i] - rank_dt[i] - rank_dprev[i]
    end
    return out
end

function _cohomology_module_from_lazy(L::LazyModuleCochainComplex, t::Int)
    if t == 0 && L.tmin == 0
        active0, _ = _lazy_ensure_active!(L, 1)
        active1 = if length(L.grades_by_dim) >= 2
            _lazy_ensure_active!(L, 2)[1]
        else
            [Int[] for _ in 1:nvertices(L.P)]
        end
        # H0 maps are induced by component merges, not by restricting the
        # kernel of the vertex coboundary under structural inclusion.  The
        # union-find path computes those merge maps directly and is therefore
        # the canonical lazy H0 route whenever the dim-1 boundary looks like a
        # graph incidence matrix.
        Muf = _cohomology_module_h0_unionfind_from_lazy(L, active0, active1)
        if Muf !== nothing
            return Muf
        end
        return _cohomology_module_h0_lowdim_from_lazy(L)
    end
    if t == 1 && _H1_COKERNEL_FASTPATH[] && L.tmin == 0 && L.tmax <= 1
        try
            return _cohomology_module_h1_cokernel_from_lazy(L)
        catch
            # Some sparse-restricted module maps can violate strict image solver
            # checks on this shortcut path; fall back to the canonical generic
            # cohomology route in that case.
        end
    end
    if _use_degree_local_module_fastpath(L, t)
        try
            return _cohomology_module_degree_local_from_lazy(L, t)
        catch
            # Keep the generic path as the canonical fallback for unusual
            # complexes where the degree-local shortcut assumptions do not hold.
        end
    end
    return _cohomology_module_from_lazy_generic(L, t)
end

function _materialize_cochain(L::LazyModuleCochainComplex; check::Bool=true)
    nd = length(L.terms)
    K = coeff_type(L.field)
    terms = Vector{PModule{K}}(undef, nd)
    for i in 1:nd
        terms[i] = _lazy_term_idx!(L, i)
    end
    diffs = Vector{PMorphism{K}}(undef, max(0, nd - 1))
    for k in eachindex(diffs)
        diffs[k] = _lazy_diff_idx!(L, k)
    end
    return ModuleCochainComplex(terms, diffs; tmin=L.tmin, check=check)
end

function euler_characteristic_surface(L::LazyModuleCochainComplex,
                                      pi::AbstractPLikeEncodingMap,
                                      opts::InvariantOptions;
                                      cache=nothing)
    return euler_characteristic_surface(_euler_dims_from_lazy(L), pi, opts; cache=cache)
end

euler_surface(L::LazyModuleCochainComplex,
              pi::AbstractPLikeEncodingMap,
              opts::InvariantOptions;
              kwargs...) =
    euler_characteristic_surface(L, pi, opts; kwargs...)

function euler_signed_measure(L::LazyModuleCochainComplex,
                              pi::AbstractPLikeEncodingMap,
                              opts::InvariantOptions;
                              kwargs...)
    return euler_signed_measure(_euler_dims_from_lazy(L), pi, opts; kwargs...)
end

function euler_signed_measure(L::LazyModuleCochainComplex,
                              pi::GridEncodingMap{1},
                              opts::InvariantOptions;
                              drop_zeros::Bool=true,
                              max_terms::Int=0,
                              min_abs_weight::Real=0,
                              cache=nothing)
    _ = cache
    if _can_use_direct_euler_signed_measure_chain_1d(L, pi, opts)
        return _euler_signed_measure_from_lazy_chain_1d(L, pi.coords[1];
            drop_zeros=drop_zeros,
            max_terms=max_terms,
            min_abs_weight=min_abs_weight)
    end
    return euler_signed_measure(_euler_dims_from_lazy(L), pi, opts;
        drop_zeros=drop_zeros,
        max_terms=max_terms,
        min_abs_weight=min_abs_weight,
        cache=cache)
end

function euler_signed_measure(L::LazyModuleCochainComplex,
                              pi::CompiledEncoding{<:GridEncodingMap{1}},
                              opts::InvariantOptions;
                              drop_zeros::Bool=true,
                              max_terms::Int=0,
                              min_abs_weight::Real=0,
                              cache=nothing)
    raw_pi = pi.pi
    if _can_use_direct_euler_signed_measure_chain_1d(L, raw_pi, opts)
        ax = axes_from_encoding(pi)
        ax === nothing || length(ax) == 1 ||
            error("euler_signed_measure: expected one encoding axis for 1D direct path.")
        if ax !== nothing
            return _euler_signed_measure_from_lazy_chain_1d(L, ax[1];
                drop_zeros=drop_zeros,
                max_terms=max_terms,
                min_abs_weight=min_abs_weight)
        end
    end
    return euler_signed_measure(_euler_dims_from_lazy(L), pi, opts;
        drop_zeros=drop_zeros,
        max_terms=max_terms,
        min_abs_weight=min_abs_weight,
        cache=cache)
end

function euler_signed_measure(L::LazyModuleCochainComplex,
                              pi::GridEncodingMap{2},
                              opts::InvariantOptions;
                              drop_zeros::Bool=true,
                              max_terms::Int=0,
                              min_abs_weight::Real=0,
                              cache=nothing)
    _ = cache
    if _can_use_direct_euler_signed_measure_grid_2d(L, pi, opts)
        return _euler_signed_measure_from_lazy_grid_2d(L, pi.coords;
            drop_zeros=drop_zeros,
            max_terms=max_terms,
            min_abs_weight=min_abs_weight)
    end
    return euler_signed_measure(_euler_dims_from_lazy(L), pi, opts;
        drop_zeros=drop_zeros,
        max_terms=max_terms,
        min_abs_weight=min_abs_weight,
        cache=cache)
end

function euler_signed_measure(L::LazyModuleCochainComplex,
                              pi::CompiledEncoding{<:GridEncodingMap{2}},
                              opts::InvariantOptions;
                              drop_zeros::Bool=true,
                              max_terms::Int=0,
                              min_abs_weight::Real=0,
                              cache=nothing)
    raw_pi = pi.pi
    if _can_use_direct_euler_signed_measure_grid_2d(L, raw_pi, opts)
        ax = axes_from_encoding(pi)
        ax === nothing || length(ax) == 2 ||
            error("euler_signed_measure: expected two encoding axes for 2D direct path.")
        if ax !== nothing
            return _euler_signed_measure_from_lazy_grid_2d(L, ax;
                drop_zeros=drop_zeros,
                max_terms=max_terms,
                min_abs_weight=min_abs_weight)
        end
    end
    return euler_signed_measure(_euler_dims_from_lazy(L), pi, opts;
        drop_zeros=drop_zeros,
        max_terms=max_terms,
        min_abs_weight=min_abs_weight,
        cache=cache)
end

function _cochain_complex_from_grades_and_boundaries(grades_by_dim::Vector,
                                                     boundaries_in::Vector{SparseMatrixCSC{Int,Int}},
                                                     P::AbstractPoset,
                                                     axes::NTuple{N,Vector{T}};
                                                     orientation::NTuple{N,Int}=ntuple(_ -> 1, N),
                                                     field::AbstractCoeffField=QQField(),
                                                     multicritical::Symbol=:union) where {N,T}
    sizes = ntuple(i -> length(axes[i]), N)
    vertex_idxs = _grid_tuples(sizes)

    active_by_dim = Vector{Vector{Vector{Int}}}(undef, length(grades_by_dim))
    for d in 1:length(grades_by_dim)
        births_d = _birth_indices(grades_by_dim[d], axes, orientation)
        active_by_dim[d] = _active_lists(births_d, vertex_idxs, orientation; multicritical=multicritical)
    end

    K = coeff_type(field)
    terms = Vector{PModule{K}}(undef, length(grades_by_dim))
    for d in 1:length(grades_by_dim)
        terms[d] = _pmodule_from_active_lists(P, active_by_dim[d]; field=field)
    end

    diffs = PMorphism{K}[]
    boundaries = _normalize_cochain_boundaries(grades_by_dim, boundaries_in)

    for k in 1:length(boundaries)
        B = boundaries[k]  # boundary C_{k+1} -> C_k
        comps = Vector{SparseMatrixCSC{K,Int}}(undef, nvertices(P))
        for i in 1:nvertices(P)
            Lk = active_by_dim[k][i]
            Lk1 = active_by_dim[k + 1][i]
            comps[i] = _restricted_coboundary_component(B, Lk, Lk1, field)
        end
        push!(diffs, PMorphism{K}(terms[k], terms[k + 1], comps))
    end

    return ModuleCochainComplex(terms, diffs; tmin=0, check=true)
end

function cochain_complex_from_graded_complex(G::Union{GradedComplex,MultiCriticalGradedComplex},
                                             P::AbstractPoset,
                                             axes::NTuple{N,Vector{T}};
                                             orientation::NTuple{N,Int}=ntuple(_ -> 1, N),
                                             field::AbstractCoeffField=QQField(),
                                             multicritical::Symbol=:union,
                                             onecritical_selector::Symbol=:lexmin,
                                             onecritical_enforce_boundary::Bool=true) where {N,T}
    Guse = if G isa MultiCriticalGradedComplex && multicritical === :one_critical
        one_criticalify(G; selector=onecritical_selector, enforce_boundary=onecritical_enforce_boundary)
    else
        G
    end
    if Guse isa MultiCriticalGradedComplex
        multicritical in (:union, :intersection, :one_critical) ||
            throw(ArgumentError("cochain_complex_from_graded_complex: multicritical must be :union, :intersection, or :one_critical"))
    end
    mc_mode = multicritical === :one_critical ? :union : multicritical
    grades_by_dim = _grades_by_dim(Guse)
    return _cochain_complex_from_grades_and_boundaries(
        grades_by_dim,
        Guse.boundaries,
        P,
        axes;
        orientation=orientation,
        field=field,
        multicritical=mc_mode,
    )
end

function _lazy_cochain_complex_from_graded_complex(G::Union{GradedComplex,MultiCriticalGradedComplex},
                                                   P::AbstractPoset,
                                                   axes::NTuple{N,Vector{T}};
                                                   orientation::NTuple{N,Int}=ntuple(_ -> 1, N),
                                                   field::AbstractCoeffField=QQField(),
                                                   multicritical::Symbol=:union,
                                                   onecritical_selector::Symbol=:lexmin,
                                                   onecritical_enforce_boundary::Bool=true) where {N,T}
    Guse = if G isa MultiCriticalGradedComplex && multicritical === :one_critical
        one_criticalify(G; selector=onecritical_selector, enforce_boundary=onecritical_enforce_boundary)
    else
        G
    end
    if Guse isa MultiCriticalGradedComplex
        multicritical in (:union, :intersection, :one_critical) ||
            throw(ArgumentError("_lazy_cochain_complex_from_graded_complex: multicritical must be :union, :intersection, or :one_critical"))
    end
    mc_mode = multicritical === :one_critical ? :union : multicritical
    grades_by_dim = _grades_by_dim(Guse)
    return _lazy_cochain_complex_from_grades_and_boundaries(
        grades_by_dim,
        Guse.boundaries,
        P,
        axes;
        orientation=orientation,
        field=field,
        multicritical=mc_mode,
    )
end

function _simplex_tree_grades_by_dim(ST::SimplexTreeMulti{N,T}) where {N,T}
    nd = length(ST.dim_offsets) - 1
    if _simplex_tree_is_onecritical(ST)
        out = Vector{Vector{NTuple{N,T}}}(undef, nd)
        for d in 1:nd
            lo = ST.dim_offsets[d]
            hi = ST.dim_offsets[d + 1] - 1
            g = Vector{NTuple{N,T}}(undef, max(0, hi - lo + 1))
            p = 1
            for sid in lo:hi
                g[p] = ST.grade_data[ST.grade_offsets[sid]]
                p += 1
            end
            out[d] = g
        end
        return out
    end
    out = Vector{Vector{Vector{NTuple{N,T}}}}(undef, nd)
    for d in 1:nd
        lo = ST.dim_offsets[d]
        hi = ST.dim_offsets[d + 1] - 1
        g = Vector{Vector{NTuple{N,T}}}(undef, max(0, hi - lo + 1))
        p = 1
        for sid in lo:hi
            s = ST.grade_offsets[sid]
            e = ST.grade_offsets[sid + 1] - 1
            g[p] = unique(Vector{NTuple{N,T}}(ST.grade_data[s:e]))
            p += 1
        end
        out[d] = g
    end
    return out
end

function _simplex_tree_boundaries(ST::SimplexTreeMulti)
    nd = length(ST.dim_offsets) - 1
    nd <= 1 && return SparseMatrixCSC{Int,Int}[]
    simplices_by_dim = Vector{Vector{Vector{Int}}}(undef, nd)
    for d in 1:nd
        simplices_by_dim[d] = _simplex_tree_dim_simplices(ST, d)
    end
    boundaries = Vector{SparseMatrixCSC{Int,Int}}(undef, nd - 1)
    for d in 1:(nd - 1)
        boundaries[d] = _simplicial_boundary(simplices_by_dim[d + 1], simplices_by_dim[d])
    end
    return boundaries
end

function _one_critical_grades_by_dim(grades_by_dim::Vector{Vector{Vector{NTuple{N,T}}}},
                                     boundaries::Vector{SparseMatrixCSC{Int,Int}};
                                     selector::Symbol=:lexmin,
                                     enforce_boundary::Bool=true) where {N,T}
    nd = length(grades_by_dim)
    out = Vector{Vector{NTuple{N,T}}}(undef, nd)
    @inbounds for d in 1:nd
        gd = grades_by_dim[d]
        od = Vector{NTuple{N,T}}(undef, length(gd))
        for i in eachindex(gd)
            od[i] = _select_one_critical_grade(gd[i], selector)
        end
        out[d] = od
    end
    if enforce_boundary
        @inbounds for d in 2:nd
            B = boundaries[d - 1]
            nrows, ncols = size(B)
            nrows == length(out[d - 1]) || error("one_criticalify(simplex-tree): boundary $(d-1) row count mismatch.")
            ncols == length(out[d]) || error("one_criticalify(simplex-tree): boundary $(d-1) col count mismatch.")
            for j in 1:ncols
                g = out[d][j]
                for p in nzrange(B, j)
                    i = rowvals(B)[p]
                    g = _tuple_componentwise_max(g, out[d - 1][i])
                end
                out[d][j] = g
            end
        end
    end
    return out
end

function cochain_complex_from_simplex_tree(ST::SimplexTreeMulti{N,T},
                                           P::AbstractPoset,
                                           axes::NTuple{N,Vector{A}};
                                           orientation::NTuple{N,Int}=ntuple(_ -> 1, N),
                                           field::AbstractCoeffField=QQField(),
                                           multicritical::Symbol=:union,
                                           onecritical_selector::Symbol=:lexmin,
                                           onecritical_enforce_boundary::Bool=true) where {N,T,A}
    if !_simplex_tree_is_onecritical(ST)
        multicritical in (:union, :intersection, :one_critical) ||
            throw(ArgumentError("cochain_complex_from_simplex_tree: multicritical must be :union, :intersection, or :one_critical"))
    end
    boundaries = _simplex_tree_boundaries(ST)
    if multicritical === :one_critical
        grades_by_dim = if _simplex_tree_is_onecritical(ST)
            _simplex_tree_grades_by_dim(ST)
        else
            _one_critical_grades_by_dim(
                _simplex_tree_grades_by_dim(ST),
                boundaries;
                selector=onecritical_selector,
                enforce_boundary=onecritical_enforce_boundary,
            )
        end
        return _cochain_complex_from_grades_and_boundaries(
            grades_by_dim,
            boundaries,
            P,
            axes;
            orientation=orientation,
            field=field,
            multicritical=:union,
        )
    end
    grades_by_dim = _simplex_tree_grades_by_dim(ST)
    return _cochain_complex_from_grades_and_boundaries(
        grades_by_dim,
        boundaries,
        P,
        axes;
        orientation=orientation,
        field=field,
        multicritical=multicritical,
    )
end

function _lazy_cochain_complex_from_simplex_tree(ST::SimplexTreeMulti{N,T},
                                                 P::AbstractPoset,
                                                 axes::NTuple{N,Vector{A}};
                                                 orientation::NTuple{N,Int}=ntuple(_ -> 1, N),
                                                 field::AbstractCoeffField=QQField(),
                                                 multicritical::Symbol=:union,
                                                 onecritical_selector::Symbol=:lexmin,
                                                 onecritical_enforce_boundary::Bool=true) where {N,T,A}
    if !_simplex_tree_is_onecritical(ST)
        multicritical in (:union, :intersection, :one_critical) ||
            throw(ArgumentError("_lazy_cochain_complex_from_simplex_tree: multicritical must be :union, :intersection, or :one_critical"))
    end
    boundaries = _simplex_tree_boundaries(ST)
    if multicritical === :one_critical
        grades_by_dim = if _simplex_tree_is_onecritical(ST)
            _simplex_tree_grades_by_dim(ST)
        else
            _one_critical_grades_by_dim(
                _simplex_tree_grades_by_dim(ST),
                boundaries;
                selector=onecritical_selector,
                enforce_boundary=onecritical_enforce_boundary,
            )
        end
        return _lazy_cochain_complex_from_grades_and_boundaries(
            grades_by_dim,
            boundaries,
            P,
            axes;
            orientation=orientation,
            field=field,
            multicritical=:union,
        )
    end
    grades_by_dim = _simplex_tree_grades_by_dim(ST)
    return _lazy_cochain_complex_from_grades_and_boundaries(
        grades_by_dim,
        boundaries,
        P,
        axes;
        orientation=orientation,
        field=field,
        multicritical=multicritical,
    )
end

function _simplicial_boundary_hash(simplices::Vector{Vector{Int}}, faces::Vector{Vector{Int}})
    K = isempty(simplices) ? (isempty(faces) ? 0 : length(faces[1]) + 1) : length(simplices[1])
    K == 0 && return spzeros(Int, length(faces), length(simplices))
    face_index = Dict{Tuple{Vararg{Int}},Int}()
    for (i, f) in enumerate(faces)
        face_index[Tuple(f)] = i
    end
    I = Int[]
    J = Int[]
    V = Int[]
    sizehint!(I, K * length(simplices))
    sizehint!(J, K * length(simplices))
    sizehint!(V, K * length(simplices))
    for (j, s) in enumerate(simplices)
        for i in 1:K
            f = [s[t] for t in 1:K if t != i]
            row = face_index[Tuple(f)]
            push!(I, row)
            push!(J, j)
            push!(V, isodd(i) ? 1 : -1)
        end
    end
    return sparse(I, J, V, length(faces), length(simplices))
end

function _simplicial_boundary(simplices::Vector{Vector{Int}}, faces::Vector{Vector{Int}})
    if !_SIMPLICIAL_BOUNDARY_SPECIALIZED[]
        return _simplicial_boundary_hash(simplices, faces)
    end

    isempty(simplices) && return spzeros(Int, length(faces), 0)
    K = length(simplices[1])
    K == 0 && return spzeros(Int, length(faces), length(simplices))

    if K == 2
        # 1-simplices -> 0-simplices: rows are direct vertex indices.
        I = Int[]
        J = Int[]
        V = Int[]
        sizehint!(I, 2 * length(simplices))
        sizehint!(J, 2 * length(simplices))
        sizehint!(V, 2 * length(simplices))
        @inbounds for (j, s) in enumerate(simplices)
            u, v = s[1], s[2]
            push!(I, v); push!(J, j); push!(V, 1)
            push!(I, u); push!(J, j); push!(V, -1)
        end
        return sparse(I, J, V, length(faces), length(simplices))
    elseif K == 3
        # 2-simplices -> 1-simplices: map edge keys to row by sorted-key binary search.
        nf = length(faces)
        face_keys = Vector{UInt64}(undef, nf)
        @inbounds for i in 1:nf
            f = faces[i]
            a, b = f[1], f[2]
            a > b && ((a, b) = (b, a))
            face_keys[i] = _pack_edge_key(a, b)
        end
        perm = sortperm(face_keys)
        sorted_keys = face_keys[perm]

        @inline function row_for_edge(a::Int, b::Int)
            a > b && ((a, b) = (b, a))
            key = _pack_edge_key(a, b)
            idx = searchsortedfirst(sorted_keys, key)
            (idx <= nf && sorted_keys[idx] == key) ||
                error("_simplicial_boundary: missing edge face for key=($(a),$(b)).")
            return perm[idx]
        end

        I = Int[]
        J = Int[]
        V = Int[]
        sizehint!(I, 3 * length(simplices))
        sizehint!(J, 3 * length(simplices))
        sizehint!(V, 3 * length(simplices))
        @inbounds for (j, s) in enumerate(simplices)
            a, b, c = s[1], s[2], s[3]
            push!(I, row_for_edge(b, c)); push!(J, j); push!(V, 1)
            push!(I, row_for_edge(a, c)); push!(J, j); push!(V, -1)
            push!(I, row_for_edge(a, b)); push!(J, j); push!(V, 1)
        end
        return sparse(I, J, V, length(faces), length(simplices))
    end

    return _simplicial_boundary_hash(simplices, faces)
end

function _simplex_tree_grade_sets(simplices::Vector{Vector{Vector{Int}}},
                                  grades::Vector{<:NTuple{N,T}}) where {N,T}
    total = sum(length, simplices)
    length(grades) == total ||
        error("simplex-tree grades length mismatch: expected $(total), got $(length(grades)).")
    out = Vector{Vector{NTuple{N,T}}}(undef, total)
    idx = 1
    for d in eachindex(simplices)
        for _ in eachindex(simplices[d])
            out[idx] = NTuple{N,T}[grades[idx]]
            idx += 1
        end
    end
    return out
end

function _simplex_tree_grade_sets(simplices::Vector{Vector{Vector{Int}}},
                                  grades::Vector{<:AbstractVector{<:NTuple{N,T}}}) where {N,T}
    total = sum(length, simplices)
    length(grades) == total ||
        error("simplex-tree grade-set length mismatch: expected $(total), got $(length(grades)).")
    out = Vector{Vector{NTuple{N,T}}}(undef, total)
    @inbounds for i in 1:total
        gi = grades[i]
        isempty(gi) && error("simplex-tree: simplex $i has empty grade set.")
        out[i] = unique(Vector{NTuple{N,T}}(gi))
    end
    return out
end

@inline function _simplex_tree_cell_storage(total::Int, dim_offsets::Vector{Int})
    cell_ids = Vector{Int}(undef, total)
    p = 1
    @inbounds for slot in 1:(length(dim_offsets) - 1)
        count = dim_offsets[slot + 1] - dim_offsets[slot]
        for local_id in 1:count
            cell_ids[p] = local_id
            p += 1
        end
    end
    return cell_ids, dim_offsets
end

function _simplex_tree_unique_grade_storage(ST::SimplexTreeMulti{N,T}) where {N,T}
    ns = simplex_count(ST)
    isempty(ST.grade_data) && error("simplex-tree: each simplex must have at least one grade.")
    out_offsets = Vector{Int}(undef, ns + 1)
    out_offsets[1] = 1
    out_data = Vector{NTuple{N,T}}()
    sizehint!(out_data, length(ST.grade_data))
    @inbounds for sid in 1:ns
        lo = ST.grade_offsets[sid]
        hi = ST.grade_offsets[sid + 1] - 1
        for k in lo:hi
            g = ST.grade_data[k]
            duplicate = false
            for prev in out_offsets[sid]:(length(out_data))
                if out_data[prev] == g
                    duplicate = true
                    break
                end
            end
            duplicate && continue
            push!(out_data, g)
        end
        (length(out_data) + 1 > out_offsets[sid]) ||
            error("simplex-tree: simplex $sid has empty grade set.")
        out_offsets[sid + 1] = length(out_data) + 1
    end
    return out_offsets, out_data
end

function _simplex_tree_multi_from_simplices_packed(simplices::Vector{Vector{Vector{Int}}},
                                                   grade_offsets::AbstractVector{Int},
                                                   grade_data::AbstractVector{<:NTuple{N,T}}) where {N,T}
    isempty(simplices) && error("simplex-tree: simplices cannot be empty.")
    total = sum(length, simplices)
    total > 0 || error("simplex-tree: simplices cannot be empty.")
    length(grade_offsets) == total + 1 ||
        error("simplex-tree grade-set length mismatch: expected $(total), got $(length(grade_offsets) - 1).")

    simplex_offsets = Int[1]
    simplex_vertices_flat = Int[]
    simplex_dims = Int[]
    dim_offsets = Int[1]
    sizehint!(simplex_dims, total)
    sizehint!(simplex_offsets, total + 1)
    sizehint!(simplex_vertices_flat, sum(length, Iterators.flatten(simplices)))

    gidx = 1
    for d in eachindex(simplices)
        dim = d - 1
        for s in simplices[d]
            ss = sort!(unique(Int[x for x in s]))
            expected = dim + 1
            length(ss) == expected ||
                error("simplex-tree: simplex at dim=$dim must have $(expected) vertices (got $(length(ss))).")
            append!(simplex_vertices_flat, ss)
            push!(simplex_offsets, length(simplex_vertices_flat) + 1)
            push!(simplex_dims, dim)
            lo = grade_offsets[gidx]
            hi = grade_offsets[gidx + 1] - 1
            hi >= lo || error("simplex-tree: simplex $gidx has empty grade set.")
            @inbounds for p in lo:hi
                length(grade_data[p]) == N || error("simplex-tree: grade arity mismatch at simplex $gidx.")
            end
            gidx += 1
        end
        push!(dim_offsets, length(simplex_dims) + 1)
    end

    return SimplexTreeMulti(
        simplex_offsets,
        simplex_vertices_flat,
        simplex_dims,
        dim_offsets,
        copy(grade_offsets),
        Vector{NTuple{N,T}}(grade_data),
    )
end

function _simplex_tree_multi_from_simplices(simplices::Vector{Vector{Vector{Int}}},
                                            grades::Vector)
    isempty(simplices) && error("simplex-tree: simplices cannot be empty.")
    total = sum(length, simplices)
    total > 0 || error("simplex-tree: simplices cannot be empty.")
    grade_sets = _simplex_tree_grade_sets(simplices, grades)
    N = length(grade_sets[1][1])
    T = eltype(grade_sets[1][1])

    simplex_offsets = Int[1]
    simplex_vertices_flat = Int[]
    simplex_dims = Int[]
    dim_offsets = Int[1]

    grade_offsets = Int[1]
    grade_data = NTuple{N,T}[]
    sizehint!(simplex_dims, total)
    sizehint!(grade_offsets, total + 1)
    sizehint!(simplex_offsets, total + 1)

    gidx = 1
    for d in eachindex(simplices)
        dim = d - 1
        for s in simplices[d]
            ss = sort!(unique(Int[x for x in s]))
            expected = dim + 1
            length(ss) == expected ||
                error("simplex-tree: simplex at dim=$dim must have $(expected) vertices (got $(length(ss))).")
            append!(simplex_vertices_flat, ss)
            push!(simplex_offsets, length(simplex_vertices_flat) + 1)
            push!(simplex_dims, dim)

            gs = grade_sets[gidx]
            for g in gs
                length(g) == N || error("simplex-tree: grade arity mismatch at simplex $gidx.")
                push!(grade_data, ntuple(i -> T(g[i]), N))
            end
            push!(grade_offsets, length(grade_data) + 1)
            gidx += 1
        end
        push!(dim_offsets, length(simplex_dims) + 1)
    end

    return SimplexTreeMulti(simplex_offsets, simplex_vertices_flat, simplex_dims,
                            dim_offsets, grade_offsets, grade_data)
end

function _simplices_from_complex(G::Union{GradedComplex,MultiCriticalGradedComplex})
    _, dim_offsets = _packed_cell_storage(G)
    nd = length(dim_offsets) - 1
    nd > 0 || error("simplex-tree: complex has no dimensions.")
    counts = _packed_dim_counts(dim_offsets)
    simplices = Vector{Vector{Vector{Int}}}(undef, nd)
    simplices[1] = [[i] for i in 1:counts[1]]
    for d in 2:nd
        B = G.boundaries[d - 1]
        size(B, 1) == counts[d - 1] || error("simplex-tree: boundary row size mismatch at dim $(d - 1).")
        size(B, 2) == counts[d] || error("simplex-tree: boundary col size mismatch at dim $(d - 1).")
        curr = Vector{Vector{Int}}(undef, counts[d])
        prev = simplices[d - 1]
        @inbounds for j in 1:counts[d]
            lo = B.colptr[j]
            hi = B.colptr[j + 1] - 1
            nfaces = hi - lo + 1
            expected_faces = d
            nfaces == expected_faces ||
                error("simplex-tree: non-simplicial boundary at dim $(d - 1), cell $j (expected $expected_faces faces, got $nfaces).")
            verts = Int[]
            for p in lo:hi
                row = B.rowval[p]
                append!(verts, prev[row])
            end
            sort!(verts)
            unique!(verts)
            expected_verts = d
            length(verts) == expected_verts ||
                error("simplex-tree: non-simplicial support at dim $(d - 1), cell $j (expected $expected_verts vertices, got $(length(verts))).")
            curr[j] = verts
        end
        simplices[d] = curr
    end
    return simplices
end

function _simplex_tree_multi_from_complex(G::GradedComplex)
    total = length(G.grades)
    grade_offsets = collect(1:(total + 1))
    return _simplex_tree_multi_from_simplices_packed(_simplices_from_complex(G), grade_offsets, G.grades)
end

function _simplex_tree_multi_from_complex(G::MultiCriticalGradedComplex)
    grade_offsets, grade_data = _packed_multigrade_storage(G)
    return _simplex_tree_multi_from_simplices_packed(_simplices_from_complex(G), grade_offsets, grade_data)
end

function _simplex_tree_dim_simplices(ST::SimplexTreeMulti, dim_slot::Int)
    lo = ST.dim_offsets[dim_slot]
    hi = ST.dim_offsets[dim_slot + 1] - 1
    out = Vector{Vector{Int}}(undef, max(0, hi - lo + 1))
    p = 1
    for sid in lo:hi
        out[p] = Vector{Int}(simplex_vertices(ST, sid))
        p += 1
    end
    return out
end

@inline function _simplex_tree_is_onecritical(ST::SimplexTreeMulti)
    @inbounds for i in 1:simplex_count(ST)
        (ST.grade_offsets[i + 1] - ST.grade_offsets[i]) == 1 || return false
    end
    return true
end

function _graded_complex_from_simplex_tree(ST::SimplexTreeMulti{N,T}) where {N,T}
    nd = length(ST.dim_offsets) - 1
    simplices = Vector{Vector{Vector{Int}}}(undef, nd)
    for d in 1:nd
        simplices[d] = _simplex_tree_dim_simplices(ST, d)
    end
    boundaries = SparseMatrixCSC{Int,Int}[]
    for d in 2:nd
        push!(boundaries, _simplicial_boundary(simplices[d], simplices[d - 1]))
    end
    cell_ids, dim_offsets = _simplex_tree_cell_storage(simplex_count(ST), copy(ST.dim_offsets))
    if _simplex_tree_is_onecritical(ST)
        grades = Vector{NTuple{N,T}}(undef, simplex_count(ST))
        @inbounds for i in 1:simplex_count(ST)
            grades[i] = ST.grade_data[ST.grade_offsets[i]]
        end
        return GradedComplex{N,T}(cell_ids, dim_offsets, boundaries, grades)
    end
    grade_offsets, grade_data = _simplex_tree_unique_grade_storage(ST)
    return MultiCriticalGradedComplex{N,T}(cell_ids, dim_offsets, boundaries, grade_offsets, grade_data)
end

function _axes_from_simplex_tree(ST::SimplexTreeMulti{N,T};
                                 orientation::NTuple{N,Int}=ntuple(_ -> 1, N)) where {N,T}
    return _axes_from_grades(ST.grade_data, N; orientation=orientation)
end

function _materialize_simplicial_output(simplices::Vector{Vector{Vector{Int}}},
                                        grades::Vector{<:NTuple{N,T}},
                                        spec::FiltrationSpec;
                                        return_simplex_tree::Bool=false) where {N,T}
    orientation = get(spec.params, :orientation, ntuple(_ -> 1, N))
    if return_simplex_tree
        ST = _simplex_tree_multi_from_simplices(simplices, grades)
        axes = get(spec.params, :axes, _axes_from_simplex_tree(ST; orientation=orientation))
        return ST, axes, orientation
    end
    boundaries = SparseMatrixCSC{Int,Int}[]
    for k in 2:length(simplices)
        push!(boundaries, _simplicial_boundary(simplices[k], simplices[k - 1]))
    end
    cells = [collect(1:length(simplices[k])) for k in 1:length(simplices)]
    G = GradedComplex(cells, boundaries, grades)
    axes = get(spec.params, :axes, _axes_from_grades(grades, N; orientation=orientation))
    return G, axes, orientation
end

function _materialize_simplicial_output(simplices::Vector{Vector{Vector{Int}}},
                                        grades::Vector{<:AbstractVector{<:NTuple{N,T}}},
                                        spec::FiltrationSpec;
                                        return_simplex_tree::Bool=false) where {N,T}
    orientation = get(spec.params, :orientation, ntuple(_ -> 1, N))
    if return_simplex_tree
        ST = _simplex_tree_multi_from_simplices(simplices, grades)
        axes = get(spec.params, :axes, _axes_from_simplex_tree(ST; orientation=orientation))
        return ST, axes, orientation
    end
    boundaries = SparseMatrixCSC{Int,Int}[]
    for k in 2:length(simplices)
        push!(boundaries, _simplicial_boundary(simplices[k], simplices[k - 1]))
    end
    cells = [collect(1:length(simplices[k])) for k in 1:length(simplices)]
    G = MultiCriticalGradedComplex(cells, boundaries, grades)
    axes = get(spec.params, :axes, _axes_from_multigrades(grades, N; orientation=orientation))
    return G, axes, orientation
end

function _combinations(n::Int, k::Int)
    if k == 0
        return [Int[]]
    end
    out = Vector{Vector{Int}}()
    function rec(start::Int, acc::Vector{Int})
        if length(acc) == k
            push!(out, copy(acc))
            return
        end
        for i in start:(n - (k - length(acc)) + 1)
            push!(acc, i)
            rec(i + 1, acc)
            pop!(acc)
        end
    end
    rec(1, Int[])
    return out
end

function _filter_params(params::NamedTuple, drop::Vector{Symbol})
    return (; (k => v for (k, v) in pairs(params) if !(k in drop))...)
end

@inline function _euclidean_distance(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})
    length(p) == length(q) || error("PointCloud: point dimension mismatch.")
    s = 0.0
    @inbounds for i in 1:length(p)
        d = Float64(p[i]) - Float64(q[i])
        s += d * d
    end
    return sqrt(s)
end

@inline function _euclidean_distance(points::AbstractMatrix{<:Real}, i::Int, j::Int)
    d = size(points, 2)
    s = 0.0
    @inbounds for k in 1:d
        diff = Float64(points[i, k]) - Float64(points[j, k])
        s += diff * diff
    end
    return sqrt(s)
end

@inline _point_rows_view(points::AbstractMatrix{<:Real}) = MatrixRowsView(points)

@inline function _ordered_pair(i::Int, j::Int)
    return i < j ? (i, j) : (j, i)
end

@inline function _pack_edge_key(u::Int, v::Int)::UInt64
    return (UInt64(u) << 32) | UInt64(v)
end

@inline function _unpack_edge_key(key::UInt64)
    return Int(key >>> 32), Int(key & 0x00000000ffffffff)
end

function _finalize_edge_pairs(edge_keys::AbstractVector{UInt64},
                              edge_dists::AbstractVector{Float64},
                              edge_count::Int)
    edge_count <= 0 && return NTuple{2,Int}[], Float64[]
    order = sortperm(view(edge_keys, 1:edge_count))

    unique_count = 0
    prev_key = UInt64(0)
    first = true
    @inbounds for idx in order
        key = edge_keys[idx]
        if first || key != prev_key
            unique_count += 1
            prev_key = key
            first = false
        end
    end

    edges = Vector{NTuple{2,Int}}(undef, unique_count)
    dists = Vector{Float64}(undef, unique_count)
    out = 0
    prev_key = UInt64(0)
    best_dist = 0.0
    first = true
    @inbounds for idx in order
        key = edge_keys[idx]
        d = edge_dists[idx]
        if first
            prev_key = key
            best_dist = d
            first = false
            continue
        end
        if key == prev_key
            d < best_dist && (best_dist = d)
            continue
        end
        out += 1
        u, v = _unpack_edge_key(prev_key)
        edges[out] = (u, v)
        dists[out] = best_dist
        prev_key = key
        best_dist = d
    end
    out += 1
    u, v = _unpack_edge_key(prev_key)
    edges[out] = (u, v)
    dists[out] = best_dist
    return edges, dists
end

function _finalize_edge_pairs_edges_only(edge_keys::AbstractVector{UInt64},
                                         edge_count::Int)
    edge_count <= 0 && return NTuple{2,Int}[]
    order = sortperm(view(edge_keys, 1:edge_count))

    unique_count = 0
    prev_key = UInt64(0)
    first = true
    @inbounds for idx in order
        key = edge_keys[idx]
        if first || key != prev_key
            unique_count += 1
            prev_key = key
            first = false
        end
    end

    edges = Vector{NTuple{2,Int}}(undef, unique_count)
    out = 0
    prev_key = UInt64(0)
    first = true
    @inbounds for idx in order
        key = edge_keys[idx]
        if first
            prev_key = key
            first = false
            continue
        end
        key == prev_key && continue
        out += 1
        edges[out] = _unpack_edge_key(prev_key)
        prev_key = key
    end
    out += 1
    edges[out] = _unpack_edge_key(prev_key)
    return edges
end

@inline function _insert_neighbor_sorted!(idxs::Vector{Int}, dists::Vector{Float64}, j::Int, d::Float64)
    k = length(idxs)
    if d >= dists[k]
        return nothing
    end
    pos = k
    @inbounds while pos > 1 && d < dists[pos - 1]
        dists[pos] = dists[pos - 1]
        idxs[pos] = idxs[pos - 1]
        pos -= 1
    end
    dists[pos] = d
    idxs[pos] = j
    return nothing
end

function _knn_graph_bruteforce(points::AbstractVector{<:AbstractVector{<:Real}}, k::Int)
    n = length(points)
    n > 0 || return Vector{Vector{Int}}(), Float64[], Float64[]
    k <= n - 1 || error("kNN k=$(k) exceeds number of neighbors.")
    k_eff = min(k, n - 1)
    k_eff > 0 || error("kNN k=$(k) exceeds number of neighbors.")
    edge_keys = Vector{UInt64}(undef, n * k_eff)
    edge_vals = Vector{Float64}(undef, n * k_eff)
    edge_count = 0
    kdist = Vector{Float64}(undef, n)
    idxs = fill(0, k_eff)
    dtmp = fill(Inf, k_eff)
    for i in 1:n
        fill!(idxs, 0)
        fill!(dtmp, Inf)
        for j in 1:n
            i == j && continue
            _insert_neighbor_sorted!(idxs, dtmp, j, _euclidean_distance(points[i], points[j]))
        end
        kdist[i] = dtmp[k_eff]
        for t in 1:k_eff
            j = idxs[t]
            j == 0 && continue
            u, v = _ordered_pair(i, j)
            edge_count += 1
            edge_keys[edge_count] = _pack_edge_key(u, v)
            edge_vals[edge_count] = dtmp[t]
        end
    end
    edges, edists = _finalize_edge_pairs(edge_keys, edge_vals, edge_count)
    return edges, edists, kdist
end

function _knn_graph_bruteforce(points::AbstractMatrix{<:Real}, k::Int)
    n = size(points, 1)
    n > 0 || return NTuple{2,Int}[], Float64[], Float64[]
    k <= n - 1 || error("kNN k=$(k) exceeds number of neighbors.")
    k_eff = min(k, n - 1)
    k_eff > 0 || error("kNN k=$(k) exceeds number of neighbors.")
    edge_keys = Vector{UInt64}(undef, n * k_eff)
    edge_vals = Vector{Float64}(undef, n * k_eff)
    edge_count = 0
    kdist = Vector{Float64}(undef, n)
    idxs = fill(0, k_eff)
    dtmp = fill(Inf, k_eff)
    @inbounds for i in 1:n
        fill!(idxs, 0)
        fill!(dtmp, Inf)
        for j in 1:n
            i == j && continue
            _insert_neighbor_sorted!(idxs, dtmp, j, _euclidean_distance(points, i, j))
        end
        kdist[i] = dtmp[k_eff]
        for t in 1:k_eff
            j = idxs[t]
            j == 0 && continue
            u, v = _ordered_pair(i, j)
            edge_count += 1
            edge_keys[edge_count] = _pack_edge_key(u, v)
            edge_vals[edge_count] = dtmp[t]
        end
    end
    edges, edists = _finalize_edge_pairs(edge_keys, edge_vals, edge_count)
    return edges, edists, kdist
end

function _knn_graph_bruteforce_edges_only(points::AbstractVector{<:AbstractVector{<:Real}}, k::Int)
    n = length(points)
    n > 0 || return NTuple{2,Int}[]
    k <= n - 1 || error("kNN k=$(k) exceeds number of neighbors.")
    k_eff = min(k, n - 1)
    k_eff > 0 || error("kNN k=$(k) exceeds number of neighbors.")
    edge_keys = Vector{UInt64}(undef, n * k_eff)
    edge_count = 0
    idxs = fill(0, k_eff)
    dtmp = fill(Inf, k_eff)
    for i in 1:n
        fill!(idxs, 0)
        fill!(dtmp, Inf)
        for j in 1:n
            i == j && continue
            _insert_neighbor_sorted!(idxs, dtmp, j, _euclidean_distance(points[i], points[j]))
        end
        for t in 1:k_eff
            j = idxs[t]
            j == 0 && continue
            u, v = _ordered_pair(i, j)
            edge_count += 1
            edge_keys[edge_count] = _pack_edge_key(u, v)
        end
    end
    return _finalize_edge_pairs_edges_only(edge_keys, edge_count)
end

function _knn_graph_bruteforce_edges_only(points::AbstractMatrix{<:Real}, k::Int)
    n = size(points, 1)
    n > 0 || return NTuple{2,Int}[]
    k <= n - 1 || error("kNN k=$(k) exceeds number of neighbors.")
    k_eff = min(k, n - 1)
    k_eff > 0 || error("kNN k=$(k) exceeds number of neighbors.")
    edge_keys = Vector{UInt64}(undef, n * k_eff)
    edge_count = 0
    idxs = fill(0, k_eff)
    dtmp = fill(Inf, k_eff)
    @inbounds for i in 1:n
        fill!(idxs, 0)
        fill!(dtmp, Inf)
        for j in 1:n
            i == j && continue
            _insert_neighbor_sorted!(idxs, dtmp, j, _euclidean_distance(points, i, j))
        end
        for t in 1:k_eff
            j = idxs[t]
            j == 0 && continue
            u, v = _ordered_pair(i, j)
            edge_count += 1
            edge_keys[edge_count] = _pack_edge_key(u, v)
        end
    end
    return _finalize_edge_pairs_edges_only(edge_keys, edge_count)
end

function _radius_graph_bruteforce(points::AbstractVector{<:AbstractVector{<:Real}}, r::Float64)
    n = length(points)
    edges = NTuple{2,Int}[]
    dists = Float64[]
    sizehint!(edges, min(max(0, 4 * n), 200_000))
    sizehint!(dists, min(max(0, 4 * n), 200_000))
    for i in 1:n
        for j in (i + 1):n
            d = _euclidean_distance(points[i], points[j])
            if d <= r
                push!(edges, (i, j))
                push!(dists, d)
            end
        end
    end
    return edges, dists
end

function _radius_graph_bruteforce(points::AbstractMatrix{<:Real}, r::Float64)
    n = size(points, 1)
    edges = NTuple{2,Int}[]
    dists = Float64[]
    sizehint!(edges, min(max(0, 4 * n), 200_000))
    sizehint!(dists, min(max(0, 4 * n), 200_000))
    @inbounds for i in 1:(n - 1)
        for j in (i + 1):n
            d = _euclidean_distance(points, i, j)
            d <= r || continue
            push!(edges, (i, j))
            push!(dists, d)
        end
    end
    return edges, dists
end

function _radius_graph_bruteforce_edges_only(points::AbstractVector{<:AbstractVector{<:Real}}, r::Float64)
    n = length(points)
    edges = NTuple{2,Int}[]
    sizehint!(edges, min(max(0, 4 * n), 200_000))
    for i in 1:n
        for j in (i + 1):n
            _euclidean_distance(points[i], points[j]) <= r || continue
            push!(edges, (i, j))
        end
    end
    return edges
end

function _radius_graph_bruteforce_edges_only(points::AbstractMatrix{<:Real}, r::Float64)
    n = size(points, 1)
    edges = NTuple{2,Int}[]
    sizehint!(edges, min(max(0, 4 * n), 200_000))
    @inbounds for i in 1:(n - 1)
        for j in (i + 1):n
            _euclidean_distance(points, i, j) <= r || continue
            push!(edges, (i, j))
        end
    end
    return edges
end

function _point_cloud_knn_graph(points::AbstractVector{<:AbstractVector{<:Real}},
                                k::Int;
                                backend::Symbol=:auto,
                                approx_candidates::Int=0)
    n = length(points)
    n > 0 || return NTuple{2,Int}[], Float64[], Float64[]
    k <= n - 1 || error("kNN k=$(k) exceeds number of neighbors.")
    d = length(points[1])
    backend0 = _resolve_pointcloud_runtime_backend(backend, n, d, 1)
    if backend0 != :bruteforce
        impl = _POINTCLOUD_KNN_GRAPH_IMPL[]
        if impl !== nothing
            out = impl(points, k; backend=backend0, approx_candidates=approx_candidates)
            out === nothing || return out
        elseif backend0 == :nearestneighbors || backend0 == :approx
            throw(ArgumentError("PointCloud nn_backend=$(backend0) requires NearestNeighbors extension."))
        end
    end
    return _knn_graph_bruteforce(points, k)
end

function _point_cloud_knn_graph(points::AbstractMatrix{<:Real},
                                k::Int;
                                backend::Symbol=:auto,
                                approx_candidates::Int=0)
    n = size(points, 1)
    n > 0 || return NTuple{2,Int}[], Float64[], Float64[]
    k <= n - 1 || error("kNN k=$(k) exceeds number of neighbors.")
    d = size(points, 2)
    backend0 = _resolve_pointcloud_runtime_backend(backend, n, d, 1)
    if backend0 != :bruteforce
        return _point_cloud_knn_graph(_point_rows_view(points), k;
                                      backend=backend0,
                                      approx_candidates=approx_candidates)
    end
    return _knn_graph_bruteforce(points, k)
end

function _point_cloud_knn_edges(points::AbstractVector{<:AbstractVector{<:Real}},
                                k::Int;
                                backend::Symbol=:auto,
                                approx_candidates::Int=0)
    n = length(points)
    n > 0 || return NTuple{2,Int}[]
    k <= n - 1 || error("kNN k=$(k) exceeds number of neighbors.")
    d = length(points[1])
    backend0 = _resolve_pointcloud_runtime_backend(backend, n, d, 1)
    if backend0 != :bruteforce
        impl = _POINTCLOUD_KNN_GRAPH_EDGES_IMPL[]
        if impl !== nothing
            out = impl(points, k; backend=backend0, approx_candidates=approx_candidates)
            out === nothing || return out
        end
        impl_full = _POINTCLOUD_KNN_GRAPH_IMPL[]
        if impl_full !== nothing
            out_full = impl_full(points, k; backend=backend0, approx_candidates=approx_candidates)
            out_full === nothing || return out_full[1]
        elseif backend0 == :nearestneighbors || backend0 == :approx
            throw(ArgumentError("PointCloud nn_backend=$(backend0) requires NearestNeighbors extension."))
        end
    end
    return _knn_graph_bruteforce_edges_only(points, k)
end

function _point_cloud_knn_edges(points::AbstractMatrix{<:Real},
                                k::Int;
                                backend::Symbol=:auto,
                                approx_candidates::Int=0)
    n = size(points, 1)
    n > 0 || return NTuple{2,Int}[]
    k <= n - 1 || error("kNN k=$(k) exceeds number of neighbors.")
    d = size(points, 2)
    backend0 = _resolve_pointcloud_runtime_backend(backend, n, d, 1)
    if backend0 != :bruteforce
        return _point_cloud_knn_edges(_point_rows_view(points), k;
                                      backend=backend0,
                                      approx_candidates=approx_candidates)
    end
    return _knn_graph_bruteforce_edges_only(points, k)
end

function _point_cloud_radius_graph(points::AbstractVector{<:AbstractVector{<:Real}},
                                   r::Float64;
                                   backend::Symbol=:auto,
                                   approx_candidates::Int=0)
    n = length(points)
    if n == 0
        return NTuple{2,Int}[], Float64[]
    end
    d = length(points[1])
    backend0 = _resolve_pointcloud_runtime_backend(backend, n, d, 2)
    if backend0 != :bruteforce
        impl = _POINTCLOUD_RADIUS_GRAPH_IMPL[]
        if impl !== nothing
            out = impl(points, r; backend=backend0, approx_candidates=approx_candidates)
            out === nothing || return out
        elseif backend0 == :nearestneighbors || backend0 == :approx
            throw(ArgumentError("PointCloud nn_backend=$(backend0) requires NearestNeighbors extension."))
        end
    end
    return _radius_graph_bruteforce(points, r)
end

function _point_cloud_radius_graph(points::AbstractMatrix{<:Real},
                                   r::Float64;
                                   backend::Symbol=:auto,
                                   approx_candidates::Int=0)
    n = size(points, 1)
    n == 0 && return NTuple{2,Int}[], Float64[]
    d = size(points, 2)
    backend0 = _resolve_pointcloud_runtime_backend(backend, n, d, 2)
    if backend0 != :bruteforce
        return _point_cloud_radius_graph(_point_rows_view(points), r;
                                         backend=backend0,
                                         approx_candidates=approx_candidates)
    end
    return _radius_graph_bruteforce(points, r)
end

function _point_cloud_radius_edges(points::AbstractVector{<:AbstractVector{<:Real}},
                                   r::Float64;
                                   backend::Symbol=:auto,
                                   approx_candidates::Int=0)
    n = length(points)
    n == 0 && return NTuple{2,Int}[]
    d = length(points[1])
    backend0 = _resolve_pointcloud_runtime_backend(backend, n, d, 2)
    if backend0 != :bruteforce
        impl = _POINTCLOUD_RADIUS_GRAPH_EDGES_IMPL[]
        if impl !== nothing
            out = impl(points, r; backend=backend0, approx_candidates=approx_candidates)
            out === nothing || return out
        end
        impl_full = _POINTCLOUD_RADIUS_GRAPH_IMPL[]
        if impl_full !== nothing
            out_full = impl_full(points, r; backend=backend0, approx_candidates=approx_candidates)
            out_full === nothing || return out_full[1]
        elseif backend0 == :nearestneighbors || backend0 == :approx
            throw(ArgumentError("PointCloud nn_backend=$(backend0) requires NearestNeighbors extension."))
        end
    end
    return _radius_graph_bruteforce_edges_only(points, r)
end

function _point_cloud_radius_edges(points::AbstractMatrix{<:Real},
                                   r::Float64;
                                   backend::Symbol=:auto,
                                   approx_candidates::Int=0)
    n = size(points, 1)
    n == 0 && return NTuple{2,Int}[]
    d = size(points, 2)
    backend0 = _resolve_pointcloud_runtime_backend(backend, n, d, 2)
    if backend0 != :bruteforce
        return _point_cloud_radius_edges(_point_rows_view(points), r;
                                         backend=backend0,
                                         approx_candidates=approx_candidates)
    end
    return _radius_graph_bruteforce_edges_only(points, r)
end

@inline function _edge_vectors_from_tuples(edges::Vector{NTuple{2,Int}})
    out = Vector{Vector{Int}}(undef, length(edges))
    @inbounds for i in eachindex(edges)
        u, v = edges[i]
        out[i] = [u, v]
    end
    return out
end

@inline function _edge_boundary_matrix(n::Int, edges::Vector{NTuple{2,Int}})
    m = length(edges)
    m == 0 && return spzeros(Int, n, 0)
    I = Vector{Int}(undef, 2m)
    J = Vector{Int}(undef, 2m)
    V = Vector{Int}(undef, 2m)
    t = 1
    @inbounds for j in 1:m
        u, v = edges[j]
        I[t] = v
        J[t] = j
        V[t] = 1
        t += 1
        I[t] = u
        J[t] = j
        V[t] = -1
        t += 1
    end
    return sparse(I, J, V, n, m)
end

function _triangle_boundary_matrix_from_edges(n::Int,
                                              edges::Vector{NTuple{2,Int}},
                                              triangles::Vector{NTuple{3,Int}})
    ne = length(edges)
    nt = length(triangles)
    nt == 0 && return spzeros(Int, ne, 0)

    pair_count_big = _combination_count(n, 2)
    pair_count_big > big(typemax(Int)) &&
        throw(ArgumentError("triangle boundary edge index overflow for n=$(n)."))
    pair_count = Int(pair_count_big)
    edge_rows = zeros(Int, pair_count)
    @inbounds for (row, (u, v)) in enumerate(edges)
        edge_rows[_packed_pair_index(n, u, v)] = row
    end

    I = Vector{Int}(undef, 3 * nt)
    J = Vector{Int}(undef, 3 * nt)
    V = Vector{Int}(undef, 3 * nt)
    t = 1
    @inbounds for col in 1:nt
        a, b, c = triangles[col]
        row_bc = edge_rows[_packed_pair_index(n, b, c)]
        row_ac = edge_rows[_packed_pair_index(n, a, c)]
        row_ab = edge_rows[_packed_pair_index(n, a, b)]
        row_bc == 0 && error("triangle boundary: missing edge ($(b),$(c)) at triangle ($(a),$(b),$(c)).")
        row_ac == 0 && error("triangle boundary: missing edge ($(a),$(c)) at triangle ($(a),$(b),$(c)).")
        row_ab == 0 && error("triangle boundary: missing edge ($(a),$(b)) at triangle ($(a),$(b),$(c)).")

        I[t] = row_bc
        J[t] = col
        V[t] = 1
        t += 1
        I[t] = row_ac
        J[t] = col
        V[t] = -1
        t += 1
        I[t] = row_ab
        J[t] = col
        V[t] = 1
        t += 1
    end
    return sparse(I, J, V, ne, nt)
end

function _simplex_tree_multi_from_packed(n::Int,
                                         edges::Vector{NTuple{2,Int}},
                                         triangles::Vector{NTuple{3,Int}},
                                         grades::Vector{<:NTuple{N,T}};
                                         max_dim::Int=2) where {N,T}
    max_dim = clamp(max_dim, 0, 2)
    ne = max_dim >= 1 ? length(edges) : 0
    nt = max_dim >= 2 ? length(triangles) : 0
    ns = n + ne + nt
    length(grades) == ns ||
        error("simplex-tree packed: grades length mismatch, expected $(ns), got $(length(grades)).")

    simplex_offsets = Vector{Int}(undef, ns + 1)
    simplex_vertices = Vector{Int}(undef, n + 2 * ne + 3 * nt)
    simplex_dims = Vector{Int}(undef, ns)
    grade_offsets = collect(1:(ns + 1))
    grade_data = Vector{NTuple{N,T}}(undef, ns)
    dim_offsets = max_dim == 0 ? Int[1, ns + 1] :
                  (max_dim == 1 ? Int[1, n + 1, ns + 1] :
                                  Int[1, n + 1, n + ne + 1, ns + 1])

    vidx = 1
    sidx = 1
    @inbounds for v in 1:n
        simplex_offsets[sidx] = vidx
        simplex_vertices[vidx] = v
        vidx += 1
        simplex_dims[sidx] = 0
        grade_data[sidx] = grades[sidx]
        sidx += 1
    end
    if max_dim >= 1
        @inbounds for j in 1:ne
            u, v = edges[j]
            simplex_offsets[sidx] = vidx
            simplex_vertices[vidx] = u
            simplex_vertices[vidx + 1] = v
            vidx += 2
            simplex_dims[sidx] = 1
            grade_data[sidx] = grades[sidx]
            sidx += 1
        end
    end
    if max_dim >= 2
        @inbounds for j in 1:nt
            a, b, c = triangles[j]
            simplex_offsets[sidx] = vidx
            simplex_vertices[vidx] = a
            simplex_vertices[vidx + 1] = b
            simplex_vertices[vidx + 2] = c
            vidx += 3
            simplex_dims[sidx] = 2
            grade_data[sidx] = grades[sidx]
            sidx += 1
        end
    end
    simplex_offsets[ns + 1] = vidx

    return SimplexTreeMulti(simplex_offsets, simplex_vertices, simplex_dims,
                            dim_offsets, grade_offsets, grade_data)
end

function _simplex_tree_multi_from_packed(n::Int,
                                         edges::Vector{NTuple{2,Int}},
                                         triangles::Vector{NTuple{3,Int}},
                                         ::Type{NTuple{N,T}},
                                         vertex_grade!::Fv,
                                         edge_grade!::Fe,
                                         tri_grade!::Ft;
                                         max_dim::Int=2) where {N,T,Fv,Fe,Ft}
    max_dim = clamp(max_dim, 0, 2)
    ne = max_dim >= 1 ? length(edges) : 0
    nt = max_dim >= 2 ? length(triangles) : 0
    ns = n + ne + nt

    simplex_offsets = Vector{Int}(undef, ns + 1)
    simplex_vertices = Vector{Int}(undef, n + 2 * ne + 3 * nt)
    simplex_dims = Vector{Int}(undef, ns)
    grade_offsets = collect(1:(ns + 1))
    grade_data = Vector{NTuple{N,T}}(undef, ns)
    dim_offsets = max_dim == 0 ? Int[1, ns + 1] :
                  (max_dim == 1 ? Int[1, n + 1, ns + 1] :
                                  Int[1, n + 1, n + ne + 1, ns + 1])

    vidx = 1
    sidx = 1
    @inbounds for v in 1:n
        simplex_offsets[sidx] = vidx
        simplex_vertices[vidx] = v
        vidx += 1
        simplex_dims[sidx] = 0
        grade_data[sidx] = vertex_grade!(v)
        sidx += 1
    end
    if max_dim >= 1
        @inbounds for j in 1:ne
            u, v = edges[j]
            simplex_offsets[sidx] = vidx
            simplex_vertices[vidx] = u
            simplex_vertices[vidx + 1] = v
            vidx += 2
            simplex_dims[sidx] = 1
            grade_data[sidx] = edge_grade!(j, u, v)
            sidx += 1
        end
    end
    if max_dim >= 2
        @inbounds for j in 1:nt
            a, b, c = triangles[j]
            simplex_offsets[sidx] = vidx
            simplex_vertices[vidx] = a
            simplex_vertices[vidx + 1] = b
            simplex_vertices[vidx + 2] = c
            vidx += 3
            simplex_dims[sidx] = 2
            grade_data[sidx] = tri_grade!(j, a, b, c)
            sidx += 1
        end
    end
    simplex_offsets[ns + 1] = vidx

    return SimplexTreeMulti(simplex_offsets, simplex_vertices, simplex_dims,
                            dim_offsets, grade_offsets, grade_data)
end

function _materialize_point_cloud_packed_simplex_tree(n::Int,
                                                      max_dim::Int,
                                                      edges::Vector{NTuple{2,Int}},
                                                      triangles::Vector{NTuple{3,Int}},
                                                      spec::FiltrationSpec,
                                                      ::Type{NTuple{N,T}},
                                                      vertex_grade!::Fv,
                                                      edge_grade!::Fe,
                                                      tri_grade!::Ft) where {N,T,Fv,Fe,Ft}
    max_dim = clamp(max_dim, 0, 2)
    orientation = get(spec.params, :orientation, ntuple(_ -> 1, N))
    ST = _simplex_tree_multi_from_packed(
        n, edges, triangles, NTuple{N,T}, vertex_grade!, edge_grade!, tri_grade!; max_dim=max_dim
    )
    axes = get(spec.params, :axes, _axes_from_simplex_tree(ST; orientation=orientation))
    return ST, axes, orientation
end

@inline function _simplex_tree_multi_from_dim2_packed(n::Int,
                                                      edges::Vector{NTuple{2,Int}},
                                                      triangles::Vector{NTuple{3,Int}},
                                                      grades::Vector{<:NTuple{N,T}}) where {N,T}
    return _simplex_tree_multi_from_packed(n, edges, triangles, grades; max_dim=2)
end

function _materialize_point_cloud_packed(n::Int,
                                         max_dim::Int,
                                         edges::Vector{NTuple{2,Int}},
                                         triangles::Vector{NTuple{3,Int}},
                                         grades::Vector{<:NTuple{N,Float64}},
                                         spec::FiltrationSpec;
                                         return_simplex_tree::Bool=false) where {N}
    max_dim = clamp(max_dim, 0, 2)
    ne = max_dim >= 1 ? length(edges) : 0
    nt = max_dim >= 2 ? length(triangles) : 0
    expected = n + ne + nt
    length(grades) == expected ||
        error("materialize_point_cloud_packed: grades length mismatch, expected $(expected), got $(length(grades)).")

    orientation = get(spec.params, :orientation, ntuple(_ -> 1, N))
    if return_simplex_tree
        ST = _simplex_tree_multi_from_packed(n, edges, triangles, grades; max_dim=max_dim)
        axes = get(spec.params, :axes, _axes_from_simplex_tree(ST; orientation=orientation))
        return ST, axes, orientation
    end

    if max_dim == 0
        cells = [collect(1:n)]
        boundaries = SparseMatrixCSC{Int,Int}[]
    elseif max_dim == 1
        cells = [collect(1:n), collect(1:ne)]
        boundaries = SparseMatrixCSC{Int,Int}[_edge_boundary_matrix(n, edges)]
    else
        b1 = _edge_boundary_matrix(n, edges)
        b2 = _triangle_boundary_matrix_from_edges(n, edges, triangles)
        cells = [collect(1:n), collect(1:ne), collect(1:nt)]
        boundaries = SparseMatrixCSC{Int,Int}[b1, b2]
    end

    G = GradedComplex(cells, boundaries, grades)
    axes = get(spec.params, :axes, _axes_from_grades(grades, N; orientation=orientation))
    return G, axes, orientation
end

function _materialize_point_cloud_packed_with_cached_boundaries(n::Int,
                                                                max_dim::Int,
                                                                entry::_PackedDelaunay2DCacheEntry,
                                                                grades::Vector{<:NTuple{N,Float64}},
                                                                spec::FiltrationSpec) where {N}
    max_dim = clamp(max_dim, 0, 2)
    packed = entry.packed
    ne = max_dim >= 1 ? length(packed.edges) : 0
    nt = max_dim >= 2 ? length(packed.triangles) : 0
    expected = n + ne + nt
    length(grades) == expected ||
        error("materialize_point_cloud_packed_with_cached_boundaries: grades length mismatch, expected $(expected), got $(length(grades)).")

    orientation = get(spec.params, :orientation, ntuple(_ -> 1, N))
    if max_dim == 0
        cells = [collect(1:n)]
        boundaries = SparseMatrixCSC{Int,Int}[]
    elseif max_dim == 1
        b1 = entry.edge_boundary === nothing ? _edge_boundary_matrix(n, packed.edges) : entry.edge_boundary
        cells = [collect(1:n), collect(1:ne)]
        boundaries = SparseMatrixCSC{Int,Int}[b1]
    else
        b1 = entry.edge_boundary === nothing ? _edge_boundary_matrix(n, packed.edges) : entry.edge_boundary
        b2 = entry.tri_boundary === nothing ? _triangle_boundary_matrix_from_edges(n, packed.edges, packed.triangles) : entry.tri_boundary
        cells = [collect(1:n), collect(1:ne), collect(1:nt)]
        boundaries = SparseMatrixCSC{Int,Int}[b1, b2]
    end

    G = GradedComplex(cells, boundaries, grades)
    axes = get(spec.params, :axes, _axes_from_grades(grades, N; orientation=orientation))
    return G, axes, orientation
end

function _materialize_point_cloud_dim012(n::Int,
                                         edges::Vector{NTuple{2,Int}},
                                         triangles::Vector{NTuple{3,Int}},
                                         grades::Vector{<:NTuple{N,Float64}},
                                         spec::FiltrationSpec;
                                         return_simplex_tree::Bool=false) where {N}
    return _materialize_point_cloud_packed(
        n, 2, edges, triangles, grades, spec; return_simplex_tree=return_simplex_tree
    )
end

function _complete_point_cloud_edges_with_dist(points::AbstractVector{<:AbstractVector{<:Real}})
    n = length(points)
    n <= 1 && return NTuple{2,Int}[], Float64[]
    edge_count_big = _combination_count(n, 2)
    edge_count_big > big(typemax(Int)) &&
        throw(ArgumentError("Ingestion combinatorial explosion: edge count C($(n),2)=$(edge_count_big) exceeds representable collection size."))
    edge_count = Int(edge_count_big)
    edges = Vector{NTuple{2,Int}}(undef, edge_count)
    dists = Vector{Float64}(undef, edge_count)
    t = 1
    @inbounds for i in 1:(n - 1)
        pi = points[i]
        for j in (i + 1):n
            edges[t] = (i, j)
            dists[t] = _euclidean_distance(pi, points[j])
            t += 1
        end
    end
    return edges, dists
end

function _complete_point_cloud_edges_with_dist(points::AbstractMatrix{<:Real})
    n = size(points, 1)
    n <= 1 && return NTuple{2,Int}[], Float64[]
    edge_count_big = _combination_count(n, 2)
    edge_count_big > big(typemax(Int)) &&
        throw(ArgumentError("Ingestion combinatorial explosion: edge count C($(n),2)=$(edge_count_big) exceeds representable collection size."))
    edge_count = Int(edge_count_big)
    edges = Vector{NTuple{2,Int}}(undef, edge_count)
    dists = Vector{Float64}(undef, edge_count)
    t = 1
    @inbounds for i in 1:(n - 1)
        for j in (i + 1):n
            edges[t] = (i, j)
            dists[t] = _euclidean_distance(points, i, j)
            t += 1
        end
    end
    return edges, dists
end

function _complete_point_cloud_edges(n::Int)
    n <= 1 && return NTuple{2,Int}[]
    edge_count_big = _combination_count(n, 2)
    edge_count_big > big(typemax(Int)) &&
        throw(ArgumentError("Ingestion combinatorial explosion: edge count C($(n),2)=$(edge_count_big) exceeds representable collection size."))
    edge_count = Int(edge_count_big)
    edges = Vector{NTuple{2,Int}}(undef, edge_count)
    t = 1
    @inbounds for i in 1:(n - 1)
        for j in (i + 1):n
            edges[t] = (i, j)
            t += 1
        end
    end
    return edges
end

function _point_cloud_edges_within_radius(points::AbstractVector{<:AbstractVector{<:Real}},
                                          radius::Float64)
    n = length(points)
    n <= 1 && return NTuple{2,Int}[], Float64[]
    isfinite(radius) || return _complete_point_cloud_edges_with_dist(points)
    edges = NTuple{2,Int}[]
    dists = Float64[]
    sizehint!(edges, min(max(0, 4 * n), 200_000))
    sizehint!(dists, min(max(0, 4 * n), 200_000))
    @inbounds for i in 1:(n - 1)
        pi = points[i]
        for j in (i + 1):n
            d = _euclidean_distance(pi, points[j])
            d <= radius || continue
            push!(edges, (i, j))
            push!(dists, d)
        end
    end
    return edges, dists
end

function _point_cloud_edges_within_radius(points::AbstractMatrix{<:Real},
                                          radius::Float64)
    n = size(points, 1)
    n <= 1 && return NTuple{2,Int}[], Float64[]
    isfinite(radius) || return _complete_point_cloud_edges_with_dist(points)
    edges = NTuple{2,Int}[]
    dists = Float64[]
    sizehint!(edges, min(max(0, 4 * n), 200_000))
    sizehint!(dists, min(max(0, 4 * n), 200_000))
    @inbounds for i in 1:(n - 1)
        for j in (i + 1):n
            d = _euclidean_distance(points, i, j)
            d <= radius || continue
            push!(edges, (i, j))
            push!(dists, d)
        end
    end
    return edges, dists
end

function _point_cloud_edges_within_radius_edges_only(points::AbstractVector{<:AbstractVector{<:Real}},
                                                     radius::Float64)
    n = length(points)
    n <= 1 && return NTuple{2,Int}[]
    isfinite(radius) || return _complete_point_cloud_edges(n)
    edges = NTuple{2,Int}[]
    sizehint!(edges, min(max(0, 4 * n), 200_000))
    @inbounds for i in 1:(n - 1)
        pi = points[i]
        for j in (i + 1):n
            _euclidean_distance(pi, points[j]) <= radius || continue
            push!(edges, (i, j))
        end
    end
    return edges
end

function _point_cloud_edges_within_radius_edges_only(points::AbstractMatrix{<:Real},
                                                     radius::Float64)
    n = size(points, 1)
    n <= 1 && return NTuple{2,Int}[]
    isfinite(radius) || return _complete_point_cloud_edges(n)
    edges = NTuple{2,Int}[]
    sizehint!(edges, min(max(0, 4 * n), 200_000))
    @inbounds for i in 1:(n - 1)
        for j in (i + 1):n
            _euclidean_distance(points, i, j) <= radius || continue
            push!(edges, (i, j))
        end
    end
    return edges
end

function _point_cloud_edges_within_radius_indexed(points::AbstractVector{<:AbstractVector{<:Real}},
                                                  idxs::AbstractVector{Int},
                                                  radius::Float64)
    m = length(idxs)
    m <= 1 && return NTuple{2,Int}[], Float64[]
    edges = NTuple{2,Int}[]
    dists = Float64[]
    sizehint!(edges, min(max(0, 4 * m), 200_000))
    sizehint!(dists, min(max(0, 4 * m), 200_000))
    @inbounds for ai in 1:(m - 1)
        pi = points[idxs[ai]]
        for aj in (ai + 1):m
            d = _euclidean_distance(pi, points[idxs[aj]])
            d <= radius || continue
            push!(edges, (ai, aj))
            push!(dists, d)
        end
    end
    return edges, dists
end

function _point_cloud_edges_within_radius_indexed(points::AbstractMatrix{<:Real},
                                                  idxs::AbstractVector{Int},
                                                  radius::Float64)
    m = length(idxs)
    m <= 1 && return NTuple{2,Int}[], Float64[]
    edges = NTuple{2,Int}[]
    dists = Float64[]
    sizehint!(edges, min(max(0, 4 * m), 200_000))
    sizehint!(dists, min(max(0, 4 * m), 200_000))
    @inbounds for ai in 1:(m - 1)
        i = idxs[ai]
        for aj in (ai + 1):m
            d = _euclidean_distance(points, i, idxs[aj])
            d <= radius || continue
            push!(edges, (ai, aj))
            push!(dists, d)
        end
    end
    return edges, dists
end

@inline _point_cloud_ref_id(points::MatrixRowsView) = UInt(objectid(getfield(points, :mat)))
@inline _point_cloud_ref_id(points::AbstractMatrix{<:Real}) = UInt(objectid(points))
@inline _point_cloud_ref_id(points) = UInt(objectid(points))

@inline function _landmark_radius_cache_key(points,
                                            landmark_hash::UInt,
                                            backend::Symbol,
                                            dim_bucket::Int,
                                            radius::Float64,
                                            approx_candidates::Int)
    return (
        :landmark_radius_subgraph,
        _point_cloud_ref_id(points),
        landmark_hash,
        backend,
        dim_bucket,
        UInt(hash(radius)),
        Int(approx_candidates),
    )
end

function _landmark_radius_subgraph_cached(points::AbstractVector{<:AbstractVector{<:Real}},
                                          landmarks::AbstractVector{Int},
                                          radius::Float64,
                                          spec::FiltrationSpec;
                                          cache::Union{Nothing,EncodingCache}=nothing)
    m = length(landmarks)
    m == 0 && return (edges=NTuple{2,Int}[], dists=Float64[])
    d = length(points[landmarks[1]])
    backend_key = _pointcloud_nn_backend(spec)
    approx_candidates = _pointcloud_nn_approx_candidates(spec)
    lhash = UInt(hash(landmarks))
    key = _landmark_radius_cache_key(
        points,
        lhash,
        backend_key,
        _pointcloud_bucket_value(d),
        radius,
        approx_candidates,
    )
    cached = _get_geometry_cached(cache, key)
    if cached isa NamedTuple && hasproperty(cached, :edges) && hasproperty(cached, :dists)
        return cached
    end
    edges, dists = _point_cloud_edges_within_radius_indexed(points, landmarks, radius)
    return _set_geometry_cached!(cache, key, (edges=edges, dists=dists))
end

function _landmark_radius_subgraph_cached(points::AbstractMatrix{<:Real},
                                          landmarks::AbstractVector{Int},
                                          radius::Float64,
                                          spec::FiltrationSpec;
                                          cache::Union{Nothing,EncodingCache}=nothing)
    m = length(landmarks)
    m == 0 && return (edges=NTuple{2,Int}[], dists=Float64[])
    d = size(points, 2)
    backend_key = _pointcloud_nn_backend(spec)
    approx_candidates = _pointcloud_nn_approx_candidates(spec)
    lhash = UInt(hash(landmarks))
    key = _landmark_radius_cache_key(
        points,
        lhash,
        backend_key,
        _pointcloud_bucket_value(d),
        radius,
        approx_candidates,
    )
    cached = _get_geometry_cached(cache, key)
    if cached isa NamedTuple && hasproperty(cached, :edges) && hasproperty(cached, :dists)
        return cached
    end
    edges, dists = _point_cloud_edges_within_radius_indexed(points, landmarks, radius)
    return _set_geometry_cached!(cache, key, (edges=edges, dists=dists))
end

@inline function _packed_pair_index(n::Int, i::Int, j::Int)
    i == j && throw(ArgumentError("_packed_pair_index expects distinct indices."))
    if i > j
        i, j = j, i
    end
    # 1-based packed upper-triangular index (excluding diagonal).
    return div((i - 1) * (2 * n - i), 2) + (j - i)
end

function _packed_edge_lookup_table(n::Int, edges::AbstractVector{<:Tuple{Int,Int}})
    ne = length(edges)
    keys = Vector{Int}(undef, ne)
    @inbounds for idx in 1:ne
        u, v = edges[idx]
        keys[idx] = _packed_pair_index(n, u, v)
    end
    if issorted(keys)
        return keys, nothing
    end
    perm = sortperm(keys)
    return keys[perm], perm
end

@inline function _packed_edge_lookup_row(sorted_keys::Vector{Int},
                                         perm::Union{Nothing,Vector{Int}},
                                         key::Int)
    pos = searchsortedfirst(sorted_keys, key)
    if pos > length(sorted_keys)
        return 0
    end
    @inbounds sorted_keys[pos] == key || return 0
    return perm === nothing ? pos : @inbounds perm[pos]
end

function _point_cloud_pairwise_packed(points::AbstractVector{<:AbstractVector{<:Real}})
    n = length(points)
    n <= 1 && return Float64[]
    pair_count_big = _combination_count(n, 2)
    pair_count_big > big(typemax(Int)) &&
        throw(ArgumentError("Ingestion combinatorial explosion: pair count C($(n),2)=$(pair_count_big) exceeds representable collection size."))
    pair_count = Int(pair_count_big)
    packed = Vector{Float64}(undef, pair_count)
    t = 1
    @inbounds for i in 1:(n - 1)
        pi = points[i]
        for j in (i + 1):n
            packed[t] = _euclidean_distance(pi, points[j])
            t += 1
        end
    end
    return packed
end

function _point_cloud_pairwise_packed(points::AbstractMatrix{<:Real})
    n = size(points, 1)
    n <= 1 && return Float64[]
    pair_count_big = _combination_count(n, 2)
    pair_count_big > big(typemax(Int)) &&
        throw(ArgumentError("Ingestion combinatorial explosion: pair count C($(n),2)=$(pair_count_big) exceeds representable collection size."))
    pair_count = Int(pair_count_big)
    packed = Vector{Float64}(undef, pair_count)
    t = 1
    @inbounds for i in 1:(n - 1)
        for j in (i + 1):n
            packed[t] = _euclidean_distance(points, i, j)
            t += 1
        end
    end
    return packed
end

@inline function _packed_pair_distance(packed::Vector{Float64}, n::Int, i::Int, j::Int)
    i == j && return 0.0
    @inbounds return packed[_packed_pair_index(n, i, j)]
end

@inline function _simplex_max_pair_distance_points(points::AbstractVector{<:AbstractVector{<:Real}},
                                                   s::Vector{Int})
    m = length(s)
    m <= 1 && return 0.0
    maxd = 0.0
    @inbounds for i in 1:(m - 1)
        pi = points[s[i]]
        for j in (i + 1):m
            d = _euclidean_distance(pi, points[s[j]])
            d > maxd && (maxd = d)
        end
    end
    return maxd
end

function _point_cloud_knn_distances(points::AbstractVector{<:AbstractVector{<:Real}},
                                    k::Int;
                                    backend::Symbol=:auto,
                                    approx_candidates::Int=0)
    n = length(points)
    n > 1 || error("kNN k=$(k) exceeds number of neighbors.")
    k <= n - 1 || error("kNN k=$(k) exceeds number of neighbors.")
    k_eff = min(k, n - 1)
    k_eff > 0 || error("kNN k=$(k) exceeds number of neighbors.")
    d = length(points[1])
    backend0 = _resolve_pointcloud_runtime_backend(backend, n, d, 1)
    if backend0 != :bruteforce
        impl = _POINTCLOUD_KNN_DISTANCES_IMPL[]
        if impl !== nothing
            out = impl(points, k_eff; backend=backend0, approx_candidates=approx_candidates)
            out === nothing || return out
        elseif backend0 == :nearestneighbors || backend0 == :approx
            throw(ArgumentError("PointCloud nn_backend=$(backend0) requires NearestNeighbors extension."))
        end
    end
    _, _, kdist = _knn_graph_bruteforce(points, k_eff)
    return kdist
end

function _point_cloud_knn_distances(points::AbstractMatrix{<:Real},
                                    k::Int;
                                    backend::Symbol=:auto,
                                    approx_candidates::Int=0)
    n = size(points, 1)
    n > 1 || error("kNN k=$(k) exceeds number of neighbors.")
    k <= n - 1 || error("kNN k=$(k) exceeds number of neighbors.")
    k_eff = min(k, n - 1)
    k_eff > 0 || error("kNN k=$(k) exceeds number of neighbors.")
    d = size(points, 2)
    backend0 = _resolve_pointcloud_runtime_backend(backend, n, d, 1)
    if backend0 != :bruteforce
        return _point_cloud_knn_distances(_point_rows_view(points), k_eff;
                                          backend=backend0,
                                          approx_candidates=approx_candidates)
    end
    _, _, kdist = _knn_graph_bruteforce(points, k_eff)
    return kdist
end

@inline function _dtm_neighbor_count(n::Int, dtm_mass::Real)::Int
    mass = Float64(dtm_mass)
    0.0 < mass <= 1.0 || error("dtm_mass must lie in (0, 1], got $(dtm_mass).")
    k = floor(Int, mass * n) + 1
    k <= n || error("dtm_mass=$(mass) yields k=$(k) > n=$(n); choose dtm_mass < 1.")
    return k
end

function _point_cloud_dtm_values(points::AbstractVector{<:AbstractVector{<:Real}},
                                 dtm_mass::Real;
                                 backend::Symbol=:auto,
                                 approx_candidates::Int=0)
    n = length(points)
    n > 1 || error("DTM codensity requires at least 2 points.")
    k_eff = _dtm_neighbor_count(n, dtm_mass)
    d = length(points[1])
    backend0 = _resolve_pointcloud_runtime_backend(backend, n, d, 1)
    if backend0 != :bruteforce
        impl = _POINTCLOUD_DTM_VALUES_IMPL[]
        if impl !== nothing
            out = impl(points, Float64(dtm_mass); backend=backend0, approx_candidates=approx_candidates)
            out === nothing || return out
        elseif backend0 == :nearestneighbors || backend0 == :approx
            throw(ArgumentError("PointCloud nn_backend=$(backend0) requires NearestNeighbors extension."))
        end
    end
    vals = Vector{Float64}(undef, n)
    idxs = fill(0, k_eff)
    dtmp = fill(Inf, k_eff)
    @inbounds for i in 1:n
        fill!(idxs, 0)
        fill!(dtmp, Inf)
        for j in 1:n
            _insert_neighbor_sorted!(idxs, dtmp, j, _euclidean_distance(points[i], points[j]))
        end
        vals[i] = sqrt(sum(abs2, dtmp) / k_eff)
    end
    return vals
end

function _point_cloud_dtm_values(points::AbstractMatrix{<:Real},
                                 dtm_mass::Real;
                                 backend::Symbol=:auto,
                                 approx_candidates::Int=0)
    n = size(points, 1)
    n > 1 || error("DTM codensity requires at least 2 points.")
    k_eff = _dtm_neighbor_count(n, dtm_mass)
    d = size(points, 2)
    backend0 = _resolve_pointcloud_runtime_backend(backend, n, d, 1)
    if backend0 != :bruteforce
        return _point_cloud_dtm_values(_point_rows_view(points), dtm_mass;
                                       backend=backend0,
                                       approx_candidates=approx_candidates)
    end
    vals = Vector{Float64}(undef, n)
    idxs = fill(0, k_eff)
    dtmp = fill(Inf, k_eff)
    @inbounds for i in 1:n
        fill!(idxs, 0)
        fill!(dtmp, Inf)
        for j in 1:n
            _insert_neighbor_sorted!(idxs, dtmp, j, _euclidean_distance(points, i, j))
        end
        vals[i] = sqrt(sum(abs2, dtmp) / k_eff)
    end
    return vals
end

function _knn_distances(points::AbstractVector{<:AbstractVector{<:Real}},
                        k::Int;
                        backend::Symbol=:auto,
                        approx_candidates::Int=0)
    return _point_cloud_knn_distances(points, k; backend=backend, approx_candidates=approx_candidates)
end

function _knn_distances(points::AbstractMatrix{<:Real},
                        k::Int;
                        backend::Symbol=:auto,
                        approx_candidates::Int=0)
    return _point_cloud_knn_distances(points, k; backend=backend, approx_candidates=approx_candidates)
end

@inline function _simplex_aggregate(vals::AbstractVector{<:Real}, agg::Symbol)
    if agg === :max
        return maximum(vals)
    elseif agg === :min
        return minimum(vals)
    elseif agg === :sum
        return sum(vals)
    elseif agg === :mean
        return sum(vals) / length(vals)
    end
    throw(ArgumentError("Unsupported simplex_agg=$(agg). Supported: :max, :min, :sum, :mean"))
end

@inline function _aggregate_pair(a::Float64, b::Float64, agg::Symbol)
    if agg === :max
        return max(a, b)
    elseif agg === :min
        return min(a, b)
    elseif agg === :sum
        return a + b
    elseif agg === :mean
        return (a + b) / 2
    end
    throw(ArgumentError("Unsupported simplex_agg=$(agg). Supported: :max, :min, :sum, :mean"))
end

@inline function _aggregate_triple(a::Float64, b::Float64, c::Float64, agg::Symbol)
    if agg === :max
        return max(a, max(b, c))
    elseif agg === :min
        return min(a, min(b, c))
    elseif agg === :sum
        return a + b + c
    elseif agg === :mean
        return (a + b + c) / 3
    end
    throw(ArgumentError("Unsupported simplex_agg=$(agg). Supported: :max, :min, :sum, :mean"))
end

@inline function _aggregate_tuple_pair(vals::AbstractVector{<:NTuple{N,Float64}},
                                       u::Int, v::Int,
                                       agg::Symbol) where {N}
    a = vals[u]
    b = vals[v]
    if agg === :max
        return ntuple(i -> max(a[i], b[i]), N)
    elseif agg === :min
        return ntuple(i -> min(a[i], b[i]), N)
    elseif agg === :sum
        return ntuple(i -> a[i] + b[i], N)
    elseif agg === :mean
        return ntuple(i -> (a[i] + b[i]) / 2.0, N)
    end
    throw(ArgumentError("Unsupported simplex_agg=$(agg). Supported: :max, :min, :sum, :mean"))
end

@inline function _aggregate_tuple_triple(vals::AbstractVector{<:NTuple{N,Float64}},
                                         aidx::Int, bidx::Int, cidx::Int,
                                         agg::Symbol) where {N}
    a = vals[aidx]
    b = vals[bidx]
    c = vals[cidx]
    if agg === :max
        return ntuple(i -> max(a[i], max(b[i], c[i])), N)
    elseif agg === :min
        return ntuple(i -> min(a[i], min(b[i], c[i])), N)
    elseif agg === :sum
        return ntuple(i -> a[i] + b[i] + c[i], N)
    elseif agg === :mean
        return ntuple(i -> (a[i] + b[i] + c[i]) / 3.0, N)
    end
    throw(ArgumentError("Unsupported simplex_agg=$(agg). Supported: :max, :min, :sum, :mean"))
end

@inline function _aggregate_tuple_indexed(vals::AbstractVector{<:NTuple{N,Float64}},
                                          idxs::AbstractVector{Int},
                                          agg::Symbol) where {N}
    isempty(idxs) && throw(ArgumentError("simplex_aggregate received empty index set"))
    if agg === :max
        return ntuple(i -> begin
            acc = vals[idxs[1]][i]
            @inbounds for t in 2:length(idxs)
                v = vals[idxs[t]][i]
                v > acc && (acc = v)
            end
            acc
        end, N)
    elseif agg === :min
        return ntuple(i -> begin
            acc = vals[idxs[1]][i]
            @inbounds for t in 2:length(idxs)
                v = vals[idxs[t]][i]
                v < acc && (acc = v)
            end
            acc
        end, N)
    elseif agg === :sum
        return ntuple(i -> begin
            acc = 0.0
            @inbounds for t in eachindex(idxs)
                acc += vals[idxs[t]][i]
            end
            acc
        end, N)
    elseif agg === :mean
        invn = 1.0 / length(idxs)
        return ntuple(i -> begin
            acc = 0.0
            @inbounds for t in eachindex(idxs)
                acc += vals[idxs[t]][i]
            end
            acc * invn
        end, N)
    end
    throw(ArgumentError("Unsupported simplex_agg=$(agg). Supported: :max, :min, :sum, :mean"))
end

@inline function _aggregate_indexed(vals::AbstractVector{<:Real},
                                    idxs::AbstractVector{Int},
                                    agg::Symbol)
    isempty(idxs) && throw(ArgumentError("simplex_aggregate received empty index set"))
    if agg === :max
        acc = Float64(vals[idxs[1]])
        @inbounds for t in 2:length(idxs)
            v = Float64(vals[idxs[t]])
            v > acc && (acc = v)
        end
        return acc
    elseif agg === :min
        acc = Float64(vals[idxs[1]])
        @inbounds for t in 2:length(idxs)
            v = Float64(vals[idxs[t]])
            v < acc && (acc = v)
        end
        return acc
    elseif agg === :sum || agg === :mean
        acc = 0.0
        @inbounds for t in eachindex(idxs)
            acc += Float64(vals[idxs[t]])
        end
        return agg === :mean ? acc / length(idxs) : acc
    end
    throw(ArgumentError("Unsupported simplex_agg=$(agg). Supported: :max, :min, :sum, :mean"))
end

function _simplex_aggregate(vals::AbstractVector{<:Tuple}, agg::Symbol)
    isempty(vals) && throw(ArgumentError("simplex_aggregate received empty tuple list"))
    N = length(vals[1])
    out = Vector{Float64}(undef, N)
    for i in 1:N
        coords = Float64[v[i] for v in vals]
        out[i] = _simplex_aggregate(coords, agg)
    end
    return ntuple(i -> out[i], N)
end

function _simplex_aggregate(vals::AbstractVector{<:AbstractVector{<:Real}}, agg::Symbol)
    isempty(vals) && throw(ArgumentError("simplex_aggregate received empty vector list"))
    N = length(vals[1])
    out = Vector{Float64}(undef, N)
    for i in 1:N
        coords = Float64[v[i] for v in vals]
        out[i] = _simplex_aggregate(coords, agg)
    end
    return ntuple(i -> out[i], N)
end

function _point_vertex_values(points::AbstractVector{<:AbstractVector{<:Real}},
                              spec::FiltrationSpec)
    n = length(points)
    if spec.kind == :rips_lowerstar
        coord = Int(get(spec.params, :coord, 1))
        n == 0 && return Float64[]
        1 <= coord <= length(points[1]) ||
            error("rips_lowerstar coordinate $(coord) is out of bounds for ambient dimension $(length(points[1])).")
        return Float64[Float64(points[i][coord]) for i in 1:n]
    end
    if haskey(spec.params, :vertex_values)
        vals = spec.params[:vertex_values]
        length(vals) == n || error("vertex_values length mismatch: expected $(n), got $(length(vals)).")
        return Float64[Float64(v) for v in vals]
    end
    if haskey(spec.params, :vertex_function)
        f = spec.params[:vertex_function]
        return Float64[Float64(f(points[i], i)) for i in 1:n]
    end
    error("function-Rips requires vertex_values or vertex_function.")
end

function _point_vertex_values(points::AbstractMatrix{<:Real},
                              spec::FiltrationSpec)
    n = size(points, 1)
    if spec.kind == :rips_lowerstar
        coord = Int(get(spec.params, :coord, 1))
        1 <= coord <= size(points, 2) ||
            error("rips_lowerstar coordinate $(coord) is out of bounds for ambient dimension $(size(points, 2)).")
        return Float64[Float64(points[i, coord]) for i in 1:n]
    end
    if haskey(spec.params, :vertex_values)
        vals = spec.params[:vertex_values]
        length(vals) == n || error("vertex_values length mismatch: expected $(n), got $(length(vals)).")
        return Float64[Float64(v) for v in vals]
    end
    if haskey(spec.params, :vertex_function)
        f = spec.params[:vertex_function]
        return Float64[Float64(f(view(points, i, :), i)) for i in 1:n]
    end
    error("function-Rips requires vertex_values or vertex_function.")
end

function _point_vertex_values_or_default(points::AbstractVector{<:AbstractVector{<:Real}},
                                         spec::FiltrationSpec;
                                         default::Float64=0.0)
    if haskey(spec.params, :vertex_values) || haskey(spec.params, :vertex_function)
        return _point_vertex_values(points, spec)
    end
    return fill(default, length(points))
end

@inline function _pointcloud_dtm_mass(spec::FiltrationSpec)::Float64
    mass = Float64(get(spec.params, :dtm_mass, 0.1))
    0.0 < mass <= 1.0 || error("dtm_mass must lie in (0, 1], got $(mass).")
    return mass
end

@inline function _point_codensity_values(points::AbstractVector{<:AbstractVector{<:Real}},
                                         spec::FiltrationSpec)
    return _point_cloud_dtm_values(points, _pointcloud_dtm_mass(spec);
                                   backend=_pointcloud_nn_backend(spec),
                                   approx_candidates=_pointcloud_nn_approx_candidates(spec))
end

@inline function _point_codensity_values(points::AbstractMatrix{<:Real},
                                         spec::FiltrationSpec)
    return _point_cloud_dtm_values(points, _pointcloud_dtm_mass(spec);
                                   backend=_pointcloud_nn_backend(spec),
                                   approx_candidates=_pointcloud_nn_approx_candidates(spec))
end

"""
    point_codensity(data::PointCloud, filtration::AbstractFiltration) -> PointCodensityResult
    point_codensity(data::PointCloud, spec::FiltrationSpec) -> PointCodensityResult

Compute the pointwise DTM-style codensity values used by the canonical
`:rips_codensity` ingestion family and return them as an inspectable
[`PointCodensityResult`](@ref).

Use this when you want the codensity values themselves for notebook-side plots
or diagnostics before asking for heavier encoded/cohomological stages.
"""
function point_codensity(data::PointCloud, filtration::AbstractFiltration)
    filtration_kind(typeof(filtration)) === :rips_codensity ||
        throw(ArgumentError("point_codensity(data, filtration) requires a :rips_codensity filtration, got $(filtration_kind(typeof(filtration)))."))
    spec = _filtration_spec(filtration)
    vals = _point_codensity_values(point_matrix(data), spec)
    k = _dtm_neighbor_count(DataTypes.npoints(data), _pointcloud_dtm_mass(spec))
    return PointCodensityResult(data, filtration, vals, k)
end

function point_codensity(data::PointCloud, spec::FiltrationSpec)
    spec.kind === :rips_codensity ||
        throw(ArgumentError("point_codensity(data, spec) requires FiltrationSpec(kind=:rips_codensity, ...), got kind=$(spec.kind)."))
    vals = _point_codensity_values(point_matrix(data), spec)
    k = _dtm_neighbor_count(DataTypes.npoints(data), _pointcloud_dtm_mass(spec))
    return PointCodensityResult(data, spec, vals, k)
end

function _codensity_function_rips_spec(data::PointCloud, spec::FiltrationSpec)
    p = spec.params
    vals = _point_codensity_values(point_matrix(data), spec)
    params = _params_with_nonnothing((;
        max_dim=Int(get(p, :max_dim, 1)),
        radius=get(p, :radius, nothing),
        knn=get(p, :knn, nothing),
        n_landmarks=get(p, :n_landmarks, nothing),
        nn_backend=Symbol(get(p, :nn_backend, :auto)),
        nn_approx_candidates=Int(get(p, :nn_approx_candidates, 0)),
        vertex_values=vals,
        simplex_agg=:max,
        construction=_construction_from_params(p),
    ))
    return FiltrationSpec(; kind=:function_rips, params...)
end

@inline function _orient2d(a::AbstractVector{<:Real}, b::AbstractVector{<:Real}, c::AbstractVector{<:Real})
    return (Float64(b[1]) - Float64(a[1])) * (Float64(c[2]) - Float64(a[2])) -
           (Float64(b[2]) - Float64(a[2])) * (Float64(c[1]) - Float64(a[1]))
end

function _circumcircle_2d(a::AbstractVector{<:Real},
                          b::AbstractVector{<:Real},
                          c::AbstractVector{<:Real}; atol::Float64=1e-12)
    ax, ay = Float64(a[1]), Float64(a[2])
    bx, by = Float64(b[1]), Float64(b[2])
    cx, cy = Float64(c[1]), Float64(c[2])
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    abs(d) <= atol && return nothing
    aa = ax * ax + ay * ay
    bb = bx * bx + by * by
    cc = cx * cx + cy * cy
    ux = (aa * (by - cy) + bb * (cy - ay) + cc * (ay - by)) / d
    uy = (aa * (cx - bx) + bb * (ax - cx) + cc * (bx - ax)) / d
    r2 = (ux - ax)^2 + (uy - ay)^2
    return (ux, uy, r2)
end

@inline function _sort_triplet(i::Int, j::Int, k::Int)::NTuple{3,Int}
    a = min(i, min(j, k))
    c = max(i, max(j, k))
    b = i + j + k - a - c
    return (a, b, c)
end

function _packed_delaunay_simplices_1d(points::AbstractVector{<:AbstractVector{<:Real}}; max_dim::Int=2)
    n = length(points)
    order = sortperm(1:n; by=i -> (Float64(points[i][1]), i))
    nedges = max(0, n - 1)
    edges = Vector{NTuple{2,Int}}(undef, nedges)
    edge_radius = Vector{Float64}(undef, nedges)
    @inbounds for t in 1:nedges
        i = order[t]
        j = order[t + 1]
        a, b = min(i, j), max(i, j)
        edges[t] = (a, b)
        edge_radius[t] = _euclidean_distance(points[a], points[b]) / 2
    end
    triangles = NTuple{3,Int}[]
    tri_radius = Float64[]
    return _PackedDelaunay2D(edges, edge_radius, triangles, tri_radius)
end

@inline function _push_delaunay_edge!(edges::Vector{NTuple{2,Int}},
                                      edge_radius::Vector{Float64},
                                      edge_seen::BitMatrix,
                                      points::AbstractVector{<:AbstractVector{<:Real}},
                                      u::Int, v::Int)
    a, b = min(u, v), max(u, v)
    @inbounds if edge_seen[a, b]
        return nothing
    end
    @inbounds edge_seen[a, b] = true
    push!(edges, (a, b))
    push!(edge_radius, _euclidean_distance(points[a], points[b]) / 2)
    return nothing
end

function _packed_delaunay_simplices_2d_naive(points::AbstractVector{<:AbstractVector{<:Real}}; max_dim::Int=2)
    n = length(points)
    edges = NTuple{2,Int}[]
    edge_radius = Float64[]
    triangles = NTuple{3,Int}[]
    tri_radius = Float64[]
    sizehint!(edges, max(0, 3n))
    sizehint!(edge_radius, max(0, 3n))
    max_dim >= 2 && sizehint!(triangles, max(0, 2n))
    max_dim >= 2 && sizehint!(tri_radius, max(0, 2n))
    edge_seen = falses(n, n)
    tol = 1e-10

    if max_dim >= 2
        @inbounds for i in 1:n, j in (i + 1):n, k in (j + 1):n
            cc = _circumcircle_2d(points[i], points[j], points[k]; atol=tol)
            cc === nothing && continue
            ux, uy, r2 = cc
            empty_ball = true
            for t in 1:n
                (t == i || t == j || t == k) && continue
                px = Float64(points[t][1])
                py = Float64(points[t][2])
                d2 = (px - ux)^2 + (py - uy)^2
                if d2 < r2 - tol
                    empty_ball = false
                    break
                end
            end
            empty_ball || continue
            push!(triangles, (i, j, k))
            push!(tri_radius, sqrt(r2))
            _push_delaunay_edge!(edges, edge_radius, edge_seen, points, i, j)
            _push_delaunay_edge!(edges, edge_radius, edge_seen, points, i, k)
            _push_delaunay_edge!(edges, edge_radius, edge_seen, points, j, k)
        end
    end

    # Ensure convex-hull edges are included.
    @inbounds for i in 1:n, j in (i + 1):n
        edge_seen[i, j] && continue
        pos = false
        neg = false
        for t in 1:n
            (t == i || t == j) && continue
            o = _orient2d(points[i], points[j], points[t])
            if o > tol
                pos = true
            elseif o < -tol
                neg = true
            end
            (pos && neg) && break
        end
        if !(pos && neg)
            _push_delaunay_edge!(edges, edge_radius, edge_seen, points, i, j)
        end
    end

    if !isempty(edges)
        perm = sortperm(edges; by=e -> (e[1], e[2]))
        edges = edges[perm]
        edge_radius = edge_radius[perm]
    end
    if !isempty(triangles)
        perm_t = sortperm(triangles; by=t -> (t[1], t[2], t[3]))
        triangles = triangles[perm_t]
        tri_radius = tri_radius[perm_t]
    end
    return _PackedDelaunay2D(edges, edge_radius, triangles, tri_radius)
end

@inline function _delaunay_cache_key(points::AbstractVector{<:AbstractVector{<:Real}},
                                     max_dim::Int,
                                     backend::Symbol)
    return (UInt64(objectid(points)), UInt64(length(points)), UInt64(max_dim), UInt64(hash(backend)))
end

@inline function _as_packed_delaunay_entry(out::Any)
    if out isa _PackedDelaunay2DCacheEntry
        return out
    elseif out isa _PackedDelaunay2D
        return _PackedDelaunay2DCacheEntry(out, nothing, nothing)
    else
        return nothing
    end
end

function _cached_packed_delaunay_entry_2d(points::AbstractVector{<:AbstractVector{<:Real}},
                                          max_dim::Int,
                                          backend::Symbol)
    if !_POINTCLOUD_DELAUNAY_CACHE_ENABLED[]
        packed = _packed_delaunay_simplices_2d(points; max_dim=max_dim, backend=backend)
        return _PackedDelaunay2DCacheEntry(packed, nothing, nothing)
    end
    key = _delaunay_cache_key(points, max_dim, backend)
    cached = lock(_POINTCLOUD_DELAUNAY_CACHE_LOCK) do
        get(_POINTCLOUD_DELAUNAY_CACHE, key, nothing)
    end
    entry = _as_packed_delaunay_entry(cached)
    if entry !== nothing
        if cached !== entry
            lock(_POINTCLOUD_DELAUNAY_CACHE_LOCK) do
                _POINTCLOUD_DELAUNAY_CACHE[key] = entry
            end
        end
        return entry
    end

    packed = _packed_delaunay_simplices_2d(points; max_dim=max_dim, backend=backend)
    entry = _PackedDelaunay2DCacheEntry(packed, nothing, nothing)
    return lock(_POINTCLOUD_DELAUNAY_CACHE_LOCK) do
        existing = get(_POINTCLOUD_DELAUNAY_CACHE, key, nothing)
        existing_entry = _as_packed_delaunay_entry(existing)
        if existing_entry !== nothing
            if existing !== existing_entry
                _POINTCLOUD_DELAUNAY_CACHE[key] = existing_entry
            end
            return existing_entry
        end
        _POINTCLOUD_DELAUNAY_CACHE[key] = entry
        push!(_POINTCLOUD_DELAUNAY_CACHE_ORDER, key)
        max_keep = max(1, _POINTCLOUD_DELAUNAY_CACHE_MAX[])
        while length(_POINTCLOUD_DELAUNAY_CACHE_ORDER) > max_keep
            old = popfirst!(_POINTCLOUD_DELAUNAY_CACHE_ORDER)
            delete!(_POINTCLOUD_DELAUNAY_CACHE, old)
        end
        return entry
    end
end

function _ensure_packed_delaunay_boundaries!(entry::_PackedDelaunay2DCacheEntry,
                                             n::Int,
                                             max_dim::Int)
    if max_dim >= 1 && entry.edge_boundary === nothing
        entry.edge_boundary = _edge_boundary_matrix(n, entry.packed.edges)
    end
    if max_dim >= 2 && entry.tri_boundary === nothing
        entry.tri_boundary = _triangle_boundary_matrix_from_edges(n, entry.packed.edges, entry.packed.triangles)
    end
    return entry
end

function _packed_delaunay_simplices_2d(points::AbstractVector{<:AbstractVector{<:Real}};
                                       max_dim::Int=2,
                                       backend::Symbol=:auto)
    if backend != :naive && _have_pointcloud_delaunay_backend()
        impl = _POINTCLOUD_DELAUNAY_2D_IMPL[]
        out = impl === nothing ? nothing : impl(points; max_dim=max_dim)
        if out !== nothing
            return out
        end
    end
    return _packed_delaunay_simplices_2d_naive(points; max_dim=max_dim)
end

function _packed_delaunay_entry(points::AbstractVector{<:AbstractVector{<:Real}},
                                spec::FiltrationSpec;
                                max_dim::Int=2)
    n = length(points)
    n == 0 && error("PointCloud has no points.")
    d = length(points[1])
    if d == 1
        packed = _packed_delaunay_simplices_1d(points; max_dim=max_dim)
        return _PackedDelaunay2DCacheEntry(packed, nothing, nothing)
    elseif d == 2
        backend = _pointcloud_delaunay_backend(spec)
        return _cached_packed_delaunay_entry_2d(points, max_dim, backend)
    else
        error("Delaunay filtrations currently support only 1D/2D point clouds (got dimension $d).")
    end
end

@inline function _packed_delaunay_simplices(points::AbstractVector{<:AbstractVector{<:Real}},
                                            spec::FiltrationSpec;
                                            max_dim::Int=2)
    return _packed_delaunay_entry(points, spec; max_dim=max_dim).packed
end

@inline function _point_angle_obtuse(points::AbstractVector{<:AbstractVector{<:Real}},
                                     i::Int,
                                     j::Int,
                                     k::Int;
                                     tol::Float64=1e-12)
    pk = points[k]
    dotv = 0.0
    @inbounds for d in eachindex(pk)
        dotv += (Float64(points[i][d]) - Float64(pk[d])) *
                (Float64(points[j][d]) - Float64(pk[d]))
    end
    return dotv < -tol
end

function _alpha_edge_grades_sq(points::AbstractVector{<:AbstractVector{<:Real}},
                               packed::_PackedDelaunay2D)
    ne = length(packed.edges)
    ne == 0 && return Float64[]

    edge_sq = Vector{Float64}(undef, ne)
    @inbounds for idx in 1:ne
        r = packed.edge_radius[idx]
        edge_sq[idx] = r * r
    end
    isempty(packed.triangles) && return edge_sq

    n = length(points)
    edge_keys, edge_perm = _packed_edge_lookup_table(n, packed.edges)

    incident_min_r2 = fill(Inf, ne)
    nongabriel = falses(ne)

    @inbounds for tidx in eachindex(packed.triangles)
        i, j, k = packed.triangles[tidx]
        tri_r = packed.tri_radius[tidx]
        tri_r2 = tri_r * tri_r

        ij_idx = _packed_edge_lookup_row(edge_keys, edge_perm, _packed_pair_index(n, i, j))
        ik_idx = _packed_edge_lookup_row(edge_keys, edge_perm, _packed_pair_index(n, i, k))
        jk_idx = _packed_edge_lookup_row(edge_keys, edge_perm, _packed_pair_index(n, j, k))
        ij_idx == 0 && error("alpha edge grading: missing edge ($(i),$(j)) for triangle ($(i),$(j),$(k)).")
        ik_idx == 0 && error("alpha edge grading: missing edge ($(i),$(k)) for triangle ($(i),$(j),$(k)).")
        jk_idx == 0 && error("alpha edge grading: missing edge ($(j),$(k)) for triangle ($(i),$(j),$(k)).")

        incident_min_r2[ij_idx] = min(incident_min_r2[ij_idx], tri_r2)
        incident_min_r2[ik_idx] = min(incident_min_r2[ik_idx], tri_r2)
        incident_min_r2[jk_idx] = min(incident_min_r2[jk_idx], tri_r2)

        nongabriel[ij_idx] |= _point_angle_obtuse(points, i, j, k)
        nongabriel[ik_idx] |= _point_angle_obtuse(points, i, k, j)
        nongabriel[jk_idx] |= _point_angle_obtuse(points, j, k, i)
    end

    @inbounds for idx in 1:ne
        if nongabriel[idx] && isfinite(incident_min_r2[idx])
            edge_sq[idx] = incident_min_r2[idx]
        end
    end
    return edge_sq
end

function _core_numbers(n::Int, edges::AbstractVector{<:Tuple{Int,Int}})
    n <= 0 && return Int[]

    adj = [Int[] for _ in 1:n]
    deg = zeros(Int, n)
    @inbounds for (u, v) in edges
        u == v && continue
        push!(adj[u], v)
        push!(adj[v], u)
        deg[u] += 1
        deg[v] += 1
    end
    isempty(deg) && return deg
    maxdeg = maximum(deg)
    maxdeg == 0 && return deg

    # Batagelj-Zaversnik linear-time k-core decomposition.
    bin = zeros(Int, maxdeg + 1) # bin[d+1] stores bucket start for degree d.
    @inbounds for v in 1:n
        bin[deg[v] + 1] += 1
    end
    start = 1
    @inbounds for d in 0:maxdeg
        c = bin[d + 1]
        bin[d + 1] = start
        start += c
    end

    pos = Vector{Int}(undef, n)
    vert = Vector{Int}(undef, n)
    @inbounds for v in 1:n
        d = deg[v]
        p = bin[d + 1]
        pos[v] = p
        vert[p] = v
        bin[d + 1] = p + 1
    end
    @inbounds for d in maxdeg:-1:1
        bin[d + 1] = bin[d]
    end
    bin[1] = 1

    core = copy(deg)
    @inbounds for i in 1:n
        v = vert[i]
        dv = core[v]
        for u in adj[v]
            du = core[u]
            du > dv || continue
            pu = pos[u]
            pw = bin[du + 1]
            w = vert[pw]
            if u != w
                vert[pu] = w
                pos[w] = pu
                vert[pw] = u
                pos[u] = pw
            end
            bin[du + 1] += 1
            core[u] = du - 1
        end
    end
    return core
end

function _core_edges_from_point_cloud(points::AbstractVector{<:AbstractVector{<:Real}},
                                      spec::FiltrationSpec)
    n = length(points)
    n <= 1 && return NTuple{2,Int}[]
    backend = _pointcloud_nn_backend(spec)
    approx_candidates = _pointcloud_nn_approx_candidates(spec)
    if haskey(spec.params, :radius)
        r = Float64(spec.params[:radius])
        edges = _point_cloud_radius_edges(points, r; backend=backend, approx_candidates=approx_candidates)
        return edges
    end
    k = Int(get(spec.params, :knn, 8))
    k = max(1, min(k, n - 1))
    edges = _point_cloud_knn_edges(points, k; backend=backend, approx_candidates=approx_candidates)
    return edges
end

function _core_edges_from_point_cloud(points::AbstractMatrix{<:Real},
                                      spec::FiltrationSpec)
    n = size(points, 1)
    n <= 1 && return NTuple{2,Int}[]
    backend = _pointcloud_nn_backend(spec)
    approx_candidates = _pointcloud_nn_approx_candidates(spec)
    if haskey(spec.params, :radius)
        r = Float64(spec.params[:radius])
        return _point_cloud_radius_edges(points, r; backend=backend, approx_candidates=approx_candidates)
    end
    k = Int(get(spec.params, :knn, 8))
    k = max(1, min(k, n - 1))
    return _point_cloud_knn_edges(points, k; backend=backend, approx_candidates=approx_candidates)
end

function _point_graph_scalar_values(points::AbstractVector{<:AbstractVector{<:Real}}, spec::FiltrationSpec)
    if haskey(spec.params, :vertex_values) || haskey(spec.params, :vertex_function)
        return _point_vertex_values(points, spec)
    end
    return fill(0.0, length(points))
end

function _graph_vertex_scalar_values(data::GraphData, spec::FiltrationSpec; required::Bool=false)
    if required
        vals = _graph_vertex_values(data, spec)
    else
        if haskey(spec.params, :vertex_grades) || haskey(spec.params, :vertex_values) || haskey(spec.params, :vertex_function)
            vals = _graph_vertex_values(data, spec)
        else
            return fill(0.0, data.n)
        end
    end
    out = Vector{Float64}(undef, length(vals))
    for i in eachindex(vals)
        vi = vals[i]
        if vi isa Tuple
            length(vi) == 1 || error("Expected scalar vertex values for this filtration.")
            out[i] = Float64(vi[1])
        else
            out[i] = Float64(vi)
        end
    end
    return out
end

function _graph_vertex_values(data::GraphData, spec::FiltrationSpec)
    _as_float_tuple(x) = x isa Number ? (Float64(x),) :
                         x isa Tuple ? ntuple(i -> Float64(x[i]), length(x)) :
                         x isa AbstractVector ? ntuple(i -> Float64(x[i]), length(x)) :
                         throw(ArgumentError("vertex values/function outputs must be numbers, tuples, or vectors."))
    n = data.n
    if haskey(spec.params, :vertex_grades)
        vg = spec.params[:vertex_grades]
        length(vg) == n || error("vertex_grades length mismatch.")
        return [_as_float_tuple(v) for v in vg]
    end
    if haskey(spec.params, :vertex_values)
        vv = spec.params[:vertex_values]
        length(vv) == n || error("vertex_values length mismatch.")
        return [_as_float_tuple(v) for v in vv]
    end
    if haskey(spec.params, :vertex_function)
        f = spec.params[:vertex_function]
        vals = Vector{Tuple}(undef, n)
        for i in 1:n
            arg = data.coords === nothing ? i : data.coords[i]
            vals[i] = _as_float_tuple(f(arg, i))
        end
        return vals
    end
    error("graph filtration requires vertex_grades, vertex_values, or vertex_function.")
end

@inline function _graph_metric(spec::FiltrationSpec)
    metric = Symbol(get(spec.params, :metric, :hop))
    (metric == :hop || metric == :weighted) ||
        throw(ArgumentError("graph filtration metric must be :hop or :weighted."))
    return metric
end

@inline function _graph_lift(spec::FiltrationSpec; default::Symbol=:lower_star)
    lift = Symbol(get(spec.params, :lift, default))
    return lift
end

function _graph_edge_weights(data::GraphData, spec::FiltrationSpec; required::Bool=false)
    if haskey(spec.params, :edge_weights)
        ew = spec.params[:edge_weights]
        length(ew) == length(data.edges) || error("edge_weights length mismatch.")
        return Float64[Float64(w) for w in ew]
    end
    if data.weights !== nothing
        length(data.weights) == length(data.edges) || error("GraphData weights length mismatch.")
        return Float64[Float64(w) for w in data.weights]
    end
    required && error("graph filtration requires edge_weights (or GraphData.weights) for metric=:weighted.")
    return fill(1.0, length(data.edges))
end

@inline function _graph_edge_pair_key(u::Int, v::Int)
    a = u
    b = v
    if b < a
        a, b = b, a
    end
    return (UInt64(a) << 32) | UInt64(b)
end

function _graph_edge_weight_lookup_table(edges::AbstractVector{<:Tuple{Int,Int}},
                                         weights::AbstractVector{<:Real})
    length(weights) == length(edges) || error("edge_weights length mismatch.")
    tbl = Dict{UInt64,Float64}()
    sizehint!(tbl, length(edges))
    @inbounds for idx in eachindex(edges)
        u, v = edges[idx]
        tbl[_graph_edge_pair_key(u, v)] = Float64(weights[idx])
    end
    return tbl
end

@inline function _graph_edge_weight_lookup(tbl::Dict{UInt64,Float64}, u::Int, v::Int)
    return get(tbl, _graph_edge_pair_key(u, v), Inf)
end

function _graph_adj_unweighted(n::Int, edges::AbstractVector{<:Tuple{Int,Int}})
    adj = [Int[] for _ in 1:n]
    for (u, v) in edges
        push!(adj[u], v)
        push!(adj[v], u)
    end
    return adj
end

function _graph_adj_weighted(n::Int, edges::AbstractVector{<:Tuple{Int,Int}}, weights::AbstractVector{<:Real})
    length(weights) == length(edges) || error("edge_weights length mismatch.")
    adj = [Vector{Tuple{Int,Float64}}() for _ in 1:n]
    for idx in eachindex(edges)
        u, v = edges[idx]
        w = Float64(weights[idx])
        w < 0 && error("weighted graph metric requires nonnegative edge weights.")
        push!(adj[u], (v, w))
        push!(adj[v], (u, w))
    end
    return adj
end

function _single_source_hop_dist(adj::Vector{Vector{Int}}, s::Int)
    n = length(adj)
    dist = fill(Inf, n)
    q = Int[s]
    head = 1
    dist[s] = 0.0
    while head <= length(q)
        v = q[head]
        head += 1
        dv = dist[v]
        for w in adj[v]
            if !isfinite(dist[w])
                dist[w] = dv + 1.0
                push!(q, w)
            end
        end
    end
    return dist
end

function _multi_source_hop_dist(adj::Vector{Vector{Int}}, sources::Vector{Int})
    n = length(adj)
    dist = fill(Inf, n)
    q = Int[]
    for s in sources
        if !isfinite(dist[s])
            dist[s] = 0.0
            push!(q, s)
        end
    end
    head = 1
    while head <= length(q)
        v = q[head]
        head += 1
        dv = dist[v]
        for w in adj[v]
            if !isfinite(dist[w])
                dist[w] = dv + 1.0
                push!(q, w)
            end
        end
    end
    return dist
end

function _single_source_weighted_dist(adj::Vector{Vector{Tuple{Int,Float64}}}, s::Int)
    n = length(adj)
    dist = fill(Inf, n)
    used = falses(n)
    dist[s] = 0.0
    for _ in 1:n
        v = 0
        best = Inf
        @inbounds for i in 1:n
            if !used[i] && dist[i] < best
                best = dist[i]
                v = i
            end
        end
        v == 0 && break
        used[v] = true
        dv = dist[v]
        @inbounds for (w, wt) in adj[v]
            nd = dv + wt
            nd < dist[w] && (dist[w] = nd)
        end
    end
    return dist
end

function _multi_source_weighted_dist(adj::Vector{Vector{Tuple{Int,Float64}}}, sources::Vector{Int})
    n = length(adj)
    dist = fill(Inf, n)
    used = falses(n)
    for s in sources
        dist[s] = 0.0
    end
    for _ in 1:n
        v = 0
        best = Inf
        @inbounds for i in 1:n
            if !used[i] && dist[i] < best
                best = dist[i]
                v = i
            end
        end
        v == 0 && break
        used[v] = true
        dv = dist[v]
        @inbounds for (w, wt) in adj[v]
            nd = dv + wt
            nd < dist[w] && (dist[w] = nd)
        end
    end
    return dist
end

function _replace_inf_with_large!(vals::Vector{Float64})
    maxf = 0.0
    for x in vals
        if isfinite(x) && x > maxf
            maxf = x
        end
    end
    repl = maxf + 1.0
    for i in eachindex(vals)
        isfinite(vals[i]) || (vals[i] = repl)
    end
    return vals
end

function _graph_closeness_centrality_hop(adj::Vector{Vector{Int}})
    n = length(adj)
    out = zeros(Float64, n)
    for s in 1:n
        d = _single_source_hop_dist(adj, s)
        reach = 0
        sdist = 0.0
        for i in 1:n
            if i != s && isfinite(d[i])
                reach += 1
                sdist += d[i]
            end
        end
        out[s] = (reach > 0 && sdist > 0.0) ? (reach / sdist) : 0.0
    end
    return out
end

function _graph_closeness_centrality_weighted(adj::Vector{Vector{Tuple{Int,Float64}}})
    n = length(adj)
    out = zeros(Float64, n)
    for s in 1:n
        d = _single_source_weighted_dist(adj, s)
        reach = 0
        sdist = 0.0
        for i in 1:n
            if i != s && isfinite(d[i])
                reach += 1
                sdist += d[i]
            end
        end
        out[s] = (reach > 0 && sdist > 0.0) ? (reach / sdist) : 0.0
    end
    return out
end

function _graph_betweenness_centrality_hop(adj::Vector{Vector{Int}})
    n = length(adj)
    bc = zeros(Float64, n)
    for s in 1:n
        S = Int[]
        P = [Int[] for _ in 1:n]
        sigma = zeros(Float64, n)
        sigma[s] = 1.0
        dist = fill(-1, n)
        dist[s] = 0
        Q = Int[s]
        head = 1
        while head <= length(Q)
            v = Q[head]
            head += 1
            push!(S, v)
            for w in adj[v]
                if dist[w] < 0
                    push!(Q, w)
                    dist[w] = dist[v] + 1
                end
                if dist[w] == dist[v] + 1
                    sigma[w] += sigma[v]
                    push!(P[w], v)
                end
            end
        end
        delta = zeros(Float64, n)
        while !isempty(S)
            w = pop!(S)
            for v in P[w]
                sigma[w] > 0 && (delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]))
            end
            w != s && (bc[w] += delta[w])
        end
    end
    return bc ./ 2.0
end

function _graph_betweenness_centrality_weighted(adj::Vector{Vector{Tuple{Int,Float64}}})
    n = length(adj)
    bc = zeros(Float64, n)
    eps = 1e-12
    for s in 1:n
        P = [Int[] for _ in 1:n]
        sigma = zeros(Float64, n)
        sigma[s] = 1.0
        dist = fill(Inf, n)
        dist[s] = 0.0
        used = falses(n)
        S = Int[]
        for _ in 1:n
            v = 0
            best = Inf
            for i in 1:n
                if !used[i] && dist[i] < best
                    best = dist[i]
                    v = i
                end
            end
            v == 0 && break
            used[v] = true
            push!(S, v)
            dv = dist[v]
            for (w, wt) in adj[v]
                nd = dv + wt
                if nd < dist[w] - eps
                    dist[w] = nd
                    sigma[w] = sigma[v]
                    empty!(P[w])
                    push!(P[w], v)
                elseif abs(nd - dist[w]) <= eps
                    sigma[w] += sigma[v]
                    push!(P[w], v)
                end
            end
        end
        delta = zeros(Float64, n)
        while !isempty(S)
            w = pop!(S)
            for v in P[w]
                sigma[w] > 0 && (delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]))
            end
            w != s && (bc[w] += delta[w])
        end
    end
    return bc ./ 2.0
end

function _graph_pagerank(data::GraphData, weights::Vector{Float64}; weighted::Bool,
                         damping::Float64=0.85, maxiter::Int=200, tol::Float64=1e-10)
    n = data.n
    adj = [Int[] for _ in 1:n]
    wadj = [Float64[] for _ in 1:n]
    for idx in eachindex(data.edges)
        u, v = data.edges[idx]
        w = weighted ? weights[idx] : 1.0
        push!(adj[u], v); push!(wadj[u], w)
        push!(adj[v], u); push!(wadj[v], w)
    end
    outw = zeros(Float64, n)
    for i in 1:n
        outw[i] = sum(wadj[i])
    end
    pr = fill(1.0 / n, n)
    for _ in 1:maxiter
        new = fill((1.0 - damping) / n, n)
        dangling = 0.0
        for i in 1:n
            if outw[i] <= 0.0
                dangling += pr[i]
                continue
            end
            pi = damping * pr[i] / outw[i]
            for t in eachindex(adj[i])
                new[adj[i][t]] += pi * wadj[i][t]
            end
        end
        if dangling > 0.0
            c = damping * dangling / n
            for i in 1:n
                new[i] += c
            end
        end
        if maximum(abs.(new .- pr)) <= tol
            return new
        end
        pr = new
    end
    return pr
end

function _graph_eigenvector_centrality(data::GraphData, weights::Vector{Float64}; weighted::Bool,
                                       maxiter::Int=500, tol::Float64=1e-10)
    n = data.n
    x = fill(1.0 / sqrt(n), n)
    for _ in 1:maxiter
        y = zeros(Float64, n)
        for idx in eachindex(data.edges)
            u, v = data.edges[idx]
            w = weighted ? weights[idx] : 1.0
            y[u] += w * x[v]
            y[v] += w * x[u]
        end
        ny = norm(y)
        ny > 0 || return zeros(Float64, n)
        y ./= ny
        if norm(y .- x) <= tol
            return abs.(y)
        end
        x = y
    end
    return abs.(x)
end

function _graph_centrality_values(data::GraphData, spec::FiltrationSpec)
    cent = Symbol(get(spec.params, :centrality, :degree))
    metric = _graph_metric(spec)
    weighted = metric == :weighted
    weights = _graph_edge_weights(data, spec; required=weighted)
    n = data.n
    if cent == :degree
        vals = zeros(Float64, n)
        for idx in eachindex(data.edges)
            u, v = data.edges[idx]
            w = weighted ? weights[idx] : 1.0
            vals[u] += w
            vals[v] += w
        end
        return vals
    elseif cent == :closeness
        if weighted
            return _graph_closeness_centrality_weighted(_graph_adj_weighted(n, data.edges, weights))
        end
        return _graph_closeness_centrality_hop(_graph_adj_unweighted(n, data.edges))
    elseif cent == :betweenness
        if weighted
            return _graph_betweenness_centrality_weighted(_graph_adj_weighted(n, data.edges, weights))
        end
        return _graph_betweenness_centrality_hop(_graph_adj_unweighted(n, data.edges))
    elseif cent == :pagerank
        return _graph_pagerank(data, weights; weighted=weighted)
    elseif cent == :eigenvector
        return _graph_eigenvector_centrality(data, weights; weighted=weighted)
    end
    throw(ArgumentError("Unsupported centrality=$(cent). Supported: :degree, :closeness, :betweenness, :pagerank, :eigenvector"))
end

function _graph_sources(spec::FiltrationSpec, n::Int)
    haskey(spec.params, :sources) || error("graph_geodesic filtration requires sources.")
    raw = spec.params[:sources]
    src = Int[raw...]
    isempty(src) && error("graph geodesic filtration requires at least one source.")
    for s in src
        (1 <= s <= n) || error("graph source index out of bounds: $(s) not in 1:$(n)")
    end
    return unique(src)
end

function _graph_geodesic_values(data::GraphData, spec::FiltrationSpec)
    metric = _graph_metric(spec)
    n = data.n
    src = _graph_sources(spec, n)
    if metric == :hop
        d = _multi_source_hop_dist(_graph_adj_unweighted(n, data.edges), src)
    else
        w = _graph_edge_weights(data, spec; required=true)
        d = _multi_source_weighted_dist(_graph_adj_weighted(n, data.edges, w), src)
    end
    return _replace_inf_with_large!(d)
end

function _graph_adjacency_lists_sorted(n::Int, edges::AbstractVector{<:Tuple{Int,Int}})
    adj = [Int[] for _ in 1:n]
    @inbounds for (u, v) in edges
        push!(adj[u], v)
        push!(adj[v], u)
    end
    @inbounds for i in 1:n
        sort!(adj[i])
    end
    return adj
end

function _clear_graph_backend_winner_cache!()
    lock(_GRAPH_BACKEND_WINNER_CACHE_LOCK) do
        empty!(_GRAPH_BACKEND_WINNER_CACHE)
        empty!(_GRAPH_BACKEND_WINNER_CACHE_ORDER)
    end
    return nothing
end

function _graph_backend_winner_cache_size()
    lock(_GRAPH_BACKEND_WINNER_CACHE_LOCK) do
        return length(_GRAPH_BACKEND_WINNER_CACHE)
    end
end

struct _PackedEdgeList
    n::Int
    edges::Vector{NTuple{2,Int}}
    offsets::Vector{Int}
    nbrs::Vector{Int}
end

function _pack_edge_list(n::Int, edges::AbstractVector{<:Tuple{Int,Int}})
    n >= 0 || throw(ArgumentError("_pack_edge_list: n must be nonnegative"))
    pedges = Vector{NTuple{2,Int}}(undef, length(edges))
    counts = zeros(Int, n)
    @inbounds for i in eachindex(edges)
        u0, v0 = edges[i]
        u = Int(u0)
        v = Int(v0)
        (1 <= u <= n && 1 <= v <= n && u != v) ||
            throw(ArgumentError("_pack_edge_list: invalid edge ($u,$v) for n=$n"))
        if v < u
            u, v = v, u
        end
        pedges[i] = (u, v)
        counts[u] += 1
        counts[v] += 1
    end
    offsets = Vector{Int}(undef, n + 1)
    offsets[1] = 1
    @inbounds for i in 1:n
        offsets[i + 1] = offsets[i] + counts[i]
    end
    nbrs = Vector{Int}(undef, max(0, offsets[end] - 1))
    cursor = copy(offsets)
    @inbounds for (u, v) in pedges
        iu = cursor[u]
        nbrs[iu] = v
        cursor[u] = iu + 1
        iv = cursor[v]
        nbrs[iv] = u
        cursor[v] = iv + 1
    end
    @inbounds for i in 1:n
        lo = offsets[i]
        hi = offsets[i + 1] - 1
        lo <= hi && sort!(view(nbrs, lo:hi))
    end
    return _PackedEdgeList(n, pedges, offsets, nbrs)
end

@inline function _graph_neighbor_scan_checksum(adj_lists::Vector{Vector{Int}})
    s = 0
    @inbounds for nbrs in adj_lists
        for v in nbrs
            s += v
        end
    end
    return s
end

@inline function _graph_neighbor_scan_checksum(packed::_PackedEdgeList)
    s = 0
    @inbounds for i in 1:packed.n
        lo = packed.offsets[i]
        hi = packed.offsets[i + 1] - 1
        for ptr in lo:hi
            s += packed.nbrs[ptr]
        end
    end
    return s
end

@inline function _graph_backend_bucket_value(x::Int)
    x <= 64 && return 64
    x <= 128 && return 128
    x <= 256 && return 256
    x <= 512 && return 512
    x <= 1024 && return 1024
    x <= 2048 && return 2048
    return 4096
end

@inline function _graph_backend_bucket_key(n::Int, m::Int, k::Int)
    max_edges = div(n * (n - 1), 2)
    density = max_edges > 0 ? (Float64(m) / Float64(max_edges)) : 0.0
    density_bucket = clamp(floor(Int, density * 20.0), 0, 20)
    return (_graph_backend_bucket_value(n), _graph_backend_bucket_value(m), Int(k), density_bucket)
end

function _graph_backend_cache_lookup(key::NTuple{4,Int})
    lock(_GRAPH_BACKEND_WINNER_CACHE_LOCK) do
        return get(_GRAPH_BACKEND_WINNER_CACHE, key, nothing)
    end
end

function _graph_backend_cache_store!(key::NTuple{4,Int}, prefer_packed::Bool)
    lock(_GRAPH_BACKEND_WINNER_CACHE_LOCK) do
        if !haskey(_GRAPH_BACKEND_WINNER_CACHE, key)
            push!(_GRAPH_BACKEND_WINNER_CACHE_ORDER, key)
        end
        _GRAPH_BACKEND_WINNER_CACHE[key] = prefer_packed
        max_entries = max(1, _GRAPH_BACKEND_WINNER_CACHE_MAX[])
        while length(_GRAPH_BACKEND_WINNER_CACHE_ORDER) > max_entries
            old = popfirst!(_GRAPH_BACKEND_WINNER_CACHE_ORDER)
            delete!(_GRAPH_BACKEND_WINNER_CACHE, old)
        end
    end
    return prefer_packed
end

@inline function _graph_backend_probe_eligible(n::Int, m::Int, k::Int)
    _GRAPH_BACKEND_WINNER_CACHE_PROBE[] || return false
    n >= 96 || return false
    m >= 192 || return false
    k >= 3 || return false
    return true
end

function _graph_backend_probe_prefers_packed(edges::AbstractVector{<:Tuple{Int,Int}},
                                             n::Int)
    sample_n = min(n, 192)
    sample_edges = Tuple{Int,Int}[]
    sizehint!(sample_edges, min(length(edges), 4096))
    @inbounds for (u, v) in edges
        if u <= sample_n && v <= sample_n
            push!(sample_edges, (u, v))
            length(sample_edges) >= 4096 && break
        end
    end

    # Too little local structure to profile meaningfully; keep packed preferred.
    length(sample_edges) >= 128 || return true

    t_adj = @elapsed begin
        adj = _graph_adjacency_lists_sorted(sample_n, sample_edges)
        _graph_neighbor_scan_checksum(adj)
    end
    t_packed = @elapsed begin
        packed = _pack_edge_list(sample_n, sample_edges)
        _graph_neighbor_scan_checksum(packed)
    end
    return t_packed <= (1.03 * t_adj)
end

function _enumerate_cliques_k_intersection(adj_lists::Vector{Vector{Int}},
                                           k::Int,
                                           spec::FiltrationSpec,
                                           total_before::BigInt)
    n = length(adj_lists)
    k == 1 && return [[i] for i in 1:n]
    out = Vector{Vector{Int}}()
    clique = Vector{Int}(undef, k)
    max_simp = _construction_max_simplices(spec)
    marks = zeros(UInt32, n)
    stamp = UInt32(0)
    cands = [Int[] for _ in 1:k]
    resize!(cands[1], n)
    @inbounds for i in 1:n
        cands[1][i] = i
    end

    function rec(depth::Int, cand::Vector{Int})
        need = k - depth + 1
        length(cand) < need && return
        max_idx = length(cand) - need + 1
        @inbounds for idx in 1:max_idx
            v = cand[idx]
            clique[depth] = v
            if depth == k
                if max_simp !== nothing && total_before + big(length(out) + 1) > big(max_simp)
                    throw(ArgumentError("Ingestion construction budget exceeded during clique enumeration at simplex dimension $(k - 1): count=$(total_before + big(length(out) + 1)) > max_simplices=$(max_simp)."))
                end
                push!(out, clique[1:k])
            else
                stamp += UInt32(1)
                if stamp == UInt32(0)
                    fill!(marks, UInt32(0))
                    stamp = UInt32(1)
                end
                @inbounds for w in adj_lists[v]
                    marks[w] = stamp
                end
                next = cands[depth + 1]
                empty!(next)
                sizehint!(next, max(0, length(cand) - idx))
                @inbounds for t in (idx + 1):length(cand)
                    u = cand[t]
                    marks[u] == stamp && push!(next, u)
                end
                rec(depth + 1, next)
            end
        end
    end

    rec(1, cands[1])
    return out
end

function _enumerate_cliques_k_intersection(packed::_PackedEdgeList,
                                           k::Int,
                                           spec::FiltrationSpec,
                                           total_before::BigInt)
    n = packed.n
    k == 1 && return [[i] for i in 1:n]
    out = Vector{Vector{Int}}()
    clique = Vector{Int}(undef, k)
    max_simp = _construction_max_simplices(spec)
    marks = zeros(UInt32, n)
    stamp = UInt32(0)
    cands = [Int[] for _ in 1:k]
    resize!(cands[1], n)
    @inbounds for i in 1:n
        cands[1][i] = i
    end

    function rec(depth::Int, cand::Vector{Int})
        need = k - depth + 1
        length(cand) < need && return
        max_idx = length(cand) - need + 1
        @inbounds for idx in 1:max_idx
            v = cand[idx]
            clique[depth] = v
            if depth == k
                if max_simp !== nothing && total_before + big(length(out) + 1) > big(max_simp)
                    throw(ArgumentError("Ingestion construction budget exceeded during clique enumeration at simplex dimension $(k - 1): count=$(total_before + big(length(out) + 1)) > max_simplices=$(max_simp)."))
                end
                push!(out, clique[1:k])
            else
                stamp += UInt32(1)
                if stamp == UInt32(0)
                    fill!(marks, UInt32(0))
                    stamp = UInt32(1)
                end
                lo = packed.offsets[v]
                hi = packed.offsets[v + 1] - 1
                for ptr in lo:hi
                    marks[packed.nbrs[ptr]] = stamp
                end
                next = cands[depth + 1]
                empty!(next)
                sizehint!(next, max(0, length(cand) - idx))
                @inbounds for t in (idx + 1):length(cand)
                    u = cand[t]
                    marks[u] == stamp && push!(next, u)
                end
                rec(depth + 1, next)
            end
        end
    end

    rec(1, cands[1])
    return out
end

function _enumerate_triangles_intersection(adj_lists::Vector{Vector{Int}},
                                           spec::FiltrationSpec,
                                           total_before::BigInt)
    n = length(adj_lists)
    out = NTuple{3,Int}[]
    max_simp = _construction_max_simplices(spec)
    marks = zeros(UInt32, n)
    stamp = UInt32(0)

    @inbounds for a in 1:(n - 2)
        stamp += UInt32(1)
        if stamp == UInt32(0)
            fill!(marks, UInt32(0))
            stamp = UInt32(1)
        end
        for w in adj_lists[a]
            w > a && (marks[w] = stamp)
        end
        for b in adj_lists[a]
            b > a || continue
            for c in adj_lists[b]
                c > b || continue
                marks[c] == stamp || continue
                if max_simp !== nothing && total_before + big(length(out) + 1) > big(max_simp)
                    throw(ArgumentError("Ingestion construction budget exceeded during clique triangle enumeration at simplex dimension 2: count=$(total_before + big(length(out) + 1)) > max_simplices=$(max_simp)."))
                end
                push!(out, (a, b, c))
            end
        end
    end
    return out
end

function _enumerate_triangles_intersection(packed::_PackedEdgeList,
                                           spec::FiltrationSpec,
                                           total_before::BigInt)
    n = packed.n
    out = NTuple{3,Int}[]
    max_simp = _construction_max_simplices(spec)
    marks = zeros(UInt32, n)
    stamp = UInt32(0)

    @inbounds for a in 1:(n - 2)
        stamp += UInt32(1)
        if stamp == UInt32(0)
            fill!(marks, UInt32(0))
            stamp = UInt32(1)
        end
        lo_a = packed.offsets[a]
        hi_a = packed.offsets[a + 1] - 1
        for ptr in lo_a:hi_a
            w = packed.nbrs[ptr]
            w > a && (marks[w] = stamp)
        end
        for ptr_b in lo_a:hi_a
            b = packed.nbrs[ptr_b]
            b > a || continue
            lo_b = packed.offsets[b]
            hi_b = packed.offsets[b + 1] - 1
            for ptr_c in lo_b:hi_b
                c = packed.nbrs[ptr_c]
                c > b || continue
                marks[c] == stamp || continue
                if max_simp !== nothing && total_before + big(length(out) + 1) > big(max_simp)
                    throw(ArgumentError("Ingestion construction budget exceeded during clique triangle enumeration at simplex dimension 2: count=$(total_before + big(length(out) + 1)) > max_simplices=$(max_simp)."))
                end
                push!(out, (a, b, c))
            end
        end
    end
    return out
end

function _enumerate_triangles_combinations(adj::BitMatrix,
                                           n::Int,
                                           spec::FiltrationSpec,
                                           total_before::BigInt;
                                           context::AbstractString="clique")
    _construction_precheck_combination_candidates!(n, 3, total_before, spec; context=context)
    out = NTuple{3,Int}[]
    max_simp = _construction_max_simplices(spec)
    @inbounds for a in 1:(n - 2)
        for b in (a + 1):(n - 1)
            adj[a, b] || continue
            for c in (b + 1):n
                (adj[a, c] && adj[b, c]) || continue
                if max_simp !== nothing && total_before + big(length(out) + 1) > big(max_simp)
                    throw(ArgumentError("Ingestion construction budget exceeded during $(context) enumeration at simplex dimension 2: count=$(total_before + big(length(out) + 1)) > max_simplices=$(max_simp)."))
                end
                push!(out, (a, b, c))
            end
        end
    end
    return out
end

function _enumerate_cliques_k_combinations(adj::BitMatrix,
                                           n::Int,
                                           k::Int,
                                           spec::FiltrationSpec,
                                           total_before::BigInt;
                                           context::AbstractString="clique")
    _construction_precheck_combination_candidates!(n, k, total_before, spec; context=context)
    out = Vector{Vector{Int}}()
    max_simp = _construction_max_simplices(spec)
    for comb in _combinations(n, k)
        ok = true
        @inbounds for i in 1:k
            for j in (i + 1):k
                if !adj[comb[i], comb[j]]
                    ok = false
                    break
                end
            end
            !ok && break
        end
        if ok
            if max_simp !== nothing && total_before + big(length(out) + 1) > big(max_simp)
                throw(ArgumentError("Ingestion construction budget exceeded during $(context) enumeration at simplex dimension $(k - 1): count=$(total_before + big(length(out) + 1)) > max_simplices=$(max_simp)."))
            end
            push!(out, comb)
        end
    end
    return out
end

@inline function _graph_clique_enum_mode()
    mode = _GRAPH_CLIQUE_ENUM_MODE[]
    (mode == :auto || mode == :intersection || mode == :combinations) ||
        throw(ArgumentError("Graph clique enumeration mode must be :auto, :intersection, or :combinations (got $(mode))."))
    return mode
end

@inline function _use_packed_edge_list_backend(n::Int, m::Int, _k::Int)
    _GRAPH_PACKED_EDGELIST_BACKEND[] || return false
    n <= 0 && return false
    m <= 0 && return false
    max_edges = div(n * (n - 1), 2)
    density = max_edges > 0 ? (Float64(m) / Float64(max_edges)) : 0.0
    # Conservative gate: keep packed backend off on sparse mid/large graphs
    # where pack-build overhead can dominate intersection traversal.
    if n <= 224
        return m >= 256
    end
    if m >= 4_000
        return true
    end
    return density >= 0.08
end

function _select_packed_edge_list_backend(edges::AbstractVector{<:Tuple{Int,Int}},
                                          n::Int,
                                          k::Int)
    _GRAPH_PACKED_EDGELIST_BACKEND[] || return false
    m = length(edges)
    heuristic = _use_packed_edge_list_backend(n, m, k)
    _GRAPH_BACKEND_WINNER_CACHE_ENABLED[] || return heuristic

    key = _graph_backend_bucket_key(n, m, k)
    cached = _graph_backend_cache_lookup(key)
    cached === nothing || return cached

    prefer_packed = heuristic
    if heuristic && _graph_backend_probe_eligible(n, m, k)
        prefer_packed = _graph_backend_probe_prefers_packed(edges, n)
    end
    return _graph_backend_cache_store!(key, prefer_packed)
end

@inline function _graph_clique_auto_prefers_combinations(n::Int,
                                                         m::Int,
                                                         k::Int,
                                                         spec::FiltrationSpec,
                                                         total_before::BigInt)
    if k <= 3 && n <= 64
        max_edges = max(1, div(n * (n - 1), 2))
        # Sparse small graphs usually benefit from intersection pruning.
        (10 * m) < (3 * max_edges) && return false
    end

    cand = _combination_count(n, k)
    ms = _construction_max_simplices(spec)
    if ms !== nothing && total_before + cand > big(ms)
        # Keep intersection path available when combination upper-bound checks
        # would fail but actual clique count could still satisfy the budget.
        return false
    end
    cand <= big(2_500) && return true

    if k <= 3 && n <= 64 && cand <= big(40_000)
        # For small dense graphs, a branch-light combinations scan can win.
        # Sparse graphs benefit from intersection pruning.
        return true
    end
    return false
end

function _enumerate_cliques_k(edges::AbstractVector{<:Tuple{Int,Int}},
                              n::Int,
                              k::Int,
                              spec::FiltrationSpec,
                              total_before::BigInt;
                              context::AbstractString="clique")
    sims, _, _ = _enumerate_cliques_k_cached(
        edges, n, k, spec, total_before;
        context=context,
        packed=nothing,
        adj_lists=nothing,
    )
    return sims
end

function _enumerate_cliques_k_cached(edges::AbstractVector{<:Tuple{Int,Int}},
                                     n::Int,
                                     k::Int,
                                     spec::FiltrationSpec,
                                     total_before::BigInt;
                                     context::AbstractString="clique",
                                     packed::Union{Nothing,_PackedEdgeList}=nothing,
                                     adj_lists::Union{Nothing,Vector{Vector{Int}}}=nothing)
    mode = _graph_clique_enum_mode()
    use_intersection = if mode == :intersection
        true
    elseif mode == :combinations
        false
    else
        !_graph_clique_auto_prefers_combinations(n, length(edges), k, spec, total_before)
    end
    if use_intersection
        if _select_packed_edge_list_backend(edges, n, k)
            packed2 = packed === nothing ? _pack_edge_list(n, edges) : packed
            sims = _enumerate_cliques_k_intersection(packed2, k, spec, total_before)
            return sims, packed2, adj_lists
        end
        adj2 = adj_lists === nothing ? _graph_adjacency_lists_sorted(n, edges) : adj_lists
        sims = _enumerate_cliques_k_intersection(adj2, k, spec, total_before)
        return sims, packed, adj2
    end
    adj = falses(n, n)
    @inbounds for (u, v) in edges
        adj[u, v] = true
        adj[v, u] = true
    end
    sims = _enumerate_cliques_k_combinations(adj, n, k, spec, total_before; context=context)
    return sims, packed, adj_lists
end

function _enumerate_triangles_cached(edges::AbstractVector{<:Tuple{Int,Int}},
                                     n::Int,
                                     spec::FiltrationSpec,
                                     total_before::BigInt;
                                     context::AbstractString="clique",
                                     packed::Union{Nothing,_PackedEdgeList}=nothing,
                                     adj_lists::Union{Nothing,Vector{Vector{Int}}}=nothing)
    mode = _graph_clique_enum_mode()
    use_intersection = if mode == :intersection
        true
    elseif mode == :combinations
        false
    else
        !_graph_clique_auto_prefers_combinations(n, length(edges), 3, spec, total_before)
    end
    if use_intersection
        if _select_packed_edge_list_backend(edges, n, 3)
            packed2 = packed === nothing ? _pack_edge_list(n, edges) : packed
            tris = _enumerate_triangles_intersection(packed2, spec, total_before)
            return tris, packed2, adj_lists
        end
        adj2 = adj_lists === nothing ? _graph_adjacency_lists_sorted(n, edges) : adj_lists
        tris = _enumerate_triangles_intersection(adj2, spec, total_before)
        return tris, packed, adj2
    end
    adj = falses(n, n)
    @inbounds for (u, v) in edges
        adj[u, v] = true
        adj[v, u] = true
    end
    tris = _enumerate_triangles_combinations(adj, n, spec, total_before; context=context)
    return tris, packed, adj_lists
end

function _graph_lifted_complex(data::GraphData,
                               spec::FiltrationSpec,
                               vertex_grades::Vector{<:Tuple};
                               return_simplex_tree::Bool=false)
    n = data.n
    edges = data.edges
    length(vertex_grades) == n || error("vertex grade length mismatch.")
    N = length(vertex_grades[1])
    grades = Vector{NTuple{N,Float64}}(undef, n)
    for i in 1:n
        gi = vertex_grades[i]
        length(gi) == N || error("vertex grade arity mismatch.")
        grades[i] = ntuple(j -> Float64(gi[j]), N)
    end
    construction = _construction_from_params(spec.params)
    agg = Symbol(get(spec.params, :simplex_agg, :max))
    lift = _graph_lift(spec; default=:lower_star)
    if lift == :lower_star
        construction.collapse == :none || error("construction.collapse=$(construction.collapse) is unsupported for graph lower-star ingestion.")
        construction.sparsify == :none || error("construction.sparsify=$(construction.sparsify) is unsupported for graph lower-star ingestion.")
        _construction_check_max_edges!(length(edges), spec)
        ne = length(edges)
        total = big(n) + big(ne)
        _construction_check_max_simplices!(total, 1, spec)
        packed_edges = Vector{NTuple{2,Int}}(undef, ne)
        @inbounds for idx in eachindex(edges)
            u, v = edges[idx]
            packed_edges[idx] = (u, v)
        end
        out_grades = Vector{NTuple{N,Float64}}(undef, n + ne)
        @inbounds for i in 1:n
            out_grades[i] = grades[i]
        end
        t = n + 1
        @inbounds for idx in eachindex(packed_edges)
            u, v = packed_edges[idx]
            out_grades[t] = _aggregate_tuple_pair(grades, u, v, agg)
            t += 1
        end
        return _materialize_point_cloud_dim01(
            n, true, packed_edges, out_grades, spec; return_simplex_tree=return_simplex_tree
        )
    elseif lift == :clique
        construction.collapse == :none || error("construction.collapse=$(construction.collapse) is unsupported for graph clique ingestion.")
        construction.sparsify == :none || error("construction.sparsify=$(construction.sparsify) is unsupported for graph clique ingestion.")
        max_dim = max(Int(get(spec.params, :max_dim, 2)), 1)
        _construction_check_max_edges!(length(edges), spec)
        if max_dim <= 2
            packed_edges = Vector{NTuple{2,Int}}(undef, length(edges))
            @inbounds for idx in eachindex(edges)
                u, v = edges[idx]
                packed_edges[idx] = (u, v)
            end

            total = big(n) + big(length(packed_edges))
            _construction_check_max_simplices!(total, 1, spec)
            triangles = NTuple{3,Int}[]
            if max_dim >= 2
                triangles, _, _ = _enumerate_triangles_cached(
                    edges, n, spec, total;
                    context="clique triangles",
                    packed=nothing,
                    adj_lists=nothing,
                )
                total += length(triangles)
                _construction_check_max_simplices!(total, 2, spec)
            end

            nt = length(triangles)
            out_grades = Vector{NTuple{N,Float64}}(undef, n + length(packed_edges) + nt)
            @inbounds for i in 1:n
                out_grades[i] = grades[i]
            end
            t = n + 1
            @inbounds for idx in eachindex(packed_edges)
                u, v = packed_edges[idx]
                out_grades[t] = _aggregate_tuple_pair(grades, u, v, agg)
                t += 1
            end
            if max_dim >= 2
                @inbounds for idx in eachindex(triangles)
                    a, b, c = triangles[idx]
                    out_grades[t] = _aggregate_tuple_triple(grades, a, b, c, agg)
                    t += 1
                end
                return _materialize_point_cloud_dim012(
                    n, packed_edges, triangles, out_grades, spec; return_simplex_tree=return_simplex_tree
                )
            end
            return _materialize_point_cloud_dim01(
                n, true, packed_edges, out_grades, spec; return_simplex_tree=return_simplex_tree
            )
        end

        simplices = Vector{Vector{Vector{Int}}}(undef, max_dim + 1)
        simplices[1] = [[i] for i in 1:n]
        packed = nothing
        adj_lists = nothing
        total = big(n)
        for k in 2:max_dim+1
            sims, packed, adj_lists = _enumerate_cliques_k_cached(
                edges, n, k, spec, total;
                context="clique",
                packed=packed,
                adj_lists=adj_lists,
            )
            simplices[k] = sims
            total += length(sims)
            _construction_check_max_simplices!(total, k - 1, spec)
        end
        for k in 2:max_dim+1
            for s in simplices[k]
                if length(s) == 2
                    push!(grades, _aggregate_tuple_pair(grades, s[1], s[2], agg))
                elseif length(s) == 3
                    push!(grades, _aggregate_tuple_triple(grades, s[1], s[2], s[3], agg))
                else
                    push!(grades, _aggregate_tuple_indexed(grades, s, agg))
                end
            end
        end
        return _materialize_simplicial_output(simplices, grades, spec; return_simplex_tree=return_simplex_tree)
    end
    throw(ArgumentError("Unsupported graph lift=$(lift). Supported: :lower_star, :clique"))
end

function _graph_weight_threshold_complex(data::GraphData, spec::FiltrationSpec;
                                         return_simplex_tree::Bool=false)
    n = data.n
    edges = data.edges
    weights = _graph_edge_weights(data, spec; required=true)
    construction = _construction_from_params(spec.params)
    lift = _graph_lift(spec; default=:graph)
    if lift == :graph
        construction.collapse == :none || error("construction.collapse=$(construction.collapse) is unsupported for graph weight-threshold ingestion.")
        construction.sparsify == :none || error("construction.sparsify=$(construction.sparsify) is unsupported for graph weight-threshold ingestion.")
        _construction_check_max_edges!(length(edges), spec)
        ne = length(edges)
        total = big(n) + big(ne)
        _construction_check_max_simplices!(total, 1, spec)
        packed_edges = Vector{NTuple{2,Int}}(undef, ne)
        @inbounds for idx in eachindex(edges)
            u, v = edges[idx]
            packed_edges[idx] = (u, v)
        end
        grades = Vector{NTuple{1,Float64}}(undef, n + ne)
        @inbounds for i in 1:n
            grades[i] = (0.0,)
        end
        @inbounds for idx in 1:ne
            grades[n + idx] = (Float64(weights[idx]),)
        end
        return _materialize_point_cloud_dim01(
            n, true, packed_edges, grades, spec; return_simplex_tree=return_simplex_tree
        )
    elseif lift == :clique
        construction.collapse == :none || error("construction.collapse=$(construction.collapse) is unsupported for graph weight-threshold clique ingestion.")
        construction.sparsify == :none || error("construction.sparsify=$(construction.sparsify) is unsupported for graph weight-threshold clique ingestion.")
        max_dim = max(Int(get(spec.params, :max_dim, 2)), 1)
        _construction_check_max_edges!(length(edges), spec)
        weight_tbl = _graph_edge_weight_lookup_table(edges, weights)
        simplices = Vector{Vector{Vector{Int}}}(undef, max_dim + 1)
        simplices[1] = [[i] for i in 1:n]
        grades = NTuple{1,Float64}[(0.0,) for _ in 1:n]
        packed = nothing
        adj_lists = nothing
        total = big(n)
        for k in 2:max_dim+1
            sims, packed, adj_lists = _enumerate_cliques_k_cached(
                edges, n, k, spec, total;
                context="graph weight-threshold clique",
                packed=packed,
                adj_lists=adj_lists,
            )
            for comb in sims
                maxw = if k == 2
                    _graph_edge_weight_lookup(weight_tbl, comb[1], comb[2])
                elseif k == 3
                    w12 = _graph_edge_weight_lookup(weight_tbl, comb[1], comb[2])
                    w13 = _graph_edge_weight_lookup(weight_tbl, comb[1], comb[3])
                    w23 = _graph_edge_weight_lookup(weight_tbl, comb[2], comb[3])
                    max(w12, max(w13, w23))
                else
                    wmax = 0.0
                    @inbounds for i in 1:k
                        ci = comb[i]
                        for j in (i + 1):k
                            wmax = max(wmax, _graph_edge_weight_lookup(weight_tbl, ci, comb[j]))
                        end
                    end
                    wmax
                end
                push!(grades, (maxw,))
            end
            simplices[k] = sims
            total += length(sims)
            _construction_check_max_simplices!(total, k - 1, spec)
        end
        return _materialize_simplicial_output(simplices, grades, spec; return_simplex_tree=return_simplex_tree)
    end
    throw(ArgumentError("Unsupported graph_weight_threshold lift=$(lift). Supported: :graph, :clique"))
end

@inline function _delaunay_highdim_policy(spec::FiltrationSpec)
    policy = Symbol(get(spec.params, :highdim_policy, :rips))
    (policy === :rips || policy === :error) ||
        throw(ArgumentError("Unsupported Delaunay highdim_policy=$(policy). Expected :rips or :error."))
    return policy
end

function _graded_complex_from_point_cloud_delaunay_highdim_fallback(data::PointCloud,
                                                                    spec::FiltrationSpec;
                                                                    function_style::Bool=false,
                                                                    return_simplex_tree::Bool=false)
    d = length(data.points[1])
    policy = _delaunay_highdim_policy(spec)
    if policy === :error
        error("Delaunay filtrations currently support only 1D/2D point clouds (got dimension $d). Set highdim_policy=:rips to enable higher-dimensional fallback.")
    end

    fallback_params = _filter_params(spec.params, [:highdim_policy, :delaunay_backend])
    fallback_spec = FiltrationSpec(; kind=:function_rips, fallback_params...)
    if function_style
        return _graded_complex_from_point_cloud(data, fallback_spec; return_simplex_tree=return_simplex_tree)
    end

    # Lower-star fallback keeps 1-parameter output by projecting function-Rips grades.
    G2, _, _ = _graded_complex_from_point_cloud(data, fallback_spec; return_simplex_tree=false)
    grades = Vector{NTuple{1,Float64}}(undef, length(G2.grades))
    @inbounds for i in eachindex(G2.grades)
        grades[i] = (Float64(G2.grades[i][2]),)
    end
    G1 = GradedComplex(G2.cells_by_dim, G2.boundaries, grades)
    orientation = get(spec.params, :orientation, (1,))
    axes = get(spec.params, :axes, _axes_from_grades(grades, 1; orientation=orientation))
    if return_simplex_tree
        return _materialize_simplicial_output(_simplices_from_complex(G1), G1.grades, spec; return_simplex_tree=true)
    end
    return G1, axes, orientation
end

function _graded_complex_from_point_cloud_delaunay(data::PointCloud,
                                                   spec::FiltrationSpec;
                                                   function_style::Bool=false,
                                                   return_simplex_tree::Bool=false)
    points = data.points
    d = length(points[1])
    if d > 2
        return _graded_complex_from_point_cloud_delaunay_highdim_fallback(
            data, spec;
            function_style=function_style,
            return_simplex_tree=return_simplex_tree,
        )
    end

    max_dim = Int(get(spec.params, :max_dim, 2))
    max_dim = min(max_dim, 2)
    entry = _packed_delaunay_entry(points, spec; max_dim=max_dim)
    packed = entry.packed
    vals = _point_vertex_values(points, spec)
    agg = get(spec.params, :simplex_agg, :max)
    n = length(points)
    ne = max_dim >= 1 ? length(packed.edges) : 0
    nt = max_dim >= 2 ? length(packed.triangles) : 0
    if return_simplex_tree
        if function_style
            return _materialize_point_cloud_packed_simplex_tree(
                n, max_dim, packed.edges, packed.triangles, spec, NTuple{2,Float64},
                v -> (0.0, vals[v]),
                (idx, i, j) -> (packed.edge_radius[idx], _aggregate_pair(Float64(vals[i]), Float64(vals[j]), agg)),
                (idx, i, j, k) -> (packed.tri_radius[idx], _aggregate_triple(Float64(vals[i]), Float64(vals[j]), Float64(vals[k]), agg)),
            )
        end
        return _materialize_point_cloud_packed_simplex_tree(
            n, max_dim, packed.edges, packed.triangles, spec, NTuple{1,Float64},
            v -> (vals[v],),
            (_idx, i, j) -> (_aggregate_pair(Float64(vals[i]), Float64(vals[j]), agg),),
            (_idx, i, j, k) -> (_aggregate_triple(Float64(vals[i]), Float64(vals[j]), Float64(vals[k]), agg),),
        )
    end

    grades = if function_style
        Vector{NTuple{2,Float64}}(undef, n + ne + nt)
    else
        Vector{NTuple{1,Float64}}(undef, n + ne + nt)
    end

    t = 1
    # Vertices
    @inbounds for v in 1:n
        if function_style
            grades[t] = (0.0, vals[v])
        else
            grades[t] = (vals[v],)
        end
        t += 1
    end
    # Edges
    if max_dim >= 1
        @inbounds for idx in 1:ne
            i, j = packed.edges[idx]
            r = packed.edge_radius[idx]
            f = _aggregate_pair(Float64(vals[i]), Float64(vals[j]), agg)
            if function_style
                grades[t] = (r, f)
            else
                grades[t] = (f,)
            end
            t += 1
        end
    end
    # Triangles
    if max_dim >= 2
        @inbounds for idx in 1:nt
            i, j, k = packed.triangles[idx]
            r = packed.tri_radius[idx]
            f = _aggregate_triple(Float64(vals[i]), Float64(vals[j]), Float64(vals[k]), agg)
            if function_style
                grades[t] = (r, f)
            else
                grades[t] = (f,)
            end
            t += 1
        end
    end

    if max_dim >= 1
        if d == 2 && _POINTCLOUD_DELAUNAY_CACHE_ENABLED[]
            lock(_POINTCLOUD_DELAUNAY_CACHE_LOCK) do
                _ensure_packed_delaunay_boundaries!(entry, n, max_dim)
            end
        else
            _ensure_packed_delaunay_boundaries!(entry, n, max_dim)
        end
    end
    return _materialize_point_cloud_packed_with_cached_boundaries(n, max_dim, entry, grades, spec)
end

function _graded_complex_from_point_cloud_alpha(data::PointCloud,
                                                spec::FiltrationSpec;
                                                return_simplex_tree::Bool=false)
    points = data.points
    d = length(points[1])
    if d > 2
        policy = _delaunay_highdim_policy(spec)
        if policy === :error
            error("Alpha filtrations currently support only 1D/2D point clouds (got dimension $d). Set highdim_policy=:rips to enable higher-dimensional fallback.")
        end
        fallback_params = _filter_params(spec.params, [:highdim_policy, :delaunay_backend])
        fallback_spec = FiltrationSpec(; kind=:rips, fallback_params...)
        return _graded_complex_from_point_cloud(data, fallback_spec; return_simplex_tree=return_simplex_tree)
    end

    max_dim = min(Int(get(spec.params, :max_dim, 2)), 2)
    entry = _packed_delaunay_entry(points, spec; max_dim=max_dim)
    packed = entry.packed
    n = length(points)
    ne = max_dim >= 1 ? length(packed.edges) : 0
    nt = max_dim >= 2 ? length(packed.triangles) : 0
    edge_alpha_sq = max_dim >= 1 ? _alpha_edge_grades_sq(points, packed) : Float64[]
    tri_alpha_sq = max_dim >= 2 ? map(r -> r * r, packed.tri_radius) : Float64[]
    if return_simplex_tree
        return _materialize_point_cloud_packed_simplex_tree(
            n, max_dim, packed.edges, packed.triangles, spec, NTuple{1,Float64},
            _v -> (0.0,),
            (idx, _i, _j) -> (edge_alpha_sq[idx],),
            (idx, _i, _j, _k) -> (tri_alpha_sq[idx],),
        )
    end

    grades = Vector{NTuple{1,Float64}}(undef, n + ne + nt)

    t = 1
    @inbounds for _ in 1:n
        grades[t] = (0.0,)
        t += 1
    end
    if max_dim >= 1
        @inbounds for idx in 1:ne
            grades[t] = (edge_alpha_sq[idx],)
            t += 1
        end
    end
    if max_dim >= 2
        @inbounds for idx in 1:nt
            grades[t] = (tri_alpha_sq[idx],)
            t += 1
        end
    end

    if max_dim >= 1
        if d == 2 && _POINTCLOUD_DELAUNAY_CACHE_ENABLED[]
            lock(_POINTCLOUD_DELAUNAY_CACHE_LOCK) do
                _ensure_packed_delaunay_boundaries!(entry, n, max_dim)
            end
        else
            _ensure_packed_delaunay_boundaries!(entry, n, max_dim)
        end
    end
    return _materialize_point_cloud_packed_with_cached_boundaries(n, max_dim, entry, grades, spec)
end

@inline function _vertex_degree_scores(n::Int, edges)
    deg = zeros(Int, n)
    @inbounds for e in edges
        u, v = e[1], e[2]
        deg[u] += 1
        deg[v] += 1
    end
    return deg
end

function _graded_complex_from_point_cloud_core_delaunay(data::PointCloud,
                                                        spec::FiltrationSpec;
                                                        return_simplex_tree::Bool=false)
    points = data.points
    n = length(points)
    d = length(points[1])
    if d > 2
        policy = _delaunay_highdim_policy(spec)
        if policy === :error
            error("Core-Delaunay filtrations currently support only 1D/2D point clouds (got dimension $d). Set highdim_policy=:rips to enable higher-dimensional fallback.")
        end
        fallback_params = _filter_params(spec.params, [:highdim_policy, :delaunay_backend])
        fallback_spec = FiltrationSpec(; kind=:degree_rips, fallback_params...)
        return _graded_complex_from_point_cloud(data, fallback_spec; return_simplex_tree=return_simplex_tree)
    end

    max_dim = min(Int(get(spec.params, :max_dim, 2)), 2)
    entry = _packed_delaunay_entry(points, spec; max_dim=max_dim)
    packed = entry.packed
    core = _core_numbers(n, packed.edges)
    ne = max_dim >= 1 ? length(packed.edges) : 0
    nt = max_dim >= 2 ? length(packed.triangles) : 0
    if return_simplex_tree
        return _materialize_point_cloud_packed_simplex_tree(
            n, max_dim, packed.edges, packed.triangles, spec, NTuple{2,Float64},
            i -> (0.0, Float64(core[i])),
            (idx, i, j) -> (packed.edge_radius[idx], Float64(max(core[i], core[j]))),
            (idx, i, j, k) -> (packed.tri_radius[idx], Float64(max(core[i], max(core[j], core[k])))),
        )
    end

    grades = Vector{NTuple{2,Float64}}(undef, n + ne + nt)

    t = 1
    @inbounds for i in 1:n
        grades[t] = (0.0, Float64(core[i]))
        t += 1
    end
    if max_dim >= 1
        @inbounds for idx in 1:ne
            i, j = packed.edges[idx]
            r = packed.edge_radius[idx]
            grades[t] = (r, Float64(max(core[i], core[j])))
            t += 1
        end
    end
    if max_dim >= 2
        @inbounds for idx in 1:nt
            i, j, k = packed.triangles[idx]
            r = packed.tri_radius[idx]
            grades[t] = (r, Float64(max(core[i], max(core[j], core[k]))))
            t += 1
        end
    end

    if max_dim >= 1
        if d == 2 && _POINTCLOUD_DELAUNAY_CACHE_ENABLED[]
            lock(_POINTCLOUD_DELAUNAY_CACHE_LOCK) do
                _ensure_packed_delaunay_boundaries!(entry, n, max_dim)
            end
        else
            _ensure_packed_delaunay_boundaries!(entry, n, max_dim)
        end
    end
    return _materialize_point_cloud_packed_with_cached_boundaries(n, max_dim, entry, grades, spec)
end

function _graded_complex_from_point_cloud_core(data::PointCloud,
                                               spec::FiltrationSpec;
                                               return_simplex_tree::Bool=false)
    points = data.points
    pmat = point_matrix(data)
    n = length(points)
    edges = _core_edges_from_point_cloud(pmat, spec)
    core = _core_numbers(n, edges)
    vals = _point_graph_scalar_values(points, spec)
    m = length(edges)
    grades = Vector{NTuple{2,Float64}}(undef, n + m)
    t = 1
    @inbounds for i in 1:n
        grades[t] = (vals[i], Float64(core[i]))
        t += 1
    end
    @inbounds for idx in 1:m
        u, v = edges[idx]
        grades[t] = (max(vals[u], vals[v]), Float64(min(core[u], core[v])))
        t += 1
    end
    return _materialize_point_cloud_dim01(
        n, true, edges, grades, spec; return_simplex_tree=return_simplex_tree
    )
end

function _rips_like_simplices_for_point_cloud(data::PointCloud,
                                              spec::FiltrationSpec)
    construction = _construction_from_params(spec.params)
    n = length(data.points)
    pmat = point_matrix(data)
    max_dim = Int(get(spec.params, :max_dim, 2))
    simplices = Vector{Vector{Vector{Int}}}(undef, max_dim + 1)
    simplices[1] = [[i] for i in 1:n]
    if construction.sparsify != :none
        edges, edge_dists, _ = _point_cloud_sparsify_edge_driven(pmat, spec, construction)
        edges, _ = _apply_construction_collapse_edge_driven(edges, edge_dists, pmat, construction)
        simplices = [simplices[1], _edge_vectors_from_tuples(edges)]
        return simplices
    end
    if max_dim >= 1
        edge_count = binomial(big(n), big(2))
        _construction_check_max_edges!(edge_count, spec)
    end
    total = big(n)
    for k in 2:max_dim+1
        count_k = _construction_precheck_combination_enumeration!(n, k, total, spec)
        simplices[k] = _combinations(n, k)
        total += count_k
    end
    return simplices
end

@inline function _next_combination!(comb::Vector{Int}, n::Int, k::Int)
    i = k
    @inbounds while i >= 1 && comb[i] == n - k + i
        i -= 1
    end
    i == 0 && return false
    @inbounds begin
        comb[i] += 1
        for j in (i + 1):k
            comb[j] = comb[j - 1] + 1
        end
    end
    return true
end

function _rhomboid_fill_dim_simplices_and_grades!(simplices_k::Vector{Vector{Int}},
                                                  grades::Vector{NTuple{2,Float64}},
                                                  vals::Vector{Float64},
                                                  k::Int)
    n = length(vals)
    isempty(simplices_k) && return nothing
    comb = collect(1:k)
    out = 1
    while true
        s = Vector{Int}(undef, k)
        vmin = vals[comb[1]]
        vmax = vmin
        @inbounds for t in 1:k
            c = comb[t]
            s[t] = c
            v = vals[c]
            v < vmin && (vmin = v)
            v > vmax && (vmax = v)
        end
        simplices_k[out] = s
        push!(grades, (vmin, vmax))
        out += 1
        _next_combination!(comb, n, k) || break
    end
    return nothing
end

function _graded_complex_from_point_cloud_rhomboid(data::PointCloud,
                                                   spec::FiltrationSpec;
                                                   return_simplex_tree::Bool=false)
    points = data.points
    construction = _construction_from_params(spec.params)
    max_dim = Int(get(spec.params, :max_dim, 2))
    vals = _point_vertex_values(points, spec)

    if max_dim <= 1 || construction.sparsify != :none || construction.collapse != :none
        simplices = _rips_like_simplices_for_point_cloud(data, spec)
        grades = Vector{NTuple{2,Float64}}()
        sizehint!(grades, sum(length.(simplices)))
        for s in simplices[1]
            v = vals[s[1]]
            push!(grades, (v, v))
        end
        for k in 2:length(simplices)
            for s in simplices[k]
                vmin = Float64(vals[s[1]])
                vmax = vmin
                @inbounds for t in 2:length(s)
                    v = Float64(vals[s[t]])
                    v < vmin && (vmin = v)
                    v > vmax && (vmax = v)
                end
                push!(grades, (vmin, vmax))
            end
        end
        return _materialize_simplicial_output(simplices, grades, spec; return_simplex_tree=return_simplex_tree)
    end

    n = length(points)
    simplices = Vector{Vector{Vector{Int}}}(undef, max_dim + 1)
    simplices[1] = [[i] for i in 1:n]
    grades = Vector{NTuple{2,Float64}}()
    sizehint!(grades, n)
    @inbounds for i in 1:n
        v = vals[i]
        push!(grades, (v, v))
    end

    total = big(n)
    for k in 2:max_dim+1
        count_k = _construction_precheck_combination_enumeration!(n, k, total, spec)
        simplices_k = Vector{Vector{Int}}(undef, count_k)
        sizehint!(grades, length(grades) + count_k)
        _rhomboid_fill_dim_simplices_and_grades!(simplices_k, grades, vals, k)
        simplices[k] = simplices_k
        total += count_k
    end
    _construction_check_memory_budget!(
        _estimate_dense_bytes_from_cell_counts(BigInt[length(s) for s in simplices]),
        spec,
    )
    return _materialize_simplicial_output(simplices, grades, spec; return_simplex_tree=return_simplex_tree)
end

function _materialize_point_cloud_dim01(n::Int,
                                        include_edge_dim::Bool,
                                        edges::Vector{NTuple{2,Int}},
                                        grades::Vector{<:NTuple{N,Float64}},
                                        spec::FiltrationSpec;
                                        return_simplex_tree::Bool=false) where {N}
    max_dim = include_edge_dim ? 1 : 0
    return _materialize_point_cloud_packed(
        n, max_dim, edges, NTuple{3,Int}[], grades, spec; return_simplex_tree=return_simplex_tree
    )
end

@inline function _is_point_cloud_lowdim_direct_lazy_kind(kind::Symbol)
    return kind === :rips || kind === :rips_density || kind === :rips_codensity ||
           kind === :rips_lowerstar || kind === :function_rips || kind === :degree_rips
end

struct _PointCloudLowdimEdgePayload{P,M}
    points::P
    pmat::M
    n::Int
    kind::Symbol
    construction::ConstructionOptions
    max_dim::Int
    include_edge_dim::Bool
    edges::Vector{NTuple{2,Int}}
    edge_dists::Vector{Float64}
    kdist::Vector{Float64}
end

struct _PointCloudLowdimLazyPayload
    grades_by_dim::Vector
    boundaries::Vector{SparseMatrixCSC{Int,Int}}
    axes::Any
    orientation::Any
end

function _point_cloud_lowdim_edge_payload(data::PointCloud, spec::FiltrationSpec)
    points = data.points
    pmat = point_matrix(data)
    n = length(points)
    kind = spec.kind
    construction = _construction_from_params(spec.params)
    max_dim = Int(get(spec.params, :max_dim, 1))
    include_edge_dim = (max_dim >= 1) || (construction.sparsify != :none)
    rhomboid_edge_only = (kind == :rhomboid)

    edges = NTuple{2,Int}[]
    edge_dists = Float64[]
    kdist = Float64[]

    if include_edge_dim
        if construction.sparsify != :none
            edges, edge_dists, kdist = _point_cloud_sparsify_edge_driven(pmat, spec, construction)
            edges, edge_dists = _apply_construction_collapse_edge_driven(edges, edge_dists, pmat, construction)
        else
            radius = haskey(spec.params, :radius) ? Float64(spec.params[:radius]) : Inf
            if rhomboid_edge_only
                if isfinite(radius)
                    edges = _point_cloud_edges_within_radius_edges_only(pmat, radius)
                    _construction_check_max_edges!(length(edges), spec)
                else
                    edge_count = _combination_count(n, 2)
                    _construction_check_max_edges!(edge_count, spec)
                    edges = _complete_point_cloud_edges(n)
                end
            else
                if isfinite(radius) && _POINTCLOUD_LOWDIM_RADIUS_STREAMING[]
                    edges, edge_dists = _point_cloud_edges_within_radius(pmat, radius)
                    _construction_check_max_edges!(length(edges), spec)
                else
                    edges_all, dists_all = _complete_point_cloud_edges_with_dist(pmat)
                    if isfinite(radius)
                        edges = NTuple{2,Int}[]
                        edge_dists = Float64[]
                        sizehint!(edges, min(length(edges_all), max(0, 4 * n)))
                        sizehint!(edge_dists, min(length(dists_all), max(0, 4 * n)))
                        @inbounds for idx in eachindex(edges_all)
                            d = dists_all[idx]
                            d <= radius || continue
                            push!(edges, edges_all[idx])
                            push!(edge_dists, d)
                        end
                        _construction_check_max_edges!(length(edges), spec)
                    else
                        _construction_check_max_edges!(length(edges_all), spec)
                        edges = edges_all
                        edge_dists = dists_all
                    end
                end
            end
        end
        total = big(n) + big(length(edges))
        _construction_check_max_simplices!(total, 1, spec)
        _construction_check_memory_budget!(_estimate_dense_bytes_from_cell_counts(BigInt[n, length(edges)]), spec)
    else
        _construction_check_max_simplices!(big(n), 0, spec)
        _construction_check_memory_budget!(_estimate_dense_bytes_from_cell_counts(BigInt[n]), spec)
    end

    return _PointCloudLowdimEdgePayload(
        points,
        pmat,
        n,
        kind,
        construction,
        max_dim,
        include_edge_dim,
        edges,
        edge_dists,
        kdist,
    )
end

function _point_cloud_lowdim_grades_by_dim(payload::_PointCloudLowdimEdgePayload,
                                           spec::FiltrationSpec)
    n = payload.n
    kind = payload.kind
    include_edge_dim = payload.include_edge_dim
    edges = payload.edges
    edge_dists = payload.edge_dists
    construction = payload.construction
    kdist = payload.kdist
    points = payload.points
    pmat = payload.pmat

    if kind == :rips
        vertex_grades = [(0.0,) for _ in 1:n]
        edge_grades = NTuple{1,Float64}[]
        sizehint!(edge_grades, include_edge_dim ? length(edges) : 0)
        if include_edge_dim
            @inbounds for d in edge_dists
                push!(edge_grades, (d,))
            end
        end
        return Any[vertex_grades, edge_grades]
    elseif kind == :rips_density
        density_k = Int(get(spec.params, :density_k, 2))
        density_k > 0 || error("density_k must be > 0.")
        nn_backend = _pointcloud_nn_backend(spec)
        approx_candidates = _pointcloud_nn_approx_candidates(spec)
        dens = if construction.sparsify == :knn && density_k == Int(get(spec.params, :knn, 8)) && length(kdist) == n
            kdist
        else
            _point_cloud_knn_distances(pmat, density_k; backend=nn_backend, approx_candidates=approx_candidates)
        end
        vertex_grades = Vector{NTuple{2,Float64}}(undef, n)
        @inbounds for i in 1:n
            vertex_grades[i] = (0.0, dens[i])
        end
        edge_grades = Vector{NTuple{2,Float64}}(undef, include_edge_dim ? length(edges) : 0)
        if include_edge_dim
            @inbounds for idx in eachindex(edges)
                u, v = edges[idx]
                edge_grades[idx] = (edge_dists[idx], max(dens[u], dens[v]))
            end
        end
        return Any[vertex_grades, edge_grades]
    elseif kind == :rips_codensity
        vals = _point_codensity_values(points, spec)
        vertex_grades = Vector{NTuple{2,Float64}}(undef, n)
        @inbounds for i in 1:n
            vertex_grades[i] = (0.0, vals[i])
        end
        edge_grades = Vector{NTuple{2,Float64}}(undef, include_edge_dim ? length(edges) : 0)
        if include_edge_dim
            @inbounds for idx in eachindex(edges)
                u, v = edges[idx]
                edge_grades[idx] = (edge_dists[idx], max(vals[u], vals[v]))
            end
        end
        return Any[vertex_grades, edge_grades]
    elseif kind == :rips_lowerstar || kind == :function_rips
        vals = _point_vertex_values(points, spec)
        agg = Symbol(get(spec.params, :simplex_agg, :max))
        vertex_grades = Vector{NTuple{2,Float64}}(undef, n)
        @inbounds for i in 1:n
            vertex_grades[i] = (0.0, vals[i])
        end
        edge_grades = Vector{NTuple{2,Float64}}(undef, include_edge_dim ? length(edges) : 0)
        if include_edge_dim
            @inbounds for idx in eachindex(edges)
                u, v = edges[idx]
                edge_grades[idx] = (edge_dists[idx], _aggregate_pair(vals[u], vals[v], agg))
            end
        end
        return Any[vertex_grades, edge_grades]
    elseif kind == :degree_rips
        deg = _vertex_degree_scores(n, edges)
        vertex_grades = Vector{NTuple{2,Float64}}(undef, n)
        @inbounds for i in 1:n
            vertex_grades[i] = (0.0, Float64(deg[i]))
        end
        edge_grades = Vector{NTuple{2,Float64}}(undef, include_edge_dim ? length(edges) : 0)
        if include_edge_dim
            @inbounds for idx in eachindex(edges)
                u, v = edges[idx]
                edge_grades[idx] = (edge_dists[idx], Float64(max(deg[u], deg[v])))
            end
        end
        return Any[vertex_grades, edge_grades]
    end

    error("_point_cloud_lowdim_grades_by_dim: unsupported kind $(kind).")
end

function _point_cloud_lowdim_lazy_payload(data::PointCloud, spec::FiltrationSpec)
    payload = _point_cloud_lowdim_edge_payload(data, spec)
    grades_by_dim = _point_cloud_lowdim_grades_by_dim(payload, spec)
    if !payload.include_edge_dim
        resize!(grades_by_dim, 1)
    end
    boundaries = payload.include_edge_dim ? SparseMatrixCSC{Int,Int}[_edge_boundary_matrix(payload.n, payload.edges)] :
                                            SparseMatrixCSC{Int,Int}[]
    orient = get(spec.params, :orientation, ntuple(_ -> 1, length(first(grades_by_dim[1]))))
    flat_grades = payload.include_edge_dim ? vcat(grades_by_dim[1], grades_by_dim[2]) : grades_by_dim[1]
    axes = get(spec.params, :axes, _axes_from_grades(flat_grades, length(orient); orientation=orient))
    return _PointCloudLowdimLazyPayload(grades_by_dim, boundaries, axes, orient)
end

function _graded_complex_from_point_cloud_lowdim(data::PointCloud,
                                                 spec::FiltrationSpec;
                                                 return_simplex_tree::Bool=false)
    payload = _point_cloud_lowdim_edge_payload(data, spec)
    n = payload.n
    kind = payload.kind
    include_edge_dim = payload.include_edge_dim
    edges = payload.edges
    edge_dists = payload.edge_dists

    if kind == :rips
        grades_by_dim = _point_cloud_lowdim_grades_by_dim(payload, spec)
        grades = include_edge_dim ? vcat(grades_by_dim[1], grades_by_dim[2]) : grades_by_dim[1]
        return _materialize_point_cloud_dim01(n, include_edge_dim, edges, grades, spec; return_simplex_tree=return_simplex_tree)
    elseif kind == :rips_density
        grades_by_dim = _point_cloud_lowdim_grades_by_dim(payload, spec)
        grades = include_edge_dim ? vcat(grades_by_dim[1], grades_by_dim[2]) : grades_by_dim[1]
        return _materialize_point_cloud_dim01(n, include_edge_dim, edges, grades, spec; return_simplex_tree=return_simplex_tree)
    elseif kind == :rips_codensity || kind == :rips_lowerstar || kind == :function_rips || kind == :degree_rips
        grades_by_dim = _point_cloud_lowdim_grades_by_dim(payload, spec)
        grades = include_edge_dim ? vcat(grades_by_dim[1], grades_by_dim[2]) : grades_by_dim[1]
        return _materialize_point_cloud_dim01(n, include_edge_dim, edges, grades, spec; return_simplex_tree=return_simplex_tree)
    elseif kind == :rhomboid
        points = payload.points
        vals = _point_vertex_values(points, spec)
        grades = Vector{NTuple{2,Float64}}(undef, n + (include_edge_dim ? length(edges) : 0))
        t = 1
        @inbounds for i in 1:n
            v = Float64(vals[i])
            grades[t] = (v, v)
            t += 1
        end
        if include_edge_dim
            @inbounds for idx in eachindex(edges)
                u, v = edges[idx]
                vu = Float64(vals[u])
                vv = Float64(vals[v])
                grades[t] = (min(vu, vv), max(vu, vv))
                t += 1
            end
        end
        return _materialize_point_cloud_dim01(n, include_edge_dim, edges, grades, spec; return_simplex_tree=return_simplex_tree)
    end
    error("Unsupported low-dimensional point cloud filtration kind: $(kind).")
end

function _graded_complex_from_point_cloud_dim2_packed(data::PointCloud,
                                                      spec::FiltrationSpec;
                                                      return_simplex_tree::Bool=false)
    points = point_matrix(data)
    n = size(points, 1)
    kind = spec.kind
    max_dim = Int(get(spec.params, :max_dim, 2))
    max_dim == 2 || error("dim2 packed point-cloud kernel requires max_dim=2 (got $(max_dim)).")

    radius = if haskey(spec.params, :radius)
        Float64(spec.params[:radius])
    else
        Inf
    end

    dist_packed = _point_cloud_pairwise_packed(points)
    pair_count = length(dist_packed)

    edges = NTuple{2,Int}[]
    edge_dists = Float64[]
    edge_mask = falses(pair_count)
    sizehint!(edges, pair_count)
    sizehint!(edge_dists, pair_count)
    adj_hi = [Int[] for _ in 1:n]

    packed_idx = 1
    @inbounds for i in 1:(n - 1)
        for j in (i + 1):n
            d = dist_packed[packed_idx]
            if d <= radius
                push!(edges, (i, j))
                push!(edge_dists, d)
                edge_mask[packed_idx] = true
                push!(adj_hi[i], j)
            end
            packed_idx += 1
        end
    end

    _construction_check_max_edges!(length(edges), spec)

    triangles = NTuple{3,Int}[]
    tri_diams = Float64[]
    sizehint!(triangles, max(length(edges), n))
    sizehint!(tri_diams, max(length(edges), n))
    @inbounds for i in 1:(n - 2)
        nbrs = adj_hi[i]
        ln = length(nbrs)
        ln < 2 && continue
        for aidx in 1:(ln - 1)
            j = nbrs[aidx]
            dij = _packed_pair_distance(dist_packed, n, i, j)
            for bidx in (aidx + 1):ln
                k = nbrs[bidx]
                jk_idx = _packed_pair_index(n, j, k)
                edge_mask[jk_idx] || continue
                dik = _packed_pair_distance(dist_packed, n, i, k)
                djk = dist_packed[jk_idx]
                push!(triangles, (i, j, k))
                push!(tri_diams, max(dij, max(dik, djk)))
            end
        end
    end

    total = big(n) + big(length(edges)) + big(length(triangles))
    _construction_check_max_simplices!(total, 2, spec)
    _construction_check_memory_budget!(
        _estimate_dense_bytes_from_cell_counts(BigInt[n, length(edges), length(triangles)]),
        spec,
    )

    if kind == :rips
        grades = Vector{NTuple{1,Float64}}(undef, n + length(edges) + length(triangles))
        t = 1
        @inbounds for _ in 1:n
            grades[t] = (0.0,)
            t += 1
        end
        @inbounds for d in edge_dists
            grades[t] = (d,)
            t += 1
        end
        @inbounds for d in tri_diams
            grades[t] = (d,)
            t += 1
        end
        return _materialize_point_cloud_dim012(
            n, edges, triangles, grades, spec; return_simplex_tree=return_simplex_tree
        )
    elseif kind == :rips_density
        density_k = Int(get(spec.params, :density_k, 2))
        density_k > 0 || error("density_k must be > 0.")
        nn_backend = _pointcloud_nn_backend(spec)
        approx_candidates = _pointcloud_nn_approx_candidates(spec)
        dens = _point_cloud_knn_distances(points, density_k; backend=nn_backend, approx_candidates=approx_candidates)
        grades = Vector{NTuple{2,Float64}}(undef, n + length(edges) + length(triangles))
        t = 1
        @inbounds for i in 1:n
            grades[t] = (0.0, dens[i])
            t += 1
        end
        @inbounds for idx in eachindex(edges)
            u, v = edges[idx]
            grades[t] = (edge_dists[idx], max(dens[u], dens[v]))
            t += 1
        end
        @inbounds for idx in eachindex(triangles)
            a, b, c = triangles[idx]
            grades[t] = (tri_diams[idx], max(dens[a], max(dens[b], dens[c])))
            t += 1
        end
        return _materialize_point_cloud_dim012(
            n, edges, triangles, grades, spec; return_simplex_tree=return_simplex_tree
        )
    elseif kind == :rips_lowerstar || kind == :function_rips
        vvals = _point_vertex_values(points, spec)
        agg = Symbol(get(spec.params, :simplex_agg, :max))
        grades = Vector{NTuple{2,Float64}}(undef, n + length(edges) + length(triangles))
        t = 1
        @inbounds for i in 1:n
            grades[t] = (0.0, vvals[i])
            t += 1
        end
        @inbounds for idx in eachindex(edges)
            u, v = edges[idx]
            grades[t] = (edge_dists[idx], _aggregate_pair(vvals[u], vvals[v], agg))
            t += 1
        end
        @inbounds for idx in eachindex(triangles)
            a, b, c = triangles[idx]
            grades[t] = (tri_diams[idx], _aggregate_triple(vvals[a], vvals[b], vvals[c], agg))
            t += 1
        end
        return _materialize_point_cloud_dim012(
            n, edges, triangles, grades, spec; return_simplex_tree=return_simplex_tree
        )
    elseif kind == :degree_rips
        deg = _vertex_degree_scores(n, edges)
        grades = Vector{NTuple{2,Float64}}(undef, n + length(edges) + length(triangles))
        t = 1
        @inbounds for i in 1:n
            grades[t] = (0.0, Float64(deg[i]))
            t += 1
        end
        @inbounds for idx in eachindex(edges)
            u, v = edges[idx]
            grades[t] = (edge_dists[idx], Float64(max(deg[u], deg[v])))
            t += 1
        end
        @inbounds for idx in eachindex(triangles)
            a, b, c = triangles[idx]
            grades[t] = (tri_diams[idx], Float64(max(deg[a], max(deg[b], deg[c]))))
            t += 1
        end
        return _materialize_point_cloud_dim012(
            n, edges, triangles, grades, spec; return_simplex_tree=return_simplex_tree
        )
    end

    error("Unsupported dim2 packed point-cloud filtration kind: $(kind).")
end

function _graded_complex_from_point_cloud(data::PointCloud, spec::FiltrationSpec;
                                          return_simplex_tree::Bool=false,
                                          cache::Union{Nothing,EncodingCache}=nothing)
    data2, spec2, _ = _maybe_greedy_perm_reduce(data, spec)
    data = data2
    spec = spec2
    if spec.kind == :rips_codensity
        spec = _codensity_function_rips_spec(data, spec)
    end

    construction = _construction_from_params(spec.params)
    max_dim = get(spec.params, :max_dim, 1)
    kind = spec.kind
    points = data.points
    pmat = point_matrix(data)
    n = length(points)
    if n == 0
        error("PointCloud has no points.")
    end

    if kind == :delaunay_lower_star
        return _graded_complex_from_point_cloud_delaunay(data, spec; function_style=false, return_simplex_tree=return_simplex_tree)
    elseif kind == :alpha
        return _graded_complex_from_point_cloud_alpha(data, spec; return_simplex_tree=return_simplex_tree)
    elseif kind == :function_delaunay
        return _graded_complex_from_point_cloud_delaunay(data, spec; function_style=true, return_simplex_tree=return_simplex_tree)
    elseif kind == :core_delaunay
        return _graded_complex_from_point_cloud_core_delaunay(data, spec; return_simplex_tree=return_simplex_tree)
    end

    if kind == :landmark_rips
        landmarks = get(spec.params, :landmarks, nothing)
        landmarks === nothing && error("landmark_rips requires landmarks.")
        length(landmarks) > 0 || error("landmark_rips: landmarks cannot be empty.")
        construction_lm = if construction.sparsify == :none && get(spec.params, :radius, nothing) !== nothing
            ConstructionOptions(;
                sparsify = :radius,
                collapse = construction.collapse,
                output_stage = construction.output_stage,
                budget = construction.budget,
            )
        else
            construction
        end
        radius = get(spec.params, :radius, nothing)
        include_edge_dim = (max_dim >= 1) || (construction_lm.sparsify != :none)
        if include_edge_dim &&
           radius !== nothing &&
           construction_lm.sparsify == :radius &&
           construction_lm.collapse == :none &&
           max_dim <= 1
            m = length(landmarks)
            r = Float64(radius)
            packed = _landmark_radius_subgraph_cached(pmat, landmarks, r, spec; cache=cache)
            edges = packed.edges
            edge_dists = packed.dists
            _construction_check_max_edges!(length(edges), spec)
            total = big(m) + big(length(edges))
            _construction_check_max_simplices!(total, include_edge_dim ? 1 : 0, spec)
            _construction_check_memory_budget!(_estimate_dense_bytes_from_cell_counts(BigInt[m, include_edge_dim ? length(edges) : 0]), spec)
            grades = Vector{NTuple{1,Float64}}(undef, m + (include_edge_dim ? length(edges) : 0))
            t = 1
            @inbounds for _ in 1:m
                grades[t] = (0.0,)
                t += 1
            end
            if include_edge_dim
                @inbounds for d in edge_dists
                    grades[t] = (d,)
                    t += 1
                end
            end
            spec_lm = if construction_lm === construction
                spec
            else
                p_lm = _filter_params(spec.params, [:construction])
                FiltrationSpec(; kind=spec.kind, p_lm..., construction=construction_lm)
            end
            return _materialize_point_cloud_dim01(
                m, include_edge_dim, edges, grades, spec_lm; return_simplex_tree=return_simplex_tree
            )
        end
        pts = [points[i] for i in landmarks]
        params2 = _filter_params(spec.params, [:landmarks])
        params2 = merge(params2, (max_dim = max_dim, construction = construction_lm))
        return _graded_complex_from_point_cloud(
            PointCloud(pts),
            FiltrationSpec(; kind=:rips, params2...);
            return_simplex_tree=return_simplex_tree,
            cache=cache,
        )
    end

    if construction.sparsify != :none && max_dim > 1
        error("construction.sparsify=$(construction.sparsify) currently supports max_dim <= 1.")
    end
    if construction.collapse != :none && max_dim > 1
        error("construction.collapse=$(construction.collapse) currently supports max_dim <= 1.")
    end

    # For large point clouds, require explicit sparse construction for Rips-like ingestion.
    if construction.sparsify == :none &&
       (kind == :rips || kind == :rips_density || kind == :rips_codensity ||
        kind == :rips_lowerstar || kind == :function_rips || kind == :degree_rips) &&
       max_dim >= 1 &&
       n >= _POINTCLOUD_LARGE_N_SPARSE_ONLY[]
        throw(ArgumentError("PointCloud with n=$(n) requires sparse construction for this filtration. Use ConstructionOptions(sparsify=:knn|:radius|:greedy_perm) and set a budget."))
    end

    if max_dim <= 1 &&
       (kind == :rips || kind == :rips_density || kind == :rips_codensity ||
        kind == :rips_lowerstar || kind == :function_rips || kind == :rhomboid || kind == :degree_rips)
        return _graded_complex_from_point_cloud_lowdim(data, spec; return_simplex_tree=return_simplex_tree)
    end

    if construction.sparsify != :none &&
       (kind == :rips || kind == :rips_density || kind == :rips_codensity ||
        kind == :rips_lowerstar || kind == :function_rips || kind == :degree_rips)
        simplices = Vector{Vector{Vector{Int}}}(undef, 2)
        simplices[1] = [[i] for i in 1:n]
        edges, edge_dists, kdist = _point_cloud_sparsify_edge_driven(pmat, spec, construction)
        edges, edge_dists = _apply_construction_collapse_edge_driven(edges, edge_dists, pmat, construction)
        simplices[2] = edges
        total = big(n + length(edges))
        _construction_check_max_simplices!(total, 1, spec)
        _construction_check_memory_budget!(_estimate_dense_bytes_from_cell_counts(BigInt[length(s) for s in simplices]), spec)

        if kind == :rips
            grades = Vector{NTuple{1,Float64}}(undef, n + length(edges))
            t = 1
            for _ in 1:n
                grades[t] = (0.0,)
                t += 1
            end
            for d in edge_dists
                grades[t] = (d,)
                t += 1
            end
        elseif kind == :rips_density
            density_k = Int(get(spec.params, :density_k, 2))
            density_k > 0 || error("density_k must be > 0.")
            nn_backend = _pointcloud_nn_backend(spec)
            approx_candidates = _pointcloud_nn_approx_candidates(spec)
            dens = if construction.sparsify == :knn && density_k == Int(get(spec.params, :knn, 8)) && length(kdist) == n
                kdist
            else
                _point_cloud_knn_distances(pmat, density_k; backend=nn_backend, approx_candidates=approx_candidates)
            end
            grades = Vector{NTuple{2,Float64}}(undef, n + length(edges))
            t = 1
            for i in 1:n
                grades[t] = (0.0, dens[i])
                t += 1
            end
            for idx in eachindex(edges)
                e = edges[idx]
                grades[t] = (edge_dists[idx], max(dens[e[1]], dens[e[2]]))
                t += 1
            end
        elseif kind == :rips_codensity || kind == :rips_lowerstar || kind == :function_rips
            vvals = _point_vertex_values(points, spec)
            agg = get(spec.params, :simplex_agg, :max)
            grades = Vector{NTuple{2,Float64}}(undef, n + length(edges))
            t = 1
            for i in 1:n
                grades[t] = (0.0, vvals[i])
                t += 1
            end
            for idx in eachindex(edges)
                e = edges[idx]
                uv = if agg == :max
                    max(vvals[e[1]], vvals[e[2]])
                elseif agg == :min
                    min(vvals[e[1]], vvals[e[2]])
                elseif agg == :sum
                    vvals[e[1]] + vvals[e[2]]
                elseif agg == :mean
                    (vvals[e[1]] + vvals[e[2]]) / 2
                else
                    throw(ArgumentError("Unsupported simplex_agg=$(agg). Supported: :max, :min, :sum, :mean"))
                end
                grades[t] = (edge_dists[idx], uv)
                t += 1
            end
        else
            deg = _vertex_degree_scores(n, edges)
            grades = Vector{NTuple{2,Float64}}(undef, n + length(edges))
            t = 1
            for i in 1:n
                grades[t] = (0.0, Float64(deg[i]))
                t += 1
            end
            for idx in eachindex(edges)
                e = edges[idx]
                grades[t] = (edge_dists[idx], Float64(max(deg[e[1]], deg[e[2]])))
                t += 1
            end
        end

        return _materialize_simplicial_output(simplices, grades, spec; return_simplex_tree=return_simplex_tree)
    end

    if _POINTCLOUD_DIM2_PACKED_KERNEL[] &&
       max_dim == 2 &&
       construction.sparsify == :none &&
       construction.collapse == :none &&
       (kind == :rips || kind == :rips_density || kind == :rips_codensity ||
        kind == :rips_lowerstar || kind == :function_rips || kind == :degree_rips)
        return _graded_complex_from_point_cloud_dim2_packed(
            data,
            spec;
            return_simplex_tree=return_simplex_tree,
        )
    end

    if kind == :core
        return _graded_complex_from_point_cloud_core(data, spec; return_simplex_tree=return_simplex_tree)
    elseif kind == :rhomboid
        return _graded_complex_from_point_cloud_rhomboid(data, spec; return_simplex_tree=return_simplex_tree)
    end

    # Dense non-sparse Rips-like path: avoid n x n dense distance matrices.
    # Streaming mode computes simplex diameters directly from point coordinates.
    # A packed upper-triangular fallback remains available for benchmarking.
    use_stream = _POINTCLOUD_STREAM_DIST_NONSPARSE[]
    dist_packed = use_stream ? Float64[] : _point_cloud_pairwise_packed(points)

    # simplices by dimension (dense non-sparse path only)
    simplices = Vector{Vector{Vector{Int}}}(undef, max_dim + 1)
    simplices[1] = [[i] for i in 1:n]
    total = big(length(simplices[1]))
    if max_dim >= 1
        edge_count = binomial(big(n), big(2))
        if edge_count > big(typemax(Int))
            throw(ArgumentError("Ingestion construction budget check overflow for edge count=$(edge_count)."))
        end
        _construction_check_max_edges!(Int(edge_count), spec)
    end
    for k in 2:max_dim+1
        count_k = _construction_precheck_combination_enumeration!(n, k, total, spec)
        sims = _combinations(n, k)
        simplices[k] = sims
        total += count_k
    end

    _construction_check_memory_budget!(_estimate_dense_bytes_from_cell_counts(BigInt[length(s) for s in simplices]), spec)

    if kind == :rips
        grades = Vector{NTuple{1,Float64}}()
        for s in simplices[1]
            push!(grades, (0.0,))
        end
        for k in 2:max_dim+1
            for s in simplices[k]
                maxd = use_stream ? _simplex_max_pair_distance_points(points, s) : begin
                    md = 0.0
                    for i in 1:length(s)
                        for j in (i+1):length(s)
                            d = _packed_pair_distance(dist_packed, n, s[i], s[j])
                            d > md && (md = d)
                        end
                    end
                    md
                end
                push!(grades, (maxd,))
            end
        end
    elseif kind == :rips_density
        knn_k = get(spec.params, :density_k, 2)
        nn_backend = _pointcloud_nn_backend(spec)
        approx_candidates = _pointcloud_nn_approx_candidates(spec)
        dens = _knn_distances(points, knn_k; backend=nn_backend, approx_candidates=approx_candidates)
        grades = Vector{NTuple{2,Float64}}()
        for s in simplices[1]
            v = s[1]
            push!(grades, (0.0, dens[v]))
        end
        for k in 2:max_dim+1
            for s in simplices[k]
                maxd = use_stream ? _simplex_max_pair_distance_points(points, s) : 0.0
                maxden = 0.0
                for i in 1:length(s)
                    vi = s[i]
                    if dens[vi] > maxden
                        maxden = dens[vi]
                    end
                    if !use_stream
                        for j in (i+1):length(s)
                            d = _packed_pair_distance(dist_packed, n, s[i], s[j])
                            if d > maxd
                                maxd = d
                            end
                        end
                    end
                end
                push!(grades, (maxd, maxden))
            end
        end
    elseif kind == :rips_codensity || kind == :rips_lowerstar || kind == :function_rips
        vvals = _point_vertex_values(points, spec)
        agg = get(spec.params, :simplex_agg, :max)
        grades = Vector{NTuple{2,Float64}}()
        for s in simplices[1]
            v = s[1]
            push!(grades, (0.0, vvals[v]))
        end
        for k in 2:max_dim+1
            for s in simplices[k]
                maxd = use_stream ? _simplex_max_pair_distance_points(points, s) : begin
                    md = 0.0
                    for i in 1:length(s)
                        for j in (i+1):length(s)
                            d = _packed_pair_distance(dist_packed, n, s[i], s[j])
                            if d > md
                                md = d
                            end
                        end
                    end
                    md
                end
                push!(grades, (maxd, _aggregate_indexed(vvals, s, agg)))
            end
        end
    elseif kind == :degree_rips
        deg = (length(simplices) >= 2) ? _vertex_degree_scores(n, simplices[2]) : zeros(Int, n)
        grades = Vector{NTuple{2,Float64}}()
        for s in simplices[1]
            v = s[1]
            push!(grades, (0.0, Float64(deg[v])))
        end
        for k in 2:max_dim+1
            for s in simplices[k]
                maxd = use_stream ? _simplex_max_pair_distance_points(points, s) : begin
                    md = 0.0
                    for i in 1:length(s)
                        for j in (i+1):length(s)
                            d = _packed_pair_distance(dist_packed, n, s[i], s[j])
                            if d > md
                                md = d
                            end
                        end
                    end
                    md
                end
                maxdeg = 0
                @inbounds for v in s
                    dv = deg[v]
                    dv > maxdeg && (maxdeg = dv)
                end
                push!(grades, (maxd, Float64(maxdeg)))
            end
        end
    else
        error("Unsupported point cloud filtration kind: $(kind).")
    end

    return _materialize_simplicial_output(simplices, grades, spec; return_simplex_tree=return_simplex_tree)
end

function _graded_complex_from_graph(data::GraphData, spec::FiltrationSpec;
                                    return_simplex_tree::Bool=false)
    kind = spec.kind
    n = data.n
    edges = data.edges
    if kind == :graph_lower_star || kind == :clique_lower_star
        vertex_grades = _graph_vertex_values(data, spec)
        if kind == :graph_lower_star
            params = _filter_params(spec.params, Symbol[:max_dim, :lift])
            return _graph_lifted_complex(
                data,
                FiltrationSpec(; kind=kind, lift=:lower_star, params...),
                vertex_grades;
                return_simplex_tree=return_simplex_tree,
            )
        end
        params = _filter_params(spec.params, Symbol[:lift])
        return _graph_lifted_complex(
            data,
            FiltrationSpec(; kind=kind, lift=:clique, params...),
            vertex_grades;
            return_simplex_tree=return_simplex_tree,
        )
    elseif kind == :graph_centrality
        vals = _graph_centrality_values(data, spec)
        return _graph_lifted_complex(data, spec, [(v,) for v in vals]; return_simplex_tree=return_simplex_tree)
    elseif kind == :graph_geodesic
        vals = _graph_geodesic_values(data, spec)
        return _graph_lifted_complex(data, spec, [(v,) for v in vals]; return_simplex_tree=return_simplex_tree)
    elseif kind == :graph_function_geodesic_bifiltration
        geo = _graph_geodesic_values(data, spec)
        usr = _graph_vertex_scalar_values(data, spec; required=true)
        length(usr) == n || error("vertex function values length mismatch.")
        pairs = NTuple{2,Float64}[(geo[i], usr[i]) for i in 1:n]
        return _graph_lifted_complex(data, spec, pairs; return_simplex_tree=return_simplex_tree)
    elseif kind == :graph_weight_threshold
        return _graph_weight_threshold_complex(data, spec; return_simplex_tree=return_simplex_tree)
    elseif kind == :core
        vals = _graph_vertex_scalar_values(data, spec; required=false)
        core = _core_numbers(n, edges)
        packed_edges = Vector{NTuple{2,Int}}(undef, length(edges))
        @inbounds for idx in eachindex(edges)
            u, v = edges[idx]
            packed_edges[idx] = (u, v)
        end
        m = length(packed_edges)
        grades = Vector{NTuple{2,Float64}}(undef, n + m)
        t = 1
        @inbounds for i in 1:n
            grades[t] = (vals[i], Float64(core[i]))
            t += 1
        end
        @inbounds for idx in 1:m
            u, v = packed_edges[idx]
            grades[t] = (max(vals[u], vals[v]), Float64(min(core[u], core[v])))
            t += 1
        end
        return _materialize_point_cloud_dim01(
            n, true, packed_edges, grades, spec; return_simplex_tree=return_simplex_tree
        )
    elseif kind == :rhomboid
        vals = _graph_vertex_scalar_values(data, spec; required=true)
        construction = _construction_from_params(spec.params)
        construction.collapse == :none || error("construction.collapse=$(construction.collapse) is unsupported for graph rhomboid ingestion.")
        construction.sparsify == :none || error("construction.sparsify=$(construction.sparsify) is unsupported for graph rhomboid ingestion.")
        max_dim = Int(get(spec.params, :max_dim, 2))
        _construction_check_max_edges!(length(edges), spec)
        simplices = Vector{Vector{Vector{Int}}}(undef, max_dim + 1)
        simplices[1] = [[i] for i in 1:n]
        packed = nothing
        adj_lists = nothing
        total = big(n)
        for k in 2:max_dim+1
            sims, packed, adj_lists = _enumerate_cliques_k_cached(
                edges, n, k, spec, total;
                context="graph rhomboid",
                packed=packed,
                adj_lists=adj_lists,
            )
            simplices[k] = sims
            total += length(sims)
            _construction_check_max_simplices!(total, k - 1, spec)
        end
        grades = NTuple{2,Float64}[]
        for s in simplices[1]
            v = vals[s[1]]
            push!(grades, (v, v))
        end
        for k in 2:max_dim+1
            for s in simplices[k]
                vv = Float64[vals[v] for v in s]
                push!(grades, (minimum(vv), maximum(vv)))
            end
        end
        return _materialize_simplicial_output(simplices, grades, spec; return_simplex_tree=return_simplex_tree)
    elseif kind == :edge_weighted
        weights = get(spec.params, :edge_weights, nothing)
        weights === nothing && error("edge_weighted requires edge_weights.")
        length(weights) == length(edges) || error("edge_weights length mismatch.")
        grades = Vector{NTuple{1,Float64}}()
        for _ in 1:n
            push!(grades, (0.0,))
        end
        for w in weights
            push!(grades, (Float64(w),))
        end
        simplices0 = [ [i] for i in 1:n ]
        simplices1 = [ [u, v] for (u, v) in edges ]
        simplices = [simplices0, simplices1]
        return _materialize_simplicial_output(simplices, grades, spec; return_simplex_tree=return_simplex_tree)
    else
        error("Unsupported graph filtration kind: $(kind).")
    end
end

function _distance_transform_bruteforce(mask::AbstractArray{Bool})
    dims = size(mask)
    N = length(dims)
    coords = CartesianIndices(mask)
    true_pts = [c for c in coords if mask[c]]
    out = zeros(Float64, dims)
    for c in coords
        if mask[c]
            out[c] = 0.0
            continue
        end
        best = Inf
        for t in true_pts
            s = 0.0
            for i in 1:N
                d = (c[i] - t[i])
                s += d * d
            end
            if s < best
                best = s
            end
        end
        out[c] = sqrt(best)
    end
    return out
end

@inline function _edt_1d_squared!(out::Vector{Float64},
                                  f::Vector{Float64},
                                  v::Vector{Int},
                                  z::Vector{Float64})
    n = length(f)
    n == 0 && return out
    first_finite = 0
    @inbounds for i in 1:n
        if isfinite(f[i])
            first_finite = i
            break
        end
    end
    if first_finite == 0
        fill!(out, Inf)
        return out
    end

    k = 1
    v[1] = first_finite
    z[1] = -Inf
    z[2] = Inf

    @inbounds for q in (first_finite + 1):n
        fq = f[q]
        isfinite(fq) || continue
        q0 = q - 1
        while true
            p = v[k]
            p0 = p - 1
            s = ((fq + q0 * q0) - (f[p] + p0 * p0)) / (2.0 * (q - p))
            if s > z[k]
                k += 1
                v[k] = q
                z[k] = s
                z[k + 1] = Inf
                break
            end
            k == 1 && break
            k -= 1
        end
        if k == 1
            p = v[1]
            p0 = p - 1
            s = ((fq + q0 * q0) - (f[p] + p0 * p0)) / (2.0 * (q - p))
            if s <= z[1]
                v[1] = q
                z[1] = -Inf
                z[2] = Inf
            end
        end
    end

    kk = 1
    @inbounds for q in 1:n
        q0 = q - 1
        while z[kk + 1] < q0
            kk += 1
        end
        dq = q - v[kk]
        out[q] = dq * dq + f[v[kk]]
    end
    return out
end

function _distance_transform_2d(mask::AbstractMatrix{Bool})
    nx, ny = size(mask)
    (nx == 0 || ny == 0) && return zeros(Float64, size(mask))
    any(mask) || return fill(Inf, size(mask))
    all(mask) && return zeros(Float64, size(mask))

    tmp = Matrix{Float64}(undef, nx, ny)
    out = Matrix{Float64}(undef, nx, ny)

    row_f = Vector{Float64}(undef, nx)
    row_out = Vector{Float64}(undef, nx)
    row_v = Vector{Int}(undef, nx)
    row_z = Vector{Float64}(undef, nx + 1)

    @inbounds for j in 1:ny
        for i in 1:nx
            row_f[i] = mask[i, j] ? 0.0 : Inf
        end
        _edt_1d_squared!(row_out, row_f, row_v, row_z)
        for i in 1:nx
            tmp[i, j] = row_out[i]
        end
    end

    col_f = Vector{Float64}(undef, ny)
    col_out = Vector{Float64}(undef, ny)
    col_v = Vector{Int}(undef, ny)
    col_z = Vector{Float64}(undef, ny + 1)
    @inbounds for i in 1:nx
        for j in 1:ny
            col_f[j] = tmp[i, j]
        end
        _edt_1d_squared!(col_out, col_f, col_v, col_z)
        for j in 1:ny
            out[i, j] = sqrt(col_out[j])
        end
    end

    return out
end

function _distance_transform(mask::AbstractArray{Bool})
    ndims(mask) == 2 && return _distance_transform_2d(mask)
    return _distance_transform_bruteforce(mask)
end

@inline function _distance_transform_cache_key(mask::AbstractArray{Bool})
    return (
        :image_distance_transform,
        UInt(objectid(mask)),
        Tuple(size(mask)),
        UInt(hash(mask)),
        eltype(mask),
    )
end

function _distance_transform_cached(mask::AbstractArray{Bool};
                                   cache::Union{Nothing,EncodingCache}=nothing)
    cache === nothing && return _distance_transform(mask)
    key = _distance_transform_cache_key(mask)
    cached = _get_geometry_cached(cache, key)
    if cached isa AbstractArray{<:Real} && size(cached) == size(mask)
        return cached
    end
    dist = _distance_transform(mask)
    return _set_geometry_cached!(cache, key, dist)
end

function _cubical_structure_2d(dims::NTuple{2,Int})
    nx, ny = dims
    TCell = Tuple{NTuple{2,Int},NTuple{2,Int}}
    mask0 = (0, 0)
    mask10 = (1, 0)
    mask01 = (0, 1)
    mask11 = (1, 1)

    nv = nx * ny
    neh = max(nx - 1, 0) * ny
    nev = nx * max(ny - 1, 0)
    ne = neh + nev
    nf = max(nx - 1, 0) * max(ny - 1, 0)

    @inline vid(i::Int, j::Int) = i + (j - 1) * nx
    @inline hid(i::Int, j::Int) = i + (j - 1) * (nx - 1)
    @inline vidx(i::Int, j::Int) = neh + i + (j - 1) * nx
    @inline fid(i::Int, j::Int) = i + (j - 1) * (nx - 1)

    cells0 = Vector{TCell}(undef, nv)
    t = 1
    @inbounds for j in 1:ny
        for i in 1:nx
            cells0[t] = ((i, j), mask0)
            t += 1
        end
    end

    cells1 = Vector{TCell}(undef, ne)
    t = 1
    @inbounds for j in 1:ny
        for i in 1:(nx - 1)
            cells1[t] = ((i, j), mask10)
            t += 1
        end
    end
    @inbounds for j in 1:(ny - 1)
        for i in 1:nx
            cells1[t] = ((i, j), mask01)
            t += 1
        end
    end

    cells2 = Vector{TCell}(undef, nf)
    t = 1
    @inbounds for j in 1:(ny - 1)
        for i in 1:(nx - 1)
            cells2[t] = ((i, j), mask11)
            t += 1
        end
    end

    I1 = Vector{Int}(undef, 2 * ne)
    J1 = Vector{Int}(undef, 2 * ne)
    V1 = Vector{Int}(undef, 2 * ne)
    t = 1
    @inbounds for j in 1:ny
        for i in 1:(nx - 1)
            col = hid(i, j)
            I1[t] = vid(i, j); J1[t] = col; V1[t] = 1; t += 1
            I1[t] = vid(i + 1, j); J1[t] = col; V1[t] = -1; t += 1
        end
    end
    @inbounds for j in 1:(ny - 1)
        for i in 1:nx
            col = vidx(i, j)
            I1[t] = vid(i, j); J1[t] = col; V1[t] = 1; t += 1
            I1[t] = vid(i, j + 1); J1[t] = col; V1[t] = -1; t += 1
        end
    end
    b1 = sparse(I1, J1, V1, nv, ne)

    I2 = Vector{Int}(undef, 4 * nf)
    J2 = Vector{Int}(undef, 4 * nf)
    V2 = Vector{Int}(undef, 4 * nf)
    t = 1
    @inbounds for j in 1:(ny - 1)
        for i in 1:(nx - 1)
            col = fid(i, j)
            # axis 1 faces: low/high vertical edges.
            I2[t] = vidx(i, j); J2[t] = col; V2[t] = 1; t += 1
            I2[t] = vidx(i + 1, j); J2[t] = col; V2[t] = -1; t += 1
            # axis 2 faces: low/high horizontal edges.
            I2[t] = hid(i, j); J2[t] = col; V2[t] = -1; t += 1
            I2[t] = hid(i, j + 1); J2[t] = col; V2[t] = 1; t += 1
        end
    end
    b2 = sparse(I2, J2, V2, ne, nf)

    cells_by_dim = Vector{Vector{TCell}}(undef, 3)
    cells_by_dim[1] = cells0
    cells_by_dim[2] = cells1
    cells_by_dim[3] = cells2
    boundaries = SparseMatrixCSC{Int,Int}[b1, b2]
    return (cells_by_dim=cells_by_dim, cell_index=nothing, boundaries=boundaries)
end

function _cubical_structure(dims::NTuple{N,Int};
                            cache::Union{Nothing,EncodingCache}=nothing) where {N}
    key = Tuple(dims)
    if cache !== nothing
        Base.lock(cache.lock)
        try
            entry = get(cache.cubical, key, nothing)
            entry === nothing || return entry.value
        finally
            Base.unlock(cache.lock)
        end
    end

    cached = if N == 2 && _CUBICAL_2D_FASTPATH[]
        _cubical_structure_2d((dims[1], dims[2]))
    else
        cells_by_dim = Vector{Vector{Tuple{NTuple{N,Int},NTuple{N,Int}}}}(undef, N + 1)
        cell_index = Vector{Dict{Tuple{NTuple{N,Int},NTuple{N,Int}},Int}}(undef, N + 1)
        for k in 0:N
            cells_by_dim[k+1] = Tuple{NTuple{N,Int},NTuple{N,Int}}[]
            cell_index[k+1] = Dict{Tuple{NTuple{N,Int},NTuple{N,Int}},Int}()
        end

        for _ in _combinations(N, 0)
            mask = ntuple(i -> 0, N)
            for coords in CartesianIndices(Tuple(dims))
                origin = ntuple(i -> coords[i], N)
                key2 = (origin, mask)
                push!(cells_by_dim[1], key2)
            end
        end
        for k in 1:N
            for mask_idxs in _combinations(N, k)
                mask = ntuple(i -> (i in mask_idxs ? 1 : 0), N)
                ranges = Vector{UnitRange{Int}}(undef, N)
                for i in 1:N
                    ranges[i] = 1:(dims[i] - mask[i])
                end
                for coords in CartesianIndices(Tuple(ranges))
                    origin = ntuple(i -> coords[i], N)
                    key2 = (origin, mask)
                    push!(cells_by_dim[k+1], key2)
                end
            end
        end

        for k in 0:N
            for (i, key2) in enumerate(cells_by_dim[k+1])
                cell_index[k+1][key2] = i
            end
        end

        boundaries = SparseMatrixCSC{Int,Int}[]
        for k in 1:N
            I = Int[]
            J = Int[]
            V = Int[]
            for (j, (origin, mask)) in enumerate(cells_by_dim[k+1])
                axes = [i for i in 1:N if mask[i] == 1]
                for (pos, axis) in enumerate(axes)
                    mask_face = ntuple(i -> (i == axis ? 0 : mask[i]), N)
                    origin_low = origin
                    origin_high = ntuple(i -> (i == axis ? origin[i] + 1 : origin[i]), N)
                    low_key = (origin_low, mask_face)
                    high_key = (origin_high, mask_face)
                    row_low = cell_index[k][low_key]
                    row_high = cell_index[k][high_key]
                    sign_low = isodd(pos) ? 1 : -1
                    sign_high = -sign_low
                    push!(I, row_low); push!(J, j); push!(V, sign_low)
                    push!(I, row_high); push!(J, j); push!(V, sign_high)
                end
            end
            push!(boundaries, sparse(I, J, V, length(cells_by_dim[k]), length(cells_by_dim[k+1])))
        end
        (cells_by_dim=cells_by_dim, cell_index=cell_index, boundaries=boundaries)
    end

    if cache !== nothing
        Base.lock(cache.lock)
        try
            entry = get(cache.cubical, key, nothing)
            if entry === nothing
                cache.cubical[key] = CubicalCachePayload(cached)
                return cached
            end
            return entry.value
        finally
            Base.unlock(cache.lock)
        end
    end
    return cached
end

function _cubical_grades_2d(channels::Vector{<:AbstractArray})
    img = channels[1]
    nx, ny = size(img)
    C = length(channels)
    neh = max(nx - 1, 0) * ny
    nev = nx * max(ny - 1, 0)
    nf = max(nx - 1, 0) * max(ny - 1, 0)
    total = nx * ny + neh + nev + nf
    grades = Vector{NTuple{C,Float64}}(undef, total)
    t = 1

    @inbounds for j in 1:ny
        for i in 1:nx
            grades[t] = ntuple(ch -> Float64(channels[ch][i, j]), C)
            t += 1
        end
    end
    @inbounds for j in 1:ny
        for i in 1:(nx - 1)
            grades[t] = ntuple(ch -> begin
                a = Float64(channels[ch][i, j])
                b = Float64(channels[ch][i + 1, j])
                a >= b ? a : b
            end, C)
            t += 1
        end
    end
    @inbounds for j in 1:(ny - 1)
        for i in 1:nx
            grades[t] = ntuple(ch -> begin
                a = Float64(channels[ch][i, j])
                b = Float64(channels[ch][i, j + 1])
                a >= b ? a : b
            end, C)
            t += 1
        end
    end
    @inbounds for j in 1:(ny - 1)
        for i in 1:(nx - 1)
            grades[t] = ntuple(ch -> begin
                a = Float64(channels[ch][i, j])
                b = Float64(channels[ch][i + 1, j])
                c = Float64(channels[ch][i, j + 1])
                d = Float64(channels[ch][i + 1, j + 1])
                m1 = a >= b ? a : b
                m2 = c >= d ? c : d
                m1 >= m2 ? m1 : m2
            end, C)
            t += 1
        end
    end
    return grades
end

function _cubical_grades_2d_image_distance(img::AbstractArray,
                                           dist::AbstractArray)
    size(img) == size(dist) || error("image and distance channels must have the same size.")
    nx, ny = size(img)
    neh = max(nx - 1, 0) * ny
    nev = nx * max(ny - 1, 0)
    nf = max(nx - 1, 0) * max(ny - 1, 0)
    total = nx * ny + neh + nev + nf
    grades = Vector{NTuple{2,Float64}}(undef, total)
    t = 1

    @inbounds for j in 1:ny
        for i in 1:nx
            grades[t] = (Float64(img[i, j]), Float64(dist[i, j]))
            t += 1
        end
    end
    @inbounds for j in 1:ny
        for i in 1:(nx - 1)
            v1a = Float64(img[i, j])
            v1b = Float64(img[i + 1, j])
            v2a = Float64(dist[i, j])
            v2b = Float64(dist[i + 1, j])
            grades[t] = (
                v1a >= v1b ? v1a : v1b,
                v2a >= v2b ? v2a : v2b,
            )
            t += 1
        end
    end
    @inbounds for j in 1:(ny - 1)
        for i in 1:nx
            v1a = Float64(img[i, j])
            v1b = Float64(img[i, j + 1])
            v2a = Float64(dist[i, j])
            v2b = Float64(dist[i, j + 1])
            grades[t] = (
                v1a >= v1b ? v1a : v1b,
                v2a >= v2b ? v2a : v2b,
            )
            t += 1
        end
    end
    @inbounds for j in 1:(ny - 1)
        for i in 1:(nx - 1)
            a1 = Float64(img[i, j]); b1 = Float64(img[i + 1, j])
            c1 = Float64(img[i, j + 1]); d1 = Float64(img[i + 1, j + 1])
            a2 = Float64(dist[i, j]); b2 = Float64(dist[i + 1, j])
            c2 = Float64(dist[i, j + 1]); d2 = Float64(dist[i + 1, j + 1])
            m11 = a1 >= b1 ? a1 : b1
            m12 = c1 >= d1 ? c1 : d1
            m21 = a2 >= b2 ? a2 : b2
            m22 = c2 >= d2 ? c2 : d2
            grades[t] = (
                m11 >= m12 ? m11 : m12,
                m21 >= m22 ? m21 : m22,
            )
            t += 1
        end
    end
    return grades
end

function _graded_complex_from_image_channels(channels::Vector{<:AbstractArray},
                                             spec::FiltrationSpec;
                                             cache::Union{Nothing,EncodingCache}=nothing)
    img = channels[1]
    dims = size(img)
    for c in channels
        size(c) == dims || error("All channels must have the same size.")
    end
    N = length(dims)
    C = length(channels)

    cached = _cubical_structure(dims; cache=cache)
    cells_by_dim = cached.cells_by_dim
    grades = if N == 2 && _CUBICAL_2D_FASTPATH[]
        if C == 2 && spec.kind == :image_distance_bifiltration
            _cubical_grades_2d_image_distance(channels[1], channels[2])
        else
            _cubical_grades_2d(channels)
        end
    else
        out = Vector{NTuple{C,Float64}}()
        for k in 0:N
            for (origin, mask) in cells_by_dim[k+1]
                maxv = fill(-Inf, C)
                ranges = Vector{UnitRange{Int}}(undef, N)
                for i in 1:N
                    ranges[i] = origin[i]:(origin[i] + mask[i])
                end
                for coords in CartesianIndices(Tuple(ranges))
                    for ch in 1:C
                        v = Float64(channels[ch][coords])
                        if v > maxv[ch]
                            maxv[ch] = v
                        end
                    end
                end
                push!(out, ntuple(i -> maxv[i], C))
            end
        end
        out
    end

    cells = [collect(1:length(cells_by_dim[k+1])) for k in 0:N]
    G = GradedComplex(cells, cached.boundaries, grades)
    orientation = get(spec.params, :orientation, ntuple(_ -> 1, C))
    axes = get(spec.params, :axes, _axes_from_grades(grades, C; orientation=orientation))
    return G, axes, orientation
end

function _graded_complex_from_image(data::ImageNd, spec::FiltrationSpec;
                                    cache::Union{Nothing,EncodingCache}=nothing)
    if haskey(spec.params, :channels)
        chans = spec.params[:channels]
        return _graded_complex_from_image_channels(chans, spec; cache=cache)
    elseif spec.kind == :image_distance_bifiltration
        mask = get(spec.params, :mask, nothing)
        mask === nothing && error("image_distance_bifiltration requires mask.")
        dist = _distance_transform_cached(mask; cache=cache)
        chans = [data.data, dist]
        return _graded_complex_from_image_channels(chans, spec; cache=cache)
    else
        return _graded_complex_from_image_channels([data.data], spec; cache=cache)
    end
end

function _graded_complex_from_data(data, spec::FiltrationSpec;
                                   cache::Union{Nothing,EncodingCache}=nothing)
    if data isa GradedComplex
        N = length(data.grades[1])
        orientation = get(spec.params, :orientation, ntuple(_ -> 1, N))
        axes = get(spec.params, :axes, _axes_from_grades(data.grades, N; orientation=orientation))
        return data, axes, orientation
    elseif data isa SimplexTreeMulti
        N = length(data.grade_data[1])
        orientation = get(spec.params, :orientation, ntuple(_ -> 1, N))
        axes = get(spec.params, :axes, _axes_from_simplex_tree(data; orientation=orientation))
        return _graded_complex_from_simplex_tree(data), axes, orientation
    elseif data isa MultiCriticalGradedComplex
        _, grade_data = _packed_multigrade_storage(data)
        isempty(grade_data) && error("MultiCriticalGradedComplex has no grades.")
        N = length(grade_data[1])
        orientation = get(spec.params, :orientation, ntuple(_ -> 1, N))
        axes = get(spec.params, :axes, _axes_from_grades(grade_data, N; orientation=orientation))
        return data, axes, orientation
    elseif data isa PointCloud
        return _graded_complex_from_point_cloud(data, spec; cache=cache)
    elseif data isa ImageNd
        return _graded_complex_from_image(data, spec; cache=cache)
    elseif data isa GraphData
        return _graded_complex_from_graph(data, spec)
    elseif data isa EmbeddedPlanarGraph2D
        if spec.kind == :wing_vein_bifiltration
            grid = get(spec.params, :grid, (32, 32))
            bbox = get(spec.params, :bbox, data.bbox)
            bbox === nothing && error("wing_vein_bifiltration requires bbox.")
            xmin, xmax, ymin, ymax = bbox
            nx, ny = grid
            xs = range(xmin, xmax; length=nx)
            ys = range(ymin, ymax; length=ny)
            distV = zeros(Float64, nx, ny)
            distE = zeros(Float64, nx, ny)
            verts = data.vertices
            segments = Vector{Tuple{Vector{Float64},Vector{Float64}}}()
            if data.polylines === nothing
                for (u, v) in data.edges
                    push!(segments, (verts[u], verts[v]))
                end
            else
                for poly in data.polylines
                    for i in 1:(length(poly)-1)
                        push!(segments, (poly[i], poly[i+1]))
                    end
                end
            end
            for i in 1:nx, j in 1:ny
                x = xs[i]; y = ys[j]
                dv = Inf
                for v in verts
                    d = hypot(x - v[1], y - v[2])
                    if d < dv
                        dv = d
                    end
                end
                distV[i, j] = dv
                de = Inf
                for (a, b) in segments
                    ax, ay = a[1], a[2]
                    bx, by = b[1], b[2]
                    vx = bx - ax
                    vy = by - ay
                    wx = x - ax
                    wy = y - ay
                    t = (vx*wx + vy*wy) / (vx*vx + vy*vy)
                    if t < 0
                        px, py = ax, ay
                    elseif t > 1
                        px, py = bx, by
                    else
                        px, py = ax + t*vx, ay + t*vy
                    end
                    d = hypot(x - px, y - py)
                    if d < de
                        de = d
                    end
                end
                distE[i, j] = de
            end
            chans = [-distV, distE]
            img = ImageNd(chans[1])
            spec2 = FiltrationSpec(; kind=:image_distance_bifiltration, channels=chans,
                                    orientation=get(spec.params, :orientation, (-1, 1)))
            return _graded_complex_from_image(img, spec2; cache=cache)
        else
            # Treat as a graph with externally supplied vertex grades
            gspec = FiltrationSpec(; kind=spec.kind, spec.params...)
            gdata = GraphData(length(data.vertices), data.edges;
                              coords=data.vertices, weights=nothing, T=eltype(data.vertices[1]))
            return _graded_complex_from_graph(gdata, gspec)
        end
    else
        error("Unsupported dataset type for encode(data, filtration).")
    end
end

"""
    build_graded_complex(data, filtration; cache=nothing) -> GradedComplexBuildResult

Build the graded-complex stage of the ingestion pipeline and return a typed
owner-level wrapper around `(G, axes, orientation)`.

Cheap-first workflow
- call `estimate_ingestion(...)` first when you only need a size/budget check;
- call `build_graded_complex(...)` when you explicitly want the graded complex
  and its grade axes without running the rest of the encoding pipeline;
- use `run_ingestion(...; stage=:graded_complex)` when you want the canonical
  staged workflow surface instead of this owner-level builder.

Custom filtration families should register a builder via
[`register_filtration_family!`](@ref). Registered builders continue to return
the internal raw tuple `(G, axes, orientation)`, which this public wrapper
normalizes into [`GradedComplexBuildResult`](@ref).
"""
@inline function _normalize_build_graded_complex_output(out, kind::Symbol)
    if out isa GradedComplexBuildResult
        return graded_complex(out), grade_axes(out), grade_orientation(out)
    elseif out isa Tuple && length(out) == 3
        return out
    end
    throw(ArgumentError("build_graded_complex kind=:$kind must return `(G, axes, orientation)` or `GradedComplexBuildResult`."))
end

function _build_graded_complex_tuple(data, filtration::AbstractFiltration;
                                     cache::Union{Nothing,EncodingCache}=nothing)
    spec = try
        _filtration_spec(filtration)
    catch err
        if !(err isa ArgumentError)
            rethrow()
        end
        nothing
    end
    if spec !== nothing
        entry = _filtration_registry_get(spec.kind)
        if entry !== nothing && !(spec.kind in _BUILTIN_FILTRATION_KINDS)
            return _normalize_build_graded_complex_output(entry.builder(data, filtration; cache=cache), spec.kind)
        end
        return _graded_complex_from_data(data, spec; cache=cache)
    end
    kind = filtration_kind(typeof(filtration))
    entry = _filtration_registry_get(kind)
    entry === nothing &&
        throw(ArgumentError("No graded-complex builder for filtration kind=:$kind. Register kind=:$kind via `register_filtration_family!` or define `_build_graded_complex_tuple(data, ::$(typeof(filtration)); cache=...)`."))
    return _normalize_build_graded_complex_output(entry.builder(data, filtration; cache=cache), kind)
end

function build_graded_complex(data, filtration::AbstractFiltration;
                              cache::Union{Nothing,EncodingCache}=nothing)
    return GradedComplexBuildResult(_build_graded_complex_tuple(data, filtration; cache=cache))
end

function _graded_complex_from_data(data, filtration::AbstractFiltration;
                                   cache::Union{Nothing,EncodingCache}=nothing)
    return _build_graded_complex_tuple(data, filtration; cache=cache)
end

@inline function _graded_complex_build_describe(res::GradedComplexBuildResult)
    G = graded_complex(res)
    d = describe(G)
    return (
        kind = :graded_complex_build_result,
        complex_kind = get(d, :kind, Symbol(nameof(typeof(G)))),
        parameter_dim = get(d, :parameter_dim, length(grade_axes(res))),
        cell_counts = get(d, :cell_counts, nothing),
        axis_sizes = ntuple(i -> length(grade_axes(res)[i]), length(grade_axes(res))),
        orientation = grade_orientation(res),
    )
end

"""
    describe(res::GradedComplexBuildResult) -> NamedTuple

Return a compact semantic summary of a direct graded-complex build result.

This is the shared inspection surface for the owner-level
[`build_graded_complex`](@ref) contract. Use the semantic accessors
[`graded_complex`](@ref), [`grade_axes`](@ref), and [`grade_orientation`](@ref)
when you need the full payload.
"""
describe(res::GradedComplexBuildResult) = _graded_complex_build_describe(res)

"""
    graded_complex_build_summary(res::GradedComplexBuildResult) -> NamedTuple

Owner-local summary helper for direct graded-complex build results.
"""
graded_complex_build_summary(res::GradedComplexBuildResult) = _graded_complex_build_describe(res)

function Base.show(io::IO, res::GradedComplexBuildResult)
    d = _graded_complex_build_describe(res)
    print(io, "GradedComplexBuildResult(kind=", d.complex_kind,
          ", parameter_dim=", d.parameter_dim,
          ", axis_sizes=", d.axis_sizes, ")")
end

function Base.show(io::IO, ::MIME"text/plain", res::GradedComplexBuildResult)
    d = _graded_complex_build_describe(res)
    print(io, "GradedComplexBuildResult",
          "\n  complex_kind: ", d.complex_kind,
          "\n  parameter_dim: ", d.parameter_dim,
          "\n  cell_counts: ", repr(d.cell_counts),
          "\n  axis_sizes: ", repr(d.axis_sizes),
          "\n  orientation: ", repr(d.orientation))
end

"""
    describe(res::PointCodensityResult) -> NamedTuple

Return a compact semantic summary of pointwise codensity values computed for a
point cloud and a `:rips_codensity` filtration contract.
"""
describe(res::PointCodensityResult) = _point_codensity_describe(res)

function Base.show(io::IO, res::PointCodensityResult)
    d = _point_codensity_describe(res)
    print(io, "PointCodensityResult(npoints=", d.npoints,
          ", ambient_dim=", d.ambient_dim,
          ", dtm_mass=", d.dtm_mass,
          ", k=", d.neighbor_count, ")")
end

function Base.show(io::IO, ::MIME"text/plain", res::PointCodensityResult)
    d = _point_codensity_describe(res)
    print(io, "PointCodensityResult",
          "\n  npoints: ", d.npoints,
          "\n  ambient_dim: ", d.ambient_dim,
          "\n  filtration_kind: ", d.filtration_kind,
          "\n  dtm_mass: ", d.dtm_mass,
          "\n  neighbor_count: ", d.neighbor_count,
          "\n  value_range: ", repr(d.value_range))
end

function _simplex_tree_from_data(data, spec::FiltrationSpec;
                                 cache::Union{Nothing,EncodingCache}=nothing)
    if data isa SimplexTreeMulti
        N = length(data.grade_data[1])
        orientation = get(spec.params, :orientation, ntuple(_ -> 1, N))
        axes = get(spec.params, :axes, _axes_from_simplex_tree(data; orientation=orientation))
        return data, axes, orientation
    end
    if data isa PointCloud
        return _graded_complex_from_point_cloud(data, spec; return_simplex_tree=true, cache=cache)
    elseif data isa GraphData
        return _graded_complex_from_graph(data, spec; return_simplex_tree=true)
    elseif data isa EmbeddedPlanarGraph2D && spec.kind != :wing_vein_bifiltration
        gspec = FiltrationSpec(; kind=spec.kind, spec.params...)
        gdata = GraphData(length(data.vertices), data.edges;
                          coords=data.vertices, weights=nothing, T=eltype(data.vertices[1]))
        return _graded_complex_from_graph(gdata, gspec; return_simplex_tree=true)
    end

    G, axes, orientation = _graded_complex_from_data(data, spec; cache=cache)
    ST = try
        _simplex_tree_multi_from_complex(G)
    catch err
        throw(ArgumentError("SimplexTreeMulti output_stage is unavailable for this ingestion (non-simplicial complex): $(sprint(showerror, err))"))
    end
    return ST, axes, orientation
end

function _simplex_tree_from_data(data, filtration::AbstractFiltration;
                                 cache::Union{Nothing,EncodingCache}=nothing)
    spec = _filtration_spec(filtration)
    return _simplex_tree_from_data(data, spec; cache=cache)
end

@inline function _resolve_ingestion_stage(stage::Symbol, construction::ConstructionOptions)::Symbol
    s = stage === :auto ? construction.output_stage : stage
    if s == :simplex_tree || s == :graded_complex || s == :cochain ||
       s == :encoded_complex ||
       s == :module || s == :fringe || s == :flange || s == :encoding_result ||
       s == :cohomology_dims
        return s
    end
    error("encode(data, filtration): stage must be :auto, :simplex_tree, :graded_complex, :cochain, :encoded_complex, :module, :fringe, :flange, :cohomology_dims, or :encoding_result.")
end

@inline function _pipeline_options_from_spec(spec::FiltrationSpec)::PipelineOptions
    p = spec.params
    return PipelineOptions(;
        orientation = get(p, :orientation, nothing),
        axes_policy = Symbol(get(p, :axes_policy, :encoding)),
        axis_kind = get(p, :axis_kind, nothing),
        eps = get(p, :eps, nothing),
        poset_kind = Symbol(get(p, :poset_kind, :signature)),
        field = get(p, :field, nothing),
        max_axis_len = get(p, :max_axis_len, nothing),
    )
end

@inline function _pipeline_options_from_filtration(filtration::AbstractFiltration)::PipelineOptions
    try
        return _pipeline_options_from_spec(_filtration_spec(filtration))
    catch err
        msg = sprint(showerror, err)
        if err isa ArgumentError && occursin("No FiltrationSpec conversion for", msg)
            return PipelineOptions()
        end
        rethrow()
    end
end

@inline function _spec_with_plan_options(spec::FiltrationSpec,
                                         construction::ConstructionOptions,
                                         pipeline::PipelineOptions)::FiltrationSpec
    p = spec.params
    p2 = (; p...,
          construction = construction,
          orientation = pipeline.orientation,
          axes_policy = pipeline.axes_policy,
          axis_kind = pipeline.axis_kind,
          eps = pipeline.eps,
          poset_kind = pipeline.poset_kind,
          field = pipeline.field,
          max_axis_len = pipeline.max_axis_len)
    return FiltrationSpec(spec.kind, p2)
end

@inline function _ingestion_route_hint(data)::Symbol
    if data isa SimplexTreeMulti
        return :simplex_tree_input
    elseif data isa PointCloud || data isa GraphData || data isa EmbeddedPlanarGraph2D
        return :simplex_tree_first
    end
    return :graded_complex_only
end

"""
    IngestionPlan

Normalized advanced ingestion plan used by `run_ingestion(plan; ...)` and
`encode(plan; degree=...)`.
"""
struct IngestionPlan{D,F,S}
    data::D
    filtration::F
    spec::S
    construction::ConstructionOptions
    pipeline::PipelineOptions
    stage::Symbol
    field::AbstractCoeffField
    cache::Union{Nothing,SessionCache}
    preflight::Union{Nothing,IngestionEstimate}
    route_hint::Symbol
    multicritical::Symbol
    onecritical_selector::Symbol
    onecritical_enforce_boundary::Bool
end

@inline source_data(plan::IngestionPlan) = getfield(plan, :data)
@inline source_data(res::PointCodensityResult) = getfield(res, :data)
@inline plan_filtration(plan::IngestionPlan) = getfield(plan, :filtration)
@inline plan_spec(plan::IngestionPlan) = getfield(plan, :spec)
@inline plan_construction(plan::IngestionPlan) = getfield(plan, :construction)
@inline plan_pipeline(plan::IngestionPlan) = getfield(plan, :pipeline)
@inline planned_stage(plan::IngestionPlan) = getfield(plan, :stage)
@inline plan_field(plan::IngestionPlan) = getfield(plan, :field)
@inline has_preflight(plan::IngestionPlan) = !isnothing(getfield(plan, :preflight))
@inline preflight_estimate(plan::IngestionPlan) = getfield(plan, :preflight)
@inline route_hint(plan::IngestionPlan) = getfield(plan, :route_hint)
@inline multicritical_mode(plan::IngestionPlan) = getfield(plan, :multicritical)
@inline onecritical_selector(plan::IngestionPlan) = getfield(plan, :onecritical_selector)
@inline enforce_boundary(plan::IngestionPlan) = getfield(plan, :onecritical_enforce_boundary)

@inline function _ingestion_plan_describe(plan::IngestionPlan)
    return (
        kind = :ingestion_plan,
        data_kind = _ingestion_data_kind(source_data(plan)),
        filtration_kind = filtration_kind(typeof(plan_filtration(plan))),
        planned_stage = planned_stage(plan),
        field = Symbol(nameof(typeof(plan_field(plan)))),
        route_hint = route_hint(plan),
        has_preflight = has_preflight(plan),
        construction_mode = _construction_mode_summary(plan_construction(plan)),
        multicritical_mode = multicritical_mode(plan),
    )
end

"""
    describe(plan::IngestionPlan) -> NamedTuple

Return a compact semantic summary of a prepared ingestion plan.

This is the shared inspection surface for advanced ingestion workflows. It
exposes the data kind, filtration family, resolved stage, field, route hint,
and whether a preflight estimate is already attached, without forcing direct
field inspection.
"""
describe(plan::IngestionPlan) = _ingestion_plan_describe(plan)

"""
    ingestion_plan_summary(plan::IngestionPlan) -> NamedTuple

Owner-local summary helper for advanced ingestion plans.

This is the preferred cheap-first inspection entrypoint after
[`plan_ingestion`](@ref) and before [`run_ingestion`](@ref). Use the semantic
accessors in this module when you want the resolved filtration, field, stage,
or attached preflight estimate separately.
"""
ingestion_plan_summary(plan::IngestionPlan) = _ingestion_plan_describe(plan)

function Base.show(io::IO, plan::IngestionPlan)
    d = _ingestion_plan_describe(plan)
    print(io, "IngestionPlan(data=", d.data_kind,
          ", filtration=", d.filtration_kind,
          ", stage=", d.planned_stage,
          ", preflight=", d.has_preflight, ")")
end

function Base.show(io::IO, ::MIME"text/plain", plan::IngestionPlan)
    d = _ingestion_plan_describe(plan)
    print(io, "IngestionPlan",
          "\n  data_kind: ", d.data_kind,
          "\n  filtration_kind: ", d.filtration_kind,
          "\n  planned_stage: ", d.planned_stage,
          "\n  field: ", d.field,
          "\n  route_hint: ", d.route_hint,
          "\n  has_preflight: ", d.has_preflight,
          "\n  construction_mode: ", repr(d.construction_mode),
          "\n  multicritical_mode: ", d.multicritical_mode)
end

"""
    plan_ingestion(data, filtration; stage=:auto, field=QQField(), cache=:auto,
                   construction=nothing, pipeline=nothing,
                   preflight=false, strict_preflight=false)
        -> IngestionPlan

Build a normalized advanced ingestion plan without yet running the heavy
encoding stages.

Cheap-first workflow
- start with `estimate_ingestion(...)` when you only need a risk/budget check;
- call `plan_ingestion(...)` when you want to inspect the resolved stage,
  filtration normalization, route hint, and cache contract before execution;
- inspect the returned [`IngestionPlan`](@ref) with `describe(...)` or
  [`ingestion_plan_summary`](@ref) before calling [`run_ingestion`](@ref).

`stage`
- `stage=:auto` resolves to `construction.output_stage`;
- otherwise `stage` explicitly selects where the ingestion pipeline stops;
- use `:graded_complex` or `:simplex_tree` when you want to stay at a cheaper
  intermediate mathematical object instead of building the final encoding.

Preflight
- `preflight=true` stores an [`IngestionEstimate`](@ref) on `plan.preflight`;
- `strict_preflight=true` makes preflight warnings throw immediately.
"""
function plan_ingestion(data, filtration::AbstractFiltration;
                        stage::Symbol=:auto,
                        field::AbstractCoeffField=QQField(),
                        cache=:auto,
                        construction::Union{Nothing,ConstructionOptions}=nothing,
                        pipeline::Union{Nothing,PipelineOptions}=nothing,
                        preflight::Bool=false,
                        strict_preflight::Bool=false)
    base_construction = _construction_from_filtration(filtration)
    construction_final = construction === nothing ? base_construction : construction
    pipeline_final = pipeline === nothing ? _pipeline_options_from_filtration(filtration) : pipeline
    session_cache = _resolve_workflow_session_cache(cache)
    stage_final = _resolve_ingestion_stage(stage, construction_final)
    enc_cache = _workflow_encoding_cache(session_cache)

    spec = nothing
    filtration_final = filtration
    preflight_report = nothing
    multicritical = :union
    onecritical_selector = :lexmin
    onecritical_enforce_boundary = true

    try
        spec0 = _filtration_spec(filtration)
        spec = _spec_with_plan_options(spec0, construction_final, pipeline_final)
        if _INGESTION_PLAN_NORM_CACHE[] && session_cache !== nothing
            norm_key = _ingestion_plan_norm_key(spec, stage_final, field)
            cached = _get_geometry_cached(enc_cache, norm_key)
            if cached isa NamedTuple &&
               hasproperty(cached, :filtration) &&
               hasproperty(cached, :multicritical) &&
               hasproperty(cached, :onecritical_selector) &&
               hasproperty(cached, :onecritical_enforce_boundary)
                filtration_final = getproperty(cached, :filtration)
                multicritical = getproperty(cached, :multicritical)
                onecritical_selector = getproperty(cached, :onecritical_selector)
                onecritical_enforce_boundary = getproperty(cached, :onecritical_enforce_boundary)
            else
                filtration_final = to_filtration(spec)
                multicritical = Symbol(get(spec.params, :multicritical, :union))
                onecritical_selector = Symbol(get(spec.params, :onecritical_selector, :lexmin))
                onecritical_enforce_boundary = Bool(get(spec.params, :onecritical_enforce_boundary, true))
                _set_geometry_cached!(enc_cache, norm_key, (
                    filtration=filtration_final,
                    multicritical=multicritical,
                    onecritical_selector=onecritical_selector,
                    onecritical_enforce_boundary=onecritical_enforce_boundary,
                ))
            end
        else
            filtration_final = to_filtration(spec)
            multicritical = Symbol(get(spec.params, :multicritical, :union))
            onecritical_selector = Symbol(get(spec.params, :onecritical_selector, :lexmin))
            onecritical_enforce_boundary = Bool(get(spec.params, :onecritical_enforce_boundary, true))
        end
        if preflight || strict_preflight
            preflight_report = estimate_ingestion(data, spec; strict=strict_preflight)
        end
    catch err
        msg = sprint(showerror, err)
        (err isa ArgumentError && occursin("No FiltrationSpec conversion for", msg)) || rethrow()
        if construction !== nothing || pipeline !== nothing
            throw(ArgumentError("plan_ingestion: construction/pipeline overrides require a filtration that round-trips through FiltrationSpec."))
        end
    end

    return IngestionPlan(
        data,
        filtration_final,
        spec,
        construction_final,
        pipeline_final,
        stage_final,
        field,
        session_cache,
        preflight_report,
        _ingestion_route_hint(data),
        multicritical,
        onecritical_selector,
        onecritical_enforce_boundary,
    )
end

function plan_ingestion(data, spec::FiltrationSpec;
                        stage::Symbol=:auto,
                        field::AbstractCoeffField=QQField(),
                        cache=:auto,
                        construction::Union{Nothing,ConstructionOptions}=nothing,
                        pipeline::Union{Nothing,PipelineOptions}=nothing,
                        preflight::Bool=false,
                        strict_preflight::Bool=false)
    base_construction = _construction_from_params(spec.params)
    construction_final = construction === nothing ? base_construction : construction
    base_pipeline = _pipeline_options_from_spec(spec)
    pipeline_final = pipeline === nothing ? base_pipeline : pipeline
    spec_final = _spec_with_plan_options(spec, construction_final, pipeline_final)
    session_cache = _resolve_workflow_session_cache(cache)
    stage_final = _resolve_ingestion_stage(stage, construction_final)
    enc_cache = _workflow_encoding_cache(session_cache)

    filtration = nothing
    multicritical = :union
    onecritical_selector = :lexmin
    onecritical_enforce_boundary = true
    if _INGESTION_PLAN_NORM_CACHE[] && session_cache !== nothing
        norm_key = _ingestion_plan_norm_key(spec_final, stage_final, field)
        cached = _get_geometry_cached(enc_cache, norm_key)
        if cached isa NamedTuple &&
           hasproperty(cached, :filtration) &&
           hasproperty(cached, :multicritical) &&
           hasproperty(cached, :onecritical_selector) &&
           hasproperty(cached, :onecritical_enforce_boundary)
            filtration = getproperty(cached, :filtration)
            multicritical = getproperty(cached, :multicritical)
            onecritical_selector = getproperty(cached, :onecritical_selector)
            onecritical_enforce_boundary = getproperty(cached, :onecritical_enforce_boundary)
        end
        if filtration === nothing
            filtration = to_filtration(spec_final)
            multicritical = Symbol(get(spec_final.params, :multicritical, :union))
            onecritical_selector = Symbol(get(spec_final.params, :onecritical_selector, :lexmin))
            onecritical_enforce_boundary = Bool(get(spec_final.params, :onecritical_enforce_boundary, true))
            _set_geometry_cached!(enc_cache, norm_key, (
                filtration=filtration,
                multicritical=multicritical,
                onecritical_selector=onecritical_selector,
                onecritical_enforce_boundary=onecritical_enforce_boundary,
            ))
        end
    else
        filtration = to_filtration(spec_final)
        multicritical = Symbol(get(spec_final.params, :multicritical, :union))
        onecritical_selector = Symbol(get(spec_final.params, :onecritical_selector, :lexmin))
        onecritical_enforce_boundary = Bool(get(spec_final.params, :onecritical_enforce_boundary, true))
    end

    preflight_report = if preflight || strict_preflight
        estimate_ingestion(data, spec_final; strict=strict_preflight)
    else
        nothing
    end

    return IngestionPlan(
        data,
        filtration,
        spec_final,
        construction_final,
        pipeline_final,
        stage_final,
        field,
        session_cache,
        preflight_report,
        _ingestion_route_hint(data),
        multicritical,
        onecritical_selector,
        onecritical_enforce_boundary,
    )
end

@inline function _ingestion_datafile_cache_key(path::AbstractString,
                                               kind::Symbol,
                                               format::Symbol,
                                               opts::DataFileOptions,
                                               load_kwargs::NamedTuple)
    st = stat(path)
    return (
        :ingestion_datafile,
        abspath(path),
        Int(st.size),
        Float64(st.mtime),
        kind,
        format,
        UInt(hash(opts)),
        UInt(hash(load_kwargs)),
    )
end

@inline function _load_ingestion_datafile(path::AbstractString,
                                          kind::Symbol,
                                          format::Symbol,
                                          opts::DataFileOptions,
                                          load_kwargs::NamedTuple,
                                          cache)
    session_cache = _resolve_workflow_session_cache(cache)
    if session_cache === nothing
        return DataFileIO.load_data(path; kind=kind, format=format, opts=opts, load_kwargs...)
    end
    enc_cache = _workflow_encoding_cache(session_cache)
    key = _ingestion_datafile_cache_key(path, kind, format, opts, load_kwargs)
    cached = _get_geometry_cached(enc_cache, key)
    if cached !== nothing
        return cached
    end
    data = DataFileIO.load_data(path; kind=kind, format=format, opts=opts, load_kwargs...)
    return _set_geometry_cached!(enc_cache, key, data)
end

"""
    plan_ingestion(path::AbstractString, filtration_or_spec; kind=:auto, format=:auto,
                   file_opts=DataFileOptions(), load_kwargs=NamedTuple(), ...)

File-based planner overload. Loads typed data via `DataFileIO.load_data` and
delegates to the canonical typed `plan_ingestion(data, ...)` path.

Use this when the data still lives on disk and you want to inspect the resolved
ingestion plan before committing to a full `encode(path, ...)` run.
"""
function plan_ingestion(path::AbstractString, filtration::AbstractFiltration;
                        kind::Symbol=:auto,
                        format::Symbol=:auto,
                        file_opts::DataFileOptions=DataFileOptions(),
                        load_kwargs::NamedTuple=NamedTuple(),
                        stage::Symbol=:auto,
                        field::AbstractCoeffField=QQField(),
                        cache=:auto,
                        construction::Union{Nothing,ConstructionOptions}=nothing,
                        pipeline::Union{Nothing,PipelineOptions}=nothing,
                        preflight::Bool=false,
                        strict_preflight::Bool=false)
    data = _load_ingestion_datafile(path, kind, format, file_opts, load_kwargs, cache)
    session_cache = _resolve_workflow_session_cache(cache)
    return plan_ingestion(data, filtration;
                          stage=stage,
                          field=field,
                          cache=session_cache,
                          construction=construction,
                          pipeline=pipeline,
                          preflight=preflight,
                          strict_preflight=strict_preflight)
end

function plan_ingestion(path::AbstractString, spec::FiltrationSpec;
                        kind::Symbol=:auto,
                        format::Symbol=:auto,
                        file_opts::DataFileOptions=DataFileOptions(),
                        load_kwargs::NamedTuple=NamedTuple(),
                        stage::Symbol=:auto,
                        field::AbstractCoeffField=QQField(),
                        cache=:auto,
                        construction::Union{Nothing,ConstructionOptions}=nothing,
                        pipeline::Union{Nothing,PipelineOptions}=nothing,
                        preflight::Bool=false,
                        strict_preflight::Bool=false)
    data = _load_ingestion_datafile(path, kind, format, file_opts, load_kwargs, cache)
    session_cache = _resolve_workflow_session_cache(cache)
    return plan_ingestion(data, spec;
                          stage=stage,
                          field=field,
                          cache=session_cache,
                          construction=construction,
                          pipeline=pipeline,
                          preflight=preflight,
                          strict_preflight=strict_preflight)
end

@inline function _throw_invalid_ingestion(name::AbstractString, issues::Vector{String})
    throw(ArgumentError(string(name, " is invalid:\n  - ", join(issues, "\n  - "))))
end

@inline _ingestion_validation_report(kind::Symbol, valid::Bool; kwargs...) = (; kind, valid, kwargs...)

@inline function _ingestion_validation_meta(report)
    return (
        kind = get(report, :kind, :ingestion_validation),
        valid = get(report, :valid, false),
        issues = get(report, :issues, String[]),
    )
end

function Base.show(io::IO, summary::IngestionValidationSummary)
    meta = _ingestion_validation_meta(getfield(summary, :report))
    print(io, "IngestionValidationSummary(kind=", meta.kind,
          ", valid=", meta.valid,
          ", issues=", length(meta.issues), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::IngestionValidationSummary)
    report = getfield(summary, :report)
    meta = _ingestion_validation_meta(report)
    println(io, "IngestionValidationSummary")
    println(io, "  kind = ", meta.kind)
    println(io, "  valid = ", meta.valid)
    for key in (:data_kind, :filtration_kind, :registered, :planned_stage, :requested_stage,
                :resolved_stage, :field, :route_hint, :arity, :nparameters,
                :spec_convertible, :estimated_cells, :estimated_poset_size, :nwarnings,
                :parameter_dim, :naxes, :orientation_length, :sparsify, :collapse,
                :output_stage)
        haskey(report, key) || continue
        println(io, "  ", key, " = ", getfield(report, key))
    end
    if isempty(meta.issues)
        println(io, "  issues = none")
    else
        println(io, "  issues:")
        for msg in meta.issues
            println(io, "    - ", msg)
        end
    end
    warnings = get(report, :warnings, String[])
    if !isempty(warnings)
        println(io, "  warnings:")
        for msg in warnings
            println(io, "    - ", msg)
        end
    end
end

"""
    check_ingestion_stage(stage; throw=false) -> NamedTuple

Validate an ingestion pipeline stage selector.

Use this before threading stage values into [`plan_ingestion`](@ref),
[`run_ingestion`](@ref), or `encode(data, filtration; stage=...)` when stage
values are coming from a notebook/UI/config surface rather than hard-coded
symbols.
"""
function check_ingestion_stage(stage; throw::Bool=false)
    issues = String[]
    resolved_stage = nothing
    if !(stage isa Symbol)
        push!(issues, "stage must be a Symbol.")
    else
        try
            resolved_stage = _resolve_ingestion_stage(stage, ConstructionOptions())
        catch err
            push!(issues, sprint(showerror, err))
        end
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_ingestion("ingestion stage", issues)
    return _ingestion_validation_report(:ingestion_stage, valid;
                                        requested_stage=stage,
                                        resolved_stage=resolved_stage,
                                        issues=issues)
end

"""
    check_preflight_mode(x; throw=false) -> NamedTuple

Validate a `preflight` / `strict_preflight` boolean control.

This is a lightweight contract helper for notebook or config-driven ingestion
workflows where preflight flags are not yet statically typed.
"""
function check_preflight_mode(x; throw::Bool=false)
    issues = String[]
    x isa Bool || push!(issues, "preflight mode must be Bool.")
    valid = isempty(issues)
    throw && !valid && _throw_invalid_ingestion("preflight mode", issues)
    return _ingestion_validation_report(:preflight_mode, valid;
                                        value=x,
                                        issues=issues)
end

"""
    check_construction_options(data, construction; throw=false) -> NamedTuple

Validate a [`ConstructionOptions`](@ref) object against a dataset kind.

This is a cheap contract check for stage and obvious data/option compatibility.
More filtration-specific route checks still belong to
[`check_data_filtration`](@ref).
"""
function check_construction_options(data, construction::ConstructionOptions; throw::Bool=false)
    issues = String[]
    warnings = String[]
    stage_report = check_ingestion_stage(construction.output_stage; throw=false)
    get(stage_report, :valid, false) || append!(issues, get(stage_report, :issues, String[]))
    data_kind = _ingestion_data_kind(data)
    if construction.sparsify != :none && !(data isa PointCloud)
        push!(issues, "construction.sparsify=$(construction.sparsify) is currently only supported for point-cloud ingestion.")
    elseif data isa PointCloud && construction.sparsify != :none
        push!(warnings, "sparse point-cloud construction also depends on filtration-specific radius/knn contracts; use check_data_filtration(...) for the full route check.")
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_ingestion("construction options", issues)
    return _ingestion_validation_report(:construction_options, valid;
                                        data_kind=data_kind,
                                        sparsify=construction.sparsify,
                                        collapse=construction.collapse,
                                        output_stage=construction.output_stage,
                                        warnings=warnings,
                                        issues=issues)
end

"""
    check_filtration_spec(spec; throw=false) -> NamedTuple

Validate a serialized [`FiltrationSpec`](@ref).

This checks that the family is known, the construction payload is canonical,
and the spec converts cleanly through [`to_filtration`](@ref).
"""
function check_filtration_spec(spec::FiltrationSpec; throw::Bool=false)
    issues = String[]
    registered = false
    arity = :unknown
    try
        sig = filtration_signature(filtration_kind(spec))
        registered = sig.registered
        arity = sig.arity
    catch err
        push!(issues, sprint(showerror, err))
    end
    try
        filtration_parameters(spec) isa NamedTuple ||
            push!(issues, "filtration spec parameters must be a NamedTuple.")
    catch err
        push!(issues, sprint(showerror, err))
    end
    try
        construction_mode(spec)
    catch err
        push!(issues, sprint(showerror, err))
    end
    try
        to_filtration(spec)
    catch err
        push!(issues, sprint(showerror, err))
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_ingestion("filtration spec", issues)
    return _ingestion_validation_report(:filtration_spec, valid;
                                        filtration_kind=filtration_kind(spec),
                                        registered=registered,
                                        arity=arity,
                                        nparameters=length(filtration_parameters(spec)),
                                        issues=issues)
end

"""
    check_graded_complex_build_result(res; throw=false) -> NamedTuple

Validate a [`GradedComplexBuildResult`](@ref).

This checks the wrapped graded-complex object and the consistency of the
returned axes/orientation payload without rebuilding the complex.
"""
function check_graded_complex_build_result(res::GradedComplexBuildResult; throw::Bool=false)
    issues = String[]
    G = graded_complex(res)
    if !(G isa GradedComplex || G isa MultiCriticalGradedComplex)
        push!(issues, "graded_complex must be GradedComplex or MultiCriticalGradedComplex.")
    else
        report = if G isa GradedComplex
            DataTypes.check_graded_complex(G; throw=false)
        else
            DataTypes.check_multicritical_complex(G; throw=false)
        end
        get(report, :valid, false) || append!(issues, get(report, :issues, String[]))
    end
    axes = grade_axes(res)
    orientation = grade_orientation(res)
    naxes = try
        length(axes)
    catch
        push!(issues, "grade_axes must support length.")
        0
    end
    orientation_length = try
        length(orientation)
    catch
        push!(issues, "grade_orientation must support length.")
        0
    end
    parameter_dim = if G isa GradedComplex || G isa MultiCriticalGradedComplex
        DataTypes.parameter_dim(G)
    else
        nothing
    end
    parameter_dim === nothing || naxes == parameter_dim ||
        push!(issues, "grade_axes length must match the complex parameter dimension.")
    naxes == orientation_length ||
        push!(issues, "grade_orientation length must match the number of grade axes.")
    if axes isa Tuple || axes isa AbstractVector
        for (i, axis) in enumerate(axes)
            axis isa AbstractVector || push!(issues, "grade_axes[$i] must be an AbstractVector.")
            axis isa AbstractVector && isempty(axis) && push!(issues, "grade_axes[$i] must be nonempty.")
        end
    else
        push!(issues, "grade_axes must be a tuple or vector of axes.")
    end
    if orientation isa Tuple || orientation isa AbstractVector
        all(o -> o == 1 || o == -1, orientation) ||
            push!(issues, "grade_orientation entries must be +1 or -1.")
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_ingestion("graded-complex build result", issues)
    return _ingestion_validation_report(:graded_complex_build_result, valid;
                                        complex_kind=Symbol(nameof(typeof(G))),
                                        parameter_dim=parameter_dim,
                                        naxes=naxes,
                                        orientation_length=orientation_length,
                                        issues=issues)
end

"""
    check_filtration(filtration; throw=false) -> NamedTuple

Validate a typed ingestion filtration contract.

This validator checks the public `DataIngestion` filtration surface: family
kind, parameter arity, parameter container shape, resolved construction mode,
and whether the filtration round-trips through `FiltrationSpec`.

Cheap-first workflow
- use [`filtration_summary`](@ref) first when you only want inspection;
- use `check_filtration(...)` before wiring hand-built typed filtrations into
  `estimate_ingestion(...)`, `plan_ingestion(...)`, or `encode(...)`;
- wrap the returned report with [`ingestion_validation_summary`](@ref) for a
  readable REPL summary.
"""
function check_filtration(f::AbstractFiltration; throw::Bool=false)
    issues = String[]
    kind = :unknown
    arity = :unknown
    spec_convertible = false
    try
        kind = filtration_kind(typeof(f))
    catch err
        push!(issues, sprint(showerror, err))
    end
    if isempty(issues)
        try
            arity = filtration_arity(f)
            if !(arity === :variable || (arity isa Int && arity > 0))
                push!(issues, "filtration_arity must be a positive Int or :variable.")
            end
        catch err
            push!(issues, sprint(showerror, err))
        end
    end
    try
        params = filtration_parameters(f)
        params isa NamedTuple || push!(issues, "filtration_parameters must return a NamedTuple.")
    catch err
        push!(issues, sprint(showerror, err))
    end
    try
        _filtration_spec(f)
        spec_convertible = true
    catch err
        if !(err isa ArgumentError && occursin("No FiltrationSpec conversion for", sprint(showerror, err)))
            push!(issues, sprint(showerror, err))
        end
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_ingestion("filtration", issues)
    return _ingestion_validation_report(:filtration, valid;
                                        filtration_kind=kind,
                                        arity=arity,
                                        construction_mode=_construction_mode_summary(construction_mode(f)),
                                        spec_convertible=spec_convertible,
                                        issues=issues)
end

"""
    check_ingestion_estimate(est; throw=false) -> NamedTuple

Validate a typed preflight estimate returned by [`estimate_ingestion`](@ref).

Use this when an estimate came from stored state, was hand-edited in a notebook,
or needs an explicit sanity check before you trust its scalar summary.
"""
function check_ingestion_estimate(est::IngestionEstimate; throw::Bool=false)
    issues = String[]
    cells = estimated_cells(est)
    cells >= 0 || push!(issues, "estimated_cells must be nonnegative.")
    counts = cell_counts_by_dim(est)
    all(x -> x >= 0, counts) || push!(issues, "cell_counts_by_dim entries must be nonnegative.")
    axis_sizes = estimated_axis_sizes(est)
    if axis_sizes !== nothing
        all(x -> x >= 1, axis_sizes) || push!(issues, "estimated_axis_sizes entries must be positive.")
    end
    poset = estimated_poset_size(est)
    poset === nothing || poset >= 0 || push!(issues, "estimated_poset_size must be nonnegative.")
    estimated_nnz(est) >= 0 || push!(issues, "estimated_nnz must be nonnegative.")
    estimated_dense_bytes(est) >= 0 || push!(issues, "estimated_dense_bytes must be nonnegative.")
    estimate_warnings(est) isa Vector{String} || push!(issues, "estimate_warnings must be a Vector{String}.")
    valid = isempty(issues)
    throw && !valid && _throw_invalid_ingestion("ingestion estimate", issues)
    return _ingestion_validation_report(:ingestion_estimate, valid;
                                        estimated_cells=cells,
                                        estimated_poset_size=poset,
                                        nwarnings=length(estimate_warnings(est)),
                                        issues=issues)
end

@inline function _ingestion_warning_is_error(msg::AbstractString)
    return occursin("will fail at ingestion", msg)
end

"""
    check_data_filtration(data, filtration; throw=false) -> NamedTuple

Validate a dataset/filtration pairing before planning or execution.

This is the highest-value ingestion validator. It checks the typed filtration
surface first, then runs the cheapest meaningful compatibility probe:
- a preflight estimate when the filtration is `FiltrationSpec`-convertible,
- otherwise the direct graded-complex builder contract.

Use this before `plan_ingestion(...)` when you need a notebook-friendly answer
to "will this data/filtration pair actually work?" without stepping through the
full pipeline.
"""
function check_data_filtration(data, filtration::AbstractFiltration; throw::Bool=false)
    issues = String[]
    warnings = String[]
    filt_report = check_filtration(filtration; throw=false)
    kind = get(filt_report, :filtration_kind, :unknown)
    get(filt_report, :valid, false) || append!(issues, get(filt_report, :issues, String[]))
    if isempty(issues)
        try
            spec = _filtration_spec(filtration)
            est = IngestionEstimate(_estimate_ingestion_report(data, spec; strict=false))
            warnings = copy(estimate_warnings(est))
            for msg in warnings
                _ingestion_warning_is_error(msg) && push!(issues, msg)
            end
        catch err
            try
                _build_graded_complex_tuple(data, filtration; cache=nothing)
            catch inner
                push!(issues, sprint(showerror, inner))
            end
            if !(err isa ArgumentError && occursin("No FiltrationSpec conversion for", sprint(showerror, err)))
                push!(issues, sprint(showerror, err))
            end
        end
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_ingestion("data/filtration pair", issues)
    return _ingestion_validation_report(:data_filtration, valid;
                                        data_kind=_ingestion_data_kind(data),
                                        filtration_kind=kind,
                                        warnings=warnings,
                                        issues=issues)
end

"""
    check_ingestion_plan(plan; throw=false) -> NamedTuple

Validate a prepared [`IngestionPlan`](@ref).

This validator checks the resolved stage contract, the attached preflight
estimate when present, and the underlying dataset/filtration pairing.
"""
function check_ingestion_plan(plan::IngestionPlan; throw::Bool=false)
    issues = String[]
    filtration_kind_sym = try
        filtration_kind(typeof(plan_filtration(plan)))
    catch
        :unknown
    end
    if !(planned_stage(plan) in (:simplex_tree, :graded_complex, :cochain, :encoded_complex, :module,
                                 :fringe, :flange, :cohomology_dims, :encoding_result))
        push!(issues, "planned_stage must be one of the canonical ingestion stages.")
    end
    pair_report = check_data_filtration(source_data(plan), plan_filtration(plan); throw=false)
    get(pair_report, :valid, false) || append!(issues, get(pair_report, :issues, String[]))
    has_preflight(plan) && begin
        pre_report = check_ingestion_estimate(preflight_estimate(plan); throw=false)
        get(pre_report, :valid, false) || append!(issues, get(pre_report, :issues, String[]))
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_ingestion("ingestion plan", issues)
    return _ingestion_validation_report(:ingestion_plan, valid;
                                        data_kind=_ingestion_data_kind(source_data(plan)),
                                        filtration_kind=filtration_kind_sym,
                                        planned_stage=planned_stage(plan),
                                        field=Symbol(nameof(typeof(plan_field(plan)))),
                                        route_hint=route_hint(plan),
                                        has_preflight=has_preflight(plan),
                                        issues=issues)
end

@inline function _build_ingestion_lazy_cochain(
    ST,
    G,
    P,
    axes_final,
    orientation_final,
    field,
    multicritical::Symbol,
    onecritical_selector::Symbol,
    onecritical_enforce_boundary::Bool,
)
    if ST !== nothing
        return _lazy_cochain_complex_from_simplex_tree(
            ST, P, axes_final;
            orientation=orientation_final,
            field=field,
            multicritical=multicritical,
            onecritical_selector=onecritical_selector,
            onecritical_enforce_boundary=onecritical_enforce_boundary,
        )
    end
    return _lazy_cochain_complex_from_graded_complex(
        G, P, axes_final;
        orientation=orientation_final,
        field=field,
        multicritical=multicritical,
        onecritical_selector=onecritical_selector,
        onecritical_enforce_boundary=onecritical_enforce_boundary,
    )
end

@inline function _build_ingestion_lazy_cochain(
    payload::_PointCloudLowdimLazyPayload,
    P,
    axes_final,
    orientation_final,
    field,
    multicritical::Symbol,
    onecritical_selector::Symbol,
    onecritical_enforce_boundary::Bool,
)
    _ = onecritical_selector
    _ = onecritical_enforce_boundary
    return _lazy_cochain_complex_from_grades_and_boundaries(
        payload.grades_by_dim,
        payload.boundaries,
        P,
        axes_final;
        orientation=orientation_final,
        field=field,
        multicritical=multicritical,
    )
end

function _run_ingestion_plan(plan::IngestionPlan;
                             degree::Int=0,
                             stage::Symbol=:auto)
    data = plan.data
    filtration = plan.filtration
    construction = plan.construction
    pipeline = plan.pipeline
    field = plan.field
    session_cache = plan.cache
    multicritical = plan.multicritical
    onecritical_selector = plan.onecritical_selector
    onecritical_enforce_boundary = plan.onecritical_enforce_boundary
    stage = _resolve_ingestion_stage(stage, construction)

    axes_override = if plan.spec isa FiltrationSpec
        get(plan.spec.params, :axes, nothing)
    else
        nothing
    end
    orientation = pipeline.orientation
    eps = pipeline.eps
    axes_policy = pipeline.axes_policy
    max_axis_len = pipeline.max_axis_len
    axis_kind = pipeline.axis_kind

    enc_cache = _workflow_encoding_cache(session_cache)

    if stage == :simplex_tree
        ST, _, _ = _simplex_tree_from_data(data, filtration; cache=enc_cache)
        return ST
    end

    ST = nothing
    G = nothing
    direct_payload = nothing
    axes0 = nothing
    orientation0 = nothing
    tree_route_ok = true
    spec_try = plan.spec isa FiltrationSpec ? plan.spec : nothing
    if filtration isa AbstractFiltration
        spec_try = try
            _filtration_spec(filtration)
        catch
            nothing
        end
        if spec_try isa FiltrationSpec
            tree_route_ok = spec_try.kind in _BUILTIN_FILTRATION_KINDS
        else
            tree_route_ok = false
        end
    end
    if stage == :encoded_complex &&
       eps === nothing &&
       data isa PointCloud &&
       spec_try isa FiltrationSpec &&
       Int(get(spec_try.params, :max_dim, 1)) <= 1 &&
       _is_point_cloud_lowdim_direct_lazy_kind(spec_try.kind)
        direct_payload = _point_cloud_lowdim_lazy_payload(data, spec_try)
        axes0 = direct_payload.axes
        orientation0 = direct_payload.orientation
    else
        can_try_tree_compute = stage != :graded_complex &&
                               tree_route_ok &&
                               (data isa SimplexTreeMulti ||
                                data isa PointCloud ||
                                data isa GraphData ||
                                data isa EmbeddedPlanarGraph2D)
        if can_try_tree_compute
            try
                ST, axes0, orientation0 = _simplex_tree_from_data(data, filtration; cache=enc_cache)
            catch err
                msg = sprint(showerror, err)
                if err isa ArgumentError && (
                    occursin("SimplexTreeMulti output_stage is unavailable", msg) ||
                    occursin("No FiltrationSpec conversion for", msg)
                )
                    ST = nothing
                else
                    rethrow()
                end
            end
        end

        if ST === nothing
            G, axes0, orientation0 = _graded_complex_from_data(data, filtration; cache=enc_cache)
        end

        if stage == :graded_complex
            if G === nothing
                G = _graded_complex_from_simplex_tree(ST)
            end
            return G
        end
    end

    if stage == :graded_complex && direct_payload !== nothing
        error("encode(data, filtration): direct lowdim lazy payload is only available for stage=:encoded_complex.")
    end
    orientation_final = orientation === nothing ? orientation0 : orientation
    axes_final = if axes_override === nothing
        if orientation === nothing
            axes0
        elseif ST !== nothing
            _axes_from_simplex_tree(ST; orientation=orientation_final)
        elseif direct_payload !== nothing
            flat_grades = length(direct_payload.grades_by_dim) == 1 ?
                direct_payload.grades_by_dim[1] :
                vcat(direct_payload.grades_by_dim[1], direct_payload.grades_by_dim[2])
            _axes_from_grades(
                flat_grades,
                length(orientation_final);
                orientation=orientation_final,
            )
        else
            _axes_from_complex_grades(G, orientation_final)
        end
    else
        axes_override
    end

    if eps !== nothing
        if ST !== nothing
            ST = _quantize_simplex_tree(ST, eps)
            axes_final = _axes_from_simplex_tree(ST; orientation=orientation_final)
        else
            G = _quantize_grades(G, eps)
            axes_final = _axes_from_complex_grades(G, orientation_final)
        end
    end

    if axes_policy == :as_given
        axes_override === nothing && error("encode(data, filtration): axes_policy=:as_given requires explicit axes.")
        axes_final = axes_override
    elseif axes_policy == :coarsen
        max_axis_len === nothing && error("encode(data, filtration): axes_policy=:coarsen requires max_axis_len.")
        axes_final = _coarsen_axes(axes_final, Int(max_axis_len))
    elseif axes_policy != :encoding
        error("encode(data, filtration): unknown axes_policy $(axes_policy).")
    end

    _validate_axes_sorted(axes_final)
    _validate_axes_kind(axes_final; axis_kind=axis_kind)
    needs_flange = (stage == :flange)
    if needs_flange && axis_kind != :zn
        error("encode(data, filtration; stage=:flange): axis_kind must be :zn (integer axes).")
    end

    poset_fast_key = _ingestion_poset_fast_key(
        data,
        filtration,
        orientation_final,
        axes_policy,
        max_axis_len,
        axis_kind,
        eps,
        multicritical,
        onecritical_selector,
        onecritical_enforce_boundary,
    )
    P = session_cache === nothing ? nothing : _get_geometry_cached(enc_cache, poset_fast_key)
    if !(P isa AbstractPoset)
        P = _poset_from_axes_cached(axes_final, orientation_final; cache=enc_cache)
        session_cache === nothing || _set_geometry_cached!(enc_cache, poset_fast_key, P)
    end
    module_fast_key = session_cache === nothing ? nothing :
        _ingestion_module_fast_key(poset_fast_key, P, degree, field)
    cached_module = module_fast_key === nothing ? nothing : _get_geometry_cached(enc_cache, module_fast_key)
    cached_module_hit = cached_module isa NamedTuple && hasproperty(cached_module, :M)

    pi = GridEncodingMap(P, axes_final; orientation=orientation_final)
    # Keep ingestion encode() results self-contained: per-result encoding cache.
    # Session-level cache reuse for ingestion happens at the poset/data level above.
    pi2 = stage == :encoded_complex ?
        _compile_encoding_without_reps(P, pi, (encoding_cache=EncodingCache(),)) :
        compile_encoding(P, pi; meta=(encoding_cache=EncodingCache(),))

    build_lazy = () -> direct_payload === nothing ?
        _build_ingestion_lazy_cochain(
            ST, G, P, axes_final, orientation_final, field,
            multicritical, onecritical_selector, onecritical_enforce_boundary,
        ) :
        _build_ingestion_lazy_cochain(
            direct_payload, P, axes_final, orientation_final, field,
            multicritical, onecritical_selector, onecritical_enforce_boundary,
        )

    lazyC = nothing
    if stage == :cochain || stage == :encoded_complex || stage == :cohomology_dims || !(_INGESTION_SKIP_LAZY_ON_MODULE_CACHE_HIT[] && cached_module_hit)
        lazyC = build_lazy()
    end
    if stage == :cochain
        lazyC === nothing && (lazyC = build_lazy())
        return _materialize_cochain(lazyC; check=true)
    end
    if stage == :encoded_complex
        lazyC === nothing && (lazyC = build_lazy())
        return EncodedComplexResult(
            P,
            lazyC,
            pi2;
            field=field,
            meta=(presentation=(data=data, filtration=filtration), backend=:data),
        )
    end
    if stage == :cohomology_dims
        lazyC === nothing && (lazyC = build_lazy())
        dims = _cohomology_dims_from_lazy(lazyC, degree)
        return CohomologyDimsResult(P, dims, pi2; degree=degree, field=field)
    end
    use_lazy_encoding_module = stage == :encoding_result &&
                               _ENCODING_RESULT_LAZY_MODULE[] &&
                               !cached_module_hit
    M = if use_lazy_encoding_module
        lazyC === nothing && (lazyC = build_lazy())
        _lazy_encoded_module_from_lazy(lazyC, degree)
    elseif cached_module_hit
        getproperty(cached_module, :M)
    else
        M_fast = nothing
        if _H0_CHAIN_SWEEP_FASTPATH[] &&
           degree == 0 &&
           ST isa SimplexTreeMulti{1} &&
           max_simplex_dim(ST) <= 1 &&
           multicritical == :union
            M_fast = _cohomology_module_h0_chain_sweep_from_simplex_tree(
                ST, P, axes_final;
                orientation=orientation_final,
                field=field,
            )
        end
        Mc = if M_fast === nothing
            lazyC === nothing && (lazyC = build_lazy())
            _cohomology_module_from_lazy(lazyC, degree)
        else
            M_fast
        end
        module_fast_key === nothing || _set_geometry_cached!(enc_cache, module_fast_key, (M=Mc, H=nothing))
        Mc
    end
    H_cached = if cached_module isa NamedTuple && hasproperty(cached_module, :H)
        getproperty(cached_module, :H)
    else
        nothing
    end
    if stage == :module
        return M
    end
    if stage == :fringe
        H = H_cached === nothing ? fringe_presentation(M) : H_cached
        if H_cached === nothing && module_fast_key !== nothing
            _set_geometry_cached!(enc_cache, module_fast_key, (M=M, H=H))
        end
        return H
    end
    if stage == :flange
        return flange_presentation(M, pi)
    end
    res = EncodingResult(P, M, pi2; H=H_cached, presentation=(data=data, filtration=filtration),
                         opts=EncodingOptions(; backend=:data, field=field), backend=:data, meta=(;))
    return res
end

"""
    run_ingestion(plan; stage=:auto, degree=0) -> Any

Run a prepared ingestion plan at a selected output stage.

This is the advanced execution surface for [`IngestionPlan`](@ref). Use it when
you intentionally want an intermediate ingestion object such as a
`SimplexTreeMulti`, `GradedComplex`, cochain complex, encoded complex result,
or fringe presentation.
For the canonical final workflow object, prefer `encode(plan)`.
"""
run_ingestion(plan::IngestionPlan; stage::Symbol=:auto, degree::Int=0) =
    _run_ingestion_plan(plan; stage=stage, degree=degree)

"""
    encode(plan::IngestionPlan; degree=0) -> EncodingResult

Canonical final-stage execution for a prepared ingestion plan.

Use this when you already called `plan_ingestion(...)` and now want the same
typed endpoint a simple `encode(data, filtration; stage=:encoding_result, ...)`
call would produce.

`output` vs `stage`
- prepared-plan execution does not use the presentation-workflow `output`
  keyword,
- instead, the plan already stores the ingestion pipeline contract and
  `encode(plan)` always means "run to the final `EncodingResult` stage",
- use `run_ingestion(plan; stage=...)` when you intentionally want an
  intermediate pipeline object.

Cheap-first workflow
- start with `encode(plan)` when you want the canonical final workflow object,
- use `describe(enc)` / `result_summary(enc)` before materializing heavier
  downstream workflows such as `resolve`, `invariant`, or `slice_barcodes`.
"""
encode(plan::IngestionPlan; degree::Int=0) =
    _run_ingestion_plan(plan; degree=degree, stage=:encoding_result)

"""
    encode(data, filtration; degree=0, stage=:auto, field=QQField(), cache=:auto,
           construction=nothing, pipeline=nothing,
           preflight=false, strict_preflight=false) -> Any

Canonical data-ingestion entrypoint.

`stage` controls the returned object:
- `:simplex_tree`, `:graded_complex`, `:cochain`, `:module`, `:fringe`,
  `:flange`, `:cohomology_dims`, `:encoded_complex`, or `:encoding_result`.
- `:auto` uses `construction.output_stage` from the typed filtration/spec.

`stage=:cohomology_dims` returns `CohomologyDimsResult(P, dims, pi; degree, field)`
and skips module-map materialization; this is intended for dim/rank-only
invariant workflows.

`stage=:encoded_complex` returns `EncodedComplexResult(P, C, pi; field)` where
`C` is an encoded cochain-complex payload, typically stored lazily until a
downstream workflow needs full term/differential materialization. This is the
canonical ingestion stage for exact whole-complex invariants such as Euler
signed measures.

`stage=:encoding_result` avoids eager fringe materialization on the hot path.
When `_ENCODING_RESULT_LAZY_MODULE[]` is enabled, `enc.M` is a lazy module
wrapper and the concrete `PModule` is materialized only when needed by
downstream workflows.

`output` vs `stage`
- data-ingestion `encode` uses `stage`, not `output`,
- `stage` chooses where the ingestion pipeline stops,
- the presentation-workflow `encode(x; output=:result/:raw)` family is the
  separate surface for already-built mathematical presentations.

Typed filtration vs `FiltrationSpec`
- use typed filtrations when you want constructor defaults, semantic parameter
  names, and inspection through `describe(...)` / `filtration_summary(...)`;
- use `FiltrationSpec` when the filtration must cross a serialized boundary or
  be stored in a plan/config artifact.

Preflight behavior:
- `preflight=false` (default) skips `estimate_ingestion(...)` on the hot path.
- `preflight=true` computes estimates and stores them in `IngestionPlan.preflight`.
- `strict_preflight=true` runs preflight in strict mode and throws on budget violations.

Best practices
- stay at `stage=:simplex_tree` or `stage=:graded_complex` when you are still
  debugging the filtration or checking combinatorial growth;
- use `stage=:encoded_complex` when the downstream task needs the full encoded
  cochain complex for exact whole-complex invariants;
- only run to `:encoding_result` when you actually need the final encoded
  module/poset object for downstream algebra or invariant workflows.
"""
function encode(data, filtration::AbstractFiltration;
                degree::Int=0,
                stage::Symbol=:auto,
                field::AbstractCoeffField=QQField(),
                cache=:auto,
                construction::Union{Nothing,ConstructionOptions}=nothing,
                pipeline::Union{Nothing,PipelineOptions}=nothing,
                preflight::Bool=false,
                strict_preflight::Bool=false)
    plan = plan_ingestion(data, filtration;
                          stage=stage,
                          field=field,
                          cache=cache,
                          construction=construction,
                          pipeline=pipeline,
                          preflight=preflight,
                          strict_preflight=strict_preflight)
    return run_ingestion(plan; stage=plan.stage, degree=degree)
end

function encode(data, spec::FiltrationSpec;
                degree::Int=0,
                stage::Symbol=:auto,
                field::AbstractCoeffField=QQField(),
                cache=:auto,
                construction::Union{Nothing,ConstructionOptions}=nothing,
                pipeline::Union{Nothing,PipelineOptions}=nothing,
                preflight::Bool=false,
                strict_preflight::Bool=false)
    plan = plan_ingestion(data, spec;
                          stage=stage,
                          field=field,
                          cache=cache,
                          construction=construction,
                          pipeline=pipeline,
                          preflight=preflight,
                          strict_preflight=strict_preflight)
    return run_ingestion(plan; stage=plan.stage, degree=degree)
end

"""
    encode(path::AbstractString, filtration_or_spec; kind=:auto, format=:auto,
           file_opts=DataFileOptions(), load_kwargs=NamedTuple(), ...)

File-based one-liner ingestion entrypoint. Loads typed data with
`DataFileIO.load_data` and then runs the canonical ingestion pipeline.

Use `plan_ingestion(path, ...)` first when you want to inspect the route,
preflight estimate, or resolved stage before executing the full pipeline.
"""
function encode(path::AbstractString, filtration::AbstractFiltration;
                kind::Symbol=:auto,
                format::Symbol=:auto,
                file_opts::DataFileOptions=DataFileOptions(),
                load_kwargs::NamedTuple=NamedTuple(),
                degree::Int=0,
                stage::Symbol=:auto,
                field::AbstractCoeffField=QQField(),
                cache=:auto,
                construction::Union{Nothing,ConstructionOptions}=nothing,
                pipeline::Union{Nothing,PipelineOptions}=nothing,
                preflight::Bool=false,
                strict_preflight::Bool=false)
    plan = plan_ingestion(path, filtration;
                          kind=kind,
                          format=format,
                          file_opts=file_opts,
                          load_kwargs=load_kwargs,
                          stage=stage,
                          field=field,
                          cache=cache,
                          construction=construction,
                          pipeline=pipeline,
                          preflight=preflight,
                          strict_preflight=strict_preflight)
    return run_ingestion(plan; stage=plan.stage, degree=degree)
end

function encode(path::AbstractString, spec::FiltrationSpec;
                kind::Symbol=:auto,
                format::Symbol=:auto,
                file_opts::DataFileOptions=DataFileOptions(),
                load_kwargs::NamedTuple=NamedTuple(),
                degree::Int=0,
                stage::Symbol=:auto,
                field::AbstractCoeffField=QQField(),
                cache=:auto,
                construction::Union{Nothing,ConstructionOptions}=nothing,
                pipeline::Union{Nothing,PipelineOptions}=nothing,
                preflight::Bool=false,
                strict_preflight::Bool=false)
    plan = plan_ingestion(path, spec;
                          kind=kind,
                          format=format,
                          file_opts=file_opts,
                          load_kwargs=load_kwargs,
                          stage=stage,
                          field=field,
                          cache=cache,
                          construction=construction,
                          pipeline=pipeline,
                          preflight=preflight,
                          strict_preflight=strict_preflight)
    return run_ingestion(plan; stage=plan.stage, degree=degree)
end

end # module DataIngestion
