module IndicatorResolutions

# Indicator resolutions: build (F,dF) and (E,dE) data (upset/downset) and provide
# utilities for converting between presentations and PModule objects.
#
# Responsibilities:
#   - turn a finite fringe module into indicator presentations (upset/downset)
#   - build longer indicator resolutions (F,dF) and (E,dE)
#   - verification routines and small caching helpers
#
# Design notes:
#   - This module does not compute Ext itself. It supplies resolutions and minimal
#     presentation helpers consumed by `DerivedFunctors.HomExtEngine`.
#   - Derived-functor drivers that interpret resolutions (e.g. first-page diagnostics
#     and Ext dimension computations) live in `DerivedFunctors.ExtTorSpaces`.

using SparseArrays, LinearAlgebra
using ..FiniteFringe
using ..IndicatorTypes: UpsetPresentation, DownsetCopresentation,
                        check_upset_presentation, check_downset_copresentation
using ..CoreModules: AbstractCoeffField, QQField, PrimeField, RealField,
                     ResolutionCache, ResolutionKey3, _resolution_key3,
                     IndicatorResolutionPayload, coeff_type, eye, field_from_eltype
using ..FieldLinAlg

using ..Modules: CoverCache, _get_cover_cache, _clear_cover_cache!,
                 CoverEdgeMapStore, _find_sorted_index,
                 PModule, PMorphism, dim_at,
                 zero_pmodule, zero_morphism, map_leq, map_leq_many, map_leq_many!,
                 MapLeqQueryBatch, prepare_map_leq_batch, id_morphism,
                 check_module, check_morphism
import ..Modules: id_morphism

import ..AbelianCategories
using ..AbelianCategories: kernel_with_inclusion, _cokernel_module, _CoverGraphLists, _cover_graph_lists

import ..FiniteFringe: AbstractPoset, FinitePoset, Upset, Downset,
                       principal_upset, principal_downset,
                       cover_edges, nvertices, leq, _succs, _preds,
                       _pred_slots_of_succ,
                       upset_indices, downset_indices
import ..FiniteFringe: build_cache!
import Base.Threads

const INDICATOR_MAP_MEMO_THRESHOLD = Ref(1_000_000)
const INDICATOR_MAP_BATCH_THRESHOLD = Ref(4)
const INDICATOR_THREADS_MIN_VERTICES = Ref(24)
const INDICATOR_THREADS_MIN_TOTAL_DIMS = Ref(192)
const INDICATOR_THREADS_MIN_WORK = Ref(12_000)
const INDICATOR_PREFIX_CACHE_ENABLED = Ref(true)
const INDICATOR_INCREMENTAL_LINALG = Ref(true)
const INDICATOR_MAP_PLAN_MIN_VERTICES = Ref(32)
const INDICATOR_MAP_PLAN_MIN_PAIRS = Ref(256)
const INDICATOR_PREFIX_CACHE_SHARDS = Ref(8)
const INDICATOR_BIRTH_PLAN_CACHE_SHARDS = Ref(8)
const INDICATOR_PMODULE_DIRECT_STORE_MIN_EDGES = Ref(128)
const INDICATOR_PMODULE_DIRECT_STORE_MIN_WORK = Ref(8_192)
const INDICATOR_PMODULE_TARGET_FACTOR_MIN_ROWS_SAVED = Ref(8)
const INDICATOR_PMODULE_TARGET_FACTOR_MIN_SHRINK = Ref(2)
const INDICATOR_UPSET_PREFIX_CACHE_MIN_STEPS = Ref(3)
const INDICATOR_UPSET_PREFIX_CACHE_MIN_VERTICES = Ref(96)
const INDICATOR_UPSET_PREFIX_CACHE_MIN_TOTAL_DIMS = Ref(384)
const _INDICATOR_UPSET_ACTIVE_SOURCE_DELTA = Ref(true)
const INDICATOR_INCREMENTAL_LINALG_MIN_MAPS = Ref(4)
const INDICATOR_INCREMENTAL_LINALG_MIN_ENTRIES = Ref(4_096)
const INDICATOR_INCREMENTAL_VERTEX_CACHE = Ref(true)
const INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_VERTICES = Ref(96)
const INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_TOTAL_DIMS = Ref(256)
const INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS = Ref(12)
const INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES = Ref(24_576)
const _INDICATOR_INJECTIVE_PACKED_PLAN_MIN_SOCLE_VERTICES = Ref(28)
const _INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_GENS = Ref(48)
const _INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_HULL_DIMS = Ref(900)
const _INDICATOR_DOWNSET_RHO_VALUE_ONLY_REUSE = Ref(true)
const _INDICATOR_DOWNSET_RHO_SPARSE_WORKSPACE = Ref(true)
const _INDICATOR_DOWNSET_COKERNEL_TRANSPORT_REUSE = Ref(true)
const _INDICATOR_DOWNSET_TRANSPORT_REUSE_MIN_VERTICES = Ref(25)
const _INDICATOR_DOWNSET_TRANSPORT_REUSE_MIN_TOTAL_DIMS = Ref(96)
const _INDICATOR_DOWNSET_SUPPORT_NARROWING = Ref(true)
const _INDICATOR_DOWNSET_FRONTIER_VERTICES = Ref(true)

@inline function _new_downset_cokernel_transport_cache(::Type{K}, n::Int) where {K}
    return AbelianCategories._new_cokernel_transport_cache(K, n)
end

mutable struct _InjectiveHullReuseCache{K}
    soc::Vector{Matrix{K}}
    linv::Vector{Matrix{K}}
    mult::Vector{Int}
    valid::BitVector
end

@inline function _new_injective_hull_reuse_cache(::Type{K}, n::Int) where {K}
    return _InjectiveHullReuseCache{K}(
        [zeros(K, 0, 0) for _ in 1:n],
        [zeros(K, 0, 0) for _ in 1:n],
        zeros(Int, n),
        falses(n),
    )
end

@inline function _ensure_injective_hull_reuse_cache!(cache::_InjectiveHullReuseCache{K}, n::Int) where {K}
    oldn = length(cache.soc)
    if oldn != n
        resize!(cache.soc, n)
        resize!(cache.linv, n)
        resize!(cache.mult, n)
        resize!(cache.valid, n)
        if n > oldn
            @inbounds for i in (oldn + 1):n
                cache.soc[i] = zeros(K, 0, 0)
                cache.linv[i] = zeros(K, 0, 0)
                cache.mult[i] = 0
                cache.valid[i] = false
            end
        end
    end
    return cache
end

const _INDICATOR_INCREMENTAL_LINALG_MIN_MAPS_DEFAULT = 4
const _INDICATOR_INCREMENTAL_LINALG_MIN_ENTRIES_DEFAULT = 4_096
const _INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_VERTICES_DEFAULT = 96
const _INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_TOTAL_DIMS_DEFAULT = 256
const _INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS_DEFAULT = 12
const _INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES_DEFAULT = 24_576

const _INDICATOR_INCREMENTAL_LINALG_MIN_MAPS_QQ = 6
const _INDICATOR_INCREMENTAL_LINALG_MIN_ENTRIES_QQ = 8_192
const _INDICATOR_INCREMENTAL_LINALG_MIN_MAPS_PRIME = 8
const _INDICATOR_INCREMENTAL_LINALG_MIN_ENTRIES_PRIME = 16_384
const _INDICATOR_INCREMENTAL_LINALG_MIN_MAPS_REAL = 8
const _INDICATOR_INCREMENTAL_LINALG_MIN_ENTRIES_REAL = 12_288

const _INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_VERTICES_QQ = 128
const _INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_TOTAL_DIMS_QQ = 384
const _INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_VERTICES_PRIME = 160
const _INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_TOTAL_DIMS_PRIME = 640
const _INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_VERTICES_REAL = 160
const _INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_TOTAL_DIMS_REAL = 512
const _INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS_QQ = 16
const _INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES_QQ = 49_152
const _INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS_PRIME = 14
const _INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES_PRIME = 40_960
const _INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS_REAL = 12
const _INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES_REAL = 32_768

const _IncrementalVertexCacheEntry{K} = AbelianCategories._VertexIncrementalCacheEntry{K}

struct _BirthPlans
    upset::Vector{Vector{Int}}
    downset::Vector{Vector{Int}}
end

const _INDICATOR_PREFIX_FAMILY_LOCK = Base.ReentrantLock()
const _INDICATOR_PREFIX_FAMILIES = IdDict{DataType,Any}()
const _INDICATOR_BIRTH_PLAN_LOCKS = [Base.ReentrantLock() for _ in 1:INDICATOR_BIRTH_PLAN_CACHE_SHARDS[]]
const _INDICATOR_BIRTH_PLAN_SHARDS = [IdDict{AbstractPoset,_BirthPlans}() for _ in 1:INDICATOR_BIRTH_PLAN_CACHE_SHARDS[]]

@inline _indicator_shard_index(x, nshards::Int) = Int(mod(UInt(objectid(x)), UInt(nshards))) + 1
@inline _indicator_use_direct_pmodule_store(nedges::Int, total_work::Int, nverts::Int=0) =
    nedges >= INDICATOR_PMODULE_DIRECT_STORE_MIN_EDGES[] &&
    total_work >= INDICATOR_PMODULE_DIRECT_STORE_MIN_WORK[] &&
    total_work >= 4 * max(1, nverts)
@inline _indicator_use_target_factor_plan(full_rows::Int, reduced_rows::Int) =
    full_rows - reduced_rows >= INDICATOR_PMODULE_TARGET_FACTOR_MIN_ROWS_SAVED[] &&
    full_rows >= INDICATOR_PMODULE_TARGET_FACTOR_MIN_SHRINK[] * max(1, reduced_rows)
@inline function _nonzero_dim_vertices(dims::AbstractVector{Int})
    verts = Int[]
    @inbounds for u in eachindex(dims)
        dims[u] == 0 && continue
        push!(verts, Int(u))
    end
    return verts
end

@inline function _vertex_mask(n::Int, verts::AbstractVector{Int})
    mask = falses(n)
    @inbounds for u in verts
        mask[u] = true
    end
    return mask
end

@inline function _downset_injective_recompute_mask(
    cc::CoverCache,
    support_mask::BitVector,
    frontier_mask::BitVector,
    changed_vertices::Vector{Int},
)
    n = length(support_mask)
    mask = copy(frontier_mask)
    @inbounds for u in changed_vertices
        mask[u] = true
        for p in _preds(cc, u)
            mask[p] = true
        end
    end
    @inbounds for u in 1:n
        mask[u] &= support_mask[u]
    end
    return mask
end

@inline function _downset_injective_recompute_mask(
    cc::CoverCache,
    support_mask::BitVector,
    frontier_vertices::Vector{Int},
    changed_vertices::Vector{Int},
)
    n = length(support_mask)
    mask = falses(n)
    @inbounds for u in frontier_vertices
        mask[u] = true
    end
    @inbounds for u in changed_vertices
        mask[u] = true
        for p in _preds(cc, u)
            mask[p] = true
        end
    end
    @inbounds for u in 1:n
        mask[u] &= support_mask[u]
    end
    return mask
end

@inline function _downset_injective_support_vertices(
    cc::CoverCache,
    support_vertices::Vector{Int},
    support_mask::BitVector,
    prev_active_socle_vertices::Vector{Int},
    changed_vertices::Vector{Int},
)
    if !_INDICATOR_DOWNSET_SUPPORT_NARROWING[] ||
       isempty(changed_vertices)
        return support_vertices, support_mask
    end

    n = length(support_mask)
    mask = falses(n)
    @inbounds for u in prev_active_socle_vertices
        support_mask[u] || continue
        mask[u] = true
    end
    @inbounds for u in changed_vertices
        if support_mask[u]
            mask[u] = true
        end
        for p in _preds(cc, u)
            support_mask[p] || continue
            mask[p] = true
        end
    end

    if count(mask) == length(support_vertices)
        return support_vertices, support_mask
    end

    verts = Int[]
    sizehint!(verts, count(mask))
    @inbounds for u in support_vertices
        mask[u] || continue
        push!(verts, u)
    end
    return verts, mask
end

@inline _indicator_incremental_linalg_enabled(::Val{:upset}) = INDICATOR_INCREMENTAL_LINALG[]
@inline _indicator_incremental_linalg_enabled(::Val{:downset}) = true

@inline _indicator_vertex_cache_enabled(::Val{:upset}) = INDICATOR_INCREMENTAL_VERTEX_CACHE[]

@inline _indicator_prefix_cache_enabled(::Val{:upset}) = INDICATOR_PREFIX_CACHE_ENABLED[]

@inline function _indicator_incremental_union_thresholds(field::AbstractCoeffField)
    return _indicator_incremental_union_thresholds(field, Val(:upset))
end

@inline function _indicator_incremental_union_thresholds(field::AbstractCoeffField, ::Val{:upset})
    maps = INDICATOR_INCREMENTAL_LINALG_MIN_MAPS[]
    entries = INDICATOR_INCREMENTAL_LINALG_MIN_ENTRIES[]
    if maps != _INDICATOR_INCREMENTAL_LINALG_MIN_MAPS_DEFAULT ||
       entries != _INDICATOR_INCREMENTAL_LINALG_MIN_ENTRIES_DEFAULT
        return maps, entries
    end
    if field isa QQField
        return _INDICATOR_INCREMENTAL_LINALG_MIN_MAPS_QQ, _INDICATOR_INCREMENTAL_LINALG_MIN_ENTRIES_QQ
    elseif field isa PrimeField
        return _INDICATOR_INCREMENTAL_LINALG_MIN_MAPS_PRIME, _INDICATOR_INCREMENTAL_LINALG_MIN_ENTRIES_PRIME
    elseif field isa RealField
        return _INDICATOR_INCREMENTAL_LINALG_MIN_MAPS_REAL, _INDICATOR_INCREMENTAL_LINALG_MIN_ENTRIES_REAL
    end
    return maps, entries
end

@inline function _indicator_incremental_union_thresholds(field::AbstractCoeffField, ::Val{:downset})
    maps = INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS[]
    entries = INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES[]
    if maps != _INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS_DEFAULT ||
       entries != _INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES_DEFAULT
        return maps, entries
    end
    if field isa QQField
        return _INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS_QQ,
               _INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES_QQ
    elseif field isa PrimeField
        return _INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS_PRIME,
               _INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES_PRIME
    elseif field isa RealField
        return _INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_MAPS_REAL,
               _INDICATOR_DOWNSET_INCREMENTAL_LINALG_MIN_ENTRIES_REAL
    end
    return maps, entries
end

@inline function _indicator_vertex_cache_thresholds(field::AbstractCoeffField)
    return _indicator_vertex_cache_thresholds(field, Val(:upset))
end

@inline function _indicator_vertex_cache_thresholds(field::AbstractCoeffField, ::Val{:upset})
    min_vertices = INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_VERTICES[]
    min_total_dims = INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_TOTAL_DIMS[]
    if min_vertices != _INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_VERTICES_DEFAULT ||
       min_total_dims != _INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_TOTAL_DIMS_DEFAULT
        return min_vertices, min_total_dims
    end
    if field isa QQField
        return _INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_VERTICES_QQ,
               _INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_TOTAL_DIMS_QQ
    elseif field isa PrimeField
        return _INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_VERTICES_PRIME,
               _INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_TOTAL_DIMS_PRIME
    elseif field isa RealField
        return _INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_VERTICES_REAL,
               _INDICATOR_INCREMENTAL_VERTEX_CACHE_MIN_TOTAL_DIMS_REAL
    end
    return min_vertices, min_total_dims
end

@inline function _indicator_use_incremental_union(
    field::AbstractCoeffField,
    nrows::Int,
    total_cols::Int,
    nmats::Int,
)
    return _indicator_use_incremental_union(field, nrows, total_cols, nmats, Val(:upset))
end

@inline function _indicator_use_incremental_union(
    field::AbstractCoeffField,
    nrows::Int,
    total_cols::Int,
    nmats::Int,
    side::Val,
)
    _indicator_incremental_linalg_enabled(side) || return false
    min_maps, min_entries = _indicator_incremental_union_thresholds(field, side)
    return nmats >= min_maps && nrows * total_cols >= min_entries
end

@inline function _indicator_use_incremental_vertex_cache(
    field::AbstractCoeffField,
    n::Int,
    total_dims::Int,
)
    return _indicator_use_incremental_vertex_cache(field, n, total_dims, Val(:upset))
end

@inline function _indicator_use_incremental_vertex_cache(
    field::AbstractCoeffField,
    n::Int,
    total_dims::Int,
    side::Val,
)
    _indicator_vertex_cache_enabled(side) || return false
    min_vertices, min_total_dims = _indicator_vertex_cache_thresholds(field, side)
    if side isa Val{:upset}
        min_total_dims = max(min_total_dims, 3 * max(1, n))
    end
    return n >= min_vertices && total_dims >= min_total_dims
end

@inline function _indicator_use_incremental_vertex_cache(
    field::AbstractCoeffField,
    n::Int,
    total_dims::Int,
    ::Val{:downset},
)
    return false
end

@inline function _indicator_use_prefix_cache(
    field::AbstractCoeffField,
    n::Int,
    total_dims::Int,
    target_steps::Int,
    side::Val,
)
    _indicator_prefix_cache_enabled(side) || return false
    if side isa Val{:upset}
        target_steps >= INDICATOR_UPSET_PREFIX_CACHE_MIN_STEPS[] || return false
        n >= INDICATOR_UPSET_PREFIX_CACHE_MIN_VERTICES[] || return false
        min_total_dims = max(INDICATOR_UPSET_PREFIX_CACHE_MIN_TOTAL_DIMS[], 3 * max(1, n))
        if field isa QQField
            min_total_dims = max(min_total_dims, 4 * max(1, n))
        end
        total_dims >= min_total_dims || return false
    end
    return true
end

@inline function _indicator_use_prefix_cache(
    field::AbstractCoeffField,
    n::Int,
    total_dims::Int,
    target_steps::Int,
    ::Val{:downset},
)
    return false
end

@inline function _indicator_upset_auto_profile(M::PModule; maxlen::Union{Nothing,Int}=nothing)
    field = M.field
    n = nvertices(M.Q)
    total_dims = sum(M.dims)
    return (
        n=n,
        total_dims=total_dims,
        vertex_cache=_indicator_use_incremental_vertex_cache(field, n, total_dims, Val(:upset)),
        prefix_cache=(maxlen !== nothing &&
                      _indicator_use_prefix_cache(field, n, total_dims, maxlen, Val(:upset))),
        incremental_linalg_thresholds=_indicator_incremental_union_thresholds(field, Val(:upset)),
    )
end

@inline function _indicator_use_downset_transport_reuse(
    field::AbstractCoeffField,
    n::Int,
    total_dims::Int,
    maxlen::Union{Nothing,Int},
)
    _INDICATOR_DOWNSET_COKERNEL_TRANSPORT_REUSE[] || return false
    maxlen !== nothing && maxlen <= 1 && return false
    if maxlen !== nothing
        n >= _INDICATOR_DOWNSET_TRANSPORT_REUSE_MIN_VERTICES[] ||
            total_dims >= _INDICATOR_DOWNSET_TRANSPORT_REUSE_MIN_TOTAL_DIMS[] || return false
    end
    return true
end

@inline function _indicator_use_downset_injective_reuse(
    field::AbstractCoeffField,
    n::Int,
    total_dims::Int,
    maxlen::Union{Nothing,Int},
)
    maxlen !== nothing && maxlen <= 1 && return false
    return true
end

@inline function _indicator_downset_auto_profile(M::PModule; maxlen::Union{Nothing,Int}=nothing)
    field = M.field
    n = nvertices(M.Q)
    total_dims = sum(M.dims)
    return (
        n=n,
        total_dims=total_dims,
        vertex_cache=false,
        prefix_cache=(maxlen !== nothing &&
                      _indicator_use_prefix_cache(field, n, total_dims, maxlen, Val(:downset))),
        transport_reuse=_indicator_use_downset_transport_reuse(field, n, total_dims, maxlen),
        injective_reuse=_indicator_use_downset_injective_reuse(field, n, total_dims, maxlen),
        incremental_linalg_thresholds=_indicator_incremental_union_thresholds(field, Val(:downset)),
    )
end

@inline _indicator_use_array_memo(n::Int) = n * n <= INDICATOR_MAP_MEMO_THRESHOLD[]
@inline _indicator_memo_index(n::Int, u::Int, v::Int) = (u - 1) * n + v

@inline function _indicator_new_array_memo(::Type{K}, n::Int) where {K}
    memo = Vector{Union{Nothing,Matrix{K}}}(undef, n * n)
    fill!(memo, nothing)
    return memo
end

@inline function _indicator_memo_get(memo::AbstractVector{Union{Nothing,Matrix{K}}}, n::Int, u::Int, v::Int) where {K}
    return memo[_indicator_memo_index(n, u, v)]
end

@inline function _indicator_memo_set!(memo::AbstractVector{Union{Nothing,Matrix{K}}}, n::Int, u::Int, v::Int, val::Matrix{K}) where {K}
    memo[_indicator_memo_index(n, u, v)] = val
    return val
end

@inline function _indicator_memo_for_module!(
    pool::IdDict{Any,Vector{Union{Nothing,Matrix{K}}}},
    M::PModule{K},
) where {K}
    return get!(pool, M) do
        _indicator_new_array_memo(K, nvertices(M.Q))
    end
end

@inline function _map_leq_cached_many_indicator(
    M::PModule{K},
    pairs::Vector{Tuple{Int,Int}},
    cc::CoverCache,
    memo::AbstractVector{Union{Nothing,Matrix{K}}},
) where {K}
    out = Vector{Matrix{K}}(undef, length(pairs))
    missing_pairs = Tuple{Int,Int}[]
    missing_idx = Int[]
    n = nvertices(M.Q)

    @inbounds for i in eachindex(pairs)
        u, v = pairs[i]
        X = _indicator_memo_get(memo, n, u, v)
        if X === nothing
            push!(missing_pairs, (u, v))
            push!(missing_idx, i)
        else
            out[i] = X
        end
    end

    if !isempty(missing_pairs)
        if length(missing_pairs) <= INDICATOR_MAP_BATCH_THRESHOLD[]
            @inbounds for t in eachindex(missing_idx)
                i = missing_idx[t]
                u, v = missing_pairs[t]
                Xraw = map_leq(M, u, v; cache=cc)
                Xmat = Xraw isa Matrix{K} ? Xraw : Matrix{K}(Xraw)
                out[i] = _indicator_memo_set!(memo, n, u, v, Xmat)
            end
        else
            fetched = Vector{AbstractMatrix{K}}(undef, length(missing_pairs))
            batch = prepare_map_leq_batch(missing_pairs)
            map_leq_many!(fetched, M, batch; cache=cc)
            @inbounds for t in eachindex(missing_idx)
                i = missing_idx[t]
                u, v = missing_pairs[t]
                Xraw = fetched[t]
                Xmat = Xraw isa Matrix{K} ? Xraw : Matrix{K}(Xraw)
                out[i] = _indicator_memo_set!(memo, n, u, v, Xmat)
            end
        end
    end

    return out
end

@inline function _map_leq_cached_many_indicator(
    M::PModule{K},
    batch::MapLeqQueryBatch,
    cc::CoverCache,
    memo::AbstractVector{Union{Nothing,Matrix{K}}},
) where {K}
    pairs = batch.pairs
    out = Vector{Matrix{K}}(undef, length(pairs))
    missing_pairs = Tuple{Int,Int}[]
    missing_idx = Int[]
    n = nvertices(M.Q)

    @inbounds for i in eachindex(pairs)
        u, v = pairs[i]
        X = _indicator_memo_get(memo, n, u, v)
        if X === nothing
            push!(missing_pairs, (u, v))
            push!(missing_idx, i)
        else
            out[i] = X
        end
    end

    if !isempty(missing_pairs)
        if length(missing_pairs) <= INDICATOR_MAP_BATCH_THRESHOLD[]
            @inbounds for t in eachindex(missing_idx)
                i = missing_idx[t]
                u, v = missing_pairs[t]
                Xraw = map_leq(M, u, v; cache=cc)
                Xmat = Xraw isa Matrix{K} ? Xraw : Matrix{K}(Xraw)
                out[i] = _indicator_memo_set!(memo, n, u, v, Xmat)
            end
        else
            fetched = Vector{AbstractMatrix{K}}(undef, length(missing_pairs))
            miss_batch = prepare_map_leq_batch(missing_pairs)
            map_leq_many!(fetched, M, miss_batch; cache=cc)
            @inbounds for t in eachindex(missing_idx)
                i = missing_idx[t]
                u, v = missing_pairs[t]
                Xraw = fetched[t]
                Xmat = Xraw isa Matrix{K} ? Xraw : Matrix{K}(Xraw)
                out[i] = _indicator_memo_set!(memo, n, u, v, Xmat)
            end
        end
    end

    return out
end


@inline function _map_leq_fill_memo_indicator!(
    M::PModule{K},
    pairs::Vector{Tuple{Int,Int}},
    cc::CoverCache,
    memo::AbstractVector{Union{Nothing,Matrix{K}}},
    ws,
) where {K}
    n = nvertices(M.Q)
    empty!(ws.missing_pairs_buf)

    @inbounds for i in eachindex(pairs)
        u, v = pairs[i]
        _indicator_memo_get(memo, n, u, v) === nothing || continue
        push!(ws.missing_pairs_buf, (u, v))
    end

    isempty(ws.missing_pairs_buf) && return nothing

    if length(ws.missing_pairs_buf) <= INDICATOR_MAP_BATCH_THRESHOLD[]
        @inbounds for i in eachindex(ws.missing_pairs_buf)
            u, v = ws.missing_pairs_buf[i]
            Xraw = map_leq(M, u, v; cache=cc)
            Xmat = Xraw isa Matrix{K} ? Xraw : Matrix{K}(Xraw)
            _indicator_memo_set!(memo, n, u, v, Xmat)
        end
        return nothing
    end

    resize!(ws.fetched_buf, length(ws.missing_pairs_buf))
    batch = _workspace_get_or_prepare_batch!(ws, ws.missing_pairs_buf)
    map_leq_many!(ws.fetched_buf, M, batch; cache=cc)
    @inbounds for i in eachindex(ws.missing_pairs_buf)
        u, v = ws.missing_pairs_buf[i]
        Xraw = ws.fetched_buf[i]
        Xmat = Xraw isa Matrix{K} ? Xraw : Matrix{K}(Xraw)
        _indicator_memo_set!(memo, n, u, v, Xmat)
    end
    return nothing
end

@inline function _pairs_signature(pairs::Vector{Tuple{Int,Int}})
    h = hash(length(pairs))
    @inbounds for p in pairs
        h = hash(p, h)
    end
    return UInt64(h)
end

@inline function _pairs_equal(a::Vector{Tuple{Int,Int}}, b::Vector{Tuple{Int,Int}})
    length(a) == length(b) || return false
    @inbounds for i in eachindex(a)
        a[i] == b[i] || return false
    end
    return true
end

@inline function _workspace_get_or_prepare_batch!(ws, pairs::Vector{Tuple{Int,Int}})
    key = _pairs_signature(pairs)
    batch = get(ws.map_batch_cache, key, nothing)
    if batch !== nothing && _pairs_equal(batch.pairs, pairs)
        return batch
    end
    b = prepare_map_leq_batch(pairs)
    ws.map_batch_cache[key] = b
    return b
end

@inline function _map_leq_fill_memo_indicator!(
    M::PModule{K},
    batch::MapLeqQueryBatch,
    cc::CoverCache,
    memo::AbstractVector{Union{Nothing,Matrix{K}}},
    ws,
) where {K}
    return _map_leq_fill_memo_indicator!(M, batch.pairs, cc, memo, ws)
end

@inline function _sparse_from_workspace!(
    ws,
    m::Int,
    n::Int,
)
    nnz_trip = length(ws.I)
    if m == 0 || n == 0 || nnz_trip == 0
        return spzeros(eltype(ws.V), m, n)
    end
    resize!(ws.klasttouch, n)
    resize!(ws.csrrowptr, m + 1)
    resize!(ws.csrcolval, nnz_trip)
    resize!(ws.csrnzval, nnz_trip)

    # Ping-pong owned CSC buffers. Each call hands a slot's arrays to the
    # returned matrix and rotates to the other slot for the next call.
    slot = ws.csc_slot
    colptr = ws.csccolptr_slots[slot]
    rowval = ws.cscrowval_slots[slot]
    nzval = ws.cscnzval_slots[slot]
    resize!(colptr, n + 1)
    resize!(rowval, nnz_trip)
    resize!(nzval, nnz_trip)

    A = SparseArrays.sparse!(
        ws.I,
        ws.J,
        ws.V,
        m,
        n,
        +,
        ws.klasttouch,
        ws.csrrowptr,
        ws.csrcolval,
        ws.csrnzval,
        colptr,
        rowval,
        nzval,
    )

    # Transfer ownership of CSC arrays to `A`; install fresh arrays in the slot.
    ws.csccolptr_slots[slot] = Int[]
    ws.cscrowval_slots[slot] = Int[]
    ws.cscnzval_slots[slot] = Vector{eltype(ws.V)}()
    ws.csc_slot = slot == 1 ? 2 : 1

    return A
end

struct _PackedIntLists
    ptr::Vector{Int}
    data::Vector{Int}
end

@inline _packed_length(p::_PackedIntLists, i::Int) = p.ptr[i + 1] - p.ptr[i]
@inline _packed_firstindex(p::_PackedIntLists, i::Int) = p.ptr[i]
@inline _packed_lastindex(p::_PackedIntLists, i::Int) = p.ptr[i + 1] - 1

@inline function _accumulate_product_entries_upset!(
    I::Vector{Int},
    J::Vector{Int},
    V::Vector{K},
    A::AbstractMatrix{K},
    B::AbstractMatrix{K},
    row0::Int,
    col0::Int,
    nrows::Int,
    ncols::Int,
    theta_gid0::Int,
    lambda_gid0::Int,
) where {K}
    kdim = size(A, 2)
    @inbounds for rr in 0:(nrows - 1)
        row = row0 + rr
        for cc in 0:(ncols - 1)
            col = col0 + cc
            s = zero(K)
            for k in 1:kdim
                s += A[row, k] * B[k, col]
            end
            if s != 0
                push!(I, theta_gid0 + cc)
                push!(J, lambda_gid0 + rr)
                push!(V, s)
            end
        end
    end
    return nothing
end

@inline function _accumulate_product_entries_upset!(
    I::Vector{Int},
    J::Vector{Int},
    V::Vector{K},
    A::AbstractMatrix{K},
    B::AbstractMatrix{K},
    active_sources::_PackedIntLists,
    theta::Int,
    counts_prev::Vector{Int},
    gid_prev_starts::Vector{Int},
    col0::Int,
    ncols::Int,
    theta_gid0::Int,
) where {K}
    row0 = 1
    data = active_sources.data
    lo = _packed_firstindex(active_sources, theta)
    hi = _packed_lastindex(active_sources, theta)
    @inbounds for idx in lo:hi
        plambda = data[idx]
        clambda = counts_prev[plambda]
        lambda_gid0 = gid_prev_starts[plambda]
        _accumulate_product_entries_upset!(
            I,
            J,
            V,
            A,
            B,
            row0,
            col0,
            clambda,
            ncols,
            theta_gid0,
            lambda_gid0,
        )
        row0 += clambda
    end
    return nothing
end

@inline function _accumulate_product_entries_downset!(
    I::Vector{Int},
    J::Vector{Int},
    V::Vector{K},
    A::AbstractMatrix{K},
    B::AbstractMatrix{K},
    row0::Int,
    col0::Int,
    nrows::Int,
    ncols::Int,
    theta_gid0::Int,
    lambda_gid0::Int,
) where {K}
    kdim = size(A, 2)
    @inbounds for rr in 0:(nrows - 1)
        row = row0 + rr
        for cc in 0:(ncols - 1)
            col = col0 + cc
            s = zero(K)
            for k in 1:kdim
                s += A[row, k] * B[k, col]
            end
            if s != 0
                push!(I, theta_gid0 + rr)
                push!(J, lambda_gid0 + cc)
                push!(V, s)
            end
        end
    end
    return nothing
end

@inline function _generator_id_ranges(counts::Vector{Int})
    ranges = Vector{UnitRange{Int}}(undef, length(counts))
    off = 0
    @inbounds for i in eachindex(counts)
        c = counts[i]
        if c > 0
            ranges[i] = (off + 1):(off + c)
            off += c
        else
            ranges[i] = 1:0
        end
    end
    return ranges, off
end

@inline function _fill_generator_starts!(
    starts::Vector{Int},
    counts::Vector{Int},
)
    n = length(counts)
    length(starts) == n + 1 || error("_fill_generator_starts!: wrong length")
    starts[1] = 1
    @inbounds for i in 1:n
        starts[i + 1] = starts[i] + counts[i]
    end
    return starts[n + 1] - 1
end

@inline function _packed_block_starts(
    active_sources::_PackedIntLists,
    counts::Vector{Int},
)
    ptr = active_sources.ptr
    data = Vector{Int}(undef, length(active_sources.data))
    @inbounds for i in 1:(length(ptr) - 1)
        off = 1
        lo = ptr[i]
        hi = ptr[i + 1] - 1
        if hi >= lo
            for idx in lo:hi
                u = active_sources.data[idx]
                data[idx] = off
                off += counts[u]
            end
        end
    end
    return _PackedIntLists(ptr, data)
end

@inline function _packed_block_start_for_source(
    active_sources::_PackedIntLists,
    starts::_PackedIntLists,
    target_idx::Int,
    source_idx::Int,
)
    lo = _packed_firstindex(active_sources, target_idx)
    hi = _packed_lastindex(active_sources, target_idx)
    @inbounds for idx in lo:hi
        active_sources.data[idx] == source_idx && return starts.data[idx]
    end
    return 0
end

@inline function _gather_selected_columns!(
    cols::Matrix{K},
    col0::Int,
    A::Matrix{K},
    chosen_cols::Vector{Int},
) where {K}
    Mi = size(A, 1)
    @inbounds for s in eachindex(chosen_cols)
        j = chosen_cols[s]
        col = col0 + s - 1
        for r in 1:Mi
            cols[r, col] = A[r, j]
        end
    end
    return col0 + length(chosen_cols)
end

struct _InjectiveGeneratorPlan
    mult::Vector{Int}
    gid_starts::Vector{Int}
    total::Int
    active_socle_vertices::Vector{Int}
    active_sources::_PackedIntLists
    active_gid_lists::_PackedIntLists
    Edims::Vector{Int}
end

struct _ProjectiveGeneratorPlan
    chosen_cols::Vector{Vector{Int}}
    counts::Vector{Int}
    gid_starts::Vector{Int}
    total::Int
    active_birth_vertices::Vector{Int}
    active_sources::_PackedIntLists
    Fdims::Vector{Int}
    self_starts_down::Vector{Int}
end

struct _ProjectiveGeneratorVertexView <: AbstractVector{Tuple{Int,Int}}
    birth_vertex::Int
    chosen_cols::Vector{Int}
end

Base.IndexStyle(::Type{_ProjectiveGeneratorVertexView}) = IndexLinear()
Base.size(view::_ProjectiveGeneratorVertexView) = (length(view.chosen_cols),)
Base.length(view::_ProjectiveGeneratorVertexView) = length(view.chosen_cols)
Base.eltype(::Type{_ProjectiveGeneratorVertexView}) = Tuple{Int,Int}
@inline Base.getindex(view::_ProjectiveGeneratorVertexView, i::Int) =
    (view.birth_vertex, view.chosen_cols[i])
@inline function Base.iterate(view::_ProjectiveGeneratorVertexView, state::Int=1)
    state > length(view.chosen_cols) && return nothing
    return ((view.birth_vertex, view.chosen_cols[state]), state + 1)
end

"""
    ProjectiveGenerators

Lazy public wrapper over the generator labels produced by `projective_cover`.
Index by birth vertex: `gens[p]` is an iterable/vector-like view of pairs
`(p, j)` naming the selected generator columns in `M_p`.

This wrapper avoids eagerly materializing `Vector{Vector{Tuple{Int,Int}}}` on the
public path while preserving the same mathematical reading and iteration model.
Internal callers that need the packed representation should use
`projective_cover(...; materialize_gens=false)`.

Use [`generator_vertices`](@ref) and [`generator_blocks`](@ref) to inspect the
birth vertices and multiplicity blocks without materializing all labels. In
ordinary workflows, keep this wrapper lazy and only materialize explicit tuple
labels when a downstream calculation truly needs them.
"""
struct ProjectiveGenerators <: AbstractVector{_ProjectiveGeneratorVertexView}
    plan::_ProjectiveGeneratorPlan
end

Base.IndexStyle(::Type{ProjectiveGenerators}) = IndexLinear()
Base.size(gens::ProjectiveGenerators) = (length(gens.plan.counts),)
Base.length(gens::ProjectiveGenerators) = length(gens.plan.counts)
Base.eltype(::Type{ProjectiveGenerators}) = _ProjectiveGeneratorVertexView
@inline Base.getindex(gens::ProjectiveGenerators, i::Int) =
    _ProjectiveGeneratorVertexView(i, gens.plan.chosen_cols[i])
@inline function Base.iterate(gens::ProjectiveGenerators, state::Int=1)
    state > length(gens) && return nothing
    return (gens[state], state + 1)
end

@inline _projective_generators_view(plan::_ProjectiveGeneratorPlan) = ProjectiveGenerators(plan)

struct _InjectiveGeneratorVertexView <: AbstractVector{Tuple{Int,Int}}
    socle_vertex::Int
    multiplicity::Int
end

Base.IndexStyle(::Type{_InjectiveGeneratorVertexView}) = IndexLinear()
Base.size(view::_InjectiveGeneratorVertexView) = (view.multiplicity,)
Base.length(view::_InjectiveGeneratorVertexView) = view.multiplicity
Base.eltype(::Type{_InjectiveGeneratorVertexView}) = Tuple{Int,Int}
@inline Base.getindex(view::_InjectiveGeneratorVertexView, i::Int) = (view.socle_vertex, i)
@inline function Base.iterate(view::_InjectiveGeneratorVertexView, state::Int=1)
    state > view.multiplicity && return nothing
    return ((view.socle_vertex, state), state + 1)
end

"""
    InjectiveGenerators

Lazy public wrapper over the generator labels produced by `injective_hull`.
Index by socle vertex: `gens[u]` is an iterable/vector-like view of pairs
`(u, j)` naming the selected socle generators at `M_u`.

This wrapper avoids eagerly materializing `Vector{Vector{Tuple{Int,Int}}}` on the
public path while preserving the same mathematical reading and iteration model.
Internal callers that need the packed representation should use
`injective_hull(...; materialize_gens=false)` or `_injective_hull(...; materialize_gens=false)`.

Use [`generator_vertices`](@ref) and [`generator_blocks`](@ref) to inspect the
socle vertices and multiplicity blocks without materializing all tuple labels.
This is the cheap/default inspection surface for hull generators.
"""
struct InjectiveGenerators <: AbstractVector{_InjectiveGeneratorVertexView}
    plan::_InjectiveGeneratorPlan
end

Base.IndexStyle(::Type{InjectiveGenerators}) = IndexLinear()
Base.size(gens::InjectiveGenerators) = (length(gens.plan.mult),)
Base.length(gens::InjectiveGenerators) = length(gens.plan.mult)
Base.eltype(::Type{InjectiveGenerators}) = _InjectiveGeneratorVertexView
@inline Base.getindex(gens::InjectiveGenerators, i::Int) =
    _InjectiveGeneratorVertexView(i, gens.plan.mult[i])
@inline function Base.iterate(gens::InjectiveGenerators, state::Int=1)
    state > length(gens) && return nothing
    return (gens[state], state + 1)
end

@inline _injective_generators_view(plan::_InjectiveGeneratorPlan) = InjectiveGenerators(plan)

function cover_module end
function cover_map end
function hull_module end
function hull_map end
function resolution_modules end
function resolution_maps end
function resolution_generators end
function projective_resolution end
function injective_resolution end
function augmentation end
function coaugmentation end
function presentation_module end
function presentation_map end
function generator_vertices end
function generator_blocks end
function materialize_generators end
function resolution_length end
function generator_count end
function generator_count_by_degree end
function cover_summary end
function hull_summary end
function resolution_summary end
function presentation_summary end
function check_projective_cover end
function check_injective_hull end
function check_resolution end
function check_fringe_presentation end
function indicator_resolution_validation_summary end

"""
    IndicatorResolutionValidationSummary

Display-oriented wrapper for validation reports returned by the
`check_*` helpers in `IndicatorResolutions`.

The canonical validation helpers return structured `NamedTuple`s for easy
programmatic use. Wrap one of those reports with
[`indicator_resolution_validation_summary`](@ref) when you want a compact
notebook/REPL summary instead of field-by-field inspection.
"""
struct IndicatorResolutionValidationSummary{R}
    report::R
end

"""
    ProjectiveCoverResult

Typed result object returned by [`projective_cover`](@ref).

The object stores the projective cover module, the augmentation map onto the
source module, and the generator data (either lazy public generator views or the
packed internal plan when `materialize_gens=false`). It iterates as
`(cover_module, cover_map, generators)` for compatibility with tuple
destructuring in existing code.

Canonical accessors are:
- [`cover_module`](@ref)
- [`cover_map`](@ref) / [`augmentation`](@ref)
- [`resolution_generators`](@ref)
- [`cover_summary`](@ref) or `describe(...)`

Prefer the summary/accessor surface over direct field inspection. The default
public path keeps generator labels lazy; packed generators are for internal or
benchmark-sensitive workflows.
"""
struct ProjectiveCoverResult{MType,MapType,GenType}
    module_object::MType
    map::MapType
    generators::GenType
    minimal_requested::Bool
    checks_requested::Bool
end

"""
    InjectiveHullResult

Typed result object returned by [`injective_hull`](@ref).

The object stores the injective hull module, the coaugmentation map from the
source module, and the generator data (lazy public views by default, or the
packed internal plan when `materialize_gens=false`). It iterates as
`(hull_module, hull_map, generators)` for compatibility with tuple
destructuring.

Canonical accessors are:
- [`hull_module`](@ref)
- [`hull_map`](@ref) / [`coaugmentation`](@ref)
- [`resolution_generators`](@ref)
- [`hull_summary`](@ref) or `describe(...)`
"""
struct InjectiveHullResult{MType,MapType,GenType}
    module_object::MType
    map::MapType
    generators::GenType
    minimal_requested::Bool
    checks_requested::Bool
end

"""
    UpsetResolutionResult

Typed result object returned by [`upset_resolution`](@ref).

It stores the presentation objects in degree order, the differential matrices,
the packed generator plans backing each stage, and the augmentation
`F_0 -> M`. It iterates as `(F, dF)` for compatibility with existing tuple
destructuring.

Use [`resolution_modules`](@ref), [`resolution_maps`](@ref),
[`resolution_generators`](@ref), and [`augmentation`](@ref) for semantic access.
Use [`resolution_summary`](@ref) or `describe(...)` when you want a cheap
overview before touching the full presentation data.
"""
struct UpsetResolutionResult{PresType,DiffType,PlanType,MapType,ModuleType}
    presentations::PresType
    differentials::DiffType
    generator_plans::PlanType
    augmentation_map::MapType
    source_module::ModuleType
    minimal_requested::Bool
    checks_requested::Bool
end

"""
    DownsetResolutionResult

Typed result object returned by [`downset_resolution`](@ref).

It stores the copresentation objects in degree order, the differential
matrices, the packed generator plans backing each stage, and the coaugmentation
`M -> E^0`. It iterates as `(E, dE)` for compatibility with existing tuple
destructuring.

Use [`resolution_modules`](@ref), [`resolution_maps`](@ref),
[`resolution_generators`](@ref), and [`coaugmentation`](@ref) for semantic
access. Use [`resolution_summary`](@ref) or `describe(...)` for cheap-first
inspection.
"""
struct DownsetResolutionResult{PresType,DiffType,PlanType,MapType,ModuleType}
    presentations::PresType
    differentials::DiffType
    generator_plans::PlanType
    coaugmentation_map::MapType
    source_module::ModuleType
    minimal_requested::Bool
    checks_requested::Bool
end

"""
    IndicatorResolutionsResult

Typed result object returned by [`indicator_resolutions`](@ref).

It stores the projective/upset and injective/downset resolution data together.
The object iterates as `(F, dF, E, dE)` so existing tuple-destructuring call
sites continue to work unchanged.

Use [`projective_resolution`](@ref) and [`injective_resolution`](@ref) instead
of reaching for raw `.upset` / `.downset` fields. Use
[`resolution_summary`](@ref) or `describe(...)` for a compact overview.
"""
struct IndicatorResolutionsResult{UpsetType,DownsetType}
    upset::UpsetType
    downset::DownsetType
end

@inline function indicator_resolution_validation_summary(report::NamedTuple)
    return IndicatorResolutionValidationSummary(report)
end

"""
    cover_module(res::ProjectiveCoverResult)
    cover_map(res::ProjectiveCoverResult)
    hull_module(res::InjectiveHullResult)
    hull_map(res::InjectiveHullResult)

Semantic accessors for the cover/hull result wrappers.

Use these instead of reaching into raw result fields. They return the underlying
`PModule` and `PMorphism` objects represented by the cover or hull result.
"""

"""
    resolution_modules(res)
    resolution_maps(res)
    resolution_generators(res)

Semantic accessors for resolution-style result objects.

- `resolution_modules(res)` returns the presentation or copresentation objects.
- `resolution_maps(res)` returns the differential matrices.
- `resolution_generators(res)` returns the generator data in the cheap/default
  lazy form whenever possible.

For combined [`IndicatorResolutionsResult`](@ref) objects, these return named
tuples with `projective`/`injective` data split by side.
"""

"""
    projective_resolution(res::IndicatorResolutionsResult)
    injective_resolution(res::IndicatorResolutionsResult)

Extract the projective/upset or injective/downset half of a combined
[`IndicatorResolutionsResult`](@ref).

These are the canonical accessors for the combined result; prefer them over raw
field access.
"""

"""
    generator_vertices(gens)
    generator_blocks(gens)

Cheap inspection helpers for lazy or packed generator data.

- `generator_vertices(gens)` returns the ambient vertices carrying nonzero
  generator blocks.
- `generator_blocks(gens)` returns the multiplicity/count for each such block.

These helpers are the preferred way to inspect lazy generator wrappers without
materializing explicit `(vertex, local_index)` tuples.
"""

"""
    materialize_generators(obj)

Explicitly materialize lazy generator wrappers used by `IndicatorResolutions`.

Mathematically, this converts the cheap lazy generator view into explicit
generator labels `(vertex, local_index)` that can be stored, serialized, or
handed to downstream code that expects fully realized tuple data.

For lazy [`ProjectiveGenerators`](@ref) / [`InjectiveGenerators`](@ref), this
returns the concrete nested generator lists `Vector{Vector{Tuple{Int,Int}}}`.
For result wrappers, it returns the materialized generator payload associated to
that result:

- cover/hull results return one materialized generator collection,
- resolution results return one materialized generator collection per degree,
- combined results return `(; projective=..., injective=...)`.

Best practice:
- stay on the lazy wrappers for inspection, summaries, and most algebraic
  workflows,
- call `materialize_generators(...)` only when a downstream consumer truly
  needs explicit tuple labels.

The lazy/default path is cheaper in memory and usually cheaper in runtime. Full
materialization is an explicit opt-in.
"""

"""
    resolution_length(res)
    generator_count(res)
    generator_count_by_degree(res)

Cheap scalar helpers for `IndicatorResolutions` result wrappers.

These are cheap inspection helpers for the mathematical size of a cover, hull,
or resolution:

- `resolution_length(res)` returns the number of differentials on the chosen
  side, so covers/hulls have length `0` and multi-step resolutions have one
  entry per computed differential.
- `generator_count(res)` returns the total number of generators stored by that
  result.
- `generator_count_by_degree(res)` returns the per-degree generator counts used
  in the compact summaries.

For combined [`IndicatorResolutionsResult`](@ref), these return named tuples
with `projective`/`injective` entries.

Use these when you want a cheap quantitative summary without touching the full
presentation/copresentation data or materializing generators.
"""

"""
    cover_summary(res::ProjectiveCoverResult)
    hull_summary(res::InjectiveHullResult)
    resolution_summary(res)
    presentation_summary(H)

Owner-local summary helpers for `IndicatorResolutions`.

Use these when you want a discoverable, cheap summary from this owner module
without relying on the shared `describe(...)` generic.
"""

"""
    check_projective_cover(res; throw=false)
    check_injective_hull(res; throw=false)
    check_resolution(res; throw=false)
    check_fringe_presentation(obj; throw=false)

Notebook-friendly validation helpers for the main `IndicatorResolutions`
results.

They return structured `NamedTuple` reports for programmatic use. Wrap those
reports with [`indicator_resolution_validation_summary`](@ref) when you want a
compact REPL/notebook display.

Set `throw=true` to raise an `ArgumentError` on failure instead of returning a
report.
"""

@inline Base.length(::ProjectiveCoverResult) = 3
@inline Base.length(::InjectiveHullResult) = 3
@inline Base.length(::UpsetResolutionResult) = 2
@inline Base.length(::DownsetResolutionResult) = 2
@inline Base.length(::IndicatorResolutionsResult) = 4

@inline function Base.iterate(res::ProjectiveCoverResult, state::Int=1)
    state == 1 && return (res.module_object, 2)
    state == 2 && return (res.map, 3)
    state == 3 && return (res.generators, 4)
    return nothing
end

@inline function Base.iterate(res::InjectiveHullResult, state::Int=1)
    state == 1 && return (res.module_object, 2)
    state == 2 && return (res.map, 3)
    state == 3 && return (res.generators, 4)
    return nothing
end

@inline function Base.iterate(res::UpsetResolutionResult, state::Int=1)
    state == 1 && return (res.presentations, 2)
    state == 2 && return (res.differentials, 3)
    return nothing
end

@inline function Base.iterate(res::DownsetResolutionResult, state::Int=1)
    state == 1 && return (res.presentations, 2)
    state == 2 && return (res.differentials, 3)
    return nothing
end

@inline function Base.iterate(res::IndicatorResolutionsResult, state::Int=1)
    state == 1 && return (res.upset.presentations, 2)
    state == 2 && return (res.upset.differentials, 3)
    state == 3 && return (res.downset.presentations, 4)
    state == 4 && return (res.downset.differentials, 5)
    return nothing
end

@inline _indicator_resolution_field_label(::QQField) = "QQ"
@inline _indicator_resolution_field_label(field::PrimeField) = "Fp($(field.p))"
@inline _indicator_resolution_field_label(field::RealField{T}) where {T<:AbstractFloat} = "RealField($(T))"
@inline _indicator_resolution_field_label(field::AbstractCoeffField) = string(nameof(typeof(field)))

@inline generator_vertices(gens::ProjectiveGenerators) = gens.plan.active_birth_vertices
@inline generator_vertices(gens::InjectiveGenerators) = gens.plan.active_socle_vertices
@inline generator_vertices(plan::_ProjectiveGeneratorPlan) = plan.active_birth_vertices
@inline generator_vertices(plan::_InjectiveGeneratorPlan) = plan.active_socle_vertices

@inline generator_blocks(gens::ProjectiveGenerators) = gens.plan.counts
@inline generator_blocks(gens::InjectiveGenerators) = gens.plan.mult
@inline generator_blocks(plan::_ProjectiveGeneratorPlan) = plan.counts
@inline generator_blocks(plan::_InjectiveGeneratorPlan) = plan.mult

@inline materialize_generators(gens::ProjectiveGenerators) = _materialize_projective_gens(gens.plan)
@inline materialize_generators(gens::InjectiveGenerators) = _materialize_injective_gens(gens.plan)
@inline materialize_generators(plan::_ProjectiveGeneratorPlan) = _materialize_projective_gens(plan)
@inline materialize_generators(plan::_InjectiveGeneratorPlan) = _materialize_injective_gens(plan)

@inline _generator_total(gens) = sum(generator_blocks(gens))
@inline _generator_storage(gens) = gens isa ProjectiveGenerators || gens isa InjectiveGenerators ? :lazy : :packed

@inline cover_module(res::ProjectiveCoverResult) = res.module_object
@inline cover_map(res::ProjectiveCoverResult) = res.map
@inline hull_module(res::InjectiveHullResult) = res.module_object
@inline hull_map(res::InjectiveHullResult) = res.map
@inline resolution_modules(res::UpsetResolutionResult) = res.presentations
@inline resolution_modules(res::DownsetResolutionResult) = res.presentations
@inline resolution_modules(res::IndicatorResolutionsResult) = (; projective=res.upset.presentations, injective=res.downset.presentations)
@inline resolution_maps(res::UpsetResolutionResult) = res.differentials
@inline resolution_maps(res::DownsetResolutionResult) = res.differentials
@inline resolution_maps(res::IndicatorResolutionsResult) = (; projective=res.upset.differentials, injective=res.downset.differentials)
@inline resolution_generators(res::ProjectiveCoverResult) = res.generators
@inline resolution_generators(res::InjectiveHullResult) = res.generators
@inline resolution_generators(res::UpsetResolutionResult) = [_projective_generators_view(plan) for plan in res.generator_plans]
@inline resolution_generators(res::DownsetResolutionResult) = [_injective_generators_view(plan) for plan in res.generator_plans]
@inline resolution_generators(res::IndicatorResolutionsResult) = (; projective=resolution_generators(res.upset), injective=resolution_generators(res.downset))
@inline materialize_generators(res::ProjectiveCoverResult) = materialize_generators(res.generators)
@inline materialize_generators(res::InjectiveHullResult) = materialize_generators(res.generators)
@inline materialize_generators(res::UpsetResolutionResult) = [materialize_generators(plan) for plan in res.generator_plans]
@inline materialize_generators(res::DownsetResolutionResult) = [materialize_generators(plan) for plan in res.generator_plans]
@inline materialize_generators(res::IndicatorResolutionsResult) = (; projective=materialize_generators(res.upset), injective=materialize_generators(res.downset))
@inline projective_resolution(res::IndicatorResolutionsResult) = res.upset
@inline injective_resolution(res::IndicatorResolutionsResult) = res.downset
@inline augmentation(res::ProjectiveCoverResult) = res.map
@inline augmentation(res::UpsetResolutionResult) = res.augmentation_map
@inline coaugmentation(res::InjectiveHullResult) = res.map
@inline coaugmentation(res::DownsetResolutionResult) = res.coaugmentation_map
@inline presentation_module(H::FiniteFringe.FringeModule) = H
@inline presentation_map(H::FiniteFringe.FringeModule) = FiniteFringe.fringe_coefficients(H)

@inline function _indicator_resolution_describe(res::ProjectiveCoverResult)
    M = cover_module(res)
    counts = (_generator_total(res.generators),)
    return (
        kind=:projective_cover,
        side=:projective,
        field=M.field,
        poset_kind=Symbol(nameof(typeof(M.Q))),
        nvertices=nvertices(M.Q),
        generator_counts=counts,
        resolution_length=0,
        generator_storage=_generator_storage(res.generators),
        lazy_generators=res.generators isa ProjectiveGenerators,
        packed_generators=res.generators isa _ProjectiveGeneratorPlan,
        minimal_requested=res.minimal_requested,
        checks_requested=res.checks_requested,
    )
end

@inline function _indicator_generator_describe(gens::ProjectiveGenerators)
    return (
        kind=:projective_generators,
        side=:projective,
        active_vertices=generator_vertices(gens),
        block_counts=generator_blocks(gens),
        total_generators=_generator_total(gens),
        storage_mode=_generator_storage(gens),
    )
end

@inline function _indicator_generator_describe(gens::InjectiveGenerators)
    return (
        kind=:injective_generators,
        side=:injective,
        active_vertices=generator_vertices(gens),
        block_counts=generator_blocks(gens),
        total_generators=_generator_total(gens),
        storage_mode=_generator_storage(gens),
    )
end

@inline function _indicator_resolution_describe(res::InjectiveHullResult)
    M = hull_module(res)
    counts = (_generator_total(res.generators),)
    return (
        kind=:injective_hull,
        side=:injective,
        field=M.field,
        poset_kind=Symbol(nameof(typeof(M.Q))),
        nvertices=nvertices(M.Q),
        generator_counts=counts,
        resolution_length=0,
        generator_storage=_generator_storage(res.generators),
        lazy_generators=res.generators isa InjectiveGenerators,
        packed_generators=res.generators isa _InjectiveGeneratorPlan,
        minimal_requested=res.minimal_requested,
        checks_requested=res.checks_requested,
    )
end

@inline function _indicator_resolution_describe(res::UpsetResolutionResult)
    M = res.source_module
    return (
        kind=:upset_resolution,
        side=:upset,
        field=M.field,
        poset_kind=Symbol(nameof(typeof(M.Q))),
        nvertices=nvertices(M.Q),
        generator_counts=Tuple(length(F.U0) for F in res.presentations),
        resolution_length=length(res.differentials),
        generator_storage=:packed_plan,
        lazy_generators=true,
        packed_generators=true,
        minimal_requested=res.minimal_requested,
        checks_requested=res.checks_requested,
    )
end

@inline function _indicator_resolution_describe(res::DownsetResolutionResult)
    M = res.source_module
    return (
        kind=:downset_resolution,
        side=:downset,
        field=M.field,
        poset_kind=Symbol(nameof(typeof(M.Q))),
        nvertices=nvertices(M.Q),
        generator_counts=Tuple(length(E.D0) for E in res.presentations),
        resolution_length=length(res.differentials),
        generator_storage=:packed_plan,
        lazy_generators=true,
        packed_generators=true,
        minimal_requested=res.minimal_requested,
        checks_requested=res.checks_requested,
    )
end

@inline function _indicator_resolution_describe(res::IndicatorResolutionsResult)
    return (
        kind=:indicator_resolutions,
        projective=_indicator_resolution_describe(res.upset),
        injective=_indicator_resolution_describe(res.downset),
    )
end

@inline function _fringe_presentation_describe(H::FiniteFringe.FringeModule)
    return (
        kind=:fringe_presentation,
        field=FiniteFringe.field(H),
        poset_kind=Symbol(nameof(typeof(H.P))),
        nvertices=nvertices(H.P),
        ngenerators=FiniteFringe.ngenerators(H),
        nrelations=FiniteFringe.nrelations(H),
        matrix_size=size(FiniteFringe.fringe_coefficients(H)),
    )
end

@inline cover_summary(res::ProjectiveCoverResult) = _indicator_resolution_describe(res)
@inline hull_summary(res::InjectiveHullResult) = _indicator_resolution_describe(res)
@inline resolution_summary(res::UpsetResolutionResult) = _indicator_resolution_describe(res)
@inline resolution_summary(res::DownsetResolutionResult) = _indicator_resolution_describe(res)
@inline resolution_summary(res::IndicatorResolutionsResult) = _indicator_resolution_describe(res)
@inline presentation_summary(H::FiniteFringe.FringeModule) = _fringe_presentation_describe(H)
@inline resolution_length(res::ProjectiveCoverResult) = 0
@inline resolution_length(res::InjectiveHullResult) = 0
@inline resolution_length(res::UpsetResolutionResult) = length(res.differentials)
@inline resolution_length(res::DownsetResolutionResult) = length(res.differentials)
@inline resolution_length(res::IndicatorResolutionsResult) = (; projective=resolution_length(res.upset), injective=resolution_length(res.downset))
@inline generator_count_by_degree(res::ProjectiveCoverResult) = (_generator_total(res.generators),)
@inline generator_count_by_degree(res::InjectiveHullResult) = (_generator_total(res.generators),)
@inline generator_count_by_degree(res::UpsetResolutionResult) = Tuple(length(F.U0) for F in res.presentations)
@inline generator_count_by_degree(res::DownsetResolutionResult) = Tuple(length(E.D0) for E in res.presentations)
@inline generator_count_by_degree(res::IndicatorResolutionsResult) = (; projective=generator_count_by_degree(res.upset), injective=generator_count_by_degree(res.downset))
@inline generator_count(res::ProjectiveCoverResult) = only(generator_count_by_degree(res))
@inline generator_count(res::InjectiveHullResult) = only(generator_count_by_degree(res))
@inline generator_count(res::UpsetResolutionResult) = sum(generator_count_by_degree(res))
@inline generator_count(res::DownsetResolutionResult) = sum(generator_count_by_degree(res))
@inline generator_count(res::IndicatorResolutionsResult) = (; projective=generator_count(res.upset), injective=generator_count(res.downset))

@noinline function _indicator_resolution_invalid_output(fname::Symbol, output)
    Base.throw(ArgumentError("$(fname): output must be :full or :summary, got $(repr(output))"))
end

@inline function _indicator_resolution_with_output(
    fname::Symbol,
    result,
    summary_fn,
    output::Symbol,
)
    output === :full && return result
    output === :summary && return summary_fn(result)
    return _indicator_resolution_invalid_output(fname, output)
end

function Base.show(io::IO, gens::ProjectiveGenerators)
    print(io, "ProjectiveGenerators(active_vertices=", length(generator_vertices(gens)),
          ", total_generators=", _generator_total(gens), ")")
end

function Base.show(io::IO, ::MIME"text/plain", gens::ProjectiveGenerators)
    print(io, "ProjectiveGenerators",
          "\n  active_birth_vertices: ", repr(generator_vertices(gens)),
          "\n  block_counts: ", repr(generator_blocks(gens)),
          "\n  total_generators: ", _generator_total(gens))
end

function Base.show(io::IO, gens::InjectiveGenerators)
    print(io, "InjectiveGenerators(active_vertices=", length(generator_vertices(gens)),
          ", total_generators=", _generator_total(gens), ")")
end

function Base.show(io::IO, ::MIME"text/plain", gens::InjectiveGenerators)
    print(io, "InjectiveGenerators",
          "\n  active_socle_vertices: ", repr(generator_vertices(gens)),
          "\n  multiplicities: ", repr(generator_blocks(gens)),
          "\n  total_generators: ", _generator_total(gens))
end

function Base.show(io::IO, res::ProjectiveCoverResult)
    d = _indicator_resolution_describe(res)
    print(io, "ProjectiveCoverResult(field=", _indicator_resolution_field_label(d.field),
          ", nvertices=", d.nvertices,
          ", generator_count=", only(d.generator_counts),
          ", generator_storage=", d.generator_storage, ")")
end

function Base.show(io::IO, ::MIME"text/plain", res::ProjectiveCoverResult)
    d = _indicator_resolution_describe(res)
    print(io, "ProjectiveCoverResult",
          "\n  side: ", d.side,
          "\n  field: ", _indicator_resolution_field_label(d.field),
          "\n  poset_kind: ", d.poset_kind,
          "\n  nvertices: ", d.nvertices,
          "\n  generator_counts: ", repr(d.generator_counts),
          "\n  resolution_length: ", d.resolution_length,
          "\n  lazy_generators: ", d.lazy_generators,
          "\n  packed_generators: ", d.packed_generators,
          "\n  minimal_requested: ", d.minimal_requested,
          "\n  checks_requested: ", d.checks_requested)
end

function Base.show(io::IO, res::InjectiveHullResult)
    d = _indicator_resolution_describe(res)
    print(io, "InjectiveHullResult(field=", _indicator_resolution_field_label(d.field),
          ", nvertices=", d.nvertices,
          ", generator_count=", only(d.generator_counts),
          ", generator_storage=", d.generator_storage, ")")
end

function Base.show(io::IO, ::MIME"text/plain", res::InjectiveHullResult)
    d = _indicator_resolution_describe(res)
    print(io, "InjectiveHullResult",
          "\n  side: ", d.side,
          "\n  field: ", _indicator_resolution_field_label(d.field),
          "\n  poset_kind: ", d.poset_kind,
          "\n  nvertices: ", d.nvertices,
          "\n  generator_counts: ", repr(d.generator_counts),
          "\n  resolution_length: ", d.resolution_length,
          "\n  lazy_generators: ", d.lazy_generators,
          "\n  packed_generators: ", d.packed_generators,
          "\n  minimal_requested: ", d.minimal_requested,
          "\n  checks_requested: ", d.checks_requested)
end

function Base.show(io::IO, res::UpsetResolutionResult)
    d = _indicator_resolution_describe(res)
    print(io, "UpsetResolutionResult(length=", d.resolution_length,
          ", field=", _indicator_resolution_field_label(d.field),
          ", generator_counts=", repr(d.generator_counts), ")")
end

function Base.show(io::IO, ::MIME"text/plain", res::UpsetResolutionResult)
    d = _indicator_resolution_describe(res)
    print(io, "UpsetResolutionResult",
          "\n  side: ", d.side,
          "\n  field: ", _indicator_resolution_field_label(d.field),
          "\n  poset_kind: ", d.poset_kind,
          "\n  nvertices: ", d.nvertices,
          "\n  generator_counts: ", repr(d.generator_counts),
          "\n  resolution_length: ", d.resolution_length,
          "\n  lazy_generators: ", d.lazy_generators,
          "\n  packed_generators: ", d.packed_generators,
          "\n  minimal_requested: ", d.minimal_requested,
          "\n  checks_requested: ", d.checks_requested)
end

function Base.show(io::IO, res::DownsetResolutionResult)
    d = _indicator_resolution_describe(res)
    print(io, "DownsetResolutionResult(length=", d.resolution_length,
          ", field=", _indicator_resolution_field_label(d.field),
          ", generator_counts=", repr(d.generator_counts), ")")
end

function Base.show(io::IO, ::MIME"text/plain", res::DownsetResolutionResult)
    d = _indicator_resolution_describe(res)
    print(io, "DownsetResolutionResult",
          "\n  side: ", d.side,
          "\n  field: ", _indicator_resolution_field_label(d.field),
          "\n  poset_kind: ", d.poset_kind,
          "\n  nvertices: ", d.nvertices,
          "\n  generator_counts: ", repr(d.generator_counts),
          "\n  resolution_length: ", d.resolution_length,
          "\n  lazy_generators: ", d.lazy_generators,
          "\n  packed_generators: ", d.packed_generators,
          "\n  minimal_requested: ", d.minimal_requested,
          "\n  checks_requested: ", d.checks_requested)
end

function Base.show(io::IO, res::IndicatorResolutionsResult)
    d = _indicator_resolution_describe(res)
    print(io, "IndicatorResolutionsResult(projective_length=", d.projective.resolution_length,
          ", injective_length=", d.injective.resolution_length, ")")
end

function Base.show(io::IO, ::MIME"text/plain", res::IndicatorResolutionsResult)
    d = _indicator_resolution_describe(res)
    print(io, "IndicatorResolutionsResult",
          "\n  projective: ", repr(d.projective),
          "\n  injective: ", repr(d.injective))
end

function Base.show(io::IO, summary::IndicatorResolutionValidationSummary)
    report = summary.report
    kind = get(report, :kind, :indicator_resolution_validation)
    valid = get(report, :valid, false)
    issues = get(report, :issues, String[])
    print(io, "IndicatorResolutionValidationSummary(kind=", kind,
          ", valid=", valid,
          ", issues=", length(issues), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::IndicatorResolutionValidationSummary)
    report = summary.report
    print(io, "IndicatorResolutionValidationSummary",
          "\n  kind: ", get(report, :kind, :indicator_resolution_validation),
          "\n  valid: ", get(report, :valid, false))
    for (key, value) in pairs(report)
        key === :kind && continue
        key === :valid && continue
        key === :issues && continue
        print(io, "\n  ", key, ": ", repr(value))
    end
    issues = get(report, :issues, String[])
    if isempty(issues)
        print(io, "\n  issues: none")
    else
        print(io, "\n  issues:")
        for issue in issues
            print(io, "\n    - ", issue)
        end
    end
end

@inline function _indicator_resolution_report(kind::Symbol, valid::Bool; kwargs...)
    return (; kind, valid, kwargs...)
end

function _throw_invalid_indicator_resolution(kind::Symbol, issues::Vector{String})
    Base.throw(ArgumentError(string(kind, ": ", isempty(issues) ? "invalid indicator-resolution object." : join(issues, " "))))
end

function check_projective_cover(res::ProjectiveCoverResult; throw::Bool=false)
    issues = String[]
    F0 = cover_module(res)
    pi0 = cover_map(res)
    modrep = check_module(F0)
    modrep.valid || append!(issues, "cover module: " .* modrep.issues)
    morrep = check_morphism(pi0)
    morrep.valid || append!(issues, "cover map: " .* morrep.issues)
    pi0.dom === F0 || push!(issues, "cover map domain must equal the returned cover module.")
    @inbounds for q in 1:nvertices(F0.Q)
        FieldLinAlg.rank(F0.field, pi0.comps[q]) == pi0.cod.dims[q] ||
            push!(issues, "component at vertex $q is not surjective onto the source stalk.")
    end
    length(generator_blocks(res.generators)) == nvertices(F0.Q) ||
        push!(issues, "generator blocks must be indexed by ambient vertices.")
    report = _indicator_resolution_report(
        :projective_cover,
        isempty(issues);
        side=:projective,
        field=F0.field,
        nvertices=nvertices(F0.Q),
        generator_storage=_generator_storage(res.generators),
        issues=issues,
    )
    throw && !report.valid && _throw_invalid_indicator_resolution(:check_projective_cover, issues)
    return report
end

function check_projective_cover(F0::PModule, pi0::PMorphism, gens; throw::Bool=false)
    return check_projective_cover(ProjectiveCoverResult(F0, pi0, gens, false, false); throw=throw)
end

function check_injective_hull(res::InjectiveHullResult; throw::Bool=false)
    issues = String[]
    E0 = hull_module(res)
    iota = hull_map(res)
    modrep = check_module(E0)
    modrep.valid || append!(issues, "hull module: " .* modrep.issues)
    morrep = check_morphism(iota)
    morrep.valid || append!(issues, "hull map: " .* morrep.issues)
    iota.cod === E0 || push!(issues, "hull map codomain must equal the returned hull module.")
    @inbounds for q in 1:nvertices(E0.Q)
        FieldLinAlg.rank(E0.field, iota.comps[q]) == iota.dom.dims[q] ||
            push!(issues, "component at vertex $q is not injective on the source stalk.")
    end
    length(generator_blocks(res.generators)) == nvertices(E0.Q) ||
        push!(issues, "generator blocks must be indexed by ambient vertices.")
    report = _indicator_resolution_report(
        :injective_hull,
        isempty(issues);
        side=:injective,
        field=E0.field,
        nvertices=nvertices(E0.Q),
        generator_storage=_generator_storage(res.generators),
        issues=issues,
    )
    throw && !report.valid && _throw_invalid_indicator_resolution(:check_injective_hull, issues)
    return report
end

function check_injective_hull(E0::PModule, iota::PMorphism, gens; throw::Bool=false)
    return check_injective_hull(InjectiveHullResult(E0, iota, gens, false, false); throw=throw)
end

function check_resolution(res::UpsetResolutionResult; throw::Bool=false)
    issues = String[]
    for (a, F) in enumerate(res.presentations)
        rep = check_upset_presentation(F)
        rep.valid || append!(issues, "degree $a presentation: " .* rep.issues)
    end
    try
        verify_upset_resolution(res.presentations, res.differentials)
    catch err
        push!(issues, sprint(showerror, err))
    end
    report = _indicator_resolution_report(
        :upset_resolution,
        isempty(issues);
        side=:upset,
        field=res.source_module.field,
        nvertices=nvertices(res.source_module.Q),
        resolution_length=length(res.differentials),
        generator_counts=Tuple(length(F.U0) for F in res.presentations),
        issues=issues,
    )
    throw && !report.valid && _throw_invalid_indicator_resolution(:check_resolution, issues)
    return report
end

function check_resolution(res::DownsetResolutionResult; throw::Bool=false)
    issues = String[]
    for (b, E) in enumerate(res.presentations)
        rep = check_downset_copresentation(E)
        rep.valid || append!(issues, "degree $b copresentation: " .* rep.issues)
    end
    try
        verify_downset_resolution(res.presentations, res.differentials)
    catch err
        push!(issues, sprint(showerror, err))
    end
    report = _indicator_resolution_report(
        :downset_resolution,
        isempty(issues);
        side=:downset,
        field=res.source_module.field,
        nvertices=nvertices(res.source_module.Q),
        resolution_length=length(res.differentials),
        generator_counts=Tuple(length(E.D0) for E in res.presentations),
        issues=issues,
    )
    throw && !report.valid && _throw_invalid_indicator_resolution(:check_resolution, issues)
    return report
end

function check_resolution(res::IndicatorResolutionsResult; throw::Bool=false)
    up = check_resolution(res.upset)
    down = check_resolution(res.downset)
    issues = String[]
    up.valid || append!(issues, "projective side: " .* up.issues)
    down.valid || append!(issues, "injective side: " .* down.issues)
    report = _indicator_resolution_report(
        :indicator_resolutions,
        isempty(issues);
        projective_valid=up.valid,
        injective_valid=down.valid,
        issues=issues,
    )
    throw && !report.valid && _throw_invalid_indicator_resolution(:check_resolution, issues)
    return report
end

function check_fringe_presentation(H::FiniteFringe.FringeModule; source::Union{Nothing,PModule}=nothing, throw::Bool=false)
    issues = String[]
    rep = FiniteFringe.check_fringe_module(H)
    rep.valid || append!(issues, rep.issues)
    if source !== nothing
        M = pmodule_from_fringe(H)
        M.dims == source.dims || push!(issues, "converted fringe presentation must match source module stalk dimensions.")
        M.edge_maps == source.edge_maps || push!(issues, "converted fringe presentation must match source module structure maps.")
    end
    report = _indicator_resolution_report(
        :fringe_presentation,
        isempty(issues);
        field=FiniteFringe.field(H),
        nvertices=nvertices(H.P),
        ngenerators=FiniteFringe.ngenerators(H),
        nrelations=FiniteFringe.nrelations(H),
        has_source_module=source !== nothing,
        issues=issues,
    )
    throw && !report.valid && _throw_invalid_indicator_resolution(:check_fringe_presentation, issues)
    return report
end

function check_fringe_presentation(M::PModule; throw::Bool=false)
    return check_fringe_presentation(fringe_presentation(M); source=M, throw=throw)
end

mutable struct _DownsetRhoPattern
    counts_prev::Vector{Int}
    counts_next::Vector{Int}
    total_prev::Int
    total_next::Int
    theta_vertices::Vector{Int}
    lambda_vertices::Vector{Int}
    row0s::Vector{Int}
    col0s::Vector{Int}
    nrowss::Vector{Int}
    ncolss::Vector{Int}
    theta_gid0s::Vector{Int}
    lambda_gid0s::Vector{Int}
    value_ptr::Vector{Int}
    I::Vector{Int}
    J::Vector{Int}
end

@inline function _same_int_vector(a::Vector{Int}, b::Vector{Int})
    length(a) == length(b) || return false
    @inbounds for i in eachindex(a)
        a[i] == b[i] || return false
    end
    return true
end

@inline function _subsequence_identity_sparse(
    ::Type{K},
    tgt_ids::Vector{Int},
    src_ids::Vector{Int},
) where {K}
    nt = length(tgt_ids)
    ns = length(src_ids)
    I = Int[]
    J = Int[]
    V = K[]
    sizehint!(I, min(nt, ns))
    sizehint!(J, min(nt, ns))
    sizehint!(V, min(nt, ns))

    i = 1
    j = 1
    oneK = one(K)
    @inbounds while i <= nt && j <= ns
        ti = tgt_ids[i]
        sj = src_ids[j]
        if ti == sj
            push!(I, i)
            push!(J, j)
            push!(V, oneK)
            i += 1
            j += 1
        elseif ti < sj
            i += 1
        else
            error("_subsequence_identity_sparse: source generator id missing from target active set")
        end
    end
    j > ns || error("_subsequence_identity_sparse: source generator id missing from target active set")
    return sparse(I, J, V, nt, ns)
end

@inline function _projection_identity_sparse(
    ::Type{K},
    tgt_ids::Vector{Int},
    src_ids::Vector{Int},
) where {K}
    nt = length(tgt_ids)
    ns = length(src_ids)
    I = Int[]
    J = Int[]
    V = K[]
    sizehint!(I, min(nt, ns))
    sizehint!(J, min(nt, ns))
    sizehint!(V, min(nt, ns))

    i = 1
    j = 1
    oneK = one(K)
    @inbounds while i <= nt && j <= ns
        ti = tgt_ids[i]
        sj = src_ids[j]
        if ti == sj
            push!(I, i)
            push!(J, j)
            push!(V, oneK)
            i += 1
            j += 1
        elseif sj < ti
            j += 1
        else
            error("_projection_identity_sparse: target generator id missing from source active set")
        end
    end
    i > nt || error("_projection_identity_sparse: target generator id missing from source active set")
    return sparse(I, J, V, nt, ns)
end

@inline function _projection_identity_sparse(
    ::Type{K},
    packed::_PackedIntLists,
    tgt_idx::Int,
    src_idx::Int,
) where {K}
    nt = _packed_length(packed, tgt_idx)
    ns = _packed_length(packed, src_idx)
    i = _packed_firstindex(packed, tgt_idx)
    j = _packed_firstindex(packed, src_idx)
    ihi = _packed_lastindex(packed, tgt_idx)
    jhi = _packed_lastindex(packed, src_idx)
    data = packed.data
    rowval = Int[]
    sizehint!(rowval, min(nt, ns))
    colptr = Vector{Int}(undef, ns + 1)
    colptr[1] = 1
    row = 1
    col = 1
    @inbounds while i <= ihi && j <= jhi
        ti = data[i]
        sj = data[j]
        if ti == sj
            push!(rowval, row)
            colptr[col + 1] = length(rowval) + 1
            row += 1
            col += 1
            i += 1
            j += 1
        elseif sj < ti
            colptr[col + 1] = length(rowval) + 1
            col += 1
            j += 1
        else
            error("_projection_identity_sparse: target generator id missing from source active set")
        end
    end
    row > nt || error("_projection_identity_sparse: target generator id missing from source active set")
    @inbounds while col <= ns
        colptr[col + 1] = length(rowval) + 1
        col += 1
    end
    return SparseMatrixCSC{K,Int}(nt, ns, colptr, rowval, fill(one(K), length(rowval)))
end

@inline function _subsequence_identity_sparse_blocks(
    ::Type{K},
    tgt_sources::Vector{Int},
    src_sources::Vector{Int},
    mult::Vector{Int},
    nt::Int,
    ns::Int,
) where {K}
    I = Vector{Int}(undef, ns)
    J = Vector{Int}(undef, ns)
    V = Vector{K}(undef, ns)
    row0 = 1
    col0 = 1
    pos = 1
    i = 1
    j = 1
    oneK = one(K)
    @inbounds while i <= length(tgt_sources) && j <= length(src_sources)
        ti = tgt_sources[i]
        sj = src_sources[j]
        if ti == sj
            m = mult[ti]
            for k in 0:(m - 1)
                I[pos] = row0 + k
                J[pos] = col0 + k
                V[pos] = oneK
                pos += 1
            end
            row0 += m
            col0 += m
            i += 1
            j += 1
        elseif ti < sj
            row0 += mult[ti]
            i += 1
        else
            error("_subsequence_identity_sparse_blocks: source active set is not a subsequence of target active set")
        end
    end
    j > length(src_sources) ||
        error("_subsequence_identity_sparse_blocks: source active set is not a subsequence of target active set")
    return sparse(I, J, V, nt, ns)
end

@inline function _subsequence_identity_sparse_blocks(
    ::Type{K},
    packed::_PackedIntLists,
    tgt_idx::Int,
    src_idx::Int,
    mult::Vector{Int},
    nt::Int,
    ns::Int,
) where {K}
    i = _packed_firstindex(packed, tgt_idx)
    j = _packed_firstindex(packed, src_idx)
    ihi = _packed_lastindex(packed, tgt_idx)
    jhi = _packed_lastindex(packed, src_idx)
    data = packed.data
    rowval = Vector{Int}(undef, ns)
    colptr = Vector{Int}(undef, ns + 1)
    colptr[1] = 1
    row0 = 1
    col = 1
    pos = 1
    @inbounds while i <= ihi && j <= jhi
        ti = data[i]
        sj = data[j]
        if ti == sj
            m = mult[ti]
            for k in 0:(m - 1)
                rowval[pos] = row0 + k
                colptr[col + 1] = pos + 1
                pos += 1
                col += 1
            end
            row0 += m
            i += 1
            j += 1
        elseif ti < sj
            row0 += mult[ti]
            i += 1
        else
            error("_subsequence_identity_sparse_blocks: source active set is not a subsequence of target active set")
        end
    end
    j > jhi || error("_subsequence_identity_sparse_blocks: source active set is not a subsequence of target active set")
    @inbounds while col <= ns
        colptr[col + 1] = pos
        col += 1
    end
    return SparseMatrixCSC{K,Int}(nt, ns, colptr, rowval, fill(one(K), ns))
end

mutable struct _ResolutionWorkspace{K}
    I_chunks::Vector{Vector{Int}}
    J_chunks::Vector{Vector{Int}}
    V_chunks::Vector{Vector{K}}
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{K}
    self_start::Vector{Int}
    # SparseArrays.sparse! scratch storage.
    klasttouch::Vector{Int}
    csrrowptr::Vector{Int}
    csrcolval::Vector{Int}
    csrnzval::Vector{K}
    csccolptr_slots::Vector{Vector{Int}}
    cscrowval_slots::Vector{Vector{Int}}
    cscnzval_slots::Vector{Vector{K}}
    csc_slot::Int
    # Batch map_leq scratch storage.
    pairs_buf::Vector{Tuple{Int,Int}}
    missing_pairs_buf::Vector{Tuple{Int,Int}}
    fetched_buf::Vector{AbstractMatrix{K}}
    map_batch_cache::Dict{UInt64,MapLeqQueryBatch}
    kernel_vertex_cache::Vector{_IncrementalVertexCacheEntry{K}}
end

@inline function _new_resolution_workspace(::Type{K}, _n::Int) where {K}
    nt = max(1, Threads.maxthreadid())
    return _ResolutionWorkspace{K}(
        [Int[] for _ in 1:nt],
        [Int[] for _ in 1:nt],
        [K[] for _ in 1:nt],
        Int[],
        Int[],
        K[],
        zeros(Int, _n),
        Int[],
        Int[],
        Int[],
        K[],
        [Int[], Int[]],
        [Int[], Int[]],
        [K[], K[]],
        1,
        Tuple{Int,Int}[],
        Tuple{Int,Int}[],
        AbstractMatrix{K}[],
        Dict{UInt64,MapLeqQueryBatch}(),
        _IncrementalVertexCacheEntry{K}[],
    )
end

@inline function _workspace_prepare!(ws::_ResolutionWorkspace{K}, _n::Int) where {K}
    if length(ws.self_start) != _n
        resize!(ws.self_start, _n)
    end
    empty!(ws.I)
    empty!(ws.J)
    empty!(ws.V)
    for t in 1:length(ws.I_chunks)
        empty!(ws.I_chunks[t])
        empty!(ws.J_chunks[t])
        empty!(ws.V_chunks[t])
    end
    empty!(ws.pairs_buf)
    empty!(ws.missing_pairs_buf)
    empty!(ws.fetched_buf)
    fill!(ws.kernel_vertex_cache, nothing)
    ws.csc_slot = 1
    return ws
end

mutable struct _UpsetPrefixState{K}
    gens_by_a::Vector{_ProjectiveGeneratorPlan}
    dF::Vector{SparseMatrixCSC{K,Int}}
    pi0::PMorphism{K}
    curr_pi::PMorphism{K}
    curr_gens::_ProjectiveGeneratorPlan
    done::Bool
end

mutable struct _IndicatorPrefixFamily{K}
    upset_locks::Vector{Base.ReentrantLock}
    upset_shards::Vector{IdDict{PModule{K},_UpsetPrefixState{K}}}
end

@inline function _new_indicator_prefix_family(::Type{K}) where {K}
    nshards = INDICATOR_PREFIX_CACHE_SHARDS[]
    return _IndicatorPrefixFamily{K}(
        [Base.ReentrantLock() for _ in 1:nshards],
        [IdDict{PModule{K},_UpsetPrefixState{K}}() for _ in 1:nshards],
    )
end

function _indicator_prefix_family(::Type{K}) where {K}
    Base.lock(_INDICATOR_PREFIX_FAMILY_LOCK)
    try
        fam = get(_INDICATOR_PREFIX_FAMILIES, K, nothing)
        if fam === nothing
            fam = _new_indicator_prefix_family(K)
            _INDICATOR_PREFIX_FAMILIES[K] = fam
        end
        return fam::_IndicatorPrefixFamily{K}
    finally
        Base.unlock(_INDICATOR_PREFIX_FAMILY_LOCK)
    end
end

function _clear_indicator_prefix_caches!()
    Base.lock(_INDICATOR_PREFIX_FAMILY_LOCK)
    try
        for fam_any in values(_INDICATOR_PREFIX_FAMILIES)
            fam = fam_any
            for i in eachindex(fam.upset_shards)
                Base.lock(fam.upset_locks[i])
                try
                    empty!(fam.upset_shards[i])
                finally
                    Base.unlock(fam.upset_locks[i])
                end
            end
        end
    finally
        Base.unlock(_INDICATOR_PREFIX_FAMILY_LOCK)
    end
    for i in eachindex(_INDICATOR_BIRTH_PLAN_SHARDS)
        Base.lock(_INDICATOR_BIRTH_PLAN_LOCKS[i])
        try
            empty!(_INDICATOR_BIRTH_PLAN_SHARDS[i])
        finally
            Base.unlock(_INDICATOR_BIRTH_PLAN_LOCKS[i])
        end
    end
    return nothing
end

@inline function _upset_prefix_steps(M::PModule{K}) where {K}
    fam = _indicator_prefix_family(K)
    idx = _indicator_shard_index(M, length(fam.upset_shards))
    Base.lock(fam.upset_locks[idx])
    try
        st = get(fam.upset_shards[idx], M, nothing)
        st === nothing && return 0
        return length(st.dF)
    finally
        Base.unlock(fam.upset_locks[idx])
    end
end

@inline function _cached_birth_plans(P::AbstractPoset)
    idx = _indicator_shard_index(P, length(_INDICATOR_BIRTH_PLAN_SHARDS))
    Base.lock(_INDICATOR_BIRTH_PLAN_LOCKS[idx])
    try
        plans = get(_INDICATOR_BIRTH_PLAN_SHARDS[idx], P, nothing)
        if plans === nothing
            n = nvertices(P)
            upset = Vector{Vector{Int}}(undef, n)
            downset = Vector{Vector{Int}}(undef, n)
            @inbounds for i in 1:n
                downset[i] = collect(downset_indices(P, i))
                upset[i] = collect(upset_indices(P, i))
            end
            plans = _BirthPlans(upset, downset)
            _INDICATOR_BIRTH_PLAN_SHARDS[idx][P] = plans
        end
        return plans::_BirthPlans
    finally
        Base.unlock(_INDICATOR_BIRTH_PLAN_LOCKS[idx])
    end
end

@inline function _fill_birth_self_start_down!(
    starts::Vector{Int},
    P::AbstractPoset,
    counts::Vector{Int},
)
    n = nvertices(P)
    length(starts) == n || error("_fill_birth_self_start_down!: wrong length")
    @inbounds for i in 1:n
        pos = 1
        for p in downset_indices(P, i)
            p == i && break
            pos += counts[p]
        end
        starts[i] = pos
    end
    return starts
end

@inline function _fill_birth_self_start_up!(
    starts::Vector{Int},
    P::AbstractPoset,
    counts::Vector{Int},
)
    n = nvertices(P)
    length(starts) == n || error("_fill_birth_self_start_up!: wrong length")
    @inbounds for i in 1:n
        pos = 1
        for u in upset_indices(P, i)
            u == i && break
            pos += counts[u]
        end
        starts[i] = pos
    end
    return starts
end

@inline function _build_downset_rho_pattern(
    birth_plan::Vector{Vector{Int}},
    counts_prev::Vector{Int},
    gid_prev_starts::Vector{Int},
    counts_next::Vector{Int},
    gid_next_starts::Vector{Int},
    starts_next::Vector{Int},
    total_prev::Int,
    total_next::Int,
)
    n = length(counts_next)
    nblocks = 0
    total_values = 0
    @inbounds for utheta in 1:n
        counts_next[utheta] == 0 && continue
        for ulambda in birth_plan[utheta]
            counts_prev[ulambda] == 0 && continue
            nblocks += 1
            total_values += counts_next[utheta] * counts_prev[ulambda]
        end
    end

    theta_vertices = Vector{Int}(undef, nblocks)
    lambda_vertices = Vector{Int}(undef, nblocks)
    row0s = Vector{Int}(undef, nblocks)
    col0s = Vector{Int}(undef, nblocks)
    nrowss = Vector{Int}(undef, nblocks)
    ncolss = Vector{Int}(undef, nblocks)
    theta_gid0s = Vector{Int}(undef, nblocks)
    lambda_gid0s = Vector{Int}(undef, nblocks)
    value_ptr = Vector{Int}(undef, nblocks + 1)
    I = Vector{Int}(undef, total_values)
    J = Vector{Int}(undef, total_values)

    pos = 1
    value_pos = 1
    @inbounds for utheta in 1:n
        ctheta = counts_next[utheta]
        ctheta == 0 && continue
        row0 = starts_next[utheta]
        col0 = 1
        theta_gid0 = gid_next_starts[utheta]
        for ulambda in birth_plan[utheta]
            clambda = counts_prev[ulambda]
            if clambda != 0
                value_ptr[pos] = value_pos
                theta_vertices[pos] = utheta
                lambda_vertices[pos] = ulambda
                row0s[pos] = row0
                col0s[pos] = col0
                nrowss[pos] = ctheta
                ncolss[pos] = clambda
                theta_gid0s[pos] = theta_gid0
                lambda_gid0s[pos] = gid_prev_starts[ulambda]
                for rr in 0:(ctheta - 1)
                    for cc in 0:(clambda - 1)
                        I[value_pos] = theta_gid0 + rr
                        J[value_pos] = gid_prev_starts[ulambda] + cc
                        value_pos += 1
                    end
                end
                pos += 1
            end
            col0 += clambda
        end
    end
    value_ptr[end] = value_pos

    return _DownsetRhoPattern(
        copy(counts_prev),
        copy(counts_next),
        total_prev,
        total_next,
        theta_vertices,
        lambda_vertices,
        row0s,
        col0s,
        nrowss,
        ncolss,
        theta_gid0s,
        lambda_gid0s,
        value_ptr,
        I,
        J,
    )
end

@inline function _downset_rho_pattern_matches(
    pat::_DownsetRhoPattern,
    counts_prev::Vector{Int},
    counts_next::Vector{Int},
)
    return _same_int_vector(pat.counts_prev, counts_prev) &&
           _same_int_vector(pat.counts_next, counts_next)
end

@inline function _downset_rho_from_pattern!(
    ws::_ResolutionWorkspace{K},
    pat::_DownsetRhoPattern,
    j_comps::AbstractVector{<:AbstractMatrix{K}},
    q_comps::AbstractVector{<:AbstractMatrix{K}},
    threaded::Bool,
) where {K}
    if threaded && Threads.nthreads() > 1
        for tid in 1:length(ws.I_chunks)
            empty!(ws.I_chunks[tid])
            empty!(ws.J_chunks[tid])
            empty!(ws.V_chunks[tid])
        end
        Threads.@threads for b in eachindex(pat.theta_vertices)
            tid = _thread_local_index(ws.I_chunks)
            I_t = ws.I_chunks[tid]
            J_t = ws.J_chunks[tid]
            V_t = ws.V_chunks[tid]
            theta = pat.theta_vertices[b]
            _accumulate_product_entries_downset!(
                I_t, J_t, V_t,
                j_comps[theta], q_comps[theta],
                pat.row0s[b], pat.col0s[b], pat.nrowss[b], pat.ncolss[b],
                pat.theta_gid0s[b], pat.lambda_gid0s[b],
            )
        end
        empty!(ws.I)
        empty!(ws.J)
        empty!(ws.V)
        for tid in 1:length(ws.I_chunks)
            append!(ws.I, ws.I_chunks[tid])
            append!(ws.J, ws.J_chunks[tid])
            append!(ws.V, ws.V_chunks[tid])
        end
    else
        empty!(ws.I)
        empty!(ws.J)
        empty!(ws.V)
        @inbounds for b in eachindex(pat.theta_vertices)
            theta = pat.theta_vertices[b]
            _accumulate_product_entries_downset!(
                ws.I, ws.J, ws.V,
                j_comps[theta], q_comps[theta],
                pat.row0s[b], pat.col0s[b], pat.nrowss[b], pat.ncolss[b],
                pat.theta_gid0s[b], pat.lambda_gid0s[b],
            )
        end
    end
    return _sparse_from_workspace!(ws, pat.total_next, pat.total_prev)
end

@inline function _downset_rho_from_value_pattern!(
    ws::_ResolutionWorkspace{K},
    pat::_DownsetRhoPattern,
    j_comps::AbstractVector{<:AbstractMatrix{K}},
    q_comps::AbstractVector{<:AbstractMatrix{K}},
    threaded::Bool,
) where {K}
    resize!(ws.V, length(pat.I))
    if threaded && Threads.nthreads() > 1
        Threads.@threads for b in eachindex(pat.theta_vertices)
            theta = pat.theta_vertices[b]
            A = j_comps[theta]
            B = q_comps[theta]
            row0 = pat.row0s[b]
            col0 = pat.col0s[b]
            nrows = pat.nrowss[b]
            ncols = pat.ncolss[b]
            pos = pat.value_ptr[b]
            kdim = size(A, 2)
            @inbounds for rr in 0:(nrows - 1)
                row = row0 + rr
                for cc in 0:(ncols - 1)
                    col = col0 + cc
                    s = zero(K)
                    for k in 1:kdim
                        s += A[row, k] * B[k, col]
                    end
                    ws.V[pos] = s
                    pos += 1
                end
            end
        end
    else
        @inbounds for b in eachindex(pat.theta_vertices)
            theta = pat.theta_vertices[b]
            A = j_comps[theta]
            B = q_comps[theta]
            row0 = pat.row0s[b]
            col0 = pat.col0s[b]
            nrows = pat.nrowss[b]
            ncols = pat.ncolss[b]
            pos = pat.value_ptr[b]
            kdim = size(A, 2)
            for rr in 0:(nrows - 1)
                row = row0 + rr
                for cc in 0:(ncols - 1)
                    col = col0 + cc
                    s = zero(K)
                    for k in 1:kdim
                        s += A[row, k] * B[k, col]
                    end
                    ws.V[pos] = s
                    pos += 1
                end
            end
        end
    end
    Rh = if _INDICATOR_DOWNSET_RHO_SPARSE_WORKSPACE[]
        resize!(ws.I, length(pat.I))
        resize!(ws.J, length(pat.J))
        copyto!(ws.I, pat.I)
        copyto!(ws.J, pat.J)
        _sparse_from_workspace!(ws, pat.total_next, pat.total_prev)
    else
        sparse(pat.I, pat.J, ws.V, pat.total_next, pat.total_prev)
    end
    dropzeros!(Rh)
    return Rh
end

@inline function _upset_delta_from_active_sources!(
    ws::_ResolutionWorkspace{K},
    next_birth_vertices,
    prev_active_sources::_PackedIntLists,
    iota_comps::AbstractVector{<:AbstractMatrix{K}},
    pinext_comps::AbstractVector{<:AbstractMatrix{K}},
    counts_prev::Vector{Int},
    gid_prev_starts::Vector{Int},
    counts_next::Vector{Int},
    gid_next_starts::Vector{Int},
    starts_next::Vector{Int},
    total_next::Int,
    total_prev::Int,
    threaded::Bool,
) where {K}
    if threaded && Threads.nthreads() > 1
        for tid in 1:length(ws.I_chunks)
            empty!(ws.I_chunks[tid])
            empty!(ws.J_chunks[tid])
            empty!(ws.V_chunks[tid])
        end
        Threads.@threads for idx in eachindex(next_birth_vertices)
            ptheta = next_birth_vertices[idx]
            tid = _thread_local_index(ws.I_chunks)
            I_t = ws.I_chunks[tid]
            J_t = ws.J_chunks[tid]
            V_t = ws.V_chunks[tid]
            _accumulate_product_entries_upset!(
                I_t,
                J_t,
                V_t,
                iota_comps[ptheta],
                pinext_comps[ptheta],
                prev_active_sources,
                ptheta,
                counts_prev,
                gid_prev_starts,
                starts_next[ptheta],
                counts_next[ptheta],
                gid_next_starts[ptheta],
            )
        end
        empty!(ws.I)
        empty!(ws.J)
        empty!(ws.V)
        for tid in 1:length(ws.I_chunks)
            append!(ws.I, ws.I_chunks[tid])
            append!(ws.J, ws.J_chunks[tid])
            append!(ws.V, ws.V_chunks[tid])
        end
    else
        empty!(ws.I)
        empty!(ws.J)
        empty!(ws.V)
        for ptheta in next_birth_vertices
            _accumulate_product_entries_upset!(
                ws.I,
                ws.J,
                ws.V,
                iota_comps[ptheta],
                pinext_comps[ptheta],
                prev_active_sources,
                ptheta,
                counts_prev,
                gid_prev_starts,
                starts_next[ptheta],
                counts_next[ptheta],
                gid_next_starts[ptheta],
            )
        end
    end
    return _sparse_from_workspace!(ws, total_next, total_prev)
end

@inline function _upset_delta_from_birth_plan!(
    ws::_ResolutionWorkspace{K},
    n::Int,
    birth_plan::Vector{Vector{Int}},
    iota_comps::AbstractVector{<:AbstractMatrix{K}},
    pinext_comps::AbstractVector{<:AbstractMatrix{K}},
    counts_prev::Vector{Int},
    gid_prev_starts::Vector{Int},
    counts_next::Vector{Int},
    gid_next_starts::Vector{Int},
    starts_next::Vector{Int},
    total_next::Int,
    total_prev::Int,
    threaded::Bool,
) where {K}
    if threaded && Threads.nthreads() > 1
        for tid in 1:length(ws.I_chunks)
            empty!(ws.I_chunks[tid])
            empty!(ws.J_chunks[tid])
            empty!(ws.V_chunks[tid])
        end
        Threads.@threads for ptheta in 1:n
            ctheta = counts_next[ptheta]
            ctheta == 0 && continue
            tid = _thread_local_index(ws.I_chunks)
            I_t = ws.I_chunks[tid]
            J_t = ws.J_chunks[tid]
            V_t = ws.V_chunks[tid]
            Ai = iota_comps[ptheta]
            Bi = pinext_comps[ptheta]
            theta_gid0 = gid_next_starts[ptheta]
            col0 = starts_next[ptheta]
            row0 = 1
            @inbounds for plambda in birth_plan[ptheta]
                clambda = counts_prev[plambda]
                clambda == 0 && continue
                lambda_gid0 = gid_prev_starts[plambda]
                _accumulate_product_entries_upset!(
                    I_t, J_t, V_t,
                    Ai, Bi,
                    row0, col0, clambda, ctheta,
                    theta_gid0, lambda_gid0,
                )
                row0 += clambda
            end
        end
        empty!(ws.I)
        empty!(ws.J)
        empty!(ws.V)
        for tid in 1:length(ws.I_chunks)
            append!(ws.I, ws.I_chunks[tid])
            append!(ws.J, ws.J_chunks[tid])
            append!(ws.V, ws.V_chunks[tid])
        end
    else
        empty!(ws.I)
        empty!(ws.J)
        empty!(ws.V)
        for ptheta in 1:n
            ctheta = counts_next[ptheta]
            ctheta == 0 && continue
            Ai = iota_comps[ptheta]
            Bi = pinext_comps[ptheta]
            theta_gid0 = gid_next_starts[ptheta]
            col0 = starts_next[ptheta]
            row0 = 1
            @inbounds for plambda in birth_plan[ptheta]
                clambda = counts_prev[plambda]
                clambda == 0 && continue
                lambda_gid0 = gid_prev_starts[plambda]
                _accumulate_product_entries_upset!(
                    ws.I, ws.J, ws.V,
                    Ai, Bi,
                    row0, col0, clambda, ctheta,
                    theta_gid0, lambda_gid0,
                )
                row0 += clambda
            end
        end
    end
    return _sparse_from_workspace!(ws, total_next, total_prev)
end

@inline function _choose_projective_generators(
    field::AbstractCoeffField,
    Img::AbstractMatrix{K},
    d::Int,
) where {K}
    r = size(Img, 2)
    beta = d - r
    beta <= 0 && return Int[]

    A = Matrix{K}(undef, d, r + d)
    if r > 0
        @views A[:, 1:r] .= Img
    end
    @views fill!(A[:, (r + 1):(r + d)], zero(K))
    @inbounds for j in 1:d
        A[j, r + j] = one(K)
    end

    _, pivs = FieldLinAlg.rref(field, A; pivots=true)
    out = Int[]
    sizehint!(out, beta)
    @inbounds for p in pivs
        if p > r
            push!(out, p - r)
            length(out) == beta && break
        end
    end
    return out
end

function _map_leq_cached_indicator(
    M::PModule{K},
    u::Int,
    v::Int,
    cc::CoverCache,
    memo::AbstractVector{Union{Nothing,Matrix{K}}}
)::Matrix{K} where {K}
    n = nvertices(M.Q)
    X = _indicator_memo_get(memo, n, u, v)
    X === nothing || return X
    Xraw = map_leq(M, u, v; cache=cc)
    Xmat = Xraw isa Matrix{K} ? Xraw : Matrix{K}(Xraw)
    return _indicator_memo_set!(memo, n, u, v, Xmat)
end

id_morphism(H::FiniteFringe.FringeModule{K}) where {K} =
    id_morphism(pmodule_from_fringe(H))

@inline _is_exact_field(field::AbstractCoeffField) = !(field isa RealField)

@inline _resolution_cache_shard_index(dicts) =
    min(length(dicts), max(1, Threads.threadid()))
@inline _thread_local_index(arr) =
    min(length(arr), max(1, Threads.threadid()))

@inline function _indicator_use_threads(
    requested::Bool,
    nverts::Int,
    total_dims::Int,
    expected_work::Int,
)
    requested || return false
    Threads.nthreads() > 1 || return false
    nverts >= INDICATOR_THREADS_MIN_VERTICES[] || return false
    total_dims >= INDICATOR_THREADS_MIN_TOTAL_DIMS[] || return false
    expected_work >= INDICATOR_THREADS_MIN_WORK[] || return false
    return true
end

@inline function _indicator_use_map_plans(
    nverts::Int,
    total_pairs::Int,
)
    nverts >= INDICATOR_MAP_PLAN_MIN_VERTICES[] || return false
    total_pairs >= INDICATOR_MAP_PLAN_MIN_PAIRS[] || return false
    return true
end

@inline function _indicator_cache_admit_locked!(
    cache::ResolutionCache,
    key::ResolutionKey3,
    val,
)::Bool
    # Explicit resolution caches have strict reuse semantics: once the caller
    # asks for caching, indicator-resolution payloads are admitted immediately.
    _ = cache
    _ = key
    _ = val
    return true
end

@inline function _indicator_primary_dict(cache::ResolutionCache, ::Type{R}) where {R}
    cache.indicator_primary_type === R || return nothing
    return cache.indicator_primary::Dict{ResolutionKey3,R}
end

@inline function _indicator_primary_shard(cache::ResolutionCache, ::Type{R}) where {R}
    cache.indicator_primary_type === R || return nothing
    shard = cache.indicator_primary_shards[_resolution_cache_shard_index(cache.indicator_primary_shards)]
    shard === nothing && return nothing
    return shard::Dict{ResolutionKey3,R}
end

@inline function _ensure_indicator_primary_locked!(cache::ResolutionCache, ::Type{R}) where {R}
    if cache.indicator_primary_type === nothing && isempty(cache.indicator)
        cache.indicator_primary_type = R
        cache.indicator_primary = Dict{ResolutionKey3,R}()
        for i in eachindex(cache.indicator_primary_shards)
            cache.indicator_primary_shards[i] = Dict{ResolutionKey3,R}()
        end
    end
    return _indicator_primary_dict(cache, R)
end

@inline function _resolution_cache_indicator_get(cache::ResolutionCache, key::ResolutionKey3, ::Type{R}) where {R}
    primary = _indicator_primary_dict(cache, R)
    if primary !== nothing
        if length(cache.indicator_primary_shards) == 1
            return get(primary, key, nothing)
        end
        shard = _indicator_primary_shard(cache, R)
        shard === nothing || begin
            v = get(shard, key, nothing)
            v === nothing || return v
        end
        Base.lock(cache.lock)
        try
            v = get(primary, key, nothing)
            if v !== nothing && shard !== nothing
                shard[key] = v
            end
            return v
        finally
            Base.unlock(cache.lock)
        end
    end

    # Single-thread fast path: avoid lock/shard indirection on misses.
    if length(cache.indicator_shards) == 1
        v = get(cache.indicator, key, nothing)
        return v === nothing ? nothing : (v.value::R)
    end

    shard = cache.indicator_shards[_resolution_cache_shard_index(cache.indicator_shards)]
    v = get(shard, key, nothing)
    v === nothing || return (v.value::R)
    Base.lock(cache.lock)
    try
        v = get(cache.indicator, key, nothing)
    finally
        Base.unlock(cache.lock)
    end
    v === nothing || begin
        vv = v.value::R
        shard[key] = v
        return vv
    end
    return nothing
end

@inline function _resolution_cache_indicator_store!(cache::ResolutionCache, key::ResolutionKey3, val::R) where {R}
    primary = _indicator_primary_dict(cache, R)
    if primary === nothing
        Base.lock(cache.lock)
        try
            primary = _ensure_indicator_primary_locked!(cache, R)
            if primary !== nothing
                extant = get(primary, key, nothing)
                extant === nothing || return extant
                primary[key] = val
                shard = cache.indicator_primary_shards[_resolution_cache_shard_index(cache.indicator_primary_shards)]::Dict{ResolutionKey3,R}
                shard[key] = val
                return val
            end
        finally
            Base.unlock(cache.lock)
        end
    else
        if length(cache.indicator_primary_shards) == 1
            extant = get(primary, key, nothing)
            extant === nothing || return extant
            _indicator_cache_admit_locked!(cache, key, val) || return val
            primary[key] = val
            return val
        end
        shard = _indicator_primary_shard(cache, R)
        shard === nothing || begin
            extant = get(shard, key, nothing)
            extant === nothing || return extant
        end
        Base.lock(cache.lock)
        try
            extant = get(primary, key, nothing)
            extant === nothing || return extant
            _indicator_cache_admit_locked!(cache, key, val) || return val
            primary[key] = val
            shard === nothing || (shard[key] = val)
            return val
        finally
            Base.unlock(cache.lock)
        end
    end

    # Single-thread fast path: lock-free get!/insert.
    if length(cache.indicator_shards) == 1
        extant = get(cache.indicator, key, nothing)
        extant === nothing || return (extant.value::R)
        _indicator_cache_admit_locked!(cache, key, val) || return val
        out = get!(cache.indicator, key) do
            IndicatorResolutionPayload(val)
        end
        return out.value::R
    end

    shard = cache.indicator_shards[_resolution_cache_shard_index(cache.indicator_shards)]
    existing = get(shard, key, nothing)
    existing === nothing || return (existing.value::R)

    Base.lock(cache.lock)
    out = try
        extant = get(cache.indicator, key, nothing)
        if extant !== nothing
            extant
        elseif !_indicator_cache_admit_locked!(cache, key, val)
            nothing
        else
            get!(cache.indicator, key) do
                IndicatorResolutionPayload(val)
            end
        end
    finally
        Base.unlock(cache.lock)
    end
    out === nothing && return val
    outR = out.value::R
    shard[key] = out
    return outR
end

function _is_zero_matrix(field::AbstractCoeffField, M)
    if field isa RealField
        tol = field.atol + field.rtol * opnorm(M, 1)
        return norm(M) <= tol
    end
    return nnz(M) == 0
end

function _rank_restricted_field(field::AbstractCoeffField, A, rows, cols)
    return FieldLinAlg.rank_restricted(field, A, rows, cols)
end

# ----------------------- from FringeModule to PModule ----------------------------

@inline function _active_summands_per_vertex(summands, n::Int)
    out = [Int[] for _ in 1:n]
    @inbounds for idx in eachindex(summands)
        mask = summands[idx].mask
        for q in 1:n
            mask[q] && push!(out[q], idx)
        end
    end
    return out
end

struct _FringeFiberImageDescriptor{K,F}
    rows::Vector{Int}
    basis::Matrix{K}
    rhs_rows::Vector{Int}
    factor::F
end

@inline function Base.convert(
    ::Type{_FringeFiberImageDescriptor{K, F}},
    d::_FringeFiberImageDescriptor{K, G},
) where {K, F, G}
    return _FringeFiberImageDescriptor{K, F}(d.rows, d.basis, d.rhs_rows, convert(F, d.factor))
end

@inline function _new_fringe_descriptor_vector(::QQField, ::Type{K}, n::Int) where {K}
    T = _FringeFiberImageDescriptor{K,Union{Nothing,FieldLinAlg.FullColumnFactor{K}}}
    return Vector{T}(undef, n)
end

@inline function _new_fringe_descriptor_vector(field::PrimeField, ::Type{K}, n::Int) where {K}
    if field.p == 2
        T = _FringeFiberImageDescriptor{K,Union{Nothing,FieldLinAlg.F2FullColumnFactor}}
        return Vector{T}(undef, n)
    elseif field.p == 3
        T = _FringeFiberImageDescriptor{K,Union{Nothing,FieldLinAlg.F3FullColumnFactor}}
        return Vector{T}(undef, n)
    end
    T = _FringeFiberImageDescriptor{K,Nothing}
    return Vector{T}(undef, n)
end

@inline function _new_fringe_descriptor_vector(::AbstractCoeffField, ::Type{K}, n::Int) where {K}
    T = _FringeFiberImageDescriptor{K,Nothing}
    return Vector{T}(undef, n)
end

@inline function _build_fringe_fiber_descriptor(
    field::QQField,
    rows::Vector{Int},
    phi_q,
    ::Type{K},
) where {K}
    T = _FringeFiberImageDescriptor{K,Union{Nothing,FieldLinAlg.FullColumnFactor{K}}}
    if isempty(rows)
        basis = zeros(K, 0, 0)
        return T(rows, basis, rows, nothing)
    end
    basis = FieldLinAlg.colspace(field, phi_q)
    factor = if size(basis, 2) == 0 || !_indicator_use_target_factor_plan(length(rows), size(basis, 2))
        nothing
    else
        FieldLinAlg._factor_fullcolumnQQ(basis)
    end
    rhs_rows = _indicator_target_rhs_rows(rows, factor)
    return T(rows, basis, rhs_rows, factor)
end

@inline function _build_fringe_fiber_descriptor(
    field::PrimeField,
    rows::Vector{Int},
    phi_q,
    ::Type{K},
) where {K}
    if field.p == 2
        T = _FringeFiberImageDescriptor{K,Union{Nothing,FieldLinAlg.F2FullColumnFactor}}
        if isempty(rows)
            basis = zeros(K, 0, 0)
            return T(rows, basis, rows, nothing)
        end
        basis = FieldLinAlg.colspace(field, phi_q)
        factor = if size(basis, 2) == 0 || !_indicator_use_target_factor_plan(length(rows), size(basis, 2))
            nothing
        else
            _indicator_target_factor(field, basis)
        end
        rhs_rows = _indicator_target_rhs_rows(rows, factor)
        return T(rows, basis, rhs_rows, factor)
    elseif field.p == 3
        T = _FringeFiberImageDescriptor{K,Union{Nothing,FieldLinAlg.F3FullColumnFactor}}
        if isempty(rows)
            basis = zeros(K, 0, 0)
            return T(rows, basis, rows, nothing)
        end
        basis = FieldLinAlg.colspace(field, phi_q)
        factor = if size(basis, 2) == 0 || !_indicator_use_target_factor_plan(length(rows), size(basis, 2))
            nothing
        else
            _indicator_target_factor(field, basis)
        end
        rhs_rows = _indicator_target_rhs_rows(rows, factor)
        return T(rows, basis, rhs_rows, factor)
    end
    T = _FringeFiberImageDescriptor{K,Nothing}
    if isempty(rows)
        basis = zeros(K, 0, 0)
        return T(rows, basis, rows, nothing)
    end
    basis = FieldLinAlg.colspace(field, phi_q)
    return T(rows, basis, rows, nothing)
end

@inline function _build_fringe_fiber_descriptor(
    field::AbstractCoeffField,
    rows::Vector{Int},
    phi_q,
    ::Type{K},
) where {K}
    T = _FringeFiberImageDescriptor{K,Nothing}
    if isempty(rows)
        basis = zeros(K, 0, 0)
        return T(rows, basis, rows, nothing)
    end
    basis = FieldLinAlg.colspace(field, phi_q)
    return T(rows, basis, rows, nothing)
end

@inline function _project_rows_to_vertex!(
    Im::Matrix{K},
    Bu::AbstractMatrix{K},
    rows_u::Vector{Int},
    rows_v::Vector{Int},
) where {K}
    rows_u === rows_v && return copyto!(Im, Bu)
    i = 1
    j = 1
    nv = length(rows_v)
    nu = length(rows_u)
    @inbounds while i <= nv && j <= nu
        rv = rows_v[i]
        ru = rows_u[j]
        if ru == rv
            copyto!(view(Im, i, :), view(Bu, j, :))
            i += 1
            j += 1
        elseif ru < rv
            j += 1
        else
            error("_project_rows_to_vertex!: projection rows mismatch")
        end
    end
    i > nv || error("_project_rows_to_vertex!: projection rows mismatch")
    return Im
end

@inline function _row_projection_slots(
    rows_src::Vector{Int},
    rows_tgt::Vector{Int},
)
    nt = length(rows_tgt)
    slots = Vector{Int}(undef, nt)
    nt <= 1 || issorted(rows_tgt) || return _row_projection_slots_unsorted(rows_src, rows_tgt, slots)
    i = 1
    j = 1
    ns = length(rows_src)
    @inbounds while i <= nt && j <= ns
        rt = rows_tgt[i]
        rs = rows_src[j]
        if rt == rs
            slots[i] = j
            i += 1
            j += 1
        elseif rs < rt
            j += 1
        else
            error("_row_projection_slots: target row missing from source row set")
        end
    end
    i > nt || error("_row_projection_slots: target row missing from source row set")
    return slots
end

@inline function _row_projection_slots_unsorted(
    rows_src::Vector{Int},
    rows_tgt::Vector{Int},
    slots::Vector{Int},
)
    @inbounds for i in eachindex(rows_tgt)
        rt = rows_tgt[i]
        j = searchsortedfirst(rows_src, rt)
        (j <= length(rows_src) && rows_src[j] == rt) ||
            error("_row_projection_slots: target row missing from source row set")
        slots[i] = j
    end
    return slots
end

@inline function _project_rows_by_slots!(
    Im::Matrix{K},
    Bu::AbstractMatrix{K},
    slots::Vector{Int},
) where {K}
    @inbounds for i in eachindex(slots)
        copyto!(view(Im, i, :), view(Bu, slots[i], :))
    end
    return Im
end

@inline _indicator_target_factor(::AbstractCoeffField, ::AbstractMatrix) = nothing

@inline function _indicator_target_factor(field::QQField, B::AbstractMatrix{K}) where {K}
    return FieldLinAlg._factor_fullcolumnQQ(B)
end

@inline function _indicator_target_factor(field::PrimeField, B::AbstractMatrix{K}) where {K}
    n = size(B, 2)
    n == 0 && return nothing
    if field.p == 2
        pivs = FieldLinAlg._pivot_cols_f2(transpose(B))
        length(pivs) == n ||
            error("_indicator_target_factor: expected full column rank, got rank $(length(pivs)) < $n")
        Bsub = Matrix{K}(B[pivs, :])
        invB = FieldLinAlg._f2_inverse_packed(Bsub)
        return FieldLinAlg.F2FullColumnFactor(pivs, invB, n)
    elseif field.p == 3
        pivs = FieldLinAlg._pivot_cols_f3(transpose(B))
        length(pivs) == n ||
            error("_indicator_target_factor: expected full column rank, got rank $(length(pivs)) < $n")
        Bsub = Matrix{K}(B[pivs, :])
        invB = FieldLinAlg._f3_inverse(Bsub)
        return FieldLinAlg.F3FullColumnFactor(pivs, invB)
    end
    return nothing
end

@inline _indicator_target_rhs_rows(rows_v::Vector{Int}, ::Nothing) = rows_v

@inline function _indicator_target_rhs_rows(rows_v::Vector{Int}, factor)
    return rows_v[factor.rows]
end

@inline function _indicator_apply_target_factor(
    ::QQField,
    factor::FieldLinAlg.FullColumnFactor{K},
    Bu::AbstractMatrix{K},
    slots::Vector{Int},
) where {K}
    rhs = Matrix{K}(undef, length(slots), size(Bu, 2))
    _project_rows_by_slots!(rhs, Bu, slots)
    X = Matrix{K}(undef, size(factor.invB, 1), size(rhs, 2))
    mul!(X, factor.invB, rhs)
    return X
end

@inline function _indicator_apply_target_factor(
    field::PrimeField,
    factor::FieldLinAlg.F3FullColumnFactor,
    Bu::AbstractMatrix{K},
    slots::Vector{Int},
) where {K}
    field.p == 3 || error("_indicator_apply_target_factor: expected F3 factor")
    rhs = Matrix{K}(undef, length(slots), size(Bu, 2))
    _project_rows_by_slots!(rhs, Bu, slots)
    X = Matrix{K}(undef, size(factor.invB, 1), size(rhs, 2))
    mul!(X, factor.invB, rhs)
    return X
end

@inline function _indicator_apply_target_factor(
    field::PrimeField,
    factor::FieldLinAlg.F2FullColumnFactor,
    Bu::AbstractMatrix{K},
    slots::Vector{Int},
) where {K}
    field.p == 2 || error("_indicator_apply_target_factor: expected F2 factor")
    n = factor.n
    rhs_cols = size(Bu, 2)
    X = Matrix{K}(undef, n, rhs_cols)
    z = zero(K)
    o = one(K)
    @inbounds for i in 1:n, j in 1:rhs_cols
        X[i, j] = z
    end
    invB = factor.invB
    nblocks = FieldLinAlg._f2_blocks(n)
    @inbounds for j in 1:rhs_cols
        packed = fill(UInt64(0), nblocks)
        for i in 1:n
            if Bu[slots[i], j].val != 0
                FieldLinAlg._f2_setbit!(packed, i)
            end
        end
        for i in 1:n
            acc = UInt64(0)
            row = invB[i]
            for blk in eachindex(row)
                acc = xor(acc, row[blk] & packed[blk])
            end
            acc = xor(acc, acc >>> 32)
            acc = xor(acc, acc >>> 16)
            acc = xor(acc, acc >>> 8)
            acc = xor(acc, acc >>> 4)
            acc = xor(acc, acc >>> 2)
            acc = xor(acc, acc >>> 1)
            X[i, j] = (acc & UInt64(1) != 0) ? o : z
        end
    end
    return X
end

@inline function _indicator_cover_map_from_basis(
    field::AbstractCoeffField,
    Bu::AbstractMatrix{K},
    Bv::AbstractMatrix{K},
    slots::Vector{Int},
    factor,
) where {K}
    if factor === nothing
        rhs = Matrix{K}(undef, length(slots), size(Bu, 2))
        _project_rows_by_slots!(rhs, Bu, slots)
        return FieldLinAlg.solve_fullcolumn(field, Bv, rhs)
    end
    return _indicator_apply_target_factor(field, factor, Bu, slots)
end

@inline function _indicator_target_factors(
    field::AbstractCoeffField,
    B::Vector{Matrix{K}},
    dims::Vector{Int},
    active_D::Vector{Vector{Int}},
) where {K}
    return fill(nothing, length(B))
end

@inline function _indicator_target_factors(
    field::QQField,
    B::Vector{Matrix{K}},
    dims::Vector{Int},
    active_D::Vector{Vector{Int}},
) where {K}
    factors = Vector{Union{Nothing,FieldLinAlg.FullColumnFactor{K}}}(undef, length(B))
    @inbounds for v in eachindex(B)
        factors[v] = (dims[v] == 0 || !_indicator_use_target_factor_plan(length(active_D[v]), dims[v])) ?
            nothing : FieldLinAlg._factor_fullcolumnQQ(B[v])
    end
    return factors
end

@inline function _indicator_target_factors(
    field::PrimeField,
    B::Vector{Matrix{K}},
    dims::Vector{Int},
    active_D::Vector{Vector{Int}},
) where {K}
    if field.p == 2
        factors = Vector{Union{Nothing,FieldLinAlg.F2FullColumnFactor}}(undef, length(B))
        @inbounds for v in eachindex(B)
            factors[v] = (dims[v] == 0 || !_indicator_use_target_factor_plan(length(active_D[v]), dims[v])) ?
                nothing : _indicator_target_factor(field, B[v])
        end
        return factors
    elseif field.p == 3
        factors = Vector{Union{Nothing,FieldLinAlg.F3FullColumnFactor}}(undef, length(B))
        @inbounds for v in eachindex(B)
            factors[v] = (dims[v] == 0 || !_indicator_use_target_factor_plan(length(active_D[v]), dims[v])) ?
                nothing : _indicator_target_factor(field, B[v])
        end
        return factors
    end
    return fill(nothing, length(B))
end

"""
    pmodule_from_fringe(H::FiniteFringe.FringeModule{K})
Return an internal `PModule{K}` whose fibers and structure maps are induced by the
fringe presentation `phi : oplus k[U_i] to oplus k[D_j]` (Defs. 3.16-3.17).
Implementation: M_q = im(phi_q) inside E_q; along a cover u<v the map is the restriction
E_u to E_v followed by projection to M_v.
"""
function pmodule_from_fringe(H::FiniteFringe.FringeModule{K}) where {K}
    field = H.field
    Q = H.P
    n = nvertices(Q)
    build_cache!(Q; cover=true)
    cc = _get_cover_cache(Q)
    active_U = _active_summands_per_vertex(H.U, n)
    active_D = _active_summands_per_vertex(H.D, n)
    preds = [_preds(cc, v) for v in 1:n]
    succs = [_succs(cc, u) for u in 1:n]

    # Basis/factor descriptor for each fiber image M_q = im(phi_q) inside E_q.
    descs = _new_fringe_descriptor_vector(field, K, n)
    dims = zeros(Int, n)
    for q in 1:n
        cols = active_U[q]
        rows = active_D[q]
        if isempty(cols) || isempty(rows)
            descs[q] = _build_fringe_fiber_descriptor(field, rows, zeros(K, length(rows), 0), K)
            dims[q] = 0
            continue
        end
        phi_q = @view H.phi[rows, cols]
        desc = _build_fringe_fiber_descriptor(field, rows, phi_q, K)
        descs[q] = desc
        dims[q] = size(desc.basis, 2)
    end

    # Build one packed row-projection plan per cover edge so the hot path does
    # not rescan row label vectors for every map assembly.
    proj_slots_to_succ = [Vector{Vector{Int}}(undef, length(succs[u])) for u in 1:n]
    @inbounds for u in 1:n
        rows_u = descs[u].rows
        su = succs[u]
        plans_u = proj_slots_to_succ[u]
        for j in eachindex(su)
            v = su[j]
            if dims[u] == 0 || dims[v] == 0
                plans_u[j] = Int[]
            else
                plans_u[j] = _row_projection_slots(rows_u, descs[v].rhs_rows)
            end
        end
    end

    # Structure maps on cover edges, aligned with the cover cache store so later
    # kernels can traverse cover edges without dict lookups.
    total_work = 0
    @inbounds for u in 1:n
        du = dims[u]
        du == 0 && continue
        for v in _succs(cc, u)
            total_work += du * dims[v]
        end
    end

    if !_indicator_use_direct_pmodule_store(cc.nedges, total_work, n)
        edge_maps = Dict{Tuple{Int,Int},Matrix{K}}()
        sizehint!(edge_maps, cc.nedges)
        @inbounds for u in 1:n
            su = succs[u]
            plans_u = proj_slots_to_succ[u]
            for j in eachindex(su)
                v = su[j]
                du = dims[u]
                dv = dims[v]
                X = if du == 0 || dv == 0
                    zeros(K, dv, du)
                else
                    _indicator_cover_map_from_basis(field, descs[u].basis, descs[v].basis, plans_u[j], descs[v].factor)
                end
                edge_maps[(u, v)] = X
            end
        end
        return PModule{K}(Q, dims, edge_maps; field=field)
    end

    maps_from_pred = [Vector{Matrix{K}}(undef, length(preds[v])) for v in 1:n]
    maps_to_succ = [Vector{Matrix{K}}(undef, length(succs[u])) for u in 1:n]

    @inbounds for u in 1:n
        su = succs[u]
        plans_u = proj_slots_to_succ[u]
        for j in eachindex(su)
            v = su[j]
            du = dims[u]
            dv = dims[v]
            X = if du == 0 || dv == 0
                zeros(K, dv, du)
            else
                _indicator_cover_map_from_basis(field, descs[u].basis, descs[v].basis, plans_u[j], descs[v].factor)
            end
            maps_to_succ[u][j] = X
            maps_from_pred[v][_pred_slots_of_succ(cc, u)[j]] = X
        end
    end
    store = CoverEdgeMapStore{K,Matrix{K}}(preds, succs, maps_from_pred, maps_to_succ, cc.nedges)
    return PModule{K}(Q, dims, store; field=field)
end

# -------------------------- projective cover (Def. 6.4.1) --------------------------

@inline function _colspace_union_incremental(
    field::AbstractCoeffField,
    nrows::Int,
    mats,
    ::Type{K},
) where {K}
    basis = zeros(K, nrows, 0)
    @inbounds for A in mats
        ncols = size(A, 2)
        ncols == 0 && continue
        if size(basis, 2) == 0
            basis = FieldLinAlg.colspace(field, A)
        else
            merged = Matrix{K}(undef, nrows, size(basis, 2) + ncols)
            @views merged[:, 1:size(basis, 2)] .= basis
            @views merged[:, size(basis, 2)+1:end] .= A
            basis = FieldLinAlg.colspace(field, merged)
        end
        size(basis, 2) == nrows && return basis
    end
    return basis
end

@inline function _colspace_union_dense(
    field::AbstractCoeffField,
    nrows::Int,
    total_cols::Int,
    mats,
    ::Type{K},
) where {K}
    total_cols == 0 && return zeros(K, nrows, 0)
    merged = Matrix{K}(undef, nrows, total_cols)
    offset = 0
    @inbounds for A in mats
        ncols = size(A, 2)
        ncols == 0 && continue
        @views merged[:, offset + 1:offset + ncols] .= A
        offset += ncols
    end
    offset == total_cols || error("_colspace_union_dense: total_cols mismatch")
    return FieldLinAlg.colspace(field, merged)
end

"Incoming image at v from immediate predecessors; basis matrix with K columns."
function _incoming_image_basis(M::PModule{K}, v::Int; cache::Union{Nothing,CoverCache}=nothing) where {K}
    field = M.field
    dv = M.dims[v]
    pv = M.edge_maps.preds[v]
    maps = M.edge_maps.maps_from_pred[v]

    if dv == 0 || isempty(pv)
        return zeros(K, dv, 0)
    end

    tot = 0
    @inbounds for u in pv
        tot += M.dims[u]
    end
    if tot == 0
        return zeros(K, dv, 0)
    end

    if length(pv) == 1
        return FieldLinAlg.colspace(field, maps[1])
    end

    if !_indicator_use_incremental_union(field, dv, tot, length(pv))
        return _colspace_union_dense(field, dv, tot, maps, K)
    end

    return _colspace_union_incremental(field, dv, maps, K)
end



"""
    projective_cover(M::PModule{K}; output=:full) -> ProjectiveCoverResult
    projective_cover(M::PModule{K}; output=:summary) -> NamedTuple

Build the one-step projective cover of `M`.

Mathematically, this constructs a projective module `F_0` together with an
augmentation `F_0 -> M` that presents the first stage of an upset/projective
resolution.

The returned [`ProjectiveCoverResult`](@ref) stores:
- the covering module `F_0`,
- the augmentation `F_0 -> M`,
- generator data, exposed lazily by default.

The result iterates as `(F0, pi0, gens_at)` for compatibility with tuple
destructuring. Internal callers may pass `materialize_gens=false` to receive the
packed `_ProjectiveGeneratorPlan` inside the result rather than the lazy public
wrapper.

Use `output=:summary` for the cheap/default inspection path when you only want:
- the side (`:projective`),
- field/poset metadata,
- generator counts,
- storage/laziness information.

Use `output=:full` when you actually need the cover module, augmentation map, or
generator wrapper for further computation. Full materialization is appropriate
for algebraic follow-up work; summary output is appropriate for notebook/REPL
exploration.
"""
function projective_cover(M::PModule{K};
                          cache::Union{Nothing,CoverCache}=nothing,
                          map_memo::Union{Nothing,AbstractVector{Union{Nothing,Matrix{K}}}}=nothing,
                          workspace::Union{Nothing,_ResolutionWorkspace{K}}=nothing,
                          materialize_gens::Bool=true,
                          output::Symbol=:full,
                          threads::Bool = (Threads.nthreads() > 1)) where {K}
    field = M.field
    Q = M.Q; n = nvertices(Q)
    build_cache!(Q; cover=true, updown=true)
    cc = cache === nothing ? _get_cover_cache(Q) : cache
    total_dims = sum(M.dims)
    expected_work = max(1, cc.nedges) * max(1, total_dims)
    threaded = _indicator_use_threads(threads, n, total_dims, expected_work)
    map_memo_local = map_memo === nothing ? _indicator_new_array_memo(K, n) : map_memo
    ws = workspace
    if ws === nothing && !threaded
        ws = _new_resolution_workspace(K, n)
    end
    memos = threaded ?
        [_indicator_new_array_memo(K, n)
         for _ in 1:max(1, Threads.maxthreadid())] : Vector{Vector{Union{Nothing,Matrix{K}}}}()

    # number of generators at each vertex = dim(M_v) - rank(incoming_image)
    chosen_at = Vector{Vector{Int}}(undef, n)
    if threaded
        Threads.@threads for v in 1:n
            Img = _incoming_image_basis(M, v; cache=cc)
            chosen = _choose_projective_generators(field, Img, M.dims[v])
            chosen_at[v] = chosen
        end
    else
        for v in 1:n
            Img = _incoming_image_basis(M, v; cache=cc)
            chosen = _choose_projective_generators(field, Img, M.dims[v])
            chosen_at[v] = chosen
        end
    end
    plan = _build_projective_generator_plan(Q, chosen_at)
    counts = plan.counts

        # F0 as a direct sum of principal upsets.
    #
    # IMPORTANT CORRECTNESS NOTE
    # On a general finite poset, the cover-edge maps of a direct sum of principal
    # upsets are NOT given by a "rectangular identity" unless the chosen basis at
    # every vertex extends the basis at each predecessor as a prefix.
    #
    # We therefore build the cover-edge maps by matching generator labels (p,j)
    # across vertices. This agrees with the representable functor structure and is
    # independent of the arbitrary ordering of vertices.
    F0_dims = plan.Fdims
    active_sources = plan.active_sources

    F0_edges = Dict{Tuple{Int,Int}, SparseMatrixCSC{K,Int}}()
    if threaded
        edges = cover_edges(Q)
        mats = Vector{SparseMatrixCSC{K,Int}}(undef, length(edges))
        Threads.@threads for idx in eachindex(edges)
            u, v = edges[idx]
            Muv = _subsequence_identity_sparse_blocks(
                K,
                active_sources,
                v,
                u,
                counts,
                F0_dims[v],
                F0_dims[u],
            )
            mats[idx] = Muv
        end
        for idx in eachindex(edges)
            F0_edges[edges[idx]] = mats[idx]
        end
    else
        @inbounds for u in 1:n
            su = _succs(cc, u)
            for v in su
                Muv = _subsequence_identity_sparse_blocks(
                    K,
                    active_sources,
                    v,
                    u,
                    counts,
                    F0_dims[v],
                    F0_dims[u],
                )
                F0_edges[(u, v)] = Muv
            end
        end
    end
    F0 = PModule{K}(Q, F0_dims, F0_edges; field=field)


    # pi0 : F0 -> M
    #
    # Preallocate each component and fill blockwise.
    # Old code used repeated hcat (allocates many temporaries).
    # Cache the chosen basis indices in each M_p once.
    J_at = chosen_at
    total_pairs = length(active_sources.data)
    use_plans = _indicator_use_map_plans(n, total_pairs)
    batch_by_i = Vector{Union{Nothing,MapLeqQueryBatch}}()

    if use_plans
        resize!(batch_by_i, n)
        @inbounds for i in 1:n
            lo = _packed_firstindex(active_sources, i)
            hi = _packed_lastindex(active_sources, i)
            if hi < lo
                batch_by_i[i] = nothing
                continue
            end
            pairs = Vector{Tuple{Int,Int}}(undef, hi - lo + 1)
            pos = 1
            for idx in lo:hi
                pairs[pos] = (active_sources.data[idx], i)
                pos += 1
            end
            batch_by_i[i] = isempty(pairs) ? nothing : prepare_map_leq_batch(pairs)
        end
    end

    comps = Vector{Matrix{K}}(undef, n)
    if threaded
        Threads.@threads for i in 1:n
            memo = memos[_thread_local_index(memos)]
            Mi = M.dims[i]
            Fi = F0_dims[i]
            cols = zeros(K, Mi, Fi)
            col = 1
            lo = _packed_firstindex(active_sources, i)
            hi = _packed_lastindex(active_sources, i)
            batch = if use_plans
                batch_by_i[i]
            else
                if hi < lo
                    nothing
                else
                    pairs = Vector{Tuple{Int,Int}}(undef, hi - lo + 1)
                    pos = 1
                    for idx in lo:hi
                        pairs[pos] = (active_sources.data[idx], i)
                        pos += 1
                    end
                    prepare_map_leq_batch(pairs)
                end
            end
            maps = batch === nothing ? Matrix{K}[] : _map_leq_cached_many_indicator(M, batch, cc, memo)
            if hi >= lo
                t = 1
                @inbounds for idx in lo:hi
                    p = active_sources.data[idx]
                    A = maps[t]
                    col = _gather_selected_columns!(cols, col, A, J_at[p])
                    t += 1
                end
            end
            comps[i] = cols
        end
    else
        for i in 1:n
            Mi = M.dims[i]
            Fi = F0_dims[i]
            cols = zeros(K, Mi, Fi)
            col = 1
            lo = _packed_firstindex(active_sources, i)
            hi = _packed_lastindex(active_sources, i)
            pairs = Tuple{Int,Int}[]
            batch = nothing
            if use_plans
                batch = batch_by_i[i]
                if ws !== nothing
                    pairs = ws.pairs_buf
                    empty!(pairs)
                    if hi >= lo
                        for idx in lo:hi
                            push!(pairs, (active_sources.data[idx], i))
                        end
                    end
                end
            elseif ws === nothing
                if hi >= lo
                    pairs = Vector{Tuple{Int,Int}}(undef, hi - lo + 1)
                    pos = 1
                    for idx in lo:hi
                        pairs[pos] = (active_sources.data[idx], i)
                        pos += 1
                    end
                end
                batch = isempty(pairs) ? nothing : prepare_map_leq_batch(pairs)
            else
                pairs = ws.pairs_buf
                empty!(pairs)
                if hi >= lo
                    for idx in lo:hi
                        push!(pairs, (active_sources.data[idx], i))
                    end
                end
            end
            if ws === nothing
                maps = batch === nothing ? Matrix{K}[] : _map_leq_cached_many_indicator(M, batch, cc, map_memo_local)
                if hi >= lo
                    t = 1
                    @inbounds for idx in lo:hi
                        p = active_sources.data[idx]
                        A = maps[t]
                        col = _gather_selected_columns!(cols, col, A, J_at[p])
                        t += 1
                    end
                end
            else
                if length(pairs) <= INDICATOR_MAP_BATCH_THRESHOLD[]
                    _map_leq_fill_memo_indicator!(M, pairs, cc, map_memo_local, ws)
                else
                    if use_plans
                        _map_leq_fill_memo_indicator!(M, batch::MapLeqQueryBatch, cc, map_memo_local, ws)
                    else
                        _map_leq_fill_memo_indicator!(M, _workspace_get_or_prepare_batch!(ws, pairs), cc, map_memo_local, ws)
                    end
                end
                if hi >= lo
                    @inbounds for idx in lo:hi
                        p = active_sources.data[idx]
                        A = _indicator_memo_get(map_memo_local, n, p, i)::Matrix{K}
                        col = _gather_selected_columns!(cols, col, A, J_at[p])
                    end
                end
            end
            comps[i] = cols
        end
    end
    pi0 = PMorphism{K}(F0, M, comps)
    gens = materialize_gens ? _projective_generators_view(plan) : plan
    result = ProjectiveCoverResult(F0, pi0, gens, false, false)
    return _indicator_resolution_with_output(:projective_cover, result, cover_summary, output)
end


# Basis of the "socle" at vertex u: kernel of the stacked outgoing map
# M_u to oplus_{u<v} M_v along cover edges u < v.
# Columns of the returned matrix span soc(M)_u subseteq M_u.
function _socle_basis(M::PModule{K}, u::Int; cache::Union{Nothing,CoverCache}=nothing) where {K}
    field = M.field
    cc = (cache === nothing ? _get_cover_cache(M.Q) : cache)
    su = _succs(cc, u)
    du = M.dims[u]

    if isempty(su) || du == 0
        return eye(field, du)
    end

    # Build the stacked outgoing map A : M_u -> (direct sum over cover successors).
    # A has size (sum_v dim(M_v)) x dim(M_u).
    tot = 0
    @inbounds for j in eachindex(su)
        tot += M.dims[su[j]]
    end
    if tot == 0
        # No nonzero target fibers; outgoing map is zero, so socle is all of M_u.
        return eye(field, du)
    end
    S = _outgoing_span_basis(M, u; cache=cc)
    size(S, 1) == 0 && return eye(field, du)
    return FieldLinAlg.nullspace(field, S)  # columns span the socle at u
end


# A canonical left-inverse for a full-column-rank matrix S: L*S = I.
# Over finite fields we cannot use the Gram matrix formula because S'S can be
# singular even when S has full column rank.
function _left_inverse_full_column(field::AbstractCoeffField, S::AbstractMatrix{K}) where {K}
    A = Matrix(S)
    m, s = size(A)
    if s == 0
        return zeros(K, 0, m)
    end
    if field isa RealField
        return A \ eye(field, m)
    end

    Aug = hcat(A, eye(field, m))
    R, pivs_all = FieldLinAlg.rref(field, Aug)
    pivs = Int[]
    @inbounds for p in pivs_all
        p <= s && push!(pivs, p)
    end
    length(pivs) == s ||
        error("_left_inverse_full_column: expected full column rank, got rank $(length(pivs)) < $s")

    L = zeros(K, s, m)
    @inbounds for (row, pcol) in enumerate(pivs)
        L[pcol, :] = R[row, s+1:end]
    end
    return L
end

@inline function _injective_active_plan(
    Q::AbstractPoset,
    mult::Vector{Int},
)
    n = nvertices(Q)
    Edims = zeros(Int, n)
    counts = zeros(Int, n)

    @inbounds for i in 1:n
        total = 0
        for u in upset_indices(Q, i)
            mu = mult[u]
            mu == 0 && continue
            counts[i] += 1
            total += mu
        end

        Edims[i] = total
    end

    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    @inbounds for i in 1:n
        ptr[i + 1] = ptr[i] + counts[i]
    end
    data = Vector{Int}(undef, ptr[end] - 1)
    fill!(counts, 0)
    @inbounds for i in 1:n
        base = ptr[i] - 1
        for u in upset_indices(Q, i)
            mu = mult[u]
            mu == 0 && continue
            counts[i] += 1
            data[base + counts[i]] = u
        end
    end

    return Edims, _PackedIntLists(ptr, data)
end

@inline function _injective_active_plan(
    Q::AbstractPoset,
    mult::Vector{Int},
    active_vertices::Vector{Int},
)
    n = nvertices(Q)
    Edims = zeros(Int, n)
    counts = zeros(Int, n)

    @inbounds for u in active_vertices
        mu = mult[u]
        mu == 0 && continue
        for i in downset_indices(Q, u)
            Edims[i] += mu
            counts[i] += 1
        end
    end

    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    @inbounds for i in 1:n
        ptr[i + 1] = ptr[i] + counts[i]
        counts[i] = 0
    end
    data = Vector{Int}(undef, ptr[end] - 1)

    @inbounds for u in active_vertices
        mult[u] == 0 && continue
        for i in downset_indices(Q, u)
            counts[i] += 1
            data[ptr[i] + counts[i] - 1] = u
        end
    end

    return Edims, _PackedIntLists(ptr, data)
end

@inline function _injective_active_ids(
    active_sources::_PackedIntLists,
    gid_starts::Vector{Int},
    Edims::Vector{Int},
)
    n = length(Edims)
    active_ids = Vector{Vector{Int}}(undef, n)
    @inbounds for i in 1:n
        ids = Vector{Int}(undef, Edims[i])
        pos = 1
        lo = _packed_firstindex(active_sources, i)
        hi = _packed_lastindex(active_sources, i)
        if hi >= lo
            for idx in lo:hi
                u = active_sources.data[idx]
                for gid in gid_starts[u]:(gid_starts[u + 1] - 1)
                    ids[pos] = gid
                    pos += 1
                end
            end
        end
        active_ids[i] = ids
    end
    return active_ids
end

@inline function _injective_active_gid_plan(
    active_sources::_PackedIntLists,
    gid_starts::Vector{Int},
    Edims::Vector{Int},
)
    n = length(Edims)
    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    @inbounds for i in 1:n
        ptr[i + 1] = ptr[i] + Edims[i]
    end
    data = Vector{Int}(undef, ptr[end] - 1)
    @inbounds for i in 1:n
        pos = ptr[i]
        lo = _packed_firstindex(active_sources, i)
        hi = _packed_lastindex(active_sources, i)
        if hi >= lo
            for idx in lo:hi
                u = active_sources.data[idx]
                for gid in gid_starts[u]:(gid_starts[u + 1] - 1)
                    data[pos] = gid
                    pos += 1
                end
            end
        end
    end
    return _PackedIntLists(ptr, data)
end

@inline function _packed_lists_to_vectors(p::_PackedIntLists, n::Int)
    out = Vector{Vector{Int}}(undef, n)
    @inbounds for i in 1:n
        lo = _packed_firstindex(p, i)
        hi = _packed_lastindex(p, i)
        out[i] = hi < lo ? Int[] : copy(view(p.data, lo:hi))
    end
    return out
end

@inline function _build_injective_generator_plan(
    Q::AbstractPoset,
    mult::Vector{Int},
    active_socle_vertices::Vector{Int},
)
    sorted_active = issorted(active_socle_vertices) ? active_socle_vertices : sort(active_socle_vertices)
    Edims, active_sources = _injective_active_plan(Q, mult, sorted_active)
    gid_starts = Vector{Int}(undef, length(mult) + 1)
    total = _fill_generator_starts!(gid_starts, mult)
    active_gid_lists = _injective_active_gid_plan(active_sources, gid_starts, Edims)
    return _InjectiveGeneratorPlan(
        mult,
        gid_starts,
        total,
        sorted_active,
        active_sources,
        active_gid_lists,
        Edims,
    )
end

@inline function _projective_active_plan(
    Q::AbstractPoset,
    counts::Vector{Int},
    active_birth_vertices::Vector{Int},
)
    n = nvertices(Q)
    Fdims = zeros(Int, n)
    per_vertex_counts = zeros(Int, n)

    @inbounds for p in active_birth_vertices
        cp = counts[p]
        cp == 0 && continue
        for i in upset_indices(Q, p)
            Fdims[i] += cp
            per_vertex_counts[i] += 1
        end
    end

    ptr = Vector{Int}(undef, n + 1)
    ptr[1] = 1
    @inbounds for i in 1:n
        ptr[i + 1] = ptr[i] + per_vertex_counts[i]
        per_vertex_counts[i] = 0
    end
    data = Vector{Int}(undef, ptr[end] - 1)

    @inbounds for p in active_birth_vertices
        counts[p] == 0 && continue
        for i in upset_indices(Q, p)
            per_vertex_counts[i] += 1
            data[ptr[i] + per_vertex_counts[i] - 1] = p
        end
    end

    return Fdims, _PackedIntLists(ptr, data)
end

@inline function _build_projective_generator_plan(
    Q::AbstractPoset,
    chosen_cols::Vector{Vector{Int}},
)
    counts = Vector{Int}(undef, length(chosen_cols))
    active_birth_vertices = Int[]
    @inbounds for p in eachindex(chosen_cols)
        cp = length(chosen_cols[p])
        counts[p] = cp
        cp == 0 || push!(active_birth_vertices, p)
    end
    gid_starts = Vector{Int}(undef, length(counts) + 1)
    total = _fill_generator_starts!(gid_starts, counts)
    Fdims, active_sources = _projective_active_plan(Q, counts, active_birth_vertices)
    self_starts_down = ones(Int, length(counts))
    data = active_sources.data
    @inbounds for p in active_birth_vertices
        pos = 1
        lo = _packed_firstindex(active_sources, p)
        hi = _packed_lastindex(active_sources, p)
        for idx in lo:hi
            u = data[idx]
            u == p && break
            pos += counts[u]
        end
        self_starts_down[p] = pos
    end
    return _ProjectiveGeneratorPlan(
        chosen_cols,
        counts,
        gid_starts,
        total,
        active_birth_vertices,
        active_sources,
        Fdims,
        self_starts_down,
    )
end

@inline function _materialize_projective_gens(plan::_ProjectiveGeneratorPlan)
    n = length(plan.counts)
    gens_at = Vector{Vector{Tuple{Int,Int}}}(undef, n)
    @inbounds for p in 1:n
        chosen = plan.chosen_cols[p]
        gens = Vector{Tuple{Int,Int}}(undef, length(chosen))
        for j in eachindex(chosen)
            gens[j] = (p, chosen[j])
        end
        gens_at[p] = gens
    end
    return gens_at
end

@inline function _empty_packed_lists(n::Int)
    return _PackedIntLists(ones(Int, n + 1), Int[])
end

@inline function _indicator_use_packed_injective_plan(
    nsocle::Int,
    total_gens::Int,
    total_hull_dims::Int,
)
    nsocle >= _INDICATOR_INJECTIVE_PACKED_PLAN_MIN_SOCLE_VERTICES[] && return true
    total_gens >= _INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_GENS[] && return true
    total_hull_dims >= _INDICATOR_INJECTIVE_PACKED_PLAN_MIN_TOTAL_HULL_DIMS[] && return true
    return false
end

@inline function _materialize_injective_gens(plan::_InjectiveGeneratorPlan)
    n = length(plan.mult)
    gens_at = Vector{Vector{Tuple{Int,Int}}}(undef, n)
    @inbounds for u in 1:n
        mu = plan.mult[u]
        if mu == 0
            gens_at[u] = Tuple{Int,Int}[]
        else
            gens_at[u] = [(u, j) for j in 1:mu]
        end
    end
    return gens_at
end

@inline function _principal_downsets_from_plan(P::PT, plan::_InjectiveGeneratorPlan) where {PT<:AbstractPoset}
    n = nvertices(P)
    D_at = Vector{Downset{PT}}(undef, n)
    @inbounds for u in 1:n
        D_at[u] = principal_downset(P, u)
    end
    D = Downset{PT}[]
    @inbounds for u in 1:n
        mu = plan.mult[u]
        for _ in 1:mu
            push!(D, D_at[u])
        end
    end
    return D
end

@inline function _support_slot_vector(n::Int, support_vertices::Vector{Int})
    slots = zeros(Int, n)
    @inbounds for (idx, u) in enumerate(support_vertices)
        slots[u] = idx
    end
    return slots
end

@inline function _refresh_support_vertex!(
    support_vertices::Vector{Int},
    support_mask::BitVector,
    support_slots::Vector{Int},
    dims::Vector{Int},
    u::Int,
)
    new_active = dims[u] != 0
    old_active = support_mask[u]
    new_active == old_active && return nothing
    if new_active
        push!(support_vertices, u)
        support_slots[u] = length(support_vertices)
        support_mask[u] = true
    else
        idx = support_slots[u]
        last = support_vertices[end]
        support_vertices[idx] = last
        support_slots[last] = idx
        pop!(support_vertices)
        support_slots[u] = 0
        support_mask[u] = false
    end
    return nothing
end

@inline function _update_support_state!(
    support_vertices::Vector{Int},
    support_mask::BitVector,
    support_slots::Vector{Int},
    dims::Vector{Int},
    changed_vertices::Vector{Int},
    frontier_mask::BitVector,
)
    @inbounds for u in changed_vertices
        _refresh_support_vertex!(support_vertices, support_mask, support_slots, dims, u)
    end
    @inbounds for u in eachindex(frontier_mask)
        frontier_mask[u] || continue
        _refresh_support_vertex!(support_vertices, support_mask, support_slots, dims, u)
    end
    return support_vertices
end

@inline function _update_support_state!(
    support_vertices::Vector{Int},
    support_mask::BitVector,
    support_slots::Vector{Int},
    dims::Vector{Int},
    changed_vertices::Vector{Int},
    frontier_vertices::Vector{Int},
)
    @inbounds for u in changed_vertices
        _refresh_support_vertex!(support_vertices, support_mask, support_slots, dims, u)
    end
    @inbounds for u in frontier_vertices
        _refresh_support_vertex!(support_vertices, support_mask, support_slots, dims, u)
    end
    return support_vertices
end

# Build the injective (downset) hull:  iota : M into E  where
# E is a direct sum of principal downsets with multiplicities = socle dimensions.
# Also return the generator labels as (u, j) with u the vertex and j the column.
function _injective_hull(M::PModule{K};
                         cache::Union{Nothing,CoverCache}=nothing,
                         map_memo::Union{Nothing,AbstractVector{Union{Nothing,Matrix{K}}}}=nothing,
                         workspace::Union{Nothing,_ResolutionWorkspace{K}}=nothing,
                         support_vertices::Union{Nothing,Vector{Int}}=nothing,
                         graph_lists::Union{Nothing,_CoverGraphLists}=nothing,
                         reuse_cache::Union{Nothing,_InjectiveHullReuseCache{K}}=nothing,
                         recompute_mask::Union{Nothing,BitVector}=nothing,
                         materialize_gens::Bool=true,
                         threads::Bool = (Threads.nthreads() > 1)) where {K}
    field = M.field
    Q = M.Q; n = nvertices(Q)
    build_cache!(Q; cover=true, updown=true)
    cc = cache === nothing ? _get_cover_cache(Q) : cache
    total_dims = sum(M.dims)
    expected_work = max(1, cc.nedges) * max(1, total_dims)
    threaded = _indicator_use_threads(threads, n, total_dims, expected_work)
    map_memo_local = map_memo === nothing ? _indicator_new_array_memo(K, n) : map_memo
    ws = workspace
    if ws === nothing && !threaded
        ws = _new_resolution_workspace(K, n)
    end
    memos = threaded ?
        [_indicator_new_array_memo(K, n)
         for _ in 1:max(1, Threads.maxthreadid())] : Vector{Vector{Union{Nothing,Matrix{K}}}}()

    # socle bases at each vertex and their multiplicities
    Soc = Vector{Matrix{K}}(undef, n)
    mult = zeros(Int, n)
    support = support_vertices === nothing ? _nonzero_dim_vertices(M.dims) : support_vertices
    reuse_cache === nothing || _ensure_injective_hull_reuse_cache!(reuse_cache, n)
    @inbounds for u in 1:n
        Soc[u] = zeros(K, M.dims[u], 0)
    end
    if threaded
        Threads.@threads for idx in eachindex(support)
            u = support[idx]
            can_reuse = reuse_cache !== nothing &&
                        reuse_cache.valid[u] &&
                        (recompute_mask === nothing || !recompute_mask[u]) &&
                        size(reuse_cache.soc[u], 1) == M.dims[u]
            if can_reuse
                Soc[u] = reuse_cache.soc[u]
                mult[u] = reuse_cache.mult[u]
            else
                Su = _socle_basis(M, u; cache=cc)
                Soc[u] = Su
                mult[u] = size(Su, 2)
                if reuse_cache !== nothing
                    reuse_cache.soc[u] = Su
                    reuse_cache.mult[u] = mult[u]
                    reuse_cache.valid[u] = true
                end
            end
        end
    else
        for u in support
            can_reuse = reuse_cache !== nothing &&
                        reuse_cache.valid[u] &&
                        (recompute_mask === nothing || !recompute_mask[u]) &&
                        size(reuse_cache.soc[u], 1) == M.dims[u]
            if can_reuse
                Soc[u] = reuse_cache.soc[u]
                mult[u] = reuse_cache.mult[u]
            else
                Su = _socle_basis(M, u; cache=cc)
                Soc[u] = Su
                mult[u] = size(Su, 2)
                if reuse_cache !== nothing
                    reuse_cache.soc[u] = Su
                    reuse_cache.mult[u] = mult[u]
                    reuse_cache.valid[u] = true
                end
            end
        end
    end
    active_socle_vertices = Int[]
    @inbounds for u in support
        mult[u] == 0 && continue
        push!(active_socle_vertices, u)
    end

    sorted_active = issorted(active_socle_vertices) ? active_socle_vertices : sort(active_socle_vertices)
    Edims, active_sources = _injective_active_plan(Q, mult, sorted_active)
    gid_starts = Vector{Int}(undef, length(mult) + 1)
    total_gens = _fill_generator_starts!(gid_starts, mult)
    total_hull_dims = sum(Edims)
    use_packed_gid_plan = _indicator_use_packed_injective_plan(length(sorted_active), total_gens, total_hull_dims)
    active_gid_lists = use_packed_gid_plan ?
        _injective_active_gid_plan(active_sources, gid_starts, Edims) :
        _empty_packed_lists(length(Edims))
    plan = _InjectiveGeneratorPlan(
        mult,
        gid_starts,
        total_gens,
        sorted_active,
        active_sources,
        active_gid_lists,
        Edims,
    )
    active_hull_vertices = Int[]
    active_iota_vertices = Int[]
    sizehint!(active_hull_vertices, length(active_socle_vertices))
    sizehint!(active_iota_vertices, length(active_socle_vertices))
    @inbounds for i in 1:n
        if Edims[i] != 0
            push!(active_hull_vertices, i)
            if M.dims[i] != 0
                push!(active_iota_vertices, i)
            end
        end
    end
    active_ids = use_packed_gid_plan ? Vector{Vector{Int}}() : _injective_active_ids(active_sources, gid_starts, Edims)

    # E structure maps along cover edges u<v are coordinate projections:
    # keep exactly those generators that are still active at v.
    preds = graph_lists === nothing ? [_preds(cc, v) for v in 1:n] : graph_lists.preds
    succs = graph_lists === nothing ? [_succs(cc, u) for u in 1:n] : graph_lists.succs
    maps_from_pred = [Vector{SparseMatrixCSC{K,Int}}(undef, length(preds[v])) for v in 1:n]
    maps_to_succ = [Vector{SparseMatrixCSC{K,Int}}(undef, length(succs[u])) for u in 1:n]
    zero_from_empty = Vector{SparseMatrixCSC{K,Int}}(undef, n)
    zero_to_empty = Vector{SparseMatrixCSC{K,Int}}(undef, n)
    @inbounds for v in 1:n
        zero_from_empty[v] = spzeros(K, Edims[v], 0)
        zero_to_empty[v] = spzeros(K, 0, Edims[v])
    end

    if threaded
        Threads.@threads for u in 1:n
            su = succs[u]
            pred_slots = graph_lists === nothing ? _pred_slots_of_succ(cc, u) : graph_lists.pred_slot_of_succ[u]
            local_maps = Vector{SparseMatrixCSC{K,Int}}(undef, length(su))
            for j in eachindex(su)
                v = su[j]
                local_maps[j] = if Edims[u] == 0
                    zero_from_empty[v]
                elseif Edims[v] == 0
                    zero_to_empty[u]
                elseif use_packed_gid_plan
                    _projection_identity_sparse(K, active_gid_lists, v, u)
                else
                    _projection_identity_sparse(K, active_ids[v], active_ids[u])
                end
            end
            maps_to_succ[u] = local_maps
            for j in eachindex(su)
                v = su[j]
                maps_from_pred[v][pred_slots[j]] = local_maps[j]
            end
        end
    else
        @inbounds for u in 1:n
            su = succs[u]
            pred_slots = graph_lists === nothing ? _pred_slots_of_succ(cc, u) : graph_lists.pred_slot_of_succ[u]
            for j in eachindex(su)
                v = su[j]
                Muv = if Edims[u] == 0
                    zero_from_empty[v]
                elseif Edims[v] == 0
                    zero_to_empty[u]
                elseif use_packed_gid_plan
                    _projection_identity_sparse(K, active_gid_lists, v, u)
                else
                    _projection_identity_sparse(K, active_ids[v], active_ids[u])
                end
                maps_to_succ[u][j] = Muv
                maps_from_pred[v][pred_slots[j]] = Muv
            end
        end
    end
    store = CoverEdgeMapStore{K,SparseMatrixCSC{K,Int}}(preds, succs, maps_from_pred, maps_to_succ, cc.nedges)
    E = PModule{K}(Q, Edims, store; field=field)

    # iota : M -> E
    Linv = Vector{Matrix{K}}(undef, n)
    @inbounds for u in 1:n
        Linv[u] = zeros(K, 0, M.dims[u])
    end
    @inbounds for u in active_socle_vertices
        can_reuse = reuse_cache !== nothing &&
                    reuse_cache.valid[u] &&
                    (recompute_mask === nothing || !recompute_mask[u]) &&
                    size(reuse_cache.linv[u], 1) == mult[u] &&
                    size(reuse_cache.linv[u], 2) == M.dims[u]
        if can_reuse
            Linv[u] = reuse_cache.linv[u]
        else
            Lu = _left_inverse_full_column(field, Soc[u])
            Linv[u] = Lu
            if reuse_cache !== nothing
                reuse_cache.linv[u] = Lu
            end
        end
    end

    active_data = active_sources.data
    active_ptr = active_sources.ptr
    total_pairs = length(active_data)
    use_plans = _indicator_use_map_plans(n, total_pairs)
    pairs_by_i = Vector{Union{Nothing,Vector{Tuple{Int,Int}}}}()
    batch_by_i = Vector{Union{Nothing,MapLeqQueryBatch}}()
    if use_plans
        resize!(pairs_by_i, n)
        resize!(batch_by_i, n)
        fill!(pairs_by_i, nothing)
        fill!(batch_by_i, nothing)
        @inbounds for i in active_iota_vertices
            lo = active_ptr[i]
            hi = active_ptr[i + 1] - 1
            pairs = Vector{Tuple{Int,Int}}(undef, max(0, hi - lo + 1))
            pos = 1
            if hi >= lo
                for idx in lo:hi
                    pairs[pos] = (i, active_data[idx])
                    pos += 1
                end
            end
            pairs_by_i[i] = pairs
            batch_by_i[i] = isempty(pairs) ? nothing : prepare_map_leq_batch(pairs)
        end
    end

    # Only vertices with nontrivial socle contribute rows.
    comps = Vector{Matrix{K}}(undef, n)
    @inbounds for i in 1:n
        comps[i] = zeros(K, Edims[i], M.dims[i])
    end
    if threaded
        Threads.@threads for idx in eachindex(active_iota_vertices)
            i = active_iota_vertices[idx]
            memo = memos[_thread_local_index(memos)]
            rows = comps[i]
            r = 1
            lo = active_ptr[i]
            hi = active_ptr[i + 1] - 1
            batch = if use_plans
                batch_by_i[i]
            else
                pairs = hi < lo ? Tuple{Int,Int}[] : Tuple{Int,Int}[(i, active_data[idx]) for idx in lo:hi]
                isempty(pairs) ? nothing : prepare_map_leq_batch(pairs)
            end
            maps = batch === nothing ? Matrix{K}[] : _map_leq_cached_many_indicator(M, batch, cc, memo)
            pos = 1
            if hi >= lo
                @inbounds for idx in lo:hi
                    u = active_data[idx]
                    m = mult[u]
                    Mi_to_Mu = maps[pos]
                    @views mul!(rows[r:r+m-1, :], Linv[u], Mi_to_Mu)
                    r += m
                    pos += 1
                end
            end
            @assert r == Edims[i] + 1
        end
    else
        for i in active_iota_vertices
            rows = comps[i]
            r = 1
            lo = active_ptr[i]
            hi = active_ptr[i + 1] - 1
            pairs = Tuple{Int,Int}[]
            batch = nothing
            if use_plans
                pairs = pairs_by_i[i]::Vector{Tuple{Int,Int}}
                batch = batch_by_i[i]
            elseif ws === nothing
                pairs = hi < lo ? Tuple{Int,Int}[] : Tuple{Int,Int}[(i, active_data[idx]) for idx in lo:hi]
                batch = isempty(pairs) ? nothing : prepare_map_leq_batch(pairs)
            else
                pairs = ws.pairs_buf
                empty!(pairs)
                if hi >= lo
                    for idx in lo:hi
                        push!(pairs, (i, active_data[idx]))
                    end
                end
            end
            if ws === nothing
                maps = batch === nothing ? Matrix{K}[] : _map_leq_cached_many_indicator(M, batch, cc, map_memo_local)
                pos = 1
                if hi >= lo
                    @inbounds for idx in lo:hi
                        u = active_data[idx]
                        m = mult[u]
                        Mi_to_Mu = maps[pos]
                        @views mul!(rows[r:r+m-1, :], Linv[u], Mi_to_Mu)
                        r += m
                        pos += 1
                    end
                end
            else
                if length(pairs) <= INDICATOR_MAP_BATCH_THRESHOLD[]
                    _map_leq_fill_memo_indicator!(M, pairs, cc, map_memo_local, ws)
                else
                    if use_plans
                        _map_leq_fill_memo_indicator!(M, batch::MapLeqQueryBatch, cc, map_memo_local, ws)
                    else
                        _map_leq_fill_memo_indicator!(M, _workspace_get_or_prepare_batch!(ws, pairs), cc, map_memo_local, ws)
                    end
                end
                if hi >= lo
                    @inbounds for idx in lo:hi
                        u = active_data[idx]
                        m = mult[u]
                        Mi_to_Mu = _indicator_memo_get(map_memo_local, n, i, u)::Matrix{K}
                        @views mul!(rows[r:r+m-1, :], Linv[u], Mi_to_Mu)
                        r += m
                    end
                end
            end
            @assert r == Edims[i] + 1
        end
    end
    iota = PMorphism{K}(M, E, comps)

    return E, iota, materialize_gens ? _injective_generators_view(plan) : plan
end

"""
    injective_hull(M::PModule{K}; cache=nothing, output=:full, threads=(Threads.nthreads() > 1))
        -> InjectiveHullResult
    injective_hull(M::PModule{K}; cache=nothing, output=:summary, threads=(Threads.nthreads() > 1))
        -> NamedTuple

Build the one-step injective hull `M -> E`.

Mathematically, this constructs an injective module `E^0` together with a
coaugmentation `M -> E^0` that begins the downset/injective resolution of `M`.

The returned [`InjectiveHullResult`](@ref) stores:
- the hull module `E^0`,
- the coaugmentation `M -> E^0`,
- generator data, exposed lazily by default.

The result iterates as `(E, iota, gens_at_E)` for compatibility with tuple
destructuring. Internal callers that need the packed `_InjectiveGeneratorPlan`
should call [`_injective_hull`](@ref) directly with `materialize_gens=false`.

Use `output=:summary` for the cheap/default inspection path when you only want:
- the side (`:injective`),
- field/poset metadata,
- generator counts,
- storage/laziness information.

Use `output=:full` when you need the hull module, coaugmentation map, or lazy
generator wrapper for further algebraic work. Summary output is for inspection;
full output is for computation.
"""
function injective_hull(
    M::PModule{K};
    cache::Union{Nothing,CoverCache}=nothing,
    map_memo::Union{Nothing,AbstractVector{Union{Nothing,Matrix{K}}}}=nothing,
    workspace::Union{Nothing,_ResolutionWorkspace{K}}=nothing,
    output::Symbol=:full,
    threads::Bool=(Threads.nthreads() > 1),
) where {K}
    E, iota, gens = _injective_hull(M; cache=cache, map_memo=map_memo, workspace=workspace, threads=threads)
    result = InjectiveHullResult(E, iota, gens, false, false)
    return _indicator_resolution_with_output(:injective_hull, result, hull_summary, output)
end

"""
    upset_presentation_one_step(Hfringe::FringeModule)
Compute the one-step upset presentation (Def. 6.4.1):
    F1 --d1--> F0 --pi0-->> M,
and return the lightweight wrapper `UpsetPresentation{K}(P, U0, U1, delta, H)`.
"""
function upset_presentation_one_step(H::FiniteFringe.FringeModule{K}) where {K}
    M = pmodule_from_fringe(H)          # internal PModule over K
    cc = _get_cover_cache(M.Q)
    # First step: projective cover of M
    F0, pi0, gens_at_F0 = projective_cover(M; cache=cc)
    # Kernel K1 with inclusion i1 : K1 \into F0
    K1, i1 = kernel_with_inclusion(pi0; cache=cc)
    # Second projective cover (of K1)
    F1, pi1, gens_at_F1 = projective_cover(K1; cache=cc)
    # Differential d1 = i1 \circ pi1 : F1 \to F0
    comps = [i1.comps[i] * pi1.comps[i] for i in 1:length(M.dims)]
    d1 = PMorphism{K}(F1, F0, comps)

    # Build indicator wrapper:
    # U0: list of principal upsets, one per generator in gens_at_F0[p]
    P = M.Q
    PT = typeof(P)
    U0 = Upset{PT}[]
    for p in 1:nvertices(P)
        for _ in gens_at_F0[p]
            push!(U0, principal_upset(P, p))
        end
    end
    U1 = Upset{PT}[]
    for p in 1:nvertices(P)
        for _ in gens_at_F1[p]
            push!(U1, principal_upset(P, p))
        end
    end

    # scalar delta block: one entry for each pair (theta in U1, lambda in U0) if ptheta <= plambda.
    # Extract the scalar from d1 at the minimal vertex i = plambda.
    m1 = length(U1); m0 = length(U0)
    delta = spzeros(K, m1, m0)
    # local index maps: at vertex i, Fk_i basis is "all generators at p with p <= i" in (p increasing) order.
    # Build offsets to find local coordinates.
    function local_index_list(gens_at)
        # return vector of vectors L[i] listing global generator indices active at vertex i
        L = Vector{Vector{Tuple{Int,Int}}}(undef, nvertices(P))
        for i in 1:nvertices(P)
            L[i] = Tuple{Int,Int}[]
            for p in downset_indices(P, i)
                append!(L[i], gens_at[p])
            end
        end
        L
    end
    L0 = local_index_list(gens_at_F0)
    L1 = local_index_list(gens_at_F1)

    # Build a map from "global generator number in U0/U1" to its vertex p and position j in M_p
    globalU0 = Tuple{Int,Int}[]  # (p,j)
    for p in 1:nvertices(P); append!(globalU0, gens_at_F0[p]); end
    globalU1 = Tuple{Int,Int}[]
    for p in 1:nvertices(P); append!(globalU1, gens_at_F1[p]); end

    # helper: find local column index of global generator g=(p,j) at vertex i
    function local_col_of(L, g::Tuple{Int,Int}, i::Int)
        for (c, gg) in enumerate(L[i])
            if gg == g; return c; end
        end
        return 0
    end

    for (lambda, (plambda, jlambda)) in enumerate(globalU0)
        for (theta, (ptheta, jtheta)) in enumerate(globalU1)
            if leq(P, plambda, ptheta)   # containment for principal upsets: Up(ptheta) subseteq Up(plambda)
                i = ptheta              # read at the minimal vertex where the domain generator exists
                col = local_col_of(L1, (ptheta, jtheta), i)
                row = local_col_of(L0, (plambda, jlambda), i)
                if col > 0 && row > 0
                    val = d1.comps[i][row, col]
                    if val != 0
                        delta[theta, lambda] = val
                    end
                end
            end

        end
    end

    UpsetPresentation{K}(P, U0, U1, delta, H)
end

# ------------------------ downset copresentation (Def. 6.4.2) ------------------------

# The dual story: compute an injective hull E0 of M and the next step E1 with rho0 : E0 \to E1.
# For brevity we implement the duals by applying the above steps to M^op using
# down-closures/up-closures symmetry. Here we implement directly degreewise.

"Outgoing coimage at u to immediate successors; basis for the span of maps M_u to oplus_{u<v} M_v."
function _outgoing_span_basis(M::PModule{K}, u::Int; cache::Union{Nothing,CoverCache}=nothing) where {K}
    field = M.field
    cc = (cache === nothing ? _get_cover_cache(M.Q) : cache)
    su = _succs(cc, u)
    du = M.dims[u]

    if isempty(su) || du == 0
        return zeros(K, 0, du)
    end

    # Stack outgoing maps B : M_u -> (direct sum over cover successors).
    tot = 0
    @inbounds for j in eachindex(su)
        tot += M.dims[su[j]]
    end
    if tot == 0
        return zeros(K, 0, du)
    end
    maps_u = M.edge_maps.maps_to_succ[u]
    if length(su) == 1
        A = maps_u[1]
        return size(A, 1) == 0 ? zeros(K, 0, du) :
               transpose(FieldLinAlg.colspace(field, transpose(A)))
    end
    cols = (transpose(A) for A in maps_u)
    if !_indicator_use_incremental_union(field, du, tot, length(su), Val(:downset))
        return transpose(_colspace_union_dense(field, du, tot, cols, K))
    end
    return transpose(_colspace_union_incremental(field, du, cols, K))
end


"Essential socle dimension at u = dim(M_u) - rank(outgoing span) (dual of generators)."
function _socle_count(M::PModule{K}, u::Int) where {K}
    field = M.field
    S = _outgoing_span_basis(M, u)
    M.dims[u] - FieldLinAlg.rank(field, S)
end

"""
    downset_copresentation_one_step(Hfringe::FringeModule)

Compute the one-step downset **copresentation** (Def. 6.4(2)):
    M = ker(rho : E^0 to E^1),
with E^0 and E^1 expressed as direct sums of principal downsets and rho assembled
from the actual vertexwise maps, not just from the partial order.  Steps:

1. Build the injective (downset) hull iota0 : M into E^0.
2. Form C = coker(iota0) as a P-module together with q : E^0 to C.
3. Build the injective (downset) hull j : C into E^1 and set rho0 = j circ q : E^0 -> E^1.
4. Read scalar entries of rho at minimal vertices, as for delta on the upset side.
"""
function downset_copresentation_one_step(H::FiniteFringe.FringeModule{K}) where {K}
    # Convert fringe to internal PModule over K
    M = pmodule_from_fringe(H)
    Q = M.Q; n = nvertices(Q)
    PT = typeof(Q)
    cc = _get_cover_cache(Q)

    # (1) Injective hull of M: E0 with inclusion iota0
    E0, iota0, gens_at_E0 = _injective_hull(M; cache=cc)

    # (2) Degreewise cokernel C and the quotient q : E0 \to C
    C, q = _cokernel_module(iota0; cache=cc)

    # (3) Injective hull of C: E1 with inclusion j : C \into E1
    E1, j, gens_at_E1 = _injective_hull(C; cache=cc)

    # Compose to get rho0 : E0 \to E1 (at each vertex i: rho0[i] = j[i] * q[i])
    comps_rho0 = [ j.comps[i] * q.comps[i] for i in 1:n ]
    rho0 = PMorphism{K}(E0, E1, comps_rho0)

    # (4) Assemble the indicator wrapper: labels D0, D1 and the scalar block rho.
    # D0/D1 each contain one principal downset for every generator (u,j) chosen above.
    D0 = Downset{PT}[]
    for u in 1:n, _ in gens_at_E0[u]
        push!(D0, principal_downset(Q, u))
    end
    D1 = Downset{PT}[]
    for u in 1:n, _ in gens_at_E1[u]
        push!(D1, principal_downset(Q, u))
    end

    # Local index lists: at vertex i, the active generators are those born at u with i <= u.
    function _local_index_list_D(gens_at)
        L = Vector{Vector{Tuple{Int,Int}}}(undef, n)
        for i in 1:n
            lst = Tuple{Int,Int}[]
            for u in upset_indices(Q, i)
                append!(lst, gens_at[u])
            end
            L[i] = lst
        end
        L
    end
    L0 = _local_index_list_D(gens_at_E0)
    L1 = _local_index_list_D(gens_at_E1)

    # Global enumerations (lambda in D0, theta in D1) with their birth vertices u_lambda, u_theta.
    globalD0 = Tuple{Int,Int}[]; for u in 1:n; append!(globalD0, gens_at_E0[u]); end
    globalD1 = Tuple{Int,Int}[]; for u in 1:n; append!(globalD1, gens_at_E1[u]); end

    # Helper to find the local column index of a global generator at vertex i
    function _local_col_of(L, g::Tuple{Int,Int}, i::Int)
        for (c, gg) in enumerate(L[i])
            if gg == g; return c; end
        end
        return 0
    end

    # Assemble the scalar monomial matrix rho by reading the minimal vertex i = u_theta.
    m1 = length(globalD1); m0 = length(globalD0)
    rho = spzeros(K, m1, m0)

    for (lambda, (ulambda, jlambda)) in enumerate(globalD0)
        for (theta, (utheta, jtheta)) in enumerate(globalD1)
            if leq(Q, utheta, ulambda) # D(utheta) subseteq D(ulambda)
                i   = utheta
                col = _local_col_of(L0, (ulambda, jlambda), i)
                row = _local_col_of(L1, (utheta, jtheta), i)
                if col > 0 && row > 0
                    val = rho0.comps[i][row, col]
                    if val != 0
                        rho[theta, lambda] = val
                    end
                end
            end
        end
    end

    return DownsetCopresentation{K}(Q, D0, D1, rho, H)
end



"""
    prune_zero_relations(F::UpsetPresentation{K}) -> UpsetPresentation{K}

Remove rows of `delta` that are identically zero (redundant relations) and drop the
corresponding entries of `U1`. The cokernel is unchanged.
"""
function prune_zero_relations(F::UpsetPresentation{K}) where {K}
    m1, m0 = size(F.delta)
    keep = trues(m1)
    # mark zero rows
    rows, _, _ = findnz(F.delta)
    seen = falses(m1); @inbounds for r in rows; seen[r] = true; end
    @inbounds for r in 1:m1
        if !seen[r]; keep[r] = false; end
    end
    new_U1 = [F.U1[i] for i in 1:m1 if keep[i]]
    new_delta = F.delta[keep, :]
    UpsetPresentation{K}(F.P, F.U0, new_U1, new_delta, F.H)
end

"""
    cancel_isolated_unit_pairs(F::UpsetPresentation{K}) -> UpsetPresentation{K}

Iteratively cancels isolated nonzero entries `delta[theta,lambda]` for which:
  * the theta-th row has exactly that one nonzero,
  * the lambda-th column has exactly that one nonzero, and
  * U1[theta] == U0[lambda] as Upsets (principal upsets match).

Each cancellation removes one generator in U0 and one relation in U1 without
changing the cokernel.
"""
function cancel_isolated_unit_pairs(F::UpsetPresentation{K}) where {K}
    P, U0, U1, Delta = F.P, F.U0, F.U1, F.delta
    while true
        m1, m0 = size(Delta)
        rows, cols, _ = findnz(Delta)
        # count nonzeros per row/col
        rcount = zeros(Int, m1)
        ccount = zeros(Int, m0)
        @inbounds for k in eachindex(rows)
            rcount[rows[k]] += 1; ccount[cols[k]] += 1
        end
        # search an isolated pair with matching principal upsets
        found = false
        theta = 0; lambda = 0
        @inbounds for k in eachindex(rows)
            r = rows[k]; c = cols[k]
            if rcount[r] == 1 && ccount[c] == 1
                # require identical principal upsets
                if U1[r].P === U0[c].P && U1[r].mask == U0[c].mask
                    theta, lambda = r, c; found = true; break
                end
            end
        end
        if !found; break; end
        # remove row theta and column lambda
        keep_rows = trues(m1); keep_rows[theta] = false
        keep_cols = trues(m0); keep_cols[lambda] = false
        U1 = [U1[i] for i in 1:m1 if keep_rows[i]]
        U0 = [U0[j] for j in 1:m0 if keep_cols[j]]
        Delta = Delta[keep_rows, keep_cols]
    end
    UpsetPresentation{K}(P, U0, U1, Delta, F.H)
end

"""
    minimal_upset_presentation_one_step(H::FiniteFringe.FringeModule)
        -> UpsetPresentation{K}

Build a one-step upset presentation and apply safe minimality passes:
1) drop zero relations; 2) cancel isolated isomorphism pairs.
"""
function minimal_upset_presentation_one_step(H::FiniteFringe.FringeModule{K}) where {K}
    F = upset_presentation_one_step(H)     # existing builder
    F = prune_zero_relations(F)
    F = cancel_isolated_unit_pairs(F)
    return F
end


"""
    prune_unused_targets(E::DownsetCopresentation{K}) -> DownsetCopresentation{K}

Drop rows of `rho` that are identically zero (unused target summands in E^1). The kernel is unchanged.
"""
function prune_unused_targets(E::DownsetCopresentation{K}) where {K}
    m1, m0 = size(E.rho)
    keep = trues(m1)
    rows, _, _ = findnz(E.rho)
    seen = falses(m1); @inbounds for r in rows; seen[r] = true; end
    @inbounds for r in 1:m1
        if !seen[r]; keep[r] = false; end
    end
    new_D1 = [E.D1[i] for i in 1:m1 if keep[i]]
    new_rho = E.rho[keep, :]
    DownsetCopresentation{K}(E.P, E.D0, new_D1, new_rho, E.H)
end

"""
    cancel_isolated_unit_pairs(E::DownsetCopresentation{K}) -> DownsetCopresentation{K}

Iteratively cancels isolated nonzero entries `rho[theta,lambda]` with matching principal downsets
(D1[theta] == D0[lambda]) and unique in their row/column.
"""
function cancel_isolated_unit_pairs(E::DownsetCopresentation{K}) where {K}
    P, D0, D1, R = E.P, E.D0, E.D1, E.rho
    while true
        m1, m0 = size(R)
        rows, cols, _ = findnz(R)
        rcount = zeros(Int, m1)
        ccount = zeros(Int, m0)
        @inbounds for k in eachindex(rows)
            rcount[rows[k]] += 1; ccount[cols[k]] += 1
        end
        found = false; theta = 0; lambda = 0
        @inbounds for k in eachindex(rows)
            r = rows[k]; c = cols[k]
            if rcount[r] == 1 && ccount[c] == 1
                if D1[r].P === D0[c].P && D1[r].mask == D0[c].mask
                    theta, lambda = r, c; found = true; break
                end
            end
        end
        if !found; break; end
        keep_rows = trues(m1); keep_rows[theta] = false
        keep_cols = trues(m0); keep_cols[lambda] = false
        D1 = [D1[i] for i in 1:m1 if keep_rows[i]]
        D0 = [D0[j] for j in 1:m0 if keep_cols[j]]
        R = R[keep_rows, keep_cols]
    end
    DownsetCopresentation{K}(P, D0, D1, R, E.H)
end

"""
    minimal_downset_copresentation_one_step(H::FiniteFringe.FringeModule)
        -> DownsetCopresentation{K}

Build a one-step downset copresentation and apply safe minimality passes:
1) drop zero target rows; 2) cancel isolated isomorphism pairs.
"""
function minimal_downset_copresentation_one_step(H::FiniteFringe.FringeModule)
    E = downset_copresentation_one_step(H)
    E = prune_unused_targets(E)
    E = cancel_isolated_unit_pairs(E)
    return E
end



# =============================================================================
# Longer indicator resolutions and high-level Ext driver
# =============================================================================
# We expose:
#   * upset_resolution(H; maxlen)     -> (F, dF)
#   * downset_resolution(H; maxlen)   -> (E, dE)
#   * indicator_resolutions(HM, HN; maxlen) -> (F, dF, E, dE)
#
# The outputs (F, dF, E, dE) are exactly the shapes expected by
# HomExt.build_hom_tot_complex / HomExt.ext_dims_via_resolutions:
#   - F is a Vector{UpsetPresentation{K}} with F[a+1].U0 = U_a
#   - dF[a] is the sparse delta_a : U_a <- U_{a+1}  (shape |U_{a+1}| x |U_a|)
#   - E is a Vector{DownsetCopresentation{K}} with E[b+1].D0 = D_b
#   - dE[b] is the sparse rho_b : D_b -> D_{b+1}    (shape |D_{b+1}| x |D_b|)
#
# Construction mirrors section 6.1 and the one-step routines already present.
# =============================================================================

# ------------------------------ small helpers --------------------------------

# Build the list of principal upsets from per-vertex generator labels returned by
# projective_cover: gens_at[v] is a vector of pairs (p, j).  Each pair contributes
# one principal upset at vertex p.
function _principal_upsets_from_gens(P::PT,
                                     gens_at::AbstractVector{<:AbstractVector{Tuple{Int,Int}}}) where {PT<:AbstractPoset}
    n = nvertices(P)
    U_at = Vector{Upset{PT}}(undef, n)
    @inbounds for p in 1:n
        U_at[p] = principal_upset(P, p)
    end
    U = Upset{PT}[]
    for p in 1:n
        for _ in gens_at[p]
            push!(U, U_at[p])
        end
    end
    return U
end

function _principal_upsets_from_plan(P::PT,
                                     plan::_ProjectiveGeneratorPlan) where {PT<:AbstractPoset}
    n = nvertices(P)
    U_at = Vector{Upset{PT}}(undef, n)
    @inbounds for p in 1:n
        U_at[p] = principal_upset(P, p)
    end
    U = Upset{PT}[]
    @inbounds for p in 1:n
        for _ in 1:plan.counts[p]
            push!(U, U_at[p])
        end
    end
    return U
end

@inline _principal_upsets_from_gens(P::PT, plan::_ProjectiveGeneratorPlan) where {PT<:AbstractPoset} =
    _principal_upsets_from_plan(P, plan)

# Build the list of principal downsets from per-vertex labels returned by _injective_hull
function _principal_downsets_from_gens(P::PT,
                                       gens_at::AbstractVector{<:AbstractVector{Tuple{Int,Int}}}) where {PT<:AbstractPoset}
    n = nvertices(P)
    D_at = Vector{Downset{PT}}(undef, n)
    @inbounds for u in 1:n
        D_at[u] = principal_downset(P, u)
    end
    D = Downset{PT}[]
    for u in 1:n
        for _ in gens_at[u]
            push!(D, D_at[u])
        end
    end
    return D
end

@inline _principal_downsets_from_gens(P::PT, plan::_InjectiveGeneratorPlan) where {PT<:AbstractPoset} =
    _principal_downsets_from_plan(P, plan)

# Local basis index lists used by Workflow presentation conversions.
function _local_index_list_up(
    P::AbstractPoset,
    gens_at::AbstractVector{<:AbstractVector{Tuple{Int,Int}}},
)
    n = nvertices(P)
    L = Vector{Vector{Tuple{Int,Int}}}(undef, n)
    @inbounds for i in 1:n
        lst = Tuple{Int,Int}[]
        for p in downset_indices(P, i)
            append!(lst, gens_at[p])
        end
        L[i] = lst
    end
    return L
end

function _local_index_list_down(
    P::AbstractPoset,
    gens_at::AbstractVector{<:AbstractVector{Tuple{Int,Int}}},
)
    n = nvertices(P)
    L = Vector{Vector{Tuple{Int,Int}}}(undef, n)
    @inbounds for i in 1:n
        lst = Tuple{Int,Int}[]
        for u in upset_indices(P, i)
            append!(lst, gens_at[u])
        end
        L[i] = lst
    end
    return L
end

"""
    fringe_presentation(M::PModule{K}) -> FringeModule{K}

Construct a canonical fringe presentation whose image recovers `M`.
"""
function _fringe_presentation_scalar(M::PModule{K}) where {K}
    P = M.Q
    field = M.field
    F0, pi0, gens_at_F0 = projective_cover(M)
    E0, iota0, gens_at_E0 = _injective_hull(M)

    comps = Vector{Matrix{K}}(undef, nvertices(P))
    for i in 1:nvertices(P)
        comps[i] = iota0.comps[i] * pi0.comps[i]
    end
    f = PMorphism{K}(F0, E0, comps)

    U = _principal_upsets_from_gens(P, gens_at_F0)
    D = _principal_downsets_from_gens(P, gens_at_E0)

    L0 = _local_index_list_up(P, gens_at_F0)
    L1 = _local_index_list_down(P, gens_at_E0)

    globalU = Tuple{Int,Int}[]
    for p in 1:nvertices(P)
        append!(globalU, gens_at_F0[p])
    end
    globalD = Tuple{Int,Int}[]
    for u in 1:nvertices(P)
        append!(globalD, gens_at_E0[u])
    end

    function local_pos(L, g::Tuple{Int,Int}, i::Int)
        for (j, gg) in enumerate(L[i])
            gg == g && return j
        end
        return 0
    end

    phi = spzeros(K, length(D), length(U))
    for (lambda, (plambda, jlambda)) in enumerate(globalU)
        for (theta, (ptheta, jtheta)) in enumerate(globalD)
            if leq(P, plambda, ptheta)
                i = plambda
                col = local_pos(L0, (plambda, jlambda), i)
                row = local_pos(L1, (ptheta, jtheta), i)
                if row > 0 && col > 0
                    phi[theta, lambda] = f.comps[i][row, col]
                end
            end
        end
    end

    return FiniteFringe.FringeModule{K}(P, U, D, phi; field=field)
end

function fringe_presentation(M::PModule{K}) where {K}
    return _fringe_presentation_scalar(M)
end

@inline function _truncate_steps(maxlen::Union{Int,Nothing}, available::Int)
    return maxlen === nothing ? available : min(available, maxlen)
end

@inline function _upset_birth_block_plan(P::AbstractPoset)
    return _cached_birth_plans(P).downset
end

@inline function _downset_birth_block_plan(P::AbstractPoset)
    return _cached_birth_plans(P).upset
end

@inline function _package_upset_from_state(
    ::Type{K},
    P::PT,
    st::_UpsetPrefixState{K},
    maxlen::Union{Int,Nothing},
) where {K,PT<:AbstractPoset}
    nd = _truncate_steps(maxlen, length(st.dF))
    ng = nd + 1
    U_by_a = Vector{Vector{Upset{PT}}}(undef, ng)
    @inbounds for a in 1:ng
        U_by_a[a] = _principal_upsets_from_plan(P, st.gens_by_a[a])
    end
    U0 = U_by_a[1]
    U1 = ng > 1 ? U_by_a[2] : Upset{PT}[]
    delta = ng > 1 ? st.dF[1] : spzeros(K, 0, length(U0))
    F1 = UpsetPresentation{K}(P, U0, U1, delta, nothing)
    F = Vector{typeof(F1)}(undef, ng)
    F[1] = F1
    @inbounds for a in 2:ng
        U0 = U_by_a[a]
        if a < ng
            U1 = U_by_a[a + 1]
            delta = st.dF[a]
        else
            U1 = Upset{PT}[]
            delta = spzeros(K, 0, length(U0))
        end
        F[a] = UpsetPresentation{K}(P, U0, U1, delta, nothing)
    end
    return F, st.dF[1:nd]
end

# ------------------------------ upset resolution ------------------------------

"""
    upset_resolution(H::FiniteFringe.FringeModule{K}; maxlen=nothing)

Compute an upset (projective) indicator resolution of the fringe module `H`.
`maxlen` is a cutoff on the number of differentials computed (it does not pad output).
"""
function _extend_upset_prefix_state!(
    st::_UpsetPrefixState{K},
    P::AbstractPoset,
    birth_plan::Union{Nothing,Vector{Vector{Int}}},
    cc::CoverCache,
    memo_pool::IdDict{Any,Vector{Union{Nothing,Matrix{K}}}},
    ws::_ResolutionWorkspace{K},
    target_steps::Int,
    threads::Bool,
) where {K}
    n = nvertices(P)
    use_vertex_cache = _indicator_use_incremental_vertex_cache(
        st.curr_pi.dom.field,
        n,
        max(sum(st.curr_pi.dom.dims), sum(st.curr_pi.cod.dims)),
        Val(:upset),
    )
    while !st.done && length(st.dF) < target_steps
        Kmod, iota = kernel_with_inclusion(
            st.curr_pi;
            cache=cc,
            incremental_cache=(use_vertex_cache ? ws.kernel_vertex_cache : nothing),
        )
        if sum(Kmod.dims) == 0
            st.done = true
            break
        end
        map_memo_next = _indicator_memo_for_module!(memo_pool, Kmod)
        Fnext, pinext, gens_at_next = projective_cover(
            Kmod;
            cache=cc,
            map_memo=map_memo_next,
            workspace=ws,
            materialize_gens=false,
            threads=threads,
        )

        iota_comps = iota.comps
        pinext_comps = pinext.comps
        counts_prev = st.curr_gens.counts
        counts_next = gens_at_next.counts
        gid_prev_starts = st.curr_gens.gid_starts
        gid_next_starts = gens_at_next.gid_starts
        total_prev = st.curr_gens.total
        total_next = gens_at_next.total
        use_active_delta = _INDICATOR_UPSET_ACTIVE_SOURCE_DELTA[]
        delta = if use_active_delta
            _upset_delta_from_active_sources!(
                ws,
                gens_at_next.active_birth_vertices,
                st.curr_gens.active_sources,
                iota_comps,
                pinext_comps,
                counts_prev,
                gid_prev_starts,
                counts_next,
                gid_next_starts,
                gens_at_next.self_starts_down,
                total_next,
                total_prev,
                threads,
            )
        else
            starts_next = _fill_birth_self_start_down!(ws.self_start, P, counts_next)
            _upset_delta_from_birth_plan!(
                ws,
                n,
                birth_plan::Vector{Vector{Int}},
                iota_comps,
                pinext_comps,
                counts_prev,
                gid_prev_starts,
                counts_next,
                gid_next_starts,
                starts_next,
                total_next,
                total_prev,
                threads,
            )
        end
        push!(st.dF, delta)
        push!(st.gens_by_a, gens_at_next)
        st.curr_pi = pinext
        st.curr_gens = gens_at_next
    end
    return st
end

"""
    upset_resolution(H::FiniteFringe.FringeModule{K}; maxlen=nothing, output=:full)
    upset_resolution(H::FiniteFringe.FringeModule{K}; maxlen=nothing, output=:summary)

Compute a projective/upset indicator resolution of the fringe module `H`.

This is the canonical high-level upset-side entrypoint when you start from a
fringe presentation rather than an already-built `PModule`.

`maxlen` is a cutoff on the number of differentials computed. Use
`output=:summary` when you want a cheap overview of the resolution without
touching the full presentation sequence.

Use `output=:full` when you want the wrapped presentation sequence and
augmentation map for further algebra. Use `output=:summary` for notebook/REPL
inspection or when you only need side/field/length/generator-count metadata.
"""
function upset_resolution(H::FiniteFringe.FringeModule{K};
                          maxlen::Union{Int,Nothing}=nothing,
                          output::Symbol=:full,
                          cache::Union{Nothing,CoverCache}=nothing,
                          map_memo::Union{Nothing,AbstractVector{Union{Nothing,Matrix{K}}}}=nothing,
                          map_memo_pool::Union{Nothing,IdDict{Any,Vector{Union{Nothing,Matrix{K}}}}}=nothing,
                          workspace::Union{Nothing,_ResolutionWorkspace{K}}=nothing,
                          threads::Bool = (Threads.nthreads() > 1)) where {K}
    return upset_resolution(
        pmodule_from_fringe(H);
        maxlen=maxlen,
        output=output,
        cache=cache,
        map_memo=map_memo,
        map_memo_pool=map_memo_pool,
        workspace=workspace,
        threads=threads,
    )
end


"""
upset_resolution(M::PModule{K}; maxlen::Union{Int,Nothing}=nothing, output=:full)
        -> UpsetResolutionResult
upset_resolution(M::PModule{K}; maxlen::Union{Int,Nothing}=nothing, output=:summary)
        -> NamedTuple

Overload of `upset_resolution` for an already-constructed finite-poset module
`M`.

The returned [`UpsetResolutionResult`](@ref) stores the upset presentation
objects in degree order, the differential matrices, the packed generator plans
backing each stage, and the augmentation `F_0 -> M`. It iterates as `(F, dF)`
for compatibility with existing tuple-destructuring code.

Use `output=:summary` for the cheapest mathematically meaningful view when you
only need side/field/length/generator-count metadata.

Mathematically, `output=:summary` answers "what projective/upset resolution did
I build?" while `output=:full` answers "give me the actual presentation objects
and differentials so I can compute with them."
"""
function upset_resolution(M::PModule{K};
                          maxlen::Union{Int,Nothing}=nothing,
                          output::Symbol=:full,
                          cache::Union{Nothing,CoverCache}=nothing,
                          map_memo::Union{Nothing,AbstractVector{Union{Nothing,Matrix{K}}}}=nothing,
                          map_memo_pool::Union{Nothing,IdDict{Any,Vector{Union{Nothing,Matrix{K}}}}}=nothing,
                          workspace::Union{Nothing,_ResolutionWorkspace{K}}=nothing,
                          threads::Bool = (Threads.nthreads() > 1)) where {K}
    field = M.field
    P = M.Q
    n = nvertices(P)
    total_dims = sum(M.dims)
    build_cache!(P; cover=true, updown=true)
    cc = cache === nothing ? _get_cover_cache(P) : cache
    use_vertex_cache = _indicator_use_incremental_vertex_cache(field, n, total_dims, Val(:upset))
    memo_pool = map_memo_pool === nothing ? IdDict{Any,Vector{Union{Nothing,Matrix{K}}}}() : map_memo_pool
    if map_memo !== nothing
        memo_pool[M] = map_memo
    end
    map_memo_local = _indicator_memo_for_module!(memo_pool, M)
    ws = workspace === nothing ? _new_resolution_workspace(K, n) : workspace
    _workspace_prepare!(ws, n)
    use_active_delta = _INDICATOR_UPSET_ACTIVE_SOURCE_DELTA[]
    birth_plan = use_active_delta ? nothing : _upset_birth_block_plan(P)

    use_prefix_cache = maxlen !== nothing &&
                       _indicator_use_prefix_cache(field, n, total_dims, Int(maxlen), Val(:upset)) &&
                       cache === nothing &&
                       map_memo === nothing &&
                       map_memo_pool === nothing &&
                       workspace === nothing

    if use_prefix_cache
        target_steps = Int(maxlen)
        fam = _indicator_prefix_family(K)
        shard = _indicator_shard_index(M, length(fam.upset_shards))
        Base.lock(fam.upset_locks[shard])
        try
            st_any = get(fam.upset_shards[shard], M, nothing)
            st = if st_any === nothing || !(st_any isa _UpsetPrefixState{K})
                F0, pi0, gens_at_F0 = projective_cover(
                    M;
                    cache=cc,
                    map_memo=map_memo_local,
                    workspace=ws,
                    materialize_gens=false,
                    threads=threads,
                )
                _ = F0
                st_new = _UpsetPrefixState{K}(
                    _ProjectiveGeneratorPlan[gens_at_F0],
                    SparseMatrixCSC{K,Int}[],
                    pi0,
                    pi0,
                    gens_at_F0,
                    false,
                )
                fam.upset_shards[shard][M] = st_new
                st_new
            else
                st_any::_UpsetPrefixState{K}
            end

            _extend_upset_prefix_state!(st, P, birth_plan, cc, memo_pool, ws, target_steps, threads)
            F, dF = _package_upset_from_state(K, P, st, maxlen)
            result = UpsetResolutionResult(
                F,
                dF,
                st.gens_by_a[1:(length(dF) + 1)],
                st.pi0,
                M,
                false,
                false,
            )
            return _indicator_resolution_with_output(:upset_resolution, result, resolution_summary, output)
        finally
            Base.unlock(fam.upset_locks[shard])
        end
    end

    # First projective cover: F0 --pi0--> M, with labels gens_at_F0
    F0, pi0, gens_at_F0 = projective_cover(
        M;
        cache=cc,
        map_memo=map_memo_local,
        workspace=ws,
        materialize_gens=false,
        threads=threads,
    )

    # Keep generator labels through the loop and materialize principal upsets only once at packaging.
    gens_by_a = _ProjectiveGeneratorPlan[]
    push!(gens_by_a, gens_at_F0)

    dF = Vector{SparseMatrixCSC{K,Int}}()

    # Iteration state
    curr_dom  = F0
    curr_pi   = pi0
    curr_gens = gens_at_F0
    steps = 0

    while true
        if maxlen !== nothing && steps >= maxlen
            break
        end

        # Next kernel and projective cover
        Kmod, iota = kernel_with_inclusion(
            curr_pi;
            cache=cc,
            incremental_cache=(use_vertex_cache ? ws.kernel_vertex_cache : nothing),
        )   # ker(F_prev -> ...)
        if sum(Kmod.dims) == 0
            break
        end
        map_memo_next = _indicator_memo_for_module!(memo_pool, Kmod)
        Fnext, pinext, gens_at_next = projective_cover(
            Kmod;
            cache=cc,
            map_memo=map_memo_next,
            workspace=ws,
            materialize_gens=false,
            threads=threads,
        )

        # d = iota o pinext : Fnext -> curr_dom
        # Build sparse delta entries directly from required comparable blocks.
        iota_comps = iota.comps
        pinext_comps = pinext.comps

        # Dense local-index starts (born-at-vertex blocks).
        counts_prev = curr_gens.counts
        counts_next = gens_at_next.counts
        gid_prev_starts = curr_gens.gid_starts
        gid_next_starts = gens_at_next.gid_starts
        total_prev = curr_gens.total
        total_next = gens_at_next.total
        delta = if use_active_delta
            _upset_delta_from_active_sources!(
                ws,
                gens_at_next.active_birth_vertices,
                curr_gens.active_sources,
                iota_comps,
                pinext_comps,
                counts_prev,
                gid_prev_starts,
                counts_next,
                gid_next_starts,
                gens_at_next.self_starts_down,
                total_next,
                total_prev,
                threads,
            )
        else
            starts_next = _fill_birth_self_start_down!(ws.self_start, P, counts_next)
            _upset_delta_from_birth_plan!(
                ws,
                n,
                birth_plan::Vector{Vector{Int}},
                iota_comps,
                pinext_comps,
                counts_prev,
                gid_prev_starts,
                counts_next,
                gid_next_starts,
                starts_next,
                total_next,
                total_prev,
                threads,
            )
        end

        push!(dF, delta)
        push!(gens_by_a, gens_at_next)

        # Advance
        curr_dom  = Fnext
        curr_pi   = pinext
        curr_gens = gens_at_next
        steps += 1
    end

    # Materialize principal upsets only once when packaging output.
    PT = typeof(P)
    U_by_a = Vector{Vector{Upset{PT}}}(undef, length(gens_by_a))
    @inbounds for a in eachindex(gens_by_a)
        U_by_a[a] = _principal_upsets_from_plan(P, gens_by_a[a])
    end

    # Package as UpsetPresentation list.
    U0 = U_by_a[1]
    U1 = length(gens_by_a) > 1 ? U_by_a[2] : Upset{PT}[]
    delta = length(gens_by_a) > 1 ? dF[1] : spzeros(K, 0, length(U0))
    F1 = UpsetPresentation{K}(P, U0, U1, delta, nothing)
    F = Vector{typeof(F1)}(undef, length(gens_by_a))
    F[1] = F1
    for a in 2:length(gens_by_a)
        U0 = U_by_a[a]
        if a < length(gens_by_a)
            U1 = U_by_a[a + 1]
            delta = dF[a]
        else
            U1 = Upset{PT}[]
            delta = spzeros(K, 0, length(U0))
        end
        F[a] = UpsetPresentation{K}(P, U0, U1, delta, nothing)
    end

    result = UpsetResolutionResult(F, dF, gens_by_a, pi0, M, false, false)
    return _indicator_resolution_with_output(:upset_resolution, result, resolution_summary, output)
end


# ---------------------------- downset resolution ------------------------------

"""
    downset_resolution(H::FiniteFringe.FringeModule{K}; maxlen=nothing, output=:full)
    downset_resolution(H::FiniteFringe.FringeModule{K}; maxlen=nothing, output=:summary)

Compute a downset (injective) indicator resolution of the fringe module `H`.
`maxlen` is a cutoff on the number of differentials computed (it does not pad output).
Use `output=:summary` for a cheap overview rather than the full copresentation
sequence.

This is the canonical high-level injective-side entrypoint when you start from a
fringe presentation rather than an already-built `PModule`.

Use `output=:full` when you need the wrapped copresentation sequence and
coaugmentation map for further algebra. Use `output=:summary` when you only want
the mathematical summary of what was built.
"""
function downset_resolution(H::FiniteFringe.FringeModule{K};
                            maxlen::Union{Int,Nothing}=nothing,
                            output::Symbol=:full,
                            cache::Union{Nothing,CoverCache}=nothing,
                            map_memo::Union{Nothing,AbstractVector{Union{Nothing,Matrix{K}}}}=nothing,
                            map_memo_pool::Union{Nothing,IdDict{Any,Vector{Union{Nothing,Matrix{K}}}}}=nothing,
                            workspace::Union{Nothing,_ResolutionWorkspace{K}}=nothing,
                            threads::Bool = (Threads.nthreads() > 1)) where {K}
    return downset_resolution(
        pmodule_from_fringe(H);
        maxlen=maxlen,
        output=output,
        cache=cache,
        map_memo=map_memo,
        map_memo_pool=map_memo_pool,
        workspace=workspace,
        threads=threads,
    )
end


"""
downset_resolution(M::PModule{K}; maxlen::Union{Int,Nothing}=nothing, output=:full)
        -> DownsetResolutionResult
downset_resolution(M::PModule{K}; maxlen::Union{Int,Nothing}=nothing, output=:summary)
        -> NamedTuple

Overload of `downset_resolution` for an already-constructed finite-poset module
`M`.

The returned [`DownsetResolutionResult`](@ref) stores the downset
copresentation objects in degree order, the differential matrices, the packed
generator plans backing each stage, and the coaugmentation `M -> E^0`. It
iterates as `(E, dE)` for compatibility with existing tuple-destructuring code.

Use `output=:summary` for the cheapest mathematically meaningful view when you
only need side/field/length/generator-count metadata.

Mathematically, `output=:summary` answers "what injective/downset resolution did
I build?" while `output=:full` gives the actual copresentation objects and
differentials for computation.
"""
function downset_resolution(M::PModule{K};
                            maxlen::Union{Int,Nothing}=nothing,
                            output::Symbol=:full,
                            cache::Union{Nothing,CoverCache}=nothing,
                            map_memo::Union{Nothing,AbstractVector{Union{Nothing,Matrix{K}}}}=nothing,
                            map_memo_pool::Union{Nothing,IdDict{Any,Vector{Union{Nothing,Matrix{K}}}}}=nothing,
                            workspace::Union{Nothing,_ResolutionWorkspace{K}}=nothing,
                            threads::Bool = (Threads.nthreads() > 1)) where {K}
    field = M.field
    P = M.Q
    n = nvertices(P)
    build_cache!(P; cover=true, updown=true)
    cc = cache === nothing ? _get_cover_cache(P) : cache
    memo_pool = map_memo_pool === nothing ? IdDict{Any,Vector{Union{Nothing,Matrix{K}}}}() : map_memo_pool
    if map_memo !== nothing
        memo_pool[M] = map_memo
    end
    map_memo_local = _indicator_memo_for_module!(memo_pool, M)
    ws = workspace === nothing ? _new_resolution_workspace(K, n) : workspace
    _workspace_prepare!(ws, n)
    birth_plan = _downset_birth_block_plan(P)
    initial_support_vertices = _nonzero_dim_vertices(M.dims)
    initial_support_mask = _vertex_mask(n, initial_support_vertices)
    total_dims = sum(M.dims)
    use_transport_reuse = _indicator_use_downset_transport_reuse(field, n, total_dims, maxlen)
    use_injective_reuse = _indicator_use_downset_injective_reuse(field, n, total_dims, maxlen)
    transport_cache = use_transport_reuse ? _new_downset_cokernel_transport_cache(K, n) : nothing
    injective_cache = use_injective_reuse ? _new_injective_hull_reuse_cache(K, n) : nothing
    graph_lists = _cover_graph_lists(cc)

    # First injective hull: iota0 : M -> E0
    E0, iota0, gens_at_E0 = _injective_hull(
        M;
        cache=cc,
        map_memo=map_memo_local,
        workspace=ws,
        support_vertices=initial_support_vertices,
        graph_lists=graph_lists,
        reuse_cache=injective_cache,
        materialize_gens=false,
        threads=threads,
    )
    # Keep generator labels through the loop and materialize principal downsets only once at packaging.
    gens_by_b = _InjectiveGeneratorPlan[]
    push!(gens_by_b, gens_at_E0)

    dE = Vector{SparseMatrixCSC{K,Int}}()

    # First cokernel
    C, q = _cokernel_module(
        iota0;
        cache=cc,
        active_vertices=initial_support_vertices,
        active_mask=initial_support_mask,
        transport_cache=transport_cache,
        graph_lists=graph_lists,
    )

    # Iteration state
    prev_E    = E0
    prev_gens = gens_at_E0
    prev_q    = q
    prev_counts = gens_at_E0.mult
    prev_gid_starts = gens_at_E0.gid_starts
    prev_total = gens_at_E0.total
    support_vertices = _nonzero_dim_vertices(C.dims)
    support_mask = _vertex_mask(n, support_vertices)
    support_slots = _support_slot_vector(n, support_vertices)
    frontier_mask = transport_cache === nothing ? falses(n) : copy(transport_cache.frontier_mask)
    frontier_vertices = transport_cache === nothing ? Int[] : copy(transport_cache.frontier_vertices)
    changed_vertices = transport_cache === nothing ? Int[] : copy(transport_cache.changed_vertices)
    rho_pattern = nothing
    steps = 0

    while true
        if sum(C.dims) == 0
            break
        end
        if maxlen !== nothing && steps >= maxlen
            break
        end

        # Injective hull of the current cokernel: j : C -> E1
        map_memo_next = _indicator_memo_for_module!(memo_pool, C)
        hull_support_vertices, hull_support_mask = _downset_injective_support_vertices(
            cc,
            support_vertices,
            support_mask,
            prev_gens.active_socle_vertices,
            changed_vertices,
        )
        recompute_mask = injective_cache === nothing ? nothing :
            (_INDICATOR_DOWNSET_FRONTIER_VERTICES[] ?
                _downset_injective_recompute_mask(cc, hull_support_mask, frontier_vertices, changed_vertices) :
                _downset_injective_recompute_mask(cc, hull_support_mask, frontier_mask, changed_vertices))
        E1, j, gens_at_E1 = _injective_hull(
            C;
            cache=cc,
            map_memo=map_memo_next,
            workspace=ws,
            support_vertices=hull_support_vertices,
            graph_lists=graph_lists,
            reuse_cache=injective_cache,
            recompute_mask=recompute_mask,
            materialize_gens=false,
            threads=threads,
        )

        # rho_b = j o prev_q : prev_E -> E1
        # Build sparse rho entries directly from comparable birth blocks.
        counts_prev = prev_counts
        counts_next = gens_at_E1.mult
        gid_next_starts = gens_at_E1.gid_starts
        total_next = gens_at_E1.total
        starts_next = _fill_birth_self_start_up!(ws.self_start, P, counts_next)
        if rho_pattern === nothing || !_downset_rho_pattern_matches(rho_pattern, counts_prev, counts_next)
            rho_pattern = _build_downset_rho_pattern(
                birth_plan,
                counts_prev,
                prev_gid_starts,
                counts_next,
                gid_next_starts,
                starts_next,
                prev_total,
                total_next,
            )
        end
        Rh = if _INDICATOR_DOWNSET_RHO_VALUE_ONLY_REUSE[]
            _downset_rho_from_value_pattern!(ws, rho_pattern, j.comps, prev_q.comps, threads)
        else
            _downset_rho_from_pattern!(ws, rho_pattern, j.comps, prev_q.comps, threads)
        end

        push!(dE, Rh)
        push!(gens_by_b, gens_at_E1)

        # Next cokernel and advance
        C, q_next = _cokernel_module(
            j;
            cache=cc,
            active_vertices=support_vertices,
            active_mask=support_mask,
            transport_cache=transport_cache,
            graph_lists=graph_lists,
        )
        prev_E    = E1
        prev_gens = gens_at_E1
        prev_q    = q_next
        prev_counts = counts_next
        prev_gid_starts = gid_next_starts
        prev_total = total_next
        if transport_cache !== nothing
            copyto!(frontier_mask, transport_cache.frontier_mask)
            empty!(frontier_vertices)
            append!(frontier_vertices, transport_cache.frontier_vertices)
            empty!(changed_vertices)
            append!(changed_vertices, transport_cache.changed_vertices)
            if _INDICATOR_DOWNSET_FRONTIER_VERTICES[]
                _update_support_state!(
                    support_vertices,
                    support_mask,
                    support_slots,
                    C.dims,
                    changed_vertices,
                    frontier_vertices,
                )
            else
                _update_support_state!(
                    support_vertices,
                    support_mask,
                    support_slots,
                    C.dims,
                    changed_vertices,
                    frontier_mask,
                )
            end
        else
            support_vertices = _nonzero_dim_vertices(C.dims)
            fill!(support_mask, false)
            @inbounds for u in support_vertices
                support_mask[u] = true
            end
            fill!(support_slots, 0)
            @inbounds for (idx, u) in enumerate(support_vertices)
                support_slots[u] = idx
            end
            fill!(frontier_mask, false)
            empty!(frontier_vertices)
            empty!(changed_vertices)
        end
        steps += 1
    end

    # Materialize principal downsets only once when packaging output.
    PT = typeof(P)
    D_by_b = Vector{Vector{Downset{PT}}}(undef, length(gens_by_b))
    @inbounds for b in eachindex(gens_by_b)
        D_by_b[b] = _principal_downsets_from_plan(P, gens_by_b[b])
    end

    # Package as DownsetCopresentation list.
    D0 = D_by_b[1]
    D1 = length(gens_by_b) > 1 ? D_by_b[2] : Downset{PT}[]
    rho_b = length(gens_by_b) > 1 ? dE[1] : spzeros(K, 0, length(D0))
    E1 = DownsetCopresentation{K}(P, D0, D1, rho_b, nothing)
    E = Vector{typeof(E1)}(undef, length(gens_by_b))
    E[1] = E1
    for b in 2:length(gens_by_b)
        D0 = D_by_b[b]
        if b < length(gens_by_b)
            D1 = D_by_b[b + 1]
            rho_b = dE[b]
        else
            D1 = Downset{PT}[]
            rho_b = spzeros(K, 0, length(D0))
        end
        E[b] = DownsetCopresentation{K}(P, D0, D1, rho_b, nothing)
    end

    result = DownsetResolutionResult(E, dE, gens_by_b, iota0, M, false, false)
    return _indicator_resolution_with_output(:downset_resolution, result, resolution_summary, output)
end


# --------------------- aggregator + high-level Ext driver ---------------------

"""
    indicator_resolutions(HM, HN; maxlen=nothing, output=:full)
        -> IndicatorResolutionsResult
    indicator_resolutions(HM, HN; maxlen=nothing, output=:summary)
        -> NamedTuple

Convenience wrapper: build an upset resolution for the source module (fringe HM)
and a downset resolution for the target module (fringe HN).  The `maxlen` keyword
cuts off each side after that many steps (useful for quick tests).

Use `output=:summary` when you want the combined projective/injective metadata
without touching the full wrapped resolution objects.

Mathematically, this is the paired-resolution entrypoint: it builds the
projective/upset side for `HM` and the injective/downset side for `HN` in one
call.

Use `output=:summary` for the cheap/default exploration path when you want to
see the length, field, generator counts, and side metadata of both resolutions.
Use `output=:full` when you need the wrapped resolution objects for subsequent
Ext/Tor-style computation or detailed inspection.
"""
@inline function _indicator_resolutions_from_pmodules(
    MM::PModule{K},
    NN::PModule{K};
    maxlen::Union{Int,Nothing}=nothing,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    PM = MM.Q
    PN = NN.Q
    build_cache!(PM; cover=true, updown=true)
    ccM = _get_cover_cache(PM)
    if PN === PM
        ccN = ccM
    else
        build_cache!(PN; cover=true, updown=true)
        ccN = _get_cover_cache(PN)
    end

    ws_M = _new_resolution_workspace(K, nvertices(PM))
    map_memo_M = _indicator_new_array_memo(K, nvertices(PM))
    map_memo_N = NN === MM ? map_memo_M : _indicator_new_array_memo(K, nvertices(PN))
    memo_pool = IdDict{Any,Vector{Union{Nothing,Matrix{K}}}}()
    upset = upset_resolution(
        MM;
        maxlen=maxlen,
        cache=ccM,
        map_memo=map_memo_M,
        map_memo_pool=memo_pool,
        workspace=ws_M,
        threads=threads,
    )

    ws_N = PN === PM ? ws_M : _new_resolution_workspace(K, nvertices(PN))
    downset = downset_resolution(
        NN;
        maxlen=maxlen,
        cache=ccN,
        map_memo=map_memo_N,
        map_memo_pool=memo_pool,
        workspace=ws_N,
        threads=threads,
    )

    return IndicatorResolutionsResult(upset, downset)
end

@inline function _indicator_resolutions_cache_payload(res::IndicatorResolutionsResult)
    return (
        res.upset.presentations,
        res.upset.differentials,
        res.downset.presentations,
        res.downset.differentials,
    )
end

function indicator_resolutions(HM::FiniteFringe.FringeModule{K},
                               HN::FiniteFringe.FringeModule{K};
                               maxlen::Union{Int,Nothing}=nothing,
                               output::Symbol=:full,
                               threads::Bool = (Threads.nthreads() > 1),
                               cache::Union{Nothing,ResolutionCache}=nothing) where {K}
    key = cache === nothing ? nothing : _resolution_key3(HM, HN, maxlen === nothing ? -1 : Int(maxlen))

    if cache !== nothing
        PT = typeof(HM.P)
        UP = UpsetPresentation{K,PT,Nothing,SparseMatrixCSC{K,Int}}
        DP = DownsetCopresentation{K,PT,Nothing,SparseMatrixCSC{K,Int}}
        cache_val_type = Tuple{
            Vector{UP},
            Vector{SparseMatrixCSC{K,Int}},
            Vector{DP},
            Vector{SparseMatrixCSC{K,Int}},
        }

        cached = _resolution_cache_indicator_get(cache, key, cache_val_type)
        if cached !== nothing
            F, dF, E, dE = cached
            MM = pmodule_from_fringe(HM)
            NN = HM === HN ? MM : pmodule_from_fringe(HN)
            upset = UpsetResolutionResult(F, dF, _ProjectiveGeneratorPlan[], nothing, MM, false, false)
            downset = DownsetResolutionResult(E, dE, _InjectiveGeneratorPlan[], nothing, NN, false, false)
            result = IndicatorResolutionsResult(upset, downset)
            return _indicator_resolution_with_output(:indicator_resolutions, result, resolution_summary, output)
        end
    end

    MM = pmodule_from_fringe(HM)
    NN = HM === HN ? MM : pmodule_from_fringe(HN)
    out = _indicator_resolutions_from_pmodules(
        MM,
        NN;
        maxlen=maxlen,
        threads=threads,
    )

    if cache !== nothing
        _resolution_cache_indicator_store!(cache, key::ResolutionKey3, _indicator_resolutions_cache_payload(out))
    end
    return _indicator_resolution_with_output(:indicator_resolutions, out, resolution_summary, output)
end

# --------------------- resolution verification (structural checks) ---------------------

"""
    verify_upset_resolution(F, dF; vertices=:all,
                            check_d2=true,
                            check_exactness=true,
                            check_connected=true)

Structural checks for an upset (projective) indicator resolution:
  * d^2 = 0 (as coefficient matrices),
  * exactness at intermediate stages (vertexwise),
  * each nonzero entry corresponds to a connected homomorphism in the principal-upset case,
    i.e. support inclusion U_{a+1} subseteq U_a.

Throws an error if a check fails; returns true otherwise.
"""
function verify_upset_resolution(F::AbstractVector{<:UpsetPresentation{K}},
                                dF::Vector{SparseMatrixCSC{K,Int}};
                                vertices = :all,
                                check_d2::Bool = true,
                                check_exactness::Bool = true,
                                check_connected::Bool = true) where {K}
    field = (F[1].H === nothing) ? field_from_eltype(K) : F[1].H.field

    @assert length(dF) == length(F) - 1 "verify_upset_resolution: expected length(dF) == length(F)-1"
    P = F[1].P
    for f in F
        @assert f.P === P "verify_upset_resolution: all presentations must use the same poset"
    end

    U_by_a = [f.U0 for f in F]  # U_0,...,U_A
    vs = (vertices === :all) ? (1:nvertices(P)) : vertices


    # Connectedness / valid monomial support (principal-upset case):
    # a nonzero entry for k[Udom] -> k[Ucod] is only allowed when Udom subseteq Ucod.
    if check_connected
        for a in 1:length(dF)
            delta = dF[a]                 # rows: U_{a+1}, cols: U_a
            Udom  = U_by_a[a+1]
            Ucod  = U_by_a[a]
            for col in 1:size(delta,2)
                for ptr in delta.colptr[col]:(delta.colptr[col+1]-1)
                    row = delta.rowval[ptr]
                    val = delta.nzval[ptr]
                    if !iszero(val) && !FiniteFringe.is_subset(Udom[row], Ucod[col])
                        error("Upset resolution: nonzero delta at (row=$row,col=$col) but U_{a+1}[row] not subset of U_a[col] (a=$(a-1))")
                    end
                end
            end
        end
    end

    # d^2 = 0
    if check_d2 && length(dF) >= 2
        for a in 1:(length(dF)-1)
            C = dF[a+1] * dF[a]
            dropzeros!(C)
            if !_is_zero_matrix(field, C)
                error("Upset resolution: dF[$(a+1)]*dF[$a] != 0 (nnz=$(nnz(C)))")
            end
        end
    end

    # Vertexwise exactness at intermediate stages:
    # For each q, check rank(d_{a}) + rank(d_{a+1}) = dim(F_a(q)).
    if check_exactness && length(dF) >= 2
        for q in vs
            for k in 2:(length(U_by_a)-1)  # check exactness at U_k (k=1..A-1)
                active_k   = findall(u -> u.mask[q], U_by_a[k])
                dim_mid = length(active_k)
                if dim_mid == 0
                    continue
                end
                active_km1 = findall(u -> u.mask[q], U_by_a[k-1])
                active_kp1 = findall(u -> u.mask[q], U_by_a[k+1])

                delta_prev = dF[k-1]  # U_k -> U_{k-1}
                delta_next = dF[k]    # U_{k+1} -> U_k

                r_prev = isempty(active_km1) ? 0 : _rank_restricted_field(field, delta_prev, active_k, active_km1)
                r_next = isempty(active_kp1) ? 0 : _rank_restricted_field(field, delta_next, active_kp1, active_k)

                if r_prev + r_next != dim_mid
                    error("Upset resolution: vertex q=$q fails exactness at degree k=$(k-1): rank(prev)=$r_prev, rank(next)=$r_next, dim=$dim_mid")
                end
            end
        end
    end

    return true
end


"""
    verify_downset_resolution(E, dE; vertices=:all,
                              check_d2=true,
                              check_exactness=true,
                              check_connected=true)

Structural checks for a downset (injective) indicator resolution:
  * d^2 = 0,
  * exactness at intermediate stages (vertexwise),
  * valid monomial support (principal-downset case): nonzero entry implies D_b subseteq D_{b+1}.

Throws an error if a check fails; returns true otherwise.
"""
function verify_downset_resolution(E::AbstractVector{<:DownsetCopresentation{K}},
                                  dE::Vector{SparseMatrixCSC{K,Int}};
                                  vertices = :all,
                                  check_d2::Bool = true,
                                  check_exactness::Bool = true,
                                  check_connected::Bool = true) where {K}
    field = (E[1].H === nothing) ? field_from_eltype(K) : E[1].H.field

    @assert length(dE) == length(E) - 1 "verify_downset_resolution: expected length(dE) == length(E)-1"
    P = E[1].P
    for e in E
        @assert e.P === P "verify_downset_resolution: all copresentations must use the same poset"
    end

    D_by_b = [e.D0 for e in E]  # D_0,...,D_B
    vs = (vertices === :all) ? (1:nvertices(P)) : vertices


    # Valid monomial support (principal-downset case):
    # rho has rows in D_{b+1} and cols in D_b, so nonzero implies D_b subseteq D_{b+1}.
    if check_connected
        for b in 1:length(dE)
            rho = dE[b]                 # rows: D_{b+1}, cols: D_b
            Ddom = D_by_b[b]
            Dcod = D_by_b[b+1]
            for col in 1:size(rho,2)
                for ptr in rho.colptr[col]:(rho.colptr[col+1]-1)
                    row = rho.rowval[ptr]
                    val = rho.nzval[ptr]
                    if !iszero(val) && !FiniteFringe.is_subset(Dcod[row], Ddom[col])
                        error("Downset resolution: nonzero rho at (row=$row,col=$col) but D_{b+1}[row] not subset of D_b[col] (b=$(b-1))")
                    end
                end
            end
        end
    end

    # d^2 = 0
    if check_d2 && length(dE) >= 2
        for b in 1:(length(dE)-1)
            C = dE[b+1] * dE[b]
            dropzeros!(C)
            if !_is_zero_matrix(field, C)
                error("Downset resolution: dE[$(b+1)]*dE[$b] != 0 (nnz=$(nnz(C)))")
            end
        end
    end

    # Vertexwise exactness at intermediate stages:
    if check_exactness && length(dE) >= 2
        for q in vs
            for k in 2:(length(D_by_b)-1)  # check exactness at D_k (k=1..B-1)
                active_k   = findall(d -> d.mask[q], D_by_b[k])
                dim_mid = length(active_k)
                if dim_mid == 0
                    continue
                end
                active_km1 = findall(d -> d.mask[q], D_by_b[k-1])
                active_kp1 = findall(d -> d.mask[q], D_by_b[k+1])

                rho_prev = dE[k-1]  # D_{k-1} -> D_k
                rho_next = dE[k]    # D_k -> D_{k+1}

                r_prev = isempty(active_km1) ? 0 : _rank_restricted_field(field, rho_prev, active_k, active_km1)
                r_next = isempty(active_kp1) ? 0 : _rank_restricted_field(field, rho_next, active_kp1, active_k)

                if r_prev + r_next != dim_mid
                    error("Downset resolution: vertex q=$q fails exactness at degree k=$(k-1): rank(prev)=$r_prev, rank(next)=$r_next, dim=$dim_mid")
                end
            end
        end
    end

    return true
end

end # module
