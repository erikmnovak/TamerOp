module Modules

using SparseArrays, LinearAlgebra
using ..CoreModules: QQ, AbstractCoeffField, QQField, PrimeField, RealField,
    BackendMatrix, coeff_type, field_from_eltype, eye, coerce
using ..Options: ModuleOptions
import ..CoreModules: change_field, _append_scaled_triplets!
import ..FiniteFringe
import ..FiniteFringe: AbstractPoset, FinitePoset, cover_edges, leq, nvertices,
       CoverCache, PosetCache, _get_cover_cache, _succs, _preds,
       _pred_slots_of_succ, _succ_slots_of_pred,
       _clear_cover_cache!, _pairkey, _chosen_predecessor, _chosen_predecessor_slow,
       _chain_parent_dense, _chain_parent_dict
import ..FieldLinAlg
import Base.Threads

const MAP_LEQ_MEMO_MAX_PER_THREAD = Ref(200_000)
const MAP_LEQ_DENSE_MEMO_MIN_ENTRIES = Ref(4_096)
const MAP_LEQ_DENSE_MEMO_MAX_ENTRIES_PER_THREAD = Ref(500_000)
const MAP_LEQ_SLOT_DENSE_MIN_ENTRIES = Ref(4_096)
const MAP_LEQ_SLOT_DENSE_MAX_ENTRIES_PER_THREAD = Ref(1_000_000)
const MAP_LEQ_MANY_PLAN_MIN_LEN = Ref(128)
const MAP_LEQ_MANY_PLAN_MAX_PER_THREAD = Ref(128)
const MAP_LEQ_MANY_ONEOFF_LONG_MIN_LEN = Ref(128)
const MAP_LEQ_MANY_ONEOFF_LONG_MIN_LONG = Ref(64)
const MAP_LEQ_MANY_ONEOFF_MIN_OVERLAP_QQ = Ref(0.28)
const MAP_LEQ_MANY_ONEOFF_MIN_OVERLAP_PRIME = Ref(0.26)
const MAP_LEQ_MANY_ONEOFF_MIN_OVERLAP_REAL = Ref(0.30)
const MAP_LEQ_MANY_ONEOFF_MIN_TARGET_REPEAT_QQ = Ref(0.45)
const MAP_LEQ_MANY_ONEOFF_MIN_TARGET_REPEAT_PRIME = Ref(0.30)
const MAP_LEQ_MANY_ONEOFF_MIN_TARGET_REPEAT_REAL = Ref(0.34)
const MAP_LEQ_MANY_LONG_SCALAR_MAX_OVERLAP_QQ = Ref(0.15)
const MAP_LEQ_MANY_LONG_SCALAR_MAX_OVERLAP_PRIME = Ref(0.10)
const MAP_LEQ_MANY_LONG_SCALAR_MAX_OVERLAP_REAL = Ref(0.12)
const MAP_LEQ_MANY_LONG_SCALAR_MAX_TARGET_REPEAT_QQ = Ref(0.35)
const MAP_LEQ_MANY_LONG_SCALAR_MAX_TARGET_REPEAT_PRIME = Ref(0.14)
const MAP_LEQ_MANY_LONG_SCALAR_MAX_TARGET_REPEAT_REAL = Ref(0.16)
const MAP_LEQ_MANY_LONG_MIN_AVG_HOPS_QQ = Ref(4.0)
const MAP_LEQ_MANY_LONG_MIN_AVG_HOPS_PRIME = Ref(4.5)
const MAP_LEQ_MANY_LONG_MIN_AVG_HOPS_REAL = Ref(5.0)
const MAP_LEQ_MANY_PLAN_MIN_SCORE_QQ = Ref(0.18)
const MAP_LEQ_MANY_PLAN_MIN_SCORE_PRIME = Ref(0.20)
const MAP_LEQ_MANY_PLAN_MIN_SCORE_REAL = Ref(0.22)
const MAP_LEQ_MANY_TRIPLET_PLAN_MIN_LEN = Ref(32)
const MAP_LEQ_QQ_NEMO_LONG_MIN_HOPS = Ref(5)
const MAP_LEQ_QQ_NEMO_LONG_MIN_WORK = Ref(192)
const DIRECT_SUM_SPARSE_MIN_TOTAL_ENTRIES = Ref(16_384)
const DIRECT_SUM_SPARSE_MIN_AVG_EDGE_ENTRIES_QQ = Ref(128.0)
const DIRECT_SUM_SPARSE_MIN_AVG_EDGE_ENTRIES_PRIME = Ref(256.0)
const DIRECT_SUM_SPARSE_MIN_AVG_EDGE_ENTRIES_REAL = Ref(256.0)
const DIRECT_SUM_SPARSE_MAX_DENSITY = Ref(0.55)
const DIRECT_SUM_SPARSE_MAX_DENSITY_PRIME = Ref(0.28)
const DIRECT_SUM_SPARSE_MAX_DENSITY_REAL = Ref(0.20)
const DIRECT_SUM_COST_DENSE_WEIGHT_QQ = Ref(1.25)
const DIRECT_SUM_COST_DENSE_WEIGHT_PRIME = Ref(1.00)
const DIRECT_SUM_COST_DENSE_WEIGHT_REAL = Ref(0.90)
const DIRECT_SUM_COST_SPARSE_NNZ_WEIGHT_QQ = Ref(0.92)
const DIRECT_SUM_COST_SPARSE_NNZ_WEIGHT_PRIME = Ref(1.10)
const DIRECT_SUM_COST_SPARSE_NNZ_WEIGHT_REAL = Ref(1.18)
const DIRECT_SUM_COST_SPARSE_EDGE_OVERHEAD_QQ = Ref(2.0)
const DIRECT_SUM_COST_SPARSE_EDGE_OVERHEAD_PRIME = Ref(4.0)
const DIRECT_SUM_COST_SPARSE_EDGE_OVERHEAD_REAL = Ref(5.0)

"""
    CoverEdgeMapStore{K,MatT}

Internal storage for the structure maps of a `PModule` on the *cover* relations
of a finite poset.

For each vertex `v` we store:

  * `preds[v]`  : cover predecessors of `v` (sorted)
  * `maps_from_pred[v]` : matrices for the maps `u -> v` in the same order

Dually we store:

  * `succs[u]`  : cover successors of `u` (sorted)
  * `maps_to_succ[u]` : the same matrices, indexed by successors

This eliminates hot-path dictionary lookups when traversing the cover graph.

Canonical access is `store[u,v]` (with `haskey(store, u, v)`).
"""
struct CoverEdgeMapStore{K,MatT<:AbstractMatrix{K}}
    preds::Vector{Vector{Int}}
    succs::Vector{Vector{Int}}
    maps_from_pred::Vector{Vector{MatT}}
    maps_to_succ::Vector{Vector{MatT}}
    nedges::Int
end

Base.length(store::CoverEdgeMapStore) = store.nedges

# Structural equality and hashing for CoverEdgeMapStore.
# Without this, `==` falls back to object identity, so two stores with the same
# content compare unequal.
function Base.:(==)(A::CoverEdgeMapStore, B::CoverEdgeMapStore)
    return A.nedges == B.nedges &&
           A.preds == B.preds &&
           A.succs == B.succs &&
           A.maps_from_pred == B.maps_from_pred &&
           A.maps_to_succ == B.maps_to_succ
end

function Base.isequal(A::CoverEdgeMapStore, B::CoverEdgeMapStore)
    return A.nedges == B.nedges &&
           isequal(A.preds, B.preds) &&
           isequal(A.succs, B.succs) &&
           isequal(A.maps_from_pred, B.maps_from_pred) &&
           isequal(A.maps_to_succ, B.maps_to_succ)
end

function Base.hash(A::CoverEdgeMapStore, h::UInt)
    h = hash(A.nedges, h)
    h = hash(A.preds, h)
    h = hash(A.succs, h)
    h = hash(A.maps_from_pred, h)
    h = hash(A.maps_to_succ, h)
    return h
end

# Iterator: yield ((u, v), map) for each cover edge u -> v, ordered by v then preds[v].
function Base.iterate(store::CoverEdgeMapStore)
    v = 1
    while v <= length(store.preds) && isempty(store.preds[v])
        v += 1
    end
    if v > length(store.preds)
        return nothing
    end
    k = 1
    return ((store.preds[v][k], v), store.maps_from_pred[v][k]), (v, k)
end

function Base.iterate(store::CoverEdgeMapStore, state::Tuple{Int,Int})
    v, k = state
    k += 1
    if k <= length(store.preds[v])
        return ((store.preds[v][k], v), store.maps_from_pred[v][k]), (v, k)
    end
    v += 1
    while v <= length(store.preds) && isempty(store.preds[v])
        v += 1
    end
    if v > length(store.preds)
        return nothing
    end
    k = 1
    return ((store.preds[v][k], v), store.maps_from_pred[v][k]), (v, k)
end

# ----- helper: binary search in sorted Int vectors -----
@inline function _find_sorted_index(xs::AbstractVector{Int}, x::Int)::Int
    i = searchsortedfirst(xs, x)
    if i <= length(xs) && @inbounds xs[i] == x
        return i
    end
    return 0
end

@inline function _find_sorted_index_range(xs::AbstractVector{Int}, lo::Int, hi::Int, x::Int)::Int
    l = lo
    h = hi
    while l <= h
        m = (l + h) >>> 1
        y = @inbounds xs[m]
        if y < x
            l = m + 1
        elseif y > x
            h = m - 1
        else
            return m - lo + 1
        end
    end
    return 0
end

# ----- type-stable zero map creation (dense/sparse) -----
@inline function _zero_edge_map(::Type{Matrix{K}}, ::Type{K}, m::Int, n::Int) where {K}
    return zeros(K, m, n)
end

@inline function _zero_edge_map(::Type{SparseMatrixCSC{K,Int}}, ::Type{K}, m::Int, n::Int) where {K}
    return spzeros(K, m, n)
end

@inline function _zero_edge_map(::Type{MatT}, ::Type{K}, m::Int, n::Int) where {K,MatT<:AbstractMatrix{K}}
    return convert(MatT, zeros(K, m, n))
end

# ----- dictionary-like API (KeyError if not a cover edge) -----
#
# Canonical access:
#   store[u,v]  (KeyError if (u,v) is not a cover edge)
#   haskey(store, u, v)
#
# This avoids tuple allocations and keeps hot loops allocation-free.
# For bulk traversal, prefer store-aligned iteration via succs/maps_to_succ.

@inline function Base.haskey(store::CoverEdgeMapStore, u::Int, v::Int)::Bool
    return _find_sorted_index(store.preds[v], u) != 0
end

@inline function Base.getindex(store::CoverEdgeMapStore{K,MatT}, u::Int, v::Int) where {K,MatT}
    i = _find_sorted_index(store.preds[v], u)
    i == 0 && throw(KeyError((u, v)))
    return @inbounds store.maps_from_pred[v][i]
end

# NOTE: We intentionally avoid generic adapter helpers for edge map access.
# We instead dispatch explicitly on the supported representations at construction time.

"""
    CoverEdgeMapStore{K,MatT}(Q, dims, edge_maps; cache=nothing, check_sizes=true)

Build a store aligned with the cover graph of `Q`.

This is the canonical internal representation used by `PModule` for cover-edge maps.
The constructor supports two input representations:

- `edge_maps::AbstractDict{Tuple{Int,Int},...}`: a tuple-keyed mapping where `(u,v)`
  stores the map for the cover relation `u \\lessdot v`.
- `edge_maps::CoverEdgeMapStore{K,MatT}`: an already-built store.

Missing cover-edge maps are filled with the appropriate zero map.

We keep this as an explicit dispatch point (Dict vs CoverEdgeMapStore) to avoid
runtime "two-world" adapter helpers.
"""
function CoverEdgeMapStore{K,MatT}(
    Q::AbstractPoset,
    dims::Vector{Int},
    edge_maps::AbstractDict{Tuple{Int,Int},<:Any};
    cache::Union{Nothing,CoverCache}=nothing,
    check_sizes::Bool=true,
) where {K,MatT<:AbstractMatrix{K}}

    cc = cache === nothing ? _get_cover_cache(Q) : cache
    n = nvertices(Q)
    preds = [_preds(cc, v) for v in 1:n]
    succs = [_succs(cc, u) for u in 1:n]

    # Incoming storage (indexed by v, then by sorted preds[v]).
    maps_from_pred = [Vector{MatT}(undef, length(preds[v])) for v in 1:n]

    for v in 1:n
        pv = preds[v]
        mv = maps_from_pred[v]
        dv = dims[v]
        @inbounds for i in eachindex(pv)
            u = pv[i]
            # Dict representation uses tuple keys (u,v).
            if haskey(edge_maps, (u, v))
                A = convert(MatT, edge_maps[(u, v)])
            else
                A = _zero_edge_map(MatT, K, dv, dims[u])
            end

            if check_sizes
                if size(A, 1) != dv || size(A, 2) != dims[u]
                    error("edge map ($u,$v) has size $(size(A)), expected ($(dv),$(dims[u]))")
                end
            end
            mv[i] = A
        end
    end

    # Outgoing storage holds references to the same matrices as maps_from_pred.
    maps_to_succ = [Vector{MatT}(undef, length(succs[u])) for u in 1:n]
    @inbounds for u in 1:n
        su = succs[u]
        mu = maps_to_succ[u]
        slot = _pred_slots_of_succ(cc, u)
        for j in eachindex(su)
            v = su[j]
            i = slot[j]
            mu[j] = maps_from_pred[v][i]
        end
    end

    nedges = sum(length, succs)
    return CoverEdgeMapStore{K,MatT}(preds, succs, maps_from_pred, maps_to_succ, nedges)
end

function CoverEdgeMapStore{K,MatT}(
    Q::AbstractPoset,
    dims::Vector{Int},
    edge_maps::CoverEdgeMapStore{K,MatT};
    cache::Union{Nothing,CoverCache}=nothing,
    check_sizes::Bool=true,
) where {K,MatT<:AbstractMatrix{K}}

    # If the caller already has a store, return it (after optional validation).
    if check_sizes
        cc = cache === nothing ? _get_cover_cache(Q) : cache
        if length(edge_maps.preds) != nvertices(Q) || length(edge_maps.succs) != nvertices(Q) ||
           any(edge_maps.preds[v] != _preds(cc, v) for v in 1:nvertices(Q)) ||
           any(edge_maps.succs[u] != _succs(cc, u) for u in 1:nvertices(Q))
            error("edge_maps store does not match the cover graph of Q")
        end

        n = nvertices(Q)
        preds = [_preds(cc, v) for v in 1:n]
        if length(edge_maps.maps_from_pred) != n
            error("edge_maps store has wrong size (expected $n vertices)")
        end

        for v in 1:n
            pv = preds[v]
            mv = edge_maps.maps_from_pred[v]
            dv = dims[v]
            if length(mv) != length(pv)
                error("edge_maps store mismatch at vertex $v")
            end
            @inbounds for i in eachindex(pv)
                u = pv[i]
                A = mv[i]
                if size(A, 1) != dv || size(A, 2) != dims[u]
                    error("edge map ($u,$v) has size $(size(A)), expected ($(dv),$(dims[u]))")
                end
            end
        end
    end

    return edge_maps
end

# ------------------------------ tiny internal model ------------------------------

mutable struct MapLeqScratch{K}
    tmp1::Matrix{K}
    tmp2::Matrix{K}
    pool1::Vector{Matrix{K}}
    pool2::Vector{Matrix{K}}
    qq_nemo_owner::Any
    qq_nemo_edge::IdDict{Any,Any}
end

const _MAP_LEQ_BATCH_ID_COUNTER = Threads.Atomic{UInt}(UInt(1))

@inline function _next_map_leq_batch_id()::UInt64
    return UInt64(Threads.atomic_add!(_MAP_LEQ_BATCH_ID_COUNTER, UInt(1)))
end

"""
    MapLeqQueryBatch

Prepared query batch for repeated `map_leq_many` calls.

Construct via `prepare_map_leq_batch(pairs)`. The constructor copies `pairs`,
so later mutation of the original vector cannot invalidate the prepared batch.
"""
struct MapLeqQueryBatch
    id::UInt64
    pairs::Vector{Tuple{Int,Int}}
end

mutable struct _MapLeqManyPlan
    npairs::Int
    pairs_hash::UInt
    first_pair::Tuple{Int,Int}
    last_pair::Tuple{Int,Int}
    has_repeats::Bool
    warmed::Bool
    kinds::Vector{UInt8}          # 0: u==v, 1: cover edge, 2: two-hop, 3: long chain
    us::Vector{Int}
    vs::Vector{Int}
    mids::Vector{Int}             # only used for two-hop entries
    chain_ptr::Vector{Int}        # start offsets in chain_data (1-based), length npairs+1
    chain_data::Vector{Int}       # packed chain vertex indices
    chain_slots::Vector{Int}      # packed predecessor slots aligned with chain_data; chain_slots[sidx] == 0
    query_suffix::Vector{Int}     # query -> suffix program id for long-chain entries
    suffix_u::Vector{Int}         # suffix source
    suffix_v::Vector{Int}         # suffix target
    suffix_child::Vector{Int}     # child suffix id (0 => direct cover tail)
    suffix_child_node::Vector{Int}# node after suffix source on chosen cover chain
    suffix_head_slot::Vector{Int} # slot of edge suffix_u -> suffix_child_node
    suffix_tail_slot::Vector{Int} # slot of direct cover tail suffix_child_node -> suffix_v when suffix_child == 0
    suffix_hops::Vector{Int}      # hop count represented by each suffix entry
end

mutable struct _MapLeqManyPlanArena
    kinds::Vector{UInt8}
    us::Vector{Int}
    vs::Vector{Int}
    mids::Vector{Int}
    chain_ptr::Vector{Int}
    chain_data::Vector{Int}
    chain_slots::Vector{Int}
    query_suffix::Vector{Int}
    suffix_u::Vector{Int}
    suffix_v::Vector{Int}
    suffix_child::Vector{Int}
    suffix_child_node::Vector{Int}
    suffix_head_slot::Vector{Int}
    suffix_tail_slot::Vector{Int}
    suffix_hops::Vector{Int}
    suffix_id_dense::Vector{Int}
    suffix_id_touched::Vector{Int}
    target_seen::BitVector
    tmp_chain::Vector{Int}
    tmp_slots::Vector{Int}
end

struct _MapComposeDenseMemo{MatT}
    seen::BitVector
    vals::Vector{MatT}
    n::Int
end

struct _MapPredSlotDenseMemo
    seen::BitVector
    vals::Vector{Int}
    n::Int
end

struct _DirectSumEdgeStats
    n_edges::Int
    total_entries::Int64
    total_nnz::Int64
    sum_dom_dims::Int64
    sum_cod_dims::Int64
end

mutable struct _MapLeqLastPairCache{MatT}
    seen::BitVector
    us::Vector{Int}
    vs::Vector{Int}
    promoted::BitVector
    vals::Vector{MatT}
end

struct _MapLeqManyBatchPlanEntry
    batch::MapLeqQueryBatch
    plan::_MapLeqManyPlan
end

struct _MapLeqManyBatchStats
    npairs::Int
    pairs_hash::UInt
    first_pair::Tuple{Int,Int}
    last_pair::Tuple{Int,Int}
    has_repeats::Bool
    nlong::Int
    total_long_hops::Int
    total_suffix_visits::Int
    nsuffix::Int
    ntargets::Int
end

"""
    PModule{K}(Q, dims, edge_maps; field=..., check_sizes=true)

A module over a finite poset `Q`, viewed as a functor `Q -> Vec_K`.

Mathematically, a `PModule` assigns:
- a finite-dimensional vector space `M(q)` to each vertex `q` of `Q`,
- a linear map `M(q <= q') : M(q) -> M(q')` to each comparable pair,
- subject to the usual functorial composition law.

The constructor is given concretely by:
- `dims[i] = dim_K M(i)`,
- one structure map for each cover relation of the poset.

Implementation notes
- For performance, only cover-edge maps are stored explicitly in a
  `CoverEdgeMapStore`.
- Maps for longer comparable pairs are derived from those cover-edge maps.

Best practices
- Prefer the canonical `PModule(...)` constructors to direct field manipulation.
- After building a module by hand, run `check_module(M)` before using it in
  invariants, derived functors, or benchmarks.
- Prefer `describe(M)`, `dimensions(M)`, `basis(M, q)`, and
  `coordinates(M, q, x)` over reading `M.dims` and `M.edge_maps` directly unless
  you explicitly need the raw storage.

See also: `PMorphism`, `check_module`, `describe`, `dimensions`, `basis`,
`coordinates`.
"""
struct PModule{K,F<:AbstractCoeffField,MatT<:AbstractMatrix{K},QT<:AbstractPoset}
    field::F
    Q::QT
    dims::Vector{Int}
    edge_maps::CoverEdgeMapStore{K,MatT}
    direct_sum_stats::_DirectSumEdgeStats
    map_compose::Vector{Dict{UInt64, MatT}}
    map_compose_dense::Union{Nothing,Vector{_MapComposeDenseMemo{MatT}}}
    map_pred_slot::Vector{Dict{UInt64, Int}}
    map_pred_slot_dense::Union{Nothing,Vector{_MapPredSlotDenseMemo}}
    map_many_plan::Vector{Dict{UInt64,_MapLeqManyPlan}}
    map_many_batch_plan::Vector{Dict{UInt64,_MapLeqManyBatchPlanEntry}}
    map_many_batch_last::Vector{Union{Nothing,_MapLeqManyBatchPlanEntry}}
    map_many_plan_arena::Vector{_MapLeqManyPlanArena}
    map_scratch::Vector{MapLeqScratch{K}}
    identity_compose::Vector{Dict{Int, MatT}}
    map_last_pair::_MapLeqLastPairCache{MatT}
end

# choose a storage matrix type from a user-provided mapping
@inline _is_nemo_field(field::AbstractCoeffField) =
    field isa QQField || (field isa PrimeField && field.p > 3)

@inline _is_backend_dense_candidate(A::AbstractMatrix) = !(A isa SparseMatrixCSC)

@inline function _should_backendize_map(field::AbstractCoeffField, A::AbstractMatrix)
    _is_nemo_field(field) || return false
    FieldLinAlg._have_nemo() || return false
    _is_backend_dense_candidate(A) || return false
    m, n = size(A)
    return m * n >= FieldLinAlg.NEMO_THRESHOLD[]
end

function _should_backendize_maps(field::AbstractCoeffField, edge_maps)::Bool
    _is_nemo_field(field) || return false
    FieldLinAlg._have_nemo() || return false
    for (_, A) in edge_maps
        if A isa AbstractMatrix && _should_backendize_map(field, A)
            return true
        end
    end
    return false
end

@inline function _pmodule_mat_type(::Type{K}, edge_maps, field::AbstractCoeffField) where {K}
    if _should_backendize_maps(field, edge_maps)
        return BackendMatrix{K}
    end
    V = Base.valtype(edge_maps)
    if V <: AbstractMatrix{K} && isconcretetype(V)
        return V
    end
    return Matrix{K}
end

@inline function _new_map_leq_memo(::Type{MatT}) where {MatT}
    nt = max(1, Threads.maxthreadid())
    return [Dict{UInt64, MatT}() for _ in 1:nt]
end

@inline function _new_map_leq_dense_memo(::Type{MatT}, n::Int) where {MatT}
    dense_entries = n * n
    use_dense = dense_entries >= MAP_LEQ_DENSE_MEMO_MIN_ENTRIES[] &&
                dense_entries <= MAP_LEQ_DENSE_MEMO_MAX_ENTRIES_PER_THREAD[]
    use_dense || return nothing
    nt = max(1, Threads.maxthreadid())
    return [_MapComposeDenseMemo{MatT}(falses(dense_entries), Vector{MatT}(undef, dense_entries), n)
            for _ in 1:nt]
end

@inline function _new_map_leq_many_plan_cache()
    nt = max(1, Threads.maxthreadid())
    return [Dict{UInt64,_MapLeqManyPlan}() for _ in 1:nt]
end

@inline function _new_map_pred_slot_memo()
    nt = max(1, Threads.maxthreadid())
    return [Dict{UInt64, Int}() for _ in 1:nt]
end

@inline function _new_map_pred_slot_dense_memo(n::Int)
    dense_entries = n * n
    use_dense = dense_entries >= MAP_LEQ_SLOT_DENSE_MIN_ENTRIES[] &&
                dense_entries <= MAP_LEQ_SLOT_DENSE_MAX_ENTRIES_PER_THREAD[]
    use_dense || return nothing
    nt = max(1, Threads.maxthreadid())
    return [_MapPredSlotDenseMemo(falses(dense_entries), zeros(Int, dense_entries), n)
            for _ in 1:nt]
end

@inline function _new_map_leq_many_batch_plan_cache()
    nt = max(1, Threads.maxthreadid())
    return [Dict{UInt64,_MapLeqManyBatchPlanEntry}() for _ in 1:nt]
end

@inline function _new_map_leq_many_batch_last_cache()
    nt = max(1, Threads.maxthreadid())
    v = Vector{Union{Nothing,_MapLeqManyBatchPlanEntry}}(undef, nt)
    fill!(v, nothing)
    return v
end

@inline function _new_map_leq_many_plan_arena()
    nt = max(1, Threads.maxthreadid())
    return [_MapLeqManyPlanArena(
                UInt8[],
                Int[],
                Int[],
                Int[],
                Int[],
                Int[],
                Int[],
                Int[],
                Int[],
                Int[],
                Int[],
                Int[],
                Int[],
                Int[],
                Int[],
                Int[],
                Int[],
                falses(0),
                Int[],
                Int[],
            ) for _ in 1:nt]
end

@inline function _new_map_leq_scratch(::Type{K}) where {K}
    nt = max(1, Threads.maxthreadid())
    return [MapLeqScratch{K}(Matrix{K}(undef, 0, 0), Matrix{K}(undef, 0, 0),
                             Matrix{K}[], Matrix{K}[], nothing, IdDict{Any,Any}())
            for _ in 1:nt]
end

@inline function _new_identity_compose(::Type{MatT}) where {MatT}
    nt = max(1, Threads.maxthreadid())
    return [Dict{Int, MatT}() for _ in 1:nt]
end

@inline function _new_map_leq_last_pair_cache(::Type{K}, ::Type{MatT}) where {K,MatT}
    nt = max(1, Threads.maxthreadid())
    seen = falses(nt)
    promoted = falses(nt)
    us = zeros(Int, nt)
    vs = zeros(Int, nt)
    vals = Vector{MatT}(undef, nt)
    empty_map = _zero_edge_map(MatT, K, 0, 0)
    @inbounds for i in 1:nt
        vals[i] = empty_map
    end
    return _MapLeqLastPairCache{MatT}(seen, us, vs, promoted, vals)
end

@inline function _edge_nnz_count(::Type{MatT}, A::AbstractMatrix{K}) where {K,MatT}
    if MatT <: SparseMatrixCSC
        return Int64(nnz(A))
    end
    return Int64(size(A, 1)) * Int64(size(A, 2))
end

function _build_direct_sum_edge_stats(dims::Vector{Int}, store::CoverEdgeMapStore{K,MatT}) where {K,MatT}
    total_entries = Int64(0)
    total_nnz = Int64(0)
    sum_dom = Int64(0)
    sum_cod = Int64(0)
    n_edges = 0
    succs = store.succs
    maps_to_succ = store.maps_to_succ
    @inbounds for u in eachindex(succs)
        du = Int64(dims[u])
        su = succs[u]
        mu = maps_to_succ[u]
        for j in eachindex(su)
            v = su[j]
            dv = Int64(dims[v])
            total_entries += dv * du
            total_nnz += _edge_nnz_count(MatT, mu[j])
            sum_dom += du
            sum_cod += dv
            n_edges += 1
        end
    end
    return _DirectSumEdgeStats(n_edges, total_entries, total_nnz, sum_dom, sum_cod)
end

"""
    PModule{K}(Q, dims, edge_maps; check_sizes=true)

Construct a `PModule` over coefficient type `K`. `edge_maps` may be a dict or
any mapping supporting `(u,v)` keys. Missing cover maps become zero maps.
"""
function PModule{K}(Q::AbstractPoset, dims::Vector{Int}, edge_maps;
                    check_sizes::Bool=true,
                    opts::ModuleOptions=ModuleOptions(),
                    field::AbstractCoeffField=field_from_eltype(K)) where {K}
    if opts != ModuleOptions()
        check_sizes == true || error("PModule: pass either check_sizes or opts, not both.")
        check_sizes = opts.check_sizes
    end
    coeff_type(field) == K || error("PModule: coeff_type(field) != K")
    MatT = _pmodule_mat_type(K, edge_maps, field)
    store = CoverEdgeMapStore{K,MatT}(Q, dims, edge_maps; check_sizes=check_sizes)
    ds_stats = _build_direct_sum_edge_stats(dims, store)
    QT = typeof(Q)
    return PModule{K, typeof(field), MatT, QT}(field, Q, dims, store, ds_stats,
                                               _new_map_leq_memo(MatT),
                                               _new_map_leq_dense_memo(MatT, nvertices(Q)),
                                               _new_map_pred_slot_memo(),
                                               _new_map_pred_slot_dense_memo(nvertices(Q)),
                                               _new_map_leq_many_plan_cache(),
                                               _new_map_leq_many_batch_plan_cache(),
                                               _new_map_leq_many_batch_last_cache(),
                                               _new_map_leq_many_plan_arena(),
                                               _new_map_leq_scratch(K),
                                               _new_identity_compose(MatT),
                                               _new_map_leq_last_pair_cache(K, MatT))
end

# rebase existing store to this poset (important for ChangeOfPosets)
function PModule{K}(Q::AbstractPoset, dims::Vector{Int}, store::CoverEdgeMapStore{K,MatT};
                    check_sizes::Bool=true,
                    opts::ModuleOptions=ModuleOptions(),
                    field::AbstractCoeffField=field_from_eltype(K)) where {K,MatT<:AbstractMatrix{K}}
    if opts != ModuleOptions()
        check_sizes == true || error("PModule: pass either check_sizes or opts, not both.")
        check_sizes = opts.check_sizes
    end
    cc = _get_cover_cache(Q)
    if MatT == Matrix{K} && _should_backendize_maps(field, store)
        dense_edge = Dict{Tuple{Int,Int}, Matrix{K}}()
        sizehint!(dense_edge, length(store))
        for ((u, v), A) in store
            dense_edge[(u, v)] = Matrix{K}(A)
        end
        return PModule{K}(Q, dims, dense_edge; check_sizes=check_sizes, field=field)
    end
    if length(store.preds) == nvertices(Q) && length(store.succs) == nvertices(Q) &&
       all(store.preds[v] == _preds(cc, v) for v in 1:nvertices(Q)) &&
       all(store.succs[u] == _succs(cc, u) for u in 1:nvertices(Q))
        coeff_type(field) == K || error("PModule: coeff_type(field) != K")
        ds_stats = _build_direct_sum_edge_stats(dims, store)
        QT = typeof(Q)
        return PModule{K, typeof(field), MatT, QT}(field, Q, dims, store, ds_stats,
                                                   _new_map_leq_memo(MatT),
                                                   _new_map_leq_dense_memo(MatT, nvertices(Q)),
                                                   _new_map_pred_slot_memo(),
                                                   _new_map_pred_slot_dense_memo(nvertices(Q)),
                                                   _new_map_leq_many_plan_cache(),
                                                   _new_map_leq_many_batch_plan_cache(),
                                                   _new_map_leq_many_batch_last_cache(),
                                                   _new_map_leq_many_plan_arena(),
                                                   _new_map_leq_scratch(K),
                                                   _new_identity_compose(MatT),
                                                   _new_map_leq_last_pair_cache(K, MatT))
    end
    new_store = CoverEdgeMapStore{K,MatT}(Q, dims, store; cache=cc, check_sizes=check_sizes)
    coeff_type(field) == K || error("PModule: coeff_type(field) != K")
    ds_stats = _build_direct_sum_edge_stats(dims, new_store)
    QT = typeof(Q)
    return PModule{K, typeof(field), MatT, QT}(field, Q, dims, new_store, ds_stats,
                                               _new_map_leq_memo(MatT),
                                               _new_map_leq_dense_memo(MatT, nvertices(Q)),
                                               _new_map_pred_slot_memo(),
                                               _new_map_pred_slot_dense_memo(nvertices(Q)),
                                               _new_map_leq_many_plan_cache(),
                                               _new_map_leq_many_batch_plan_cache(),
                                               _new_map_leq_many_batch_last_cache(),
                                               _new_map_leq_many_plan_arena(),
                                               _new_map_leq_scratch(K),
                                               _new_identity_compose(MatT),
                                               _new_map_leq_last_pair_cache(K, MatT))
end

# infer coefficient type from first map
function PModule(Q::AbstractPoset, dims::Vector{Int}, edge_maps;
                 check_sizes::Bool=true,
                 opts::ModuleOptions=ModuleOptions(),
                 field::Union{AbstractCoeffField,Nothing}=nothing)
    if opts != ModuleOptions()
        check_sizes == true || error("PModule: pass either check_sizes or opts, not both.")
        check_sizes = opts.check_sizes
    end
    for (_, A) in edge_maps
        return PModule{eltype(A)}(Q, dims, edge_maps; check_sizes=check_sizes,
                                  field=field === nothing ? field_from_eltype(eltype(A)) : field)
    end
    error("Cannot infer coefficient type K from empty edge_maps; use PModule{K}(...)")
end


"""
    dim_at(M::PModule, q::Integer) -> Int

Return the dimension of the stalk (fiber) of the P-module `M` at the vertex `q`.

Mathematically: `dim_at(M, q) = dim_k M(q)`.

This mirrors the existing `dim_at` query used in the Zn/Flange layer, but for
internal `PModule`s used by IndicatorResolutions / DerivedFunctors.

Notes
-----
- Vertices are encoded as integers `1:Q.n`.
- This is a pure convenience method: it simply returns `M.dims[q]`.

Best practices
- Treat `dim_at(M, q)` as the canonical public stalk-dimension query.
- Prefer this over reading `M.dims[q]` directly in examples and notebook code.
- Pair it with `basis(M, q)` and `coordinates(M, q, x)` when you want an
  explicit coordinate convention on the stalk.
"""
function dim_at(M::PModule{K}, q::Integer) where {K}
    return M.dims[Int(q)]
end

"""
    support(M::PModule) -> Vector{Int}

Return the vertices where the module has nonzero stalk dimension.

Mathematically, this is the support of the module viewed as a representation of
the finite poset.

Best practices
- use this to understand where a sparse module is actually nontrivial,
- pair it with `support_dims(M)` when you also want the corresponding stalk
  dimensions.
"""
function support(M::PModule)
    out = Int[]
    sizehint!(out, length(M.dims))
    @inbounds for q in eachindex(M.dims)
        M.dims[q] != 0 && push!(out, q)
    end
    return out
end

"""
    nonzero_vertices(M::PModule) -> Vector{Int}

Explicit synonym for `support(M)`.

This name is useful when you want the output to read purely in terms of poset
vertices rather than representation-theoretic support.
"""
@inline nonzero_vertices(M::PModule) = support(M)

"""
    support_dims(M::PModule) -> NamedTuple

Return the support of `M` together with the corresponding nonzero stalk
dimensions.

The result has fields:
- `vertices`
- `dims`

where `vertices[i]` is in the support of `M` and `dims[i] = dim_at(M, vertices[i])`.
"""
function support_dims(M::PModule)
    verts = support(M)
    dims = Vector{Int}(undef, length(verts))
    @inbounds for i in eachindex(verts)
        dims[i] = M.dims[verts[i]]
    end
    return (vertices = verts, dims = dims)
end

"""
    structure_map(M; source, target, cache=nothing, opts=ModuleOptions()) -> AbstractMatrix

Return the structure map `M(source <= target)` of the module `M`.

This is the named-keyword companion to `map_leq(M, source, target; ...)`. It is
the more didactic call style for mathematicians, since the roles of the two
vertices are explicit at the call site.

Best practices
- prefer `structure_map(M; source=u, target=v)` in examples and notebook code,
- use `map_leq(M, u, v; ...)` in tight internal loops when positional calling is
  clearer,
- treat the returned matrix as read-only because it may alias cached internal
  storage.
"""
function structure_map(M::PModule{K};
                       source::Integer,
                       target::Integer,
                       cache::Union{Nothing,CoverCache}=nothing,
                       opts::ModuleOptions=ModuleOptions()) where {K}
    return map_leq(M, Int(source), Int(target); cache=cache, opts=opts)
end

@inline function _module_field_label(field::AbstractCoeffField)
    field isa QQField && return "QQ"
    field isa PrimeField && return "F$(field.p)"
    field isa RealField && return "Real"
    return string(typeof(field))
end

@inline function _module_matrix_zero(A::AbstractMatrix{K}) where {K}
    if K <: AbstractFloat
        tol = sqrt(eps(K))
        @inbounds for x in A
            abs(x) <= tol || return false
        end
        return true
    end
    if A isa SparseMatrixCSC
        return nnz(A) == 0
    end
    @inbounds for x in A
        iszero(x) || return false
    end
    return true
end

@inline function _module_nonzero_components(comps)
    count(A -> !_module_matrix_zero(A), comps)
end

@inline function _pmodule_total_dim(M::PModule)
    return sum(M.dims)
end

@inline function _pmodule_dimensions(M::PModule)
    return (
        vertices = nvertices(M.Q),
        stalks = copy(M.dims),
        total = _pmodule_total_dim(M),
        maximum_stalk = isempty(M.dims) ? 0 : maximum(M.dims),
        edges = length(M.edge_maps),
    )
end

@inline function _pmodule_dimensions(M::PModule, q::Int)
    return (
        vertex = q,
        stalk = M.dims[q],
    )
end

@inline function _pmodule_basis(M::PModule{K}, q::Int) where {K}
    1 <= q <= length(M.dims) || error("basis: vertex $(q) is out of range for a module on $(length(M.dims)) vertices")
    return eye(M.field, M.dims[q])
end

@inline function _pmodule_coordinates(M::PModule{K}, q::Int, x::AbstractVector{K}) where {K}
    1 <= q <= length(M.dims) || error("coordinates: vertex $(q) is out of range for a module on $(length(M.dims)) vertices")
    length(x) == M.dims[q] || error("coordinates: expected a vector of length $(M.dims[q]) at vertex $(q); got $(length(x))")
    return copy(x)
end

@inline function _pmodule_coordinates(M::PModule{K}, q::Int, X::AbstractMatrix{K}) where {K}
    1 <= q <= length(M.dims) || error("coordinates: vertex $(q) is out of range for a module on $(length(M.dims)) vertices")
    size(X, 1) == M.dims[q] || error("coordinates: expected $(M.dims[q]) rows at vertex $(q); got $(size(X, 1))")
    return Matrix{K}(X)
end

@inline function _pmodule_describe(M::PModule)
    dims = _pmodule_dimensions(M)
    return (
        kind = :pmodule,
        field = _module_field_label(M.field),
        vertices = dims.vertices,
        total_dimension = dims.total,
        maximum_stalk = dims.maximum_stalk,
        edge_count = dims.edges,
        poset_type = nameof(typeof(M.Q)),
    )
end

@inline function _pmorphism_dimensions(f)
    return (
        vertices = length(f.comps),
        domain = copy(f.dom.dims),
        codomain = copy(f.cod.dims),
        nonzero_components = _module_nonzero_components(f.comps),
    )
end

@inline function _pmorphism_describe(f)
    dims = _pmorphism_dimensions(f)
    return (
        kind = :pmorphism,
        field = _module_field_label(f.dom.field),
        vertices = dims.vertices,
        domain_total_dimension = sum(dims.domain),
        codomain_total_dimension = sum(dims.codomain),
        nonzero_components = dims.nonzero_components,
    )
end

@inline function _modules_validation_error(fn::AbstractString, issues::Vector{String})
    error("$(fn): validation failed:\n - " * join(issues, "\n - "))
end

"""
    check_module_data(Q, dims, edge_maps; throw=false) -> NamedTuple

Validate raw constructor data for a `PModule` before building the object.

Inputs
- `Q`: the underlying finite poset,
- `dims`: proposed stalk dimensions,
- `edge_maps`: proposed cover-edge structure maps, keyed by `(u,v)`.

The returned report has fields:
- `kind = :pmodule_data`
- `valid::Bool`
- `vertices`
- `edge_count`
- `issues::Vector{String}`

What is checked
- `length(dims) == nvertices(Q)`,
- all stalk dimensions are nonnegative,
- every cover edge of `Q` has exactly one proposed map,
- no non-cover edges are supplied,
- each supplied map has size `(dims[v], dims[u])`.

Best practices
- use this before `PModule(Q, dims, edge_maps; ...)` when building examples or
  notebook fixtures by hand,
- use `throw=true` in tests when invalid raw data should abort immediately.
"""
function check_module_data(Q::AbstractPoset,
                           dims::AbstractVector{<:Integer},
                           edge_maps;
                           throw::Bool=false)
    issues = String[]
    n = nvertices(Q)
    length(dims) == n || push!(issues,
        "expected dims to have length $(n); got $(length(dims))")
    any(<(0), dims) && push!(issues, "expected all stalk dimensions to be nonnegative")

    cover = collect(cover_edges(Q))
    cover_set = Set(cover)
    sizehint!(issues, length(cover))

    for (u, v) in cover
        if !haskey(edge_maps, (u, v))
            push!(issues, "expected one map for each cover edge; missing edge ($(u), $(v))")
            continue
        end
        A = edge_maps[(u, v)]
        du = 1 <= u <= length(dims) ? Int(dims[u]) : 0
        dv = 1 <= v <= length(dims) ? Int(dims[v]) : 0
        size(A) == (dv, du) || push!(issues,
            "expected map $(u) <= $(v) to have size ($(dv), $(du)); got $(size(A))")
    end

    for (uv, A) in pairs(edge_maps)
        uv isa Tuple{Int,Int} || begin
            push!(issues, "expected edge-map keys to be integer pairs; got $(typeof(uv))")
            continue
        end
        u, v = uv
        ((u, v) in cover_set) || push!(issues,
            "expected edge-map data only on cover edges; got ($(u), $(v))")
        A isa AbstractMatrix || push!(issues,
            "expected edge-map data at ($(u), $(v)) to be a matrix; got $(typeof(A))")
    end

    report = (
        kind = :pmodule_data,
        valid = isempty(issues),
        vertices = n,
        edge_count = length(edge_maps),
        issues = issues,
    )
    throw && !report.valid && _modules_validation_error("check_module_data", issues)
    return report
end

"""
    check_module(M; throw=false) -> NamedTuple

Validate a hand-built `PModule` without invoking heavier algebraic routines.

The returned report has fields:
- `kind = :pmodule`
- `valid::Bool`
- `vertices`
- `dims`
- `edge_count`
- `issues::Vector{String}`

Set `throw=true` to raise an error when validation fails.

What is checked
- the length and nonnegativity of the stalk-dimension vector,
- predecessor/successor cover-edge storage consistency,
- matrix sizes on every stored cover-edge map,
- agreement between predecessor and successor edge views,
- agreement between the number of stored maps and the cover graph of the poset.

When to use this
- immediately after building a module by hand,
- before debugging `map_leq`, kernels, cokernels, or derived functors,
- in tests with hand-computable examples.

Best practices
- Prefer `check_module(M)` over ad hoc inspection of internal fields.
- Use `check_module(M; throw=true)` in scripts and tests when invalid input
  should abort execution immediately.

This is a structural validation helper. It checks the module-storage contract,
not deeper semantic properties of a downstream computation.
"""
function check_module(M::PModule; throw::Bool=false)
    issues = String[]
    n = nvertices(M.Q)
    length(M.dims) == n || push!(issues,
        "expected dims to have length $(n); got $(length(M.dims))")
    any(<(0), M.dims) && push!(issues, "expected all stalk dimensions to be nonnegative")

    store = M.edge_maps
    length(store.preds) == n || push!(issues, "expected edge-map predecessor storage for $(n) vertices; got $(length(store.preds))")
    length(store.succs) == n || push!(issues, "expected edge-map successor storage for $(n) vertices; got $(length(store.succs))")
    length(store.maps_from_pred) == n || push!(issues, "expected predecessor maps for $(n) vertices; got $(length(store.maps_from_pred))")
    length(store.maps_to_succ) == n || push!(issues, "expected successor maps for $(n) vertices; got $(length(store.maps_to_succ))")

    nscan = min(n, length(store.preds), length(store.succs), length(store.maps_from_pred), length(store.maps_to_succ))
    for v in 1:nscan
        length(store.preds[v]) == length(store.maps_from_pred[v]) || push!(issues,
            "expected predecessor list and incoming maps to have the same length at vertex $(v)")
        issorted(store.preds[v]) || push!(issues, "expected predecessor list at vertex $(v) to be sorted")
        for (k, u) in pairs(store.preds[v])
            (1 <= u <= n) || push!(issues, "expected predecessor indices to lie in 1:$(n); got $(u) at vertex $(v)")
            size(store.maps_from_pred[v][k]) == (M.dims[v], M.dims[u]) || push!(issues,
                "expected map $(u) <= $(v) to have size ($(M.dims[v]), $(M.dims[u])); got $(size(store.maps_from_pred[v][k]))")
            haskey(store, u, v) || push!(issues, "expected cover-edge map store to contain edge ($(u), $(v))")
        end

        length(store.succs[v]) == length(store.maps_to_succ[v]) || push!(issues,
            "expected successor list and outgoing maps to have the same length at vertex $(v)")
        issorted(store.succs[v]) || push!(issues, "expected successor list at vertex $(v) to be sorted")
        for (k, w) in pairs(store.succs[v])
            (1 <= w <= n) || push!(issues, "expected successor indices to lie in 1:$(n); got $(w) at vertex $(v)")
            size(store.maps_to_succ[v][k]) == (M.dims[w], M.dims[v]) || push!(issues,
                "expected map $(v) <= $(w) to have size ($(M.dims[w]), $(M.dims[v])); got $(size(store.maps_to_succ[v][k]))")
            slot = _find_sorted_index(store.preds[w], v)
            slot != 0 || push!(issues, "expected successor edge ($(v), $(w)) to appear in predecessor storage as well")
        end
    end

    length(store) == length(collect(cover_edges(M.Q))) || push!(issues,
        "expected one structure map for each cover edge of the poset; got $(length(store)) maps for $(length(collect(cover_edges(M.Q)))) cover edges")

    report = (
        kind = :pmodule,
        valid = isempty(issues),
        vertices = n,
        dims = copy(M.dims),
        edge_count = length(store),
        issues = issues,
    )
    throw && !report.valid && _modules_validation_error("check_module", issues)
    return report
end

# ----------------------------
# Field coercion helpers
# ----------------------------

function _coerce_matrix(field::AbstractCoeffField, A::AbstractMatrix{K}) where {K}
    K2 = coeff_type(field)
    if A isa SparseMatrixCSC{K,Int}
        S = spzeros(K2, size(A, 1), size(A, 2))
        @inbounds for j in 1:size(A, 2)
            for idx in A.colptr[j]:(A.colptr[j + 1] - 1)
                i = A.rowval[idx]
                S[i, j] = coerce(field, A.nzval[idx])
            end
        end
        return S
    end

    M = Matrix{K2}(undef, size(A, 1), size(A, 2))
    @inbounds for j in 1:size(A, 2), i in 1:size(A, 1)
        M[i, j] = coerce(field, A[i, j])
    end
    return M
end

"""
    change_field(M, field)

Return a P-module obtained by coercing all structure maps into `field`.
"""
function change_field(M::PModule{K}, field::AbstractCoeffField) where {K}
    K2 = coeff_type(field)
    edge = Dict{Tuple{Int,Int}, AbstractMatrix{K2}}()
    sizehint!(edge, length(M.edge_maps))
    for ((u, v), A) in M.edge_maps
        edge[(u, v)] = _coerce_matrix(field, A)
    end
    return PModule{K2}(M.Q, M.dims, edge; field=field)
end

"""
    PMorphism(dom, cod, comps)

A morphism of modules on a common poset.

If `dom` and `cod` are `PModule`s on the same poset `Q`, then a `PMorphism`
stores a component map

`f_q : dom(q) -> cod(q)`

at each vertex `q`, and these component maps must commute with all structure
maps of the modules.

Best practices
- Construct morphisms with `PMorphism(...)` so shape checks happen immediately.
- Use `check_morphism(f)` for hand-built examples and debugging.
- Prefer `describe(f)` and `dimensions(f)` for inspection before dropping to
  raw component matrices in `f.comps`.

See also: `PModule`, `check_morphism`, `describe`, `dimensions`.
"""
struct PMorphism{K, F<:AbstractCoeffField, MatT<:AbstractMatrix{K}}
    dom::PModule{K,F}
    cod::PModule{K,F}
    comps::Vector{MatT}   # comps[i] has size cod.dims[i] x dom.dims[i]
    function PMorphism{K,F,MatT}(dom::PModule{K,F}, cod::PModule{K,F},
                                 comps::Vector{MatT}) where {K,F<:AbstractCoeffField,MatT<:AbstractMatrix{K}}
        return new{K,F,MatT}(dom, cod, comps)
    end
end

@inline function _morphism_mat_type(::Type{K}, comps, field::AbstractCoeffField) where {K}
    if _is_nemo_field(field) && FieldLinAlg._have_nemo()
        for A in comps
            if A isa AbstractMatrix{K} && _should_backendize_map(field, A)
                return BackendMatrix{K}
            end
        end
    end
    T = Base.eltype(comps)
    if T <: AbstractMatrix{K} && isconcretetype(T)
        return T
    end
    return Matrix{K}
end

function _build_p_morphism(dom::PModule{K,F}, cod::PModule{K,F},
                           comps::AbstractVector{<:AbstractMatrix{K}}) where {K,F}
    dom.field == cod.field || error("PMorphism: field mismatch")
    dom.Q === cod.Q || error("PMorphism: domain/codomain posets differ")
    n = nvertices(dom.Q)
    length(comps) == n || error("PMorphism: expected $n vertex components, got $(length(comps))")

    MatT = _morphism_mat_type(K, comps, dom.field)
    cvec = Vector{MatT}(undef, n)
    @inbounds for i in 1:n
        A = comps[i]
        if size(A, 1) != cod.dims[i] || size(A, 2) != dom.dims[i]
            error("PMorphism: component $i has size $(size(A)), expected ($(cod.dims[i]),$(dom.dims[i]))")
        end
        cvec[i] = convert(MatT, A)
    end
    return PMorphism{K,F,MatT}(dom, cod, cvec)
end

function PMorphism{K}(dom::PModule{K,F}, cod::PModule{K,F},
                      comps::AbstractVector{<:AbstractMatrix{K}}) where {K,F}
    return _build_p_morphism(dom, cod, comps)
end

function PMorphism{K,F}(dom::PModule{K,F}, cod::PModule{K,F},
                        comps::AbstractVector{<:AbstractMatrix{K}}) where {K,F}
    return _build_p_morphism(dom, cod, comps)
end

function PMorphism{K}(dom::PModule{K,F}, cod::PModule{K,F},
                      comps::Vector{MatT}) where {K,F,MatT<:AbstractMatrix{K}}
    return _build_p_morphism(dom, cod, comps)
end

function PMorphism{K,F}(dom::PModule{K,F}, cod::PModule{K,F},
                        comps::Vector{MatT}) where {K,F,MatT<:AbstractMatrix{K}}
    return _build_p_morphism(dom, cod, comps)
end

PMorphism(dom::PModule{K,F}, cod::PModule{K,F},
          comps::AbstractVector{<:AbstractMatrix{K}}) where {K,F} =
    _build_p_morphism(dom, cod, comps)

PMorphism(dom::PModule{K,F}, cod::PModule{K,F},
          comps::Vector{MatT}) where {K,F,MatT<:AbstractMatrix{K}} =
    _build_p_morphism(dom, cod, comps)

"""
    component(f::PMorphism, q::Integer) -> AbstractMatrix

Return the vertexwise component map `f_q` of a module morphism.

If `f : M -> N` is a `PMorphism`, then `component(f, q)` is the linear map

`f_q : M(q) -> N(q)`.

This is the canonical accessor for stalkwise morphism data. It is preferable to
reaching into `f.comps[q]` directly in user-facing code.
"""
function component(f::PMorphism, q::Integer)
    n = length(f.comps)
    1 <= q <= n || error("component: vertex $(q) is out of range for a morphism with $(n) components")
    return f.comps[Int(q)]
end

"""
    check_morphism_data(dom, cod, comps; throw=false) -> NamedTuple

Validate raw constructor data for a `PMorphism` before building the object.

Inputs
- `dom`, `cod`: the proposed domain and codomain modules,
- `comps`: one proposed component map per vertex.

The returned report has fields:
- `kind = :pmorphism_data`
- `valid::Bool`
- `vertices`
- `issues::Vector{String}`

What is checked
- structural validity of `dom` and `cod`,
- field agreement,
- poset identity,
- one component map per vertex,
- component sizes,
- commutativity on every cover-edge square.

Best practices
- use this before calling `PMorphism(dom, cod, comps)` in hand-built examples,
- use `throw=true` in tests when any invalid raw data should stop execution.
"""
function check_morphism_data(dom::PModule{K},
                             cod::PModule{K},
                             comps::AbstractVector{<:AbstractMatrix{K}};
                             throw::Bool=false) where {K}
    issues = String[]
    rep_dom = check_module(dom)
    rep_cod = check_module(cod)
    rep_dom.valid || append!(issues, "domain module: " .* rep_dom.issues)
    rep_cod.valid || append!(issues, "codomain module: " .* rep_cod.issues)

    dom.field == cod.field || push!(issues, "expected domain and codomain to use the same field")
    dom.Q === cod.Q || push!(issues, "expected domain and codomain to live on the same poset object")

    n = nvertices(dom.Q)
    length(comps) == n || push!(issues, "expected one component per vertex, so length(comps) = $(n); got $(length(comps))")

    nscan = min(length(comps), n)
    for v in 1:nscan
        size(comps[v]) == (cod.dims[v], dom.dims[v]) || push!(issues,
            "expected component $(v) to have size ($(cod.dims[v]), $(dom.dims[v])); got $(size(comps[v]))")
    end

    if dom.Q === cod.Q && length(comps) >= n
        for (u, v) in cover_edges(dom.Q)
            if !(haskey(dom.edge_maps, u, v) && haskey(cod.edge_maps, u, v))
                push!(issues, "expected both modules to carry the cover-edge map ($(u), $(v))")
                continue
            end
            size(comps[u]) == (cod.dims[u], dom.dims[u]) || continue
            size(comps[v]) == (cod.dims[v], dom.dims[v]) || continue
            lhs = cod.edge_maps[u, v] * comps[u]
            rhs = comps[v] * dom.edge_maps[u, v]
            _module_matrix_zero(lhs - rhs) || push!(issues,
                "expected the square on cover edge ($(u), $(v)) to commute")
        end
    end

    report = (
        kind = :pmorphism_data,
        valid = isempty(issues),
        vertices = n,
        issues = issues,
    )
    throw && !report.valid && _modules_validation_error("check_morphism_data", issues)
    return report
end

"""
    check_morphism(f; throw=false) -> NamedTuple

Validate a hand-built `PMorphism`.

The returned report has fields:
- `kind = :pmorphism`
- `valid::Bool`
- `vertices`
- `issues::Vector{String}`

Set `throw=true` to raise an error when validation fails.

What is checked
- domain/codomain field agreement,
- domain/codomain poset identity,
- one component map per vertex,
- component matrix sizes,
- commutativity on each cover-edge square.

When to use this
- after assembling a morphism by hand,
- before calling kernels, images, cokernels, or derived constructions,
- when debugging a morphism that should commute but appears not to.

Best practices
- Prefer `check_morphism(f)` over manually sampling a few component sizes.
- Use `check_morphism(f; throw=true)` when invalid input should stop execution
  immediately.

This checks the algebraic contract of a module morphism, but it does not infer
user intent beyond those commutativity and shape constraints.
"""
function check_morphism(f::PMorphism{K}; throw::Bool=false) where {K}
    issues = String[]
    dom = f.dom
    cod = f.cod
    n = nvertices(dom.Q)

    dom.field == cod.field || push!(issues, "expected domain and codomain to use the same field")
    dom.Q === cod.Q || push!(issues, "expected domain and codomain to live on the same poset object")
    length(f.comps) == n || push!(issues, "expected one component per vertex, so length(comps) = $(n); got $(length(f.comps))")

    nscan = min(length(f.comps), n)
    for v in 1:nscan
        size(f.comps[v]) == (cod.dims[v], dom.dims[v]) || push!(issues,
            "expected component $(v) to have size ($(cod.dims[v]), $(dom.dims[v])); got $(size(f.comps[v]))")
    end

    if dom.Q === cod.Q
        for (u, v) in cover_edges(dom.Q)
            if !(haskey(dom.edge_maps, u, v) && haskey(cod.edge_maps, u, v))
                push!(issues, "expected both modules to carry the cover-edge map ($(u), $(v))")
                continue
            end
            if u > length(f.comps) || v > length(f.comps)
                continue
            end
            try
                lhs = cod.edge_maps[u, v] * f.comps[u]
                rhs = f.comps[v] * dom.edge_maps[u, v]
                _module_matrix_zero(lhs - rhs) || push!(issues,
                    "expected the square on cover edge ($(u), $(v)) to commute")
            catch
                push!(issues, "could not compare the commutative square on cover edge ($(u), $(v)) because the component sizes are incompatible")
            end
        end
    end

    report = (
        kind = :pmorphism,
        valid = isempty(issues),
        vertices = n,
        issues = issues,
    )
    throw && !report.valid && _modules_validation_error("check_morphism", issues)
    return report
end

function Base.show(io::IO, M::PModule)
    d = _pmodule_describe(M)
    print(io, "PModule(field=", d.field, ", vertices=", d.vertices,
          ", total_dim=", d.total_dimension, ", edge_count=", d.edge_count, ")")
end

function Base.show(io::IO, ::MIME"text/plain", M::PModule)
    d = _pmodule_describe(M)
    print(io,
          "PModule\n",
          "  field: ", d.field, "\n",
          "  vertices: ", d.vertices, "\n",
          "  total dimension: ", d.total_dimension, "\n",
          "  max stalk dimension: ", d.maximum_stalk, "\n",
          "  edge count: ", d.edge_count, "\n",
          "  poset type: ", d.poset_type)
end

function Base.show(io::IO, f::PMorphism)
    d = _pmorphism_describe(f)
    print(io, "PMorphism(field=", d.field, ", vertices=", d.vertices,
          ", nonzero_components=", d.nonzero_components, ")")
end

function Base.show(io::IO, ::MIME"text/plain", f::PMorphism)
    d = _pmorphism_describe(f)
    print(io,
          "PMorphism\n",
          "  field: ", d.field, "\n",
          "  vertices: ", d.vertices, "\n",
          "  domain total dimension: ", d.domain_total_dimension, "\n",
          "  codomain total dimension: ", d.codomain_total_dimension, "\n",
          "  nonzero components: ", d.nonzero_components)
end

"""
    change_field(f, field)

Return a PMorphism obtained by coercing domain, codomain, and components into `field`.
"""
function change_field(f::PMorphism{K}, field::AbstractCoeffField) where {K}
    dom2 = change_field(f.dom, field)
    cod2 = change_field(f.cod, field)
    K2 = coeff_type(field)
    comps2 = Vector{Matrix{K2}}(undef, length(f.comps))
    @inbounds for i in 1:length(f.comps)
        comps2[i] = _coerce_matrix(field, f.comps[i])
    end
    return PMorphism{K2, typeof(field)}(dom2, cod2, comps2)
end

"Identity morphism."
id_morphism(M::PModule{K,F}) where {K,F} =
    PMorphism{K,F}(M, M, [eye(M.field, M.dims[i]) for i in 1:length(M.dims)])

@inline _map_leq_key(u::Int, v::Int)::UInt64 = _pairkey(u, v)

@inline function _thread_slot(n::Int)::Int
    return min(n, max(1, Threads.threadid()))
end

@inline function _map_leq_memo_dict(M::PModule{K,F,MatT}) where {K,F,MatT}
    return M.map_compose[_thread_slot(length(M.map_compose))]
end

@inline function _map_leq_memo_dense(M::PModule{K,F,MatT}) where {K,F,MatT}
    d = M.map_compose_dense
    d === nothing && return nothing
    return d[_thread_slot(length(d))]
end

@inline function _map_leq_many_plan_dict(M::PModule)
    return M.map_many_plan[_thread_slot(length(M.map_many_plan))]
end

@inline function _map_leq_pred_slot_dict(M::PModule)
    return M.map_pred_slot[_thread_slot(length(M.map_pred_slot))]
end

@inline function _map_leq_pred_slot_dense(M::PModule)
    d = M.map_pred_slot_dense
    d === nothing && return nothing
    return d[_thread_slot(length(d))]
end

@inline function _map_leq_many_batch_plan_dict(M::PModule)
    return M.map_many_batch_plan[_thread_slot(length(M.map_many_batch_plan))]
end

@inline function _map_leq_many_plan_arena(M::PModule)
    return M.map_many_plan_arena[_thread_slot(length(M.map_many_plan_arena))]
end

@inline function _map_leq_scratch(M::PModule{K}) where {K}
    return M.map_scratch[_thread_slot(length(M.map_scratch))]
end

@inline function _identity_compose_dict(M::PModule{K,F,MatT}) where {K,F,MatT}
    return M.identity_compose[_thread_slot(length(M.identity_compose))]
end

@inline function _map_leq_memo_get(M::PModule{K,F,MatT}, u::Int, v::Int) where {K,F,MatT}
    dense = _map_leq_memo_dense(M)
    if dense !== nothing
        @inbounds idx = (u - 1) * dense.n + v
        if dense.seen[idx]
            return dense.vals[idx]
        end
        return nothing
    end

    memo = _map_leq_memo_dict(M)
    A = get(memo, _map_leq_key(u, v), nothing)
    A === nothing && return nothing
    return A::MatT
end

@inline function _map_leq_memo_set!(M::PModule{K,F,MatT}, u::Int, v::Int, A) where {K,F,MatT}
    dense = _map_leq_memo_dense(M)
    if dense !== nothing
        @inbounds idx = (u - 1) * dense.n + v
        Atyped = A isa MatT ? A : convert(MatT, A)
        dense.vals[idx] = Atyped
        dense.seen[idx] = true
        return Atyped
    end

    memo = _map_leq_memo_dict(M)
    if length(memo) >= MAP_LEQ_MEMO_MAX_PER_THREAD[]
        empty!(memo)
    end
    Atyped = A isa MatT ? A : convert(MatT, A)
    memo[_map_leq_key(u, v)] = Atyped
    return Atyped
end

@inline function _map_leq_pred_slot_get(M::PModule, u::Int, v::Int)::Int
    dense = _map_leq_pred_slot_dense(M)
    if dense !== nothing
        @inbounds idx = (u - 1) * dense.n + v
        return dense.seen[idx] ? dense.vals[idx] : 0
    end
    return get(_map_leq_pred_slot_dict(M), _map_leq_key(u, v), 0)
end

@inline function _map_leq_pred_slot_set!(M::PModule, u::Int, v::Int, slot::Int)::Int
    dense = _map_leq_pred_slot_dense(M)
    if dense !== nothing
        @inbounds idx = (u - 1) * dense.n + v
        dense.vals[idx] = slot
        dense.seen[idx] = true
        return slot
    end
    _map_leq_pred_slot_dict(M)[_map_leq_key(u, v)] = slot
    return slot
end

@inline function _map_leq_last_pair_get(M::PModule{K,F,MatT}, u::Int, v::Int) where {K,F,MatT}
    c = M.map_last_pair
    slot = _thread_slot(length(c.us))
    @inbounds if c.seen[slot] && c.us[slot] == u && c.vs[slot] == v
        A = c.vals[slot]
        if !c.promoted[slot]
            _map_leq_memo_set!(M, u, v, A)
            c.promoted[slot] = true
        end
        return A
    end
    return nothing
end

@inline function _map_leq_last_pair_set!(M::PModule{K,F,MatT}, u::Int, v::Int, A) where {K,F,MatT}
    c = M.map_last_pair
    slot = _thread_slot(length(c.us))
    Atyped = A isa MatT ? A : convert(MatT, A)
    @inbounds begin
        c.seen[slot] = true
        c.us[slot] = u
        c.vs[slot] = v
        c.promoted[slot] = false
        c.vals[slot] = Atyped
    end
    return Atyped
end

@inline function _identity_map(M::PModule{K,F,MatT}, v::Int) where {K,F,MatT}
    d = M.dims[v]
    memo = _identity_compose_dict(M)
    A = get(memo, d, nothing)
    if A === nothing
        Aeye = eye(M.field, d)
        Atyped = Aeye isa MatT ? Aeye : convert(MatT, Aeye)
        memo[d] = Atyped
        return Atyped
    end
    return A::MatT
end

function _clear_map_leq_memo!(M::PModule)
    for d in M.map_compose
        empty!(d)
    end
    if M.map_compose_dense !== nothing
        for d in M.map_compose_dense
            fill!(d.seen, false)
        end
    end
    for d in M.map_pred_slot
        empty!(d)
    end
    if M.map_pred_slot_dense !== nothing
        for d in M.map_pred_slot_dense
            fill!(d.seen, false)
        end
    end
    fill!(M.map_last_pair.seen, false)
    fill!(M.map_last_pair.promoted, false)
    return nothing
end

function _clear_map_leq_many_plan_cache!(M::PModule)
    for d in M.map_many_plan
        empty!(d)
    end
    for d in M.map_many_batch_plan
        empty!(d)
    end
    fill!(M.map_many_batch_last, nothing)
    return nothing
end

@inline _map_leq_many_plan_key(pairs) = UInt(objectid(pairs))
@inline _map_leq_many_plan_key(batch::MapLeqQueryBatch) = UInt(batch.id)

@inline function _pairs_signature(pairs::AbstractVector{<:Tuple{Int,Int}})
    h = UInt(0x9e3779b97f4a7c15)
    @inbounds for i in eachindex(pairs)
        u, v = pairs[i]
        h = hash(v, hash(u, h))
    end
    return h
end

@inline function _is_cover_pair(cc::CoverCache, u::Int, v::Int)::Bool
    if cc.C !== nothing
        @inbounds return cc.C[u, v]
    end
    return _find_sorted_index(_succs(cc, u), v) != 0
end

@inline function _plan_signature_matches(plan::_MapLeqManyPlan, pairs)::Bool
    length(pairs) == plan.npairs || return false
    isempty(pairs) && return true
    @inbounds return pairs[firstindex(pairs)] == plan.first_pair &&
                     pairs[lastindex(pairs)] == plan.last_pair &&
                     _pairs_signature(pairs) == plan.pairs_hash
end

@inline function _reset_suffix_id_dense!(arena::_MapLeqManyPlanArena)
    dense = arena.suffix_id_dense
    touched = arena.suffix_id_touched
    @inbounds for idx in touched
        dense[idx] = 0
    end
    empty!(touched)
    return dense
end

@inline function _ensure_suffix_id_dense!(arena::_MapLeqManyPlanArena, n::Int)
    dense = arena.suffix_id_dense
    needed = n * n
    if length(dense) != needed
        resize!(dense, needed)
        fill!(dense, 0)
        empty!(arena.suffix_id_touched)
    else
        _reset_suffix_id_dense!(arena)
    end
    return dense
end

function _build_map_leq_many_suffix_program!(arena::_MapLeqManyPlanArena,
                                             pairs::AbstractVector{<:Tuple{Int,Int}},
                                             n::Int)
    query_suffix = arena.query_suffix
    suffix_u = arena.suffix_u
    suffix_v = arena.suffix_v
    suffix_child = arena.suffix_child
    suffix_child_node = arena.suffix_child_node
    suffix_head_slot = arena.suffix_head_slot
    suffix_tail_slot = arena.suffix_tail_slot
    suffix_hops = arena.suffix_hops
    chain_ptr = arena.chain_ptr
    chain_data = arena.chain_data
    chain_slots = arena.chain_slots
    kinds = arena.kinds
    dense = _ensure_suffix_id_dense!(arena, n)
    touched = arena.suffix_id_touched
    target_seen = arena.target_seen

    resize!(query_suffix, length(pairs))
    fill!(query_suffix, 0)
    empty!(suffix_u)
    empty!(suffix_v)
    empty!(suffix_child)
    empty!(suffix_child_node)
    empty!(suffix_head_slot)
    empty!(suffix_tail_slot)
    empty!(suffix_hops)
    resize!(target_seen, n)
    fill!(target_seen, false)

    total_long_hops = 0
    total_suffix_visits = 0
    ntargets = 0

    @inbounds for i in eachindex(pairs)
        kinds[i] == 0x03 || continue
        sidx = chain_ptr[i]
        eidx = chain_ptr[i + 1] - 1
        hops = eidx - sidx
        hops >= 3 || continue
        total_long_hops += hops
        total_suffix_visits += hops - 1

        v = chain_data[eidx]
        if !target_seen[v]
            target_seen[v] = true
            ntargets += 1
        end

        child_id = 0
        child_node = chain_data[eidx - 1]
        child_tail_slot = chain_slots[eidx]
        @inbounds for k in (eidx - 2):-1:sidx
            u = chain_data[k]
            idx = (u - 1) * n + v
            sid = dense[idx]
            if sid == 0
                sid = length(suffix_u) + 1
                dense[idx] = sid
                push!(touched, idx)
                push!(suffix_u, u)
                push!(suffix_v, v)
                push!(suffix_child, child_id)
                push!(suffix_child_node, child_node)
                push!(suffix_head_slot, chain_slots[k + 1])
                push!(suffix_tail_slot, child_id == 0 ? child_tail_slot : 0)
                push!(suffix_hops, child_id == 0 ? 2 : (suffix_hops[child_id] + 1))
            end
            child_id = sid
            child_node = u
            child_tail_slot = 0
        end
        query_suffix[i] = child_id
    end

    return total_long_hops, total_suffix_visits, length(suffix_u), ntargets
end

function _fill_map_leq_many_plan_arena!(arena::_MapLeqManyPlanArena,
                                        M::PModule,
                                        pairs::AbstractVector{<:Tuple{Int,Int}},
                                        Q::AbstractPoset,
                                        cc::CoverCache)
    m = length(pairs)
    kinds = arena.kinds
    us = arena.us
    vs = arena.vs
    mids = arena.mids
    chain_ptr = arena.chain_ptr
    chain_data = arena.chain_data
    chain_slots = arena.chain_slots
    tmp_chain = arena.tmp_chain
    tmp_slots = arena.tmp_slots

    resize!(kinds, m)
    resize!(us, m)
    resize!(vs, m)
    resize!(mids, m)
    fill!(mids, 0)
    resize!(chain_ptr, m + 1)
    chain_ptr[1] = 1
    empty!(chain_data)
    empty!(chain_slots)
    sizehint!(chain_data, max(16, 3m))
    sizehint!(chain_slots, max(16, 3m))
    n = nvertices(Q)
    empty!(tmp_chain)
    empty!(tmp_slots)
    h = UInt(0x9e3779b97f4a7c15)
    seen_pairs = Dict{UInt64,Nothing}()
    sizehint!(seen_pairs, m)
    has_repeats = false
    nlong = 0

    @inbounds for i in eachindex(pairs)
        u, v = pairs[i]
        h = hash(v, hash(u, h))
        pk = _map_leq_key(u, v)
        if !has_repeats
            if haskey(seen_pairs, pk)
                has_repeats = true
            else
                seen_pairs[pk] = nothing
            end
        end
        (1 <= u <= n && 1 <= v <= n) || error("map_leq_many!: indices out of range at i=$i (u=$u, v=$v)")
        us[i] = u
        vs[i] = v

        if u == v
            kinds[i] = 0x00
            chain_ptr[i + 1] = chain_ptr[i]
            continue
        end
        leq(Q, u, v) || error("map_leq_many!: need u <= v at i=$i (u=$u, v=$v)")

        if _is_cover_pair(cc, u, v)
            kinds[i] = 0x01
            chain_ptr[i + 1] = chain_ptr[i]
            continue
        end

        p = _chosen_predecessor(cc, u, v)
        if p == u
            kinds[i] = 0x01
            chain_ptr[i + 1] = chain_ptr[i]
            continue
        end

        if _is_cover_pair(cc, u, p)
            kinds[i] = 0x02
            mids[i] = p
            chain_ptr[i + 1] = chain_ptr[i]
            continue
        end

        kinds[i] = 0x03
        nlong += 1
        _build_cover_chain_with_slots!(tmp_chain, tmp_slots, M, u, v, cc)
        append!(chain_data, tmp_chain)
        push!(chain_slots, 0)
        append!(chain_slots, tmp_slots)
        chain_ptr[i + 1] = length(chain_data) + 1
    end

    first_pair = m == 0 ? (0, 0) : pairs[firstindex(pairs)]
    last_pair = m == 0 ? (0, 0) : pairs[lastindex(pairs)]
    total_long_hops, total_suffix_visits, nsuffix, ntargets =
        _build_map_leq_many_suffix_program!(arena, pairs, n)
    return _MapLeqManyBatchStats(m, h, first_pair, last_pair, has_repeats, nlong,
                                 total_long_hops, total_suffix_visits, nsuffix, ntargets)
end

@inline function _copy_map_leq_many_plan(arena::_MapLeqManyPlanArena,
                                         stats::_MapLeqManyBatchStats)
    return _MapLeqManyPlan(stats.npairs, stats.pairs_hash, stats.first_pair, stats.last_pair, stats.has_repeats, false,
                           copy(arena.kinds), copy(arena.us), copy(arena.vs), copy(arena.mids),
                           copy(arena.chain_ptr), copy(arena.chain_data), copy(arena.chain_slots),
                           copy(arena.query_suffix), copy(arena.suffix_u), copy(arena.suffix_v),
                           copy(arena.suffix_child), copy(arena.suffix_child_node),
                           copy(arena.suffix_head_slot), copy(arena.suffix_tail_slot),
                           copy(arena.suffix_hops))
end

function _build_map_leq_many_plan(M::PModule,
                                  pairs::AbstractVector{<:Tuple{Int,Int}},
                                  Q::AbstractPoset,
                                  cc::CoverCache,
                                  arena::_MapLeqManyPlanArena):: _MapLeqManyPlan
    stats = _fill_map_leq_many_plan_arena!(arena, M, pairs, Q, cc)
    return _copy_map_leq_many_plan(arena, stats)
end

function _lookup_map_leq_many_plan(M::PModule,
                                   pairs::AbstractVector{<:Tuple{Int,Int}})
    length(pairs) >= MAP_LEQ_MANY_PLAN_MIN_LEN[] || return nothing
    cache = _map_leq_many_plan_dict(M)
    k = _map_leq_many_plan_key(pairs)
    plan = get(cache, k, nothing)
    if plan !== nothing && _plan_signature_matches(plan, pairs)
        plan.warmed = true
        return plan
    end
    return nothing
end

function _get_or_build_map_leq_many_plan!(M::PModule,
                                          pairs::AbstractVector{<:Tuple{Int,Int}},
                                          Q::AbstractPoset,
                                          cc::CoverCache)
    plan = _lookup_map_leq_many_plan(M, pairs)
    plan === nothing || return plan
    length(pairs) >= MAP_LEQ_MANY_PLAN_MIN_LEN[] || return nothing
    cache = _map_leq_many_plan_dict(M)
    k = _map_leq_many_plan_key(pairs)
    arena = _map_leq_many_plan_arena(M)
    new_plan = _build_map_leq_many_plan(M, pairs, Q, cc, arena)
    if length(cache) >= MAP_LEQ_MANY_PLAN_MAX_PER_THREAD[]
        empty!(cache)
    end
    cache[k] = new_plan
    return new_plan
end

function _get_or_build_map_leq_many_plan!(M::PModule,
                                          batch::MapLeqQueryBatch,
                                          Q::AbstractPoset,
                                          cc::CoverCache)
    pairs = batch.pairs
    length(pairs) >= MAP_LEQ_MANY_PLAN_MIN_LEN[] || return nothing
    slot = _thread_slot(length(M.map_many_batch_last))
    last = M.map_many_batch_last[slot]
    if last !== nothing
        e = last::_MapLeqManyBatchPlanEntry
        if e.batch === batch
            e.plan.warmed = true
            return e.plan
        end
    end

    cache = _map_leq_many_batch_plan_dict(M)
    k = _map_leq_many_plan_key(batch)
    entry = get(cache, k, nothing)
    if entry !== nothing
        e = entry::_MapLeqManyBatchPlanEntry
        if e.batch === batch
            e.plan.warmed = true
            M.map_many_batch_last[slot] = e
            return e.plan
        end
    end

    arena = _map_leq_many_plan_arena(M)
    new_plan = _build_map_leq_many_plan(M, pairs, Q, cc, arena)
    if length(cache) >= MAP_LEQ_MANY_PLAN_MAX_PER_THREAD[]
        empty!(cache)
    end
    entry_new = _MapLeqManyBatchPlanEntry(batch, new_plan)
    cache[k] = entry_new
    M.map_many_batch_last[slot] = entry_new
    return new_plan
end

function _cache_map_leq_many_plan_from_arena!(M::PModule,
                                              pairs::AbstractVector{<:Tuple{Int,Int}},
                                              stats::_MapLeqManyBatchStats,
                                              arena::_MapLeqManyPlanArena)
    stats.npairs >= MAP_LEQ_MANY_PLAN_MIN_LEN[] || return nothing
    cache = _map_leq_many_plan_dict(M)
    k = _map_leq_many_plan_key(pairs)
    new_plan = _copy_map_leq_many_plan(arena, stats)
    if length(cache) >= MAP_LEQ_MANY_PLAN_MAX_PER_THREAD[]
        empty!(cache)
    end
    cache[k] = new_plan
    return new_plan
end

@inline function _map_leq_many_overlap_ratio(stats::_MapLeqManyBatchStats)
    stats.total_suffix_visits == 0 && return 0.0
    return 1.0 - stats.nsuffix / stats.total_suffix_visits
end

@inline function _map_leq_many_target_repeat_ratio(stats::_MapLeqManyBatchStats)
    stats.nlong == 0 && return 0.0
    return 1.0 - stats.ntargets / stats.nlong
end

@inline function _map_leq_many_avg_hops(stats::_MapLeqManyBatchStats)
    stats.nlong == 0 && return 0.0
    return stats.total_long_hops / stats.nlong
end

@inline function _map_leq_many_oneoff_overlap_threshold(field::AbstractCoeffField)
    if field isa QQField
        return MAP_LEQ_MANY_ONEOFF_MIN_OVERLAP_QQ[]
    elseif field isa PrimeField
        return MAP_LEQ_MANY_ONEOFF_MIN_OVERLAP_PRIME[]
    else
        return MAP_LEQ_MANY_ONEOFF_MIN_OVERLAP_REAL[]
    end
end

@inline function _map_leq_many_oneoff_target_repeat_threshold(field::AbstractCoeffField)
    if field isa QQField
        return MAP_LEQ_MANY_ONEOFF_MIN_TARGET_REPEAT_QQ[]
    elseif field isa PrimeField
        return MAP_LEQ_MANY_ONEOFF_MIN_TARGET_REPEAT_PRIME[]
    else
        return MAP_LEQ_MANY_ONEOFF_MIN_TARGET_REPEAT_REAL[]
    end
end

@inline function _map_leq_many_scalar_overlap_threshold(field::AbstractCoeffField)
    if field isa QQField
        return MAP_LEQ_MANY_LONG_SCALAR_MAX_OVERLAP_QQ[]
    elseif field isa PrimeField
        return MAP_LEQ_MANY_LONG_SCALAR_MAX_OVERLAP_PRIME[]
    else
        return MAP_LEQ_MANY_LONG_SCALAR_MAX_OVERLAP_REAL[]
    end
end

@inline function _map_leq_many_scalar_target_repeat_threshold(field::AbstractCoeffField)
    if field isa QQField
        return MAP_LEQ_MANY_LONG_SCALAR_MAX_TARGET_REPEAT_QQ[]
    elseif field isa PrimeField
        return MAP_LEQ_MANY_LONG_SCALAR_MAX_TARGET_REPEAT_PRIME[]
    else
        return MAP_LEQ_MANY_LONG_SCALAR_MAX_TARGET_REPEAT_REAL[]
    end
end

@inline function _map_leq_many_min_avg_hops(field::AbstractCoeffField)
    if field isa QQField
        return MAP_LEQ_MANY_LONG_MIN_AVG_HOPS_QQ[]
    elseif field isa PrimeField
        return MAP_LEQ_MANY_LONG_MIN_AVG_HOPS_PRIME[]
    else
        return MAP_LEQ_MANY_LONG_MIN_AVG_HOPS_REAL[]
    end
end

@inline function _map_leq_many_plan_score_threshold(field::AbstractCoeffField)
    if field isa QQField
        return MAP_LEQ_MANY_PLAN_MIN_SCORE_QQ[]
    elseif field isa PrimeField
        return MAP_LEQ_MANY_PLAN_MIN_SCORE_PRIME[]
    else
        return MAP_LEQ_MANY_PLAN_MIN_SCORE_REAL[]
    end
end

@inline function _map_leq_many_plan_score(stats::_MapLeqManyBatchStats)::Float64
    stats.npairs == 0 && return 0.0
    long_frac = stats.nlong / stats.npairs
    overlap = _map_leq_many_overlap_ratio(stats)
    target_repeat = _map_leq_many_target_repeat_ratio(stats)
    hop_term = clamp((_map_leq_many_avg_hops(stats) - 3.0) / 4.0, 0.0, 1.0)
    return 0.55 * overlap + 0.30 * target_repeat + 0.10 * hop_term + 0.05 * long_frac
end

@inline function _prefer_map_leq_many_plan_build(stats::_MapLeqManyBatchStats,
                                                 field::AbstractCoeffField)::Bool
    stats.npairs >= MAP_LEQ_MANY_PLAN_MIN_LEN[] || return false
    stats.has_repeats && return true
    stats.nlong == 0 && return true
    4 * stats.nlong < 3 * stats.npairs && return true
    2 * stats.nlong >= stats.npairs || return false
    _map_leq_many_avg_hops(stats) >= _map_leq_many_min_avg_hops(field) || return false
    return _map_leq_many_plan_score(stats) >= _map_leq_many_plan_score_threshold(field)
end

@inline function _use_map_leq_many_oneoff_long(stats::_MapLeqManyBatchStats,
                                               field::AbstractCoeffField)::Bool
    stats.has_repeats && return false
    stats.npairs >= MAP_LEQ_MANY_ONEOFF_LONG_MIN_LEN[] || return false
    stats.nlong >= MAP_LEQ_MANY_ONEOFF_LONG_MIN_LONG[] || return false
    2 * stats.nlong >= stats.npairs || return false
    avg_hops = _map_leq_many_avg_hops(stats)
    avg_hops >= _map_leq_many_min_avg_hops(field) || return false
    overlap = _map_leq_many_overlap_ratio(stats)
    target_repeat = _map_leq_many_target_repeat_ratio(stats)
    return overlap >= _map_leq_many_oneoff_overlap_threshold(field) ||
           target_repeat >= _map_leq_many_oneoff_target_repeat_threshold(field)
end

@inline function _prefer_map_leq_many_scalar_long(stats::_MapLeqManyBatchStats,
                                                  field::AbstractCoeffField)::Bool
    stats.has_repeats && return false
    stats.npairs >= MAP_LEQ_MANY_ONEOFF_LONG_MIN_LEN[] || return false
    stats.nlong >= MAP_LEQ_MANY_ONEOFF_LONG_MIN_LONG[] || return false
    2 * stats.nlong >= stats.npairs || return false
    avg_hops = _map_leq_many_avg_hops(stats)
    avg_hops >= _map_leq_many_min_avg_hops(field) || return false
    overlap = _map_leq_many_overlap_ratio(stats)
    target_repeat = _map_leq_many_target_repeat_ratio(stats)
    return overlap <= _map_leq_many_scalar_overlap_threshold(field) &&
           target_repeat <= _map_leq_many_scalar_target_repeat_threshold(field)
end

@inline function _new_suffix_vals(::Type{MatT}, n::Int) where {MatT}
    return Vector{MatT}(undef, n)
end

@inline function _use_qq_nemo_suffix_product(M::PModule{QQ},
                                             tail::AbstractMatrix{QQ},
                                             head::AbstractMatrix{QQ},
                                             hops::Int)::Bool
    FieldLinAlg._have_nemo() || return false
    hops >= MAP_LEQ_QQ_NEMO_LONG_MIN_HOPS[] || return false
    work = size(tail, 1) * size(head, 1) * size(head, 2)
    return work >= MAP_LEQ_QQ_NEMO_LONG_MIN_WORK[]
end

@inline function _mul_qq_nemo_into!(out::Matrix{QQ},
                                    s::MapLeqScratch{QQ},
                                    M::PModule{QQ},
                                    A::AbstractMatrix{QQ},
                                    B::AbstractMatrix{QQ})
    An = _qq_nemo_edge_matrix!(s, M, A)
    Bn = _qq_nemo_edge_matrix!(s, M, B)
    return _copy_from_fmpq_mat!(out, An * Bn)
end

@inline function _execute_suffix_program_dense!(suffix_vals::Vector{MatT},
                                                M::PModule{K,F,Matrix{K}},
                                                suffix_u::AbstractVector{Int},
                                                suffix_v::AbstractVector{Int},
                                                suffix_child::AbstractVector{Int},
                                                suffix_child_node::AbstractVector{Int},
                                                suffix_head_slot::AbstractVector{Int},
                                                suffix_tail_slot::AbstractVector{Int},
                                                suffix_hops::AbstractVector{Int};
                                                store_results::Bool) where {K,F,MatT}
    store = M.edge_maps
    @inbounds for sid in eachindex(suffix_u)
        u = suffix_u[sid]
        v = suffix_v[sid]
        memoA = _map_leq_memo_get(M, u, v)
        if memoA !== nothing
            suffix_vals[sid] = memoA
            continue
        end

        child = suffix_child[sid]
        child_node = suffix_child_node[sid]
        tail = child == 0 ? store.maps_from_pred[v][suffix_tail_slot[sid]] : suffix_vals[child]
        head = store.maps_from_pred[child_node][suffix_head_slot[sid]]

        A = if M.dims[u] == 1 && M.dims[v] == 1
            out = Matrix{K}(undef, 1, 1)
            out[1, 1] = tail[1, 1] * head[1, 1]
            _as_mattype(MatT, out)
        else
            out = Matrix{K}(undef, size(tail, 1), size(head, 2))
            if K === QQ && _use_qq_nemo_suffix_product(M, tail, head, suffix_hops[sid])
                s = _map_leq_scratch(M)
                _mul_qq_nemo_into!(out, s, M, tail, head)
            else
                _mul_dense_maybe_tiny!(out, tail, head)
            end
            _as_mattype(MatT, out)
        end
        suffix_vals[sid] = store_results ? _map_leq_memo_set!(M, u, v, A) : A
    end
    return suffix_vals
end

@inline function _execute_suffix_program_generic!(suffix_vals::Vector{MatT},
                                                  M::PModule{K,F,MatT},
                                                  suffix_u::AbstractVector{Int},
                                                  suffix_v::AbstractVector{Int},
                                                  suffix_child::AbstractVector{Int},
                                                  suffix_child_node::AbstractVector{Int},
                                                  suffix_head_slot::AbstractVector{Int},
                                                  suffix_tail_slot::AbstractVector{Int};
                                                  store_results::Bool) where {K,F,MatT}
    store = M.edge_maps
    @inbounds for sid in eachindex(suffix_u)
        u = suffix_u[sid]
        v = suffix_v[sid]
        memoA = _map_leq_memo_get(M, u, v)
        if memoA !== nothing
            suffix_vals[sid] = memoA
            continue
        end

        child = suffix_child[sid]
        child_node = suffix_child_node[sid]
        tail = child == 0 ? store.maps_from_pred[v][suffix_tail_slot[sid]] : suffix_vals[child]
        head = store.maps_from_pred[child_node][suffix_head_slot[sid]]

        A = if M.dims[u] == 1 && M.dims[v] == 1
            out = Matrix{K}(undef, 1, 1)
            out[1, 1] = tail[1, 1] * head[1, 1]
            _as_mattype(MatT, out)
        else
            _as_mattype(MatT, FieldLinAlg._matmul(tail, head))
        end
        suffix_vals[sid] = store_results ? _map_leq_memo_set!(M, u, v, A) : A
    end
    return suffix_vals
end

function _map_leq_many_with_plan_unchecked!(dest::AbstractVector,
                                            M::PModule{K,F,MatT},
                                            pairs::AbstractVector{<:Tuple{Int,Int}},
                                            plan::_MapLeqManyPlan) where {K,F,MatT}
    length(pairs) == plan.npairs || return false
    store_results = plan.has_repeats || plan.warmed
    suffix_vals = isempty(plan.suffix_u) ? nothing : _new_suffix_vals(MatT, length(plan.suffix_u))
    if suffix_vals !== nothing
        if MatT <: Matrix{K}
            _execute_suffix_program_dense!(suffix_vals, M, plan.suffix_u, plan.suffix_v,
                                           plan.suffix_child, plan.suffix_child_node,
                                           plan.suffix_head_slot, plan.suffix_tail_slot,
                                           plan.suffix_hops;
                                           store_results=store_results)
        else
            _execute_suffix_program_generic!(suffix_vals, M, plan.suffix_u, plan.suffix_v,
                                             plan.suffix_child, plan.suffix_child_node,
                                             plan.suffix_head_slot, plan.suffix_tail_slot;
                                             store_results=store_results)
        end
    end

    @inbounds for i in eachindex(pairs)
        kind = plan.kinds[i]
        u = plan.us[i]
        v = plan.vs[i]

        if kind == 0x00
            dest[i] = _identity_map(M, v)
            continue
        elseif kind == 0x01
            dest[i] = M.edge_maps[u, v]
            continue
        elseif kind == 0x03
            dest[i] = suffix_vals[plan.query_suffix[i]]
            continue
        end

        if M.dims[u] == 1 && M.dims[v] == 1
            p = plan.mids[i]
            coeff = M.edge_maps[p, v][1, 1] * M.edge_maps[u, p][1, 1]
            A = Matrix{K}(undef, 1, 1)
            A[1, 1] = coeff
            dest[i] = _as_mattype(MatT, A)
            continue
        end

        memoA = _map_leq_memo_get(M, u, v)
        if memoA !== nothing
            dest[i] = memoA
            continue
        end

        p = plan.mids[i]
        A = if MatT <: Matrix{K}
            s = _map_leq_scratch(M)
            E2 = M.edge_maps[p, v]
            E1 = M.edge_maps[u, p]
            out = _scratch_mat!(s, true, size(E2, 1), size(E1, 2))
            _mul_dense_maybe_tiny!(out, E2, E1)
            _as_mattype(MatT, copy(out))
        else
            _as_mattype(MatT, FieldLinAlg._matmul(M.edge_maps[p, v], M.edge_maps[u, p]))
        end

        dest[i] = store_results ? _map_leq_memo_set!(M, u, v, A) : A
    end
    return true
end

function _map_leq_many_with_plan!(dest::AbstractVector,
                                  M::PModule{K,F,MatT},
                                  pairs::AbstractVector{<:Tuple{Int,Int}},
                                  plan::_MapLeqManyPlan) where {K,F,MatT}
    _plan_signature_matches(plan, pairs) || return false
    return _map_leq_many_with_plan_unchecked!(dest, M, pairs, plan)
end
function _map_leq_many_oneoff_long_batch!(dest::AbstractVector,
                                          M::PModule{K,F,MatT},
                                          arena::_MapLeqManyPlanArena,
                                          stats::_MapLeqManyBatchStats) where {K,F,MatT}
    _use_map_leq_many_oneoff_long(stats, M.field) || return false
    suffix_vals = isempty(arena.suffix_u) ? nothing : _new_suffix_vals(MatT, length(arena.suffix_u))
    if suffix_vals !== nothing
        if MatT <: Matrix{K}
            _execute_suffix_program_dense!(suffix_vals, M, arena.suffix_u, arena.suffix_v,
                                           arena.suffix_child, arena.suffix_child_node,
                                           arena.suffix_head_slot, arena.suffix_tail_slot,
                                           arena.suffix_hops;
                                           store_results=false)
        else
            _execute_suffix_program_generic!(suffix_vals, M, arena.suffix_u, arena.suffix_v,
                                             arena.suffix_child, arena.suffix_child_node,
                                             arena.suffix_head_slot, arena.suffix_tail_slot;
                                             store_results=false)
        end
    end

    @inbounds for i in 1:stats.npairs
        kind = arena.kinds[i]
        u = arena.us[i]
        v = arena.vs[i]

        if kind == 0x00
            dest[i] = _identity_map(M, v)
        elseif kind == 0x01
            dest[i] = M.edge_maps[u, v]
        elseif kind == 0x03
            dest[i] = suffix_vals[arena.query_suffix[i]]
        elseif M.dims[u] == 1 && M.dims[v] == 1
            p = arena.mids[i]
            A = Matrix{K}(undef, 1, 1)
            A[1, 1] = M.edge_maps[p, v][1, 1] * M.edge_maps[u, p][1, 1]
            dest[i] = _as_mattype(MatT, A)
        elseif MatT <: Matrix{K}
            s = _map_leq_scratch(M)
            p = arena.mids[i]
            E2 = M.edge_maps[p, v]
            E1 = M.edge_maps[u, p]
            out = _scratch_mat!(s, true, size(E2, 1), size(E1, 2))
            _mul_dense_maybe_tiny!(out, E2, E1)
            dest[i] = _as_mattype(MatT, copy(out))
        else
            p = arena.mids[i]
            dest[i] = _as_mattype(MatT, FieldLinAlg._matmul(M.edge_maps[p, v], M.edge_maps[u, p]))
        end
    end
    return true
end

function _map_leq_many_scalar_long_batch!(dest::AbstractVector,
                                          M::PModule{K,F,MatT},
                                          arena::_MapLeqManyPlanArena,
                                          stats::_MapLeqManyBatchStats) where {K,F,MatT}
    _prefer_map_leq_many_scalar_long(stats, M.field) || return false
    chain_ptr = arena.chain_ptr
    chain_data = arena.chain_data
    chain_slots = arena.chain_slots
    @inbounds for i in 1:stats.npairs
        kind = arena.kinds[i]
        u = arena.us[i]
        v = arena.vs[i]
        if kind == 0x00
            dest[i] = _identity_map(M, v)
        elseif kind == 0x01
            dest[i] = M.edge_maps[u, v]
        elseif kind == 0x02
            p = arena.mids[i]
            if M.dims[u] == 1 && M.dims[v] == 1
                if MatT <: Matrix{K}
                    out = _ensure_dense_dest_matrix!(dest, i, K, 1, 1)
                    out[1, 1] = M.edge_maps[p, v][1, 1] * M.edge_maps[u, p][1, 1]
                else
                    A = Matrix{K}(undef, 1, 1)
                    A[1, 1] = M.edge_maps[p, v][1, 1] * M.edge_maps[u, p][1, 1]
                    dest[i] = _as_mattype(MatT, A)
                end
            elseif MatT <: Matrix{K}
                s = _map_leq_scratch(M)
                E2 = M.edge_maps[p, v]
                E1 = M.edge_maps[u, p]
                out = _ensure_dense_dest_matrix!(dest, i, K, size(E2, 1), size(E1, 2))
                _mul_dense_maybe_tiny!(out, E2, E1)
            else
                dest[i] = _as_mattype(MatT, FieldLinAlg._matmul(M.edge_maps[p, v], M.edge_maps[u, p]))
            end
        else
            sidx = chain_ptr[i]
            eidx = chain_ptr[i + 1] - 1
            if MatT <: Matrix{K}
                s = _map_leq_scratch(M)
                out = _ensure_dense_dest_matrix!(dest, i, K, M.dims[v], M.dims[u])
                if K === QQ && _use_qq_nemo_long_product(M, chain_data, chain_slots, sidx, eidx)
                    _compose_chain_dense_packed_slots_nemo_into!(out, M, chain_data, chain_slots, sidx, eidx, s)
                else
                    _compose_chain_dense_packed_slots_into!(out, M, chain_data, chain_slots, sidx, eidx, s)
                end
            else
                dest[i] = _as_mattype(MatT, _compose_chain_generic_packed_slots(M, chain_data, chain_slots, sidx, eidx))
            end
        end
    end
    return true
end


# ----------------------------
# Zero objects + direct sums (PModules)
# ----------------------------

"""
    zero_pmodule(Q::AbstractPoset; field=QQField())

The zero P-module on a finite poset Q (all stalks 0 and all structure maps 0).
"""
function zero_pmodule(Q::AbstractPoset; field::AbstractCoeffField=QQField())
    K = coeff_type(field)
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (u,v) in cover_edges(Q)
        edge[(u,v)] = zeros(K, 0, 0)
    end
    return PModule{K}(Q, zeros(Int, nvertices(Q)), edge; field=field)
end

"""
    zero_morphism(M::PModule{K}, N::PModule{K}) -> PMorphism{K}

Return the zero morphism `M -> N`.

This is the canonical way to build a morphism whose component at every vertex is
the zero matrix.

Best practices
- use this instead of assembling zero component matrices by hand,
- domain and codomain must live on the same poset and field,
- this is often the right starting point for hand-built exact-sequence and
  snake-lemma examples.
"""
function zero_morphism(M::PModule{K}, N::PModule{K}) where {K}
    Q = M.Q
    @assert N.Q === Q
    M.field == N.field || error("zero_morphism: field mismatch")
    n = nvertices(Q)
    comps = Vector{Matrix{K}}(undef, n)
    for i in 1:n
        comps[i] = zeros(K, N.dims[i], M.dims[i])
    end
    return PMorphism{K, typeof(M.field)}(M, N, comps)
end

@inline _is_sparse_map_type(::Type{MatT}) where {MatT} = MatT <: SparseMatrixCSC

@inline function _direct_sum_sparse_density_cap(field::AbstractCoeffField)
    if field isa RealField
        return DIRECT_SUM_SPARSE_MAX_DENSITY_REAL[]
    elseif field isa PrimeField
        return DIRECT_SUM_SPARSE_MAX_DENSITY_PRIME[]
    end
    return DIRECT_SUM_SPARSE_MAX_DENSITY[]
end

@inline function _direct_sum_sparse_min_avg_edge_entries(field::AbstractCoeffField)
    if field isa RealField
        return DIRECT_SUM_SPARSE_MIN_AVG_EDGE_ENTRIES_REAL[]
    elseif field isa PrimeField
        return DIRECT_SUM_SPARSE_MIN_AVG_EDGE_ENTRIES_PRIME[]
    end
    return DIRECT_SUM_SPARSE_MIN_AVG_EDGE_ENTRIES_QQ[]
end

@inline function _direct_sum_dense_weight(field::AbstractCoeffField)
    if field isa RealField
        return DIRECT_SUM_COST_DENSE_WEIGHT_REAL[]
    elseif field isa PrimeField
        return DIRECT_SUM_COST_DENSE_WEIGHT_PRIME[]
    end
    return DIRECT_SUM_COST_DENSE_WEIGHT_QQ[]
end

@inline function _direct_sum_sparse_nnz_weight(field::AbstractCoeffField)
    if field isa RealField
        return DIRECT_SUM_COST_SPARSE_NNZ_WEIGHT_REAL[]
    elseif field isa PrimeField
        return DIRECT_SUM_COST_SPARSE_NNZ_WEIGHT_PRIME[]
    end
    return DIRECT_SUM_COST_SPARSE_NNZ_WEIGHT_QQ[]
end

@inline function _direct_sum_sparse_edge_overhead(field::AbstractCoeffField)
    if field isa RealField
        return DIRECT_SUM_COST_SPARSE_EDGE_OVERHEAD_REAL[]
    elseif field isa PrimeField
        return DIRECT_SUM_COST_SPARSE_EDGE_OVERHEAD_PRIME[]
    end
    return DIRECT_SUM_COST_SPARSE_EDGE_OVERHEAD_QQ[]
end

@inline function _direct_sum_estimated_entries(sa::_DirectSumEdgeStats, sb::_DirectSumEdgeStats)
    n_edges = max(1, min(sa.n_edges, sb.n_edges))
    cross_est = (Float64(sa.sum_cod_dims) * Float64(sb.sum_dom_dims) +
                 Float64(sb.sum_cod_dims) * Float64(sa.sum_dom_dims)) / Float64(n_edges)
    return Float64(sa.total_entries + sb.total_entries) + cross_est
end

function _direct_sum_sparse_preferred(A::PModule{K,FA,MatA},
                                      B::PModule{K,FB,MatB}) where {K,FA,FB,MatA,MatB}
    # Sparse output only pays off reliably when both inputs are already sparse.
    (_is_sparse_map_type(MatA) && _is_sparse_map_type(MatB)) || return false
    sa = A.direct_sum_stats
    sb = B.direct_sum_stats
    sa.n_edges == sb.n_edges || return false

    total_entries = _direct_sum_estimated_entries(sa, sb)
    total_entries == 0.0 && return false
    total_entries < DIRECT_SUM_SPARSE_MIN_TOTAL_ENTRIES[] && return false
    avg_entries = total_entries / Float64(sa.n_edges)
    avg_entries >= _direct_sum_sparse_min_avg_edge_entries(A.field) || return false

    total_nnz = Float64(sa.total_nnz + sb.total_nnz)
    density_est = total_nnz / total_entries
    density_est <= _direct_sum_sparse_density_cap(A.field) || return false

    dense_cost = _direct_sum_dense_weight(A.field) * total_entries
    sparse_cost = _direct_sum_sparse_nnz_weight(A.field) * total_nnz +
                  _direct_sum_sparse_edge_overhead(A.field) * Float64(sa.n_edges)
    return sparse_cost <= dense_cost
end

@inline function _direct_sum_mat_type(::Type{K},
                                      A::PModule{K,FA,MatA},
                                      B::PModule{K,FB,MatB}) where {K,FA,FB,MatA,MatB}
    return _direct_sum_sparse_preferred(A, B) ? SparseMatrixCSC{K,Int} : Matrix{K}
end

@inline _col_nnz(A::SparseMatrixCSC, j::Int) = @inbounds A.colptr[j + 1] - A.colptr[j]
@inline function _col_nnz(A::AbstractMatrix, j::Int)
    c = 0
    @inbounds for i in 1:size(A, 1)
        iszero(A[i, j]) || (c += 1)
    end
    return c
end

@inline function _write_col_nonzeros!(rowval::Vector{Int},
                                      nzval::Vector{K},
                                      ptr::Int,
                                      A::SparseMatrixCSC{K,Int},
                                      j::Int,
                                      row_shift::Int) where {K}
    @inbounds for idx in A.colptr[j]:(A.colptr[j + 1] - 1)
        rowval[ptr] = A.rowval[idx] + row_shift
        nzval[ptr] = A.nzval[idx]
        ptr += 1
    end
    return ptr
end

@inline function _write_col_nonzeros!(rowval::Vector{Int},
                                      nzval::Vector{K},
                                      ptr::Int,
                                      A::AbstractMatrix{K},
                                      j::Int,
                                      row_shift::Int) where {K}
    @inbounds for i in 1:size(A, 1)
        v = A[i, j]
        iszero(v) && continue
        rowval[ptr] = i + row_shift
        nzval[ptr] = v
        ptr += 1
    end
    return ptr
end

@inline function _direct_sum_edge_map(::Type{Matrix{K}},
                                      Au::AbstractMatrix{K},
                                      Bu::AbstractMatrix{K},
                                      aU::Int, bU::Int, aV::Int, bV::Int) where {K}
    Muv = zeros(K, aV + bV, aU + bU)
    if aV != 0 && aU != 0
        copyto!(view(Muv, 1:aV, 1:aU), Au)
    end
    if bV != 0 && bU != 0
        copyto!(view(Muv, aV+1:aV+bV, aU+1:aU+bU), Bu)
    end
    return Muv
end

@inline function _direct_sum_edge_map(::Type{SparseMatrixCSC{K,Int}},
                                      Au::AbstractMatrix{K},
                                      Bu::AbstractMatrix{K},
                                      aU::Int, bU::Int, aV::Int, bV::Int) where {K}
    out_m = aV + bV
    out_n = aU + bU
    out_n == 0 && return spzeros(K, out_m, out_n)

    colptr = Vector{Int}(undef, out_n + 1)
    colptr[1] = 1
    nextptr = 1

    @inbounds for j in 1:aU
        nextptr += _col_nnz(Au, j)
        colptr[j + 1] = nextptr
    end
    @inbounds for j in 1:bU
        nextptr += _col_nnz(Bu, j)
        colptr[aU + j + 1] = nextptr
    end

    nnz_total = nextptr - 1
    rowval = Vector{Int}(undef, nnz_total)
    nzval = Vector{K}(undef, nnz_total)

    @inbounds for j in 1:aU
        ptr = colptr[j]
        _write_col_nonzeros!(rowval, nzval, ptr, Au, j, 0)
    end
    @inbounds for j in 1:bU
        ptr = colptr[aU + j]
        _write_col_nonzeros!(rowval, nzval, ptr, Bu, j, aV)
    end

    return SparseMatrixCSC(out_m, out_n, colptr, rowval, nzval)
end

"""
    direct_sum(A::PModule{K}, B::PModule{K}) -> PModule{K}

Return the binary direct sum `A oplus B` as a `PModule`.

At each vertex `q`, the stalk is `A(q) oplus B(q)`, and each structure map is the
block-diagonal direct sum of the corresponding structure maps of `A` and `B`.

Best practices
- use this as the canonical binary coproduct/product object in the module
  category,
- use `direct_sum_with_maps` when you also want the canonical injections and
  projections,
- both modules must live on the same poset and use the same field.
"""
function direct_sum(A::PModule{K,FA,MatA}, B::PModule{K,FB,MatB}) where {K,FA,FB,MatA,MatB}
    Q = A.Q
    n = nvertices(Q)
    @assert B.Q === Q
    A.field == B.field || error("direct_sum: field mismatch")

    dims = [A.dims[i] + B.dims[i] for i in 1:n]
    OutMatT = _direct_sum_mat_type(K, A, B)

    # Fast path: traverse cover edges via store-aligned succ lists and grab maps
    # by index (no tuple allocation, no search).
    cc = _get_cover_cache(Q)
    if (all(A.edge_maps.succs[u] == _succs(cc, u) for u in 1:n) &&
        all(A.edge_maps.preds[v] == _preds(cc, v) for v in 1:n) &&
        all(B.edge_maps.succs[u] == _succs(cc, u) for u in 1:n) &&
        all(B.edge_maps.preds[v] == _preds(cc, v) for v in 1:n))

        preds = [_preds(cc, v) for v in 1:n]
        succs = [_succs(cc, u) for u in 1:n]

        maps_from_pred = [Vector{OutMatT}(undef, length(preds[v])) for v in 1:n]
        maps_to_succ   = [Vector{OutMatT}(undef, length(succs[u])) for u in 1:n]

        @inbounds for u in 1:n
            su = succs[u]
            pred_slots = _pred_slots_of_succ(cc, u)
            Au = A.edge_maps.maps_to_succ[u]
            Bu = B.edge_maps.maps_to_succ[u]
            outu = maps_to_succ[u]

            aU, bU = A.dims[u], B.dims[u]

            for j in eachindex(su)
                v = su[j]
                aV, bV = A.dims[v], B.dims[v]

                Muv = _direct_sum_edge_map(OutMatT, Au[j], Bu[j], aU, bU, aV, bV)

                outu[j] = Muv
                ip = pred_slots[j]
                @inbounds maps_from_pred[v][ip] = Muv
            end
        end

        store = CoverEdgeMapStore{K,OutMatT}(preds, succs, maps_from_pred, maps_to_succ, cc.nedges)
        return PModule{K}(Q, dims, store; field=A.field)
    end

    # Fallback (rare): use keyed access.
    edge = Dict{Tuple{Int,Int}, OutMatT}()
    sizehint!(edge, length(A.edge_maps))
    for (u, v) in cover_edges(Q)
        Au = A.edge_maps[u, v]
        Bu = B.edge_maps[u, v]
        aU, bU = A.dims[u], B.dims[u]
        aV, bV = A.dims[v], B.dims[v]
        edge[(u, v)] = _direct_sum_edge_map(OutMatT, Au, Bu, aU, bU, aV, bV)
    end
    return PModule{K}(Q, dims, edge; field=A.field)
end


"""
    direct_sum_with_maps(A,B) -> (S, iA, iB, pA, pB)

Direct sum together with canonical injections/projections.
"""
function direct_sum_with_maps(A::PModule{K}, B::PModule{K}) where {K}
    S = direct_sum(A, B)
    Q = A.Q
    n = nvertices(Q)
    @assert B.Q === Q

    iA_comps = Vector{Matrix{K}}(undef, n)
    iB_comps = Vector{Matrix{K}}(undef, n)
    pA_comps = Vector{Matrix{K}}(undef, n)
    pB_comps = Vector{Matrix{K}}(undef, n)

    @inbounds for u in 1:n
        a = A.dims[u]
        b = B.dims[u]

        iA = zeros(K, a + b, a)
        iB = zeros(K, a + b, b)
        pA = zeros(K, a, a + b)
        pB = zeros(K, b, a + b)

        for t in 1:a
            iA[t, t] = one(K)
            pA[t, t] = one(K)
        end
        for t in 1:b
            iB[a + t, t] = one(K)
            pB[t, a + t] = one(K)
        end

        iA_comps[u] = iA
        iB_comps[u] = iB
        pA_comps[u] = pA
        pB_comps[u] = pB
    end

    F = typeof(S.field)
    iA = PMorphism{K,F}(A, S, iA_comps)
    iB = PMorphism{K,F}(B, S, iB_comps)
    pA = PMorphism{K,F}(S, A, pA_comps)
    pB = PMorphism{K,F}(S, B, pB_comps)
    return S, iA, iB, pA, pB
end




@inline function _scratch_pool_lookup!(current::Matrix{K},
                                       pool::Vector{Matrix{K}},
                                       m::Int, n::Int)::Matrix{K} where {K}
    if size(current, 1) == m && size(current, 2) == n
        return current
    end
    @inbounds for A in pool
        if size(A, 1) == m && size(A, 2) == n
            return A
        end
    end
    A = Matrix{K}(undef, m, n)
    push!(pool, A)
    return A
end

@inline function _scratch_mat!(s::MapLeqScratch{K}, which::Bool, m::Int, n::Int)::Matrix{K} where {K}
    if which
        s.tmp1 = _scratch_pool_lookup!(s.tmp1, s.pool1, m, n)
        return s.tmp1
    else
        s.tmp2 = _scratch_pool_lookup!(s.tmp2, s.pool2, m, n)
        return s.tmp2
    end
end

@inline function _qq_nemo_long_product_work(M::PModule{QQ},
                                            chain_data::Vector{Int},
                                            chain_slots::Vector{Int},
                                            sidx::Int, eidx::Int)::Int
    dom = M.dims[chain_data[sidx]]
    work = 0
    store = M.edge_maps
    @inbounds for k in (sidx + 2):eidx
        E = store.maps_from_pred[chain_data[k]][chain_slots[k]]
        work += size(E, 1) * size(E, 2) * dom
    end
    return work
end

@inline function _use_qq_nemo_long_product(M::PModule{QQ},
                                           chain_data::Vector{Int},
                                           chain_slots::Vector{Int},
                                           sidx::Int, eidx::Int)::Bool
    FieldLinAlg._have_nemo() || return false
    hops = eidx - sidx
    hops >= MAP_LEQ_QQ_NEMO_LONG_MIN_HOPS[] || return false
    return _qq_nemo_long_product_work(M, chain_data, chain_slots, sidx, eidx) >=
           MAP_LEQ_QQ_NEMO_LONG_MIN_WORK[]
end

@inline function _qq_nemo_edge_matrix!(s::MapLeqScratch{QQ},
                                       M::PModule{QQ},
                                       A::AbstractMatrix{QQ})
    if s.qq_nemo_owner !== M
        empty!(s.qq_nemo_edge)
        s.qq_nemo_owner = M
    end
    cached = get(s.qq_nemo_edge, A, nothing)
    cached === nothing || return cached
    cached = FieldLinAlg._to_fmpq_mat(A)
    s.qq_nemo_edge[A] = cached
    return cached
end

@inline function _copy_from_fmpq_mat!(out::Matrix{QQ}, M)
    size(out, 1) == size(M, 1) || throw(DimensionMismatch("row mismatch in _copy_from_fmpq_mat!"))
    size(out, 2) == size(M, 2) || throw(DimensionMismatch("column mismatch in _copy_from_fmpq_mat!"))
    Nemo = FieldLinAlg.Nemo
    @inbounds for j in 1:size(M, 2), i in 1:size(M, 1)
        x = M[i, j]
        out[i, j] = QQ(BigInt(Nemo.numerator(x)), BigInt(Nemo.denominator(x)))
    end
    return out
end

@inline function _build_cover_chain!(chain::Vector{Int}, u::Int, v::Int, cc::CoverCache)
    empty!(chain)
    push!(chain, v)
    d = v
    while d != u
        d = _chosen_predecessor(cc, u, d)
        push!(chain, d)
    end
    reverse!(chain)
    return chain
end

@inline function _build_cover_chain_with_slots!(chain::Vector{Int},
                                                slots::Vector{Int},
                                                M::PModule,
                                                u::Int, v::Int,
                                                cc::CoverCache)
    empty!(chain)
    empty!(slots)
    push!(chain, v)
    d = v
    while d != u
        p, slot = _chosen_predecessor_with_slot(M, cc, u, d)
        push!(chain, p)
        push!(slots, slot)
        d = p
    end
    reverse!(chain)
    reverse!(slots)
    return chain, slots
end

@inline function _chosen_predecessor_with_slot(M::PModule, cc::CoverCache, a::Int, d::Int)::Tuple{Int,Int}
    lo = cc.pred_ptr[d]
    hi = cc.pred_ptr[d + 1] - 1
    slot = _map_leq_pred_slot_get(M, a, d)
    if slot != 0
        @inbounds return cc.pred_idx[lo + slot - 1], slot
    end

    dense = _chain_parent_dense(cc)
    if dense !== nothing
        idx = (a - 1) * dense.n + d
        @inbounds if dense.seen[idx]
            p = dense.vals[idx]
            slot = _find_sorted_index_range(cc.pred_idx, lo, hi, p)
            slot == 0 || _map_leq_pred_slot_set!(M, a, d, slot)
            return p, slot
        end
        slot_a = 0
        @inbounds for k in lo:hi
            b = cc.pred_idx[k]
            slot = k - lo + 1
            if b == a
                slot_a = slot
            elseif leq(cc.Q, a, b)
                dense.vals[idx] = b
                dense.seen[idx] = true
                _map_leq_pred_slot_set!(M, a, d, slot)
                return b, slot
            end
        end
        dense.vals[idx] = a
        dense.seen[idx] = true
        slot_a == 0 || _map_leq_pred_slot_set!(M, a, d, slot_a)
        return a, slot_a
    end

    cache = _chain_parent_dict(cc)
    key = _pairkey(a, d)
    p = get(cache, key, 0)
    if p != 0
        slot = _find_sorted_index_range(cc.pred_idx, lo, hi, p)
        slot == 0 || _map_leq_pred_slot_set!(M, a, d, slot)
        return p, slot
    end

    slot_a = 0
    @inbounds for k in lo:hi
        b = cc.pred_idx[k]
        slot = k - lo + 1
        if b == a
            slot_a = slot
        elseif leq(cc.Q, a, b)
            cache[key] = b
            _map_leq_pred_slot_set!(M, a, d, slot)
            return b, slot
        end
    end
    cache[key] = a
    slot_a == 0 || _map_leq_pred_slot_set!(M, a, d, slot_a)
    return a, slot_a
end

@inline function _compose_chain_dense_packed!(M::PModule{K},
                                              chain_data::Vector{Int},
                                              sidx::Int, eidx::Int,
                                              s::MapLeqScratch{K}) where {K}
    first_map = M.edge_maps[chain_data[sidx], chain_data[sidx + 1]]
    acc = _scratch_mat!(s, true, size(first_map, 1), size(first_map, 2))
    copyto!(acc, first_map)
    acc_is_tmp1 = true

    @inbounds for k in (sidx + 1):(eidx - 1)
        E = M.edge_maps[chain_data[k], chain_data[k + 1]]
        out = _scratch_mat!(s, !acc_is_tmp1, size(E, 1), size(acc, 2))
        _mul_dense_maybe_tiny!(out, E, acc)
        acc = out
        acc_is_tmp1 = !acc_is_tmp1
    end
    return copy(acc)
end

@inline function _compose_chain_dense_packed_slots!(M::PModule{K},
                                                    chain_data::Vector{Int},
                                                    chain_slots::Vector{Int},
                                                    sidx::Int, eidx::Int,
                                                    s::MapLeqScratch{K}) where {K}
    store = M.edge_maps
    first_map = store.maps_from_pred[chain_data[sidx + 1]][chain_slots[sidx + 1]]
    acc = _scratch_mat!(s, true, size(first_map, 1), size(first_map, 2))
    copyto!(acc, first_map)
    acc_is_tmp1 = true

    @inbounds for k in (sidx + 2):eidx
        E = store.maps_from_pred[chain_data[k]][chain_slots[k]]
        out = _scratch_mat!(s, !acc_is_tmp1, size(E, 1), size(acc, 2))
        _mul_dense_maybe_tiny!(out, E, acc)
        acc = out
        acc_is_tmp1 = !acc_is_tmp1
    end
    return copy(acc)
end

@inline function _compose_chain_dense_packed_slots_into!(out::Matrix{K},
                                                         M::PModule{K},
                                                         chain_data::Vector{Int},
                                                         chain_slots::Vector{Int},
                                                         sidx::Int, eidx::Int,
                                                         s::MapLeqScratch{K}) where {K}
    store = M.edge_maps
    first_map = store.maps_from_pred[chain_data[sidx + 1]][chain_slots[sidx + 1]]
    if sidx + 1 == eidx
        copyto!(out, first_map)
        return out
    end

    acc = _scratch_mat!(s, true, size(first_map, 1), size(first_map, 2))
    copyto!(acc, first_map)
    acc_is_tmp1 = true

    @inbounds for k in (sidx + 2):(eidx - 1)
        E = store.maps_from_pred[chain_data[k]][chain_slots[k]]
        tmp = _scratch_mat!(s, !acc_is_tmp1, size(E, 1), size(acc, 2))
        _mul_dense_maybe_tiny!(tmp, E, acc)
        acc = tmp
        acc_is_tmp1 = !acc_is_tmp1
    end

    last_edge = store.maps_from_pred[chain_data[eidx]][chain_slots[eidx]]
    _mul_dense_maybe_tiny!(out, last_edge, acc)
    return out
end

@inline function _compose_chain_dense_packed_slots_nemo_into!(out::Matrix{QQ},
                                                              M::PModule{QQ},
                                                              chain_data::Vector{Int},
                                                              chain_slots::Vector{Int},
                                                              sidx::Int, eidx::Int,
                                                              s::MapLeqScratch{QQ})
    store = M.edge_maps
    acc = _qq_nemo_edge_matrix!(s, M, store.maps_from_pred[chain_data[sidx + 1]][chain_slots[sidx + 1]])
    @inbounds for k in (sidx + 2):eidx
        E = _qq_nemo_edge_matrix!(s, M, store.maps_from_pred[chain_data[k]][chain_slots[k]])
        acc = E * acc
    end
    return _copy_from_fmpq_mat!(out, acc)
end

@inline function _compose_chain_generic_packed(M::PModule,
                                               chain_data::Vector{Int},
                                               sidx::Int, eidx::Int)
    A = M.edge_maps[chain_data[sidx], chain_data[sidx + 1]]
    @inbounds for k in (sidx + 1):(eidx - 1)
        E = M.edge_maps[chain_data[k], chain_data[k + 1]]
        A = FieldLinAlg._matmul(E, A)
    end
    return A
end

@inline function _compose_chain_generic_packed_slots(M::PModule,
                                                     chain_data::Vector{Int},
                                                     chain_slots::Vector{Int},
                                                     sidx::Int, eidx::Int)
    store = M.edge_maps
    A = store.maps_from_pred[chain_data[sidx + 1]][chain_slots[sidx + 1]]
    @inbounds for k in (sidx + 2):eidx
        E = store.maps_from_pred[chain_data[k]][chain_slots[k]]
        A = FieldLinAlg._matmul(E, A)
    end
    return A
end

@inline function _ensure_dense_dest_matrix!(dest::AbstractVector, i::Int,
                                            ::Type{K}, m::Int, n::Int)::Matrix{K} where {K}
    if Base.isassigned(dest, i)
        A = dest[i]
        if A isa Matrix{K} && size(A, 1) == m && size(A, 2) == n
            return A
        end
    end
    A = Matrix{K}(undef, m, n)
    dest[i] = A
    return A
end

@inline function _mul_tiny_dense!(out::Matrix{K},
                                  A::AbstractMatrix{K},
                                  B::AbstractMatrix{K}) where {K}
    fill!(out, zero(K))
    m, k = size(A)
    _, n = size(B)
    @inbounds for j in 1:n
        for t in 1:k
            bt = B[t, j]
            iszero(bt) && continue
            for i in 1:m
                out[i, j] += A[i, t] * bt
            end
        end
    end
    return out
end

@inline function _mul_dense_maybe_tiny!(out::Matrix{K},
                                        A::AbstractMatrix{K},
                                        B::AbstractMatrix{K}) where {K}
    if FieldLinAlg._is_tiny_mul(A, B)
        return _mul_tiny_dense!(out, A, B)
    end
    mul!(out, A, B)
    return out
end

@inline function _compose_chain_dense_backward!(M::PModule{K},
                                                u::Int, v::Int,
                                                cc::CoverCache,
                                                s::MapLeqScratch{K}) where {K}
    store = M.edge_maps
    d = v
    p, slot = _chosen_predecessor_with_slot(M, cc, u, d)
    slot == 0 && error("missing cover-edge map for ($p,$d)")
    first_map = store.maps_from_pred[d][slot]
    acc = _scratch_mat!(s, true, size(first_map, 1), size(first_map, 2))
    copyto!(acc, first_map)
    acc_is_tmp1 = true

    while p != u
        d = p
        p, slot = _chosen_predecessor_with_slot(M, cc, u, d)
        slot == 0 && error("missing cover-edge map for ($p,$d)")
        E = store.maps_from_pred[d][slot]
        out = _scratch_mat!(s, !acc_is_tmp1, size(acc, 1), size(E, 2))
        _mul_dense_maybe_tiny!(out, acc, E)
        acc = out
        acc_is_tmp1 = !acc_is_tmp1
    end
    return copy(acc)
end

@inline function _compose_chain_generic_backward(M::PModule,
                                                 u::Int, v::Int,
                                                 cc::CoverCache)
    store = M.edge_maps
    d = v
    p, slot = _chosen_predecessor_with_slot(M, cc, u, d)
    slot == 0 && error("missing cover-edge map for ($p,$d)")
    A = store.maps_from_pred[d][slot]

    while p != u
        d = p
        p, slot = _chosen_predecessor_with_slot(M, cc, u, d)
        slot == 0 && error("missing cover-edge map for ($p,$d)")
        E = store.maps_from_pred[d][slot]
        A = FieldLinAlg._matmul(A, E)
    end
    return A
end

@inline _as_mattype(::Type{MatT}, A) where {MatT} = A isa MatT ? A : convert(MatT, A)

@inline function _map_leq_short_chain_no_memo(M::PModule{K,F,MatT}, u::Int, v::Int, cc::CoverCache) where {K,F,MatT}
    # Try a no-memo fast path when the chosen chain has length <= 2 edges.
    # We intentionally avoid touching the cover-cache predecessor memo here,
    # since this path targets cold one-off queries.
    lo_v = cc.pred_ptr[v]
    hi_v = cc.pred_ptr[v + 1] - 1
    p = 0
    p_slot = 0
    @inbounds for idx in lo_v:hi_v
        b = cc.pred_idx[idx]
        if b != u && leq(M.Q, u, b)
            p = b
            p_slot = idx - lo_v + 1
            break
        end
    end
    p == 0 && return M.edge_maps[u, v]

    lo_p = cc.pred_ptr[p]
    hi_p = cc.pred_ptr[p + 1] - 1
    up_slot = 0
    @inbounds for idx in lo_p:hi_p
        b = cc.pred_idx[idx]
        if b == u
            up_slot = idx - lo_p + 1
        elseif leq(M.Q, u, b)
            return nothing
        end
    end
    up_slot == 0 && return nothing

    E2 = M.edge_maps.maps_from_pred[v][p_slot]
    E1 = M.edge_maps.maps_from_pred[p][up_slot]
    if MatT <: Matrix{K}
        A = Matrix{K}(undef, size(E2, 1), size(E1, 2))
        _mul_dense_maybe_tiny!(A, E2, E1)
    else
        A = FieldLinAlg._matmul(E2, E1)
    end
    return _as_mattype(MatT, A)
end

@inline function _map_leq_scalar_chain_no_memo(M::PModule{K,F,MatT}, u::Int, v::Int, cc::CoverCache) where {K,F,MatT}
    # For 1x1 fibers, avoid matrix multiplications and compose scalars directly.
    (M.dims[u] == 1 && M.dims[v] == 1) || return nothing
    d = v
    coeff = one(K)
    while d != u
        p = _chosen_predecessor_slow(cc, u, d)
        (M.dims[p] == 1 && M.dims[d] == 1) || return nothing
        coeff *= M.edge_maps[p, d][1, 1]
        d = p
    end
    A = Matrix{K}(undef, 1, 1)
    A[1, 1] = coeff
    return _as_mattype(MatT, A)
end

@inline function _map_leq_cover_chain(M::PModule{K,F,MatT}, u::Int, v::Int, cc::CoverCache) where {K,F,MatT}
    # Compute M(u<=v) by composing cover-edge maps along a cached cover chain.
    # Assumes u < v and u <= v.
    @inbounds if cc.C !== nothing && cc.C[u, v]
        return M.edge_maps[u, v]
    end
    if cc.C === nothing && haskey(M.edge_maps, u, v)
        return M.edge_maps[u, v]
    end
    lastA = _map_leq_last_pair_get(M, u, v)
    lastA === nothing || return lastA
    memoA = _map_leq_memo_get(M, u, v)
    memoA === nothing || return memoA

    # Fast path: skip memo/scratch machinery for tiny queries.
    # - scalar-chain composition for 1x1 fibers
    # - short chosen chains (<=2 edges)
    Afast = _map_leq_scalar_chain_no_memo(M, u, v, cc)
    Afast === nothing || return _map_leq_last_pair_set!(M, u, v, Afast)
    Afast = _map_leq_short_chain_no_memo(M, u, v, cc)
    Afast === nothing || return _map_leq_last_pair_set!(M, u, v, Afast)

    seen_before = _map_leq_pred_slot_get(M, u, v) != 0
    s = _map_leq_scratch(M)
    A = MatT <: Matrix{K} ?
        _compose_chain_dense_backward!(M, u, v, cc, s) :
        _compose_chain_generic_backward(M, u, v, cc)
    if seen_before
        Atyped = _map_leq_memo_set!(M, u, v, A)
        _map_leq_last_pair_set!(M, u, v, Atyped)
        return Atyped
    end
    return _map_leq_last_pair_set!(M, u, v, A)
end

"""
    map_leq(M::PModule, u, v; cache=nothing) -> Matrix

Return the structure map `M(u <= v)` for a comparable pair `u <= v`.

The internal `PModule` stores only the maps on *cover* edges of the Hasse
diagram.  This function composes those cover maps along a (poset-dependent)
chosen cover chain.

For a functorial module, the resulting map is independent of the chosen chain;
the chain is only used as a witness that `u <= v`.

Performance notes:
  * If `cache` is omitted, the per-poset lazy `CoverCache` for `M.Q` is used.
  * The chosen cover chain is cached inside the `CoverCache` via parent pointers,
    so repeated calls avoid rescanning predecessor lists.

Warning:
  The returned matrix may alias internal storage or memoized composed maps.
  Treat it as read-only.

Best practices
- use `map_leq(M, u, v)` as the canonical low-level query for a comparable-pair
  structure map,
- use `structure_map(M; source=u, target=v)` in examples and notebooks when the
  named call style is clearer,
- for repeated workloads, prefer `prepare_map_leq_batch` together with
  `map_leq_many!`.
"""
function map_leq(M::PModule{K}, u::Int, v::Int;
                 cache::Union{Nothing,CoverCache}=nothing,
                 opts::ModuleOptions=ModuleOptions()) where {K}
    Q = M.Q
    n = nvertices(Q)
    (1 <= u <= n && 1 <= v <= n) || error("map_leq: indices out of range")

    u == v && return _identity_map(M, v)
    leq(Q, u, v) || error("map_leq: need u <= v in the poset (got u=$u, v=$v)")

    if opts != ModuleOptions()
        cache === nothing || error("map_leq: pass either cache or opts, not both.")
        cache = opts.cache
    end
    cc = cache === nothing ? _get_cover_cache(Q) : cache
    return _map_leq_cover_chain(M, u, v, cc)
end

function _map_leq_many_raw_route_kind(M::PModule,
                                      pairs::AbstractVector{<:Tuple{Int,Int}},
                                      cc::CoverCache)
    plan = _lookup_map_leq_many_plan(M, pairs)
    plan === nothing || return :plan_cached
    Q = M.Q
    arena = _map_leq_many_plan_arena(M)
    stats = _fill_map_leq_many_plan_arena!(arena, M, pairs, Q, cc)
    if _use_map_leq_many_oneoff_long(stats, M.field)
        return :oneoff_long
    elseif _prefer_map_leq_many_scalar_long(stats, M.field)
        return :scalar_fallback
    elseif _prefer_map_leq_many_plan_build(stats, M.field)
        return :plan_build
    end
    return :scalar_fallback
end

@inline function _map_leq_many_fallback!(dest::AbstractVector,
                                         M::PModule{K},
                                         pairs::AbstractVector{<:Tuple{Int,Int}},
                                         Q::AbstractPoset,
                                         n::Int,
                                         cc::CoverCache) where {K}
    @inbounds for i in eachindex(pairs)
        u, v = pairs[i]
        (1 <= u <= n && 1 <= v <= n) || error("map_leq_many!: indices out of range at i=$i (u=$u, v=$v)")
        if u == v
            dest[i] = _identity_map(M, v)
        else
            leq(Q, u, v) || error("map_leq_many!: need u <= v at i=$i (u=$u, v=$v)")
            dest[i] = _map_leq_cover_chain(M, u, v, cc)
        end
    end
    return dest
end

"""
    prepare_map_leq_batch(pairs) -> MapLeqQueryBatch

Prepare a reusable query batch for repeated `map_leq_many` calls.

The input pairs are copied on construction. Use this for hot repeated workloads
to avoid per-call mutable-vector signature checks.

Best practices
- prefer this when the same comparable pairs are queried many times,
- pass the resulting batch to `map_leq_many!`,
- use the raw vector form only for one-off or tiny workloads.
"""
function prepare_map_leq_batch(pairs::AbstractVector{<:Tuple{Int,Int}})
    copied = Vector{Tuple{Int,Int}}(undef, length(pairs))
    @inbounds for i in eachindex(pairs)
        copied[i] = pairs[i]
    end
    return _prepare_map_leq_batch_owned(copied)
end

@inline function _prepare_map_leq_batch_owned(pairs::Vector{Tuple{Int,Int}})
    return MapLeqQueryBatch(_next_map_leq_batch_id(), pairs)
end

"""
    map_leq_many!(dest, M, pairs; cache=nothing, opts=ModuleOptions())

Batch form of `map_leq` for comparable pairs `(u,v)`.
Writes one map per pair into `dest` and returns `dest`.

For large one-off batches, this may use an internal batch-local long-chain
executor that avoids per-query scalar `map_leq` calls and avoids cached-plan
copies. For repeated hot batches, prefer `prepare_map_leq_batch`.

Best practices
- use this when you need many comparable-pair maps at once,
- pass a prepared batch when the same query pattern repeats,
- size `dest` to match the number of requested pairs exactly,
- treat returned matrices as read-only because they may alias cached storage.
"""
function map_leq_many!(dest::AbstractVector,
                       M::PModule{K,F,MatT},
                       pairs::AbstractVector{<:Tuple{Int,Int}};
                       cache::Union{Nothing,CoverCache}=nothing,
                       opts::ModuleOptions=ModuleOptions()) where {K,F,MatT}
    if opts != ModuleOptions()
        cache === nothing || error("map_leq_many!: pass either cache or opts, not both.")
        cache = opts.cache
    end
    length(dest) == length(pairs) ||
        error("map_leq_many!: destination length $(length(dest)) does not match pair count $(length(pairs)).")

    Q = M.Q
    n = nvertices(Q)
    cc = cache === nothing ? _get_cover_cache(Q) : cache
    plan = _lookup_map_leq_many_plan(M, pairs)
    if plan !== nothing && _map_leq_many_with_plan!(dest, M, pairs, plan)
        return dest
    end
    arena = _map_leq_many_plan_arena(M)
    stats = _fill_map_leq_many_plan_arena!(arena, M, pairs, Q, cc)
    if _map_leq_many_oneoff_long_batch!(dest, M, arena, stats)
        return dest
    end
    if _map_leq_many_scalar_long_batch!(dest, M, arena, stats)
        return dest
    end
    plan = _prefer_map_leq_many_plan_build(stats, M.field) ?
        _cache_map_leq_many_plan_from_arena!(M, pairs, stats, arena) : nothing
    if plan !== nothing && _map_leq_many_with_plan!(dest, M, pairs, plan)
        return dest
    end
    return _map_leq_many_fallback!(dest, M, pairs, Q, n, cc)
end

"""
    map_leq_many!(dest, M, batch::MapLeqQueryBatch; cache=nothing, opts=ModuleOptions())

Batch form of `map_leq` using a prepared query batch from
`prepare_map_leq_batch`.

This is the canonical hot-path entrypoint for repeated map queries on a stable
set of comparable pairs.
"""
function map_leq_many!(dest::AbstractVector,
                       M::PModule{K,F,MatT},
                       batch::MapLeqQueryBatch;
                       cache::Union{Nothing,CoverCache}=nothing,
                       opts::ModuleOptions=ModuleOptions()) where {K,F,MatT}
    if opts != ModuleOptions()
        cache === nothing || error("map_leq_many!: pass either cache or opts, not both.")
        cache = opts.cache
    end
    pairs = batch.pairs
    length(dest) == length(pairs) ||
        error("map_leq_many!: destination length $(length(dest)) does not match pair count $(length(pairs)).")

    Q = M.Q
    n = nvertices(Q)
    cc = cache === nothing ? _get_cover_cache(Q) : cache
    plan = _get_or_build_map_leq_many_plan!(M, batch, Q, cc)
    if plan !== nothing && _map_leq_many_with_plan_unchecked!(dest, M, pairs, plan)
        return dest
    end
    return _map_leq_many_fallback!(dest, M, pairs, Q, n, cc)
end

@inline function _append_scaled_identity_triplets!(I::Vector{Int},
                                                   J::Vector{Int},
                                                   V::Vector{K},
                                                   dim::Int,
                                                   row_off::Int,
                                                   col_off::Int;
                                                   scale::K) where {K}
    iszero(scale) && return nothing
    @inbounds for t in 1:dim
        push!(I, row_off + t)
        push!(J, col_off + t)
        push!(V, scale)
    end
    return nothing
end

@inline function _append_map_leq_triplets_from_plan_at!(I::Vector{Int},
                                                        J::Vector{Int},
                                                        V::Vector{K},
                                                        M::PModule{K,F,MatT},
                                                        plan::_MapLeqManyPlan,
                                                        suffix_vals,
                                                        idx::Int,
                                                        row_off::Int,
                                                        col_off::Int,
                                                        scale::K) where {K,F,MatT}
    iszero(scale) && return nothing

    kind = plan.kinds[idx]
    u = plan.us[idx]
    v = plan.vs[idx]

    if kind == 0x00
        return _append_scaled_identity_triplets!(I, J, V, M.dims[v], row_off, col_off; scale=scale)
    elseif kind == 0x01
        return _append_scaled_triplets!(I, J, V, M.edge_maps[u, v], row_off, col_off; scale=scale)
    end

    memoA = _map_leq_memo_get(M, u, v)
    memoA === nothing || return _append_scaled_triplets!(I, J, V, memoA, row_off, col_off; scale=scale)

    if kind == 0x03
        store_results = plan.has_repeats || plan.warmed
        sidx = plan.chain_ptr[idx]
        eidx = plan.chain_ptr[idx + 1] - 1
        if MatT <: Matrix{K}
            s = _map_leq_scratch(M)
            out = _scratch_mat!(s, true, M.dims[v], M.dims[u])
            if K === QQ && _use_qq_nemo_long_product(M, plan.chain_data, plan.chain_slots, sidx, eidx)
                _compose_chain_dense_packed_slots_nemo_into!(out, M, plan.chain_data, plan.chain_slots, sidx, eidx, s)
            else
                _compose_chain_dense_packed_slots_into!(out, M, plan.chain_data, plan.chain_slots, sidx, eidx, s)
            end
            if store_results
                A = _map_leq_memo_set!(M, u, v, _as_mattype(MatT, copy(out)))
                return _append_scaled_triplets!(I, J, V, A, row_off, col_off; scale=scale)
            end
            return _append_scaled_triplets!(I, J, V, out, row_off, col_off; scale=scale)
        end

        A = _as_mattype(MatT, _compose_chain_generic_packed_slots(M, plan.chain_data, plan.chain_slots, sidx, eidx))
        if store_results
            A = _map_leq_memo_set!(M, u, v, A)
        end
        return _append_scaled_triplets!(I, J, V, A, row_off, col_off; scale=scale)
    end

    p = plan.mids[idx]
    if M.dims[u] == 1 && M.dims[v] == 1
        coeff = M.edge_maps[p, v][1, 1] * M.edge_maps[u, p][1, 1]
        iszero(coeff) || begin
            push!(I, row_off + 1)
            push!(J, col_off + 1)
            push!(V, scale * coeff)
        end
        return nothing
    end

    store_results = plan.has_repeats || plan.warmed
    if MatT <: Matrix{K}
        s = _map_leq_scratch(M)
        E2 = M.edge_maps[p, v]
        E1 = M.edge_maps[u, p]
        out = _scratch_mat!(s, true, size(E2, 1), size(E1, 2))
        _mul_dense_maybe_tiny!(out, E2, E1)
        if store_results
            A = _map_leq_memo_set!(M, u, v, _as_mattype(MatT, copy(out)))
            return _append_scaled_triplets!(I, J, V, A, row_off, col_off; scale=scale)
        end
        return _append_scaled_triplets!(I, J, V, out, row_off, col_off; scale=scale)
    end

    A = _as_mattype(MatT, FieldLinAlg._matmul(M.edge_maps[p, v], M.edge_maps[u, p]))
    if store_results
        A = _map_leq_memo_set!(M, u, v, A)
    end
    return _append_scaled_triplets!(I, J, V, A, row_off, col_off; scale=scale)
end

function _append_map_leq_many_scaled_triplets!(I::Vector{Int},
                                               J::Vector{Int},
                                               V::Vector{K},
                                               M::PModule{K,F,MatT},
                                               batch::MapLeqQueryBatch,
                                               row_ids::AbstractVector{<:Integer},
                                               col_ids::AbstractVector{<:Integer},
                                               row_offsets::AbstractVector{<:Integer},
                                               col_offsets::AbstractVector{<:Integer},
                                               scales::AbstractVector{K};
                                               cache::Union{Nothing,CoverCache}=nothing) where {K,F,MatT}
    pairs = batch.pairs
    npairs = length(pairs)
    length(row_ids) == npairs || error("_append_map_leq_many_scaled_triplets!: row_ids length mismatch")
    length(col_ids) == npairs || error("_append_map_leq_many_scaled_triplets!: col_ids length mismatch")
    length(scales) == npairs || error("_append_map_leq_many_scaled_triplets!: scales length mismatch")

    Q = M.Q
    cc = cache === nothing ? _get_cover_cache(Q) : cache
    plan = _get_or_build_map_leq_many_plan!(M, batch, Q, cc)
    if plan === nothing && npairs >= MAP_LEQ_MANY_TRIPLET_PLAN_MIN_LEN[]
        arena = _map_leq_many_plan_arena(M)
        stats = _fill_map_leq_many_plan_arena!(arena, M, pairs, Q, cc)
        plan = _copy_map_leq_many_plan(arena, stats)
    end
    if plan === nothing
        @inbounds for idx in 1:npairs
            scale = scales[idx]
            iszero(scale) && continue
            row_off = Int(row_offsets[Int(row_ids[idx])])
            col_off = Int(col_offsets[Int(col_ids[idx])])
            A = map_leq(M, pairs[idx][1], pairs[idx][2]; cache=cc)
            _append_scaled_triplets!(I, J, V, A, row_off, col_off; scale=scale)
        end
        return nothing
    end

    @inbounds for idx in 1:npairs
        row_off = Int(row_offsets[Int(row_ids[idx])])
        col_off = Int(col_offsets[Int(col_ids[idx])])
        _append_map_leq_triplets_from_plan_at!(I, J, V, M, plan, nothing,
                                               idx, row_off, col_off, scales[idx])
    end
    return nothing
end

@inline function _accum_scaled_vec!(dst::AbstractVector{K},
                                    src::AbstractVector{K},
                                    scale::K) where {K}
    iszero(scale) && return nothing
    @inbounds for t in eachindex(src)
        dst[t] += scale * src[t]
    end
    return nothing
end

@inline function _accum_scaled_matvec!(dst::AbstractVector{K},
                                       A::AbstractMatrix{K},
                                       x::AbstractVector{K},
                                       scale::K,
                                       tmpbuf::Vector{K}) where {K}
    iszero(scale) && return nothing
    nrows = size(A, 1)
    resize!(tmpbuf, nrows)
    tmp = view(tmpbuf, 1:nrows)
    mul!(tmp, A, x)
    @inbounds for t in 1:nrows
        dst[t] += scale * tmp[t]
    end
    return nothing
end

@inline function _accum_map_leq_matvec_from_plan_at!(dst::AbstractVector{K},
                                                     M::PModule{K,F,MatT},
                                                     plan::_MapLeqManyPlan,
                                                     idx::Int,
                                                     x::AbstractVector{K},
                                                     scale::K,
                                                     tmp1::Vector{K},
                                                     tmp2::Vector{K}) where {K,F,MatT}
    iszero(scale) && return nothing

    kind = plan.kinds[idx]
    u = plan.us[idx]
    v = plan.vs[idx]

    if kind == 0x00
        return _accum_scaled_vec!(dst, x, scale)
    elseif kind == 0x01
        return _accum_scaled_matvec!(dst, M.edge_maps[u, v], x, scale, tmp1)
    end

    memoA = _map_leq_memo_get(M, u, v)
    memoA === nothing || return _accum_scaled_matvec!(dst, memoA, x, scale, tmp1)

    if kind == 0x03
        store = M.edge_maps
        sidx = plan.chain_ptr[idx]
        eidx = plan.chain_ptr[idx + 1] - 1

        E = store.maps_from_pred[plan.chain_data[sidx + 1]][plan.chain_slots[sidx + 1]]
        resize!(tmp1, size(E, 1))
        cur = view(tmp1, 1:size(E, 1))
        mul!(cur, E, x)
        if sidx + 1 == eidx
            return _accum_scaled_vec!(dst, cur, scale)
        end

        cur_is_tmp1 = true
        @inbounds for k in (sidx + 2):eidx
            E = store.maps_from_pred[plan.chain_data[k]][plan.chain_slots[k]]
            if cur_is_tmp1
                resize!(tmp2, size(E, 1))
                nxt = view(tmp2, 1:size(E, 1))
            else
                resize!(tmp1, size(E, 1))
                nxt = view(tmp1, 1:size(E, 1))
            end
            mul!(nxt, E, cur)
            cur = nxt
            cur_is_tmp1 = !cur_is_tmp1
        end
        return _accum_scaled_vec!(dst, cur, scale)
    end

    p = plan.mids[idx]
    if M.dims[u] == 1 && M.dims[v] == 1
        coeff = M.edge_maps[p, v][1, 1] * M.edge_maps[u, p][1, 1]
        iszero(coeff) || (dst[1] += scale * coeff * x[1])
        return nothing
    end

    E1 = M.edge_maps[u, p]
    resize!(tmp1, size(E1, 1))
    mid = view(tmp1, 1:size(E1, 1))
    mul!(mid, E1, x)
    E2 = M.edge_maps[p, v]
    resize!(tmp2, size(E2, 1))
    out = view(tmp2, 1:size(E2, 1))
    mul!(out, E2, mid)
    return _accum_scaled_vec!(dst, out, scale)
end

function _accum_map_leq_many_scaled_matvecs!(out::AbstractVector{K},
                                             M::PModule{K,F,MatT},
                                             batch::MapLeqQueryBatch,
                                             dest_ids::AbstractVector{<:Integer},
                                             src_ids::AbstractVector{<:Integer},
                                             dest_offsets::AbstractVector{<:Integer},
                                             src_parts::AbstractVector,
                                             scales::AbstractVector{K};
                                             cache::Union{Nothing,CoverCache}=nothing) where {K,F,MatT}
    pairs = batch.pairs
    npairs = length(pairs)
    length(dest_ids) == npairs || error("_accum_map_leq_many_scaled_matvecs!: dest_ids length mismatch")
    length(src_ids) == npairs || error("_accum_map_leq_many_scaled_matvecs!: src_ids length mismatch")
    length(scales) == npairs || error("_accum_map_leq_many_scaled_matvecs!: scales length mismatch")

    Q = M.Q
    cc = cache === nothing ? _get_cover_cache(Q) : cache
    plan = _get_or_build_map_leq_many_plan!(M, batch, Q, cc)
    if plan === nothing && npairs >= MAP_LEQ_MANY_TRIPLET_PLAN_MIN_LEN[]
        arena = _map_leq_many_plan_arena(M)
        stats = _fill_map_leq_many_plan_arena!(arena, M, pairs, Q, cc)
        plan = _copy_map_leq_many_plan(arena, stats)
    end

    tmp1 = Vector{K}()
    tmp2 = Vector{K}()
    @inbounds for idx in 1:npairs
        scale = scales[idx]
        iszero(scale) && continue
        did = Int(dest_ids[idx])
        sid = Int(src_ids[idx])
        dst = view(out, Int(dest_offsets[did]) + 1:Int(dest_offsets[did + 1]))
        x = src_parts[sid]
        if plan === nothing
            A = map_leq(M, pairs[idx][1], pairs[idx][2]; cache=cc)
            _accum_scaled_matvec!(dst, A, x, scale, tmp1)
        else
            _accum_map_leq_matvec_from_plan_at!(dst, M, plan, idx, x, scale, tmp1, tmp2)
        end
    end
    return out
end

function _accum_map_leq_many_scaled_sourcevec!(out::AbstractVector{K},
                                               M::PModule{K,F,MatT},
                                               batch::MapLeqQueryBatch,
                                               dest_ids::AbstractVector{<:Integer},
                                               src_ids::AbstractVector{<:Integer},
                                               dest_offsets::AbstractVector{<:Integer},
                                               src_offsets::AbstractVector{<:Integer},
                                               src::AbstractVector{K},
                                               scales::AbstractVector{K};
                                               cache::Union{Nothing,CoverCache}=nothing) where {K,F,MatT}
    pairs = batch.pairs
    npairs = length(pairs)
    length(dest_ids) == npairs || error("_accum_map_leq_many_scaled_sourcevec!: dest_ids length mismatch")
    length(src_ids) == npairs || error("_accum_map_leq_many_scaled_sourcevec!: src_ids length mismatch")
    length(scales) == npairs || error("_accum_map_leq_many_scaled_sourcevec!: scales length mismatch")

    Q = M.Q
    cc = cache === nothing ? _get_cover_cache(Q) : cache
    plan = _get_or_build_map_leq_many_plan!(M, batch, Q, cc)
    if plan === nothing && npairs >= MAP_LEQ_MANY_TRIPLET_PLAN_MIN_LEN[]
        arena = _map_leq_many_plan_arena(M)
        stats = _fill_map_leq_many_plan_arena!(arena, M, pairs, Q, cc)
        plan = _copy_map_leq_many_plan(arena, stats)
    end

    tmp1 = Vector{K}()
    tmp2 = Vector{K}()
    @inbounds for idx in 1:npairs
        scale = scales[idx]
        iszero(scale) && continue
        did = Int(dest_ids[idx])
        sid = Int(src_ids[idx])
        dst = view(out, Int(dest_offsets[did]) + 1:Int(dest_offsets[did + 1]))
        x = view(src, Int(src_offsets[sid]) + 1:Int(src_offsets[sid + 1]))
        if plan === nothing
            A = map_leq(M, pairs[idx][1], pairs[idx][2]; cache=cc)
            _accum_scaled_matvec!(dst, A, x, scale, tmp1)
        else
            _accum_map_leq_matvec_from_plan_at!(dst, M, plan, idx, x, scale, tmp1, tmp2)
        end
    end
    return out
end

"""
    map_leq_many(M, pairs; cache=nothing, opts=ModuleOptions()) -> Vector

Allocate and return a vector of maps `M(u<=v)` for each pair `(u,v)`.
"""
function map_leq_many(M::PModule{K,F,MatT},
                      pairs::AbstractVector{<:Tuple{Int,Int}};
                      cache::Union{Nothing,CoverCache}=nothing,
                      opts::ModuleOptions=ModuleOptions()) where {K,F,MatT}
    out = Vector{MatT}(undef, length(pairs))
    return map_leq_many!(out, M, pairs; cache=cache, opts=opts)
end

"""
    map_leq_many(M, batch::MapLeqQueryBatch; cache=nothing, opts=ModuleOptions()) -> Vector

Allocate and return a vector of maps using a prepared query batch.
"""
function map_leq_many(M::PModule{K,F,MatT},
                      batch::MapLeqQueryBatch;
                      cache::Union{Nothing,CoverCache}=nothing,
                      opts::ModuleOptions=ModuleOptions()) where {K,F,MatT}
    out = Vector{MatT}(undef, length(batch.pairs))
    return map_leq_many!(out, M, batch; cache=cache, opts=opts)
end

end
