module Modules

using SparseArrays, LinearAlgebra
using ..CoreModules: QQ, ModuleOptions
import ..FiniteFringe
import ..FiniteFringe: AbstractPoset, FinitePoset, cover_edges, leq, nvertices
import Base.Threads


export PModule, PMorphism,
       ModuleOptions,
       zero_pmodule, zero_morphism, direct_sum, direct_sum_with_maps, map_leq

"""
    CoverCache(Q)

An internal cache for poset cover data and a hot-path memo used by `map_leq`.

Thread-safety:

- `succs`, `preds`, and `C` are read-only once constructed.
- `chain_parent` is a *vector of dicts*, one dict per Julia thread, so the hot-path
  memo writes are thread-local and do not require locks.
"""
# --- src/IndicatorResolutions.jl ---

# In the existing struct CoverCache, add the nedges field at the end.
struct CoverCache
    Q::AbstractPoset
    C::Union{BitMatrix,Nothing}
    succs::Vector{Vector{Int}}
    preds::Vector{Vector{Int}}

    # For each thread, chain_parent[tid][pairkey(a,d)] = chosen predecessor b in preds[d]
    # with a <= b (used to speed up witness chains in kernel/image constructions).
    chain_parent::Vector{Dict{UInt64, Int}}

    # Number of cover edges in Q (used to size edge-indexed stores).
    nedges::Int
end


const _COVER_CACHE_MEMO = IdDict{AbstractPoset, CoverCache}()

# Global memo table is shared across threads: must be protected.
const _COVER_CACHE_LOCK = Base.ReentrantLock()

function cover_cache(Q::AbstractPoset)
    # Access to IdDict must be locked if there may be concurrent writers.
    Base.lock(_COVER_CACHE_LOCK)
    cc = get(_COVER_CACHE_MEMO, Q, nothing)
    Base.unlock(_COVER_CACHE_LOCK)

    if cc === nothing
        # Build outside the lock to avoid blocking other threads on heavy work.
        newcc = _cover_cache(Q)

        Base.lock(_COVER_CACHE_LOCK)
        cc = get(_COVER_CACHE_MEMO, Q, nothing)
        if cc === nothing
            _COVER_CACHE_MEMO[Q] = newcc
            cc = newcc
        end
        Base.unlock(_COVER_CACHE_LOCK)
    end

    return cc
end

"""
    clear_cover_cache!()

Clear the global `cover_cache` memo table.

This is mostly useful in tests or benchmarks; it is safe to call in a threaded
session.
"""
function clear_cover_cache!()
    Base.lock(_COVER_CACHE_LOCK)
    empty!(_COVER_CACHE_MEMO)
    Base.unlock(_COVER_CACHE_LOCK)
    empty!(FiniteFringe._COVER_EDGES_OBJ_CACHE)
    return nothing
end

# Packs two Int32-ish values into a UInt64 for faster Dict keys.
# This is used both here and in the `CoverCache.chain_parent` hot-path memo.
@inline function _pairkey(u::Int, v::Int)::UInt64
    return (UInt64(u) << 32) | UInt64(v)
end

function _cover_cache(Q::AbstractPoset)
    # Cache enough to quickly traverse the cover graph and to build edge-indexed stores.

    # Cover edges in Q (stores both Ce.edges and Ce.mat).
    Ce = cover_edges(Q)

    # BitMatrix adjacency for O(1) cover checks (only for FinitePoset).
    C = Q isa FinitePoset ? BitMatrix(Ce) : nothing

    # Total number of cover edges (O(1) since CoverEdges stores Ce.edges).
    nedges = length(Ce)

    n = nvertices(Q)
    outdeg = zeros(Int, n)
    indeg = zeros(Int, n)

    # Count degrees.
    for (a, b) in Ce
        outdeg[a] += 1
        indeg[b] += 1
    end

    succs = [Vector{Int}(undef, outdeg[u]) for u in 1:n]
    preds = [Vector{Int}(undef, indeg[u]) for u in 1:n]

    outk = ones(Int, n)
    ink = ones(Int, n)

    for (a, b) in Ce
        succs[a][outk[a]] = b
        preds[b][ink[b]] = a
        outk[a] += 1
        ink[b] += 1
    end

    # Ensure sorted adjacency lists for binary-search helpers.
    @inbounds for u in 1:n
        sort!(succs[u])
        sort!(preds[u])
    end

    # One dict per thread to make the hot-path memo thread-safe without locks.
    chain_parent = [Dict{UInt64, Int}() for _ in 1:Threads.nthreads()]

    return CoverCache(Q, C, succs, preds, chain_parent, nedges)
end


# Thread-local accessor for the hot-path memo dict.
@inline function _chain_parent_dict(cc::CoverCache)::Dict{UInt64, Int}
    return cc.chain_parent[Threads.threadid()]
end

# choose b in preds[d] with a <= b and b != a, using memo
function _chosen_predecessor(cc::CoverCache, a::Int, d::Int)
    k = _pairkey(a, d)

    chain_parent = _chain_parent_dict(cc)
    b = get(chain_parent, k, 0)

    if b == 0
        b = findfirst(x -> x != a && leq(cc.Q, a, x), cc.preds[d])
        b = (b === nothing) ? a : cc.preds[d][b]
        chain_parent[k] = b
    end

    return b
end


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
@inline function _find_sorted_index(xs::Vector{Int}, x::Int)::Int
    i = searchsortedfirst(xs, x)
    if i <= length(xs) && @inbounds xs[i] == x
        return i
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

    cc = cache === nothing ? cover_cache(Q) : cache
    preds = cc.preds
    succs = cc.succs
    n = nvertices(Q)

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
        for j in eachindex(su)
            v = su[j]
            i = _find_sorted_index(preds[v], u)
            i == 0 && error("internal error: missing cover edge ($u,$v)")
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
        cc = cache === nothing ? cover_cache(Q) : cache
        if edge_maps.preds != cc.preds || edge_maps.succs != cc.succs
            error("edge_maps store does not match the cover graph of Q")
        end

        preds = cc.preds
        n = nvertices(Q)
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

"""
A minimal module over a finite poset `Q`.

A `PModule` is a functor `Q -> Vec_K` specified by:

  * `dims[i] = dim_K M(i)`
  * maps on cover relations `u leq with dot v`

For performance, cover-edge maps are stored in a `CoverEdgeMapStore`,
aligned with the cover graph, rather than in a dictionary keyed by `(u,v)`.
"""
struct PModule{K,MatT<:AbstractMatrix{K}}
    Q::AbstractPoset
    dims::Vector{Int}
    edge_maps::CoverEdgeMapStore{K,MatT}
end

# choose a storage matrix type from a user-provided mapping
@inline function _pmodule_mat_type(::Type{K}, edge_maps) where {K}
    V = Base.valtype(edge_maps)
    if V <: AbstractMatrix{K} && isconcretetype(V)
        return V
    end
    return Matrix{K}
end

"""
    PModule{K}(Q, dims, edge_maps; check_sizes=true)

Construct a `PModule` over coefficient type `K`. `edge_maps` may be a dict or
any mapping supporting `(u,v)` keys. Missing cover maps become zero maps.
"""
function PModule{K}(Q::AbstractPoset, dims::Vector{Int}, edge_maps;
                    check_sizes::Bool=true,
                    opts::Union{ModuleOptions,Nothing}=nothing) where {K}
    if opts !== nothing
        check_sizes == true || error("PModule: pass either check_sizes or opts, not both.")
        check_sizes = opts.check_sizes
    end
    MatT = _pmodule_mat_type(K, edge_maps)
    store = CoverEdgeMapStore{K,MatT}(Q, dims, edge_maps; check_sizes=check_sizes)
    return PModule{K,MatT}(Q, dims, store)
end

# rebase existing store to this poset (important for ChangeOfPosets)
function PModule{K}(Q::AbstractPoset, dims::Vector{Int}, store::CoverEdgeMapStore{K,MatT};
                    check_sizes::Bool=true,
                    opts::Union{ModuleOptions,Nothing}=nothing) where {K,MatT<:AbstractMatrix{K}}
    if opts !== nothing
        check_sizes == true || error("PModule: pass either check_sizes or opts, not both.")
        check_sizes = opts.check_sizes
    end
    cc = _cover_cache(Q)
    if store.preds === cc.preds && store.succs === cc.succs
        return PModule{K,MatT}(Q, dims, store)
    end
    new_store = CoverEdgeMapStore{K,MatT}(Q, dims, store; cache=cc, check_sizes=check_sizes)
    return PModule{K,MatT}(Q, dims, new_store)
end

# infer coefficient type from first map
function PModule(Q::AbstractPoset, dims::Vector{Int}, edge_maps;
                 check_sizes::Bool=true,
                 opts::Union{ModuleOptions,Nothing}=nothing)
    if opts !== nothing
        check_sizes == true || error("PModule: pass either check_sizes or opts, not both.")
        check_sizes = opts.check_sizes
    end
    for (_, A) in edge_maps
        return PModule{eltype(A)}(Q, dims, edge_maps; check_sizes=check_sizes)
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
"""
function dim_at(M::PModule{K}, q::Integer) where {K}
    return M.dims[Int(q)]
end

"Vertexwise morphism of P-modules (components are M_i \to N_i)."
struct PMorphism{K}
    dom::PModule{K}
    cod::PModule{K}
    comps::Vector{Matrix{K}}   # comps[i] :: Matrix{K} of size cod.dims[i] \times dom.dims[i]
end

"Identity morphism."
id_morphism(M::PModule{K}) where {K} =
    PMorphism{K}(M, M, [Matrix{K}(I, M.dims[i], M.dims[i]) for i in 1:length(M.dims)])


    
function _predecessors(Q::AbstractPoset)
    return _cover_cache(Q).preds
end


# ----------------------------
# Zero objects + direct sums (PModules)
# ----------------------------

"""
    zero_pmodule(Q::AbstractPoset, ::Type{K}=QQ)

The zero P-module on a finite poset Q (all stalks 0 and all structure maps 0).
"""
function zero_pmodule(Q::AbstractPoset, ::Type{K}=QQ) where {K}
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (u,v) in cover_edges(Q)
        edge[(u,v)] = zeros(K, 0, 0)
    end
    return PModule{K}(Q, zeros(Int, nvertices(Q)), edge)
end

"""
    zero_morphism(M::PModule{K}, N::PModule{K}) -> PMorphism{K}

Zero morphism M -> N.
"""
function zero_morphism(M::PModule{K}, N::PModule{K}) where {K}
    Q = M.Q
    @assert N.Q === Q
    n = nvertices(Q)
    comps = Vector{Matrix{K}}(undef, n)
    for i in 1:n
        comps[i] = zeros(K, N.dims[i], M.dims[i])
    end
    return PMorphism{K}(M, N, comps)
end

"""
    direct_sum(A::PModule{K}, B::PModule{K}) -> PModule{K}

Binary direct sum A oplus B as a P-module.
"""
function direct_sum(A::PModule{K}, B::PModule{K}) where {K}
    Q = A.Q
    n = nvertices(Q)
    @assert B.Q === Q

    dims = [A.dims[i] + B.dims[i] for i in 1:n]

    # Fast path: traverse cover edges via store-aligned succ lists and grab maps
    # by index (no tuple allocation, no search).
    cc = cover_cache(Q)
    if (A.edge_maps.succs == cc.succs && A.edge_maps.preds == cc.preds &&
        B.edge_maps.succs == cc.succs && B.edge_maps.preds == cc.preds)

        preds = cc.preds
        succs = cc.succs

        maps_from_pred = [Vector{Matrix{K}}(undef, length(preds[v])) for v in 1:n]
        maps_to_succ   = [Vector{Matrix{K}}(undef, length(succs[u])) for u in 1:n]

        @inbounds for u in 1:n
            su = succs[u]
            Au = A.edge_maps.maps_to_succ[u]
            Bu = B.edge_maps.maps_to_succ[u]
            outu = maps_to_succ[u]

            aU, bU = A.dims[u], B.dims[u]

            for j in eachindex(su)
                v = su[j]
                aV, bV = A.dims[v], B.dims[v]

                Muv = zeros(K, aV + bV, aU + bU)

                if aV != 0 && aU != 0
                    copyto!(view(Muv, 1:aV, 1:aU), Au[j])
                end
                if bV != 0 && bU != 0
                    copyto!(view(Muv, aV+1:aV+bV, aU+1:aU+bU), Bu[j])
                end

                outu[j] = Muv
                ip = _find_sorted_index(preds[v], u)
                @inbounds maps_from_pred[v][ip] = Muv
            end
        end

        store = CoverEdgeMapStore{K,Matrix{K}}(preds, succs, maps_from_pred, maps_to_succ, cc.nedges)
        return PModule{K,Matrix{K}}(Q, dims, store)
    end

    # Fallback (rare): use keyed access.
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    sizehint!(edge, length(A.edge_maps))
    for (u, v) in cover_edges(Q)
        Au = A.edge_maps[u, v]
        Bu = B.edge_maps[u, v]
        aU, bU = A.dims[u], B.dims[u]
        aV, bV = A.dims[v], B.dims[v]
        Muv = zeros(K, aV + bV, aU + bU)
        if aV != 0 && aU != 0
            copyto!(view(Muv, 1:aV, 1:aU), Au)
        end
        if bV != 0 && bU != 0
            copyto!(view(Muv, aV+1:aV+bV, aU+1:aU+bU), Bu)
        end
        edge[(u, v)] = Muv
    end
    return PModule{K}(Q, dims, edge)
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

    iA = PMorphism{K}(A, S, iA_comps)
    iB = PMorphism{K}(B, S, iB_comps)
    pA = PMorphism{K}(S, A, pA_comps)
    pB = PMorphism{K}(S, B, pB_comps)
    return S, iA, iB, pA, pB
end




@inline function _map_leq_cover_chain(M::PModule{K}, u::Int, v::Int, cc::CoverCache) where {K}
    # Compute M(u<=v) by composing cover-edge maps along the chosen chain.
    # Assumes u < v and u <= v.
    if u == v
        return Matrix{K}(I, M.dims[v], M.dims[u])
    end
    @inbounds if cc.C !== nothing && cc.C[u, v]
        return M.edge_maps[u, v]
    end
    if cc.C === nothing && haskey(M.edge_maps, u, v)
        return M.edge_maps[u, v]
    end
    w = _chosen_predecessor(cc, u, v)
    return M.edge_maps[w, v] * _map_leq_cover_chain(M, u, w, cc)
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
  * If `cache` is omitted, a memoized `CoverCache` for `M.Q` is used.
  * The chosen cover chain is cached inside the `CoverCache` via parent pointers,
    so repeated calls avoid rescanning predecessor lists.

Warning:
  The returned matrix may alias internal storage when `u < v` is a *cover edge*.
  Treat it as read-only.
"""
function map_leq(M::PModule{K}, u::Int, v::Int;
                 cache::Union{Nothing,CoverCache}=nothing,
                 opts::Union{ModuleOptions,Nothing}=nothing) where {K}
    Q = M.Q
    n = nvertices(Q)
    (1 <= u <= n && 1 <= v <= n) || error("map_leq: indices out of range")

    u == v && return Matrix{K}(I, M.dims[v], M.dims[u])
    leq(Q, u, v) || error("map_leq: need u <= v in the poset (got u=$u, v=$v)")

    if opts !== nothing
        cache === nothing || error("map_leq: pass either cache or opts, not both.")
        cache = opts.cache
    end
    cc = cache === nothing ? cover_cache(Q) : cache
    return _map_leq_cover_chain(M, u, v, cc)
end

end
