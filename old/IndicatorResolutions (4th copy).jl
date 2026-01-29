module IndicatorResolutions
# =============================================================================
# Indicator (co)presentations over a finite poset, harmonized with FiniteFringe.
#
# What this file does:
#   * Convert a FringeModule to an internal PModule with explicit structure maps.
#   * Build the ONE-STEP upset presentation (Def. 6.4.1) and downset copresentation
#     (Def. 6.4.2), returning the lightweight wrappers:
#        UpsetPresentation{K}(P, U0, U1, delta)
#        DownsetCopresentation{K}(P, D0, D1, rho)
#   * Provide a first-page Hom/Ext dimension routine from these one-step data.
#
# Design notes:
#   - We keep a small internal PModule/PMorphism type here, to avoid imposing a new
#     surface type on the rest of the code base. The public surface uses the
#     IndicatorTypes wrappers exclusively, which are already consumed by HomExt.jl.
#   - All computations use QQ = Rational{BigInt} for exactness (ExactQQ.jl).
#   - The one-step routines are sufficient to feed HomExt.hom_ext_first_page and
#     also the general Tot* builder when longer resolutions are later added.
# =============================================================================

using SparseArrays, LinearAlgebra
using ..FiniteFringe
using ..IndicatorTypes: UpsetPresentation, DownsetCopresentation
using ..ExactQQ: QQ, rrefQQ, rankQQ, rankQQ_restricted, nullspaceQQ, solve_fullcolumnQQ, colspaceQQ
using ..HomExt: pi0_count

import ..FiniteFringe: FinitePoset, Upset, Downset, principal_upset, principal_downset, cover_edges


# ---- Cached cover edges and adjacency on the Hasse diagram -------------------

# --- Threading support (Base only, no extra deps) -----------------------------

import Base.Threads

"""
    CoverCache(Q)

An internal cache for poset cover data and a hot-path memo used by `map_leq`.

Thread-safety:

- `succs`, `preds`, and `C` are read-only once constructed.
- `chain_parent` is a *vector of dicts*, one dict per Julia thread, so the hot-path
  memo writes are thread-local and do not require locks.
"""
struct CoverCache
    Q::FinitePoset
    C::BitMatrix
    succs::Vector{Vector{Int}}
    preds::Vector{Vector{Int}}

    # Hot-path memo for `_chosen_predecessor` in `map_leq`.
    #
    # Key is `_pairkey(a, d)` packing (a, d) into a UInt64.
    #
    # Thread safety: one dict per thread; each thread only writes to its own dict.
    chain_parent::Vector{Dict{UInt64, Int}}
end

const _COVER_CACHE_MEMO = IdDict{FinitePoset, CoverCache}()

# Global memo table is shared across threads: must be protected.
const _COVER_CACHE_LOCK = Base.ReentrantLock()

function cover_cache(Q::FinitePoset)
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
    return nothing
end

# Packs two Int32-ish values into a UInt64 for faster Dict keys.
# This is used both here and in the `CoverCache.chain_parent` hot-path memo.
@inline function _pairkey(u::Int, v::Int)::UInt64
    return (UInt64(u) << 32) | UInt64(v)
end

function _cover_cache(Q::FinitePoset)
    # FiniteFringe already caches cover edges in a compact representation.
    # Pull it once here and build predecessor/successor adjacency lists.
    Ce = cover_edges(Q)     # CoverEdges object (cached in FiniteFringe)
    C  = BitMatrix(Ce)      # view of Ce.mat (no copy)
    n  = Q.n

    # Build adjacency lists in O(|E|) time rather than scanning all n^2 pairs.
    succs = [Int[] for _ in 1:n]
    preds = [Int[] for _ in 1:n]

    # Pre-size the adjacency lists to avoid repeated reallocation.
    outdeg = zeros(Int, n)
    indeg  = zeros(Int, n)
    for (a, b) in Ce
        outdeg[a] += 1
        indeg[b]  += 1
    end
    for v in 1:n
        sizehint!(succs[v], outdeg[v])
        sizehint!(preds[v], indeg[v])
    end

    for (a, b) in Ce
        push!(succs[a], b)
        push!(preds[b], a)
    end

    # One dict per thread to make the hot-path memo thread-safe without locks.
    chain_parent = [Dict{UInt64, Int}() for _ in 1:Threads.nthreads()]

    return CoverCache(Q, C, succs, preds, chain_parent)
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
        b = findfirst(x -> x != a && cc.Q.leq[a, x], cc.preds[d])
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
  stores the map for the cover relation `u ⋖ v`.
- `edge_maps::CoverEdgeMapStore{K,MatT}`: an already-built store.

Missing cover-edge maps are filled with the appropriate zero map.

We keep this as an explicit dispatch point (Dict vs CoverEdgeMapStore) to avoid
runtime "two-world" adapter helpers.
"""
function CoverEdgeMapStore{K,MatT}(
    Q::FinitePoset,
    dims::Vector{Int},
    edge_maps::AbstractDict{Tuple{Int,Int},<:Any};
    cache::Union{Nothing,CoverCache}=nothing,
    check_sizes::Bool=true,
) where {K,MatT<:AbstractMatrix{K}}

    cc = cache === nothing ? cover_cache(Q) : cache
    preds = cc.preds
    succs = cc.succs
    n = Q.n

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
    Q::FinitePoset,
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
        n = Q.n
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
    Q::FinitePoset
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
function PModule{K}(Q::FinitePoset, dims::Vector{Int}, edge_maps; check_sizes::Bool=true) where {K}
    MatT = _pmodule_mat_type(K, edge_maps)
    store = CoverEdgeMapStore{K,MatT}(Q, dims, edge_maps; check_sizes=check_sizes)
    return PModule{K,MatT}(Q, dims, store)
end

# rebase existing store to this poset (important for ChangeOfPosets)
function PModule{K}(Q::FinitePoset, dims::Vector{Int}, store::CoverEdgeMapStore{K,MatT}; check_sizes::Bool=true) where {K,MatT<:AbstractMatrix{K}}
    cc = _cover_cache(Q)
    if store.preds === cc.preds && store.succs === cc.succs
        return PModule{K,MatT}(Q, dims, store)
    end
    new_store = CoverEdgeMapStore{K,MatT}(Q, dims, store; cache=cc, check_sizes=check_sizes)
    return PModule{K,MatT}(Q, dims, new_store)
end

# infer coefficient type from first map
function PModule(Q::FinitePoset, dims::Vector{Int}, edge_maps; check_sizes::Bool=true)
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

    
function _predecessors(Q::FinitePoset)
    return _cover_cache(Q).preds
end


# ----------------------------
# Zero objects + direct sums (PModules)
# ----------------------------

"""
    zero_pmodule(Q::FinitePoset, ::Type{K}=QQ)

The zero P-module on a finite poset Q (all stalks 0 and all structure maps 0).
"""
function zero_pmodule(Q::FinitePoset, ::Type{K}=QQ) where {K}
    edge = Dict{Tuple{Int,Int}, Matrix{K}}()
    for (u,v) in cover_edges(Q)
        edge[(u,v)] = zeros(K, 0, 0)
    end
    return PModule{K}(Q, zeros(Int, Q.n), edge)
end

"""
    zero_morphism(M::PModule{K}, N::PModule{K}) -> PMorphism{K}

Zero morphism M -> N.
"""
function zero_morphism(M::PModule{K}, N::PModule{K}) where {K}
    Q = M.Q
    @assert N.Q === Q
    comps = Vector{Matrix{K}}(undef, Q.n)
    for i in 1:Q.n
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
    n = Q.n
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
    n = Q.n
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
    @inbounds if cc.C[u, v]
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
function map_leq(M::PModule{K}, u::Int, v::Int; cache::Union{Nothing,CoverCache}=nothing) where {K}
    Q = M.Q
    (1 <= u <= Q.n && 1 <= v <= Q.n) || error("map_leq: indices out of range")

    u == v && return Matrix{K}(I, M.dims[v], M.dims[u])
    Q.leq[u, v] || error("map_leq: need u <= v in the poset (got u=$u, v=$v)")

    cc = cache === nothing ? cover_cache(Q) : cache
    return _map_leq_cover_chain(M, u, v, cc)
end



# ----------------------- from FringeModule to PModule ----------------------------

"""
    pmodule_from_fringe(H::FiniteFringe.FringeModule{K})
Return an internal `PModule{QQ}` whose fibers and structure maps are induced by the
fringe presentation `phi : oplus k[U_i] to oplus k[D_j]` (Defs. 3.16-3.17).
Implementation: M_q = im(phi_q) inside E_q; along a cover u<v the map is the restriction
E_u to E_v followed by projection to M_v.
"""
function pmodule_from_fringe(H::FiniteFringe.FringeModule{K}) where {K}
    Q = H.P
    n = Q.n

    # Basis for each fiber M_q as columns of a QQ matrix B[q] spanning im(phi_q).
    B = Vector{Matrix{QQ}}(undef, n)
    dims = zeros(Int, n)
    for q in 1:n
        cols = findall(U -> U.mask[q], H.U)
        rows = findall(D -> D.mask[q], H.D)
        if isempty(cols) || isempty(rows)
            B[q] = zeros(QQ, length(rows), 0)
            dims[q] = 0
            continue
        end
        phi_q = Matrix{QQ}(Matrix(H.phi[rows, cols]))
        B[q] = colspaceQQ(phi_q)
        dims[q] = size(B[q], 2)
    end

    # Death projection E_u \to E_v on a cover u<v: keep row indices j that remain active at v.
    function death_projection(u::Int, v::Int)
        rows_u = findall(D -> D.mask[u], H.D)
        rows_v = findall(D -> D.mask[v], H.D)
        pos_v = Dict{Int,Int}(rows_v[i] => i for i in 1:length(rows_v))
        P = zeros(QQ, length(rows_v), length(rows_u))
        for (jpos, jidx) in enumerate(rows_u)
            if haskey(pos_v, jidx)
                P[pos_v[jidx], jpos] = 1//1
            end
        end
        P
    end

    # Structure map on a cover u<v: M_u --incl--> E_u --proj--> E_v --coords--> M_v
    edge_maps = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    C = cover_edges(Q)

    @inbounds for (u, v) in C
        du = dims[u]
        dv = dims[v]
        if du == 0 || dv == 0
            edge_maps[(u, v)] = zeros(QQ, dv, du)
        else
            Puv = death_projection(u, v)      # E_u -> E_v
            Im  = Puv * B[u]                  # in E_v coordinates
            X   = solve_fullcolumnQQ(B[v], Im)
            edge_maps[(u, v)] = X             # M_u -> M_v
        end
    end

    PModule{QQ}(Q, dims, edge_maps)
end

# -------------------------- projective cover (Def. 6.4.1) --------------------------

"Incoming image at v from immediate predecessors; basis matrix with QQ columns."
function _incoming_image_basis(M::PModule{QQ}, v::Int; cache::Union{Nothing,CoverCache}=nothing)
    dv = M.dims[v]
    pv = M.edge_maps.preds[v]
    maps = M.edge_maps.maps_from_pred[v]

    if dv == 0 || isempty(pv)
        return zeros(QQ, dv, 0)
    end

    tot = 0
    @inbounds for u in pv
        tot += M.dims[u]
    end
    if tot == 0
        return zeros(QQ, dv, 0)
    end

    # dense fast path
    if eltype(maps) <: Matrix{QQ}
        A = Matrix{QQ}(undef, dv, tot)
        col = 1
        @inbounds for i in eachindex(pv)
            u = pv[i]
            du = M.dims[u]
            if du > 0
                A[:, col:col+du-1] .= maps[i]
                col += du
            end
        end
        return colspaceQQ(A)
    end

    # fallback: sparse/abstract matrix types
    return colspaceQQ(hcat(maps...))
end



"""
    projective_cover(M::PModule{QQ})
Return (F0, pi0, gens_at) where F0 is a direct sum of principal upsets covering M,
pi0 : F0 \to M is the natural surjection, and `gens_at[v]` lists the generators activated
at vertex v (each item is a pair (p, local_index_in_Mp)).
"""
function projective_cover(M::PModule{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    Q = M.Q; n = Q.n
    cc = cache === nothing ? cover_cache(Q) : cache

    # number of generators at each vertex = dim(M_v) - rank(incoming_image)
    gens_at = Vector{Vector{Tuple{Int,Int}}}(undef, n)
    gen_of_p = fill(0, n)
    for v in 1:n
        Img = _incoming_image_basis(M, v; cache=cc)
        beta = M.dims[v] - size(Img, 2)
        chosen = Int[]
        if beta > 0 && M.dims[v] > 0
            S = Img
            Id = Matrix{QQ}(I, M.dims[v], M.dims[v])
            rS = size(S, 2)
            for j in 1:M.dims[v]
                T = hcat(S, Id[:, j])
                if rankQQ(T) > rS
                    push!(chosen, j); S = colspaceQQ(T); rS += 1
                    length(chosen) == beta && break
                end
            end
        end
        gens_at[v] = [(v, j) for j in chosen]
        gen_of_p[v] = length(chosen)
    end

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
    F0_dims = [sum(gen_of_p[p] for p in 1:n if Q.leq[p,i]) for i in 1:n]

    # Active generator lists (and positions) at each vertex i.
    active_at = Vector{Vector{Tuple{Int,Int}}}(undef, n)
    pos_at    = Vector{Dict{Tuple{Int,Int},Int}}(undef, n)
    for i in 1:n
        lst = Tuple{Int,Int}[]
        for p in 1:n
            if gen_of_p[p] > 0 && Q.leq[p,i]
                append!(lst, gens_at[p])
            end
        end
        active_at[i] = lst
        d = Dict{Tuple{Int,Int},Int}()
        for (k, g) in enumerate(lst)
            d[g] = k
        end
        pos_at[i] = d
    end

    F0_edges = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    @inbounds for u in 1:n
        su = cc.succs[u]
        for v in su
            Muv = zeros(QQ, F0_dims[v], F0_dims[u])
            # Inclusion: every generator active at u is also active at v.
            for (j, g) in enumerate(active_at[u])
                i = pos_at[v][g]
                Muv[i, j] = 1//1
            end
            F0_edges[(u, v)] = Muv
        end
    end
    F0 = PModule{QQ}(Q, F0_dims, F0_edges)


    # pi0 : F0 -> M
    #
    # Preallocate each component and fill blockwise.
    # Old code used repeated hcat (allocates many temporaries).
    gen_vertices = [p for p in 1:n if gen_of_p[p] > 0]

    # Cache the chosen basis indices in each M_p once.
    J_at = Vector{Vector{Int}}(undef, n)
    for p in 1:n
        J_at[p] = Int[]
    end
    for p in gen_vertices
        J_at[p] = [pair[2] for pair in gens_at[p]]
    end

    comps = Vector{Matrix{QQ}}(undef, n)
    for i in 1:n
        Mi = M.dims[i]
        Fi = F0_dims[i]
        cols = zeros(QQ, Mi, Fi)
        col = 1
        for p in gen_vertices
            k = gen_of_p[p]
            if !Q.leq[p, i]
                continue
            end
            A = map_leq(M, p, i; cache=cc)  # M_p -> M_i
            Jp = J_at[p]
            @inbounds for t in 1:k
                j = Jp[t]
                copyto!(view(cols, :, col), view(A, :, j))
                col += 1
            end
        end
        comps[i] = cols
    end
    pi0 = PMorphism{QQ}(F0, M, comps)
    return F0, pi0, gens_at
end


# --------------------------- kernel and upset presentation --------------------------

"Kernel of f with inclusion iota : ker(f) to dom(f), degreewise."
function kernel_with_inclusion(f::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    M = f.dom
    n = M.Q.n

    basisK = Vector{Matrix{QQ}}(undef, n)
    K_dims = zeros(Int, n)
    for i in 1:n
        B = nullspaceQQ(f.comps[i])
        basisK[i] = B
        K_dims[i] = size(B, 2)
    end

    # Build the kernel's structure maps directly in store-aligned form.
    cc = (cache === nothing ? cover_cache(M.Q) : cache)
    preds = cc.preds
    succs = cc.succs

    maps_from_pred = [Vector{Matrix{QQ}}(undef, length(preds[v])) for v in 1:n]
    maps_to_succ   = [Vector{Matrix{QQ}}(undef, length(succs[u])) for u in 1:n]

    @inbounds for u in 1:n
        su = succs[u]
        maps_u_M = M.edge_maps.maps_to_succ[u]   # aligned with su
        outu = maps_to_succ[u]

        for j in eachindex(su)
            v = su[j]

            # If either stalk is 0, the induced map is the unique 0 map.
            if K_dims[u] == 0 || K_dims[v] == 0
                X = zeros(QQ, K_dims[v], K_dims[u])
                outu[j] = X
                ip = _find_sorted_index(preds[v], u)
                maps_from_pred[v][ip] = X
                continue
            end

            # Induced map K(u) -> K(v): express M(u->v)*basisK[u] in basisK[v].
            T  = maps_u_M[j]
            Im = T * basisK[u]
            X  = solve_fullcolumnQQ(basisK[v], Im)

            outu[j] = X
            ip = _find_sorted_index(preds[v], u)
            maps_from_pred[v][ip] = X
        end
    end

    storeK = CoverEdgeMapStore{QQ,Matrix{QQ}}(preds, succs, maps_from_pred, maps_to_succ, cc.nedges)
    K = PModule{QQ,Matrix{QQ}}(M.Q, K_dims, storeK)

    iota = PMorphism{QQ}(K, M, [basisK[i] for i in 1:n])
    return K, iota
end


# ----------------------------
# Image with inclusion (dual to kernel_with_inclusion)
# ----------------------------

"""
    image_with_inclusion(f::PMorphism{QQ}) -> (Im, iota)

Compute the image submodule Im subseteq cod(f) with the inclusion morphism iota: Im -> cod(f).
"""
function image_with_inclusion(f::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    N = f.cod
    Q = N.Q
    n = Q.n

    bases = Vector{Matrix{QQ}}(undef, n)
    dims  = zeros(Int, n)

    for i in 1:n
        B = column_basisQQ(f.comps[i])
        bases[i] = B
        dims[i]  = size(B, 2)
    end

    # Build the image's structure maps directly in store-aligned form.
    storeN = N.edge_maps
    preds  = storeN.preds
    succs  = storeN.succs

    maps_from_pred = [Vector{Matrix{QQ}}(undef, length(preds[v])) for v in 1:n]
    maps_to_succ   = [Vector{Matrix{QQ}}(undef, length(succs[u])) for u in 1:n]

    @inbounds for u in 1:n
        su = succs[u]
        Nu = storeN.maps_to_succ[u]
        outu = maps_to_succ[u]

        Bu = bases[u]
        du = size(Bu, 2)

        for j in eachindex(su)
            v = su[j]
            Bv = bases[v]
            dv = size(Bv, 2)

            # Induced map Im(u) -> Im(v): express N(u->v)*Bu in Bv.
            Auv = if du == 0
                zeros(QQ, dv, 0)
            elseif dv == 0
                zeros(QQ, 0, du)
            else
                T = Nu[j] * Bu
                solve_fullcolumnQQ(Bv, T)
            end

            outu[j] = Auv
            ip = _find_sorted_index(preds[v], u)
            maps_from_pred[v][ip] = Auv
        end
    end

    storeIm = CoverEdgeMapStore{QQ,Matrix{QQ}}(preds, succs, maps_from_pred, maps_to_succ, storeN.nedges)
    Im = PModule{QQ,Matrix{QQ}}(Q, dims, storeIm)

    iota = PMorphism(Im, N, [bases[i] for i in 1:n])
    return Im, iota
end




# Basis of the "socle" at vertex u: kernel of the stacked outgoing map
# M_u to oplus_{u<v} M_v along cover edges u < v.
# Columns of the returned matrix span soc(M)_u subseteq M_u.
function _socle_basis(M::PModule{QQ}, u::Int; cache::Union{Nothing,CoverCache}=nothing)
    cc = (cache === nothing ? cover_cache(M.Q) : cache)
    su = cc.succs[u]
    du = M.dims[u]

    if isempty(su) || du == 0
        return Matrix{QQ}(I, du, du)
    end

    # Build the stacked outgoing map A : M_u -> (direct sum over cover successors).
    # A has size (sum_v dim(M_v)) x dim(M_u).
    tot = 0
    @inbounds for j in eachindex(su)
        tot += M.dims[su[j]]
    end
    if tot == 0
        # No nonzero target fibers; outgoing map is zero, so socle is all of M_u.
        return Matrix{QQ}(I, du, du)
    end

    A = Matrix{QQ}(undef, tot, du)
    row = 1
    maps_u = M.edge_maps.maps_to_succ[u]

    @inbounds for j in eachindex(su)
        v = su[j]
        dv = M.dims[v]
        if dv > 0
            A[row:row+dv-1, :] .= maps_u[j]
            row += dv
        end
    end

    return nullspaceQQ(A)  # columns span the socle at u
end


# A canonical left-inverse for a full-column-rank matrix S: L*S = I.
# Implemented as L = (S^T S)^{-1} S^T using exact QQ solves.
function _left_inverse_full_column(S::AbstractMatrix{QQ})
    s = size(S,2)
    if s == 0
        return zeros(QQ, 0, size(S,1))
    end
    G = transpose(S) * S                       # s*s Gram matrix, invertible over QQ
    return solve_fullcolumnQQ(G, transpose(S)) # returns (S^T S)^{-1} S^T with size s * m
end

# Build the injective (downset) hull:  iota : M into E  where
# E is a direct sum of principal downsets with multiplicities = socle dimensions.
# Also return the generator labels as (u, j) with u the vertex and j the column.
function _injective_hull(M::PModule{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    Q = M.Q; n = Q.n
    cc = cache === nothing ? cover_cache(Q) : cache

    # socle bases at each vertex and their multiplicities
    Soc = Vector{Matrix{QQ}}(undef, n)
    mult = zeros(Int, n)
    for u in 1:n
        Soc[u]  = _socle_basis(M, u; cache=cc)
        mult[u] = size(Soc[u], 2)
    end

        # generator labels for the chosen downset summands
    gens_at = Vector{Vector{Tuple{Int,Int}}}(undef, n)
    for u in 1:n
        gens_at[u] = [(u, j) for j in 1:mult[u]]
    end

    # fiber dimensions of E
    Edims = [sum(mult[u] for u in 1:n if Q.leq[i,u]) for i in 1:n]

    # Active generator lists (and positions) at each vertex i.
    # This ordering matches the row-stacking order used in the inclusion iota below.
    active_at = Vector{Vector{Tuple{Int,Int}}}(undef, n)
    pos_at    = Vector{Dict{Tuple{Int,Int},Int}}(undef, n)
    for i in 1:n
        lst = Tuple{Int,Int}[]
        for u in 1:n
            if mult[u] > 0 && Q.leq[i,u]
                append!(lst, gens_at[u])
            end
        end
        active_at[i] = lst
        d = Dict{Tuple{Int,Int},Int}()
        for (k, g) in enumerate(lst)
            d[g] = k
        end
        pos_at[i] = d
    end

    # E structure maps along cover edges u<v are coordinate projections:
    # keep exactly those generators that are still active at v.
    Eedges = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    @inbounds for u in 1:n
        su = cc.succs[u]
        for v in su
            Muv = zeros(QQ, Edims[v], Edims[u])
            for (i, g) in enumerate(active_at[v])
                j = pos_at[u][g]
                Muv[i, j] = 1//1
            end
            Eedges[(u, v)] = Muv
        end
    end
    E = PModule{QQ}(Q, Edims, Eedges)

    # iota : M -> E
    Linv = [ _left_inverse_full_column(Soc[u]) for u in 1:n ]

    # Only vertices with nontrivial socle contribute rows.
    mult_vertices = [u for u in 1:n if mult[u] > 0]

    comps = Vector{Matrix{QQ}}(undef, n)
    for i in 1:n
        rows = zeros(QQ, Edims[i], M.dims[i])
        r = 1
        for u in mult_vertices
            if Q.leq[i, u]
                Mi_to_Mu = map_leq(M, i, u; cache=cc)
                m = mult[u]
                @views mul!(rows[r:r+m-1, :], Linv[u], Mi_to_Mu)
                r += m
            end
        end
        @assert r == Edims[i] + 1
        comps[i] = rows
    end
    iota = PMorphism{QQ}(M, E, comps)

    return E, iota, gens_at
end


# Degreewise cokernel of iota : E0 <- M, produced as a P-module C together with
# the quotient q : E0 -> C.  The quotient is represented by surjections q_i whose
# kernels are colspace(iota_i).
function _cokernel_module(iota::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    E = iota.cod; Q = E.Q; n = Q.n
    Cdims  = zeros(Int, n)
    qcomps = Vector{Matrix{QQ}}(undef, n)     # each is (dim C_i) x (dim E_i)

    # degreewise quotients
    for i in 1:n
        Bi = colspaceQQ(iota.comps[i])        # dim E_i x rank
        Ni = nullspaceQQ(transpose(Bi))       # dim E_i x (dim E_i - rank)
        Cdims[i]  = size(Ni, 2)
        qcomps[i] = transpose(Ni)
    end

    # structure maps of C
    Cedges = Dict{Tuple{Int,Int}, Matrix{QQ}}()
    cc = (cache === nothing ? cover_cache(Q) : cache)

    @inbounds for u in 1:n
        su     = cc.succs[u]
        maps_u = E.edge_maps.maps_to_succ[u]   # aligned with su

        for j in eachindex(su)
            v = su[j]

            if Cdims[u] > 0 && Cdims[v] > 0
                T = maps_u[j]  # E_u -> E_v along this cover edge

                # Induced quotient map: enforce q_v * T = A * q_u.
                X = solve_fullcolumnQQ(transpose(qcomps[u]), transpose(qcomps[v] * T))
                Cedges[(u, v)] = transpose(X)  # dim C_v x dim C_u
            else
                Cedges[(u, v)] = zeros(QQ, Cdims[v], Cdims[u])
            end
        end
    end

    Cmod = PModule{QQ}(Q, Cdims, Cedges)
    q = PMorphism{QQ}(E, Cmod, qcomps)
    return Cmod, q
end


# =============================================================================
# Abelian category API (kernels/cokernels/images/coimages/quotients)
# =============================================================================

"""
    cokernel_with_projection(f; cache=nothing) -> (C, q)

Compute the cokernel of a morphism f : A -> B as the quotient module C = B / im(f),
together with the quotient map q : B -> C.

In this functor category (finite poset modules over a field), cokernels are computed
pointwise (vertexwise) and the structure maps of the quotient are induced.

This is the dual companion of `kernel_with_inclusion`.
"""
function cokernel_with_projection(f::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    return _cokernel_module(f; cache=cache)
end

# ---------------------------------------------------------------------------
# Equalizers / coequalizers (abelian category = kernels / cokernels of f - g)
# ---------------------------------------------------------------------------

# Internal: difference of parallel maps (same dom/cod).
function _difference_morphism(f::PMorphism{QQ}, g::PMorphism{QQ})
    if f.dom !== g.dom || f.cod !== g.cod
        error("need parallel morphisms with the same domain and codomain")
    end
    comps = Matrix{QQ}[f.comps[i] - g.comps[i] for i in 1:f.dom.Q.n]
    return PMorphism{QQ}(f.dom, f.cod, comps)
end

"""
    equalizer(f, g; cache=nothing) -> (E, e)

Equalizer of parallel morphisms `f, g : A -> B`.

In an abelian category, the equalizer is `ker(f - g)` with inclusion map
`e : E -> A`.

Returns:
- `E` : the equalizer object
- `e` : the inclusion `E -> A`

See also: `coequalizer`, `kernel_with_inclusion`.
"""
function equalizer(f::PMorphism{QQ}, g::PMorphism{QQ}; cache=nothing)
    h = _difference_morphism(f, g)
    return kernel_with_inclusion(h; cache=cache)
end

"""
    coequalizer(f, g; cache=nothing) -> (Q, q)

Coequalizer of parallel morphisms `f, g : A -> B`.

In an abelian category, the coequalizer is `coker(f - g)` with projection map
`q : B -> Q`.

Returns:
- `Q` : the coequalizer object
- `q` : the projection `B -> Q`

See also: `equalizer`, `cokernel_with_projection`.
"""
function coequalizer(f::PMorphism{QQ}, g::PMorphism{QQ}; cache=nothing)
    h = _difference_morphism(f, g)
    return cokernel_with_projection(h; cache=cache)
end


"""
    cokernel(f; cache=nothing) -> C

Return only the cokernel module.
"""
cokernel(f::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing) =
    (cokernel_with_projection(f; cache=cache))[1]

"""
    kernel(f; cache=nothing) -> K

Return only the kernel module.
"""
kernel(f::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing) =
    (kernel_with_inclusion(f; cache=cache))[1]

"""
    image(f) -> Im

Return only the image module (a submodule of `codomain(f)`).
Use `image_with_inclusion` if you also want the inclusion map `Im -> codomain(f)`.
"""
image(f::PMorphism{QQ}) = (image_with_inclusion(f))[1]

"""
    coimage_with_projection(f; cache=nothing) -> (Coim, p)

Compute the coimage of f : A -> B, defined as Coim = A / ker(f),
together with the canonical projection p : A -> Coim.

In an abelian category, the canonical map Coim -> Im is an isomorphism.
"""
function coimage_with_projection(f::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    K, iK = kernel_with_inclusion(f; cache=cache)
    Coim, p = _cokernel_module(iK; cache=cache)
    return Coim, p
end

coimage(f::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing) =
    (coimage_with_projection(f; cache=cache))[1]

# Quotients of submodules: for now we represent a submodule by its inclusion morphism.
"""
    quotient_with_projection(iota; cache=nothing) -> (Q, q)

Given a monomorphism iota : N -> M (typically an inclusion), compute the quotient module
Q = M / N together with the projection q : M -> Q.

This is an alias for `cokernel_with_projection(iota)`.
"""
function quotient_with_projection(iota::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    return cokernel_with_projection(iota; cache=cache)
end

"""
    quotient(iota; cache=nothing) -> Q

Return only the quotient module M/N for an inclusion iota : N -> M.
"""
quotient(iota::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing) =
    (quotient_with_projection(iota; cache=cache))[1]

# Small predicates

"""
    is_zero_morphism(f) -> Bool

Return true if all components of f are the zero matrix.
"""
function is_zero_morphism(f::PMorphism{K}) where {K}
    for A in f.comps
        all(iszero, A) || return false
    end
    return true
end

"""
    is_monomorphism(f) -> Bool

Check whether f is a monomorphism in the functor category of P-modules.
For these representations, this is equivalent to f_i being injective for every vertex i.

This method is implemented for QQ using exact rank computations.
"""
function is_monomorphism(f::PMorphism{QQ})
    Q = f.dom.Q
    @assert f.cod.Q === Q
    for i in 1:Q.n
        if rankQQ(f.comps[i]) != f.dom.dims[i]
            return false
        end
    end
    return true
end

"""
    is_epimorphism(f) -> Bool

Check whether f is an epimorphism (pointwise surjective).
Implemented for QQ via exact ranks.
"""
function is_epimorphism(f::PMorphism{QQ})
    Q = f.dom.Q
    @assert f.cod.Q === Q
    for i in 1:Q.n
        if rankQQ(f.comps[i]) != f.cod.dims[i]
            return false
        end
    end
    return true
end


# -----------------------------------------------------------------------------
# Submodules as first-class objects
# -----------------------------------------------------------------------------

"""
    Submodule(incl)

A lightweight wrapper around an inclusion morphism `incl : N -> M` representing the submodule
N <= M. The ambient module is `ambient(S)` and the underlying module is `sub(S)`.

This wrapper is intentionally minimal: it stores only the inclusion map.
"""
struct Submodule{K}
    incl::PMorphism{K}  # incl : sub -> ambient
end

"""
    submodule(incl; check_mono=true) -> Submodule

Build a `Submodule` from an inclusion map.

If `check_mono=true`, verify that `incl` is a monomorphism (QQ only).
"""
function submodule(incl::PMorphism{QQ}; check_mono::Bool=true)
    check_mono && !is_monomorphism(incl) && error("submodule: given inclusion is not a monomorphism")
    return Submodule{QQ}(incl)
end
function submodule(incl::PMorphism{K}; check_mono::Bool=false) where {K}
    check_mono && error("submodule: check_mono is only implemented for QQ")
    return Submodule{K}(incl)
end

"Underlying submodule N (as a P-module) for a `Submodule` N <= M."
@inline sub(S::Submodule) = S.incl.dom

"Ambient module M for a `Submodule` N <= M."
@inline ambient(S::Submodule) = S.incl.cod

"Inclusion map N -> M for a `Submodule`."
@inline inclusion(S::Submodule) = S.incl

"""
    quotient_with_projection(S::Submodule; cache=nothing) -> (Q, q)
    quotient(S::Submodule; cache=nothing) -> Q

Compute the quotient M/N given a submodule S representing N <= M.
"""
quotient_with_projection(S::Submodule{QQ}; cache::Union{Nothing,CoverCache}=nothing) =
    quotient_with_projection(S.incl; cache=cache)
quotient(S::Submodule{QQ}; cache::Union{Nothing,CoverCache}=nothing) =
    quotient(S.incl; cache=cache)

"""
    quotient_with_projection(M, S::Submodule; cache=nothing) -> (Q, q)
    quotient(M, S::Submodule; cache=nothing) -> Q

Convenience methods matching the common mathematical notation `M/N`.
These verify that `ambient(S) === M`.
"""
function quotient_with_projection(M::PModule{QQ}, S::Submodule{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    ambient(S) === M || error("quotient_with_projection: submodule is not a submodule of the given ambient module")
    return quotient_with_projection(S; cache=cache)
end
quotient(M::PModule{QQ}, S::Submodule{QQ}; cache::Union{Nothing,CoverCache}=nothing) =
    (quotient_with_projection(M, S; cache=cache))[1]

"""
    quotient_with_projection(M, iota; cache=nothing) -> (Q, q)
    quotient(M, iota; cache=nothing) -> Q

Convenience overloads where the submodule is given as an inclusion morphism iota : N -> M.
"""
function quotient_with_projection(M::PModule{QQ}, iota::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    iota.cod === M || error("quotient_with_projection: inclusion morphism does not target the given ambient module")
    return quotient_with_projection(iota; cache=cache)
end
quotient(M::PModule{QQ}, iota::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing) =
    (quotient_with_projection(M, iota; cache=cache))[1]

"""
    kernel_submodule(f; cache=nothing) -> Submodule

Return ker(f) <= dom(f) as a `Submodule`. (The inclusion is provided by `kernel_with_inclusion`.)
"""
function kernel_submodule(f::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    K, iota = kernel_with_inclusion(f; cache=cache)
    return Submodule{QQ}(iota)
end

"""
    image_submodule(f) -> Submodule

Return im(f) <= cod(f) as a `Submodule`.
"""
function image_submodule(f::PMorphism{QQ})
    Im, iota = image_with_inclusion(f)
    return Submodule{QQ}(iota)
end

# -----------------------------------------------------------------------------
# Pushouts and pullbacks
# -----------------------------------------------------------------------------

# Internal: compute a right inverse for a full-row-rank matrix Q (r x m), so Q * rinv = I_r.
function _right_inverse_full_rowQQ(Q::Matrix{QQ})
    r, m = size(Q)
    if r == 0
        return zeros(QQ, m, 0)
    end
    G = Q * transpose(Q)  # r x r, invertible if Q has full row rank
    invG = solve_fullcolumnQQ(G, Matrix{QQ}(I, r, r))
    return transpose(Q) * invG  # m x r
end

"""
    pushout(f, g; cache=nothing) -> (P, inB, inC, q, phi)

Compute the pushout of a span A --f--> B and A --g--> C.

Construction:
    P = (B oplus C) / im( (f, -g) : A -> B oplus C )

Returns:
- P : the pushout module
- inB : B -> P
- inC : C -> P
- q : (B oplus C) -> P (the quotient map)
- phi : A -> (B oplus C) (the map whose cokernel defines the pushout)

The maps satisfy inB o f == inC o g.
"""
function pushout(f::PMorphism{QQ}, g::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    @assert f.dom === g.dom
    A = f.dom
    B = f.cod
    C = g.cod
    S, iB, iC, pB, pC = direct_sum_with_maps(B, C)
    Q = S.Q
    # phi = iB o f - iC o g : A -> B oplus C
    phi_comps = Vector{Matrix{QQ}}(undef, Q.n)
    for u in 1:Q.n
        phi_comps[u] = iB.comps[u] * f.comps[u] - iC.comps[u] * g.comps[u]
    end
    phi = PMorphism{QQ}(A, S, phi_comps)

    P, q = _cokernel_module(phi; cache=cache)

    # inB = q o iB, inC = q o iC
    inB_comps = Vector{Matrix{QQ}}(undef, Q.n)
    inC_comps = Vector{Matrix{QQ}}(undef, Q.n)
    for u in 1:Q.n
        inB_comps[u] = q.comps[u] * iB.comps[u]
        inC_comps[u] = q.comps[u] * iC.comps[u]
    end
    inB = PMorphism{QQ}(B, P, inB_comps)
    inC = PMorphism{QQ}(C, P, inC_comps)

    return P, inB, inC, q, phi
end

# ---------------------------------------------------------------------------
# Diagram objects + limit/colimit dispatch (small but useful categorical layer)
# ---------------------------------------------------------------------------

"""
Abstract supertype for small diagram objects valued in `PModule{K}`.

This is intentionally minimal: it supports the common finite shapes needed in
everyday categorical algebra:
- discrete pair (product/coproduct)
- parallel pair (equalizer/coequalizer)
- span (pushout)
- cospan (pullback)
"""
abstract type AbstractDiagram{K} end

"""
    DiscretePairDiagram(A, B)

The discrete diagram on two objects `A` and `B` (no arrows).
"""
struct DiscretePairDiagram{K} <: AbstractDiagram{K}
    A::PModule{K}
    B::PModule{K}
end

"""
    ParallelPairDiagram(f, g)

A parallel pair of morphisms `f, g : A -> B`.
"""
struct ParallelPairDiagram{K} <: AbstractDiagram{K}
    f::PMorphism{K}
    g::PMorphism{K}
end

"""
    SpanDiagram(f, g)

A span-shaped diagram `A --f--> B` and `A --g--> C`.

Its colimit is the pushout.
"""
struct SpanDiagram{K} <: AbstractDiagram{K}
    f::PMorphism{K}  # A -> B
    g::PMorphism{K}  # A -> C
end

"""
    CospanDiagram(f, g)

A cospan-shaped diagram `B --f--> D` and `C --g--> D`.

Its limit is the pullback.
"""
struct CospanDiagram{K} <: AbstractDiagram{K}
    f::PMorphism{K}  # B -> D
    g::PMorphism{K}  # C -> D
end

"""
    limit(D; cache=nothing)

Compute a limit of a supported small diagram object.

Supported shapes:
- `DiscretePairDiagram`: product
- `ParallelPairDiagram` (over QQ): equalizer
- `CospanDiagram` (over QQ): pullback
"""
function limit(D::DiscretePairDiagram{K}; cache=nothing) where {K}
    return product(D.A, D.B)
end

function limit(D::ParallelPairDiagram{QQ}; cache=nothing)
    return equalizer(D.f, D.g; cache=cache)
end

function limit(D::CospanDiagram{QQ}; cache=nothing)
    return pullback(D.f, D.g; cache=cache)
end

"""
    colimit(D; cache=nothing)

Compute a colimit of a supported small diagram object.

Supported shapes:
- `DiscretePairDiagram`: coproduct
- `ParallelPairDiagram` (over QQ): coequalizer
- `SpanDiagram` (over QQ): pushout
"""
function colimit(D::DiscretePairDiagram{K}; cache=nothing) where {K}
    return coproduct(D.A, D.B)
end

function colimit(D::ParallelPairDiagram{QQ}; cache=nothing)
    return coequalizer(D.f, D.g; cache=cache)
end

function colimit(D::SpanDiagram{QQ}; cache=nothing)
    return pushout(D.f, D.g; cache=cache)
end


# Internal: solve A*X = B over QQ, returning one particular solution with free variables set to 0.
# This duplicates ChainComplexes.solve_particularQQ but avoids introducing a dependency.
function _solve_particularQQ(A::AbstractMatrix{QQ}, B::AbstractMatrix{QQ})
    A0 = Matrix{QQ}(A)
    B0 = Matrix{QQ}(B)
    m, n = size(A0)
    @assert size(B0, 1) == m

    Aug = hcat(A0, B0)
    R, pivs_all = rrefQQ(Aug)
    rhs = size(B0, 2)

    # consistency check: zero row in A-part with nonzero in RHS-part
    for i in 1:m
        if all(R[i, 1:n] .== 0)
            if any(R[i, n+1:n+rhs] .!= 0)
                error("_solve_particularQQ: inconsistent system")
            end
        end
    end

    pivs = Int[]
    for p in pivs_all
        p <= n && push!(pivs, p)
    end

    X = zeros(QQ, n, rhs)
    for (row, pcol) in enumerate(pivs)
        X[pcol, :] = R[row, n+1:n+rhs]
    end
    return X
end

"""
    pullback(f, g; cache=nothing) -> (P, prB, prC, iota, psi)

Compute the pullback of a cospan B --f--> D and C --g--> D.

Construction:
    P = ker( (f, -g) : B oplus C -> D )

Returns:
- P : the pullback module
- prB : P -> B
- prC : P -> C
- iota : P -> (B oplus C) (the kernel inclusion)
- psi : (B oplus C) -> D (the map whose kernel defines the pullback)

The projections satisfy f o prB == g o prC.
"""
function pullback(f::PMorphism{QQ}, g::PMorphism{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    @assert f.cod === g.cod
    B = f.dom
    C = g.dom
    D = f.cod
    S, iB, iC, pB, pC = direct_sum_with_maps(B, C)
    Q = S.Q

    # psi = f o pB - g o pC : B oplus C -> D
    psi_comps = Vector{Matrix{QQ}}(undef, Q.n)
    for u in 1:Q.n
        psi_comps[u] = f.comps[u] * pB.comps[u] - g.comps[u] * pC.comps[u]
    end
    psi = PMorphism{QQ}(S, D, psi_comps)

    P, iota = kernel_with_inclusion(psi; cache=cache)  # iota : P -> S

    prB_comps = Vector{Matrix{QQ}}(undef, Q.n)
    prC_comps = Vector{Matrix{QQ}}(undef, Q.n)
    for u in 1:Q.n
        prB_comps[u] = pB.comps[u] * iota.comps[u]
        prC_comps[u] = pC.comps[u] * iota.comps[u]
    end
    prB = PMorphism{QQ}(P, B, prB_comps)
    prC = PMorphism{QQ}(P, C, prC_comps)

    return P, prB, prC, iota, psi
end

# -----------------------------------------------------------------------------
# Exactness utilities: short exact sequences and snake lemma
# -----------------------------------------------------------------------------

"""
    ShortExactSequence(i, p; check=true, cache=nothing)

Record a short exact sequence 0 -> A --i--> B --p--> C -> 0.

If `check=true` (default), we verify:
- p o i = 0
- i is a monomorphism (pointwise injective)
- p is an epimorphism (pointwise surjective)
- im(i) = ker(p) as submodules of B (pointwise equality)

The check is exact over QQ.

This object caches `ker(p)` and `im(i)` once computed, since many downstream
constructions (Ext/Tor LES, snake lemma, etc.) use them repeatedly.
"""
mutable struct ShortExactSequence{K}
    A::PModule{K}
    B::PModule{K}
    C::PModule{K}
    i::PMorphism{K}
    p::PMorphism{K}
    checked::Bool
    exact::Bool
    ker_p::Union{Nothing,Tuple{PModule{K},PMorphism{K}}}
    img_i::Union{Nothing,Tuple{PModule{K},PMorphism{K}}}
end

function ShortExactSequence(i::PMorphism{QQ}, p::PMorphism{QQ};
                           check::Bool=true,
                           cache::Union{Nothing,CoverCache}=nothing)
    @assert i.cod === p.dom
    ses = ShortExactSequence{QQ}(i.dom, i.cod, p.cod, i, p, false, false, nothing, nothing)
    if check
        ok = is_exact(ses; cache=cache)
        ok || error("ShortExactSequence: maps do not form a short exact sequence")
    end
    return ses
end

"""
    short_exact_sequence(i, p; check=true, cache=nothing) -> ShortExactSequence

Alias for `ShortExactSequence(i, p; check=..., cache=...)`.
"""
short_exact_sequence(i::PMorphism{QQ}, p::PMorphism{QQ};
                     check::Bool=true,
                     cache::Union{Nothing,CoverCache}=nothing) =
    ShortExactSequence(i, p; check=check, cache=cache)

""" 
    is_exact(ses; cache=nothing) -> Bool

Check whether the stored maps define a short exact sequence.
Results are cached inside the object.
"""
function is_exact(ses::ShortExactSequence{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    if ses.checked
        return ses.exact
    end

    i = ses.i
    p = ses.p
    A = ses.A
    B = ses.B
    C = ses.C
    @assert i.dom === A
    @assert i.cod === B
    @assert p.dom === B
    @assert p.cod === C

    # p o i = 0
    for u in 1:B.Q.n
        comp = p.comps[u] * i.comps[u]
        all(iszero, comp) || (ses.checked = true; ses.exact = false; return false)
    end

    # i mono, p epi
    if !is_monomorphism(i) || !is_epimorphism(p)
        ses.checked = true
        ses.exact = false
        return false
    end

    # Compute and cache ker(p) and im(i) as submodules of B.
    if ses.ker_p === nothing
        ses.ker_p = kernel_with_inclusion(p; cache=cache)
    end
    if ses.img_i === nothing
        ses.img_i = image_with_inclusion(i)
    end
    (K, incK) = ses.ker_p
    (Im, incIm) = ses.img_i

    # Compare subspaces at each vertex using ranks.
    Q = B.Q
    for u in 1:Q.n
        Au = incK.comps[u]
        Bu = incIm.comps[u]
        rA = rankQQ(Au)
        rB = rankQQ(Bu)
        if rA != rB
            ses.checked = true
            ses.exact = false
            return false
        end
        # span(Au,Bu) must have same dimension if they are equal.
        rAB = rankQQ(hcat(Au, Bu))
        if rAB != rA
            ses.checked = true
            ses.exact = false
            return false
        end
    end

    ses.checked = true
    ses.exact = true
    return true
end

""" 
    assert_exact(ses; cache=nothing)

Throw an error if `ses` is not exact.
"""
function assert_exact(ses::ShortExactSequence{QQ}; cache::Union{Nothing,CoverCache}=nothing)
    is_exact(ses; cache=cache) || error("ShortExactSequence: sequence is not exact")
    return nothing
end

# Internal: induced map between kernels, given g : A -> B and kernel inclusions kA : ker(fA) -> A, kB : ker(fB) -> B.
function _induced_map_to_kernelQQ(g::PMorphism{QQ},
                                 kA::PMorphism{QQ},
                                 kB::PMorphism{QQ})
    Q = g.dom.Q
    @assert kA.cod === g.dom
    @assert g.cod === kB.cod
    Kdom = kA.dom
    Kcod = kB.dom
    comps = Vector{Matrix{QQ}}(undef, Q.n)
    for u in 1:Q.n
        rhs = g.comps[u] * kA.comps[u]  # B_u x dim kerA_u
        comps[u] = solve_fullcolumnQQ(kB.comps[u], rhs)  # dim kerB_u x dim kerA_u
    end
    return PMorphism{QQ}(Kdom, Kcod, comps)
end

# Internal: induced map between cokernels, given h : A -> B and cokernel projections qA : A -> cokerA, qB : B -> cokerB.
function _induced_map_from_cokernelQQ(h::PMorphism{QQ},
                                     qA::PMorphism{QQ},
                                     qB::PMorphism{QQ})
    Q = h.dom.Q
    @assert qA.dom === h.dom
    @assert h.cod === qB.dom
    Cdom = qA.cod
    Ccod = qB.cod
    comps = Vector{Matrix{QQ}}(undef, Q.n)
    for u in 1:Q.n
        Qsrc = qA.comps[u]
        rinv = _right_inverse_full_rowQQ(Qsrc)
        rhs = qB.comps[u] * h.comps[u]  # cokerB_u x A_u
        comps[u] = rhs * rinv           # cokerB_u x cokerA_u
    end
    return PMorphism{QQ}(Cdom, Ccod, comps)
end

"""
    SnakeLemmaResult

Result of `snake_lemma`: it packages the six objects and five maps in the snake lemma
exact sequence, including the connecting morphism delta.

The long exact sequence has the form:

    ker(fA) -> ker(fB) -> ker(fC) --delta--> coker(fA) -> coker(fB) -> coker(fC)

All maps are returned as actual morphisms of P-modules (natural transformations).
"""
struct SnakeLemmaResult{K}
    kerA::Tuple{PModule{K},PMorphism{K}}   # (KerA, inclusion KerA -> A)
    kerB::Tuple{PModule{K},PMorphism{K}}
    kerC::Tuple{PModule{K},PMorphism{K}}
    cokA::Tuple{PModule{K},PMorphism{K}}   # (CokA, projection A' -> CokA) where A' is cod(fA)
    cokB::Tuple{PModule{K},PMorphism{K}}
    cokC::Tuple{PModule{K},PMorphism{K}}
    k1::PMorphism{K}                       # ker(fA) -> ker(fB)
    k2::PMorphism{K}                       # ker(fB) -> ker(fC)
    delta::PMorphism{K}                    # ker(fC) -> coker(fA)
    c1::PMorphism{K}                       # coker(fA) -> coker(fB)
    c2::PMorphism{K}                       # coker(fB) -> coker(fC)
end

# Internal: check commutativity of a square g1 o f1 == g2 o f2 at all vertices.
function _check_commutative_squareQQ(g1::PMorphism{QQ}, f1::PMorphism{QQ},
                                     g2::PMorphism{QQ}, f2::PMorphism{QQ})
    Q = f1.dom.Q
    @assert f1.cod === g1.dom
    @assert f2.cod === g2.dom
    @assert f1.dom === f2.dom
    @assert g1.cod === g2.cod
    for u in 1:Q.n
        left = g1.comps[u] * f1.comps[u]
        right = g2.comps[u] * f2.comps[u]
        left == right || return false
    end
    return true
end

"""
    snake_lemma(top, bottom, fA, fB, fC; check=true, cache=nothing) -> SnakeLemmaResult

Compute the maps and objects in the snake lemma exact sequence for a commutative diagram
with exact rows:

    0 -> A  --i-->  B  --p-->  C  -> 0
          |        |        |
         fA       fB       fC
          |        |        |
    0 -> A' --i'->  B' --p'-> C' -> 0

Inputs:
- top    : ShortExactSequence for the top row (A,B,C,i,p)
- bottom : ShortExactSequence for the bottom row (A',B',C',i',p')
- fA, fB, fC : vertical morphisms A->A', B->B', C->C'

If `check=true`, we verify that both rows are exact and that the two squares commute.

The connecting morphism `delta : ker(fC) -> coker(fA)` is computed explicitly using
linear algebra in each stalk.
"""
function snake_lemma(top::ShortExactSequence{QQ},
                     bottom::ShortExactSequence{QQ},
                     fA::PMorphism{QQ},
                     fB::PMorphism{QQ},
                     fC::PMorphism{QQ};
                     check::Bool=true,
                     cache::Union{Nothing,CoverCache}=nothing)

    if check
        assert_exact(top; cache=cache)
        assert_exact(bottom; cache=cache)

        # Squares must commute:
        # fB o i = i' o fA
        ok1 = _check_commutative_squareQQ(fB, top.i, bottom.i, fA)
        ok1 || error("snake_lemma: left square does not commute")
        # p' o fB = fC o p
        ok2 = _check_commutative_squareQQ(bottom.p, fB, fC, top.p)
        ok2 || error("snake_lemma: right square does not commute")
    end

    # Kernels of vertical maps
    kerA = kernel_with_inclusion(fA; cache=cache)
    kerB = kernel_with_inclusion(fB; cache=cache)
    kerC = kernel_with_inclusion(fC; cache=cache)

    # Cokernels of vertical maps
    cokA = cokernel_with_projection(fA; cache=cache)
    cokB = cokernel_with_projection(fB; cache=cache)
    cokC = cokernel_with_projection(fC; cache=cache)

    (KerA, incKerA) = kerA
    (KerB, incKerB) = kerB
    (KerC, incKerC) = kerC

    (CokA, qA) = cokA
    (CokB, qB) = cokB
    (CokC, qC) = cokC

    Q = top.A.Q

    # Induced maps on kernels: ker(fA) -> ker(fB) -> ker(fC)
    # k1 : KerA -> KerB induced by top.i : A -> B
    k1 = _induced_map_to_kernelQQ(top.i, incKerA, incKerB)

    # k2 : KerB -> KerC induced by top.p : B -> C
    k2 = _induced_map_to_kernelQQ(top.p, incKerB, incKerC)

    # Induced maps on cokernels: coker(fA) -> coker(fB) -> coker(fC)
    # c1 induced by bottom.i : A' -> B'
    c1 = _induced_map_from_cokernelQQ(bottom.i, qA, qB)

    # c2 induced by bottom.p : B' -> C'
    c2 = _induced_map_from_cokernelQQ(bottom.p, qB, qC)

    # Connecting morphism delta : ker(fC) -> coker(fA)
    delta_comps = Vector{Matrix{QQ}}(undef, Q.n)
    for u in 1:Q.n
        kdim = KerC.dims[u]
        if kdim == 0 || CokA.dims[u] == 0
            delta_comps[u] = zeros(QQ, CokA.dims[u], kdim)
            continue
        end

        # Kc is C_u x kdim, columns are a basis of ker(fC_u) (embedded into C_u).
        Kc = incKerC.comps[u]

        # Lift basis elements in C_u to B_u via p : B -> C (top row).
        # Since p_u is surjective in a short exact sequence, we can use a right inverse.
        rinv_p = _right_inverse_full_rowQQ(top.p.comps[u])  # B_u x C_u
        B_lift = rinv_p * Kc  # B_u x kdim

        # Apply fB to get elements in B'_u.
        Bp = fB.comps[u] * B_lift  # B'_u x kdim

        # Since Kc is in ker(fC), commutativity implies Bp is in ker(p') = im(i').
        Ap = solve_fullcolumnQQ(bottom.i.comps[u], Bp)  # A'_u x kdim

        # Project to coker(fA): qA : A' -> CokA
        delta_comps[u] = qA.comps[u] * Ap  # CokA_u x kdim
    end
    delta = PMorphism{QQ}(KerC, CokA, delta_comps)

    return SnakeLemmaResult{QQ}(kerA, kerB, kerC, cokA, cokB, cokC, k1, k2, delta, c1, c2)
end

"""
    snake_lemma(i, p, i2, p2, fA, fB, fC; check=true, cache=nothing) -> SnakeLemmaResult

Convenience overload: provide the four row maps directly instead of pre-constructing
`ShortExactSequence` objects for the top and bottom rows.
"""
function snake_lemma(i::PMorphism{QQ}, p::PMorphism{QQ},
                     i2::PMorphism{QQ}, p2::PMorphism{QQ},
                     fA::PMorphism{QQ}, fB::PMorphism{QQ}, fC::PMorphism{QQ};
                     check::Bool=true,
                     cache::Union{Nothing,CoverCache}=nothing)
    top = ShortExactSequence(i, p; check=check, cache=cache)
    bottom = ShortExactSequence(i2, p2; check=check, cache=cache)
    return snake_lemma(top, bottom, fA, fB, fC; check=check, cache=cache)
end

# ---------------------------------------------------------------------------
# Categorical biproduct/product/coproduct wrappers (mathematician-friendly API)
# ---------------------------------------------------------------------------

"""
    biproduct(M, N) -> (S, iM, iN, pM, pN)

Binary biproduct in the abelian category of `PModule`s.

Concretely, this is the direct sum `S = M \\oplus N` with its canonical
injections and projections:

- `iM : M -> S`, `iN : N -> S`
- `pM : S -> M`, `pN : S -> N`

This is a readability wrapper around `direct_sum_with_maps`.

See also: `direct_sum_with_maps`, `product`, `coproduct`.
"""
biproduct(M::PModule{K}, N::PModule{K}) where {K} = direct_sum_with_maps(M, N)

"""
    coproduct(M, N) -> (S, iM, iN)

Categorical coproduct of two modules.

In an additive category (in particular for modules), the coproduct is the same
object as the product: the biproduct `M \\oplus N`.

Returns the object `S` and the canonical injections.
"""
function coproduct(M::PModule{K}, N::PModule{K}) where {K}
    S, iM, iN, _, _ = direct_sum_with_maps(M, N)
    return S, iM, iN
end

"""
    product(M, N) -> (P, pM, pN)

Categorical product of two modules.

In an additive category (in particular for modules), the product is the same
object as the coproduct: the biproduct `M \\oplus N`.

Returns the object `P` and the canonical projections.
"""
function product(M::PModule{K}, N::PModule{K}) where {K}
    P, _, _, pM, pN = direct_sum_with_maps(M, N)
    return P, pM, pN
end

# --- Finite products/coproducts (n-ary) ------------------------------------
#
# These are convenient in categorical workflows and are implemented using a
# single-pass block construction for speed (rather than iterating binary sums).

"""
    coproduct(mods::AbstractVector{<:PModule}) -> (S, injections)

Finite coproduct of a list of modules.

Returns:
- `S` : the direct sum object
- `injections[i] : mods[i] -> S` : the canonical injection maps

For an empty list, throws an error.
"""
function coproduct(mods::AbstractVector{<:PModule{K}}) where {K}
    if isempty(mods)
        error("coproduct: need at least one module")
    elseif length(mods) == 1
        M = mods[1]
        return M, PMorphism{K}[id_morphism(M)]
    end
    S, injections, _ = _direct_sum_many_with_maps(mods)
    return S, injections
end

"""
    product(mods::AbstractVector{<:PModule}) -> (P, projections)

Finite product of a list of modules.

Returns:
- `P` : the direct sum object
- `projections[i] : P -> mods[i]` : the canonical projection maps

For an empty list, throws an error.
"""
function product(mods::AbstractVector{<:PModule{K}}) where {K}
    if isempty(mods)
        error("product: need at least one module")
    elseif length(mods) == 1
        M = mods[1]
        return M, PMorphism{K}[id_morphism(M)]
    end
    P, _, projections = _direct_sum_many_with_maps(mods)
    return P, projections
end

# Vararg convenience:
coproduct(M::PModule{K}, N::PModule{K}, rest::PModule{K}...) where {K} =
    coproduct(PModule{K}[M, N, rest...])

product(M::PModule{K}, N::PModule{K}, rest::PModule{K}...) where {K} =
    product(PModule{K}[M, N, rest...])

# --- Internal helper: build one big direct sum in a single pass -------------
#
# Returns (S, injections, projections), where:
# - injections[i] : mods[i] -> S
# - projections[i] : S -> mods[i]
#
# This keeps the common "poset + cover edge maps" structure and uses:
# - block-diagonal edge maps
# - block-identity injections/projections at each vertex
function _direct_sum_many_with_maps(mods::AbstractVector{<:PModule{K}}) where {K}
    m = length(mods)
    m == 0 && error("_direct_sum_many_with_maps: need at least one module")

    Q = mods[1].Q
    n = Q.n
    for M in mods
        if M.Q !== Q
            error("_direct_sum_many_with_maps: modules must live on the same poset")
        end
    end

    # offsets[u][i] is the 0-based starting index of the i-th summand inside the
    # direct-sum fiber at vertex u. (So offsets[u][1] = 0.)
    offsets = [Vector{Int}(undef, m + 1) for _ in 1:n]
    for u in 1:n
        off = offsets[u]
        off[1] = 0
        for i in 1:m
            off[i+1] = off[i] + mods[i].dims[u]
        end
    end

    Sdims = [offsets[u][end] for u in 1:n]

    # Build cover-edge maps for the direct sum.
    cc = cover_cache(Q)

    aligned = true
    for M in mods
        if M.edge_maps.preds != cc.preds || M.edge_maps.succs != cc.succs
            aligned = false
            break
        end
    end

    if aligned
        preds = cc.preds
        succs = cc.succs

        maps_from_pred = [Vector{Matrix{K}}(undef, length(preds[v])) for v in 1:n]
        maps_to_succ   = [Vector{Matrix{K}}(undef, length(succs[u])) for u in 1:n]

        @inbounds for u in 1:n
            su = succs[u]
            outu = maps_to_succ[u]
            for j in eachindex(su)
                v = su[j]
                Auv = zeros(K, Sdims[v], Sdims[u])

                # Fill the block diagonal with summand maps.
                for i in 1:m
                    du = mods[i].dims[u]
                    dv = mods[i].dims[v]
                    if du == 0 || dv == 0
                        continue
                    end
                    r0 = offsets[v][i] + 1
                    r1 = offsets[v][i+1]
                    c0 = offsets[u][i] + 1
                    c1 = offsets[u][i+1]
                    block = mods[i].edge_maps.maps_to_succ[u][j]
                    copyto!(view(Auv, r0:r1, c0:c1), block)
                end

                outu[j] = Auv
                ip = _find_sorted_index(preds[v], u)
                @inbounds maps_from_pred[v][ip] = Auv
            end
        end

        store = CoverEdgeMapStore{K,Matrix{K}}(preds, succs, maps_from_pred, maps_to_succ, cc.nedges)
        S = PModule{K,Matrix{K}}(Q, Sdims, store)
    else
        # Fallback: keyed access (still O(|E|), but slower).
        edge_maps = Dict{Tuple{Int,Int}, Matrix{K}}()
        sizehint!(edge_maps, cc.nedges)
        for (u, v) in cover_edges(Q)
            Auv = zeros(K, Sdims[v], Sdims[u])
            for i in 1:m
                du = mods[i].dims[u]
                dv = mods[i].dims[v]
                if du == 0 || dv == 0
                    continue
                end
                r0 = offsets[v][i] + 1
                r1 = offsets[v][i+1]
                c0 = offsets[u][i] + 1
                c1 = offsets[u][i+1]
                block = mods[i].edge_maps[u, v]
                copyto!(view(Auv, r0:r1, c0:c1), block)
            end
            edge_maps[(u, v)] = Auv
        end
        S = PModule{K}(Q, Sdims, edge_maps)
    end

    # Injections and projections (fill diagonal entries directly).
    injections = PMorphism{K}[]
    projections = PMorphism{K}[]
    for i in 1:m
        inj_comps = Vector{Matrix{K}}(undef, n)
        proj_comps = Vector{Matrix{K}}(undef, n)
        for u in 1:n
            du = mods[i].dims[u]
            Su = Sdims[u]
            Iu = zeros(K, Su, du)
            Pu = zeros(K, du, Su)
            if du != 0
                r0 = offsets[u][i] + 1
                c0 = offsets[u][i] + 1
                @inbounds for t in 1:du
                    Iu[r0 + t - 1, t] = one(K)
                    Pu[t, c0 + t - 1] = one(K)
                end
            end
            inj_comps[u] = Iu
            proj_comps[u] = Pu
        end
        push!(injections, PMorphism{K}(mods[i], S, inj_comps))
        push!(projections, PMorphism{K}(S, mods[i], proj_comps))
    end

    return S, injections, projections
end




# -----------------------------------------------------------------------------
# Pretty-printing / display (ASCII-only)
# -----------------------------------------------------------------------------
#
# Goals:
# - Mathematician-friendly summaries for 
# -  PModule / PMorphism (for REPL ergonomics when doing lots of algebra),
# Submodule, ShortExactSequence, and SnakeLemmaResult.
# - ASCII-only output (no Unicode arrows, symbols, etc).
# - Do NOT trigger heavy computations during printing:
#     * show(ShortExactSequence) must NOT call is_exact(ses)
#     * show(SnakeLemmaResult) must NOT recompute anything
# - Respect IOContext(:limit=>true) by truncating long dim vectors.

# Internal: human-readable scalar name (keeps QQ readable).
_scalar_name(::Type{QQ}) = "QQ"
_scalar_name(::Type{K}) where {K} = string(K)

# Internal: cheap stats for a dims vector (no allocations).
# Returns (sum, nnz, max).
function _dims_stats(dims::AbstractVector{<:Integer})
    total = 0
    nnz = 0
    maxd = 0
    @inbounds for d0 in dims
        d = Int(d0)
        total += d
        if d != 0
            nnz += 1
            if d > maxd
                maxd = d
            end
        end
    end
    return total, nnz, maxd
end

# Internal: print integer vector, truncating if the IOContext requests it.
function _print_int_vec(io::IO, v::AbstractVector{<:Integer};
                        max_elems::Int=12, head::Int=4, tail::Int=3)
    n = length(v)
    print(io, "[")
    if n == 0
        print(io, "]")
        return
    end

    limit = get(io, :limit, false)

    # Full print if not limiting or short enough.
    if !limit || n <= max_elems
        @inbounds for i in 1:n
            i > 1 && print(io, ", ")
            print(io, Int(v[i]))
        end
        print(io, "]")
        return
    end

    # Truncated print: head entries, "...", tail entries.
    h = min(head, n)
    t = min(tail, max(0, n - h))

    @inbounds for i in 1:h
        i > 1 && print(io, ", ")
        print(io, Int(v[i]))
    end

    if h < n - t
        print(io, ", ..., ")
    elseif h < n && t > 0
        print(io, ", ")
    end

    @inbounds for i in (n - t + 1):n
        i > (n - t + 1) && print(io, ", ")
        print(io, Int(v[i]))
    end

    print(io, "]")
    return
end

"""
    Base.show(io::IO, M::PModule{K}) where {K}

Compact one-line summary for a `PModule`.

This is intended for quick REPL inspection. It prints:
- nverts: number of vertices in the underlying finite poset,
- sum/nnz/max: cheap statistics of the stalk dimension vector,
- dims: the stalk dimensions (truncated if `IOContext(io, :limit=>true)`),
- cover_maps: number of stored structure maps along cover edges.

ASCII-only by design.
"""
function Base.show(io::IO, M::PModule{K}) where {K}
    nverts = M.Q.n
    s, nnz, mx = _dims_stats(M.dims)

    print(io, "PModule(")
    print(io, "nverts=", nverts)
    print(io, ", sum=", s)
    print(io, ", nnz=", nnz)
    print(io, ", max=", mx)
    print(io, ", dims=")
    _print_int_vec(io, M.dims)
    print(io, ", cover_maps=", length(M.edge_maps))
    print(io, ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", M::PModule{K}) where {K}

Verbose multi-line summary for a `PModule` (what the REPL typically shows).

This is still cheap: it only scans `dims` once and does not compute ranks,
images, kernels, etc.
"""
function Base.show(io::IO, ::MIME"text/plain", M::PModule{K}) where {K}
    nverts = M.Q.n
    s, nnz, mx = _dims_stats(M.dims)

    println(io, "PModule")
    println(io, "  scalars = ", _scalar_name(K))
    println(io, "  nverts = ", nverts)

    print(io, "  dims = ")
    _print_int_vec(io, M.dims)
    println(io)

    println(io, "    sum = ", s, ", nnz = ", nnz, ", max = ", mx)
    println(io, "  cover_maps = ", length(M.edge_maps), "  (maps stored along cover edges)")
end

"""
    Base.show(io::IO, f::PMorphism{K}) where {K}

Compact one-line summary for a vertexwise morphism of P-modules.

We intentionally avoid any expensive linear algebra (ranks, images, etc.) here.
"""
function Base.show(io::IO, f::PMorphism{K}) where {K}
    n_dom = f.dom.Q.n
    n_cod = f.cod.Q.n

    dom_sum, _, _ = _dims_stats(f.dom.dims)
    cod_sum, _, _ = _dims_stats(f.cod.dims)

    print(io, "PMorphism(")
    if f.dom === f.cod
        print(io, "endo, ")
    end

    if n_dom == n_cod
        print(io, "nverts=", n_dom)
    else
        # Should not happen in well-formed inputs, but keep printing robust.
        print(io, "nverts_dom=", n_dom, ", nverts_cod=", n_cod)
    end

    print(io, ", dom_sum=", dom_sum)
    print(io, ", cod_sum=", cod_sum)
    print(io, ", comps=", length(f.comps))
    print(io, ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", f::PMorphism{K}) where {K}

Verbose multi-line summary for `PMorphism`.

We report basic size information about the domain/codomain and indicate the
intended per-vertex matrix sizes, without verifying them (verification can be
added as a separate validator; printing should stay cheap and noninvasive).
"""
function Base.show(io::IO, ::MIME"text/plain", f::PMorphism{K}) where {K}
    n_dom = f.dom.Q.n
    n_cod = f.cod.Q.n

    dom_sum, dom_nnz, dom_max = _dims_stats(f.dom.dims)
    cod_sum, cod_nnz, cod_max = _dims_stats(f.cod.dims)

    println(io, "PMorphism")
    println(io, "  scalars = ", _scalar_name(K))

    if n_dom == n_cod
        println(io, "  nverts = ", n_dom)
    else
        println(io, "  nverts_dom = ", n_dom)
        println(io, "  nverts_cod = ", n_cod)
    end

    println(io, "  endomorphism = ", (f.dom === f.cod))

    print(io, "  dom dims = ")
    _print_int_vec(io, f.dom.dims)
    println(io)
    println(io, "    sum = ", dom_sum, ", nnz = ", dom_nnz, ", max = ", dom_max)

    print(io, "  cod dims = ")
    _print_int_vec(io, f.cod.dims)
    println(io)
    println(io, "    sum = ", cod_sum, ", nnz = ", cod_nnz, ", max = ", cod_max)

    println(io, "  comps: ", length(f.comps), " vertexwise linear maps")
    println(io, "    comps[i] has size cod.dims[i] x dom.dims[i] (for each vertex i)")
end

"""
    Base.show(io::IO, S::Submodule)

Compact one-line summary for `Submodule`.
"""
function Base.show(io::IO, S::Submodule{K}) where {K}
    N = sub(S)
    M = ambient(S)
    nverts = M.Q.n
    sub_sum, _, _ = _dims_stats(N.dims)
    amb_sum, _, _ = _dims_stats(M.dims)
    print(io,
          "Submodule(",
          "nverts=", nverts,
          ", sub_sum=", sub_sum,
          ", ambient_sum=", amb_sum,
          ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", S::Submodule)

Verbose multi-line summary for `Submodule`. ASCII-only.
"""
function Base.show(io::IO, ::MIME"text/plain", S::Submodule{K}) where {K}
    N = sub(S)
    M = ambient(S)
    nverts = M.Q.n

    sub_sum, sub_nnz, sub_max = _dims_stats(N.dims)
    amb_sum, amb_nnz, amb_max = _dims_stats(M.dims)

    println(io, "Submodule")
    println(io, "  scalars = ", _scalar_name(K))
    println(io, "  nverts = ", nverts)

    print(io, "  sub dims = ")
    _print_int_vec(io, N.dims)
    println(io)
    println(io, "    sum = ", sub_sum, ", nnz = ", sub_nnz, ", max = ", sub_max)

    print(io, "  ambient dims = ")
    _print_int_vec(io, M.dims)
    println(io)
    println(io, "    sum = ", amb_sum, ", nnz = ", amb_nnz, ", max = ", amb_max)

    println(io, "  inclusion : sub(S) -> ambient(S)  (use inclusion(S) to access)")
end


"""
    Base.show(io::IO, ses::ShortExactSequence)

Compact one-line summary for `ShortExactSequence`.

NOTE: This does NOT call `is_exact(ses)`; it only reports cached status.
"""
function Base.show(io::IO, ses::ShortExactSequence{K}) where {K}
    nverts = ses.B.Q.n
    Asum, _, _ = _dims_stats(ses.A.dims)
    Bsum, _, _ = _dims_stats(ses.B.dims)
    Csum, _, _ = _dims_stats(ses.C.dims)

    print(io,
          "ShortExactSequence(",
          "nverts=", nverts,
          ", A_sum=", Asum,
          ", B_sum=", Bsum,
          ", C_sum=", Csum,
          ", checked=", ses.checked,
          ", exact=")
    if ses.checked
        print(io, ses.exact)
    else
        print(io, "unknown")
    end
    print(io, ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", ses::ShortExactSequence)

Verbose multi-line summary for `ShortExactSequence`. ASCII-only.

NOTE: This does NOT call `is_exact(ses)`; it only reports cached status.
"""
function Base.show(io::IO, ::MIME"text/plain", ses::ShortExactSequence{K}) where {K}
    nverts = ses.B.Q.n

    Asum, Annz, Amax = _dims_stats(ses.A.dims)
    Bsum, Bnnz, Bmax = _dims_stats(ses.B.dims)
    Csum, Cnnz, Cmax = _dims_stats(ses.C.dims)

    println(io, "ShortExactSequence")
    println(io, "  0 -> A -(i)-> B -(p)-> C -> 0")
    println(io, "  scalars = ", _scalar_name(K))
    println(io, "  nverts = ", nverts)

    println(io, "  checked = ", ses.checked)
    if ses.checked
        println(io, "  exact = ", ses.exact)
    else
        println(io, "  exact = unknown (call is_exact(ses) to check and cache)")
    end
    println(io, "  caches: ker(p) = ", ses.ker_p !== nothing,
                ", im(i) = ", ses.img_i !== nothing)

    print(io, "  A dims = ")
    _print_int_vec(io, ses.A.dims)
    println(io)
    println(io, "    sum = ", Asum, ", nnz = ", Annz, ", max = ", Amax)

    print(io, "  B dims = ")
    _print_int_vec(io, ses.B.dims)
    println(io)
    println(io, "    sum = ", Bsum, ", nnz = ", Bnnz, ", max = ", Bmax)

    print(io, "  C dims = ")
    _print_int_vec(io, ses.C.dims)
    println(io)
    println(io, "    sum = ", Csum, ", nnz = ", Cnnz, ", max = ", Cmax)

    println(io, "  maps: use ses.i and ses.p (both are PMorphism objects)")
end


"""
    Base.show(io::IO, sn::SnakeLemmaResult)

Compact one-line summary for `SnakeLemmaResult`.
"""
function Base.show(io::IO, sn::SnakeLemmaResult{K}) where {K}
    nverts = sn.delta.dom.Q.n
    kerCsum, _, _ = _dims_stats(sn.kerC[1].dims)
    cokAsum, _, _ = _dims_stats(sn.cokA[1].dims)

    print(io,
          "SnakeLemmaResult(",
          "nverts=", nverts,
          ", delta: kerC_sum=", kerCsum,
          " -> cokerA_sum=", cokAsum,
          ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", sn::SnakeLemmaResult)

Verbose multi-line summary for `SnakeLemmaResult`. ASCII-only.
"""
function Base.show(io::IO, ::MIME"text/plain", sn::SnakeLemmaResult{K}) where {K}
    nverts = sn.delta.dom.Q.n

    println(io, "SnakeLemmaResult")
    println(io, "  kerA -> kerB -> kerC --delta--> cokerA -> cokerB -> cokerC")
    println(io, "  scalars = ", _scalar_name(K))
    println(io, "  nverts = ", nverts)

    # Helper to print one object entry (kerX/cokerX).
    function _print_obj(io2::IO, name::AbstractString, tup)
        M = tup[1]
        s, nnz, mx = _dims_stats(M.dims)
        print(io2, "  ", name, " dims = ")
        _print_int_vec(io2, M.dims)
        println(io2)
        println(io2, "    sum = ", s, ", nnz = ", nnz, ", max = ", mx)
    end

    _print_obj(io, "kerA", sn.kerA)
    _print_obj(io, "kerB", sn.kerB)
    _print_obj(io, "kerC", sn.kerC)
    _print_obj(io, "cokerA", sn.cokA)
    _print_obj(io, "cokerB", sn.cokB)
    _print_obj(io, "cokerC", sn.cokC)

    println(io, "  maps: k1, k2, delta, c1, c2  (access as fields on the result)")
end



"""
    upset_presentation_one_step(Hfringe::FringeModule)
Compute the one-step upset presentation (Def. 6.4.1):
    F1 --d1--> F0 --pi0-->> M,
and return the lightweight wrapper `UpsetPresentation{QQ}(P, U0, U1, delta)`.
"""
function upset_presentation_one_step(H::FiniteFringe.FringeModule)
    M = pmodule_from_fringe(H)          # internal PModule over QQ
    cc = _cover_cache(M.Q)
    # First step: projective cover of M
    F0, pi0, gens_at_F0 = projective_cover(M; cache=cc)
    # Kernel K1 with inclusion i1 : K1 \into F0
    K1, i1 = kernel_with_inclusion(pi0; cache=cc)
    # Second projective cover (of K1)
    F1, pi1, gens_at_F1 = projective_cover(K1; cache=cc)
    # Differential d1 = i1 \circ pi1 : F1 \to F0
    comps = [i1.comps[i] * pi1.comps[i] for i in 1:length(M.dims)]
    d1 = PMorphism{QQ}(F1, F0, comps)

    # Build indicator wrapper:
    # U0: list of principal upsets, one per generator in gens_at_F0[p]
    P = M.Q
    U0 = Upset[]
    for p in 1:P.n
        for _ in gens_at_F0[p]
            push!(U0, principal_upset(P, p))
        end
    end
    U1 = Upset[]
    for p in 1:P.n
        for _ in gens_at_F1[p]
            push!(U1, principal_upset(P, p))
        end
    end

    # scalar delta block: one entry for each pair (theta in U1, lambda in U0) if ptheta <= plambda.
    # Extract the scalar from d1 at the minimal vertex i = plambda.
    m1 = length(U1); m0 = length(U0)
    delta = spzeros(QQ, m1, m0)
    # local index maps: at vertex i, Fk_i basis is "all generators at p with p <= i" in (p increasing) order.
    # Build offsets to find local coordinates.
    function local_index_list(gens_at)
        # return vector of vectors L[i] listing global generator indices active at vertex i
        L = Vector{Vector{Tuple{Int,Int}}}(undef, P.n)
        for i in 1:P.n
            L[i] = Tuple{Int,Int}[]
            for p in 1:P.n
                if P.leq[p,i]
                    append!(L[i], gens_at[p])
                end
            end
        end
        L
    end
    L0 = local_index_list(gens_at_F0)
    L1 = local_index_list(gens_at_F1)

    # Build a map from "global generator number in U0/U1" to its vertex p and position j in M_p
    globalU0 = Tuple{Int,Int}[]  # (p,j)
    for p in 1:P.n; append!(globalU0, gens_at_F0[p]); end
    globalU1 = Tuple{Int,Int}[]
    for p in 1:P.n; append!(globalU1, gens_at_F1[p]); end

    # helper: find local column index of global generator g=(p,j) at vertex i
    function local_col_of(L, g::Tuple{Int,Int}, i::Int)
        for (c, gg) in enumerate(L[i])
            if gg == g; return c; end
        end
        return 0
    end

    for (lambda, (plambda, jlambda)) in enumerate(globalU0)
        for (theta, (ptheta, jtheta)) in enumerate(globalU1)
            if P.leq[plambda, ptheta]   # containment for principal upsets: Up(ptheta) subseteq Up(plambda)
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

    UpsetPresentation{QQ}(P, U0, U1, delta)
end

# ------------------------ downset copresentation (Def. 6.4.2) ------------------------

# The dual story: compute an injective hull E0 of M and the next step E1 with rho0 : E0 \to E1.
# For brevity we implement the duals by applying the above steps to M^op using
# down-closures/up-closures symmetry. Here we implement directly degreewise.

"Outgoing coimage at u to immediate successors; basis for the span of maps M_u to oplus_{u<v} M_v."
function _outgoing_span_basis(M::PModule{QQ}, u::Int; cache::Union{Nothing,CoverCache}=nothing)
    cc = (cache === nothing ? cover_cache(M.Q) : cache)
    su = cc.succs[u]
    du = M.dims[u]

    if isempty(su) || du == 0
        return zeros(QQ, 0, du)
    end

    # Stack outgoing maps B : M_u -> (direct sum over cover successors).
    tot = 0
    @inbounds for j in eachindex(su)
        tot += M.dims[su[j]]
    end
    if tot == 0
        return zeros(QQ, 0, du)
    end

    B = Matrix{QQ}(undef, tot, du)
    row = 1
    maps_u = M.edge_maps.maps_to_succ[u]

    @inbounds for j in eachindex(su)
        v = su[j]
        dv = M.dims[v]
        if dv > 0
            B[row:row+dv-1, :] .= maps_u[j]
            row += dv
        end
    end

    # Return a basis for the row space of B as an r x du matrix with full row rank.
    return transpose(colspaceQQ(transpose(B)))
end


"Essential socle dimension at u = dim(M_u) - rank(outgoing span) (dual of generators)."
function _socle_count(M::PModule{QQ}, u::Int)
    S = _outgoing_span_basis(M, u)
    M.dims[u] - rankQQ(S)
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
function downset_copresentation_one_step(H::FiniteFringe.FringeModule)
    # Convert fringe to internal PModule over QQ
    M = pmodule_from_fringe(H)
    Q = M.Q; n = Q.n
    cc = _cover_cache(Q)

    # (1) Injective hull of M: E0 with inclusion iota0
    E0, iota0, gens_at_E0 = _injective_hull(M; cache=cc)

    # (2) Degreewise cokernel C and the quotient q : E0 \to C
    C, q = _cokernel_module(iota0; cache=cc)

    # (3) Injective hull of C: E1 with inclusion j : C \into E1
    E1, j, gens_at_E1 = _injective_hull(C; cache=cc)

    # Compose to get rho0 : E0 \to E1 (at each vertex i: rho0[i] = j[i] * q[i])
    comps_rho0 = [ j.comps[i] * q.comps[i] for i in 1:n ]
    rho0 = PMorphism{QQ}(E0, E1, comps_rho0)

    # (4) Assemble the indicator wrapper: labels D0, D1 and the scalar block rho.
    # D0/D1 each contain one principal downset for every generator (u,j) chosen above.
    D0 = Downset[]
    for u in 1:n, _ in gens_at_E0[u]
        push!(D0, principal_downset(Q, u))
    end
    D1 = Downset[]
    for u in 1:n, _ in gens_at_E1[u]
        push!(D1, principal_downset(Q, u))
    end

    # Local index lists: at vertex i, the active generators are those born at u with i <= u.
    function _local_index_list_D(gens_at)
        L = Vector{Vector{Tuple{Int,Int}}}(undef, n)
        for i in 1:n
            lst = Tuple{Int,Int}[]
            for u in 1:n
                if Q.leq[i,u]
                    append!(lst, gens_at[u])
                end
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
    rho = spzeros(QQ, m1, m0)

    for (lambda, (ulambda, jlambda)) in enumerate(globalD0)
        for (theta, (utheta, jtheta)) in enumerate(globalD1)
            if Q.leq[utheta, ulambda] # D(utheta) subseteq D(ulambda)
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

    return DownsetCopresentation{QQ}(Q, D0, D1, rho)
end



"""
    prune_zero_relations(F::UpsetPresentation{QQ}) -> UpsetPresentation{QQ}

Remove rows of `delta` that are identically zero (redundant relations) and drop the
corresponding entries of `U1`. The cokernel is unchanged.
"""
function prune_zero_relations(F::UpsetPresentation{QQ})
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
    UpsetPresentation{QQ}(F.P, F.U0, new_U1, new_delta)
end

"""
    cancel_isolated_unit_pairs(F::UpsetPresentation{QQ}) -> UpsetPresentation{QQ}

Iteratively cancels isolated nonzero entries `delta[theta,lambda]` for which:
  * the theta-th row has exactly that one nonzero,
  * the lambda-th column has exactly that one nonzero, and
  * U1[theta] == U0[lambda] as Upsets (principal upsets match).

Each cancellation removes one generator in U0 and one relation in U1 without
changing the cokernel.
"""
function cancel_isolated_unit_pairs(F::UpsetPresentation{QQ})
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
    UpsetPresentation{QQ}(P, U0, U1, Delta)
end

"""
    minimal_upset_presentation_one_step(H::FiniteFringe.FringeModule)
        -> UpsetPresentation{QQ}

Build a one-step upset presentation and apply safe minimality passes:
1) drop zero relations; 2) cancel isolated isomorphism pairs.
"""
function minimal_upset_presentation_one_step(H::FiniteFringe.FringeModule)
    F = upset_presentation_one_step(H)     # existing builder
    F = prune_zero_relations(F)
    F = cancel_isolated_unit_pairs(F)
    return F
end


"""
    prune_unused_targets(E::DownsetCopresentation{QQ}) -> DownsetCopresentation{QQ}

Drop rows of `rho` that are identically zero (unused target summands in E^1). The kernel is unchanged.
"""
function prune_unused_targets(E::DownsetCopresentation{QQ})
    m1, m0 = size(E.rho)
    keep = trues(m1)
    rows, _, _ = findnz(E.rho)
    seen = falses(m1); @inbounds for r in rows; seen[r] = true; end
    @inbounds for r in 1:m1
        if !seen[r]; keep[r] = false; end
    end
    new_D1 = [E.D1[i] for i in 1:m1 if keep[i]]
    new_rho = E.rho[keep, :]
    DownsetCopresentation{QQ}(E.P, E.D0, new_D1, new_rho)
end

"""
    cancel_isolated_unit_pairs(E::DownsetCopresentation{QQ}) -> DownsetCopresentation{QQ}

Iteratively cancels isolated nonzero entries `rho[theta,lambda]` with matching principal downsets
(D1[theta] == D0[lambda]) and unique in their row/column.
"""
function cancel_isolated_unit_pairs(E::DownsetCopresentation{QQ})
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
    DownsetCopresentation{QQ}(P, D0, D1, R)
end

"""
    minimal_downset_copresentation_one_step(H::FiniteFringe.FringeModule)
        -> DownsetCopresentation{QQ}

Build a one-step downset copresentation and apply safe minimality passes:
1) drop zero target rows; 2) cancel isolated isomorphism pairs.
"""
function minimal_downset_copresentation_one_step(H::FiniteFringe.FringeModule)
    E = downset_copresentation_one_step(H)
    E = prune_unused_targets(E)
    E = cancel_isolated_unit_pairs(E)
    return E
end


# --------------------- First page dimensions from one-step data ---------------------

"""
    hom_ext_first_page(F0F1::UpsetPresentation{QQ}, E0E1::DownsetCopresentation{QQ})
Return `(dimHom, dimExt1)` computed from the first page (Defs. 6.1 & 6.4).
This delegates to the HomExt block assembly through the one-step data.
"""
function hom_ext_first_page(F::UpsetPresentation{QQ}, E::DownsetCopresentation{QQ})
    P = F.P
    @assert E.P === P "hom_ext_first_page: posets must match"

    # Wrap one-step data as length-1 (co)resolutions so we can reuse HomExt's
    # block assembly to get the induced maps on Hom-spaces.
    Fvec = UpsetPresentation{QQ}[
        F,
        UpsetPresentation{QQ}(P, F.U1, Upset[], spzeros(QQ, 0, length(F.U1))),
    ]
    dF = SparseMatrixCSC{QQ,Int}[F.delta]

    Evec = DownsetCopresentation{QQ}[
        E,
        DownsetCopresentation{QQ}(P, E.D1, Downset[], spzeros(QQ, 0, length(E.D1))),
    ]
    dE = SparseMatrixCSC{QQ,Int}[E.rho]

    # This gives the component-basis Hom-blocks and the total differential matrices.
    dimsCt, dts = build_hom_tot_complex(Fvec, dF, Evec, dE)

    # dimsCt[1] = dim Hom(F0, E0)
    n00 = dimsCt[1]

    # In build_hom_tot_complex, the (a,b) blocks in total degree t are appended
    # in the order produced by the nested loops (a outer, b inner).
    # For t=1 that means (a=0,b=1) comes before (a=1,b=0), i.e.
    #   C^1 = Hom(F0,E1) oplus Hom(F1,E0).
    #
    # We need the split point n01 = dim Hom(F0,E1).
    n01 = 0
    for D in E.D1, U in F.U0
        n01 += pi0_count(P, U, D)
    end
    n10 = dimsCt[2] - n01  # dim Hom(F1,E0)

    # d0 : C^0 -> C^1 is stacked as [rho0; delta0]
    d0 = dts[1]

    rho0 = (n01 == 0) ? zeros(QQ, 0, n00) : d0[1:n01, :]          # Hom(F0,E0) -> Hom(F0,E1)
    # delta0 is the remaining block: Hom(F0,E0) -> Hom(F1,E0)
    # (We don't need it explicitly; it's part of d0.)

    # d1 : C^1 -> C^2, and the columns after n01 correspond to Hom(F1,E0).
    d1 = dts[2]
    rho1 = (n10 == 0) ? zeros(QQ, size(d1,1), 0) : d1[:, n01+1:end]  # Hom(F1,E0) -> Hom(F1,E1)

    # Ranks we need
    r_rho0 = rankQQ(rho0)
    r_d0   = rankQQ(d0)       # rank([rho0; delta0])
    r_rho1 = rankQQ(rho1)

    # Using:
    #   Hom(F0,N) = ker(rho0)
    #   Hom(F1,N) = ker(rho1)
    #   Hom(M,N)  = ker(delta^* : Hom(F0,N)->Hom(F1,N)) = ker([rho0;delta0])
    dimHom = n00 - r_d0

    # rank(delta^* restricted to ker(rho0)) = rank([rho0;delta0]) - rank(rho0)
    rank_delta_on_ker = r_d0 - r_rho0

    dimHomF1N = n10 - r_rho1
    dimExt1 = dimHomF1N - rank_delta_on_ker

    return dimHom, dimExt1
end



# =============================================================================
# Longer indicator resolutions and high-level Ext driver
# =============================================================================
# We expose:
#   * upset_resolution(H; maxlen)     -> (F, dF)
#   * downset_resolution(H; maxlen)   -> (E, dE)
#   * indicator_resolutions(HM, HN; maxlen) -> (F, dF, E, dE)
#   * ext_dimensions_via_indicator_resolutions(HM, HN; maxlen) -> Dict{Int,Int}
#
# The outputs (F, dF, E, dE) are exactly the shapes expected by
# HomExt.build_hom_tot_complex / HomExt.ext_dims_via_resolutions:
#   - F is a Vector{UpsetPresentation{QQ}} with F[a+1].U0 = U_a
#   - dF[a] is the sparse delta_a : U_a <- U_{a+1}  (shape |U_{a+1}| x |U_a|)
#   - E is a Vector{DownsetCopresentation{QQ}} with E[b+1].D0 = D_b
#   - dE[b] is the sparse rho_b : D_b -> D_{b+1}    (shape |D_{b+1}| x |D_b|)
#
# Construction mirrors section 6.1 and the one-step routines already present.
# =============================================================================

using ..HomExt: build_hom_tot_complex, ext_dims_via_resolutions  # re-exported API. :contentReference[oaicite:2]{index=2}

# ------------------------------ small helpers --------------------------------

# Build the list of principal upsets from per-vertex generator labels returned by
# projective_cover: gens_at[v] is a vector of pairs (p, j).  Each pair contributes
# one principal upset at vertex p.
function _principal_upsets_from_gens(P::FinitePoset,
                                     gens_at::Vector{Vector{Tuple{Int,Int}}})
    U = Upset[]
    for p in 1:P.n
        for _ in gens_at[p]
            push!(U, principal_upset(P, p))
        end
    end
    U
end

# Build the list of principal downsets from per-vertex labels returned by _injective_hull
function _principal_downsets_from_gens(P::FinitePoset,
                                       gens_at::Vector{Vector{Tuple{Int,Int}}})
    D = Downset[]
    for u in 1:P.n
        for _ in gens_at[u]
            push!(D, principal_downset(P, u))
        end
    end
    D
end

# For upset side: at vertex i, which global generators are active (born at p <= i)?
# Returns L[i] = vector of global generator labels (p,j) visible at i.
function _local_index_list_up(P::FinitePoset,
                              gens_at::Vector{Vector{Tuple{Int,Int}}})
    L = Vector{Vector{Tuple{Int,Int}}}(undef, P.n)
    for i in 1:P.n
        lst = Tuple{Int,Int}[]
        for p in 1:P.n
            if P.leq[p,i]
                append!(lst, gens_at[p])
            end
        end
        L[i] = lst
    end
    L
end

# For downset side: at vertex i, which global generators are active (born at u with i <= u)?
# Returns L[i] = vector of global generator labels (u,j) visible at i.
function _local_index_list_down(P::FinitePoset,
                                gens_at::Vector{Vector{Tuple{Int,Int}}})
    L = Vector{Vector{Tuple{Int,Int}}}(undef, P.n)
    for i in 1:P.n
        lst = Tuple{Int,Int}[]
        for u in 1:P.n
            if P.leq[i,u]
                append!(lst, gens_at[u])
            end
        end
        L[i] = lst
    end
    L
end

# Find the local column index (1-based) of a global generator g=(p,j) or (u,j) in L[i].
# Returns 0 if not present.
function _local_col_of(L::Vector{Vector{Tuple{Int,Int}}}, g::Tuple{Int,Int}, i::Int)
    for (c, gg) in enumerate(L[i])
        if gg == g; return c; end
    end
    return 0
end

# Dense -> sparse helper over QQ (already have a general helper in FiniteFringe, but here
# we build directly from triplets to avoid materializing full dense blocks).
function _empty_sparse_QQ(nr::Int, nc::Int)
    return spzeros(QQ, nr, nc)
end

# ------------------------------ upset resolution ------------------------------

"""
    upset_resolution(H::FiniteFringe.FringeModule{QQ}; maxlen=nothing)

Compute an upset (projective) indicator resolution of the fringe module `H`.
`maxlen` is a cutoff on the number of differentials computed (it does not pad output).
"""
function upset_resolution(H::FiniteFringe.FringeModule{QQ}; maxlen::Union{Int,Nothing}=nothing)
    return upset_resolution(pmodule_from_fringe(H); maxlen=maxlen)
end


"""
upset_resolution(M::PModule{QQ}; maxlen::Union{Int,Nothing}=nothing)
        -> (F::Vector{UpsetPresentation{QQ}}, dF::Vector{SparseMatrixCSC{QQ,Int}})

Overload of `upset_resolution` for an already-constructed finite-poset module `M`.

This is the same construction as `upset_resolution(::FiniteFringe.FringeModule{QQ})`,
but it skips the `pmodule_from_fringe` conversion step. This matters when callers
already have a `PModule{QQ}` (for example, after encoding a Z^n or R^n module to a
finite poset, or after explicitly calling `pmodule_from_fringe`).
"""
function upset_resolution(M::PModule{QQ}; maxlen::Union{Int,Nothing}=nothing)
    P = M.Q

    # First projective cover: F0 --pi0--> M, with labels gens_at_F0
    F0, pi0, gens_at_F0 = projective_cover(M)

    U_by_a = Vector{Vector{Upset}}()
    push!(U_by_a, _principal_upsets_from_gens(P, gens_at_F0))

    dF = Vector{SparseMatrixCSC{QQ,Int}}()

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
        K, iota = kernel_with_inclusion(curr_pi)   # K = ker(F_prev -> ...)
        if sum(K.dims) == 0
            break
        end
        Fnext, pinext, gens_at_next = projective_cover(K)

        # d = iota o pinext : Fnext -> curr_dom
        comps = Vector{Matrix{QQ}}(undef, P.n)
        for i in 1:P.n
            comps[i] = iota.comps[i] * pinext.comps[i]
        end
        d = PMorphism{QQ}(Fnext, curr_dom, comps)

        # Labels and local index lists
        U_next = _principal_upsets_from_gens(P, gens_at_next)
        Lprev  = _local_index_list_up(P, curr_gens)
        Lnext  = _local_index_list_up(P, gens_at_next)

        # Global enumerations of generators (record their birth vertices)
        global_prev = Tuple{Int,Int}[]
        for p in 1:P.n
            append!(global_prev, curr_gens[p])
        end
        global_next = Tuple{Int,Int}[]
        for p in 1:P.n
            append!(global_next, gens_at_next[p])
        end

        # Assemble sparse delta: rows index next, cols index prev
        delta = _empty_sparse_QQ(length(global_next), length(global_prev))
        for (lambda, (plambda, jlambda)) in enumerate(global_prev)
            for (theta, (ptheta, jtheta)) in enumerate(global_next)
                # Containment for principal upsets: Up(ptheta) subseteq Up(plambda)
                if P.leq[plambda, ptheta]
                    # Read at the minimal vertex where the domain generator exists
                    i = ptheta
                    col = _local_col_of(Lnext, (ptheta, jtheta), i)
                    row = _local_col_of(Lprev, (plambda, jlambda), i)
                    if col > 0 && row > 0
                        val = d.comps[i][row, col]
                        if val != 0
                            delta[theta, lambda] = val
                        end
                    end
                end
            end
        end

        push!(dF, delta)
        push!(U_by_a, U_next)

        # Advance
        curr_dom  = Fnext
        curr_pi   = pinext
        curr_gens = gens_at_next
        steps += 1
    end

    # Package as UpsetPresentation list
    F = Vector{UpsetPresentation{QQ}}(undef, length(U_by_a))
    for a in 1:length(U_by_a)
        U0 = U_by_a[a]
        if a < length(U_by_a)
            U1 = U_by_a[a+1]
            delta = dF[a]
        else
            U1 = Upset[]
            delta = spzeros(QQ, 0, length(U0))
        end
        F[a] = UpsetPresentation{QQ}(P, U0, U1, delta)
    end

    return F, dF
end


# ---------------------------- downset resolution ------------------------------

"""
    downset_resolution(H::FiniteFringe.FringeModule{QQ}; maxlen=nothing)

Compute a downset (injective) indicator resolution of the fringe module `H`.
`maxlen` is a cutoff on the number of differentials computed (it does not pad output).
"""
function downset_resolution(H::FiniteFringe.FringeModule{QQ}; maxlen::Union{Int,Nothing}=nothing)
    return downset_resolution(pmodule_from_fringe(H); maxlen=maxlen)
end


"""
downset_resolution(M::PModule{QQ}; maxlen::Union{Int,Nothing}=nothing)
        -> (E::Vector{DownsetCopresentation{QQ}}, dE::Vector{SparseMatrixCSC{QQ,Int}})

Overload of `downset_resolution` for an already-constructed finite-poset module `M`.

This is the same construction as `downset_resolution(::FiniteFringe.FringeModule{QQ})`,
but it skips the `pmodule_from_fringe` conversion step.
"""
function downset_resolution(M::PModule{QQ}; maxlen::Union{Int,Nothing}=nothing)
    P = M.Q

    # First injective hull: iota0 : M -> E0
    E0, iota0, gens_at_E0 = _injective_hull(M)
    D_by_b = Vector{Vector{Downset}}()
    push!(D_by_b, _principal_downsets_from_gens(P, gens_at_E0))

    dE = Vector{SparseMatrixCSC{QQ,Int}}()

    # First cokernel
    C, q = _cokernel_module(iota0)

    # Iteration state
    prev_E    = E0
    prev_gens = gens_at_E0
    prev_q    = q
    steps = 0

    while true
        if sum(C.dims) == 0
            break
        end
        if maxlen !== nothing && steps >= maxlen
            break
        end

        # Injective hull of the current cokernel: j : C -> E1
        E1, j, gens_at_E1 = _injective_hull(C)

        # rho_b = j o prev_q : prev_E -> E1
        comps = Vector{Matrix{QQ}}(undef, P.n)
        for i in 1:P.n
            comps[i] = j.comps[i] * prev_q.comps[i]
        end
        rho = PMorphism{QQ}(prev_E, E1, comps)

        # Labels and local index lists
        D_next = _principal_downsets_from_gens(P, gens_at_E1)
        L0     = _local_index_list_down(P, prev_gens)
        L1     = _local_index_list_down(P, gens_at_E1)

        globalD0 = Tuple{Int,Int}[]
        for u in 1:P.n
            append!(globalD0, prev_gens[u])
        end
        globalD1 = Tuple{Int,Int}[]
        for u in 1:P.n
            append!(globalD1, gens_at_E1[u])
        end

        # Assemble sparse rho: rows index D1 (next), cols index D0 (prev)
        Rh = _empty_sparse_QQ(length(globalD1), length(globalD0))
        for (lambda, (ulambda, jlambda)) in enumerate(globalD0)      # col in D0
            for (theta,  (utheta,  jtheta))  in enumerate(globalD1)  # row in D1
                # Containment for principal downsets: D(utheta) subseteq D(ulambda)
                if P.leq[utheta, ulambda]
                    i   = utheta
                    col = _local_col_of(L0, (ulambda, jlambda), i)
                    row = _local_col_of(L1, (utheta,  jtheta),  i)
                    if col > 0 && row > 0
                        val = rho.comps[i][row, col]
                        if val != 0
                            Rh[theta, lambda] = val
                        end
                    end
                end
            end
        end

        push!(dE, Rh)
        push!(D_by_b, D_next)

        # Next cokernel and advance
        C, q_next = _cokernel_module(j)
        prev_E    = E1
        prev_gens = gens_at_E1
        prev_q    = q_next
        steps += 1
    end

    # Package as DownsetCopresentation list
    E = Vector{DownsetCopresentation{QQ}}(undef, length(D_by_b))
    for b in 1:length(D_by_b)
        D0 = D_by_b[b]
        if b < length(D_by_b)
            D1 = D_by_b[b+1]
            rho_b = dE[b]
        else
            D1 = Downset[]
            rho_b = spzeros(QQ, 0, length(D0))
        end
        E[b] = DownsetCopresentation{QQ}(P, D0, D1, rho_b)
    end

    return E, dE
end


# --------------------- aggregator + high-level Ext driver ---------------------

"""
    indicator_resolutions(HM, HN; maxlen=nothing)
        -> (F, dF, E, dE)

Convenience wrapper: build an upset resolution for the source module (fringe HM)
and a downset resolution for the target module (fringe HN).  The `maxlen` keyword
cuts off each side after that many steps (useful for quick tests).
"""
function indicator_resolutions(HM::FiniteFringe.FringeModule{QQ},
                               HN::FiniteFringe.FringeModule{QQ};
                               maxlen::Union{Int,Nothing}=nothing)
    F, dF = upset_resolution(HM; maxlen=maxlen)
    E, dE = downset_resolution(HN; maxlen=maxlen)
    return F, dF, E, dE
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
function verify_upset_resolution(F::Vector{UpsetPresentation{QQ}},
                                dF::Vector{SparseMatrixCSC{QQ,Int}};
                                vertices = :all,
                                check_d2::Bool = true,
                                check_exactness::Bool = true,
                                check_connected::Bool = true)

    @assert length(dF) == length(F) - 1 "verify_upset_resolution: expected length(dF) == length(F)-1"
    P = F[1].P
    for f in F
        @assert f.P === P "verify_upset_resolution: all presentations must use the same poset"
    end

    U_by_a = [f.U0 for f in F]  # U_0,...,U_A
    vs = (vertices === :all) ? (1:P.n) : vertices


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
            if nnz(C) != 0
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

                r_prev = isempty(active_km1) ? 0 : rankQQ_restricted(delta_prev, active_k, active_km1)
                r_next = isempty(active_kp1) ? 0 : rankQQ_restricted(delta_next, active_kp1, active_k)

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
function verify_downset_resolution(E::Vector{DownsetCopresentation{QQ}},
                                  dE::Vector{SparseMatrixCSC{QQ,Int}};
                                  vertices = :all,
                                  check_d2::Bool = true,
                                  check_exactness::Bool = true,
                                  check_connected::Bool = true)

    @assert length(dE) == length(E) - 1 "verify_downset_resolution: expected length(dE) == length(E)-1"
    P = E[1].P
    for e in E
        @assert e.P === P "verify_downset_resolution: all copresentations must use the same poset"
    end

    D_by_b = [e.D0 for e in E]  # D_0,...,D_B
    vs = (vertices === :all) ? (1:P.n) : vertices


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
            if nnz(C) != 0
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

                r_prev = isempty(active_km1) ? 0 : rankQQ_restricted(rho_prev, active_k, active_km1)
                r_next = isempty(active_kp1) ? 0 : rankQQ_restricted(rho_next, active_kp1, active_k)

                if r_prev + r_next != dim_mid
                    error("Downset resolution: vertex q=$q fails exactness at degree k=$(k-1): rank(prev)=$r_prev, rank(next)=$r_next, dim=$dim_mid")
                end
            end
        end
    end

    return true
end


"""
    ext_dimensions_via_indicator_resolutions(HM, HN; maxlen=nothing)::Dict{Int,Int}

Build indicator resolutions for HM and HN and return a dictionary mapping total
degree t to dim Ext^t(HM, HN) calculated from the Tot complex.  This is exactly
HomExt.ext_dims_via_resolutions(F, dF, E, dE) after constructing (F,dF,E,dE).
"""
function ext_dimensions_via_indicator_resolutions(HM::FiniteFringe.FringeModule{QQ},
                                                  HN::FiniteFringe.FringeModule{QQ};
                                                  maxlen::Union{Int,Nothing}=nothing,
                                                  verify::Bool=true,
                                                  vertices=:all)

    F, dF, E, dE = indicator_resolutions(HM, HN; maxlen=maxlen)

    if verify
        verify_upset_resolution(F, dF; vertices=vertices)
        verify_downset_resolution(E, dE; vertices=vertices)
    end

    return ext_dims_via_resolutions(F, dF, E, dE)
end



# =============================================================================

export PModule, PMorphism,
       pmodule_from_fringe, map_leq, projective_cover, injective_hull,
       upset_resolution, downset_resolution,
       indicator_resolutions, ext_dimensions_via_indicator_resolutions,
       verify_upset_resolution, verify_downset_resolution,
       hom_ext_first_page,
       # Abelian-category / submodule / diagram-chasing API
       kernel_with_inclusion, kernel, image_with_inclusion, image,
       cokernel_with_projection, cokernel,
       coimage_with_projection, coimage,
       quotient_with_projection, quotient,
       is_zero_morphism, is_monomorphism, is_epimorphism,
       Submodule, submodule, sub, ambient, inclusion,
       kernel_submodule, image_submodule,
       pushout, pullback,
       ShortExactSequence, short_exact_sequence, is_exact, assert_exact,
       snake_lemma, SnakeLemmaResult,
       # Basic constructors
       zero_pmodule, zero_morphism, direct_sum, direct_sum_with_maps,

       biproduct, product, coproduct,
       equalizer, coequalizer,
       AbstractDiagram, DiscretePairDiagram, ParallelPairDiagram, SpanDiagram, CospanDiagram,
       limit, colimit


end # module
