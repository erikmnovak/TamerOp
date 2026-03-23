# finite_fringe/posets.jl
# Scope: iterator helpers, poset interfaces, cover caches, and concrete finite-poset types.

# =========================================
# Iterator helpers (tuple/iterator-first APIs)
# =========================================

struct IndicesView
    data::Vector{Int}
end

Base.IteratorSize(::Type{IndicesView}) = Base.HasLength()
Base.eltype(::Type{IndicesView}) = Int
Base.length(view::IndicesView) = length(view.data)
Base.size(view::IndicesView) = (length(view.data),)
Base.getindex(view::IndicesView, i::Int) = view.data[i]
Base.iterate(view::IndicesView, state::Int=1) =
    state > length(view.data) ? nothing : (view.data[state], state + 1)

struct PosetLeqIter{P}
    P::P
    i::Int
    is_upset::Bool
end

Base.IteratorSize(::Type{PosetLeqIter}) = Base.HasLength()
Base.eltype(::Type{PosetLeqIter}) = Int
function Base.length(it::PosetLeqIter)
    n = nvertices(it.P)
    cnt = 0
    if it.is_upset
        @inbounds for j in 1:n
            cnt += leq(it.P, it.i, j) ? 1 : 0
        end
    else
        @inbounds for j in 1:n
            cnt += leq(it.P, j, it.i) ? 1 : 0
        end
    end
    return cnt
end
function Base.iterate(it::PosetLeqIter, state::Int=1)
    n = nvertices(it.P)
    j = state
    if it.is_upset
        while j <= n
            if leq(it.P, it.i, j)
                return j, j + 1
            end
            j += 1
        end
    else
        while j <= n
            if leq(it.P, j, it.i)
                return j, j + 1
            end
            j += 1
        end
    end
    return nothing
end

struct BitRowIter{A<:AbstractVector{Bool}}
    row::A
end

Base.IteratorSize(::Type{BitRowIter}) = Base.HasLength()
Base.eltype(::Type{BitRowIter}) = Int
Base.length(it::BitRowIter) = count(it.row)
Base.iterate(it::BitRowIter, state::Int=1) = begin
    j = findnext(it.row, state)
    j === nothing ? nothing : (j, j + 1)
end

struct ProductIndexIter{N}
    cart::CartesianIndices{N,NTuple{N,UnitRange{Int}}}
    strides::NTuple{N,Int}
end

Base.IteratorSize(::Type{ProductIndexIter}) = Base.HasLength()
Base.eltype(::Type{ProductIndexIter}) = Int
Base.length(it::ProductIndexIter) = length(it.cart)
function Base.iterate(it::ProductIndexIter{N}, state...) where {N}
    nxt = iterate(it.cart, state...)
    nxt === nothing && return nothing
    I, st = nxt
    lin = 1
    @inbounds for k in 1:N
        lin += (I[k] - 1) * it.strides[k]
    end
    return lin, st
end

# =========================================
# Poset interface + finite poset and indicator sets
# =========================================

abstract type AbstractPoset end

const UPDOWN_CACHE_MODE = Ref(:auto)  # :auto | :always | :never
const UPDOWN_CACHE_THRESHOLD_FINITE = Ref(500_000)
const UPDOWN_CACHE_THRESHOLD_GENERIC = Ref(200_000)
const CHAIN_PARENT_DENSE_MIN_ENTRIES = Ref(4_096)
const CHAIN_PARENT_DENSE_MAX_ENTRIES_PER_THREAD = Ref(1_000_000)
const CHAIN_PARENT_DENSE_MAX_TOTAL_ENTRIES = Ref(8_000_000)

@inline _updown_cache_skip_auto(::AbstractPoset) = false
@inline _updown_cache_threshold(::AbstractPoset) = UPDOWN_CACHE_THRESHOLD_GENERIC[]

@inline function _should_cache_updown(P::AbstractPoset, n::Int)
    mode = UPDOWN_CACHE_MODE[]
    mode === :always && return true
    mode === :never && return false
    mode === :auto || error("UPDOWN cache mode must be one of :auto, :always, :never (got $(repr(mode))).")
    _updown_cache_skip_auto(P) && return false
    return n * n <= _updown_cache_threshold(P)
end

function _ensure_updown_cache!(P::AbstractPoset)
    hasproperty(P, :cache) || return nothing
    pc = getproperty(P, :cache)
    hasproperty(pc, :upsets) || return nothing
    hasproperty(pc, :downsets) || return nothing
    upsets = getproperty(pc, :upsets)
    downsets = getproperty(pc, :downsets)
    if upsets !== nothing && downsets !== nothing
        return (upsets, downsets)
    end
    n = nvertices(P)
    _should_cache_updown(P, n) || return nothing
    lock = hasproperty(pc, :lock) ? getproperty(pc, :lock) : nothing
    if lock !== nothing
        Base.lock(lock)
    end
    try
        upsets = getproperty(pc, :upsets)
        downsets = getproperty(pc, :downsets)
        if upsets === nothing || downsets === nothing
            up = Vector{Vector{Int}}(undef, n)
            down = Vector{Vector{Int}}(undef, n)
            @inbounds for i in 1:n
                up[i] = Int[]
                down[i] = Int[]
            end
            @inbounds for i in 1:n
                for j in 1:n
                    if leq(P, i, j)
                        push!(up[i], j)
                    end
                    if leq(P, j, i)
                        push!(down[i], j)
                    end
                end
            end
            setproperty!(pc, :upsets, up)
            setproperty!(pc, :downsets, down)
            upsets = up
            downsets = down
        end
        return (upsets, downsets)
    finally
        if lock !== nothing
            Base.unlock(lock)
        end
    end
end

function _cached_upset_indices(P::AbstractPoset, i::Int)
    cache = _ensure_updown_cache!(P)
    cache === nothing && return nothing
    return cache[1][i]
end

function _cached_downset_indices(P::AbstractPoset, i::Int)
    cache = _ensure_updown_cache!(P)
    cache === nothing && return nothing
    return cache[2][i]
end

"""
    CoverEdges

Lightweight cover-relation wrapper that supports:
- adjacency queries via `C[u,v]`,
- iteration over cover edges `(u,v)`,
- matrix recovery via `BitMatrix(C)` / `Matrix(C)`.
"""
struct CoverEdges
    mat::BitMatrix
    edges::Vector{Tuple{Int,Int}}
end

Base.size(C::CoverEdges) = size(C.mat)
Base.getindex(C::CoverEdges, i::Int) = C.edges[i]
Base.getindex(C::CoverEdges, i::Int, j::Int) = C.mat[i,j]
Base.length(C::CoverEdges) = length(C.edges)
Base.firstindex(C::CoverEdges) = firstindex(C.edges)
Base.lastindex(C::CoverEdges) = lastindex(C.edges)
Base.eachindex(C::CoverEdges) = eachindex(C.edges)
Base.eltype(::Type{CoverEdges}) = Tuple{Int,Int}
Base.IteratorSize(::Type{CoverEdges}) = Base.HasLength()
Base.iterate(C::CoverEdges, state::Int=1) =
    state > length(C.edges) ? nothing : (C.edges[state], state + 1)
Base.convert(::Type{BitMatrix}, C::CoverEdges) = C.mat
Base.convert(::Type{Matrix{Bool}}, C::CoverEdges) = Matrix(C.mat)
Base.BitMatrix(C::CoverEdges) = C.mat
Base.Matrix(C::CoverEdges) = Matrix(C.mat)
Base.findall(C::CoverEdges) = C.edges

mutable struct _ChainParentDenseMemo
    seen::BitVector
    vals::Vector{Int}
    n::Int
end

struct _PackedAdjacency
    ptr::Vector{Int}
    idx::Vector{Int}
end

@inline function _adj_bounds(adj::_PackedAdjacency, v::Int)
    return adj.ptr[v], adj.ptr[v + 1] - 1
end

struct _PackedIntSlice <: AbstractVector{Int}
    data::Vector{Int}
    lo::Int
    hi::Int
end

Base.IndexStyle(::Type{_PackedIntSlice}) = IndexLinear()
Base.firstindex(::_PackedIntSlice) = 1
@inline Base.length(s::_PackedIntSlice) = max(0, s.hi - s.lo + 1)
Base.lastindex(s::_PackedIntSlice) = length(s)
Base.size(s::_PackedIntSlice) = (length(s),)
Base.axes(s::_PackedIntSlice) = (Base.OneTo(length(s)),)
Base.isempty(s::_PackedIntSlice) = length(s) == 0

@inline function Base.getindex(s::_PackedIntSlice, i::Int)
    @boundscheck checkbounds(s, i)
    return @inbounds s.data[s.lo + i - 1]
end

@inline function Base.iterate(s::_PackedIntSlice, state::Int=1)
    state > length(s) && return nothing
    return s[state], state + 1
end

"""
    CoverCache(Q)

An internal cache for poset cover data and a hot-path memo used by `map_leq`.
This cache is stored lazily on each poset object (via `PosetCache`).

Thread-safety:

- `succs`, `preds`, and `C` are read-only once constructed.
- `chain_parent` / `chain_parent_dense` are per-thread hot-path memos for witness
  predecessor lookups. Writes are thread-local and do not require locks.
"""
struct CoverCache
    Q::AbstractPoset
    C::Union{BitMatrix,Nothing}
    succ_ptr::Vector{Int}
    succ_idx::Vector{Int}
    succ_pred_slot::Vector{Int}
    pred_ptr::Vector{Int}
    pred_idx::Vector{Int}
    pred_succ_slot::Vector{Int}
    undir::Union{Nothing,_PackedAdjacency}

    # Sparse fallback memo: chain_parent[tid][pairkey(a,d)] = chosen predecessor b.
    chain_parent::Vector{Dict{UInt64, Int}}
    # Dense memo for finite posets when n^2 is moderate; optional for memory control.
    chain_parent_dense::Union{Nothing,Vector{_ChainParentDenseMemo}}

    # Number of cover edges in Q (used to size edge-indexed stores).
    nedges::Int
end

@inline function _succs(cc::CoverCache, u::Int)
    lo = cc.succ_ptr[u]
    hi = cc.succ_ptr[u + 1] - 1
    return _PackedIntSlice(cc.succ_idx, lo, hi)
end

@inline function _preds(cc::CoverCache, v::Int)
    lo = cc.pred_ptr[v]
    hi = cc.pred_ptr[v + 1] - 1
    return _PackedIntSlice(cc.pred_idx, lo, hi)
end

@inline function _pred_slots_of_succ(cc::CoverCache, u::Int)
    lo = cc.succ_ptr[u]
    hi = cc.succ_ptr[u + 1] - 1
    return _PackedIntSlice(cc.succ_pred_slot, lo, hi)
end

@inline function _succ_slots_of_pred(cc::CoverCache, v::Int)
    lo = cc.pred_ptr[v]
    hi = cc.pred_ptr[v + 1] - 1
    return _PackedIntSlice(cc.pred_succ_slot, lo, hi)
end

mutable struct PosetCache
    cover_edges::Union{Nothing,CoverEdges}
    cover::Union{Nothing,CoverCache}
    lock::Base.ReentrantLock
    upsets::Union{Nothing,Vector{Vector{Int}}}
    downsets::Union{Nothing,Vector{Vector{Int}}}
    hom_route_choice::Dict{UInt64,Symbol}
    PosetCache() = new(nothing, nothing, Base.ReentrantLock(), nothing, nothing, Dict{UInt64,Symbol}())
end

function Base.propertynames(::CoverCache, private::Bool=false)
    base = (:Q, :C, :succ_ptr, :succ_idx, :succ_pred_slot,
            :pred_ptr, :pred_idx, :pred_succ_slot,
            :undir, :chain_parent, :chain_parent_dense, :nedges)
    return private ? base : base
end

# Internal lazy accessor for per-poset cover cache.
function _get_cover_cache(Q::AbstractPoset)
    if hasproperty(Q, :cache)
        pc = getproperty(Q, :cache)
        if pc isa PosetCache
            if pc.cover === nothing
                Base.lock(pc.lock)  # lock only on miss path
                try
                    if pc.cover === nothing
                        pc.cover = _build_cover_cache(Q)
                    end
                finally
                    Base.unlock(pc.lock)
                end
            end
            return pc.cover
        end
    end
    error("_get_cover_cache: poset type $(typeof(Q)) does not support caching")
end

"""
    build_cache!(Q; cover=true, updown=true)

Canonical public cache-build entrypoint for a poset. Use this before entering threaded
read-only loops to avoid lock contention and duplicate lazy-initialization work.
"""
function build_cache!(Q::AbstractPoset; cover::Bool=true, updown::Bool=true)
    cover && _get_cover_cache(Q)
    updown && _ensure_updown_cache!(Q)
    return Q
end

"""
    _clear_cover_cache!(Q)

Clear the cached cover data stored on a poset object.
"""
function _clear_cover_cache!(Q::AbstractPoset)
    if hasproperty(Q, :cache)
        pc = getproperty(Q, :cache)
        if pc isa PosetCache
            Base.lock(pc.lock)
            try
                pc.cover_edges = nothing
                pc.cover = nothing
                pc.upsets = nothing
                pc.downsets = nothing
                empty!(pc.hom_route_choice)
            finally
                Base.unlock(pc.lock)
            end
            return nothing
        end
    end
    error("_clear_cover_cache!: poset type $(typeof(Q)) does not support caching")
end

# Packs two Int32-ish values into a UInt64 for faster Dict keys.
# This is used both here and in the `CoverCache.chain_parent` hot-path memo.
@inline function _pairkey(u::Int, v::Int)::UInt64
    return (UInt64(u) << 32) | UInt64(v)
end

function _build_cover_cache(Q::AbstractPoset)
    # Cache enough to quickly traverse the cover graph and to build edge-indexed stores.
    Ce = cover_edges(Q)

    # BitMatrix adjacency for O(1) cover checks (only for FinitePoset).
    C = Q isa FinitePoset ? BitMatrix(Ce) : nothing

    nedges = length(Ce)
    n = nvertices(Q)
    outdeg = zeros(Int, n)
    indeg = zeros(Int, n)

    for (a, b) in Ce
        outdeg[a] += 1
        indeg[b] += 1
    end

    succ_ptr = Vector{Int}(undef, n + 1)
    pred_ptr = Vector{Int}(undef, n + 1)
    succ_ptr[1] = 1
    pred_ptr[1] = 1
    @inbounds for u in 1:n
        succ_ptr[u + 1] = succ_ptr[u] + outdeg[u]
        pred_ptr[u + 1] = pred_ptr[u] + indeg[u]
    end

    succ_idx = Vector{Int}(undef, nedges)
    pred_idx = Vector{Int}(undef, nedges)
    outk = copy(succ_ptr)
    ink = copy(pred_ptr)

    for (a, b) in Ce
        succ_idx[outk[a]] = b
        pred_idx[ink[b]] = a
        outk[a] += 1
        ink[b] += 1
    end

    @inbounds for u in 1:n
        slo, shi = succ_ptr[u], succ_ptr[u + 1] - 1
        plo, phi = pred_ptr[u], pred_ptr[u + 1] - 1
        slo <= shi && sort!(@view succ_idx[slo:shi])
        plo <= phi && sort!(@view pred_idx[plo:phi])
    end

    succ_pred_slot = Vector{Int}(undef, nedges)
    pred_succ_slot = Vector{Int}(undef, nedges)
    @inbounds for u in 1:n
        slo, shi = succ_ptr[u], succ_ptr[u + 1] - 1
        for sp in slo:shi
            v = succ_idx[sp]
            plo, phi = pred_ptr[v], pred_ptr[v + 1] - 1
            i = searchsortedfirst(@view(pred_idx[plo:phi]), u)
            predp = plo + i - 1
            succ_pred_slot[sp] = i
            pred_succ_slot[predp] = sp - slo + 1
        end
    end

    nt = max(1, Base.Threads.maxthreadid())
    chain_parent = [Dict{UInt64, Int}() for _ in 1:max(1, Base.Threads.maxthreadid())]
    chain_parent_dense = nothing
    dense_entries = n * n
    use_dense_parent = (Q isa FinitePoset) &&
                       (dense_entries >= CHAIN_PARENT_DENSE_MIN_ENTRIES[]) &&
                       (dense_entries <= CHAIN_PARENT_DENSE_MAX_ENTRIES_PER_THREAD[]) &&
                       (dense_entries * nt <= CHAIN_PARENT_DENSE_MAX_TOTAL_ENTRIES[])
    if use_dense_parent
        chain_parent_dense = [
            _ChainParentDenseMemo(falses(dense_entries), zeros(Int, dense_entries), n)
            for _ in 1:nt
        ]
    end

    undir_ptr = Vector{Int}(undef, n + 1)
    undir_ptr[1] = 1
    @inbounds for u in 1:n
        undir_ptr[u + 1] = undir_ptr[u] + outdeg[u] + indeg[u]
    end
    undir_idx = Vector{Int}(undef, undir_ptr[end] - 1)
    undir_k = copy(undir_ptr)
    @inbounds for u in 1:n
        for p in succ_ptr[u]:(succ_ptr[u + 1] - 1)
            undir_idx[undir_k[u]] = succ_idx[p]
            undir_k[u] += 1
        end
        for p in pred_ptr[u]:(pred_ptr[u + 1] - 1)
            undir_idx[undir_k[u]] = pred_idx[p]
            undir_k[u] += 1
        end
    end

    return CoverCache(Q, C, succ_ptr, succ_idx, succ_pred_slot,
                      pred_ptr, pred_idx, pred_succ_slot,
                      _PackedAdjacency(undir_ptr, undir_idx),
                      chain_parent, chain_parent_dense, nedges)
end

@inline function _chain_parent_dict(cc::CoverCache)::Dict{UInt64, Int}
    return cc.chain_parent[min(length(cc.chain_parent), max(1, Base.Threads.threadid()))]
end

@inline function _chain_parent_dense(cc::CoverCache)
    dense = cc.chain_parent_dense
    dense === nothing && return nothing
    return dense[min(length(dense), max(1, Base.Threads.threadid()))]
end

function _clear_chain_parent_cache!(cc::CoverCache)
    for d in cc.chain_parent
        empty!(d)
    end
    if cc.chain_parent_dense !== nothing
        for m in cc.chain_parent_dense
            fill!(m.seen, false)
        end
    end
    return nothing
end

@inline function _chosen_predecessor_slow(cc::CoverCache, a::Int, d::Int)
    lo, hi = cc.pred_ptr[d], cc.pred_ptr[d + 1] - 1
    @inbounds for p in lo:hi
        b = cc.pred_idx[p]
        (b != a && leq(cc.Q, a, b)) && return b
    end
    return a
end

function _chosen_predecessor(cc::CoverCache, a::Int, d::Int)
    dense = _chain_parent_dense(cc)
    if dense !== nothing
        idx = (a - 1) * dense.n + d
        @inbounds if dense.seen[idx]
            return dense.vals[idx]
        end
        b = _chosen_predecessor_slow(cc, a, d)
        @inbounds begin
            dense.vals[idx] = b
            dense.seen[idx] = true
        end
        return b
    end

    k = _pairkey(a, d)
    chain_parent = _chain_parent_dict(cc)
    b = get(chain_parent, k, 0)
    if b == 0
        b = _chosen_predecessor_slow(cc, a, d)
        chain_parent[k] = b
    end
    return b
end

"""
    nvertices(P) -> Int

Return the number of vertices in the finite poset `P`.
"""
nvertices(::AbstractPoset) = error("nvertices(P) is not implemented for $(typeof(P)).")

"""
    leq(P, i, j) -> Bool

Return `true` iff `i <= j` in the poset `P`.
"""
leq(::AbstractPoset, ::Int, ::Int) =
    error("leq(P, i, j) is not implemented for $(typeof(P)).")

"""
    leq_matrix(P) -> BitMatrix

Materialize the order matrix for `P`. This is a fallback and should be avoided
for large structured posets.
"""
function leq_matrix(P::AbstractPoset)
    n = nvertices(P)
    L = falses(n, n)
    @inbounds for i in 1:n, j in 1:n
        L[i, j] = leq(P, i, j)
    end
    return L
end

"""
    upset_indices(P, i)
    downset_indices(P, i)

Return an iterable of indices in the principal upset/downset of `i`.
Fallback implementations scan the whole poset.
"""
function upset_indices(P::AbstractPoset, i::Int)
    cached = _cached_upset_indices(P, i)
    cached === nothing || return IndicesView(cached)
    return PosetLeqIter(P, i, true)
end

function downset_indices(P::AbstractPoset, i::Int)
    cached = _cached_downset_indices(P, i)
    cached === nothing || return IndicesView(cached)
    return PosetLeqIter(P, i, false)
end

"""
    leq_row(P, i)
    leq_col(P, j)

Return indices in the i-th row / j-th column of the order relation.
Defaults to upset/downset indices.
"""
leq_row(P::AbstractPoset, i::Int) = upset_indices(P, i)
leq_col(P::AbstractPoset, j::Int) = downset_indices(P, j)

"""
    poset_equal(P, Q) -> Bool

Structural equality of finite posets, defaulting to order-matrix comparison.
"""
function poset_equal(P::AbstractPoset, Q::AbstractPoset)
    nvertices(P) == nvertices(Q) || return false
    return leq_matrix(P) == leq_matrix(Q)
end

"""
    poset_equal_opposite(P, Q) -> Bool

Return true iff `P` equals the opposite of `Q`.
"""
function poset_equal_opposite(P::AbstractPoset, Q::AbstractPoset)
    nvertices(P) == nvertices(Q) || return false
    return leq_matrix(P) == transpose(leq_matrix(Q))
end

"""
    FinitePoset(leq; check=true) -> FinitePoset
    FinitePoset(n, leq; check=true) -> FinitePoset

Construct a finite poset from an explicit Boolean order matrix.

# Mathematical meaning

`FinitePoset` represents a finite partially ordered set on vertices
`1, ..., n`.  The order is stored by a Boolean matrix `leq` with the convention

    leq[i, j] == true    if and only if    i <= j.

So rows represent principal upsets and columns represent principal downsets.

# Inputs

- `leq::AbstractMatrix{Bool}`: an `n x n` Boolean matrix encoding the order.
- `n::Int`: optional size check when the ambient size is already known.
- `check::Bool=true`: whether to validate that `leq` is a genuine partial order.

# Output

- A `FinitePoset` object with ambient vertex set `{1, ..., n}`.

# Invariants

When `check=true`, the constructor verifies that `leq` is:

- reflexive,
- antisymmetric,
- transitive.

The stored matrix is normalized to a `BitMatrix` and the object carries a lazy
`PosetCache` for cover and upset/downset data.

# Failure / contract behavior

- Throws if `leq` is not square.
- Throws if `n` does not match the matrix size.
- Throws if `check=true` and the matrix is not a partial order.

# Best practices

- Use `check=true` for hand-built or uncertain order matrices.
- Use `check=false` only for known-good programmatic constructions where the
  order axioms are already guaranteed; this skips validation and is a genuine
  performance feature.
- Prefer `principal_upset`, `principal_downset`, `upset_indices`, and
  `downset_indices` for semantic interaction with the poset rather than reading
  the raw order matrix in user code.

# Examples

```jldoctest
julia> using TamerOp

julia> const FF = TamerOp.FiniteFringe
Main.TamerOp.FiniteFringe

julia> P = FF.FinitePoset(Bool[1 1 1;
                              0 1 1;
                              0 0 1]);

julia> describe(P).nvertices
3

julia> FF.support(FF.principal_upset(P, 2))
[2, 3]
```
"""
struct FinitePoset <: AbstractPoset
    n::Int
    _leq::BitMatrix           # _leq[i,j] = true  iff i <= j
    cache::PosetCache
    function FinitePoset(leq::AbstractMatrix{Bool};
                         check::Bool=true)
        n1, n2 = size(leq)
        @assert n1 == n2 "leq must be square"

        # Normalize storage.
        L = leq isa BitMatrix ? leq : BitMatrix(leq)

        if check
            _validate_partial_order_matrix!(L)
        end

        new(n1, L, PosetCache())
    end
end

@inline function _fringe_describe(P::AbstractPoset)
    return (kind=Symbol(nameof(typeof(P))),
            nvertices=nvertices(P))
end

@inline function _fringe_describe(P::FinitePoset)
    return (kind=:finite_poset,
            nvertices=P.n,
            cover_cache_built=P.cache.cover !== nothing,
            updown_cache_built=P.cache.upsets !== nothing && P.cache.downsets !== nothing)
end

# Convenience constructor when you already know n.
# (This is also useful as a search/replace target in performance hot paths.)
FinitePoset(n::Int, leq::AbstractMatrix{Bool};
            check::Bool=true) = begin
    @assert size(leq, 1) == n && size(leq, 2) == n "leq must be an n x n Boolean matrix"
    FinitePoset(leq; check=check)
end

nvertices(P::FinitePoset) = P.n
leq(P::FinitePoset, i::Int, j::Int) = P._leq[i,j]
leq_matrix(P::FinitePoset) = P._leq
@inline _updown_cache_threshold(::FinitePoset) = UPDOWN_CACHE_THRESHOLD_FINITE[]

function upset_indices(P::FinitePoset, i::Int)
    cached = _cached_upset_indices(P, i)
    cached === nothing || return IndicesView(cached)
    row = @view P._leq[i, :]
    return BitRowIter(row)
end

function downset_indices(P::FinitePoset, i::Int)
    cached = _cached_downset_indices(P, i)
    cached === nothing || return IndicesView(cached)
    col = @view P._leq[:, i]
    return BitRowIter(col)
end

leq_row(P::FinitePoset, i::Int) = P._leq[i, :]
leq_col(P::FinitePoset, j::Int) = P._leq[:, j]

"""
    ProductOfChainsPoset(sizes)

Poset on a product of chains with sizes given by `sizes`.
Vertex indices use mixed radix ordering with the first axis varying fastest.
"""
struct ProductOfChainsPoset{N} <: AbstractPoset
    sizes::NTuple{N,Int}
    strides::NTuple{N,Int}
    cache::PosetCache
end

@inline function _fringe_describe(P::ProductOfChainsPoset)
    return (kind=:product_of_chains_poset,
            sizes=P.sizes,
            nvertices=nvertices(P))
end

function _poset_strides(sizes::NTuple{N,Int}) where {N}
    strides = Vector{Int}(undef, N)
    strides[1] = 1
    for i in 2:N
        strides[i] = strides[i-1] * sizes[i-1]
    end
    return ntuple(i -> strides[i], N)
end

ProductOfChainsPoset(sizes::NTuple{N,Int}) where {N} =
    ProductOfChainsPoset{N}(sizes, _poset_strides(sizes), PosetCache())

ProductOfChainsPoset(sizes::NTuple{N,Int}, strides::NTuple{N,Int}) where {N} =
    ProductOfChainsPoset{N}(sizes, strides, PosetCache())

ProductOfChainsPoset(sizes::AbstractVector{<:Integer}) =
    ProductOfChainsPoset(ntuple(i -> Int(sizes[i]), length(sizes)))
@inline _updown_cache_skip_auto(::ProductOfChainsPoset) = true

function nvertices(P::ProductOfChainsPoset)
    n = 1
    @inbounds for s in P.sizes
        n *= s
    end
    return n
end

@inline _coord_at(idx::Int, size::Int, stride::Int) =
    (div(idx - 1, stride) % size) + 1

@inline function _index_to_coords!(coords::Vector{Int}, idx::Int,
                                  sizes::NTuple{N,Int}, strides::NTuple{N,Int}) where {N}
    x = idx - 1
    @inbounds for i in 1:N
        coords[i] = div(x, strides[i]) % sizes[i] + 1
    end
    return coords
end

@inline function leq(P::ProductOfChainsPoset{N}, i::Int, j::Int) where {N}
    @inbounds for k in 1:N
        ci = _coord_at(i, P.sizes[k], P.strides[k])
        cj = _coord_at(j, P.sizes[k], P.strides[k])
        if ci > cj
            return false
        end
    end
    return true
end

function upset_indices(P::ProductOfChainsPoset{N}, i::Int) where {N}
    cached = _cached_upset_indices(P, i)
    cached === nothing || return IndicesView(cached)
    ranges = ntuple(k -> _coord_at(i, P.sizes[k], P.strides[k]):P.sizes[k], N)
    return ProductIndexIter(CartesianIndices(ranges), P.strides)
end

function downset_indices(P::ProductOfChainsPoset{N}, i::Int) where {N}
    cached = _cached_downset_indices(P, i)
    cached === nothing || return IndicesView(cached)
    ranges = ntuple(k -> 1:_coord_at(i, P.sizes[k], P.strides[k]), N)
    return ProductIndexIter(CartesianIndices(ranges), P.strides)
end

"""
    GridPoset(coords)

Axis-aligned grid poset with coordinate vectors per axis. Indices follow the
same mixed radix ordering as `ProductOfChainsPoset`.
"""
struct GridPoset{N,T} <: AbstractPoset
    coords::NTuple{N,Vector{T}}
    sizes::NTuple{N,Int}
    strides::NTuple{N,Int}
    cache::PosetCache
end

@inline function _fringe_describe(P::GridPoset)
    return (kind=:grid_poset,
            sizes=P.sizes,
            nvertices=nvertices(P))
end

function GridPoset(coords::NTuple{N,Vector{T}}) where {N,T}
    for i in 1:N
        axis = coords[i]
        for j in 2:length(axis)
            if axis[j] <= axis[j - 1]
                error("GridPoset: coords[$i] must be strictly increasing (no duplicates).")
            end
        end
    end
    sizes = ntuple(i -> length(coords[i]), N)
    return GridPoset{N,T}(coords, sizes, _poset_strides(sizes), PosetCache())
end

GridPoset(coords::AbstractVector{<:AbstractVector}) = GridPoset(ntuple(i -> Vector(coords[i]), length(coords)))
@inline _updown_cache_skip_auto(::GridPoset) = true

nvertices(P::GridPoset) = nvertices(ProductOfChainsPoset(P.sizes))
@inline function leq(P::GridPoset{N}, i::Int, j::Int) where {N}
    @inbounds for k in 1:N
        ci = _coord_at(i, P.sizes[k], P.strides[k])
        cj = _coord_at(j, P.sizes[k], P.strides[k])
        if ci > cj
            return false
        end
    end
    return true
end

upset_indices(P::GridPoset{N}, i::Int) where {N} =
    upset_indices(ProductOfChainsPoset(P.sizes, P.strides, P.cache), i)
downset_indices(P::GridPoset{N}, i::Int) where {N} =
    downset_indices(ProductOfChainsPoset(P.sizes, P.strides, P.cache), i)

"""
    ProductPoset(P1, P2)

Product of two finite posets with the first factor varying fastest.
"""
struct ProductPoset{P1<:AbstractPoset,P2<:AbstractPoset} <: AbstractPoset
    P1::P1
    P2::P2
    cache::PosetCache
end

@inline function _fringe_describe(P::ProductPoset)
    return (kind=:product_poset,
            nvertices=nvertices(P),
            left_nvertices=nvertices(P.P1),
            right_nvertices=nvertices(P.P2))
end

ProductPoset(P1::AbstractPoset, P2::AbstractPoset) =
    ProductPoset{typeof(P1),typeof(P2)}(P1, P2, PosetCache())

nvertices(P::ProductPoset) = nvertices(P.P1) * nvertices(P.P2)

@inline function _prod_indices(n1::Int, idx::Int)
    i1 = ((idx - 1) % n1) + 1
    i2 = div(idx - 1, n1) + 1
    return i1, i2
end

@inline function leq(P::ProductPoset, i::Int, j::Int)
    n1 = nvertices(P.P1)
    i1, i2 = _prod_indices(n1, i)
    j1, j2 = _prod_indices(n1, j)
    return leq(P.P1, i1, j1) && leq(P.P2, i2, j2)
end

"""
    RegionsPoset(Q, regions)

Structured poset on regions of a finite poset `Q`, with order defined by:
`A <= B` iff there exist `a in A`, `b in B` with `a <= b` in `Q` (Prop. 4.15).
"""
struct RegionsPoset{P<:AbstractPoset} <: AbstractPoset
    Q::P
    regions::Vector{Vector{Int}}
    n::Int
    cache::PosetCache
end

@inline function _fringe_describe(P::RegionsPoset)
    return (kind=:regions_poset,
            nvertices=P.n,
            base_nvertices=nvertices(P.Q),
            region_sizes=map(length, P.regions))
end

@inline function _region_size_summary(sizes::AbstractVector{<:Integer})
    isempty(sizes) && return "(min=0, max=0, avg=0.0)"
    return "(min=$(minimum(sizes)), max=$(maximum(sizes)), avg=$(round(sum(sizes) / length(sizes); digits=2)))"
end

function RegionsPoset(Q::AbstractPoset, regions::Vector{Vector{Int}})
    return RegionsPoset{typeof(Q)}(Q, regions, length(regions), PosetCache())
end

nvertices(P::RegionsPoset) = P.n

function leq(P::RegionsPoset, i::Int, j::Int)
    @inbounds for a in P.regions[i]
        for b in P.regions[j]
            if leq(P.Q, a, b)
                return true
            end
        end
    end
    return false
end


# ---- internal helpers ---------------------------------------------------------

# Return the smallest index j for which sub[j] is true and sup[j] is false.
# If sub is a subset of sup, return 0.
@inline function _subset_violation_index(sub::BitVector, sup::BitVector, n::Int)::Int
    @assert length(sub) == length(sup) == n
    sc = sub.chunks
    uc = sup.chunks
    nchunks = length(sc)

    # Mask off unused bits in the final word, to be robust to any garbage bits.
    r = n & 63
    lastmask = (r == 0) ? typemax(UInt64) : (UInt64(1) << r) - 1

    @inbounds for w in 1:nchunks
        diff = sc[w] & ~uc[w]
        if w == nchunks
            diff &= lastmask
        end
        if diff != 0
            tz = trailing_zeros(diff)
            j = (w - 1) * 64 + tz + 1
            return (j <= n) ? j : 0
        end
    end
    return 0
end

function _validate_partial_order_matrix!(L::BitMatrix)
    n1, n2 = size(L)
    @assert n1 == n2 "leq must be square"
    n = n1

    # Reflexive.
    @inbounds for i in 1:n
        if !L[i, i]
            error("FinitePoset: leq must be reflexive; missing leq[$i,$i] == true")
        end
    end

    # Antisymmetric: i<=j and j<=i implies i=j.
    @inbounds for i in 1:n
        for j in (i + 1):n
            if L[i, j] && L[j, i]
                error("FinitePoset: leq violates antisymmetry: leq[$i,$j] and leq[$j,$i] are both true")
            end
        end
    end

    # Transitive: i<=k and k<=j implies i<=j.
    #
    # Bitset formulation: whenever i<=k, the principal upset of k must be a subset
    # of the principal upset of i. We check subset failures via chunk-wise
    # operations to avoid scalar triple loops.
    rows = Vector{BitVector}(undef, n)
    @inbounds for i in 1:n
        rows[i] = L[i, :]
    end

    @inbounds for i in 1:n
        ri = rows[i]
        # NOTE: for BitVectors, the supported API is `findnext(bitvec, start)`.
        # The 3-argument form `findnext(bitvec, true, start)` is not defined for
        # BitArrays in Julia 1.12.
        k = findnext(ri, 1)
        while k !== nothing
            j = _subset_violation_index(rows[k], ri, n)
            if j != 0
                error("FinitePoset: leq violates transitivity at (i,k,j)=($i,$k,$j)")
            end
            k = findnext(ri, k + 1)
        end
    end

    return nothing
end

"""
    check_poset(P; throw=false) -> NamedTuple

Validate a hand-built finite-fringe poset object.

This helper checks structural invariants appropriate to the concrete poset
type:
- [`FinitePoset`](@ref): square relation matrix, matching `n`, and partial
  order laws;
- [`ProductOfChainsPoset`](@ref): positive axis sizes and stride consistency;
- [`GridPoset`](@ref): positive axis sizes, stride consistency, and strictly
  increasing coordinate axes;
- [`ProductPoset`](@ref): validity of both factors;
- [`RegionsPoset`](@ref): validity of the base poset and in-bounds region
  entries.

Set `throw=true` to raise an `ArgumentError` instead of returning an invalid
report. Pair the returned report with `finite_fringe_validation_summary(...)`
or `fringe_summary(P)` when working interactively.
"""
function check_poset(P::FinitePoset; throw::Bool=false)
    issues = String[]
    size(P._leq, 1) == P.n || push!(issues, "stored n=$(P.n) must match relation size $(size(P._leq, 1))")
    size(P._leq, 1) == size(P._leq, 2) || push!(issues, "relation matrix must be square")
    if isempty(issues)
        try
            _validate_partial_order_matrix!(copy(P._leq))
        catch err
            push!(issues, sprint(showerror, err))
        end
    end
    report = (kind=:finite_poset,
              valid=isempty(issues),
              nvertices=P.n,
              issues=issues)
    if throw && !report.valid
        Base.throw(ArgumentError("check_poset: invalid FinitePoset: " * join(report.issues, "; ")))
    end
    return report
end

function check_poset(P::ProductOfChainsPoset; throw::Bool=false)
    issues = String[]
    all(>(0), P.sizes) || push!(issues, "all chain sizes must be positive")
    P.strides == _poset_strides(P.sizes) || push!(issues, "stored strides are inconsistent with sizes")
    report = (kind=:product_of_chains_poset,
              valid=isempty(issues),
              sizes=P.sizes,
              nvertices=nvertices(P),
              issues=issues)
    if throw && !report.valid
        Base.throw(ArgumentError("check_poset: invalid ProductOfChainsPoset: " * join(report.issues, "; ")))
    end
    return report
end

function check_poset(P::GridPoset; throw::Bool=false)
    issues = String[]
    all(>(0), P.sizes) || push!(issues, "all grid axis sizes must be positive")
    P.strides == _poset_strides(P.sizes) || push!(issues, "stored strides are inconsistent with sizes")
    @inbounds for a in 1:length(P.coords)
        length(P.coords[a]) == P.sizes[a] || push!(issues, "axis $a length must equal stored size $(P.sizes[a])")
        axis = P.coords[a]
        for j in 2:length(axis)
            axis[j] > axis[j - 1] || begin
                push!(issues, "axis $a must be strictly increasing")
                break
            end
        end
    end
    report = (kind=:grid_poset,
              valid=isempty(issues),
              sizes=P.sizes,
              nvertices=nvertices(P),
              issues=issues)
    if throw && !report.valid
        Base.throw(ArgumentError("check_poset: invalid GridPoset: " * join(report.issues, "; ")))
    end
    return report
end

function check_poset(P::ProductPoset; throw::Bool=false)
    left = check_poset(P.P1)
    right = check_poset(P.P2)
    issues = String[]
    left.valid || append!(issues, "left factor: " .* left.issues)
    right.valid || append!(issues, "right factor: " .* right.issues)
    report = (kind=:product_poset,
              valid=isempty(issues),
              nvertices=nvertices(P),
              issues=issues)
    if throw && !report.valid
        Base.throw(ArgumentError("check_poset: invalid ProductPoset: " * join(report.issues, "; ")))
    end
    return report
end

function check_poset(P::RegionsPoset; throw::Bool=false)
    base = check_poset(P.Q)
    issues = String[]
    base.valid || append!(issues, "base poset: " .* base.issues)
    length(P.regions) == P.n || push!(issues, "stored n=$(P.n) must equal number of regions $(length(P.regions))")
    nQ = nvertices(P.Q)
    @inbounds for (k, reg) in enumerate(P.regions)
        for v in reg
            1 <= v <= nQ || begin
                push!(issues, "region $k contains out-of-bounds vertex $v for base poset with $nQ vertices")
                break
            end
        end
    end
    report = (kind=:regions_poset,
              valid=isempty(issues),
              nvertices=P.n,
              base_nvertices=nQ,
              issues=issues)
    if throw && !report.valid
        Base.throw(ArgumentError("check_poset: invalid RegionsPoset: " * join(report.issues, "; ")))
    end
    return report
end

function Base.show(io::IO, P::FinitePoset)
    d = _fringe_describe(P)
    print(io, "FinitePoset(nvertices=", d.nvertices, ")")
end

function Base.show(io::IO, ::MIME"text/plain", P::FinitePoset)
    d = _fringe_describe(P)
    print(io, "FinitePoset\n  nvertices: ", d.nvertices,
          "\n  cover_cache_built: ", d.cover_cache_built,
          "\n  updown_cache_built: ", d.updown_cache_built)
end

function Base.show(io::IO, P::ProductOfChainsPoset)
    d = _fringe_describe(P)
    print(io, "ProductOfChainsPoset(axes=", repr(d.sizes),
          ", nvertices=", d.nvertices, ")")
end

function Base.show(io::IO, ::MIME"text/plain", P::ProductOfChainsPoset)
    d = _fringe_describe(P)
    print(io, "ProductOfChainsPoset\n  axes: ", repr(d.sizes),
          "\n  nvertices: ", d.nvertices,
          "\n  model: finite product of chains")
end

function Base.show(io::IO, P::GridPoset)
    d = _fringe_describe(P)
    print(io, "GridPoset(axes=", repr(d.sizes),
          ", nvertices=", d.nvertices, ")")
end

function Base.show(io::IO, ::MIME"text/plain", P::GridPoset)
    d = _fringe_describe(P)
    print(io, "GridPoset\n  axes: ", repr(d.sizes),
          "\n  nvertices: ", d.nvertices,
          "\n  model: axis-aligned coordinate grid")
end

function Base.show(io::IO, P::ProductPoset)
    d = _fringe_describe(P)
    print(io, "ProductPoset(nvertices=", d.nvertices,
          ", factors=(", d.left_nvertices, ", ", d.right_nvertices, "))")
end

function Base.show(io::IO, ::MIME"text/plain", P::ProductPoset)
    d = _fringe_describe(P)
    print(io, "ProductPoset\n  nvertices: ", d.nvertices,
          "\n  left_nvertices: ", d.left_nvertices,
          "\n  right_nvertices: ", d.right_nvertices,
          "\n  model: categorical product of finite posets")
end

function Base.show(io::IO, P::RegionsPoset)
    d = _fringe_describe(P)
    print(io, "RegionsPoset(nregions=", d.nvertices,
          ", base_nvertices=", d.base_nvertices,
          ", region_sizes=", _region_size_summary(d.region_sizes), ")")
end

function Base.show(io::IO, ::MIME"text/plain", P::RegionsPoset)
    d = _fringe_describe(P)
    print(io, "RegionsPoset\n  nregions: ", d.nvertices,
          "\n  base_nvertices: ", d.base_nvertices,
          "\n  region_sizes: ", _region_size_summary(d.region_sizes),
          "\n  model: region poset over a finite base poset")
end

# Cover-edge wrappers and constructor specializations are defined near the
# poset-cache declarations so cache storage can be strongly typed.


# BitVector helpers (chunk-level) for allocation-free set operations.
@inline function _tailmask(nbits::Int)::UInt64
    r = nbits & 63
    return (r == 0) ? typemax(UInt64) : (UInt64(1) << r) - 1
end

@inline function _or_chunks!(dest::BitVector, src::BitVector)
    dc = dest.chunks
    sc = src.chunks
    nchunks = length(dc)
    @inbounds for w in 1:nchunks
        dc[w] |= sc[w]
    end
    # Ensure unused tail bits are always zeroed.
    dc[end] &= _tailmask(length(dest))
    return dest
end

@inline function _andnot_chunks!(dest::BitVector, mask::BitVector)
    dc = dest.chunks
    mc = mask.chunks
    nchunks = length(dc)
    @inbounds for w in 1:nchunks
        dc[w] &= ~mc[w]
    end
    # Ensure unused tail bits are always zeroed.
    dc[end] &= _tailmask(length(dest))
    return dest
end

@inline function _foreach_setbit(f::F, mask::BitVector) where {F<:Function}
    n = length(mask)
    nchunks = length(mask.chunks)
    nchunks == 0 && return nothing
    lastmask = _tailmask(n)
    @inbounds for w in 1:nchunks
        bits = mask.chunks[w]
        w == nchunks && (bits &= lastmask)
        while bits != 0
            tz = trailing_zeros(bits)
            f(((w - 1) << 6) + tz + 1)
            bits &= bits - UInt64(1)
        end
    end
    return nothing
end

function _compute_cover_edges_bitset(L::BitMatrix)::CoverEdges
    n = size(L, 1)

    # Copy the rows of the order matrix as BitVectors so we can do chunk-wise
    # OR/AND operations. This is the key speedup over scalar triple loops.
    rows = Vector{BitVector}(undef, n)
    @inbounds for i in 1:n
        rows[i] = L[i, :]
    end

    mat = falses(n, n)
    edges = Tuple{Int,Int}[]

    # Workspace reused across i.
    upper = falses(n)      # strict upper set of i, then the set of covers
    redundant = falses(n)  # elements proved non-minimal in upper

    @inbounds for i in 1:n
        copyto!(upper, rows[i])
        upper[i] = false
        fill!(redundant, false)

        # Mark all non-minimal elements of upper(i).
        #
        # For each k in upper(i), any element j > k is not minimal in upper(i),
        # hence i is not covered by j.
        k = findnext(upper, 1)
        while k !== nothing
            # We want strict_upper(k), i.e. exclude k itself. Since rows[k] is
            # reflexive, we OR it in and then restore redundant[k].
            prev = redundant[k]
            _or_chunks!(redundant, rows[k])
            redundant[k] = prev
            k = findnext(upper, k + 1)
        end

        # covers(i) = upper(i) \ redundant
        _andnot_chunks!(upper, redundant)

        j = findnext(upper, 1)
        while j !== nothing
            mat[i, j] = true
            push!(edges, (i, j))
            j = findnext(upper, j + 1)
        end
    end

    return CoverEdges(mat, edges)
end

function _set_cover_edges_cache!(P::AbstractPoset, C::CoverEdges)
    hasproperty(P, :cache) || return C
    pc = getproperty(P, :cache)
    pc isa PosetCache || return C
    Base.lock(pc.lock)
    try
        pc.cover_edges = C
    finally
        Base.unlock(pc.lock)
    end
    return C
end

function _cover_edges_cached_or_build!(builder::Function, P::AbstractPoset, cached::Bool)
    cached || return builder()
    hasproperty(P, :cache) || return builder()
    pc = getproperty(P, :cache)
    pc isa PosetCache || return builder()

    C = pc.cover_edges
    C === nothing || return C

    Base.lock(pc.lock)  # lock only on miss path
    try
        C = pc.cover_edges
        if C === nothing
            C = builder()
            pc.cover_edges = C
        end
        return C
    finally
        Base.unlock(pc.lock)
    end
end

"""
    cover_edges(P; cached=true)

Return the cover relation (Hasse diagram) of the finite poset `P`.

The return value is a `CoverEdges` object `C` supporting:

  * `C[i,j]`          (adjacency query),
  * iteration         (`for (i,j) in C`),
  * `findall(C)`      (edge list),
  * `BitMatrix(C)`    (adjacency matrix).

For performance, the result is cached per `FinitePoset` instance by default.
Pass `cached=false` to force recomputation.
"""
function cover_edges(P::FinitePoset;
                     cached::Bool=true)
    return _cover_edges_cached_or_build!(P, cached) do
        _compute_cover_edges_bitset(P._leq)
    end
end

function _cover_edges_from_edges(n::Int, edges::Vector{Tuple{Int,Int}})
    mat = falses(n, n)
    @inbounds for (i, j) in edges
        mat[i, j] = true
    end
    return CoverEdges(mat, edges)
end

function cover_edges(P::AbstractPoset;
                     cached::Bool=true)
    return _cover_edges_cached_or_build!(P, cached) do
        L = leq_matrix(P)
        _compute_cover_edges_bitset(L isa BitMatrix ? L : BitMatrix(L))
    end
end

function cover_edges(P::ProductOfChainsPoset{N};
                     cached::Bool=true) where {N}
    return _cover_edges_cached_or_build!(P, cached) do
        n = nvertices(P)
        edges = Tuple{Int,Int}[]
        coords = Vector{Int}(undef, N)
        @inbounds for idx in 1:n
            _index_to_coords!(coords, idx, P.sizes, P.strides)
            for k in 1:N
                if coords[k] < P.sizes[k]
                    coords[k] += 1
                    lin = 1
                    for t in 1:N
                        lin += (coords[t] - 1) * P.strides[t]
                    end
                    push!(edges, (idx, lin))
                    coords[k] -= 1
                end
            end
        end
        _cover_edges_from_edges(n, edges)
    end
end

function cover_edges(P::GridPoset{N};
                     cached::Bool=true) where {N}
    return _cover_edges_cached_or_build!(P, cached) do
        cover_edges(ProductOfChainsPoset(P.sizes, P.strides, P.cache); cached=true)
    end
end

function cover_edges(P::ProductPoset;
                     cached::Bool=true)
    return _cover_edges_cached_or_build!(P, cached) do
        n1 = nvertices(P.P1)
        n2 = nvertices(P.P2)
        n = n1 * n2
        edges = Tuple{Int,Int}[]
        C1 = cover_edges(P.P1; cached=cached)
        C2 = cover_edges(P.P2; cached=cached)
        @inbounds for (i1, j1) in C1
            for i2 in 1:n2
                src = i1 + (i2 - 1) * n1
                dst = j1 + (i2 - 1) * n1
                push!(edges, (src, dst))
            end
        end
        @inbounds for (i2, j2) in C2
            for i1 in 1:n1
                src = i1 + (i2 - 1) * n1
                dst = i1 + (j2 - 1) * n1
                push!(edges, (src, dst))
            end
        end
        _cover_edges_from_edges(n, edges)
    end
end
