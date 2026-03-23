# This fragment owns the basic rank/Hilbert invariant surface for TamerOp.Invariants.
# It defines the Hilbert-function alias, box normalization, rank invariant/query,
# restricted Hilbert function helpers, and Hilbert-distance computations. It does
# not own higher-level module summaries, support geometry, or moved sibling
# invariant families. Later invariant fragments depend on `HilbertFunction` and
# `_coerce_box` defined here.

# A "Hilbert function" in this codebase is the vector of fiber dimensions per region.
# Keep it as a type alias (not a new struct) to make dispatch readable.
const HilbertFunction = AbstractVector{<:Integer}


# Convert "box-like" input to a normalized (lo, hi) tuple of Float64 vectors.
@inline function _coerce_box(box)
    box === nothing && return nothing
    return _normalize_box(box)
end



# ----- Rank invariant ----------------------------------------------------------


"""
    rank_invariant(M::PModule{K}; store_zeros=false, threads=(Threads.nthreads() > 1)) -> RankInvariantResult

Compute the rank invariant of a module `M`, returning a typed
`RankInvariantResult` that remains dictionary-like.

The stored entries map `(a, b)` with `a <= b` to the rank of the structure map
`M(a <= b)`.

Keyword arguments:

- `store_zeros`: if `true`, include all comparable pairs `(a, b)` including rank 0.
  If `false`, store only positive ranks (sparser and usually faster).
- `threads`: if `true` and Julia has more than one thread, parallelize over the
  outer vertex index `a`. This is safe: `CoverCache` is thread-safe and each
  thread uses its own map memo and output dictionary.
"""
function rank_invariant(
    M::PModule{K},
    opts::InvariantOptions;
    store_zeros::Bool = false
) where {K}
    threads = _default_threads(opts.threads)
    Q = M.Q
    # Two-phase threading: build caches once, then threaded loops read-only.
    build_cache!(Q; cover=true, updown=true)
    cc = _get_cover_cache(Q)
    n = nvertices(Q)
    use_array_memo = _use_array_memo(n)

    if threads && Threads.nthreads() > 1
        vals = zeros(Int, n * n)
        filled = falses(n * n)
        nT = Threads.nthreads()
        memo_by_thread = use_array_memo ?
            [_new_array_memo(K, n) for _ in 1:nT] :
            [Dict{Tuple{Int, Int}, AbstractMatrix{K}}() for _ in 1:nT]

        Threads.@threads for a in 1:nvertices(Q)
            tid = Threads.threadid()
            memo = memo_by_thread[tid]
            for b in upset_indices(Q, a)
                r = rank_map(M, a, b; cache = cc, memo = memo)
                if store_zeros || r > 0
                    idx = _memo_index(n, a, b)
                    vals[idx] = r
                    filled[idx] = true
                end
            end
        end
        ranks = Dict{Tuple{Int, Int}, Int}()
        sizehint!(ranks, count(filled))
        @inbounds for idx in eachindex(filled)
            filled[idx] || continue
            a = Int(div(idx - 1, n)) + 1
            b = ((idx - 1) % n) + 1
            ranks[(a, b)] = vals[idx]
        end
        return RankInvariantResult(Q, ranks, store_zeros)
    end

    vals = zeros(Int, n * n)
    filled = falses(n * n)

    memo = use_array_memo ? _new_array_memo(K, n) : Dict{Tuple{Int, Int}, AbstractMatrix{K}}()
    for a in 1:nvertices(Q)
        for b in upset_indices(Q, a)
            r = rank_map(M, a, b; cache = cc, memo = memo)
            if store_zeros || r > 0
                idx = _memo_index(n, a, b)
                vals[idx] = r
                filled[idx] = true
            end
        end
    end
    ranks = Dict{Tuple{Int, Int}, Int}()
    sizehint!(ranks, count(filled))
    @inbounds for idx in eachindex(filled)
        filled[idx] || continue
        a = Int(div(idx - 1, n)) + 1
        b = ((idx - 1) % n) + 1
        ranks[(a, b)] = vals[idx]
    end
    return RankInvariantResult(Q, ranks, store_zeros)
end

function rank_invariant(H::FringeModule{K}, opts::InvariantOptions; store_zeros::Bool = false) where {K}
    Mp = pmodule_from_fringe(H)
    return rank_invariant(Mp, opts; store_zeros = store_zeros)
end


#--------------------------------------------------------------------
# Signed barcodes / rectangle measures (Mobius inversion of rank).
#--------------------------------------------------------------------

"""
    rank_query(M, pi, x, y, opts::InvariantOptions; rq_cache=nothing, cache=nothing, memo=nothing) -> Int

Fast rank-query helper for repeated rank-map evaluations.

- For `ZnEncodingMap`, this reuses `RankQueryCache` to memoize both `locate(pi, x)` and
  `(a, b) -> rank_map(M, a, b)` values.
- For other encoding maps, this falls back to `rank_map(M, pi, x, y, opts; ...)`.

Unknown points follow `opts.strict` semantics:
- strict (default): throw if `locate` returns 0,
- non-strict: return 0.
"""
function rank_query(M::PModule{K},
                    pi::ZnEncodingMap,
                    x,
                    y,
                    opts::InvariantOptions;
                    rq_cache::Union{Nothing,RankQueryCache}=nothing,
                    cache=nothing,
                    memo=nothing)::Int where {K}
    # Do not build a RankQueryCache for one-off queries.
    # The cache pays off only when the caller reuses it across repeated
    # encoded rank queries on the same Zn encoding.
    if rq_cache === nothing
        return rank_map(M, pi, x, y, opts; cache=cache, memo=memo)
    end

    rc = _resolve_rank_query_cache(pi, rq_cache)
    xt = _rank_query_point_tuple(x, rc.n)
    yt = _rank_query_point_tuple(y, rc.n)

    a = _rank_query_locate!(rc, xt)
    b = _rank_query_locate!(rc, yt)

    strict0 = opts.strict === nothing ? true : opts.strict
    if a == 0 || b == 0
        strict0 && error("rank_query: locate(pi, x) or locate(pi, y) returned 0 (unknown region)")
        return 0
    end

    return _rank_cache_get!(rc, a, b) do
        rank_map(M, a, b; cache=cache, memo=memo)
    end
end

function rank_query(M::PModule{K},
                    pi::CompiledEncoding{<:ZnEncodingMap},
                    x,
                    y,
                    opts::InvariantOptions;
                    kwargs...)::Int where {K}
    return rank_query(M, pi.pi, x, y, opts; kwargs...)
end

function rank_query(M::PModule{K},
                    pi::ZnEncodingMap,
                    a::Int,
                    b::Int;
                    rq_cache::Union{Nothing,RankQueryCache}=nothing,
                    cache=nothing,
                    memo=nothing)::Int where {K}
    rq_cache === nothing && return rank_map(M, a, b; cache=cache, memo=memo)
    rc = _resolve_rank_query_cache(pi, rq_cache)
    return _rank_cache_get!(rc, a, b) do
        rank_map(M, a, b; cache=cache, memo=memo)
    end
end

function rank_query(M::PModule{K},
                    pi::CompiledEncoding{<:ZnEncodingMap},
                    a::Int,
                    b::Int;
                    kwargs...)::Int where {K}
    return rank_query(M, pi.pi, a, b; kwargs...)
end

function rank_query(M::PModule{K},
                    pi::ZnEncodingMap,
                    a::Int,
                    b::Int,
                    opts::InvariantOptions;
                    kwargs...)::Int where {K}
    return rank_query(M, pi, a, b; kwargs...)
end

function rank_query(M::PModule{K},
                    pi::CompiledEncoding{<:ZnEncodingMap},
                    a::Int,
                    b::Int,
                    opts::InvariantOptions;
                    kwargs...)::Int where {K}
    return rank_query(M, pi.pi, a, b; kwargs...)
end

function rank_query(M::PModule{K}, pi, x, y, opts::InvariantOptions;
                    cache=nothing, memo=nothing)::Int where {K}
    return rank_map(M, pi, x, y, opts; cache=cache, memo=memo)
end

function rank_query(H::FringeModule{K}, pi, x, y, opts::InvariantOptions; kwargs...)::Int where {K}
    Mp = pmodule_from_fringe(H)
    return rank_query(Mp, pi, x, y, opts; kwargs...)
end

function rank_query(H::FringeModule{K}, pi::ZnEncodingMap, a::Int, b::Int; kwargs...)::Int where {K}
    Mp = pmodule_from_fringe(H)
    return rank_query(Mp, pi, a, b; kwargs...)
end

function rank_query(H::FringeModule{K}, pi::CompiledEncoding{<:ZnEncodingMap}, a::Int, b::Int; kwargs...)::Int where {K}
    Mp = pmodule_from_fringe(H)
    return rank_query(Mp, pi, a, b; kwargs...)
end

function rank_query(H::FringeModule{K}, pi::ZnEncodingMap, a::Int, b::Int,
                    opts::InvariantOptions; kwargs...)::Int where {K}
    Mp = pmodule_from_fringe(H)
    return rank_query(Mp, pi, a, b; kwargs...)
end

function rank_query(H::FringeModule{K}, pi::CompiledEncoding{<:ZnEncodingMap}, a::Int, b::Int,
                    opts::InvariantOptions; kwargs...)::Int where {K}
    Mp = pmodule_from_fringe(H)
    return rank_query(Mp, pi, a, b; kwargs...)
end

# ----- Restricted Hilbert function --------------------------------------------

"""
    restricted_hilbert(M) -> Vector{Int}

Restricted Hilbert function on a finite poset: the dimension surface
`q |-> dim(M_q)` represented as a vector indexed by poset vertices.

Methods:
- `restricted_hilbert(M::PModule{K})` returns `copy(M.dims)`
- `restricted_hilbert(H::FringeModule{K})` computes fiber dimensions via
  `fiber_dimension(H, q)` for each vertex
"""
restricted_hilbert(M::PModule{K}) where {K} = copy(M.dims)

function restricted_hilbert(H::FringeModule{K}) where {K}
    return [fiber_dimension(H, q) for q in 1:nvertices(H.P)]
end


"""
    restricted_hilbert(M, pi, x; strict=true) -> Int

Evaluate the restricted Hilbert function at a point `x` in the original domain,
by first locating its region via `locate(pi, x)`.

If `strict=true` and `locate` returns 0, an error is thrown.
If `strict=false`, unknown regions return 0.
"""
function restricted_hilbert(M::PModule{K}, pi, x, opts::InvariantOptions)::Int where {K}
    strict0 = opts.strict === nothing ? true : opts.strict

    p = locate(pi, x)
    if p == 0
        strict0 && error("restricted_hilbert: locate(pi, x) returned 0 (unknown region)")
        return 0
    end

    (1 <= p <= length(M.dims)) ||
        error("restricted_hilbert: locate returned out-of-range index")

    return M.dims[p]
end

function restricted_hilbert(H::FringeModule{K}, pi, x, opts::InvariantOptions)::Int where {K}
    Mp = pmodule_from_fringe(H)
    return restricted_hilbert(Mp, pi, x, opts)
end

"""
    restricted_hilbert(dims::AbstractVector{<:Integer}) -> Vector{Int}

Convenience overload: interpret `dims` as an already computed restricted Hilbert
function (dimension per region). This makes it easy to use the "module-level"
summary/statistics functions without constructing a module object.
"""
function restricted_hilbert(dims::AbstractVector{<:Integer})
    return [Int(d) for d in dims]
end


# ----- Distances between restricted Hilbert functions --------------------------

function _normalize_weights(n::Int, weights)
    if weights === nothing
        return nothing
    elseif weights isa AbstractVector
        length(weights) == n || error("weights must have length $n")
        return weights
    elseif weights isa AbstractDict
        w = ones(Float64, n)
        for (k, v) in weights
            (1 <= k <= n) || error("weight key out of range: $k")
            w[Int(k)] = float(v)
        end
        return w
    else
        error("weights must be nothing, a vector, or a dictionary")
    end
end


"""
    hilbert_distance(M, N; norm=:L1, weights=nothing) -> Real

Compute a fast distance between two modules based on their restricted Hilbert
functions (dimension surfaces) on a fixed finite encoding.

Inputs:
* `M`, `N`: `PModule{K}` or `FringeModule{K}` on the same finite poset.
* `norm`: one of `:L1`, `:L2`, `:Linf`.
* `weights`: optional per-region weights.

If `weights` is provided, it should be either:
* a vector `w` of length `nvertices(P)`, or
* a dictionary mapping region indices to weights (others default to 1).

The weighted norms are computed as:
* `L1`: sum_i w[i] * abs(hM[i] - hN[i])
* `L2`: sqrt( sum_i w[i] * (hM[i] - hN[i])^2 )
* `Linf`: max_i w[i] * abs(hM[i] - hN[i])

To weight by region size for a particular encoding map `pi`, use
`weights = region_weights(pi; box=...)` when available.
"""
function hilbert_distance(M, N; norm::Symbol=:L1, weights=nothing)
    hM = restricted_hilbert(M)
    hN = restricted_hilbert(N)
    length(hM) == length(hN) || error("hilbert_distance: modules must live on the same finite poset")
    n = length(hM)

    w = _normalize_weights(n, weights)

    if norm == :L1
        if w === nothing
            s = 0
            for i in 1:n
                s += abs(hM[i] - hN[i])
            end
            return s
        end
        s = zero(promote_type(eltype(w), Int))
        for i in 1:n
            s += w[i] * abs(hM[i] - hN[i])
        end
        return s
    elseif norm == :L2
        acc = 0.0
        if w === nothing
            for i in 1:n
                d = float(hM[i] - hN[i])
                acc += d * d
            end
        else
            for i in 1:n
                d = float(hM[i] - hN[i])
                acc += float(w[i]) * d * d
            end
        end
        return sqrt(acc)
    elseif norm == :Linf
        if w === nothing
            m = 0
            for i in 1:n
                m = max(m, abs(hM[i] - hN[i]))
            end
            return m
        end
        m = 0.0
        for i in 1:n
            m = max(m, float(w[i]) * float(abs(hM[i] - hN[i])))
        end
        return m
    else
        error("hilbert_distance: norm must be :L1, :L2, or :Linf")
    end
end


"""
    hilbert_distance(M, N, pi, opts::InvariantOptions; norm=:L1, kwargs...)

Convenience wrapper: compute region weights from the encoding map and then call
`hilbert_distance(M, N; norm=..., weights=...)`.

This method uses:
- opts.box (and the special value :auto) to select a window.
- opts.strict (if not `nothing`) to control how `region_weights` handles points
  not found in the encoding.

Remaining keywords:
- norm (passed to `hilbert_distance(M, N; ...)`)

Any remaining keyword arguments are forwarded to `region_weights(pi; ...)`.
"""
function hilbert_distance(M, N, pi, opts::InvariantOptions; norm::Symbol=:L1, kwargs...)
    bb = _resolve_box(pi, opts.box)

    w = if opts.strict === nothing
        region_weights(pi; box=bb, kwargs...)
    else
        region_weights(pi; box=bb, strict=opts.strict, kwargs...)
    end

    return hilbert_distance(M, N; norm=norm, weights=w)
end
