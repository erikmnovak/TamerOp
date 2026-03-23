# =============================================================================
# rank_api.jl
#
# Shared basic rank-query API used across invariant families.
#
# Owns:
# - rank_map entrypoints on modules/fringe modules
#
# Does not own:
# - higher invariant families or slice/signed-measure algorithms
# =============================================================================

# Call path
# - `PModule` direct query      -> `map_leq` or `_map_leq_cached` -> `FieldLinAlg.rank`
# - `FringeModule` direct query -> `pmodule_from_fringe`          -> same path
# - encoded query               -> `locate`                       -> same path

"""
    rank_map(M, a, b; cache=nothing, memo=nothing) -> Int

Return the rank of the structure map `M(a <= b)` for comparable vertex labels
`a <= b` in a finite poset.

This direct-poset overload is the shared low-level rank-query surface reused by
multiple invariant owners.

Inputs:
- `M::PModule{K}` is the finite-poset module being queried,
- `a`, `b` are vertex indices in the underlying ambient poset,
- the `FringeModule` overload first converts with `pmodule_from_fringe(H)` and
  then follows this same path.

Keyword arguments:
- `cache`: optional cover-cache object used by `map_leq`,
- `memo`: optional memoization cache for structure maps; when present, the
  query uses `_map_leq_cached(...)` instead of recomputing maps via `map_leq`.

Notes for performance:
- repeated `FringeModule` queries should usually convert once with
  `pmodule_from_fringe(H)`,
- repeated direct queries over the same finite module can reuse `memo`.
"""
function rank_map(M::PModule{K}, a::Int, b::Int; cache=nothing, memo=nothing)::Int where {K}
    Q = M.Q
    (1 <= a <= nvertices(Q)) || error("rank_map: a out of range")
    (1 <= b <= nvertices(Q)) || error("rank_map: b out of range")
    leq(Q, a, b) || error("rank_map: a and b are not comparable (need a <= b)")

    if a == b
        return M.dims[a]
    end

    # Use shared memoization if provided; otherwise fall back to map_leq.
    if memo !== nothing
        cc = cache === nothing ? _get_cover_cache(Q) : cache
        A = _map_leq_cached(M, a, b, cc, memo)
        return FieldLinAlg.rank(M.field, A)
    end

    A = map_leq(M, a, b; cache=cache)
    return FieldLinAlg.rank(M.field, A)
end

function rank_map(H::FringeModule{K}, a::Int, b::Int; kwargs...)::Int where {K}
    Mp = pmodule_from_fringe(H)
    return rank_map(Mp, a, b; kwargs...)
end

"""
    rank_map(M, pi, x, y, opts::InvariantOptions) -> Int

Compute `rank_map(M, a, b)` after locating points `x` and `y` in an encoding.

This encoded-point overload resolves:
- `a = locate(pi, x)`
- `b = locate(pi, y)`

and then delegates to the direct-poset overload above.

Strictness semantics:
- if `opts.strict === nothing`, the canonical default is `strict=true`,
- if `strict=true` and either point maps to `0`, this throws,
- if `strict=false` and either point maps to `0`, the query returns `0`,
- any `cache` / `memo` keyword is then forwarded to the direct-poset query.
"""
function rank_map(M::PModule{K}, pi, x, y, opts::InvariantOptions;
                  cache = nothing, memo = nothing)::Int where {K}
    strict0 = opts.strict === nothing ? true : opts.strict

    a = locate(pi, x)
    b = locate(pi, y)

    if (a == 0 || b == 0)
        strict0 && error("rank_map: locate(pi, x) or locate(pi, y) returned 0 (unknown region)")
        return 0
    end

    return rank_map(M, a, b; cache = cache, memo = memo)
end

function rank_map(H::FringeModule{K}, pi, x, y, opts::InvariantOptions;
                  cache = nothing, memo = nothing)::Int where {K}
    Mp = pmodule_from_fringe(H)
    return rank_map(Mp, pi, x, y, opts; cache = cache, memo = memo)
end
