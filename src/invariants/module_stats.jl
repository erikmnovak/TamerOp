# This fragment owns global module-size summaries built from region weights and
# restricted Hilbert vectors for TamerOp.Invariants. It includes integrated
# mass, dimension summaries, entropies, and interface measures. It does not own
# lower-level rank/Hilbert primitives, support geometry, or moved sibling
# invariant families. It depends on the basic invariant helpers loaded earlier.

# =============================================================================
# (3b) Global "module size" summaries built from region sizes
# =============================================================================

# Internal helper: interpret box=:auto uniformly.
@inline _resolve_box(pi, box) = (box === :auto ? (pi isa PLikeEncodingMap ? window_box(pi) : nothing) : box)

@inline function _region_weights_from_opts(pi, opts::InvariantOptions; weights=nothing, kwargs...)
    weights === nothing || return weights
    bb = _resolve_box(pi, opts.box)
    if opts.strict === nothing
        return region_weights(pi; box=bb, kwargs...)
    end
    return region_weights(pi; box=bb, strict=opts.strict, kwargs...)
end

function _module_size_stats_from_dims_weights(h::AbstractVector{<:Integer}, w)
    length(w) == length(h) || error("module_size_summary: weights length does not match restricted_hilbert length")
    T = promote_type(eltype(w), Int)
    total = zero(T)
    mass = zero(T)
    mass2 = zero(T)
    supp = zero(T)
    bydim = Dict{Int,T}()
    zT = zero(T)
    @inbounds for i in eachindex(h)
        wi = w[i]
        di = h[i]
        total += wi
        mass += wi * di
        mass2 += wi * (di * di)
        di >= 1 && (supp += wi)
        bydim[di] = get(bydim, di, zT) + wi
    end
    if total == 0
        return (;
            total_measure=0.0,
            integrated_hilbert_mass=0.0,
            support_measure=supp,
            measure_by_dimension=bydim,
            mean_dim=NaN,
            var_dim=NaN,
        )
    end
    total_f = float(total)
    mean = float(mass) / total_f
    mean2 = float(mass2) / total_f
    return (;
        total_measure=total,
        integrated_hilbert_mass=mass,
        support_measure=supp,
        measure_by_dimension=bydim,
        mean_dim=mean,
        var_dim=mean2 - mean * mean,
    )
end

"""
    integrated_hilbert_mass(M, pi, opts::InvariantOptions; weights=nothing, kwargs...) -> Real

Approximate the integral of the Hilbert function over a window.

In a finite encoding, the Hilbert function is constant on each region. If
`w_r = vol(R_r \\cap W)` denotes the region weight inside window W, then:

    \\int_W dim M(x) dx  \\approx  \\sum_r w_r * dim(M|_{R_r})

Arguments:
- `M`: a `PModule{K}`, a `FringeModule{K}` (delegates via `restricted_hilbert`),
  or a vector of integers interpreted as a precomputed restricted Hilbert function.
- `pi`: an encoding map implementing `region_weights(pi; ...)` (typically a
  `PLikeEncodingMap`, e.g. `PLEncodingMapBoxes` or `ZnEncodingMap`).

Uses fields of opts:
- opts.box: window W (via _resolve_box)
- opts.strict: passed to region_weights when not nothing

Keywords:
- weights: optional precomputed region weights
- kwargs...: forwarded to region_weights when weights not provided
"""
function integrated_hilbert_mass(M::PModule{K}, pi, opts::InvariantOptions; weights=nothing, kwargs...) where {K}
    if haskey(kwargs, :box) || haskey(kwargs, :strict) || haskey(kwargs, :threads)
        throw(ArgumentError("integrated_hilbert_mass: pass box/strict/threads via opts, not kwargs"))
    end

    strict0 = _default_strict(opts.strict)

    w = weights
    if w === nothing
        opts.box === nothing && error("integrated_hilbert_mass: provide opts.box (or opts.box=:auto) or pass weights=...")
        bb = _coerce_box(_resolve_box(pi, opts.box))
        w = region_weights(pi; box=bb, strict=strict0, kwargs...)
    end

    h = restricted_hilbert(M)
    length(w) == length(h) || error("integrated_hilbert_mass: weights length does not match restricted_hilbert length")

    acc = 0.0
    @inbounds for i in eachindex(h)
        acc += float(w[i]) * float(h[i])
    end
    return acc
end

function integrated_hilbert_mass(H::FringeModule{K}, pi, opts::InvariantOptions;
    weights=nothing, kwargs...) where {K}
    # Convert to restricted Hilbert dims to reuse the vector-based implementation.
    return integrated_hilbert_mass(restricted_hilbert(H), pi, opts; weights=weights, kwargs...)
end

"""
    integrated_hilbert_mass(dims::AbstractVector{<:Integer}, pi, opts::InvariantOptions;
                            weights=nothing, kwargs...) -> Real

Convenience overload when you already have the restricted Hilbert function `dims`.
Semantics match the module-based method.
"""
function integrated_hilbert_mass(dims::AbstractVector{<:Integer}, pi, opts::InvariantOptions;
    weights=nothing, kwargs...)
    if haskey(kwargs, :box) || haskey(kwargs, :strict) || haskey(kwargs, :threads)
        throw(ArgumentError("integrated_hilbert_mass: pass box/strict/threads via opts, not kwargs"))
    end

    strict0 = _default_strict(opts.strict)

    w = weights
    if w === nothing
        opts.box === nothing && error("integrated_hilbert_mass: provide opts.box (or opts.box=:auto) or pass weights=...")
        bb = _coerce_box(_resolve_box(pi, opts.box))
        w = region_weights(pi; box=bb, strict=strict0, kwargs...)
    end

    h = restricted_hilbert(dims)
    length(w) == length(h) || error("integrated_hilbert_mass: weights length does not match restricted_hilbert length")

    acc = 0.0
    @inbounds for i in eachindex(h)
        acc += float(w[i]) * float(h[i])
    end
    return acc
end


"""
    measure_by_dimension(M, pi, opts::InvariantOptions; weights=nothing, kwargs...) -> Dict{Int,T}

"Histogram" of region sizes by module dimension.

Uses opts.box and opts.strict exactly like integrated_hilbert_mass.

Returns a dictionary mapping d -> total measure of {x in box : dim M(x) == d}.
Values have type `T = promote_type(eltype(weights), Int)`.
"""
function measure_by_dimension(M, pi, opts::InvariantOptions; weights=nothing, kwargs...)
    h = restricted_hilbert(M)
    w = if weights === nothing
        bb = _resolve_box(pi, opts.box)
        if opts.strict === nothing
            region_weights(pi; box=bb, kwargs...)
        else
            region_weights(pi; box=bb, strict=opts.strict, kwargs...)
        end
    else
        weights
    end
    length(w) == length(h) || error("measure_by_dimension: weights length does not match restricted_hilbert length")
    T = promote_type(eltype(w), Int)
    out = Dict{Int,T}()
    zT = zero(T)
    @inbounds for i in eachindex(h)
        d = h[i]
        out[d] = get(out, d, zT) + w[i]
    end
    return out
end

"""
    support_measure(M, pi, opts::InvariantOptions; weights=nothing, min_dim=1, kwargs...) -> Real

Total measure of the support (or high-rank) region:

    measure({x in box : dim M(x) >= min_dim}).

Uses opts.box and opts.strict.

Default `min_dim=1` corresponds to the usual support {dim > 0}.
"""
function support_measure(M, pi, opts::InvariantOptions; weights=nothing, min_dim::Int=1, kwargs...)
    h = restricted_hilbert(M)
    w = if weights === nothing
        bb = _resolve_box(pi, opts.box)
        if opts.strict === nothing
            region_weights(pi; box=bb, kwargs...)
        else
            region_weights(pi; box=bb, strict=opts.strict, kwargs...)
        end
    else
        weights
    end
    length(w) == length(h) || error("support_measure: weights length does not match restricted_hilbert length")
    T = promote_type(eltype(w), Int)
    s = zero(T)
    @inbounds for i in eachindex(h)
        if h[i] >= min_dim
            s += w[i]
        end
    end
    return s
end

"""
    dim_stats(M, pi, opts::InvariantOptions; weights=nothing, kwargs...) -> NamedTuple

Basic region-weighted statistics of the dimension surface.

Uses opts.box and opts.strict.

Returns:
- `total_measure`: sum of weights
- `integrated_mass`: sum(w_r * dim_r)
- `mean`: integrated_mass / total_measure
- `var`: weighted variance of dim_r

If `total_measure == 0`, `mean` and `var` are `NaN`.
"""
function dim_stats(M, pi, opts::InvariantOptions; weights=nothing, kwargs...)
    h = restricted_hilbert(M)
    w = if weights === nothing
        bb = _resolve_box(pi, opts.box)
        if opts.strict === nothing
            region_weights(pi; box=bb, kwargs...)
        else
            region_weights(pi; box=bb, strict=opts.strict, kwargs...)
        end
    else
        weights
    end
    length(w) == length(h) || error("dim_stats: weights length does not match restricted_hilbert length")
    T = promote_type(eltype(w), Int)
    total = zero(T)
    mass = zero(T)
    mass2 = zero(T)
    @inbounds for i in eachindex(h)
        wi = w[i]
        di = h[i]
        total += wi
        mass += wi * di
        mass2 += wi * (di * di)
    end
    if total == 0
        return (total_measure=0.0, integrated_mass=0.0, mean=NaN, var=NaN)
    end
    total_f = float(total)
    mean = float(mass) / total_f
    mean2 = float(mass2) / total_f
    var = mean2 - mean * mean
    return (total_measure=total, integrated_mass=mass, mean=mean, var=var)
end

"""
    dim_norm(M::PModule{K}, pi::PLikeEncodingMap, opts::InvariantOptions; p=2, weights=nothing, kwargs...) -> Float64
    dim_norm(H::FringeModule{K}, pi::PLikeEncodingMap, opts::InvariantOptions; p=2, weights=nothing, kwargs...) -> Float64
    dim_norm(dims::AbstractVector{<:Integer}, pi::PLikeEncodingMap, opts::InvariantOptions; p=2, weights=nothing, kwargs...) -> Float64

Compute an L^p-type norm of the dimension surface of `M` over a window in `pi`.

The `FringeModule` and `dims` overloads use `restricted_hilbert` to obtain the
dimension values per region.

This opts-primary overload uses:
- `opts.box` (with `:auto` interpreted as `window_box(pi)`)
- `opts.strict` (passed to `region_weights` only when not `nothing`)

For finite p:
    ( sum_r w_r * |dim_r|^p )^(1/p)

For p == Inf:
    max_{r : w_r > 0} |dim_r|

If `weights` is provided, it is used directly and `region_weights` is not called.
"""
# Shared implementation once we have a restricted Hilbert vector.
function _dim_norm_impl(h::AbstractVector{<:Integer}, pi::PLikeEncodingMap, opts::InvariantOptions;
    p=2, weights=nothing, kwargs...)
    bb = _resolve_box(pi, opts.box)

    w = if weights === nothing
        if opts.strict === nothing
            region_weights(pi; box=bb, kwargs...)
        else
            region_weights(pi; box=bb, strict=opts.strict, kwargs...)
        end
    else
        weights
    end

    length(h) == length(w) || error("dim_norm: length mismatch between restricted_hilbert and weights")

    if p == Inf
        return maximum(abs.(h))
    elseif p == 1
        return sum(abs.(h) .* w)
    else
        return (sum((abs.(h) .^ p) .* w))^(1 / p)
    end
end

function dim_norm(M::PModule{K}, pi::PLikeEncodingMap, opts::InvariantOptions; p=2, weights=nothing, kwargs...) where {K}
    return _dim_norm_impl(restricted_hilbert(M), pi, opts; p=p, weights=weights, kwargs...)
end

function dim_norm(H::FringeModule{K}, pi::PLikeEncodingMap, opts::InvariantOptions; p=2, weights=nothing, kwargs...) where {K}
    # Delegate through restricted_hilbert to keep logic and weighting uniform.
    return _dim_norm_impl(restricted_hilbert(H), pi, opts; p=p, weights=weights, kwargs...)
end

function dim_norm(dims::AbstractVector{<:Integer}, pi::PLikeEncodingMap, opts::InvariantOptions; p=2, weights=nothing, kwargs...)
    return _dim_norm_impl(restricted_hilbert(dims), pi, opts; p=p, weights=weights, kwargs...)
end


"""
    region_weight_entropy(pi, opts::InvariantOptions; weights=nothing, base=exp(1), kwargs...) -> Float64

Shannon entropy of the region-size distribution inside the window selected by `opts`.

Let `w[i]` be the region weights (volumes) inside the window. Define the normalized
distribution p[i] = w[i] / sum(w). This returns:

    H = - sum_i p[i] * log(p[i]) / log(base)

Uses fields of `opts`
---------------------
- `opts.box`: bounding window for weight computation (via `_resolve_box(pi, opts.box)`).
  You may set `opts.box = :auto` to use `window_box(pi)`.
- `opts.strict`: if not `nothing`, forwarded to `region_weights` as `strict=opts.strict`.

Keywords
--------
- `weights`: optional precomputed region weights. If provided, `opts.box` is not needed.
- `base`: logarithm base (default is `exp(1)` for natural logs).
- `kwargs...`: forwarded to `region_weights(pi; ...)` when weights are not provided.
  Do not pass `box` here. Avoid passing `strict` here; use `opts.strict` instead.

Returns
-------
- `0.0` if `sum(weights) == 0`.

Notes
-----
This is useful as a coarse "how uniformly is the window partitioned?" diagnostic:
larger values indicate many similarly-sized regions; smaller values indicate
mass concentrated in fewer regions.
"""
function region_weight_entropy(pi, opts::InvariantOptions; weights=nothing, base::Real=exp(1), kwargs...)
    # Validate base (must be positive and not 1).
    b = float(base)
    (b > 0.0 && b != 1.0) || error("region_weight_entropy: base must be positive and != 1")

    # Obtain region weights (volumes) either from cache or by calling region_weights.
    w = if weights === nothing
        opts.box === nothing && error("region_weight_entropy: provide opts.box (or opts.box=:auto) or pass weights=...")
        bb = _resolve_box(pi, opts.box)
        if opts.strict === nothing
            region_weights(pi; box=bb, kwargs...)
        else
            region_weights(pi; box=bb, strict=opts.strict, kwargs...)
        end
    else
        weights
    end

    # Compute total mass.
    total = 0.0
    @inbounds for wi in w
        total += float(wi)
    end
    total > 0.0 || return 0.0

    invtotal = 1.0 / total
    logbase = log(b)

    # Shannon entropy in natural logs, then convert to base `b`.
    H = 0.0
    @inbounds for wi in w
        p = float(wi) * invtotal
        if p > 0.0
            H -= p * log(p)
        end
    end
    return H / logbase
end


"""
    aspect_ratio_stats(pi, opts::InvariantOptions; weights=nothing, kwargs...) -> NamedTuple

Volume-weighted summary statistics for region anisotropy inside a window.

For each region r, we compute:
- weight w[r] (volume in the window)
- aspect ratio ar[r] = region_aspect_ratio(pi, r; box=...)

We then return a NamedTuple:
- `total_measure`: sum of weights included in the statistic
- `mean`: weighted mean of aspect ratios
- `min`: minimum aspect ratio among regions with positive weight
- `max`: maximum aspect ratio among regions with positive weight

Uses fields of `opts`
---------------------
- `opts.box`: required, because geometry queries (aspect ratios) require a window.
  You may set `opts.box = :auto` to use `window_box(pi)`.
- `opts.strict`: if not `nothing`, forwarded to both `region_weights` and
  `region_aspect_ratio` (as backend keyword `strict=...`).

Keywords
--------
- `weights`: optional precomputed region weights.
- `kwargs...`: forwarded to `region_weights` / `region_aspect_ratio` when called.
  Do not pass `box` here. Avoid passing `strict` here; use `opts.strict` instead.

Notes
-----
- Regions with zero weight are skipped.
- If total measure is zero, returns mean/min/max as NaN.
"""
function aspect_ratio_stats(pi, opts::InvariantOptions; weights=nothing, kwargs...)
    # Geometry statistics require a box even if weights are provided.
    opts.box === nothing && error("aspect_ratio_stats: opts.box is required (or set opts.box=:auto)")
    bb = _resolve_box(pi, opts.box)

    # Obtain region weights.
    w = if weights === nothing
        if opts.strict === nothing
            region_weights(pi; box=bb, kwargs...)
        else
            region_weights(pi; box=bb, strict=opts.strict, kwargs...)
        end
    else
        weights
    end

    # Accumulate weighted stats.
    total = 0.0
    mean_num = 0.0
    amin = Inf
    amax = 0.0

    @inbounds for r in 1:length(w)
        wi = float(w[r])
        wi == 0.0 && continue

        # Aspect ratio uses region_bbox/region_widths under the hood.
        ar = if opts.strict === nothing
            region_aspect_ratio(pi, r; box=bb, kwargs...)
        else
            region_aspect_ratio(pi, r; box=bb, strict=opts.strict, kwargs...)
        end

        ari = float(ar)
        total += wi
        mean_num += wi * ari
        amin = min(amin, ari)
        amax = max(amax, ari)
    end

    if total == 0.0
        return (total_measure=0.0, mean=NaN, min=NaN, max=NaN)
    end
    return (total_measure=total, mean=mean_num / total, min=amin, max=amax)
end


"""
    module_size_summary(M, pi, opts::InvariantOptions; weights=nothing, kwargs...) -> ModuleSizeSummary

Convenience wrapper collecting:
- total_measure
- integrated_hilbert_mass
- support_measure
- measure_by_dimension
- mean_dim, var_dim

Uses opts.box and opts.strict.

"""
function module_size_summary(M, pi, opts::InvariantOptions; weights=nothing, kwargs...)
    h = restricted_hilbert(M)
    w = _region_weights_from_opts(pi, opts; weights=weights, kwargs...)
    stats = _module_size_stats_from_dims_weights(h, w)
    return ModuleSizeSummary((
        total_measure = stats.total_measure,
        integrated_hilbert_mass = stats.integrated_hilbert_mass,
        support_measure = stats.support_measure,
        measure_by_dimension = stats.measure_by_dimension,
        mean_dim = stats.mean_dim,
        var_dim = stats.var_dim
    ))
end

# =============================================================================
# (3c) Interface measures from the region adjacency graph
# =============================================================================

"""
    interface_measure(pi, opts::InvariantOptions; adjacency=nothing, kwargs...) -> Float64

Total (n-1)-dimensional interface measure between adjacent regions inside the window.

This sums the edge weights returned by `region_adjacency`:

    sum( m_rs for (r,s) in edges )

where `edges = region_adjacency(pi; box=...)` is a dictionary keyed by unordered
pairs `(r,s)` with `r < s`, and each value is the estimated interface measure
(length/area/hyperarea) between the regions inside the window.

Uses fields of `opts`
---------------------
- `opts.box`: required if `adjacency` is not provided. You may set `opts.box=:auto`.
- `opts.strict`: if not `nothing`, forwarded to `region_adjacency` as `strict=...`.

Keywords
--------
- `adjacency`: optionally provide precomputed adjacency dictionary to avoid recomputation.
- `kwargs...`: forwarded to `region_adjacency(pi; ...)` when adjacency is not provided.
  Do not pass `box` here. Avoid passing `strict` here; use `opts.strict` instead.
"""
function interface_measure(pi, opts::InvariantOptions; adjacency=nothing, kwargs...)
    edges = if adjacency === nothing
        opts.box === nothing && error("interface_measure: provide opts.box (or opts.box=:auto) or pass adjacency=...")
        bb = _resolve_box(pi, opts.box)
        if opts.strict === nothing
            region_adjacency(pi; box=bb, kwargs...)
        else
            region_adjacency(pi; box=bb, strict=opts.strict, kwargs...)
        end
    else
        adjacency
    end

    s = 0.0
    for m in values(edges)
        s += float(m)
    end
    return s
end


"""
    interface_measure_by_dim_pair(M, pi, opts::InvariantOptions; adjacency=nothing, kwargs...) -> Dict{Tuple{Int,Int},Float64}

Partition total interface measure by the pair of Hilbert dimensions on each side.

For each adjacency edge (r,s) with interface measure m_rs, let:
- a = dim(M on region r) = restricted_hilbert(M)[r]
- b = dim(M on region s) = restricted_hilbert(M)[s]

We accumulate m_rs into a dictionary keyed by (min(a,b), max(a,b)).

Uses fields of `opts`
---------------------
- `opts.box`: required if `adjacency` not provided. You may set `opts.box=:auto`.
- `opts.strict`: if not `nothing`, forwarded to `region_adjacency` as `strict=...`.

Keywords
--------
- `adjacency`: optionally provide a precomputed adjacency dictionary.
- `kwargs...`: forwarded to `region_adjacency(pi; ...)` when adjacency is not provided.
"""
function interface_measure_by_dim_pair(M, pi, opts::InvariantOptions; adjacency=nothing, kwargs...)
    dims = restricted_hilbert(M)

    edges = if adjacency === nothing
        opts.box === nothing && error("interface_measure_by_dim_pair: provide opts.box (or opts.box=:auto) or pass adjacency=...")
        bb = _resolve_box(pi, opts.box)
        if opts.strict === nothing
            region_adjacency(pi; box=bb, kwargs...)
        else
            region_adjacency(pi; box=bb, strict=opts.strict, kwargs...)
        end
    else
        adjacency
    end

    out = Dict{Tuple{Int,Int}, Float64}()
    z = 0.0
    for ((r, s), m) in edges
        a = dims[r]
        b = dims[s]
        i = min(a, b)
        j = max(a, b)
        out[(i, j)] = get(out, (i, j), z) + float(m)
    end
    return out
end


"""
    interface_measure_dim_changes(M, pi, opts::InvariantOptions; adjacency=nothing, kwargs...) -> Float64

Total interface measure across region boundaries where the Hilbert dimension changes.

That is, we sum interface measures m_rs over adjacency edges (r,s) such that
restricted_hilbert(M)[r] != restricted_hilbert(M)[s].

Uses fields of `opts`
---------------------
- `opts.box`: required if `adjacency` not provided. You may set `opts.box=:auto`.
- `opts.strict`: if not `nothing`, forwarded to `region_adjacency` as `strict=...`.

Keywords
--------
- `adjacency`: optionally provide a precomputed adjacency dictionary.
- `kwargs...`: forwarded to `region_adjacency(pi; ...)` when adjacency is not provided.
"""
function interface_measure_dim_changes(M, pi, opts::InvariantOptions; adjacency=nothing, kwargs...)
    dims = restricted_hilbert(M)

    edges = if adjacency === nothing
        opts.box === nothing && error("interface_measure_dim_changes: provide opts.box (or opts.box=:auto) or pass adjacency=...")
        bb = _resolve_box(pi, opts.box)
        if opts.strict === nothing
            region_adjacency(pi; box=bb, kwargs...)
        else
            region_adjacency(pi; box=bb, strict=opts.strict, kwargs...)
        end
    else
        adjacency
    end

    s = 0.0
    for ((r, t), m) in edges
        if dims[r] != dims[t]
            s += float(m)
        end
    end
    return s
end
