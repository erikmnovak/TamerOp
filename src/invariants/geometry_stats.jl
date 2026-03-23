# This fragment owns additional region-geometry and asymptotic module summaries
# for TamerOp.Invariants. It builds on region weights/boundary/adjacency and
# the restricted Hilbert machinery loaded earlier. It does not own pretty
# printing, Betti/Bass support helpers, or support-geometry queries.

# ------------------------------------------------------------------------------
# (3b) Additional module-level geometry statistics
#
# These build on:
# - region_weights (volumes)
# - region_boundary_measure (boundary measures)
# - region_adjacency (adjacency graph, weighted by boundary measure)
#
# and stratify by the restricted Hilbert function (dimension per region).
# ------------------------------------------------------------------------------

# Internal histogram helper (no StatsBase dependency).
# Returns (edges, counts) where edges has length nbins+1.
function _histogram_1d(values::Vector{Float64}; nbins::Integer=10, range=nothing)
    nbins >= 1 || error("histogram: nbins must be >= 1")
    v = [x for x in values if isfinite(x)]
    isempty(v) && return (edges=Float64[], counts=Int[])

    lo = range === nothing ? minimum(v) : float(range[1])
    hi = range === nothing ? maximum(v) : float(range[2])

    if hi == lo
        hi = lo + 1.0
    end

    edges = collect(Base.range(lo, hi; length=nbins + 1))
    counts = zeros(Int, nbins)

    for x in v
        # Place x into a bin; include hi in the last bin.
        if x <= lo
            counts[1] += 1
        elseif x >= hi
            counts[end] += 1
        else
            t = (x - lo) / (hi - lo)
            b = Int(floor(t * nbins)) + 1
            b = clamp(b, 1, nbins)
            counts[b] += 1
        end
    end

    return (edges=edges, counts=counts)
end

"""
    region_volume_samples_by_dim(M, pi, opts::InvariantOptions; weights=nothing, kwargs...) -> Dict{Int,Vector{Float64}}

Collect per-region volume samples grouped by Hilbert dimension.

For each region r:
- let d = restricted_hilbert(M)[r]
- let v = region weight (volume) w[r]
Append v to the vector stored at key d.

Uses fields of `opts`
---------------------
- `opts.box`: required if `weights` not provided (or set `opts.box=:auto`).
- `opts.strict`: if not `nothing`, forwarded to `region_weights` as `strict=...`.

Keywords
--------
- `weights`: optionally provide precomputed region weights.
- `kwargs...`: forwarded to `region_weights(pi; ...)` when weights are not provided.
"""
function region_volume_samples_by_dim(M, pi, opts::InvariantOptions; weights=nothing, kwargs...)
    dims = restricted_hilbert(M)

    w = if weights === nothing
        opts.box === nothing && error("region_volume_samples_by_dim: provide opts.box (or opts.box=:auto) or pass weights=...")
        bb = _resolve_box(pi, opts.box)
        if opts.strict === nothing
            region_weights(pi; box=bb, kwargs...)
        else
            region_weights(pi; box=bb, strict=opts.strict, kwargs...)
        end
    else
        weights
    end

    length(w) == length(dims) || error("region_volume_samples_by_dim: size mismatch")

    out = Dict{Int, Vector{Float64}}()
    @inbounds for i in eachindex(dims)
        d = dims[i]
        push!(get!(out, d, Float64[]), float(w[i]))
    end
    return out
end


"""
    region_volume_histograms_by_dim(M, pi, opts::InvariantOptions;
                                   weights=nothing, nbins=10, range=nothing, kwargs...) -> Dict{Int,NamedTuple}

Compute per-dimension histograms of region volumes.

This calls `region_volume_samples_by_dim` and then bins each sample set with
`_histogram_1d`.

Uses fields of `opts`
---------------------
- `opts.box`: required if `weights` not provided (or set `opts.box=:auto`).
- `opts.strict`: if not `nothing`, forwarded to `region_weights` as `strict=...`.

Keywords
--------
- `weights`: optionally provide precomputed region weights.
- `nbins`: number of histogram bins.
- `range`: optional (lo, hi) range passed to `_histogram_1d`.
- `kwargs...`: forwarded to `region_weights` via `region_volume_samples_by_dim`.
"""
function region_volume_histograms_by_dim(M, pi, opts::InvariantOptions;
    weights=nothing,
    nbins::Integer=10,
    range=nothing,
    kwargs...
)
    samples = region_volume_samples_by_dim(M, pi, opts; weights=weights, kwargs...)
    out = Dict{Int, NamedTuple}()
    for (d, v) in samples
        out[d] = _histogram_1d(v; nbins=nbins, range=range)
    end
    return out
end


"""
    region_boundary_to_volume_samples_by_dim(M, pi, opts::InvariantOptions;
                                            weights=nothing, boundary_measures=nothing) -> Dict{Int,Vector{Float64}}

Collect per-region (boundary measure)/(volume) ratios grouped by Hilbert dimension.

For each region r with weight w[r] > 0, define:

    ratio[r] = region_boundary_measure(pi, r; box=..., strict=...)/w[r]

Non-finite or negative ratios are converted to NaN.

Uses fields of `opts`
---------------------
- `opts.box`: required unless both `weights` and `boundary_measures` are provided.
  You may set `opts.box=:auto`.
- `opts.strict`: controls the `strict` flag used in `region_boundary_measure`.
  Old keyword API default was `strict=true`, so we use:
      strict0 = (opts.strict === nothing ? true : opts.strict)

Keywords
--------
- `weights`: optionally provide precomputed region weights.
- `boundary_measures`: optionally provide precomputed boundary measures (per region).

Notes
-----
- If `opts.strict` is `nothing`, we preserve the old default `strict=true` for
  boundary measure computation.
- If you want to avoid any geometry calls, pass both `weights` and `boundary_measures`.
"""
function region_boundary_to_volume_samples_by_dim(M, pi, opts::InvariantOptions;
    weights=nothing,
    boundary_measures=nothing
)
    dims = restricted_hilbert(M)
    n = length(dims)

    # Old keyword default was strict=true.
    strict0 = (opts.strict === nothing ? true : opts.strict)

    w = if weights === nothing
        opts.box === nothing && error("region_boundary_to_volume_samples_by_dim: provide opts.box (or opts.box=:auto) or pass weights=...")
        bb = _resolve_box(pi, opts.box)
        # Also apply strict0 to weights for consistency.
        region_weights(pi; box=bb, strict=strict0)
    else
        weights
    end
    length(w) == n || error("region_boundary_to_volume_samples_by_dim: size mismatch")

    bms = if boundary_measures === nothing
        opts.box === nothing && error("region_boundary_to_volume_samples_by_dim: provide opts.box (or opts.box=:auto) or pass boundary_measures=...")
        bb = _resolve_box(pi, opts.box)
        [region_boundary_measure(pi, i; box=bb, strict=strict0) for i in 1:n]
    else
        boundary_measures
    end
    length(bms) == n || error("region_boundary_to_volume_samples_by_dim: boundary size mismatch")

    out = Dict{Int, Vector{Float64}}()
    @inbounds for i in 1:n
        wi = float(w[i])
        wi == 0.0 && continue

        ri = float(bms[i]) / wi
        if !isfinite(ri) || ri < 0.0
            ri = NaN
        end

        d = dims[i]
        push!(get!(out, d, Float64[]), ri)
    end
    return out
end


"""
    region_boundary_to_volume_histograms_by_dim(M, pi, opts::InvariantOptions;
                                                weights=nothing, boundary_measures=nothing,
                                                nbins=10, range=nothing) -> Dict{Int,NamedTuple}

Compute per-dimension histograms of (boundary measure)/(volume) ratios.

This calls `region_boundary_to_volume_samples_by_dim` and bins each sample set
with `_histogram_1d`.

Uses fields of `opts`
---------------------
- `opts.box`, `opts.strict` (see `region_boundary_to_volume_samples_by_dim`).
"""
function region_boundary_to_volume_histograms_by_dim(M, pi, opts::InvariantOptions;
    weights=nothing,
    boundary_measures=nothing,
    nbins::Integer=10,
    range=nothing
)
    samples = region_boundary_to_volume_samples_by_dim(M, pi, opts;
        weights=weights,
        boundary_measures=boundary_measures
    )
    out = Dict{Int, NamedTuple}()
    for (d, v) in samples
        out[d] = _histogram_1d(v; nbins=nbins, range=range)
    end
    return out
end


# ------------------------------------------------------------------------------
# Graph statistics for the region adjacency graph
# The adjacency is assumed undirected with each edge appearing once as (i,j), i<j.
# Edge weights are typically boundary measures (interface sizes).
# ------------------------------------------------------------------------------

"""
    graph_degrees(adjacency, nregions) -> (degrees, weighted_degrees)

Compute (unweighted) degrees and weighted degrees for an undirected graph.

- `adjacency` should be a Dict{Tuple{Int,Int},<:Real} with edges stored once.
- `degrees[i]` is the number of incident edges.
- `weighted_degrees[i]` is the sum of incident edge weights.
"""
function graph_degrees(adjacency, nregions::Integer)
    deg = zeros(Int, nregions)
    wdeg = zeros(Float64, nregions)
    for ((i, j), w) in adjacency
        wi = float(w)
        deg[i] += 1
        deg[j] += 1
        wdeg[i] += wi
        wdeg[j] += wi
    end
    return (degrees=deg, weighted_degrees=wdeg)
end

"""
    graph_connected_components(adjacency, nregions) -> Vector{Vector{Int}}

Connected components of an undirected graph.
"""
function graph_connected_components(adjacency, nregions::Integer)
    nbrs = [Int[] for _ in 1:nregions]
    for ((i, j), _) in adjacency
        push!(nbrs[i], j)
        push!(nbrs[j], i)
    end

    visited = falses(nregions)
    comps = Vector{Vector{Int}}()

    for v in 1:nregions
        if visited[v]
            continue
        end
        stack = [v]
        visited[v] = true
        comp = Int[]
        while !isempty(stack)
            u = pop!(stack)
            push!(comp, u)
            for w in nbrs[u]
                if !visited[w]
                    visited[w] = true
                    push!(stack, w)
                end
            end
        end
        push!(comps, comp)
    end

    return comps
end

"""
    graph_modularity(labels, adjacency; nregions=length(labels)) -> Float64

Weighted Newman-Girvan modularity for an undirected graph.

Let m = total edge weight (edges counted once). Let k_i be weighted degree of node i.
Then:

    Q = sum_c (w_in(c)/m - (k_tot(c)/(2m))^2),

where w_in(c) is total weight of edges with both endpoints in community c, and
k_tot(c) = sum_{i in c} k_i.

Returns 0 when m == 0.
"""
function graph_modularity(labels::AbstractVector{<:Integer}, adjacency; nregions::Integer=length(labels))
    length(labels) == nregions || error("graph_modularity: label length mismatch")

    m = 0.0
    wdeg = zeros(Float64, nregions)
    for ((i, j), w) in adjacency
        wi = float(w)
        m += wi
        wdeg[i] += wi
        wdeg[j] += wi
    end
    if m == 0.0
        return 0.0
    end

    # k_tot per community
    ktot = Dict{Int,Float64}()
    for i in 1:nregions
        lab = Int(labels[i])
        ktot[lab] = get(ktot, lab, 0.0) + wdeg[i]
    end

    # w_in per community (edges counted once)
    win = Dict{Int,Float64}()
    for ((i, j), w) in adjacency
        li = Int(labels[i])
        lj = Int(labels[j])
        if li == lj
            win[li] = get(win, li, 0.0) + float(w)
        end
    end

    Q = 0.0
    for (lab, ksum) in ktot
        Q += get(win, lab, 0.0) / m - (ksum / (2.0 * m))^2
    end
    return Q
end

"""
    region_adjacency_graph_stats(M, pi, opts::InvariantOptions; adjacency=nothing, kwargs...) -> NamedTuple

Graph statistics of the region adjacency graph inside a window.

We build (or accept) an adjacency dictionary `edges = region_adjacency(...)` and
then compute:
- weighted degrees (using edge weights as interface measures)
- connected components
- size distribution of components (in number of regions)
- a simple modularity heuristic based on Hilbert dimension labels

Uses fields of `opts`
---------------------
- `opts.box`: required if `adjacency` not provided. You may set `opts.box=:auto`.
- `opts.strict`: old keyword API default was `strict=true`. We use:
      strict0 = (opts.strict === nothing ? true : opts.strict)

Keywords
--------
- `adjacency`: optionally provide a precomputed adjacency dictionary.
- `kwargs...`: forwarded to `region_adjacency(pi; ...)` when adjacency is not provided.
"""
function region_adjacency_graph_stats(M, pi, opts::InvariantOptions; adjacency=nothing, kwargs...)
    dims = restricted_hilbert(M)
    nregions = length(dims)

    # Old keyword default was strict=true.
    strict0 = (opts.strict === nothing ? true : opts.strict)

    edges = if adjacency === nothing
        opts.box === nothing && error("region_adjacency_graph_stats: provide opts.box (or opts.box=:auto) or pass adjacency=...")
        bb = _resolve_box(pi, opts.box)
        region_adjacency(pi; box=bb, strict=strict0, kwargs...)
    else
        adjacency
    end

    degrees = graph_degrees(edges, nregions)
    comps = graph_connected_components(edges, nregions)
    comp_sizes = sort([length(c) for c in comps]; rev=true)
    ncomps = length(comps)

    # Each region is labeled by its Hilbert dimension; use that as a crude "community label".
    labels = [dims[i] for i in 1:nregions]
    Q = graph_modularity(labels, edges; nregions=nregions)

    return (
        nregions=nregions,
        nedges=length(edges),
        ncomponents=ncomps,
        component_sizes=comp_sizes,
        degrees=degrees,
        modularity=Q
    )
end


"""
    module_geometry_summary(M, pi, opts::InvariantOptions;
                            weights=nothing, adjacency=nothing, boundary_measures=nothing,
                            nbins=10, range=nothing) -> ModuleGeometrySummary

Compute a bundle of "geometry of support" summaries for a module M over a window.

This is a convenience aggregator that combines:
- module_size_summary (mass/support/entropy/stats)
- interface measures from region adjacency
- per-dimension region volume samples and histograms
- per-dimension boundary/volume ratio samples and histograms
- adjacency graph stats

Uses fields of `opts`
---------------------
- `opts.box`: required unless you provide all caches (`weights`, `adjacency`,
  `boundary_measures`) needed by the subcomputations. You may set `opts.box=:auto`.
- `opts.strict`: old keyword API default was `strict=true`. We use:
      strict0 = (opts.strict === nothing ? true : opts.strict)

Keywords
--------
- `weights`: optional precomputed region weights.
- `adjacency`: optional precomputed region adjacency dictionary.
- `boundary_measures`: optional precomputed per-region boundary measures.
- `nbins`, `range`: histogram settings (passed through).

Notes
-----
This function intentionally avoids forwarding arbitrary backend keywords. If you
need fine control, precompute `weights`, `adjacency`, and/or `boundary_measures`
with backend-specific calls and pass them in.
"""
function module_geometry_summary(M, pi, opts::InvariantOptions;
    weights=nothing,
    adjacency=nothing,
    boundary_measures=nothing,
    nbins::Integer=10,
    range=nothing
)
    # Old keyword default was strict=true.
    strict0 = (opts.strict === nothing ? true : opts.strict)

    # Compute caches if not provided.
    w = if weights === nothing
        opts.box === nothing && error("module_geometry_summary: provide opts.box (or opts.box=:auto) or pass weights=...")
        bb = _resolve_box(pi, opts.box)
        region_weights(pi; box=bb, strict=strict0)
    else
        weights
    end

    edges = if adjacency === nothing
        opts.box === nothing && error("module_geometry_summary: provide opts.box (or opts.box=:auto) or pass adjacency=...")
        bb = _resolve_box(pi, opts.box)
        region_adjacency(pi; box=bb, strict=strict0)
    else
        adjacency
    end

    bms = if boundary_measures === nothing
        opts.box === nothing && error("module_geometry_summary: provide opts.box (or opts.box=:auto) or pass boundary_measures=...")
        bb = _resolve_box(pi, opts.box)
        [region_boundary_measure(pi, i; box=bb, strict=strict0) for i in 1:length(w)]
    else
        boundary_measures
    end

    # Size summary is already opts-primary.
    size_sum = module_size_summary(M, pi, opts; weights=w)

    # Interface-related summaries.
    iface = interface_measure(pi, opts; adjacency=edges)
    iface_pairs = interface_measure_by_dim_pair(M, pi, opts; adjacency=edges)
    iface_changes = interface_measure_dim_changes(M, pi, opts; adjacency=edges)

    # Volume and boundary/volume summaries by dimension.
    vol_samples = region_volume_samples_by_dim(M, pi, opts; weights=w)
    vol_hists = region_volume_histograms_by_dim(M, pi, opts; weights=w, nbins=nbins, range=range)

    b2v_samples = region_boundary_to_volume_samples_by_dim(M, pi, opts; weights=w, boundary_measures=bms)
    b2v_hists = region_boundary_to_volume_histograms_by_dim(M, pi, opts;
        weights=w,
        boundary_measures=bms,
        nbins=nbins,
        range=range
    )

    # Graph stats.
    gstats = region_adjacency_graph_stats(M, pi, opts; adjacency=edges)

    return ModuleGeometrySummary((
        size_summary=size_sum,
        interface_measure=iface,
        interface_by_dim_pair=iface_pairs,
        interface_dim_changes=iface_changes,
        volume_samples_by_dim=vol_samples,
        volume_histograms_by_dim=vol_hists,
        boundary_to_volume_samples_by_dim=b2v_samples,
        boundary_to_volume_histograms_by_dim=b2v_hists,
        graph_stats=gstats
    ))
end


# -----------------------------------------------------------------------------
# Asymptotic growth summaries (expanding windows)
# -----------------------------------------------------------------------------

# small internal helper: scale a box about its center and (optionally) integerize outward
function _scale_box_about_center(ell0, u0, s::Real, padding::Real, integerize::Bool)
    n = length(ell0)
    if integerize
        ell = Vector{Int}(undef, n)
        u   = Vector{Int}(undef, n)
        for i in 1:n
            lo0 = float(ell0[i])
            hi0 = float(u0[i])
            c = (lo0 + hi0) / 2
            h = (hi0 - lo0) / 2
            lo = c - float(s) * h - padding
            hi = c + float(s) * h + padding
            ell[i] = floor(Int, lo)
            u[i]   = ceil(Int,  hi)
        end
        return (ell, u)
    else
        ell = Vector{Float64}(undef, n)
        u   = Vector{Float64}(undef, n)
        for i in 1:n
            lo0 = float(ell0[i])
            hi0 = float(u0[i])
            c = (lo0 + hi0) / 2
            h = (hi0 - lo0) / 2
            ell[i] = c - float(s) * h - padding
            u[i]   = c + float(s) * h + padding
        end
        return (ell, u)
    end
end

# log-log linear regression fit of log(y) vs log(scale)
function _loglog_fit(scales, ys; strict::Bool=false)
    xs = Float64[]
    zs = Float64[]
    used = Int[]
    for i in eachindex(scales)
        s = float(scales[i])
        y = float(ys[i])
        if isfinite(s) && isfinite(y) && s > 0 && y > 0
            push!(xs, log(s))
            push!(zs, log(y))
            push!(used, i)
        end
    end
    if length(xs) < 2
        strict && error("loglog fit needs at least two positive points")
        return (exponent=NaN, intercept=NaN, r2=NaN, used_indices=used)
    end

    # compute means manually (no Statistics dependency)
    mx = 0.0
    mz = 0.0
    for i in eachindex(xs)
        mx += xs[i]
        mz += zs[i]
    end
    mx /= length(xs)
    mz /= length(xs)

    vx = 0.0
    cov = 0.0
    for i in eachindex(xs)
        dx = xs[i] - mx
        vx += dx * dx
        cov += dx * (zs[i] - mz)
    end
    b = cov / vx
    a = mz - b * mx

    sse = 0.0
    sst = 0.0
    for i in eachindex(xs)
        pred = a + b * xs[i]
        err = zs[i] - pred
        sse += err * err
        dz = zs[i] - mz
        sst += dz * dz
    end
    r2 = (sst == 0.0 ? 1.0 : 1.0 - sse / sst)

    return (exponent=b, intercept=a, r2=r2, used_indices=used)
end

# Polynomial least squares fit y ~ sum_{k=0}^deg c[k+1] * x^k.
function _polyfit(xs::AbstractVector{<:Real}, ys::AbstractVector{<:Real}, deg::Int)
    m = length(xs)
    A = Matrix{Float64}(undef, m, deg + 1)
    for i in 1:m
        x = float(xs[i])
        A[i, 1] = 1.0
        for k in 2:(deg + 1)
            A[i, k] = A[i, k - 1] * x
        end
    end
    y = Float64.(ys)
    c = A \ y
    yhat = A * c
    mu = sum(y) / m
    sst = sum((y[i] - mu)^2 for i in 1:m)
    sse = sum((y[i] - yhat[i])^2 for i in 1:m)
    r2 = sst == 0.0 ? NaN : 1.0 - sse/sst
    return (degree=deg, coeffs=c, r2=r2)
end

# Quasi-polynomial fit: separate polynomial fits on each residue class mod period.
function _quasipolyfit(xs::AbstractVector{<:Integer}, ys::AbstractVector{<:Real}, deg::Int, period::Int)
    QuasiPolyFitEntry = Union{
        Nothing,
        NamedTuple{
            (:degree, :coeffs, :r2, :residue, :npoints),
            Tuple{Int,Vector{Float64},Float64,Int,Int}
        }
    }
    fits = Vector{QuasiPolyFitEntry}(undef, period)
    for r in 0:(period - 1)
        idx = [i for i in eachindex(xs) if mod(xs[i], period) == r]
        if length(idx) < deg + 1
            fits[r + 1] = nothing
            continue
        end
        xsr = [xs[i] for i in idx]
        ysr = [ys[i] for i in idx]
        fits[r + 1] = merge(_polyfit(xsr, ysr, deg), (residue=r, npoints=length(idx)))
    end
    return (period=period, degree=deg, fits=fits)
end


"""
    module_geometry_asymptotics(M::PModule{K}, pi::PLikeEncodingMap, opts::InvariantOptions;
        scales=[1,2,4,8],
        padding=0.0,
        fit=:loglog,
        include_interface=true,
        include_ehrhart=false,
        ehrhart_period=1,
        ehrhart_degree=:auto) -> ModuleGeometryAsymptoticsSummary
    module_geometry_asymptotics(dims::AbstractVector{<:Integer}, pi::PLikeEncodingMap, opts::InvariantOptions;
        scales=[1,2,4,8],
        padding=0.0,
        fit=:loglog,
        include_interface=true,
        include_ehrhart=false,
        ehrhart_period=1,
        ehrhart_degree=:auto) -> ModuleGeometryAsymptoticsSummary

Compute window-dependent size and geometry summaries as the window expands.

The `dims` overload treats the vector as a precomputed restricted Hilbert
function (dimension per region).

If `base_box == :auto`, uses `window_box(pi)`.

For each `s in scales`, expands the base window about its center by factor `s`,
then adds absolute `padding`.

For lattice encodings, padding is applied in real coordinates and then the
endpoints are rounded outward to integers so the window remains an integer box.

This is an opts-primary API:
- `opts.box` provides the base window. If `opts.box === nothing`, we use the
  default `:auto` (equivalently `window_box(pi)`).
- `opts.strict` is forwarded to region computations (defaults to `true`).

The returned `ModuleGeometryAsymptoticsSummary` preserves the previous report
fields while adding typed `show` / `describe` support. Windowing and strictness
are controlled via `opts`.

The wrapped report contains:
  * `windows`: per-scale windows used
  * `total_measure`: sum of region weights (volume in R^n or lattice count in Z^n)
  * `integrated_hilbert_mass`: integrated mass over the window
  * `interface_measure`: total adjacency weight, if supported
  * fitted log-log growth exponents
"""
# Shared implementation for PModule and dims to keep behavior uniform.
function _module_geometry_asymptotics_impl(M, pi::PLikeEncodingMap, opts::InvariantOptions;
    scales::AbstractVector{<:Real} = [1,2,4,8],
    padding::Real = 0.0,
    fit::Symbol = :loglog,
    include_interface::Bool = true,
    include_ehrhart::Bool = false,
    ehrhart_period::Integer = 1,
    ehrhart_degree = :auto)

    fit == :loglog || error("module_geometry_asymptotics: only fit=:loglog is implemented")

    # Legacy defaults when opts fields are unset.
    base_box = opts.box === nothing ? :auto : opts.box
    strict0 = opts.strict === nothing ? true : opts.strict

    # Resolve the base window and check dimensional consistency.
    bb = (base_box === :auto ? window_box(pi) : base_box)
    ell0, u0 = bb
    length(ell0) == length(u0) || error("module_geometry_asymptotics: base_box has mismatched endpoints")

    integerize = (eltype(ell0) <: Integer) && (eltype(u0) <: Integer)
    dim = length(ell0)

    windows = Vector{typeof(bb)}()
    totals  = Float64[]
    masses  = Float64[]
    ifaces  = Float64[]

    # Only compute interface terms if the encoding supplies adjacency.
    do_iface = include_interface && hasmethod(region_adjacency, Tuple{typeof(pi)})

    for s in scales
        if integerize
            # Keep lattice backends on integer boxes after padding.
            ell = ceil.(Int, (ell0 .* s) .- padding)
            u = floor.(Int, (u0 .* s) .+ padding)
            win = (ell, u)
        else
            ell = (ell0 .* s) .- padding
            u = (u0 .* s) .+ padding
            win = (ell, u)
        end
        push!(windows, win)

        w = region_weights(pi; box=win, strict=strict0)
        push!(totals, sum(float, values(w)))

        # We provide weights explicitly, so an empty options object is sufficient.
        push!(masses, integrated_hilbert_mass(M, pi, InvariantOptions(); weights=w))

        if do_iface
            adj = region_adjacency(pi; box=win, strict=strict0)
            push!(ifaces, sum(float, values(adj)))
        end
    end

    fit_total = _loglog_fit(scales, totals)
    fit_mass  = _loglog_fit(scales, masses)
    fit_iface = do_iface ? _loglog_fit(scales, ifaces) : nothing

    ehrhart_degree = (ehrhart_degree === :auto ? dim : ehrhart_degree)
    ehrhart_total = include_ehrhart ? _quasipolyfit(scales, totals, ehrhart_degree, ehrhart_period) : nothing
    ehrhart_iface = include_ehrhart ? (do_iface ? _quasipolyfit(scales, ifaces, ehrhart_degree, ehrhart_period) : nothing) : nothing

    return ModuleGeometryAsymptoticsSummary((
        base_box = bb,
        scales = scales,
        windows = windows,
        total_measure = totals,
        integrated_hilbert_mass = masses,
        interface_measure = (do_iface ? ifaces : nothing),
        exponent_total_measure = fit_total.exponent,
        exponent_integrated_hilbert_mass = fit_mass.exponent,
        exponent_interface_measure = (do_iface ? fit_iface.exponent : nothing),
        fit_total = fit_total,
        fit_integrated_hilbert_mass = fit_mass,
        fit_interface = fit_iface,
        ehrhart_total_measure = ehrhart_total,
        ehrhart_interface_measure = ehrhart_iface,
        ehrhart_period = ehrhart_period,
        ehrhart_degree = ehrhart_degree,
    ))
end

function module_geometry_asymptotics(M::PModule{K}, pi::PLikeEncodingMap, opts::InvariantOptions;
    scales::AbstractVector{<:Real} = [1,2,4,8],
    padding::Real = 0.0,
    fit::Symbol = :loglog,
    include_interface::Bool = true,
    include_ehrhart::Bool = false,
    ehrhart_period::Integer = 1,
    ehrhart_degree = :auto) where {K}
    return _module_geometry_asymptotics_impl(M, pi, opts;
        scales=scales,
        padding=padding,
        fit=fit,
        include_interface=include_interface,
        include_ehrhart=include_ehrhart,
        ehrhart_period=ehrhart_period,
        ehrhart_degree=ehrhart_degree)
end

function module_geometry_asymptotics(dims::AbstractVector{<:Integer}, pi::PLikeEncodingMap, opts::InvariantOptions;
    scales::AbstractVector{<:Real} = [1,2,4,8],
    padding::Real = 0.0,
    fit::Symbol = :loglog,
    include_interface::Bool = true,
    include_ehrhart::Bool = false,
    ehrhart_period::Integer = 1,
    ehrhart_degree = :auto)
    return _module_geometry_asymptotics_impl(dims, pi, opts;
        scales=scales,
        padding=padding,
        fit=fit,
        include_interface=include_interface,
        include_ehrhart=include_ehrhart,
        ehrhart_period=ehrhart_period,
        ehrhart_degree=ehrhart_degree)
end


