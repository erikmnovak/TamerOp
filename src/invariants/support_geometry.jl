# This fragment owns support-geometry queries for TamerOp.Invariants:
# support masks/components, graph/geometric diameters, bounding boxes, and
# support-measure summaries. It depends on the Hilbert-function alias and region
# utilities loaded earlier. It does not own opts-default wrapper methods.

# -----------------------------------------------------------------------------
# Support geometry: unions of regions with dim(M_x) >= k
# -----------------------------------------------------------------------------

"""
    support_mask(H; min_dim=1)

Return a BitVector indicating which regions are in the support:
mask[r] == true iff H[r] >= min_dim.
"""
function support_mask(H::AbstractVector{<:Integer}; min_dim::Integer=1)
    m = BitVector(undef, length(H))
    @inbounds for i in eachindex(H)
        m[i] = (H[i] >= min_dim)
    end
    return m
end

"""
    support_vertices(H; min_dim=1)

Return the list of region indices where H[r] >= min_dim.
"""
support_vertices(H::AbstractVector{<:Integer}; min_dim::Integer=1) =
    findall(support_mask(H; min_dim=min_dim))

"""
    support_vertices(M, pi; min_dim=1)

Compute restricted Hilbert function on pi and return its support vertices.
"""
function support_vertices(M::PModule{K}, pi; min_dim::Integer = 1) where {K}
    H = restricted_hilbert(M)
    return support_vertices(H; min_dim=min_dim)
end

# Internal: induced components on a mask from an edge dictionary adjacency.
function _masked_components_from_edges(n::Int, edges::Dict{Tuple{Int,Int},<:Real}, mask::BitVector)
    nbrs = Dict{Int, Vector{Int}}()
    for ((a,b), _) in edges
        a == b && continue
        if mask[a] && mask[b]
            push!(get!(nbrs, a, Int[]), b)
            push!(get!(nbrs, b, Int[]), a)
        end
    end

    visited = falses(n)
    comps = Vector{Vector{Int}}()

    for v in 1:n
        if mask[v] && !visited[v]
            stack = Int[v]
            visited[v] = true
            comp = Int[]
            while !isempty(stack)
                u = pop!(stack)
                push!(comp, u)
                for w in get(nbrs, u, Int[])
                    if mask[w] && !visited[w]
                        visited[w] = true
                        push!(stack, w)
                    end
                end
            end
            sort!(comp)
            push!(comps, comp)
        end
    end
    return comps
end

"""
    support_components(H, pi, opts::InvariantOptions; min_dim=1, adjacency=nothing)

Compute connected components of the support mask on the region adjacency graph.

Opts usage:
- `opts.box` chooses the working window (default `:auto` when unset).
- `opts.strict` controls region computations (defaults to true).

If `adjacency` is not provided, this calls `region_adjacency(pi; box=...)`.

Returns a typed `SupportComponentsSummary`.
"""
function support_components(H::HilbertFunction, pi::PLikeEncodingMap, opts::InvariantOptions;
    min_dim::Integer = 1,
    adjacency = nothing)

    strict0 = opts.strict === nothing ? true : opts.strict
    box0 = opts.box === nothing ? :auto : opts.box

    mask = support_mask(H; min_dim=min_dim)

    adj = (adjacency === nothing ? region_adjacency(pi; box=box0, strict=strict0) : adjacency)
    comps = _masked_components_from_edges(length(mask), adj, mask)
    return SupportComponentsSummary(comps, [length(comp) for comp in comps])
end

"""
    support_components(M, pi, opts::InvariantOptions; min_dim=1, adjacency=nothing, kwargs...)

Convenience overload that first computes `restricted_hilbert(M; pi=pi, kwargs...)`.
"""
function support_components(M::PModule{K}, pi::PLikeEncodingMap, opts::InvariantOptions;
    min_dim::Integer = 1,
    adjacency = nothing,
    kwargs...) where {K}

    isempty(kwargs) || throw(ArgumentError("support_components(::PModule{K}, ...): keyword arguments are not supported; pass invariant options via opts"))
    H = restricted_hilbert(M)
    return support_components(H, pi, opts; min_dim=min_dim, adjacency=adjacency)
end

# Graph diameter helpers (exact for small components, double-sweep approx for large).
function _bfs_eccentricity(nbrs::Dict{Int,Vector{Int}}, start::Int, allowed::BitVector)
    q = Int[start]
    dist = Dict{Int,Int}(start => 0)
    head = 1
    maxd = 0
    while head <= length(q)
        v = q[head]; head += 1
        dv = dist[v]
        maxd = max(maxd, dv)
        for w in get(nbrs, v, Int[])
            if allowed[w] && !haskey(dist, w)
                dist[w] = dv + 1
                push!(q, w)
            end
        end
    end
    return maxd
end

"""
    support_graph_diameter(H, pi, opts::InvariantOptions; min_dim=1, adjacency=nothing)

Compute the diameter of the support graph (support mask restricted to region adjacency).

Opts usage:
- `opts.box` chooses the working window (default `:auto` when unset).
- `opts.strict` controls region computations (defaults to true).
"""
function support_graph_diameter(H::HilbertFunction, pi::PLikeEncodingMap, opts::InvariantOptions;
    min_dim::Integer = 1,
    adjacency = nothing)

    strict0 = opts.strict === nothing ? true : opts.strict
    box0 = opts.box === nothing ? :auto : opts.box

    mask = support_mask(H; min_dim=min_dim)
    adj = (adjacency === nothing ? region_adjacency(pi; box=box0, strict=strict0) : adjacency)
    comps = _masked_components_from_edges(length(mask), adj, mask)

    isempty(comps) && return SupportGraphDiameterSummary((
        component_sizes = Int[],
        component_diameters = Int[],
        overall_diameter = 0,
    ))

    # Build neighbor lists once and compute exact eccentricities per component.
    nbrs = Dict{Int, Vector{Int}}()
    for ((a, b), _) in adj
        a == b && continue
        if mask[a] && mask[b]
            push!(get!(nbrs, a, Int[]), b)
            push!(get!(nbrs, b, Int[]), a)
        end
    end

    diams = zeros(Int, length(comps))
    for (i, comp) in enumerate(comps)
        maxd = 0
        for v in comp
            d = _bfs_eccentricity(nbrs, v, mask)
            if d > maxd
                maxd = d
            end
        end
        diams[i] = maxd
    end
    return SupportGraphDiameterSummary((
        component_sizes = [length(comp) for comp in comps],
        component_diameters = diams,
        overall_diameter = maximum(diams),
    ))
end

"""
    support_graph_diameter(M, pi, opts::InvariantOptions; min_dim=1, adjacency=nothing, kwargs...)

Convenience overload that first computes `restricted_hilbert`.
"""
function support_graph_diameter(M::PModule{K}, pi::PLikeEncodingMap, opts::InvariantOptions;
    min_dim::Integer = 1,
    adjacency = nothing,
    kwargs...) where {K}

    isempty(kwargs) || throw(ArgumentError("support_graph_diameter(::PModule{K}, ...): keyword arguments are not supported; pass invariant options via opts"))
    H = restricted_hilbert(M)
    return support_graph_diameter(H, pi, opts; min_dim=min_dim, adjacency=adjacency)
end


"""
    support_bbox(M, pi, opts::InvariantOptions; sep=0.0, min_dim=1, kwargs...) -> SupportBoundingBox

Compute an axis-aligned bounding box of the *support* of a module (where Hilbert mass is nonzero).

Opts usage:
- `opts.box` selects the working window. If unset (`nothing`), we use the default `:auto`.
- `opts.strict` controls region computations (defaults to true).
"""
function support_bbox(M::PModule{K}, pi::PLikeEncodingMap, opts::InvariantOptions;
    sep::Real = 0.0,
    min_dim::Integer = 1,
    kwargs...)::Tuple{Vector{Float64}, Vector{Float64}} where {K}

    strict0 = opts.strict === nothing ? true : opts.strict
    box0 = opts.box === nothing ? :auto : opts.box

    isempty(kwargs) || throw(ArgumentError("support_bbox(::PModule{K}, ...): keyword arguments are not supported; pass invariant options via opts"))
    H = restricted_hilbert(M)
    w = region_weights(pi; box=box0, strict=strict0)

    return support_bbox(H, pi, opts; weights=w, sep=sep, min_dim=min_dim)
end

"""
    support_bbox(H, pi, opts::InvariantOptions; weights=nothing, sep=0.0, min_dim=1) -> SupportBoundingBox

Support bbox from a Hilbert function directly. `weights` may be provided to avoid recomputation.
"""
function _support_bbox_from_mask_weights(mask::BitVector, w, pi::PLikeEncodingMap, bb;
    sep::Real = 0.0)
    ell, u = bb
    lo = fill(Inf, length(ell))
    hi = fill(-Inf, length(u))

    @inbounds for rid in eachindex(mask, w)
        mask[rid] || continue
        w[rid] == 0 && continue
        # Use per-region bboxes so backend-specific geometry is respected.
        ell_r, u_r = region_bbox(pi, rid; box=bb)
        for i in eachindex(lo)
            xlo = float(ell_r[i])
            xhi = float(u_r[i])
            if xlo < lo[i]
                lo[i] = xlo
            end
            if xhi > hi[i]
                hi[i] = xhi
            end
        end
    end

    # If support is empty, fall back to the working region bbox.
    if any(isinf, lo) || any(isinf, hi)
        lo = float.(ell)
        hi = float.(u)
    end

    if sep != 0.0
        lo .-= sep
        hi .+= sep
    end

    return SupportBoundingBox((lo = lo, hi = hi))
end

function support_bbox(H::HilbertFunction, pi::PLikeEncodingMap, opts::InvariantOptions;
    weights = nothing,
    sep::Real = 0.0,
    min_dim::Integer = 1)

    strict0 = opts.strict === nothing ? true : opts.strict
    box0 = opts.box === nothing ? :auto : opts.box

    bb = _resolve_box(pi, box0)
    w = (weights === nothing ? region_weights(pi; box=box0, strict=strict0) : weights)
    mask = support_mask(H; min_dim=min_dim)
    return _support_bbox_from_mask_weights(mask, w, pi, bb; sep=sep)
end

"""
    support_geometric_diameter(M::PModule{K}, pi::PLikeEncodingMap, opts::InvariantOptions; metric=:L2, sep=0.0, kwargs...)
    support_geometric_diameter(H::HilbertFunction, pi::PLikeEncodingMap, opts::InvariantOptions; metric=:L2, sep=0.0, min_dim=1) -> Float64
    support_geometric_diameter(dims::AbstractVector{<:Integer}, pi::PLikeEncodingMap, opts::InvariantOptions; metric=:L2, sep=0.0, min_dim=1) -> Float64

Compute a coarse geometric diameter of the support by:
1) computing a support bounding box, then
2) measuring its diameter in the chosen metric.

metric:
- :L2   sqrt(sum((u-ell)^2))
- :Linf max(u-ell)
- :L1   sum(u-ell)

Opts usage:
- `opts.box` chooses the working window (default `:auto` when unset).
- `opts.strict` controls region computations (defaults to true).
"""
function support_geometric_diameter(M::PModule{K}, pi::PLikeEncodingMap, opts::InvariantOptions;
    metric::Symbol = :L2,
    sep::Real = 0.0,
    kwargs...)::Float64 where {K}

    if haskey(kwargs, :box) || haskey(kwargs, :strict) || haskey(kwargs, :threads)
        throw(ArgumentError("support_geometric_diameter: pass box/strict/threads via opts, not kwargs"))
    end

    lo, hi = support_bbox(M, pi, opts; sep=sep, kwargs...)

    if metric == :L2
        return norm(hi .- lo)
    elseif metric == :L1
        return sum(abs.(hi .- lo))
    elseif metric == :Linf
        return maximum(abs.(hi .- lo))
    else
        error("support_geometric_diameter: unsupported metric=$metric (use :L2, :L1, or :Linf)")
    end
end

function support_geometric_diameter(H::HilbertFunction, pi::PLikeEncodingMap, opts::InvariantOptions;
    metric::Symbol = :L2,
    sep::Real = 0.0,
    min_dim::Integer = 1)

    lo, hi = support_bbox(H, pi, opts; sep=sep, min_dim=min_dim)

    if metric == :L2
        return norm(hi .- lo)
    elseif metric == :L1
        return sum(abs.(hi .- lo))
    elseif metric == :Linf
        return maximum(abs.(hi .- lo))
    else
        error("support_geometric_diameter: unsupported metric=$metric (use :L2, :L1, or :Linf)")
    end
end

"""
    support_measure_stats(M::PModule{K}, pi::PLikeEncodingMap, opts::InvariantOptions; sep=0.0, min_dim=1) -> SupportMeasureSummary
    support_measure_stats(H::FringeModule{K}, pi::PLikeEncodingMap, opts::InvariantOptions; sep=0.0, min_dim=1) -> SupportMeasureSummary
    support_measure_stats(dims::AbstractVector{<:Integer}, pi::PLikeEncodingMap, opts::InvariantOptions; sep=0.0, min_dim=1) -> SupportMeasureSummary

Compute simple summary statistics of the support measure on a working window.

The `FringeModule` and `dims` overloads use `restricted_hilbert` to obtain the
dimension values per region.

Support measure with uncertainty: calls `region_weights(...; return_info=true)` if
available and returns (estimate, stderr, ci, info).

For exact backends, stderr=0 and ci=(estimate,estimate).
For Monte Carlo/sample backends, uses binomial Wilson interval on the *subset* count.

Opts usage:
- `opts.box` chooses the working window (default `:auto` when unset).
- `opts.strict` controls region computations (defaults to true).

Returns a typed `SupportMeasureSummary` with fields like:
- `estimate`, `stderr`, `ci`, `info`,
- `total_measure`, `support_measure`, `support_fraction`,
- and basic bbox summaries.
"""
# Shared implementation for uniform behavior across module/dims inputs.
function _support_measure_stats_impl(H::AbstractVector{<:Integer}, pi::PLikeEncodingMap, opts::InvariantOptions;
    sep::Real = 0.0,
    min_dim::Int = 1)

    strict0 = opts.strict === nothing ? true : opts.strict
    box0 = opts.box === nothing ? :auto : opts.box

    info = nothing
    w = nothing
    # Prefer return_info when supported to expose uncertainty fields.
    try
        info = region_weights(pi; box=box0, strict=strict0, return_info=true)
        w = info.weights
    catch e
        if !(e isa MethodError)
            rethrow()
        end
    end
    if w === nothing
        w = region_weights(pi; box=box0, strict=strict0)
    end

    mask = support_mask(H; min_dim=min_dim)
    support_measure = 0.0
    total_measure = 0.0
    @inbounds for rid in eachindex(mask, w)
        wi = float(w[rid])
        total_measure += wi
        mask[rid] || continue
        support_measure += wi
    end

    bbox = _support_bbox_from_mask_weights(mask, w, pi, _resolve_box(pi, box0); sep=sep)
    lo, hi = bbox

    estimate = support_measure
    stderr = 0.0
    ci = (estimate, estimate)

    if info !== nothing
        counts = (haskey(info, :counts) ? info.counts : nothing)
        nsamples = (haskey(info, :nsamples) ? info.nsamples : 0)
        if counts !== nothing && nsamples > 0
            subset = 0
            for (rid, keep) in pairs(mask)
                keep || continue
                subset += counts[rid]
            end
            alpha = (haskey(info, :alpha) ? info.alpha : 0.05)
            (plo, phi) = _wilson_interval(subset, nsamples; alpha=alpha)
            total = if haskey(info, :total_volume)
                float(info.total_volume)
            elseif haskey(info, :total_points)
                float(info.total_points)
            else
                NaN
            end
            if isfinite(total)
                stderr = total * sqrt((subset / nsamples) * (1 - subset / nsamples) / nsamples)
                ci = (total * plo, total * phi)
            end
        end
    end

    return SupportMeasureSummary((
        estimate = estimate,
        stderr = stderr,
        ci = ci,
        info = info,
        total_measure = total_measure,
        support_measure = support_measure,
        support_fraction = (total_measure == 0.0 ? 0.0 : support_measure / total_measure),
        support_bbox = bbox,
        support_bbox_diameter = norm(hi .- lo),
    ))
end

function support_measure_stats(M::PModule{K}, pi::PLikeEncodingMap, opts::InvariantOptions;
    sep::Real = 0.0,
    min_dim::Int = 1,
    kwargs...) where {K}

    isempty(kwargs) || throw(ArgumentError("support_measure_stats(::PModule{K}, ...): keyword arguments are not supported; pass invariant options via opts"))
    return _support_measure_stats_impl(restricted_hilbert(M), pi, opts; sep=sep, min_dim=min_dim)
end

function support_measure_stats(H::FringeModule{K}, pi::PLikeEncodingMap, opts::InvariantOptions;
    sep::Real = 0.0,
    min_dim::Int = 1,
    kwargs...) where {K}

    isempty(kwargs) || throw(ArgumentError("support_measure_stats(::FringeModule{K}, ...): keyword arguments are not supported; pass invariant options via opts"))
    return _support_measure_stats_impl(restricted_hilbert(H), pi, opts; sep=sep, min_dim=min_dim)
end

function support_measure_stats(dims::AbstractVector{<:Integer}, pi::PLikeEncodingMap, opts::InvariantOptions;
    sep::Real = 0.0,
    min_dim::Int = 1,
    kwargs...)

    isempty(kwargs) || throw(ArgumentError("support_measure_stats(dims, ...): keyword arguments are not supported; pass invariant options via opts"))
    return _support_measure_stats_impl(restricted_hilbert(dims), pi, opts; sep=sep, min_dim=min_dim)
end
