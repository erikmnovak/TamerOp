"""
    RegionGeometry

Geometry-related extension points and generic geometry utilities for encoding maps.

This module intentionally lives outside `CoreModules` to keep `CoreModules` a small
project prelude (scalars, tiny utilities, and core interface hooks).

There are two distinct audiences here:

- Backend implementers extend the low-level hook functions such as
  `region_weights`, `region_bbox`, `region_adjacency`,
  `region_boundary_measure`, and `region_chebyshev_ball` for a concrete
  encoding-map type.
- End users usually call the derived region-geometry queries such as
  `region_volume`, `region_diameter`, `region_aspect_ratio`,
  `region_isoperimetric_ratio`, `region_anisotropy_scores`,
  `check_region_geometry`, and `region_geometry_summary`.

Design:
- Backends extend hook functions such as `region_weights`, `region_bbox`,
  `region_adjacency`, and `region_boundary_measure`.
- This module provides generic wrappers and derived quantities (diameter,
  aspect ratio, isoperimetric ratio, anisotropy scores, etc.) built on those hooks.

Note:
- The classifier hook `locate(pi, x)` lives in `EncodingCore` and is imported here
  for algorithms that need point membership queries.
"""
module RegionGeometry

using LinearAlgebra
using Random

using ..CoreModules: EncodingCache, GeometryCachePayload
import ..EncodingCore: locate, locate_many!, CompiledEncoding,
                       _geometry_fingerprint, _locate_call_style

# Internal runtime gates/thresholds. Keep only knobs that remain meaningful
# runtime policy choices; benchmark-era always-on branches should be collapsed.
const _REGION_BATCHED_LOCATE = Ref(true)
const _REGION_FAST_WRAPPERS = Ref(true)
const _REGION_BATCHED_LOCATE_MIN_PROPOSALS = Ref(128)
const _REGION_LOCATE_BATCH_SIZE = Ref(256)
const _REGION_DIRECT_VOLUME = Ref(true)
const _REGION_WORKSPACE_REUSE = Ref(true)
const _REGION_BLOCKED_PROJECTION_MIN_ACCEPTED = Ref(8)
const _REGION_SAMPLED_SUMMARY_CACHE = Ref(true)
const _REGION_SAMPLE_CACHE_MAX = Ref(256)
const _REGION_CACHE_SHARDS = 16

"""
    RegionGeometryValidationSummary

Notebook/REPL-friendly wrapper for reports returned by `check_region_geometry`.

Use `region_geometry_validation_summary(report)` to pretty-print backend support
for region-geometry hooks and derived user-facing queries.
"""
struct RegionGeometryValidationSummary{R}
    report::R
end

"""
    RegionGeometrySummary

Typed exploratory wrapper returned by `region_geometry_summary(pi, r; ...)`.

This is the preferred owner-local summary object for quick region inspection.
It records:
- which quantities were available,
- which quantities were unavailable and why,
- whether a finite box was used,
- whether each reported quantity came from a bbox-based, hook-based, or
  sampling-based computation path.

Use semantic accessors such as `region_bbox(summary)`, `region_volume(summary)`,
and `region_boundary(summary)` instead of inspecting the raw report directly.
"""
struct RegionGeometrySummary{R}
    report::R
end

"""
    RegionPrincipalDirectionsSummary

Typed exploratory wrapper for principal-direction and covariance summaries of a
single region.

This wraps the richer result produced by `region_principal_directions_summary`
and provides a stable owner-local inspection surface for:
- principal values/eigenvalues,
- principal vectors/eigenvectors,
- covariance matrix,
- anisotropy scores derived from the same covariance estimate.
"""
struct RegionPrincipalDirectionsSummary{R}
    report::R
end

"""
    region_geometry_validation_summary(report) -> RegionGeometryValidationSummary

Wrap the raw `NamedTuple` report returned by `check_region_geometry(pi)` in a
plain-text display helper.

This is the preferred presentation layer for notebooks and the REPL. Keep the raw
report when you want programmatic access to booleans such as
`report.hooks.boundary_measure` or `report.queries.region_circumradius`.
"""
@inline region_geometry_validation_summary(report::NamedTuple) = RegionGeometryValidationSummary(report)

"""
    region_bbox(summary::RegionGeometrySummary)

Return the bounding box stored in a `RegionGeometrySummary`, or `nothing` when
it was unavailable.
"""
@inline region_bbox(summary::RegionGeometrySummary) = summary.report.bbox

"""
    region_volume(summary::RegionGeometrySummary)

Return the region volume stored in a `RegionGeometrySummary`, or `nothing` when
it was unavailable.
"""
@inline region_volume(summary::RegionGeometrySummary) = summary.report.volume

"""
    region_boundary(summary::RegionGeometrySummary)

Return the boundary measure stored in a `RegionGeometrySummary`, or `nothing`
when it was unavailable.
"""
@inline region_boundary(summary::RegionGeometrySummary) = summary.report.boundary_measure

"""
    available_quantities(summary::RegionGeometrySummary)

Return the names of the quantities that were successfully computed in
`region_geometry_summary(...)`.
"""
@inline available_quantities(summary::RegionGeometrySummary) = summary.report.available

"""
    unavailable_quantities(summary::RegionGeometrySummary)

Return `(name => reason)` pairs for quantities that were unavailable in
`region_geometry_summary(...)`.
"""
@inline unavailable_quantities(summary::RegionGeometrySummary) = summary.report.unavailable

"""
    quantity_sources(summary::RegionGeometrySummary)

Return a `NamedTuple` describing how each quantity is computed:
- `:bbox_based`
- `:hook_based`
- `:sampling_based`
- `:diagnostic_heavy`
"""
@inline quantity_sources(summary::RegionGeometrySummary) = summary.report.quantity_sources

"""
    finite_box_used(summary::RegionGeometrySummary)

Return `true` when `region_geometry_summary(...)` evaluated the region inside a
finite `box=(a,b)`.
"""
@inline finite_box_used(summary::RegionGeometrySummary) = summary.report.finite_box

"""
    principal_values(summary::RegionPrincipalDirectionsSummary)

Return the principal values (eigenvalues of the covariance matrix), sorted in
descending order.
"""
@inline principal_values(summary::RegionPrincipalDirectionsSummary) = summary.report.evals

"""
    principal_vectors(summary::RegionPrincipalDirectionsSummary)

Return the principal directions as a matrix whose columns are the eigenvectors
matching `principal_values(summary)`.
"""
@inline principal_vectors(summary::RegionPrincipalDirectionsSummary) = summary.report.evecs

"""
    covariance_matrix(summary::RegionPrincipalDirectionsSummary)

Return the covariance matrix estimated for the sampled region.
"""
@inline covariance_matrix(summary::RegionPrincipalDirectionsSummary) = summary.report.cov

@inline _region_backend_type(pi) = typeof(pi)

function _unsupported_region_geometry_message(fname::Symbol, pi;
    note::AbstractString="",
    fallback::AbstractString="")
    msg = string(
        fname,
        " is not available for backend ",
        _region_backend_type(pi),
        ". Call `check_region_geometry(pi)` to inspect supported hooks and derived queries for this backend."
    )
    isempty(fallback) || (msg *= " " * fallback)
    isempty(note) || (msg *= " " * note)
    msg *= " Backend implementers should add a typed RegionGeometry method in the owning encoding/backend module."
    return msg
end

# -----------------------------------------------------------------------------
"""
    region_weights(pi; box=nothing, kwargs...) -> AbstractVector

Optional hook for weighting invariants computed on a finite encoding.

Backend implementers add methods here. End users usually prefer the derived
queries `region_volume(...)` or `region_geometry_summary(...)` unless they
specifically need the full weight vector.

Many invariants in multiparameter persistence reduce, after choosing a finite
encoding map `pi : Q -> P`, to data indexed by regions/vertices in a finite
poset `P`. When comparing two modules via the restricted Hilbert function
(dimension surface), it is often useful to weight each region by its "size"
(e.g. volume in R^n, or lattice-point count in Z^n) inside a bounding box.

This function is an *extension point* (like `locate`). Concrete encoding map
types may provide methods.

Common conventions:
- `region_weights(pi; box=(a,b))` returns a vector `w` with `w[t] >= 0`.
- `length(w)` should equal the number of encoded regions (i.e. `P.n`).
- If no meaningful weighting is available, a method may return all ones.
"""
function region_weights(pi; box=nothing, kwargs...)
    throw(ArgumentError(_unsupported_region_geometry_message(
        :region_weights,
        pi;
        fallback="If you only need a support check, use `check_region_geometry(pi)` first."
    )))
end

"""
    region_bbox(pi, r; box=nothing, kwargs...)

Geometric hook: return a bounding box for region `r` of an encoding map `pi`.

Backend implementers add methods here. End users usually call higher-level
queries such as `region_diameter`, `region_widths`, `region_centroid`, or
`region_geometry_summary`, all of which use `region_bbox` when available.

Intended meaning:
- For encodings of R^n: return an axis-aligned bounding box for the region in R^n.
- Many regions are naturally unbounded; in that case, you should either:
  * require a user-supplied finite `box=(a,b)` and compute the bbox of (region intersect box), or
  * return `-Inf` / `Inf` in unbounded directions.

Return convention:
- Return `nothing` if the region has empty intersection with the supplied `box`
  (or is otherwise empty for the intended semantics).
- Otherwise return `(a, b)` where `a[i] <= b[i]` are the lower/upper corners.

This is an extension point: concrete encoding map types may provide methods.
"""
function region_bbox(pi, r::Integer; box=nothing, kwargs...)
    throw(ArgumentError(_unsupported_region_geometry_message(
        :region_bbox,
        pi;
        fallback="Many downstream geometry queries fall back to `region_bbox`, so this is one of the most valuable hooks for backend implementers."
    )))
end

"""
    region_volume(pi, r; box=nothing, kwargs...) -> Real

Geometric hook: return the volume/weight of a single region.

Concrete backends are encouraged to implement this directly when a single-region
query is materially cheaper than building the full vector returned by
`region_weights`. The default fallback still extracts the `r`-th entry from
`region_weights`.
"""
function region_volume(pi, r::Integer; box=nothing, closure::Bool=true, cache=nothing, kwargs...)
    box = _resolve_box(box, cache)
    if _REGION_DIRECT_VOLUME[]
        fast = _region_volume_fast(pi, r; box=box, closure=closure, cache=cache)
        fast === nothing || return float(fast)
    end
    return _region_volume_from_weights(pi, r; box=box, closure=closure, cache=cache, kwargs...)
end

"""
    region_diameter(pi, r; metric=:L2, box=nothing, method=:bbox, kwargs...) -> Real

Estimate the diameter of region `r` under the chosen metric.

Default implementation (`method == :bbox`):
Compute the diameter of the bounding box returned by
`region_bbox(pi, r; box=box, kwargs...)`.

This default is conservative: it is an upper bound on the true diameter inside
the box (if the region is non-convex, disconnected, etc.).

Metrics supported by the default method:
- `:L2`   (Euclidean)
- `:Linf` (max norm)
- `:L1`   (taxicab)

Concrete encodings may provide specialized methods, e.g. a vertex-based diameter
for convex polyhedral regions.

Cost profile:
- cheap / bbox-based by default,
- typically one of the cheapest geometric summaries once `region_bbox` is available.
"""
function region_diameter(
    pi,
    r::Integer;
    metric::Symbol=:L2,
    box=nothing,
    method::Symbol=:bbox,
    kwargs...
)
    method == :bbox || error("region_diameter: method must be :bbox (or implement a specialized method)")

    bb = region_bbox(pi, r; box=box, kwargs...)
    bb === nothing && return 0.0

    a, b = bb
    length(a) == length(b) || error("region_diameter: region_bbox returned mismatched endpoints")
    n = length(a)

    if metric == :Linf
        d = 0.0
        for i in 1:n
            li = float(b[i]) - float(a[i])
            isfinite(li) || return Inf
            d = max(d, abs(li))
        end
        return d
    elseif metric == :L2
        acc = 0.0
        for i in 1:n
            li = float(b[i]) - float(a[i])
            isfinite(li) || return Inf
            acc += li * li
        end
        return sqrt(acc)
    elseif metric == :L1
        s = 0.0
        for i in 1:n
            li = float(b[i]) - float(a[i])
            isfinite(li) || return Inf
            s += abs(li)
        end
        return s
    else
        error("region_diameter: metric must be :L1, :L2, or :Linf")
    end
end

"""
    region_widths(pi, r; box=nothing, kwargs...) -> Union{Nothing,Vector{Float64}}

Side lengths of the axis-aligned bounding box of region `r`.

This is computed from `region_bbox(pi, r; box=box, kwargs...)` as `b - a`,
where `(a,b)` is the returned bounding box.

Returns `nothing` if `region_bbox` returns `nothing` (empty intersection).

Note:
- If you do not supply a finite `box`, some coordinates may be unbounded and
  the corresponding widths may be `Inf`.
"""
function region_widths(pi, r::Integer; box=nothing, kwargs...)
    bb = region_bbox(pi, r; box=box, kwargs...)
    bb === nothing && return nothing
    a, b = bb
    length(a) == length(b) || error("region_widths: region_bbox returned mismatched endpoints")
    n = length(a)
    w = Vector{Float64}(undef, n)
    for i in 1:n
        w[i] = float(b[i]) - float(a[i])
    end
    return w
end

"""
    region_centroid(pi, r; box=nothing, method=:bbox, kwargs...) -> Union{Nothing,Vector{Float64}}

A convenient representative point ("centroid") for region `r`.

By default (`method == :bbox`), this returns the midpoint of the bounding box
returned by `region_bbox(pi, r; box=box, kwargs...)`.

Returns `nothing` if:
- the region does not intersect the box, or
- the bounding box is unbounded in some coordinate (supply a finite `box`).

Encodings may provide more refined notions (e.g. true centroid, Monte Carlo
centroid) by adding specialized methods.
"""
function region_centroid(pi, r::Integer; box=nothing, method::Symbol=:bbox, kwargs...)
    method == :bbox || error("region_centroid: unsupported method $method (default is :bbox)")
    cache = _kwget(kwargs, :cache, nothing)
    closure = _kwget(kwargs, :closure, true)
    box = _resolve_box(box, cache)
    return _region_centroid_maybe_cached(pi, r; box=box, method=:bbox, closure=closure, cache=cache)
end

"""
    region_aspect_ratio(pi, r; box=nothing, kwargs...) -> Real

A simple anisotropy proxy for region `r` based on its axis-aligned bounding box.

Let `w = region_widths(pi, r; ...)`. This function returns:

    max(w) / min_positive(w)

where `min_positive(w)` is the smallest strictly positive width.

Conventions:
- Returns 0.0 if the region has empty intersection with the given box.
- Returns 1.0 if the region collapses to a point (all widths are 0).
- Returns Inf if the region is unbounded (some width is Inf) or degenerate
  (some widths are 0 while others are positive).
"""
function region_aspect_ratio(pi, r::Integer; box=nothing, kwargs...)
    w = region_widths(pi, r; box=box, kwargs...)
    w === nothing && return 0.0

    maxw = 0.0
    minw = Inf
    for wi in w
        aw = abs(float(wi))
        isfinite(aw) || return Inf
        maxw = max(maxw, aw)
        if aw > 0.0
            minw = min(minw, aw)
        end
    end

    if maxw == 0.0
        return 1.0
    elseif minw == Inf
        return Inf
    else
        return maxw / minw
    end
end

"""
    region_adjacency(pi; box, kwargs...) -> AbstractDict

Geometric hook: report how regions meet inside a finite bounding window.

Backend implementers add methods here. End users usually call this directly
only when they specifically need adjacency-driven invariants.

Implementations should return a dictionary whose keys are unordered region pairs
`(r,s)` with `r < s`. The value is the estimated (n-1)-dimensional measure of
the interface between regions `r` and `s` inside `box=(a,b)`.

Notes:
- For n=2 this is (a notion of) boundary length.
- For n=3 this is boundary area.
- For n=1 this is a 0-dimensional measure (counting boundary points), so each
  interior boundary typically contributes 1.

This is currently implemented for axis-aligned box grids (`PLEncodingMapBoxes`).
"""
function region_adjacency(pi; box=nothing, kwargs...)
    throw(ArgumentError(_unsupported_region_geometry_message(
        :region_adjacency,
        pi;
        fallback="If you only need scalar shape statistics, many of them can still work without adjacency support."
    )))
end

"""
    region_facet_count(pi, r; kwargs...) -> Int

Geometric hook: return a facet/constraint count for region `r`.

This is primarily a backend/diagnostic hook rather than a first-line end-user
query.

For polyhedral backends, a reasonable default is the number of inequalities in
the stored H-representation. This is a complexity proxy (it may count redundant
inequalities and thus may exceed the true number of facets).
"""
function region_facet_count(pi, r::Integer; kwargs...)
    throw(ArgumentError(_unsupported_region_geometry_message(:region_facet_count, pi)))
end

"""
    region_vertex_count(pi, r; box, kwargs...) -> Union{Nothing,Int}

Geometric hook: return the number of vertices of `region r` inside the window
`box=(a,b)`.

This is primarily a backend/diagnostic hook rather than a first-line end-user
query.

This may be expensive. Implementations are allowed to return `nothing` if vertex
enumeration is not attempted (e.g. too many constraints / combinations).
"""
function region_vertex_count(pi, r::Integer; box=nothing, kwargs...)
    throw(ArgumentError(_unsupported_region_geometry_message(:region_vertex_count, pi)))
end


"""
    region_boundary_measure(pi, r; box=nothing, strict=true)

Boundary measure of region `r` inside the window `box=(a,b)`.

This is the (n-1)-dimensional Hausdorff measure of the boundary of `R_r cap box`:
* n == 2: perimeter
* n == 3: surface area
* general n: hypersurface measure

Backends should implement this when feasible. The default method throws an error.

Cost profile:
- hook-dependent,
- often substantially more expensive than bbox-based queries.
"""
function region_boundary_measure(pi, r; box=nothing, strict=true)
    throw(ArgumentError(_unsupported_region_geometry_message(
        :region_boundary_measure,
        pi;
        fallback="If you only need a coarse size statistic, try `region_volume` or `region_diameter` when those are available."
    )))
end

"""
    region_boundary_measure_breakdown(pi, r; box=nothing, kwargs...) -> Vector{NamedTuple}

Diagnostic geometric hook: a per-facet decomposition of the boundary measure of
region `r` inside the window `box=(a,b)`.

Each entry is a NamedTuple and should contain at least:
- `measure::Float64`

Backends may optionally provide additional fields such as:
- `normal::Vector{Float64}`: an (outward) normal of the facet (not necessarily unit).
- `point::Vector{Float64}`: a representative point on the facet (e.g. barycenter).
- `neighbor::Union{Nothing,Int}`: the neighboring region across the facet, if detectable.
- `kind::Symbol`: e.g. `:internal` (between regions) or `:box` (window boundary).

The total boundary measure should satisfy, approximately:

    region_boundary_measure(pi, r; box=box) approx sum(e.measure for e in breakdown)

This is a diagnostic-level query and may not be implemented for all encodings.

Cost profile:
- diagnostic / heavy,
- typically more expensive than the scalar `region_boundary_measure`.
"""
function region_boundary_measure_breakdown(pi, r; box=nothing, kwargs...)
    throw(ArgumentError(_unsupported_region_geometry_message(
        :region_boundary_measure_breakdown,
        pi;
        fallback="This is a diagnostic-level query; many backends only implement the scalar `region_boundary_measure`."
    )))
end


# --- helper hooks ------------------------------------------------------------

# Owner backends may provide a concrete cached box and direct cached geometry
# answers. RegionGeometry keeps the default path generic and only uses these
# hooks when a backend opts in with a typed method.
@inline _cache_box(::Nothing) = nothing
@inline _cache_box(cache) = nothing
@inline _resolve_box(box, cache) = box === nothing ? _cache_box(cache) : box

@inline _region_bbox_fast(pi, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache=nothing) = nothing
@inline _region_centroid_fast(pi, r::Integer;
    box, method::Symbol=:bbox, closure::Bool=true, cache=nothing) = nothing
@inline _region_volume_fast(pi, r::Integer; box, closure::Bool=true, cache=nothing) = nothing
@inline _region_boundary_measure_fast(pi, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache=nothing) = nothing
@inline _region_circumradius_fast(pi, r::Integer;
    box, center=:bbox, metric::Symbol=:L2, method::Symbol=:bbox,
    strict::Bool=true, closure::Bool=true, cache=nothing) = nothing
@inline _region_minkowski_functionals_fast(pi, r::Integer;
    box, volume=nothing, boundary=nothing, mean_width_method::Symbol=:auto,
    mean_width_ndirs::Integer=256, mean_width_rng=Random.default_rng(),
    mean_width_directions=nothing, strict::Bool=true, closure::Bool=true,
    cache=nothing) = nothing
@inline _region_geometry_summary_fast(pi, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache=nothing,
    mean_width_method::Symbol=:auto, mean_width_ndirs::Integer=256,
    mean_width_rng=Random.default_rng(), mean_width_directions=nothing,
    need_mean_width::Bool=false) = nothing
@inline _region_weights_cached(pi; box, closure::Bool=true, cache=nothing, kwargs...) = nothing
@inline _region_weights_closure(pi; box, closure::Bool=true, kwargs...) = nothing
@inline _region_boundary_measure_cached(pi, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache=nothing) = nothing
@inline _region_boundary_measure_closure(pi, r::Integer;
    box, strict::Bool=true, closure::Bool=true) = nothing
@inline _region_boundary_measure_strict(pi, r::Integer; box, strict::Bool=true) = nothing
@inline _region_bbox_cached(pi, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache=nothing) = nothing
@inline _region_bbox_closure(pi, r::Integer;
    box, strict::Bool=true, closure::Bool=true) = nothing
@inline _region_bbox_strict(pi, r::Integer; box, strict::Bool=true) = nothing
@inline _region_centroid_cached(pi, r::Integer;
    box, method::Symbol=:bbox, closure::Bool=true, cache=nothing) = nothing
@inline _region_centroid_closure(pi, r::Integer;
    box, method::Symbol=:bbox, closure::Bool=true) = nothing

const _REGION_WEIGHTS_FALLBACK = which(region_weights, Tuple{Any})
const _REGION_BBOX_FALLBACK = which(region_bbox, Tuple{Any, Int})
const _REGION_ADJACENCY_FALLBACK = which(region_adjacency, Tuple{Any})
const _REGION_FACET_COUNT_FALLBACK = which(region_facet_count, Tuple{Any, Int})
const _REGION_VERTEX_COUNT_FALLBACK = which(region_vertex_count, Tuple{Any, Int})
const _REGION_BOUNDARY_MEASURE_FALLBACK = which(region_boundary_measure, Tuple{Any, Int})
const _REGION_BOUNDARY_BREAKDOWN_FALLBACK = which(region_boundary_measure_breakdown, Tuple{Any, Int})
const _REGION_VOLUME_FAST_FALLBACK = which(_region_volume_fast, Tuple{Any, Int})
const _REGION_CENTROID_FAST_FALLBACK = which(_region_centroid_fast, Tuple{Any, Int})
const _REGION_BOUNDARY_FAST_FALLBACK = which(_region_boundary_measure_fast, Tuple{Any, Int})
const _REGION_CIRCUMRADIUS_FAST_FALLBACK = which(_region_circumradius_fast, Tuple{Any, Int})
const _REGION_MINKOWSKI_FAST_FALLBACK = which(_region_minkowski_functionals_fast, Tuple{Any, Int})
const _REGION_GEOMETRY_SUMMARY_FAST_FALLBACK = which(_region_geometry_summary_fast, Tuple{Any, Int})

@inline function _region_has_custom_method(f, fallback::Method, argtypes::Type{<:Tuple})
    return which(f, argtypes) !== fallback
end

@inline _region_supports_weights(pi) = _region_has_custom_method(region_weights, _REGION_WEIGHTS_FALLBACK, Tuple{typeof(pi)})
@inline _region_supports_bbox(pi) = _region_has_custom_method(region_bbox, _REGION_BBOX_FALLBACK, Tuple{typeof(pi), Int})
@inline _region_supports_adjacency(pi) = _region_has_custom_method(region_adjacency, _REGION_ADJACENCY_FALLBACK, Tuple{typeof(pi)})
@inline _region_supports_facet_count(pi) = _region_has_custom_method(region_facet_count, _REGION_FACET_COUNT_FALLBACK, Tuple{typeof(pi), Int})
@inline _region_supports_vertex_count(pi) = _region_has_custom_method(region_vertex_count, _REGION_VERTEX_COUNT_FALLBACK, Tuple{typeof(pi), Int})
@inline _region_supports_boundary_measure(pi) = _region_has_custom_method(region_boundary_measure, _REGION_BOUNDARY_MEASURE_FALLBACK, Tuple{typeof(pi), Int})
@inline _region_supports_boundary_breakdown(pi) = _region_has_custom_method(region_boundary_measure_breakdown, _REGION_BOUNDARY_BREAKDOWN_FALLBACK, Tuple{typeof(pi), Int})
@inline _region_supports_fast_volume(pi) = _region_has_custom_method(_region_volume_fast, _REGION_VOLUME_FAST_FALLBACK, Tuple{typeof(pi), Int})
@inline _region_supports_fast_centroid(pi) = _region_has_custom_method(_region_centroid_fast, _REGION_CENTROID_FAST_FALLBACK, Tuple{typeof(pi), Int})
@inline _region_supports_fast_boundary_measure(pi) = _region_has_custom_method(_region_boundary_measure_fast, _REGION_BOUNDARY_FAST_FALLBACK, Tuple{typeof(pi), Int})
@inline _region_supports_fast_circumradius(pi) = _region_has_custom_method(_region_circumradius_fast, _REGION_CIRCUMRADIUS_FAST_FALLBACK, Tuple{typeof(pi), Int})
@inline _region_supports_fast_minkowski(pi) = _region_has_custom_method(_region_minkowski_functionals_fast, _REGION_MINKOWSKI_FAST_FALLBACK, Tuple{typeof(pi), Int})
@inline _region_supports_fast_summary(pi) = _region_has_custom_method(_region_geometry_summary_fast, _REGION_GEOMETRY_SUMMARY_FAST_FALLBACK, Tuple{typeof(pi), Int})

@inline _region_anytrue(nt::NamedTuple) = any(values(nt))

@inline function _region_count_hint(pi)
    if hasproperty(pi, :P)
        P = getproperty(pi, :P)
        hasproperty(P, :n) && return Int(getproperty(P, :n))
    end
    hasproperty(pi, :sig_y) && return length(getproperty(pi, :sig_y))
    hasproperty(pi, :regions) && return length(getproperty(pi, :regions))
    return nothing
end

@inline function _region_ambient_dim_hint(pi)
    hasproperty(pi, :n) && return Int(getproperty(pi, :n))
    return nothing
end

@inline function _region_query_requires_box(query::Symbol)
    return query in (
        :summary,
        :region_bbox, :region_widths, :region_diameter, :region_centroid,
        :region_aspect_ratio, :region_boundary_measure, :region_boundary_measure_breakdown,
        :region_adjacency, :region_chebyshev_ball, :region_circumradius,
        :region_boundary_to_volume_ratio, :region_isoperimetric_ratio,
        :region_principal_directions, :region_mean_width,
        :region_minkowski_functionals, :region_covariance_anisotropy,
        :region_covariance_eccentricity, :region_anisotropy_scores,
        :region_perimeter, :region_surface_area,
    )
end

@inline function _supported_region_syms(nt::NamedTuple)
    return [k for (k, v) in pairs(nt) if v]
end

const _REGION_EXTRA_QUERIES = (
    :region_mean_width,
    :region_principal_directions,
    :region_anisotropy_scores,
)

const _REGION_RECOMMENDED_QUERIES = (
    :region_bbox,
    :region_widths,
    :region_volume,
    :region_centroid,
    :region_aspect_ratio,
    :region_boundary_measure,
    :region_circumradius,
)

@inline _region_supports_sampling_queries(pi) =
    (_region_ambient_dim_hint(pi) !== nothing) || _region_anytrue(check_region_geometry(pi).queries)

function _supports_extra_region_query(pi, query::Symbol)
    if query === :region_mean_width
        return _region_supports_sampling_queries(pi)
    elseif query === :region_principal_directions
        return _region_supports_sampling_queries(pi)
    elseif query === :region_anisotropy_scores
        return _region_supports_sampling_queries(pi)
    end
    return false
end

"""
    check_region_geometry(pi; box=nothing, throw=false) -> NamedTuple

Inspect which RegionGeometry hooks and common derived queries are available for
backend `pi`.

This helper is intended for both advanced users and backend implementers:
- End users can call it before a geometry workflow to see which queries are
  actually supported on a given encoding backend.
- Backend implementers can use it as a quick capability report while adding new
  geometry hooks.

The returned report contains:
- `hooks`: support for extension-hook functions such as `region_weights`,
  `region_bbox`, `region_adjacency`, and `region_boundary_measure`
- `fast_paths`: support for optional backend-provided fast wrappers
- `queries`: support for common end-user queries derived from those hooks
- `issues`: actionable guidance when support is partial or missing

Use `region_geometry_validation_summary(report)` for notebook/REPL-friendly
display.
"""
function check_region_geometry(pi; box=nothing, throw::Bool=false)
    hooks = (
        weights = _region_supports_weights(pi),
        bbox = _region_supports_bbox(pi),
        adjacency = _region_supports_adjacency(pi),
        facet_count = _region_supports_facet_count(pi),
        vertex_count = _region_supports_vertex_count(pi),
        boundary_measure = _region_supports_boundary_measure(pi),
        boundary_breakdown = _region_supports_boundary_breakdown(pi),
        chebyshev_ball = _region_supports_chebyshev_ball(pi),
    )
    fast_paths = (
        volume = _region_supports_fast_volume(pi),
        centroid = _region_supports_fast_centroid(pi),
        boundary_measure = _region_supports_fast_boundary_measure(pi),
        circumradius = _region_supports_fast_circumradius(pi),
        minkowski_functionals = _region_supports_fast_minkowski(pi),
        geometry_summary = _region_supports_fast_summary(pi),
    )
    queries = (
        region_volume = hooks.weights || fast_paths.volume || fast_paths.geometry_summary,
        region_bbox = hooks.bbox,
        region_widths = hooks.bbox,
        region_diameter = hooks.bbox,
        region_centroid = hooks.bbox || fast_paths.centroid,
        region_aspect_ratio = hooks.bbox,
        region_boundary_measure = hooks.boundary_measure || fast_paths.boundary_measure || fast_paths.geometry_summary,
        region_boundary_measure_breakdown = hooks.boundary_breakdown,
        region_adjacency = hooks.adjacency,
        region_chebyshev_ball = hooks.chebyshev_ball,
        region_circumradius = hooks.bbox || hooks.chebyshev_ball || fast_paths.circumradius,
        region_boundary_to_volume_ratio = (hooks.weights || fast_paths.volume || fast_paths.geometry_summary) &&
            (hooks.boundary_measure || fast_paths.boundary_measure || fast_paths.geometry_summary),
        region_isoperimetric_ratio = (hooks.weights || fast_paths.volume || fast_paths.geometry_summary) &&
            (hooks.boundary_measure || fast_paths.boundary_measure || fast_paths.geometry_summary),
        region_minkowski_functionals = fast_paths.minkowski_functionals || fast_paths.geometry_summary ||
            ((hooks.weights || fast_paths.volume) && (hooks.boundary_measure || fast_paths.boundary_measure)),
    )

    issues = String[]
    _region_anytrue(hooks) || push!(issues,
        "No RegionGeometry hook is implemented for this backend. End users should expect geometry queries to fail until the backend adds at least one typed hook such as `region_bbox` or `region_weights`.")
    box === nothing && push!(issues,
        "Many region-geometry queries require a finite `box=(a,b)`; support booleans above only indicate whether the backend and generic wrappers know how to answer the query once a box is supplied.")
    hooks.boundary_measure || fast_paths.boundary_measure || fast_paths.geometry_summary || push!(issues,
        "Boundary-based quantities (perimeter, surface area, boundary-to-volume ratio, isoperimetric ratio) are unavailable for this backend.")
    hooks.chebyshev_ball || push!(issues,
        "Inscribed-ball queries are unavailable; if you only need an outer-radius statistic, `region_circumradius(...; method=:bbox)` can still work when `region_bbox` is available.")

    valid = _region_anytrue(queries)
    report = (
        kind = :region_geometry,
        backend = _region_backend_type(pi),
        valid = valid,
        hooks = hooks,
        fast_paths = fast_paths,
        queries = queries,
        issues = issues,
    )
    if throw && !valid
        throw(ArgumentError("check_region_geometry: backend $(_region_backend_type(pi)) does not expose any supported region-geometry query. " *
            (isempty(issues) ? "" : join(issues, " "))))
    end
    return report
end

"""
    supported_region_hooks(pi) -> Vector{Symbol}

Return the supported RegionGeometry extension hooks for backend `pi`.

This is a thin semantic accessor on top of `check_region_geometry(pi)` and is
useful in notebooks when you want a quick list rather than the full report.
"""
function supported_region_hooks(pi)
    return _supported_region_syms(check_region_geometry(pi).hooks)
end

"""
    supported_region_queries(pi) -> Vector{Symbol}

Return the supported user-facing region-geometry queries for backend `pi`.

This is a thin semantic accessor on top of `check_region_geometry(pi)` and is
intended for quick exploratory inspection.
"""
function supported_region_queries(pi)
    syms = Set(_supported_region_syms(check_region_geometry(pi).queries))
    for q in _REGION_EXTRA_QUERIES
        _supports_extra_region_query(pi, q) && push!(syms, q)
    end
    return sort!(collect(syms); by=string)
end

"""
    supports_region_query(pi, query::Symbol) -> Bool

Return `true` if `check_region_geometry(pi)` reports support for the named
region-geometry query.

Use the canonical query names, e.g.
- `:region_bbox`
- `:region_volume`
- `:region_boundary_measure`
- `:region_circumradius`
"""
function supports_region_query(pi, query::Symbol)
    queries = check_region_geometry(pi).queries
    if haskey(queries, query)
        return getproperty(queries, query)
    elseif query in _REGION_EXTRA_QUERIES
        return _supports_extra_region_query(pi, query)
    end
    throw(ArgumentError(
        "supports_region_query: unknown query=$query. Call `supported_region_queries(pi)` to inspect the canonical query names."
    ))
end

"""
    recommended_region_queries(pi) -> Vector{Symbol}

Return a notebook-friendly subset of supported region queries that are usually
the cheapest and safest first inspection calls once a finite `box=(a,b)` is
available.

These are not the only useful queries; they are the recommended first pass for
interactive exploration before calling heavier sampling-based or diagnostic
routines.
"""
function recommended_region_queries(pi)
    supported = Set(supported_region_queries(pi))
    return [q for q in _REGION_RECOMMENDED_QUERIES if q in supported]
end

"""
    check_region_query(pi, r; box=nothing, query=:summary, throw=false) -> NamedTuple

Validate the basic contract for a region-geometry query before calling a heavier
`region_*` routine.

This helper checks:
- whether the region index `r` is plausible when the backend exposes a region count,
- whether a supplied `box=(a,b)` has the expected ambient dimension when that
  dimension is inspectable,
- whether the chosen query class normally requires a finite `box`.

This is a query-shape validator, not a geometry computation. Use it to surface
common user mistakes before entering a heavier workflow.
"""
function check_region_query(pi, r::Integer; box=nothing, query::Symbol=:summary, throw::Bool=false)
    issues = String[]
    nregions = _region_count_hint(pi)
    ambient = _region_ambient_dim_hint(pi)

    if nregions !== nothing && !(1 <= r <= nregions)
        push!(issues, "region index r=$r is out of range; expected 1 <= r <= $nregions.")
    end

    if box !== nothing
        try
            a, b = box
            if ambient !== nothing
                length(a) == ambient || push!(issues, "box lower corner has length $(length(a)), expected ambient dimension $ambient.")
                length(b) == ambient || push!(issues, "box upper corner has length $(length(b)), expected ambient dimension $ambient.")
            elseif length(a) != length(b)
                push!(issues, "box endpoints have mismatched lengths $(length(a)) and $(length(b)).")
            end
        catch err
            push!(issues, "box must have the form `(a,b)` with indexable endpoints. " * sprint(showerror, err))
        end
    elseif _region_query_requires_box(query)
        push!(issues, "query=$query normally requires a finite `box=(a,b)`.")
    end

    if query !== :summary
        try
            supports_region_query(pi, query) || push!(issues,
                "backend $(_region_backend_type(pi)) does not report support for query=$query. Call `check_region_geometry(pi)` for the full capability report.")
        catch err
            push!(issues, sprint(showerror, err))
        end
    end

    report = (
        kind = :region_query,
        backend = _region_backend_type(pi),
        query = query,
        region = Int(r),
        ambient_dim = ambient,
        region_count = nregions,
        valid = isempty(issues),
        issues = issues,
    )
    if throw && !report.valid
        throw(ArgumentError("check_region_query: " * join(issues, " ")))
    end
    return report
end

"""
    region_geometry_summary(pi, r; box=nothing, strict=true, closure=true,
                            cache=nothing, include_breakdown=false, kwargs...) -> RegionGeometrySummary

Return a compact, user-facing summary of region `r` for backend `pi`.

This is a cheap-first convenience wrapper for exploratory work. It collects the
most common scalar/box summaries when they are supported by the backend:
- `volume`
- `bbox`
- `widths`
- `centroid`
- `aspect_ratio`
- `boundary_measure`
- `circumradius`

Unavailable quantities are returned as `nothing`. The typed summary records:
- `available_quantities(summary)`,
- `unavailable_quantities(summary)` with reasons,
- whether a finite `box` was used,
- whether each quantity came from a bbox-based, hook-based, or sampling-based
  computation path.

Best practices:
- supply a finite `box=(a,b)` whenever possible; many geometry queries are only
  meaningful on a bounded window,
- call `check_region_geometry(pi)` first if you are unsure which quantities a
  backend can provide,
- use the individual `region_*` functions when you need one quantity only.

Cost profile:
- cheap-first wrapper,
- it only evaluates supported quantities and records missing/heavy ones in
  `issues`, so it is the preferred entrypoint for interactive exploration.
"""
function region_geometry_summary(pi, r::Integer; box=nothing, strict::Bool=true,
    closure::Bool=true, cache=nothing, include_breakdown::Bool=false, kwargs...)
    report = check_region_geometry(pi; box=box)
    issues = String[]
    q = report.queries
    quantity_sources = (
        volume = :hook_based,
        bbox = :bbox_based,
        widths = :bbox_based,
        centroid = :bbox_based,
        aspect_ratio = :bbox_based,
        boundary_measure = :hook_based,
        circumradius = :bbox_based,
        boundary_breakdown = :diagnostic_heavy,
    )
    unavailable = Pair{Symbol,String}[]
    available = Symbol[]

    box_eff = _resolve_box(box, cache)
    finite_box = box_eff !== nothing
    finite_box || push!(issues,
        "No finite `box=(a,b)` was supplied or recoverable from the cache; box-dependent quantities may be unavailable.")

    function maybe_compute(thunk, name::Symbol, supported::Bool; requires_box::Bool=false)
        if !supported
            push!(unavailable, name => "backend $(_region_backend_type(pi)) does not report support for $name.")
            return nothing
        elseif requires_box && !finite_box
            push!(unavailable, name => "requires a finite `box=(a,b)`.")
            return nothing
        end
        try
            value = thunk()
            value === nothing ? push!(unavailable, name => "returned `nothing` for this region/query.") : push!(available, name)
            return value
        catch err
            reason = sprint(showerror, err)
            push!(issues, string(name, ": ", reason))
            push!(unavailable, name => reason)
            return nothing
        end
    end

    bbox = maybe_compute(:bbox, q.region_bbox; requires_box=true) do
        _region_bbox_maybe_cached(pi, r; box=box_eff, strict=strict, closure=closure, cache=cache)
    end

    widths = bbox === nothing ? nothing : maybe_compute(:widths, q.region_widths) do
        lo, hi = bbox
        [float(hi[i]) - float(lo[i]) for i in eachindex(lo, hi)]
    end

    volume = maybe_compute(:volume, q.region_volume; requires_box=true) do
        _region_volume_maybe_cached(pi, r; box=box_eff, closure=closure, cache=cache)
    end

    centroid = maybe_compute(:centroid, q.region_centroid; requires_box=true) do
        _region_centroid_maybe_cached(pi, r; box=box_eff, method=:bbox, closure=closure, cache=cache)
    end

    aspect_ratio = widths === nothing ? nothing : maybe_compute(:aspect_ratio, q.region_aspect_ratio) do
        maxw = 0.0
        minw = Inf
        for wi in widths
            aw = abs(float(wi))
            isfinite(aw) || return Inf
            maxw = max(maxw, aw)
            if aw > 0.0
                minw = min(minw, aw)
            end
        end
        maxw == 0.0 ? 1.0 : (minw == Inf ? Inf : maxw / minw)
    end

    boundary_measure = maybe_compute(:boundary_measure, q.region_boundary_measure; requires_box=true) do
        _region_boundary_measure_maybe_cached(pi, r; box=box_eff, strict=strict, closure=closure, cache=cache)
    end

    circumradius = maybe_compute(:circumradius, q.region_circumradius; requires_box=true) do
        region_circumradius(pi, r; box=box_eff, strict=strict, closure=closure, cache=cache, kwargs...)
    end

    breakdown = include_breakdown ? maybe_compute(:boundary_breakdown,
        q.region_boundary_measure_breakdown; requires_box=true) do
            region_boundary_measure_breakdown(pi, r; box=box_eff, strict=strict, kwargs...)
        end : nothing

    return RegionGeometrySummary((
        kind = :region_geometry_summary,
        backend = _region_backend_type(pi),
        region = Int(r),
        box = box_eff,
        finite_box = finite_box,
        capabilities = report.queries,
        quantity_sources = quantity_sources,
        available = unique!(available),
        unavailable = unavailable,
        volume = volume,
        bbox = bbox,
        widths = widths,
        centroid = centroid,
        aspect_ratio = aspect_ratio,
        boundary_measure = boundary_measure,
        circumradius = circumradius,
        boundary_breakdown = breakdown,
        issues = issues,
    ))
end

@inline _kwget(kwargs, key::Symbol, default) = get(kwargs, key, default)

@inline function _locate_dispatch(pi, x, ::Val{:strict_closure}; strict::Bool, closure::Bool)
    return locate(pi, x; strict=strict, closure=closure)
end

@inline function _locate_dispatch(pi, x, ::Val{:strict_only}; strict::Bool, closure::Bool)
    _ = closure
    return locate(pi, x; strict=strict)
end

@inline function _locate_dispatch(pi, x, ::Val{:plain}; strict::Bool, closure::Bool)
    _ = (strict, closure)
    return locate(pi, x)
end

function _resolve_locate_style(pi, x0::AbstractVector{<:Real}; strict::Bool, closure::Bool)
    return _locate_call_style(pi, x0; strict=strict, closure=closure, batched=false)
end

@inline function _locate_many_dispatch!(dest, pi, X, ::Val{:strict_closure}; strict::Bool, closure::Bool)
    return locate_many!(dest, pi, X; strict=strict, closure=closure)
end

@inline function _locate_many_dispatch!(dest, pi, X, ::Val{:strict_only}; strict::Bool, closure::Bool)
    _ = closure
    return locate_many!(dest, pi, X; strict=strict)
end

@inline function _locate_many_dispatch!(dest, pi, X, ::Val{:plain}; strict::Bool, closure::Bool)
    _ = (strict, closure)
    return locate_many!(dest, pi, X)
end

function _resolve_locate_many_style(pi, x0::AbstractVector{<:Real}; strict::Bool, closure::Bool)
    return _locate_call_style(pi, x0; strict=strict, closure=closure, batched=true)
end

@inline function _zero_moments!(sumx::AbstractVector{Float64}, sumxx::AbstractMatrix{Float64})
    fill!(sumx, 0.0)
    fill!(sumxx, 0.0)
    return nothing
end

@inline function _compute_block_moments!(ws,
    X::AbstractMatrix{Float64}, ncols::Int)
    ncols <= 0 && return nothing
    n = size(X, 1)
    block_sum = ws.block_sum
    fill!(block_sum, 0.0)
    gram = ws.gram
    if ncols <= 1
        fill!(gram, 0.0)
        @inbounds for j in 1:ncols
            for i in 1:n
                xij = X[i, j]
                block_sum[i] += xij
            end
            for c in 1:n, r in 1:n
                gram[r, c] += X[r, j] * X[c, j]
            end
        end
        return nothing
    end
    @inbounds for j in 1:ncols, i in 1:n
        block_sum[i] += X[i, j]
    end
    mul!(gram, view(X, 1:n, 1:ncols), transpose(view(X, 1:n, 1:ncols)))
    return nothing
end

@inline function _accumulate_block_moments!(sumx::AbstractVector{Float64},
    sumxx::AbstractMatrix{Float64}, ws)
    @inbounds for i in eachindex(sumx)
        sumx[i] += ws.block_sum[i]
    end
    @inbounds for j in axes(sumxx, 2), i in axes(sumxx, 1)
        sumxx[i, j] += ws.gram[i, j]
    end
    return nothing
end

@inline _region_encoding_cache(::Any) = nothing
@inline _region_encoding_cache(cache::EncodingCache) = cache
@inline _region_encoding_cache(enc::CompiledEncoding) = enc.meta isa EncodingCache ? enc.meta : nothing
@inline function _region_encoding_cache(pi, cache)
    cache isa EncodingCache && return cache
    return _region_encoding_cache(pi)
end

@inline function _region_geometry_cache_get(cache::Union{Nothing,EncodingCache}, key)
    cache === nothing && return nothing
    Base.lock(cache.lock)
    try
        entry = get(cache.geometry, key, nothing)
        return entry === nothing ? nothing : entry.value
    finally
        Base.unlock(cache.lock)
    end
end

@inline function _region_geometry_cache_set!(cache::Union{Nothing,EncodingCache}, key, value)
    cache === nothing && return value
    Base.lock(cache.lock)
    try
        cache.geometry[key] = GeometryCachePayload(value)
    finally
        Base.unlock(cache.lock)
    end
    return value
end

mutable struct _RegionBatchWorkspace
    n::Int
    batchsize::Int
    ndirs::Int
    mu::Vector{Float64}
    C::Matrix{Float64}
    mu_b::Vector{Float64}
    C_b::Matrix{Float64}
    block_sum::Vector{Float64}
    gram::Matrix{Float64}
    X::Matrix{Float64}
    locs::Vector{Int}
    x::Vector{Float64}
    accepted::Matrix{Float64}
    proj::Matrix{Float64}
    minproj::Vector{Float64}
    maxproj::Vector{Float64}
end

@inline function _RegionBatchWorkspace(n::Integer, batchsize::Integer, ndirs::Integer=0)
    return _RegionBatchWorkspace(
        Int(n),
        Int(batchsize),
        Int(ndirs),
        Vector{Float64}(undef, n),
        Matrix{Float64}(undef, n, n),
        Vector{Float64}(undef, n),
        Matrix{Float64}(undef, n, n),
        Vector{Float64}(undef, n),
        Matrix{Float64}(undef, n, n),
        Matrix{Float64}(undef, n, batchsize),
        Vector{Int}(undef, batchsize),
        Vector{Float64}(undef, n),
        Matrix{Float64}(undef, n, batchsize),
        Matrix{Float64}(undef, max(1, ndirs), batchsize),
        Vector{Float64}(undef, max(1, ndirs)),
        Vector{Float64}(undef, max(1, ndirs)),
    )
end

const _REGION_WORKSPACES = let n = max(1, Base.Threads.nthreads())
    [Dict{Tuple{Int,Int,Int},_RegionBatchWorkspace}() for _ in 1:n]
end

mutable struct _RegionSampleCache{T}
    locks::Vector{Base.ReentrantLock}
    shards::Vector{Dict{UInt64,T}}
end

@inline _region_cache_nshards() = Threads.nthreads() == 1 ? 1 : _REGION_CACHE_SHARDS

function _RegionSampleCache(::Type{T}) where {T}
    nshards = _region_cache_nshards()
    return _RegionSampleCache([Base.ReentrantLock() for _ in 1:nshards],
        [Dict{UInt64,T}() for _ in 1:nshards])
end

@inline function _region_cache_shard(cache::_RegionSampleCache, key::UInt64)
    nshards = length(cache.shards)
    return Int(mod(key, UInt64(nshards))) + 1
end

@inline function _region_cache_shard_cap(cache::_RegionSampleCache)
    return max(1, cld(_REGION_SAMPLE_CACHE_MAX[], max(1, length(cache.shards))))
end

function Base.length(cache::_RegionSampleCache)
    total = 0
    for shard in cache.shards
        total += length(shard)
    end
    return total
end

struct _PrincipalSummaryEntry
    mean::Vector{Float64}
    cov::Matrix{Float64}
    evals::Vector{Float64}
    evecs::Matrix{Float64}
    mean_stderr::Vector{Float64}
    evals_stderr::Vector{Float64}
    batch_evals::Matrix{Float64}
    batch_n_accepted::Vector{Int}
    nbatches::Int
    n_accepted::Int
    n_proposed::Int
end

const _REGION_PRINCIPAL_SUMMARY_CACHE = _RegionSampleCache(_PrincipalSummaryEntry)
const _REGION_MEAN_WIDTH_CACHE = _RegionSampleCache(Float64)

function _clear_region_geometry_runtime_caches!()
    for cache in (_REGION_PRINCIPAL_SUMMARY_CACHE, _REGION_MEAN_WIDTH_CACHE)
        for i in eachindex(cache.shards)
            Base.lock(cache.locks[i])
            try
                empty!(cache.shards[i])
            finally
                Base.unlock(cache.locks[i])
            end
        end
    end
    for ws in _REGION_WORKSPACES
        empty!(ws)
    end
    return nothing
end

@inline function _region_workspace(n::Integer, batchsize::Integer, ndirs::Integer=0)
    if !_REGION_WORKSPACE_REUSE[]
        return _RegionBatchWorkspace(n, batchsize, ndirs)
    end
    tid = min(Base.Threads.threadid(), length(_REGION_WORKSPACES))
    key = (Int(n), Int(batchsize), Int(ndirs))
    return get!(_REGION_WORKSPACES[tid], key) do
        _RegionBatchWorkspace(n, batchsize, ndirs)
    end
end

@inline function _sample_cache_get(cache::_RegionSampleCache{T}, key::UInt64) where {T}
    shard = _region_cache_shard(cache, key)
    Base.lock(cache.locks[shard])
    try
        return get(cache.shards[shard], key, nothing)
    finally
        Base.unlock(cache.locks[shard])
    end
end

@inline function _sample_cache_insert!(cache::_RegionSampleCache{T}, key::UInt64, value::T) where {T}
    shard = _region_cache_shard(cache, key)
    Base.lock(cache.locks[shard])
    try
        bucket = cache.shards[shard]
        if length(bucket) >= _region_cache_shard_cap(cache)
            empty!(bucket)
        end
        bucket[key] = value
    finally
        Base.unlock(cache.locks[shard])
    end
    return value
end

@inline _region_summary_cache_key(kind::Symbol, key::UInt64) = (kind, key)
@inline _region_direction_cache_key(n::Integer, ndirs::Integer, rngh) = (:region_direction_bank, Int(n), Int(ndirs), UInt64(rngh))

@inline function _region_summary_cache_get(global_cache::_RegionSampleCache{T},
    enc_cache::Union{Nothing,EncodingCache}, kind::Symbol, key::UInt64) where {T}
    if enc_cache !== nothing
        cached = _region_geometry_cache_get(enc_cache, _region_summary_cache_key(kind, key))
        cached === nothing || return cached
    end
    return _sample_cache_get(global_cache, key)
end

@inline function _region_summary_cache_set!(global_cache::_RegionSampleCache{T},
    enc_cache::Union{Nothing,EncodingCache}, kind::Symbol, key::UInt64, value::T) where {T}
    if enc_cache !== nothing
        return _region_geometry_cache_set!(enc_cache, _region_summary_cache_key(kind, key), value)
    end
    return _sample_cache_insert!(global_cache, key, value)
end

@inline function _hash_float_sequence(xs)
    h = hash(length(xs))
    @inbounds for x in xs
        h = hash(float(x), h)
    end
    return h
end

@inline _box_cache_hash(box) = hash((_hash_float_sequence(box[1]), _hash_float_sequence(box[2])))

@inline _pi_geometry_cache_hash(pi) = hash(_geometry_fingerprint(pi))

function _rng_cache_hash_slow(rng)
    rc = try
        copy(rng)
    catch
        try
            deepcopy(rng)
        catch
            return nothing
        end
    end
    vals = ntuple(_ -> rand(rc, UInt64), 4)
    return hash((typeof(rng), vals))
end

function _rng_cache_hash(rng::Random.MersenneTwister)
    st = getfield(rng, :state)
    return hash((typeof(rng), getfield(rng, :seed), getfield(rng, :idxF), getfield(rng, :idxI),
        getfield(rng, :adv), getfield(rng, :adv_jump), getfield(rng, :adv_vals),
        getfield(rng, :adv_ints), hash(getfield(st, :val))))
end

function _rng_cache_hash(rng::Random.Xoshiro)
    return hash((typeof(rng), getfield(rng, :s0), getfield(rng, :s1),
        getfield(rng, :s2), getfield(rng, :s3), getfield(rng, :s4)))
end

_rng_cache_hash(rng) = _rng_cache_hash_slow(rng)

@inline _batch_eval_count(batch_evals::AbstractMatrix{<:Real}) = size(batch_evals, 2)
@inline _batch_eval_count(batch_evals::AbstractVector) = length(batch_evals)
@inline _batch_eval_column(batch_evals::AbstractMatrix{<:Real}, j::Integer) = view(batch_evals, :, j)
@inline _batch_eval_column(batch_evals::AbstractVector, j::Integer) = batch_evals[j]

function _directions_cache_hash(directions, ndirs::Integer)
    directions === nothing && return hash((:random, ndirs))
    return hash((size(directions), _hash_float_sequence(directions)))
end

function _principal_summary_cache_key(pi, r::Integer;
    box, strict::Bool, closure::Bool, nsamples::Integer,
    max_proposals::Integer, return_info::Bool, nbatches::Integer, rng)
    rngh = _rng_cache_hash(rng)
    rngh === nothing && return nothing
    return UInt64(hash((_pi_geometry_cache_hash(pi), Int(r), _box_cache_hash(box),
        strict, closure, Int(nsamples), Int(max_proposals),
        return_info, Int(nbatches), rngh)))
end

function _mean_width_cache_key(pi, r::Integer;
    box, strict::Bool, closure::Bool, nsamples::Integer,
    max_proposals::Integer, ndirs::Integer, directions, rng)
    rngh = _rng_cache_hash(rng)
    rngh === nothing && return nothing
    dirh = _directions_cache_hash(directions, ndirs)
    return UInt64(hash((_pi_geometry_cache_hash(pi), Int(r), _box_cache_hash(box),
        strict, closure, Int(nsamples), Int(max_proposals), Int(ndirs), dirh, rngh)))
end

@inline function _box_lower_widths(box)
    a_in, b_in = box
    n = length(a_in)
    a = Vector{Float64}(undef, n)
    widths = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        ai = float(a_in[i])
        bi = float(b_in[i])
        a[i] = ai
        widths[i] = bi - ai
    end
    return a, widths
end

@inline function _fill_random_point!(x::AbstractVector{Float64},
    a::AbstractVector{Float64}, widths::AbstractVector{Float64}, rng)
    rand!(rng, x)
    @inbounds for i in eachindex(x)
        x[i] = muladd(x[i], widths[i], a[i])
    end
    return x
end

@inline function _fill_random_points!(X::AbstractMatrix{Float64},
    a::AbstractVector{Float64}, widths::AbstractVector{Float64}, rng, ncols::Integer)
    Xv = view(X, :, 1:ncols)
    rand!(rng, Xv)
    @inbounds for j in 1:ncols, i in eachindex(a)
        X[i, j] = muladd(X[i, j], widths[i], a[i])
    end
    return X
end


"""
    region_perimeter(pi, r; box, kwargs...) -> Float64

Convenience wrapper for 2D regions. This calls
`region_boundary_measure(pi, r; box=box, kwargs...)`.
"""
function region_perimeter(pi, r; box=nothing, cache=nothing, kwargs...)
    box = _resolve_box(box, cache)
    box === nothing && error("region_perimeter: please provide box=(a,b)")

    a, _ = box
    length(a) == 2 || error("region_perimeter: expected 2D box, got length(a)=$(length(a))")
    if cache === nothing
        return float(region_boundary_measure(pi, r; box=box, kwargs...))
    end
    return float(region_boundary_measure(pi, r; box=box, cache=cache, kwargs...))
end

"""
    region_surface_area(pi, r; box, kwargs...) -> Float64

Convenience wrapper for 3D and higher. This calls
`region_boundary_measure(pi, r; box=box, kwargs...)`.
"""
function region_surface_area(pi, r; box=nothing, cache=nothing, kwargs...)
    box = _resolve_box(box, cache)
    box === nothing && error("region_surface_area: please provide box=(a,b)")

    a, _ = box
    length(a) >= 3 || error("region_surface_area: expected 3D+ box, got length(a)=$(length(a))")
    if cache === nothing
        return float(region_boundary_measure(pi, r; box=box, kwargs...))
    end
    return float(region_boundary_measure(pi, r; box=box, cache=cache, kwargs...))
end

"""
    region_principal_directions(pi, r; box, nsamples=20_000, rng=Random.default_rng(),
        strict=true, closure=true, max_proposals=10*nsamples,
        return_info=false, nbatches=0)

Estimate the mean, covariance matrix, and principal directions for the region `r`
inside a *finite* `box = (ell, u)` by rejection sampling.

Return value (always):
- `mean`  : estimated mean of a uniform point in the region (within the box)
- `cov`   : estimated covariance matrix
- `evals` : eigenvalues of `cov`, sorted in descending order
- `evecs` : corresponding eigenvectors (columns), matching `evals`
- `n_accepted`, `n_proposed` : acceptance diagnostics

If `return_info=true`, additional uncertainty estimates are returned:
- `mean_stderr` : coordinatewise standard error for the mean estimate
- `evals_stderr`: standard error for the eigenvalue estimates, based on batching
- `batch_evals` : per-batch eigenvalue vectors (each sorted)
- `batch_n_accepted` : accepted samples per batch
- `nbatches`    : number of batches actually used

Batching behavior:
- If `return_info=true` and `nbatches == 0`, a default of 10 batches is used.
- Batch statistics are computed on *accepted* samples.

Speed notes:
- Uses preallocated vectors and in-place Welford updates to reduce allocations.
- Cost profile: sampling-based. This is heavier than bbox-based queries such as
  `region_widths` or `region_diameter`, but more informative for anisotropy and
  covariance structure.
"""
function region_principal_directions(pi, r::Integer;
    box=nothing,
    nsamples::Int=20_000,
    rng::AbstractRNG=Random.default_rng(),
    strict::Bool=true,
    closure::Bool=true,
    max_proposals::Int=10*nsamples,
    return_info::Bool=false,
    nbatches::Int=0
)
    use_batched = _REGION_BATCHED_LOCATE[] &&
        max_proposals >= _REGION_BATCHED_LOCATE_MIN_PROPOSALS[]
    if use_batched
        return _region_principal_directions_batched(pi, r;
            box=box, nsamples=nsamples, rng=rng, strict=strict, closure=closure,
            max_proposals=max_proposals, return_info=return_info, nbatches=nbatches)
    end
    return _region_principal_directions_scalar(pi, r;
        box=box, nsamples=nsamples, rng=rng, strict=strict, closure=closure,
        max_proposals=max_proposals, return_info=return_info, nbatches=nbatches)
end

"""
    region_principal_directions_summary(pi, r; box, epsilon=0.0, kwargs...) -> RegionPrincipalDirectionsSummary

Typed exploratory wrapper around `region_principal_directions`.

This is the preferred owner-local summary object when you want:
- principal values and directions,
- covariance accessors,
- anisotropy scores derived from the same covariance estimate,
- a stable object with compact/plain-text display.

Use `principal_values`, `principal_vectors`, and `covariance_matrix` to inspect
the result without field archaeology.
"""
function region_principal_directions_summary(pi, r::Integer; box=nothing,
    epsilon::Real=0.0, kwargs...)
    pd = region_principal_directions(pi, r; box=box, return_info=true, kwargs...)
    evals = pd.evals
    anisotropy = (
        ratio = _covariance_anisotropy_from_evals(evals; kind=:ratio, epsilon=epsilon),
        log_ratio = _covariance_anisotropy_from_evals(evals; kind=:log_ratio, epsilon=epsilon),
        normalized = _covariance_anisotropy_from_evals(evals; kind=:normalized, epsilon=epsilon),
        eccentricity = _covariance_eccentricity_from_evals(evals; epsilon=epsilon),
    )
    return RegionPrincipalDirectionsSummary((
        backend = _region_backend_type(pi),
        region = Int(r),
        box = box,
        epsilon = float(epsilon),
        mean = pd.mean,
        cov = pd.cov,
        evals = pd.evals,
        evecs = pd.evecs,
        mean_stderr = pd.mean_stderr,
        evals_stderr = pd.evals_stderr,
        batch_evals = pd.batch_evals,
        batch_n_accepted = pd.batch_n_accepted,
        nbatches = pd.nbatches,
        n_accepted = pd.n_accepted,
        n_proposed = pd.n_proposed,
        anisotropy = anisotropy,
    ))
end

@inline function _batch_eval_from_moments!(dest::AbstractVector{Float64},
    sumx::AbstractVector{Float64}, sumxx::AbstractMatrix{Float64}, nacc::Int)
    n = length(sumx)
    if nacc <= 1
        fill!(dest, 0.0)
        return dest
    end
    invn = inv(float(nacc))
    cov = Matrix{Float64}(undef, n, n)
    @inbounds for j in 1:n, i in 1:n
        centered = sumxx[i, j] - (sumx[i] * sumx[j]) * invn
        cov[i, j] = centered / (nacc - 1)
    end
    E = eigen(Symmetric(cov))
    p = sortperm(E.values, rev=true)
    @inbounds for i in 1:n
        dest[i] = E.values[p[i]]
    end
    return dest
end

function _principal_directions_result(sumx::Vector{Float64}, sumxx::Matrix{Float64},
    batch_evals::Matrix{Float64}, nbatches::Int, batch_n::Vector{Int},
    nacc::Int, nprop::Int, return_info::Bool)
    n = length(sumx)
    if nacc <= 1
        mean = zeros(Float64, n)
        cov = zeros(Float64, n, n)
        evals = zeros(Float64, n)
        evecs = Matrix{Float64}(I, n, n)
        mean_stderr = fill(NaN, n)
        evals_stderr = fill(NaN, n)
    else
        invn = inv(float(nacc))
        mean = copy(sumx)
        rmul!(mean, invn)
        cov = Matrix{Float64}(undef, n, n)
        @inbounds for j in 1:n, i in 1:n
            centered = sumxx[i, j] - (sumx[i] * sumx[j]) * invn
            cov[i, j] = centered / (nacc - 1)
        end
        E = eigen(Symmetric(cov))
        p = sortperm(E.values, rev=true)
        evals = E.values[p]
        evecs = E.vectors[:, p]
        mean_stderr = sqrt.(diag(cov) ./ nacc)

        if nbatches >= 2
            k = nbatches
            evals_stderr = Vector{Float64}(undef, n)
            @inbounds for i in 1:n
                s = 0.0
                ss = 0.0
                for j in 1:nbatches
                    v = batch_evals[i, j]
                    s += v
                    ss += v * v
                end
                m = s / k
                var = (ss - k * m * m) / (k - 1)
                evals_stderr[i] = sqrt(var < 0.0 ? 0.0 : var) / sqrt(k)
            end
        else
            evals_stderr = fill(NaN, n)
        end
    end

    if !return_info
        return (mean=mean, cov=cov, evals=evals, evecs=evecs,
            n_accepted=nacc, n_proposed=nprop)
    end

    return (mean=mean, cov=cov, evals=evals, evecs=evecs,
        mean_stderr=mean_stderr, evals_stderr=evals_stderr,
        batch_evals=copy(view(batch_evals, :, 1:nbatches)), batch_n_accepted=copy(batch_n),
        nbatches=nbatches,
        n_accepted=nacc, n_proposed=nprop)
end

@inline _principal_cov_blocksize(nsamples::Integer) =
    max(1, min(Int(nsamples), _REGION_LOCATE_BATCH_SIZE[]))

@inline function _principal_flush_buffer!(sumx::AbstractVector{Float64}, sumxx::AbstractMatrix{Float64},
    sumx_b::AbstractVector{Float64}, sumxx_b::AbstractMatrix{Float64},
    accepted::AbstractMatrix{Float64}, nbuf::Int, ws::_RegionBatchWorkspace, want_batches::Int)
    nbuf <= 0 && return 0
    _compute_block_moments!(ws, accepted, nbuf)
    _accumulate_block_moments!(sumx, sumxx, ws)
    if want_batches > 0
        _accumulate_block_moments!(sumx_b, sumxx_b, ws)
    end
    return 0
end

@inline function _principal_finish_batch!(batch_evals::AbstractMatrix{Float64}, batch_n::Vector{Int},
    nbatches::Int, sumx_b::AbstractVector{Float64}, sumxx_b::AbstractMatrix{Float64}, nacc_b::Int)
    if nacc_b > 1
        _batch_eval_from_moments!(view(batch_evals, :, nbatches + 1), sumx_b, sumxx_b, nacc_b)
        push!(batch_n, nacc_b)
        nbatches += 1
    end
    _zero_moments!(sumx_b, sumxx_b)
    return nbatches
end

function _region_principal_directions_scalar(pi, r::Integer;
    box=nothing,
    nsamples::Int=20_000,
    rng::AbstractRNG=Random.default_rng(),
    strict::Bool=true,
    closure::Bool=true,
    max_proposals::Int=10 * nsamples,
    return_info::Bool=false,
    nbatches::Int=0
)
    box === nothing && error("A finite box is required for region_principal_directions.")
    a, b = box
    n = length(a)

    ws = _region_workspace(n, _principal_cov_blocksize(nsamples), 0)
    sumx = ws.mu
    sumxx = ws.C
    sumx_b = ws.mu_b
    sumxx_b = ws.C_b
    x = ws.x
    accepted = ws.accepted
    _zero_moments!(sumx, sumxx)
    _zero_moments!(sumx_b, sumxx_b)
    a_f, widths = _box_lower_widths(box)

    nacc = 0
    nprop = 0
    nbuf = 0
    nacc_b = 0

    want_batches = return_info ? (nbatches > 0 ? nbatches : 10) : 0
    batch_evals = Matrix{Float64}(undef, n, want_batches > 0 ? want_batches : 0)
    batch_n = Int[]
    nbatch_used = 0
    if want_batches > 0
        sizehint!(batch_n, want_batches)
    end
    batch_target = want_batches > 0 ? max(2, Int(floor(nsamples / want_batches))) : 0

    x0 = Float64[a_f[i] + 0.5 * widths[i] for i in 1:n]
    locate_style = _resolve_locate_style(pi, x0; strict=strict, closure=closure)

    while (nacc < nsamples) && (nprop < max_proposals)
        _fill_random_point!(x, a_f, widths, rng)
        nprop += 1

        if _locate_dispatch(pi, x, locate_style; strict=strict, closure=closure) == r
            nacc += 1
            nbuf += 1
            @inbounds for i in 1:n
                accepted[i, nbuf] = x[i]
            end
            if want_batches > 0
                nacc_b += 1
            end

            if nbuf >= size(accepted, 2) || (want_batches > 0 && nacc_b >= batch_target) || nacc >= nsamples
                nbuf = _principal_flush_buffer!(sumx, sumxx, sumx_b, sumxx_b, accepted, nbuf, ws, want_batches)
            end
            if want_batches > 0 && nacc_b >= batch_target
                nbatch_used = _principal_finish_batch!(batch_evals, batch_n, nbatch_used, sumx_b, sumxx_b, nacc_b)
                nacc_b = 0
            end
        end
    end

    nbuf = _principal_flush_buffer!(sumx, sumxx, sumx_b, sumxx_b, accepted, nbuf, ws, want_batches)
    if want_batches > 0 && nacc_b > 0
        nbatch_used = _principal_finish_batch!(batch_evals, batch_n, nbatch_used, sumx_b, sumxx_b, nacc_b)
    end

    return _principal_directions_result(sumx, sumxx, batch_evals, nbatch_used, batch_n, nacc, nprop, return_info)
end

function _region_principal_directions_batched(pi, r::Integer;
    box=nothing,
    nsamples::Int=20_000,
    rng::AbstractRNG=Random.default_rng(),
    strict::Bool=true,
    closure::Bool=true,
    max_proposals::Int=10 * nsamples,
    return_info::Bool=false,
    nbatches::Int=0
)
    box === nothing && error("A finite box is required for region_principal_directions.")
    a, b = box
    n = length(a)

    ws = _region_workspace(n, max(1, _REGION_LOCATE_BATCH_SIZE[]), 0)
    sumx = ws.mu
    sumxx = ws.C
    sumx_b = ws.mu_b
    sumxx_b = ws.C_b
    accepted = ws.accepted
    _zero_moments!(sumx, sumxx)
    _zero_moments!(sumx_b, sumxx_b)
    a_f, widths = _box_lower_widths(box)

    nacc = 0
    nprop = 0
    nbuf = 0
    nacc_b = 0

    want_batches = return_info ? (nbatches > 0 ? nbatches : 10) : 0
    batch_evals = Matrix{Float64}(undef, n, want_batches > 0 ? want_batches : 0)
    batch_n = Int[]
    nbatch_used = 0
    if want_batches > 0
        sizehint!(batch_n, want_batches)
    end
    batch_target = want_batches > 0 ? max(2, Int(floor(nsamples / want_batches))) : 0

    x0 = Float64[a_f[i] + 0.5 * widths[i] for i in 1:n]
    locate_style = _resolve_locate_many_style(pi, x0; strict=strict, closure=closure)

    while (nacc < nsamples) && (nprop < max_proposals)
        nbatch = min(size(ws.X, 2), max_proposals - nprop)
        _fill_random_points!(ws.X, a_f, widths, rng, nbatch)
        Xbatch = view(ws.X, :, 1:nbatch)
        locbatch = view(ws.locs, 1:nbatch)
        _locate_many_dispatch!(locbatch, pi, Xbatch, locate_style; strict=strict, closure=closure)
        nprop += nbatch

        @inbounds for j in 1:nbatch
            locbatch[j] == r || continue
            nacc += 1
            nbuf += 1
            for i in 1:n
                accepted[i, nbuf] = ws.X[i, j]
            end
            if want_batches > 0
                nacc_b += 1
            end

            if nbuf >= size(accepted, 2) || (want_batches > 0 && nacc_b >= batch_target) || nacc >= nsamples
                nbuf = _principal_flush_buffer!(sumx, sumxx, sumx_b, sumxx_b, accepted, nbuf, ws, want_batches)
            end
            if want_batches > 0 && nacc_b >= batch_target
                nbatch_used = _principal_finish_batch!(batch_evals, batch_n, nbatch_used, sumx_b, sumxx_b, nacc_b)
                nacc_b = 0
            end

            nacc >= nsamples && break
        end
    end

    nbuf = _principal_flush_buffer!(sumx, sumxx, sumx_b, sumxx_b, accepted, nbuf, ws, want_batches)
    if want_batches > 0 && nacc_b > 0
        nbatch_used = _principal_finish_batch!(batch_evals, batch_n, nbatch_used, sumx_b, sumxx_b, nacc_b)
    end

    return _principal_directions_result(sumx, sumxx, batch_evals, nbatch_used, batch_n, nacc, nprop, return_info)
end



# ------------------------------------------------------------------------------
# Additional geometric descriptors for regions
#
# These are intended as "mathematician-friendly" higher-level quantities built
# on top of the primitive geometry routines (volume, boundary measure, bbox,
# principal directions).
#
# Most of these routines only make sense for bounded regions, so they generally
# require a finite `box=(a,b)` to intersect with.
# ------------------------------------------------------------------------------

# Internal helper: volume of the unit n-ball in Euclidean space (Float64).
# We avoid SpecialFunctions by using the recursion:
#   omega_0 = 1, omega_1 = 2, omega_n = (2*pi/n) * omega_{n-2}.
@inline function _unit_ball_volume(n::Integer)
    n < 0 && error("unit ball volume: dimension must be >= 0, got $n")
    n == 0 && return 1.0
    n == 1 && return 2.0

    if iseven(n)
        omega = 1.0
        k = 2
        while k <= n
            omega *= (2.0 * pi) / k
            k += 2
        end
        return omega
    else
        omega = 2.0
        k = 3
        while k <= n
            omega *= (2.0 * pi) / k
            k += 2
        end
        return omega
    end
end

@inline function _isoperimetric_constant(n::Integer)
    # Sharp Euclidean isoperimetric inequality:
    #   S >= c_n * V^((n-1)/n),  where c_n = n * omega_n^(1/n).
    n < 1 && error("isoperimetric constant: dimension must be >= 1, got $n")
    omega = _unit_ball_volume(n)
    return float(n) * omega^(1.0 / float(n))
end

# Internal helper: circumradius of an axis-aligned bounding box around a center.
@inline function _bbox_circumradius(lo::AbstractVector, hi::AbstractVector,
    c::AbstractVector, metric::Symbol)
    n = length(lo)
    length(hi) == n || error("bbox_circumradius: lo/hi length mismatch")
    length(c) == n || error("bbox_circumradius: center length mismatch")

    if metric === :L2
        s2 = 0.0
        @inbounds for i in 1:n
            di = max(abs(lo[i] - c[i]), abs(hi[i] - c[i]))
            s2 += di * di
        end
        return sqrt(s2)
    elseif metric === :Linf
        dmax = 0.0
        @inbounds for i in 1:n
            di = max(abs(lo[i] - c[i]), abs(hi[i] - c[i]))
            dmax = max(dmax, di)
        end
        return dmax
    elseif metric === :L1
        s = 0.0
        @inbounds for i in 1:n
            di = max(abs(lo[i] - c[i]), abs(hi[i] - c[i]))
            s += di
        end
        return s
    else
        error("bbox_circumradius: unknown metric=$metric (use :L2, :L1, or :Linf)")
    end
end

# A small helper to call region_volume / region_boundary_measure with optional
# closure/cache/strict keywords, while gracefully degrading when the backend
# does not accept these keywords.
function _region_volume_maybe_cached_slow(pi, r::Integer; box, closure::Bool=true, cache=nothing)
    return _region_volume_from_weights(pi, r; box=box, closure=closure, cache=cache)
end

function _region_volume_maybe_cached(pi, r::Integer; box, closure::Bool=true, cache=nothing)
    if _REGION_DIRECT_VOLUME[]
        fast = _region_volume_fast(pi, r; box=box, closure=closure, cache=cache)
        fast === nothing || return float(fast)
    end
    return _region_volume_maybe_cached_slow(pi, r; box=box, closure=closure, cache=cache)
end

function _region_weights_maybe_cached(pi; box=nothing, closure::Bool=true, cache=nothing, kwargs...)
    if cache !== nothing
        cached = _region_weights_cached(pi; box=box, closure=closure, cache=cache, kwargs...)
        cached === nothing || return cached
    end
    if closure
        cached = _region_weights_closure(pi; box=box, closure=closure, kwargs...)
        cached === nothing || return cached
    end
    return region_weights(pi; box=box, kwargs...)
end

function _region_volume_from_weights(pi, r::Integer; box=nothing, closure::Bool=true, cache=nothing, kwargs...)
    w = _region_weights_maybe_cached(pi; box=box, closure=closure, cache=cache, kwargs...)
    (1 <= r <= length(w)) || error("region_volume: region index out of range")
    return float(w[Int(r)])
end

function _region_boundary_measure_maybe_cached_slow(pi, r::Integer; box, strict::Bool=true,
    closure::Bool=true, cache=nothing)
    if cache !== nothing
        cached = _region_boundary_measure_cached(pi, r;
            box=box, strict=strict, closure=closure, cache=cache)
        cached === nothing || return float(cached)
    end
    if closure
        cached = _region_boundary_measure_closure(pi, r;
            box=box, strict=strict, closure=closure)
        cached === nothing || return float(cached)
    end
    cached = _region_boundary_measure_strict(pi, r; box=box, strict=strict)
    cached === nothing || return float(cached)
    return float(region_boundary_measure(pi, r; box=box))
end

function _region_boundary_measure_maybe_cached(pi, r::Integer; box, strict::Bool=true,
    closure::Bool=true, cache=nothing)
    if _REGION_FAST_WRAPPERS[]
        fast = _region_boundary_measure_fast(pi, r;
            box=box, strict=strict, closure=closure, cache=cache)
        fast === nothing || return float(fast)
    end
    return _region_boundary_measure_maybe_cached_slow(pi, r;
        box=box, strict=strict, closure=closure, cache=cache)
end

function _region_bbox_maybe_cached(pi, r::Integer; box, strict::Bool=true,
    closure::Bool=true, cache=nothing)
    if _REGION_FAST_WRAPPERS[]
        fast = _region_bbox_fast(pi, r; box=box, strict=strict, closure=closure, cache=cache)
        fast === nothing || return fast
    end
    if cache !== nothing
        cached = _region_bbox_cached(pi, r; box=box, strict=strict, closure=closure, cache=cache)
        cached === nothing || return cached
    end
    if closure
        cached = _region_bbox_closure(pi, r; box=box, strict=strict, closure=closure)
        cached === nothing || return cached
    end
    cached = _region_bbox_strict(pi, r; box=box, strict=strict)
    cached === nothing || return cached
    return region_bbox(pi, r; box=box)
end

@inline function _centroid_from_bbox(bb)
    bb === nothing && return nothing
    a, b = bb
    n = length(a)
    c = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        ai = float(a[i])
        bi = float(b[i])
        (isfinite(ai) && isfinite(bi)) || return nothing
        c[i] = (ai + bi) / 2.0
    end
    return c
end

function _region_centroid_maybe_cached(pi, r::Integer; box, method::Symbol=:bbox,
    closure::Bool=true, cache=nothing)
    if _REGION_FAST_WRAPPERS[]
        fast = _region_centroid_fast(pi, r;
            box=box, method=method, closure=closure, cache=cache)
        fast === nothing || return fast
    end
    if method === :bbox
        return _centroid_from_bbox(_region_bbox_maybe_cached(pi, r;
            box=box, strict=true, closure=closure, cache=cache))
    end
    if cache !== nothing
        cached = _region_centroid_cached(pi, r;
            box=box, method=method, closure=closure, cache=cache)
        cached === nothing || return cached
    end
    if closure
        cached = _region_centroid_closure(pi, r; box=box, method=method, closure=closure)
        cached === nothing || return cached
    end
    return region_centroid(pi, r; box=box, method=method)
end

@inline function _principal_summary_result(entry::_PrincipalSummaryEntry, return_info::Bool)
    if !return_info
        return (mean=copy(entry.mean), cov=copy(entry.cov), evals=copy(entry.evals),
            evecs=copy(entry.evecs), n_accepted=entry.n_accepted, n_proposed=entry.n_proposed)
    end
    return (mean=copy(entry.mean), cov=copy(entry.cov), evals=copy(entry.evals),
        evecs=copy(entry.evecs), mean_stderr=copy(entry.mean_stderr),
        evals_stderr=copy(entry.evals_stderr),
        batch_evals=copy(entry.batch_evals),
        batch_n_accepted=copy(entry.batch_n_accepted), nbatches=entry.nbatches,
        n_accepted=entry.n_accepted, n_proposed=entry.n_proposed)
end

function _principal_summary_entry(mu, cov, evals, evecs, mean_stderr, evals_stderr,
    batch_evals, batch_n, nacc::Int, nprop::Int)
    batch_eval_mat = if batch_evals isa Matrix{Float64}
        copy(batch_evals)
    elseif isempty(batch_evals)
        Matrix{Float64}(undef, length(evals), 0)
    else
        hcat(batch_evals...)
    end
    nbatches = size(batch_eval_mat, 2)
    return _PrincipalSummaryEntry(copy(mu), copy(cov), copy(evals), copy(evecs),
        copy(mean_stderr), copy(evals_stderr), batch_eval_mat, copy(batch_n),
        nbatches, nacc, nprop)
end

function _principal_directions_compute(pi, r::Integer; box, nsamples::Integer,
    rng, strict::Bool=true, closure::Bool=true, max_proposals::Integer=10*nsamples,
    return_info::Bool=false, nbatches::Int=0)
    return region_principal_directions(pi, r; box=box, nsamples=nsamples, rng=rng,
        strict=strict, closure=closure, max_proposals=max_proposals,
        return_info=return_info, nbatches=nbatches)
end

function _principal_summary_entry_maybe_closure(pi, r::Integer; box, nsamples::Integer,
    rng, strict::Bool=true, closure::Bool=true, max_proposals::Integer=10*nsamples,
    return_info::Bool=false, nbatches::Int=0)
    enc_cache = _region_encoding_cache(pi, nothing)
    key = _REGION_SAMPLED_SUMMARY_CACHE[] ? _principal_summary_cache_key(pi, r;
        box=box, strict=strict, closure=closure, nsamples=nsamples,
        max_proposals=max_proposals, return_info=return_info, nbatches=nbatches, rng=rng) : nothing
    if key !== nothing
        cached = _region_summary_cache_get(_REGION_PRINCIPAL_SUMMARY_CACHE,
            enc_cache, :principal_summary, key)
        cached === nothing || return cached
    end

    pd = _principal_directions_compute(pi, r; box=box, nsamples=nsamples, rng=rng,
        strict=strict, closure=closure, max_proposals=max_proposals,
        return_info=return_info, nbatches=nbatches)
    entry = _principal_summary_entry(pd.mean, pd.cov, pd.evals, pd.evecs,
        get(pd, :mean_stderr, fill(NaN, length(pd.mean))),
        get(pd, :evals_stderr, fill(NaN, length(pd.evals))),
        get(pd, :batch_evals, Matrix{Float64}(undef, length(pd.evals), 0)),
        get(pd, :batch_n_accepted, Int[]),
        pd.n_accepted, pd.n_proposed)

    if key !== nothing
        _region_summary_cache_set!(_REGION_PRINCIPAL_SUMMARY_CACHE,
            enc_cache, :principal_summary, key, entry)
    end
    return entry
end

function _principal_directions_maybe_closure(pi, r::Integer; box, nsamples::Integer,
    rng, strict::Bool=true, closure::Bool=true, max_proposals::Integer=10*nsamples,
    return_info::Bool=false, nbatches::Int=0)
    entry = _principal_summary_entry_maybe_closure(pi, r; box=box, nsamples=nsamples,
        rng=rng, strict=strict, closure=closure, max_proposals=max_proposals,
        return_info=return_info, nbatches=nbatches)
    return _principal_summary_result(entry, return_info)
end

@inline function _region_geometry_summary_maybe_fast(pi, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache=nothing,
    mean_width_method::Symbol=:auto, mean_width_ndirs::Integer=256,
    mean_width_rng=Random.default_rng(), mean_width_directions=nothing,
    need_mean_width::Bool=false)
    !_REGION_FAST_WRAPPERS[] && return nothing
    return _region_geometry_summary_fast(pi, r;
        box=box, strict=strict, closure=closure, cache=cache,
        mean_width_method=mean_width_method, mean_width_ndirs=mean_width_ndirs,
        mean_width_rng=mean_width_rng, mean_width_directions=mean_width_directions,
        need_mean_width=need_mean_width)
end

function _mean_width_cached(pi, r::Integer; box, method::Symbol=:auto,
    ndirs::Integer=256, nsamples::Integer=4000, max_proposals::Integer=10*nsamples,
    rng=Random.default_rng(), directions=nothing,
    strict::Bool=true, closure::Bool=true, cache=nothing)
    return region_mean_width(pi, r; box=box, method=method, ndirs=ndirs,
        nsamples=nsamples, max_proposals=max_proposals, rng=rng, directions=directions,
        strict=strict, closure=closure, cache=cache)
end


"""
    region_chebyshev_ball(pi, r; box, metric=:L2, method=:auto, kwargs...) -> NamedTuple

Return a `(center, radius)` pair describing a large inscribed ball of region `r`,
intersected with `box=(a,b)`.

This is backend-dependent:
- Convex polyhedral backends can compute a true Chebyshev (largest inscribed) ball.
- Non-convex backends may return a lower bound (still a valid inscribed ball).

The ball is interpreted in the norm specified by `metric`:
- `:L2`   Euclidean ball
- `:Linf` axis-aligned cube (L_infinity ball)
- `:L1`   cross-polytope (L1 ball)

Backends that do not implement this should throw an error.
"""
function region_chebyshev_ball(pi, r::Integer; box=nothing, metric::Symbol=:L2,
    method::Symbol=:auto, kwargs...)
    throw(ArgumentError(_unsupported_region_geometry_message(
        :region_chebyshev_ball,
        pi;
        fallback="If you only need an outer-radius statistic, `region_circumradius(...; method=:bbox)` works whenever `region_bbox` is available."
    )))
end

const _REGION_CHEBYSHEV_BALL_FALLBACK = which(region_chebyshev_ball, Tuple{Any, Int})
@inline _region_supports_chebyshev_ball(pi) = _region_has_custom_method(region_chebyshev_ball, _REGION_CHEBYSHEV_BALL_FALLBACK, Tuple{typeof(pi), Int})

"""
    region_chebyshev_center(pi, r; kwargs...) -> Vector{Float64}

Return the center of `region_chebyshev_ball(pi,r; ...)`.
"""
function region_chebyshev_center(pi, r::Integer; kwargs...)
    return region_chebyshev_ball(pi, r; kwargs...).center
end

"""
    region_inradius(pi, r; kwargs...) -> Float64

Return the radius of `region_chebyshev_ball(pi,r; ...)`.
"""
function region_inradius(pi, r::Integer; kwargs...)
    return region_chebyshev_ball(pi, r; kwargs...).radius
end

"""
    region_circumradius(pi, r; box, center=:bbox, metric=:L2, method=:bbox, kwargs...) -> Float64

Return an outer radius for region `r` intersected with `box=(a,b)`.

This is the radius of a ball (in the specified norm) centered at `center` that
contains the set.

Default behavior is robust across backends: it uses only the region bounding box.

Arguments:
- `center` can be:
  - `:bbox`       (default) center of the region bounding box
  - `:centroid`   region centroid (exact for some backends, otherwise approximate)
  - `:chebyshev`  Chebyshev center (requires `region_chebyshev_ball`)
  - an explicit vector
- `method=:bbox` uses only the axis-aligned bounding box.

Backends may define more accurate methods (e.g. using vertices or cell corners).
"""
function region_circumradius(pi, r::Integer; box=nothing, center=:bbox,
    metric::Symbol=:L2, method::Symbol=:bbox, kwargs...)
    cache = _kwget(kwargs, :cache, nothing)
    closure = _kwget(kwargs, :closure, true)
    strict = _kwget(kwargs, :strict, true)
    box = _resolve_box(box, cache)
    box === nothing && error("region_circumradius: box=(a,b) is required")
    metric = Symbol(metric)
    method = Symbol(method)

    if _REGION_FAST_WRAPPERS[]
        fast = _region_circumradius_fast(pi, r;
            box=box, center=center, metric=metric, method=method,
            strict=strict, closure=closure, cache=cache)
        fast === nothing || return float(fast)
    end

    # Choose the center.
    c = nothing
    if center === :chebyshev
        c = region_chebyshev_center(pi, r; box=box, metric=metric, kwargs...)
    elseif center === :centroid
        c = _region_centroid_maybe_cached(pi, r; box=box, method=:bbox,
            closure=closure, cache=cache)
    elseif center === :bbox
        c = _centroid_from_bbox(_region_bbox_maybe_cached(pi, r; box=box,
            strict=strict, closure=closure, cache=cache))
    else
        c = center
    end

    if method === :bbox
        lo, hi = _region_bbox_maybe_cached(pi, r; box=box,
            strict=strict, closure=closure, cache=cache)
        return _bbox_circumradius(lo, hi, c, metric)
    else
        throw(ArgumentError(
            "region_circumradius: method=$method is not available for backend $(_region_backend_type(pi)). " *
            "Use `method=:bbox` when `region_bbox` is supported, or call `check_region_geometry(pi)` to inspect available geometry queries."
        ))
    end
end

"""
    region_boundary_to_volume_ratio(pi, r; box, volume=nothing, boundary=nothing,
                                    strict=true, closure=true, cache=nothing) -> Float64

Return boundary_measure / volume for region `r` intersected with `box=(a,b)`.

This is not scale-invariant, but is often a useful "boundary-to-volume" statistic.
If volume is zero, returns `Inf` if boundary is positive, and `NaN` if both are zero.

If you already computed `volume` and/or `boundary`, pass them in to avoid recomputation.
"""
function region_boundary_to_volume_ratio(pi, r::Integer; box=nothing,
    volume=nothing, boundary=nothing,
    strict::Bool=true, closure::Bool=true, cache=nothing)
    box === nothing && error("region_boundary_to_volume_ratio: box=(a,b) is required")

    summary = (volume === nothing || boundary === nothing) ? _region_geometry_summary_maybe_fast(pi, r;
        box=box, strict=strict, closure=closure, cache=cache, need_mean_width=false) : nothing
    V = volume === nothing ? (summary === nothing ?
        _region_volume_maybe_cached(pi, r; box=box, closure=closure, cache=cache) :
        float(summary.volume)) : float(volume)
    S = boundary === nothing ? (summary === nothing ?
        _region_boundary_measure_maybe_cached(pi, r; box=box, strict=strict, closure=closure, cache=cache) :
        float(summary.boundary_measure)) : float(boundary)

    if V == 0.0
        return (S == 0.0) ? NaN : Inf
    end
    return S / V
end

"""
    region_isoperimetric_ratio(pi, r; box, volume=nothing, boundary=nothing,
                              kind=:quotient, strict=true, closure=true, cache=nothing) -> Float64

Compute an isoperimetric shape statistic for region `r` intersected with `box=(a,b)`.

Let `V` be the n-dimensional volume and `S` the (n-1)-dimensional boundary measure.
For `n >= 2`, the sharp Euclidean isoperimetric inequality states

    S >= c_n * V^((n-1)/n),   c_n = n * omega_n^(1/n),

where `omega_n` is the volume of the unit n-ball.

Supported `kind` values:
- `:quotient`           returns `c_n * V^((n-1)/n) / S` (1 for Euclidean balls; typically <= 1)
- `:ratio`              reciprocal of `:quotient`
- `:deficit`            `1 - :quotient`
- `:planar`             planar quotient `4*pi*V / S^2` (requires n==2)
- `:boundary_to_volume` returns `S / V` (not scale-invariant)

If `V <= 0` or `S <= 0`, returns `NaN`.
"""
function region_isoperimetric_ratio(pi, r::Integer; box=nothing,
    volume=nothing, boundary=nothing, kind::Symbol=:quotient,
    strict::Bool=true, closure::Bool=true, cache=nothing)
    box === nothing && error("region_isoperimetric_ratio: box=(a,b) is required")

    summary = (volume === nothing || boundary === nothing) ? _region_geometry_summary_maybe_fast(pi, r;
        box=box, strict=strict, closure=closure, cache=cache, need_mean_width=false) : nothing
    V = volume === nothing ? (summary === nothing ?
        _region_volume_maybe_cached(pi, r; box=box, closure=closure, cache=cache) :
        float(summary.volume)) : float(volume)
    S = boundary === nothing ? (summary === nothing ?
        _region_boundary_measure_maybe_cached(pi, r; box=box, strict=strict, closure=closure, cache=cache) :
        float(summary.boundary_measure)) : float(boundary)

    if kind === :boundary_to_volume
        return region_boundary_to_volume_ratio(pi, r; box=box, volume=V, boundary=S,
            strict=strict, closure=closure, cache=cache)
    end

    a, b = box
    n = length(a)
    n >= 2 || error("region_isoperimetric_ratio: requires ambient dimension n>=2, got n=$n")

    if V <= 0.0 || S <= 0.0
        return NaN
    end

    if kind === :planar
        n == 2 || error("region_isoperimetric_ratio(kind=:planar) requires n==2, got n=$n")
        return (4.0 * Base.MathConstants.pi * V) / (S * S)
    end

    cn = _isoperimetric_constant(n)
    q = cn * V^((float(n) - 1.0) / float(n)) / S

    if kind === :quotient
        return q
    elseif kind === :ratio
        return 1.0 / q
    elseif kind === :deficit
        return 1.0 - q
    else
        error("region_isoperimetric_ratio: unknown kind=$kind")
    end
end

# Random directions on S^{n-1}. Columns are unit vectors.
function _random_unit_directions(n::Integer, ndirs::Integer; rng=Random.default_rng())
    n >= 1 || error("random_unit_directions: n must be >= 1")
    ndirs >= 1 || error("random_unit_directions: ndirs must be >= 1")

    U = Matrix{Float64}(undef, n, ndirs)
    @inbounds for j in 1:ndirs
        s2 = 0.0
        for i in 1:n
            u = randn(rng)
            U[i, j] = u
            s2 += u * u
        end
        if s2 == 0.0
            fill!(view(U, :, j), 0.0)
            U[1, j] = 1.0
        else
            invs = inv(sqrt(s2))
            for i in 1:n
                U[i, j] *= invs
            end
        end
    end
    return U
end

@inline function _direction_matrix(directions::Matrix{Float64}, n::Integer, ndirs::Integer;
    rng=Random.default_rng(), enc_cache::Union{Nothing,EncodingCache}=nothing)
    _ = (rng, enc_cache, ndirs)
    size(directions, 1) == n || error("region_mean_width: directions must have size (n,ndirs)")
    return directions
end

@inline function _direction_matrix(directions, n::Integer, ndirs::Integer;
    rng=Random.default_rng(), enc_cache::Union{Nothing,EncodingCache}=nothing)
    if directions !== nothing
        U = Matrix{Float64}(directions)
        size(U, 1) == n || error("region_mean_width: directions must have size (n,ndirs)")
        return U
    end
    if enc_cache !== nothing
        rngh = _rng_cache_hash(rng)
        if rngh !== nothing
            key = _region_direction_cache_key(n, ndirs, rngh)
            cached = _region_geometry_cache_get(enc_cache, key)
            cached === nothing || return cached
            return _region_geometry_cache_set!(enc_cache, key,
                _random_unit_directions(n, ndirs; rng=rng))
        end
    end
    return _random_unit_directions(n, ndirs; rng=rng)
end

function _update_projection_extrema_scalar!(minproj::AbstractVector{Float64}, maxproj::AbstractVector{Float64},
    U::AbstractMatrix{<:Real}, X::AbstractMatrix{Float64}, col::Int)
    @inbounds for j in 1:size(U, 2)
        s = 0.0
        for i in 1:size(U, 1)
            s += U[i, j] * X[i, col]
        end
        if s < minproj[j]
            minproj[j] = s
        end
        if s > maxproj[j]
            maxproj[j] = s
        end
    end
    return nothing
end

function _update_projection_extrema_blocked!(minproj::AbstractVector{Float64}, maxproj::AbstractVector{Float64},
    U::AbstractMatrix{Float64}, accepted::AbstractMatrix{Float64}, naccepted::Int,
    ws::_RegionBatchWorkspace)
    if naccepted <= 0
        return nothing
    end
    if naccepted < _REGION_BLOCKED_PROJECTION_MIN_ACCEPTED[]
        @inbounds for j in 1:naccepted
            _update_projection_extrema_scalar!(minproj, maxproj, U, accepted, j)
        end
        return nothing
    end

    proj = view(ws.proj, 1:size(U, 2), 1:naccepted)
    mul!(proj, transpose(U), view(accepted, 1:size(U, 1), 1:naccepted))
    @inbounds for i in 1:size(U, 2)
        lo = minproj[i]
        hi = maxproj[i]
        for j in 1:naccepted
            s = proj[i, j]
            if s < lo
                lo = s
            end
            if s > hi
                hi = s
            end
        end
        minproj[i] = lo
        maxproj[i] = hi
    end
    return nothing
end

"""
    region_mean_width(pi, r; box, method=:auto, ndirs=256, nsamples=4000,
                      max_proposals=10*nsamples, rng=Random.default_rng(),
                      directions=nothing, strict=true, closure=true, cache=nothing) -> Float64

Estimate the mean width of region `r` intersected with `box=(a,b)`.

Mean width is the average of the width
`w(u) = sup_{x in K} <u,x> - inf_{x in K} <u,x>`
over unit directions `u` on the sphere.

Supported methods:
- `method=:cauchy` (only n==2, convex planar sets): mean_width = perimeter/pi
- `method=:mc`     Monte Carlo estimate using random directions and random points
- `method=:auto`   uses `:cauchy` when n==2, otherwise `:mc`

Backends may implement more accurate methods (e.g. using vertices or cell corners)
by defining a more specific method.

Cost profile:
- `method=:cauchy` is cheap when available (boundary-based, planar),
- `method=:mc` is sampling-based and substantially heavier.
"""
function region_mean_width(pi, r::Integer; box=nothing, method::Symbol=:auto,
    ndirs::Integer=256, nsamples::Integer=4000, max_proposals::Integer=10*nsamples,
    rng=Random.default_rng(), directions=nothing,
    strict::Bool=true, closure::Bool=true, cache=nothing)
    method = Symbol(method)
    box === nothing && error("region_mean_width: box=(a,b) is required")
    a, _ = box
    n = length(a)

    if method === :auto
        method = (n == 2) ? :cauchy : :mc
    end

    if method === :cauchy
        n == 2 || error("region_mean_width(method=:cauchy) requires n==2, got n=$n")
        return region_perimeter(pi, r; box=box, strict=strict, cache=cache) / pi
    elseif method !== :mc
        error("region_mean_width: unknown method=$method (use :auto, :cauchy, :mc)")
    end

    enc_cache = _region_encoding_cache(pi, cache)
    key = _REGION_SAMPLED_SUMMARY_CACHE[] ? _mean_width_cache_key(pi, r;
        box=box, strict=strict, closure=closure, nsamples=nsamples,
        max_proposals=max_proposals, ndirs=ndirs, directions=directions, rng=rng) : nothing
    if key !== nothing
        cached = _region_summary_cache_get(_REGION_MEAN_WIDTH_CACHE,
            enc_cache, :mean_width, key)
        cached === nothing || return cached
    end

    use_batched = _REGION_BATCHED_LOCATE[] &&
        max_proposals >= _REGION_BATCHED_LOCATE_MIN_PROPOSALS[]
    mw = if use_batched
        _region_mean_width_batched(pi, r; box=box, ndirs=ndirs, nsamples=nsamples,
            max_proposals=max_proposals, rng=rng, directions=directions,
            strict=strict, closure=closure, enc_cache=enc_cache)
    else
        _region_mean_width_scalar(pi, r; box=box, ndirs=ndirs, nsamples=nsamples,
        max_proposals=max_proposals, rng=rng, directions=directions,
        strict=strict, closure=closure, enc_cache=enc_cache)
    end

    if key !== nothing
        _region_summary_cache_set!(_REGION_MEAN_WIDTH_CACHE,
            enc_cache, :mean_width, key, float(mw))
    end
    return mw
end

function _region_mean_width_scalar(pi, r::Integer; box,
    ndirs::Integer=256, nsamples::Integer=4000, max_proposals::Integer=10*nsamples,
    rng=Random.default_rng(), directions=nothing,
    strict::Bool=true, closure::Bool=true,
    enc_cache::Union{Nothing,EncodingCache}=nothing)
    box === nothing && error("region_mean_width: box=(a,b) is required")
    a_f, widths = _box_lower_widths(box)
    n = length(a_f)

    U = _direction_matrix(directions, n, ndirs; rng=rng, enc_cache=enc_cache)
    ws = _region_workspace(n, 1, size(U, 2))
    minproj = ws.minproj
    maxproj = ws.maxproj
    fill!(minproj, Inf)
    fill!(maxproj, -Inf)
    x = ws.x
    x0 = Float64[a_f[i] + 0.5 * widths[i] for i in 1:n]
    locate_style = _resolve_locate_style(pi, x0; strict=strict, closure=closure)
    nacc = 0
    proposals = 0
    @inbounds while nacc < nsamples && proposals < max_proposals
        proposals += 1
        _fill_random_point!(x, a_f, widths, rng)
        q = _locate_dispatch(pi, x, locate_style; strict=strict, closure=closure)
        if q == r
            nacc += 1
            ws.accepted[:, 1] = x
            _update_projection_extrema_blocked!(minproj, maxproj, U, ws.accepted, 1, ws)
        elseif q == 0 && strict
            error("region_mean_width: encountered locate()==0; use strict=false or closure=true.")
        end
    end

    if nacc == 0
        return 0.0
    end

    wsum = 0.0
    @inbounds for j in 1:length(minproj)
        if isfinite(minproj[j])
            wsum += (maxproj[j] - minproj[j])
        end
    end
    return wsum / float(length(minproj))
end

function _region_mean_width_batched(pi, r::Integer; box,
    ndirs::Integer=256, nsamples::Integer=4000, max_proposals::Integer=10*nsamples,
    rng=Random.default_rng(), directions=nothing,
    strict::Bool=true, closure::Bool=true,
    enc_cache::Union{Nothing,EncodingCache}=nothing)
    box === nothing && error("region_mean_width: box=(a,b) is required")
    a_f, widths = _box_lower_widths(box)
    n = length(a_f)

    U = _direction_matrix(directions, n, ndirs; rng=rng, enc_cache=enc_cache)
    ws = _region_workspace(n, max(1, _REGION_LOCATE_BATCH_SIZE[]), size(U, 2))
    minproj = ws.minproj
    maxproj = ws.maxproj
    fill!(minproj, Inf)
    fill!(maxproj, -Inf)
    x0 = Float64[a_f[i] + 0.5 * widths[i] for i in 1:n]
    locate_style = _resolve_locate_many_style(pi, x0; strict=strict, closure=closure)

    nacc = 0
    proposals = 0
    @inbounds while nacc < nsamples && proposals < max_proposals
        nbatch = min(size(ws.X, 2), max_proposals - proposals)
        _fill_random_points!(ws.X, a_f, widths, rng, nbatch)
        Xbatch = view(ws.X, :, 1:nbatch)
        locbatch = view(ws.locs, 1:nbatch)
        _locate_many_dispatch!(locbatch, pi, Xbatch, locate_style; strict=strict, closure=closure)
        proposals += nbatch

        naccepted_batch = 0
        for j in 1:nbatch
            q = locbatch[j]
            if q == r
                nacc += 1
                naccepted_batch += 1
                @inbounds for i in 1:n
                    ws.accepted[i, naccepted_batch] = ws.X[i, j]
                end
                nacc >= nsamples && break
            elseif q == 0 && strict
                error("region_mean_width: encountered locate()==0; use strict=false or closure=true.")
            end
        end
        _update_projection_extrema_blocked!(minproj, maxproj, U, ws.accepted, naccepted_batch, ws)
    end

    if nacc == 0
        return 0.0
    end

    wsum = 0.0
    @inbounds for j in eachindex(minproj)
        if isfinite(minproj[j])
            wsum += (maxproj[j] - minproj[j])
        end
    end
    return wsum / float(length(minproj))
end

"""
    region_minkowski_functionals(pi, r; box, volume=nothing, boundary=nothing,
                                 mean_width_method=:auto, mean_width_ndirs=256,
                                 mean_width_rng=Random.default_rng(), mean_width_directions=nothing,
                                 strict=true, closure=true, cache=nothing) -> NamedTuple

Compute a small bundle of Minkowski-type functionals for region `r` intersected with `box=(a,b)`.

Returned fields:
- `volume`           n-dimensional volume
- `boundary_measure` (n-1)-dimensional boundary measure
- `mean_width`       estimated mean width (see `region_mean_width`)

Pass `volume` and/or `boundary` if you already computed them.
"""
function region_minkowski_functionals(pi, r::Integer; box=nothing,
    volume=nothing, boundary=nothing,
    mean_width_method::Symbol=:auto,
    mean_width_ndirs::Integer=256,
    mean_width_rng=Random.default_rng(),
    mean_width_directions=nothing,
    strict::Bool=true, closure::Bool=true, cache=nothing)

    box === nothing && error("region_minkowski_functionals: box=(a,b) is required")

    if _REGION_FAST_WRAPPERS[]
        summary = _region_geometry_summary_maybe_fast(pi, r;
            box=box, strict=strict, closure=closure, cache=cache,
            mean_width_method=mean_width_method, mean_width_ndirs=mean_width_ndirs,
            mean_width_rng=mean_width_rng, mean_width_directions=mean_width_directions,
            need_mean_width=true)
        summary === nothing || return (volume=float(summary.volume),
            boundary_measure=float(summary.boundary_measure),
            mean_width=float(summary.mean_width))
        fast = _region_minkowski_functionals_fast(pi, r;
            box=box, volume=volume, boundary=boundary,
            mean_width_method=mean_width_method, mean_width_ndirs=mean_width_ndirs,
            mean_width_rng=mean_width_rng, mean_width_directions=mean_width_directions,
            strict=strict, closure=closure, cache=cache)
        fast === nothing || return fast
    end

    V = volume === nothing ? _region_volume_maybe_cached(pi, r; box=box, closure=closure, cache=cache) : float(volume)
    S = boundary === nothing ? _region_boundary_measure_maybe_cached(pi, r; box=box, strict=strict, closure=closure, cache=cache) : float(boundary)

    mw = float(_mean_width_cached(pi, r; box=box, method=mean_width_method,
        ndirs=mean_width_ndirs, rng=mean_width_rng,
        directions=mean_width_directions, strict=strict, closure=closure, cache=cache))

    return (volume=V, boundary_measure=S, mean_width=mw)
end

"""
    region_covariance_anisotropy(pi, r; box, kind=:ratio, epsilon=0.0,
                                 nsamples=20000, max_proposals=10*nsamples,
                                 rng=Random.default_rng(), strict=true, closure=true) -> Float64

Compute an anisotropy score from the covariance eigenvalues returned by
`region_principal_directions(pi,r; ...)`.

Let `lambda_max >= ... >= lambda_min` be covariance eigenvalues.
Supported `kind`:
- `:ratio`      lambda_max / max(lambda_min, epsilon)
- `:log_ratio`  log(lambda_max / max(lambda_min, epsilon))
- `:normalized` (lambda_max - lambda_min) / (lambda_max + lambda_min + epsilon)

If lambda_max == 0 (degenerate region), returns 1 for `:ratio`, and 0 otherwise.
"""
function region_covariance_anisotropy(pi, r::Integer; box=nothing,
    kind::Symbol=:ratio, epsilon::Real=0.0,
    nsamples::Integer=20000, max_proposals::Integer=10*nsamples,
    rng=Random.default_rng(), strict::Bool=true, closure::Bool=true)

    box === nothing && error("region_covariance_anisotropy: box=(a,b) is required")

    entry = _principal_summary_entry_maybe_closure(pi, r; box=box, nsamples=nsamples,
        rng=rng, strict=strict, closure=closure, max_proposals=max_proposals)

    evals = entry.evals
    isempty(evals) && return NaN
    lmax = float(evals[1])
    lmin = float(evals[end])
    eps = float(epsilon)

    if lmax == 0.0
        return (kind === :ratio) ? 1.0 : 0.0
    end

    if kind === :ratio
        return lmax / max(lmin, eps)
    elseif kind === :log_ratio
        return log(lmax / max(lmin, eps))
    elseif kind === :normalized
        return (lmax - lmin) / (lmax + lmin + eps)
    else
        error("region_covariance_anisotropy: unknown kind=$kind")
    end
end

"""
    region_covariance_eccentricity(pi, r; box, epsilon=0.0, kwargs...) -> Float64

An "eccentricity" score derived from covariance eigenvalues:

    ecc = sqrt(max(0, 1 - lambda_min / max(lambda_max, epsilon))).

Returns 0 for isotropic or point-like regions.
"""
function region_covariance_eccentricity(pi, r::Integer; box=nothing,
    epsilon::Real=0.0,
    nsamples::Integer=20000, max_proposals::Integer=10*nsamples,
    rng=Random.default_rng(), strict::Bool=true, closure::Bool=true)

    box === nothing && error("region_covariance_eccentricity: box=(a,b) is required")

    entry = _principal_summary_entry_maybe_closure(pi, r; box=box, nsamples=nsamples,
        rng=rng, strict=strict, closure=closure, max_proposals=max_proposals)

    evals = entry.evals
    isempty(evals) && return NaN
    lmax = float(evals[1])
    lmin = float(evals[end])
    eps = float(epsilon)

    if lmax == 0.0
        return 0.0
    end

    t = 1.0 - (lmin / max(lmax, eps))
    return sqrt(max(t, 0.0))
end

# -----------------------------------------------------------------------------
# Internal helpers for covariance-based anisotropy scores.
#
# These are intentionally small, pure numeric helpers. They are kept unexported.
# -----------------------------------------------------------------------------

@inline function _covariance_anisotropy_from_evals(evals::AbstractVector{<:Real};
    kind::Symbol = :ratio,
    epsilon::Real = 0.0
)
    isempty(evals) && return NaN
    # Ensure we use float arithmetic.
    lmax = float(evals[1])
    lmin = float(evals[end])
    eps = float(epsilon)

    if !(isfinite(lmax) && isfinite(lmin))
        return NaN
    end

    if lmax <= 0.0
        return kind == :log_ratio ? 0.0 : 1.0
    end

    denom = max(lmin, eps)
    denom > 0.0 || return kind == :log_ratio ? Inf : Inf

    if kind == :ratio
        return lmax / denom
    elseif kind == :log_ratio
        return log(lmax / denom)
    elseif kind == :normalized
        # Scale-free proxy in [0,1) when evals are nonnegative.
        return (lmax - denom) / (lmax + denom + eps)
    else
        error("_covariance_anisotropy_from_evals: unknown kind=$(kind); use :ratio, :log_ratio, or :normalized")
    end
end

@inline function _covariance_eccentricity_from_evals(evals::AbstractVector{<:Real};
    epsilon::Real = 0.0
)
    isempty(evals) && return NaN
    lmax = float(evals[1])
    lmin = float(evals[end])
    eps = float(epsilon)

    if !(isfinite(lmax) && isfinite(lmin))
        return NaN
    end

    if lmax <= 0.0
        return 0.0
    end

    denom = max(lmax, eps)
    denom > 0.0 || return 0.0
    t = 1.0 - lmin / denom
    return sqrt(max(0.0, t))
end


"""
    region_anisotropy_scores(pi, r; box, epsilon=0.0, nsamples=20_000,
        max_proposals=10*nsamples, rng=Random.default_rng(),
        strict=true, closure=true, return_info=false, nbatches=0)

Compute several scale-invariant anisotropy scores derived from the covariance
matrix of a uniform point in region `r` (restricted to the finite `box`).

Output fields:
- `ratio`       : lambda_max / lambda_min (>= 1)
- `log_ratio`   : log(ratio)
- `normalized`  : (lambda_max - lambda_min) / (lambda_max + lambda_min) in [0,1)
- `eccentricity`: sqrt(1 - lambda_min/lambda_max) in [0,1)

If `epsilon > 0`, eigenvalues smaller than `epsilon` are clamped.

If `return_info=true`, requests batched diagnostics from
`region_principal_directions` and adds:
- `ratio_stderr`, `log_ratio_stderr`, `normalized_stderr`, `eccentricity_stderr`
computed from batch-to-batch variability (standard error of the estimate),
and includes `pca=pd` with sampling diagnostics.

This answers: "how stable is this feature under sampling?"

Cost profile:
- sampling-based / diagnostic-heavy,
- best used after cheaper bbox-based summaries suggest anisotropy is worth
  measuring more carefully.
"""
function region_anisotropy_scores(pi, r::Integer;
    box=nothing,
    epsilon::Real=0.0,
    nsamples::Int=20_000,
    max_proposals::Int=10*nsamples,
    rng::AbstractRNG=Random.default_rng(),
    strict::Bool=true,
    closure::Bool=true,
    return_info::Bool=false,
    nbatches::Int=0
)
    box === nothing && error("A finite box is required for region_anisotropy_scores.")

    if !return_info
        entry = _principal_summary_entry_maybe_closure(pi, r;
            box=box,
            nsamples=nsamples,
            max_proposals=max_proposals,
            rng=rng,
            strict=strict,
            closure=closure,
            return_info=false,
            nbatches=0)
        evals = entry.evals
        ratio = _covariance_anisotropy_from_evals(evals; kind=:ratio, epsilon=epsilon)
        log_ratio = _covariance_anisotropy_from_evals(evals; kind=:log_ratio, epsilon=epsilon)
        normalized = _covariance_anisotropy_from_evals(evals; kind=:normalized, epsilon=epsilon)
        ecc = _covariance_eccentricity_from_evals(evals; epsilon=epsilon)
        return (ratio=ratio, log_ratio=log_ratio, normalized=normalized, eccentricity=ecc)
    end

    pd = _principal_directions_maybe_closure(pi, r;
        box=box,
        nsamples=nsamples,
        max_proposals=max_proposals,
        rng=rng,
        strict=strict,
        closure=closure,
        return_info=true,
        nbatches=nbatches)

    evals = pd.evals

    ratio = _covariance_anisotropy_from_evals(evals; kind=:ratio, epsilon=epsilon)
    log_ratio = _covariance_anisotropy_from_evals(evals; kind=:log_ratio, epsilon=epsilon)
    normalized = _covariance_anisotropy_from_evals(evals; kind=:normalized, epsilon=epsilon)
    ecc = _covariance_eccentricity_from_evals(evals; epsilon=epsilon)

    ratio_se = NaN
    log_ratio_se = NaN
    normalized_se = NaN
    ecc_se = NaN

    if haskey(pd, :batch_evals) && (pd.batch_evals !== nothing) && (_batch_eval_count(pd.batch_evals) >= 2)
        k = _batch_eval_count(pd.batch_evals)

        sr = 0.0; ssr = 0.0
        slr = 0.0; sslr = 0.0
        sn = 0.0; ssn = 0.0
        se = 0.0; sse = 0.0

        for j in 1:k
            bevals = _batch_eval_column(pd.batch_evals, j)
            rb = _covariance_anisotropy_from_evals(bevals; kind=:ratio, epsilon=epsilon)
            lrb = _covariance_anisotropy_from_evals(bevals; kind=:log_ratio, epsilon=epsilon)
            nb = _covariance_anisotropy_from_evals(bevals; kind=:normalized, epsilon=epsilon)
            eb = _covariance_eccentricity_from_evals(bevals; epsilon=epsilon)

            sr += rb; ssr += rb*rb
            slr += lrb; sslr += lrb*lrb
            sn += nb; ssn += nb*nb
            se += eb; sse += eb*eb
        end

        mr = sr / k
        var_r = (ssr - k*mr*mr) / (k - 1)
        var_r = var_r < 0.0 ? 0.0 : var_r
        ratio_se = sqrt(var_r) / sqrt(k)

        mlr = slr / k
        var_lr = (sslr - k*mlr*mlr) / (k - 1)
        var_lr = var_lr < 0.0 ? 0.0 : var_lr
        log_ratio_se = sqrt(var_lr) / sqrt(k)

        mn = sn / k
        var_n = (ssn - k*mn*mn) / (k - 1)
        var_n = var_n < 0.0 ? 0.0 : var_n
        normalized_se = sqrt(var_n) / sqrt(k)

        me = se / k
        var_e = (sse - k*me*me) / (k - 1)
        var_e = var_e < 0.0 ? 0.0 : var_e
        ecc_se = sqrt(var_e) / sqrt(k)
    end

    return (ratio=ratio, ratio_stderr=ratio_se,
        log_ratio=log_ratio, log_ratio_stderr=log_ratio_se,
        normalized=normalized, normalized_stderr=normalized_se,
        eccentricity=ecc, eccentricity_stderr=ecc_se,
        pca=pd)
end

"""
    region_anisotropy_scores(summary::RegionPrincipalDirectionsSummary; epsilon=summary.report.epsilon)

Compute anisotropy scores from an already-built principal-direction summary.

This avoids re-running the sampling workflow when you already have a
`RegionPrincipalDirectionsSummary`.
"""
function region_anisotropy_scores(summary::RegionPrincipalDirectionsSummary; epsilon::Real=summary.report.epsilon)
    evals = principal_values(summary)
    return (
        ratio = _covariance_anisotropy_from_evals(evals; kind=:ratio, epsilon=epsilon),
        log_ratio = _covariance_anisotropy_from_evals(evals; kind=:log_ratio, epsilon=epsilon),
        normalized = _covariance_anisotropy_from_evals(evals; kind=:normalized, epsilon=epsilon),
        eccentricity = _covariance_eccentricity_from_evals(evals; epsilon=epsilon),
    )
end

"""
    Base.show(io::IO, summary::RegionGeometryValidationSummary)

Compact display for the report returned by `region_geometry_validation_summary`.
"""
function Base.show(io::IO, summary::RegionGeometryValidationSummary)
    report = summary.report
    nhooks = count(identity, values(report.hooks))
    nqueries = count(identity, values(report.queries))
    print(io, "RegionGeometryValidationSummary(backend=", report.backend,
        ", valid=", report.valid,
        ", hooks=", nhooks, "/", length(report.hooks),
        ", queries=", nqueries, "/", length(report.queries), ")")
end

"""
    Base.show(io::IO, ::MIME\"text/plain\", summary::RegionGeometryValidationSummary)

Plain-text display for the report returned by `region_geometry_validation_summary`.
"""
function Base.show(io::IO, ::MIME"text/plain", summary::RegionGeometryValidationSummary)
    report = summary.report
    println(io, "RegionGeometryValidationSummary")
    println(io, "  backend: ", report.backend)
    println(io, "  valid:   ", report.valid)
    supported_hooks = [String(k) for (k, v) in pairs(report.hooks) if v]
    supported_queries = [String(k) for (k, v) in pairs(report.queries) if v]
    println(io, "  supported hooks:   ", isempty(supported_hooks) ? "(none)" : join(supported_hooks, ", "))
    println(io, "  supported queries: ", isempty(supported_queries) ? "(none)" : join(supported_queries, ", "))
    if isempty(report.issues)
        print(io, "  issues: none")
    else
        println(io, "  issues:")
        for issue in report.issues
            println(io, "    - ", issue)
        end
    end
end

"""
    Base.show(io::IO, summary::RegionGeometrySummary)

Compact display for region geometry summaries.
"""
function Base.show(io::IO, summary::RegionGeometrySummary)
    report = summary.report
    print(io, "RegionGeometrySummary(backend=", report.backend,
        ", region=", report.region,
        ", finite_box=", report.finite_box,
        ", available=", length(report.available),
        ", unavailable=", length(report.unavailable), ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain\", summary::RegionGeometrySummary)

Plain-text display for region geometry summaries.
"""
function Base.show(io::IO, ::MIME"text/plain", summary::RegionGeometrySummary)
    report = summary.report
    println(io, "RegionGeometrySummary")
    println(io, "  backend:         ", report.backend)
    println(io, "  region:          ", report.region)
    println(io, "  finite_box:      ", report.finite_box)
    println(io, "  available:       ", isempty(report.available) ? "(none)" : join(string.(report.available), ", "))
    println(io, "  quantity_sources:")
    for (name, source) in pairs(report.quantity_sources)
        println(io, "    - ", name, " => ", source)
    end
    if isempty(report.unavailable)
        println(io, "  unavailable:     (none)")
    else
        println(io, "  unavailable:")
        for pair in report.unavailable
            println(io, "    - ", first(pair), ": ", last(pair))
        end
    end
    if isempty(report.issues)
        print(io, "  issues:          none")
    else
        println(io, "  issues:")
        for issue in report.issues
            println(io, "    - ", issue)
        end
    end
end

"""
    Base.show(io::IO, summary::RegionPrincipalDirectionsSummary)

Compact display for principal-direction summaries.
"""
function Base.show(io::IO, summary::RegionPrincipalDirectionsSummary)
    report = summary.report
    print(io, "RegionPrincipalDirectionsSummary(backend=", report.backend,
        ", region=", report.region,
        ", ambient_dim=", length(report.evals),
        ", n_accepted=", report.n_accepted,
        ", n_proposed=", report.n_proposed, ")")
end

"""
    Base.show(io::IO, ::MIME\"text/plain\", summary::RegionPrincipalDirectionsSummary)

Plain-text display for principal-direction summaries.
"""
function Base.show(io::IO, ::MIME"text/plain", summary::RegionPrincipalDirectionsSummary)
    report = summary.report
    println(io, "RegionPrincipalDirectionsSummary")
    println(io, "  backend:      ", report.backend)
    println(io, "  region:       ", report.region)
    println(io, "  ambient_dim:  ", length(report.evals))
    println(io, "  n_accepted:   ", report.n_accepted)
    println(io, "  n_proposed:   ", report.n_proposed)
    println(io, "  principal_values: ", join(string.(report.evals), ", "))
    print(io, "  anisotropy: ratio=", report.anisotropy.ratio,
        ", normalized=", report.anisotropy.normalized,
        ", eccentricity=", report.anisotropy.eccentricity)
end

end # module
