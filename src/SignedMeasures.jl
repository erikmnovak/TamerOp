module SignedMeasures
# -----------------------------------------------------------------------------
# SignedMeasures.jl
#
# SignedMeasures owner module extracted from Invariants.jl.
# -----------------------------------------------------------------------------

"""
    SignedMeasures

Owner module for signed-measure style multiparameter invariants, including:

- Euler characteristic surfaces and Euler signed measures on finite grids
- signed point measures and their kernels
- signed rectangle barcodes and their derived summaries
- MMA-style wrappers that package those signed outputs together

This module owns the signed-measure families themselves. It does not own the
basic rank/Hilbert invariant layer (`Invariants`) or the generic shared
plumbing (`InvariantCore`).
"""

using LinearAlgebra
using JSON3
using ..CoreModules: EncodingCache, SessionCache, AbstractCoeffField, GeometryCachePayload,
                     RegionPosetCachePayload, AbstractSlicePlanCache,
                     _resolve_workflow_session_cache, _workflow_encoding_cache
using ..Options: InvariantOptions
using ..EncodingCore: PLikeEncodingMap, CompiledEncoding, GridEncodingMap,
                      locate, axes_from_encoding, dimension, representatives
using Statistics: mean
using ..Stats: _wilson_interval
using ..Encoding: EncodingMap
using ..PLPolyhedra
using ..RegionGeometry: region_weights, region_volume, region_bbox, region_widths,
                        region_centroid, region_aspect_ratio, region_diameter,
                        region_adjacency, region_facet_count, region_vertex_count,
                        region_boundary_measure, region_boundary_measure_breakdown,
                        region_perimeter, region_surface_area,
                        region_principal_directions,
                        region_chebyshev_ball, region_chebyshev_center, region_inradius,
                        region_circumradius,
                        region_boundary_to_volume_ratio, region_isoperimetric_ratio,
                        region_mean_width, region_minkowski_functionals,
                        region_covariance_anisotropy, region_covariance_eccentricity, region_anisotropy_scores
using ..FieldLinAlg
using ..InvariantCore: SliceSpec, RankQueryCache,
                       _unwrap_compiled,
                       _default_strict, _default_threads, _drop_keys,
                       orthant_directions, _selection_kwargs_from_opts, _axes_kwargs_from_opts,
                       _eye, rank_map,
                       RANK_INVARIANT_MEMO_THRESHOLD, RECTANGLE_LOC_LINEAR_CACHE_THRESHOLD,
                       _use_array_memo, _new_array_memo, _grid_cache_index,
                       _memo_get, _memo_set!, _map_leq_cached,
                       _rank_cache_get!, _resolve_rank_query_cache,
                       _rank_query_point_tuple, _rank_query_locate!
import ..FiniteFringe: AbstractPoset, FinitePoset, FringeModule, Upset, Downset, fiber_dimension,
                       leq, leq_matrix, upset_indices, downset_indices, leq_col, nvertices, build_cache!,
                       _preds
import ..ZnEncoding
import ..IndicatorTypes: UpsetPresentation, DownsetCopresentation
import ..ModuleComplexes: ModuleCochainComplex
import ..ChangeOfPosets: pushforward_left, pushforward_right
import Base.Threads
import ..ChainComplexes: describe
import ..Serialization: save_mpp_decomposition_json, load_mpp_decomposition_json,
                        save_mpp_image_json, load_mpp_image_json
import ..Modules: PModule, map_leq, CoverCache, _get_cover_cache
import ..IndicatorResolutions: pmodule_from_fringe
import ..ZnEncoding: ZnEncodingMap

@inline _slice_invariants_module() = getfield(parentmodule(@__MODULE__), :SliceInvariants)
@inline _multiparameter_images_module() = getfield(parentmodule(@__MODULE__), :MultiparameterImages)

"""
    Rect{N}

An axis-aligned hyperrectangle in Z^N, represented by two corners `lo` and `hi`
with `lo <= hi` coordinatewise.

This type is used to represent "rectangle signed barcodes" (also called
"signed rectangle measures") obtained by Mobius inversion of the rank invariant.
"""
struct Rect{N}
    lo::NTuple{N,Int}
    hi::NTuple{N,Int}
    function Rect{N}(lo::NTuple{N,Int}, hi::NTuple{N,Int}) where {N}
        _tuple_leq(lo, hi) || error("Rect: expected lo <= hi coordinatewise")
        return new{N}(lo, hi)
    end
end

# Coordinatewise partial order on integer tuples.
_tuple_leq(a::NTuple{N,Int}, b::NTuple{N,Int}) where {N} = all(a[i] <= b[i] for i in 1:N)

function Base.show(io::IO, r::Rect{N}) where {N}
    print(io, "Rect", r.lo, " => ", r.hi)
end

"""
    RectValidationSummary

Typed validation report returned by [`check_rect`](@ref).

Use `summary.valid` for a cheap notebook-friendly success flag and
`summary.errors` for the explicit list of contract violations.
"""
struct RectValidationSummary
    valid::Bool
    errors::Vector{String}
end

"""
    RectSignedBarcodeValidationSummary

Typed validation report for [`check_rect_signed_barcode`](@ref) and
`validate(sb::RectSignedBarcode; ...)`.
"""
struct RectSignedBarcodeValidationSummary
    valid::Bool
    errors::Vector{String}
end

"""
    PointSignedMeasureValidationSummary

Typed validation report for [`check_point_signed_measure`](@ref) and
`validate(pm::PointSignedMeasure; ...)`.
"""
struct PointSignedMeasureValidationSummary
    valid::Bool
    errors::Vector{String}
end

"""
    SignedMeasureDecompositionValidationSummary

Typed validation report for [`check_signed_measure_decomposition`](@ref) and
`validate(result::SignedMeasureDecomposition; ...)`.
"""
struct SignedMeasureDecompositionValidationSummary
    valid::Bool
    errors::Vector{String}
end

@inline _validation_issue_count(summary) = length(summary.errors)

function Base.show(io::IO, summary::RectValidationSummary)
    print(io, "RectValidationSummary(valid=", summary.valid,
          ", errors=", _validation_issue_count(summary), ")")
end

function Base.show(io::IO, summary::RectSignedBarcodeValidationSummary)
    print(io, "RectSignedBarcodeValidationSummary(valid=", summary.valid,
          ", errors=", _validation_issue_count(summary), ")")
end

function Base.show(io::IO, summary::PointSignedMeasureValidationSummary)
    print(io, "PointSignedMeasureValidationSummary(valid=", summary.valid,
          ", errors=", _validation_issue_count(summary), ")")
end

function Base.show(io::IO, summary::SignedMeasureDecompositionValidationSummary)
    print(io, "SignedMeasureDecompositionValidationSummary(valid=", summary.valid,
          ", errors=", _validation_issue_count(summary), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::RectValidationSummary)
    print(io,
          "RectValidationSummary",
          "\n  valid: ", summary.valid,
          "\n  errors: ", repr(summary.errors))
end

function Base.show(io::IO, ::MIME"text/plain", summary::RectSignedBarcodeValidationSummary)
    print(io,
          "RectSignedBarcodeValidationSummary",
          "\n  valid: ", summary.valid,
          "\n  errors: ", repr(summary.errors))
end

function Base.show(io::IO, ::MIME"text/plain", summary::PointSignedMeasureValidationSummary)
    print(io,
          "PointSignedMeasureValidationSummary",
          "\n  valid: ", summary.valid,
          "\n  errors: ", repr(summary.errors))
end

function Base.show(io::IO, ::MIME"text/plain", summary::SignedMeasureDecompositionValidationSummary)
    print(io,
          "SignedMeasureDecompositionValidationSummary",
          "\n  valid: ", summary.valid,
          "\n  errors: ", repr(summary.errors))
end

"""
    RectSignedBarcode{N,T}

A finite signed multiset of axis-aligned hyperrectangles in Z^N.

Fields:
- `axes`: the coordinate grids used for the inversion, one sorted vector per axis.
- `rects`: the rectangles (elements of the free abelian group on rectangles).
- `weights`: the integer weights (can be negative).

Interpretation:
Given a rank invariant function `r(p,q)` on comparable pairs `p <= q` in the grid,
Mobius inversion produces weights `w(lo,hi)` such that

    r(p,q) = sum_{lo <= p, q <= hi} w(lo,hi)

for all grid pairs `p <= q`. For N=1 this recovers the usual barcode multiplicities.
For N>1 the result is typically signed, hence the name "signed barcode".
"""
struct RectSignedBarcode{N,T<:Integer}
    axes::NTuple{N,Vector{Int}}
    rects::Vector{Rect{N}}
    weights::Vector{T}
end

# =============================================================================
# Signed point measures (Euler signed-measure / Mobius inversion on a grid)
# =============================================================================

"""
    PointSignedMeasure(axes, inds, wts)

A sparse signed measure supported on a rectangular grid.

- `axes` is an `N`-tuple of coordinate vectors `(a1, ..., aN)` describing the grid.
- `inds` is a vector of `N`-tuples of *indices* into those axes.
- `wts` is the corresponding vector of signed weights.

This is the point-measure analogue of `RectSignedBarcode`.  It is the natural
output type for an Euler signed-measure decomposition, but it can also be used
for any grid function whose Mobius inversion (on the product-of-chains poset)
is desired.

Notes for mathematicians:
- If `f : A1 x ... x AN -> Z` is a function on a finite product of chains,
  then the Mobius inversion `mu` satisfies

      f(x) = sum_{y <= x} mu(y),

  where `<=` is the product order.  `PointSignedMeasure` stores `mu`.
"""
struct PointSignedMeasure{N,T,W}
    axes::NTuple{N,Vector{T}}
    inds::Vector{NTuple{N,Int}}
    wts::Vector{W}
end

Base.length(pm::PointSignedMeasure) = length(pm.wts)

@inline ambient_dimension(::Rect{N}) where {N} = N

"""
    lower_corner(rect)
    upper_corner(rect)
    side_lengths(rect)
    grid_span(rect)
    center(rect)

Cheap geometric accessors for a `Rect`.

- `lower_corner(rect)` and `upper_corner(rect)` return the integer corners.
- `side_lengths(rect)` returns `hi - lo` coordinatewise.
- `grid_span(rect)` returns the inclusive lattice span `hi - lo + 1`.
- `center(rect)` returns the midpoint as floating-point coordinates.
"""
@inline lower_corner(rect::Rect{N}) where {N} = rect.lo
@inline upper_corner(rect::Rect{N}) where {N} = rect.hi
@inline side_lengths(rect::Rect{N}) where {N} = ntuple(i -> rect.hi[i] - rect.lo[i], N)
@inline grid_span(rect::Rect{N}) where {N} = ntuple(i -> rect.hi[i] - rect.lo[i] + 1, N)
@inline center(rect::Rect{N}) where {N} = ntuple(i -> 0.5 * (rect.lo[i] + rect.hi[i]), N)

"""
    is_degenerate(rect)
    contains(rect, p)
    intersects(r1, r2)
    intersection(r1, r2)

Cheap query helpers for rectangles.

- `is_degenerate(rect)` reports whether at least one side has zero length.
- `contains(rect, p)` checks whether an integer lattice point lies in `rect`.
- `intersects(r1, r2)` checks axis-aligned intersection.
- `intersection(r1, r2)` returns the overlap rectangle, or `nothing`.
"""
@inline is_degenerate(rect::Rect{N}) where {N} = any(i -> rect.lo[i] == rect.hi[i], 1:N)

@inline function contains(rect::Rect{N}, point::NTuple{N,<:Integer}) where {N}
    return all(i -> rect.lo[i] <= point[i] <= rect.hi[i], 1:N)
end

@inline contains(rect::Rect, point::AbstractVector{<:Integer}) = contains(rect, Tuple(point))

@inline function intersects(r1::Rect{N}, r2::Rect{N}) where {N}
    return all(i -> max(r1.lo[i], r2.lo[i]) <= min(r1.hi[i], r2.hi[i]), 1:N)
end

function intersection(r1::Rect{N}, r2::Rect{N}) where {N}
    intersects(r1, r2) || return nothing
    lo = ntuple(i -> max(r1.lo[i], r2.lo[i]), N)
    hi = ntuple(i -> min(r1.hi[i], r2.hi[i]), N)
    return Rect{N}(lo, hi)
end

Base.length(sb::RectSignedBarcode) = length(sb.rects)
Base.axes(sb::RectSignedBarcode) = sb.axes
@inline rectangles(sb::RectSignedBarcode) = sb.rects
@inline weights(sb::RectSignedBarcode) = sb.weights
@inline ambient_dimension(sb::RectSignedBarcode) = length(sb.axes)
@inline nterms(sb::RectSignedBarcode) = length(sb)
@inline positive_terms(sb::RectSignedBarcode) = count(>(0), sb.weights)
@inline negative_terms(sb::RectSignedBarcode) = count(<(0), sb.weights)
@inline total_mass(sb::RectSignedBarcode) = sum(sb.weights)
@inline total_variation(sb::RectSignedBarcode) = sum(abs, sb.weights)
@inline axis_lengths(sb::RectSignedBarcode) = map(length, sb.axes)
@inline weight_range(sb::RectSignedBarcode) = isempty(sb.weights) ? nothing : (minimum(sb.weights), maximum(sb.weights))
@inline max_abs_weight(sb::RectSignedBarcode) = isempty(sb.weights) ? zero(eltype(sb.weights)) : maximum(abs, sb.weights)
@inline support_size(sb::RectSignedBarcode) = length(Set(sb.rects))

Base.axes(pm::PointSignedMeasure) = pm.axes
@inline support_indices(pm::PointSignedMeasure) = pm.inds
@inline weights(pm::PointSignedMeasure) = pm.wts
@inline ambient_dimension(pm::PointSignedMeasure) = length(pm.axes)
@inline nterms(pm::PointSignedMeasure) = length(pm)
@inline positive_terms(pm::PointSignedMeasure) = count(>(0), pm.wts)
@inline negative_terms(pm::PointSignedMeasure) = count(<(0), pm.wts)
@inline total_mass(pm::PointSignedMeasure) = sum(pm.wts)
@inline total_variation(pm::PointSignedMeasure) = sum(abs, pm.wts)
@inline axis_lengths(pm::PointSignedMeasure) = map(length, pm.axes)
@inline weight_range(pm::PointSignedMeasure) = isempty(pm.wts) ? nothing : (minimum(pm.wts), maximum(pm.wts))
@inline max_abs_weight(pm::PointSignedMeasure) = isempty(pm.wts) ? zero(eltype(pm.wts)) : maximum(abs, pm.wts)
@inline support_size(pm::PointSignedMeasure) = length(Set(pm.inds))

@doc """
    axis_lengths(x)
    weight_range(x)
    max_abs_weight(x)
    support_size(x)

Cheap scalar exploration helpers for `RectSignedBarcode` and
`PointSignedMeasure`.

These are meant for notebook/REPL inspection before heavier queries such as
full `describe(...)`, `largest_terms(...)`, or kernel/image computations.
""" axis_lengths

@inline _axes_sorted_unique(ax::AbstractVector) = issorted(ax) && allunique(ax)

@inline function _largest_term_indices(weights_vec::AbstractVector, n::Int)
    n <= 0 && return Int[]
    idx = collect(eachindex(weights_vec))
    sort!(idx, by = i -> (-abs(weights_vec[i]), i))
    resize!(idx, min(length(idx), n))
    return idx
end

@inline function _point_from_index(pm::PointSignedMeasure{N}, idx::NTuple{N,Int}) where {N}
    return ntuple(k -> pm.axes[k][idx[k]], N)
end

function points(pm::PointSignedMeasure{N}) where {N}
    out = Vector{NTuple{N,eltype(pm.axes[1])}}(undef, length(pm))
    @inbounds for i in eachindex(pm.inds)
        out[i] = _point_from_index(pm, pm.inds[i])
    end
    return out
end

"""
    term(obj, i)

Return the `i`th signed term of a signed-measure container as a semantic named
tuple.

- For `RectSignedBarcode`, the result contains `rect`, `lower`, `upper`, and
  `weight`.
- For `PointSignedMeasure`, the result contains `index`, `point`, and `weight`.
"""
@inline function term(sb::RectSignedBarcode, i::Integer)
    rect = sb.rects[i]
    return (rect = rect, lower = lower_corner(rect), upper = upper_corner(rect), weight = sb.weights[i])
end

@inline function term(pm::PointSignedMeasure, i::Integer)
    idx = pm.inds[i]
    return (index = idx, point = _point_from_index(pm, idx), weight = pm.wts[i])
end

function largest_terms(sb::RectSignedBarcode; n::Int = 10)
    return [term(sb, i) for i in _largest_term_indices(sb.weights, n)]
end

function largest_terms(pm::PointSignedMeasure; n::Int = 10)
    return [term(pm, i) for i in _largest_term_indices(pm.wts, n)]
end

@doc """
    largest_terms(obj; n=10)

Return up to `n` terms with largest absolute weight.

This is the recommended cheap-first inspection helper for signed-measure
objects in notebooks and REPL sessions.
""" largest_terms

@doc """
    coefficient(sb, rect)
    coefficient(pm, point)

Return the total coefficient of an exact rectangle or point in the signed
object.

Repeated terms are summed, so the result is the coefficient in the free
abelian-group sense rather than just the first matching stored term.
""" coefficient
function coefficient(sb::RectSignedBarcode{N}, rect::Rect{N}) where {N}
    coeff = zero(eltype(sb.weights))
    @inbounds for i in eachindex(sb.rects)
        sb.rects[i] == rect || continue
        coeff += sb.weights[i]
    end
    return coeff
end

function coefficient(pm::PointSignedMeasure{N}, point::NTuple{N,<:Real}) where {N}
    coeff = zero(eltype(pm.wts))
    @inbounds for i in eachindex(pm.inds)
        _point_from_index(pm, pm.inds[i]) == point || continue
        coeff += pm.wts[i]
    end
    return coeff
end

function coefficient(pm::PointSignedMeasure, point::AbstractVector{<:Real})
    return coefficient(pm, Tuple(point))
end

function support_bbox(sb::RectSignedBarcode{N}) where {N}
    isempty(sb.rects) && return nothing
    lo = collect(lower_corner(sb.rects[1]))
    hi = collect(upper_corner(sb.rects[1]))
    @inbounds for rect in sb.rects
        for k in 1:N
            lo[k] = min(lo[k], rect.lo[k])
            hi[k] = max(hi[k], rect.hi[k])
        end
    end
    return (lo = Tuple(lo), hi = Tuple(hi))
end

function support_bbox(pm::PointSignedMeasure{N}) where {N}
    isempty(pm.inds) && return nothing
    first_pt = _point_from_index(pm, pm.inds[1])
    lo = collect(first_pt)
    hi = collect(first_pt)
    @inbounds for idx in pm.inds
        pt = _point_from_index(pm, idx)
        for k in 1:N
            lo[k] = min(lo[k], pt[k])
            hi[k] = max(hi[k], pt[k])
        end
    end
    return (lo = Tuple(lo), hi = Tuple(hi))
end

@doc """
    support_bbox(obj)

Return the axis-aligned bounding box of the support of a `RectSignedBarcode` or
`PointSignedMeasure`, or `nothing` if the object is empty.
""" support_bbox

function describe(sb::RectSignedBarcode; nlargest::Int = 5)
    return (
        kind = :rect_signed_barcode,
        ambient_dimension = ambient_dimension(sb),
        axis_lengths = axis_lengths(sb),
        nterms = nterms(sb),
        positive_terms = positive_terms(sb),
        negative_terms = negative_terms(sb),
        total_mass = total_mass(sb),
        total_variation = total_variation(sb),
        largest_terms = largest_terms(sb; n = nlargest),
    )
end

function describe(pm::PointSignedMeasure; nlargest::Int = 5)
    return (
        kind = :point_signed_measure,
        ambient_dimension = ambient_dimension(pm),
        axis_lengths = axis_lengths(pm),
        nterms = nterms(pm),
        positive_terms = positive_terms(pm),
        negative_terms = negative_terms(pm),
        total_mass = total_mass(pm),
        total_variation = total_variation(pm),
        largest_terms = largest_terms(pm; n = nlargest),
    )
end

function Base.show(io::IO, sb::RectSignedBarcode)
    d = describe(sb; nlargest = 3)
    print(io,
          "RectSignedBarcode(dim=", d.ambient_dimension,
          ", terms=", d.nterms,
          ", mass=", d.total_mass,
          ", variation=", d.total_variation, ")")
end

function Base.show(io::IO, ::MIME"text/plain", sb::RectSignedBarcode)
    d = describe(sb; nlargest = 5)
    print(io,
          "RectSignedBarcode",
          "\n  ambient_dimension: ", d.ambient_dimension,
          "\n  axis_lengths: ", repr(d.axis_lengths),
          "\n  nterms: ", d.nterms,
          "\n  positive_terms: ", d.positive_terms,
          "\n  negative_terms: ", d.negative_terms,
          "\n  total_mass: ", d.total_mass,
          "\n  total_variation: ", d.total_variation,
          "\n  largest_terms: ", repr(d.largest_terms))
end

function Base.show(io::IO, pm::PointSignedMeasure)
    d = describe(pm; nlargest = 3)
    print(io,
          "PointSignedMeasure(dim=", d.ambient_dimension,
          ", terms=", d.nterms,
          ", mass=", d.total_mass,
          ", variation=", d.total_variation, ")")
end

function Base.show(io::IO, ::MIME"text/plain", pm::PointSignedMeasure)
    d = describe(pm; nlargest = 5)
    print(io,
          "PointSignedMeasure",
          "\n  ambient_dimension: ", d.ambient_dimension,
          "\n  axis_lengths: ", repr(d.axis_lengths),
          "\n  nterms: ", d.nterms,
          "\n  positive_terms: ", d.positive_terms,
          "\n  negative_terms: ", d.negative_terms,
          "\n  total_mass: ", d.total_mass,
          "\n  total_variation: ", d.total_variation,
          "\n  largest_terms: ", repr(d.largest_terms))
end

@doc """
    signed_measure_summary(x)
    rect_signed_barcode_summary(sb)
    point_signed_measure_summary(pm)
    signed_measure_decomposition_summary(result)

Owner-local summary aliases for SignedMeasures notebooks and REPL workflows.

Examples
--------
```julia
sb = rectangle_signed_barcode(M, pi; ...)
rect_signed_barcode_summary(sb)

pm = euler_signed_measure(M, pi, opts)
point_signed_measure_summary(pm)

out = mma_decomposition(M, pi; method=:all, ...)
signed_measure_decomposition_summary(out)
```
""" signed_measure_summary
@inline signed_measure_summary(x) = describe(x)
@inline rect_signed_barcode_summary(sb::RectSignedBarcode) = describe(sb)
@inline point_signed_measure_summary(pm::PointSignedMeasure) = describe(pm)

function _validation_result(::Type{S}, errors::Vector{String}; throw::Bool) where {S}
    if throw && !isempty(errors)
        Base.throw(ArgumentError(join(errors, "; ")))
    end
    return S(isempty(errors), errors)
end

"""
    check_rect(rect; axes=nothing, throw=false)

Validate a hand-built `Rect`.

If `axes` is supplied, this also checks that the rectangle corners lie on the
declared axis grids. With `throw=false`, returns a `RectValidationSummary`.
"""
function check_rect(rect::Rect{N}; axes=nothing, throw::Bool = false) where {N}
    errors = String[]
    if axes !== nothing
        length(axes) == N || push!(errors, "axes must have length $N")
        if length(axes) == N
            for k in 1:N
                _axes_sorted_unique(axes[k]) || push!(errors, "axis $k must be sorted with unique entries")
                rect.lo[k] in axes[k] || push!(errors, "rectangle lower corner is not on axis $k")
                rect.hi[k] in axes[k] || push!(errors, "rectangle upper corner is not on axis $k")
            end
        end
    end
    return _validation_result(RectValidationSummary, errors; throw = throw)
end

"""
    check_rect_signed_barcode(sb; throw=false)
    validate(sb::RectSignedBarcode; throw=true)

Validate a `RectSignedBarcode`.

Checks:
- each axis is sorted with unique entries,
- rectangle and weight counts agree,
- every rectangle is valid and lies on the declared axes.
"""
function check_rect_signed_barcode(sb::RectSignedBarcode; throw::Bool = false)
    errors = String[]
    for (k, ax) in pairs(sb.axes)
        _axes_sorted_unique(ax) || push!(errors, "axis $k must be sorted with unique entries")
    end
    length(sb.rects) == length(sb.weights) || push!(errors, "rectangles and weights must have the same length")
    @inbounds for (i, rect) in enumerate(sb.rects)
        rect_summary = check_rect(rect; axes = sb.axes, throw = false)
        append!(errors, ["rectangle $i: " * err for err in rect_summary.errors])
    end
    return _validation_result(RectSignedBarcodeValidationSummary, errors; throw = throw)
end

function validate(sb::RectSignedBarcode; throw::Bool = true)
    return check_rect_signed_barcode(sb; throw = throw)
end

"""
    check_point_signed_measure(pm; throw=false)
    validate(pm::PointSignedMeasure; throw=true)

Validate a `PointSignedMeasure`.

Checks:
- each axis is sorted with unique entries,
- support-index and weight counts agree,
- each support index lies within the declared axis bounds.
"""
function check_point_signed_measure(pm::PointSignedMeasure; throw::Bool = false)
    errors = String[]
    for (k, ax) in pairs(pm.axes)
        _axes_sorted_unique(ax) || push!(errors, "axis $k must be sorted with unique entries")
    end
    length(pm.inds) == length(pm.wts) || push!(errors, "support indices and weights must have the same length")
    @inbounds for (i, idx) in enumerate(pm.inds)
        for k in 1:ambient_dimension(pm)
            1 <= idx[k] <= length(pm.axes[k]) || push!(errors, "support index $i is out of bounds on axis $k")
        end
    end
    return _validation_result(PointSignedMeasureValidationSummary, errors; throw = throw)
end

function validate(pm::PointSignedMeasure; throw::Bool = true)
    return check_point_signed_measure(pm; throw = throw)
end

"""
    truncate_point_signed_measure(pm; max_terms=0, min_abs_weight=0)

Return a new `PointSignedMeasure` keeping only the "largest" terms.

- Drops all terms with `abs(w) < min_abs_weight`.
- If `max_terms > 0`, keeps only the top `max_terms` by `abs(w)`.

This mirrors `truncate_signed_barcode` for rectangles.
"""
function truncate_point_signed_measure(pm::PointSignedMeasure;
                                       max_terms::Int=0,
                                       min_abs_weight::Real=0)
    n = length(pm)
    if n == 0
        return pm
    end

    keep = Int[]
    sizehint!(keep, n)
    @inbounds for i in 1:n
        if abs(pm.wts[i]) >= min_abs_weight
            push!(keep, i)
        end
    end

    if max_terms > 0 && length(keep) > max_terms
        # sort kept indices by descending abs(weight)
        sort!(keep, by=i -> -abs(pm.wts[i]))
        keep = keep[1:max_terms]
        sort!(keep)  # restore increasing index order for stability
    end

    new_inds = Vector{NTuple{length(pm.axes),Int}}(undef, length(keep))
    new_wts  = Vector{eltype(pm.wts)}(undef, length(keep))
    @inbounds for (j,i) in enumerate(keep)
        new_inds[j] = pm.inds[i]
        new_wts[j]  = pm.wts[i]
    end

    return PointSignedMeasure(pm.axes, new_inds, new_wts)
end

# -----------------------------------------------------------------------------
# Internal: in-place Mobius inversion on product of chains via iterated diffs
# -----------------------------------------------------------------------------

# Apply 1D "difference" operator along each axis in-place.
# For 1D: w[i] <- f[i] - f[i-1] (with f[0]=0)
# For ND: iterated differences gives the product-poset Mobius inversion.
function _mobius_inversion_product_chains!(w::AbstractArray)
    N = ndims(w)
    if N == 1
        @inbounds for k in length(w):-1:2
            w[k] -= w[k-1]
        end
        return w
    end
    for d in 1:N
        # eachslice keeps dimension d and fixes all others, producing 1D views
        for sl in eachslice(w; dims=d)
            @inbounds for k in length(sl):-1:2
                sl[k] -= sl[k-1]
            end
        end
    end
    return w
end

# Inverse of Mobius inversion: prefix sums along each axis in-place.
function _prefix_sum_product_chains!(f::AbstractArray)
    N = ndims(f)
    if N == 1
        @inbounds for k in 2:length(f)
            f[k] += f[k-1]
        end
        return f
    end
    for d in 1:N
        for sl in eachslice(f; dims=d)
            @inbounds for k in 2:length(sl)
                sl[k] += sl[k-1]
            end
        end
    end
    return f
end

# -----------------------------------------------------------------------------
# Mixed-orientation Mobius inversion on products of chains
#
# Rectangle signed barcodes live on "interval posets":
# pairs (p,q) with p <= q, ordered by inclusion:
#   (lo,hi) <= (p,q)  iff  lo <= p and q <= hi.
#
# This is a product of chains in the "lo/p" coordinates and REVERSED chains in
# the "hi/q" coordinates.  Computationally, Mobius inversion is still just an
# iterated 1D difference along each axis; reversed axes use a forward difference.
# -----------------------------------------------------------------------------

# In-place Mobius inversion on a product of chains where some axes are reversed.
#
# For a non-reversed axis:
#   w[k] <- f[k] - f[k-1]  (with f[0] = 0)
# For a reversed axis:
#   w[k] <- f[k] - f[k+1]  (with f[end+1] = 0)
#
# This is the natural transform for interval-posets / rectangle measures.
function _mobius_inversion_product_chains_mixed!(w::AbstractArray,
                                                reverse_axis::NTuple{N,Bool}) where {N}
    ndims(w) == N || error("_mobius_inversion_product_chains_mixed!: reverse_axis must have length ndims(w)")
    for d in 1:N
        if reverse_axis[d]
            # Forward difference (uses the next entry).
            for sl in eachslice(w; dims=d)
                @inbounds for k in 1:(length(sl)-1)
                    sl[k] -= sl[k+1]
                end
            end
        else
            # Backward difference (uses the previous entry).
            for sl in eachslice(w; dims=d)
                @inbounds for k in length(sl):-1:2
                    sl[k] -= sl[k-1]
                end
            end
        end
    end
    return w
end

# Inverse transform to `_mobius_inversion_product_chains_mixed!`:
# in-place "mixed prefix sums".
#
# For a non-reversed axis:
#   f[k] <- f[k] + f[k-1]  (prefix sum)
# For a reversed axis:
#   f[k] <- f[k] + f[k+1]  (reverse prefix sum / suffix sum)
function _prefix_sum_product_chains_mixed!(f::AbstractArray,
                                          reverse_axis::NTuple{N,Bool}) where {N}
    ndims(f) == N || error("_prefix_sum_product_chains_mixed!: reverse_axis must have length ndims(f)")
    for d in 1:N
        if reverse_axis[d]
            for sl in eachslice(f; dims=d)
                @inbounds for k in (length(sl)-1):-1:1
                    sl[k] += sl[k+1]
                end
            end
        else
            for sl in eachslice(f; dims=d)
                @inbounds for k in 2:length(sl)
                    sl[k] += sl[k-1]
                end
            end
        end
    end
    return f
end

# Interval-poset Mobius inversion on a product of chains.
#
# For rectangle signed barcodes, each coordinate contributes a triangular
# interval domain `(p_k, q_k)` with `p_k <= q_k`. The 2N-dimensional tensor only
# stores comparable entries; values on `p_k > q_k` are structural zeros, not the
# full mixed-prefix extension. The correct inversion therefore acts on each
# `(p_k, q_k)` interval slice directly:
#
#   w[p, q] <- f[p, q] - f[p - 1, q] - f[p, q + 1] + f[p - 1, q + 1]
#
# applied one coordinate pair at a time.
function _mobius_inversion_interval_product!(w::AbstractArray, N::Int)
    ndims(w) == 2N || error("_mobius_inversion_interval_product!: expected a 2N-dimensional tensor")
    for d in 1:N
        perm = (d, N + d, ntuple(i -> i, d - 1)..., ntuple(i -> d + i, N - d)...,
                ntuple(i -> N + i, d - 1)..., ntuple(i -> N + d + i, N - d)...)
        wp = PermutedDimsArray(w, perm)
        dd = size(w, d)
        wr = reshape(wp, dd, dd, :)
        for s in axes(wr, 3)
            sl = view(wr, :, :, s)
            @inbounds for q in 1:dd
                for p in q:-1:1
                    v = sl[p, q]
                    if p > 1
                        v -= sl[p - 1, q]
                    end
                    if q < dd
                        v -= sl[p, q + 1]
                    end
                    if p > 1 && q < dd
                        v += sl[p - 1, q + 1]
                    end
                    sl[p, q] = v
                end
            end
        end
    end
    return w
end


"""
    point_signed_measure(surface, axes; drop_zeros=true)

Compute the Mobius inversion of `surface` on the product-of-chains grid given by `axes`.

- `surface` is an `N`-dimensional array of values on the grid.
- `axes` is an `N`-tuple of coordinate vectors, whose lengths match `size(surface)`.

Returns a `PointSignedMeasure` supported on grid points.

Performance notes:
- Uses in-place iterated differences (O(N * prod(size))) rather than O(2^N) inclusion-exclusion.
"""
function point_signed_measure(surface::AbstractArray{<:Integer,N},
                              axes::NTuple{N,AbstractVector};
                              drop_zeros::Bool=true) where {N}
    # Normalize axes to concrete vectors (no views/ranges stored in the measure)
    ax = ntuple(i -> collect(axes[i]), N)
    size(surface) == ntuple(i -> length(ax[i]), N) ||
        throw(ArgumentError("axes lengths must match surface size"))

    w = copy(surface)
    _mobius_inversion_product_chains!(w)

    inds = Vector{NTuple{N,Int}}()
    wts  = Vector{eltype(w)}()
    sizehint!(inds, length(w))
    sizehint!(wts,  length(w))

    @inbounds for I in CartesianIndices(w)
        val = w[I]
        if !drop_zeros || val != 0
            push!(inds, ntuple(k -> I[k], N))
            push!(wts, val)
        end
    end

    return PointSignedMeasure(ax, inds, wts)
end

function point_signed_measure(surface::AbstractArray{<:Integer,1},
                              axes::NTuple{1,AbstractVector};
                              drop_zeros::Bool=true)
    ax1 = collect(axes[1])
    length(surface) == length(ax1) ||
        throw(ArgumentError("axes lengths must match surface size"))

    inds = Vector{NTuple{1,Int}}()
    wts = Vector{eltype(surface)}()
    sizehint!(inds, length(surface))
    sizehint!(wts, length(surface))

    prev = zero(eltype(surface))
    @inbounds for i in eachindex(ax1)
        cur = surface[i]
        wt = cur - prev
        prev = cur
        if !drop_zeros || wt != 0
            push!(inds, (i,))
            push!(wts, wt)
        end
    end

    return PointSignedMeasure((ax1,), inds, wts)
end

"""
    surface_from_point_signed_measure(pm)

Reconstruct the grid function `surface` from its Mobius inversion stored in `pm`.

This is the inverse operation to `point_signed_measure` (up to dropped zeros).
"""
function surface_from_point_signed_measure(pm::PointSignedMeasure{N,T,W}) where {N,T,W}
    dims = ntuple(i -> length(pm.axes[i]), N)
    f = zeros(Int, dims...)
    @inbounds for i in 1:length(pm)
        I = pm.inds[i]
        f[I...] += pm.wts[i]
    end
    _prefix_sum_product_chains!(f)
    return f
end

"""
    point_signed_measure_kernel(pm1, pm2; sigma=1.0, kind=:gaussian)

A simple weighted point-cloud kernel for signed point measures.

- `kind=:gaussian` uses `exp(-||x-y||^2 / (2*sigma^2))`
- `kind=:laplacian` uses `exp(-||x-y|| / sigma)`

This is optional but convenient for ML-style pipelines on Euler signed measures.
"""
function point_signed_measure_kernel(pm1::PointSignedMeasure{N},
                                     pm2::PointSignedMeasure{N};
                                     sigma::Real=1.0,
                                     kind::Symbol=:gaussian) where {N}
    sigma > 0 || throw(ArgumentError("sigma must be positive"))
    s2 = float(sigma*sigma)
    acc = 0.0
    @inbounds for i in 1:length(pm1)
        I = pm1.inds[i]
        w1 = float(pm1.wts[i])
        x = ntuple(d -> pm1.axes[d][I[d]], N)
        for j in 1:length(pm2)
            J = pm2.inds[j]
            w2 = float(pm2.wts[j])
            y = ntuple(d -> pm2.axes[d][J[d]], N)

            # Euclidean distance in coordinate space
            dsq = 0.0
            for d in 1:N
                t = float(x[d] - y[d])
                dsq += t*t
            end

            k = if kind === :gaussian
                exp(-dsq/(2*s2))
            elseif kind === :laplacian
                exp(-sqrt(dsq)/float(sigma))
            else
                throw(ArgumentError("kind must be :gaussian or :laplacian"))
            end

            acc += w1*w2*k
        end
    end
    return acc
end

# Build the nonzero Mobius coefficients for a product of chains.
# For a chain, mu(i,j) is nonzero only when j in {i, i+1}, with values {+1, -1}.
# For a product of N chains, mu is the product, hence 2^N patterns.
function _mobius_eps(N::Int)
    m = 1 << N
    eps = Vector{NTuple{N,Int}}(undef, m)
    sgn = Vector{Int}(undef, m)
    for mask in 0:(m-1)
        eps[mask+1] = ntuple(i -> Int((mask >> (i-1)) & 1), N)
        sgn[mask+1] = isodd(Base.count_ones(mask)) ? -1 : 1
    end
    return eps, sgn
end

_shift_plus(t::NTuple{N,Int}, eps::NTuple{N,Int}) where {N} = ntuple(i -> t[i] + eps[i], N)
_shift_minus(t::NTuple{N,Int}, eps::NTuple{N,Int}) where {N} = ntuple(i -> t[i] - eps[i], N)


###############################################################################
# Performance helpers (memoization and grid restriction) for rectangle signed
# barcodes on large grids.
###############################################################################


"""
    coarsen_axis(axis; max_len, method=:uniform) -> Vector{Int}

Restrict an integer sampling axis to at most `max_len` points, preserving order
and endpoints. This is a lightweight performance helper for large-grid signed
barcode computations.
"""
function coarsen_axis(axis::AbstractVector{<:Integer}; max_len::Int, method::Symbol=:uniform)
    ax = sort(unique(Int.(axis)))
    L = length(ax)
    if max_len <= 0 || L <= max_len
        return ax
    end
    if method != :uniform
        error("coarsen_axis: unsupported method = $(method); use :uniform")
    end
    # Uniformly sample indices in [1, L], always including endpoints.
    idxs = unique(round.(Int, range(1, L; length=max_len)))
    idxs = sort(unique(vcat(1, idxs, L)))
    return ax[idxs]
end

"""
    coarsen_axes(axes; max_len, method=:uniform)

Apply `coarsen_axis` to each axis in an `N`-tuple `axes`.
"""
function coarsen_axes(axes::NTuple{N,<:AbstractVector{<:Integer}}; max_len::Int, method::Symbol=:uniform) where {N}
    return ntuple(k -> coarsen_axis(axes[k]; max_len=max_len, method=method), N)
end

"""
    restrict_axes_to_encoding(axes, pi; keep_endpoints=true)

Grid restriction heuristic for `ZnEncodingMap`.

Given user-provided axes and an encoding map `pi`, keep only those axis values
that appear in the encoding-derived axes `axes_from_encoding(pi)` and lie between
the user axis endpoints. If `keep_endpoints=true`, also keep the user endpoints.

For typical `ZnEncodingMap`s, the rank invariant is constant on the axis-aligned
cells determined by the encoding, so this restriction often removes redundant grid
points without changing nonzero signed weights.
"""
function restrict_axes_to_encoding(axes::NTuple{N,<:AbstractVector{<:Integer}}, pi::ZnEncodingMap; keep_endpoints::Bool=true) where {N}
    enc = axes_from_encoding(pi)
    return ntuple(k -> _restrict_axis_to_encoding(axes[k], enc[k]; keep_endpoints=keep_endpoints), N)
end

restrict_axes_to_encoding(axes::NTuple{N,<:AbstractVector{<:Integer}}, pi::CompiledEncoding{<:ZnEncodingMap};
                           keep_endpoints::Bool=true) where {N} =
    restrict_axes_to_encoding(axes, pi.pi; keep_endpoints=keep_endpoints)

function _restrict_axis_to_encoding(axis::AbstractVector{<:Integer}, enc_axis::Vector{Int}; keep_endpoints::Bool=true)
    ax = sort(unique(Int.(axis)))
    if isempty(ax)
        return copy(enc_axis)
    end
    lo, hi = first(ax), last(ax)
    vals = Int[]
    for v in enc_axis
        if lo <= v <= hi
            push!(vals, v)
        end
    end
    if keep_endpoints
        push!(vals, lo)
        push!(vals, hi)
    end
    return sort(unique(vals))
end


# Apply encoding-axis restriction policy to an axis tuple (Integer grids).
# This is the implementation used by rectangle_signed_barcode(...; axes_policy=:encoding).
function _axes_policy_encoding(axes::NTuple{N,Vector{Int}},
                               enc_axes::NTuple{N,Vector{Int}};
                               keep_endpoints::Bool=true) where {N}
    return ntuple(k -> _restrict_axis_to_encoding(axes[k], enc_axes[k]; keep_endpoints=keep_endpoints), N)
end

# Internal: normalize an axis tuple to an `NTuple{N,Vector{Int}}` of sorted unique axes.
function _normalize_axes(axes::NTuple{N,<:AbstractVector{<:Integer}}) where {N}
    return ntuple(k -> sort(unique(Int.(axes[k]))), N)
end


# -----------------------------------------------------------------------------
# Real-valued axes support (needed for PL/box encodings and general R^n grids)
# -----------------------------------------------------------------------------

function _normalize_axes_real(axes::NTuple{N,AbstractVector}) where {N}
    ax = ntuple(i -> sort(collect(axes[i])), N)
    for i in 1:N
        length(ax[i]) > 0 || throw(ArgumentError("axis cannot be empty"))
    end
    return ax
end

# Intersection of two sorted vectors (generic Real). Used by :encoding axis policy.
function _axis_intersection_real(a::AbstractVector{<:Real},
                                 b::AbstractVector{<:Real})
    i = 1
    j = 1
    out = eltype(a)[]
    sizehint!(out, min(length(a), length(b)))
    while i <= length(a) && j <= length(b)
        if a[i] == b[j]
            push!(out, a[i]); i += 1; j += 1
        elseif a[i] < b[j]
            i += 1
        else
            j += 1
        end
    end
    return out
end

# Restrict a proposed axis to one supported by the encoding axis (generic Real).
# Keeps only values that appear in the encoding axis.
function _restrict_axis_to_encoding(axis::AbstractVector{<:Real},
                                   enc_axis::AbstractVector{<:Real})
    axis2 = sort(collect(axis))
    enc2  = sort(collect(enc_axis))
    out = _axis_intersection_real(axis2, enc2)
    length(out) > 0 || throw(ArgumentError("axis restriction is empty"))
    return out
end

# Generic fallback: if an encoding supports axes_from_encoding, we can restrict.
function restrict_axes_to_encoding(axes::NTuple{N,AbstractVector}, pi) where {N}
    enc_axes = axes_from_encoding(pi)  # requires a method; may be provided by Zn or PLBackend
    length(enc_axes) == N || throw(ArgumentError("axes dimension mismatch"))
    return ntuple(i -> _restrict_axis_to_encoding(axes[i], enc_axes[i]), N)
end

# Coarsen any sorted axis (Integer or Real) by downsampling every other point.
function coarsen_axis(axis::AbstractVector{<:Real})
    length(axis) <= 2 && return collect(axis)
    out = eltype(axis)[]
    sizehint!(out, (length(axis)+1) >>> 1)
    @inbounds for i in 1:2:length(axis)
        push!(out, axis[i])
    end
    # Ensure last point included (preserve extent)
    if out[end] != axis[end]
        push!(out, axis[end])
    end
    return out
end


# =============================================================================
# Euler characteristic surface and Euler signed-measure pipeline
# =============================================================================

"""
    euler_characteristic_surface(obj, pi, opts::InvariantOptions)

Compute the Euler characteristic surface of `obj` on a finite grid induced by
`pi` and the axis policy encoded in `opts`.

- For a `PModule`, this agrees with the restricted Hilbert surface.
- For a `ModuleCochainComplex`, this is the alternating sum of chain-group
  dimensions on each encoded region:

      chi(x) = sum_t (-1)^t dim(C^t_x)

  where `x` is first located in the finite encoding via `locate(pi, x)`.

The grid is chosen from `opts.axes`, `opts.axes_policy`, and
`opts.max_axis_len` in the same way as the signed-measure helpers in this
module.
"""
function _euler_characteristic_surface_on_axes(chi_dims::AbstractVector{<:Integer},
                                               pi::PLikeEncodingMap,
                                               ax::NTuple{N,AbstractVector},
                                               use_threads::Bool) where {N}
    surf = zeros(Int, length.(ax)...)
    n = dimension(pi)
    if use_threads
        xs = [zeros(Float64, n) for _ in 1:Threads.nthreads()]
        Threads.@threads for I in CartesianIndices(size(surf))
            x = xs[Threads.threadid()]
            for i in 1:n
                x[i] = float(ax[i][I[i]])
            end
            u = locate(pi, x)
            surf[I] = (u == 0) ? 0 : chi_dims[u]
        end
    else
        x = zeros(Float64, n)
        for I in CartesianIndices(size(surf))
            for i in 1:n
                x[i] = float(ax[i][I[i]])
            end
            u = locate(pi, x)
            surf[I] = (u == 0) ? 0 : chi_dims[u]
        end
    end
    return surf
end

function _euler_characteristic_surface_on_axes(chi_dims::AbstractVector{<:Integer},
                                               pi::GridEncodingMap{1},
                                               ax::NTuple{1,AbstractVector},
                                               use_threads::Bool)
    axis = ax[1]
    coords = pi.coords[1]
    surf = Vector{Int}(undef, length(axis))
    isempty(coords) && return fill!(surf, 0)

    if pi.orientation[1] == 1
        rid = 0
        @inbounds for i in eachindex(axis)
            xi = float(axis[i])
            while rid < length(coords) && coords[rid + 1] <= xi
                rid += 1
            end
            surf[i] = rid == 0 ? 0 : chi_dims[rid]
        end
    else
        rid = 0
        @inbounds for i in length(axis):-1:1
            xi = -float(axis[i])
            while rid < length(coords) && coords[rid + 1] <= xi
                rid += 1
            end
            surf[i] = rid == 0 ? 0 : chi_dims[rid]
        end
    end
    return surf
end

function _euler_characteristic_surface_on_axes(chi_dims::AbstractVector{<:Integer},
                                               pi::CompiledEncoding{PiType},
                                               ax::NTuple{1,AbstractVector},
                                               use_threads::Bool) where {PiType<:GridEncodingMap{1}}
    return _euler_characteristic_surface_on_axes(chi_dims, pi.pi, ax, use_threads)
end

function _cached_euler_characteristic_surface(obj,
                                              pi::PLikeEncodingMap,
                                              opts::InvariantOptions,
                                              session_cache::Union{Nothing,SessionCache})
    chi_dims = _euler_dims(obj)
    ax = _choose_axes_real(pi;
        axes=opts.axes, axes_policy=opts.axes_policy, max_axis_len=opts.max_axis_len)
    use_threads = opts.threads === nothing ? (Threads.nthreads() > 1) : opts.threads
    session_cache === nothing &&
        return _euler_characteristic_surface_on_axes(chi_dims, pi, ax, use_threads), ax
    enc_cache = _workflow_encoding_cache(session_cache)
    key = _signed_measures_euler_surface_key(obj, pi, ax)
    cached = _signed_measures_cache_get(enc_cache, key)
    if cached isa AbstractArray{<:Integer}
        return cached, ax
    end
    surf = _euler_characteristic_surface_on_axes(chi_dims, pi, ax, use_threads)
    _signed_measures_cache_set!(enc_cache, key, surf)
    return surf, ax
end

function euler_characteristic_surface(obj, pi::PLikeEncodingMap, opts::InvariantOptions;
                                      cache=nothing)
    session_cache = _signed_measures_session_cache(cache)
    surf, _ = _cached_euler_characteristic_surface(obj, pi, opts, session_cache)
    return session_cache === nothing ? surf : copy(surf)
end

function _choose_axes_real(pi::PLikeEncodingMap; axes=nothing, axes_policy::Symbol=:encoding, max_axis_len::Int=64)
    if axes === nothing
        if axes_policy === :as_given
            throw(ArgumentError("axes_policy=:as_given requires explicit axes"))
        elseif axes_policy === :encoding
            return _normalize_axes_real(axes_from_encoding(pi))
        elseif axes_policy === :coarsen
            return coarsen_axes(_normalize_axes_real(axes_from_encoding(pi)), max_axis_len)
        else
            error("unknown axes_policy=$axes_policy")
        end
    else
        ax = _normalize_axes_real(axes)
        if axes_policy === :as_given
            return ax
        elseif axes_policy === :encoding
            return _normalize_axes_real(restrict_axes_to_encoding(ax, pi))
        elseif axes_policy === :coarsen
            return coarsen_axes(ax, max_axis_len)
        else
            error("unknown axes_policy=$axes_policy")
        end
    end
end

const _SIGNED_MEASURES_RQ_CACHE_TAG = :signed_measures_rank_query_cache
const _SIGNED_MEASURES_EULER_SURFACE_TAG = :signed_measures_euler_surface
const _SIGNED_MEASURES_RECT_BARCODE_TAG = :signed_measures_rectangle_barcode
const _SIGNED_MEASURES_REGION_GRID_TAG = :signed_measures_region_grid
const _SIGNED_MEASURES_REGION_BLOCKS_2D_TAG = :signed_measures_region_blocks_2d
const _SIGNED_MEASURES_PACKED_TENSOR_2D_TAG = :signed_measures_packed_tensor_2d
const _SIGNED_MEASURES_RQ_RECT_CACHE_LOCK = ReentrantLock()
const _SIGNED_MEASURES_RQ_RECT_CACHES = WeakKeyDict{Any,Dict{Any,Any}}()
const _SIGNED_MEASURES_RQ_TENSOR_CACHE_LOCK = ReentrantLock()
const _SIGNED_MEASURES_RQ_TENSOR_CACHES = WeakKeyDict{Any,Dict{Any,Any}}()
const _RECT_BARCODE_EMBEDDING_CACHE_LOCK = ReentrantLock()
const _RECT_BARCODE_EMBEDDING_CACHE = IdDict{Any,Any}()
const _RECT_BARCODE_EMBEDDING_CACHE_MAX = 256

@inline function _signed_measures_session_cache(cache)
    cache === nothing && return nothing
    return _resolve_workflow_session_cache(cache)
end

@inline function _signed_measures_cache_get(cache::Union{Nothing,EncodingCache}, key)
    cache === nothing && return nothing
    Base.lock(cache.lock)
    try
        entry = get(cache.geometry, key, nothing)
        return entry === nothing ? nothing : entry.value
    finally
        Base.unlock(cache.lock)
    end
end

@inline function _signed_measures_cache_set!(cache::Union{Nothing,EncodingCache}, key, value)
    cache === nothing && return value
    Base.lock(cache.lock)
    try
        cache.geometry[key] = GeometryCachePayload(value)
    finally
        Base.unlock(cache.lock)
    end
    return value
end

@inline _signed_measures_axes_key(ax::NTuple{N,AbstractVector}) where {N} =
    ntuple(i -> Tuple(ax[i]), N)

@inline function _signed_measures_rank_query_cache(M::PModule,
                                                   pi::ZnEncodingMap,
                                                   rq_cache::Union{Nothing,RankQueryCache},
                                                   session_cache::Union{Nothing,SessionCache})
    if rq_cache !== nothing
        rq_cache.pi === pi || error("rectangle_signed_barcode: rq_cache.pi must match pi")
        rq_cache.n == pi.n || error("rectangle_signed_barcode: rq_cache has wrong dimension")
        return rq_cache
    end
    enc_cache = _workflow_encoding_cache(session_cache)
    key = (_SIGNED_MEASURES_RQ_CACHE_TAG, UInt(objectid(M)), UInt(objectid(pi)))
    cached = _signed_measures_cache_get(enc_cache, key)
    if cached isa RankQueryCache
        cached.pi === pi || error("rectangle_signed_barcode: cached rank query cache has wrong encoding")
        return cached
    end
    return _signed_measures_cache_set!(enc_cache, key, RankQueryCache(pi))
end

@inline _signed_measures_euler_surface_key(obj, pi, ax) =
    (_SIGNED_MEASURES_EULER_SURFACE_TAG, UInt(objectid(obj)), UInt(objectid(pi)), _signed_measures_axes_key(ax))

@inline _signed_measures_region_grid_key(pi, axes, strict::Bool) =
    (_SIGNED_MEASURES_REGION_GRID_TAG, UInt(objectid(pi)), _signed_measures_axes_key(axes), strict)

@inline _signed_measures_region_blocks_2d_key(pi, axes, strict::Bool) =
    (_SIGNED_MEASURES_REGION_BLOCKS_2D_TAG, UInt(objectid(pi)), _signed_measures_axes_key(axes), strict)

@inline _signed_measures_packed_tensor_2d_key(M, pi, axes, strict::Bool) =
    (_SIGNED_MEASURES_PACKED_TENSOR_2D_TAG, UInt(objectid(M)), UInt(objectid(pi)), _signed_measures_axes_key(axes), strict)

@inline function _signed_measures_rectangle_barcode_key(M::PModule,
                                                        pi::ZnEncodingMap,
                                                        axes::NTuple{N,Vector{Int}},
                                                        meth::Symbol;
                                                        drop_zeros::Bool,
                                                        tol::Int,
                                                        max_span,
                                                        strict::Bool) where {N}
    span_key = _normalize_rectangle_span(max_span, axes)
    return (
        _SIGNED_MEASURES_RECT_BARCODE_TAG,
        UInt(objectid(M)),
        UInt(objectid(pi)),
        _signed_measures_axes_key(axes),
        meth,
        drop_zeros,
        tol,
        span_key,
        strict,
    )
end

@inline function _signed_measures_rectangle_barcode_key(M::PModule,
                                                        pi::GridEncodingMap,
                                                        axes::NTuple{N,Vector{Int}},
                                                        meth::Symbol;
                                                        drop_zeros::Bool,
                                                        tol::Int,
                                                        max_span,
                                                        strict::Bool) where {N}
    span_key = _normalize_rectangle_span(max_span, axes)
    return (
        _SIGNED_MEASURES_RECT_BARCODE_TAG,
        UInt(objectid(M)),
        UInt(objectid(pi)),
        _signed_measures_axes_key(axes),
        meth,
        drop_zeros,
        tol,
        span_key,
        strict,
    )
end

@inline function _rectangle_signed_barcode_embedding_cached(sb::RectSignedBarcode{N}) where {N}
    Base.lock(_RECT_BARCODE_EMBEDDING_CACHE_LOCK)
    try
        cached = get(_RECT_BARCODE_EMBEDDING_CACHE, sb, nothing)
        if cached isa Tuple{Matrix{Float64},Vector{Float64}}
            return cached
        end
    finally
        Base.unlock(_RECT_BARCODE_EMBEDDING_CACHE_LOCK)
    end

    packed = _rectangle_signed_barcode_embedding(sb)
    Base.lock(_RECT_BARCODE_EMBEDDING_CACHE_LOCK)
    try
        cached = get(_RECT_BARCODE_EMBEDDING_CACHE, sb, nothing)
        if cached isa Tuple{Matrix{Float64},Vector{Float64}}
            return cached
        end
        if length(_RECT_BARCODE_EMBEDDING_CACHE) >= _RECT_BARCODE_EMBEDDING_CACHE_MAX
            empty!(_RECT_BARCODE_EMBEDDING_CACHE)
        end
        _RECT_BARCODE_EMBEDDING_CACHE[sb] = packed
        return packed
    finally
        Base.unlock(_RECT_BARCODE_EMBEDDING_CACHE_LOCK)
    end
end

@inline function _signed_measures_rq_rectangle_get(rq_cache::Union{Nothing,RankQueryCache}, key)
    rq_cache === nothing && return nothing
    Base.lock(_SIGNED_MEASURES_RQ_RECT_CACHE_LOCK)
    try
        store = get(_SIGNED_MEASURES_RQ_RECT_CACHES, rq_cache, nothing)
        store === nothing && return nothing
        return get(store, key, nothing)
    finally
        Base.unlock(_SIGNED_MEASURES_RQ_RECT_CACHE_LOCK)
    end
end

@inline function _signed_measures_rq_rectangle_set!(rq_cache::Union{Nothing,RankQueryCache}, key, value)
    rq_cache === nothing && return value
    Base.lock(_SIGNED_MEASURES_RQ_RECT_CACHE_LOCK)
    try
        store = get!(() -> Dict{Any,Any}(), _SIGNED_MEASURES_RQ_RECT_CACHES, rq_cache)
        store[key] = value
    finally
        Base.unlock(_SIGNED_MEASURES_RQ_RECT_CACHE_LOCK)
    end
    return value
end

@inline function _signed_measures_rq_tensor_get(rq_cache::Union{Nothing,RankQueryCache}, key)
    rq_cache === nothing && return nothing
    Base.lock(_SIGNED_MEASURES_RQ_TENSOR_CACHE_LOCK)
    try
        store = get(_SIGNED_MEASURES_RQ_TENSOR_CACHES, rq_cache, nothing)
        store === nothing && return nothing
        return get(store, key, nothing)
    finally
        Base.unlock(_SIGNED_MEASURES_RQ_TENSOR_CACHE_LOCK)
    end
end

@inline function _signed_measures_rq_tensor_set!(rq_cache::Union{Nothing,RankQueryCache}, key, value)
    rq_cache === nothing && return value
    Base.lock(_SIGNED_MEASURES_RQ_TENSOR_CACHE_LOCK)
    try
        store = get!(() -> Dict{Any,Any}(), _SIGNED_MEASURES_RQ_TENSOR_CACHES, rq_cache)
        store[key] = value
    finally
        Base.unlock(_SIGNED_MEASURES_RQ_TENSOR_CACHE_LOCK)
    end
    return value
end

"""
    euler_surface(obj, pi, opts::InvariantOptions)
    euler_surface(obj, pi; opts::InvariantOptions=InvariantOptions(), kwargs...)

Canonical short alias for `euler_characteristic_surface`.
"""
euler_surface(obj, pi, opts::InvariantOptions; kwargs...) =
    euler_characteristic_surface(obj, pi, opts; kwargs...)
euler_surface(obj, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    euler_surface(obj, pi, opts; kwargs...)

# Internal: Euler dims on encoding elements.
_euler_dims(M::PModule) = M.dims
_euler_dims(dims::AbstractVector{<:Integer}) = [Int(d) for d in dims]

function _euler_dims(C::ModuleCochainComplex)
    n = length(C.terms[1].dims)
    chi = zeros(Int, n)
    for (i, M) in enumerate(C.terms)
        t = C.tmin + (i - 1)
        sgn = isodd(t) ? -1 : 1
        @inbounds for u in 1:n
            chi[u] += sgn * M.dims[u]
        end
    end
    return chi
end

"""
    euler_signed_measure(obj, pi, opts::InvariantOptions;
                         drop_zeros=true, max_terms=0, min_abs_weight=0, cache=nothing)
    euler_signed_measure(obj, pi; opts::InvariantOptions=InvariantOptions(), kwargs...)

Compute the Euler signed measure of `obj` on the grid determined by `opts`.

This evaluates the Euler characteristic surface and then applies
`point_signed_measure` on that grid.

Pass `cache=SessionCache()` to reuse the Euler surface across repeated
Euler/MMA workflows on the same `(obj, pi, opts)` inputs.
"""
function _euler_signed_measure_from_surface(surf::AbstractArray{<:Integer},
                                            ax::NTuple{N,AbstractVector};
                                            drop_zeros::Bool=true,
                                            max_terms::Int=0,
                                            min_abs_weight::Real=0) where {N}
    pm = point_signed_measure(surf, ax; drop_zeros=drop_zeros)
    if max_terms > 0 || min_abs_weight > 0
        pm = truncate_point_signed_measure(pm;
            max_terms=max_terms, min_abs_weight=min_abs_weight)
    end
    return pm
end

function euler_signed_measure(obj, pi, opts::InvariantOptions;
    drop_zeros::Bool=true,
    max_terms::Int=0,
    min_abs_weight::Real=0,
    cache=nothing)

    session_cache = _signed_measures_session_cache(cache)
    surf, ax = _cached_euler_characteristic_surface(obj, pi, opts, session_cache)
    return _euler_signed_measure_from_surface(surf, ax;
        drop_zeros=drop_zeros,
        max_terms=max_terms,
        min_abs_weight=min_abs_weight,
    )
end

euler_signed_measure(obj, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    euler_signed_measure(obj, pi, opts; kwargs...)

"""
    euler_distance(A, B; ord=1)
    euler_distance(obj1, obj2, pi, opts::InvariantOptions; ord=1, cache=nothing)
    euler_distance(obj1, obj2, pi; opts::InvariantOptions=InvariantOptions(), kwargs...)

Compute the `L^p` distance between Euler characteristic surfaces on the same
grid. The object-level method evaluates both surfaces first and then applies the
array-level distance. Pass `cache=SessionCache()` to reuse those surfaces across
repeated comparisons.
"""
function euler_distance(A::AbstractArray, B::AbstractArray; ord::Real=1)
    size(A) == size(B) || throw(ArgumentError("surface sizes must match"))
    if ord == Inf
        m = 0.0
        @inbounds for I in eachindex(A)
            v = abs(float(A[I] - B[I]))
            if v > m
                m = v
            end
        end
        return m
    else
        ord >= 1 || throw(ArgumentError("ord must be >= 1 or Inf"))
        s = 0.0
        @inbounds for I in eachindex(A)
            s += abs(float(A[I] - B[I]))^ord
        end
        return s^(1/ord)
    end
end

function euler_distance(obj1, obj2, pi, opts::InvariantOptions; ord::Real=1, cache=nothing)
    session_cache = _signed_measures_session_cache(cache)
    A, _ = _cached_euler_characteristic_surface(obj1, pi, opts, session_cache)
    B, _ = _cached_euler_characteristic_surface(obj2, pi, opts, session_cache)
    return euler_distance(A, B; ord=ord)
end

euler_distance(obj1, obj2, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    euler_distance(obj1, obj2, pi, opts; kwargs...)



"""
    _rectangle_signed_barcode_local(rank_idx, axes; drop_zeros=true, tol=0, max_span=nothing)

Compute the rectangle signed barcode by N-dimensional Mobius inversion of a rank invariant
sampled on a finite grid.

Arguments:
- `axes` is a tuple of sorted coordinate lists `(a1, ..., aN)`. The grid points are the
  cartesian product of these axes.
- `rank_idx(p, q)` must accept index tuples `p::NTuple{N,Int}`, `q::NTuple{N,Int}` with
  `p <= q` coordinatewise (indices into `axes`), and return the rank invariant value at the
  corresponding grid points.

Keyword arguments:
- `drop_zeros`: if true, return only rectangles with `|w| > tol`; otherwise include all
  enumerated rectangles and store `0` when `|w| <= tol`.
- `tol`: integer threshold for treating small weights as zero.
- `max_span`: optional grid restriction heuristic. If `max_span` is an `Int` or an `N`-tuple,
  only rectangles with coordinate spans `axes[k][q[k]] - axes[k][p[k]] <= max_span[k]` are
  enumerated. This can dramatically reduce runtime on very large grids.

Returns:
A `RectSignedBarcode{N,Int}`.

Notes:
This is the higher-dimensional analog of recovering a 1-parameter barcode from the rank function.
For `N > 1` the output is a signed measure on rectangles (not, in general, a literal decomposition
of the module as a direct sum of rectangle modules).
"""
function _rectangle_signed_barcode_local(rank_idx::Function, axes::NTuple{N,Vector{Int}};
    drop_zeros::Bool=true,
    tol::Int=0,
    max_span=nothing,
    threads::Bool=false) where {N}

    @assert tol >= 0 "rectangle_signed_barcode: tol must be nonnegative"

    rects = Rect{N}[]
    weights = Int[]

    dims = ntuple(k -> length(axes[k]), N)
    CI = CartesianIndices(dims)

    # Precompute epsilon-shifts and signs for Mobius inversion.
    eps_list, eps_sign = _mobius_eps(N)

    # Normalize max_span to an NTuple{N,Int} or `nothing`.
    span = nothing
    if max_span !== nothing
        if max_span isa Integer
            s = Int(max_span)
            @assert s >= 0 "rectangle_signed_barcode: max_span must be nonnegative"
            span = ntuple(_ -> s, N)
        else
            @assert length(max_span) == N "rectangle_signed_barcode: max_span must be an Int or an N-tuple"
            span = ntuple(k -> Int(max_span[k]), N)
            @assert all(span[k] >= 0 for k in 1:N) "rectangle_signed_barcode: max_span must be nonnegative"
        end
    end

    if threads && Threads.nthreads() > 1
        total = length(CI)
        nchunks = min(total, Threads.nthreads())
        chunk_rects = [Rect{N}[] for _ in 1:nchunks]
        chunk_weights = [Int[] for _ in 1:nchunks]
        chunk_size = cld(total, nchunks)

        Threads.@threads for c in 1:nchunks
            start_idx = (c - 1) * chunk_size + 1
            end_idx = min(c * chunk_size, total)
            start_idx > end_idx && continue

            rects_local = chunk_rects[c]
            weights_local = chunk_weights[c]

            for idx in start_idx:end_idx
                pCI = CI[idx]
                p = pCI.I

                # Enumerate q indices with p <= q coordinatewise, optionally restricting spans.
                ranges = ntuple(k -> begin
                    start = p[k]
                    stop = dims[k]
                    if span !== nothing
                        pval = @inbounds axes[k][start]
                        qmax = pval + span[k]
                        qstop = searchsortedlast(axes[k], qmax)
                        stop = min(stop, qstop)
                    end
                    start:stop
                end, N)

                for qCI in CartesianIndices(ranges)
                    q = qCI.I

                    w = 0
                    @inbounds for (i, eps_p) in enumerate(eps_list), (j, eps_q) in enumerate(eps_list)
                        p2 = _shift_minus(p, eps_p)
                        q2 = _shift_plus(q, eps_q)

                        # Ensure shifted indices still satisfy 1 <= p2 <= q2 <= dims.
                        ok = true
                        for k in 1:N
                            if p2[k] < 1 || q2[k] > dims[k] || p2[k] > q2[k]
                                ok = false
                                break
                            end
                        end
                        if ok
                            w += eps_sign[i] * eps_sign[j] * rank_idx(p2, q2)
                        end
                    end

                    if abs(w) > tol
                        # NOTE (parsing pitfall):
                        # Avoid writing `ntuple(k -> @inbounds axes[k][p[k]], N)` because `@inbounds`
                        # can accidentally capture the trailing `, N`, leading to a one-argument call
                        # `ntuple(closure)` and a MethodError at runtime.
                        lo = ntuple(Val(N)) do k
                            @inbounds axes[k][p[k]]
                        end
                        hi = ntuple(Val(N)) do k
                            @inbounds axes[k][q[k]]
                        end
                        push!(rects_local, Rect{N}(lo, hi))
                        push!(weights_local, w)
                    elseif !drop_zeros
                       lo = ntuple(Val(N)) do k
                            @inbounds axes[k][p[k]]
                        end
                        hi = ntuple(Val(N)) do k
                            @inbounds axes[k][q[k]]
                        end
                        push!(rects_local, Rect{N}(lo, hi))
                        push!(weights_local, 0)
                    end
                end
            end
        end

        for c in 1:nchunks
            append!(rects, chunk_rects[c])
            append!(weights, chunk_weights[c])
        end
    else
        for pCI in CI
            p = pCI.I

            # Enumerate q indices with p <= q coordinatewise, optionally restricting spans.
            ranges = ntuple(k -> begin
                start = p[k]
                stop = dims[k]
                if span !== nothing
                    pval = @inbounds axes[k][start]
                    qmax = pval + span[k]
                    qstop = searchsortedlast(axes[k], qmax)
                    stop = min(stop, qstop)
                end
                start:stop
            end, N)

            for qCI in CartesianIndices(ranges)
                q = qCI.I

                w = 0
                @inbounds for (i, eps_p) in enumerate(eps_list), (j, eps_q) in enumerate(eps_list)
                    p2 = _shift_minus(p, eps_p)
                    q2 = _shift_plus(q, eps_q)

                    # Ensure shifted indices still satisfy 1 <= p2 <= q2 <= dims.
                    ok = true
                    for k in 1:N
                        if p2[k] < 1 || q2[k] > dims[k] || p2[k] > q2[k]
                            ok = false
                            break
                        end
                    end
                    if ok
                        w += eps_sign[i] * eps_sign[j] * rank_idx(p2, q2)
                    end
                end

                if abs(w) > tol
                    # NOTE (parsing pitfall):
                    # Avoid writing `ntuple(k -> @inbounds axes[k][p[k]], N)` because `@inbounds`
                    # can accidentally capture the trailing `, N`, leading to a one-argument call
                    # `ntuple(closure)` and a MethodError at runtime.
                    lo = ntuple(Val(N)) do k
                        @inbounds axes[k][p[k]]
                    end
                    hi = ntuple(Val(N)) do k
                        @inbounds axes[k][q[k]]
                    end
                    push!(rects, Rect{N}(lo, hi))
                    push!(weights, w)
                elseif !drop_zeros
                   lo = ntuple(Val(N)) do k
                        @inbounds axes[k][p[k]]
                    end
                    hi = ntuple(Val(N)) do k
                        @inbounds axes[k][q[k]]
                    end
                    push!(rects, Rect{N}(lo, hi))
                    push!(weights, 0)
                end
            end
        end
    end

    return RectSignedBarcode{N,Int}(axes, rects, weights)
end

# -----------------------------------------------------------------------------
# Fast rectangle signed barcode computation (bulk rank + Mobius inversion)
# -----------------------------------------------------------------------------

const _RECTANGLE_BULK_MIN_COMPARABLE_PAIRS = Ref(64)
const _RECTANGLE_BULK_MAX_AXIS_ASPECT = Ref(4.0)
const _USE_PACKED_RECTANGLE_BULK_2D = Ref(true)
const _USE_PACKED_TENSOR_CACHE_2D = Ref(true)
const _RECTANGLE_BULK_AUTO_MAX_COMPRESSED_COMPARABLE_PAIRS = Ref(4_096)

struct _RectanglePackedTensor2D
    axes_comp::NTuple{2,Vector{Int}}
    weights::Matrix{Int}
end

function _estimate_rectangle_rank_array_elems(axes::NTuple{N,Vector{Int}}) where {N}
    # We allocate an Int array of size prod(length.(axes))^2.
    # Use overflow-safe arithmetic to avoid nonsense sizes on large grids.
    m = 1
    for k in 1:N
        mk, of = Base.mul_with_overflow(m, length(axes[k]))
        if of
            return typemax(Int)
        end
        m = mk
    end
    mm, of = Base.mul_with_overflow(m, m)
    return of ? typemax(Int) : mm
end

function _estimate_rectangle_comparable_pairs(axes::NTuple{N,Vector{Int}}) where {N}
    total = 1
    for k in 1:N
        d = length(axes[k])
        term_big = div(big(d) * big(d + 1), 2)
        term_big > typemax(Int) && return typemax(Int)
        term = Int(term_big)
        total2, of = Base.mul_with_overflow(total, term)
        of && return typemax(Int)
        total = total2
    end
    return total
end

@inline function _rectangle_axis_aspect(axes::NTuple{N,Vector{Int}}) where {N}
    lengths = ntuple(i -> length(axes[i]), N)
    minlen = minimum(lengths)
    minlen == 0 && return Inf
    return maximum(lengths) / minlen
end

function _choose_rectangle_signed_barcode_method(method::Symbol,
                                                 axes::NTuple{N,Vector{Int}};
                                                 max_span,
                                                 bulk_max_elems::Int) where {N}
    if method == :auto
        if max_span === nothing &&
           _estimate_rectangle_rank_array_elems(axes) <= bulk_max_elems &&
           _estimate_rectangle_comparable_pairs(axes) >= _RECTANGLE_BULK_MIN_COMPARABLE_PAIRS[] &&
           _rectangle_axis_aspect(axes) <= _RECTANGLE_BULK_MAX_AXIS_ASPECT[]
            return :bulk
        end
        return :local
    elseif method == :local
        return :local
    elseif method == :bulk
        ne = _estimate_rectangle_rank_array_elems(axes)
        if ne > bulk_max_elems
            throw(ArgumentError(
                "method=:bulk would allocate an array with $ne Int entries " *
                "(> bulk_max_elems=$bulk_max_elems). " *
                "Increase bulk_max_elems, coarsen the axes, or use method=:local."
            ))
        end
        return :bulk
    else
        throw(ArgumentError("Unknown method=$(method). Use :auto, :bulk, or :local."))
    end
end

function _normalize_rectangle_span(max_span, axes::NTuple{N,Vector{Int}}) where {N}
    span = nothing
    if max_span !== nothing
        if max_span isa Integer
            s = Int(max_span)
            @assert s >= 0 "rectangle_signed_barcode: max_span must be nonnegative"
            span = ntuple(_ -> s, N)
        else
            @assert length(max_span) == N "rectangle_signed_barcode: max_span must be an Int or an N-tuple"
            span = ntuple(k -> Int(max_span[k]), N)
            @assert all(span[k] >= 0 for k in 1:N) "rectangle_signed_barcode: max_span must be nonnegative"
        end
    end
    return span
end

function _extract_rectangles_from_mobius_tensor(
    w::AbstractArray{<:Integer},
    axes::NTuple{N,Vector{Int}};
    drop_zeros::Bool=true,
    tol::Int=0,
    max_span=nothing,
    threads::Bool=false,
) where {N}
    dims = ntuple(k -> length(axes[k]), N)
    span = _normalize_rectangle_span(max_span, axes)
    CI = CartesianIndices(dims)
    @inline function _q_ranges(p)
        return ntuple(k -> begin
            start = p[k]
            stop = dims[k]
            if span !== nothing
                pval = @inbounds axes[k][start]
                qmax = pval + span[k]
                qstop = searchsortedlast(axes[k], qmax)
                stop = min(stop, qstop)
            end
            start:stop
        end, N)
    end

    rects = Rect{N}[]
    weights = Int[]

    if threads && Threads.nthreads() > 1
        total = length(CI)
        nchunks = min(total, Threads.nthreads())
        chunk_rects = [Rect{N}[] for _ in 1:nchunks]
        chunk_weights = [Int[] for _ in 1:nchunks]
        chunk_size = cld(total, nchunks)

        Threads.@threads for c in 1:nchunks
            start_idx = (c - 1) * chunk_size + 1
            end_idx = min(c * chunk_size, total)
            start_idx > end_idx && continue

            rects_local = chunk_rects[c]
            weights_local = chunk_weights[c]

            for idx in start_idx:end_idx
                p = CI[idx].I
                lo = ntuple(Val(N)) do k
                    @inbounds axes[k][p[k]]
                end
                q_ranges = _q_ranges(p)

                for qCI in CartesianIndices(q_ranges)
                    q = qCI.I
                    wt = @inbounds Int(w[p..., q...])
                    if abs(wt) > tol || !drop_zeros
                        hi = ntuple(Val(N)) do k
                            @inbounds axes[k][q[k]]
                        end
                        push!(rects_local, Rect{N}(lo, hi))
                        push!(weights_local, abs(wt) > tol ? wt : 0)
                    end
                end
            end
        end

        for c in 1:nchunks
            append!(rects, chunk_rects[c])
            append!(weights, chunk_weights[c])
        end
    else
        for pCI in CI
            p = pCI.I
            lo = ntuple(Val(N)) do k
                @inbounds axes[k][p[k]]
            end
            q_ranges = _q_ranges(p)

            for qCI in CartesianIndices(q_ranges)
                q = qCI.I
                wt = @inbounds Int(w[p..., q...])
                if abs(wt) > tol || !drop_zeros
                    hi = ntuple(Val(N)) do k
                        @inbounds axes[k][q[k]]
                    end
                    push!(rects, Rect{N}(lo, hi))
                    push!(weights, abs(wt) > tol ? wt : 0)
                end
            end
        end
    end

    return RectSignedBarcode{N,Int}(axes, rects, weights)
end

@inline _triangle_number(n::Int) = (n * (n + 1)) >>> 1
@inline _interval_linear_index(p::Int, q::Int) = (((q - 1) * q) >>> 1) + p

@inline function _region_rows_equal(reg::AbstractMatrix{Int}, i::Int, j::Int)
    @inbounds for k in axes(reg, 2)
        reg[i, k] == reg[j, k] || return false
    end
    return true
end

@inline function _region_cols_equal(reg::AbstractMatrix{Int}, i::Int, j::Int)
    @inbounds for k in axes(reg, 1)
        reg[k, i] == reg[k, j] || return false
    end
    return true
end

function _run_boundary_keep_indices(n::Int, equal_adjacent::Function)
    n == 0 && return Int[]
    keep = Int[]
    run_start = 1
    for i in 2:n
        equal_adjacent(i - 1, i) && continue
        push!(keep, run_start)
        (i - 1) != run_start && push!(keep, i - 1)
        run_start = i
    end
    push!(keep, run_start)
    n != run_start && push!(keep, n)
    return keep
end

function _compress_region_grid_2d(reg::Matrix{Int}, axes::NTuple{2,Vector{Int}})
    keep1 = collect(Base.axes(reg, 1))
    keep2 = collect(Base.axes(reg, 2))
    regc = reg
    changed = true
    while changed
        changed = false
        row_keep = _run_boundary_keep_indices(size(regc, 1), (i, j) -> _region_rows_equal(regc, i, j))
        if length(row_keep) < size(regc, 1)
            keep1 = keep1[row_keep]
            regc = regc[row_keep, :]
            changed = true
        end
        col_keep = _run_boundary_keep_indices(size(regc, 2), (i, j) -> _region_cols_equal(regc, i, j))
        if length(col_keep) < size(regc, 2)
            keep2 = keep2[col_keep]
            regc = regc[:, col_keep]
            changed = true
        end
    end
    return regc, keep1, keep2, (axes[1][keep1], axes[2][keep2])
end

function _fill_rectangle_rank_tensor_dense(rank_idx::Function,
                                           dims::NTuple{N,Int};
                                           threads::Bool=false) where {N}
    r = zeros(Int, (dims..., dims...))
    if N == 1
        d1 = dims[1]
        for p in 1:d1
            for q in p:d1
                @inbounds r[p, q] = rank_idx((p,), (q,))
            end
        end
        return r
    end

    CI = CartesianIndices(dims)
    if threads && Threads.nthreads() > 1
        total = length(CI)
        nchunks = min(total, Threads.nthreads())
        chunk_size = cld(total, nchunks)
        Threads.@threads for c in 1:nchunks
            start_idx = (c - 1) * chunk_size + 1
            end_idx = min(c * chunk_size, total)
            start_idx > end_idx && continue
            for idx in start_idx:end_idx
                pCI = CI[idx]
                p = pCI.I
                q_ranges = ntuple(k -> p[k]:dims[k], N)
                for qCI in CartesianIndices(q_ranges)
                    q = qCI.I
                    @inbounds r[p..., q...] = rank_idx(p, q)
                end
            end
        end
    else
        for pCI in CI
            p = pCI.I
            q_ranges = ntuple(k -> p[k]:dims[k], N)
            for qCI in CartesianIndices(q_ranges)
                q = qCI.I
                @inbounds r[p..., q...] = rank_idx(p, q)
            end
        end
    end
    return r
end

function _fill_rectangle_rank_tensor_dense_from_regions_2d(reg::AbstractMatrix{Int},
                                                           rank_ab::Function;
                                                           threads::Bool=false)
    d1, d2 = size(reg)
    r = zeros(Int, d1, d2, d1, d2)
    if threads && Threads.nthreads() > 1
        Threads.@threads for p1 in 1:d1
            for p2 in 1:d2
                a = @inbounds reg[p1, p2]
                for q1 in p1:d1
                    for q2 in p2:d2
                        b = @inbounds reg[q1, q2]
                        @inbounds r[p1, p2, q1, q2] = (a == 0 || b == 0) ? 0 : rank_ab(a, b)
                    end
                end
            end
        end
    else
        @inbounds for p1 in 1:d1
            for p2 in 1:d2
                a = reg[p1, p2]
                for q1 in p1:d1
                    for q2 in p2:d2
                        b = reg[q1, q2]
                        r[p1, p2, q1, q2] = (a == 0 || b == 0) ? 0 : rank_ab(a, b)
                    end
                end
            end
        end
    end
    return r
end

function _fill_rectangle_rank_tensor_packed_2d(rank_idx::Function,
                                               d1::Int,
                                               d2::Int;
                                               threads::Bool=false)
    n1 = _triangle_number(d1)
    n2 = _triangle_number(d2)
    r = Matrix{Int}(undef, n1, n2)
    if threads && Threads.nthreads() > 1
        Threads.@threads for q1 in 1:d1
            base1 = ((q1 - 1) * q1) >>> 1
            for p1 in 1:q1
                i1 = base1 + p1
                for q2 in 1:d2
                    base2 = ((q2 - 1) * q2) >>> 1
                    for p2 in 1:q2
                        @inbounds r[i1, base2 + p2] = rank_idx((p1, p2), (q1, q2))
                    end
                end
            end
        end
    else
        for q1 in 1:d1
            base1 = ((q1 - 1) * q1) >>> 1
            for p1 in 1:q1
                i1 = base1 + p1
                for q2 in 1:d2
                    base2 = ((q2 - 1) * q2) >>> 1
                    for p2 in 1:q2
                        @inbounds r[i1, base2 + p2] = rank_idx((p1, p2), (q1, q2))
                    end
                end
            end
        end
    end
    return r
end

function _fill_rectangle_rank_tensor_packed_from_regions_2d(reg::AbstractMatrix{Int},
                                                            rank_ab::Function;
                                                            threads::Bool=false)
    d1, d2 = size(reg)
    n1 = _triangle_number(d1)
    n2 = _triangle_number(d2)
    r = Matrix{Int}(undef, n1, n2)
    if threads && Threads.nthreads() > 1
        Threads.@threads for q1 in 1:d1
            base1 = ((q1 - 1) * q1) >>> 1
            for p1 in 1:q1
                i1 = base1 + p1
                for q2 in 1:d2
                    base2 = ((q2 - 1) * q2) >>> 1
                    for p2 in 1:q2
                        a = @inbounds reg[p1, p2]
                        b = @inbounds reg[q1, q2]
                        @inbounds r[i1, base2 + p2] = (a == 0 || b == 0) ? 0 : rank_ab(a, b)
                    end
                end
            end
        end
    else
        @inbounds for q1 in 1:d1
            base1 = ((q1 - 1) * q1) >>> 1
            for p1 in 1:q1
                i1 = base1 + p1
                for q2 in 1:d2
                    base2 = ((q2 - 1) * q2) >>> 1
                    for p2 in 1:q2
                        a = reg[p1, p2]
                        b = reg[q1, q2]
                        r[i1, base2 + p2] = (a == 0 || b == 0) ? 0 : rank_ab(a, b)
                    end
                end
            end
        end
    end
    return r
end

function _mobius_inversion_interval_product_packed_2d!(w::AbstractMatrix{<:Integer},
                                                       d1::Int,
                                                       d2::Int;
                                                       threads::Bool=false)
    n1 = _triangle_number(d1)
    n2 = _triangle_number(d2)
    size(w, 1) == n1 || error("_mobius_inversion_interval_product_packed_2d!: axis1 size mismatch")
    size(w, 2) == n2 || error("_mobius_inversion_interval_product_packed_2d!: axis2 size mismatch")

    if threads && Threads.nthreads() > 1
        Threads.@threads for j2 in 1:n2
            for q1 in 1:d1
                base1 = ((q1 - 1) * q1) >>> 1
                for p1 in q1:-1:1
                    i1 = base1 + p1
                    v = @inbounds w[i1, j2]
                    if p1 > 1
                        v -= @inbounds w[_interval_linear_index(p1 - 1, q1), j2]
                    end
                    if q1 < d1
                        v -= @inbounds w[_interval_linear_index(p1, q1 + 1), j2]
                    end
                    if p1 > 1 && q1 < d1
                        v += @inbounds w[_interval_linear_index(p1 - 1, q1 + 1), j2]
                    end
                    @inbounds w[i1, j2] = v
                end
            end
        end
        Threads.@threads for i1 in 1:n1
            for q2 in 1:d2
                base2 = ((q2 - 1) * q2) >>> 1
                for p2 in q2:-1:1
                    i2 = base2 + p2
                    v = @inbounds w[i1, i2]
                    if p2 > 1
                        v -= @inbounds w[i1, _interval_linear_index(p2 - 1, q2)]
                    end
                    if q2 < d2
                        v -= @inbounds w[i1, _interval_linear_index(p2, q2 + 1)]
                    end
                    if p2 > 1 && q2 < d2
                        v += @inbounds w[i1, _interval_linear_index(p2 - 1, q2 + 1)]
                    end
                    @inbounds w[i1, i2] = v
                end
            end
        end
        return w
    end

    for j2 in 1:n2
        for q1 in 1:d1
            base1 = ((q1 - 1) * q1) >>> 1
            for p1 in q1:-1:1
                i1 = base1 + p1
                v = @inbounds w[i1, j2]
                if p1 > 1
                    v -= @inbounds w[_interval_linear_index(p1 - 1, q1), j2]
                end
                if q1 < d1
                    v -= @inbounds w[_interval_linear_index(p1, q1 + 1), j2]
                end
                if p1 > 1 && q1 < d1
                    v += @inbounds w[_interval_linear_index(p1 - 1, q1 + 1), j2]
                end
                @inbounds w[i1, j2] = v
            end
        end
    end
    for i1 in 1:n1
        for q2 in 1:d2
            base2 = ((q2 - 1) * q2) >>> 1
            for p2 in q2:-1:1
                i2 = base2 + p2
                v = @inbounds w[i1, i2]
                if p2 > 1
                    v -= @inbounds w[i1, _interval_linear_index(p2 - 1, q2)]
                end
                if q2 < d2
                    v -= @inbounds w[i1, _interval_linear_index(p2, q2 + 1)]
                end
                if p2 > 1 && q2 < d2
                    v += @inbounds w[i1, _interval_linear_index(p2 - 1, q2 + 1)]
                end
                @inbounds w[i1, i2] = v
            end
        end
    end
    return w
end

function _extract_rectangles_from_packed_mobius_2d(w::AbstractMatrix{<:Integer},
                                                   axes::NTuple{2,Vector{Int}};
                                                   axes_out::NTuple{2,Vector{Int}}=axes,
                                                   drop_zeros::Bool=true,
                                                   tol::Int=0,
                                                   max_span=nothing)
    ax1, ax2 = axes
    d1 = length(ax1)
    d2 = length(ax2)
    span = _normalize_rectangle_span(max_span, axes)
    rects = Rect{2}[]
    weights = Int[]
    sizehint!(rects, min(length(w), _estimate_rectangle_comparable_pairs(axes)))
    sizehint!(weights, min(length(w), _estimate_rectangle_comparable_pairs(axes)))
    @inbounds for q1 in 1:d1
        hi1 = ax1[q1]
        base1 = ((q1 - 1) * q1) >>> 1
        for p1 in 1:q1
            lo1 = ax1[p1]
            if span !== nothing && hi1 - lo1 > span[1]
                continue
            end
            i1 = base1 + p1
            for q2 in 1:d2
                hi2 = ax2[q2]
                base2 = ((q2 - 1) * q2) >>> 1
                for p2 in 1:q2
                    lo2 = ax2[p2]
                    if span !== nothing && hi2 - lo2 > span[2]
                        continue
                    end
                    wt = Int(w[i1, base2 + p2])
                    if abs(wt) > tol || !drop_zeros
                        push!(rects, Rect{2}((lo1, lo2), (hi1, hi2)))
                        push!(weights, abs(wt) > tol ? wt : 0)
                    end
                end
            end
        end
    end
    return RectSignedBarcode{2,Int}(axes_out, rects, weights)
end

function _rectangle_signed_barcode_bulk_packed_2d(rank_idx::Function,
                                                  axes::NTuple{2,Vector{Int}};
                                                  axes_out::NTuple{2,Vector{Int}}=axes,
                                                  drop_zeros::Bool=true,
                                                  tol::Int=0,
                                                  max_span=nothing,
                                                  threads::Bool=false)
    r = _fill_rectangle_rank_tensor_packed_2d(rank_idx, length(axes[1]), length(axes[2]); threads=threads)
    _mobius_inversion_interval_product_packed_2d!(r, length(axes[1]), length(axes[2]); threads=threads)
    return _extract_rectangles_from_packed_mobius_2d(
        r, axes; axes_out=axes_out, drop_zeros=drop_zeros, tol=tol, max_span=max_span
    )
end

function _rectangle_signed_barcode_bulk_dense(rank_idx::Function,
                                              axes::NTuple{N,Vector{Int}};
                                              drop_zeros::Bool=true,
                                              tol::Int=0,
                                              max_span=nothing,
                                              threads::Bool=false) where {N}
    dims = ntuple(i -> length(axes[i]), N)
    r = _fill_rectangle_rank_tensor_dense(rank_idx, dims; threads=threads)
    _mobius_inversion_interval_product!(r, N)
    return _extract_rectangles_from_mobius_tensor(
        r, axes; drop_zeros=drop_zeros, tol=tol, max_span=max_span, threads=threads
    )
end

function _rectangle_signed_barcode_bulk(rank_idx::Function,
                                        axes::NTuple{N,Vector{Int}};
                                        drop_zeros::Bool=true,
                                        tol::Int=0,
                                        max_span=nothing,
                                        threads::Bool=false) where {N}
    if N == 2 && _USE_PACKED_RECTANGLE_BULK_2D[]
        return _rectangle_signed_barcode_bulk_packed_2d(
            rank_idx, axes; drop_zeros=drop_zeros, tol=tol, max_span=max_span, threads=threads
        )
    end
    return _rectangle_signed_barcode_bulk_dense(
        rank_idx, axes; drop_zeros=drop_zeros, tol=tol, max_span=max_span, threads=threads
    )
end

"""
    rectangle_signed_barcode(rank_idx, axes; drop_zeros=true, tol=0, max_span=nothing,
                             method=:auto, bulk_max_elems=20_000_000)

Compute the rectangle signed barcode (a signed decomposition into axis-aligned
rectangles) induced by a rank function on a finite grid.

The rank function is provided as `rank_idx(p,q)`, where `p` and `q` are
N-tuples of *indices* into the coordinate axes, and should be interpreted as
a rank invariant on comparable pairs `p <= q` (coordinatewise).

Algorithm choices:

- `method=:local` reproduces the original inclusion-exclusion formula, evaluating
  `rank_idx` up to `2^(2N)` times per rectangle (but allowing early truncation by
  `max_span`).
- `method=:bulk` evaluates `rank_idx` once per comparable pair `(p,q)` and then
  performs a single mixed-orientation Mobius inversion (finite differences) on a
  `prod(length.(axes))^2` array. This is typically much faster when you want the
  full barcode and rank queries are cheap enough to make overhead matter.
- `method=:auto` (default) uses `:bulk` when `max_span === nothing` and the
  required rank array is at most `bulk_max_elems` entries; otherwise it uses
  `:local`.

The output is a `RectSignedBarcode`, a list of rectangles with signed integer
weights. Zero-weight rectangles are omitted by default.
"""
function rectangle_signed_barcode(rank_idx::Function,
                                  axes::NTuple{N,<:AbstractVector{<:Integer}};
                                  drop_zeros::Bool=true,
                                  tol::Int=0,
                                  max_span=nothing,
                                  method::Symbol=:auto,
                                  bulk_max_elems::Int=20_000_000,
                                  threads::Bool = (Threads.nthreads() > 1)) where {N}
    ax = ntuple(i -> collect(Int, axes[i]), N)
    meth = _choose_rectangle_signed_barcode_method(
        method, ax; max_span=max_span, bulk_max_elems=bulk_max_elems
    )
    if meth == :bulk
        return _rectangle_signed_barcode_bulk(
            rank_idx, ax; drop_zeros=drop_zeros, tol=tol, max_span=max_span, threads=threads
        )
    else
        return _rectangle_signed_barcode_local(
            rank_idx, ax; drop_zeros=drop_zeros, tol=tol, max_span=max_span, threads=false
        )
    end
end

function _axis_slab_runs(axis::Vector{Int}, coords::Vector{Int})
    n = length(axis)
    starts = Int[]
    ends = Int[]
    slabs = Int[]
    n == 0 && return starts, ends, slabs

    sizehint!(starts, n)
    sizehint!(ends, n)
    sizehint!(slabs, n)

    ncoords = length(coords)
    current_slab = searchsortedlast(coords, axis[1])
    run_start = 1
    for i in 2:n
        next_slab = current_slab
        x = @inbounds axis[i]
        while next_slab < ncoords && x >= @inbounds(coords[next_slab + 1])
            next_slab += 1
        end
        next_slab == current_slab && continue
        push!(starts, run_start)
        push!(ends, i - 1)
        push!(slabs, current_slab)
        run_start = i
        current_slab = next_slab
    end
    push!(starts, run_start)
    push!(ends, n)
    push!(slabs, current_slab)
    return starts, ends, slabs
end

function _rectangle_region_run_grid_2d(pi::ZnEncodingMap{2},
                                       axes::NTuple{2,Vector{Int}},
                                       rq_cache::RankQueryCache;
                                       strict::Bool=false)
    starts1, ends1, slabs1 = _axis_slab_runs(axes[1], pi.coords[1])
    starts2, ends2, slabs2 = _axis_slab_runs(axes[2], pi.coords[2])
    reg_runs = Matrix{Int}(undef, length(starts1), length(starts2))

    if pi.cell_to_region !== nothing
        stride1 = pi.cell_strides[1]
        stride2 = pi.cell_strides[2]
        @inbounds for j in eachindex(starts2)
            base2 = 1 + slabs2[j] * stride2
            for i in eachindex(starts1)
                a = pi.cell_to_region[base2 + slabs1[i] * stride1]
                if strict && a == 0
                    error("rectangle_signed_barcode: point not found in encoding")
                end
                reg_runs[i, j] = a
            end
        end
        return reg_runs, starts1, ends1, starts2, ends2
    end

    @inbounds for j in eachindex(starts2)
        y = axes[2][starts2[j]]
        for i in eachindex(starts1)
            x = (axes[1][starts1[i]], y)
            a = get!(rq_cache.loc_cache, x) do
                locate(pi, x)
            end
            if strict && a == 0
                error("rectangle_signed_barcode: point not found in encoding")
            end
            reg_runs[i, j] = a
        end
    end
    return reg_runs, starts1, ends1, starts2, ends2
end

function _expand_region_run_grid_2d(reg_runs::Matrix{Int},
                                    starts1::Vector{Int},
                                    ends1::Vector{Int},
                                    starts2::Vector{Int},
                                    ends2::Vector{Int},
                                    dims::NTuple{2,Int})
    reg = Matrix{Int}(undef, dims...)
    @inbounds for j in eachindex(starts2)
        jlo = starts2[j]
        jhi = ends2[j]
        for i in eachindex(starts1)
            a = reg_runs[i, j]
            ilo = starts1[i]
            ihi = ends1[i]
            for jj in jlo:jhi
                for ii in ilo:ihi
                    reg[ii, jj] = a
                end
            end
        end
    end
    return reg
end

function _cached_rectangle_region_grid(pi::ZnEncodingMap,
                                       axes::NTuple{N,Vector{Int}},
                                       rq_cache::RankQueryCache,
                                       session_cache::Union{Nothing,SessionCache};
                                       strict::Bool=false) where {N}
    session_cache === nothing && return _rectangle_region_grid(pi, axes, rq_cache; strict=strict)
    enc_cache = _workflow_encoding_cache(session_cache)
    key = _signed_measures_region_grid_key(pi, axes, strict)
    cached = _signed_measures_cache_get(enc_cache, key)
    if cached isa AbstractArray{<:Integer,N}
        return cached
    end
    reg = _rectangle_region_grid(pi, axes, rq_cache; strict=strict)
    return _signed_measures_cache_set!(enc_cache, key, reg)
end

function _cached_rectangle_region_blocks_2d(pi::ZnEncodingMap{2},
                                            axes::NTuple{2,Vector{Int}},
                                            rq_cache::RankQueryCache,
                                            session_cache::Union{Nothing,SessionCache};
                                            strict::Bool=false)
    if session_cache === nothing
        reg = _cached_rectangle_region_grid(pi, axes, rq_cache, session_cache; strict=strict)
        return _compress_region_grid_2d(reg::Matrix{Int}, axes)
    end
    enc_cache = _workflow_encoding_cache(session_cache)
    key = _signed_measures_region_blocks_2d_key(pi, axes, strict)
    cached = _signed_measures_cache_get(enc_cache, key)
    cached === nothing || return cached
    reg = _cached_rectangle_region_grid(pi, axes, rq_cache, session_cache; strict=strict)
    blocks = _compress_region_grid_2d(reg::Matrix{Int}, axes)
    return _signed_measures_cache_set!(enc_cache, key, blocks)
end

@inline function _signed_measures_packed_tensor_2d_get(rq_cache::Union{Nothing,RankQueryCache},
                                                       session_cache::Union{Nothing,SessionCache},
                                                       key)
    if _USE_PACKED_TENSOR_CACHE_2D[]
        enc_cache = _workflow_encoding_cache(session_cache)
        cached = _signed_measures_cache_get(enc_cache, key)
        cached isa _RectanglePackedTensor2D && return cached
        cached = _signed_measures_rq_tensor_get(session_cache === nothing ? rq_cache : nothing, key)
        cached isa _RectanglePackedTensor2D && return cached
    end
    return nothing
end

@inline function _signed_measures_packed_tensor_2d_set!(rq_cache::Union{Nothing,RankQueryCache},
                                                        session_cache::Union{Nothing,SessionCache},
                                                        key,
                                                        value::_RectanglePackedTensor2D)
    _USE_PACKED_TENSOR_CACHE_2D[] || return value
    enc_cache = _workflow_encoding_cache(session_cache)
    _signed_measures_rq_tensor_set!(session_cache === nothing ? rq_cache : nothing, key, value)
    return _signed_measures_cache_set!(enc_cache, key, value)
end

function _cached_rectangle_packed_tensor_2d(M::PModule,
                                            pi::ZnEncodingMap{2},
                                            axes::NTuple{2,Vector{Int}},
                                            reg_comp::AbstractMatrix{Int},
                                            axes_comp::NTuple{2,Vector{Int}},
                                            rq_cache::RankQueryCache,
                                            session_cache::Union{Nothing,SessionCache},
                                            cc::CoverCache;
                                            strict::Bool=false,
                                            threads::Bool=false)
    key = _signed_measures_packed_tensor_2d_key(M, pi, axes, strict)
    cached = _signed_measures_packed_tensor_2d_get(rq_cache, session_cache, key)
    cached === nothing || return cached

    function rank_ab(a::Int, b::Int)
        return _rank_cache_get!(rq_cache, a, b, () -> rank_map(M, a, b; cache=cc))
    end
    r = _fill_rectangle_rank_tensor_packed_from_regions_2d(reg_comp, rank_ab; threads=threads)
    _mobius_inversion_interval_product_packed_2d!(r, size(reg_comp, 1), size(reg_comp, 2); threads=threads)
    return _signed_measures_packed_tensor_2d_set!(
        rq_cache, session_cache, key, _RectanglePackedTensor2D(axes_comp, r)
    )
end

function _choose_rectangle_signed_barcode_method_module_with_blocks(M::PModule,
                                                                    pi::ZnEncodingMap,
                                                                    axes::NTuple{N,Vector{Int}},
                                                                    rq_cache::RankQueryCache,
                                                                    session_cache::Union{Nothing,SessionCache};
                                                                    strict::Bool,
                                                                    method::Symbol,
                                                                    max_span,
                                                                    bulk_max_elems::Int) where {N}
    method == :auto || return (
        _choose_rectangle_signed_barcode_method(
            method, axes; max_span=max_span, bulk_max_elems=bulk_max_elems
        ),
        nothing,
    )
    max_span === nothing || return :local, nothing
    _estimate_rectangle_rank_array_elems(axes) <= bulk_max_elems || return :local, nothing
    _rectangle_axis_aspect(axes) <= _RECTANGLE_BULK_MAX_AXIS_ASPECT[] || return :local, nothing
    if N != 2 || !_USE_PACKED_RECTANGLE_BULK_2D[]
        return :bulk, nothing
    end
    tensor_key = _signed_measures_packed_tensor_2d_key(M, pi, axes, strict)
    _signed_measures_packed_tensor_2d_get(rq_cache, session_cache, tensor_key) === nothing || return :bulk, nothing
    blocks = _cached_rectangle_region_blocks_2d(pi, axes, rq_cache, session_cache; strict=strict)
    reg_comp, _, _, _ = blocks
    comparable_pairs = _triangle_number(size(reg_comp, 1)) * _triangle_number(size(reg_comp, 2))
    if comparable_pairs <= _RECTANGLE_BULK_AUTO_MAX_COMPRESSED_COMPARABLE_PAIRS[]
        return :bulk, blocks
    end
    return :local, blocks
end

function _choose_rectangle_signed_barcode_method_module(M::PModule,
                                                        pi::ZnEncodingMap,
                                                        axes::NTuple{N,Vector{Int}},
                                                        rq_cache::RankQueryCache,
                                                        session_cache::Union{Nothing,SessionCache};
                                                        strict::Bool,
                                                        method::Symbol,
                                                        max_span,
                                                        bulk_max_elems::Int) where {N}
    return first(_choose_rectangle_signed_barcode_method_module_with_blocks(
        M, pi, axes, rq_cache, session_cache;
        strict=strict,
        method=method,
        max_span=max_span,
        bulk_max_elems=bulk_max_elems,
    ))
end

function _rectangle_region_grid(pi::ZnEncodingMap{2},
                                axes::NTuple{2,Vector{Int}},
                                rq_cache::RankQueryCache;
                                strict::Bool=false)
    reg_runs, starts1, ends1, starts2, ends2 = _rectangle_region_run_grid_2d(
        pi, axes, rq_cache; strict=strict
    )
    return _expand_region_run_grid_2d(
        reg_runs, starts1, ends1, starts2, ends2, (length(axes[1]), length(axes[2]))
    )
end

function _rectangle_region_grid(pi::ZnEncodingMap,
                                axes::NTuple{N,Vector{Int}},
                                rq_cache::RankQueryCache;
                                strict::Bool=false) where {N}
    dims = ntuple(i -> length(axes[i]), N)
    npoints_big = foldl(*, (big(d) for d in dims); init=big(1))
    use_linear_loc_cache = npoints_big <= RECTANGLE_LOC_LINEAR_CACHE_THRESHOLD[]
    loc_cache_linear = use_linear_loc_cache ? zeros(Int, Int(npoints_big)) : Int[]
    loc_cache_filled = use_linear_loc_cache ? falses(Int(npoints_big)) : falses(0)
    reg = Array{Int,N}(undef, dims...)

    @inline function region_idx_from_grid(p::NTuple{N,Int}) where {N}
        if use_linear_loc_cache
            idx = _grid_cache_index(p, dims)
            idx == 0 && return 0
            if loc_cache_filled[idx]
                return loc_cache_linear[idx]
            end
            x = ntuple(Val(N)) do k
                @inbounds axes[k][p[k]]
            end
            a = locate(pi, x)
            loc_cache_linear[idx] = a
            loc_cache_filled[idx] = true
            return a
        end
        x = ntuple(Val(N)) do k
            @inbounds axes[k][p[k]]
        end
        return get!(rq_cache.loc_cache, x) do
            locate(pi, x)
        end
    end

    for pCI in CartesianIndices(dims)
        a = region_idx_from_grid(pCI.I)
        if strict && a == 0
            error("rectangle_signed_barcode: point not found in encoding")
        end
        @inbounds reg[pCI] = a
    end
    return reg
end

"""
    rectangle_signed_barcode(M, pi, opts::InvariantOptions; drop_zeros=true,
                             tol=0, max_span=nothing, rq_cache=nothing,
                             cache=nothing, keep_endpoints=true, method=:auto,
                             bulk_max_elems=20_000_000)

Compute the rectangle signed barcode of a Z^n module `M` using a `ZnEncodingMap` `pi`.

The barcode is the Mobius inversion of the rank invariant restricted to the lattice grid
`axes` (one integer coordinate vector per dimension).

This is the opts-primary overload. Axes selection and strictness are driven by `opts`:
- `opts.axes`, `opts.axes_policy`, `opts.max_axis_len`
- `opts.strict` (if `nothing`, treated as `false` here)

Keyword arguments are forwarded to the underlying rectangle signed barcode algorithms.

Axis selection:
- If `axes === nothing`, use `axes_from_encoding(pi)` (encoding-derived "critical" values).
- Otherwise `axes` is normalized to sorted unique integer vectors, then optionally modified by
  `axes_policy`:
    * `:as_given`  keep the provided axes.
    * `:encoding`  intersect with encoding-derived axes (and keep endpoints if requested).
    * `:coarsen`   uniformly downsample each axis to length <= `max_axis_len`.

Caching:
- Pass `rq_cache = RankQueryCache(pi)` to reuse cached `locate(pi, x)` values and cached region-pair
  ranks `rank_map(M, a, b)` across calls. Repeated identical rectangle-barcode calls with the same
  `rq_cache` also reuse the final `RectSignedBarcode`.
- Pass `cache=SessionCache()` to make repeated rectangle/Euler/MMA calls reuse the canonical
  workflow-level path without manually managing `RankQueryCache`. Identical rectangle-barcode
  calls on the same session cache also reuse the final `RectSignedBarcode` directly. For 2D bulk
  workflows, the same session cache also reuses the region grid and compressed block grid for the
  chosen `(pi, axes, strict)` inputs.

Algorithm choices:
- `method=:bulk` evaluates the rank function once per comparable pair on the grid, stores it in a
  `prod(length.(axes))^2` array, and performs one mixed Mobius inversion (finite differences).
  This avoids the `2^(2n)` rank evaluations per rectangle of the local formula and is usually
  much faster on moderate grids.
- `method=:local` uses the original inclusion-exclusion formula and can be preferable when `max_span`
  is set (because it avoids building the full rank array).
- `method=:auto` (default) uses `:bulk` when `max_span === nothing` and the rank array would have at
  most `bulk_max_elems` entries.

`strict=false` treats grid points not found in the encoding as rank 0.
"""
function rectangle_signed_barcode(M::PModule, pi::ZnEncodingMap, opts::InvariantOptions;
                                  drop_zeros::Bool=true,
                                  tol::Int=0,
                                  max_span=nothing,
                                  rq_cache::Union{Nothing,RankQueryCache}=nothing,
                                  cache=nothing,
                                  keep_endpoints::Bool=true,
                                  method::Symbol=:auto,
                                  bulk_max_elems::Int=20_000_000,
                                  threads::Bool = (Threads.nthreads() > 1))
    rq_cache_input = rq_cache

    axes = opts.axes
    axes_policy = opts.axes_policy
    max_axis_len = opts.max_axis_len
    strict = opts.strict === nothing ? false : opts.strict

    axesN = axes === nothing ? axes_from_encoding(pi) : _normalize_axes(axes)

    if axes !== nothing && axes_policy != :as_given
        enc_axes = axes_from_encoding(pi)
        if axes_policy == :encoding
            axesN = _axes_policy_encoding(axesN, enc_axes; keep_endpoints=keep_endpoints)
        elseif axes_policy == :coarsen
            axesN = ntuple(i -> coarsen_axis(axesN[i]; max_len=max_axis_len, method=:uniform),
                           pi.n)
        else
            error("Unknown axes_policy=$(axes_policy)")
        end
    end

    session_cache = _signed_measures_session_cache(cache)
    enc_cache = _workflow_encoding_cache(session_cache)
    axesN isa NTuple{pi.n,Vector{Int}} || error("axes length mismatch: expected pi.n axes")
    dims = ntuple(i -> length(axesN[i]), pi.n)

    @inline function _cached_rect_for_method(meth::Symbol)
        rect_key = _signed_measures_rectangle_barcode_key(
            M, pi, axesN, meth; drop_zeros=drop_zeros, tol=tol, max_span=max_span, strict=strict
        )
        cached = _signed_measures_cache_get(enc_cache, rect_key)
        cached isa RectSignedBarcode && return meth, rect_key, cached
        cached = _signed_measures_rq_rectangle_get(session_cache === nothing ? rq_cache_input : nothing, rect_key)
        cached isa RectSignedBarcode && return meth, rect_key, cached
        return meth, rect_key, nothing
    end

    if method == :auto
        for meth_try in (:bulk, :local)
            meth0, rect_key0, cached0 = _cached_rect_for_method(meth_try)
            cached0 === nothing || return cached0
        end
    end

    meth = method == :auto ? :auto :
        _choose_rectangle_signed_barcode_method(method, axesN; max_span=max_span, bulk_max_elems=bulk_max_elems)
    rect_key = nothing
    if method != :auto
        meth, rect_key, cached_rect = _cached_rect_for_method(meth)
        cached_rect === nothing || return cached_rect
    end

    rq_cache = _signed_measures_rank_query_cache(M, pi, rq_cache, session_cache)

    blocks_hint = nothing
    if method == :auto
        meth, blocks_hint = _choose_rectangle_signed_barcode_method_module_with_blocks(
            M, pi, axesN, rq_cache, session_cache;
            strict=strict,
            method=method,
            max_span=max_span,
            bulk_max_elems=bulk_max_elems,
        )
        meth, rect_key, cached_rect = _cached_rect_for_method(meth)
        cached_rect === nothing || return cached_rect
    end

    build_cache!(M.Q; cover=true, updown=false)
    cc = _get_cover_cache(M.Q)

    function rank_ab(a::Int, b::Int)
        return _rank_cache_get!(rq_cache, a, b, () -> rank_map(M, a, b; cache=cc))
    end

    npoints_big = foldl(*, (big(d) for d in dims); init=big(1))
    use_linear_loc_cache = npoints_big <= RECTANGLE_LOC_LINEAR_CACHE_THRESHOLD[]
    loc_cache_linear = use_linear_loc_cache ? zeros(Int, Int(npoints_big)) : Int[]
    loc_cache_filled = use_linear_loc_cache ? falses(Int(npoints_big)) : falses(0)

    @inline function region_idx_from_grid(p::NTuple{N,Int}) where {N}
        if use_linear_loc_cache
            idx = _grid_cache_index(p, dims)
            idx == 0 && return 0
            if loc_cache_filled[idx]
                return loc_cache_linear[idx]
            end
            x = ntuple(Val(N)) do k
                @inbounds axesN[k][p[k]]
            end
            a = locate(pi, x)
            loc_cache_linear[idx] = a
            loc_cache_filled[idx] = true
            return a
        end
        x = ntuple(Val(N)) do k
            @inbounds axesN[k][p[k]]
        end
        return get!(rq_cache.loc_cache, x) do
            locate(pi, x)
        end
    end

    if meth == :bulk
        if pi.n == 2 && _USE_PACKED_RECTANGLE_BULK_2D[]
            reg_comp, _, _, axes_comp =
                blocks_hint === nothing ?
                _cached_rectangle_region_blocks_2d(pi, axesN, rq_cache, session_cache; strict=strict) :
                blocks_hint
            packed = _cached_rectangle_packed_tensor_2d(
                M, pi, axesN, reg_comp, axes_comp, rq_cache, session_cache, cc;
                strict=strict,
                threads=threads,
            )
            sb = _extract_rectangles_from_packed_mobius_2d(
                packed.weights, packed.axes_comp;
                axes_out=axesN,
                drop_zeros=drop_zeros,
                tol=tol,
                max_span=max_span,
            )
        else
            reg = _cached_rectangle_region_grid(pi, axesN, rq_cache, session_cache; strict=strict)
            r = if pi.n == 2
                _fill_rectangle_rank_tensor_dense_from_regions_2d(reg::Matrix{Int}, rank_ab; threads=threads)
            else
                function rank_idx_reg(p::NTuple{N,Int}, q::NTuple{N,Int}) where {N}
                    a = @inbounds reg[p...]
                    b = @inbounds reg[q...]
                    return (a == 0 || b == 0) ? 0 : rank_ab(a, b)
                end
                _fill_rectangle_rank_tensor_dense(rank_idx_reg, dims; threads=threads)
            end
            _mobius_inversion_interval_product!(r, pi.n)
            sb = _extract_rectangles_from_mobius_tensor(
                r, axesN; drop_zeros=drop_zeros, tol=tol, max_span=max_span, threads=threads
            )
        end
        _signed_measures_rq_rectangle_set!(session_cache === nothing ? rq_cache_input : nothing, rect_key, sb)
        return _signed_measures_cache_set!(enc_cache, rect_key, sb)
    end

    function rank_idx_cached(p::NTuple{N,Int}, q::NTuple{N,Int}) where {N}
        a = region_idx_from_grid(p)
        b = region_idx_from_grid(q)
        if a == 0 || b == 0
            if strict
                error("Point not found in encoding")
            end
            return 0
        end
        return rank_ab(a, b)
    end

    sb = _rectangle_signed_barcode_local(
        rank_idx_cached, axesN; drop_zeros=drop_zeros, tol=tol, max_span=max_span, threads=false
    )
    _signed_measures_rq_rectangle_set!(session_cache === nothing ? rq_cache_input : nothing, rect_key, sb)
    return _signed_measures_cache_set!(enc_cache, rect_key, sb)
end

@inline function _coarsen_axis_real_to_length(axis::AbstractVector{<:Real}, max_len::Int)
    max_len > 0 || throw(ArgumentError("coarsen_axis: max_len must be > 0"))
    ax = sort(unique(float.(axis)))
    while length(ax) > max_len
        ax = coarsen_axis(ax)
    end
    return ax
end

@inline function _grid_encoding_axis_index(coords::AbstractVector{<:Real}, x::Real)
    xi = float(x)
    if xi == 0
        xi = zero(xi)
    end
    return searchsortedlast(coords, xi)
end

function _grid_encoding_axis_semantic_pairs(coords::AbstractVector{<:Real},
                                            axis_sem::AbstractVector{<:Real})
    idxs = Int[]
    vals = Float64[]
    last_idx = 0
    for x in axis_sem
        idx = _grid_encoding_axis_index(coords, x)
        if idx < 1
            continue
        end
        if isempty(idxs) || idx != last_idx
            push!(idxs, idx)
            push!(vals, float(x))
            last_idx = idx
        end
    end
    return vals, idxs
end

@inline function _restrict_grid_axis_to_encoding(axis::AbstractVector{<:Real},
                                                 enc_axis::AbstractVector{<:Real};
                                                 keep_endpoints::Bool=true)
    ax = sort(unique(float.(axis)))
    isempty(ax) && return sort(unique(float.(enc_axis)))
    lo = first(ax)
    hi = last(ax)
    vals = Float64[]
    sizehint!(vals, length(enc_axis) + (keep_endpoints ? 2 : 0))
    for v in enc_axis
        lo <= v <= hi && push!(vals, float(v))
    end
    if keep_endpoints
        push!(vals, lo)
        push!(vals, hi)
    end
    return sort(unique(vals))
end

function _rectangle_signed_barcode_grid_axes(pi::GridEncodingMap{N},
                                             opts::InvariantOptions;
                                             keep_endpoints::Bool=true) where {N}
    axes = opts.axes
    axes_policy = opts.axes_policy
    max_axis_len = opts.max_axis_len

    axes_sem = axes === nothing ? ntuple(i -> copy(pi.coords[i]), N) : _normalize_axes_real(axes)

    if axes !== nothing && axes_policy != :as_given
        enc_axes = axes_from_encoding(pi)
        if axes_policy == :encoding
            axes_sem = ntuple(i -> _restrict_grid_axis_to_encoding(
                axes_sem[i], enc_axes[i]; keep_endpoints=keep_endpoints
            ), N)
        elseif axes_policy == :coarsen
            axes_sem = ntuple(i -> _coarsen_axis_real_to_length(axes_sem[i], max_axis_len), N)
        else
            error("Unknown axes_policy=$(axes_policy)")
        end
    end

    sem_aligned = Vector{Vector{Float64}}(undef, N)
    idx_aligned = Vector{Vector{Int}}(undef, N)
    for i in 1:N
        vals, idxs = _grid_encoding_axis_semantic_pairs(pi.coords[i], axes_sem[i])
        sem_aligned[i] = vals
        idx_aligned[i] = idxs
    end
    return ntuple(i -> sem_aligned[i], N), ntuple(i -> idx_aligned[i], N)
end

function _rectangle_signed_barcode_grid_semantic_axes(pi::GridEncodingMap{N},
                                                      opts::InvariantOptions;
                                                      keep_endpoints::Bool=true) where {N}
    birth_axes, _ = _rectangle_signed_barcode_grid_axes(pi, opts; keep_endpoints=keep_endpoints)
    death_axes = ntuple(i -> begin
        ax = birth_axes[i]
        n = length(ax)
        vals = Vector{Float64}(undef, n)
        @inbounds for j in 1:n
            vals[j] = j < n ? Float64(ax[j + 1]) : Inf
        end
        vals
    end, N)
    return birth_axes, death_axes
end

function rectangle_signed_barcode(M::PModule, pi::GridEncodingMap{N}, opts::InvariantOptions;
                                  drop_zeros::Bool=true,
                                  tol::Int=0,
                                  max_span=nothing,
                                  rq_cache=nothing,
                                  cache=nothing,
                                  keep_endpoints::Bool=true,
                                  method::Symbol=:auto,
                                  bulk_max_elems::Int=20_000_000,
                                  threads::Bool = (Threads.nthreads() > 1)) where {N}
    rq_cache === nothing || throw(ArgumentError("rectangle_signed_barcode: rq_cache is only supported for ZnEncodingMap encodings."))

    _, axes_idx = _rectangle_signed_barcode_grid_axes(pi, opts; keep_endpoints=keep_endpoints)
    strict = opts.strict === nothing ? false : opts.strict

    session_cache = _signed_measures_session_cache(cache)
    enc_cache = _workflow_encoding_cache(session_cache)
    meth = _choose_rectangle_signed_barcode_method(
        method, axes_idx; max_span=max_span, bulk_max_elems=bulk_max_elems
    )
    rect_key = _signed_measures_rectangle_barcode_key(
        M, pi, axes_idx, meth;
        drop_zeros=drop_zeros,
        tol=tol,
        max_span=max_span,
        strict=strict,
    )
    cached = _signed_measures_cache_get(enc_cache, rect_key)
    cached isa RectSignedBarcode && return cached

    build_cache!(M.Q; cover=true, updown=false)
    cc = _get_cover_cache(M.Q)
    rank_cache = Dict{Tuple{Int,Int},Int}()

    @inline function region_idx_from_grid(p::NTuple{K,Int}) where {K}
        idx = 1
        @inbounds for i in 1:K
            slab = axes_idx[i][p[i]]
            slab == 0 && return 0
            idx += (slab - 1) * pi.strides[i]
        end
        return idx
    end

    @inline function rank_ab(a::Int, b::Int)
        return get!(rank_cache, (a, b)) do
            rank_map(M, a, b; cache=cc)
        end
    end

    function rank_idx(p::NTuple{K,Int}, q::NTuple{K,Int}) where {K}
        a = region_idx_from_grid(p)
        b = region_idx_from_grid(q)
        if a == 0 || b == 0
            strict && error("rectangle_signed_barcode: point not found in encoding")
            return 0
        end
        return rank_ab(a, b)
    end

    sb = rectangle_signed_barcode(
        rank_idx,
        axes_idx;
        drop_zeros=drop_zeros,
        tol=tol,
        max_span=max_span,
        method=meth,
        bulk_max_elems=bulk_max_elems,
        threads=threads,
    )
    return _signed_measures_cache_set!(enc_cache, rect_key, sb)
end

function rectangle_signed_barcode(M::PModule, pi::CompiledEncoding{<:ZnEncodingMap}, opts::InvariantOptions;
                                  kwargs...)
    return rectangle_signed_barcode(M, pi.pi, opts; kwargs...)
end

function rectangle_signed_barcode(M::PModule, pi::CompiledEncoding{<:GridEncodingMap}, opts::InvariantOptions;
                                  kwargs...)
    return rectangle_signed_barcode(M, pi.pi, opts; kwargs...)
end

rectangle_signed_barcode(M::PModule, pi::ZnEncodingMap;
    axes = nothing,
    axes_policy::Symbol = :encoding,
    max_axis_len::Int = 64,
    kwargs...) =
    rectangle_signed_barcode(
        M, pi,
        InvariantOptions(axes = axes, axes_policy = axes_policy, max_axis_len = max_axis_len);
        kwargs...
    )

rectangle_signed_barcode(M::PModule, pi::CompiledEncoding{<:ZnEncodingMap}; kwargs...) =
    rectangle_signed_barcode(M, pi.pi; kwargs...)

rectangle_signed_barcode(M::PModule, pi::GridEncodingMap;
    axes = nothing,
    axes_policy::Symbol = :encoding,
    max_axis_len::Int = 64,
    kwargs...) =
    rectangle_signed_barcode(
        M, pi,
        InvariantOptions(axes = axes, axes_policy = axes_policy, max_axis_len = max_axis_len);
        kwargs...
    )

rectangle_signed_barcode(M::PModule, pi::CompiledEncoding{<:GridEncodingMap}; kwargs...) =
    rectangle_signed_barcode(M, pi.pi; kwargs...)




"""
    rank_from_signed_barcode(sb, p, q)

Evaluate the rank function reconstructed from a rectangle signed barcode.

Inputs `p` and `q` are points in Z^N (represented as NTuples) with `p <= q`.
The value returned is

    sum_{rect in sb} weight(rect) * 1[ rect.lo <= p and q <= rect.hi ].

Warning:
This reconstruction is exact on the grid used to compute `sb`. Off-grid evaluation is meaningful
when the underlying rank function is constant on the corresponding encoded cells.
"""
function rank_from_signed_barcode(sb::RectSignedBarcode{N}, p::NTuple{N,Int}, q::NTuple{N,Int}) where {N}
    _tuple_leq(p, q) || error("rank_from_signed_barcode: expected p <= q")
    r = 0
    for (rect, w) in zip(sb.rects, sb.weights)
        if _tuple_leq(rect.lo, p) && _tuple_leq(q, rect.hi)
            r += w
        end
    end
    return r
end

"""
    rectangles_from_grid(axes; max_span=nothing)

Enumerate all grid-aligned rectangles with corners in the given axes.

The order matches the iteration order used by `rectangle_signed_barcode`:
lexicographic in the lower corner, and lexicographic in the upper corner
restricted by `lo <= hi` (coordinatewise). This is useful for debugging and for
building feature matrices indexed by rectangles.
"""
function rectangles_from_grid(axes::NTuple{N,<:AbstractVector{<:Integer}}; max_span=nothing) where {N}
    ax = ntuple(i -> collect(Int, axes[i]), N)
    span = max_span === nothing ? nothing :
        (max_span isa Integer ? ntuple(_ -> Int(max_span), N) :
                               ntuple(i -> Int(max_span[i]), N))
    dims = ntuple(i -> length(ax[i]), N)
    rects = Rect{N}[]
    CI = CartesianIndices(dims)
    for pCI in CI
        p = pCI.I
        q_ranges = ntuple(k -> p[k]:dims[k], N)
        for qCI in CartesianIndices(q_ranges)
            q = qCI.I
            if span !== nothing
                bad = false
                @inbounds for k in 1:N
                    if ax[k][q[k]] - ax[k][p[k]] > span[k]
                        bad = true
                        break
                    end
                end
                bad && continue
            end
            lo = ntuple(k -> ax[k][p[k]], N)
            hi = ntuple(k -> ax[k][q[k]], N)
            push!(rects, Rect{N}(lo, hi))
        end
    end
    return rects
end

"""
    rectangle_signed_barcode_rank(sb; zero_noncomparable=true)

Reconstruct the rank function on the underlying grid from a rectangle signed barcode.

Returns a 2N-dimensional Int array `R` with indices `(p..., q...)`, where `p` and `q`
range over grid indices. For comparable pairs `p <= q`, `R[p..., q...]` equals the
rank invariant reconstructed from `sb`.

Implementation note:
This is the inverse of the Mobius inversion used by `rectangle_signed_barcode`, and is
implemented via a mixed-orientation prefix-sum transform (no per-rectangle scanning).
"""
function rectangle_signed_barcode_rank(sb::RectSignedBarcode{N};
                                       zero_noncomparable::Bool=true,
                                       threads::Bool = (Threads.nthreads() > 1)) where {N}
    axes = sb.axes
    dims = ntuple(i -> length(axes[i]), N)

    # Coordinate-to-index maps for each axis (axes are assumed unique and sorted).
    idx = ntuple(i -> Dict{Int,Int}(axes[i][j] => j for j in 1:dims[i]), N)

    # Directly accumulate rank values over comparable pairs.
    w = zeros(Int, (dims..., dims...))
    for (rect, wt) in zip(sb.rects, sb.weights)
        plo = ntuple(i -> idx[i][rect.lo[i]], N)
        phi = ntuple(i -> idx[i][rect.hi[i]], N)
        p_ranges = ntuple(k -> plo[k]:phi[k], N)
        for pCI in CartesianIndices(p_ranges)
            p = pCI.I
            q_ranges = ntuple(k -> p[k]:phi[k], N)
            for qCI in CartesianIndices(q_ranges)
                @inbounds w[pCI, qCI] += Int(wt)
            end
        end
    end

    if zero_noncomparable
        # Convention: rank invariant is only meaningful for comparable pairs p <= q.
        # We zero out noncomparable index pairs for a cleaner array representation.
        CI = CartesianIndices(dims)
        if threads && Threads.nthreads() > 1
            Threads.@threads for idx in 1:length(CI)
                pCI = CI[idx]
                p = pCI.I
                for qCI in CI
                    q = qCI.I
                    comparable = true
                    @inbounds for k in 1:N
                        if p[k] > q[k]
                            comparable = false
                            break
                        end
                    end
                    if !comparable
                        @inbounds w[pCI, qCI] = 0
                    end
                end
            end
        else
            for pCI in CI
                p = pCI.I
                for qCI in CI
                    q = qCI.I
                    comparable = true
                    @inbounds for k in 1:N
                        if p[k] > q[k]
                            comparable = false
                            break
                        end
                    end
                    if !comparable
                        @inbounds w[pCI, qCI] = 0
                    end
                end
            end
        end
    end

    return w
end



"""
    truncate_signed_barcode(sb; max_terms=nothing, min_abs_weight=1)

Return a truncated signed barcode by discarding rectangles with small weights.

- Keep only rectangles with `abs(weight) >= min_abs_weight`.
- If `max_terms` is set, keep at most that many rectangles, chosen by decreasing `abs(weight)`.

This is a simple "module approximation (MMA) style" step: the full signed barcode is exact
(on the chosen grid), while truncation yields a controlled-size approximation suitable for
feature extraction pipelines.
"""
function truncate_signed_barcode(sb::RectSignedBarcode{N}; max_terms=nothing, min_abs_weight::Int=1) where {N}
    keep = [i for i in eachindex(sb.weights) if abs(sb.weights[i]) >= min_abs_weight]
    if max_terms !== nothing && length(keep) > max_terms
        sort!(keep, by=i -> abs(sb.weights[i]), rev=true)
        keep = keep[1:max_terms]
        sort!(keep)
    end
    rects = sb.rects[keep]
    weights = sb.weights[keep]
    return RectSignedBarcode{N,eltype(weights)}(sb.axes, rects, weights)
end

"""
    rectangle_signed_barcode_kernel(sb1, sb2; kind=:linear, sigma=1.0)

Kernels for rectangle signed barcodes.

- `kind=:linear` computes the dot product of rectangle weights on exactly matching rectangles.
- `kind=:gaussian` uses a Gaussian kernel on rectangle endpoints in R^(2N):

      k(sb1,sb2) = sum_{i,j} w1[i] * w2[j] * exp(-||emb(rect1[i]) - emb(rect2[j])||^2 / (2*sigma^2))

  where `emb(rect) = (lo..., hi...)`.

The Gaussian version is a standard RKHS embedding of signed measures.
"""
@inline function _rectangle_signed_barcode_embedding(sb::RectSignedBarcode{N}) where {N}
    nrect = length(sb.rects)
    emb = Matrix{Float64}(undef, 2 * N, nrect)
    weights = Vector{Float64}(undef, nrect)
    @inbounds for j in 1:nrect
        rect = sb.rects[j]
        for k in 1:N
            emb[k, j] = float(rect.lo[k])
            emb[N + k, j] = float(rect.hi[k])
        end
        weights[j] = float(sb.weights[j])
    end
    return emb, weights
end

@inline function _rectangle_signed_barcode_kernel_gaussian(emb1::Matrix{Float64},
                                                           wts1::Vector{Float64},
                                                           emb2::Matrix{Float64},
                                                           wts2::Vector{Float64},
                                                           inv_two_sig2::Float64,
                                                           ::Val{D}) where {D}
    n1 = size(emb1, 2)
    n2 = size(emb2, 2)
    s = 0.0
    @inbounds for j in 1:n2
        w2 = wts2[j]
        for i in 1:n1
            d2 = 0.0
            @simd for k in 1:D
                dk = emb1[k, i] - emb2[k, j]
                d2 += dk * dk
            end
            s += wts1[i] * w2 * exp(-d2 * inv_two_sig2)
        end
    end
    return s
end

function rectangle_signed_barcode_kernel(sb1::RectSignedBarcode{N}, sb2::RectSignedBarcode{N}; kind::Symbol=:linear, sigma::Real=1.0) where {N}
    if kind == :linear
        d2 = Dict{Rect{N},Int}()
        for (r,w) in zip(sb2.rects, sb2.weights)
            d2[r] = get(d2, r, 0) + w
        end
        s = 0.0
        for (r,w1) in zip(sb1.rects, sb1.weights)
            w2 = get(d2, r, 0)
            s += w1 * w2
        end
        return s
    elseif kind == :gaussian
        sig2 = float(sigma)^2
        sig2 > 0 || error("rectangle_signed_barcode_kernel: sigma must be > 0")
        emb1, wts1 = _rectangle_signed_barcode_embedding_cached(sb1)
        emb2, wts2 = _rectangle_signed_barcode_embedding_cached(sb2)
        return _rectangle_signed_barcode_kernel_gaussian(
            emb1, wts1, emb2, wts2, 0.5 / sig2, Val(2 * N)
        )
    else
        error("rectangle_signed_barcode_kernel: unknown kind $(kind)")
    end
end

"""
    rectangle_signed_barcode_image(sb; xs=nothing, ys=nothing, sigma=1.0,
                                   mode=:center, cutoff_tol=0.0)

A simple 2D "image-like" vectorization of a rectangle signed barcode.

Each rectangle contributes its signed weight to a 2D grid via a Gaussian bump centered at:
- `mode=:center` (default): the rectangle center (average of `lo` and `hi`)
- `mode=:lo`: the lower corner `lo`
- `mode=:hi`: the upper corner `hi`

If `cutoff_tol > 0`, contributions whose Gaussian weight would be below
`cutoff_tol` are skipped via a finite radius cutoff. The default `cutoff_tol=0`
keeps the exact image and still uses a separable 2D accumulation path.

This is intentionally lightweight and meant for data-analysis pipelines.
For N != 2, this function errors (a higher-dimensional tensorization would be needed).
"""
@inline function _rectangle_image_anchor(rect::Rect{2}, mode::Symbol)
    if mode == :center
        return ((rect.lo[1] + rect.hi[1]) / 2, (rect.lo[2] + rect.hi[2]) / 2)
    elseif mode == :lo
        return (rect.lo[1], rect.lo[2])
    elseif mode == :hi
        return (rect.hi[1], rect.hi[2])
    end
    error("rectangle_signed_barcode_image: unknown mode $(mode)")
end

@inline function _rectangle_image_cutoff_radius(sigma::Float64, cutoff_tol::Real)
    cutoff_tol <= 0 && return Inf
    0 < cutoff_tol < 1 || throw(ArgumentError("rectangle_signed_barcode_image: cutoff_tol must lie in [0, 1)"))
    return sigma * sqrt(-2 * log(float(cutoff_tol)))
end

function _accumulate_rectangle_image!(img::AbstractMatrix{Float64},
                                      xs::AbstractVector{<:Real},
                                      ys::AbstractVector{<:Real},
                                      cx::Float64,
                                      cy::Float64,
                                      w::Float64,
                                      inv_two_sig2::Float64,
                                      cutoff_radius::Float64)
    w == 0.0 && return img
    if isfinite(cutoff_radius)
        ixlo = searchsortedfirst(xs, cx - cutoff_radius)
        ixhi = searchsortedlast(xs, cx + cutoff_radius)
        iylo = searchsortedfirst(ys, cy - cutoff_radius)
        iyhi = searchsortedlast(ys, cy + cutoff_radius)
    else
        ixlo, ixhi = 1, length(xs)
        iylo, iyhi = 1, length(ys)
    end
    (ixlo > ixhi || iylo > iyhi) && return img

    gx = Vector{Float64}(undef, ixhi - ixlo + 1)
    @inbounds for (j, ix) in enumerate(ixlo:ixhi)
        dx = float(xs[ix]) - cx
        gx[j] = exp(-(dx * dx) * inv_two_sig2)
    end
    @inbounds for iy in iylo:iyhi
        dy = float(ys[iy]) - cy
        wy = w * exp(-(dy * dy) * inv_two_sig2)
        for (j, ix) in enumerate(ixlo:ixhi)
            img[ix, iy] += wy * gx[j]
        end
    end
    return img
end

function rectangle_signed_barcode_image(sb::RectSignedBarcode{2};
    xs=nothing,
    ys=nothing,
    sigma::Real=1.0,
    mode::Symbol=:center,
    cutoff_tol::Real=0.0,
    threads::Bool = (Threads.nthreads() > 1)
)
    xs === nothing && (xs = sb.axes[1])
    ys === nothing && (ys = sb.axes[2])
    sigmaf = float(sigma)
    sigmaf > 0 || error("rectangle_signed_barcode_image: sigma must be > 0")
    inv_two_sig2 = 0.5 / (sigmaf * sigmaf)
    cutoff_radius = _rectangle_image_cutoff_radius(sigmaf, cutoff_tol)
    anchors = [_rectangle_image_anchor(rect, mode) for rect in sb.rects]
    weights = Float64[float(w) for w in sb.weights]

    img = zeros(Float64, length(xs), length(ys))
    if threads && Threads.nthreads() > 1 && !isempty(sb.rects)
        nrect = length(sb.rects)
        nchunks = min(nrect, Threads.nthreads())
        chunk_size = cld(nrect, nchunks)
        chunk_imgs = [zeros(Float64, length(xs), length(ys)) for _ in 1:nchunks]
        Threads.@threads for c in 1:nchunks
            lo = (c - 1) * chunk_size + 1
            hi = min(c * chunk_size, nrect)
            lo > hi && continue
            local_img = chunk_imgs[c]
            for i in lo:hi
                cx, cy = anchors[i]
                _accumulate_rectangle_image!(local_img, xs, ys, float(cx), float(cy), weights[i],
                                             inv_two_sig2, cutoff_radius)
            end
        end
        for local_img in chunk_imgs
            img .+= local_img
        end
    else
        for i in eachindex(sb.rects)
            cx, cy = anchors[i]
            _accumulate_rectangle_image!(img, xs, ys, float(cx), float(cy), weights[i],
                                         inv_two_sig2, cutoff_radius)
        end
    end
    return img
end

"""
    SignedMeasureDecomposition(; rectangles=nothing, slices=nothing,
                               euler_surface=nothing, euler_signed_measure=nothing,
                               mpp_image=nothing)

Typed result container for `mma_decomposition`.

This keeps the signed-measure workflow surface inspectable and consistent with
other user-visible result objects in the library. Use semantic accessors such as
[`rectangles`](@ref), [`slices`](@ref), [`euler_surface`](@ref),
[`euler_signed_measure`](@ref), and [`mpp_image`](@ref) instead of direct field
access when writing ordinary user code.
"""
struct SignedMeasureDecomposition{R,S,E,P,I}
    rectangles::R
    slices::S
    euler_surface::E
    euler_signed_measure::P
    mpp_image::I
end

SignedMeasureDecomposition(; rectangles=nothing,
                           slices=nothing,
                           euler_surface=nothing,
                           euler_signed_measure=nothing,
                           mpp_image=nothing) =
    SignedMeasureDecomposition(rectangles, slices, euler_surface, euler_signed_measure, mpp_image)

@inline rectangles(result::SignedMeasureDecomposition) = result.rectangles
@inline slices(result::SignedMeasureDecomposition) = result.slices
@inline euler_surface(result::SignedMeasureDecomposition) = result.euler_surface
@inline euler_signed_measure(result::SignedMeasureDecomposition) = result.euler_signed_measure
@inline mpp_image(result::SignedMeasureDecomposition) = result.mpp_image
@inline has_rectangles(result::SignedMeasureDecomposition) = result.rectangles !== nothing
@inline has_slices(result::SignedMeasureDecomposition) = result.slices !== nothing
@inline has_euler_surface(result::SignedMeasureDecomposition) = result.euler_surface !== nothing
@inline has_euler_signed_measure(result::SignedMeasureDecomposition) = result.euler_signed_measure !== nothing
@inline has_mpp_image(result::SignedMeasureDecomposition) = result.mpp_image !== nothing
@inline has_euler(result::SignedMeasureDecomposition) = has_euler_surface(result) || has_euler_signed_measure(result)
@inline has_image(result::SignedMeasureDecomposition) = has_mpp_image(result)
@inline nrectangles(result::SignedMeasureDecomposition) = has_rectangles(result) ? nterms(result.rectangles) : 0
@inline nslices(result::SignedMeasureDecomposition) = has_slices(result) && hasproperty(result.slices, :barcodes) ? length(result.slices.barcodes) : 0
@inline ncomponents(result::SignedMeasureDecomposition) =
    Int(has_rectangles(result)) + Int(has_slices(result)) + Int(has_euler_surface(result)) +
    Int(has_euler_signed_measure(result)) + Int(has_mpp_image(result))
@inline signed_measure_decomposition_summary(result::SignedMeasureDecomposition) = describe(result)

@doc """
    nrectangles(result)
    nslices(result)
    has_euler(result)
    has_image(result)
    ncomponents(result)
    isempty(result)

Cheap scalar exploration helpers for `SignedMeasureDecomposition`.
""" nrectangles

"""
    components(result)
    component_names(result)

Return the present components of a `SignedMeasureDecomposition` as a named
tuple, and the corresponding component names.

Examples
--------
```julia
out = mma_decomposition(M, pi; method=:all, ...)
component_names(out)
components(out).rectangles
```
"""
function components(result::SignedMeasureDecomposition)
    parts = Pair{Symbol,Any}[]
    has_rectangles(result) && push!(parts, :rectangles => result.rectangles)
    has_slices(result) && push!(parts, :slices => result.slices)
    has_euler_surface(result) && push!(parts, :euler_surface => result.euler_surface)
    has_euler_signed_measure(result) && push!(parts, :euler_signed_measure => result.euler_signed_measure)
    has_mpp_image(result) && push!(parts, :mpp_image => result.mpp_image)
    return (; parts...)
end

@inline component_names(result::SignedMeasureDecomposition) = Tuple(propertynames(components(result)))
Base.isempty(result::SignedMeasureDecomposition) = ncomponents(result) == 0

"""
    check_signed_measure_decomposition(result; throw=false)
    validate(result::SignedMeasureDecomposition; throw=true)

Validate a `SignedMeasureDecomposition`.

This checks each present component with its owner-level validator and also
checks cross-component consistency for Euler data when both `euler_surface` and
`euler_signed_measure` are present.
"""
function check_signed_measure_decomposition(result::SignedMeasureDecomposition; throw::Bool = false)
    errors = String[]

    if has_rectangles(result)
        rect_summary = check_rect_signed_barcode(result.rectangles; throw = false)
        append!(errors, ["rectangles: " * err for err in rect_summary.errors])
    end

    if has_euler_signed_measure(result)
        pm_summary = check_point_signed_measure(result.euler_signed_measure; throw = false)
        append!(errors, ["euler_signed_measure: " * err for err in pm_summary.errors])
    end

    if has_euler_surface(result) && has_euler_signed_measure(result)
        expected_size = Tuple(axis_lengths(result.euler_signed_measure))
        size(result.euler_surface) == expected_size ||
            push!(errors, "euler_surface size must match Euler signed-measure axes")
        surface_from_point_signed_measure(result.euler_signed_measure) == result.euler_surface ||
            push!(errors, "euler_surface is inconsistent with euler_signed_measure")
    end

    if has_slices(result)
        hasproperty(result.slices, :barcodes) || push!(errors, "slices component must expose :barcodes")
        hasproperty(result.slices, :weights) || push!(errors, "slices component must expose :weights")
    end

    if has_mpp_image(result)
        for sym in (:xgrid, :ygrid, :img, :sigma)
            hasproperty(result.mpp_image, sym) || push!(errors, "mpp_image component must expose :$sym")
        end
    end

    return _validation_result(SignedMeasureDecompositionValidationSummary, errors; throw = throw)
end

function validate(result::SignedMeasureDecomposition; throw::Bool = true)
    return check_signed_measure_decomposition(result; throw = throw)
end

function describe(result::SignedMeasureDecomposition)
    return (
        kind = :signed_measure_decomposition,
        has_rectangles = has_rectangles(result),
        has_slices = has_slices(result),
        has_euler_surface = has_euler_surface(result),
        has_euler_signed_measure = has_euler_signed_measure(result),
        has_mpp_image = has_mpp_image(result),
        has_euler = has_euler(result),
        has_image = has_image(result),
        component_names = component_names(result),
        ncomponents = ncomponents(result),
        rectangle_terms = nrectangles(result),
        slice_shape = has_slices(result) && hasproperty(result.slices, :barcodes) ? size(result.slices.barcodes) : nothing,
        nslices = nslices(result),
        euler_surface_size = has_euler_surface(result) ? size(result.euler_surface) : nothing,
        mpp_resolution = has_mpp_image(result) ? (length(result.mpp_image.xgrid), length(result.mpp_image.ygrid)) : nothing,
    )
end

function Base.show(io::IO, result::SignedMeasureDecomposition)
    d = describe(result)
    parts = String[]
    d.has_rectangles && push!(parts, "rectangles")
    d.has_slices && push!(parts, "slices")
    d.has_euler_surface && push!(parts, "euler")
    d.has_mpp_image && push!(parts, "mpp_image")
    print(io, "SignedMeasureDecomposition(",
          isempty(parts) ? "empty" : join(parts, ", "),
          ")")
end

function Base.show(io::IO, ::MIME"text/plain", result::SignedMeasureDecomposition)
    d = describe(result)
    print(io,
          "SignedMeasureDecomposition",
          "\n  has_rectangles: ", d.has_rectangles,
          "\n  has_slices: ", d.has_slices,
          "\n  has_euler_surface: ", d.has_euler_surface,
          "\n  has_euler_signed_measure: ", d.has_euler_signed_measure,
          "\n  has_mpp_image: ", d.has_mpp_image,
          "\n  component_names: ", repr(d.component_names),
          "\n  ncomponents: ", d.ncomponents,
          "\n  rectangle_terms: ", d.rectangle_terms,
          "\n  slice_shape: ", repr(d.slice_shape),
          "\n  nslices: ", d.nslices,
          "\n  euler_surface_size: ", repr(d.euler_surface_size),
          "\n  mpp_resolution: ", repr(d.mpp_resolution))
end

"""
    mma_decomposition(M, pi, opts::InvariantOptions; method=:rectangles,
                      rect_kwargs=NamedTuple(), slice_kwargs=NamedTuple(), mpp_kwargs=NamedTuple(),
                      truncate=true, max_terms=nothing, min_abs_weight=1,
                      euler_drop_zeros=true, euler_max_terms=0, euler_min_abs_weight=0)

A small "MMA style" front-end.

Compute an MMA-style decomposition / summary for a module `M` encoded by `pi`.

This is the opts-primary API. Axis selection / strictness / box settings are taken from `opts`
(see `InvariantOptions`).

This is not a full MMA solver. It provides the core algebraic objects commonly used in
multi-parameter matching distance (MMA) pipelines:
  - rectangle signed measures (Mobius inversion of the rank invariant),
  - directional slice barcodes,
  - optional Euler characteristic objects on a grid, and
  - optional multiparameter persistence images (MPPI; Carriere et al.).

Methods

- `method=:rectangles` returns a `SignedMeasureDecomposition` whose `rectangles`
  component is a `RectSignedBarcode` obtained by Mobius inversion of the rank
  invariant, optionally truncated via `truncate_signed_barcode`.
- `method=:slices` returns a `SignedMeasureDecomposition` whose `slices`
  component is the object produced by `slice_barcodes(M, pi; ...)`.
  Note: `slice_barcodes` requires `directions` and `offsets`; pass them via `slice_kwargs`.
- `method=:both` returns a `SignedMeasureDecomposition` with `rectangles` and `slices`.
- `method=:euler` returns a `SignedMeasureDecomposition` with
  `euler_surface` and `euler_signed_measure`.
- `method=:mpp_image` returns a `SignedMeasureDecomposition` with an
  `mpp_image` component carrying the Carriere multiparameter persistence image.
  Note: this requires a 2-parameter (2D) encoding and an exact-field module.
- `method=:all` returns a `SignedMeasureDecomposition` carrying all available components.

Keyword argument routing

- `rect_kwargs` is forwarded to `rectangle_signed_barcode(M, pi; ...)`.
- `slice_kwargs` is forwarded to `slice_barcodes(M, pi; ...)`.
- `mpp_kwargs` is forwarded to `mpp_image(M, pi; ...)` (only used for `method=:mpp_image`).

Euler-related keywords

The Euler outputs are computed on a rectangular grid controlled by `(axes, axes_policy, max_axis_len)`.
Truncation of the Euler signed measure is controlled separately via the `euler_*` keywords.

If you want Euler output for an object that is not a `PModule` or for an encoding that is not
a `ZnEncodingMap`, use the generic method

    mma_decomposition(obj, pi; method=:euler, axes=..., axes_policy=..., max_axis_len=...)

which returns only the Euler outputs.
"""
function mma_decomposition(M::PModule, pi::ZnEncodingMap, opts::InvariantOptions;
    method::Symbol=:rectangles,
    rect_kwargs::NamedTuple=NamedTuple(),
    slice_kwargs::NamedTuple=NamedTuple(),
    mpp_kwargs::NamedTuple=NamedTuple(),
    cache=:auto,
    truncate::Bool=true,
    max_terms=nothing,
    min_abs_weight::Int=1,
    euler_drop_zeros::Bool=true,
    euler_max_terms::Int=0,
    euler_min_abs_weight::Real=0)

    meths = Set([:rectangles, :slices, :both, :euler, :all, :mpp_image])
    method in meths || throw(ArgumentError("mma_decomposition(M,pi,opts): unsupported method=$(method)"))

    if method == :mpp_image
        return SignedMeasureDecomposition(; mpp_image = _multiparameter_images_module().mpp_image(M, pi; mpp_kwargs...))
    end

    session_cache = _signed_measures_session_cache(cache)
    rects = nothing
    slices = nothing
    surf = nothing
    pm = nothing

    if method == :rectangles || method == :both || method == :all
        sb = rectangle_signed_barcode(M, pi, opts; cache=session_cache, rect_kwargs...)
        if truncate
            sb = truncate_signed_barcode(sb; max_terms=max_terms, min_abs_weight=min_abs_weight)
        end
        rects = sb
    end

    if method == :slices || method == :both || method == :all
        slices = _slice_invariants_module().slice_barcodes(M, pi, opts; slice_kwargs...)
    end

    if method == :euler || method == :all
        surf, ax = _cached_euler_characteristic_surface(M, pi, opts, session_cache)
        pm = _euler_signed_measure_from_surface(surf, ax;
            drop_zeros=euler_drop_zeros,
            max_terms=euler_max_terms,
            min_abs_weight=euler_min_abs_weight,
        )
        session_cache === nothing || (surf = copy(surf))
    end

    if method == :rectangles
        return SignedMeasureDecomposition(; rectangles = rects)
    elseif method == :slices
        return SignedMeasureDecomposition(; slices = slices)
    elseif method == :both
        return SignedMeasureDecomposition(; rectangles = rects, slices = slices)
    elseif method == :euler
        return SignedMeasureDecomposition(; euler_surface = surf, euler_signed_measure = pm)
    else
        return SignedMeasureDecomposition(;
            rectangles = rects,
            slices = slices,
            euler_surface = surf,
            euler_signed_measure = pm,
        )
    end
end

mma_decomposition(M::PModule, pi::ZnEncodingMap; kwargs...) =
    mma_decomposition(M, pi, InvariantOptions(); kwargs...)

function mma_decomposition(M::PModule, pi::CompiledEncoding{<:ZnEncodingMap}, opts::InvariantOptions;
                           kwargs...)
    return mma_decomposition(M, pi.pi, opts; kwargs...)
end

mma_decomposition(M::PModule, pi::CompiledEncoding{<:ZnEncodingMap}; kwargs...) =
    mma_decomposition(M, pi.pi; kwargs...)

"""
    mma_decomposition(M, pi, opts::InvariantOptions; method=:euler, mpp_kwargs=NamedTuple(), ...)


A lightweight wrapper for Euler and MPPI output when `pi` is not a `ZnEncodingMap`.

This method exists so that you can write

    mma_decomposition(M, pi; method=:mpp_image, mpp_kwargs=(...))

for 2D encodings coming from backends such as `PLBackend`.

Supported methods

- `method=:euler` returns a `SignedMeasureDecomposition` with
  `euler_surface` and `euler_signed_measure`.
- `method=:mpp_image` returns a `SignedMeasureDecomposition` whose `mpp_image`
  component is filled via `mpp_image(M, pi; mpp_kwargs...)`.
  Note: this requires a 2-parameter (2D) encoding and a `PModule{K}`.

Other `method` values are not supported here (use the `ZnEncodingMap` method if you
need rectangles or slice barcodes).
"""
function mma_decomposition(M::PModule, pi, opts::InvariantOptions;
    method::Symbol=:euler,
    mpp_kwargs::NamedTuple=NamedTuple(),
    cache=:auto,
    euler_drop_zeros::Bool=true,
    euler_max_terms::Int=0,
    euler_min_abs_weight::Real=0)

    method in Set([:euler, :mpp_image]) ||
        throw(ArgumentError("mma_decomposition(M,pi,opts): supported methods are :euler and :mpp_image for this signature"))

    if method == :mpp_image
        return SignedMeasureDecomposition(; mpp_image = _multiparameter_images_module().mpp_image(M, pi; mpp_kwargs...))
    end

    session_cache = _signed_measures_session_cache(cache)
    surf, ax = _cached_euler_characteristic_surface(M, pi, opts, session_cache)
    pm = _euler_signed_measure_from_surface(surf, ax;
        drop_zeros=euler_drop_zeros,
        max_terms=euler_max_terms,
        min_abs_weight=euler_min_abs_weight,
    )
    session_cache === nothing || (surf = copy(surf))
    return SignedMeasureDecomposition(; euler_surface = surf, euler_signed_measure = pm)
end

mma_decomposition(M::PModule, pi; opts::InvariantOptions=InvariantOptions(), kwargs...) =
    mma_decomposition(M, pi, opts; kwargs...)


"""
    mma_decomposition(obj, pi, opts::InvariantOptions; method=:euler, ...)

Euler-only front-end for arbitrary objects supporting `restricted_hilbert(obj; pi=pi)`.

This method is useful for objects such as `ModuleCochainComplex` (Euler surface = alternating sum of
chain-group dimensions) or for encodings `pi` that are not `ZnEncodingMap`s.

Only `method=:euler` is supported for this signature, and the result is a
`SignedMeasureDecomposition` with Euler components filled.
"""
function mma_decomposition(obj, pi, opts::InvariantOptions;
    method::Symbol=:euler,
    cache=:auto,
    euler_drop_zeros::Bool=true,
    euler_max_terms::Int=0,
    euler_min_abs_weight::Real=0)

    method === :euler ||
        throw(ArgumentError("mma_decomposition(obj,pi,opts): only method=:euler is supported for this signature"))

    session_cache = _signed_measures_session_cache(cache)
    surf, ax = _cached_euler_characteristic_surface(obj, pi, opts, session_cache)
    pm = _euler_signed_measure_from_surface(surf, ax;
        drop_zeros=euler_drop_zeros,
        max_terms=euler_max_terms,
        min_abs_weight=euler_min_abs_weight,
    )
    session_cache === nothing || (surf = copy(surf))
    return SignedMeasureDecomposition(; euler_surface = surf, euler_signed_measure = pm)
end

end # module SignedMeasures
