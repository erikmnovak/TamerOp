module MultiparameterImages
# -----------------------------------------------------------------------------
# MultiparameterImages.jl
#
# MultiparameterImages owner module extracted from Invariants.jl.
# -----------------------------------------------------------------------------

"""
    MultiparameterImages

Owner module for multiparameter persistence images and landscapes built from
finite families of 1D slice barcodes.
"""

using LinearAlgebra
using JSON3
using ..CoreModules: EncodingCache, AbstractCoeffField, RegionPosetCachePayload,
                     AbstractSlicePlanCache
using ..Options: InvariantOptions
import ..DataTypes: bounding_box
using ..EncodingCore: PLikeEncodingMap, CompiledEncoding, locate, axes_from_encoding, dimension, representatives
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
                       orthant_directions, _normalize_dir,
                       _selection_kwargs_from_opts, _axes_kwargs_from_opts,
                       _eye, rank_map,
                       RANK_INVARIANT_MEMO_THRESHOLD, RECTANGLE_LOC_LINEAR_CACHE_THRESHOLD,
                       _use_array_memo, _new_array_memo, _grid_cache_index,
                       _memo_get, _memo_set!, _map_leq_cached,
                       _rank_cache_get!, _resolve_rank_query_cache,
                       _rank_query_point_tuple, _rank_query_locate!
import ..ChainComplexes: describe
import ..FiniteFringe: AbstractPoset, FinitePoset, FringeModule, Upset, Downset, fiber_dimension,
                       leq, leq_matrix, upset_indices, downset_indices, leq_col, nvertices, build_cache!,
                       _preds
import ..ZnEncoding
import ..IndicatorTypes: UpsetPresentation, DownsetCopresentation
import ..ModuleComplexes: ModuleCochainComplex
import ..ChangeOfPosets: pushforward_left, pushforward_right
import Base.Threads
import ..Serialization: save_mpp_decomposition_json, load_mpp_decomposition_json,
                        save_mpp_image_json, load_mpp_image_json
import ..Modules: PModule, map_leq, CoverCache, _get_cover_cache
import ..IndicatorResolutions: pmodule_from_fringe
import ..ZnEncoding: ZnEncodingMap
import ..SliceInvariants: collect_slices, CompiledSlicePlan, slice_barcode, _slice_barcode_packed,
                          _slice_barcode_packed_with_workspace, _extended_values_view,
                          _persistence_landscape_values!,
                          _clean_tgrid, _plan_idx, compile_slices, default_offsets, uniform2d,
                          encoding_box,
                          landscape_grid, landscape_values, landscape_layers, feature_dimension,
                          image_xgrid, image_ygrid, image_values, image_shape,
                          slice_weights, slice_directions, slice_offsets
import ..Fibered2D: FiberedArrangement2D, FiberedBarcodeCache2D,
                    fibered_arrangement_2d, fibered_barcode_cache_2d,
                    fibered_barcode, fibered_cell_id,
                    _unique_points_2d!, _line_basepoint_from_normal_offset_2d,
                    _normal_from_dir_2d, _as_float2

# -----------------------------------------------------------------------------
# ----- Multiparameter persistence images (Carriere et al. construction) -------
# -----------------------------------------------------------------------------
#
# This section implements the "Multiparameter Persistence Image" (MPPI)
# construction described by Carriere et al.
#
# The goal is NOT "just take 1D persistence images on slices".
# Instead, the construction:
#   1) chooses specific families of slices (lines) in the parameter plane,
#   2) computes fibered barcodes along these slices,
#   3) connects bars across neighboring slices by optimal matchings ("vineyards"),
#   4) treats each connected track as a summand I,
#   5) weights each summand by a geometric hull weight w(I),
#   6) produces an image by evaluating the Carriere kernel
#        k(x, I) = omega(l*) * exp(-d(x,I)^2 / sigma^2)
#      where l* is the slice containing the closest segment in I.
#
# This produces a stable, low-dimensional representation that explicitly
# accounts for multiparameter coherence across slices.

"""
    MPPLineSpec

A single slice line for the Carriere MPPI construction.

This is the advanced line object stored inside an [`MPPDecomposition`](@ref).
In the usual workflow you inspect it rather than constructing it yourself:

```julia
decomp = mpp_decomposition(M, pi; N=8)
line = first(line_specs(decomp))
describe(line)
mpp_line_spec_summary(line)
check_mpp_line_spec(line)
```

Fields
- `dir`: direction vector (Float64, length 2), normalized according to the
         arrangement's direction normalization convention (usually L1).
- `off`: normal offset (Float64), so the line is { x : n . x = off } where
         n is a unit normal associated to dir.
- `x0`: a basepoint on the line (Float64, length 2), compatible with (dir, off).
- `omega`: the Carriere direction weight omega(dir) = min(cos(theta), sin(theta))
           where theta is the direction angle in [0, pi/2]. Equivalently,
           if `u = dir / norm(dir,2)` then omega = min(u[1], u[2]).
"""
struct MPPLineSpec
    dir::Vector{Float64}
    off::Float64
    x0::Vector{Float64}
    omega::Float64
end

"""
    MPPLineSpecValidationSummary

Typed validation report returned by [`check_mpp_line_spec`](@ref).
"""
struct MPPLineSpecValidationSummary
    valid::Bool
    errors::Vector{String}
end

Base.:(==)(a::MPPLineSpec, b::MPPLineSpec) =
    a.dir == b.dir && a.off == b.off && a.x0 == b.x0 && a.omega == b.omega

Base.isequal(a::MPPLineSpec, b::MPPLineSpec) =
    isequal(a.dir, b.dir) &&
    isequal(a.off, b.off) &&
    isequal(a.x0, b.x0) &&
    isequal(a.omega, b.omega)

Base.hash(line::MPPLineSpec, h::UInt) = hash((line.dir, line.off, line.x0, line.omega), h)

@inline line_direction(line::MPPLineSpec) = line.dir
@inline line_offset(line::MPPLineSpec) = line.off
@inline line_basepoint(line::MPPLineSpec) = line.x0
@inline line_omega(line::MPPLineSpec) = line.omega

function describe(line::MPPLineSpec)
    d = line_direction(line)
    return (
        kind = :mpp_line_spec,
        direction = d,
        offset = line_offset(line),
        basepoint = line_basepoint(line),
        omega = line_omega(line),
        direction_norm_l1 = abs(d[1]) + abs(d[2]),
        direction_norm_l2 = sqrt(d[1]^2 + d[2]^2),
    )
end

"""
    mpp_line_spec_summary(line::MPPLineSpec)

Owner-local summary alias for [`describe(line)`](@ref).

Typical workflow:

```julia
line = first(line_specs(decomp))
mpp_line_spec_summary(line)
check_mpp_line_spec(line)
```
"""
@inline mpp_line_spec_summary(line::MPPLineSpec) = describe(line)

function Base.show(io::IO, line::MPPLineSpec)
    d = describe(line)
    print(io,
          "MPPLineSpec(",
          "dir=", repr(d.direction),
          ", off=", d.offset,
          ", omega=", d.omega,
          ")")
end

function Base.show(io::IO, ::MIME"text/plain", line::MPPLineSpec)
    d = describe(line)
    print(io,
          "MPPLineSpec",
          "\n  direction: ", repr(d.direction),
          "\n  offset: ", d.offset,
          "\n  basepoint: ", repr(d.basepoint),
          "\n  omega: ", d.omega,
          "\n  direction_norm_l1: ", d.direction_norm_l1,
          "\n  direction_norm_l2: ", d.direction_norm_l2)
end

"""
    check_mpp_line_spec(line; throw=false)

Validate a hand-built [`MPPLineSpec`](@ref).

Typical workflow:

```julia
line = first(line_specs(decomp))
check_mpp_line_spec(line)
```
"""
function check_mpp_line_spec(line::MPPLineSpec; throw::Bool=false)
    errors = String[]
    d = line_direction(line)
    x0 = line_basepoint(line)

    _mppi_is_finite_pair(d) || push!(errors, "direction must be a finite length-2 vector")
    _mppi_is_finite_pair(x0) || push!(errors, "basepoint must be a finite length-2 vector")
    if _mppi_is_finite_pair(d)
        abs(d[1]) + abs(d[2]) > 0.0 || push!(errors, "direction must be nonzero")
    end
    isfinite(line_offset(line)) || push!(errors, "offset must be finite")
    isfinite(line_omega(line)) || push!(errors, "omega must be finite")
    (0.0 <= line_omega(line) <= 1.0 + 1e-12) || push!(errors, "omega must lie in [0,1]")

    return _mppi_validation_result(MPPLineSpecValidationSummary, errors; throw=throw)
end

"""
    MPPDecomposition

A decomposition of a module into "summands" I_k as used by Carriere MPPI.

Each summand is represented as a list of segments in R^2, where each segment
lies on one of the chosen slice lines and corresponds to one barcode interval
(birth -> death mapped into the plane).

This is the canonical intermediate object for the multiparameter-image
workflow:

```julia
decomp = mpp_decomposition(M, pi; N=8, delta=:auto)
describe(decomp)
mpp_decomposition_summary(decomp)
check_mpp_decomposition(decomp)
img = mpp_image(decomp; resolution=24, sigma=0.2)
```

Fields
- `lines`: vector of `MPPLineSpec` used.
- `summands`: vector of summands; each summand is a vector of segments.
  A segment is stored as an `MPPSegment` with packed 2D endpoints `(p, q)` and
  direction weight `omega`.
- `weights`: w(I_k) geometric weights for each summand (Float64).
- `box`: the ambient bounding box (a,b) in R^2 used for normalization.
"""
struct MPPSegment
    p::NTuple{2,Float64}
    q::NTuple{2,Float64}
    omega::Float64
end

@inline function MPPSegment(p::NTuple{2,<:Real}, q::NTuple{2,<:Real}, omega::Real)
    return MPPSegment(
        (Float64(p[1]), Float64(p[2])),
        (Float64(q[1]), Float64(q[2])),
        Float64(omega),
    )
end

@inline function MPPSegment(p::AbstractVector{<:Real}, q::AbstractVector{<:Real}, omega::Real)
    length(p) == 2 || throw(ArgumentError("MPPSegment: p must have length 2"))
    length(q) == 2 || throw(ArgumentError("MPPSegment: q must have length 2"))
    return MPPSegment((Float64(p[1]), Float64(p[2])), (Float64(q[1]), Float64(q[2])), Float64(omega))
end

@inline Base.length(::MPPSegment) = 3

@inline function Base.iterate(seg::MPPSegment, state::Int=1)
    if state == 1
        return seg.p, 2
    elseif state == 2
        return seg.q, 3
    elseif state == 3
        return seg.omega, 4
    end
    return nothing
end

struct MPPDecomposition
    lines::Vector{MPPLineSpec}
    summands::Vector{Vector{MPPSegment}}
    weights::Vector{Float64}
    box::Tuple{Vector{Float64},Vector{Float64}}
end

function MPPDecomposition(lines::Vector{MPPLineSpec},
                          summands::AbstractVector,
                          weights::AbstractVector,
                          box)
    segs = Vector{Vector{MPPSegment}}(undef, length(summands))
    @inbounds for i in eachindex(summands)
        src = summands[i]
        dst = Vector{MPPSegment}(undef, length(src))
        for j in eachindex(src)
            seg = src[j]
            dst[j] = seg isa MPPSegment ? seg : MPPSegment(seg[1], seg[2], seg[3])
        end
        segs[i] = dst
    end
    lo, hi = box
    return MPPDecomposition(lines, segs, Float64.(collect(weights)), (Float64.(collect(lo)), Float64.(collect(hi))))
end

"""
    MPPImage

A Carriere multiparameter persistence image.

This is the canonical image object returned by [`mpp_image`](@ref). The cheap
workflow is:

```julia
img = mpp_image(M, pi; resolution=24, sigma=0.2, N=8)
describe(img)
mpp_image_summary(img)
check_mpp_image(img)
```

Use [`mpp_image_distance`](@ref) and [`mpp_image_kernel`](@ref) for direct
comparison once the image has been built.

Fields
- `xgrid`, `ygrid`: coordinate grids for evaluation.
- `img`: matrix of values, size (length(ygrid), length(xgrid)).
- `sigma`: Gaussian scale used in kernel exp(-d^2/sigma^2).
- `decomp`: the underlying `MPPDecomposition` (stored for reproducibility).
"""
struct MPPImage
    xgrid::Vector{Float64}
    ygrid::Vector{Float64}
    img::Matrix{Float64}
    sigma::Float64
    decomp::MPPDecomposition
end

"""
    MPPDecompositionValidationSummary

Typed validation report returned by [`check_mpp_decomposition`](@ref).
"""
struct MPPDecompositionValidationSummary
    valid::Bool
    errors::Vector{String}
end

"""
    MPPImageValidationSummary

Typed validation report returned by [`check_mpp_image`](@ref).
"""
struct MPPImageValidationSummary
    valid::Bool
    errors::Vector{String}
end

"""
    MPLandscapeValidationSummary

Typed validation report returned by [`check_mp_landscape`](@ref).
"""
struct MPLandscapeValidationSummary
    valid::Bool
    errors::Vector{String}
end

@inline _mppi_issue_count(summary) = length(summary.errors)

function Base.show(io::IO, summary::MPPLineSpecValidationSummary)
    print(io, "MPPLineSpecValidationSummary(valid=", summary.valid,
          ", errors=", _mppi_issue_count(summary), ")")
end

function Base.show(io::IO, summary::MPPDecompositionValidationSummary)
    print(io, "MPPDecompositionValidationSummary(valid=", summary.valid,
          ", errors=", _mppi_issue_count(summary), ")")
end

function Base.show(io::IO, summary::MPPImageValidationSummary)
    print(io, "MPPImageValidationSummary(valid=", summary.valid,
          ", errors=", _mppi_issue_count(summary), ")")
end

function Base.show(io::IO, summary::MPLandscapeValidationSummary)
    print(io, "MPLandscapeValidationSummary(valid=", summary.valid,
          ", errors=", _mppi_issue_count(summary), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::MPPDecompositionValidationSummary)
    print(io,
          "MPPDecompositionValidationSummary",
          "\n  valid: ", summary.valid,
          "\n  errors: ", repr(summary.errors))
end

function Base.show(io::IO, ::MIME"text/plain", summary::MPPLineSpecValidationSummary)
    print(io,
          "MPPLineSpecValidationSummary",
          "\n  valid: ", summary.valid,
          "\n  errors: ", repr(summary.errors))
end

function Base.show(io::IO, ::MIME"text/plain", summary::MPPImageValidationSummary)
    print(io,
          "MPPImageValidationSummary",
          "\n  valid: ", summary.valid,
          "\n  errors: ", repr(summary.errors))
end

function Base.show(io::IO, ::MIME"text/plain", summary::MPLandscapeValidationSummary)
    print(io,
          "MPLandscapeValidationSummary",
          "\n  valid: ", summary.valid,
          "\n  errors: ", repr(summary.errors))
end

@inline function _mppi_validation_result(::Type{S}, errors::Vector{String}; throw::Bool=false) where {S}
    if throw && !isempty(errors)
        Base.throw(ArgumentError(join(errors, "; ")))
    end
    return S(isempty(errors), errors)
end

@inline _mppi_is_finite_pair(v) = length(v) == 2 && isfinite(v[1]) && isfinite(v[2])

@inline function _mppi_is_strictly_increasing(v::AbstractVector{<:Real})
    length(v) >= 2 || return false
    @inbounds for i in 1:length(v)-1
        v[i] < v[i + 1] || return false
    end
    return true
end

@inline function _append_prefixed_errors!(errors::Vector{String}, prefix::AbstractString, report)
    for err in report.errors
        push!(errors, prefix * err)
    end
    return errors
end

@inline nlines(decomp::MPPDecomposition)::Int = length(decomp.lines)
@inline nsummands(decomp::MPPDecomposition)::Int = length(decomp.summands)
@inline line_specs(decomp::MPPDecomposition) = decomp.lines
@inline summand_weights(decomp::MPPDecomposition) = decomp.weights
@inline summand_segments(decomp::MPPDecomposition, k::Integer) = decomp.summands[Int(k)]
@inline total_segments(decomp::MPPDecomposition)::Int = sum(length, decomp.summands)
@inline weight_sum(decomp::MPPDecomposition)::Float64 = sum(decomp.weights)
@inline bounding_box(decomp::MPPDecomposition) = decomp.box

@inline image_xgrid(img::MPPImage) = img.xgrid
@inline image_ygrid(img::MPPImage) = img.ygrid
@inline image_values(img::MPPImage) = img.img
@inline image_shape(img::MPPImage) = size(img.img)
@inline decomposition(img::MPPImage) = img.decomp

function describe(decomp::MPPDecomposition)
    w = summand_weights(decomp)
    return (
        kind = :mpp_decomposition,
        nlines = nlines(decomp),
        nsummands = nsummands(decomp),
        total_segments = total_segments(decomp),
        weight_sum = weight_sum(decomp),
        weight_range = isempty(w) ? nothing : (minimum(w), maximum(w)),
        box = bounding_box(decomp),
    )
end

"""
    mpp_decomposition_summary(decomp::MPPDecomposition)

Owner-local summary alias for [`describe(decomp)`](@ref).

Typical workflow:

```julia
decomp = mpp_decomposition(M, pi; N=8)
mpp_decomposition_summary(decomp)
check_mpp_decomposition(decomp)
```
"""
@inline mpp_decomposition_summary(decomp::MPPDecomposition) = describe(decomp)

function Base.show(io::IO, decomp::MPPDecomposition)
    d = describe(decomp)
    print(io,
          "MPPDecomposition(",
          "nsummands=", d.nsummands,
          ", nlines=", d.nlines,
          ", total_segments=", d.total_segments,
          ")")
end

function Base.show(io::IO, ::MIME"text/plain", decomp::MPPDecomposition)
    d = describe(decomp)
    print(io,
          "MPPDecomposition",
          "\n  nsummands: ", d.nsummands,
          "\n  nlines: ", d.nlines,
          "\n  total_segments: ", d.total_segments,
          "\n  weight_sum: ", d.weight_sum,
          "\n  weight_range: ", repr(d.weight_range),
          "\n  box: ", repr(d.box))
end

function describe(img::MPPImage)
    xg = image_xgrid(img)
    yg = image_ygrid(img)
    return (
        kind = :mpp_image,
        image_shape = image_shape(img),
        sigma = img.sigma,
        x_range = isempty(xg) ? nothing : (xg[1], xg[end]),
        y_range = isempty(yg) ? nothing : (yg[1], yg[end]),
        nsummands = nsummands(decomposition(img)),
        nlines = nlines(decomposition(img)),
        weight_sum = weight_sum(decomposition(img)),
    )
end

"""
    mpp_image_summary(img::MPPImage)

Owner-local summary alias for [`describe(img)`](@ref).

Typical workflow:

```julia
img = mpp_image(M, pi; resolution=24, sigma=0.2, N=8)
mpp_image_summary(img)
check_mpp_image(img)
```
"""
@inline mpp_image_summary(img::MPPImage) = describe(img)

function Base.show(io::IO, img::MPPImage)
    d = describe(img)
    print(io,
          "MPPImage(",
          "shape=", d.image_shape,
          ", sigma=", d.sigma,
          ", nsummands=", d.nsummands,
          ")")
end

function Base.show(io::IO, ::MIME"text/plain", img::MPPImage)
    d = describe(img)
    print(io,
          "MPPImage",
          "\n  image_shape: ", repr(d.image_shape),
          "\n  sigma: ", d.sigma,
          "\n  x_range: ", repr(d.x_range),
          "\n  y_range: ", repr(d.y_range),
          "\n  nsummands: ", d.nsummands,
          "\n  nlines: ", d.nlines,
          "\n  weight_sum: ", d.weight_sum)
end

# ------------------------- internal geometry helpers --------------------------

# Squared Euclidean distance from point (x,y) to segment with endpoints (p,q).
#
# Important for performance:
# The MPPI image evaluation loops over many grid points, so we provide a scalar
# version to avoid allocations in the innermost loops.
@inline function _dist2_point_segment_xy(x::Float64, y::Float64,
                                        px::Float64, py::Float64,
                                        qx::Float64, qy::Float64)::Float64
    vx = qx - px
    vy = qy - py
    wx = x - px
    wy = y - py
    vv = vx*vx + vy*vy
    if vv == 0.0
        dx = x - px
        dy = y - py
        return dx*dx + dy*dy
    end
    t = (wx*vx + wy*vy) / vv
    if t <= 0.0
        dx = x - px
        dy = y - py
        return dx*dx + dy*dy
    elseif t >= 1.0
        dx = x - qx
        dy = y - qy
        return dx*dx + dy*dy
    else
        projx = px + t*vx
        projy = py + t*vy
        dx = x - projx
        dy = y - projy
        return dx*dx + dy*dy
    end
end

# Squared distance from point (x,y) to an axis-aligned bounding box.
# Returns 0.0 if (x,y) lies inside the box.
@inline function _bbox_dist2_xy(x::Float64, y::Float64,
                               xmin::Float64, xmax::Float64,
                               ymin::Float64, ymax::Float64)::Float64
    dx = 0.0
    if x < xmin
        dx = xmin - x
    elseif x > xmax
        dx = x - xmax
    end
    dy = 0.0
    if y < ymin
        dy = ymin - y
    elseif y > ymax
        dy = y - ymax
    end
    return dx*dx + dy*dy
end


# Convex hull (monotone chain) and area for weight w(I).
@inline _cross2(o::NTuple{2,Float64}, a::NTuple{2,Float64}, b::NTuple{2,Float64}) =
    (a[1]-o[1])*(b[2]-o[2]) - (a[2]-o[2])*(b[1]-o[1])

function _convex_hull_2d(pts::Vector{NTuple{2,Float64}}; atol::Float64=0.0)
    n = length(pts)
    n <= 1 && return pts
    sort!(pts, by=p->(p[1],p[2]))
    _unique_points_2d!(pts; atol=atol)
    length(pts) <= 1 && return pts

    lower = NTuple{2,Float64}[]
    for p in pts
        while length(lower) >= 2 &&
              _cross2(lower[end-1], lower[end], p) <= atol
            pop!(lower)
        end
        push!(lower, p)
    end

    upper = NTuple{2,Float64}[]
    for p in reverse(pts)
        while length(upper) >= 2 &&
              _cross2(upper[end-1], upper[end], p) <= atol
            pop!(upper)
        end
        push!(upper, p)
    end

    return vcat(lower[1:end-1], upper[1:end-1])
end

function _polygon_area_2d(poly::Vector{NTuple{2,Float64}})::Float64
    n = length(poly)
    n < 3 && return 0.0
    s = 0.0
    @inbounds for i in 1:n
        x1,y1 = poly[i]
        x2,y2 = poly[i == n ? 1 : i+1]
        s += x1*y2 - y1*x2
    end
    return 0.5 * abs(s)
end

# Carriere omega(dir). Uses unit L2 direction.
@inline function _omega_from_dir(d::Vector{Float64})::Float64
    nrm = sqrt(d[1]*d[1] + d[2]*d[2])
    nrm == 0.0 && return 0.0
    u1 = d[1]/nrm
    u2 = d[2]/nrm
    return min(u1, u2)
end

# Convert a barcode dict to an explicit multiset of (b,d) points.
function _barcode_points(bc::Dict{Tuple{Float64,Float64},Int})
    pts = Tuple{Float64,Float64}[]
    for (k,m) in bc
        for _ in 1:m
            push!(pts, k)
        end
    end
    return pts
end

@inline function _barcode_point_count(bc::Dict{Tuple{Float64,Float64},Int})::Int
    n = 0
    for m in values(bc)
        n += m
    end
    return n
end

function _fill_barcode_points!(dest::Vector{Tuple{Float64,Float64}},
                               start::Int,
                               bc::Dict{Tuple{Float64,Float64},Int})::Int
    idx = start
    for (k, m) in bc
        for _ in 1:m
            dest[idx] = k
            idx += 1
        end
    end
    return idx
end

# Bottleneck matching between two multisets of points.
# Returns a vector `match` of length length(A) with entries in 1:length(B) or 0 (diagonal).
#
# We reuse the internal 1D bottleneck machinery already present in this file:
# `bottleneck_distance(barA, barB)` exists, but it does not expose matching.
# Therefore, we implement a small exact bipartite matching specialized to the
# Carriere construction where sizes are typically modest.
#
# This is O(n^3) Hungarian-style on the max metric, but n is usually small
# (number of bars per slice).
function _bottleneck_matching_points(A::AbstractVector{<:Tuple{Float64,Float64}},
                                    B::AbstractVector{<:Tuple{Float64,Float64}})
    n = length(A)
    m = length(B)
    # Cost to diagonal for a point (b,d): half persistence in L_infty.
    diagcost(p) = 0.5*abs(p[2]-p[1])

    # Build square cost matrix by padding with diagonal nodes.
    N = max(n,m)
    C = fill(0.0, N, N)
    for i in 1:N
        for j in 1:N
            if i <= n && j <= m
                # L_infty distance between points.
                C[i,j] = max(abs(A[i][1]-B[j][1]), abs(A[i][2]-B[j][2]))
            elseif i <= n
                C[i,j] = diagcost(A[i])
            elseif j <= m
                C[i,j] = diagcost(B[j])
            else
                C[i,j] = 0.0
            end
        end
    end

    # Find minimal bottleneck threshold t such that a perfect matching exists.
    vals = unique(vec(C))
    sort!(vals)
    function feasible(t)
        # Build adjacency: i connects j if C[i,j] <= t.
        adj = [Int[] for _ in 1:N]
        for i in 1:N
            for j in 1:N
                C[i,j] <= t && push!(adj[i], j)
            end
        end
        # Standard bipartite matching (DFS augment).
        matchR = fill(0, N)
        function dfs(u, seen)
            for v in adj[u]
                seen[v] && continue
                seen[v] = true
                if matchR[v] == 0 || dfs(matchR[v], seen)
                    matchR[v] = u
                    return true
                end
            end
            return false
        end
        for u in 1:N
            seen = falses(N)
            dfs(u, seen) || return false
        end
        return true
    end

    lo = 1
    hi = length(vals)
    while lo < hi
        mid = (lo+hi) >>> 1
        feasible(vals[mid]) ? (hi = mid) : (lo = mid+1)
    end
    thr = vals[lo]

    # Extract one matching at threshold thr.
    adj = [Int[] for _ in 1:N]
    for i in 1:N
        for j in 1:N
            C[i,j] <= thr && push!(adj[i], j)
        end
    end
    matchR = fill(0, N)
    function dfs(u, seen)
        for v in adj[u]
            seen[v] && continue
            seen[v] = true
            if matchR[v] == 0 || dfs(matchR[v], seen)
                matchR[v] = u
                return true
            end
        end
        return false
    end
    for u in 1:N
        seen = falses(N)
        dfs(u, seen) || error("internal error: expected feasible matching")
    end

    # Convert matchingR (right->left) into left->right for original sizes.
    matchL = fill(0, n)
    for j in 1:N
        i = matchR[j]
        if i >= 1 && i <= n && j >= 1 && j <= m
            matchL[i] = j
        end
    end
    return matchL
end

function _bottleneck_matching_points_flat(pool::Vector{Tuple{Float64,Float64}},
                                          startA::Int,
                                          countA::Int,
                                          startB::Int,
                                          countB::Int)
    countA >= 0 || throw(ArgumentError("_bottleneck_matching_points_flat: countA must be nonnegative"))
    countB >= 0 || throw(ArgumentError("_bottleneck_matching_points_flat: countB must be nonnegative"))
    countA == 0 && return Int[]
    (1 <= startA <= length(pool) + 1) || throw(ArgumentError("_bottleneck_matching_points_flat: startA out of range"))
    (1 <= startB <= length(pool) + 1) || throw(ArgumentError("_bottleneck_matching_points_flat: startB out of range"))
    startA + countA - 1 <= length(pool) || throw(ArgumentError("_bottleneck_matching_points_flat: A range out of bounds"))
    startB + countB - 1 <= length(pool) || throw(ArgumentError("_bottleneck_matching_points_flat: B range out of bounds"))

    diagcost(b::Float64, d::Float64) = 0.5 * abs(d - b)

    n = countA
    m = countB
    N = max(n, m)
    C = fill(0.0, N, N)
    @inbounds for i in 1:N
        ai = startA + i - 1
        if i <= n
            Ab = pool[ai][1]
            Ad = pool[ai][2]
            for j in 1:N
                if j <= m
                    bj = startB + j - 1
                    Bb = pool[bj][1]
                    Bd = pool[bj][2]
                    C[i, j] = max(abs(Ab - Bb), abs(Ad - Bd))
                else
                    C[i, j] = diagcost(Ab, Ad)
                end
            end
        else
            for j in 1:N
                if j <= m
                    bj = startB + j - 1
                    Bb = pool[bj][1]
                    Bd = pool[bj][2]
                    C[i, j] = diagcost(Bb, Bd)
                else
                    C[i, j] = 0.0
                end
            end
        end
    end

    vals = unique(vec(C))
    sort!(vals)
    function feasible(t)
        adj = [Int[] for _ in 1:N]
        @inbounds for i in 1:N
            for j in 1:N
                C[i, j] <= t && push!(adj[i], j)
            end
        end
        matchR = fill(0, N)
        function dfs(u, seen)
            for v in adj[u]
                seen[v] && continue
                seen[v] = true
                if matchR[v] == 0 || dfs(matchR[v], seen)
                    matchR[v] = u
                    return true
                end
            end
            return false
        end
        for u in 1:N
            seen = falses(N)
            dfs(u, seen) || return false
        end
        return true
    end

    lo = 1
    hi = length(vals)
    while lo < hi
        mid = (lo + hi) >>> 1
        feasible(vals[mid]) ? (hi = mid) : (lo = mid + 1)
    end
    thr = vals[lo]

    adj = [Int[] for _ in 1:N]
    @inbounds for i in 1:N
        for j in 1:N
            C[i, j] <= thr && push!(adj[i], j)
        end
    end
    matchR = fill(0, N)
    function dfs(u, seen)
        for v in adj[u]
            seen[v] && continue
            seen[v] = true
            if matchR[v] == 0 || dfs(matchR[v], seen)
                matchR[v] = u
                return true
            end
        end
        return false
    end
    for u in 1:N
        seen = falses(N)
        dfs(u, seen) || error("internal error: expected feasible matching")
    end

    matchL = fill(0, n)
    @inbounds for j in 1:N
        i = matchR[j]
        if i >= 1 && i <= n && j >= 1 && j <= m
            matchL[i] = j
        end
    end
    return matchL
end

# -------------------- line families L_m^N, L_M^N, L_delta ---------------------

function _box_corners_2d(box::Tuple{AbstractVector{<:Real},AbstractVector{<:Real}})
    a_in,b_in = box
    a = Float64[float(a_in[1]), float(a_in[2])]
    b = Float64[float(b_in[1]), float(b_in[2])]
    lo = Float64[min(a[1],b[1]), min(a[2],b[2])]
    hi = Float64[max(a[1],b[1]), max(a[2],b[2])]
    return lo, hi
end

function _make_line_spec(arr::FiberedArrangement2D,
                         dir_in::AbstractVector{<:Real},
                         off_in::Real;
                         tie_break::Symbol=:center)
    d = _normalize_dir(_as_float2(dir_in), arr.normalize_dirs)
    off = Float64(off_in)
    # Nudge offsets away from boundaries if needed.
    cid = fibered_cell_id(arr, d, off; tie_break=tie_break)
    if cid === nothing
        eps = max(10.0*arr.atol, 1e-12)
        cid2 = fibered_cell_id(arr, d, off+eps; tie_break=tie_break)
        if cid2 !== nothing
            off += eps
        else
            cid2 = fibered_cell_id(arr, d, off-eps; tie_break=tie_break)
            if cid2 !== nothing
                off -= eps
            end
        end
    end
    x0 = _line_basepoint_from_normal_offset_2d(d, off)
    omega = _omega_from_dir(d)
    return MPPLineSpec(d, off, x0, omega)
end

function _line_families_carriere(arr::FiberedArrangement2D;
                                 N::Int=16,
                                 delta::Union{Real,Symbol}=:auto,
                                 tie_break::Symbol=:center)
    N > 1 || error("mpp_image: N must be >= 2")
    lo, hi = _box_corners_2d(arr.box)
    mpt = lo
    Mpt = hi

    # Directions in (0,pi/2). We avoid axes by default; axes contributions are
    # automatically damped since omega=0 there anyway.
    dirs = Vector{Vector{Float64}}()
    for i in 1:(N-1)
        th = (i*pi) / (2.0*N)
        push!(dirs, Float64[cos(th), sin(th)])
    end

    lines = MPPLineSpec[]

    # L_m^N: lines through bottom-left corner m.
    for d in dirs
        (n1,n2) = _normal_from_dir_2d(d)
        off = n1*mpt[1] + n2*mpt[2]
        push!(lines, _make_line_spec(arr, d, off; tie_break=tie_break))
    end

    # L_M^N: lines through top-right corner M.
    for d in dirs
        (n1,n2) = _normal_from_dir_2d(d)
        off = n1*Mpt[1] + n2*Mpt[2]
        push!(lines, _make_line_spec(arr, d, off; tie_break=tie_break))
    end

    # L_delta: slope-1 lines sweeping across the box.
    d45 = Float64[1.0, 1.0]
    (n1,n2) = _normal_from_dir_2d(d45)
    # compute min/max normal offsets over corners
    corners = [
        (lo[1], lo[2]),
        (lo[1], hi[2]),
        (hi[1], lo[2]),
        (hi[1], hi[2]),
    ]
    offs = Float64[n1*c[1] + n2*c[2] for c in corners]
    omin = minimum(offs)
    omax = maximum(offs)

    dstep = 0.0
    if delta === :auto
        # automatic: about N steps across the span
        dstep = (omax-omin)/max(1, N)
    else
        dstep = Float64(delta)
    end
    dstep > 0.0 || error("mpp_image: delta must be positive")

    off = omin
    while off <= omax + 1e-12
        push!(lines, _make_line_spec(arr, d45, off; tie_break=tie_break))
        off += dstep
    end

    return lines
end

# ---------------------- vineyard-style decomposition --------------------------

"""
    mpp_decomposition(cache::FiberedBarcodeCache2D; N=16, delta=:auto, q=1.0, tie_break=:center)

Compute the Carriere MPPI summand decomposition from a fibered barcode cache.

This is the cheap-first entrypoint when you already have a
[`FiberedBarcodeCache2D`](@ref). It builds only the geometric decomposition, not
the final image. A typical staged workflow is:

```julia
cache = fibered_barcode_cache_2d(M, pi; precompute=:barcodes)
decomp = mpp_decomposition(cache; N=8, delta=:auto)
mpp_decomposition_summary(decomp)
img = mpp_image(decomp; resolution=24, sigma=0.2)
```

Keyword arguments
- `N`: number of slice directions used in the Carriere construction.
- `delta`: vineyard matching threshold. Use `:auto` for the default heuristic.
- `q`: exponent in the geometric summand-weight formula.
- `tie_break`: deterministic rule for ambiguous local assignments.

Returns an [`MPPDecomposition`](@ref).
"""
function mpp_decomposition(cache::FiberedBarcodeCache2D;
                           N::Int=16,
                           delta::Union{Real,Symbol}=:auto,
                           q::Real=1.0,
                           tie_break::Symbol=:center)

    arr = cache.arrangement
    lines = _line_families_carriere(arr; N=N, delta=delta, tie_break=tie_break)
    nlines = length(lines)

    # Compute barcodes per line, then flatten them into one pool so the
    # decomposition path can work with offset arithmetic instead of nested
    # vectors of tiny interval tuples.
    bc_dicts = Vector{Dict{Tuple{Float64,Float64},Int}}(undef, nlines)
    counts = Vector{Int}(undef, nlines)
    total = 0
    for i in 1:nlines
        spec = lines[i]
        bc = fibered_barcode(cache, spec.dir, spec.off; values=:t, tie_break=:center)
        bc_dicts[i] = bc
        cnt = _barcode_point_count(bc)
        counts[i] = cnt
        total += cnt
    end

    # Union-find for tracking connected components across matchings.
    offsets = Vector{Int}(undef, nlines)
    next_offset = 0
    for i in 1:nlines
        offsets[i] = next_offset
        next_offset += counts[i]
    end

    bcs = Vector{Tuple{Float64,Float64}}(undef, total)
    cursor = 1
    for i in 1:nlines
        cursor = _fill_barcode_points!(bcs, cursor, bc_dicts[i])
    end

    parent = collect(1:total)
    rankv = fill(0, total)
    findp(x) = (parent[x] == x ? x : (parent[x] = findp(parent[x])))
    function unite(a,b)
        ra = findp(a)
        rb = findp(b)
        ra == rb && return
        if rankv[ra] < rankv[rb]
            parent[ra] = rb
        elseif rankv[ra] > rankv[rb]
            parent[rb] = ra
        else
            parent[rb] = ra
            rankv[ra] += 1
        end
    end

    # Match successive barcodes and union matched bars.
    for i in 1:(nlines - 1)
        countA = counts[i]
        countB = counts[i + 1]
        (countA == 0 || countB == 0) && continue
        match = _bottleneck_matching_points_flat(
            bcs,
            offsets[i] + 1,
            countA,
            offsets[i + 1] + 1,
            countB,
        )
        for a in 1:length(match)
            b = match[a]
            b == 0 && continue
            ida = offsets[i] + a
            idb = offsets[i+1] + b
            unite(ida, idb)
        end
    end

    lo, hi = _box_corners_2d(arr.box)
    areaR = (hi[1]-lo[1])*(hi[2]-lo[2])
    areaR > 0.0 || error("mpp_decomposition: box has zero area")

    total == 0 && return MPPDecomposition(lines, Vector{Vector{MPPSegment}}(), Float64[], (lo, hi))

    # Compact DSU roots to component ids and count nodes per component.
    root_to_comp = zeros(Int, total)
    item_comp = Vector{Int}(undef, total)
    comp_counts = zeros(Int, total)
    ncomp = 0
    @inbounds for id in 1:total
        r = findp(id)
        cid = root_to_comp[r]
        if cid == 0
            ncomp += 1
            cid = ncomp
            root_to_comp[r] = cid
        end
        item_comp[id] = cid
        comp_counts[cid] += 1
    end

    resize!(comp_counts, ncomp)
    summands = Vector{Vector{MPPSegment}}(undef, ncomp)
    @inbounds for cid in 1:ncomp
        summands[cid] = Vector{MPPSegment}(undef, comp_counts[cid])
    end
    next_pos = ones(Int, ncomp)

    # Fill each summand directly into its preallocated segment buffer.
    @inbounds for li in 1:nlines
        spec = lines[li]
        dx = spec.dir[1]
        dy = spec.dir[2]
        x0x = spec.x0[1]
        x0y = spec.x0[2]
        omega = spec.omega
        start = offsets[li]
        for bj in 1:counts[li]
            id = start + bj
            cid = item_comp[id]
            pos = next_pos[cid]
            b, d = bcs[id]
            summands[cid][pos] = MPPSegment(
                (x0x + b * dx, x0y + b * dy),
                (x0x + d * dx, x0y + d * dy),
                omega,
            )
            next_pos[cid] = pos + 1
        end
    end

    weights = Vector{Float64}(undef, ncomp)
    @inbounds for cid in 1:ncomp
        segs = summands[cid]
        pts_for_hull = Vector{NTuple{2,Float64}}(undef, 2 * length(segs))
        idx = 1
        for seg in segs
            pts_for_hull[idx] = seg.p
            pts_for_hull[idx + 1] = seg.q
            idx += 2
        end
        hull = _convex_hull_2d(pts_for_hull; atol=arr.atol)
        areaI = _polygon_area_2d(hull)
        weights[cid] = q == 0.0 ? 1.0 : (areaI / areaR)^Float64(q)
    end

    return MPPDecomposition(lines, summands, weights, (lo,hi))
end

# -------------------------- internal evaluation cache -------------------------

# For fast MPPI image evaluation, we precompute:
# - a per-summand bounding box (used for optional cutoff pruning), and
# - per-segment scalar data including a segment-level bounding box (used for exact pruning).
#
# Each segment is stored as a tuple
#   (px, py, qx, qy, omega, xmin, xmax, ymin, ymax)
# where (xmin,xmax,ymin,ymax) is the segment's axis-aligned bounding box.
struct _MPPImageEvalCache
    segdata::Vector{Vector{NTuple{9,Float64}}}
    bboxes::Vector{NTuple{4,Float64}}  # (xmin, xmax, ymin, ymax) per summand
end

function _mpp_image_eval_cache(decomp::MPPDecomposition)::_MPPImageEvalCache
    ns = length(decomp.summands)
    segdata = Vector{Vector{NTuple{9,Float64}}}(undef, ns)
    bboxes = Vector{NTuple{4,Float64}}(undef, ns)

    for k in 1:ns
        segs = decomp.summands[k]
        m = length(segs)

        v = Vector{NTuple{9,Float64}}(undef, m)

        xmin = Inf
        xmax = -Inf
        ymin = Inf
        ymax = -Inf

        for i in 1:m
            seg = segs[i]
            px = seg.p[1]; py = seg.p[2]
            qx = seg.q[1]; qy = seg.q[2]
            om = seg.omega

            sxmin = (px < qx ? px : qx)
            sxmax = (px < qx ? qx : px)
            symin = (py < qy ? py : qy)
            symax = (py < qy ? qy : py)

            v[i] = (px, py, qx, qy, om, sxmin, sxmax, symin, symax)

            # Update summand bounding box.
            if px < xmin; xmin = px; end
            if qx < xmin; xmin = qx; end
            if px > xmax; xmax = px; end
            if qx > xmax; xmax = qx; end
            if py < ymin; ymin = py; end
            if qy < ymin; ymin = qy; end
            if py > ymax; ymax = py; end
            if qy > ymax; ymax = qy; end
        end

        segdata[k] = v
        if m == 0
            # Should not happen for MPPI summands, but keep a sensible default.
            bboxes[k] = (0.0, 0.0, 0.0, 0.0)
        else
            bboxes[k] = (xmin, xmax, ymin, ymax)
        end
    end

    return _MPPImageEvalCache(segdata, bboxes)
end

# Determine an effective cutoff radius from either an explicit radius or a kernel-value tolerance.
#
# - If both are omitted, returns Inf (no cutoff).
# - If cutoff_tol is provided, we use exp(-d^2/sigma^2) <= cutoff_tol to define the radius.
function _mpp_effective_cutoff_radius(sig::Float64,
                                      cutoff_radius::Union{Nothing,Real},
                                      cutoff_tol::Union{Nothing,Real})::Float64
    if cutoff_radius !== nothing && cutoff_tol !== nothing
        throw(ArgumentError("mpp_image: use at most one of cutoff_radius and cutoff_tol"))
    end

    if cutoff_radius !== nothing
        r = Float64(cutoff_radius)
        r >= 0.0 || throw(ArgumentError("mpp_image: cutoff_radius must be >= 0"))
        return r
    end

    if cutoff_tol !== nothing
        tol = Float64(cutoff_tol)
        (0.0 < tol < 1.0) || throw(ArgumentError("mpp_image: cutoff_tol must satisfy 0 < cutoff_tol < 1"))
        # exp(-d^2/sigma^2) <= tol  <=>  d >= sigma * sqrt(log(1/tol))
        return sig * sqrt(log(1.0 / tol))
    end

    return Inf
end


# ---------------------------- MPPI evaluation --------------------------------

"""
    mpp_image(decomp::MPPDecomposition;
              resolution=32, xgrid=nothing, ygrid=nothing,
              sigma=0.05, cutoff_radius=nothing, cutoff_tol=nothing,
              segment_prune=true)

Evaluate a Carriere multiparameter persistence image from a precomputed vineyard decomposition.

This method does not recompute barcodes or matchings; it only evaluates the kernel sum
on a grid in R^2.

Keyword arguments
- `resolution`: number of grid points per axis (used when `xgrid` or `ygrid` is `nothing`).
- `xgrid`, `ygrid`: explicit grids. If provided, they are converted to `Vector{Float64}`.
- `sigma`: Gaussian scale in exp(-d^2/sigma^2).
- `cutoff_radius`: optional distance cutoff. If set, summands whose distance to the query
  point exceeds this cutoff are skipped. This is an approximation.
- `cutoff_tol`: alternative to `cutoff_radius`. Defines the cutoff radius by
  `exp(-d^2/sigma^2) <= cutoff_tol`. Requires `0 < cutoff_tol < 1`.
- `segment_prune`: if true, use exact bounding-box pruning inside the nearest-segment search
  (this does not change the result; it only reduces work).

Typical workflow:

```julia
decomp = mpp_decomposition(M, pi; N=8)
img = mpp_image(decomp; resolution=24, sigma=0.2)
mpp_image_summary(img)
```

Use [`mpp_image_distance`](@ref) and [`mpp_image_kernel`](@ref) once images are
on the same grid.

Returns an [`MPPImage`](@ref).
"""
function mpp_image(decomp::MPPDecomposition;
                   resolution::Int=32,
                   xgrid=nothing,
                   ygrid=nothing,
                   sigma::Real=0.05,
                   cutoff_radius::Union{Nothing,Real}=nothing,
                   cutoff_tol::Union{Nothing,Real}=nothing,
                   segment_prune::Bool=true,
                   threads::Bool = (Threads.nthreads() > 1))

    sig = Float64(sigma)
    sig > 0.0 || error("mpp_image: sigma must be positive")

    # If at least one axis uses `resolution`, enforce it.
    if xgrid === nothing || ygrid === nothing
        resolution > 1 || error("mpp_image: resolution must be >= 2")
    end

    lo, hi = decomp.box

    xg = if xgrid === nothing
        collect(range(lo[1], hi[1], length=resolution))
    else
        xgtmp = Float64.(collect(xgrid))
        length(xgtmp) > 1 || error("mpp_image: xgrid must have length >= 2")
        xgtmp
    end

    yg = if ygrid === nothing
        collect(range(lo[2], hi[2], length=resolution))
    else
        ygtmp = Float64.(collect(ygrid))
        length(ygtmp) > 1 || error("mpp_image: ygrid must have length >= 2")
        ygtmp
    end

    cutoff = _mpp_effective_cutoff_radius(sig, cutoff_radius, cutoff_tol)
    cutoff2 = isfinite(cutoff) ? cutoff*cutoff : Inf

    eval_cache = _mpp_image_eval_cache(decomp)

    img = zeros(Float64, length(yg), length(xg))
    invsig2 = 1.0/(sig*sig)

    ns = length(decomp.weights)
    weights = decomp.weights
    segdata = eval_cache.segdata
    bboxes = eval_cache.bboxes

    # Loop order:
    # - ix outer, iy inner writes contiguous memory in Julia's column-major layout
    #   because img[iy, ix] has the first index varying fastest.
    if threads && Threads.nthreads() > 1
        Threads.@threads for ix in 1:length(xg)
            x = xg[ix]
            for iy in 1:length(yg)
                y = yg[iy]

                acc = 0.0

                for k in 1:ns
                    wI = weights[k]
                    wI == 0.0 && continue

                    # Optional approximate pruning: skip summands whose bounding box is farther
                    # than the cutoff radius from the query point.
                    if cutoff2 < Inf
                        (xmin, xmax, ymin, ymax) = bboxes[k]
                        if _bbox_dist2_xy(x, y, xmin, xmax, ymin, ymax) > cutoff2
                            continue
                        end
                    end

                    bestd2 = Inf
                    bestomega = 0.0

                    segs = segdata[k]
                    @inbounds for j in 1:length(segs)
                        seg = segs[j]
                        (px, py, qx, qy, om, sxmin, sxmax, symin, symax) = seg

                        # Exact pruning: if this segment's bounding box is already farther
                        # than the best distance found so far, it cannot improve the minimum.
                        if segment_prune && bestd2 < Inf
                            if _bbox_dist2_xy(x, y, sxmin, sxmax, symin, symax) >= bestd2
                                continue
                            end
                        end

                        d2 = _dist2_point_segment_xy(x, y, px, py, qx, qy)
                        if d2 < bestd2
                            bestd2 = d2
                            bestomega = om
                            if bestd2 == 0.0
                                break
                            end
                        end
                    end

                    # Optional cutoff: if the nearest segment is still beyond the cutoff,
                    # the kernel value is below exp(-cutoff^2/sigma^2).
                    if cutoff2 < Inf && bestd2 > cutoff2
                        continue
                    end

                    acc += wI * bestomega * exp(-bestd2 * invsig2)
                end

                img[iy, ix] = acc
            end
        end
    else
        @inbounds for ix in 1:length(xg)
            x = xg[ix]
            for iy in 1:length(yg)
                y = yg[iy]

                acc = 0.0

                for k in 1:ns
                    wI = weights[k]
                    wI == 0.0 && continue

                    # Optional approximate pruning: skip summands whose bounding box is farther
                    # than the cutoff radius from the query point.
                    if cutoff2 < Inf
                        (xmin, xmax, ymin, ymax) = bboxes[k]
                        if _bbox_dist2_xy(x, y, xmin, xmax, ymin, ymax) > cutoff2
                            continue
                        end
                    end

                    bestd2 = Inf
                    bestomega = 0.0

                    segs = segdata[k]
                    @inbounds for j in 1:length(segs)
                        seg = segs[j]
                        (px, py, qx, qy, om, sxmin, sxmax, symin, symax) = seg

                        # Exact pruning: if this segment's bounding box is already farther
                        # than the best distance found so far, it cannot improve the minimum.
                        if segment_prune && bestd2 < Inf
                            if _bbox_dist2_xy(x, y, sxmin, sxmax, symin, symax) >= bestd2
                                continue
                            end
                        end

                        d2 = _dist2_point_segment_xy(x, y, px, py, qx, qy)
                        if d2 < bestd2
                            bestd2 = d2
                            bestomega = om
                            if bestd2 == 0.0
                                break
                            end
                        end
                    end

                    # Optional cutoff: if the nearest segment is still beyond the cutoff,
                    # the kernel value is below exp(-cutoff^2/sigma^2).
                    if cutoff2 < Inf && bestd2 > cutoff2
                        continue
                    end

                    acc += wI * bestomega * exp(-bestd2 * invsig2)
                end

                img[iy, ix] = acc
            end
        end
    end

    return MPPImage(xg, yg, img, sig, decomp)
end

"""
    mpp_image(cache::FiberedBarcodeCache2D;
              resolution=32, xgrid=nothing, ygrid=nothing,
              sigma=0.05, N=16, delta=:auto, q=1.0,
              tie_break=:center,
              cutoff_radius=nothing, cutoff_tol=nothing, segment_prune=true)

Compute a Carriere multiparameter persistence image (MPPI) from a fibered-barcode cache.

This does two stages:
1) `mpp_decomposition(cache; N, delta, q, tie_break)`  (barcodes + matchings + vineyard tracks)
2) `mpp_image(decomp; resolution, xgrid, ygrid, sigma, cutoff_radius, cutoff_tol, segment_prune)`

Keyword arguments
- `resolution`, `xgrid`, `ygrid`, `sigma`: image evaluation parameters.
- `N`, `delta`, `q`, `tie_break`: Carriere decomposition parameters (see `mpp_decomposition`).
- `cutoff_radius` / `cutoff_tol`: optional approximate cutoff for faster evaluation.
- `segment_prune`: exact bounding-box pruning inside each summand's nearest-segment search.

Use this when you want the final image directly from an already-built fibered
cache. For explicit inspection of the intermediate decomposition, call
[`mpp_decomposition(cache)`](@ref) first and then [`mpp_image(decomp)`](@ref).

Returns an [`MPPImage`](@ref).
"""
function mpp_image(cache::FiberedBarcodeCache2D;
                   resolution::Int=32,
                   xgrid=nothing,
                   ygrid=nothing,
                   sigma::Real=0.05,
                   N::Int=16,
                   delta::Union{Real,Symbol}=:auto,
                   q::Real=1.0,
                   tie_break::Symbol=:center,
                   cutoff_radius::Union{Nothing,Real}=nothing,
                   cutoff_tol::Union{Nothing,Real}=nothing,
                   segment_prune::Bool=true,
                   threads::Bool = (Threads.nthreads() > 1))

    decomp = mpp_decomposition(cache; N=N, delta=delta, q=q, tie_break=tie_break)
    return mpp_image(decomp;
                     resolution=resolution,
                     xgrid=xgrid,
                     ygrid=ygrid,
                     sigma=sigma,
                     cutoff_radius=cutoff_radius,
                     cutoff_tol=cutoff_tol,
                     segment_prune=segment_prune,
                     threads=threads)
end

"""
    mpp_decomposition(M, pi, opts::InvariantOptions; kwargs...)

Compute a 2D multiparameter persistence (MPP) decomposition for `M` over `pi`.

This wrapper:
1) builds an exact 2D fibered arrangement (including axes),
2) builds a fibered barcode cache,
3) calls `mpp_decomposition(cache)`.

Opts usage:
- `opts.box` and `opts.strict` control arrangement/windowing behavior.

This is the canonical high-level entrypoint when you start from a module and
encoding map. The decomposition-specific keywords `N`, `delta`, `q`, and
`tie_break` are threaded through to the owner-level decomposition routine.

Typical workflow:

```julia
decomp = mpp_decomposition(M, pi; N=8, delta=:auto)
mpp_decomposition_summary(decomp)
```
"""
function mpp_decomposition(
    M::PModule{K},
    pi::Union{PLikeEncodingMap,CompiledEncoding},
    opts::InvariantOptions;
    N::Int=16,
    delta::Union{Real,Symbol}=:auto,
    q::Real=1.0,
    tie_break::Symbol=:center,
    kwargs...,
) where {K}
    cache_kwargs = _drop_keys(kwargs, (:N, :delta, :q, :tie_break))
    if !haskey(cache_kwargs, :include_axes)
        cache_kwargs = merge((include_axes=true,), cache_kwargs)
    end
    cache = fibered_barcode_cache_2d(M, pi, opts; cache_kwargs...)
    return mpp_decomposition(cache; N=N, delta=delta, q=q, tie_break=tie_break)
end


"""
    mpp_image(M, pi, opts::InvariantOptions; kwargs...)

Compute the 2D MPP image for `M` over `pi` via an exact fibered cache.

Opts usage:
- `opts.box` and `opts.strict` control arrangement/windowing behavior.

This is the canonical direct workflow when you want the final image and do not
need to inspect the intermediate decomposition:

```julia
img = mpp_image(M, pi; resolution=24, sigma=0.2, N=8)
mpp_image_summary(img)
```
"""
function mpp_image(
    M::PModule{K},
    pi::Union{PLikeEncodingMap,CompiledEncoding},
    opts::InvariantOptions;
    resolution::Int=32,
    xgrid=nothing,
    ygrid=nothing,
    sigma::Real=0.05,
    N::Int=16,
    delta::Union{Real,Symbol}=:auto,
    q::Real=1.0,
    tie_break::Symbol=:center,
    cutoff_radius::Union{Nothing,Real}=nothing,
    cutoff_tol::Union{Nothing,Real}=nothing,
    segment_prune::Bool=true,
    threads::Bool = (Threads.nthreads() > 1),
    kwargs...,
) where {K}
    cache_kwargs = _drop_keys(kwargs, (
        :resolution, :xgrid, :ygrid, :sigma,
        :N, :delta, :q, :tie_break,
        :cutoff_radius, :cutoff_tol, :segment_prune, :threads,
    ))
    if !haskey(cache_kwargs, :include_axes)
        cache_kwargs = merge((include_axes=true,), cache_kwargs)
    end
    cache = fibered_barcode_cache_2d(M, pi, opts; cache_kwargs..., threads=threads)
    return mpp_image(cache;
                     resolution=resolution,
                     xgrid=xgrid,
                     ygrid=ygrid,
                     sigma=sigma,
                     N=N,
                     delta=delta,
                     q=q,
                     tie_break=tie_break,
                     cutoff_radius=cutoff_radius,
                     cutoff_tol=cutoff_tol,
                     segment_prune=segment_prune,
                     threads=threads)
end

mpp_image(M::PModule{K}, pi::PLikeEncodingMap; opts::InvariantOptions=InvariantOptions(), kwargs...) where {K} =
    mpp_image(M, pi, opts; kwargs...)

# ------------------------ MPPI image operations -------------------------------

"""
    mpp_image_inner_product(A::MPPImage, B::MPPImage)

L2 inner product of two MPPI images on a common grid.

This is the linear-kernel building block for image-space comparisons. Use
[`mpp_image_distance`](@ref) for the associated metric and
[`mpp_image_kernel`](@ref) for the Gaussian kernel.
"""
function mpp_image_inner_product(A::MPPImage, B::MPPImage)
    A.xgrid == B.xgrid || error("mpp_image_inner_product: xgrid mismatch")
    A.ygrid == B.ygrid || error("mpp_image_inner_product: ygrid mismatch")
    size(A.img) == size(B.img) || error("mpp_image_inner_product: img size mismatch")
    return sum(A.img .* B.img)
end

"""
    mpp_image_distance(A::MPPImage, B::MPPImage)

L2 distance between two MPPI images on a common grid.

If the grids differ, rebuild one image so that both use the same `xgrid` and
`ygrid` before comparing them.
"""
function mpp_image_distance(A::MPPImage, B::MPPImage)::Float64
    ipAA = mpp_image_inner_product(A,A)
    ipBB = mpp_image_inner_product(B,B)
    ipAB = mpp_image_inner_product(A,B)
    val = ipAA + ipBB - 2.0*ipAB
    return sqrt(max(val, 0.0))
end

"""
    mpp_image_kernel(A::MPPImage, B::MPPImage; sigma=1.0)

Gaussian kernel on MPPI images:
    exp(-||A-B||^2 / sigma^2)

This is the canonical image-space kernel once two [`MPPImage`](@ref) objects
have already been constructed on the same grid.
"""
function mpp_image_kernel(A::MPPImage, B::MPPImage; sigma::Real=1.0)::Float64
    s = Float64(sigma)
    s > 0.0 || error("mpp_image_kernel: sigma must be positive")
    d = mpp_image_distance(A,B)
    return exp(-(d*d)/(s*s))
end

# ----- Multiparameter persistence landscapes ----------------------------------
#
# We implement a practical, finite-sampled version of Vipond's multiparameter
# persistence landscapes. The key computational primitive is already in place:
# for a chosen geometric slice (line) we can restrict a multiparameter module to
# a chain in a finite encoding, compute its 1D barcode, and then convert that
# barcode into an ordinary 1D persistence landscape (Bubenik).
#
# The resulting "mp landscape" is represented here as a family of 1D landscapes
# indexed by a finite list of slice directions and offsets, together with a
# common evaluation grid and slice weights. This makes it directly usable as a
# stable feature map for statistics and kernel methods.

"""
    MPLandscape

A sampled multiparameter persistence landscape.

This object stores a finite family of one-parameter persistence landscapes,
indexed by slice direction and offset, together with slice weights and a common
evaluation grid. The cheap workflow is:

```julia
L = mp_landscape(M, pi; directions=dirs, offsets=offs, ts=0:40, kmax=3)
describe(L)
mp_landscape_summary(L)
check_mp_landscape(L)
```

Use [`mp_landscape_distance`](@ref), [`mp_landscape_inner_product`](@ref), and
[`mp_landscape_kernel`](@ref) for comparison once the object has been built.
"""
struct MPLandscape{D,O}
    kmax::Int
    tgrid::Vector{Float64}
    values::Array{Float64,4}
    weights::Matrix{Float64}
    directions::Vector{D}
    offsets::Vector{O}
end

"""
    L[idir, ioff]

Return a (kmax x length(tgrid)) view of the stored 1D landscape values for the
slice with direction index `idir` and offset index `ioff`.
"""
Base.getindex(L::MPLandscape, idir::Int, ioff::Int) = view(L.values, idir, ioff, :, :)

@inline landscape_grid(L::MPLandscape) = L.tgrid
@inline landscape_values(L::MPLandscape) = L.values
@inline landscape_layers(L::MPLandscape)::Int = L.kmax
@inline feature_dimension(L::MPLandscape)::Int = length(L.values)
@inline slice_weights(L::MPLandscape) = L.weights
@inline slice_directions(L::MPLandscape) = L.directions
@inline slice_offsets(L::MPLandscape) = L.offsets
@inline ndirections(L::MPLandscape)::Int = size(L.values, 1)
@inline noffsets(L::MPLandscape)::Int = size(L.values, 2)
@inline weight_sum(L::MPLandscape)::Float64 = sum(L.weights)
@inline landscape_slice(L::MPLandscape, idir::Integer, ioff::Integer) =
    view(L.values, Int(idir), Int(ioff), :, :)

function describe(L::MPLandscape)
    tg = landscape_grid(L)
    return (
        kind = :mp_landscape,
        ndirections = ndirections(L),
        noffsets = noffsets(L),
        kmax = landscape_layers(L),
        grid_length = length(tg),
        values_shape = size(landscape_values(L)),
        weight_shape = size(slice_weights(L)),
        total_weight = weight_sum(L),
        t_range = isempty(tg) ? nothing : (tg[1], tg[end]),
        weights_normalized = abs(weight_sum(L) - 1.0) < 1e-8,
    )
end

"""
    mp_landscape_summary(L::MPLandscape)

Owner-local summary alias for [`describe(L)`](@ref).

Typical workflow:

```julia
L = mp_landscape(M, pi; directions=dirs, offsets=offs, ts=0:40, kmax=3)
mp_landscape_summary(L)
check_mp_landscape(L)
```
"""
@inline mp_landscape_summary(L::MPLandscape) = describe(L)

# -----------------------------------------------------------------------------
# Pretty-printing / ergonomics
# -----------------------------------------------------------------------------

"""
    Base.show(io::IO, L::MPLandscape)

Print a compact one-line summary for a sampled multiparameter persistence
landscape. The summary is meant to be "mathematician-friendly" and includes:

- number of directions (ndirs)
- number of offsets (noffsets)
- kmax (number of landscape layers stored)
- nt (length of the common t-grid)
- whether the stored slice weights appear normalized (sum(weights) ~= 1)

For a more verbose multi-line summary, use:

    show(io, MIME("text/plain"), L)
"""
function Base.show(io::IO, L::MPLandscape)
    d = describe(L)

    print(io,
          "MPLandscape(",
          "ndirs=", d.ndirections,
          ", noffsets=", d.noffsets,
          ", kmax=", d.kmax,
          ", nt=", d.grid_length,
          ", weights_normalized=", d.weights_normalized,
          ")")
end

"""
    Base.show(io::IO, ::MIME"text/plain", L::MPLandscape)

Verbose multi-line summary for `MPLandscape`.
"""
function Base.show(io::IO, ::MIME"text/plain", L::MPLandscape)
    d = describe(L)

    println(io, "MPLandscape")
    println(io, "  ndirs = ", d.ndirections)
    println(io, "  noffsets = ", d.noffsets)
    println(io, "  kmax = ", d.kmax)
    println(io, "  nt = ", d.grid_length)

    if d.t_range !== nothing
        println(io, "  tmin = ", d.t_range[1])
        println(io, "  tmax = ", d.t_range[2])
    end

    println(io, "  weights_sum = ", d.total_weight)
    println(io, "  weights_normalized = ", d.weights_normalized)
end

"""
    check_mpp_decomposition(decomp; throw=false)

Validate a hand-built [`MPPDecomposition`](@ref).

Typical workflow:

```julia
decomp = mpp_decomposition(M, pi; N=8)
check_mpp_decomposition(decomp)
```
"""
function check_mpp_decomposition(decomp::MPPDecomposition; throw::Bool=false)
    errors = String[]

    length(decomp.summands) == length(decomp.weights) ||
        push!(errors, "summands and weights must have the same length")

    lo, hi = bounding_box(decomp)
    _mppi_is_finite_pair(lo) || push!(errors, "box lower corner must be a finite length-2 pair")
    _mppi_is_finite_pair(hi) || push!(errors, "box upper corner must be a finite length-2 pair")
    if _mppi_is_finite_pair(lo) && _mppi_is_finite_pair(hi)
        lo[1] <= hi[1] || push!(errors, "box must satisfy lo[1] <= hi[1]")
        lo[2] <= hi[2] || push!(errors, "box must satisfy lo[2] <= hi[2]")
    end

    @inbounds for (i, line) in enumerate(line_specs(decomp))
        _append_prefixed_errors!(errors, "line $i: ", check_mpp_line_spec(line; throw=false))
    end

    @inbounds for (i, w) in enumerate(summand_weights(decomp))
        isfinite(w) || push!(errors, "weight $i must be finite")
        w >= 0.0 || push!(errors, "weight $i must be nonnegative")
    end

    @inbounds for i in 1:nsummands(decomp)
        segs = summand_segments(decomp, i)
        isempty(segs) && push!(errors, "summand $i must not be empty")
        for (j, seg) in enumerate(segs)
            p, q, om = seg
            _mppi_is_finite_pair(p) || push!(errors, "summand $i segment $j: p must be a finite length-2 pair")
            _mppi_is_finite_pair(q) || push!(errors, "summand $i segment $j: q must be a finite length-2 pair")
            isfinite(om) || push!(errors, "summand $i segment $j: omega must be finite")
            om >= 0.0 || push!(errors, "summand $i segment $j: omega must be nonnegative")
        end
    end

    return _mppi_validation_result(MPPDecompositionValidationSummary, errors; throw=throw)
end

"""
    check_mpp_image(img; throw=false)

Validate a hand-built [`MPPImage`](@ref).

Typical workflow:

```julia
img = mpp_image(M, pi; resolution=24, sigma=0.2, N=8)
check_mpp_image(img)
```
"""
function check_mpp_image(img::MPPImage; throw::Bool=false)
    errors = String[]

    xg = image_xgrid(img)
    yg = image_ygrid(img)
    vals = image_values(img)

    _mppi_is_strictly_increasing(xg) || push!(errors, "xgrid must be strictly increasing with length >= 2")
    _mppi_is_strictly_increasing(yg) || push!(errors, "ygrid must be strictly increasing with length >= 2")
    size(vals) == (length(yg), length(xg)) ||
        push!(errors, "image matrix shape must equal (length(ygrid), length(xgrid))")
    all(isfinite, vals) || push!(errors, "image matrix must contain only finite values")
    isfinite(img.sigma) || push!(errors, "sigma must be finite")
    img.sigma > 0.0 || push!(errors, "sigma must be positive")

    _append_prefixed_errors!(errors, "decomposition: ", check_mpp_decomposition(decomposition(img); throw=false))
    return _mppi_validation_result(MPPImageValidationSummary, errors; throw=throw)
end

"""
    check_mp_landscape(L; throw=false)

Validate a hand-built [`MPLandscape`](@ref).

Typical workflow:

```julia
L = mp_landscape(M, pi; directions=dirs, offsets=offs, ts=0:40, kmax=3)
check_mp_landscape(L)
```
"""
function check_mp_landscape(L::MPLandscape; throw::Bool=false)
    errors = String[]

    tg = landscape_grid(L)
    vals = landscape_values(L)
    W = slice_weights(L)
    nd = ndirections(L)
    no = noffsets(L)

    _mppi_is_strictly_increasing(tg) || push!(errors, "tgrid must be strictly increasing with length >= 2")
    landscape_layers(L) > 0 || push!(errors, "kmax must be positive")
    size(vals, 3) == landscape_layers(L) || push!(errors, "values third dimension must equal kmax")
    size(W) == (nd, no) || push!(errors, "weights shape must equal (ndirections, noffsets)")
    length(slice_directions(L)) == nd || push!(errors, "directions length must equal ndirections")
    length(slice_offsets(L)) == no || push!(errors, "offsets length must equal noffsets")
    all(isfinite, vals) || push!(errors, "values must contain only finite entries")
    all(isfinite, W) || push!(errors, "weights must contain only finite entries")
    all(w -> w >= 0.0, W) || push!(errors, "weights must be nonnegative")

    return _mppi_validation_result(MPLandscapeValidationSummary, errors; throw=throw)
end

# Internal: trapezoidal rule on a common grid.
function _trapz(tg::Vector{Float64}, f::AbstractVector{Float64})::Float64
    nt = length(tg)
    length(f) == nt || error("_trapz: length mismatch")
    nt < 2 && return 0.0

    s = 0.0
    for i in 1:nt-1
        dt = tg[i + 1] - tg[i]
        dt >= 0 || error("_trapz: tgrid must be sorted increasing")
        s += 0.5 * (f[i] + f[i + 1]) * dt
    end
    return s
end

# Internal: choose weights according to a policy.
function _combine_weights(LA::MPLandscape, LB::MPLandscape, mode::Symbol)::Matrix{Float64}
    if mode == :check
        maximum(abs.(LA.weights .- LB.weights)) < 1e-12 || error("mp_landscape_*: weight mismatch between inputs")
        return LA.weights
    elseif mode == :left
        return LA.weights
    elseif mode == :right
        return LB.weights
    elseif mode == :average
        return 0.5 .* (LA.weights .+ LB.weights)
    else
        error("_combine_weights: mode must be :check, :left, :right, or :average")
    end
end

"""
    mp_landscape(M, slices; kmax=5, tgrid, default_weight=1.0, normalize_weights=true) -> MPLandscape
    mp_landscape(M, pi; directions, offsets, kmax=5, ...) -> MPLandscape

Compute a sampled multiparameter persistence landscape.

Two calling patterns are supported.

1) Explicit slices (purely combinatorial):
    slices = [
        [q1,q2,...,qm],                                # a chain in M.Q
        (chain=[...], values=[...], weight=1.0),       # richer spec (NamedTuple)
        (chain, values, weight)                        # tuple form
    ]
    L = mp_landscape(M, slices; kmax=5, tgrid=...)

2) Geometric slices via a finite encoding map `pi`:
    L = mp_landscape(M, pi;
        directions=[v1,v2,...],
        offsets=[x01,x02,...],
        tgrid=...,
        strict=true,
        direction_weight=:none,
        normalize_weights=true
    )

The geometric version builds chains via `slice_chain`, computes 1D barcodes via
`slice_barcode`, and converts them to 1D persistence landscapes via
`persistence_landscape`.

Remarks for Z^n
---------------
For `pi::ZnEncodingMap`, prefer passing an explicit discrete parameter grid via
`ts=...`, for example `ts=0:50`.

Contract notes
--------------
- Use explicit `slices` when you already know the exact chains or slice values
  you want to compare.
- Use `pi` when you want the owner to build geometric slices from directions and
  offsets automatically.
- `tgrid` is the common real evaluation grid for the resulting 1D landscapes.
- `ts` is the discrete parameter grid used only for `ZnEncodingMap` slicing;
  it does not replace `tgrid`.
- `directions` and `offsets` parameterize the slice family in the source
  geometry. They are part of the mathematical definition of the finite
  landscape family, not low-level performance knobs.
- If you will reuse the same slice family across many modules or queries, the
  intended advanced route is to precompile it with [`compile_slice_plan`](@ref)
  and reuse the resulting plan via the `cache=` surface.

See also `mp_landscape_distance` and `mp_landscape_kernel`.
"""
function mp_landscape(
    M::PModule{K},
    slices::AbstractVector;
    kmax::Int=5,
    tgrid,
    default_weight=1.0,
    normalize_weights::Bool=true
)::MPLandscape where {K}
    specs = collect_slices(slices; default_weight=default_weight)
    return mp_landscape(M, specs;
                        kmax=kmax,
                        tgrid=tgrid,
                        default_weight=default_weight,
                        normalize_weights=normalize_weights)
end

function mp_landscape(
    M::PModule{K},
    slices::AbstractVector{<:SliceSpec{<:Real,Nothing}};
    kmax::Int=5,
    tgrid,
    default_weight=1.0,
    normalize_weights::Bool=true
)::MPLandscape where {K}
    _ = default_weight
    tg = _clean_tgrid(tgrid)
    nt = length(tg)

    nslices = length(slices)
    nslices > 0 || error("mp_landscape: slices list is empty")

    vals = zeros(Float64, nslices, 1, kmax, nt)
    W = zeros(Float64, nslices, 1)
    dirs = collect(slices)
    offs = [nothing]
    points_scratch = Tuple{Float64,Float64}[]
    tent_scratch = Float64[]

    @inbounds for i in 1:nslices
        spec = slices[i]
        bc = _slice_barcode_packed(M, spec.chain; values=nothing)
        _persistence_landscape_values!(
            @view(vals[i, 1, :, :]),
            bc,
            tg;
            points_scratch=points_scratch,
            tent_scratch=tent_scratch,
        )
        W[i, 1] = float(spec.weight)
    end

    if normalize_weights
        s = sum(W)
        s > 0 || error("mp_landscape: total slice weight is zero")
        W ./= s
    end

    return MPLandscape(kmax, tg, vals, W, dirs, offs)
end

function mp_landscape(
    M::PModule{K},
    slices::AbstractVector{<:SliceSpec{<:Real,<:AbstractVector}};
    kmax::Int=5,
    tgrid,
    default_weight=1.0,
    normalize_weights::Bool=true
)::MPLandscape where {K}
    _ = default_weight
    tg = _clean_tgrid(tgrid)
    nt = length(tg)

    nslices = length(slices)
    nslices > 0 || error("mp_landscape: slices list is empty")

    vals = zeros(Float64, nslices, 1, kmax, nt)
    W = zeros(Float64, nslices, 1)
    dirs = collect(slices)
    offs = [nothing]
    points_scratch = Tuple{Float64,Float64}[]
    tent_scratch = Float64[]

    @inbounds for i in 1:nslices
        spec = slices[i]
        bc = _slice_barcode_packed(M, spec.chain; values=spec.values)
        _persistence_landscape_values!(
            @view(vals[i, 1, :, :]),
            bc,
            tg;
            points_scratch=points_scratch,
            tent_scratch=tent_scratch,
        )
        W[i, 1] = float(spec.weight)
    end

    if normalize_weights
        s = sum(W)
        s > 0 || error("mp_landscape: total slice weight is zero")
        W ./= s
    end

    return MPLandscape(kmax, tg, vals, W, dirs, offs)
end

function mp_landscape(
    M::PModule{K},
    chain::AbstractVector{Int};
    kwargs...
) where {K}
    return mp_landscape(M, [chain]; kwargs...)
end

function mp_landscape(
    M::PModule{K},
    pi,
    opts::InvariantOptions;
    directions = nothing,
    offsets = nothing,
    kmax::Integer = 10,
    tgrid = nothing,
    tmin = nothing,
    tmax = nothing,
    n_t::Integer = 100,
    direction_weight::Symbol = :uniform,
    normalize_weights::Bool = true,
    drop_unknown::Bool = true,
    dedup::Bool = true,
    cache = nothing,
    ts = nothing,
    kwargs...,
)::MPLandscape where {K}
    kwargs_nt = NamedTuple(kwargs)
    if haskey(kwargs_nt, :box) || haskey(kwargs_nt, :strict) || haskey(kwargs_nt, :threads) ||
        haskey(kwargs_nt, :axes) || haskey(kwargs_nt, :axes_policy) || haskey(kwargs_nt, :max_axis_len)
        throw(ArgumentError("mp_landscape: do not pass options fields as keywords; use opts::InvariantOptions"))
    end
    if haskey(kwargs_nt, :kmin) || haskey(kwargs_nt, :kmax_param)
        throw(ArgumentError("mp_landscape: use ts=... for discrete parameter samples instead of kmin/kmax_param"))
    end

    strict0 = opts.strict === nothing ? true : opts.strict
    if opts.box === nothing
        box_kw = pi isa PLikeEncodingMap ? :auto : nothing
    else
        box_kw = opts.box
    end

    opts_chain = InvariantOptions(
        axes = opts.axes,
        axes_policy = opts.axes_policy,
        max_axis_len = opts.max_axis_len,
        box = box_kw,
        threads = opts.threads,
        strict = strict0,
        pl_mode = opts.pl_mode,
    )

    forward = _drop_keys(kwargs_nt, (:ts,))

    d = dimension(pi)
    if directions === nothing
        directions = orthant_directions(d)
    end
    dirs_in = orthant_directions(d, directions)

    if tmin === nothing || tmax === nothing || offsets === nothing
        lo, hi = encoding_box(pi, opts_chain)
        if tmin === nothing
            tmin = -minimum(abs.(vcat(lo, hi)))
        end
        if tmax === nothing
            tmax = maximum(abs.(vcat(lo, hi)))
        end
        if offsets === nothing
            offsets = default_offsets(pi, opts_chain)
        end
    end

    tg = tgrid === nothing ? collect(LinRange(tmin, tmax, n_t)) : collect(tgrid)

    direction_weight_spec = if direction_weight == :uniform
        :uniform
    elseif direction_weight == :uniform2d
        uniform2d
    else
        error("mp_landscape: unknown direction_weight mode $(direction_weight)")
    end

    ts_arg = ts === nothing ? tg : ts

    threads0 = opts.threads === nothing ? (Threads.nthreads() > 1) : opts.threads
    plan = compile_slices(
        pi,
        opts_chain;
        directions = dirs_in,
        offsets = offsets,
        normalize_dirs = :none,
        direction_weight = direction_weight_spec,
        offset_weights = nothing,
        normalize_weights = normalize_weights,
        drop_unknown = drop_unknown,
        dedup = dedup,
        threads = threads0,
        cache = cache,
        ts = ts_arg,
        forward...,
    )

    return mp_landscape(
        M,
        plan;
        kmax = kmax,
        tgrid = tg,
        normalize_weights = false,
        threads = threads0,
    )
end

mp_landscape(M::PModule{K}, pi; kwargs...) where {K} =
    mp_landscape(M, pi, InvariantOptions(); kwargs...)



"""
    mp_landscape_distance(LM, LN; p=2, weight_mode=:check) -> Float64
    mp_landscape_distance(M, N, slices; p=2, ...) -> Float64
    mp_landscape_distance(M, N, pi; p=2, ...) -> Float64

Compute an L^p distance between multiparameter persistence landscapes.

- If `LM` and `LN` are `MPLandscape` objects, this compares them directly.
- If modules `M` and `N` are provided, the landscapes are constructed using the
  provided slice data (explicit `slices` or geometric `pi`) and then compared.

The integral in the t-parameter is approximated by the trapezoidal rule on the
common grid stored in the landscape object(s). The sum over k is truncated at
the stored `kmax`.

`p` may be any positive real number or `Inf`.

Use the object form when landscapes are already built. Use the module forms when
you want the owner to construct comparable landscapes from the same slice family
first.
"""
function mp_landscape_distance(
    LA::MPLandscape,
    LB::MPLandscape;
    p::Real=2,
    weight_mode::Symbol=:check
)::Float64
    LA.kmax == LB.kmax || error("mp_landscape_distance: kmax mismatch")
    size(LA.values) == size(LB.values) || error("mp_landscape_distance: values array shape mismatch")
    length(LA.tgrid) == length(LB.tgrid) || error("mp_landscape_distance: tgrid length mismatch")
    maximum(abs.(LA.tgrid .- LB.tgrid)) < 1e-12 || error("mp_landscape_distance: tgrid mismatch")

    W = _combine_weights(LA, LB, weight_mode)

    nd, no, kmax, nt = size(LA.values)
    tg = LA.tgrid
    pp = float(p)

    if isinf(pp)
        best = 0.0
        for i in 1:nd, j in 1:no
            w = W[i, j]
            for k in 1:kmax
                dv = maximum(abs.(view(LA.values, i, j, k, :) .- view(LB.values, i, j, k, :)))
                best = max(best, w * dv)
            end
        end
        return best
    end

    pp > 0 || error("mp_landscape_distance: p must be > 0")
    acc = 0.0

    for i in 1:nd, j in 1:no
        w = W[i, j]
        w == 0.0 && continue

        for k in 1:kmax
            # Integrate |lambda_M - lambda_N|^p over t using trapezoid rule.
            s = 0.0
            for it in 1:nt-1
                dt = tg[it + 1] - tg[it]
                a1 = abs(LA.values[i, j, k, it] - LB.values[i, j, k, it])
                a2 = abs(LA.values[i, j, k, it + 1] - LB.values[i, j, k, it + 1])
                s += 0.5 * (a1^pp + a2^pp) * dt
            end
            acc += w * s
        end
    end

    return acc^(1 / pp)
end

function mp_landscape_distance(
    M::PModule{K},
    N::PModule{K},
    slices::AbstractVector;
    p::Real=2,
    weight_mode::Symbol=:check,
    mp_kwargs...
)::Float64 where {K}
    LM = mp_landscape(M, slices; mp_kwargs...)
    LN = mp_landscape(N, slices; mp_kwargs...)
    return mp_landscape_distance(LM, LN; p=p, weight_mode=weight_mode)
end

function mp_landscape_distance(
    M::PModule{K},
    N::PModule{K},
    pi;
    p::Real=2,
    weight_mode::Symbol=:check,
    mp_kwargs...
)::Float64 where {K}
    cache = SlicePlanCache()
    LM = mp_landscape(M, pi; cache=cache, mp_kwargs...)
    LN = mp_landscape(N, pi; cache=cache, mp_kwargs...)
    return mp_landscape_distance(LM, LN; p=p, weight_mode=weight_mode)
end


"""
    mp_landscape_inner_product(LM, LN; weight_mode=:check) -> Float64

Approximate the L^2 inner product between two multiparameter persistence landscapes:

    <LM, LN> = sum_k int LM_k(t) * LN_k(t) dt,

with the integral approximated by the trapezoidal rule on the common grid and
with slice weights applied.

This is the linear-kernel building block for landscape-based comparisons.
"""
function mp_landscape_inner_product(
    LA::MPLandscape,
    LB::MPLandscape;
    weight_mode::Symbol=:check
)::Float64
    LA.kmax == LB.kmax || error("mp_landscape_inner_product: kmax mismatch")
    size(LA.values) == size(LB.values) || error("mp_landscape_inner_product: values array shape mismatch")
    length(LA.tgrid) == length(LB.tgrid) || error("mp_landscape_inner_product: tgrid length mismatch")
    maximum(abs.(LA.tgrid .- LB.tgrid)) < 1e-12 || error("mp_landscape_inner_product: tgrid mismatch")

    W = _combine_weights(LA, LB, weight_mode)

    nd, no, kmax, nt = size(LA.values)
    tg = LA.tgrid
    acc = 0.0

    for i in 1:nd, j in 1:no
        w = W[i, j]
        w == 0.0 && continue

        for k in 1:kmax
            s = 0.0
            for it in 1:nt-1
                dt = tg[it + 1] - tg[it]
                a1 = LA.values[i, j, k, it] * LB.values[i, j, k, it]
                a2 = LA.values[i, j, k, it + 1] * LB.values[i, j, k, it + 1]
                s += 0.5 * (a1 + a2) * dt
            end
            acc += w * s
        end
    end

    return acc
end


"""
    mp_landscape_kernel(LM, LN; kind=:gaussian, sigma=1.0, p=2, gamma=nothing) -> Float64
    mp_landscape_kernel(M, N, slices; ...) -> Float64
    mp_landscape_kernel(M, N, pi; ...) -> Float64

Kernels derived from the multiparameter persistence landscape.

Supported kinds
---------------
- `kind=:gaussian` (default): `exp(-gamma * d^2)` where `d` is the `mp_landscape_distance`
  with exponent `p` (default `p=2`). If `gamma` is not provided, it uses
  `gamma = 1/(2*sigma^2)`.
- `kind=:laplacian`: `exp(-d/sigma)`.
- `kind=:linear`: L^2 inner product (only meaningful with `p=2`).

These are intended as practical building blocks for statistics and kernel
methods using multiparameter persistence.

Typical workflow:

```julia
LM = mp_landscape(M, pi; directions=dirs, offsets=offs, ts=0:40)
LN = mp_landscape(N, pi; directions=dirs, offsets=offs, ts=0:40)
mp_landscape_kernel(LM, LN; kind=:gaussian, sigma=1.0)
```
"""
function mp_landscape_kernel(
    LA::MPLandscape,
    LB::MPLandscape;
    kind::Symbol=:gaussian,
    sigma::Real=1.0,
    p::Real=2,
    gamma=nothing,
    weight_mode::Symbol=:check
)::Float64
    if kind == :linear
        return mp_landscape_inner_product(LA, LB; weight_mode=weight_mode)
    end

    d = mp_landscape_distance(LA, LB; p=p, weight_mode=weight_mode)

    if kind == :gaussian
        if gamma === nothing
            sigma > 0 || error("mp_landscape_kernel: sigma must be > 0")
            gamma = 1.0 / (2.0 * float(sigma)^2)
        end
        return exp(-float(gamma) * d^2)
    elseif kind == :laplacian
        sigma > 0 || error("mp_landscape_kernel: sigma must be > 0")
        return exp(-d / float(sigma))
    else
        error("mp_landscape_kernel: kind must be :gaussian, :laplacian, or :linear")
    end
end

function mp_landscape_kernel(
    M::PModule{K},
    N::PModule{K},
    slices::AbstractVector;
    kind::Symbol=:gaussian,
    sigma::Real=1.0,
    p::Real=2,
    gamma=nothing,
    weight_mode::Symbol=:check,
    mp_kwargs...
)::Float64 where {K}
    LM = mp_landscape(M, slices; mp_kwargs...)
    LN = mp_landscape(N, slices; mp_kwargs...)
    return mp_landscape_kernel(LM, LN;
                               kind=kind,
                               sigma=sigma,
                               p=p,
                               gamma=gamma,
                               weight_mode=weight_mode)
end

function mp_landscape_kernel(
    M::PModule{K},
    N::PModule{K},
    pi;
    directions=nothing,
    offsets=nothing,
    kind::Symbol=:gaussian,
    sigma::Real=1.0,
    p::Real=2,
    gamma=nothing,
    weight_mode::Symbol=:check,
    mp_kwargs...
)::Float64 where {K}
    directions === nothing && error("mp_landscape_kernel: provide directions=...")
    offsets === nothing && error("mp_landscape_kernel: provide offsets=...")

    cache = SlicePlanCache()
    LM = mp_landscape(M, pi; directions=directions, offsets=offsets, cache=cache, mp_kwargs...)
    LN = mp_landscape(N, pi; directions=directions, offsets=offsets, cache=cache, mp_kwargs...)
    return mp_landscape_kernel(LM, LN;
                               kind=kind,
                               sigma=sigma,
                               p=p,
                               gamma=gamma,
                               weight_mode=weight_mode)
end


function mp_landscape(
    M::PModule{K},
    plan::CompiledSlicePlan;
    kmax::Integer = 10,
    tgrid,
    normalize_weights::Bool = false,
    threads::Bool = (Threads.nthreads() > 1),
)::MPLandscape where {K}
    kk = Int(kmax)
    kk > 0 || throw(ArgumentError("mp_landscape(plan): kmax must be positive"))
    tg = _clean_tgrid(tgrid)
    nt = length(tg)
    nd = plan.nd
    no = plan.no
    ns = nd * no

    vals = zeros(Float64, nd, no, kk, nt)
    build_cache!(M.Q; cover=true, updown=true)
    cc = _get_cover_cache(M.Q)
    nQ = nvertices(M.Q)
    use_array_memo = _use_array_memo(nQ)
    max_chain = isempty(plan.chain_len) ? 0 : maximum(plan.chain_len)
    if threads && Threads.nthreads() > 1
        nT = Threads.nthreads()
        memo_by_thread = use_array_memo ?
            [_new_array_memo(K, nQ) for _ in 1:nT] :
            [Dict{Tuple{Int,Int}, AbstractMatrix{K}}() for _ in 1:nT]
        rank_by_thread = [Matrix{Int}(undef, max_chain, max_chain) for _ in 1:nT]
        point_scratch_by_thread = [Tuple{Float64,Float64}[] for _ in 1:Threads.nthreads()]
        tent_scratch_by_thread = [Float64[] for _ in 1:Threads.nthreads()]
        Threads.@threads for idx in 1:ns
            chain = plan.chains[idx]
            s = plan.vals_start[idx]
            l = plan.vals_len[idx]
            if isempty(chain) || s == 0 || l == 0
                continue
            end
            i = div(idx - 1, no) + 1
            j = (idx - 1) % no + 1
            tid = Threads.threadid()
            endpoints = _extended_values_view(@view(plan.vals_pool[s:s + l - 1]))
            bc = _slice_barcode_packed_with_workspace(
                M,
                chain,
                endpoints,
                cc,
                memo_by_thread[tid],
                rank_by_thread[tid],
            )
            _persistence_landscape_values!(
                @view(vals[i, j, :, :]),
                bc,
                tg;
                points_scratch=point_scratch_by_thread[tid],
                tent_scratch=tent_scratch_by_thread[tid],
            )
        end
    else
        memo = use_array_memo ? _new_array_memo(K, nQ) : Dict{Tuple{Int,Int}, AbstractMatrix{K}}()
        rank_work = Matrix{Int}(undef, max_chain, max_chain)
        points_scratch = Tuple{Float64,Float64}[]
        tent_scratch = Float64[]
        @inbounds for i in 1:nd, j in 1:no
            idx = _plan_idx(no, i, j)
            chain = plan.chains[idx]
            s = plan.vals_start[idx]
            l = plan.vals_len[idx]
            if isempty(chain) || s == 0 || l == 0
                continue
            end
            endpoints = _extended_values_view(@view(plan.vals_pool[s:s + l - 1]))
            bc = _slice_barcode_packed_with_workspace(
                M,
                chain,
                endpoints,
                cc,
                memo,
                rank_work,
            )
            _persistence_landscape_values!(
                @view(vals[i, j, :, :]),
                bc,
                tg;
                points_scratch=points_scratch,
                tent_scratch=tent_scratch,
            )
        end
    end

    W = copy(plan.weights)
    if normalize_weights
        s = sum(W)
        s > 0 || error("mp_landscape(plan): total slice weight is zero")
        W ./= s
    end

    return MPLandscape(kk, tg, vals, W, copy(plan.dirs), copy(plan.offs))
end

end # module MultiparameterImages
