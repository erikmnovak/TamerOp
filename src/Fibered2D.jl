module Fibered2D
# -----------------------------------------------------------------------------
# Fibered2D.jl
#
# Fibered2D owner module extracted from Invariants.jl.
# -----------------------------------------------------------------------------

"""
    Fibered2D

Owner module for exact 2D fibered/matching-distance machinery and repeated
slice/barcode query structures in two parameters.
"""

using LinearAlgebra
using JSON3
using ..CoreModules: EncodingCache, AbstractCoeffField, RegionPosetCachePayload,
                     AbstractSlicePlanCache
using ..Options: InvariantOptions
import ..DataTypes: ambient_dim
using ..EncodingCore: PLikeEncodingMap, CompiledEncoding, locate, axes_from_encoding, dimension, representatives
using Statistics: mean
using ..Stats: _wilson_interval
using ..Encoding: EncodingMap
import ..Encoding: source_poset
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
                       FloatBarcode, IndexBarcode,
                       _empty_index_barcode, _empty_float_barcode,
                       EndpointPair,
                       PackedBarcode, PackedIndexBarcode, PackedFloatBarcode, PackedBarcodeGrid,
                       _empty_packed_index_barcode, _empty_packed_float_barcode,
                       _packed_grid_undef, _packed_grid_from_matrix,
                       _packed_total_multiplicity,
                       _to_float_barcode,
                       _pack_index_barcode, _pack_float_barcode,
                       _points_from_packed!, _points_from_index_packed_and_values!,
                       _packed_barcode_from_rank,
                       _barcode_from_packed,
                       _float_barcode_from_index_packed_values,
                       _float_packed_from_index_packed_values,
                       _float_dict_matrix_from_packed_grid,
                       _index_dict_matrix_from_packed_grid,
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
import ..ChainComplexes: describe
import ..DerivedFunctors: source_module
import Base.Threads
import ..Serialization: save_mpp_decomposition_json, load_mpp_decomposition_json,
                        save_mpp_image_json, load_mpp_image_json
import ..Modules: PModule, map_leq, CoverCache, _get_cover_cache
import ..IndicatorResolutions: pmodule_from_fringe
import ..ZnEncoding: ZnEncodingMap
import ..SliceInvariants: SliceBarcodesResult, nslices,
                          slice_chain, slice_barcodes, slice_kernel, slice_barcode,
                          direction_weight, _direction_weight, encoding_box, window_box,
                          default_directions, _extend_values,
                          _offset_sample_weights,
                          bottleneck_distance, wasserstein_distance, _barcode_kernel

@inline _resolve_box(pi, box) = (box === :auto ? window_box(pi) : box)
@inline _invariants_module() = getfield(parentmodule(@__MODULE__), :Invariants)
@inline _region_poset(pi; kwargs...) = getfield(_invariants_module(), :region_poset)(pi; kwargs...)
@inline _region_values(pi, f; kwargs...) = getfield(_invariants_module(), :region_values)(pi, f; kwargs...)
@inline _nregions_encoding(pi) = getfield(_invariants_module(), :_nregions_encoding)(pi)

# -----------------------------------------------------------------------------
# Exact 2D matching distance (deterministic, no RNG)
#
# This implements the screenshot algorithm:
#   - Collect critical points (grid vertices or polyhedral vertices) + box corners.
#   - Build critical slopes from dy/dx over pairs, then use midpoints between slopes
#     as representative directions (optionally include axes).
#   - For each direction, compute critical offsets c_i = dot(n, p_i) with
#     n = [-dir[2], dir[1]], and use midpoints between consecutive c_i as
#     representative offsets.
#   - For each (dir, offset) representative, extract the slice chain exactly:
#       * coords-based backend: intersect with all vertical/horizontal grid lines.
#       * polyhedral backend: compute all region t-interval endpoints, then label
#         segments via locate at midpoints.
#   - Compute max over slices of w(dir) * bottleneck_distance(slice_barcode(...)).
#
# Notes:
#   - Values passed to slice_barcode are boundary parameter values (length n+1).
#   - Determinism is enforced by sorting and stable de-duplication.
#   - For speed, we avoid sampling and use O(K^2) slope enumeration over critical points,
#     which is feasible for modest K (typical in exact use).
# -----------------------------------------------------------------------------

# In-place unique of a sorted float vector using an absolute tolerance.
function _unique_sorted_floats!(v::Vector{Float64}; atol::Float64=1e-12)
    sort!(v)
    n = length(v)
    if n <= 1
        return v
    end
    j = 1
    last = v[1]
    @inbounds for i in 2:n
        x = v[i]
        if abs(x - last) > atol
            j += 1
            v[j] = x
            last = x
        end
    end
    resize!(v, j)
    return v
end

# In-place unique of a sorted vector of points (x,y) with lexicographic sorting.
function _unique_points_2d!(pts::Vector{NTuple{2,Float64}}; atol::Float64=1e-12)
    sort!(pts, by=p->(p[1], p[2]))
    n = length(pts)
    if n <= 1
        return pts
    end
    j = 1
    last = pts[1]
    @inbounds for i in 2:n
        p = pts[i]
        if (abs(p[1]-last[1]) > atol) || (abs(p[2]-last[2]) > atol)
            j += 1
            pts[j] = p
            last = p
        end
    end
    resize!(pts, j)
    return pts
end

# Basepoint x0 on the line with normal offset:
#   n = [-dir[2], dir[1]]
#   dot(n, x0) = offset
# We choose x0 = (offset / ||n||^2) * n, the projection of the origin onto the line.
function _line_basepoint_from_normal_offset_2d(dir::Vector{Float64}, offset::Float64)
    n1 = -dir[2]
    n2 =  dir[1]
    nn = n1*n1 + n2*n2
    if nn == 0.0
        error("slice_chain_exact_2d: direction vector must be nonzero")
    end
    s = offset / nn
    return [s*n1, s*n2]
end

# Parameter interval [tmin, tmax] for which x(t)=x0+t*dir lies inside axis-aligned box (a,b).
function _line_param_range_in_box_2d(x0::Vector{Float64}, dir::Vector{Float64},
                                     box::Tuple{Vector{Float64},Vector{Float64}};
                                     atol::Float64=1e-12)
    a, b = box
    tmin = -Inf
    tmax =  Inf
    @inbounds for k in 1:2
        dk = dir[k]
        ak = a[k]
        bk = b[k]
        if dk == 0.0
            # Coordinate is constant: must already lie in [a,b].
            if (x0[k] < min(ak,bk) - atol) || (x0[k] > max(ak,bk) + atol)
                return (Inf, -Inf)  # empty
            end
        else
            t1 = (ak - x0[k]) / dk
            t2 = (bk - x0[k]) / dk
            lo = min(t1, t2)
            hi = max(t1, t2)
            tmin = max(tmin, lo)
            tmax = min(tmax, hi)
        end
    end
    return (tmin, tmax)
end

# Convert a list of segment labels and boundary times into a compressed chain and boundary list.
# boundaries has length = length(labels)+1.
function _chain_values_from_boundaries(labels::Vector{Int}, boundaries::Vector{Float64};
                                       strict::Bool=true, atol::Float64=1e-12)
    if isempty(labels)
        return Int[], Float64[]
    end
    chain = Int[]
    values = Float64[]
    sizehint!(chain, length(labels))
    sizehint!(values, length(labels)+1)

    push!(values, boundaries[1])
    @inbounds for i in 1:length(labels)
        q = labels[i]
        if q == 0
            if strict
                error("slice_chain_exact_2d: encountered region label 0 inside the requested box; " *
                      "shrink box or pass strict=false")
            else
                # Non-strict behavior: skip. Note that interior 0-segments will be glued.
                continue
            end
        end
        if isempty(chain)
            push!(chain, q)
        elseif q != chain[end]
            push!(values, boundaries[i])   # boundary between segments i-1 and i
            push!(chain, q)
        end
    end
    push!(values, boundaries[end])

    return chain, values
end

# Critical points for coords-based (axis-aligned) backend:
# all grid vertices coords[1] x coords[2] (finite, inside box) + box corners.
function _critical_points_boxes_2d(pi, box::Tuple{Vector{Float64},Vector{Float64}}; atol::Float64=1e-12)
    a, b = box
    coords = getproperty(pi, :coords)
    xs_all = coords[1]
    ys_all = coords[2]

    xmin = min(a[1], b[1]) - atol
    xmax = max(a[1], b[1]) + atol
    ymin = min(a[2], b[2]) - atol
    ymax = max(a[2], b[2]) + atol

    xs = Float64[]
    ys = Float64[]
    for c in xs_all
        if isfinite(c)
            cf = float(c)
            if (cf >= xmin) && (cf <= xmax)
                push!(xs, cf)
            end
        end
    end
    for c in ys_all
        if isfinite(c)
            cf = float(c)
            if (cf >= ymin) && (cf <= ymax)
                push!(ys, cf)
            end
        end
    end

    pts = NTuple{2,Float64}[]
    sizehint!(pts, length(xs)*length(ys) + 4)
    for x in xs, y in ys
        push!(pts, (x,y))
    end
    # box corners
    push!(pts, (a[1],a[2]))
    push!(pts, (a[1],b[2]))
    push!(pts, (b[1],a[2]))
    push!(pts, (b[1],b[2]))

    _unique_points_2d!(pts; atol=atol)
    return pts
end

# Critical points for polyhedral backend:
# all vertices of each region inside box (when enumerable) + box corners.
function _critical_points_poly_2d(pi::PLPolyhedra.PLEncodingMap,
                                  box::Tuple{Vector{Float64},Vector{Float64}};
                                  max_combinations::Int=200000,
                                  max_vertices::Int=20000,
                                  atol::Float64=1e-12)
    a, b = box
    pts = NTuple{2,Float64}[]
    sizehint!(pts, 32)

    # Collect region vertices in the box. This is deterministic up to sorting below.
    for hp in pi.regions
        verts = PLPolyhedra._vertices_of_hpoly_in_box(hp, a, b;
                                                     max_combinations=max_combinations,
                                                     max_vertices=max_vertices)
        if verts !== nothing
            for v in verts
                # v is a tuple of scalars (dimension 2 here).
                push!(pts, (float(v[1]), float(v[2])))
            end
        else
            # Fallback: include witness points if enumeration fails.
            # This does not guarantee exactness, but avoids crashes.
            # The exact algorithm is intended for cases where vertex enumeration succeeds.
            # (Witnesses are Float64 already.)
        end
    end

    # Add corners to ensure offset range covers box.
    push!(pts, (a[1],a[2]))
    push!(pts, (a[1],b[2]))
    push!(pts, (b[1],a[2]))
    push!(pts, (b[1],b[2]))

    _unique_points_2d!(pts; atol=atol)
    return pts
end

# Representative slopes: given sorted unique slopes s1<s2<...<sk, return midpoints of
# intervals (0,s1), (s1,s2), ..., (sk, +inf) as finite representatives.
function _representative_slopes(slopes::Vector{Float64}; atol::Float64=1e-12)
    _unique_sorted_floats!(slopes; atol=atol)
    if isempty(slopes)
        return [1.0]  # degenerate fallback
    end
    reps = Float64[]
    sizehint!(reps, length(slopes)+1)

    # (0, s1)
    push!(reps, 0.5*slopes[1])

    # (si, s(i+1))
    for i in 1:length(slopes)-1
        push!(reps, 0.5*(slopes[i] + slopes[i+1]))
    end

    # (sk, +inf) -> pick 2*sk
    push!(reps, 2.0*slopes[end])

    return reps
end

# Compute representative directions (normalized) from critical points.
function _critical_directions_2d(points::Vector{NTuple{2,Float64}};
                                 normalize_dirs::Symbol=:L1,
                                 include_axes::Bool=false,
                                 atol::Float64=1e-12)
    # Sort points by x so dx>=0 for j>i.
    pts = copy(points)
    sort!(pts, by=p->(p[1], p[2]))

    slopes = Float64[]
    n = length(pts)
    sizehint!(slopes, max(0, n*(n-1)>>>2))
    @inbounds for i in 1:n-1
        xi, yi = pts[i]
        for j in i+1:n
            xj, yj = pts[j]
            dx = xj - xi
            dy = yj - yi
            if dx > atol && dy > atol
                push!(slopes, dy/dx)
            end
        end
    end

    reps = _representative_slopes(slopes; atol=atol)

    dirs = Vector{Vector{Float64}}()
    sizehint!(dirs, length(reps) + (include_axes ? 2 : 0))

    for s in reps
        # direction with slope s in positive quadrant
        d = [1.0, s]
        d = _normalize_dir(d, normalize_dirs)
        # skip zero directions
        if (d[1] > 0.0) && (d[2] > 0.0)
            push!(dirs, d)
        end
    end

    if include_axes
        # normalized axis directions (robustly handled in slice extraction)
        push!(dirs, _normalize_dir([1.0, 0.0], normalize_dirs))
        push!(dirs, _normalize_dir([0.0, 1.0], normalize_dirs))
    end

    return dirs
end

# For a fixed direction, compute representative offsets (midpoints between consecutive dot products).
function _critical_offsets_2d(points::Vector{NTuple{2,Float64}}, dir::NTuple{2,Float64};
                              atol::Float64=1e-12)
    n1 = -dir[2]
    n2 =  dir[1]
    cs = Float64[]
    sizehint!(cs, length(points))
    for p in points
        push!(cs, n1*p[1] + n2*p[2])
    end
    _unique_sorted_floats!(cs; atol=atol)
    if length(cs) < 2
        return Float64[]
    end
    offs = Float64[]
    sizehint!(offs, length(cs)-1)
    for i in 1:length(cs)-1
        push!(offs, 0.5*(cs[i] + cs[i+1]))
    end
    return offs
end

function _direction_representatives(slopes; normalize_dirs::Symbol=:L1, include_axes::Bool=false, atol::Float64=1e-12)
    reps = _representative_slopes(Float64.(slopes); atol=atol)
    dirs = Vector{NTuple{2,Float64}}()
    sizehint!(dirs, length(reps) + (include_axes ? 2 : 0))

    for s in reps
        d = _normalize_dir((1.0, s), normalize_dirs)
        if d[1] > 0.0 && d[2] > 0.0
            push!(dirs, d)
        end
    end

    if include_axes
        push!(dirs, _normalize_dir((1.0, 0.0), normalize_dirs))
        push!(dirs, _normalize_dir((0.0, 1.0), normalize_dirs))
    end

    return dirs
end

function _line_order(points::Vector{NTuple{2,Float64}}, dir::NTuple{2,Float64};
                     strict::Bool=true, atol::Float64=1e-12)
    n1 = -dir[2]
    n2 =  dir[1]
    idx = collect(1:length(points))
    sort!(idx, by=i->(n1*points[i][1] + n2*points[i][2], points[i][1], points[i][2]))
    return idx
end

@inline function _line_order(points::Vector{NTuple{2,Float64}}, dir::Vector{Float64};
                             strict::Bool=true, atol::Float64=1e-12)
    length(dir) == 2 || throw(ArgumentError("_line_order: expected 2D direction"))
    return _line_order(points, (float(dir[1]), float(dir[2])); strict=strict, atol=atol)
end

function _unique_positions_for_order(points::Vector{NTuple{2,Float64}}, order::Vector{Int},
                                     dir::NTuple{2,Float64}; atol::Float64=1e-12)
    n1 = -dir[2]
    n2 =  dir[1]
    pos = Int[]
    lastc = NaN
    for (k, idx) in enumerate(order)
        c = n1*points[idx][1] + n2*points[idx][2]
        if isempty(pos) || abs(c - lastc) > atol
            push!(pos, k)
            lastc = c
        end
    end
    return pos
end

@inline function _unique_positions_for_order(points::Vector{NTuple{2,Float64}}, order::Vector{Int},
                                             dir::Vector{Float64}; atol::Float64=1e-12)
    length(dir) == 2 || throw(ArgumentError("_unique_positions_for_order: expected 2D direction"))
    return _unique_positions_for_order(points, order, (float(dir[1]), float(dir[2])); atol=atol)
end

# Exact slice-chain extraction for coords-based backend.
function _slice_chain_exact_boxes_2d(pi, dir::Vector{Float64}, offset::Float64;
                                     box::Tuple{Vector{Float64},Vector{Float64}},
                                     strict::Bool=true, atol::Float64=1e-12)
    x0 = _line_basepoint_from_normal_offset_2d(dir, offset)
    tmin, tmax = _line_param_range_in_box_2d(x0, dir, box; atol=atol)
    if !(tmin < tmax)
        return Int[], Float64[]
    end

    a, b = box
    xmin = min(a[1], b[1])
    xmax = max(a[1], b[1])
    ymin = min(a[2], b[2])
    ymax = max(a[2], b[2])

    coords = getproperty(pi, :coords)
    xs = coords[1]
    ys = coords[2]

    ts = Float64[tmin, tmax]

    # Intersections with vertical grid lines x=c.
    if dir[1] != 0.0
        for c in xs
            if isfinite(c)
                cf = float(c)
                if (cf > xmin + atol) && (cf < xmax - atol)
                    t = (cf - x0[1]) / dir[1]
                    if (t > tmin + atol) && (t < tmax - atol)
                        push!(ts, t)
                    end
                end
            end
        end
    end

    # Intersections with horizontal grid lines y=c.
    if dir[2] != 0.0
        for c in ys
            if isfinite(c)
                cf = float(c)
                if (cf > ymin + atol) && (cf < ymax - atol)
                    t = (cf - x0[2]) / dir[2]
                    if (t > tmin + atol) && (t < tmax - atol)
                        push!(ts, t)
                    end
                end
            end
        end
    end

    _unique_sorted_floats!(ts; atol=atol)

    # Label each open segment by a midpoint query.
    labels = Vector{Int}(undef, length(ts)-1)
    x = [0.0, 0.0]
    @inbounds for i in 1:length(labels)
        tmid = 0.5*(ts[i] + ts[i+1])
        x[1] = x0[1] + tmid*dir[1]
        x[2] = x0[2] + tmid*dir[2]
        labels[i] = locate(pi, x)
    end

    return _chain_values_from_boundaries(labels, ts; strict=strict, atol=atol)
end

# Compute line-interval intersection with a single convex polyhedron Ax<=b.
function _line_interval_in_hpoly_2d(A::Matrix{Float64}, b::Vector{Float64},
                                    x0::Vector{Float64}, dir::Vector{Float64};
                                    atol::Float64=1e-12)
    tlo = -Inf
    thi =  Inf
    m = size(A,1)
    @inbounds for i in 1:m
        a1 = A[i,1]
        a2 = A[i,2]
        alpha = a1*dir[1] + a2*dir[2]
        beta  = b[i] - (a1*x0[1] + a2*x0[2])

        if abs(alpha) <= atol
            if beta < -atol
                return (Inf, -Inf)
            end
        elseif alpha > 0.0
            thi = min(thi, beta/alpha)
        else
            tlo = max(tlo, beta/alpha)
        end

        if tlo >= thi - atol
            return (Inf, -Inf)
        end
    end
    return (tlo, thi)
end

# Exact slice-chain extraction for polyhedral backend.
# We compute all region interval endpoints, sort them, then label segments by locate at midpoints.
function _slice_chain_exact_poly_2d(pi::PLPolyhedra.PLEncodingMap,
                                    dir::Vector{Float64}, offset::Float64;
                                    box::Tuple{Vector{Float64},Vector{Float64}},
                                    strict::Bool=true, atol::Float64=1e-12,
                                    region_cache=nothing)
    x0 = _line_basepoint_from_normal_offset_2d(dir, offset)
    tmin, tmax = _line_param_range_in_box_2d(x0, dir, box; atol=atol)
    if !(tmin < tmax)
        return Int[], Float64[]
    end

    # Precompute float A,b per region for speed if provided.
    A_list, b_list = if region_cache === nothing
        ([Float64.(hp.A) for hp in pi.regions],
         [Float64.(hp.b) for hp in pi.regions])
    else
        region_cache
    end

    ts = Float64[tmin, tmax]
    for r in 1:length(pi.regions)
        A = A_list[r]
        b = b_list[r]
        tlo, thi = _line_interval_in_hpoly_2d(A, b, x0, dir; atol=atol)
        tlo = max(tlo, tmin)
        thi = min(thi, tmax)
        if thi > tlo + atol
            push!(ts, tlo)
            push!(ts, thi)
        end
    end

    _unique_sorted_floats!(ts; atol=atol)

    labels = Vector{Int}(undef, length(ts)-1)
    x = [0.0, 0.0]
    @inbounds for i in 1:length(labels)
        tmid = 0.5*(ts[i] + ts[i+1])
        x[1] = x0[1] + tmid*dir[1]
        x[2] = x0[2] + tmid*dir[2]
        labels[i] = locate(pi, x)
    end

    return _chain_values_from_boundaries(labels, ts; strict=strict, atol=atol)
end

"""
    slice_chain_exact_2d(pi, dir, offset, opts::InvariantOptions;
        normalize_dirs=:L1,
        atol=1e-12)

Compute an *exact* 2D slice chain by intersecting the slicing line with the
arrangement induced by representative points.

Opts usage:
- `opts.box` provides the working 2D box. If `opts.box === nothing`, we use
  `encoding_box(pi, InvariantOptions())` (default behavior).
- `opts.strict` controls locate strictness (defaults to true).

We consider the line with direction `dir` and normal offset `offset`:
  - Normalize `dir` according to `normalize_dirs`.
  - Define the normal `n = [-dir[2], dir[1]]`.
  - Choose basepoint `x0 = (offset/||n||^2) * n` so that `dot(n, x0) = offset`.
  - Consider the line segment inside `box` (an axis-aligned box given as `(a,b)`).

Return:
  - `chain::Vector{Int}`: region labels along the segment, merged by constant regions.
  - `values::Vector{Float64}`: boundary parameter values of length `length(chain)+1`.

Backends:
  - If `pi` has a `coords` field, we treat regions as an axis-aligned grid and intersect
    the line with all vertical/horizontal grid lines.
  - If `pi isa PLPolyhedra.PLEncodingMap`, we intersect the line with each region polyhedron.

Returns `(chain, tvals)` where `tvals` are the exact parameter values where the
line crosses arrangement events.
"""
function slice_chain_exact_2d(
    pi::PLikeEncodingMap,
    dir::AbstractVector{<:Real},
    offset::Real,
    opts::InvariantOptions;
    normalize_dirs::Symbol = :L1,
    atol::Real = 1e-12
)
    pi0 = _unwrap_compiled(pi)
    strict0 = opts.strict === nothing ? true : opts.strict

    # Legacy default for this exact routine: if opts.box is unset, use encoding_box(pi).
    # (Note: encoding_box itself supports :auto if the caller explicitly requests it.)
    bx_raw = (opts.box === nothing ? encoding_box(pi, InvariantOptions()) : _resolve_box(pi, opts.box))
    bx = ([float(bx_raw[1][1]), float(bx_raw[1][2])],
          [float(bx_raw[2][1]), float(bx_raw[2][2])])

    # Normalize direction representation.
    d = _normalize_dir(float.(dir), normalize_dirs)

    if hasproperty(pi0, :coords)
        coords = getproperty(pi0, :coords)
        xs = [float(x) for x in coords[1] if isfinite(x)]
        ys = [float(y) for y in coords[2] if isfinite(y)]
        return _slice_chain_exact_boxes_2d_fast(
            pi0, d, float(offset);
            box = bx,
            xs = xs,
            ys = ys,
            strict = strict0,
            atol = atol
        )
    elseif pi0 isa PLPolyhedra.PLEncodingMap
        return _slice_chain_exact_poly_2d(
            pi0, d, float(offset);
            box = bx,
            strict = strict0,
            atol = atol
        )
    end

    error("slice_chain_exact_2d: unsupported encoding map backend $(typeof(pi0))")
end


"""
    matching_distance_exact_slices_2d(pi, opts::InvariantOptions;
        normalize_dirs=:L1,
        include_axes=false,
        atol=1e-12)

Deterministically enumerate the slice representatives used in the exact 2D matching distance.

Opts usage:
- `opts.box` provides the working 2D box. If unset (`nothing`), we use `encoding_box(pi, InvariantOptions())`
  (default behavior).
- `opts.strict` is forwarded to `locate` as needed (defaults to true).

Returns a vector of NamedTuples with fields:
  - `dir`    (Vector{Float64}, length 2)
  - `offset` (Float64), the normal offset c = dot(n, x) with n = [-dir[2], dir[1]]
  - `chain`  (Vector{Int})
  - `values` (Vector{Float64}, length chain+1)
  - `weight` (Float64)

One representative is produced per arrangement cell determined by the critical points.
"""
function matching_distance_exact_slices_2d(
    pi::PLikeEncodingMap,
    opts::InvariantOptions;
    normalize_dirs::Symbol = :L1,
    include_axes::Bool = false,
    atol::Real = 1e-12
)
    arr = fibered_arrangement_2d(pi, opts;
        normalize_dirs = normalize_dirs,
        include_axes = include_axes,
        atol = atol,
        precompute = :cells
    )

    slices = NamedTuple[]
    offsets_by_dir = Vector{Vector{Float64}}(undef, length(arr.dir_reps))

    for di in 1:length(arr.dir_reps)
        d = arr.dir_reps[di]
        w = _direction_weight(d; scheme=:lesnick_l1)
        offs = Float64[]
        for oi in 1:arr.noff[di]
            cell = _arr2d_cell_linear_index(arr, di, oi)
            cid = arr.cell_chain_id[cell]
            cid > 0 || continue
            _, _, cmid = _arr2d_cell_offset_interval(arr, di, oi)
            chain, vals = _arr2d_slice_chain_and_values(arr, d, cmid)
            isempty(chain) && continue
            push!(offs, cmid)
            push!(slices, (chain = chain, values = vals, weight = w))
        end
        offsets_by_dir[di] = offs
    end

    return (
        box = arr.box,
        strict = arr.strict,
        normalize_dirs = normalize_dirs,
        include_axes = include_axes,
        slopes = arr.slope_breaks,
        directions = arr.dir_reps,
        offsets_by_dir = offsets_by_dir,
        slices = slices,
    )
end
function _slice_barcode_packed(
    M::PModule{K},
    chain::AbstractVector{Int};
    values=nothing,
    check_chain::Bool=true
) where {K}
    check_chain && _assert_chain(M.Q, chain)
    m = length(chain)

    endpoints = if values === nothing
        1:(m+1)
    else
        length(values) == m || length(values) == m + 1 ||
            error("slice_barcode: values must have length m or m+1")
        length(values) == m ? _extend_values(values) : values
    end

    cc = _get_cover_cache(M.Q)
    nQ = nvertices(M.Q)
    memo = _use_array_memo(nQ) ? _new_array_memo(K, nQ) : Dict{Tuple{Int,Int}, AbstractMatrix{K}}()
    R = zeros(Int, m, m)
    @inbounds for i in 1:m
        for j in i:m
            R[i, j] = rank_map(M, chain[i], chain[j]; cache=cc, memo=memo)
        end
    end

    return _packed_barcode_from_rank(R, endpoints)
end

@inline _barcode_points(bar::PackedFloatBarcode)::Vector{Tuple{Float64,Float64}} =
    _points_from_packed!(Tuple{Float64,Float64}[], bar)

@inline function _barcode_points(bar::PackedIndexBarcode)::Vector{Tuple{Float64,Float64}}
    out = Tuple{Float64,Float64}[]
    sizehint!(out, _packed_total_multiplicity(bar))
    @inbounds for i in eachindex(bar.pairs)
        p = bar.pairs[i]
        m = bar.mults[i]
        b = float(p.b)
        d = float(p.d)
        for _ in 1:m
            push!(out, (b, d))
        end
    end
    return out
end

@inline function _values_are_float_vector(v)
    v isa AbstractVector || return false
    @inbounds for x in v
        x isa AbstractFloat || return false
    end
    return true
end

@inline function _values_are_int_vector(v)
    v isa AbstractVector || return false
    @inbounds for x in v
        x isa Integer || return false
    end
    return true
end

"""
Thread-local reusable scratch arena for invariant kernels.
"""
mutable struct InvariantScratch
    points_a::Vector{Tuple{Float64,Float64}}
    points_b::Vector{Tuple{Float64,Float64}}
    intervals::Vector{Tuple{Float64,Float64}}
    mids::Vector{Float64}
    perm::Vector{Int}
    values::Vector{Float64}
end

InvariantScratch() = InvariantScratch(
    Tuple{Float64,Float64}[],
    Tuple{Float64,Float64}[],
    Tuple{Float64,Float64}[],
    Float64[],
    Int[],
    Float64[],
)

@inline _scratch_arenas(threads::Bool) =
    [InvariantScratch() for _ in 1:(threads ? Threads.maxthreadid() : 1)]

# ----- Fast fibered-barcode queries in 2D: RIVET-style augmented arrangement -----
#
# RIVET's key idea:
#   In 2D, the slice chain (and therefore the index barcode) is constant on the
#   2-cells of a line arrangement in (direction, offset) space.
#
# This section implements a mathematician-friendly and reusable cache:
#   - FiberedArrangement2D: geometry-only point-location structure + per-cell chain/events
#   - FiberedBarcodeCache2D: module-augmented cache of index barcodes per chain
#
# Query time (typical, nondegenerate):
#   O(log N + m + |barcode|)
# where m = chain length of the slice (usually small/moderate).
#
# This provides the "fast repeated queries" capability requested in the screenshot.

const _AxisCoordIndex2D = NTuple{2,Int}

struct FiberedSliceFamilyKey
    direction_weight::Symbol
    store_values::Bool
end

mutable struct _FiberedFamilyBarcodePayload2D
    key::FiberedSliceFamilyKey
    packed_barcodes::Vector{Union{Nothing,PackedFloatBarcode}}
    n_computed::Int
end

mutable struct _FiberedDistancePayload2D
    key::FiberedSliceFamilyKey
    points::Vector{Union{Nothing,Vector{Tuple{Float64,Float64}}}}
    n_computed::Int
end

"""
    FiberedArrangement2D

Geometry-only cache for fast point-location of 2D lines in the space of
(direction, normal offset).  It stores:

- critical slopes (direction changes where order of offsets changes)
- for each direction cell, a fixed ordering of critical points by normal offset
- for each (direction cell, offset cell), a cached slice chain (computed lazily or eagerly)
- for the boxes backend, a compact event description for reconstructing boundary parameter
  values quickly for any query line in the same cell

This object is independent of any module `M` and can be shared between many modules.

Construct with [`fibered_arrangement_2d`](@ref).
Augment with [`fibered_barcode_cache_2d`](@ref).
"""
mutable struct FiberedArrangement2D{PI,RC,SFC}
    pi::PI
    box::Tuple{Vector{Float64},Vector{Float64}}
    normalize_dirs::Symbol
    include_axes::Bool
    strict::Bool
    atol::Float64

    backend::Symbol
    points::Vector{NTuple{2,Float64}}
    slope_breaks::Vector{Float64}
    dir_reps::Vector{Vector{Float64}}

    # For each direction cell:
    # - orders[k] is a permutation of critical points sorted by normal offset
    # - unique_pos[k] are positions in that ordering representing unique offset levels
    orders::Vector{Vector{Int}}
    unique_pos::Vector{Vector{Int}}

    # number of offset cells per direction cell
    noff::Vector{Int}

    # prefix starts for flattened storage
    start::Vector{Int}
    total_cells::Int

    # Per-cell cache:
    #  0 = not computed yet
    # -1 = empty slice
    # >0 = chain_id
    cell_chain_id::Vector{Int}

    # Boxes backend events are stored in a pool; each cell stores (start,len)
    cell_event_start::Vector{Int}
    cell_event_len::Vector{Int}
    event_pool::Vector{_AxisCoordIndex2D}

    # coordinate lists for boxes backend (filtered to box)
    xs::Vector{Float64}
    ys::Vector{Float64}

    # poly backend cache: (A_list, b_list)
    region_cache::RC

    # chain registry
    chain_key_to_id::Dict{Vector{Int},Int}
    chains::Vector{Vector{Int}}

    # stats: how many cells have been computed
    n_cell_computed::Int

    # Cache of precomputed slice families keyed by typed options.
    slice_family_cache::SFC
end

"""
    FiberedBarcodeCache2D

Module-augmented cache on top of [`FiberedArrangement2D`](@ref).
It memoizes the *index barcode* for each chain encountered so that repeated
fibered barcode queries are fast.

Construct with [`fibered_barcode_cache_2d`](@ref).
Query with [`fibered_barcode`](@ref).
"""
mutable struct FiberedBarcodeCache2D{K,FBP,DPP}
    arrangement::FiberedArrangement2D
    M::PModule{K}
    # Packed barcodes are the internal default for all hot loops.
    index_barcodes_packed::Vector{Union{Nothing,PackedIndexBarcode}}
    n_barcode_computed::Int
    family_barcode_payloads::FBP
    distance_payloads::DPP
end

"""
    FiberedSliceResult

Typed owner-level result wrapper for one exact 2D fibered slice query.

This stores:
- `chain`: the slice chain in the arrangement/source poset,
- `values`: the boundary parameter values aligned with that chain,
- `barcode`: the 1D barcode of the queried slice.

Mathematically, this is the full 1-parameter restriction of a module along one
chosen line in the exact 2D fibered arrangement, together with the barcode of
that restricted module. Use [`fibered_query_summary`](@ref) when you only need
cell-selection and cardinality diagnostics for a single query. Use
`FiberedSliceResult` when you actually need the chain, boundary values, or
barcode itself.

Use [`slice_chain`](@ref), [`slice_values`](@ref), and [`slice_barcode`](@ref)
instead of reading fields directly. Prefer `describe(result)` for cheap-first
inspection before unpacking the full chain/value/barcode payload.
"""
struct FiberedSliceResult{C,V,B}
    chain::C
    values::V
    barcode::B
end

@inline _fibered_slice_result(chain, values, barcode) = FiberedSliceResult(chain, values, barcode)
@inline _empty_fibered_slice_result() =
    _fibered_slice_result(Int[], Float64[], Dict{Tuple{Float64,Float64},Int}())

# ------------------------ internal helpers ------------------------

@inline function _as_float2(v)::Vector{Float64}
    if length(v) != 2
        throw(ArgumentError("expected a 2-vector"))
    end
    return [Float64(v[1]), Float64(v[2])]
end

@inline function _normal_from_dir_2d(d::Union{AbstractVector{<:Real},NTuple{2,<:Real}})
    return (-float(d[2]), float(d[1]))
end

function _critical_slopes_2d(points::Vector{NTuple{2,Float64}}; atol::Float64=1e-12)
    n = length(points)
    slopes = Float64[]
    for i in 1:n-1
        xi, yi = points[i]
        for j in i+1:n
            xj, yj = points[j]
            dx = xj - xi
            dy = yj - yi
            if dx > atol && dy > atol
                push!(slopes, dy/dx)
            end
        end
    end
    sort!(slopes)
    _unique_sorted_floats!(slopes; atol=atol)
    return slopes
end

function _unique_sorted_slopes(points; atol::Float64=1e-12)
    pts = NTuple{2,Float64}[]
    sizehint!(pts, length(points))
    for p in points
        length(p) == 2 || throw(ArgumentError("expected 2D points for slope computation"))
        push!(pts, (float(p[1]), float(p[2])))
    end
    return _critical_slopes_2d(pts; atol=atol)
end

function _fibered_dir_cell_index(arr::FiberedArrangement2D, d::Vector{Float64})::Int
    if d[1] < -arr.atol || d[2] < -arr.atol
        throw(ArgumentError("direction must lie in the nonnegative quadrant"))
    end
    if abs(d[1]) <= arr.atol && abs(d[2]) <= arr.atol
        throw(ArgumentError("direction must be nonzero"))
    end

    ndir_pos = length(arr.slope_breaks) + 1

    if arr.include_axes
        if abs(d[2]) <= arr.atol && d[1] > arr.atol
            return ndir_pos + 1
        elseif abs(d[1]) <= arr.atol && d[2] > arr.atol
            return ndir_pos + 2
        end
    else
        if abs(d[1]) <= arr.atol || abs(d[2]) <= arr.atol
            throw(ArgumentError("axis directions require include_axes=true"))
        end
    end

    if d[1] <= arr.atol || d[2] <= arr.atol
        throw(ArgumentError("direction must have strictly positive entries unless include_axes=true"))
    end

    slope = d[2] / d[1]
    k = searchsortedfirst(arr.slope_breaks, slope)
    return k
end

function _fibered_offset_cell_index(
    arr::FiberedArrangement2D,
    dir_idx::Int,
    d::Union{AbstractVector{<:Real},NTuple{2,<:Real}},
    off::Float64;
    tie_break::Symbol = :up,
)
    tie_break === :center && (tie_break = :up)
    pos = arr.unique_pos[dir_idx]
    order = arr.orders[dir_idx]
    pts = arr.points
    (n1, n2) = _normal_from_dir_2d(d)

    nu = length(pos)
    if nu < 2
        return 0
    end

    pmin = pts[order[pos[1]]]
    pmax = pts[order[pos[end]]]
    cmin = n1*pmin[1] + n2*pmin[2]
    cmax = n1*pmax[1] + n2*pmax[2]

    if off <= cmin + arr.atol || off >= cmax - arr.atol
        return 0
    end

    lo = 1
    hi = nu
    while lo + 1 < hi
        mid = (lo + hi) >>> 1
        pmid = pts[order[pos[mid]]]
        cmid = n1*pmid[1] + n2*pmid[2]

        if tie_break === :up
            if cmid <= off + arr.atol
                lo = mid
            else
                hi = mid
            end
        elseif tie_break === :down
            if cmid < off - arr.atol
                lo = mid
            else
                hi = mid
            end
        else
            throw(ArgumentError("tie_break must be :up or :down"))
        end
    end

    return lo
end

@inline function _arr2d_cell_linear_index(arr::FiberedArrangement2D, dir_idx::Int, off_idx::Int)
    return arr.start[dir_idx] + off_idx - 1
end

function _arr2d_chain_id!(arr::FiberedArrangement2D, chain::Vector{Int})::Int
    # IMPORTANT: `chain` is used as a Dict key. Do not mutate it after insertion.
    id = get(arr.chain_key_to_id, chain, 0)
    if id == 0
        id = length(arr.chains) + 1
        arr.chain_key_to_id[chain] = id
        push!(arr.chains, chain)
    end
    return id
end

function _nearest_coord_index(coords::Vector{Float64}, v::Float64; atol::Float64=1e-8)
    n = length(coords)
    if n == 0
        return 0
    end
    i = searchsortedfirst(coords, v)
    best = 0
    bestd = Inf

    for j in (i-1, i, i+1)
        if 1 <= j <= n
            d = abs(coords[j] - v)
            if d < bestd
                bestd = d
                best = j
            end
        end
    end

    return (bestd <= atol) ? best : 0
end

function _slice_chain_exact_boxes_2d_fast(
    pi,
    dir::Vector{Float64},
    offset::Float64;
    box,
    xs::Vector{Float64},
    ys::Vector{Float64},
    strict::Bool,
    atol::Float64,
)
    x0 = _line_basepoint_from_normal_offset_2d(dir, offset)
    tmin, tmax = _line_param_range_in_box_2d(x0, dir, box; atol=atol)
    if tmax <= tmin + atol
        return Int[], Float64[]
    end

    a, b = box
    xmin, ymin = a
    xmax, ymax = b

    ts = Float64[tmin, tmax]

    if abs(dir[1]) > atol
        for x in xs
            if x > xmin + atol && x < xmax - atol
                t = (x - x0[1]) / dir[1]
                if t > tmin + atol && t < tmax - atol
                    push!(ts, t)
                end
            end
        end
    end
    if abs(dir[2]) > atol
        for y in ys
            if y > ymin + atol && y < ymax - atol
                t = (y - x0[2]) / dir[2]
                if t > tmin + atol && t < tmax - atol
                    push!(ts, t)
                end
            end
        end
    end

    sort!(ts)
    _unique_sorted_floats!(ts; atol=atol)

    labels = Vector{Int}(undef, length(ts)-1)
    for i in 1:length(labels)
        tm = 0.5*(ts[i] + ts[i+1])
        xm = x0 .+ tm .* dir
        labels[i] = locate(pi, xm)
    end

    chain, values = _chain_values_from_boundaries(labels, ts; strict=strict, atol=atol)
    return chain, values
end

function _arr2d_slice_chain_and_values(arr::FiberedArrangement2D, dir::Union{AbstractVector{<:Real},NTuple{2,<:Real}}, off::Float64)
    if arr.backend === :boxes
        return _slice_chain_exact_boxes_2d_fast(
            arr.pi, dir, off;
            box=arr.box,
            xs=arr.xs,
            ys=arr.ys,
            strict=arr.strict,
            atol=arr.atol,
        )
    else
        return _slice_chain_exact_poly_2d(
            arr.pi, dir, off;
            box=arr.box,
            strict=arr.strict,
            atol=arr.atol,
            region_cache=arr.region_cache,
        )
    end
end

function _events_from_values_boxes(arr::FiberedArrangement2D, dir::Vector{Float64}, off::Float64, values::Vector{Float64})
    if length(values) <= 2
        return _AxisCoordIndex2D[]
    end
    x0 = _line_basepoint_from_normal_offset_2d(dir, off)
    events = Vector{_AxisCoordIndex2D}(undef, length(values)-2)

    snap_tol = max(1e-8, 100*arr.atol)

    for k in 1:length(events)
        t = values[k+1]
        x = x0[1] + t*dir[1]
        y = x0[2] + t*dir[2]

        ix = _nearest_coord_index(arr.xs, x; atol=snap_tol)
        iy = _nearest_coord_index(arr.ys, y; atol=snap_tol)

        if ix != 0
            events[k] = (1, ix)
        elseif iy != 0
            events[k] = (2, iy)
        else
            throw(ErrorException("could not snap boundary point to a grid line"))
        end
    end

    return events
end

function _arr2d_compute_cell!(arr::FiberedArrangement2D, dir_idx::Int, off_idx::Int)
    cell = _arr2d_cell_linear_index(arr, dir_idx, off_idx)
    cached = arr.cell_chain_id[cell]
    if cached != 0
        return cached
    end

    # Use representative direction for the direction cell.
    drep = arr.dir_reps[dir_idx]

    # Representative offset: midpoint between consecutive unique levels for drep.
    order = arr.orders[dir_idx]
    pos = arr.unique_pos[dir_idx]
    pL = arr.points[order[pos[off_idx]]]
    pR = arr.points[order[pos[off_idx+1]]]
    (n1, n2) = _normal_from_dir_2d(drep)
    cL = n1*pL[1] + n2*pL[2]
    cR = n1*pR[1] + n2*pR[2]
    off_mid = 0.5*(cL + cR)

    chain, vals = _arr2d_slice_chain_and_values(arr, drep, off_mid)
    arr.n_cell_computed += 1

    if isempty(chain)
        arr.cell_chain_id[cell] = -1
        return -1
    end

    cid = _arr2d_chain_id!(arr, chain)
    arr.cell_chain_id[cell] = cid

    if arr.backend === :boxes
        ev = _events_from_values_boxes(arr, drep, off_mid, vals)
        s = length(arr.event_pool) + 1
        append!(arr.event_pool, ev)
        arr.cell_event_start[cell] = s
        arr.cell_event_len[cell] = length(ev)
    end

    return cid
end

function _arr2d_values_from_cell_boxes(
    arr::FiberedArrangement2D,
    cell::Int,
    d::Vector{Float64},
    off::Float64,
    chain_len::Int,
)
    x0 = _line_basepoint_from_normal_offset_2d(d, off)
    tmin, tmax = _line_param_range_in_box_2d(x0, d, arr.box; atol=arr.atol)
    if tmax <= tmin + arr.atol
        return Float64[]
    end

    nvals = chain_len + 1
    values = Vector{Float64}(undef, nvals)
    values[1] = tmin
    values[end] = tmax

    s = arr.cell_event_start[cell]
    l = arr.cell_event_len[cell]
    if l != nvals - 2
        throw(ErrorException("event count does not match chain length"))
    end

    for k in 1:l
        axis, idx = arr.event_pool[s + k - 1]
        if axis == 1
            x = arr.xs[idx]
            values[k+1] = (x - x0[1]) / d[1]
        else
            y = arr.ys[idx]
            values[k+1] = (y - x0[2]) / d[2]
        end
    end

    # Enforce monotonicity against numerical noise.
    for i in 2:length(values)
        if values[i] < values[i-1]
            values[i] = values[i-1]
        end
    end

    return values
end

function _arr2d_values_for_chain_poly!(scratch::InvariantScratch, arr::FiberedArrangement2D, d::Vector{Float64}, off::Float64, chain::Vector{Int})
    x0 = _line_basepoint_from_normal_offset_2d(d, off)
    tmin, tmax = _line_param_range_in_box_2d(x0, d, arr.box; atol=arr.atol)
    if tmax <= tmin + arr.atol
        resize!(scratch.values, 0)
        return scratch.values
    end

    A_list, b_list = arr.region_cache
    m = length(chain)
    resize!(scratch.intervals, m)
    resize!(scratch.mids, m)
    resize!(scratch.perm, m)

    for i in 1:m
        r = chain[i]
        tlo, thi = _line_interval_in_hpoly_2d(A_list[r], b_list[r], x0, d; atol=arr.atol)
        tlo = max(tlo, tmin)
        thi = min(thi, tmax)
        if thi <= tlo + arr.atol
            throw(ErrorException("empty region-line intersection; likely degenerate query"))
        end
        scratch.intervals[i] = (tlo, thi)
        scratch.mids[i] = 0.5*(tlo + thi)
        scratch.perm[i] = i
    end

    sortperm!(scratch.perm, scratch.mids)
    resize!(scratch.values, m + 1)
    values = scratch.values
    values[1] = tmin
    for k in 1:m-1
        values[k+1] = scratch.intervals[scratch.perm[k]][2]
    end
    values[end] = tmax

    for i in 2:length(values)
        if values[i] < values[i-1]
            values[i] = values[i-1]
        end
    end

    return values
end

function _arr2d_values_for_chain_poly(arr::FiberedArrangement2D, d::Vector{Float64}, off::Float64, chain::Vector{Int})
    scratch = InvariantScratch()
    return copy(_arr2d_values_for_chain_poly!(scratch, arr, d, off, chain))
end

function _sync_index_barcodes!(cache::FiberedBarcodeCache2D)
    n = length(cache.arrangement.chains)
    while length(cache.index_barcodes_packed) < n
        push!(cache.index_barcodes_packed, nothing)
    end
    return nothing
end

function _index_barcode_for_chain!(cache::FiberedBarcodeCache2D, chain_id::Int)
    return _barcode_from_packed(_index_packed_for_chain!(cache, chain_id))
end

function _index_packed_for_chain!(cache::FiberedBarcodeCache2D, chain_id::Int)
    _sync_index_barcodes!(cache)
    pb = cache.index_barcodes_packed[chain_id]
    if pb !== nothing
        return pb
    end
    chain = cache.arrangement.chains[chain_id]
    pb = _slice_barcode_packed(cache.M, chain; values=nothing, check_chain=false)::PackedIndexBarcode
    cache.index_barcodes_packed[chain_id] = pb
    cache.n_barcode_computed += 1
    return pb
end

function _precompute_cells!(arr::FiberedArrangement2D;
                            threads::Bool = (Threads.nthreads() > 1))
    if threads && Threads.nthreads() > 1
        Threads.@threads for dir_idx in 1:length(arr.dir_reps)
            for off_idx in 1:arr.noff[dir_idx]
                _arr2d_compute_cell!(arr, dir_idx, off_idx)
            end
        end
    else
        for dir_idx in 1:length(arr.dir_reps)
            for off_idx in 1:arr.noff[dir_idx]
                _arr2d_compute_cell!(arr, dir_idx, off_idx)
            end
        end
    end
    return nothing
end

function _precompute_index_barcodes!(cache::FiberedBarcodeCache2D;
                                     threads::Bool = (Threads.nthreads() > 1))
    arr = cache.arrangement
    _sync_index_barcodes!(cache)
    if threads && Threads.nthreads() > 1
        Threads.@threads for i in 1:length(arr.chains)
            pb = cache.index_barcodes_packed[i]
            if pb === nothing
                chain = arr.chains[i]
                cache.index_barcodes_packed[i] =
                    _slice_barcode_packed(cache.M, chain; values=nothing, check_chain=false)::PackedIndexBarcode
            end
        end
        cache.n_barcode_computed = count(!isnothing, cache.index_barcodes_packed)
    else
        for i in 1:length(arr.chains)
            _index_packed_for_chain!(cache, i)
        end
    end
    return nothing
end

function _precompute_index_barcodes_for_chain_ids!(
    cache::FiberedBarcodeCache2D,
    chain_ids::AbstractVector{Int};
    threads::Bool = (Threads.nthreads() > 1),
)
    isempty(chain_ids) && return nothing
    arr = cache.arrangement
    _sync_index_barcodes!(cache)
    if threads && Threads.nthreads() > 1
        Threads.@threads for k in eachindex(chain_ids)
            i = chain_ids[k]
            pb = cache.index_barcodes_packed[i]
            if pb === nothing
                chain = arr.chains[i]
                cache.index_barcodes_packed[i] =
                    _slice_barcode_packed(cache.M, chain; values=nothing, check_chain=false)::PackedIndexBarcode
            end
        end
        cache.n_barcode_computed = count(!isnothing, cache.index_barcodes_packed)
    else
        for i in chain_ids
            _index_packed_for_chain!(cache, i)
        end
    end
    return nothing
end

@inline function _prepare_fibered_arrangement_readonly!(arr::FiberedArrangement2D)
    # Two-phase policy: build mutable arrangement caches sequentially, then
    # treat them as read-only in threaded compute regions.
    if arr.n_cell_computed != arr.total_cells
        _precompute_cells!(arr; threads=false)
    end
    return nothing
end

@inline function _prepare_fibered_cache_readonly!(cache::FiberedBarcodeCache2D)
    _prepare_fibered_arrangement_readonly!(cache.arrangement)
    if cache.n_barcode_computed != length(cache.arrangement.chains)
        _precompute_index_barcodes!(cache; threads=false)
    end
    return nothing
end

# ------------------------ public constructors ------------------------

"""
    fibered_arrangement_2d(pi, opts::InvariantOptions; ...)

Build and return a [`FiberedArrangement2D`](@ref) for exact repeated 2D slice
queries.

Build the exact 2D fibered arrangement for `pi`, including:
- direction representatives,
- line orders,
- the induced cell complex over a 2D box.

Cheap-first workflow:
- build the arrangement once,
- inspect it with [`describe`](@ref) or [`fibered_arrangement_summary`](@ref),
- then attach one or more modules with [`fibered_barcode_cache_2d`](@ref).

Use this exact arrangement path when many line queries, exact matching-distance
queries, or arrangement-integrated kernels will share the same 2D encoding
geometry. For one-off sampled slice workflows, stay on the lighter
`SliceInvariants` APIs instead of paying arrangement-build cost.

Opts usage:
- `opts.box` supplies the working box (if unset, we infer via `encoding_box(pi, opts)`).
- `opts.strict` controls locate strictness for "boxes backend" (defaults to true).
  For `:poly` backend we force `strict=false` (default behavior).

`precompute`:
- `:none`  : build point-location structure; compute cells lazily on demand
- `:cells` : compute all cells immediately

This object can be shared across modules.
"""
function fibered_arrangement_2d(
    pi,
    opts::InvariantOptions;
    normalize_dirs = :L1,
    include_axes = false,
    atol = 1e-12,
    max_combinations = 200_000,
    max_vertices = 20_000,
    max_cells = 5_000_000,
    precompute = :none,
    threads::Bool = (Threads.nthreads() > 1)
)
    pi0 = _unwrap_compiled(pi)
    backend = (pi0 isa PLPolyhedra.PLEncodingMap ? :poly : :boxes)

    strict_arg = opts.strict === nothing ? true : opts.strict
    strict0 = (backend == :poly ? false : strict_arg)

    # Working box: inferred from opts via encoding_box.
    bx_raw = encoding_box(pi, opts)
    bx = ([float(bx_raw[1][1]), float(bx_raw[1][2])],
          [float(bx_raw[2][1]), float(bx_raw[2][2])])

    points = if backend == :boxes && hasproperty(pi0, :coords)
        _critical_points_boxes_2d(pi0, bx; atol=atol)
    elseif backend == :poly && (pi0 isa PLPolyhedra.PLEncodingMap)
        _critical_points_poly_2d(pi0, bx; max_combinations=max_combinations, max_vertices=max_vertices, atol=atol)
    else
        pts = NTuple{2,Float64}[]
        for p in representatives(pi0)
            length(p) == 2 || throw(ArgumentError("fibered_arrangement_2d: expected 2D representatives"))
            push!(pts, (float(p[1]), float(p[2])))
        end
        pts
    end

    slopes = _unique_sorted_slopes(points; atol=atol)
    dir_reps_raw = _direction_representatives(slopes; normalize_dirs=normalize_dirs, include_axes=include_axes, atol=atol)
    dir_reps = [Float64[d[1], d[2]] for d in dir_reps_raw]

    ndirs = length(dir_reps)
    orders = Vector{Vector{Int}}(undef, ndirs)
    unique_pos = Vector{Vector{Int}}(undef, ndirs)
    noff = Vector{Int}(undef, ndirs)

    for i in 1:ndirs
        d = dir_reps[i]
        ord = _line_order(points, d; strict=strict0, atol=atol)
        orders[i] = ord
        pos = _unique_positions_for_order(points, ord, d; atol=atol)
        unique_pos[i] = pos
        noff[i] = max(0, length(pos) - 1)
    end

    start = Vector{Int}(undef, ndirs)
    total_cells = 0
    for i in 1:ndirs
        start[i] = total_cells + 1
        total_cells += noff[i]
    end

    total_cells > max_cells && error("fibered_arrangement_2d: too many cells (total_cells=$total_cells > max_cells=$max_cells)")

    cell_chain_id = zeros(Int, total_cells)
    cell_event_start = zeros(Int, total_cells)
    cell_event_len = zeros(Int, total_cells)
    event_pool = _AxisCoordIndex2D[]

    xs = Float64[]
    ys = Float64[]
    region_cache = nothing

    if backend == :boxes && hasproperty(pi0, :coords)
        coords = getproperty(pi0, :coords)
        xs = [float(x) for x in coords[1] if isfinite(x)]
        ys = [float(y) for y in coords[2] if isfinite(y)]
        _unique_sorted_floats!(xs; atol=atol)
        _unique_sorted_floats!(ys; atol=atol)
    elseif backend == :boxes
        xs = [p[1] for p in points]
        ys = [p[2] for p in points]
        _unique_sorted_floats!(xs; atol=atol)
        _unique_sorted_floats!(ys; atol=atol)
    elseif backend == :poly && (pi0 isa PLPolyhedra.PLEncodingMap)
        region_cache = ([Float64.(hp.A) for hp in pi0.regions],
                        [Float64.(hp.b) for hp in pi0.regions])
    end

    arr = FiberedArrangement2D(
        pi0, bx, normalize_dirs, include_axes, strict0, atol,
        backend, points, slopes, dir_reps, orders, unique_pos,
        noff, start, total_cells, cell_chain_id,
        cell_event_start, cell_event_len, event_pool,
        xs, ys, region_cache,
        Dict{Vector{Int},Int}(), Vector{Vector{Int}}(),
        0,
        Dict{FiberedSliceFamilyKey,FiberedSliceFamily2D}(),
    )

    _precompute_arrangement_cache!(arr, precompute; threads=threads)
    return arr
end

function _precompute_arrangement_cache!(arr::FiberedArrangement2D, precompute::Symbol;
                                        threads::Bool = (Threads.nthreads() > 1))
    if precompute in (:cells, :cells_barcodes, :full, :all)
        _precompute_cells!(arr; threads=threads)
    end
    return nothing
end


"""
    fibered_barcode_cache_2d(M, pi, opts::InvariantOptions; ...)
    fibered_barcode_cache_2d(M, arrangement::FiberedArrangement2D; precompute=:none)

Create and return a [`FiberedBarcodeCache2D`](@ref) for exact repeated fibered
barcode queries.

This is the "augmented arrangement" layer (RIVET-style caching):

- A `FiberedArrangement2D` (geometry only) partitions the space of slice parameters
  (direction, normal offset) into 2-cells on which the *slice chain* is constant.
- A `FiberedBarcodeCache2D` then augments that arrangement with a module `M` and memoizes
  the 1D *index barcode* for each slice chain as it is encountered.

The recommended workflow is to build *one* arrangement and then build *many* module caches
on top of it. This makes repeated slicing, matching distances, and kernels fast because
the expensive geometric bookkeeping is shared.

Cheap-first workflow:
- inspect the arrangement with [`fibered_arrangement_summary`](@ref),
- build one cache per module,
- inspect the cache with [`fibered_cache_summary`](@ref),
- then call [`fibered_barcode`](@ref), [`slice_barcodes`](@ref),
  [`matching_distance_exact_2d`](@ref), or [`slice_kernel`](@ref).

Convenience constructor:
- builds (or reuses) a `FiberedArrangement2D` for `pi`, then
- builds a `FiberedBarcodeCache2D` for module `M` on that arrangement,
- optionally precomputes barcodes over cells.

Opts usage:
- `opts.box` / `opts.strict` are used when building the arrangement (unless `arrangement` is provided).

Precomputation options
- `:none`     : lazy cells, lazy families, lazy payloads (default)
- `:family`   : precompute arrangement cells and the default slice family only
- `:barcodes` : additionally precompute representative packed barcodes per family cell
- `:distance` : additionally precompute point payloads used by exact distances/kernels

Compatibility aliases
- `:cells`, `:cells_barcodes`, `:full` map to `:barcodes`
- `:all` maps to `:distance`

Example: shared arrangement workflow

    using TamerOp
    const TO  = TamerOp
    const Inv = TO.Invariants

    # Inputs:
    #   pi : an encoding map in R^2 (boxes backend or PLEncodingMap)
    #   M,N: 2-parameter persistence modules over the same base field
    pi = ...
    M  = ...
    N  = ...

    opts = TO.InvariantOptions(box = Inv.encoding_box(pi))

    # 1. Build ONE geometry-only arrangement (share across many modules).
    #
    # The "arrangement" is the expensive part: it organizes the (dir, offset) plane
    # into cells where the combinatorics of the slice restriction is constant.
    arr = Inv.fibered_arrangement_2d(pi, opts;
        normalize_dirs = :L1,         # must match later queries
        include_axes = false,         # set true to include axis directions
        precompute = :cells,          # optional: eager cell computation
    )

    # 2. Build MANY module caches on the SAME arrangement object.
    #
    # Each cache stores only the module `M` plus memoized barcodes for slice chains.
    cacheM = Inv.fibered_barcode_cache_2d(M, arr)
    cacheN = Inv.fibered_barcode_cache_2d(N, arr)

    # 3a. Single slice query (direction + normal offset).
    # This returns a 1D barcode as a Dict((birth, death) => multiplicity).
    bar = Inv.fibered_barcode(cacheM, [1.0, 1.0], 0.25)

    # 3b. Many slice queries at once (matrix of barcodes).
    #
    # `offsets` can be either:
    #   - normal offsets (Real), or
    #   - basepoints in R^2 (length-2 vector/tuple), which are converted to normal offsets.
    out = Inv.slice_barcodes(cacheM;
        dirs = [[1.0, 1.0], [1.0, 2.0]],
        offsets = [0.0, 0.25],        # normal offsets; or basepoints like [x0, y0]
        values = :t,                  # barcode endpoints in the 1D parameter t
        direction_weight = :lesnick_l1,
    )
    barcodes = out.barcodes  # ndirs x noffsets matrix of Dicts
    weights  = out.weights   # ndirs x noffsets matrix (normalized by default)

    # 4. Higher-level computations that reuse the same arrangement.
    #
    # Exact matching distance by enumerating one representative slice per arrangement cell.
    d_match = Inv.matching_distance_exact_2d(cacheM, cacheN; weight = :lesnick_l1)

    # Kernel computed by integrating a per-slice kernel over arrangement cells.
    k = Inv.slice_kernel(cacheM, cacheN;
        kind = :gaussian,
        direction_weight = :lesnick_l1,
        cell_weight = :uniform,
    )

Notes
- `matching_distance_exact_2d(cacheM, cacheN)` and `slice_kernel(cacheM, cacheN)` require
  `cacheM.arrangement === cacheN.arrangement` (the same arrangement object).
- If you do not pass an existing arrangement, `fibered_barcode_cache_2d(M, pi; ...)` will
  build a fresh one (convenient, but not shared across modules unless you pass it around).
"""
function fibered_barcode_cache_2d(
    M::PModule{K},
    pi,
    opts::InvariantOptions;
    arrangement = nothing,
    precompute = :none,
    normalize_dirs = :L1,
    include_axes = false,
    atol = 1e-12,
    max_combinations = 200_000,
    max_vertices = 20_000,
    max_cells = 5_000_000,
    arr_precompute = :none,
    threads::Bool = (Threads.nthreads() > 1)
) where {K}
    level = _normalize_fibered_cache_precompute(precompute)

    # Family/barcode/distance payloads all require concrete arrangement cells.
    if arr_precompute == :none && level != :none
        arr_precompute = :cells
    end

    arr = arrangement === nothing ? fibered_arrangement_2d(pi, opts;
        normalize_dirs = normalize_dirs,
        include_axes = include_axes,
        atol = atol,
        max_combinations = max_combinations,
        max_vertices = max_vertices,
        max_cells = max_cells,
        precompute = arr_precompute,
        threads = threads
    ) : arrangement

    cache = fibered_barcode_cache_2d(M, arr; precompute = level, threads = threads)
    return cache
end

"""
    fibered_barcode_cache_2d(M, arr; precompute=:none)

Build a module-specific cache over a shared arrangement.

`precompute` can include:
- :none
- :family
- :barcodes
- :distance
- :cells_barcodes / :full (legacy aliases for `:barcodes`)
"""
function fibered_barcode_cache_2d(
    M::PModule{K},
    arr::FiberedArrangement2D;
    precompute::Symbol = :none,
    threads::Bool = (Threads.nthreads() > 1),
) where {K}
    level = _normalize_fibered_cache_precompute(precompute)

    cache = FiberedBarcodeCache2D(
        arr,
        M,
        Union{Nothing,PackedIndexBarcode}[],
        0,
        Dict{FiberedSliceFamilyKey,_FiberedFamilyBarcodePayload2D}(),
        Dict{FiberedSliceFamilyKey,_FiberedDistancePayload2D}(),
    )

    if level != :none
        _precompute_cells!(arr; threads=threads)
    end
    if level in (:family, :barcodes, :distance)
        fam = fibered_slice_family_2d(arr; direction_weight=:lesnick_l1, store_values=true)
        if level == :barcodes
            _precompute_family_barcodes!(cache, fam; threads=threads)
        end
        if level == :distance
            _precompute_distance_payload!(cache, fam; threads=threads)
        end
    end
    return cache
end


# ------------------------ public query API ------------------------

"""
    fibered_cell_id(arr, dir, offset; tie_break=:up)
    fibered_cell_id(arr, dir, x0::AbstractVector; tie_break=:up)

Return the arrangement cell id `(dir_cell, offset_cell)` for the line determined by:

- `dir` and normal-offset `offset`
- `dir` and basepoint `x0` (offset is computed as dot(n, x0))

Returns `nothing` if the line misses the arrangement box or lies on its boundary.
"""
function fibered_cell_id(arr::FiberedArrangement2D, dir, offset::Real; tie_break::Symbol=:up)
    d = _normalize_dir(_as_float2(dir), arr.normalize_dirs)
    dir_idx = _fibered_dir_cell_index(arr, d)
    off_idx = _fibered_offset_cell_index(arr, dir_idx, d, Float64(offset); tie_break=tie_break)
    if off_idx == 0
        return nothing
    end
    return (dir_idx, off_idx)
end

function fibered_cell_id(arr::FiberedArrangement2D, dir, x0::AbstractVector{<:Real}; tie_break::Symbol=:up)
    d = _normalize_dir(_as_float2(dir), arr.normalize_dirs)
    (n1, n2) = _normal_from_dir_2d(d)
    off = n1*Float64(x0[1]) + n2*Float64(x0[2])
    return fibered_cell_id(arr, d, off; tie_break=tie_break)
end

"""
    fibered_chain(arr, dir, offset; tie_break=:up, copy=true)

Return the slice chain (as a vector of region labels) for the given line.
The chain is computed lazily and then cached in the arrangement cell.
"""
function fibered_chain(arr::FiberedArrangement2D, dir, offset::Real; tie_break::Symbol=:up, copy::Bool=true)
    cid = fibered_cell_id(arr, dir, offset; tie_break=tie_break)
    if cid === nothing
        return Int[]
    end
    d = _normalize_dir(_as_float2(dir), arr.normalize_dirs)
    dir_idx, off_idx = cid
    chain_id = _arr2d_compute_cell!(arr, dir_idx, off_idx)
    if chain_id <= 0
        return Int[]
    end
    return copy ? Base.copy(arr.chains[chain_id]) : arr.chains[chain_id]
end

"""
    fibered_values(arr, dir, offset; tie_break=:up)

Return boundary parameter values along the line segment inside `arr.box`.
These are the `t` values used by `slice_barcode` to map index endpoints to real endpoints.
"""
function fibered_values(arr::FiberedArrangement2D, dir, offset::Real; tie_break::Symbol=:up)
    cid = fibered_cell_id(arr, dir, offset; tie_break=tie_break)
    if cid === nothing
        return Float64[]
    end
    d = _normalize_dir(_as_float2(dir), arr.normalize_dirs)
    off = Float64(offset)
    dir_idx, off_idx = cid
    cell = _arr2d_cell_linear_index(arr, dir_idx, off_idx)
    chain_id = _arr2d_compute_cell!(arr, dir_idx, off_idx)
    if chain_id <= 0
        return Float64[]
    end
    chain = arr.chains[chain_id]

    if arr.backend === :boxes
        return _arr2d_values_from_cell_boxes(arr, cell, d, off, length(chain))
    else
        # poly: try chain-only reconstruction; fallback to exact if degenerate
        try
            return _arr2d_values_for_chain_poly(arr, d, off, chain)
        catch
            opts_exact = InvariantOptions(box = arr.box, strict = arr.strict)
            _, vals2 = slice_chain_exact_2d(arr.pi, d, off, opts_exact; normalize_dirs = :none, atol = arr.atol)
            return vals2
        end
    end
end

"""
    fibered_barcode(cache, dir, offset; values=:t, tie_break=:up, verify=false)
    fibered_barcode(cache, dir, x0::AbstractVector; values=:t, tie_break=:up, verify=false)

Return the 1D barcode of the slice of `cache.M` along the line determined by:

- `dir` and normal-offset `offset`
- `dir` and basepoint `x0`

`values=:t` returns a barcode whose endpoints are real parameter values.
`values=:index` returns the index barcode.

This is the cheap direct query path once a [`FiberedBarcodeCache2D`](@ref) has
already been built. When a query lies on an arrangement boundary, `tie_break`
chooses the adjacent cell; inspect such cases with [`check_fibered_query`](@ref)
if you need explicit confirmation.

If `verify=true`, cross-check with `slice_chain_exact_2d` (slow, debug only).
"""
function fibered_barcode(
    cache::FiberedBarcodeCache2D,
    dir,
    offset::Real;
    values::Symbol = :t,
    tie_break::Symbol = :up,
    verify::Bool = false,
)
    arr = cache.arrangement
    d = _normalize_dir(_as_float2(dir), arr.normalize_dirs)
    off = Float64(offset)

    cid = fibered_cell_id(arr, d, off; tie_break=tie_break)
    if cid === nothing
        return Dict{Tuple{Float64,Float64},Int}()
    end
    dir_idx, off_idx = cid
    cell = _arr2d_cell_linear_index(arr, dir_idx, off_idx)
    chain_id = _arr2d_compute_cell!(arr, dir_idx, off_idx)
    if chain_id <= 0
        return Dict{Tuple{Float64,Float64},Int}()
    end

    bidx_packed = _index_packed_for_chain!(cache, chain_id)
    if values === :index
        return _barcode_from_packed(bidx_packed)
    elseif values !== :t
        throw(ArgumentError("values must be :t or :index"))
    end

    chain = arr.chains[chain_id]
    vals = Float64[]
    ok = true

    if arr.backend === :boxes
        try
            vals = _arr2d_values_from_cell_boxes(arr, cell, d, off, length(chain))
        catch
            ok = false
        end
    else
        try
            vals = _arr2d_values_for_chain_poly(arr, d, off, chain)
        catch
            ok = false
        end
    end

    if !ok || isempty(vals)
        # fallback exact (handles degenerate/boundary queries robustly)
        opts_exact = InvariantOptions(box = arr.box, strict = arr.strict)
        chain2, vals2 = slice_chain_exact_2d(arr.pi, d, off, opts_exact; normalize_dirs = :none, atol = arr.atol)
        if isempty(chain2)
            return Dict{Tuple{Float64,Float64},Int}()
        end
        cid2 = _arr2d_chain_id!(arr, chain2)
        bidx2_packed = _index_packed_for_chain!(cache, cid2)
        return _float_barcode_from_index_packed_values(bidx2_packed, vals2)
    end

    b = _float_barcode_from_index_packed_values(bidx_packed, vals)

    if verify
        opts_exact = InvariantOptions(box = arr.box, strict = arr.strict)
        chain2, vals2 = slice_chain_exact_2d(arr.pi, d, off, opts_exact; normalize_dirs = :none, atol = arr.atol)
        if chain2 != chain
            throw(ErrorException("verification failed: cached chain differs from exact chain"))
        end
        if length(vals2) == length(vals)
            for k in 1:length(vals)
                if abs(vals2[k] - vals[k]) > 1e-8
                    throw(ErrorException("verification failed: cached values differ from exact values"))
                end
            end
        end
    end

    return b
end

function fibered_barcode(cache::FiberedBarcodeCache2D, dir, x0::AbstractVector{<:Real}; kwargs...)
    arr = cache.arrangement
    d = _normalize_dir(_as_float2(dir), arr.normalize_dirs)
    (n1, n2) = _normal_from_dir_2d(d)
    off = n1*Float64(x0[1]) + n2*Float64(x0[2])
    return fibered_barcode(cache, d, off; kwargs...)
end

"""
    fibered_barcode_index(cache, dir, offset; tie_break=:up)

Convenience wrapper for `fibered_barcode(...; values=:index)`.
"""
function fibered_barcode_index(cache::FiberedBarcodeCache2D, dir, offset::Real; tie_break::Symbol=:up)
    return fibered_barcode(cache, dir, offset; values=:index, tie_break=tie_break)
end

@inline function _fibered_barcode_packed(
    cache::FiberedBarcodeCache2D,
    dir,
    offset::Real;
    values::Symbol = :t,
    tie_break::Symbol = :up,
)
    arr = cache.arrangement
    d = _normalize_dir(_as_float2(dir), arr.normalize_dirs)
    off = Float64(offset)

    cid = fibered_cell_id(arr, d, off; tie_break=tie_break)
    if cid === nothing
        return values === :index ? _empty_packed_index_barcode() : _empty_packed_float_barcode()
    end

    dir_idx, off_idx = cid
    cell = _arr2d_cell_linear_index(arr, dir_idx, off_idx)
    chain_id = _arr2d_compute_cell!(arr, dir_idx, off_idx)
    if chain_id <= 0
        return values === :index ? _empty_packed_index_barcode() : _empty_packed_float_barcode()
    end

    bidx_packed = _index_packed_for_chain!(cache, chain_id)
    if values === :index
        return bidx_packed
    elseif values !== :t
        throw(ArgumentError("values must be :t or :index"))
    end

    chain = arr.chains[chain_id]
    vals = Float64[]
    ok = true

    if arr.backend === :boxes
        try
            vals = _arr2d_values_from_cell_boxes(arr, cell, d, off, length(chain))
        catch
            ok = false
        end
    else
        try
            vals = _arr2d_values_for_chain_poly(arr, d, off, chain)
        catch
            ok = false
        end
    end

    if !ok || isempty(vals)
        opts_exact = InvariantOptions(box = arr.box, strict = arr.strict)
        chain2, vals2 = slice_chain_exact_2d(arr.pi, d, off, opts_exact; normalize_dirs = :none, atol = arr.atol)
        if isempty(chain2)
            return _empty_packed_float_barcode()
        end
        cid2 = _arr2d_chain_id!(arr, chain2)
        bidx2_packed = _index_packed_for_chain!(cache, cid2)
        return _float_packed_from_index_packed_values(bidx2_packed, vals2)
    end

    return _float_packed_from_index_packed_values(bidx_packed, vals)
end

@inline function _fibered_barcode_packed(cache::FiberedBarcodeCache2D, dir, x0::AbstractVector{<:Real}; kwargs...)
    arr = cache.arrangement
    d = _normalize_dir(_as_float2(dir), arr.normalize_dirs)
    (n1, n2) = _normal_from_dir_2d(d)
    off = n1 * Float64(x0[1]) + n2 * Float64(x0[2])
    return _fibered_barcode_packed(cache, d, off; kwargs...)
end

"""
    fibered_slice(cache, dir, offset; tie_break=:up)
    fibered_slice(cache, dir, x0::AbstractVector; tie_break=:up)

Return a [`FiberedSliceResult`](@ref) for one exact 2D slice query.

This is the owner-level "give me the whole slice payload" path once a
[`FiberedBarcodeCache2D`](@ref) is available. The result keeps the chain,
boundary values, and barcode aligned, including the exact fallback used on
degenerate/boundary queries.
"""
function fibered_slice(
    cache::FiberedBarcodeCache2D,
    dir,
    offset::Real;
    tie_break::Symbol = :up,
)
    arr = cache.arrangement
    d = _normalize_dir(_as_float2(dir), arr.normalize_dirs)
    off = Float64(offset)

    cid = fibered_cell_id(arr, d, off; tie_break=tie_break)
    if cid === nothing
        return _empty_fibered_slice_result()
    end

    dir_idx, off_idx = cid
    cell = _arr2d_cell_linear_index(arr, dir_idx, off_idx)
    chain_id = _arr2d_compute_cell!(arr, dir_idx, off_idx)

    if chain_id <= 0
        return _empty_fibered_slice_result()
    end

    bidx_packed = _index_packed_for_chain!(cache, chain_id)
    chain = arr.chains[chain_id]
    vals = Float64[]
    ok = true

    if arr.backend === :boxes
        try
            vals = _arr2d_values_from_cell_boxes(arr, cell, d, off, length(chain))
        catch
            ok = false
        end
    else
        try
            vals = _arr2d_values_for_chain_poly(arr, d, off, chain)
        catch
            ok = false
        end
    end

    if !ok || isempty(vals)
        opts_exact = InvariantOptions(box = arr.box, strict = arr.strict)
        chain2, vals2 = slice_chain_exact_2d(arr.pi, d, off, opts_exact; normalize_dirs = :none, atol = arr.atol)
        if isempty(chain2)
            return _empty_fibered_slice_result()
        end
        cid2 = _arr2d_chain_id!(arr, chain2)
        bidx2_packed = _index_packed_for_chain!(cache, cid2)
        b = _float_barcode_from_index_packed_values(bidx2_packed, vals2)
        return _fibered_slice_result(chain2, vals2, b)
    end

    b = _float_barcode_from_index_packed_values(bidx_packed, vals)

    return _fibered_slice_result(Base.copy(chain), vals, b)
end

function fibered_slice(cache::FiberedBarcodeCache2D, dir, x0::AbstractVector{<:Real}; kwargs...)
    arr = cache.arrangement
    d = _normalize_dir(_as_float2(dir), arr.normalize_dirs)
    x = _as_real_vector2(x0)
    x === nothing && throw(ArgumentError("basepoint must be given as a 2-vector or 2-tuple of reals."))
    (n1, n2) = _normal_from_dir_2d(d)
    off = n1 * x[1] + n2 * x[2]
    return fibered_slice(cache, d, off; kwargs...)
end

@inline fibered_slice(cache::FiberedBarcodeCache2D, dir, x0::NTuple{2,<:Real}; kwargs...) =
    fibered_slice(cache, dir, Float64[x0[1], x0[2]]; kwargs...)

"""
    fibered_barcode_cache_stats(cache)

Return a NamedTuple with basic statistics describing the cache state.
Useful to confirm caching/precomputation.
"""
function fibered_barcode_cache_stats(cache::FiberedBarcodeCache2D)
    arr = cache.arrangement
    n_family_barcode_payloads = length(cache.family_barcode_payloads)
    n_distance_payloads = length(cache.distance_payloads)
    n_family_barcode_slices_computed = 0
    n_distance_slices_computed = 0
    for payload in values(cache.family_barcode_payloads)
        n_family_barcode_slices_computed += payload.n_computed
    end
    for payload in values(cache.distance_payloads)
        n_distance_slices_computed += payload.n_computed
    end
    return (
        n_points = length(arr.points),
        n_dir_cells = length(arr.dir_reps),
        total_cells = arr.total_cells,
        n_cells_computed = arr.n_cell_computed,
        n_chains = length(arr.chains),
        n_index_barcodes_computed = cache.n_barcode_computed,
        n_slice_families_cached = length(arr.slice_family_cache),
        n_family_barcode_payloads = n_family_barcode_payloads,
        n_family_barcode_slices_computed = n_family_barcode_slices_computed,
        n_distance_payloads = n_distance_payloads,
        n_distance_slices_computed = n_distance_slices_computed,
        normalize_dirs = arr.normalize_dirs,
        include_axes = arr.include_axes,
        strict = arr.strict,
        atol = arr.atol,
    )
end


# -----------------------------------------------------------------------------
# Convenience wrappers and higher-level integrations for the 2D cache
# -----------------------------------------------------------------------------

# Internal helper: representative normal-offset for a given arrangement cell.
# Returns (off_mid, off_left, off_right), where off_mid is the midpoint
# between consecutive distinct critical offsets in that direction cell.
@inline function _arr2d_cell_representative_offset(
    arr::FiberedArrangement2D,
    dir_idx::Int,
    off_idx::Int,
)
    drep = arr.dir_reps[dir_idx]
    order = arr.orders[dir_idx]
    pos = arr.unique_pos[dir_idx]

    pL = arr.points[order[pos[off_idx]]]
    pR = arr.points[order[pos[off_idx + 1]]]

    (n1, n2) = _normal_from_dir_2d(drep)
    cL = n1 * pL[1] + n2 * pL[2]
    cR = n1 * pR[1] + n2 * pR[2]
    off_mid = 0.5 * (cL + cR)

    return off_mid, cL, cR
end

# Internal helper: angular width (in radians) of a direction cell, using the
# slope parameter s = d2/d1 and theta = atan(s).
#
# If include_axes=true, the axis directions are represented explicitly and have
# measure zero in theta, so we return 0.0 for those extra cells.
function _arr2d_direction_cell_theta_width(arr::FiberedArrangement2D, dir_idx::Int)
    n_base = length(arr.slope_breaks) + 1
    if dir_idx > n_base
        return 0.0
    end
    if isempty(arr.slope_breaks)
        return 0.5 * Base.MathConstants.pi
    end

    if dir_idx == 1
        sL = 0.0
        sR = arr.slope_breaks[1]
    elseif dir_idx == n_base
        sL = arr.slope_breaks[end]
        sR = Inf
    else
        sL = arr.slope_breaks[dir_idx - 1]
        sR = arr.slope_breaks[dir_idx]
    end

    thL = atan(sL)
    thR = isfinite(sR) ? atan(sR) : (0.5 * Base.MathConstants.pi)
    return thR - thL
end

# Internal helper: compute a safe global (tmin,tmax) range for representative
# slices over all nonempty arrangement cells. This is mainly used to produce a
# default tgrid for landscape kernels without materializing all barcodes.
function _arr2d_representative_tmin_tmax(arr::FiberedArrangement2D)
    tmin = Inf
    tmax = -Inf
    scratch = InvariantScratch()

    ndir = length(arr.dir_reps)
    for dir_idx in 1:ndir
        d = arr.dir_reps[dir_idx]
        for off_idx in 1:arr.noff[dir_idx]
            chain_id = _arr2d_compute_cell!(arr, dir_idx, off_idx)
            chain_id <= 0 && continue

            off_mid, _, _ = _arr2d_cell_representative_offset(arr, dir_idx, off_idx)
            chain = arr.chains[chain_id]
            cell = _arr2d_cell_linear_index(arr, dir_idx, off_idx)

            vals = Float64[]
            ok = true
            if arr.backend === :boxes
                try
                    vals = _arr2d_values_from_cell_boxes(arr, cell, d, off_mid, length(chain))
                catch
                    ok = false
                end
            else
                try
                    vals = _arr2d_values_for_chain_poly!(scratch, arr, d, off_mid, chain)
                catch
                    ok = false
                end
            end

            if !ok || isempty(vals)
                opts_exact = InvariantOptions(box = arr.box, strict = arr.strict)
                _, vals = slice_chain_exact_2d(arr.pi, d, off, opts_exact; normalize_dirs = :none, atol = arr.atol)
            end
            isempty(vals) && continue

            tmin = min(tmin, vals[1])
            tmax = max(tmax, vals[end])
        end
    end

    if !isfinite(tmin) || !isfinite(tmax)
        return 0.0, 1.0
    end
    if tmin == tmax
        return tmin - 0.5, tmax + 0.5
    end
    return tmin, tmax
end

@inline function _barcode_from_index_and_values!(
    out::FloatBarcode,
    bidx::IndexBarcode,
    vals::AbstractVector{<:Real},
)::FloatBarcode
    empty!(out)
    sizehint!(out, length(bidx))
    @inbounds for ((i, j), m) in bidx
        out[(Float64(vals[i]), Float64(vals[j]))] = m
    end
    return out
end

@inline function _barcode_from_index_and_values!(
    out::FloatBarcode,
    bidx::IndexBarcode,
    pool::Vector{Float64},
    start::Int,
)::FloatBarcode
    empty!(out)
    sizehint!(out, length(bidx))
    @inbounds for ((i, j), m) in bidx
        out[(pool[start + i - 1], pool[start + j - 1])] = m
    end
    return out
end

function _barcode_from_index_and_values(bidx::IndexBarcode, vals::AbstractVector{<:Real})::FloatBarcode
    out = FloatBarcode()
    return _barcode_from_index_and_values!(out, bidx, vals)
end

"""
    FiberedSliceFamily2D

Geometry-only precomputation for arrangement-exact slice families in R^2.

A `FiberedArrangement2D` partitions (direction, offset) space into finitely many 2-cells.
On each nonempty cell the sliced module is constant. A `FiberedSliceFamily2D` chooses
one representative slice per nonempty cell and (optionally) stores the boundary
parameter values for that slice.

This object is intended for many-pair workloads: build once per dataset, reuse across
many module pairs.
"""
struct FiberedSliceFamily2D
    arrangement::FiberedArrangement2D

    # one entry per nonempty arrangement cell
    dir_idx::Vector{Int}
    off_idx::Vector{Int}
    cell_id::Vector{Int}
    chain_id::Vector{Int}

    # representative offsets and interval endpoints
    off_mid::Vector{Float64}
    off0::Vector{Float64}
    off1::Vector{Float64}

    # concatenated boundary values storage
    vals_pool::Vector{Float64}
    vals_start::Vector{Int}
    vals_len::Vector{Int}

    # per-direction precomputations
    dir_weight::Vector{Float64}
    theta_width::Vector{Float64}

    direction_weight_scheme::Symbol
    store_values::Bool
    unique_chain_ids::Vector{Int}
end

@inline nslices(fam::FiberedSliceFamily2D)::Int = length(fam.cell_id)

@inline function fibered_values(fam::FiberedSliceFamily2D, k::Int)
    s = fam.vals_start[k]
    l = fam.vals_len[k]
    if s == 0 || l == 0
        return Float64[]
    end
    return @view fam.vals_pool[s:(s + l - 1)]
end

@inline function _arr2d_cell_offset_interval(arr::FiberedArrangement2D, dir_idx::Int, off_idx::Int)
    d = arr.dir_reps[dir_idx]
    order = arr.orders[dir_idx]
    pos = arr.unique_pos[dir_idx]
    @inbounds begin
        pL = arr.points[order[pos[off_idx]]]
        pR = arr.points[order[pos[off_idx + 1]]]
        n1 = -d[2]
        n2 =  d[1]
        cL = n1 * pL[1] + n2 * pL[2]
        cR = n1 * pR[1] + n2 * pR[2]
    end
    return cL, cR, 0.5 * (cL + cR)
end

"""
    fibered_slice_family_2d(arr; direction_weight=:lesnick_l1, store_values=true)

Build and cache a [`FiberedSliceFamily2D`](@ref) for a given arrangement.

The result is cached in `arr.slice_family_cache[FiberedSliceFamilyKey(direction_weight, store_values)]`.

Use this when you will reuse one representative slice per nonempty arrangement
cell across many pairwise exact matching-distance or kernel computations.
Inspect the cached family first with [`fibered_family_summary`](@ref).
"""
function fibered_slice_family_2d(
    arr::FiberedArrangement2D;
    direction_weight::Symbol = :lesnick_l1,
    store_values::Bool = true,
)
    key = FiberedSliceFamilyKey(direction_weight, store_values)
    cached = get(arr.slice_family_cache, key, nothing)
    if cached !== nothing
        return cached::FiberedSliceFamily2D
    end

    if arr.n_cell_computed != arr.total_cells
        _precompute_cells!(arr)
    end

    ndirs = length(arr.dir_reps)
    dir_w = Vector{Float64}(undef, ndirs)
    theta_w = Vector{Float64}(undef, ndirs)
    for i in 1:ndirs
        d = arr.dir_reps[i]
        dir_w[i] = _direction_weight(d; scheme=direction_weight)
        theta_w[i] = _arr2d_direction_cell_theta_width(arr, i)
    end

    dir_idx = Int[]
    off_idx = Int[]
    cell_id = Int[]
    chain_id = Int[]
    off_mid = Float64[]
    off0 = Float64[]
    off1 = Float64[]

    vals_pool = Float64[]
    vals_start = Int[]
    vals_len = Int[]
    scratch = InvariantScratch()

    for di in 1:ndirs
        d = arr.dir_reps[di]
        for oi in 1:arr.noff[di]
            cell = _arr2d_cell_linear_index(arr, di, oi)
            cid = arr.cell_chain_id[cell]
            cid > 0 || continue

            c0, c1, cmid = _arr2d_cell_offset_interval(arr, di, oi)

            push!(dir_idx, di)
            push!(off_idx, oi)
            push!(cell_id, cell)
            push!(chain_id, cid)
            push!(off_mid, cmid)
            push!(off0, c0)
            push!(off1, c1)

            if store_values
                chain = arr.chains[cid]
                m = length(chain)

                vals = if arr.backend == :boxes
                    _arr2d_values_from_cell_boxes(arr, cell, d, cmid, m)
                else
                    try
                        _arr2d_values_for_chain_poly!(scratch, arr, d, cmid, chain)
                    catch
                        _, v = _arr2d_slice_chain_and_values(arr, d, cmid)
                        v
                    end
                end

                s = length(vals_pool) + 1
                append!(vals_pool, vals)
                push!(vals_start, s)
                push!(vals_len, length(vals))
            else
                push!(vals_start, 0)
                push!(vals_len, 0)
            end
        end
    end

    uniq = sort!(unique(chain_id))

    fam = FiberedSliceFamily2D(
        arr, dir_idx, off_idx, cell_id, chain_id,
        off_mid, off0, off1,
        vals_pool, vals_start, vals_len,
        dir_w, theta_w,
        direction_weight, store_values, uniq,
    )

    arr.slice_family_cache[key] = fam
    return fam
end

@inline function _normalize_fibered_cache_precompute(precompute::Symbol)
    if precompute in (:none, :family, :barcodes, :distance)
        return precompute
    elseif precompute in (:cells, :cells_barcodes, :full)
        return :barcodes
    elseif precompute == :all
        return :distance
    end
    throw(ArgumentError("precompute must be one of :none, :family, :barcodes, :distance"))
end

@inline _family_key(fam::FiberedSliceFamily2D) =
    FiberedSliceFamilyKey(fam.direction_weight_scheme, fam.store_values)

@inline function _float_packed_from_index_packed_pool(
    pb::PackedIndexBarcode,
    vals_pool::AbstractVector{Float64},
    start::Int,
)::PackedFloatBarcode
    pairs = Vector{EndpointPair{Float64}}(undef, length(pb.pairs))
    mults = copy(pb.mults)
    @inbounds for i in eachindex(pb.pairs)
        p = pb.pairs[i]
        pairs[i] = EndpointPair{Float64}(vals_pool[start + p.b - 1], vals_pool[start + p.d - 1])
    end
    return PackedFloatBarcode(pairs, mults)
end

@inline function _family_barcode_payload!(cache::FiberedBarcodeCache2D, fam::FiberedSliceFamily2D)
    key = _family_key(fam)
    return get!(cache.family_barcode_payloads, key) do
        _FiberedFamilyBarcodePayload2D(
            key,
            Vector{Union{Nothing,PackedFloatBarcode}}(fill(nothing, nslices(fam))),
            0,
        )
    end
end

@inline function _distance_payload!(cache::FiberedBarcodeCache2D, fam::FiberedSliceFamily2D)
    key = _family_key(fam)
    return get!(cache.distance_payloads, key) do
        _FiberedDistancePayload2D(
            key,
            Vector{Union{Nothing,Vector{Tuple{Float64,Float64}}}}(fill(nothing, nslices(fam))),
            0,
        )
    end
end

@inline function _family_barcode_packed_uncached!(
    cache::FiberedBarcodeCache2D,
    fam::FiberedSliceFamily2D,
    k::Int,
    scratch::InvariantScratch,
)
    arr = cache.arrangement
    cid = fam.chain_id[k]
    bidx = _index_packed_for_chain!(cache, cid)
    s = fam.vals_start[k]
    if s != 0
        return _float_packed_from_index_packed_pool(bidx, fam.vals_pool, s)
    end

    di = fam.dir_idx[k]
    d = arr.dir_reps[di]
    if arr.backend === :boxes
        _, vals = _arr2d_slice_chain_and_values(arr, d, fam.off_mid[k])
        return _float_packed_from_index_packed_values(bidx, vals)
    end
    vals = _arr2d_values_for_chain_poly!(scratch, arr, d, fam.off_mid[k], arr.chains[cid])
    return _float_packed_from_index_packed_values(bidx, vals)
end

@inline function _family_barcode_packed!(
    cache::FiberedBarcodeCache2D,
    fam::FiberedSliceFamily2D,
    payload::_FiberedFamilyBarcodePayload2D,
    k::Int,
    scratch::InvariantScratch,
)
    pb = payload.packed_barcodes[k]
    if pb !== nothing
        return pb::PackedFloatBarcode
    end
    out = _family_barcode_packed_uncached!(cache, fam, k, scratch)
    payload.packed_barcodes[k] = out
    payload.n_computed += 1
    return out
end

@inline function _distance_points_for_slice_uncached!(
    cache::FiberedBarcodeCache2D,
    fam::FiberedSliceFamily2D,
    k::Int,
    scratch::InvariantScratch,
)
    cid = fam.chain_id[k]
    bidx = _index_packed_for_chain!(cache, cid)
    out = Tuple{Float64,Float64}[]
    s = fam.vals_start[k]
    if s != 0
        _points_from_index_packed_and_values!(out, bidx, fam.vals_pool, s)
        return out
    end

    arr = cache.arrangement
    di = fam.dir_idx[k]
    d = arr.dir_reps[di]
    if arr.backend === :boxes
        _, vals = _arr2d_slice_chain_and_values(arr, d, fam.off_mid[k])
        _points_from_index_packed_and_values!(out, bidx, vals)
        return out
    end
    vals = _arr2d_values_for_chain_poly!(scratch, arr, d, fam.off_mid[k], arr.chains[cid])
    _points_from_index_packed_and_values!(out, bidx, vals)
    return out
end

@inline function _distance_points_for_slice!(
    cache::FiberedBarcodeCache2D,
    fam::FiberedSliceFamily2D,
    distance_payload::_FiberedDistancePayload2D,
    k::Int,
    scratch::InvariantScratch,
)
    pts = distance_payload.points[k]
    if pts !== nothing
        return pts::Vector{Tuple{Float64,Float64}}
    end
    out = _distance_points_for_slice_uncached!(cache, fam, k, scratch)
    distance_payload.points[k] = out
    distance_payload.n_computed += 1
    return out
end

function _precompute_family_barcodes!(
    cache::FiberedBarcodeCache2D,
    fam::FiberedSliceFamily2D;
    threads::Bool = (Threads.nthreads() > 1),
)
    _prepare_fibered_arrangement_readonly!(cache.arrangement)
    _precompute_index_barcodes!(cache; threads=false)
    payload = _family_barcode_payload!(cache, fam)
    if payload.n_computed == nslices(fam)
        return payload
    end

    if threads && Threads.nthreads() > 1
        Threads.@threads for k in 1:nslices(fam)
            payload.packed_barcodes[k] === nothing || continue
            scratch = InvariantScratch()
            payload.packed_barcodes[k] = _family_barcode_packed_uncached!(cache, fam, k, scratch)
        end
        payload.n_computed = count(!isnothing, payload.packed_barcodes)
    else
        scratch = InvariantScratch()
        for k in 1:nslices(fam)
            payload.packed_barcodes[k] === nothing || continue
            payload.packed_barcodes[k] = _family_barcode_packed_uncached!(cache, fam, k, scratch)
        end
        payload.n_computed = nslices(fam)
    end
    return payload
end

function _precompute_distance_payload!(
    cache::FiberedBarcodeCache2D,
    fam::FiberedSliceFamily2D;
    threads::Bool = (Threads.nthreads() > 1),
)
    _prepare_fibered_arrangement_readonly!(cache.arrangement)
    _precompute_index_barcodes_for_chain_ids!(cache, fam.unique_chain_ids; threads=false)
    distance_payload = _distance_payload!(cache, fam)
    if distance_payload.n_computed == nslices(fam)
        return distance_payload
    end

    if threads && Threads.nthreads() > 1
        Threads.@threads for k in 1:nslices(fam)
            distance_payload.points[k] === nothing || continue
            scratch = InvariantScratch()
            distance_payload.points[k] = _distance_points_for_slice_uncached!(cache, fam, k, scratch)
        end
        distance_payload.n_computed = count(!isnothing, distance_payload.points)
    else
        scratch = InvariantScratch()
        for k in 1:nslices(fam)
            distance_payload.points[k] === nothing || continue
            distance_payload.points[k] = _distance_points_for_slice_uncached!(cache, fam, k, scratch)
        end
        distance_payload.n_computed = nslices(fam)
    end
    return distance_payload
end

"""
    slice_barcodes(cache::FiberedBarcodeCache2D; dirs, offsets, ...) -> SliceBarcodesResult

Convenience wrapper to compute a matrix of fibered barcodes using the augmented
arrangement cache.

This is analogous to `slice_barcodes(M, pi; ...)`, but it uses `fibered_barcode`
and therefore benefits from the arrangement and barcode cache.

This returns the same owner-level [`SliceBarcodesResult`](@ref) used by
`SliceInvariants`. Prefer [`describe`](@ref) on the result before unpacking the
full barcode matrix. Use this exact path when arrangement construction is
already amortized across many queries; for one-off sampled slices, use the
lighter `SliceInvariants.slice_barcodes(M, pi; ...)` surface directly.

Inputs
- `dirs` (or `directions`): iterable of 2-vectors specifying slice directions.
  Directions are interpreted using the arrangement normalization stored in
  `cache.arrangement.normalize_dirs`.
- `offsets`: iterable where each element is either
    * a `Real` normal-offset `off = dot(n, x)` with `n = (-d2, d1)`, or
    * a length-2 basepoint (vector or 2-tuple) lying on the slice line.

Keywords
- `values`: `:t` (default) or `:index` (index barcodes).
- `packed`: if `true`, return `PackedBarcodeGrid` internally; otherwise convert
  to dict barcodes at the API boundary.
- `tie_break`: forwarded to `fibered_barcode` for boundary cases. It matters
  when the query direction or offset lands on an arrangement boundary.
- `verify`: forwarded to `fibered_barcode` (slow; checks against exact slicing).
- `direction_weight`: one of `:none`, `:lesnick_l1`, `:lesnick_linf`.
- `offset_weights`: `nothing` (default), a vector of length `length(offsets)`,
  a function `w(off)` (called on each offset element as provided), or a scalar.
- `normalize_weights`: normalize the weight matrix to sum to 1.
"""
function slice_barcodes(
    cache::FiberedBarcodeCache2D;
    dirs = nothing,
    directions = nothing,
    offsets,
    values::Symbol = :t,
    packed::Bool = false,
    direction_weight::Union{Symbol,Function,Real} = :none,
    offset_weights = nothing,
    normalize_weights::Bool = true,
    tie_break::Symbol = :up,
    threads::Bool = (Threads.nthreads() > 1),
)
    if threads && Threads.nthreads() > 1
        _prepare_fibered_cache_readonly!(cache)
    end

    if dirs === nothing
        dirs = directions
    end
    dirs === nothing && error("slice_barcodes: dirs/directions is required")

    nd = length(dirs)
    no = length(offsets)
    dirs_used = Vector{Vector{Float64}}(undef, nd)
    for i in 1:nd
        dirs_used[i] = _normalize_dir(_as_float2(dirs[i]), cache.arrangement.normalize_dirs)
    end

    wdir = Vector{Float64}(undef, nd)
    for i in 1:nd
        wdir[i] = _direction_weight(dirs_used[i]; scheme=direction_weight)
    end
    woff = _offset_sample_weights(offsets, offset_weights)

    weights = wdir * woff'
    if normalize_weights
        s = sum(weights)
        if s > 0
            weights ./= s
        end
    end

    if values == :index
        grids = _packed_grid_undef(PackedIndexBarcode, nd, no)
        if threads
            Threads.@threads for idx in 1:(nd * no)
                i = div((idx - 1), no) + 1
                j = (idx - 1) % no + 1
                grids[i, j] = _fibered_barcode_packed(cache, dirs_used[i], offsets[j]; values=:index, tie_break=tie_break)
            end
        else
            for i in 1:nd, j in 1:no
                grids[i, j] = _fibered_barcode_packed(cache, dirs_used[i], offsets[j]; values=:index, tie_break=tie_break)
            end
        end
        if packed
            return SliceBarcodesResult(grids, weights, dirs_used, offsets)
        end
        return SliceBarcodesResult(_index_dict_matrix_from_packed_grid(grids), weights, dirs_used, offsets)
    elseif values == :t
        grids = _packed_grid_undef(PackedFloatBarcode, nd, no)
        if threads
            Threads.@threads for idx in 1:(nd * no)
                i = div((idx - 1), no) + 1
                j = (idx - 1) % no + 1
                grids[i, j] = _fibered_barcode_packed(cache, dirs_used[i], offsets[j]; values=:t, tie_break=tie_break)
            end
        else
            for i in 1:nd, j in 1:no
                grids[i, j] = _fibered_barcode_packed(cache, dirs_used[i], offsets[j]; values=:t, tie_break=tie_break)
            end
        end
        if packed
            return SliceBarcodesResult(grids, weights, dirs_used, offsets)
        end
        return SliceBarcodesResult(_float_dict_matrix_from_packed_grid(grids), weights, dirs_used, offsets)
    else
        throw(ArgumentError("values must be :t or :index"))
    end
end

slice_barcodes(cache::FiberedBarcodeCache2D, dirs, offsets; kwargs...) =
    slice_barcodes(cache; dirs=dirs, offsets=offsets, kwargs...)




"""
    matching_distance_exact_2d(M, N, pi, opts::InvariantOptions; ...) -> Float64

Exact 2D matching distance computed by enumerating *all* arrangement cells
(one representative slice per cell), reusing the same augmented arrangement
across many module pairs.

This returns an exact distance over the full arrangement-cell family. Use it
when the same arrangement will be reused across many module pairs and you want
deterministic exactness. For cheaper exploratory work, prefer the sampled
slice-based approximations in `SliceInvariants`.

This overload is intended for the workflow:
1. Build one `FiberedArrangement2D` once (from `pi` and `box`).
2. Build one `FiberedBarcodeCache2D` per module on that same arrangement.
3. Call `matching_distance_exact_2d(cacheM, cacheN)` repeatedly.

Requirements
- `cacheM.arrangement === cacheN.arrangement` (same arrangement object)
- same base field `Q` in both modules.

Opts usage:
- `opts.box` controls the arrangement window and slice clipping.
- `opts.strict` controls strictness when locating regions (defaults to true, but poly backend forces false).
- `opts.threads` controls parallel evaluation in the final exact distance stage.

All other keyword arguments match the previous API (except `box`, `strict`, `threads`
which are now read from opts).
"""
function _matching_distance_exact_2d_from_caches(
    cacheM::FiberedBarcodeCache2D,
    cacheN::FiberedBarcodeCache2D,
    fam::FiberedSliceFamily2D;
    threads::Bool = (Threads.nthreads() > 1),
)::Float64
    arr = cacheM.arrangement
    ns = nslices(fam)

    if threads && Threads.nthreads() > 1
        scratch_by_thread = _scratch_arenas(true)
        best_by_thread = fill(0.0, length(scratch_by_thread))
        Threads.@threads for k in 1:ns
            tid = Threads.threadid()
            scratch = scratch_by_thread[tid]
            di = fam.dir_idx[k]
            cid = fam.chain_id[k]
            w = fam.dir_weight[di]

            bidxM = cacheM.index_barcodes_packed[cid]::PackedIndexBarcode
            bidxN = cacheN.index_barcodes_packed[cid]::PackedIndexBarcode
            s = fam.vals_start[k]

            if s == 0
                d = arr.dir_reps[di]
                vals = if arr.backend === :boxes
                    _, vals0 = _arr2d_slice_chain_and_values(arr, d, fam.off_mid[k])
                    vals0
                else
                    _arr2d_values_for_chain_poly!(scratch, arr, d, fam.off_mid[k], arr.chains[cid])
                end
                _points_from_index_packed_and_values!(scratch.points_a, bidxM, vals)
                _points_from_index_packed_and_values!(scratch.points_b, bidxN, vals)
            else
                _points_from_index_packed_and_values!(scratch.points_a, bidxM, fam.vals_pool, s)
                _points_from_index_packed_and_values!(scratch.points_b, bidxN, fam.vals_pool, s)
            end
            best_by_thread[tid] = max(best_by_thread[tid], w * bottleneck_distance(scratch.points_a, scratch.points_b))
        end
        return maximum(best_by_thread)
    end

    best = 0.0
    scratch = _scratch_arenas(false)[1]
    for k in 1:ns
        di = fam.dir_idx[k]
        cid = fam.chain_id[k]
        w = fam.dir_weight[di]

        bidxM = _index_packed_for_chain!(cacheM, cid)
        bidxN = _index_packed_for_chain!(cacheN, cid)
        s = fam.vals_start[k]

        if s == 0
            d = arr.dir_reps[di]
            vals = if arr.backend === :boxes
                _, vals0 = _arr2d_slice_chain_and_values(arr, d, fam.off_mid[k])
                vals0
            else
                _arr2d_values_for_chain_poly!(scratch, arr, d, fam.off_mid[k], arr.chains[cid])
            end
            _points_from_index_packed_and_values!(scratch.points_a, bidxM, vals)
            _points_from_index_packed_and_values!(scratch.points_b, bidxN, vals)
        else
            _points_from_index_packed_and_values!(scratch.points_a, bidxM, fam.vals_pool, s)
            _points_from_index_packed_and_values!(scratch.points_b, bidxN, fam.vals_pool, s)
        end

        best = max(best, w * bottleneck_distance(scratch.points_a, scratch.points_b))
    end

    return best
end

function matching_distance_exact_2d(
    M::PModule{K},
    N::PModule{K},
    pi::PLikeEncodingMap,
    opts::InvariantOptions;
    weight = :lesnick_l1,
    normalize_dirs = :L1,
    include_axes = false,
    atol = 1e-12,
    max_combinations = 200_000,
    max_vertices = 20_000,
    max_cells = 5_000_000,
    precompute = :cells_barcodes,
    arrangement = nothing,
    store_values::Bool = true,
    family = nothing
)::Float64 where {K}

    threads0 = opts.threads === nothing ? (Threads.nthreads() > 1) : opts.threads
    arr_precompute = arrangement === nothing && family === nothing ? (precompute === :none ? :none : :cells) : :none

    arr = if arrangement !== nothing
        arrangement
    elseif family !== nothing
        family.arrangement
    else
        fibered_arrangement_2d(pi, opts;
            normalize_dirs = normalize_dirs,
            include_axes = include_axes,
            atol = atol,
            max_combinations = max_combinations,
            max_vertices = max_vertices,
            max_cells = max_cells,
            precompute = arr_precompute,
            threads = threads0,
        )
    end

    fam = family === nothing ? fibered_slice_family_2d(arr; direction_weight=weight, store_values=store_values) : family
    fam.arrangement === arr || error("matching_distance_exact_2d: provided family does not match arrangement")

    cacheM = fibered_barcode_cache_2d(M, arr; precompute=precompute, threads=threads0)
    cacheN = fibered_barcode_cache_2d(N, arr; precompute=precompute, threads=threads0)

    return matching_distance_exact_2d(cacheM, cacheN;
        weight = weight,
        family = fam,
        store_values = store_values,
        threads = threads0,
    )
end

function matching_distance_exact_2d(
    cacheM::FiberedBarcodeCache2D,
    cacheN::FiberedBarcodeCache2D;
    weight::Symbol = :lesnick_l1,
    family::Union{Nothing,FiberedSliceFamily2D} = nothing,
    store_values::Bool = true,
    threads::Bool = (Threads.nthreads() > 1),
)::Float64
    arr = cacheM.arrangement
    arr === cacheN.arrangement || error("matching_distance_exact_2d: caches must share the same arrangement")

    fam = family === nothing ? fibered_slice_family_2d(arr; direction_weight = weight, store_values = store_values) : family
    fam.arrangement === arr || error("matching_distance_exact_2d: provided family does not match arrangement")

    if threads && Threads.nthreads() > 1
        _prepare_fibered_arrangement_readonly!(arr)
        _precompute_index_barcodes_for_chain_ids!(cacheM, fam.unique_chain_ids; threads=false)
        _precompute_index_barcodes_for_chain_ids!(cacheN, fam.unique_chain_ids; threads=false)
    end

    return _matching_distance_exact_2d_from_caches(cacheM, cacheN, fam; threads=threads)
end




"""
    slice_kernel(cacheM::FiberedBarcodeCache2D, cacheN::FiberedBarcodeCache2D; ...) -> Float64

Compute a sliced kernel by integrating over *all* arrangement cells (one
representative slice per cell), using the shared augmented arrangement.

This is the arrangement-exact analogue of sampling-based `slice_kernel(M,N,pi; ...)`.

Use this when arrangement construction and per-chain barcode caching will be
amortized across many kernel evaluations. Inspect the shared arrangement and
cached family first with [`fibered_arrangement_summary`](@ref) and
[`fibered_family_summary`](@ref).

Keywords
- `kind`, `sigma`, `gamma`, `tgrid`, `kmax`: forwarded to `_barcode_kernel`.
- `direction_weight`: direction weighting scheme (`:none`, `:lesnick_l1`, `:lesnick_linf`).
- `cell_weight`: how to weight arrangement cells. Supported symbols:
    * `:uniform`      (each nonempty cell weight 1)
    * `:offset_length` (weight by the length of the offset interval in normal-offset)
    * `:theta`        (weight by angular width of the direction cell)
    * `:theta_offset` (product of `:theta` and `:offset_length`)
  or a function `w(dir, off, dir_cell, off_cell, arr)::Real`.
- `normalize_weights`: kept for API symmetry; the result is always a
  weighted average as in `slice_kernel` for explicit slice lists.
- `tgrid_nsteps`: if `tgrid` is not provided and `kind` is landscape-based,
  we build a default grid with this many points.
"""
function slice_kernel(
    cacheM::FiberedBarcodeCache2D,
    cacheN::FiberedBarcodeCache2D;
    kind::Symbol = :bottleneck_gaussian,
    sigma::Real = 1.0,
    direction_weight::Symbol = :lesnick_l1,
    cell_weight::Symbol = :uniform,
    normalize_weights::Bool = true,
    family::Union{Nothing,FiberedSliceFamily2D} = nothing,
    store_values::Bool = true,
    threads::Bool = (Threads.nthreads() > 1),
)
    arr = cacheM.arrangement
    arr === cacheN.arrangement || error("slice_kernel: caches must share the same arrangement")

    if threads && Threads.nthreads() > 1
        _prepare_fibered_arrangement_readonly!(arr)
    end
    fam = family === nothing ? fibered_slice_family_2d(arr; direction_weight=direction_weight, store_values=store_values) : family
    fam.arrangement === arr || error("slice_kernel: provided family does not match arrangement")

    # Kernel on barcodes (reuse existing kernel definitions)
    kernel_fn = (bA, bB) -> _barcode_kernel(bA, bB; kind = kind, sigma = sigma)
    use_distance_payload = kind in (:bottleneck_gaussian, :bottleneck_laplacian, :wasserstein_gaussian, :wasserstein_laplacian)

    if threads && Threads.nthreads() > 1
        if use_distance_payload
            _precompute_distance_payload!(cacheM, fam; threads=false)
            _precompute_distance_payload!(cacheN, fam; threads=false)
        else
            _precompute_family_barcodes!(cacheM, fam; threads=false)
            _precompute_family_barcodes!(cacheN, fam; threads=false)
        end
    end

    scratch_by_thread = use_distance_payload ? InvariantScratch[] : _scratch_arenas(threads)
    acc_by_thread = fill(0.0, max(length(scratch_by_thread), Threads.nthreads()))
    sumw_by_thread = fill(0.0, max(length(scratch_by_thread), Threads.nthreads()))

    ns = nslices(fam)

    # helper: cell weight
    @inline function _cell_w(k::Int)::Float64
        if cell_weight === :uniform || cell_weight === :none
            return 1.0
        elseif cell_weight === :offset_length
            return fam.off1[k] - fam.off0[k]
        elseif cell_weight === :theta
            return fam.theta_width[fam.dir_idx[k]]
        elseif cell_weight === :theta_offset
            return fam.theta_width[fam.dir_idx[k]] * (fam.off1[k] - fam.off0[k])
        else
            throw(ArgumentError("slice_kernel: unknown cell_weight=$(cell_weight)"))
        end
    end

    if threads
        if use_distance_payload
            payloadM = _distance_payload!(cacheM, fam)
            payloadN = _distance_payload!(cacheN, fam)
            Threads.@threads for k in 1:ns
                tid = Threads.threadid()
                di = fam.dir_idx[k]
                w = fam.dir_weight[di] * _cell_w(k)
                ptsM = payloadM.points[k]::Vector{Tuple{Float64,Float64}}
                ptsN = payloadN.points[k]::Vector{Tuple{Float64,Float64}}
                acc_by_thread[tid] += w * kernel_fn(ptsM, ptsN)
                sumw_by_thread[tid] += w
            end
            acc = sum(acc_by_thread)
            sumw = sum(sumw_by_thread)
            return normalize_weights && sumw > 0 ? acc / sumw : acc
        end

        Threads.@threads for k in 1:ns
            tid = Threads.threadid()
            scratch = scratch_by_thread[tid]
            di = fam.dir_idx[k]
            cid = fam.chain_id[k]
            w = fam.dir_weight[di] * _cell_w(k)

            bidxM = cacheM.index_barcodes_packed[cid]::PackedIndexBarcode
            bidxN = cacheN.index_barcodes_packed[cid]::PackedIndexBarcode
            s = fam.vals_start[k]
            if s == 0
                d = arr.dir_reps[di]
                vals = if arr.backend === :boxes
                    _, vals0 = _arr2d_slice_chain_and_values(arr, d, fam.off_mid[k])
                    vals0
                else
                    _arr2d_values_for_chain_poly!(scratch, arr, d, fam.off_mid[k], arr.chains[cid])
                end
                _points_from_index_packed_and_values!(scratch.points_a, bidxM, vals)
                _points_from_index_packed_and_values!(scratch.points_b, bidxN, vals)
            else
                _points_from_index_packed_and_values!(scratch.points_a, bidxM, fam.vals_pool, s)
                _points_from_index_packed_and_values!(scratch.points_b, bidxN, fam.vals_pool, s)
            end

            acc_by_thread[tid] += w * kernel_fn(scratch.points_a, scratch.points_b)
            sumw_by_thread[tid] += w
        end
        acc = sum(acc_by_thread)
        sumw = sum(sumw_by_thread)
        return normalize_weights && sumw > 0 ? acc / sumw : acc
    else
        if use_distance_payload
            distance_payloadM = _distance_payload!(cacheM, fam)
            distance_payloadN = _distance_payload!(cacheN, fam)
            scratchM = InvariantScratch()
            scratchN = InvariantScratch()
            acc = 0.0
            sumw = 0.0
            for k in 1:ns
                di = fam.dir_idx[k]
                w = fam.dir_weight[di] * _cell_w(k)
                ptsM = _distance_points_for_slice!(cacheM, fam, distance_payloadM, k, scratchM)
                ptsN = _distance_points_for_slice!(cacheN, fam, distance_payloadN, k, scratchN)
                acc += w * kernel_fn(ptsM, ptsN)
                sumw += w
            end
            return normalize_weights && sumw > 0 ? acc / sumw : acc
        end

        acc = 0.0
        sumw = 0.0
        scratch = scratch_by_thread[1]
        for k in 1:ns
            di = fam.dir_idx[k]
            cid = fam.chain_id[k]
            w = fam.dir_weight[di] * _cell_w(k)

            bidxM = _index_packed_for_chain!(cacheM, cid)
            bidxN = _index_packed_for_chain!(cacheN, cid)
            s = fam.vals_start[k]
            if s == 0
                d = arr.dir_reps[di]
                vals = if arr.backend === :boxes
                    _, vals0 = _arr2d_slice_chain_and_values(arr, d, fam.off_mid[k])
                    vals0
                else
                    _arr2d_values_for_chain_poly!(scratch, arr, d, fam.off_mid[k], arr.chains[cid])
                end
                _points_from_index_packed_and_values!(scratch.points_a, bidxM, vals)
                _points_from_index_packed_and_values!(scratch.points_b, bidxN, vals)
            else
                _points_from_index_packed_and_values!(scratch.points_a, bidxM, fam.vals_pool, s)
                _points_from_index_packed_and_values!(scratch.points_b, bidxN, fam.vals_pool, s)
            end

            acc += w * kernel_fn(scratch.points_a, scratch.points_b)
            sumw += w
        end
        return normalize_weights && sumw > 0 ? acc / sumw : acc
    end
end




# -----------------------------------------------------------------------------
# Projected distances / projected invariants (pushforward along projections)
# -----------------------------------------------------------------------------

"""
    ProjectedArrangement1D

A single monotone projection from a finite region poset `Q` to a chain `C`.

Fields:
- `Q`: source region poset
- `C`: target chain poset
- `f`: EncodingMap Q -> C
- `chain`: the chain indices 1:C.n (stored as a UnitRange for zero allocation)
- `values`: endpoint coordinates for the chain (length C.n+1), suitable for `slice_barcode`
- `dir`: the direction vector used to define this projection (for bookkeeping)
"""
struct ProjectedArrangement1D{P<:AbstractPoset}
    Q::P
    C::FinitePoset
    f::EncodingMap
    chain::UnitRange{Int}
    values::Vector{Float64}
    dir::Vector{Float64}
end

"""
    ProjectedArrangement

A family of 1D projections sharing the same source poset `Q`.
"""
struct ProjectedArrangement{P<:AbstractPoset}
    Q::P
    projections::Vector{ProjectedArrangement1D{P}}
end

# Internal: build the chain poset {1,...,k} with i <= j ordering.
function _chain_poset(k::Int)
    leq = BitMatrix(undef, k, k)
    @inbounds for i in 1:k, j in 1:k
        leq[i,j] = (i <= j)
    end
    return FinitePoset(k, leq; check=false)
end

# Internal: minimal isotone majorant (upper closure) on a finite poset.
# t(p) = max_{q <= p} s(q). This guarantees monotonicity.
function _monotone_upper_closure(Q::AbstractPoset, s::AbstractVector{<:Real})
    n = nvertices(Q)
    t = Vector{Float64}(undef, n)
    @inbounds for p in 1:n
        m = -Inf
        for q in downset_indices(Q, p)
            v = float(s[q])
            if v > m
                m = v
            end
        end
        t[p] = m
    end
    return t
end

"""
    projected_arrangement(Q::AbstractPoset, values; enforce_monotone=:upper)

Build and return a [`ProjectedArrangement`](@ref) containing one 1D projection
map determined by a real-valued function on the elements of `Q`.

If `enforce_monotone=:upper`, we replace `values` by its minimal isotone majorant
so that the resulting map `Q -> chain` is guaranteed monotone.

Inspect the result first with [`projected_arrangement_summary`](@ref) before
building barcode caches on top of it.
"""
function projected_arrangement(Q::AbstractPoset, values::AbstractVector{<:Real};
                               enforce_monotone::Symbol = :upper,
                               dir = nothing)
    length(values) == nvertices(Q) || error("values must have length nvertices(Q)")

    t = enforce_monotone == :upper ? _monotone_upper_closure(Q, values) :
        Float64.(values)

    uvals = sort(unique(t))
    k = length(uvals)
    C = _chain_poset(k)

    # Map each region value to its rank in the sorted unique list.
    pi_of_q = Vector{Int}(undef, nvertices(Q))
    @inbounds for i in 1:nvertices(Q)
        pi_of_q[i] = searchsortedfirst(uvals, t[i])
    end

    f = EncodingMap(Q, C, pi_of_q)
    vals_ext = _extend_values(uvals)

    d = dir === nothing ? Float64[] : Float64.(collect(dir))
    proj = ProjectedArrangement1D(Q, C, f, 1:k, vals_ext, d)
    return ProjectedArrangement(Q, [proj])
end

# Dot product helper (works for tuples or vectors).
@inline function _dot(dir::AbstractVector{<:Real}, x)
    s = 0.0
    @inbounds for i in eachindex(dir)
        s += float(dir[i]) * float(x[i])
    end
    return s
end

@inline function _dot(dir::NTuple{N,<:Real}, x) where {N}
    s = 0.0
    @inbounds for i in 1:N
        s += float(dir[i]) * float(x[i])
    end
    return s
end

@inline function _l2_norm(dir)
    return sqrt(_dot(dir, dir))
end

"""
    projected_arrangement(pi::PLikeEncodingMap; dirs=nothing, n_dirs=32, normalize=:L1, ...)

Build and return a [`ProjectedArrangement`](@ref) family from the encoding
geometry `pi`.

Each direction `dir` defines a linear functional x -> dot(dir, x) evaluated on a
representative point for each region. The resulting real values are then
monotone-closed on the region poset, collapsed to a chain, and stored for reuse.

This is the cheap projection-family workflow for repeated projected distances or
kernels. Build the arrangement once, inspect it with
[`projected_arrangement_summary`](@ref), then attach one cache per module.
"""
function projected_arrangement(pi::PLikeEncodingMap;
                               dirs::Union{Nothing,AbstractVector}=nothing,
                               n_dirs::Int = 32,
                               max_den::Int = 8,
                               include_axes::Bool = false,
                               normalize::Symbol = :L1,
                               enforce_monotone::Symbol = :upper,
                               Q::Union{Nothing,AbstractPoset}=nothing,
                               poset_kind::Symbol = :signature,
                               cache::Union{Nothing,EncodingCache}=nothing,
                               threads::Bool = (Threads.nthreads() > 1))

    # Determine the region poset Q (either provided or reconstructed from pi).
    Qposet = (Q === nothing) ? _region_poset(pi; poset_kind = poset_kind, cache=cache) : Q
    nvertices(Qposet) == _nregions_encoding(pi) || error(
        "projected_arrangement: incompatible Q; nvertices(Q)=$(nvertices(Qposet)) but encoding has $(_nregions_encoding(pi)) regions"
    )

    dirs === nothing && (dirs = default_directions(dimension(pi); n_dirs=n_dirs,
                                                   max_den=max_den,
                                                   include_axes=include_axes,
                                                   normalize=normalize))

    arrs = Vector{ProjectedArrangement1D{typeof(Qposet)}}(undef, length(dirs))
    if threads && Threads.nthreads() > 1
        Threads.@threads for j in eachindex(dirs)
            dir = dirs[j]
            vals = _region_values(pi, x -> _dot(dir, x))
            tmp = projected_arrangement(Qposet, vals; enforce_monotone=enforce_monotone, dir=dir)
            arrs[j] = tmp.projections[1]
        end
    else
        for (j,dir) in enumerate(dirs)
            vals = _region_values(pi, x -> _dot(dir, x))
            tmp = projected_arrangement(Qposet, vals; enforce_monotone=enforce_monotone, dir=dir)
            arrs[j] = tmp.projections[1]
        end
    end
    return ProjectedArrangement(Qposet, arrs)
end

"""
    ProjectedBarcodeCache

Cache the 1D barcodes obtained by pushing a module forward along each projection
in a fixed `ProjectedArrangement`.

This is the recommended workflow:
- build `arr = projected_arrangement(pi; ...)` once,
- build caches for many modules,
- compare rapidly via `projected_distance` / `projected_kernel`.
"""
mutable struct ProjectedBarcodeCache{K}
    arrangement::ProjectedArrangement
    M::PModule{K}
    side::Symbol
    packed_barcodes::Vector{Union{Nothing,PackedFloatBarcode}}
    n_computed::Int
end

"""
    ProjectedBarcodesResult

Typed owner-level result wrapper for a batch of projected barcodes.

This stores:
- `barcodes`: the projected 1D barcodes,
- `inds`: the projection indices used to extract them,
- `dirs`: the corresponding projection directions.

Mathematically, this is a selected family of 1-parameter pushforward barcodes
obtained from a fixed projected arrangement. Use this when you want the actual
per-projection barcodes. If you only need a scalar comparison between two
modules, prefer [`projected_distance`](@ref) or [`projected_kernel`](@ref)
instead of materializing the whole family.

Use [`projected_barcodes`](@ref), [`projection_indices`](@ref), and
[`projection_directions`](@ref) instead of reading fields directly. Prefer
`describe(result)` when you only need a cheap summary of the batch.
"""
struct ProjectedBarcodesResult{B,I,D}
    barcodes::B
    inds::I
    dirs::D
end

"""
    ProjectedDistancesResult

Typed owner-level result wrapper for a batch of projected barcode distances.

This stores:
- `distances`: the per-projection distance values,
- `inds`: the projection indices used in the batch,
- `dirs`: the corresponding projection directions,
- `dist`: the distance family used to compute the values.

Mathematically, this is the per-projection distance family before any final
aggregation. Use it when you want to inspect how disagreement varies by
projection direction. If you only need one scalar summary, prefer
[`projected_distance`](@ref), which aggregates this result using the requested
weighting/aggregation mode.

Use [`projected_distances`](@ref), [`projection_indices`](@ref),
[`projection_directions`](@ref), and [`distance_metric`](@ref) instead of field
archaeology. Prefer `describe(result)` for cheap-first inspection before
materializing downstream aggregates.
"""
struct ProjectedDistancesResult{V,I,D,M}
    distances::V
    inds::I
    dirs::D
    dist::M
end

# -----------------------------------------------------------------------------
# Fibered2D UX surface: accessors, summaries, and validation
# -----------------------------------------------------------------------------

"""
    Fibered2DValidationSummary

Notebook-friendly wrapper for raw validation reports returned by the
`Fibered2D` `check_*` helpers.
"""
struct Fibered2DValidationSummary{R}
    report::R
end

"""
    fibered2d_validation_summary(report) -> Fibered2DValidationSummary

Wrap a raw `Fibered2D` validation report in a compact display-oriented
container.
"""
@inline fibered2d_validation_summary(report::NamedTuple) = Fibered2DValidationSummary(report)

@inline _fibered_issue_report(kind::Symbol, valid::Bool; kwargs...) = (; kind, valid, kwargs...)

@inline function _throw_invalid_fibered2d(fn::Symbol, issues::Vector{String})
    throw(ArgumentError(string(fn) * ": " * join(issues, " ")))
end

@inline _projected_dir_dim(proj::ProjectedArrangement1D) =
    isempty(proj.dir) ? nothing : length(proj.dir)

@inline _vector_range(v) = isempty(v) ? nothing : (first(v), last(v))

@inline function _all_finite_real(xs)
    return all(x -> x isa Real && isfinite(Float64(x)), xs)
end

@inline function _as_real_vector2(x)
    if x isa AbstractVector || x isa Tuple
        try
            return Float64.(collect(x))
        catch
            return nothing
        end
    end
    return nothing
end

@inline function _fibered_boundary_note(flag::Bool)
    return flag ? "query lies on an arrangement boundary; tie_break changes the selected cell." : nothing
end

"""
    source_encoding(arr::FiberedArrangement2D)

Return the encoding geometry used to build a fibered arrangement.
"""
@inline source_encoding(arr::FiberedArrangement2D) = arr.pi

"""
    working_box(arr)

Return the axis-aligned box used to clip exact 2D fibered queries.
"""
@inline working_box(arr::FiberedArrangement2D) = arr.box
@inline working_box(cache::FiberedBarcodeCache2D) = working_box(shared_arrangement(cache))
@inline working_box(fam::FiberedSliceFamily2D) = working_box(source_arrangement(fam))

"""
    direction_representatives(arr::FiberedArrangement2D)

Return the representative directions, one per direction cell of the augmented
fibered arrangement.
"""
@inline direction_representatives(arr::FiberedArrangement2D) = arr.dir_reps

"""
    slope_breaks(arr::FiberedArrangement2D)

Return the critical slope breaks separating direction cells in a
[`FiberedArrangement2D`](@ref).
"""
@inline slope_breaks(arr::FiberedArrangement2D) = arr.slope_breaks

@inline ambient_dim(::FiberedArrangement2D) = 2
@inline ambient_dim(cache::FiberedBarcodeCache2D) = ambient_dim(shared_arrangement(cache))
@inline ambient_dim(fam::FiberedSliceFamily2D) = ambient_dim(source_arrangement(fam))

"""
    ncells(arr::FiberedArrangement2D) -> Int

Return the total number of arrangement cells in the exact 2D fibered cache.
"""
@inline ncells(arr::FiberedArrangement2D)::Int = arr.total_cells

"""
    computed_cell_count(arr::FiberedArrangement2D) -> Int

Return how many arrangement cells have already been materialized in the lazy
fibered arrangement cache.
"""
@inline computed_cell_count(arr::FiberedArrangement2D)::Int = arr.n_cell_computed

"""
    chain_count(arr::FiberedArrangement2D) -> Int

Return the number of distinct slice chains currently registered in the
arrangement cache.
"""
@inline chain_count(arr::FiberedArrangement2D)::Int = length(arr.chains)

"""
    backend(arr::FiberedArrangement2D)

Return the exact geometry backend used by a fibered arrangement (`:boxes` or
`:poly`).
"""
@inline backend(arr::FiberedArrangement2D) = arr.backend
@inline backend(cache::FiberedBarcodeCache2D) = backend(shared_arrangement(cache))
@inline backend(fam::FiberedSliceFamily2D) = backend(source_arrangement(fam))

"""
    shared_arrangement(cache)

Return the arrangement object shared by a fibered or projected barcode cache.
"""
@inline shared_arrangement(cache::FiberedBarcodeCache2D) = cache.arrangement
@inline shared_arrangement(cache::ProjectedBarcodeCache) = cache.arrangement

"""
    cached_barcode_count(cache::FiberedBarcodeCache2D) -> Int

Return how many chain-index barcodes have been materialized in a fibered cache.
"""
@inline cached_barcode_count(cache::FiberedBarcodeCache2D)::Int = cache.n_barcode_computed

"""
    source_arrangement(fam::FiberedSliceFamily2D)

Return the arrangement from which a cached exact slice family was built.
"""
@inline source_arrangement(fam::FiberedSliceFamily2D) = fam.arrangement

"""
    unique_chain_count(fam::FiberedSliceFamily2D) -> Int

Return the number of distinct slice chains represented in a cached exact slice
family.
"""
@inline unique_chain_count(fam::FiberedSliceFamily2D)::Int = length(fam.unique_chain_ids)

"""
    direction_weight_scheme(fam::FiberedSliceFamily2D)

Return the direction-weighting scheme used when building a fibered slice
family.
"""
@inline direction_weight_scheme(fam::FiberedSliceFamily2D) = fam.direction_weight_scheme

"""
    stores_values(fam::FiberedSliceFamily2D) -> Bool

Return whether the fibered slice family stores boundary parameter values for its
representative slices.
"""
@inline stores_values(fam::FiberedSliceFamily2D)::Bool = fam.store_values

"""
    slice_direction(fam, k)

Return the representative slice direction used for the `k`th cached family
slice.
"""
@inline slice_direction(fam::FiberedSliceFamily2D, k::Integer) =
    direction_representatives(source_arrangement(fam))[fam.dir_idx[k]]

"""
    slice_offset(fam, k)

Return the representative normal offset for the `k`th cached family slice.
"""
@inline slice_offset(fam::FiberedSliceFamily2D, k::Integer) = fam.off_mid[k]

"""
    slice_offset_interval(fam, k)

Return the offset interval delimiting the arrangement cell represented by the
`k`th cached family slice.
"""
@inline slice_offset_interval(fam::FiberedSliceFamily2D, k::Integer) = (fam.off0[k], fam.off1[k])

"""
    slice_chain_id(fam, k) -> Int

Return the arrangement chain identifier attached to the `k`th cached family
slice.
"""
@inline slice_chain_id(fam::FiberedSliceFamily2D, k::Integer)::Int = fam.chain_id[k]

@inline source_poset(proj::ProjectedArrangement1D) = proj.Q
@inline source_poset(arr::ProjectedArrangement) = arr.Q

"""
    projections(arr::ProjectedArrangement)

Return the stored 1D projections in a projected-arrangement family.
"""
@inline projections(arr::ProjectedArrangement) = arr.projections

"""
    nprojections(arr::ProjectedArrangement) -> Int

Return how many 1D projections are stored in a projected arrangement family.
"""
@inline nprojections(arr::ProjectedArrangement)::Int = length(projections(arr))

"""
    projection_direction(proj::ProjectedArrangement1D)

Return the direction vector used to define a projected arrangement slice.
"""
@inline projection_direction(proj::ProjectedArrangement1D) = proj.dir

"""
    projection_values(proj::ProjectedArrangement1D)

Return the extended chain endpoint values used to read barcodes on a projected
slice.
"""
@inline projection_values(proj::ProjectedArrangement1D) = proj.values

"""
    projection_chain(proj::ProjectedArrangement1D)

Return the target chain indexing a projected arrangement slice.
"""
@inline projection_chain(proj::ProjectedArrangement1D) = proj.chain

@inline source_module(cache::FiberedBarcodeCache2D) = cache.M
@inline source_module(cache::ProjectedBarcodeCache) = cache.M

"""
    computed_projection_count(cache::ProjectedBarcodeCache) -> Int

Return how many projected barcodes have already been realized in a projected
barcode cache.
"""
@inline computed_projection_count(cache::ProjectedBarcodeCache)::Int = cache.n_computed

"""
    slice_chain(result::FiberedSliceResult)
    slice_values(result::FiberedSliceResult)
    slice_barcode(result::FiberedSliceResult)

Semantic accessors for a [`FiberedSliceResult`](@ref).

Use these accessors instead of reading `result.chain`, `result.values`, or
`result.barcode` directly.
"""
@inline slice_chain(result::FiberedSliceResult) = result.chain
@inline slice_values(result::FiberedSliceResult) = result.values
@inline slice_barcode(result::FiberedSliceResult) = result.barcode

"""
    projected_barcodes(result::ProjectedBarcodesResult)
    projected_distances(result::ProjectedDistancesResult)
    projection_indices(result)
    projection_directions(result)
    distance_metric(result)

Semantic accessors for typed projected batch-query results.
"""
@inline projected_barcodes(result::ProjectedBarcodesResult) = result.barcodes
@inline projected_distances(result::ProjectedDistancesResult) = result.distances
@inline projection_indices(result::Union{ProjectedBarcodesResult,ProjectedDistancesResult}) = result.inds
@inline projection_directions(result::Union{ProjectedBarcodesResult,ProjectedDistancesResult}) = result.dirs
@inline distance_metric(result::ProjectedDistancesResult) = result.dist

@inline _fibered_barcode_interval_count(bc) = length(bc)
@inline _fibered_barcode_total_multiplicity(bc) = isempty(bc) ? 0 : sum(values(bc))

@inline function _fibered2d_describe(arr::FiberedArrangement2D)
    return (;
        kind=:fibered_arrangement_2d,
        backend=backend(arr),
        ambient_dim=ambient_dim(arr),
        working_box=working_box(arr),
        normalize_dirs=arr.normalize_dirs,
        include_axes=arr.include_axes,
        strict=arr.strict,
        direction_cells=length(direction_representatives(arr)),
        offset_cells_per_direction=Tuple(arr.noff),
        total_cells=ncells(arr),
        computed_cells=computed_cell_count(arr),
        chain_count=chain_count(arr),
        cached_slice_families=length(arr.slice_family_cache),
    )
end

@inline function _fibered2d_describe(cache::FiberedBarcodeCache2D)
    arr = shared_arrangement(cache)
    return (;
        kind=:fibered_barcode_cache_2d,
        backend=backend(cache),
        ambient_dim=ambient_dim(cache),
        working_box=working_box(cache),
        normalize_dirs=arr.normalize_dirs,
        include_axes=arr.include_axes,
        strict=arr.strict,
        direction_cells=length(direction_representatives(arr)),
        total_cells=ncells(arr),
        computed_cells=computed_cell_count(arr),
        chain_count=chain_count(arr),
        cached_barcodes=cached_barcode_count(cache),
        cached_slice_families=length(arr.slice_family_cache),
        family_payloads=length(cache.family_barcode_payloads),
        distance_payloads=length(cache.distance_payloads),
    )
end

@inline function _fibered2d_describe(fam::FiberedSliceFamily2D)
    arr = source_arrangement(fam)
    return (;
        kind=:fibered_slice_family_2d,
        backend=backend(fam),
        ambient_dim=ambient_dim(fam),
        working_box=working_box(fam),
        direction_cells=length(direction_representatives(arr)),
        total_cells=ncells(arr),
        nslices=nslices(fam),
        unique_chain_count=unique_chain_count(fam),
        direction_weight_scheme=direction_weight_scheme(fam),
        stores_values=stores_values(fam),
    )
end

@inline function _fibered2d_describe(proj::ProjectedArrangement1D)
    return (;
        kind=:projected_arrangement_1d,
        source_poset_size=nvertices(source_poset(proj)),
        ambient_dim=_projected_dir_dim(proj),
        chain_length=length(projection_chain(proj)),
        value_count=length(projection_values(proj)),
        has_direction=!isempty(projection_direction(proj)),
        value_range=_vector_range(projection_values(proj)),
    )
end

@inline function _fibered2d_describe(arr::ProjectedArrangement)
    projs = projections(arr)
    chain_lengths = isempty(projs) ? Int[] : [length(projection_chain(proj)) for proj in projs]
    dims = Tuple(unique(_projected_dir_dim(proj) for proj in projs))
    return (;
        kind=:projected_arrangement,
        source_poset_size=nvertices(source_poset(arr)),
        nprojections=nprojections(arr),
        ambient_dims=dims,
        chain_length_range=isempty(chain_lengths) ? nothing : (minimum(chain_lengths), maximum(chain_lengths)),
        with_direction_count=count(proj -> !isempty(projection_direction(proj)), projs),
    )
end

@inline function _fibered2d_describe(cache::ProjectedBarcodeCache)
    arr = shared_arrangement(cache)
    return (;
        kind=:projected_barcode_cache,
        source_poset_size=nvertices(source_poset(arr)),
        source_module_type=typeof(source_module(cache)),
        side=cache.side,
        nprojections=nprojections(arr),
        computed_projections=computed_projection_count(cache),
    )
end

@inline function _fibered2d_describe(result::FiberedSliceResult)
    bc = slice_barcode(result)
    return (;
        kind=:fibered_slice_result,
        chain_length=length(slice_chain(result)),
        value_count=length(slice_values(result)),
        barcode_intervals=_fibered_barcode_interval_count(bc),
        barcode_total_multiplicity=_fibered_barcode_total_multiplicity(bc),
        empty=isempty(slice_chain(result)),
    )
end

@inline function _fibered2d_describe(result::ProjectedBarcodesResult)
    bars = projected_barcodes(result)
    interval_counts = isempty(bars) ? Int[] : [length(bc) for bc in bars]
    return (;
        kind=:projected_barcodes_result,
        nprojections=length(projection_indices(result)),
        projection_indices=Tuple(projection_indices(result)),
        direction_count=length(projection_directions(result)),
        interval_count_range=isempty(interval_counts) ? nothing : (minimum(interval_counts), maximum(interval_counts)),
    )
end

@inline function _fibered2d_describe(result::ProjectedDistancesResult)
    dists = projected_distances(result)
    return (;
        kind=:projected_distances_result,
        nprojections=length(projection_indices(result)),
        projection_indices=Tuple(projection_indices(result)),
        distance_metric=distance_metric(result),
        distance_range=isempty(dists) ? nothing : (minimum(dists), maximum(dists)),
        mean_distance=isempty(dists) ? nothing : mean(dists),
    )
end

describe(arr::FiberedArrangement2D) = _fibered2d_describe(arr)
describe(cache::FiberedBarcodeCache2D) = _fibered2d_describe(cache)
describe(fam::FiberedSliceFamily2D) = _fibered2d_describe(fam)
describe(proj::ProjectedArrangement1D) = _fibered2d_describe(proj)
describe(arr::ProjectedArrangement) = _fibered2d_describe(arr)
describe(cache::ProjectedBarcodeCache) = _fibered2d_describe(cache)
describe(result::FiberedSliceResult) = _fibered2d_describe(result)
describe(result::ProjectedBarcodesResult) = _fibered2d_describe(result)
describe(result::ProjectedDistancesResult) = _fibered2d_describe(result)

function Base.show(io::IO, arr::FiberedArrangement2D)
    d = _fibered2d_describe(arr)
    print(io, "FiberedArrangement2D(backend=", d.backend,
          ", cells=", d.total_cells, ", computed=", d.computed_cells, ")")
end

function Base.show(io::IO, ::MIME"text/plain", arr::FiberedArrangement2D)
    d = _fibered2d_describe(arr)
    print(io, "FiberedArrangement2D",
          "\n  backend = ", d.backend,
          "\n  ambient_dim = ", d.ambient_dim,
          "\n  working_box = ", d.working_box,
          "\n  normalize_dirs = ", d.normalize_dirs,
          "\n  include_axes = ", d.include_axes,
          "\n  strict = ", d.strict,
          "\n  direction_cells = ", d.direction_cells,
          "\n  total_cells = ", d.total_cells,
          "\n  computed_cells = ", d.computed_cells,
          "\n  chain_count = ", d.chain_count)
end

function Base.show(io::IO, cache::FiberedBarcodeCache2D)
    d = _fibered2d_describe(cache)
    print(io, "FiberedBarcodeCache2D(backend=", d.backend,
          ", cached_barcodes=", d.cached_barcodes, ")")
end

function Base.show(io::IO, ::MIME"text/plain", cache::FiberedBarcodeCache2D)
    d = _fibered2d_describe(cache)
    print(io, "FiberedBarcodeCache2D",
          "\n  backend = ", d.backend,
          "\n  ambient_dim = ", d.ambient_dim,
          "\n  working_box = ", d.working_box,
          "\n  total_cells = ", d.total_cells,
          "\n  computed_cells = ", d.computed_cells,
          "\n  chain_count = ", d.chain_count,
          "\n  cached_barcodes = ", d.cached_barcodes,
          "\n  cached_slice_families = ", d.cached_slice_families)
end

function Base.show(io::IO, fam::FiberedSliceFamily2D)
    d = _fibered2d_describe(fam)
    print(io, "FiberedSliceFamily2D(nslices=", d.nslices,
          ", unique_chains=", d.unique_chain_count, ")")
end

function Base.show(io::IO, ::MIME"text/plain", fam::FiberedSliceFamily2D)
    d = _fibered2d_describe(fam)
    print(io, "FiberedSliceFamily2D",
          "\n  backend = ", d.backend,
          "\n  ambient_dim = ", d.ambient_dim,
          "\n  working_box = ", d.working_box,
          "\n  nslices = ", d.nslices,
          "\n  unique_chain_count = ", d.unique_chain_count,
          "\n  direction_weight_scheme = ", d.direction_weight_scheme,
          "\n  stores_values = ", d.stores_values)
end

function Base.show(io::IO, proj::ProjectedArrangement1D)
    d = _fibered2d_describe(proj)
    print(io, "ProjectedArrangement1D(chain_length=", d.chain_length, ")")
end

function Base.show(io::IO, ::MIME"text/plain", proj::ProjectedArrangement1D)
    d = _fibered2d_describe(proj)
    print(io, "ProjectedArrangement1D",
          "\n  source_poset_size = ", d.source_poset_size,
          "\n  ambient_dim = ", d.ambient_dim,
          "\n  chain_length = ", d.chain_length,
          "\n  value_count = ", d.value_count,
          "\n  has_direction = ", d.has_direction,
          "\n  value_range = ", d.value_range)
end

function Base.show(io::IO, arr::ProjectedArrangement)
    d = _fibered2d_describe(arr)
    print(io, "ProjectedArrangement(nprojections=", d.nprojections, ")")
end

function Base.show(io::IO, ::MIME"text/plain", arr::ProjectedArrangement)
    d = _fibered2d_describe(arr)
    print(io, "ProjectedArrangement",
          "\n  source_poset_size = ", d.source_poset_size,
          "\n  nprojections = ", d.nprojections,
          "\n  ambient_dims = ", d.ambient_dims,
          "\n  chain_length_range = ", d.chain_length_range,
          "\n  with_direction_count = ", d.with_direction_count)
end

function Base.show(io::IO, cache::ProjectedBarcodeCache)
    d = _fibered2d_describe(cache)
    print(io, "ProjectedBarcodeCache(side=", d.side,
          ", computed=", d.computed_projections, ")")
end

function Base.show(io::IO, ::MIME"text/plain", cache::ProjectedBarcodeCache)
    d = _fibered2d_describe(cache)
    print(io, "ProjectedBarcodeCache",
          "\n  source_poset_size = ", d.source_poset_size,
          "\n  side = ", d.side,
          "\n  nprojections = ", d.nprojections,
          "\n  computed_projections = ", d.computed_projections,
          "\n  source_module_type = ", d.source_module_type)
end

function Base.show(io::IO, result::FiberedSliceResult)
    d = _fibered2d_describe(result)
    print(io, "FiberedSliceResult(chain_length=", d.chain_length,
          ", barcode_intervals=", d.barcode_intervals, ")")
end

function Base.show(io::IO, ::MIME"text/plain", result::FiberedSliceResult)
    d = _fibered2d_describe(result)
    print(io, "FiberedSliceResult",
          "\n  chain_length = ", d.chain_length,
          "\n  value_count = ", d.value_count,
          "\n  barcode_intervals = ", d.barcode_intervals,
          "\n  barcode_total_multiplicity = ", d.barcode_total_multiplicity,
          "\n  empty = ", d.empty)
end

function Base.show(io::IO, result::ProjectedBarcodesResult)
    d = _fibered2d_describe(result)
    print(io, "ProjectedBarcodesResult(nprojections=", d.nprojections, ")")
end

function Base.show(io::IO, ::MIME"text/plain", result::ProjectedBarcodesResult)
    d = _fibered2d_describe(result)
    print(io, "ProjectedBarcodesResult",
          "\n  nprojections = ", d.nprojections,
          "\n  projection_indices = ", d.projection_indices,
          "\n  direction_count = ", d.direction_count,
          "\n  interval_count_range = ", d.interval_count_range)
end

function Base.show(io::IO, result::ProjectedDistancesResult)
    d = _fibered2d_describe(result)
    print(io, "ProjectedDistancesResult(metric=", d.distance_metric,
          ", nprojections=", d.nprojections, ")")
end

function Base.show(io::IO, ::MIME"text/plain", result::ProjectedDistancesResult)
    d = _fibered2d_describe(result)
    print(io, "ProjectedDistancesResult",
          "\n  distance_metric = ", d.distance_metric,
          "\n  nprojections = ", d.nprojections,
          "\n  projection_indices = ", d.projection_indices,
          "\n  distance_range = ", d.distance_range,
          "\n  mean_distance = ", d.mean_distance)
end

function Base.show(io::IO, summary::Fibered2DValidationSummary)
    r = summary.report
    print(io, "Fibered2DValidationSummary(kind=", r.kind,
          ", valid=", r.valid,
          ", issues=", length(get(r, :issues, ())), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::Fibered2DValidationSummary)
    r = summary.report
    println(io, "Fibered2DValidationSummary")
    println(io, "  kind = ", r.kind)
    println(io, "  valid = ", r.valid)
    println(io, "  issues = ", length(get(r, :issues, ())))
end

"""
    check_fibered_direction(dir; throw=false) -> NamedTuple

Validate a 2D direction intended for exact fibered queries.

This helper checks that `dir` is a finite nonzero 2-vector in the nonnegative
quadrant. Axis directions are reported but not rejected, since whether they are
allowed depends on the arrangement's `include_axes` contract. Use
`check_fibered_query(...)` when you want arrangement-aware validation.
"""
function check_fibered_direction(dir; throw::Bool=false)
    issues = String[]
    vec = _as_real_vector2(dir)
    vec === nothing && push!(issues, "direction must be given as a 2-vector or 2-tuple of reals.")
    if vec !== nothing
        length(vec) == 2 || push!(issues, "direction must have length 2.")
        _all_finite_real(vec) || push!(issues, "direction entries must be finite reals.")
        length(vec) == 2 && any(x -> x < 0.0, vec) && push!(issues, "direction must lie in the nonnegative quadrant.")
        length(vec) == 2 && iszero(vec[1]) && iszero(vec[2]) && push!(issues, "direction must be nonzero.")
    end
    axis_direction = vec === nothing ? nothing : any(iszero, vec)
    valid = isempty(issues)
    throw && !valid && _throw_invalid_fibered2d(:check_fibered_direction, issues)
    return _fibered_issue_report(:fibered_direction, valid;
                                 direction=vec,
                                 ambient_dim=vec === nothing ? nothing : length(vec),
                                 axis_direction=axis_direction,
                                 issues=issues)
end

"""
    check_projected_direction(dir; throw=false) -> NamedTuple

Validate a direction vector used to build projected arrangements.

Unlike exact fibered directions, projected directions need only be finite and
nonzero; they are not restricted to the nonnegative quadrant.
"""
function check_projected_direction(dir; throw::Bool=false)
    issues = String[]
    vec = _as_real_vector2(dir)
    vec === nothing && push!(issues, "projected direction must be given as a real vector or tuple.")
    if vec !== nothing
        !isempty(vec) || push!(issues, "projected direction must have positive length.")
        _all_finite_real(vec) || push!(issues, "projected direction entries must be finite reals.")
        norm(vec) > 0.0 || push!(issues, "projected direction must be nonzero.")
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_fibered2d(:check_projected_direction, issues)
    return _fibered_issue_report(:projected_direction, valid;
                                 direction=vec,
                                 ambient_dim=vec === nothing ? nothing : length(vec),
                                 issues=issues)
end

"""
    check_fibered_offset(offset; throw=false) -> NamedTuple

Validate a normal-offset scalar for an exact fibered query.
"""
function check_fibered_offset(offset; throw::Bool=false)
    issues = String[]
    offset isa Real || push!(issues, "offset must be a real scalar.")
    offset isa Real && !isfinite(Float64(offset)) && push!(issues, "offset must be finite.")
    valid = isempty(issues)
    throw && !valid && _throw_invalid_fibered2d(:check_fibered_offset, issues)
    return _fibered_issue_report(:fibered_offset, valid;
                                 offset=offset isa Real ? Float64(offset) : nothing,
                                 issues=issues)
end

"""
    check_fibered_basepoint(x0; throw=false) -> NamedTuple

Validate a basepoint used to specify a 2D fibered slice line.
"""
function check_fibered_basepoint(x0; throw::Bool=false)
    issues = String[]
    vec = _as_real_vector2(x0)
    vec === nothing && push!(issues, "basepoint must be given as a 2-vector or 2-tuple of reals.")
    if vec !== nothing
        length(vec) == 2 || push!(issues, "basepoint must have length 2.")
        _all_finite_real(vec) || push!(issues, "basepoint entries must be finite reals.")
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_fibered2d(:check_fibered_basepoint, issues)
    return _fibered_issue_report(:fibered_basepoint, valid;
                                 basepoint=vec,
                                 ambient_dim=vec === nothing ? nothing : length(vec),
                                 issues=issues)
end

@inline function _fibered_query_arrangement(x)
    if x isa FiberedArrangement2D
        return x
    elseif x isa FiberedBarcodeCache2D
        return shared_arrangement(x)
    end
    throw(ArgumentError("expected a FiberedArrangement2D or FiberedBarcodeCache2D"))
end

"""
    check_fibered_query(arr_or_cache, dir, off_or_x0; throw=false) -> NamedTuple

Validate an arrangement-aware exact fibered query.

This helper checks the direction and offset/basepoint shapes, confirms that the
query is compatible with the arrangement normalization/axis contract, and
reports whether the query lies on a boundary where `tie_break` changes the
selected arrangement cell.
"""
function check_fibered_query(arr_or_cache, dir, off_or_x0; throw::Bool=false)
    arr = _fibered_query_arrangement(arr_or_cache)
    issues = String[]
    dir_report = check_fibered_direction(dir; throw=false)
    append!(issues, dir_report.issues)
    query_kind = off_or_x0 isa Real ? :offset : (off_or_x0 isa AbstractVector || off_or_x0 isa Tuple ? :basepoint : :invalid)
    off_report = query_kind === :offset ? check_fibered_offset(off_or_x0; throw=false) : nothing
    base_report = query_kind === :basepoint ? check_fibered_basepoint(off_or_x0; throw=false) : nothing
    query_kind === :offset && append!(issues, off_report.issues)
    query_kind === :basepoint && append!(issues, base_report.issues)
    query_kind === :invalid && push!(issues, "query must be specified by a real offset or a 2D basepoint.")

    cell_up = nothing
    cell_down = nothing
    tie_break_relevant = false
    if isempty(issues)
        try
            query_arg = query_kind === :offset ? Float64(off_or_x0) : Float64.(collect(off_or_x0))
            cell_up = fibered_cell_id(arr, dir, query_arg; tie_break=:up)
            cell_down = fibered_cell_id(arr, dir, query_arg; tie_break=:down)
            tie_break_relevant = cell_up != cell_down
        catch err
            push!(issues, sprint(showerror, err))
        end
    end

    valid = isempty(issues)
    throw && !valid && _throw_invalid_fibered2d(:check_fibered_query, issues)
    return _fibered_issue_report(:fibered_query, valid;
                                 backend=backend(arr),
                                 query_kind=query_kind,
                                 normalize_dirs=arr.normalize_dirs,
                                 include_axes=arr.include_axes,
                                 strict=arr.strict,
                                 cell_up=cell_up,
                                 cell_down=cell_down,
                                 tie_break_relevant=tie_break_relevant,
                                 boundary_note=_fibered_boundary_note(tie_break_relevant),
                                 issues=issues)
end

"""
    check_fibered_arrangement_2d(arr; throw=false) -> NamedTuple

Validate a hand-built or externally modified [`FiberedArrangement2D`](@ref).

The report checks basic geometry/cache invariants such as cell counts, direction
metadata lengths, backend validity, and chain-id bounds. Prefer this helper
before debugging a malformed arrangement by inspecting raw fields.
"""
function check_fibered_arrangement_2d(arr::FiberedArrangement2D; throw::Bool=false)
    issues = String[]
    box = working_box(arr)
    length(box[1]) == 2 && length(box[2]) == 2 || push!(issues, "working box must have 2D endpoints.")
    backend(arr) in (:boxes, :poly) || push!(issues, "backend must be :boxes or :poly.")
    all(length(d) == 2 for d in direction_representatives(arr)) || push!(issues, "all representative directions must have length 2.")
    ndir = length(direction_representatives(arr))
    length(arr.orders) == ndir || push!(issues, "orders must have one entry per direction cell.")
    length(arr.unique_pos) == ndir || push!(issues, "unique_pos must have one entry per direction cell.")
    length(arr.noff) == ndir || push!(issues, "noff must have one entry per direction cell.")
    length(arr.start) == ndir || push!(issues, "start must have one entry per direction cell.")
    arr.total_cells == sum(arr.noff) || push!(issues, "total_cells must equal sum(noff).")
    length(arr.cell_chain_id) == ncells(arr) || push!(issues, "cell_chain_id length must equal total_cells.")
    length(arr.cell_event_start) == ncells(arr) || push!(issues, "cell_event_start length must equal total_cells.")
    length(arr.cell_event_len) == ncells(arr) || push!(issues, "cell_event_len length must equal total_cells.")
    0 <= computed_cell_count(arr) <= ncells(arr) || push!(issues, "n_cell_computed must lie between 0 and total_cells.")
    for i in 1:ndir
        arr.noff[i] >= 0 || push!(issues, "offset-cell counts must be nonnegative.")
        length(arr.unique_pos[i]) >= 1 || push!(issues, "each direction cell must record at least one unique position.")
        max(length(arr.unique_pos[i]) - 1, 0) == arr.noff[i] || push!(issues, "noff[$i] must equal length(unique_pos[$i]) - 1.")
        all(1 <= idx <= length(arr.points) for idx in arr.orders[i]) || push!(issues, "orders[$i] contains an out-of-bounds critical-point index.")
        all(1 <= idx <= length(arr.orders[i]) for idx in arr.unique_pos[i]) || push!(issues, "unique_pos[$i] contains an out-of-bounds order index.")
    end
    max_chain_id = isempty(arr.cell_chain_id) ? 0 : maximum(arr.cell_chain_id)
    max_chain_id <= chain_count(arr) || push!(issues, "cell_chain_id references a chain beyond the registered chain list.")
    valid = isempty(issues)
    throw && !valid && _throw_invalid_fibered2d(:check_fibered_arrangement_2d, issues)
    return _fibered_issue_report(:fibered_arrangement_2d, valid;
                                 backend=backend(arr),
                                 ambient_dim=ambient_dim(arr),
                                 direction_cells=ndir,
                                 total_cells=ncells(arr),
                                 computed_cells=computed_cell_count(arr),
                                 chain_count=chain_count(arr),
                                 issues=issues)
end

"""
    check_fibered_barcode_cache_2d(cache; throw=false) -> NamedTuple

Validate a [`FiberedBarcodeCache2D`](@ref).

The report checks the shared arrangement, chain-barcode storage length, and the
cached family/distance payload bookkeeping.
"""
function check_fibered_barcode_cache_2d(cache::FiberedBarcodeCache2D; throw::Bool=false)
    issues = String[]
    arr_report = check_fibered_arrangement_2d(shared_arrangement(cache); throw=false)
    append!(issues, arr_report.issues)
    length(cache.index_barcodes_packed) == chain_count(shared_arrangement(cache)) ||
        push!(issues, "index_barcodes_packed must have one slot per registered chain.")
    0 <= cached_barcode_count(cache) <= length(cache.index_barcodes_packed) ||
        push!(issues, "n_barcode_computed must lie between 0 and the number of cached chain barcodes.")
    for payload in values(cache.family_barcode_payloads)
        0 <= payload.n_computed <= length(payload.packed_barcodes) ||
            push!(issues, "family barcode payload counts must lie within their storage length.")
    end
    for payload in values(cache.distance_payloads)
        0 <= payload.n_computed <= length(payload.points) ||
            push!(issues, "distance payload counts must lie within their storage length.")
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_fibered2d(:check_fibered_barcode_cache_2d, issues)
    return _fibered_issue_report(:fibered_barcode_cache_2d, valid;
                                 backend=backend(cache),
                                 chain_count=chain_count(shared_arrangement(cache)),
                                 cached_barcodes=cached_barcode_count(cache),
                                 family_payloads=length(cache.family_barcode_payloads),
                                 distance_payloads=length(cache.distance_payloads),
                                 issues=issues)
end

"""
    check_fibered_slice_family_2d(fam; throw=false) -> NamedTuple

Validate a cached exact fibered slice family.

This checks per-slice vector lengths, representative offset intervals, cached
value-pool bounds, and chain references back into the source arrangement.
"""
function check_fibered_slice_family_2d(fam::FiberedSliceFamily2D; throw::Bool=false)
    arr = source_arrangement(fam)
    issues = String[]
    arr_report = check_fibered_arrangement_2d(arr; throw=false)
    append!(issues, arr_report.issues)
    ns = nslices(fam)
    lengths = (
        length(fam.dir_idx), length(fam.off_idx), length(fam.cell_id), length(fam.chain_id),
        length(fam.off_mid), length(fam.off0), length(fam.off1),
        length(fam.vals_start), length(fam.vals_len),
    )
    all(==(ns), lengths) || push!(issues, "all per-slice storage vectors must have length nslices(fam).")
    length(fam.dir_weight) == length(direction_representatives(arr)) ||
        push!(issues, "dir_weight must have one entry per direction cell.")
    length(fam.theta_width) == length(direction_representatives(arr)) ||
        push!(issues, "theta_width must have one entry per direction cell.")
    all(cid in fam.chain_id for cid in fam.unique_chain_ids) ||
        push!(issues, "unique_chain_ids must be a subset of the stored chain ids.")
    ncheck = min(ns, minimum(lengths))
    for k in 1:ncheck
        1 <= fam.dir_idx[k] <= length(direction_representatives(arr)) ||
            push!(issues, "slice $k references an invalid direction cell.")
        1 <= fam.cell_id[k] <= ncells(arr) ||
            push!(issues, "slice $k references an invalid arrangement cell.")
        1 <= fam.chain_id[k] <= chain_count(arr) ||
            push!(issues, "slice $k references an invalid chain id.")
        fam.off0[k] <= fam.off_mid[k] <= fam.off1[k] ||
            push!(issues, "slice $k has an invalid offset interval.")
        s = fam.vals_start[k]
        l = fam.vals_len[k]
        if s == 0 || l == 0
            stores_values(fam) && l != 0 && push!(issues, "slice $k has inconsistent empty value storage.")
        else
            1 <= s <= length(fam.vals_pool) || push!(issues, "slice $k value start is out of range.")
            s + l - 1 <= length(fam.vals_pool) || push!(issues, "slice $k value range exceeds the stored value pool.")
            l == length(arr.chains[fam.chain_id[k]]) + 1 ||
                push!(issues, "slice $k value count must match chain length plus one.")
        end
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_fibered2d(:check_fibered_slice_family_2d, issues)
    return _fibered_issue_report(:fibered_slice_family_2d, valid;
                                 nslices=ns,
                                 unique_chain_count=unique_chain_count(fam),
                                 stores_values=stores_values(fam),
                                 issues=issues)
end

"""
    check_projected_arrangement(arr; throw=false) -> NamedTuple

Validate a projected arrangement family or one of its individual projections.
"""
function check_projected_arrangement(proj::ProjectedArrangement1D; throw::Bool=false)
    issues = String[]
    first(projection_chain(proj)) == 1 || push!(issues, "projected chain must start at 1.")
    last(projection_chain(proj)) == nvertices(proj.C) || push!(issues, "projected chain must end at nvertices(C).")
    length(projection_values(proj)) == length(projection_chain(proj)) + 1 ||
        push!(issues, "projected values must have length chain_length + 1.")
    if !isempty(projection_direction(proj))
        dir_report = check_projected_direction(projection_direction(proj); throw=false)
        append!(issues, dir_report.issues)
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_fibered2d(:check_projected_arrangement, issues)
    return _fibered_issue_report(:projected_arrangement_1d, valid;
                                 source_poset_size=nvertices(source_poset(proj)),
                                 chain_length=length(projection_chain(proj)),
                                 value_count=length(projection_values(proj)),
                                 has_direction=!isempty(projection_direction(proj)),
                                 issues=issues)
end

function check_projected_arrangement(arr::ProjectedArrangement; throw::Bool=false)
    issues = String[]
    for proj in projections(arr)
        source_poset(proj) === source_poset(arr) || push!(issues, "all projections must share the same source poset.")
        append!(issues, check_projected_arrangement(proj; throw=false).issues)
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_fibered2d(:check_projected_arrangement, issues)
    return _fibered_issue_report(:projected_arrangement, valid;
                                 source_poset_size=nvertices(source_poset(arr)),
                                 nprojections=nprojections(arr),
                                 issues=issues)
end

"""
    check_projected_barcode_cache(cache; throw=false) -> NamedTuple

Validate a [`ProjectedBarcodeCache`](@ref).
"""
function check_projected_barcode_cache(cache::ProjectedBarcodeCache; throw::Bool=false)
    issues = String[]
    arr_report = check_projected_arrangement(shared_arrangement(cache); throw=false)
    append!(issues, arr_report.issues)
    length(cache.packed_barcodes) == nprojections(shared_arrangement(cache)) ||
        push!(issues, "packed_barcodes must have one slot per projection.")
    0 <= computed_projection_count(cache) <= length(cache.packed_barcodes) ||
        push!(issues, "n_computed must lie between 0 and the number of projections.")
    cache.side in (:left, :right) || push!(issues, "side must be :left or :right.")
    valid = isempty(issues)
    throw && !valid && _throw_invalid_fibered2d(:check_projected_barcode_cache, issues)
    return _fibered_issue_report(:projected_barcode_cache, valid;
                                 side=cache.side,
                                 nprojections=nprojections(shared_arrangement(cache)),
                                 computed_projections=computed_projection_count(cache),
                                 issues=issues)
end

"""
    check_fibered_cache_pair(cacheM, cacheN; throw=false) -> NamedTuple

Validate a pair of exact fibered barcode caches intended for pairwise
distance/kernel queries.
"""
function check_fibered_cache_pair(cacheM::FiberedBarcodeCache2D, cacheN::FiberedBarcodeCache2D; throw::Bool=false)
    issues = String[]
    append!(issues, check_fibered_barcode_cache_2d(cacheM; throw=false).issues)
    append!(issues, check_fibered_barcode_cache_2d(cacheN; throw=false).issues)
    shared_arrangement(cacheM) === shared_arrangement(cacheN) ||
        push!(issues, "fibered cache pairs must share the same arrangement object.")
    source_module(cacheM).field == source_module(cacheN).field ||
        push!(issues, "fibered cache pairs must use the same coefficient field.")
    valid = isempty(issues)
    throw && !valid && _throw_invalid_fibered2d(:check_fibered_cache_pair, issues)
    return _fibered_issue_report(:fibered_cache_pair, valid;
                                 shared_arrangement=(shared_arrangement(cacheM) === shared_arrangement(cacheN)),
                                 left_field=source_module(cacheM).field,
                                 right_field=source_module(cacheN).field,
                                 issues=issues)
end

"""
    check_projected_cache_pair(cacheM, cacheN; throw=false) -> NamedTuple

Validate a pair of projected barcode caches intended for per-projection
distances or kernels.
"""
function check_projected_cache_pair(cacheM::ProjectedBarcodeCache, cacheN::ProjectedBarcodeCache; throw::Bool=false)
    issues = String[]
    append!(issues, check_projected_barcode_cache(cacheM; throw=false).issues)
    append!(issues, check_projected_barcode_cache(cacheN; throw=false).issues)
    shared_arrangement(cacheM) === shared_arrangement(cacheN) ||
        push!(issues, "projected cache pairs must share the same projected arrangement object.")
    source_module(cacheM).field == source_module(cacheN).field ||
        push!(issues, "projected cache pairs must use the same coefficient field.")
    valid = isempty(issues)
    throw && !valid && _throw_invalid_fibered2d(:check_projected_cache_pair, issues)
    return _fibered_issue_report(:projected_cache_pair, valid;
                                 shared_arrangement=(shared_arrangement(cacheM) === shared_arrangement(cacheN)),
                                 left_field=source_module(cacheM).field,
                                 right_field=source_module(cacheN).field,
                                 issues=issues)
end

"""
    fibered_arrangement_summary(arr)
    fibered_cache_summary(cache)
    fibered_family_summary(fam)
    projected_arrangement_summary(arr)
    projected_cache_summary(cache)

Owner-local cheap-first summaries for the main `Fibered2D` containers.

Typical workflow:

```julia
arr = TamerOp.Advanced.fibered_arrangement_2d(pi, opts; include_axes=true)
TamerOp.Advanced.fibered_arrangement_summary(arr)

cacheM = TamerOp.Advanced.fibered_barcode_cache_2d(M, arr)
cacheN = TamerOp.Advanced.fibered_barcode_cache_2d(N, arr)
TamerOp.Advanced.fibered_cache_summary(cacheM)

barres = TamerOp.slice_barcodes(cacheM; dirs=[(1.0, 1.0)], offsets=[0.0])
fam = TamerOp.Advanced.fibered_slice_family_2d(arr)
d = TamerOp.Advanced.matching_distance_exact_2d(cacheM, cacheN; family=fam)

parr = TamerOp.Advanced.projected_arrangement(pi; dirs=[(1.0, 0.0), (0.0, 1.0)])
TamerOp.Advanced.projected_arrangement_summary(parr)
```
"""
@inline fibered_arrangement_summary(arr::FiberedArrangement2D) = describe(arr)
@inline fibered_cache_summary(cache::FiberedBarcodeCache2D) = describe(cache)
@inline fibered_family_summary(fam::FiberedSliceFamily2D) = describe(fam)
@inline projected_arrangement_summary(proj::ProjectedArrangement1D) = describe(proj)
@inline projected_arrangement_summary(arr::ProjectedArrangement) = describe(arr)
@inline projected_cache_summary(cache::ProjectedBarcodeCache) = describe(cache)

"""
    fibered_query_summary(arr_or_cache, dir, off_or_x0; tie_break=:up)

Return a cheap-first summary of one exact fibered query.

The summary reports the normalized direction, offset or basepoint, selected
cell, whether the query lies on a boundary where `tie_break` changes the cell,
and the resulting chain length. When a [`FiberedBarcodeCache2D`](@ref) is
provided, the summary also includes barcode-cardinality hints for that query.

Use this before [`fibered_slice`](@ref) when you first want to confirm which
arrangement cell a query lands in and whether `tie_break` changes the answer.
If you pass a bare [`FiberedArrangement2D`](@ref), the summary stays at the
geometry/chain-selection level. If you pass a [`FiberedBarcodeCache2D`](@ref),
it also inspects the corresponding barcode payload for that one query.
"""
function fibered_query_summary(arr_or_cache, dir, off_or_x0; tie_break::Symbol=:up)
    report = check_fibered_query(arr_or_cache, dir, off_or_x0; throw=false)
    arr = _fibered_query_arrangement(arr_or_cache)

    dir_report = check_fibered_direction(dir; throw=false)
    dnorm = dir_report.valid ? _normalize_dir(dir_report.direction, arr.normalize_dirs) : nothing
    query_kind = report.query_kind
    basepoint = query_kind === :basepoint ? _as_real_vector2(off_or_x0) : nothing
    offset = nothing
    if report.valid
        if query_kind === :offset
            offset = Float64(off_or_x0)
        else
            (n1, n2) = _normal_from_dir_2d(dnorm)
            offset = n1 * basepoint[1] + n2 * basepoint[2]
        end
    end

    cell_id = report.valid ? (tie_break === :down ? report.cell_down : report.cell_up) : nothing
    chain_len = nothing
    barcode_intervals = nothing
    barcode_total_multiplicity = nothing
    cached_chain_barcode = nothing
    if report.valid
        chain = fibered_chain(arr, dnorm, offset; tie_break=tie_break, copy=false)
        chain_len = length(chain)
        if arr_or_cache isa FiberedBarcodeCache2D
            slice = fibered_slice(arr_or_cache, dnorm, offset; tie_break=tie_break)
            bc = slice_barcode(slice)
            barcode_intervals = _fibered_barcode_interval_count(bc)
            barcode_total_multiplicity = _fibered_barcode_total_multiplicity(bc)
            if cell_id !== nothing
                chain_id = _arr2d_compute_cell!(arr, cell_id[1], cell_id[2])
                cached_chain_barcode = chain_id > 0 &&
                    chain_id <= length(arr_or_cache.index_barcodes_packed) &&
                    !isnothing(arr_or_cache.index_barcodes_packed[chain_id])
            end
        end
    end

    return (;
        kind=:fibered_query_summary,
        valid=report.valid,
        backend=backend(arr),
        query_kind=query_kind,
        normalized_direction=dnorm,
        basepoint=basepoint,
        offset=offset,
        tie_break=tie_break,
        cell_id=cell_id,
        cell_up=report.cell_up,
        cell_down=report.cell_down,
        tie_break_relevant=report.tie_break_relevant,
        boundary_note=report.boundary_note,
        chain_length=chain_len,
        barcode_intervals=barcode_intervals,
        barcode_total_multiplicity=barcode_total_multiplicity,
        cached_chain_barcode=cached_chain_barcode,
        issues=report.issues,
    )
end

"""
    projected_barcode_cache(M, arr; side=:left, precompute=false)

Build and return a [`ProjectedBarcodeCache`](@ref) on top of a precomputed
[`ProjectedArrangement`](@ref).

Cheap-first workflow:
- build the projected arrangement once,
- inspect it with [`projected_arrangement_summary`](@ref),
- build one cache per module,
- inspect cache state with [`projected_cache_summary`](@ref),
- then call [`projected_barcodes`](@ref), [`projected_distance`](@ref), or
  [`projected_kernel`](@ref).
"""
function projected_barcode_cache(M::PModule{K}, arr::ProjectedArrangement;
                                 side::Symbol = :left,
                                 precompute::Bool = false) where {K}
    side in (:left, :right) || error("side must be :left or :right")
    packed_barcodes = Vector{Union{Nothing,PackedFloatBarcode}}(undef, length(arr.projections))
    fill!(packed_barcodes, nothing)
    cache = ProjectedBarcodeCache(arr, M, side, packed_barcodes, 0)
    precompute && projected_barcodes(cache)
    return cache
end

# Lazy compute ith barcode.
function _projected_barcode(cache::ProjectedBarcodeCache, i::Int)
    return _barcode_from_packed(_projected_packed_barcode(cache, i))
end

function _projected_packed_barcode(cache::ProjectedBarcodeCache, i::Int)
    pb = cache.packed_barcodes[i]
    if pb !== nothing
        return pb
    end
    proj = cache.arrangement.projections[i]
    Mp = cache.side == :left ? pushforward_left(proj.f, cache.M; check=false) :
                               pushforward_right(proj.f, cache.M; check=false)
    pb0 = _slice_barcode_packed(Mp, proj.chain; values=proj.values, check_chain=false)
    pb = pb0 isa PackedFloatBarcode ? pb0 : _pack_float_barcode(_barcode_from_packed(pb0))
    cache.packed_barcodes[i] = pb
    cache.n_computed += 1
    return pb
end

function _prepare_projected_cache_readonly!(cache::ProjectedBarcodeCache)
    n = length(cache.arrangement.projections)
    if cache.n_computed == n
        return nothing
    end
    for i in 1:n
        cache.packed_barcodes[i] === nothing || continue
        _projected_packed_barcode(cache, i)
    end
    cache.n_computed = n
    return nothing
end

"""
    projected_barcodes(cache; inds=nothing)

Return a [`ProjectedBarcodesResult`](@ref) for all or selected projected
barcodes from a [`ProjectedBarcodeCache`](@ref).

This is the projected workflow's direct barcode-materialization path. Internal
storage stays packed until the API boundary; use [`projected_cache_summary`](@ref)
first when you only need cache state rather than the actual barcodes. The
returned wrapper keeps the selected projection indices and directions aligned
with the materialized barcode vector.

Use [`projected_barcodes`](@ref), [`projection_indices`](@ref), and
[`projection_directions`](@ref) on the result for cheap-first inspection before
working with the barcode vector itself. If you only need scalar comparisons,
prefer [`projected_distance`](@ref) or [`projected_kernel`](@ref).
"""
function projected_barcodes(cache::ProjectedBarcodeCache;
                            inds=nothing,
                            threads::Bool = (Threads.nthreads() > 1))
    n = length(cache.arrangement.projections)
    inds_vec = inds === nothing ? collect(1:n) : collect(inds)
    out = Vector{FloatBarcode}(undef, length(inds_vec))
    if threads && Threads.nthreads() > 1
        _prepare_projected_cache_readonly!(cache)
        Threads.@threads for k in eachindex(inds_vec)
            i = inds_vec[k]
            pb = cache.packed_barcodes[i]::PackedFloatBarcode
            out[k] = _barcode_from_packed(pb)
        end
    else
        for (k, i) in enumerate(inds_vec)
            out[k] = _projected_barcode(cache, i)
        end
    end
    dirs = [projection_direction(cache.arrangement.projections[i]) for i in inds_vec]
    return ProjectedBarcodesResult(out, inds_vec, dirs)
end

"""
    projected_distances(cacheM, cacheN; dist=:bottleneck, p=1, q=1)

Return a [`ProjectedDistancesResult`](@ref) containing the per-projection
barcode distances for a shared projected arrangement family.

Use this when you want the raw per-direction distance vector before any final
aggregation in [`projected_distance`](@ref). The returned wrapper keeps the
distance vector aligned with the selected projection indices and directions, so
you do not need to recover that geometry by field archaeology.

Use [`projected_distances`](@ref), [`projection_indices`](@ref), and
[`projection_directions`](@ref) on the result for cheap-first inspection. If
you only need one scalar aggregate, prefer [`projected_distance`](@ref).
"""
function projected_distances(cacheM::ProjectedBarcodeCache,
                             cacheN::ProjectedBarcodeCache;
                             dist::Symbol = :bottleneck,
                             p::Real = 1,
                             q::Real = 1,
                             threads::Bool = (Threads.nthreads() > 1))
    n = length(cacheM.arrangement.projections)
    n == length(cacheN.arrangement.projections) || error("different projection families")
    (dist == :bottleneck || dist == :wasserstein) || error("unknown dist=$dist")

    out = Vector{Float64}(undef, n)
    scratch_by_thread = _scratch_arenas(threads)
    if threads && Threads.nthreads() > 1
        _prepare_projected_cache_readonly!(cacheM)
        _prepare_projected_cache_readonly!(cacheN)
        Threads.@threads for i in 1:n
            tid = Threads.threadid()
            scratch = scratch_by_thread[tid]
            pb1 = cacheM.packed_barcodes[i]::PackedFloatBarcode
            pb2 = cacheN.packed_barcodes[i]::PackedFloatBarcode
            _points_from_packed!(scratch.points_a, pb1)
            _points_from_packed!(scratch.points_b, pb2)
            out[i] = dist == :bottleneck ? bottleneck_distance(scratch.points_a, scratch.points_b) :
                     wasserstein_distance(scratch.points_a, scratch.points_b; p=p, q=q)
        end
    else
        scratch = scratch_by_thread[1]
        for i in 1:n
            pb1 = _projected_packed_barcode(cacheM, i)
            pb2 = _projected_packed_barcode(cacheN, i)
            _points_from_packed!(scratch.points_a, pb1)
            _points_from_packed!(scratch.points_b, pb2)
            out[i] = dist == :bottleneck ? bottleneck_distance(scratch.points_a, scratch.points_b) :
                     wasserstein_distance(scratch.points_a, scratch.points_b; p=p, q=q)
        end
    end
    inds = collect(1:n)
    dirs = [projection_direction(cacheM.arrangement.projections[i]) for i in inds]
    return ProjectedDistancesResult(out, inds, dirs, dist)
end

"""
    projected_distance(cacheM, cacheN; dist=:bottleneck, agg=:mean, dir_weights=nothing, ...)

Aggregate per-projection distances (mean or maximum by default).
Weights are normalized to sum 1 when using mean/sum aggregation.

This is the usual cheap-first projected-distance summary once the projected
caches are built. Internal barcodes remain packed until distance evaluation.
"""
function projected_distance(cacheM::ProjectedBarcodeCache,
                            cacheN::ProjectedBarcodeCache;
                            dist::Symbol = :bottleneck,
                            p::Real = 1,
                            q::Real = 1,
                            agg::Symbol = :mean,
                            dir_weights = nothing,
                            threads::Bool = (Threads.nthreads() > 1))
    dres = projected_distances(cacheM, cacheN; dist=dist, p=p, q=q, threads=threads)
    dvec = projected_distances(dres)
    n = length(dvec)

    w = dir_weights === nothing ? fill(1.0/n, n) : Float64.(collect(dir_weights))
    length(w) == n || error("dir_weights must have length $n")
    s = sum(w)
    s > 0 || error("dir_weights must sum positive")
    w ./= s

    if agg == :mean
        return sum(w .* dvec)
    elseif agg == :sum
        return sum(w .* dvec)
    elseif agg == :maximum
        return maximum(dvec)
    else
        error("unknown agg=$agg")
    end
end

"""
    projected_kernel(cacheM, cacheN; kind=:wasserstein_gaussian, sigma=1.0, agg=:mean, dir_weights=nothing)

Aggregate per-projection barcode kernels using `_barcode_kernel`.

This is the projected analogue of repeated sliced-kernel work: build the
projection family once, build caches per module, then reuse them across kernel
queries.
"""
function projected_kernel(cacheM::ProjectedBarcodeCache,
                          cacheN::ProjectedBarcodeCache;
                          kind::Symbol = :wasserstein_gaussian,
                          sigma::Real = 1.0,
                          p::Real = 1,
                          q::Real = 1,
                          agg::Symbol = :mean,
                          dir_weights = nothing,
                          threads::Bool = (Threads.nthreads() > 1))
    n = length(cacheM.arrangement.projections)
    n == length(cacheN.arrangement.projections) || error("different projection families")

    w = dir_weights === nothing ? fill(1.0/n, n) : Float64.(collect(dir_weights))
    length(w) == n || error("dir_weights must have length $n")
    s = sum(w)
    s > 0 || error("dir_weights must sum positive")
    w ./= s

    vals = Vector{Float64}(undef, n)
    fast_points_kernel = kind in (:bottleneck_gaussian, :bottleneck_laplacian,
                                  :wasserstein_gaussian, :wasserstein_laplacian)
    scratch_by_thread = _scratch_arenas(threads)
    if threads && Threads.nthreads() > 1
        _prepare_projected_cache_readonly!(cacheM)
        _prepare_projected_cache_readonly!(cacheN)
        Threads.@threads for i in 1:n
            if fast_points_kernel
                tid = Threads.threadid()
                scratch = scratch_by_thread[tid]
                pb1 = cacheM.packed_barcodes[i]::PackedFloatBarcode
                pb2 = cacheN.packed_barcodes[i]::PackedFloatBarcode
                _points_from_packed!(scratch.points_a, pb1)
                _points_from_packed!(scratch.points_b, pb2)
                vals[i] = _barcode_kernel(scratch.points_a, scratch.points_b; kind=kind, sigma=sigma, p=p, q=q)
            else
                b1 = _projected_barcode(cacheM, i)
                b2 = _projected_barcode(cacheN, i)
                vals[i] = _barcode_kernel(b1, b2; kind=kind, sigma=sigma, p=p, q=q)
            end
        end
    else
        scratch = scratch_by_thread[1]
        for i in 1:n
            if fast_points_kernel
                pb1 = _projected_packed_barcode(cacheM, i)
                pb2 = _projected_packed_barcode(cacheN, i)
                _points_from_packed!(scratch.points_a, pb1)
                _points_from_packed!(scratch.points_b, pb2)
                vals[i] = _barcode_kernel(scratch.points_a, scratch.points_b; kind=kind, sigma=sigma, p=p, q=q)
            else
                b1 = _projected_barcode(cacheM, i)
                b2 = _projected_barcode(cacheN, i)
                vals[i] = _barcode_kernel(b1, b2; kind=kind, sigma=sigma, p=p, q=q)
            end
        end
    end

    if agg == :mean
        return sum(w .* vals)
    elseif agg == :sum
        return sum(w .* vals)
    elseif agg == :maximum
        return maximum(vals)
    else
        error("unknown agg=$agg")
    end
end

end # module Fibered2D
