module PLBackend
# =============================================================================
# Axis-aligned PL backend (no external deps).
#
# Shapes:
#   BoxUpset  : { x in R^n : x[i] >= ell[i] for all i }
#   BoxDownset: { x in R^n : x[i] <= u[i]   for all i }
#
# Encoding algorithm:
#   1) Collect all coordinate thresholds from ell's and u's.
#   2) Form the product-of-chains cell complex (rectangular grid cells).
#   3) Each cell gets a representative point x (strictly inside).
#   4) Y-signature of a cell is computed from contains(U_i, x) and the
#      complement for D_j. Cells with equal signatures are the uptight
#      regions (Defs. 4.12-4.17).
#   5) Build P on signatures by inclusion. Build Uhat,Dhat images on P.
#   6) Enforce monomial condition and return FringeModule + classifier pi.
#
# Complexity knobs:
#   - max_regions caps the number of grid cells (early stop if too large)
# =============================================================================

using ..FiniteFringe
using ..CoreModules: QQ, AbstractPLikeEncodingMap, EncodingOptions

import ..CoreModules: locate, dimension, representatives, axes_from_encoding
import ..RegionGeometry: region_weights, region_bbox, region_diameter, region_adjacency,
                         region_boundary_measure, region_boundary_measure_breakdown,
                         region_centroid, region_principal_directions,
                         region_chebyshev_ball, region_circumradius, region_mean_width

export BoxUpset, BoxDownset, PLEncodingMapBoxes, encode_fringe_boxes

# ------------------------------- Shapes ---------------------------------------

"Axis-aligned upset: x[i] >= ell[i] for all i."
struct BoxUpset
    ell::Vector{Float64}
end

"Axis-aligned downset: x[i] <= u[i] for all i."
struct BoxDownset
    u::Vector{Float64}
end

# -----------------------------------------------------------------------------
# Convenience constructors (mathematician-friendly):
#
# In the axis-aligned backend, an upset is determined by its lower threshold
# vector ell, and a downset is determined by its upper threshold vector u.
#
# However, lots of code naturally carries boxes as (lo, hi) corner pairs.
# To reduce friction (and to support tests that were written that way),
# we accept a 2-argument form:
#
#   BoxUpset(lo, hi)   is interpreted as BoxUpset(lo)    (hi is ignored)
#   BoxDownset(lo, hi) is interpreted as BoxDownset(hi)  (lo is ignored)
#
# The lengths are checked to avoid silent dimension mismatches.
# -----------------------------------------------------------------------------

"Construct a BoxUpset from any real vector (converted to Float64)."
BoxUpset(ell::AbstractVector{<:Real}) = BoxUpset([Float64(e) for e in ell])

"""
    BoxUpset(lo, hi)

Convenience overload: interpreted as `BoxUpset(lo)` (the second vector is ignored),
with a length check to prevent accidental dimension mismatches.
"""
function BoxUpset(lo::AbstractVector{<:Real}, hi::AbstractVector{<:Real})
    length(lo) == length(hi) || error("BoxUpset(lo, hi): expected length(lo) == length(hi)")
    return BoxUpset(lo)
end

"Construct a BoxDownset from any real vector (converted to Float64)."
BoxDownset(u::AbstractVector{<:Real}) = BoxDownset([Float64(v) for v in u])

"""
    BoxDownset(lo, hi)

Convenience overload: interpreted as `BoxDownset(hi)` (the first vector is ignored),
with a length check to prevent accidental dimension mismatches.
"""
function BoxDownset(lo::AbstractVector{<:Real}, hi::AbstractVector{<:Real})
    length(lo) == length(hi) || error("BoxDownset(lo, hi): expected length(lo) == length(hi)")
    return BoxDownset(hi)
end


# Predicate: is x in upset/downset?
#
# We keep a dedicated "contains" predicate (instead of Base.in) because it is
# used in the tight inner loops of encoding and point location.
#
# IMPORTANT: The signature convention in this backend is:
#   y[i] = contains(Ups[i], x)        (>= ell, closed)
#   z[j] = !contains(Downs[j], x)     (strictly outside the downset box)
#
# This matches the existing _signature() implementation below.

@inline function _contains_upset(U::BoxUpset, x, n::Int)
    @inbounds for i in 1:n
        if x[i] < U.ell[i]
            return false
        end
    end
    return true
end

@inline function _contains_downset(D::BoxDownset, x, n::Int)
    @inbounds for i in 1:n
        if x[i] > D.u[i]
            return false
        end
    end
    return true
end

# Public wrappers. These accept both vectors and tuples (no allocations).
@inline contains(U::BoxUpset, x::AbstractVector{<:Real}) = _contains_upset(U, x, length(x))
@inline contains(D::BoxDownset, x::AbstractVector{<:Real}) = _contains_downset(D, x, length(x))
@inline contains(U::BoxUpset, x::NTuple{N,T}) where {N,T<:Real} = _contains_upset(U, x, N)
@inline contains(D::BoxDownset, x::NTuple{N,T}) where {N,T<:Real} = _contains_downset(D, x, N)

##############################
# Packed signature keys
##############################

"""
    SigKey{MY,MZ}

Internal, allocation-free key for region lookup.

`SigKey` stores the (y,z) signature of a point, packed into 64-bit words:

- `MY = cld(m, 64)` where `m = length(Ups)` (number of upset generators).
- `MZ = cld(r, 64)` where `r = length(Downs)` (number of downset generators).

This is used as a `Dict` key in `pi.sig_to_region` to make signature lookup O(1)
without allocating `Tuple(Bool, ...)` keys.
"""
struct SigKey{MY,MZ}
    y::NTuple{MY,UInt64}
    z::NTuple{MZ,UInt64}
end

@inline function _pack_signature_words(Ups::Vector{BoxUpset}, x, n::Int, ::Val{MY}) where {MY}
    m = length(Ups)
    return ntuple(w -> begin
        word = UInt64(0)
        base = (w - 1) * 64
        @inbounds for j in 1:64
            i = base + j
            i > m && break
            if _contains_upset(Ups[i], x, n)
                word |= (UInt64(1) << (j - 1))
            end
        end
        word
    end, MY)
end

@inline function _pack_signature_words(Downs::Vector{BoxDownset}, x, n::Int, ::Val{MZ}) where {MZ}
    r = length(Downs)
    return ntuple(w -> begin
        word = UInt64(0)
        base = (w - 1) * 64
        @inbounds for j in 1:64
            i = base + j
            i > r && break
            # NOTE: z[j] is the complement of downset membership.
            if !_contains_downset(Downs[i], x, n)
                word |= (UInt64(1) << (j - 1))
            end
        end
        word
    end, MZ)
end

@inline function _pack_bitvector_words(sig::BitVector, ::Val{MW}) where {MW}
    len = length(sig)
    return ntuple(w -> begin
        word = UInt64(0)
        base = (w - 1) * 64
        @inbounds for j in 1:64
            i = base + j
            i > len && break
            if sig[i]
                word |= (UInt64(1) << (j - 1))
            end
        end
        word
    end, MW)
end

function _bitvector_from_words(words::NTuple{MW,UInt64}, len::Int) where {MW}
    bv = BitVector(undef, len)
    @inbounds for i in 1:len
        w = (i - 1) >>> 6          # div 64
        b = (i - 1) & 0x3f         # mod 64
        bv[i] = ((words[w + 1] >>> b) & UInt64(1)) == UInt64(1)
    end
    return bv
end

@inline function _sigkey_from_bitvectors(y::BitVector, z::BitVector, ::Val{MY}, ::Val{MZ}) where {MY,MZ}
    SigKey{MY,MZ}(_pack_bitvector_words(y, Val(MY)),
                  _pack_bitvector_words(z, Val(MZ)))
end


# Collect and sort split coordinates per axis.
# This is the single source of truth for the axis-aligned cell grid.
function _coords_from_generators(Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset})
    n = 0
    if !isempty(Ups)
        n = length(Ups[1].ell)
    elseif !isempty(Downs)
        n = length(Downs[1].u)
    else
        return Vector{Vector{Float64}}()
    end

    coords = [Float64[] for _ in 1:n]

    for U in Ups
        length(U.ell) == n || error("_coords_from_generators: inconsistent upset dimension")
        @inbounds for i in 1:n
            v = U.ell[i]
            isfinite(v) || error("_coords_from_generators: upset lower bounds must be finite")
            push!(coords[i], v)
        end
    end

    for D in Downs
        length(D.u) == n || error("_coords_from_generators: inconsistent downset dimension")
        @inbounds for i in 1:n
            v = D.u[i]
            isfinite(v) || error("_coords_from_generators: downset upper bounds must be finite")
            push!(coords[i], v)
        end
    end

    @inbounds for i in 1:n
        sort!(coords[i])
        unique!(coords[i])
    end

    return coords
end

##############################
# Grid helpers for O(1) locate
##############################

@inline function _cell_shape(coords::Vector{Vector{Float64}})
    n = length(coords)
    shape = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        shape[i] = length(coords[i]) + 1
    end
    return shape
end

@inline function _cell_strides(shape::Vector{Int})
    n = length(shape)
    strides = Vector{Int}(undef, n)
    strides[1] = 1
    @inbounds for i in 2:n
        strides[i] = strides[i - 1] * shape[i - 1]
    end
    return strides
end

function _axis_meta(coords::Vector{Vector{Float64}})
    n = length(coords)
    axis_is_uniform = BitVector(undef, n)
    axis_step = Vector{Float64}(undef, n)
    axis_min = Vector{Float64}(undef, n)

    @inbounds for i in 1:n
        ci = coords[i]
        k = length(ci)
        if k >= 2
            step = ci[2] - ci[1]
            # Conservative "uniform" test (tolerant to tiny rounding).
            tol = 16 * eps(Float64) * max(abs(step), 1.0)
            uniform = step > 0
            for j in 3:k
                if abs((ci[j] - ci[j - 1]) - step) > tol
                    uniform = false
                    break
                end
            end
            axis_is_uniform[i] = uniform
            axis_step[i] = uniform ? step : 0.0
            axis_min[i] = uniform ? ci[1] : 0.0
        else
            axis_is_uniform[i] = false
            axis_step[i] = 0.0
            axis_min[i] = 0.0
        end
    end
    return axis_is_uniform, axis_step, axis_min
end

# For each axis i and each split coordinate coords[i][j], store bit flags:
#   0x01 => this coordinate appears as an upset lower bound ell[i]
#   0x02 => this coordinate appears as a downset upper bound u[i]
function _coord_flags(coords::Vector{Vector{Float64}}, Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset})
    n = length(coords)
    flags = [zeros(UInt8, length(coords[i])) for i in 1:n]
    idx = [Dict{Float64,Int}() for _ in 1:n]

    @inbounds for i in 1:n
        for (j, c) in pairs(coords[i])
            idx[i][c] = j
        end
    end

    @inbounds for U in Ups
        for i in 1:n
            j = idx[i][U.ell[i]]
            flags[i][j] |= 0x01
        end
    end
    @inbounds for D in Downs
        for i in 1:n
            j = idx[i][D.u[i]]
            flags[i][j] |= 0x02
        end
    end
    return flags
end

"""
    PLEncodingMapBoxes

Backend structure describing the region decomposition induced by axis-aligned
box generators `Ups` and `Downs`.

The full-dimensional cells are determined by `coords[i]`, the sorted unique split
coordinates along axis `i`. Each cell has a constant membership signature
(y,z) with respect to `Ups` and `Downs` (using the signature convention in
`_signature` below).

For fast point location, we precompute:

- `sig_to_region`: `Dict(SigKey => region_id)` for O(1) signature lookup.
- `cell_to_region`: a dense lookup table mapping each grid cell to its region id.
  `locate` uses this for O(1) lookup in the number of regions.

The `cell_to_region` table is exact for interior points. For points lying exactly
on split coordinates, `locate` applies a cheap correction based on whether the
split came from an upset lower bound (>=) or a downset upper bound (<=), and
falls back to signature lookup only in truly ambiguous "both" cases.
"""
struct PLEncodingMapBoxes{MY,MZ} <: AbstractPLikeEncodingMap
    n::Int
    coords::Vector{Vector{Float64}}
    sig_y::Vector{BitVector}
    sig_z::Vector{BitVector}
    reps::Vector{Vector{Float64}}
    Ups::Vector{BoxUpset}
    Downs::Vector{BoxDownset}

    # Fast point location caches (built once per encoding):
    sig_to_region::Dict{SigKey{MY,MZ},Int}
    cell_shape::Vector{Int}
    cell_strides::Vector{Int}
    cell_to_region::Vector{Int}

    # Boundary handling: for each axis and split coordinate, record whether that
    # coordinate appears as an ell (upset) or u (downset) boundary.
    coord_flags::Vector{Vector{UInt8}}

    # Micro-optimization: detect uniform per-axis grids so we can index a slab
    # in O(1) arithmetic (instead of a binary search).
    axis_is_uniform::BitVector
    axis_step::Vector{Float64}
    axis_min::Vector{Float64}
end

n(pi::PLEncodingMapBoxes) = pi.n
m(pi::PLEncodingMapBoxes) = length(pi.Ups)
r(pi::PLEncodingMapBoxes) = length(pi.Downs)
N(pi::PLEncodingMapBoxes) = length(pi.reps)

# --- Core encoding-map interface ------------------------------------------------

dimension(pi::PLEncodingMapBoxes) = pi.n
representatives(pi::PLEncodingMapBoxes) = pi.reps

function _signature(x::Vector{Float64}, Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset})
    y = BitVector(undef, length(Ups))
    z = BitVector(undef, length(Downs))
    for i in 1:length(Ups)
        y[i] = contains(Ups[i], x)
    end
    for j in 1:length(Downs)
        z[j] = !contains(Downs[j], x)
    end
    return y, z
end

# Return the axis coordinate lists for this encoding.
axes_from_encoding(pi::PLEncodingMapBoxes) = ntuple(i -> pi.coords[i], pi.n)

# --- Fast locate -------------------------------------------------------------

@inline function _slab_index(ci::Vector{Float64}, xi::Real,
                             is_uniform::Bool, x0::Float64, step::Float64)
    k = length(ci)
    k == 0 && return 0
    x = Float64(xi)

    if is_uniform
        if x < ci[1]
            return 0
        elseif x >= ci[end]
            return k
        else
            j = Int(floor((x - x0) / step)) + 1
            # clamp to [0,k]
            if j < 0
                return 0
            elseif j > k
                return k
            else
                return j
            end
        end
    else
        return searchsortedlast(ci, x)
    end
end

@inline function _cell_index_and_ambiguous(pi::PLEncodingMapBoxes, x)
    lin = 1
    ambiguous = false
    @inbounds for i in 1:pi.n
        ci = pi.coords[i]
        s = _slab_index(ci, x[i], pi.axis_is_uniform[i], pi.axis_min[i], pi.axis_step[i])

        # Boundary correction: if x[i] is exactly a split coordinate, decide whether
        # it belongs to the cell on the left (<= boundary from a downset u) or
        # to the cell on the right (>= boundary from an upset ell).
        if 1 <= s <= length(ci) && x[i] == ci[s]
            flag = pi.coord_flags[i][s]
            if flag == 0x02
                s -= 1               # u-only: treat equality as "left"
            elseif flag == 0x03
                ambiguous = true     # both ell and u: fall back to signature
            end
        end

        lin += s * pi.cell_strides[i]
    end
    return lin, ambiguous
end

@inline function _sigkey(pi::PLEncodingMapBoxes{MY,MZ}, x) where {MY,MZ}
    ywords = _pack_signature_words(pi.Ups, x, pi.n, Val(MY))
    zwords = _pack_signature_words(pi.Downs, x, pi.n, Val(MZ))
    return SigKey{MY,MZ}(ywords, zwords)
end

"""
    locate(pi::PLEncodingMapBoxes, x) -> Int

Locate a point `x` in the region decomposition defined by the axis-aligned
box generators stored in `pi`.

This backend uses a precomputed dense cell table (`pi.cell_to_region`) for fast
lookup:

1. Compute the slab index in each axis (binary search, or O(1) arithmetic on
   uniform grids).
2. Apply a boundary correction on exact split coordinates (>= for upset bounds,
   <= for downset bounds).
3. Do a single array lookup in `pi.cell_to_region`.

If a coordinate lies on an "ambiguous" split value that appears as both an upset
and a downset boundary, the cell choice is not well-defined; in that case we
compute the packed signature key and fall back to `pi.sig_to_region`.

Returns `0` only in the ambiguous-boundary fallback when the signature does not
match any full-dimensional region (a measure-zero situation).
"""
function locate(pi::PLEncodingMapBoxes{MY,MZ}, x::AbstractVector{<:Real}) where {MY,MZ}
    length(x) == pi.n || error("locate: expected x of length $(pi.n), got $(length(x))")
    isempty(pi.cell_to_region) && error("locate: missing cell_to_region table; construct via encode_fringe_boxes")

    lin, ambiguous = _cell_index_and_ambiguous(pi, x)
    if !ambiguous
        return pi.cell_to_region[lin]
    end
    return get(pi.sig_to_region, _sigkey(pi, x), 0)
end

# Allocation-free tuple dispatch for point location.
# Without this, CoreModules.locate(::AbstractPLikeEncodingMap, ::NTuple) falls back
# to collect(x), which allocates (and breaks the @allocated == 0 tests).
function locate(pi::PLEncodingMapBoxes, x::NTuple{N,T}) where {N,T<:Real}
    N == pi.n || error("locate: expected x of length $(pi.n), got $N")
    isempty(pi.cell_to_region) && error("locate: missing cell_to_region table; construct via encode_fringe_boxes")

    lin = 1
    ambiguous = false

    @inbounds for i in 1:N
        ci = pi.coords[i]
        xi = x[i]

        s = _slab_index(ci, xi, pi.axis_is_uniform[i], pi.axis_min[i], pi.axis_step[i])

        # Boundary correction: if xi is exactly a split coordinate, decide whether
        # it belongs to the cell on the left (<= boundary from a downset u) or
        # to the cell on the right (>= boundary from an upset ell).
        if 1 <= s <= length(ci) && xi == ci[s]
            flag = pi.coord_flags[i][s]
            if flag == 0x02
                s -= 1               # u-only: treat equality as "left"
            elseif flag == 0x03
                ambiguous = true     # both ell and u: fall back to signature
            end
        end

        lin += s * pi.cell_strides[i]
    end

    if !ambiguous
        return pi.cell_to_region[lin]
    end

    # Only for measure-zero ambiguous boundary points:
    return get(pi.sig_to_region, _sigkey(pi, x), 0)
end

function locate(pi::PLEncodingMapBoxes{MY,MZ}, x::NTuple{N,T}) where {MY,MZ,N,T<:Real}
    N == pi.n || error("locate: expected x of length $(pi.n), got $N")
    isempty(pi.cell_to_region) && error("locate: missing cell_to_region table; construct via encode_fringe_boxes")

    lin, ambiguous = _cell_index_and_ambiguous(pi, x)
    if !ambiguous
        return pi.cell_to_region[lin]
    end
    return get(pi.sig_to_region, _sigkey(pi, x), 0)
end

# ------------------------- Region geometry / sizes ----------------------------

"""
    region_weights(pi::PLEncodingMapBoxes; box=nothing, strict=true,
        return_info=false, alpha=0.05)

Return region weights for a PLEncodingMapBoxes backend.

- If `return_info=false` (default), returns a vector `w` where `w[r]` is the
  volume/measure of region `r` inside the query `box`.

- If `return_info=true`, returns a NamedTuple with diagnostics:
    * `weights` : region weights
    * `stderr`  : standard errors (zero for this exact backend)
    * `ci`      : confidence intervals (degenerate: (w,w))
    * `alpha`   : confidence parameter
    * `method`  : `:exact` (or `:unscaled` if box===nothing)
    * `total_volume` : volume of query box (NaN if box===nothing)
    * `nsamples` : 0 (exact)
    * `counts`   : nothing (exact)
"""
function region_weights(pi::PLEncodingMapBoxes;
    box=nothing,
    strict::Bool=true,
    return_info::Bool=false,
    alpha::Real=0.05
)
    nregions = length(pi.sig_y)

    if box === nothing
        w = ones(Float64, nregions)
        if !return_info
            return w
        end
        ci = Vector{Tuple{Float64, Float64}}(undef, length(w))
        for i in eachindex(w)
            ci[i] = (w[i], w[i])
        end
        return (weights=w,
            stderr=zeros(Float64, length(w)),
            ci=ci,
            alpha=float(alpha),
            method=:unscaled,
            total_volume=NaN,
            nsamples=0,
            counts=nothing)
    end

    a_in, b_in = box
    length(a_in) == pi.n || error("region_weights: box lower corner has wrong dimension")
    length(b_in) == pi.n || error("region_weights: box upper corner has wrong dimension")

    a = Vector{Float64}(undef, pi.n)
    b = Vector{Float64}(undef, pi.n)
    @inbounds for i in 1:pi.n
        a[i] = float(a_in[i])
        b[i] = float(b_in[i])
        a[i] <= b[i] || error("region_weights: box must satisfy a[i] <= b[i] for all i")
    end

    isempty(pi.cell_to_region) && error("region_weights: missing cell_to_region table; construct via encode_fringe_boxes")

    w = zeros(Float64, nregions)
    total_vol = 1.0
    @inbounds for i in 1:pi.n
        total_vol *= (b[i] - a[i])
    end

    slo = Vector{Int}(undef, pi.n)
    shi = Vector{Int}(undef, pi.n)
    shape_sub = Vector{Int}(undef, pi.n)
    @inbounds for i in 1:pi.n
        ci = pi.coords[i]
        lo = searchsortedlast(ci, a[i])
        hi = searchsortedlast(ci, b[i])
        if lo < 0
            lo = 0
        elseif lo > length(ci)
            lo = length(ci)
        end
        if hi < 0
            hi = 0
        elseif hi > length(ci)
            hi = length(ci)
        end
        slo[i] = lo
        shi[i] = hi
        shape_sub[i] = hi - lo + 1
    end

    @inbounds for I in CartesianIndices(Tuple(shape_sub))
        lin = 1
        vol = 1.0

        for i in 1:pi.n
            s = slo[i] + (I[i] - 1)  # slab index in 0:length(coords[i])
            lb, ub = _slab_interval_axis(s, pi.coords[i])

            lo = max(a[i], lb)
            hi = min(b[i], ub)
            len = hi - lo
            if len <= 0.0
                vol = 0.0
                break
            end
            vol *= len
            lin += s * pi.cell_strides[i]
        end

        vol == 0.0 && continue

        t = pi.cell_to_region[lin]
        if t == 0
            strict && error("region_weights: encountered a cell with unknown region")
            continue
        end
        w[t] += vol
    end

    if !return_info
        return w
    end

    ci = Vector{Tuple{Float64, Float64}}(undef, length(w))
    for i in eachindex(w)
        ci[i] = (w[i], w[i])
    end

    return (weights=w,
        stderr=zeros(Float64, length(w)),
        ci=ci,
        alpha=float(alpha),
        method=:exact,
        total_volume=total_vol,
        nsamples=0,
        counts=nothing)
end


"""
    region_bbox(pi::PLEncodingMapBoxes, r; box=nothing, strict=true)
        -> Union{Nothing,Tuple{Vector{Float64},Vector{Float64}}}

Bounding box of region `r`, optionally intersected with a user-supplied ambient
box `box=(a,b)`.

- If `box === nothing`, the returned bounds may contain `-Inf` or `Inf` in
  unbounded directions.
- Return `nothing` if the intersection is empty (or has zero volume).
"""
function region_bbox(
    pi::PLEncodingMapBoxes,
    r::Integer;
    box::Union{Nothing,Tuple{AbstractVector{<:Real},AbstractVector{<:Real}}}=nothing,
    strict::Bool=true
)
    nregions = length(pi.sig_y)
    (1 <= r <= nregions) || error("region_bbox: region index out of range")
    isempty(pi.cell_to_region) && error("region_bbox: missing cell_to_region table; construct via encode_fringe_boxes")

    # Ambient box (possibly infinite).
    a = fill(-Inf, pi.n)
    b = fill(Inf,  pi.n)
    if box !== nothing
        a_in, b_in = box
        length(a_in) == pi.n || error("region_bbox: box lower corner has wrong dimension")
        length(b_in) == pi.n || error("region_bbox: box upper corner has wrong dimension")
        @inbounds for i in 1:pi.n
            a[i] = float(a_in[i])
            b[i] = float(b_in[i])
            a[i] <= b[i] || error("region_bbox: box must satisfy a[i] <= b[i] for all i")
        end
    end

    # Restrict to cells that could intersect the query box.
    slo = Vector{Int}(undef, pi.n)
    shi = Vector{Int}(undef, pi.n)
    shape_sub = Vector{Int}(undef, pi.n)
    @inbounds for i in 1:pi.n
        ci = pi.coords[i]
        lo = searchsortedlast(ci, a[i])
        hi = searchsortedlast(ci, b[i])
        if lo < 0
            lo = 0
        elseif lo > length(ci)
            lo = length(ci)
        end
        if hi < 0
            hi = 0
        elseif hi > length(ci)
            hi = length(ci)
        end
        slo[i] = lo
        shi[i] = hi
        shape_sub[i] = hi - lo + 1
    end

    lo_out = fill(Inf,  pi.n)
    hi_out = fill(-Inf, pi.n)
    hit = false

    sidx = Vector{Int}(undef, pi.n)

    @inbounds for I in CartesianIndices(Tuple(shape_sub))
        lin = 1
        for i in 1:pi.n
            s = slo[i] + (I[i] - 1)
            sidx[i] = s
            lin += s * pi.cell_strides[i]
        end

        t = pi.cell_to_region[lin]
        if t == 0
            strict && error("region_bbox: encountered a cell with unknown region")
            continue
        end
        t == r || continue

        ok = true
        for i in 1:pi.n
            lb, ub = _slab_interval_axis(sidx[i], pi.coords[i])
            lo_i = max(a[i], lb)
            hi_i = min(b[i], ub)
            if hi_i <= lo_i
                ok = false
                break
            end
            lo_out[i] = min(lo_out[i], lo_i)
            hi_out[i] = max(hi_out[i], hi_i)
        end
        if ok
            hit = true
        end
    end

    hit || return nothing
    return (lo_out, hi_out)
end

# ------------------------------------------------------------------------------
# Extra region geometry for the axis-aligned cell backend (PLEncodingMapBoxes)
#
# Key idea: in this backend, each region is a union of disjoint axis-aligned
# grid cells. Many geometric quantities can be computed exactly by iterating
# over these cells (intersected with a finite box).
# ------------------------------------------------------------------------------

@inline function _slab_interval_axis(ci::Int, s::Vector{Float64})
    # The slabs are indexed by ci in 0:length(s).
    # Empty threshold list means a single slab (-Inf, Inf).
    if isempty(s)
        return (-Inf, Inf)
    end
    if ci == 0
        return (-Inf, s[1])
    elseif ci == length(s)
        return (s[end], Inf)
    else
        return (s[ci], s[ci + 1])
    end
end

# Representative coordinate for a 1D axis-slab cell.
# `cj` is a 0-based slab index in 0:length(s).
@inline function _cell_rep_axis(s::Vector{Float64}, cj::Int)::Float64
    isempty(s) && return 0.0
    if cj == 0
        return s[1] - 1.0
    elseif cj == length(s)
        return s[end] + 1.0
    else
        return (s[cj] + s[cj + 1]) / 2.0
    end
end

# Representative point for a cell (for signature evaluation).
# `idx0` is 0-based cell indices: each idx0[j] in 0:length(coords[j]).
function _cell_rep_axis(coords::Vector{Vector{Float64}}, idx0::NTuple{N,Int}) where {N}
    x = Vector{Float64}(undef, N)
    @inbounds for j in 1:N
        s = coords[j]
        if isempty(s)
            x[j] = 0.0
        else
            cj = idx0[j]
            if cj == 0
                x[j] = s[1] - 1.0
            elseif cj == length(s)
                x[j] = s[end] + 1.0
            else
                x[j] = (s[cj] + s[cj + 1]) / 2.0
            end
        end
    end
    return x
end

# Collect all region-cells (axis-aligned boxes) inside a given finite `box`.
# Returns two vectors of length ncells: lows[k], highs[k] are the lo/hi corners.
function _cells_in_region_in_box(pi::PLEncodingMapBoxes, r::Integer, box;
    strict::Bool=true)

    box === nothing && error("_cells_in_region_in_box: box=(a,b) is required")
    a_box, b_box = box
    n = pi.n
    length(a_box) == n || error("_cells_in_region_in_box: box dimension mismatch")
    length(b_box) == n || error("_cells_in_region_in_box: box dimension mismatch")
    if !(all(isfinite, a_box) && all(isfinite, b_box))
        error("_cells_in_region_in_box: requires a finite box")
    end
    isempty(pi.cell_to_region) && error("_cells_in_region_in_box: missing cell_to_region table")

    nregions = length(pi.sig_y)
    (1 <= r <= nregions) || error("_cells_in_region_in_box: region index out of range")

    # Restrict to slabs that could intersect the query box.
    slo = Vector{Int}(undef, n)
    shi = Vector{Int}(undef, n)
    shape_sub = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        ci = pi.coords[i]
        lo = searchsortedlast(ci, float(a_box[i]))
        hi = searchsortedlast(ci, float(b_box[i]))
        if lo < 0
            lo = 0
        elseif lo > length(ci)
            lo = length(ci)
        end
        if hi < 0
            hi = 0
        elseif hi > length(ci)
            hi = length(ci)
        end
        slo[i] = lo
        shi[i] = hi
        shape_sub[i] = hi - lo + 1
    end

    lows = Vector{Vector{Float64}}()
    highs = Vector{Vector{Float64}}()

    sidx = Vector{Int}(undef, n)

    @inbounds for I in CartesianIndices(Tuple(shape_sub))
        lin = 1
        for i in 1:n
            s = slo[i] + (I[i] - 1)
            sidx[i] = s
            lin += s * pi.cell_strides[i]
        end

        t = pi.cell_to_region[lin]
        if t == 0
            strict && error("_cells_in_region_in_box: encountered a cell with unknown region")
            continue
        end
        t == r || continue

        lo = Vector{Float64}(undef, n)
        hi = Vector{Float64}(undef, n)
        ok = true
        for j in 1:n
            a_s, b_s = _slab_interval_axis(sidx[j], pi.coords[j])
            lo_j = max(float(a_box[j]), a_s)
            hi_j = min(float(b_box[j]), b_s)
            if hi_j <= lo_j
                ok = false
                break
            end
            lo[j] = lo_j
            hi[j] = hi_j
        end
        ok || continue
        push!(lows, lo)
        push!(highs, hi)
    end

    return lows, highs
end

"""
    region_chebyshev_ball(pi::PLEncodingMapBoxes, r; box, metric=:L2, method=:auto, strict=true) -> NamedTuple

Compute a large inscribed ball (Chebyshev ball) for an axis-aligned region.

In this backend, each region is a union of disjoint axis-aligned cells. Intersecting
with a finite `box=(a,b)` yields a union of (possibly clipped) rectangles/boxes.

We return the *largest axis-aligned-cell inscribed ball*:
- We scan all cells in region `r` intersected with `box`.
- For each cell intersection, the largest inscribed ball (for :L2, :Linf, or :L1)
  has radius `0.5 * min(side_lengths)` and center at the box midpoint.
- We take the maximum over cells.

This is exact for each cell, and therefore a valid global lower bound for the
(possibly nonconvex) region.
"""
function region_chebyshev_ball(pi::PLEncodingMapBoxes, r::Integer; box=nothing,
    metric::Symbol=:L2, method::Symbol=:auto, strict::Bool=true)

    box === nothing && error("region_chebyshev_ball: box=(a,b) is required")
    a_box, b_box = box
    if !(all(isfinite, a_box) && all(isfinite, b_box))
        error("region_chebyshev_ball: requires a finite box")
    end

    lows, highs = _cells_in_region_in_box(pi, r, box; strict=strict)

    # If the intersection is empty, return a clamped representative and radius 0.
    if isempty(lows)
        c = copy(pi.reps[r])
        @inbounds for j in 1:pi.n
            c[j] = clamp(c[j], a_box[j], b_box[j])
        end
        return (center=c, radius=0.0)
    end

    best_r = -Inf
    best_c = copy(pi.reps[r])

    @inbounds for k in 1:length(lows)
        lo = lows[k]
        hi = highs[k]
        # candidate center: midpoint of the (clipped) cell
        c = (lo .+ hi) ./ 2.0
        # candidate radius: half the minimum side length
        minlen = Inf
        for j in 1:pi.n
            minlen = min(minlen, hi[j] - lo[j])
        end
        rad = 0.5 * minlen
        if rad > best_r
            best_r = rad
            best_c = c
        end
    end

    return (center=best_c, radius=max(best_r, 0.0))
end

"""
    region_circumradius(pi::PLEncodingMapBoxes, r; box, center=:bbox, metric=:L2,
                        method=:cells, strict=true) -> Float64

Exact circumradius (about a chosen center) for a region represented as a union of
axis-aligned cells.

For each cell (intersection with the finite box), the farthest point from the
center is at a corner. For L2/L1/Linf norms this reduces to per-coordinate extremes,
so we do not enumerate all 2^n corners.
"""
function region_circumradius(pi::PLEncodingMapBoxes, r::Integer; box=nothing,
    center=:bbox, metric::Symbol=:L2, method::Symbol=:cells, strict::Bool=true)

    box === nothing && error("region_circumradius: box=(a,b) is required")
    a_box, b_box = box
    if !(all(isfinite, a_box) && all(isfinite, b_box))
        error("region_circumradius: requires a finite box")
    end

    metric = Symbol(metric)
    method = Symbol(method)
    method === :cells || error("region_circumradius(PLEncodingMapBoxes): method must be :cells")

    # Choose center.
    c = nothing
    if center === :bbox
        lo, hi = region_bbox(pi, r; box=box, strict=strict)
        c = (lo .+ hi) ./ 2.0
    elseif center === :centroid
        c = region_centroid(pi, r; box=box, strict=strict)
    elseif center === :chebyshev
        c = region_chebyshev_ball(pi, r; box=box, metric=metric, strict=strict).center
    else
        c = center
    end

    lows, highs = _cells_in_region_in_box(pi, r, box; strict=strict)
    isempty(lows) && return 0.0

    rad = 0.0
    n = pi.n

    @inbounds for k in 1:length(lows)
        lo = lows[k]
        hi = highs[k]

        if metric === :L2
            s2 = 0.0
            for j in 1:n
                dj = max(abs(lo[j] - c[j]), abs(hi[j] - c[j]))
                s2 += dj * dj
            end
            rad = max(rad, sqrt(s2))
        elseif metric === :Linf
            dmax = 0.0
            for j in 1:n
                dj = max(abs(lo[j] - c[j]), abs(hi[j] - c[j]))
                dmax = max(dmax, dj)
            end
            rad = max(rad, dmax)
        elseif metric === :L1
            s = 0.0
            for j in 1:n
                dj = max(abs(lo[j] - c[j]), abs(hi[j] - c[j]))
                s += dj
            end
            rad = max(rad, s)
        else
            error("region_circumradius: unknown metric=$metric (use :L2, :L1, :Linf)")
        end
    end

    return rad
end

# Random directions in R^n.
function _random_unit_directions_axis(n::Integer, ndirs::Integer; rng=Random.default_rng())
    U = Matrix{Float64}(undef, n, ndirs)
    @inbounds for j in 1:ndirs
        s2 = 0.0
        for i in 1:n
            u = randn(rng)
            U[i, j] = u
            s2 += u * u
        end
        invs = inv(sqrt(s2))
        for i in 1:n
            U[i, j] *= invs
        end
    end
    return U
end

"""
    region_mean_width(pi::PLEncodingMapBoxes, r; box, method=:auto, ndirs=256,
                      rng=Random.default_rng(), directions=nothing,
                      strict=true, closure=true, cache=nothing) -> Float64

Mean width of a region represented as a union of axis-aligned cells.

Important: This backend can represent nonconvex unions, so the planar Cauchy formula
(perimeter/pi) is not generally valid. Therefore:
- `method=:auto` defaults to `:cells` (direction sampling with exact per-direction width)
- `method=:cauchy` is available only if the user *knows* the region is convex in 2D.

`method=:cells`:
- Sample directions u.
- Compute width w(u) exactly by scanning cells:
    sup u cdot x is achieved by taking hi/lo corner depending on sign of u.
"""
function region_mean_width(pi::PLEncodingMapBoxes, r::Integer; box=nothing,
    method::Symbol=:auto, ndirs::Integer=256,
    rng=Random.default_rng(), directions=nothing,
    strict::Bool=true, closure::Bool=true, cache=nothing)

    box === nothing && error("region_mean_width: box=(a,b) is required")
    a_box, b_box = box
    if !(all(isfinite, a_box) && all(isfinite, b_box))
        error("region_mean_width: requires a finite box")
    end

    n = pi.n
    method = Symbol(method)
    if method === :auto
        method = :cells
    end

    if method === :cauchy
        n == 2 || error("region_mean_width(method=:cauchy) requires n==2, got n=$n")
        return region_boundary_measure(pi, r; box=box, strict=strict) / pi
    elseif method !== :cells
        error("region_mean_width: unknown method=$method (use :auto, :cells, :cauchy)")
    end

    lows, highs = _cells_in_region_in_box(pi, r, box; strict=strict)
    isempty(lows) && return 0.0

    U = directions === nothing ? _random_unit_directions_axis(n, ndirs; rng=rng) : directions
    size(U, 1) == n || error("region_mean_width: directions must have size (n,ndirs)")

    wsum = 0.0
    @inbounds for j in 1:size(U, 2)
        maxv = -Inf
        minv = Inf
        for k in 1:length(lows)
            lo = lows[k]
            hi = highs[k]

            # max u cdot x over rectangle: choose hi_i if u_i>=0 else lo_i
            smax = 0.0
            smin = 0.0
            for i in 1:n
                ui = U[i, j]
                if ui >= 0.0
                    smax += ui * hi[i]
                    smin += ui * lo[i]
                else
                    smax += ui * lo[i]
                    smin += ui * hi[i]
                end
            end

            maxv = max(maxv, smax)
            minv = min(minv, smin)
        end
        wsum += (maxv - minv)
    end

    return wsum / float(size(U, 2))
end

"""
    region_principal_directions(pi::PLEncodingMapBoxes, r; box, strict=true, closure=true,
        return_info=false, nbatches=0)

Compute mean/cov/principal directions for region `r` intersected with a finite
query `box=(a,b)`.

This backend computes these quantities exactly by integrating over the union of
axis-aligned grid cells that make up the region within the box.

Returns a named tuple with fields:

  * `mean`  :: Vector{Float64}
  * `cov`   :: Matrix{Float64}
  * `evals` :: Vector{Float64} (descending)
  * `evecs` :: Matrix{Float64} (columns correspond to `evals`)

Diagnostics:

  * `n_accepted`: number of contributing cells from region `r`
  * `n_proposed`: number of candidate cells intersecting the query box

If `return_info=true`, additional (mostly zero) fields are included for API
compatibility with sampling-based backends.
"""
function region_principal_directions(pi::PLEncodingMapBoxes, r;
    box=nothing,
    strict::Bool=true,
    closure::Bool=true,
    return_info::Bool=false,
    nbatches::Integer=0)

    closure = closure  # silence unused kw warning
    nbatches = nbatches

    box === nothing && error("region_principal_directions: requires a finite box=(a,b)")
    a_in, b_in = box
    length(a_in) == pi.n || error("region_principal_directions: expected box a of length $(pi.n)")
    length(b_in) == pi.n || error("region_principal_directions: expected box b of length $(pi.n)")

    Nreg = length(pi.sig_y)
    (1 <= r <= Nreg) || error("region_principal_directions: region index r out of bounds")
    isempty(pi.cell_to_region) && error("region_principal_directions: missing cell_to_region table")

    n = pi.n
    a = Vector{Float64}(undef, n)
    b = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        ai = Float64(a_in[i])
        bi = Float64(b_in[i])
        ai <= bi || error("region_principal_directions: invalid box (a[$i] > b[$i])")
        a[i] = ai
        b[i] = bi
    end

    # Restrict to slabs that can intersect the box.
    slo = Vector{Int}(undef, n)
    shi = Vector{Int}(undef, n)
    shape_sub = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        ci = pi.coords[i]
        lo = searchsortedlast(ci, a[i])
        hi = searchsortedlast(ci, b[i])
        L = length(ci)
        lo < 0 && (lo = 0)
        hi < 0 && (hi = 0)
        lo > L && (lo = L)
        hi > L && (hi = L)
        slo[i] = lo
        shi[i] = hi
        shape_sub[i] = hi - lo + 1
    end

    # Accumulate volume and raw moments.
    V = 0.0
    M1 = zeros(Float64, n)
    M2 = zeros(Float64, n, n)

    sidx = Vector{Int}(undef, n)
    len = Vector{Float64}(undef, n)
    s1 = Vector{Float64}(undef, n)
    s2 = Vector{Float64}(undef, n)

    n_used = 0
    n_checked = 0

    @inbounds for I in CartesianIndices(Tuple(shape_sub))
        lin = 1
        ok = true
        for i in 1:n
            s = slo[i] + (I[i] - 1)
            sidx[i] = s
            lin += s * pi.cell_strides[i]
        end

        # Compute intersection with the query box and its side lengths.
        vol = 1.0
        for i in 1:n
            lb, ub = _slab_interval_axis(sidx[i], pi.coords[i])
            lo_i = max(a[i], lb)
            hi_i = min(b[i], ub)
            li = hi_i - lo_i
            if li <= 0
                ok = false
                break
            end
            len[i] = li
            vol *= li

            # 1D integrals used to build moments.
            s1[i] = 0.5 * (hi_i * hi_i - lo_i * lo_i)
            s2[i] = (hi_i * hi_i * hi_i - lo_i * lo_i * lo_i) / 3.0
        end
        ok || continue

        n_checked += 1
        t = pi.cell_to_region[lin]
        if t == 0
            strict && error("region_principal_directions: unknown region id 0 in cell table")
            continue
        end
        t == r || continue

        n_used += 1
        V += vol

        # First moments and diagonal second moments.
        for i in 1:n
            prod_others = vol / len[i]
            M1[i] += s1[i] * prod_others
            M2[i,i] += s2[i] * prod_others
        end

        # Off-diagonal second moments.
        for i in 1:n
            for j in (i+1):n
                prod_others = vol / (len[i] * len[j])
                v = s1[i] * s1[j] * prod_others
                M2[i,j] += v
                M2[j,i] += v
            end
        end
    end

    if V <= 0
        # Empty intersection. Use a clamped representative as a benign mean.
        mu = Vector{Float64}(undef, n)
        rep = pi.reps[r]
        @inbounds for i in 1:n
            xi = Float64(rep[i])
            xi < a[i] && (xi = a[i])
            xi > b[i] && (xi = b[i])
            mu[i] = xi
        end
        cov = zeros(Float64, n, n)
        evals = zeros(Float64, n)
        evecs = Matrix{Float64}(I, n, n)

        if !return_info
            return (mean=mu, cov=cov, evals=evals, evecs=evecs,
                n_accepted=n_used, n_proposed=n_checked)
        end
        return (mean=mu, cov=cov, evals=evals, evecs=evecs,
            mean_stderr=zeros(Float64, n), evals_stderr=zeros(Float64, n),
            batch_evals=Vector{Vector{Float64}}(), batch_n_accepted=Int[], nbatches=0,
            n_accepted=n_used, n_proposed=n_checked)
    end

    mu = M1 ./ V
    cov = (M2 ./ V) .- (mu * transpose(mu))

    # Eigen-decomposition of the symmetric covariance.
    E = eigen(Symmetric(cov))
    p = sortperm(E.values; rev=true)
    evals = E.values[p]
    evecs = E.vectors[:, p]

    if !return_info
        return (mean=mu, cov=cov, evals=evals, evecs=evecs,
            n_accepted=n_used, n_proposed=n_checked)
    end

    return (mean=mu, cov=cov, evals=evals, evecs=evecs,
        mean_stderr=zeros(Float64, n), evals_stderr=zeros(Float64, n),
        batch_evals=Vector{Vector{Float64}}(), batch_n_accepted=Int[], nbatches=0,
        n_accepted=n_used, n_proposed=n_checked)
end


"""
    region_adjacency(pi::PLEncodingMapBoxes, box; strict=true) -> Dict{Tuple{Int,Int},Float64}

Compute the (n-1)-dimensional interface measure between every pair of distinct
regions inside the window `box=(a,b)`.

The dictionary maps an unordered region pair `(u,v)` (with `u < v`) to the
measure of the shared interface inside `box`.

This implementation uses the precomputed dense cell table (`pi.cell_to_region`)
and does *not* recompute signatures per cell.
"""
function region_adjacency(pi::PLEncodingMapBoxes,
    box; strict::Bool=true)

    box === nothing && error("region_adjacency: box=(a,b) is required")
    a_in, b_in = box
    n = pi.n
    length(a_in) == n || error("region_adjacency: a has length $(length(a_in)) but n=$n")
    length(b_in) == n || error("region_adjacency: b has length $(length(b_in)) but n=$n")

    isempty(pi.cell_to_region) && error("region_adjacency: missing cell_to_region table; construct via encode_fringe_boxes")

    a = Vector{Float64}(undef, n)
    b = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        ai = float(a_in[i])
        bi = float(b_in[i])
        ai <= bi || error("region_adjacency: require a[i] <= b[i] for all i")
        a[i] = ai
        b[i] = bi
    end

    # Slab index ranges per axis that could intersect box.
    slo = Vector{Int}(undef, n)
    shi = Vector{Int}(undef, n)
    shape_sub = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        ci = pi.coords[i]
        Li = length(ci)
        lo = searchsortedlast(ci, a[i])
        hi = searchsortedlast(ci, b[i])
        lo < 0 && (lo = 0)
        hi < 0 && (hi = 0)
        lo > Li && (lo = Li)
        hi > Li && (hi = Li)
        slo[i] = lo
        shi[i] = hi
        shape_sub[i] = hi - lo + 1
    end

    sidx = Vector{Int}(undef, n)
    lens = Vector{Float64}(undef, n)
    strides = pi.cell_strides
    edges = Dict{Tuple{Int,Int},Float64}()

    @inbounds for I in CartesianIndices(Tuple(shape_sub))
        lin = 1
        vol = 1.0

        # Compute linear index + intersection side lengths for this cell.
        for i in 1:n
            s = slo[i] + (I[i] - 1)
            sidx[i] = s
            lin += s * strides[i]

            lb, ub = _slab_interval_axis(s, pi.coords[i])
            lo = max(a[i], lb)
            hi = min(b[i], ub)
            len = hi - lo
            if len <= 0.0
                vol = 0.0
                break
            end
            lens[i] = len
            vol *= len
        end

        vol == 0.0 && continue

        t = pi.cell_to_region[lin]
        if t == 0
            strict && error("region_adjacency: cell has unknown region id (0)")
            continue
        end

        # Check +direction neighbors only (count each interface once).
        for j in 1:n
            cj = pi.coords[j]
            Lj = length(cj)
            sj = sidx[j]
            sj < Lj || continue

            # Neighbor cell across axis j.
            lin2 = lin + strides[j]

            # Make sure neighbor cell has positive length inside the box.
            lb2, ub2 = _slab_interval_axis(sj + 1, cj)
            lo2 = max(a[j], lb2)
            hi2 = min(b[j], ub2)
            if hi2 - lo2 <= 0.0
                continue
            end

            t2 = pi.cell_to_region[lin2]
            if t2 == 0
                strict && error("region_adjacency: neighbor cell has unknown region id (0)")
                continue
            end

            t == t2 && continue

            # Face measure is product of lengths in all other axes.
            face = vol / lens[j]

            u = min(t, t2)
            v = max(t, t2)
            key = (u, v)
            edges[key] = get(edges, key, 0.0) + face
        end
    end

    return edges
end

"""
    region_boundary_measure(pi::PLEncodingMapBoxes, r; box, strict=true) -> Float64

Exact boundary measure of region `r` inside a finite window `box=(a,b)`.

The region is a union of axis-aligned grid cells. The boundary measure is the
(n-1)-dimensional measure of the boundary of `(region r) cap box`. In 2D this is a
perimeter, in 3D a surface area, etc.
"""
function region_boundary_measure(pi::PLEncodingMapBoxes, r::Integer; box=nothing, strict::Bool=true)
    box === nothing && error("region_boundary_measure: please provide box=(a,b)")
    a_in, b_in = box
    n = pi.n
    length(a_in) == n || error("region_boundary_measure: expected length(a)==$n")
    length(b_in) == n || error("region_boundary_measure: expected length(b)==$n")

    a = Vector{Float64}(undef, n)
    b = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        a[i] = float(a_in[i])
        b[i] = float(b_in[i])
        (isfinite(a[i]) && isfinite(b[i])) || error("region_boundary_measure: box bounds must be finite")
        a[i] <= b[i] || error("region_boundary_measure: expected a[i] <= b[i]")
    end

    R = length(pi.sig_y)
    (1 <= r <= R) || error("region_boundary_measure: region index out of range")
    isempty(pi.cell_to_region) && error("region_boundary_measure: missing cell_to_region table; construct via encode_fringe_boxes")

    # Restrict to the slab subgrid that can intersect the box.
    slo = Vector{Int}(undef, n)
    shi = Vector{Int}(undef, n)
    shape_sub = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        ci = pi.coords[i]
        lo = searchsortedlast(ci, a[i])
        hi = searchsortedlast(ci, b[i])
        lo = max(0, min(lo, length(ci)))
        hi = max(0, min(hi, length(ci)))
        slo[i] = lo
        shi[i] = hi
        shape_sub[i] = hi - lo + 1
    end

    # Tolerance for detecting faces that lie on the box boundary.
    scale = 0.0
    @inbounds for i in 1:n
        scale = max(scale, abs(a[i]))
        scale = max(scale, abs(b[i]))
    end
    tol = 1e-12 * max(1.0, scale)

    idx0 = Vector{Int}(undef, n)
    slab_lo = Vector{Float64}(undef, n)
    slab_hi = Vector{Float64}(undef, n)
    lens = Vector{Float64}(undef, n)

    total = 0.0
    strides = pi.cell_strides

    @inbounds for I in CartesianIndices(Tuple(shape_sub))
        lin = 1
        vol = 1.0
        ok = true

        for j in 1:n
            s = slo[j] + (I[j] - 1)
            idx0[j] = s
            lo0, hi0 = _slab_interval_axis(s, pi.coords[j])
            slab_lo[j] = lo0
            slab_hi[j] = hi0
            lo = max(a[j], lo0)
            hi = min(b[j], hi0)
            len = hi - lo
            if len <= 0
                ok = false
                break
            end
            lens[j] = len
            vol *= len
            lin += s * strides[j]
        end
        ok || continue

        t = pi.cell_to_region[lin]
        if t == 0
            strict && error("region_boundary_measure: unknown cell_to_region entry at lin=$lin")
            continue
        end
        t == r || continue

        for j in 1:n
            face = vol / lens[j]

            # Faces on the box boundary.
            lo = max(a[j], slab_lo[j])
            hi = min(b[j], slab_hi[j])
            if abs(lo - a[j]) <= tol
                total += face
            end
            if abs(hi - b[j]) <= tol
                total += face
            end

            # Internal faces across grid hyperplanes.
            if idx0[j] > 0
                bd = slab_lo[j]
                if isfinite(bd) && (bd > a[j] + tol) && (bd < b[j] - tol)
                    t2 = pi.cell_to_region[lin - strides[j]]
                    if t2 == 0
                        strict && error("region_boundary_measure: unknown neighbor cell at lin=$(lin - strides[j])")
                    elseif t2 != r
                        total += face
                    end
                end
            end

            if idx0[j] < length(pi.coords[j])
                bd = slab_hi[j]
                if isfinite(bd) && (bd > a[j] + tol) && (bd < b[j] - tol)
                    t2 = pi.cell_to_region[lin + strides[j]]
                    if t2 == 0
                        strict && error("region_boundary_measure: unknown neighbor cell at lin=$(lin + strides[j])")
                    elseif t2 != r
                        total += face
                    end
                end
            end
        end
    end

    return total
end


"""
    region_boundary_measure_breakdown(pi::PLEncodingMapBoxes, r; box, strict=true) -> Vector{NamedTuple}

Return a diagnostic decomposition of the boundary of `(region r) cap box`.

Each entry has fields:

- `measure`  : (n-1)-dimensional measure of the face patch
- `normal`   : outward unit normal (axis-aligned)
- `point`    : a representative point on the patch (midpoint)
- `neighbor` : adjacent region id, or 0 for box boundary faces
- `kind`     : `:internal` or `:box`

This is intended for debugging/visualization; it may return many small patches.
"""
function region_boundary_measure_breakdown(pi::PLEncodingMapBoxes, r::Integer; box=nothing, strict::Bool=true)
    box === nothing && error("region_boundary_measure_breakdown: please provide box=(a,b)")
    a_in, b_in = box
    n = pi.n
    length(a_in) == n || error("region_boundary_measure_breakdown: expected length(a)==$n")
    length(b_in) == n || error("region_boundary_measure_breakdown: expected length(b)==$n")

    a = Vector{Float64}(undef, n)
    b = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        a[i] = float(a_in[i])
        b[i] = float(b_in[i])
        (isfinite(a[i]) && isfinite(b[i])) || error("region_boundary_measure_breakdown: box bounds must be finite")
        a[i] <= b[i] || error("region_boundary_measure_breakdown: expected a[i] <= b[i]")
    end

    R = length(pi.sig_y)
    (1 <= r <= R) || error("region_boundary_measure_breakdown: region index out of range")
    isempty(pi.cell_to_region) && error("region_boundary_measure_breakdown: missing cell_to_region table; construct via encode_fringe_boxes")

    slo = Vector{Int}(undef, n)
    shi = Vector{Int}(undef, n)
    shape_sub = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        ci = pi.coords[i]
        lo = searchsortedlast(ci, a[i])
        hi = searchsortedlast(ci, b[i])
        lo = max(0, min(lo, length(ci)))
        hi = max(0, min(hi, length(ci)))
        slo[i] = lo
        shi[i] = hi
        shape_sub[i] = hi - lo + 1
    end

    scale = 0.0
    @inbounds for i in 1:n
        scale = max(scale, abs(a[i]))
        scale = max(scale, abs(b[i]))
    end
    tol = 1e-12 * max(1.0, scale)

    idx0 = Vector{Int}(undef, n)
    slab_lo = Vector{Float64}(undef, n)
    slab_hi = Vector{Float64}(undef, n)
    lens = Vector{Float64}(undef, n)
    mids = Vector{Float64}(undef, n)

    strides = pi.cell_strides

    pieces = Vector{NamedTuple}()

    @inbounds for I in CartesianIndices(Tuple(shape_sub))
        lin = 1
        vol = 1.0
        ok = true

        for j in 1:n
            s = slo[j] + (I[j] - 1)
            idx0[j] = s
            lo0, hi0 = _slab_interval_axis(s, pi.coords[j])
            slab_lo[j] = lo0
            slab_hi[j] = hi0
            lo = max(a[j], lo0)
            hi = min(b[j], hi0)
            len = hi - lo
            if len <= 0
                ok = false
                break
            end
            lens[j] = len
            mids[j] = 0.5 * (lo + hi)
            vol *= len
            lin += s * strides[j]
        end
        ok || continue

        t = pi.cell_to_region[lin]
        if t == 0
            strict && error("region_boundary_measure_breakdown: unknown cell_to_region entry at lin=$lin")
            continue
        end
        t == r || continue

        for j in 1:n
            face = vol / lens[j]

            # Box boundary faces.
            lo = max(a[j], slab_lo[j])
            hi = min(b[j], slab_hi[j])
            if abs(lo - a[j]) <= tol
                normal = zeros(Float64, n)
                normal[j] = -1.0
                point = copy(mids)
                point[j] = lo
                push!(pieces, (measure=face, normal=normal, point=point, neighbor=0, kind=:box))
            end
            if abs(hi - b[j]) <= tol
                normal = zeros(Float64, n)
                normal[j] = 1.0
                point = copy(mids)
                point[j] = hi
                push!(pieces, (measure=face, normal=normal, point=point, neighbor=0, kind=:box))
            end

            # Internal faces across grid hyperplanes.
            if idx0[j] > 0
                bd = slab_lo[j]
                if isfinite(bd) && (bd > a[j] + tol) && (bd < b[j] - tol)
                    t2 = pi.cell_to_region[lin - strides[j]]
                    if t2 == 0
                        strict && error("region_boundary_measure_breakdown: unknown neighbor cell at lin=$(lin - strides[j])")
                    elseif t2 != r
                        normal = zeros(Float64, n)
                        normal[j] = -1.0
                        point = copy(mids)
                        point[j] = bd
                        push!(pieces, (measure=face, normal=normal, point=point, neighbor=t2, kind=:internal))
                    end
                end
            end

            if idx0[j] < length(pi.coords[j])
                bd = slab_hi[j]
                if isfinite(bd) && (bd > a[j] + tol) && (bd < b[j] - tol)
                    t2 = pi.cell_to_region[lin + strides[j]]
                    if t2 == 0
                        strict && error("region_boundary_measure_breakdown: unknown neighbor cell at lin=$(lin + strides[j])")
                    elseif t2 != r
                        normal = zeros(Float64, n)
                        normal[j] = 1.0
                        point = copy(mids)
                        point[j] = bd
                        push!(pieces, (measure=face, normal=normal, point=point, neighbor=t2, kind=:internal))
                    end
                end
            end
        end
    end

    return pieces
end


# ------------------------------- Encoding -------------------------------------

# Construct the uptight poset on region signatures.
# The order is inclusion on the (y,z) bit-vectors.
function _uptight_from_signatures(sig_y::Vector{BitVector}, sig_z::Vector{BitVector})
    N = length(sig_y)
    N == length(sig_z) || error("_uptight_from_signatures: length mismatch")
    leq = falses(N, N)
    @inbounds for i in 1:N
        leq[i, i] = true
    end
    @inbounds for i in 1:N
        yi = sig_y[i]
        zi = sig_z[i]
        for j in 1:N
            if i == j
                continue
            end
            if FiniteFringe.is_subset(yi, sig_y[j]) && FiniteFringe.is_subset(zi, sig_z[j])
                leq[i, j] = true
            end
        end
    end
    # Inclusion of signatures is already a partial order.
    return FiniteFringe.FinitePoset(leq; check=false)
end

# Push the original generators forward to the signature poset.
function _images_on_P(P::FiniteFringe.FinitePoset,
    sig_y::Vector{BitVector},
    sig_z::Vector{BitVector})

    N = length(sig_y)
    N == length(sig_z) || error("_images_on_P: length mismatch")
    m = isempty(sig_y) ? 0 : length(sig_y[1])
    r = isempty(sig_z) ? 0 : length(sig_z[1])

    Uhat = Vector{FiniteFringe.Upset}(undef, m)
    @inbounds for i in 1:m
        mask = BitVector(undef, N)
        for t in 1:N
            mask[t] = sig_y[t][i]
        end
        # This is already an upset by construction of P.
        Uhat[i] = FiniteFringe.Upset(P, mask)
    end

    Dhat = Vector{FiniteFringe.Downset}(undef, r)
    @inbounds for j in 1:r
        mask = BitVector(undef, N)
        for t in 1:N
            # sig_z is the complement of downset membership.
            mask[t] = !sig_z[t][j]
        end
        Dhat[j] = FiniteFringe.Downset(P, mask)
    end

    return Uhat, Dhat
end

# Enforce the monomial condition on Phi for the pushed-forward generators.
function _monomialize_phi(Phi_in::AbstractMatrix{QQ}, Uhat, Dhat)
    r = length(Dhat)
    m = length(Uhat)
    size(Phi_in, 1) == r || error("Phi has wrong number of rows")
    size(Phi_in, 2) == m || error("Phi has wrong number of columns")

    Phi = Matrix{QQ}(Phi_in)
    @inbounds for j in 1:r
        for i in 1:m
            if !FiniteFringe.intersects(Uhat[i], Dhat[j])
                Phi[j, i] = zero(QQ)
            end
        end
    end
    return Phi
end

"""
    encode_fringe_boxes(Ups, Downs, Phi, opts::EncodingOptions) -> (P, H, pi)

Encode a box-generated fringe module on `R^n` into a finite poset model.

Inputs
- `Ups::Vector{BoxUpset}`: birth upsets (axis-aligned lower-orthants).
- `Downs::Vector{BoxDownset}`: death downsets (axis-aligned upper-orthants).
- `Phi::AbstractMatrix{QQ}`: an `r x m` matrix (where `m=length(Ups)`, `r=length(Downs)`).
- `opts::EncodingOptions`: required.
  - `opts.backend` must be `:auto` or `:pl_backend` (synonyms `:plbackend`, `:boxes` are accepted).
  - `opts.max_regions` caps the number of grid cells in the axis grid (default: 200_000).

Returns
- `P`: the finite encoding poset
- `H`: a `FiniteFringe.FringeModule{QQ}` on `P`
- `pi`: a `PLEncodingMapBoxes` classifier map
"""
function encode_fringe_boxes(Ups::Vector{BoxUpset},
                             Downs::Vector{BoxDownset},
                             Phi_in::AbstractMatrix{QQ},
                             opts::EncodingOptions=EncodingOptions())
    if opts.backend != :auto && opts.backend != :pl_backend &&
        opts.backend != :pl_backend_boxes && opts.backend != :boxes && opts.backend != :axis
        error("encode_fringe_boxes: EncodingOptions.backend must be :auto or :pl_backend (or :pl_backend_boxes/:boxes/:axis)")
    end
    max_regions = opts.max_regions === nothing ? 200_000 : Int(opts.max_regions)

    m = length(Ups)
    r = length(Downs)
    size(Phi_in) == (r, m) || error("encode_fringe_boxes: Phi must be size (length(Downs), length(Ups)) = ($r,$m)")

    coords = _coords_from_generators(Ups, Downs)
    n = length(coords)

    # Basic dimension sanity checks (also enforced inside _coords_from_generators).
    for U in Ups
        length(U.ell) == n || error("encode_fringe_boxes: upset has inconsistent dimension")
    end
    for D in Downs
        length(D.u) == n || error("encode_fringe_boxes: downset has inconsistent dimension")
    end

    cell_shape = _cell_shape(coords)
    cell_strides = _cell_strides(cell_shape)
    shapeT = Tuple(cell_shape)
    n_cells = prod(cell_shape)

    n_cells <= max_regions || error("Too many grid cells (>$max_regions); increase opts.max_regions or reduce splits")

    coord_flags = _coord_flags(coords, Ups, Downs)
    axis_is_uniform, axis_step, axis_min = _axis_meta(coords)

    # Deduplicate cells by packed (y,z) signature.
    MY = cld(m, 64)
    MZ = cld(r, 64)

    sig_to_region = Dict{SigKey{MY,MZ},Int}()

    sig_y = BitVector[]
    sig_z = BitVector[]
    reps = Vector{Vector{Float64}}()

    cell_to_region = Vector{Int}(undef, n_cells)
    LI = LinearIndices(shapeT)

    x = Vector{Float64}(undef, n)

    for I in CartesianIndices(shapeT)
        @inbounds for j in 1:n
            sj = I[j] - 1
            cj = coords[j]
            k = length(cj)
            if k == 0
                x[j] = 0.0
            elseif sj == 0
                x[j] = cj[1] - 1.0
            elseif sj == k
                x[j] = cj[end] + 1.0
            else
                x[j] = (cj[sj] + cj[sj + 1]) / 2.0
            end
        end

        ywords = _pack_signature_words(Ups, x, n, Val(MY))
        zwords = _pack_signature_words(Downs, x, n, Val(MZ))
        key = SigKey{MY,MZ}(ywords, zwords)

        rid = get!(sig_to_region, key) do
            new_id = length(sig_y) + 1
            push!(sig_y, _bitvector_from_words(ywords, m))
            push!(sig_z, _bitvector_from_words(zwords, r))
            push!(reps, copy(x))
            return new_id
        end

        cell_to_region[LI[I]] = rid
    end

    # Build the region poset on distinct signatures, and push the module to it.
    P = _uptight_from_signatures(sig_y, sig_z)
    Uhat, Dhat = _images_on_P(P, sig_y, sig_z)
    Phi = _monomialize_phi(Phi_in, Uhat, Dhat)
    H = FiniteFringe.FringeModule{QQ}(P, Uhat, Dhat, Phi)

    pi = PLEncodingMapBoxes{MY,MZ}(n,
                                  coords,
                                  sig_y, sig_z,
                                  reps,
                                  Ups, Downs,
                                  sig_to_region,
                                  cell_shape,
                                  cell_strides,
                                  cell_to_region,
                                  coord_flags,
                                  axis_is_uniform,
                                  axis_step,
                                  axis_min)
    return P, H, pi
end


# Convenience overload: Phi defaults to all-ones.
function encode_fringe_boxes(Ups::Vector{BoxUpset}, 
                             Downs::Vector{BoxDownset}, 
                             opts::EncodingOptions=EncodingOptions())
    m = length(Ups)
    r = length(Downs)
    Phi = reshape(ones(QQ, r * m), r, m)
    return encode_fringe_boxes(Ups, Downs, Phi, opts)
end

# Convenience overload: accept Phi as a length (r*m) vector.
function encode_fringe_boxes(Ups::Vector{BoxUpset},
                             Downs::Vector{BoxDownset},
                             Phi_vec::AbstractVector{QQ},
                             opts::EncodingOptions=EncodingOptions())
    m = length(Ups)
    r = length(Downs)
    length(Phi_vec) == r * m || error("Phi vector has wrong length")
    Phi = reshape(Phi_vec, r, m)
    return encode_fringe_boxes(Ups, Downs, Phi, opts)
end

end # module PLBackend
