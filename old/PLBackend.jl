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
#   - max_regions caps the number of regions (early stop if too large).
# =============================================================================

using ..FiniteFringe
using ..CoreModules: QQ, AbstractPLikeEncodingMap

import ..CoreModules: locate, region_weights, region_bbox, region_diameter, region_adjacency,
                      region_boundary_measure, region_boundary_measure_breakdown,
                      region_centroid, region_principal_directions,
                      region_chebyshev_ball, region_circumradius, region_mean_width

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

# Backward-compatible constructor: build caches from the region signatures.
function PLEncodingMapBoxes(n::Int,
                            coords::Vector{Vector{Float64}},
                            sig_y::Vector{BitVector},
                            sig_z::Vector{BitVector},
                            reps::Vector{Vector{Float64}},
                            Ups::Vector{BoxUpset},
                            Downs::Vector{BoxDownset})
    m = length(Ups)
    r = length(Downs)
    MY = cld(m, 64)
    MZ = cld(r, 64)

    sig_to_region = Dict{SigKey{MY,MZ},Int}()
    @inbounds for t in 1:length(sig_y)
        sig_to_region[_sigkey_from_bitvectors(sig_y[t], sig_z[t], Val(MY), Val(MZ))] = t
    end

    shape = _cell_shape(coords)
    strides = _cell_strides(shape)
    # Leave the dense cell map empty; encoders will fill it. locate() will fall back.
    cell_to_region = Int[]
    flags = _coord_flags(coords, Ups, Downs)
    axis_is_uniform, axis_step, axis_min = _axis_meta(coords)

    return PLEncodingMapBoxes{MY,MZ}(n, coords, sig_y, sig_z, reps, Ups, Downs,
                                     sig_to_region, shape, strides, cell_to_region,
                                     flags, axis_is_uniform, axis_step, axis_min)
end

n(pi::PLEncodingMapBoxes) = pi.n
m(pi::PLEncodingMapBoxes) = length(pi.Ups)
r(pi::PLEncodingMapBoxes) = length(pi.Downs)
N(pi::PLEncodingMapBoxes) = length(pi.reps)

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

Return the region id containing point `x`.

This method is designed to be fast in the regime where you call `locate` many
times on the same encoding map (typical in statistics / sampling workflows).

Implementation overview
----------------------
`PLEncodingMapBoxes` stores two precomputed lookup structures:

1. `pi.cell_to_region`: a dense table mapping each grid cell (given by a slab
   index in every axis) to a region id.

2. `pi.sig_to_region`: a `Dict` from the packed signature key to region id.

Fast path:
- Compute the slab index in each axis by binary search (`searchsortedlast`) or
  O(1) arithmetic on uniform grids.
- Apply a cheap boundary correction on exact split coordinates.
- Do a single array lookup in `pi.cell_to_region`.

Fallback:
- If the dense table is missing (hand-constructed encoding), or if the point
  lies on an "ambiguous" split value that is both an upset and a downset
  boundary, compute the packed signature key and use `pi.sig_to_region`.

The complexity is O(n log k) in the number of split coordinates per axis, and
O(1) in the number of regions.
"""
function locate(pi::PLEncodingMapBoxes{MY,MZ}, x::AbstractVector{<:Real}) where {MY,MZ}
    length(x) == pi.n || error("locate: expected x of length $(pi.n), got $(length(x))")

    if !isempty(pi.cell_to_region)
        lin, ambiguous = _cell_index_and_ambiguous(pi, x)
        if !ambiguous
            return pi.cell_to_region[lin]
        end
        # ambiguous boundary -> signature fallback
    end

    return get(pi.sig_to_region, _sigkey(pi, x), 0)
end

function locate(pi::PLEncodingMapBoxes{MY,MZ}, x::NTuple{N,T}) where {MY,MZ,N,T<:Real}
    N == pi.n || error("locate: expected x of length $(pi.n), got $N")

    if !isempty(pi.cell_to_region)
        lin, ambiguous = _cell_index_and_ambiguous(pi, x)
        if !ambiguous
            return pi.cell_to_region[lin]
        end
    end

    return get(pi.sig_to_region, _sigkey(pi, x), 0)
end

function locate(pi::PLEncodingMapBoxes{MY,MZ}, x::Dict{Int,<:Real}) where {MY,MZ}
    xv = Vector{Float64}(undef, pi.n)
    @inbounds for i in 1:pi.n
        xv[i] = Float64(get(x, i, 0.0))
    end
    return locate(pi, xv)
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
    if box === nothing
        w = ones(Float64, pi.nregions)
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

    (ell, u) = box
    n = length(ell)

    v0 = pi.grid.values[1][end] - pi.grid.values[1][1]
    for i in 2:n
        v0 *= pi.grid.values[i][end] - pi.grid.values[i][1]
    end
    w = fill(v0, pi.nregions)

    for r in 1:pi.nregions
        for i in 1:n
            a = max(pi.cells[r].a[i], ell[i])
            b = min(pi.cells[r].b[i], u[i])
            if b < a
                w[r] = 0.0
                break
            end
            w[r] *= (b - a) / (pi.cells[r].b[i] - pi.cells[r].a[i])
        end
    end

    if !strict
        w[1] = v0 - sum(w[2:end])
    end

    if !return_info
        return w
    end

    ci = Vector{Tuple{Float64, Float64}}(undef, length(w))
    for i in eachindex(w)
        ci[i] = (w[i], w[i])
    end
    total_vol = 1.0
    for i in 1:n
        total_vol *= (u[i] - ell[i])
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

    # Ambient box (possibly infinite).
    a_box = fill(-Inf, pi.n)
    b_box = fill(Inf,  pi.n)
    if box !== nothing
        a_in, b_in = box
        length(a_in) == pi.n || error("region_bbox: box lower corner has wrong dimension")
        length(b_in) == pi.n || error("region_bbox: box upper corner has wrong dimension")
        for i in 1:pi.n
            a_box[i] = float(a_in[i])
            b_box[i] = float(b_in[i])
            a_box[i] <= b_box[i] || error("region_bbox: box must satisfy a[i] <= b[i] for all i")
        end
    end

    # Fast lookup from (y,z) signature to region index.
    sig_to_region = Dict{Tuple{Tuple{Vararg{Bool}},Tuple{Vararg{Bool}}},Int}()
    for t in 1:nregions
        sig_to_region[(Tuple(pi.sig_y[t]), Tuple(pi.sig_z[t]))] = t
    end

    function cell_rep(idx::Vector{Int})
        x = Vector{Float64}(undef, pi.n)
        for j in 1:pi.n
            cj = pi.coords[j]
            if idx[j] == 0
                x[j] = cj[1] - 1.0
            elseif idx[j] == length(cj)
                x[j] = cj[end] + 1.0
            else
                x[j] = (cj[idx[j]] + cj[idx[j] + 1]) / 2.0
            end
        end
        return x
    end

    function slab_interval(ci::Vector{Float64}, s::Int)
        if s == 0
            return (-Inf, ci[1])
        elseif s == length(ci)
            return (ci[end], Inf)
        else
            return (ci[s], ci[s + 1])
        end
    end

    lo_out = fill(Inf, pi.n)
    hi_out = fill(-Inf, pi.n)
    hit = false

    lo_tmp = Vector{Float64}(undef, pi.n)
    hi_tmp = Vector{Float64}(undef, pi.n)

    cell_shape = ntuple(i -> length(pi.coords[i]) + 1, pi.n)
    for I in CartesianIndices(cell_shape)
        idx = [I[k] - 1 for k in 1:pi.n]
        x = cell_rep(idx)
        y, z = _signature(x, pi.Ups, pi.Downs)
        t = get(sig_to_region, (Tuple(y), Tuple(z)), 0)
        if t == 0
            strict && error("region_bbox: encountered a cell with unknown signature")
            continue
        end
        t == r || continue

        ok = true
        for j in 1:pi.n
            lo, hi = slab_interval(pi.coords[j], idx[j])
            lo = max(a_box[j], lo)
            hi = min(b_box[j], hi)
            if hi <= lo
                ok = false
                break
            end
            lo_tmp[j] = lo
            hi_tmp[j] = hi
        end
        if ok
            for j in 1:pi.n
                lo_out[j] = min(lo_out[j], lo_tmp[j])
                hi_out[j] = max(hi_out[j], hi_tmp[j])
            end
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

    y_target = pi.sig_y[r]
    z_target = pi.sig_z[r]

    cell_shape = ntuple(i -> length(pi.coords[i]) + 1, n)

    lows = Vector{Vector{Float64}}()
    highs = Vector{Vector{Float64}}()

    @inbounds for ci in CartesianIndices(cell_shape)
        idx0 = ntuple(j -> ci.I[j] - 1, n)

        xrep = _cell_rep_axis(pi.coords, idx0)
        sigY, sigZ = _signature(xrep, pi.Ups, pi.Downs)

        if sigY != y_target || sigZ != z_target
            continue
        end

        lo = Vector{Float64}(undef, n)
        hi = Vector{Float64}(undef, n)
        ok = true
        for j in 1:n
            a_s, b_s = _slab_interval_axis(idx0[j], pi.coords[j])
            lo_j = max(a_box[j], a_s)
            hi_j = min(b_box[j], b_s)
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
                      strict=true, closure=true, cache=nothing, nsamples=0, max_proposals=0) -> Float64

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
    nsamples::Integer=0, max_proposals::Integer=0,  # accepted for API compatibility
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

Compute mean/cov/principal directions for region `r` intersected with a finite `box`,
using exact integration over the cell geometry.

Exact backend: no Monte Carlo. Standard errors are zero. Keywords `closure` and
`nbatches` exist only for API uniformity across backends.

Return value (always):
- `mean`, `cov`, `evals`, `evecs`
- `n_accepted`, `n_proposed` (diagnostics; for this exact backend these are symbolic)
- `naccepted`, `nproposed`   (backward-compatible aliases)

If `return_info=true`, also returns:
- `mean_stderr`, `evals_stderr` (zeros for exact backend)
- `batch_evals`, `batch_n_accepted`, `nbatches` (empty / zero for exact backend)
"""

function region_principal_directions(pi::PLEncodingMapBoxes, r;
    box=nothing,
    strict::Bool=true,
    closure::Bool=true,
    return_info::Bool=false,
    nbatches::Int=0
)
    box === nothing && error("A finite box is required for region_principal_directions.")
    cell = pi.cells[r]
    (mu, cov) = _cell_union_geomoments([cell], box)
    E = eigen(Symmetric(cov))
    p = sortperm(E.values, rev=true)
    evals = E.values[p]
    evecs = E.vectors[:, p]
    (nacc, nprop) = (1, pi.nregions)

    if !return_info
        return (mean=mu, cov=cov, evals=evals, evecs=evecs,
            n_accepted=nacc, n_proposed=nprop,
            naccepted=nacc, nproposed=nprop)
    end

    return (mean=mu, cov=cov, evals=evals, evecs=evecs,
        mean_stderr=zeros(Float64, length(mu)),
        evals_stderr=zeros(Float64, length(evals)),
        batch_evals=Vector{Vector{Float64}}(),
        batch_n_accepted=Int[],
        nbatches=0,
        n_accepted=nacc, n_proposed=nprop,
        naccepted=nacc, nproposed=nprop)
end



"""
    region_adjacency(pi::PLEncodingMapBoxes; box, strict=true) -> Dict{Tuple{Int,Int},Float64}

Compute region adjacencies inside a bounding window `box=(a,b)`.

Returns a dictionary mapping unordered region pairs `(r,s)` with `r < s` to the
(n-1)-dimensional measure of their shared interface inside the window.

Implementation details:
- The PLBackend encoding induces an axis-aligned grid of slabs in each coordinate.
- We iterate over all grid cells, and for each coordinate direction look at the
  neighboring cell across the next grid hyperplane.
- When the two cells map to different regions and both cells have positive
  n-dimensional measure inside the window, we add the shared face measure.

For n=1 this counts interior boundary points (each contributes 1).
"""
function region_adjacency(
    pi::PLEncodingMapBoxes;
    box::Tuple{AbstractVector{<:Real},AbstractVector{<:Real}},
    strict::Bool=true
)
    a_in, b_in = box
    n = pi.n
    length(a_in) == n || error("region_adjacency: box lower corner has wrong dimension")
    length(b_in) == n || error("region_adjacency: box upper corner has wrong dimension")

    a = [float(a_in[i]) for i in 1:n]
    b = [float(b_in[i]) for i in 1:n]
    for i in 1:n
        a[i] <= b[i] || error("region_adjacency: need a[i] <= b[i] in box=(a,b)")
    end

    # Map signature -> region index t
    sig_to_region = Dict{Tuple{Tuple{Vararg{Bool}},Tuple{Vararg{Bool}}},Int}()
    for t in 1:length(pi.sig_y)
        sig_to_region[(Tuple(pi.sig_y[t]), Tuple(pi.sig_z[t]))] = t
    end

    slab_interval(ci::Vector{Float64}, s::Int) = s == 0 ? (-Inf, ci[1]) :
                                                (s == length(ci) ? (ci[end], Inf) : (ci[s], ci[s+1]))

    function cell_rep(idx::Vector{Int})
        x = Vector{Float64}(undef, n)
        for j in 1:n
            ci = pi.coords[j]
            s = idx[j]
            if s == 0
                x[j] = ci[1] - 1.0
            elseif s == length(ci)
                x[j] = ci[end] + 1.0
            else
                x[j] = (ci[s] + ci[s+1]) / 2.0
            end
        end
        return x
    end

    function cell_region(idx::Vector{Int})
        x = cell_rep(idx)
        y, z = _signature(x, pi.Ups, pi.Downs)
        key = (Tuple(y), Tuple(z))
        if haskey(sig_to_region, key)
            return sig_to_region[key]
        end
        strict && error("region_adjacency: found unknown signature; encoding may be inconsistent")
        return 0
    end

    edges = Dict{Tuple{Int,Int},Float64}()
    idx = zeros(Int, n)

    # Recursively iterate over all slab indices.
    function rec(dim::Int)
        if dim > n
            # For this cell, compute intersection lengths in each coordinate.
            lens = Vector{Float64}(undef, n)
            for k in 1:n
                lo, hi = slab_interval(pi.coords[k], idx[k])
                lk = max(lo, a[k])
                hk = min(hi, b[k])
                lens[k] = max(0.0, hk - lk)
            end

            # Only consider cells with positive volume in the window.
            vol = 1.0
            for k in 1:n
                vol *= lens[k]
            end
            vol == 0.0 && return

            t = cell_region(idx)
            t == 0 && return

            # Check neighbors in each axis direction once (idx[j] -> idx[j]+1).
            for j in 1:n
                s = idx[j]
                Lj = length(pi.coords[j])
                s < Lj || continue

                # Boundary hyperplane is at the next threshold coordinate.
                boundary = pi.coords[j][s+1]
                (a[j] <= boundary <= b[j]) || continue

                # Neighbor cell volume-in-window must be positive too.
                lo2, hi2 = slab_interval(pi.coords[j], s + 1)
                lj2 = max(lo2, a[j])
                hj2 = min(hi2, b[j])
                len_j2 = max(0.0, hj2 - lj2)
                (lens[j] > 0.0 && len_j2 > 0.0) || continue

                # Face measure is product of lengths in all other coordinates.
                face = 1.0
                for k in 1:n
                    k == j && continue
                    face *= lens[k]
                end
                face == 0.0 && continue

                idx[j] += 1
                t2 = cell_region(idx)
                idx[j] -= 1
                (t2 == 0 || t2 == t) && continue

                u = min(t, t2)
                v = max(t, t2)
                key = (u, v)
                edges[key] = get(edges, key, 0.0) + face
            end
            return
        end

        Ld = length(pi.coords[dim])
        for s in 0:Ld
            idx[dim] = s
            rec(dim + 1)
        end
    end

    rec(1)
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

    a = Float64[a_in[i] for i in 1:n]
    b = Float64[b_in[i] for i in 1:n]
    any(!isfinite, a) && error("region_boundary_measure: box lower bounds must be finite")
    any(!isfinite, b) && error("region_boundary_measure: box upper bounds must be finite")
    any(a .> b) && error("region_boundary_measure: expected a[i] <= b[i]")

    R = length(pi.sig_y)
    (1 <= r <= R) || error("region_boundary_measure: region index out of range")

    sig_to_region = Dict{Tuple{Vector{Bool},Vector{Bool}},Int}()
    for t in 1:R
        sig_to_region[(pi.sig_y[t], pi.sig_z[t])] = t
    end

    function cell_rep(idxs::Vector{Int})
        x = Vector{Float64}(undef, n)
        @inbounds for j in 1:n
            coords = pi.coords[j]
            s = idxs[j]
            if s == 0
                x[j] = coords[1] - 1.0
            elseif s == length(coords)
                x[j] = coords[end] + 1.0
            else
                x[j] = 0.5 * (coords[s] + coords[s + 1])
            end
        end
        return x
    end

    function slab_interval(coords::Vector{Float64}, s::Int)
        if s == 0
            return -Inf, coords[1]
        elseif s == length(coords)
            return coords[end], Inf
        else
            return coords[s], coords[s + 1]
        end
    end

    total = 0.0
    cell_shape = ntuple(i -> length(pi.coords[i]) + 1, n)
    idx = Vector{Int}(undef, n)

    scale = maximum(abs.(vcat(a, b)))
    tol = 1e-12 * max(1.0, scale)

    for I in CartesianIndices(cell_shape)
        @inbounds for j in 1:n
            idx[j] = I[j] - 1
        end

        xrep = cell_rep(idx)
        y, z = _signature(xrep, pi.Ups, pi.Downs)
        t = get(sig_to_region, (y, z), 0)
        if t == 0
            strict && error("unknown signature encountered at idx=$(idx)")
            continue
        end
        t == r || continue

        slab_lo = Vector{Float64}(undef, n)
        slab_hi = Vector{Float64}(undef, n)
        lens = Vector{Float64}(undef, n)
        ok = true
        @inbounds for j in 1:n
            lo0, hi0 = slab_interval(pi.coords[j], idx[j])
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
        end
        ok || continue

        prod_all = 1.0
        @inbounds for j in 1:n
            prod_all *= lens[j]
        end

        @inbounds for j in 1:n
            face = prod_all / lens[j]

            # Box boundary faces.
            lo = max(a[j], slab_lo[j])
            hi = min(b[j], slab_hi[j])
            if abs(lo - a[j]) <= tol
                total += face
            end
            if abs(hi - b[j]) <= tol
                total += face
            end

            # Internal faces across grid hyperplanes.
            if idx[j] > 0
                bd = slab_lo[j]
                if isfinite(bd) && (bd > a[j] + tol) && (bd < b[j] - tol)
                    idx2 = copy(idx)
                    idx2[j] -= 1
                    x2 = cell_rep(idx2)
                    y2, z2 = _signature(x2, pi.Ups, pi.Downs)
                    t2 = get(sig_to_region, (y2, z2), 0)
                    if t2 == 0
                        strict && error("unknown signature encountered at idx=$(idx2)")
                    elseif t2 != r
                        total += face
                    end
                end
            end

            if idx[j] < length(pi.coords[j])
                bd = slab_hi[j]
                if isfinite(bd) && (bd > a[j] + tol) && (bd < b[j] - tol)
                    idx2 = copy(idx)
                    idx2[j] += 1
                    x2 = cell_rep(idx2)
                    y2, z2 = _signature(x2, pi.Ups, pi.Downs)
                    t2 = get(sig_to_region, (y2, z2), 0)
                    if t2 == 0
                        strict && error("unknown signature encountered at idx=$(idx2)")
                    elseif t2 != r
                        total += face
                    end
                end
            end
        end
    end

    return total
end


# ------------------------------- Encoding -------------------------------------

"""
    encode_fringe_boxes(Ups, Downs; max_regions=200_000)

Convenience overload for the axis-aligned PL backend.

If you do not supply a monomial matrix `Phi`, we default to the all-ones matrix
of size (r, m), where r = length(Downs) and m = length(Ups). The core encoding
routine will still enforce the monomial/intersection condition and zero out
entries for pairs (U_i, D_j) whose images do not intersect.
"""
function encode_fringe_boxes(Ups::Vector{BoxUpset},
                             Downs::Vector{BoxDownset};
                             max_regions::Int=200_000)
    r = length(Downs)
    m = length(Ups)
    Phi = ones(QQ, r, m)
    return encode_fringe_boxes(Ups, Downs, Phi; max_regions=max_regions)
end

"""
encode_fringe_boxes(Ups, Downs, Phi; max_regions=200000)

Axis-aligned finite encoding with return (P, H_hat, pi).
"""
function encode_fringe_boxes(Ups::Vector{BoxUpset},
                             Downs::Vector{BoxDownset},
                             Phi_in::AbstractMatrix{QQ};
                             max_regions::Int=200_000)

    n = length(Ups) > 0 ? length(Ups[1].ell) : (length(Downs) > 0 ? length(Downs[1].u) : 0)

    # 1) thresholds per axis
    coords = [Float64[] for _ in 1:n]
    for i in 1:n
        for U in Ups; push!(coords[i], U.ell[i]); end
        for D in Downs; push!(coords[i], D.u[i]); end
        sort!(coords[i]); unique!(coords[i])
    end

    # 2) cells are products of segments between consecutive thresholds plus two unbounded sides
    #    We encode a cell by integer tuple (k1,...,kn) where k_i in 0..length(coords[i])
    #    and pick a point strictly inside that interval.
    function cell_rep(idx::NTuple{N,Int}) where {N}
        x = Vector{Float64}(undef, N)
        for j in 1:N
            if idx[j] == 0
                # below the first threshold
                x[j] = coords[j][1] - 1.0
            elseif idx[j] == length(coords[j])
                # above the last threshold
                x[j] = coords[j][end] + 1.0
            else
                a = coords[j][idx[j]]; b = coords[j][idx[j]+1]
                # strict interior point
                x[j] = (a + b) / 2.0
            end
        end
        x
    end

    # enumerate all cells, collecting DISTINCT region signatures.
    # max_regions is a cap on the number of DISTINCT signatures (regions),
    # not on the number of cells visited.
    axes = [collect(0:length(coords[i])) for i in 1:n]
    cells_iter = Base.Iterators.product((axes[i] for i in 1:n)...)

    seen = Set{Tuple{Tuple{Vararg{Bool}},Tuple{Vararg{Bool}}}}()
    sigY = BitVector[]              # unique y-signatures
    sigZ = BitVector[]              # unique z-signatures
    Reps = Vector{Float64}[]        # representative points (one per signature)

    for t in cells_iter
        x = cell_rep(t)
        y, z = _signature(x, Ups, Downs)
        key = (Tuple(y), Tuple(z))
        if !(key in seen)
            push!(seen, key)
            push!(sigY, y)
            push!(sigZ, z)
            push!(Reps, x)

            if length(sigY) > max_regions
                error("encode_fringe_boxes: requires more than max_regions=$(max_regions) regions; increase max_regions")
            end
        end
    end


    # 4) build uptight poset P by inclusion of signatures
    function uptight(sigY::Vector{BitVector}, sigZ::Vector{BitVector})
        rN = length(sigY)
        leq = falses(rN, rN)
        for i in 1:rN; leq[i,i] = true; end
        for i in 1:rN, j in 1:rN
            yi, zi = sigY[i], sigZ[i]
            yj, zj = sigY[j], sigZ[j]
            leq[i,j] = all(yi .<= yj) && all(zi .<= zj)
        end
        for k in 1:rN, i in 1:rN, j in 1:rN
            leq[i,j] = leq[i,j] || (leq[i,k] && leq[k,j])
        end
        return FiniteFringe.FinitePoset(leq)
    end
    P = uptight(sigY, sigZ)

    # 5. Record the signature for each full-dimensional cell, and build fast
    #    point-location caches.
    #
    # The grid cells are products of 1D slabs induced by coords[i].
    # Within each cell, the (y,z) signature is constant, so we can precompute:
    #   - a Dict signature -> region id
    #   - a dense cell index -> region id table
    m = length(Ups)
    r = length(Downs)
    MY = cld(m, 64)
    MZ = cld(r, 64)

    shape = _cell_shape(coords)
    strides = _cell_strides(shape)
    ncells = prod(shape)
    cell_to_region = Vector{Int}(undef, ncells)

    sig_to_region = Dict{SigKey{MY,MZ},Int}()
    sigY = BitVector[]
    sigZ = BitVector[]
    Reps = Vector{Float64}[]

    # Precompute boundary flags for correct and fast handling of x[i] == split coordinates.
    flags = _coord_flags(coords, Ups, Downs)

    # Optional micro-optimization for uniform grids.
    axis_is_uniform, axis_step, axis_min = _axis_meta(coords)

    # Reuse a single representative point buffer to avoid per-cell allocations.
    x = Vector{Float64}(undef, n)

    shape_tup = Tuple(shape)
    for I in CartesianIndices(shape_tup)
        lin = 1
        @inbounds for i in 1:n
            s = I[i] - 1
            lin += s * strides[i]

            ci = coords[i]
            if isempty(ci)
                x[i] = 0.0
            elseif s == 0
                x[i] = ci[1] - 1.0
            elseif s == length(ci)
                x[i] = ci[end] + 1.0
            else
                x[i] = (ci[s] + ci[s + 1]) / 2
            end
        end

        key = SigKey{MY,MZ}(_pack_signature_words(Ups, x, n, Val(MY)),
                            _pack_signature_words(Downs, x, n, Val(MZ)))

        t = get(sig_to_region, key, 0)
        if t == 0
            t = length(sigY) + 1
            if t > max_regions
                error("Exceeded max_regions=$max_regions while encoding fringe; consider increasing max_regions.")
            end
            sig_to_region[key] = t
            push!(sigY, _bitvector_from_words(key.y, m))
            push!(sigZ, _bitvector_from_words(key.z, r))
            push!(Reps, copy(x))
        end

        cell_to_region[lin] = t
    end

    # 6. Build the region poset and images (unchanged, but remove the old re-definition of m/r).
    N = length(sigY)
    P = [Set{Int}() for _ in 1:N]
    for t in 1:N
        for s in 1:N
            if t != s && all(sigY[t][i] <= sigY[s][i] for i in 1:m) &&
               all(sigZ[t][j] <= sigZ[s][j] for j in 1:r)
                push!(P[t], s)
            end
        end
    end

    Uhat = [Set{Int}() for _ in 1:m]
    for i in 1:m, t in 1:N
        if sigY[t][i]
            push!(Uhat[i], t)
        end
    end
    Dhat = [Set{Int}() for _ in 1:r]
    for j in 1:r, t in 1:N
        if sigZ[t][j]
            push!(Dhat[j], t)
        end
    end

    # 7. Build the backend structure *with caches already filled*.
    pi = PLEncodingMapBoxes{MY,MZ}(n, coords, sigY, sigZ, Reps, Ups, Downs,
                                   sig_to_region, shape, strides, cell_to_region,
                                   flags, axis_is_uniform, axis_step, axis_min)

    H = CoreModules.EncodingMap(pi, Uhat, Dhat)
    return P, H, pi


    # 6) monomial condition
    Phi = Matrix{QQ}(Phi_in)
    for j in 1:r, i in 1:m
        if !FiniteFringe.intersects(Uhat[i], Dhat[j])
            Phi[j,i] = zero(QQ)
        end
    end
    H = FiniteFringe.FringeModule{QQ}(P, Uhat, Dhat, Phi)

    # 7) return classifier
    pi = PLEncodingMapBoxes(n, coords, sigY, sigZ, Reps, Ups, Downs)
    return P, H, pi
end

# Overload so encode_fringe_boxes accepts a length-r*m vector Phi as a convenience
function encode_fringe_boxes(Ups::Vector{BoxUpset},
                             Downs::Vector{BoxDownset},
                             Phi_in::AbstractVector{QQ};
                             max_regions::Int=200_000)
    r = length(Downs)
    m = length(Ups)
    length(Phi_in) == r*m || error("encode_fringe_boxes: Phi has length $(length(Phi_in)) but expected r*m = $(r*m); pass a matrix of size (r,m)")
    return encode_fringe_boxes(Ups, Downs, reshape(Phi_in, r, m); max_regions=max_regions)
end

export BoxUpset, BoxDownset, PLEncodingMapBoxes, encode_fringe_boxes

end # module
