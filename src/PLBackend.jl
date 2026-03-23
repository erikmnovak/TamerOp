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
import ..FiniteFringe: AbstractPoset, nvertices, birth_upsets, death_downsets
import ..ZnEncoding: SignaturePoset, nregions, critical_coordinates,
                     critical_coordinate_counts, region_representatives,
                     region_representative, region_signature, cell_shape,
                     has_direct_lookup, region_poset, poset_kind
import ..DataTypes: ambient_dim
import ..FlangeZn: generator_counts
using ..CoreModules: QQ
using ..Options: EncodingOptions, validate_pl_mode
using ..EncodingCore: AbstractPLikeEncodingMap, CompiledEncoding
using ..CoreModules.CoeffFields: QQField
using Random
using LinearAlgebra

import ..EncodingCore: locate, locate_many!, locate_many, dimension, representatives, axes_from_encoding
import ..RegionGeometry: region_weights, region_volume, region_bbox, region_diameter, region_adjacency,
                         region_boundary_measure, region_boundary_measure_breakdown,
                         region_centroid, region_principal_directions,
                         region_chebyshev_ball, region_circumradius, region_mean_width,
                         _region_bbox_fast, _region_centroid_fast,
                         _region_volume_fast, _region_boundary_measure_fast,
                         _region_circumradius_fast, _region_minkowski_functionals_fast,
                         _region_geometry_summary_fast, _region_weights_closure,
                         _region_boundary_measure_strict, _region_bbox_strict,
                         _region_centroid_closure

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
        return ()
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

    return ntuple(i -> coords[i], n)
end

##############################
# Grid helpers for O(1) locate
##############################

@inline function _cell_shape(coords::NTuple{N,Vector{Float64}}) where {N}
    shape = Vector{Int}(undef, N)
    @inbounds for i in 1:N
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

function _axis_meta(coords::NTuple{N,Vector{Float64}}) where {N}
    axis_is_uniform = BitVector(undef, N)
    axis_step = Vector{Float64}(undef, N)
    axis_min = Vector{Float64}(undef, N)

    @inbounds for i in 1:N
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
function _coord_flags(coords::NTuple{N,Vector{Float64}}, Ups::Vector{BoxUpset}, Downs::Vector{BoxDownset}) where {N}
    flags = [zeros(UInt8, length(coords[i])) for i in 1:N]
    idx = [Dict{Float64,Int}() for _ in 1:N]

    @inbounds for i in 1:N
        for (j, c) in pairs(coords[i])
            idx[i][c] = j
        end
    end

    @inbounds for U in Ups
        for i in 1:N
            j = idx[i][U.ell[i]]
            flags[i][j] |= 0x01
        end
    end
    @inbounds for D in Downs
        for i in 1:N
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
struct PLEncodingMapBoxes{N,MY,MZ} <: AbstractPLikeEncodingMap
    n::Int
    coords::NTuple{N,Vector{Float64}}
    sig_y::Vector{BitVector}
    sig_z::Vector{BitVector}
    reps::Vector{NTuple{N,Float64}}
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
axes_from_encoding(pi::PLEncodingMapBoxes) = pi.coords

function lower_bounds end
function upper_bounds end
function axes_uniformity end

"""
    ambient_dim(U::BoxUpset) -> Int
    ambient_dim(D::BoxDownset) -> Int
    ambient_dim(pi::PLEncodingMapBoxes) -> Int

Return the ambient Euclidean dimension of an axis-aligned PL-backend object.
"""
@inline ambient_dim(U::BoxUpset) = length(U.ell)
@inline ambient_dim(D::BoxDownset) = length(D.u)
@inline ambient_dim(pi::PLEncodingMapBoxes) = pi.n
@inline ambient_dim(enc::CompiledEncoding{<:PLEncodingMapBoxes}) = ambient_dim(enc.pi)

"""
    lower_bounds(U::BoxUpset) -> Vector{Float64}
    upper_bounds(D::BoxDownset) -> Vector{Float64}

Return the defining threshold vector of an axis-aligned upset or downset.

These are the semantic alternatives to inspecting `U.ell` or `D.u` directly.
For notebook and REPL exploration, prefer these accessors over field
inspection so code reads in the mathematical language of lower and upper
bounds.
"""
@inline lower_bounds(U::BoxUpset) = U.ell
@inline upper_bounds(D::BoxDownset) = D.u

"""
    birth_upsets(pi::PLEncodingMapBoxes)
    death_downsets(pi::PLEncodingMapBoxes)

Return the birth-upset and death-downset generators used to build the
axis-aligned encoding map.

These are the canonical accessors for the generator families carried by the
box backend. Use them when you want to inspect the original box presentation
without reading raw fields.
"""
@inline birth_upsets(pi::PLEncodingMapBoxes) = pi.Ups
@inline birth_upsets(enc::CompiledEncoding{<:PLEncodingMapBoxes}) = birth_upsets(enc.pi)
@inline death_downsets(pi::PLEncodingMapBoxes) = pi.Downs
@inline death_downsets(enc::CompiledEncoding{<:PLEncodingMapBoxes}) = death_downsets(enc.pi)

"""
    nregions(pi) -> Int

Return the number of encoded full-dimensional regions carried by a box backend
encoding object.

This is the preferred cheap scalar accessor when you only need the size of the
region decomposition, not the region representatives or signatures themselves.
"""
@inline nregions(pi::PLEncodingMapBoxes) = length(pi.reps)
@inline nregions(enc::CompiledEncoding{<:PLEncodingMapBoxes}) = nregions(enc.pi)

"""
    critical_coordinates(pi)

Return the stored split coordinates along each axis of a box backend encoding.

This is the owner-local alias to the coordinate grid used by the dense
cell-to-region lookup table. Keep this as the cheap/default inspection path for
the axis grid; only inspect cell-level or region-level data when you need a
specific query or bounded geometry computation.
"""
@inline critical_coordinates(pi::PLEncodingMapBoxes) = pi.coords
@inline critical_coordinates(enc::CompiledEncoding{<:PLEncodingMapBoxes}) = critical_coordinates(enc.pi)

"""
    critical_coordinate_counts(pi) -> Tuple

Return the number of stored split coordinates along each axis.

This is the preferred cheap scalar accessor when you only need the per-axis
grid complexity, not the full coordinate lists returned by
[`critical_coordinates`](@ref).
"""
@inline critical_coordinate_counts(pi::PLEncodingMapBoxes) = _box_critical_coordinate_counts(pi)
@inline critical_coordinate_counts(enc::CompiledEncoding{<:PLEncodingMapBoxes}) =
    critical_coordinate_counts(enc.pi)

"""
    generator_counts(pi) -> NamedTuple

Return the number of birth-upset and death-downset generators as
`(; upsets=..., downsets=...)`.

This is the preferred cheap scalar accessor when you only need presentation
sizes rather than the full generator objects returned by
[`birth_upsets`](@ref) and [`death_downsets`](@ref).
"""
@inline generator_counts(pi::PLEncodingMapBoxes) = _box_generator_counts(pi)
@inline generator_counts(enc::CompiledEncoding{<:PLEncodingMapBoxes}) = generator_counts(enc.pi)

"""
    axes_uniformity(pi) -> Tuple

Return a boolean tuple indicating which axes have uniformly spaced split
coordinates.

This is the semantic scalar view of the per-axis arithmetic fast-path metadata
used by the box backend locator.
"""
@inline axes_uniformity(pi::PLEncodingMapBoxes) = _box_axis_uniformity(pi)
@inline axes_uniformity(enc::CompiledEncoding{<:PLEncodingMapBoxes}) = axes_uniformity(enc.pi)

"""
    region_representatives(pi)
    region_representative(pi, r)

Return all stored region representatives, or the representative for region `r`,
of a box backend encoding object.

Use these as the cheap/default region-level accessors before asking for heavier
bounded geometry such as exact region boxes or adjacency.
"""
@inline region_representatives(pi::PLEncodingMapBoxes) = pi.reps
@inline region_representatives(enc::CompiledEncoding{<:PLEncodingMapBoxes}) = region_representatives(enc.pi)
@inline region_representative(pi::PLEncodingMapBoxes, r::Integer) =
    (@boundscheck checkbounds(pi.reps, r); pi.reps[r])
@inline region_representative(enc::CompiledEncoding{<:PLEncodingMapBoxes}, r::Integer) =
    region_representative(enc.pi, r)

"""
    region_signature(pi, r) -> NamedTuple

Return the `(y, z)` signature of region `r` as materialized boolean vectors.

Use this for notebook or REPL inspection when you want the actual signature
bits of a specific region rather than the cheap/default summary surfaces such
as [`box_encoding_summary`](@ref) or [`box_query_summary`](@ref).
"""
@inline function region_signature(pi::PLEncodingMapBoxes, r::Integer)
    @boundscheck begin
        checkbounds(pi.sig_y, r)
        checkbounds(pi.sig_z, r)
    end
    return (; y=collect(Bool, pi.sig_y[r]), z=collect(Bool, pi.sig_z[r]))
end
@inline region_signature(enc::CompiledEncoding{<:PLEncodingMapBoxes}, r::Integer) =
    region_signature(enc.pi, r)

"""
    cell_shape(pi) -> Vector{Int}

Return the slab-cell grid shape used by the box backend direct lookup table.
"""
@inline cell_shape(pi::PLEncodingMapBoxes) = pi.cell_shape
@inline cell_shape(enc::CompiledEncoding{<:PLEncodingMapBoxes}) = cell_shape(enc.pi)

"""
    has_direct_lookup(pi) -> Bool

Return whether the box backend encoding stores the dense cell-to-region lookup
table needed for the cheap/default [`locate`](@ref) path.

For `encode_fringe_boxes(...)` outputs this should normally be `true`. It is
useful as an inspectable contract when comparing raw owner objects and compiled
wrappers.
"""
@inline has_direct_lookup(pi::PLEncodingMapBoxes) = !isempty(pi.cell_to_region)
@inline has_direct_lookup(enc::CompiledEncoding{<:PLEncodingMapBoxes}) = has_direct_lookup(enc.pi)

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

@inline function _sigkey(pi::PLEncodingMapBoxes{N,MY,MZ}, x) where {N,MY,MZ}
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
function locate(pi::PLEncodingMapBoxes{N,MY,MZ}, x::AbstractVector{<:Real}; mode::Symbol=:fast) where {N,MY,MZ}
    _ = validate_pl_mode(mode)
    length(x) == pi.n || error("locate: expected x of length $(pi.n), got $(length(x))")
    isempty(pi.cell_to_region) && error("locate: missing cell_to_region table; construct via encode_fringe_boxes")

    lin, ambiguous = _cell_index_and_ambiguous(pi, x)
    if !ambiguous
        return pi.cell_to_region[lin]
    end
    return get(pi.sig_to_region, _sigkey(pi, x), 0)
end

# Allocation-free tuple dispatch for point location.
# Without this, EncodingCore.locate(::AbstractPLikeEncodingMap, ::NTuple) falls back
# to collect(x), which allocates (and breaks the @allocated == 0 tests).
function locate(pi::PLEncodingMapBoxes{N,MY,MZ}, x::NTuple{N,T}; mode::Symbol=:fast) where {N,MY,MZ,T<:Real}
    _ = validate_pl_mode(mode)
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

function locate_many!(
    dest::AbstractVector{<:Integer},
    pi::PLEncodingMapBoxes{N,MY,MZ},
    X::AbstractMatrix{<:AbstractFloat};
    mode::Symbol = :fast,
    threaded::Bool = false,
) where {N,MY,MZ}
    _ = validate_pl_mode(mode)
    _ = threaded
    size(X, 1) == pi.n || error("locate_many!: X must have size ($(pi.n), npoints)")
    length(dest) == size(X, 2) || error("locate_many!: destination length mismatch")
    isempty(pi.cell_to_region) && error("locate_many!: missing cell_to_region table; construct via encode_fringe_boxes")

    @inbounds for j in 1:size(X, 2)
        lin = 1
        ambiguous = false
        for i in 1:N
            ci = pi.coords[i]
            xi = X[i, j]
            s = _slab_index(ci, xi, pi.axis_is_uniform[i], pi.axis_min[i], pi.axis_step[i])
            if 1 <= s <= length(ci) && xi == ci[s]
                flag = pi.coord_flags[i][s]
                if flag == 0x02
                    s -= 1
                elseif flag == 0x03
                    ambiguous = true
                end
            end
            lin += s * pi.cell_strides[i]
        end
        dest[j] = ambiguous ? get(pi.sig_to_region, _sigkey(pi, view(X, :, j)), 0) : pi.cell_to_region[lin]
    end
    return dest
end

function locate_many!(
    dest::AbstractVector{<:Integer},
    pi::PLEncodingMapBoxes{N,MY,MZ},
    X::AbstractMatrix{<:Real};
    kwargs...,
) where {N,MY,MZ}
    Xf = Matrix{Float64}(undef, size(X, 1), size(X, 2))
    @inbounds for j in axes(X, 2), i in axes(X, 1)
        Xf[i, j] = float(X[i, j])
    end
    return locate_many!(dest, pi, Xf; kwargs...)
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
    mode::Symbol=:fast,
    return_info::Bool=false,
    alpha::Real=0.05
)
    _ = validate_pl_mode(mode)
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

function region_volume(pi::PLEncodingMapBoxes, r::Integer;
    box=nothing,
    strict::Bool=true,
    mode::Symbol=:fast,
    closure::Bool=true,
    cache=nothing)
    _ = (closure, cache)
    _ = validate_pl_mode(mode)
    nregions = length(pi.sig_y)
    (1 <= r <= nregions) || error("region_volume: region index out of range")

    if box === nothing
        return 1.0
    end

    a_in, b_in = box
    length(a_in) == pi.n || error("region_volume: box lower corner has wrong dimension")
    length(b_in) == pi.n || error("region_volume: box upper corner has wrong dimension")

    a = Vector{Float64}(undef, pi.n)
    b = Vector{Float64}(undef, pi.n)
    @inbounds for i in 1:pi.n
        a[i] = float(a_in[i])
        b[i] = float(b_in[i])
        a[i] <= b[i] || error("region_volume: box must satisfy a[i] <= b[i] for all i")
    end

    isempty(pi.cell_to_region) && error("region_volume: missing cell_to_region table; construct via encode_fringe_boxes")

    volume = 0.0
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
            s = slo[i] + (I[i] - 1)
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
            strict && error("region_volume: encountered a cell with unknown region")
            continue
        end
        t == r || continue
        volume += vol
    end
    return volume
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
function _cell_rep_axis(coords::NTuple{N,Vector{Float64}}, idx0::NTuple{N,Int}) where {N}
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
        rep = pi.reps[r]
        c = ntuple(j -> clamp(rep[j], a_box[j], b_box[j]), pi.n)
        return (center=c, radius=0.0)
    end

    best_r = -Inf
    best_c = pi.reps[r]

    @inbounds for k in 1:length(lows)
        lo = lows[k]
        hi = highs[k]
        # candidate center: midpoint of the (clipped) cell
        c = ntuple(j -> (lo[j] + hi[j]) / 2.0, pi.n)
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
    nsamples::Integer=0, max_proposals::Integer=0,
    rng=Random.default_rng(), directions=nothing,
    strict::Bool=true, closure::Bool=true, cache=nothing)
    _ = (nsamples, max_proposals)

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
        return region_boundary_measure(pi, r; box=box, strict=strict) / Base.MathConstants.pi
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
    region_principal_directions(pi::PLEncodingMapBoxes, r::Integer; box, nsamples=20_000,
        rng=Random.default_rng(), strict=true, closure=true, max_proposals=10*nsamples,
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
compatibility with sampling-based backends. The keywords `nsamples`, `rng`,
and `max_proposals` are accepted for API consistency but ignored because this
backend computes exact moments over grid cells.
"""
function region_principal_directions(pi::PLEncodingMapBoxes, r::Integer;
    box=nothing,
    nsamples::Integer=20_000,
    rng=Random.default_rng(),
    strict::Bool=true,
    closure::Bool=true,
    max_proposals::Integer=10*nsamples,
    return_info::Bool=false,
    nbatches::Integer=0)

    closure = closure  # silence unused kw warning
    nbatches = nbatches
    nsamples = nsamples
    rng = rng
    max_proposals = max_proposals

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
    region_adjacency(pi::PLEncodingMapBoxes; box, strict=true) -> Dict{Tuple{Int,Int},Float64}

Compute the (n-1)-dimensional interface measure between every pair of distinct
regions inside the window `box=(a,b)`.

The dictionary maps an unordered region pair `(u,v)` (with `u < v`) to the
measure of the shared interface inside `box`.

This implementation uses the precomputed dense cell table (`pi.cell_to_region`)
and does *not* recompute signatures per cell.
"""
function region_adjacency(pi::PLEncodingMapBoxes;
    box, strict::Bool=true, mode::Symbol=:fast)
    _ = validate_pl_mode(mode)

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
function region_boundary_measure(pi::PLEncodingMapBoxes, r::Integer; box=nothing, strict::Bool=true, mode::Symbol=:fast)
    _ = validate_pl_mode(mode)
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
function region_boundary_measure_breakdown(pi::PLEncodingMapBoxes, r::Integer; box=nothing, strict::Bool=true, mode::Symbol=:fast)
    _ = validate_pl_mode(mode)
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
function _images_on_P(P::AbstractPoset,
    sig_y::Vector{BitVector},
    sig_z::Vector{BitVector})

    N = length(sig_y)
    N == length(sig_z) || error("_images_on_P: length mismatch")
    nvertices(P) == N || error("_images_on_P: poset size mismatch")
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
    encode_fringe_boxes(Ups, Downs, Phi, opts::EncodingOptions; poset_kind=:signature) -> (P, H, pi)

Encode a box-generated fringe module on `R^n` into a finite poset model.

Inputs
- `Ups::Vector{BoxUpset}`: birth upsets (axis-aligned lower-orthants).
- `Downs::Vector{BoxDownset}`: death downsets (axis-aligned upper-orthants).
- `Phi::AbstractMatrix{QQ}`: an `r x m` matrix (where `m=length(Ups)`, `r=length(Downs)`).
- `opts::EncodingOptions`: required.
  - `opts.backend` must be `:auto` or `:pl_backend` (synonyms `:plbackend`, `:boxes` are accepted).
  - `opts.max_regions` caps the number of grid cells in the axis grid (default: 200_000).
- `poset_kind`: `:signature` (structured, default) or `:dense` (materialized `FinitePoset`).

Returns
- `P`: the finite encoding poset
- `H`: a `FiniteFringe.FringeModule{QQ}` on `P`
- `pi`: a `PLEncodingMapBoxes` classifier map
"""
function encode_fringe_boxes(Ups::Vector{BoxUpset},
                             Downs::Vector{BoxDownset},
                             Phi_in::AbstractMatrix{QQ},
                             opts::EncodingOptions=EncodingOptions();
                             poset_kind::Symbol = :signature)
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
    reps = Vector{NTuple{n,Float64}}()

    # Precompute generator-threshold crossing events per axis boundary.
    # When a slab index on axis `j` increments from `s` to `s+1`, only generators
    # with threshold index `t == s+1` can change membership on that axis.
    up_events = [Vector{Vector{Int}}(undef, length(coords[j])) for j in 1:n]
    down_events = [Vector{Vector{Int}}(undef, length(coords[j])) for j in 1:n]
    @inbounds for j in 1:n
        kj = length(coords[j])
        for t in 1:kj
            up_events[j][t] = Int[]
            down_events[j][t] = Int[]
        end
    end
    @inbounds for i in 1:m
        for j in 1:n
            t = searchsortedfirst(coords[j], Ups[i].ell[j])
            push!(up_events[j][t], i)
        end
    end
    @inbounds for d in 1:r
        for j in 1:n
            t = searchsortedfirst(coords[j], Downs[d].u[j])
            push!(down_events[j][t], d)
        end
    end

    @inline _set_word_bit!(words::Vector{UInt64}, idx::Int) = begin
        wi = ((idx - 1) >>> 6) + 1
        bi = (idx - 1) & 0x3f
        words[wi] |= (UInt64(1) << bi)
        nothing
    end
    @inline _clear_word_bit!(words::Vector{UInt64}, idx::Int) = begin
        wi = ((idx - 1) >>> 6) + 1
        bi = (idx - 1) & 0x3f
        words[wi] &= ~(UInt64(1) << bi)
        nothing
    end

    sat_up = zeros(Int, m)      # number of satisfied upset axis constraints per generator
    good_down = fill(n, r)      # number of satisfied downset axis constraints (x_j <= u_j) per generator
    ywords_state = fill(UInt64(0), MY)
    zwords_state = fill(UInt64(0), MZ)

    # 0-based slab indices for current cell. Traversal uses axis-1-fast odometer order,
    # matching Julia's column-major linearization of the Cartesian grid.
    idx0 = zeros(Int, n)
    x = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        x[j] = _cell_rep_axis(coords[j], 0)
    end

    @inline function _apply_axis_inc!(axis::Int, boundary::Int)
        evy = up_events[axis][boundary + 1]
        @inbounds for t in eachindex(evy)
            i = evy[t]
            sat_up[i] += 1
            if sat_up[i] == n
                _set_word_bit!(ywords_state, i)
            end
        end
        evz = down_events[axis][boundary + 1]
        @inbounds for t in eachindex(evz)
            d = evz[t]
            good_down[d] -= 1
            if good_down[d] == n - 1
                _set_word_bit!(zwords_state, d)
            end
        end
        return nothing
    end

    @inline function _apply_axis_dec!(axis::Int, boundary::Int)
        evy = up_events[axis][boundary + 1]
        @inbounds for t in eachindex(evy)
            i = evy[t]
            if sat_up[i] == n
                _clear_word_bit!(ywords_state, i)
            end
            sat_up[i] -= 1
        end
        evz = down_events[axis][boundary + 1]
        @inbounds for t in eachindex(evz)
            d = evz[t]
            if good_down[d] == n - 1
                _clear_word_bit!(zwords_state, d)
            end
            good_down[d] += 1
        end
        return nothing
    end

    cell_to_region = Vector{Int}(undef, n_cells)
    @inline function _record_cell!(lin::Int)
        ywords = ntuple(w -> ywords_state[w], MY)
        zwords = ntuple(w -> zwords_state[w], MZ)
        key = SigKey{MY,MZ}(ywords, zwords)

        rid = get!(sig_to_region, key) do
            new_id = length(sig_y) + 1
            push!(sig_y, _bitvector_from_words(ywords, m))
            push!(sig_z, _bitvector_from_words(zwords, r))
            push!(reps, ntuple(i -> x[i], n))
            return new_id
        end
        cell_to_region[lin] = rid
        return nothing
    end

    _record_cell!(1)
    @inbounds for lin in 2:n_cells
        axis = 1
        while true
            kj = length(coords[axis])
            if idx0[axis] < kj
                boundary = idx0[axis]
                _apply_axis_inc!(axis, boundary)
                idx0[axis] = boundary + 1
                x[axis] = _cell_rep_axis(coords[axis], idx0[axis])
                break
            end

            # Carry: reset this axis to 0 and restore signature state incrementally.
            while idx0[axis] > 0
                boundary = idx0[axis] - 1
                _apply_axis_dec!(axis, boundary)
                idx0[axis] -= 1
            end
            x[axis] = _cell_rep_axis(coords[axis], 0)
            axis += 1
            axis <= n || error("encode_fringe_boxes: internal cell traversal overflow")
        end
        _record_cell!(lin)
    end

    # Build the region poset on distinct signatures, and push the module to it.
    if poset_kind == :signature
        P = SignaturePoset(sig_y, sig_z)
    elseif poset_kind == :dense
        P = _uptight_from_signatures(sig_y, sig_z)
    else
        error("encode_fringe_boxes: poset_kind must be :signature or :dense")
    end
    Uhat, Dhat = _images_on_P(P, sig_y, sig_z)
    Phi = _monomialize_phi(Phi_in, Uhat, Dhat)
    H = FiniteFringe.FringeModule{QQ}(P, Uhat, Dhat, Phi; field=QQField())

    pi = PLEncodingMapBoxes{n,MY,MZ}(n,
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
                             opts::EncodingOptions=EncodingOptions();
                             poset_kind::Symbol = :signature)
    m = length(Ups)
    r = length(Downs)
    Phi = reshape(ones(QQ, r * m), r, m)
    return encode_fringe_boxes(Ups, Downs, Phi, opts; poset_kind = poset_kind)
end

# Convenience overload: accept Phi as a length (r*m) vector.
function encode_fringe_boxes(Ups::Vector{BoxUpset},
                             Downs::Vector{BoxDownset},
                             Phi_vec::AbstractVector{QQ},
                             opts::EncodingOptions=EncodingOptions();
                             poset_kind::Symbol = :signature)
    m = length(Ups)
    r = length(Downs)
    length(Phi_vec) == r * m || error("Phi vector has wrong length")
    Phi = reshape(Phi_vec, r, m)
    return encode_fringe_boxes(Ups, Downs, Phi, opts; poset_kind = poset_kind)
end

# -----------------------------------------------------------------------------
# UX layer: summaries, validation, and semantic accessors
# -----------------------------------------------------------------------------

@inline _unwrap_box_pi(pi::PLEncodingMapBoxes) = pi
@inline _unwrap_box_pi(enc::CompiledEncoding{<:PLEncodingMapBoxes}) = enc.pi
@inline _maybe_unwrap_box_pi(pi::PLEncodingMapBoxes) = pi
@inline _maybe_unwrap_box_pi(enc::CompiledEncoding{<:PLEncodingMapBoxes}) = enc.pi
@inline _maybe_unwrap_box_pi(::Any) = nothing

@inline _box_generator_counts(pi::PLEncodingMapBoxes) = (; upsets=m(pi), downsets=r(pi))
@inline _box_critical_coordinate_counts(pi::PLEncodingMapBoxes) = ntuple(i -> length(pi.coords[i]), pi.n)
@inline _box_axis_uniformity(pi::PLEncodingMapBoxes) = ntuple(i -> Bool(pi.axis_is_uniform[i]), length(pi.axis_is_uniform))
@inline _box_issue_report(kind::Symbol, valid::Bool; kwargs...) = (; kind, valid, kwargs...)

@inline function _throw_invalid_plbackend(fn::Symbol, issues::Vector{String})
    msg = isempty(issues) ? "invalid PLBackend object." :
          "invalid PLBackend object:\n - " * join(issues, "\n - ")
    throw(ArgumentError(string(fn) * ": " * msg))
end

@inline _box_point_length(x::AbstractVector) = length(x)
@inline _box_point_length(x::Tuple) = length(x)
@inline _box_point_length(::Any) = nothing

@inline _box_point_kind(x::AbstractVector{<:Integer}) = :integer_vector
@inline _box_point_kind(x::AbstractVector{<:AbstractFloat}) = :float_vector
@inline _box_point_kind(x::AbstractVector{<:Real}) = :real_vector
@inline _box_point_kind(x::Tuple) = all(v -> v isa Integer, x) ? :integer_tuple :
                                    all(v -> v isa AbstractFloat, x) ? :float_tuple :
                                    all(v -> v isa Real, x) ? :real_tuple : :invalid
@inline _box_point_kind(::Any) = :invalid

@inline _box_matrix_kind(X::AbstractMatrix{<:Integer}) = :integer_matrix
@inline _box_matrix_kind(X::AbstractMatrix{<:AbstractFloat}) = :float_matrix
@inline _box_matrix_kind(X::AbstractMatrix{<:Real}) = :real_matrix
@inline _box_matrix_kind(::Any) = :invalid

@inline _box_endpoint_kind(x::AbstractVector{<:Integer}) = :integer_vector
@inline _box_endpoint_kind(x::AbstractVector{<:AbstractFloat}) = :float_vector
@inline _box_endpoint_kind(x::AbstractVector{<:Real}) = :real_vector
@inline _box_endpoint_kind(x::Tuple) = all(v -> v isa Integer, x) ? :integer_tuple :
                                       all(v -> v isa AbstractFloat, x) ? :float_tuple :
                                       all(v -> v isa Real, x) ? :real_tuple : :invalid
@inline _box_endpoint_kind(::Any) = :invalid

@inline _box_endpoint_length(x::AbstractVector) = length(x)
@inline _box_endpoint_length(x::Tuple) = length(x)
@inline _box_endpoint_length(::Any) = nothing

"""
    PLBackendValidationSummary

Notebook-friendly wrapper for PLBackend validation reports.
"""
struct PLBackendValidationSummary{R}
    report::R
end

"""
    plbackend_validation_summary(report) -> PLBackendValidationSummary

Wrap a raw validation report returned by `check_box_*` in a compact user-facing
container with a readable `show`.
"""
@inline plbackend_validation_summary(report::NamedTuple) = PLBackendValidationSummary(report)

"""
    check_box_upset(U; throw=false) -> NamedTuple

Validate a hand-built [`BoxUpset`](@ref).

This is the preferred validation helper when users construct axis-aligned birth
generators directly. The report checks ambient dimension and finiteness of the
stored lower bounds. Use `throw=true` to turn invalid reports into
`ArgumentError`s for early contract enforcement.
"""
function check_box_upset(U::BoxUpset; throw::Bool=false)
    issues = String[]
    n0 = ambient_dim(U)
    all(isfinite, U.ell) || push!(issues, "lower bounds must be finite.")
    valid = isempty(issues)
    throw && !valid && _throw_invalid_plbackend(:check_box_upset, issues)
    return _box_issue_report(:box_upset, valid;
                             ambient_dim=n0,
                             lower_bounds=lower_bounds(U),
                             issues=issues)
end

"""
    check_box_downset(D; throw=false) -> NamedTuple

Validate a hand-built [`BoxDownset`](@ref).

This is the preferred validation helper when users construct axis-aligned death
generators directly. The report checks ambient dimension and finiteness of the
stored upper bounds. Use `throw=true` to request strict contract enforcement.
"""
function check_box_downset(D::BoxDownset; throw::Bool=false)
    issues = String[]
    n0 = ambient_dim(D)
    all(isfinite, D.u) || push!(issues, "upper bounds must be finite.")
    valid = isempty(issues)
    throw && !valid && _throw_invalid_plbackend(:check_box_downset, issues)
    return _box_issue_report(:box_downset, valid;
                             ambient_dim=n0,
                             upper_bounds=upper_bounds(D),
                             issues=issues)
end

"""
    check_box_encoding_map(pi; throw=false) -> NamedTuple

Validate a box-backend encoding map or compiled encoding wrapper.

The report checks ambient dimension consistency, region/signature array lengths,
grid-shape metadata, and whether the dense direct-lookup table is present.

This is the main notebook-friendly validation helper for owner-level box
encodings. Prefer it before manually inspecting `coords`, `cell_shape`,
`sig_y`, `sig_z`, or the cached direct-lookup tables.
"""
function check_box_encoding_map(pi_or_enc; throw::Bool=false)
    pi = _maybe_unwrap_box_pi(pi_or_enc)
    issues = String[]
    if pi === nothing
        push!(issues, "expected PLEncodingMapBoxes or CompiledEncoding{<:PLEncodingMapBoxes}.")
        throw && _throw_invalid_plbackend(:check_box_encoding_map, issues)
        return _box_issue_report(:box_encoding_map, false;
                                 ambient_dim=nothing,
                                 nregions=nothing,
                                 generator_counts=nothing,
                                 cell_shape=nothing,
                                 direct_lookup_enabled=nothing,
                                 axes_uniform=nothing,
                                 issues=issues)
    end

    n0 = ambient_dim(pi)
    length(pi.coords) == n0 || push!(issues, "coordinate tuple has length $(length(pi.coords)), expected $n0.")
    length(pi.cell_shape) == n0 || push!(issues, "cell_shape has length $(length(pi.cell_shape)), expected $n0.")
    length(pi.cell_strides) == n0 || push!(issues, "cell_strides has length $(length(pi.cell_strides)), expected $n0.")
    length(pi.coord_flags) == n0 || push!(issues, "coord_flags has length $(length(pi.coord_flags)), expected $n0.")
    length(pi.axis_is_uniform) == n0 || push!(issues, "axis_is_uniform has length $(length(pi.axis_is_uniform)), expected $n0.")
    length(pi.axis_step) == n0 || push!(issues, "axis_step has length $(length(pi.axis_step)), expected $n0.")
    length(pi.axis_min) == n0 || push!(issues, "axis_min has length $(length(pi.axis_min)), expected $n0.")

    @inbounds for i in 1:min(length(pi.coords), n0)
        axis = pi.coords[i]
        all(isfinite, axis) || push!(issues, "axis $i coordinates must be finite.")
        issorted(axis) || push!(issues, "axis $i coordinates must be sorted.")
        allunique(axis) || push!(issues, "axis $i coordinates must be unique.")
    end

    expected_shape = [length(axis) + 1 for axis in pi.coords]
    pi.cell_shape == expected_shape || push!(issues, "cell_shape $(pi.cell_shape) does not match coordinates $(expected_shape).")
    pi.cell_strides == _cell_strides(pi.cell_shape) || push!(issues, "cell_strides do not match cell_shape.")

    expected_regions = length(pi.reps)
    length(pi.sig_y) == expected_regions || push!(issues, "sig_y has length $(length(pi.sig_y)), expected $expected_regions.")
    length(pi.sig_z) == expected_regions || push!(issues, "sig_z has length $(length(pi.sig_z)), expected $expected_regions.")
    @inbounds for t in eachindex(pi.reps)
        length(pi.reps[t]) == n0 || push!(issues, "representative $t has length $(length(pi.reps[t])), expected $n0.")
    end
    @inbounds for t in eachindex(pi.sig_y)
        length(pi.sig_y[t]) == m(pi) || push!(issues, "sig_y[$t] has length $(length(pi.sig_y[t])), expected $(m(pi)).")
    end
    @inbounds for t in eachindex(pi.sig_z)
        length(pi.sig_z[t]) == r(pi) || push!(issues, "sig_z[$t] has length $(length(pi.sig_z[t])), expected $(r(pi)).")
    end

    direct_lookup_enabled = has_direct_lookup(pi)
    direct_lookup_enabled || push!(issues, "missing cell_to_region table for direct lookup.")
    expected_cells = isempty(pi.cell_shape) ? 1 : prod(pi.cell_shape)
    length(pi.cell_to_region) == expected_cells ||
        push!(issues, "cell_to_region has length $(length(pi.cell_to_region)), expected $expected_cells.")
    @inbounds for rid in pi.cell_to_region
        (1 <= rid <= expected_regions) || push!(issues, "cell_to_region contains out-of-range region id $rid.")
    end
    for rid in values(pi.sig_to_region)
        (1 <= rid <= expected_regions) || push!(issues, "sig_to_region contains out-of-range region id $rid.")
    end
    @inbounds for i in 1:min(length(pi.coord_flags), length(pi.coords))
        length(pi.coord_flags[i]) == length(pi.coords[i]) ||
            push!(issues, "coord_flags[$i] has length $(length(pi.coord_flags[i])), expected $(length(pi.coords[i])).")
    end

    valid = isempty(issues)
    throw && !valid && _throw_invalid_plbackend(:check_box_encoding_map, issues)
    return _box_issue_report(:box_encoding_map, valid;
                             ambient_dim=n0,
                             nregions=nregions(pi),
                             generator_counts=_box_generator_counts(pi),
                             cell_shape=cell_shape(pi),
                             direct_lookup_enabled=direct_lookup_enabled,
                             axes_uniform=_box_axis_uniformity(pi),
                             issues=issues)
end

"""
    check_box_point(pi, x; throw=false) -> NamedTuple

Validate a single tuple/vector point query for [`locate`](@ref).

Accepted queries are real tuples or real vectors of ambient dimension
`ambient_dim(pi)`. This is the preferred cheap validation path before repeated
REPL experimentation with `locate`.
"""
function check_box_point(pi_or_enc, x; throw::Bool=false)
    pi = _maybe_unwrap_box_pi(pi_or_enc)
    issues = String[]
    ambient = pi === nothing ? nothing : ambient_dim(pi)
    kind = _box_point_kind(x)
    len = _box_point_length(x)

    pi === nothing && push!(issues, "expected PLEncodingMapBoxes or CompiledEncoding{<:PLEncodingMapBoxes}.")
    kind === :invalid && push!(issues, "point query must be a real tuple or real vector.")
    if ambient !== nothing && len !== nothing && len != ambient
        push!(issues, "point query has length $len, expected ambient dimension $ambient.")
    end
    if kind !== :invalid && (x isa AbstractVector{<:Real} || x isa Tuple)
        all(isfinite, x) || push!(issues, "point query coordinates must be finite.")
    end

    valid = isempty(issues)
    throw && !valid && _throw_invalid_plbackend(:check_box_point, issues)
    return _box_issue_report(:box_point, valid;
                             ambient_dim=ambient,
                             query_kind=kind,
                             point_length=len,
                             direct_lookup_enabled=(pi === nothing ? nothing : has_direct_lookup(pi)),
                             issues=issues)
end

"""
    check_box_points(pi, X; throw=false) -> NamedTuple

Validate a column-batch matrix query for [`locate_many!`](@ref).

Accepted queries are real matrices whose columns are points in `R^n`, so the
matrix must have `ambient_dim(pi)` rows. Use this before batched query
workflows when the point-shape contract might be ambiguous.
"""
function check_box_points(pi_or_enc, X; throw::Bool=false)
    pi = _maybe_unwrap_box_pi(pi_or_enc)
    issues = String[]
    ambient = pi === nothing ? nothing : ambient_dim(pi)
    kind = _box_matrix_kind(X)
    matrix_size = X isa AbstractMatrix ? size(X) : nothing

    pi === nothing && push!(issues, "expected PLEncodingMapBoxes or CompiledEncoding{<:PLEncodingMapBoxes}.")
    kind === :invalid && push!(issues, "query matrix must be a real matrix with points stored by columns.")
    if ambient !== nothing && X isa AbstractMatrix && size(X, 1) != ambient
        push!(issues, "query matrix has $(size(X, 1)) rows, expected ambient dimension $ambient.")
    end
    if X isa AbstractMatrix{<:Real}
        all(isfinite, X) || push!(issues, "query matrix entries must be finite.")
    end

    valid = isempty(issues)
    throw && !valid && _throw_invalid_plbackend(:check_box_points, issues)
    return _box_issue_report(:box_points, valid;
                             ambient_dim=ambient,
                             query_kind=kind,
                             matrix_size=matrix_size,
                             direct_lookup_enabled=(pi === nothing ? nothing : has_direct_lookup(pi)),
                             issues=issues)
end

"""
    check_box_query_box(pi, box; throw=false) -> NamedTuple

Validate a finite axis-aligned query box `box=(a,b)` for bounded region-geometry
calls.

The report explicitly records that a finite box is required for bounded region
queries in this backend. This is the preferred validation path before
`region_weights`, `region_bbox`, `region_adjacency`, and related box-sensitive
geometry calls.
"""
function check_box_query_box(pi_or_enc, box; throw::Bool=false)
    pi = _maybe_unwrap_box_pi(pi_or_enc)
    issues = String[]
    ambient = pi === nothing ? nothing : ambient_dim(pi)
    endpoint_types = nothing
    endpoint_lengths = nothing

    pi === nothing && push!(issues, "expected PLEncodingMapBoxes or CompiledEncoding{<:PLEncodingMapBoxes}.")
    if !(box isa Tuple && length(box) == 2)
        push!(issues, "box must be a pair (a, b) of real endpoints.")
    else
        a, b = box
        kind_a = _box_endpoint_kind(a)
        kind_b = _box_endpoint_kind(b)
        endpoint_types = (kind_a, kind_b)
        kind_a === :invalid && push!(issues, "box lower endpoint must be a real tuple or real vector.")
        kind_b === :invalid && push!(issues, "box upper endpoint must be a real tuple or real vector.")
        len_a = _box_endpoint_length(a)
        len_b = _box_endpoint_length(b)
        endpoint_lengths = (len_a, len_b)
        if ambient !== nothing && len_a !== nothing && len_a != ambient
            push!(issues, "box lower endpoint has length $len_a, expected ambient dimension $ambient.")
        end
        if ambient !== nothing && len_b !== nothing && len_b != ambient
            push!(issues, "box upper endpoint has length $len_b, expected ambient dimension $ambient.")
        end
        if isempty(issues)
            av = Float64[float(v) for v in a]
            bv = Float64[float(v) for v in b]
            all(isfinite, av) || push!(issues, "box lower endpoint must be finite.")
            all(isfinite, bv) || push!(issues, "box upper endpoint must be finite.")
            @inbounds for i in eachindex(av, bv)
                av[i] <= bv[i] || push!(issues, "box endpoint mismatch on axis $i: expected a[$i] <= b[$i].")
            end
        end
    end

    valid = isempty(issues)
    throw && !valid && _throw_invalid_plbackend(:check_box_query_box, issues)
    return _box_issue_report(:box_query_box, valid;
                             ambient_dim=ambient,
                             endpoint_types=endpoint_types,
                             endpoint_lengths=endpoint_lengths,
                             finite_box_required=true,
                             issues=issues)
end

@inline function _box_region_bbox_if_available(pi_or_enc, r::Int, box)
    box === nothing && return nothing
    try
        return region_bbox(pi_or_enc, r; box=box, strict=false)
    catch
        return nothing
    end
end

"""
    check_box_region(pi, r; box=nothing, throw=false) -> NamedTuple

Validate a region id `r` against a box-backend encoding object.

This helper reports whether `r` is in range, exposes the stored region
representative and signature support sizes when available, and explicitly marks
that bounded region geometry requires a finite `box=(a,b)` in this backend.

If a valid finite box is supplied, the report also includes a best-effort
`bbox` payload for quick inspection.
"""
function check_box_region(pi_or_enc, r; box=nothing, throw::Bool=false)
    pi = _maybe_unwrap_box_pi(pi_or_enc)
    issues = String[]
    ambient = pi === nothing ? nothing : ambient_dim(pi)
    nreg = pi === nothing ? nothing : nregions(pi)
    region_in_range = false
    representative = nothing
    signature_support_sizes = nothing
    bbox = nothing
    box_report = nothing

    pi === nothing && push!(issues, "expected PLEncodingMapBoxes or CompiledEncoding{<:PLEncodingMapBoxes}.")
    r isa Integer || push!(issues, "region id must be an integer.")
    if pi !== nothing && r isa Integer
        region_in_range = 1 <= Int(r) <= nreg
        region_in_range || push!(issues, "region index $(repr(r)) is out of range for nregions=$nreg.")
        if region_in_range
            rr = Int(r)
            representative = region_representative(pi, rr)
            sig = region_signature(pi, rr)
            signature_support_sizes = (; y=count(identity, sig.y), z=count(identity, sig.z))
        end
    end

    if box !== nothing
        box_report = check_box_query_box(pi_or_enc, box; throw=false)
        box_report.valid || append!(issues, ["box: " * issue for issue in box_report.issues])
    end
    if pi !== nothing && region_in_range && box !== nothing && (box_report === nothing || box_report.valid)
        bbox = _box_region_bbox_if_available(pi_or_enc, Int(r), box)
    end

    valid = isempty(issues)
    throw && !valid && _throw_invalid_plbackend(:check_box_region, issues)
    return _box_issue_report(:box_region, valid;
                             ambient_dim=ambient,
                             nregions=nreg,
                             region=(r isa Integer ? Int(r) : r),
                             region_in_range=region_in_range,
                             representative=representative,
                             signature_support_sizes=signature_support_sizes,
                             finite_box_required=true,
                             box_provided=(box !== nothing),
                             bbox=bbox,
                             issues=issues)
end

@inline function _plbackend_describe(U::BoxUpset)
    return (;
        kind=:box_upset,
        ambient_dim=ambient_dim(U),
        lower_bounds=lower_bounds(U),
    )
end

@inline function _plbackend_describe(D::BoxDownset)
    return (;
        kind=:box_downset,
        ambient_dim=ambient_dim(D),
        upper_bounds=upper_bounds(D),
    )
end

@inline function _plbackend_describe(pi::PLEncodingMapBoxes)
    return (;
        kind=:pl_backend_encoding_map,
        ambient_dim=ambient_dim(pi),
        generator_counts=generator_counts(pi),
        nregions=nregions(pi),
        cell_shape=cell_shape(pi),
        direct_lookup_enabled=has_direct_lookup(pi),
        axes_uniform=axes_uniformity(pi),
        all_axes_uniform=all(pi.axis_is_uniform),
        critical_coordinate_counts=critical_coordinate_counts(pi),
    )
end

function Base.show(io::IO, U::BoxUpset)
    d = _plbackend_describe(U)
    print(io, "BoxUpset(n=", d.ambient_dim, ", ell=", d.lower_bounds, ")")
end

function Base.show(io::IO, ::MIME"text/plain", U::BoxUpset)
    d = _plbackend_describe(U)
    print(io, "BoxUpset",
          "\n  ambient_dim = ", d.ambient_dim,
          "\n  lower_bounds = ", d.lower_bounds)
end

function Base.show(io::IO, D::BoxDownset)
    d = _plbackend_describe(D)
    print(io, "BoxDownset(n=", d.ambient_dim, ", u=", d.upper_bounds, ")")
end

function Base.show(io::IO, ::MIME"text/plain", D::BoxDownset)
    d = _plbackend_describe(D)
    print(io, "BoxDownset",
          "\n  ambient_dim = ", d.ambient_dim,
          "\n  upper_bounds = ", d.upper_bounds)
end

function Base.show(io::IO, pi::PLEncodingMapBoxes)
    d = _plbackend_describe(pi)
    print(io, "PLEncodingMapBoxes(n=", d.ambient_dim,
          ", regions=", d.nregions,
          ", generators=", d.generator_counts,
          ", cell_shape=", d.cell_shape,
          ", direct_lookup=", d.direct_lookup_enabled, ")")
end

function Base.show(io::IO, ::MIME"text/plain", pi::PLEncodingMapBoxes)
    d = _plbackend_describe(pi)
    print(io, "PLEncodingMapBoxes",
          "\n  ambient_dim = ", d.ambient_dim,
          "\n  generator_counts = ", d.generator_counts,
          "\n  nregions = ", d.nregions,
          "\n  cell_shape = ", d.cell_shape,
          "\n  direct_lookup_enabled = ", d.direct_lookup_enabled,
          "\n  axes_uniform = ", d.axes_uniform,
          "\n  all_axes_uniform = ", d.all_axes_uniform,
          "\n  critical_coordinate_counts = ", d.critical_coordinate_counts)
end

function Base.show(io::IO, summary::PLBackendValidationSummary)
    r0 = summary.report
    print(io, "PLBackendValidationSummary(kind=", r0.kind,
          ", valid=", r0.valid,
          ", issues=", length(r0.issues), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::PLBackendValidationSummary)
    r0 = summary.report
    println(io, "PLBackendValidationSummary")
    println(io, "  kind: ", r0.kind)
    println(io, "  valid: ", r0.valid)
    for key in propertynames(r0)
        key in (:kind, :valid, :issues) && continue
        println(io, "  ", key, ": ", repr(getproperty(r0, key)))
    end
    if isempty(r0.issues)
        print(io, "  issues: []")
    else
        println(io, "  issues:")
        for issue in r0.issues
            println(io, "    - ", issue)
        end
    end
end

"""
    box_encoding_summary(pi) -> NamedTuple

Owner-local inspection surface for box-backend encoding objects.

Use this as the discoverable `PLBackend` summary entrypoint when you want the
ambient dimension, generator counts, region count, cell-grid shape, and direct
lookup status without inspecting storage fields directly.

Cheap/default path:
- call `box_encoding_summary(pi)` first when you want high-level information
  about the axis grid and region decomposition;
- keep [`describe(pi)`](@ref) as the shared cross-subsystem inspection surface;
- only ask for region-level or bounded-geometry data when you actually need a
  specific query result.

# Examples

```julia
using TamerOp

PLB = TamerOp.PLBackend
CC = TamerOp.ChainComplexes
opts = TamerOp.Options.EncodingOptions(backend=:pl_backend)

Ups = [PLB.BoxUpset([0.0])]
Downs = [PLB.BoxDownset([2.0])]
Phi = reshape(TamerOp.QQ[1], 1, 1)

P, H, pi = PLB.encode_fringe_boxes(Ups, Downs, Phi, opts)

CC.describe(pi)
PLB.box_encoding_summary(pi)

r = PLB.locate(pi, [1.0])
PLB.box_query_summary(pi, [1.0])
PLB.box_region_summary(pi, r; box=(Float64[0.0], Float64[2.0]))
```
"""
@inline box_encoding_summary(pi::PLEncodingMapBoxes) = _plbackend_describe(pi)
@inline function box_encoding_summary(enc::CompiledEncoding{<:PLEncodingMapBoxes})
    return (; _plbackend_describe(enc.pi)...,
            compiled=true,
            poset_kind=(enc.P isa SignaturePoset ? :signature : :dense),
            poset_size=nvertices(enc.P))
end

@inline _box_query_point_key(x::Tuple) = x
@inline _box_query_point_key(x::AbstractVector) = Tuple(x)
@inline _box_query_point_key(x) = x

"""
    box_query_summary(pi, x) -> NamedTuple

Return a cheap semantic summary of a single point query against a box backend
encoding map.

The summary reports:
- the query point and query kind,
- the located region id,
- the stored region representative when a region is found,
- the support sizes of the region signature,
- whether direct cell lookup was available, and
- whether the query landed outside the represented region set (`0` from
  [`locate`](@ref)).

Use this as the cheap/default notebook or REPL inspection path for a single
query before asking for heavier region geometry.
"""
function box_query_summary(pi_or_enc, x)
    pi = _unwrap_box_pi(pi_or_enc)
    report = check_box_point(pi_or_enc, x; throw=true)
    rid = locate(pi_or_enc, x)
    sig_counts = if rid == 0
        nothing
    else
        sig = region_signature(pi, rid)
        (; y=count(identity, sig.y), z=count(identity, sig.z))
    end
    return (;
        kind=:box_query,
        point=_box_query_point_key(x),
        query_kind=report.query_kind,
        region=rid,
        representative=(rid == 0 ? nothing : region_representative(pi, rid)),
        signature_support_sizes=sig_counts,
        direct_lookup_enabled=has_direct_lookup(pi),
        outside=(rid == 0),
    )
end

"""
    box_region_summary(pi, r; box=nothing) -> NamedTuple

Return a compact semantic summary of region `r` in a box-backend encoding.

The summary reports region validity, the stored representative, signature
support sizes, whether a finite query box is required for bounded geometry, and
an optional `bbox` payload when `box=(a,b)` is supplied.

Use this when you already know the region id and want the canonical region-level
inspection payload in one place instead of separately calling
[`region_representative`](@ref), [`region_signature`](@ref), and
[`region_bbox`](@ref).
"""
function box_region_summary(pi_or_enc, r; box=nothing)
    pi = _unwrap_box_pi(pi_or_enc)
    report = check_box_region(pi_or_enc, r; box=box, throw=true)
    return (;
        kind=:box_region,
        ambient_dim=ambient_dim(pi),
        nregions=nregions(pi),
        region=Int(r),
        representative=report.representative,
        signature_support_sizes=report.signature_support_sizes,
        finite_box_required=report.finite_box_required,
        box_provided=(box !== nothing),
        bbox=report.bbox,
        direct_lookup_enabled=has_direct_lookup(pi),
    )
end

# -----------------------------------------------------------------------------
# CompiledEncoding forwarding (treat compiled encodings as primary)
# -----------------------------------------------------------------------------

@inline _unwrap_encoding(pi::CompiledEncoding) = pi.pi

region_weights(pi::CompiledEncoding{<:PLEncodingMapBoxes}; kwargs...) =
    region_weights(_unwrap_encoding(pi); kwargs...)
region_volume(pi::CompiledEncoding{<:PLEncodingMapBoxes}, r; kwargs...) =
    region_volume(_unwrap_encoding(pi), r; kwargs...)
region_bbox(pi::CompiledEncoding{<:PLEncodingMapBoxes}, r; kwargs...) =
    region_bbox(_unwrap_encoding(pi), r; kwargs...)
region_diameter(pi::CompiledEncoding{<:PLEncodingMapBoxes}, r; kwargs...) =
    region_diameter(_unwrap_encoding(pi), r; kwargs...)
region_adjacency(pi::CompiledEncoding{<:PLEncodingMapBoxes}; kwargs...) =
    region_adjacency(_unwrap_encoding(pi); kwargs...)
region_boundary_measure(pi::CompiledEncoding{<:PLEncodingMapBoxes}, r; kwargs...) =
    region_boundary_measure(_unwrap_encoding(pi), r; kwargs...)
region_boundary_measure_breakdown(pi::CompiledEncoding{<:PLEncodingMapBoxes}, r; kwargs...) =
    region_boundary_measure_breakdown(_unwrap_encoding(pi), r; kwargs...)
region_centroid(pi::CompiledEncoding{<:PLEncodingMapBoxes}, r; kwargs...) =
    region_centroid(_unwrap_encoding(pi), r; kwargs...)
region_principal_directions(pi::CompiledEncoding{<:PLEncodingMapBoxes}, r; kwargs...) =
    region_principal_directions(_unwrap_encoding(pi), r; kwargs...)
region_chebyshev_ball(pi::CompiledEncoding{<:PLEncodingMapBoxes}, r; kwargs...) =
    region_chebyshev_ball(_unwrap_encoding(pi), r; kwargs...)
region_circumradius(pi::CompiledEncoding{<:PLEncodingMapBoxes}, r; kwargs...) =
    region_circumradius(_unwrap_encoding(pi), r; kwargs...)
region_mean_width(pi::CompiledEncoding{<:PLEncodingMapBoxes}, r; kwargs...) =
    region_mean_width(_unwrap_encoding(pi), r; kwargs...)

_region_weights_closure(pi::PLEncodingMapBoxes; box, closure::Bool=true, kwargs...) =
    region_weights(pi; box=box, strict=true, kwargs...)

_region_weights_closure(pi::CompiledEncoding{<:PLEncodingMapBoxes}; box, closure::Bool=true, kwargs...) =
    _region_weights_closure(_unwrap_encoding(pi); box=box, closure=closure, kwargs...)

_region_boundary_measure_strict(pi::PLEncodingMapBoxes, r::Integer; box, strict::Bool=true) =
    region_boundary_measure(pi, r; box=box, strict=strict)

_region_boundary_measure_strict(pi::CompiledEncoding{<:PLEncodingMapBoxes}, r::Integer; box, strict::Bool=true) =
    _region_boundary_measure_strict(_unwrap_encoding(pi), r; box=box, strict=strict)

_region_bbox_strict(pi::PLEncodingMapBoxes, r::Integer; box, strict::Bool=true) =
    region_bbox(pi, r; box=box, strict=strict)

_region_bbox_strict(pi::CompiledEncoding{<:PLEncodingMapBoxes}, r::Integer; box, strict::Bool=true) =
    _region_bbox_strict(_unwrap_encoding(pi), r; box=box, strict=strict)

_region_centroid_closure(pi::PLEncodingMapBoxes, r::Integer;
    box, method::Symbol=:bbox, closure::Bool=true) =
    _region_centroid_fast(pi, r; box=box, method=method, closure=closure, cache=nothing)

_region_centroid_closure(pi::CompiledEncoding{<:PLEncodingMapBoxes}, r::Integer;
    box, method::Symbol=:bbox, closure::Bool=true) =
    _region_centroid_closure(_unwrap_encoding(pi), r; box=box, method=method, closure=closure)

function _region_bbox_fast(pi::PLEncodingMapBoxes, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache=nothing)
    _ = (closure, cache)
    return region_bbox(pi, r; box=box, strict=strict)
end

_region_bbox_fast(pi::CompiledEncoding{<:PLEncodingMapBoxes}, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache=nothing) =
    _region_bbox_fast(_unwrap_encoding(pi), r; box=box, strict=strict, closure=closure, cache=cache)

function _region_centroid_fast(pi::PLEncodingMapBoxes, r::Integer;
    box, method::Symbol=:bbox, closure::Bool=true, cache=nothing)
    _ = (closure, cache)
    method === :bbox || return nothing
    bb = region_bbox(pi, r; box=box, strict=true)
    bb === nothing && return nothing
    lo, hi = bb
    return 0.5 .* (lo .+ hi)
end

_region_centroid_fast(pi::CompiledEncoding{<:PLEncodingMapBoxes}, r::Integer;
    box, method::Symbol=:bbox, closure::Bool=true, cache=nothing) =
    _region_centroid_fast(_unwrap_encoding(pi), r; box=box, method=method, closure=closure, cache=cache)

function _region_volume_fast(pi::PLEncodingMapBoxes, r::Integer; box, closure::Bool=true, cache=nothing)
    _ = cache
    return region_volume(pi, r; box=box, strict=true, closure=closure)
end

_region_volume_fast(pi::CompiledEncoding{<:PLEncodingMapBoxes}, r::Integer; box, closure::Bool=true, cache=nothing) =
    _region_volume_fast(_unwrap_encoding(pi), r; box=box, closure=closure, cache=cache)

function _region_boundary_measure_fast(pi::PLEncodingMapBoxes, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache=nothing)
    _ = (closure, cache)
    return region_boundary_measure(pi, r; box=box, strict=strict)
end

function _region_boundary_measure_fast(pi::CompiledEncoding{<:PLEncodingMapBoxes}, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache=nothing)
    _ = (closure, cache)
    return region_boundary_measure(pi, r; box=box, strict=strict)
end

function _region_circumradius_fast(pi::PLEncodingMapBoxes, r::Integer;
    box, center=:bbox, metric::Symbol=:L2, method::Symbol=:bbox,
    strict::Bool=true, closure::Bool=true, cache=nothing)
    _ = (closure, cache)
    return method === :bbox ? region_circumradius(pi, r;
        box=box, center=center, metric=metric, method=:cells, strict=strict) : nothing
end

_region_circumradius_fast(pi::CompiledEncoding{<:PLEncodingMapBoxes}, r::Integer;
    box, center=:bbox, metric::Symbol=:L2, method::Symbol=:bbox,
    strict::Bool=true, closure::Bool=true, cache=nothing) =
    _region_circumradius_fast(_unwrap_encoding(pi), r;
        box=box, center=center, metric=metric, method=method,
        strict=strict, closure=closure, cache=cache)

function _region_minkowski_functionals_fast(pi::PLEncodingMapBoxes, r::Integer;
    box, volume=nothing, boundary=nothing, mean_width_method::Symbol=:auto,
    mean_width_ndirs::Integer=256, mean_width_rng=Random.default_rng(),
    mean_width_directions=nothing, strict::Bool=true, closure::Bool=true,
    cache=nothing)
    _ = (closure, cache)
    V = volume === nothing ? region_volume(pi, r; box=box, strict=strict) : float(volume)
    S = boundary === nothing ? region_boundary_measure(pi, r; box=box, strict=strict) : float(boundary)
    mw = if (mean_width_method === :auto || mean_width_method === :cauchy) && length(box[1]) == 2
        S / Base.MathConstants.pi
    else
        region_mean_width(pi, r; box=box, method=mean_width_method,
            ndirs=mean_width_ndirs, rng=mean_width_rng,
            directions=mean_width_directions, strict=strict)
    end
    return (volume=V, boundary_measure=S, mean_width=float(mw))
end

_region_minkowski_functionals_fast(pi::CompiledEncoding{<:PLEncodingMapBoxes}, r::Integer;
    box, volume=nothing, boundary=nothing, mean_width_method::Symbol=:auto,
    mean_width_ndirs::Integer=256, mean_width_rng=Random.default_rng(),
    mean_width_directions=nothing, strict::Bool=true, closure::Bool=true,
    cache=nothing) =
    _region_minkowski_functionals_fast(_unwrap_encoding(pi), r;
        box=box, volume=volume, boundary=boundary,
        mean_width_method=mean_width_method, mean_width_ndirs=mean_width_ndirs,
        mean_width_rng=mean_width_rng, mean_width_directions=mean_width_directions,
        strict=strict, closure=closure, cache=cache)

function _region_geometry_summary_fast(pi::PLEncodingMapBoxes, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache=nothing,
    mean_width_method::Symbol=:auto, mean_width_ndirs::Integer=256,
    mean_width_rng=Random.default_rng(), mean_width_directions=nothing,
    need_mean_width::Bool=false)
    _ = (closure, cache)
    V = region_volume(pi, r; box=box, strict=strict)
    S = region_boundary_measure(pi, r; box=box, strict=strict)
    mw = if !need_mean_width
        NaN
    elseif (mean_width_method === :auto || mean_width_method === :cauchy) && length(box[1]) == 2
        S / Base.MathConstants.pi
    else
        region_mean_width(pi, r; box=box, method=mean_width_method,
            ndirs=mean_width_ndirs, rng=mean_width_rng,
            directions=mean_width_directions, strict=strict)
    end
    # Keep the fast summary contract minimal. Boundary-to-volume and Minkowski
    # callers only consume these scalars, so computing bbox/centroid/circumradius
    # here just adds overhead without improving the hot path.
    return (volume=float(V), boundary_measure=float(S), mean_width=float(mw))
end

_region_geometry_summary_fast(pi::CompiledEncoding{<:PLEncodingMapBoxes}, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache=nothing,
    mean_width_method::Symbol=:auto, mean_width_ndirs::Integer=256,
    mean_width_rng=Random.default_rng(), mean_width_directions=nothing,
    need_mean_width::Bool=false) =
    _region_geometry_summary_fast(_unwrap_encoding(pi), r;
        box=box, strict=strict, closure=closure, cache=cache,
        mean_width_method=mean_width_method, mean_width_ndirs=mean_width_ndirs,
        mean_width_rng=mean_width_rng, mean_width_directions=mean_width_directions,
        need_mean_width=need_mean_width)

end # module PLBackend
