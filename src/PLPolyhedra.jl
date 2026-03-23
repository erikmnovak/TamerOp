module PLPolyhedra
# =============================================================================
# Piecewise-linear (PL) backend using Polyhedra + CDD exact arithmetic.
#
# NOTE ON OPTIONAL DEPENDENCIES
# - Polyhedra/CDDLib are optional.
# - This module still loads without them.
# - Membership tests in HPoly / PolyUnion are implemented directly from stored
#   H-representations A*x <= b (exact QQ) and DO NOT require Polyhedra.
# - Feasibility enumeration (encode_from_PL_fringe) DOES require Polyhedra/CDDLib.
#
# Key robustness change:
#   We store the H-rep matrix (A,b) inside HPoly and avoid Polyhedra API helpers
#   like hrep_matrix (which may not exist in the installed Polyhedra version).
# =============================================================================

using ..FiniteFringe
import ..FiniteFringe: AbstractPoset, nvertices, birth_upsets, death_downsets, field
import ..ZnEncoding: SignaturePoset, nregions, region_signature
using ..CoreModules: QQ, EncodingCache, GeometryCachePayload
using ..Options: EncodingOptions, validate_pl_mode
using ..EncodingCore: AbstractPLikeEncodingMap, CompiledEncoding
import ..DataTypes: ambient_dim
import ..FlangeZn: coefficient_matrix
using ..CoreModules.CoeffFields: QQField
using ..Stats: _wilson_interval
using Random

import ..EncodingCore: locate, locate_many!, locate_many, dimension, representatives, axes_from_encoding
import ..RegionGeometry: region_weights, region_volume, region_bbox, region_diameter,
                         region_facet_count, region_vertex_count, region_adjacency,
                         region_boundary_measure, region_boundary_measure_breakdown, 
                         region_centroid, region_perimeter, region_principal_directions,
                         region_chebyshev_ball, region_circumradius, region_mean_width,
                         _cache_box, _region_bbox_fast, _region_centroid_fast,
                         _region_volume_fast, _region_boundary_measure_fast,
                         _region_circumradius_fast, _region_minkowski_functionals_fast,
                         _region_geometry_summary_fast, _region_weights_cached,
                         _region_weights_closure, _region_boundary_measure_cached,
                         _region_boundary_measure_closure, _region_bbox_cached,
                         _region_bbox_closure, _region_centroid_cached,
                         _region_centroid_closure, _region_encoding_cache

using LinearAlgebra



# ------------------------------ Optional deps ---------------------------------
const HAVE_POLY = try
    @eval begin
        import Polyhedra
        import CDDLib
    end
    true
catch
    false
end

const _CDD = HAVE_POLY ? CDDLib.Library(:exact) : nothing
const _CDD_FLOAT = HAVE_POLY ? CDDLib.Library(:float) : nothing

# Small positive slack for strict facet violation (kept exact)
const STRICT_EPS_QQ = 1//(big(1) << 40)
const LOCATE_FLOAT_TOL = 1e-12
const LOCATE_BOUNDARY_TOL = 1e-9
const LOCATE_GUARD_REL = 64.0 * eps(Float64)
const LOCATE_GUARD_ABS = 1e-15

# ------------------------------ QQ conversion ---------------------------------
# Robust conversion to QQ = Rational{BigInt}.
function _toQQ(u)::QQ
    if u isa QQ
        return u
    elseif u isa Integer
        return BigInt(u) // BigInt(1)
    elseif u isa Rational
        return BigInt(numerator(u)) // BigInt(denominator(u))
    elseif u isa AbstractFloat
        # rationalize expects an INTEGER type and returns Rational{T}.
        return rationalize(BigInt, u)
    else
        # Fallback for other Reals (e.g. IrrationalConstants, etc.)
        return rationalize(BigInt, float(u))
    end
end


function _toQQ_vec(v::AbstractVector)
    out = Vector{QQ}(undef, length(v))
    for i in eachindex(v)
        out[i] = _toQQ(v[i])
    end
    return out
end

function _toQQ_mat(A::AbstractMatrix)
    m, n = size(A)
    out = Matrix{QQ}(undef, m, n)
    for i in 1:m, j in 1:n
        out[i,j] = _toQQ(A[i,j])
    end
    return out
end

# ------------------------------ Basic types -----------------------------------

"""
HPoly

A single convex polyhedron in H-representation:
    { x in R^n : A*x <= b }.

We store A,b explicitly (QQ) to avoid relying on Polyhedra internals for
membership tests and facet extraction.

Field `poly` is an optional Polyhedra object (Any). It is present only when
HAVE_POLY is true and the polyhedron was constructed with Polyhedra.
"""
struct HPoly
    n::Int
    A::Matrix{QQ}
    b::Vector{QQ}
    poly::Any
    strict_mask::BitVector
    strict_eps::QQ
end

"""
PolyUnion

Finite union of convex HPolys. Membership is disjunction.
"""
struct PolyUnion
    n::Int
    parts::Vector{HPoly}
end

"""
PLUpset / PLDownset

Birth and death shapes as unions of convex parts.
"""
struct PLUpset
    U::PolyUnion
end
struct PLDownset
    D::PolyUnion
end

# ------------------------------------------------------------------------------
# PL fringe presentations (modules over R^n presented by PL birth/death data)
# ------------------------------------------------------------------------------

"""
    PLFringe(Ups, Downs, Phi)

A "fringe presentation" for a module over the poset R^n (coordinatewise order),
specified by:

- `Ups`   : birth upsets (columns), each a `PLUpset`
- `Downs` : death downsets (rows), each a `PLDownset`
- `Phi`   : a matrix over `QQ` of size length(Downs) x length(Ups)

Interpretation (informal but useful):
this is the data of a monomial matrix presenting a module as the image of a map
from a direct sum of upset indicators to a direct sum of downset indicators.

This object is not yet a finite-poset module. Use `encode_from_PL_fringe` (single)
or `encode_from_PL_fringes` (common encoding for several) to compress to a finite
encoding poset `P` together with a classifier `pi : R^n -> P`.
"""
struct PLFringe
    n::Int
    Ups::Vector{PLUpset}
    Downs::Vector{PLDownset}
    Phi::Matrix{QQ}
end

"""
    PLFringe(Ups, Downs, Phi_in; check=true) -> PLFringe

Constructor that validates ambient dimension consistency and coerces coefficients
to `QQ` exactly (floating inputs are rationalized via `_toQQ`).

The `check` flag controls whether we assert the shapes are PL-typed.
"""
function PLFringe(Ups::Vector{PLUpset},
                  Downs::Vector{PLDownset},
                  Phi_in::AbstractMatrix; check::Bool=true)

    check && _assert_PL_inputs(Ups, Downs)

    n = (length(Ups) > 0 ? Ups[1].U.n : (length(Downs) > 0 ? Downs[1].D.n : 0))

    for (i, U) in enumerate(Ups)
        U.U.n == n || error("PLFringe: Ups[$i] lives in R^$(U.U.n) but expected R^$n")
    end
    for (j, D) in enumerate(Downs)
        D.D.n == n || error("PLFringe: Downs[$j] lives in R^$(D.D.n) but expected R^$n")
    end

    size(Phi_in, 1) == length(Downs) || error("PLFringe: Phi has wrong number of rows")
    size(Phi_in, 2) == length(Ups)   || error("PLFringe: Phi has wrong number of columns")

    Phi = Matrix{QQ}(undef, size(Phi_in, 1), size(Phi_in, 2))
    for j in 1:size(Phi, 1), i in 1:size(Phi, 2)
        Phi[j, i] = _toQQ(Phi_in[j, i])
    end

    return PLFringe(n, Ups, Downs, Phi)
end

function polyhedra end
function npolyhedra end
function region_witnesses end
function region_witness end
function has_spatial_index end
function cache_box end
function cache_level end
function cached_region_count end
function check_hpoly end
function check_poly_union end
function check_pl_upset end
function check_pl_downset end
function check_pl_fringe end
function check_pl_encoding_map end
function check_poly_in_box_cache end
function check_pl_point end
function check_pl_points end
function check_pl_box end
function check_pl_region end
function plpolyhedra_validation_summary end
function pl_fringe_summary end
function pl_encoding_summary end
function pl_query_summary end
function pl_region_summary end
function poly_cache_summary end

"""
    encode_from_PL_fringe(F::PLFringe, opts::EncodingOptions; poset_kind=:signature) -> (P, H, pi)

Encode a single `PLFringe` presentation over R^n to a finite encoding poset `P`,
returning the pushed-down `FiniteFringe.FringeModule{QQ}` and the classifier
`pi : R^n -> P` (as `PLEncodingMap`).

Mathematical meaning:
- `P` is the finite poset of feasible polyhedral regions cut out by the birth
  and death pieces of `F`;
- `H` is the compressed fringe module on `P`;
- `pi` is the region classifier used to locate points of `R^n` in that finite
  encoding.

Best practice:
- keep `poset_kind=:signature` as the cheap/default path; it preserves the
  structured region-signature poset and is usually the better notebook/REPL
  surface;
- use `poset_kind=:dense` only when a fully materialized finite-poset
  realization is specifically needed downstream.

`opts` is required.
- `opts.backend` must be `:auto` or `:pl`.
- `opts.max_regions` caps region enumeration (default: 10_000).
- `opts.strict_eps` controls strict inequality handling in feasibility checks
  (default: `STRICT_EPS_QQ`).
"""
function encode_from_PL_fringe(F::PLFringe, opts::EncodingOptions; poset_kind::Symbol = :signature)
    return encode_from_PL_fringe(F.Ups, F.Downs, F.Phi, opts; poset_kind = poset_kind)
end

encode_from_PL_fringe(F::PLFringe;
                      opts::EncodingOptions=EncodingOptions(),
                      poset_kind::Symbol = :signature) =
    encode_from_PL_fringe(F, opts; poset_kind = poset_kind)



# ----------------------------- Geometry classes --------------------------------
const GeometryClass = Symbol
geometry_class(::HPoly)      = :PL::GeometryClass
geometry_class(::PolyUnion)  = :PL::GeometryClass
geometry_class(::PLUpset)    = :PL::GeometryClass
geometry_class(::PLDownset)  = :PL::GeometryClass

function _assert_PL_inputs(Ups::Vector{PLUpset}, Downs::Vector{PLDownset})
    for u in Ups
        @assert geometry_class(u) === :PL "Non-PL upset encountered"
    end
    for d in Downs
        @assert geometry_class(d) === :PL "Non-PL downset encountered"
    end
    nothing
end

# --------------------------- Construction helpers -----------------------------

"""
    make_hpoly(A, b) -> HPoly

Build a convex polyhedron { x : A*x <= b }.

- Stores A,b exactly in QQ inside the returned HPoly.
- If Polyhedra/CDDLib are available, also builds an exact Polyhedra object
  for feasibility testing and witness extraction used by encode_from_PL_fringe.
"""
function make_hpoly(A::AbstractMatrix, b::AbstractVector)
    m, n = size(A)
    length(b) == m || error("make_hpoly: size mismatch A (m,n) vs b (m).")

    Aqq = _toQQ_mat(A)
    bqq = _toQQ_vec(b)

    poly = nothing
    if HAVE_POLY
        hrep = Polyhedra.hrep(Aqq, bqq)
        poly = Polyhedra.polyhedron(hrep, _CDD)
    end

    return HPoly(n, Aqq, bqq, poly, falses(m), STRICT_EPS_QQ)
end

"Convenience: single-part PolyUnion."
poly_union(h::HPoly) = PolyUnion(h.n, [h])

# Convenience overloads: allow a single inequality a'*x <= b to be specified
# with a as a vector.
function make_hpoly(a::AbstractVector, b::AbstractVector)
    # Interpret a as a single row of A.
    return make_hpoly(reshape(a, 1, length(a)), b)
end

function make_hpoly(a::AbstractVector, b::Real)
    return make_hpoly(a, [b])
end


# ---------------------------- Membership checks -------------------------------

"Exact membership using stored A*x <= b in QQ."
function _in_hpoly(h::HPoly, x::AbstractVector)
    length(x) == h.n || error("dimension mismatch in _in_hpoly")
    xqq = _toQQ_vec(x)
    m = size(h.A, 1)
    for i in 1:m
        s = zero(QQ)
        @inbounds for j in 1:h.n
            s += h.A[i,j] * xqq[j]
        end
        if s > h.b[i]
            return false
        end
    end
    return true
end

function _in_hpoly(h::HPoly, x::NTuple{N,<:Real}) where {N}
    N == h.n || error("dimension mismatch in _in_hpoly")
    m = size(h.A, 1)
    for i in 1:m
        s = zero(QQ)
        @inbounds for j in 1:h.n
            s += h.A[i,j] * _toQQ(x[j])
        end
        if s > h.b[i]
            return false
        end
    end
    return true
end

function _in_hpoly_open(h::HPoly, x::AbstractVector)
    length(x) == h.n || error("dimension mismatch in _in_hpoly_open")
    xqq = _toQQ_vec(x)
    m = size(h.A, 1)
    for i in 1:m
        s = zero(QQ)
        @inbounds for j in 1:h.n
            s += h.A[i, j] * xqq[j]
        end
        if s >= h.b[i]
            return false
        end
    end
    return true
end

# For strict (>=) constraints we store them as (-a) * x <= -(b + strict_eps).
# For closure computations we want (-a) * x <= -b, i.e. relax those rows by adding strict_eps.
function _relaxed_b(h::HPoly)
    b = copy(h.b)
    eps = h.strict_eps
    @inbounds for i in eachindex(b)
        if h.strict_mask[i]
            b[i] += eps
        end
    end
    return b
end

"Point membership in a union of polytopes."
function _in_union(U::PolyUnion, x::AbstractVector)
    for p in U.parts
        if _in_hpoly(p, x)
            return true
        end
    end
    return false
end

"""
    contains(U::PLUpset, x)
    contains(D::PLDownset, x)

Test whether the query point `x` lies in the PL upset or PL downset.

This is a cheap membership probe against the stored H-representation pieces. It
does not build Polyhedra objects or consult any geometry cache. Use it for
one-off semantic checks on the birth/death data itself; use [`locate`](@ref) on
an encoded [`PLEncodingMap`](@ref) when you want region ids in the finite
encoding.
"""
contains(U::PLUpset, x::AbstractVector)   = _in_union(U.U, x)
contains(D::PLDownset, x::AbstractVector) = _in_union(D.D, x)

# ---------------------- Feasibility helper for enumeration --------------------

# Collect facet inequalities (a, b) representing a'*x <= b from stored A,b.
function _facets_of(hp::HPoly)::Vector{Tuple{Vector{QQ},QQ}}
    m, n = size(hp.A)
    out = Vector{Tuple{Vector{QQ},QQ}}(undef, m)
    @inbounds for i in 1:m
        ai = Vector{QQ}(undef, n)
        for j in 1:n
            ai[j] = hp.A[i,j]
        end
        out[i] = (ai, hp.b[i])
    end
    return out
end


# Build intersection of in_parts plus "outside" constraints represented as:
#   a'*x >= b0  (to emulate strict violation we use b0 + STRICT_EPS_QQ)
#
# Returns a triple (hp, witness, is_empty):
# - hp::Union{HPoly,Nothing}
# - witness::Union{Vector{Float64},Nothing}
# - is_empty::Bool
function _internal_build_poly(in_parts::Vector{HPoly},
                              out_halfspaces::Vector{Tuple{Vector{T},T}};
                              strict_eps::QQ=STRICT_EPS_QQ) where {T<:Real}
    HAVE_POLY || error("Polyhedra/CDDLib not available; install Polyhedra.jl and CDDLib.jl.")

    # Determine ambient dimension.
    n = if !isempty(in_parts)
        in_parts[1].n
    elseif !isempty(out_halfspaces)
        length(out_halfspaces[1][1])
    else
        0
    end
    n == 0 && return (nothing, nothing, true)

    # Count total constraints.
    m_in = 0
    for p in in_parts
        p.n == n || error("_internal_build_poly: dimension mismatch among in_parts")
        m_in += size(p.A, 1)
    end
    m_out = length(out_halfspaces)
    m_tot = m_in + m_out

    A = Matrix{QQ}(undef, m_tot, n)
    b = Vector{QQ}(undef, m_tot)

    # Fill with in-parts inequalities.
    row = 1
    for p in in_parts
        mp = size(p.A, 1)
        if mp > 0
            A[row:row+mp-1, :] .= p.A
            b[row:row+mp-1]    .= p.b
            row += mp
        end
    end

    # Add outside constraints: a'*x >= b0 + eps  <=>  (-a)'*x <= -(b0 + eps)
    for (a0, b0) in out_halfspaces
        length(a0) == n || error("_internal_build_poly: outside halfspace has wrong dimension")
        for j in 1:n
            A[row, j] = -_toQQ(a0[j])
        end
        b[row] = -(_toQQ(b0) + strict_eps)
        row += 1
    end

    # Mark which rows correspond to strict "outside" constraints (the last m_out rows).
    strict_mask = BitVector(vcat(fill(false, m_in), fill(true, m_out)))

    hrep = Polyhedra.hrep(A, b)
    P = Polyhedra.polyhedron(hrep, _CDD)

    if Polyhedra.isempty(P)
        return (nothing, nothing, true)
    end

    # Best-effort witness (not required for correctness; safe if Polyhedra API differs).
    witness = nothing
    try
        V = Polyhedra.vrep(P)
        pts = Polyhedra.points(V)
        firstpt = iterate(pts)
        if firstpt !== nothing
            witness = Vector{Float64}(firstpt[1])
        end
    catch
        witness = nothing
    end

    return (HPoly(n, A, b, P, strict_mask, strict_eps), witness, false)
end

# ------------------------- Region enumeration (Y-signatures) ------------------

function enumerate_feasible_regions(Ups::Vector{PLUpset}, Downs::Vector{PLDownset};
                                    max_regions::Int=10_000,
                                    strict_eps::QQ=STRICT_EPS_QQ)
    HAVE_POLY || error("Polyhedra/CDDLib not available; install Polyhedra.jl and CDDLib.jl.")

    m = length(Ups)
    r = length(Downs)
    n = (m > 0 ? Ups[1].U.n : (r > 0 ? Downs[1].D.n : 0))

    results = Vector{Tuple{BitVector,BitVector,HPoly,Tuple}}()

    # Helper: all ways to force OUTSIDE a union of HPolys:
    # pick one facet inequality to violate for each part.
    function outside_choices(union::PolyUnion)
        if isempty(union.parts)
            return [Tuple{Vector{QQ},QQ}[]]  # outside(empty) = whole space, no constraints
        end
        facet_lists = Vector{Vector{Tuple{Vector{QQ},QQ}}}(undef, length(union.parts))
        for i in 1:length(union.parts)
            facet_lists[i] = _facets_of(union.parts[i])
            if isempty(facet_lists[i])
                return Vector{Vector{Tuple{Vector{QQ},QQ}}}() # cannot be outside full space part
            end
        end
        out = Vector{Vector{Tuple{Vector{QQ},QQ}}}()
        function rec(i::Int, acc::Vector{Tuple{Vector{QQ},QQ}})
            if i > length(facet_lists)
                push!(out, copy(acc))
                return
            end
            for f in facet_lists[i]
                push!(acc, f)
                rec(i+1, acc)
                pop!(acc)
            end
        end
        rec(1, Tuple{Vector{QQ},QQ}[])
        return out
    end

    # Iterate over all signatures y in {0,1}^m and z in {0,1}^r.
    total = 1 << (m + r)
    for mask in 0:(total-1)
        y = falses(m)
        z = falses(r)
        for i in 1:m
            y[i] = ((mask >> (i-1)) & 1) == 1
        end
        for j in 1:r
            z[j] = ((mask >> (m + j - 1)) & 1) == 1
        end

        # Build "inside" disjunction choices (pick one part for each inside union constraint).
        in_choices = Vector{Vector{HPoly}}()
        push!(in_choices, HPoly[])

        # For Upsets: y[i] == 1 means inside U_i; y[i] == 0 means outside U_i.
        out_choices = Vector{Vector{Tuple{Vector{QQ},QQ}}}()
        push!(out_choices, Tuple{Vector{QQ},QQ}[])

        # Upset constraints
        feasible = true
        for i in 1:m
            if y[i]
                # inside union: choose one part
                parts = Ups[i].U.parts
                if isempty(parts)
                    feasible = false
                    break
                end
                new_in = Vector{Vector{HPoly}}()
                for base in in_choices, part in parts
                    push!(new_in, vcat(base, [part]))
                end
                in_choices = new_in
            else
                # outside union
                choices = outside_choices(Ups[i].U)
                if isempty(choices)
                    feasible = false
                    break
                end
                out_choices = [vcat(base, ch) for base in out_choices, ch in choices]
            end
        end
        feasible || continue

        # Downset constraints: z[j] == 0 means inside D_j; z[j] == 1 means outside D_j.
        for j in 1:r
            if !z[j]
                parts = Downs[j].D.parts
                if isempty(parts)
                    feasible = false
                    break
                end
                new_in = Vector{Vector{HPoly}}()
                for base in in_choices, part in parts
                    push!(new_in, vcat(base, [part]))
                end
                in_choices = new_in
            else
                choices = outside_choices(Downs[j].D)
                if isempty(choices)
                    feasible = false
                    break
                end
                out_choices = [vcat(base, ch) for base in out_choices, ch in choices]
            end
        end
        feasible || continue

        # Test feasibility for each branch.
        for iparts in in_choices
            for out_hs in out_choices
                hp, wit, isemp = _internal_build_poly(iparts, out_hs; strict_eps=strict_eps)
                if !isemp
                    push!(results, (BitVector(y), BitVector(z), hp, wit === nothing ? () : Tuple(wit)))
                    if length(results) >= max_regions
                        break
                    end
                end
            end
            if length(results) >= max_regions
                break
            end
        end

        if length(results) >= max_regions
            @warn "enumerate_feasible_regions: reached max_regions cap; stopping early."
            break
        end
    end

    # Collapse equivalent (y,z) signatures. Use tuple-of-bools keys (content-hashable).
    seen = Dict{Tuple{Tuple,Tuple},Int}()
    collapsed = Vector{Tuple{BitVector,BitVector,HPoly,Tuple}}()
    for rec in results
        key = (Tuple(rec[1]), Tuple(rec[2]))
        if !haskey(seen, key)
            seen[key] = 1
            push!(collapsed, rec)
        end
    end
    return collapsed
end

# --------------------------- Encoding (P, H_hat, pi) --------------------------

# --- Core encoding-map interface ------------------------------------------------

struct _PackedBucketSpan
    idx::Vector{Int}
    lo::Int
    hi::Int
end

Base.isempty(span::_PackedBucketSpan) = span.lo > span.hi
Base.length(span::_PackedBucketSpan) = max(0, span.hi - span.lo + 1)

@inline function Base.iterate(span::_PackedBucketSpan, state::Int=span.lo)
    state > span.hi && return nothing
    return (span.idx[state], state + 1)
end

struct _PackedBuckets
    ptr::Vector{Int}
    idx::Vector{Int}
end

Base.isempty(b::_PackedBuckets) = length(b.ptr) <= 1

@inline function Base.getindex(b::_PackedBuckets, i::Int)
    lo = b.ptr[i]
    hi = b.ptr[i + 1] - 1
    return _PackedBucketSpan(b.idx, lo, hi)
end

@inline _empty_packed_buckets() = _PackedBuckets(Int[], Int[])

struct _LocateSpatialIndex
    enabled::Bool
    nx::Int
    ny::Int
    x0::Float64
    y0::Float64
    dx::Float64
    dy::Float64
    buckets::_PackedBuckets
    avg_len::Float64
end

struct _MultiProjPrefilter
    enabled::Bool
    ndims::Int
    dims::NTuple{3,Int}
    x_sorted::NTuple{3,Vector{Float64}}
    perm::NTuple{3,Vector{Int}}
    pos::NTuple{3,Vector{Int}}
    window::Int
end

struct _LocatePrefilter
    enabled::Bool
    x1_sorted::Vector{Float64}
    perm::Vector{Int}
    pos::Vector{Int}
    window::Int
    spatial::_LocateSpatialIndex
    multiproj::_MultiProjPrefilter
end

@inline _empty_spatial_index() = _LocateSpatialIndex(false, 1, 1, 0.0, 0.0, 1.0, 1.0, _empty_packed_buckets(), 0.0)

@inline _empty_multiproj_prefilter() = _MultiProjPrefilter(
    false,
    0,
    (0, 0, 0),
    (Float64[], Float64[], Float64[]),
    (Int[], Int[], Int[]),
    (Int[], Int[], Int[]),
    0,
)

struct PLEncodingMap <: AbstractPLikeEncodingMap
    n::Int
    sig_y::Vector{BitVector}
    sig_z::Vector{BitVector}
    regions::Vector{HPoly}
    witnesses::Vector{Tuple}
    Af::Vector{Matrix{Float64}}
    bf_strict::Vector{Vector{Float64}}
    bf_relaxed::Vector{Vector{Float64}}
    prefilter::_LocatePrefilter
    cache::EncodingCache
end

function _region_bbox_from_poly(hp::HPoly, n::Int)
    HAVE_POLY || return nothing
    hp.poly === nothing && return nothing
    pts = try
        collect(Polyhedra.points(Polyhedra.vrep(hp.poly)))
    catch
        return nothing
    end
    isempty(pts) && return nothing
    lo = fill(Inf, n)
    hi = fill(-Inf, n)
    @inbounds for p in pts
        x = _point_to_floatvec(p, n)
        for j in 1:n
            xj = x[j]
            if xj < lo[j]
                lo[j] = xj
            end
            if xj > hi[j]
                hi[j] = xj
            end
        end
    end
    (all(isfinite, lo) && all(isfinite, hi)) || return nothing
    @inbounds for j in 1:n
        hi[j] >= lo[j] || return nothing
    end
    return (lo, hi)
end

function _build_locate_spatial_index(n::Int, regions::Vector{HPoly})
    nr = length(regions)
    if !HAVE_POLY || n == 0 || n > 2 || nr < 64
        return _empty_spatial_index()
    end

    lo1 = fill(Inf, nr)
    hi1 = fill(-Inf, nr)
    lo2 = fill(0.0, nr)
    hi2 = fill(0.0, nr)
    valid = falses(nr)

    g_lo1 = Inf
    g_hi1 = -Inf
    g_lo2 = Inf
    g_hi2 = -Inf

    @inbounds for r in 1:nr
        bb = _region_bbox_from_poly(regions[r], n)
        bb === nothing && continue
        lo, hi = bb
        isfinite(lo[1]) && isfinite(hi[1]) && hi[1] >= lo[1] || continue
        ylo = (n >= 2 ? lo[2] : 0.0)
        yhi = (n >= 2 ? hi[2] : 0.0)
        isfinite(ylo) && isfinite(yhi) && yhi >= ylo || continue

        lo1[r] = lo[1]
        hi1[r] = hi[1]
        lo2[r] = ylo
        hi2[r] = yhi
        valid[r] = true

        if lo[1] < g_lo1
            g_lo1 = lo[1]
        end
        if hi[1] > g_hi1
            g_hi1 = hi[1]
        end
        if ylo < g_lo2
            g_lo2 = ylo
        end
        if yhi > g_hi2
            g_hi2 = yhi
        end
    end

    nvalid = count(valid)
    if nvalid < max(32, Int(cld(nr, 4))) || !isfinite(g_lo1) || !isfinite(g_hi1) || g_hi1 < g_lo1
        return _empty_spatial_index()
    end

    nx = max(8, min(256, round(Int, sqrt(nvalid))))
    ny = (n >= 2 ? max(8, min(256, round(Int, sqrt(nvalid)))) : 1)
    dx = max((g_hi1 - g_lo1) / nx, 1e-12)
    dy = (n >= 2 ? max((g_hi2 - g_lo2) / ny, 1e-12) : 1.0)
    x0 = g_lo1
    y0 = (n >= 2 ? g_lo2 : 0.0)
    nb = nx * ny
    counts = zeros(Int, nb)

    @inbounds for r in 1:nr
        valid[r] || continue
        ix0 = clamp(Int(floor((lo1[r] - x0) / dx)) + 1, 1, nx)
        ix1 = clamp(Int(floor((hi1[r] - x0) / dx)) + 1, 1, nx)
        iy0 = 1
        iy1 = 1
        if n >= 2
            iy0 = clamp(Int(floor((lo2[r] - y0) / dy)) + 1, 1, ny)
            iy1 = clamp(Int(floor((hi2[r] - y0) / dy)) + 1, 1, ny)
        end
        for iy in iy0:iy1, ix in ix0:ix1
            counts[(iy - 1) * nx + ix] += 1
        end
    end

    ptr = Vector{Int}(undef, nb + 1)
    ptr[1] = 1
    @inbounds for bid in 1:nb
        ptr[bid + 1] = ptr[bid] + counts[bid]
    end
    idx = Vector{Int}(undef, ptr[end] - 1)
    writepos = copy(ptr[1:end-1])

    @inbounds for r in 1:nr
        valid[r] || continue
        ix0 = clamp(Int(floor((lo1[r] - x0) / dx)) + 1, 1, nx)
        ix1 = clamp(Int(floor((hi1[r] - x0) / dx)) + 1, 1, nx)
        iy0 = 1
        iy1 = 1
        if n >= 2
            iy0 = clamp(Int(floor((lo2[r] - y0) / dy)) + 1, 1, ny)
            iy1 = clamp(Int(floor((hi2[r] - y0) / dy)) + 1, 1, ny)
        end
        for iy in iy0:iy1, ix in ix0:ix1
            bid = (iy - 1) * nx + ix
            pos = writepos[bid]
            idx[pos] = r
            writepos[bid] = pos + 1
        end
    end

    total_bucket = ptr[end] - 1
    avg_len = nb == 0 ? 0.0 : (total_bucket / nb)
    return _LocateSpatialIndex(true, nx, ny, x0, y0, dx, dy, _PackedBuckets(ptr, idx), avg_len)
end

function _build_multiproj_prefilter(n::Int, witnesses::Vector{Tuple})
    nr = length(witnesses)
    if n < 3 || nr < 64
        return _empty_multiproj_prefilter()
    end
    @inbounds for i in 1:nr
        length(witnesses[i]) == n || return _empty_multiproj_prefilter()
    end

    means = zeros(Float64, n)
    @inbounds for i in 1:nr
        wi = witnesses[i]
        for j in 1:n
            means[j] += float(wi[j])
        end
    end
    inv_nr = 1.0 / nr
    @inbounds for j in 1:n
        means[j] *= inv_nr
    end

    vars = zeros(Float64, n)
    @inbounds for i in 1:nr
        wi = witnesses[i]
        for j in 1:n
            d = float(wi[j]) - means[j]
            vars[j] += d * d
        end
    end

    dims_ord = sortperm(vars; rev=true)
    k = min(3, n)
    vars[dims_ord[1]] > 0.0 || return _empty_multiproj_prefilter()

    dims = (0, 0, 0)
    x_sorted = [Float64[], Float64[], Float64[]]
    perm = [Int[], Int[], Int[]]
    pos = [Int[], Int[], Int[]]
    d1 = dims_ord[1]
    d2 = (k >= 2 ? dims_ord[2] : 0)
    d3 = (k >= 3 ? dims_ord[3] : 0)
    dims = (d1, d2, d3)

    @inbounds for slot in 1:k
        d = dims[slot]
        vals = Vector{Float64}(undef, nr)
        for i in 1:nr
            vals[i] = float(witnesses[i][d])
        end
        p = sortperm(vals)
        xs = vals[p]
        po = Vector{Int}(undef, nr)
        for (rank, rid) in enumerate(p)
            po[rid] = rank
        end
        x_sorted[slot] = xs
        perm[slot] = p
        pos[slot] = po
    end

    root = nr^(1.0 / k)
    # Keep the projection window conservative: misses are safe (we fall back to
    # canonical prefilter/full scan), and a smaller window lowers hot-path scan cost.
    window = max(4, min(96, Int(round(root))))
    return _MultiProjPrefilter(
        true,
        k,
        dims,
        (x_sorted[1], x_sorted[2], x_sorted[3]),
        (perm[1], perm[2], perm[3]),
        (pos[1], pos[2], pos[3]),
        window,
    )
end

function _build_locate_prefilter(n::Int, regions::Vector{HPoly}, witnesses::Vector{Tuple})
    nr = length(regions)
    spatial = _build_locate_spatial_index(n, regions)
    multiproj = _build_multiproj_prefilter(n, witnesses)
    if n == 0 || nr <= 8 || length(witnesses) != nr
        return _LocatePrefilter(false, Float64[], Int[], Int[], 0, spatial, multiproj)
    end
    x1 = Vector{Float64}(undef, nr)
    @inbounds for i in 1:nr
        wi = witnesses[i]
        length(wi) == n || return _LocatePrefilter(false, Float64[], Int[], Int[], 0, spatial, multiproj)
        x1[i] = float(wi[1])
    end
    perm = sortperm(x1)
    x1s = x1[perm]
    pos = Vector{Int}(undef, nr)
    @inbounds for (k, rid) in enumerate(perm)
        pos[rid] = k
    end
    # Use an adaptive fixed-width candidate window around x[1].
    window = max(4, min(64, Int(cld(isqrt(nr), 1))))
    return _LocatePrefilter(true, x1s, perm, pos, window, spatial, multiproj)
end

@inline function _spatial_prefilter_candidates(pf::_LocatePrefilter, x::AbstractVector{<:Real})
    si = pf.spatial
    if !si.enabled || isempty(si.buckets) || length(x) < 1
        return nothing
    end
    x1 = float(x[1])
    ix = clamp(Int(floor((x1 - si.x0) / si.dx)) + 1, 1, si.nx)
    iy = 1
    if length(x) >= 2
        x2 = float(x[2])
        iy = clamp(Int(floor((x2 - si.y0) / si.dy)) + 1, 1, si.ny)
    end
    return si.buckets[(iy - 1) * si.nx + ix]
end

@inline function _spatial_prefilter_candidates_col(pf::_LocatePrefilter, X::AbstractMatrix{<:Real}, col::Int)
    si = pf.spatial
    if !si.enabled || isempty(si.buckets) || size(X, 1) < 1
        return nothing
    end
    x1 = float(X[1, col])
    ix = clamp(Int(floor((x1 - si.x0) / si.dx)) + 1, 1, si.nx)
    iy = 1
    if size(X, 1) >= 2
        x2 = float(X[2, col])
        iy = clamp(Int(floor((x2 - si.y0) / si.dy)) + 1, 1, si.ny)
    end
    return si.buckets[(iy - 1) * si.nx + ix]
end

@inline function _mp_bounds(xs::Vector{Float64}, x::Float64, window::Int)
    c = searchsortedlast(xs, x)
    lo = max(1, c - window)
    hi = min(length(xs), c + window)
    return lo, hi
end

@inline function _multiproj_ranges(mp::_MultiProjPrefilter, x::AbstractVector{<:Real})
    mp.enabled || return nothing
    nd = mp.ndims
    d1 = mp.dims[1]
    lo1, hi1 = _mp_bounds(mp.x_sorted[1], float(x[d1]), mp.window)
    if nd == 1
        return (lo1, hi1, 0, 0, 0, 0, 1)
    end
    d2 = mp.dims[2]
    lo2, hi2 = _mp_bounds(mp.x_sorted[2], float(x[d2]), mp.window)
    if nd == 2
        w1 = hi1 - lo1
        w2 = hi2 - lo2
        base = (w1 <= w2 ? 1 : 2)
        return (lo1, hi1, lo2, hi2, 0, 0, base)
    end
    d3 = mp.dims[3]
    lo3, hi3 = _mp_bounds(mp.x_sorted[3], float(x[d3]), mp.window)
    w1 = hi1 - lo1
    w2 = hi2 - lo2
    w3 = hi3 - lo3
    base = 1
    bw = w1
    if w2 < bw
        base = 2
        bw = w2
    end
    if w3 < bw
        base = 3
    end
    return (lo1, hi1, lo2, hi2, lo3, hi3, base)
end

@inline function _multiproj_ranges_col(mp::_MultiProjPrefilter, X::AbstractMatrix{<:Real}, col::Int)
    mp.enabled || return nothing
    nd = mp.ndims
    d1 = mp.dims[1]
    lo1, hi1 = _mp_bounds(mp.x_sorted[1], float(X[d1, col]), mp.window)
    if nd == 1
        return (lo1, hi1, 0, 0, 0, 0, 1)
    end
    d2 = mp.dims[2]
    lo2, hi2 = _mp_bounds(mp.x_sorted[2], float(X[d2, col]), mp.window)
    if nd == 2
        w1 = hi1 - lo1
        w2 = hi2 - lo2
        base = (w1 <= w2 ? 1 : 2)
        return (lo1, hi1, lo2, hi2, 0, 0, base)
    end
    d3 = mp.dims[3]
    lo3, hi3 = _mp_bounds(mp.x_sorted[3], float(X[d3, col]), mp.window)
    w1 = hi1 - lo1
    w2 = hi2 - lo2
    w3 = hi3 - lo3
    base = 1
    bw = w1
    if w2 < bw
        base = 2
        bw = w2
    end
    if w3 < bw
        base = 3
    end
    return (lo1, hi1, lo2, hi2, lo3, hi3, base)
end

@inline function _multiproj_accept(mp::_MultiProjPrefilter, idx::Int, base::Int,
                                   lo1::Int, hi1::Int, lo2::Int, hi2::Int, lo3::Int, hi3::Int)
    nd = mp.ndims
    if nd >= 1 && base != 1
        p1 = mp.pos[1][idx]
        (lo1 <= p1 <= hi1) || return false
    end
    if nd >= 2 && base != 2
        p2 = mp.pos[2][idx]
        (lo2 <= p2 <= hi2) || return false
    end
    if nd >= 3 && base != 3
        p3 = mp.pos[3][idx]
        (lo3 <= p3 <= hi3) || return false
    end
    return true
end

@inline function _should_use_multiproj_prefilter(pf::_LocatePrefilter, nregions::Int, nqueries::Int, threaded::Bool)
    mp = pf.multiproj
    mp.enabled || return false
    threaded || return false
    nregions >= 256 || return false
    nqueries >= 2048 || return false
    mp.ndims >= 2 || return false

    # Only prefer multi-projection when x1-window prefilter is likely weak:
    # either the dominant witness axis is not x1, or x1 spread is effectively zero.
    if mp.dims[1] != 1
        return true
    end
    isempty(pf.x1_sorted) && return false
    spread = pf.x1_sorted[end] - pf.x1_sorted[1]
    return isfinite(spread) && spread <= 1e-10
end

const _LOCATE_BUCKET_GROUPING = Ref(true)
const _LOCATE_THREAD_MIN_WORK = Ref(20_000.0)
const _LOCATE_THREAD_MIN_QUERIES = Ref(1024)
const _LOCATE_THREAD_MIN_QUERIES_N3 = Ref(100_000)
const _LOCATE_THREAD_MIN_REGIONS_N3 = Ref(512)
const _LOCATE_THREAD_MIN_CANDS_N3 = Ref(12.0)
const _LOCATE_GROUP_THREAD_MIN_QUERIES = Ref(2048)
const _LOCATE_GROUP_MIN_QUERIES = Ref(2048)
const _LOCATE_GROUP_MAX_QUERIES = Ref(100_000)
const _LOCATE_GROUP_MIN_QUERIES_PER_BUCKET = Ref(4.0)
const _LOCATE_GROUP_MAX_AVG_BUCKET_CANDS = Ref(96.0)
const _LOCATE_GROUP_MAX_CAND_DENSITY = Ref(0.35)
const _LOCATE_GROUP_MAX_UNKNOWN_FRAC = Ref(0.20)
const _LOCATE_GROUP_MIN_NONEMPTY_BUCKET_FRAC = Ref(0.05)
const _LOCATE_GROUP_MAX_BUCKET_SKEW = Ref(12.0)
const _GEOM_THREAD_MIN_ITEMS = Ref(16)
const _GEOM_THREAD_MIN_WORK_UNITS = Ref(8192)
const _GEOM_BUILD_EST_OPS = Ref(256)
const _AUTO_GEOM_NO_CACHE_MAX_REGIONS = Ref(96)
const _AUTO_GEOM_NO_CACHE_MAX_DIM = Ref(4)
const _FACET_PROBE_BATCH = Ref(true)
const _FACET_BATCH_MIN_NF_CACHED = Ref(8)
const _FACET_BATCH_MIN_NF_BOUNDARY = Ref(4)
const _FACET_BATCH_MIN_NF_ADJACENCY = Ref(8)

mutable struct _LocateBucketGroupScratch
    counts::Vector{Int}
    offsets::Vector{Int}
    writepos::Vector{Int}
    cols::Vector{Int}
end

const _LOCATE_GROUP_SCRATCH = [IdDict{Any,_LocateBucketGroupScratch}() for _ in 1:max(1, Threads.maxthreadid())]
const _LOCATE_COL_QQ_CACHE = Ref(true)
const _LOCATE_ROW_DOT_CACHE = Ref(true)

struct _LocateQQDotEntry
    row::Vector{QQ}
    value::QQ
end

mutable struct _LocateQQColScratch
    qcol::Vector{QQ}
    dot_cache::Dict{UInt64, Vector{_LocateQQDotEntry}}
end

const _LOCATE_QQCOL_SCRATCH = [Dict{Int,_LocateQQColScratch}() for _ in 1:max(1, Threads.maxthreadid())]

@inline _locate_group_key(pi::PLEncodingMap, cache) = (cache === nothing ? pi.cache : cache)

function _locate_bucket_group_scratch!(pi::PLEncodingMap, cache, nb::Int, npts::Int)
    ng = nb + 1
    store = _LOCATE_GROUP_SCRATCH[Threads.threadid()]
    key = _locate_group_key(pi, cache)
    scratch = get(store, key, nothing)
    if scratch === nothing
        scratch = _LocateBucketGroupScratch(
            zeros(Int, ng),
            zeros(Int, ng + 1),
            zeros(Int, ng),
            Vector{Int}(undef, npts),
        )
        store[key] = scratch
        return scratch
    end
    length(scratch.counts) >= ng || resize!(scratch.counts, ng)
    length(scratch.writepos) >= ng || resize!(scratch.writepos, ng)
    length(scratch.offsets) >= ng + 1 || resize!(scratch.offsets, ng + 1)
    length(scratch.cols) >= npts || resize!(scratch.cols, npts)
    return scratch
end

function _locate_qcol_scratch!(n::Int)
    store = _LOCATE_QQCOL_SCRATCH[Threads.threadid()]
    scratch = get(store, n, nothing)
    if scratch === nothing
        scratch = _LocateQQColScratch(Vector{QQ}(undef, n), Dict{UInt64, Vector{_LocateQQDotEntry}}())
        store[n] = scratch
        return scratch
    end
    length(scratch.qcol) == n || resize!(scratch.qcol, n)
    return scratch
end

@inline function _estimate_locate_candidate_width(pi::PLEncodingMap, cache)::Float64
    if cache isa PolyInBoxCache && cache.bucket_enabled
        return max(1.0, cache.bucket_avg_len)
    end
    si = pi.prefilter.spatial
    if si.enabled && !isempty(si.buckets)
        return max(1.0, si.avg_len)
    end
    if pi.prefilter.enabled
        return max(1.0, min(length(pi.regions), 2 * pi.prefilter.window + 1))
    end
    return max(1.0, length(pi.regions))
end

@inline function _locate_group_bucket_stats(pi::PLEncodingMap, cache)
    if cache isa PolyInBoxCache && cache.bucket_enabled
        nb = cache.bucket_nx * cache.bucket_ny
        return true, nb, cache.bucket_avg_len
    end
    si = pi.prefilter.spatial
    if si.enabled && !isempty(si.buckets)
        nb = si.nx * si.ny
        return true, nb, si.avg_len
    end
    return false, 0, 0.0
end

@inline function _grouped_live_bucket_ok(counts::AbstractVector{<:Integer}, nb::Int, npts::Int)::Bool
    nb > 0 || return false
    npts > 0 || return false
    unknown = Int(counts[1])
    frac_unknown = float(unknown) / npts
    frac_unknown <= _LOCATE_GROUP_MAX_UNKNOWN_FRAC[] || return false

    nonempty = 0
    load_sum = 0
    load_max = 0
    @inbounds for b in 1:nb
        c = Int(counts[b + 1])
        if c > 0
            nonempty += 1
            load_sum += c
            c > load_max && (load_max = c)
        end
    end
    frac_nonempty = float(nonempty) / nb
    frac_nonempty >= _LOCATE_GROUP_MIN_NONEMPTY_BUCKET_FRAC[] || return false
    nonempty > 0 || return false
    mean_load = float(load_sum) / nonempty
    mean_load > 0.0 || return false
    skew = load_max / mean_load
    skew <= _LOCATE_GROUP_MAX_BUCKET_SKEW[] || return false
    return true
end

@inline function _should_use_grouped_locate(pi::PLEncodingMap, cache, npts::Int)::Bool
    _LOCATE_BUCKET_GROUPING[] || return false
    npts >= _LOCATE_GROUP_MIN_QUERIES[] || return false
    max_queries = Threads.nthreads() >= 10 ? min(_LOCATE_GROUP_MAX_QUERIES[], 20_000) : _LOCATE_GROUP_MAX_QUERIES[]
    npts <= max_queries || return false
    pi.n <= 2 || return false
    ok, nb, avg_cands = _locate_group_bucket_stats(pi, cache)
    ok || return false
    nb > 0 || return false
    qpb = float(npts) / nb
    qpb >= _LOCATE_GROUP_MIN_QUERIES_PER_BUCKET[] || return false
    avg_cands >= 1.0 || return false
    avg_cands <= _LOCATE_GROUP_MAX_AVG_BUCKET_CANDS[] || return false
    dens = avg_cands / max(1, length(pi.regions))
    dens <= _LOCATE_GROUP_MAX_CAND_DENSITY[] || return false
    return true
end

@inline function _should_thread_locate_many(pi::PLEncodingMap, cache, npts::Int; grouped::Bool=false)::Bool
    Threads.nthreads() > 1 || return false
    npts > 0 || return false
    min_q = grouped ? _LOCATE_GROUP_THREAD_MIN_QUERIES[] : _LOCATE_THREAD_MIN_QUERIES[]
    npts >= min_q || return false
    nregions = length(pi.regions)
    cands = _estimate_locate_candidate_width(pi, cache)
    if pi.n >= 3
        npts >= _LOCATE_THREAD_MIN_QUERIES_N3[] || return false
        nregions >= _LOCATE_THREAD_MIN_REGIONS_N3[] || return false
        cands >= _LOCATE_THREAD_MIN_CANDS_N3[] || return false
    end
    grouped && (cands *= 0.75)
    work = float(npts) * max(1, pi.n) * cands
    return work >= _LOCATE_THREAD_MIN_WORK[]
end

@inline function _should_thread_region_loop(nitems::Int, est_ops_per_item::Int=64)::Bool
    Threads.nthreads() > 1 || return false
    nitems >= _GEOM_THREAD_MIN_ITEMS[] || return false
    return nitems * max(1, est_ops_per_item) >= _GEOM_THREAD_MIN_WORK_UNITS[]
end

@inline function _should_batch_facet_probes(kind::Symbol, nf::Int, n::Int)::Bool
    _FACET_PROBE_BATCH[] || return false
    nf > 0 || return false
    n >= 2 || return false
    if kind === :cached
        return nf >= _FACET_BATCH_MIN_NF_CACHED[]
    elseif kind === :boundary
        return nf >= _FACET_BATCH_MIN_NF_BOUNDARY[]
    else
        return nf >= _FACET_BATCH_MIN_NF_ADJACENCY[]
    end
end

@inline function _cache_bucket_id_col(cache, X::AbstractMatrix{<:Real}, col::Int)::Int
    if !cache.bucket_enabled || size(X, 1) < 1
        return 0
    end
    x1 = float(X[1, col])
    if x1 < cache.box_f[1][1] || x1 > cache.box_f[2][1]
        return 0
    end
    ix = clamp(Int(floor((x1 - cache.bucket_x0) / cache.bucket_dx)) + 1, 1, cache.bucket_nx)
    iy = 1
    if cache.pi.n >= 2
        x2 = float(X[2, col])
        if x2 < cache.box_f[1][2] || x2 > cache.box_f[2][2]
            return 0
        end
        iy = clamp(Int(floor((x2 - cache.bucket_y0) / cache.bucket_dy)) + 1, 1, cache.bucket_ny)
    end
    return (iy - 1) * cache.bucket_nx + ix
end

@inline function _spatial_bucket_id_col(pf::_LocatePrefilter, X::AbstractMatrix{<:Real}, col::Int)::Int
    si = pf.spatial
    if !si.enabled || isempty(si.buckets) || size(X, 1) < 1
        return 0
    end
    x1 = float(X[1, col])
    ix = clamp(Int(floor((x1 - si.x0) / si.dx)) + 1, 1, si.nx)
    iy = 1
    if size(X, 1) >= 2
        x2 = float(X[2, col])
        iy = clamp(Int(floor((x2 - si.y0) / si.dy)) + 1, 1, si.ny)
    end
    return (iy - 1) * si.nx + ix
end

@inline function _locate_hybrid_col_prefetched(
    pi::PLEncodingMap,
    X::AbstractMatrix{<:Real},
    col::Int,
    cands::Vector{Int};
    verify_safe::Bool=false,
    use_multiproj::Bool=false,
    tol::Float64=LOCATE_FLOAT_TOL,
    boundary_tol::Float64=LOCATE_BOUNDARY_TOL,
)
    safe_idx = 0
    qscratch = _LOCATE_COL_QQ_CACHE[] ? _locate_qcol_scratch!(size(X, 1)) : nothing
    qready = false
    @inbounds for idx in cands
        st = _hpoly_float_state_col(pi.Af[idx], pi.bf_strict[idx], X, col; tol=tol, boundary_tol=boundary_tol)
        if st == 1
            if safe_idx == 0
                safe_idx = idx
            else
                ok, qready = _in_hpoly_col_cached(pi.regions[safe_idx], X, col, qscratch, qready)
                if ok
                    return safe_idx
                end
                ok, qready = _in_hpoly_col_cached(pi.regions[idx], X, col, qscratch, qready)
                if ok
                    return idx
                end
                safe_idx = 0
            end
        elseif st == 0
            ok, qready = _in_hpoly_col_cached(pi.regions[idx], X, col, qscratch, qready)
            if ok
                return idx
            end
        end
    end
    if safe_idx != 0
        if verify_safe
            ok, qready = _in_hpoly_col_cached(pi.regions[safe_idx], X, col, qscratch, qready)
            return ok ? safe_idx : 0
        end
        return safe_idx
    end
    return _locate_hybrid_col(
        pi, X, col;
        cache=nothing,
        verify_safe=verify_safe,
        use_multiproj=use_multiproj,
        tol=tol,
        boundary_tol=boundary_tol,
    )
end

@inline function _locate_hybrid_col_prefetched(
    pi::PLEncodingMap,
    X::AbstractMatrix{<:Real},
    col::Int,
    cands::_PackedBucketSpan;
    verify_safe::Bool=false,
    use_multiproj::Bool=false,
    tol::Float64=LOCATE_FLOAT_TOL,
    boundary_tol::Float64=LOCATE_BOUNDARY_TOL,
)
    safe_idx = 0
    qscratch = _LOCATE_COL_QQ_CACHE[] ? _locate_qcol_scratch!(size(X, 1)) : nothing
    qready = false
    @inbounds for p in cands.lo:cands.hi
        idx = cands.idx[p]
        st = _hpoly_float_state_col(pi.Af[idx], pi.bf_strict[idx], X, col; tol=tol, boundary_tol=boundary_tol)
        if st == 1
            if safe_idx == 0
                safe_idx = idx
            else
                ok, qready = _in_hpoly_col_cached(pi.regions[safe_idx], X, col, qscratch, qready)
                if ok
                    return safe_idx
                end
                ok, qready = _in_hpoly_col_cached(pi.regions[idx], X, col, qscratch, qready)
                if ok
                    return idx
                end
                safe_idx = 0
            end
        elseif st == 0
            ok, qready = _in_hpoly_col_cached(pi.regions[idx], X, col, qscratch, qready)
            if ok
                return idx
            end
        end
    end
    if safe_idx != 0
        if verify_safe
            ok, qready = _in_hpoly_col_cached(pi.regions[safe_idx], X, col, qscratch, qready)
            return ok ? safe_idx : 0
        end
        return safe_idx
    end
    return _locate_hybrid_col(
        pi, X, col;
        cache=nothing,
        verify_safe=verify_safe,
        use_multiproj=use_multiproj,
        tol=tol,
        boundary_tol=boundary_tol,
    )
end

function _locate_many_grouped!(
    dest::AbstractVector{<:Integer},
    pi::PLEncodingMap,
    cache,
    X::AbstractMatrix{<:Real};
    npts::Int,
    threaded::Bool,
    verify_safe::Bool,
    use_multiproj::Bool,
    tol::Float64,
    boundary_tol::Float64,
)::Bool
    npts <= size(X, 2) || return false
    length(dest) >= npts || return false
    _should_use_grouped_locate(pi, cache, npts) || return false

    use_cache_buckets = (cache !== nothing && hasproperty(cache, :bucket_enabled) && getproperty(cache, :bucket_enabled))
    use_spatial_buckets = (!use_cache_buckets && pi.prefilter.spatial.enabled && !isempty(pi.prefilter.spatial.buckets))
    (use_cache_buckets || use_spatial_buckets) || return false

    cachep = (use_cache_buckets ? cache : nothing)
    nb = use_cache_buckets ? (cachep.bucket_nx * cachep.bucket_ny) : (pi.prefilter.spatial.nx * pi.prefilter.spatial.ny)
    nb > 0 || return false

    ng = nb + 1
    scratch = _locate_bucket_group_scratch!(pi, cachep, nb, npts)
    counts = scratch.counts
    offsets = scratch.offsets
    writepos = scratch.writepos
    cols = scratch.cols

    fill!(@view(counts[1:ng]), 0)
    @inbounds for j in 1:npts
        bid = use_cache_buckets ? _cache_bucket_id_col(cachep, X, j) : _spatial_bucket_id_col(pi.prefilter, X, j)
        counts[bid + 1] += 1
    end
    _grouped_live_bucket_ok(counts, nb, npts) || return false
    offsets[1] = 1
    @inbounds for g in 1:ng
        offsets[g + 1] = offsets[g] + counts[g]
        writepos[g] = offsets[g]
    end
    @inbounds for j in 1:npts
        bid = use_cache_buckets ? _cache_bucket_id_col(cachep, X, j) : _spatial_bucket_id_col(pi.prefilter, X, j)
        g = bid + 1
        idx = writepos[g]
        cols[idx] = j
        writepos[g] = idx + 1
    end

    empty_cands = Int[]
    do_thread = threaded && _should_thread_locate_many(pi, cache, npts; grouped=true)

    if do_thread
        Threads.@threads :static for g in 1:ng
            lo = offsets[g]
            hi = offsets[g + 1] - 1
            lo <= hi || continue
            cands = if g == 1
                empty_cands
            elseif use_cache_buckets
                cachep.bucket_regions[g - 1]
            else
                pi.prefilter.spatial.buckets[g - 1]
            end
            if g == 1 || isempty(cands)
                @inbounds for p in lo:hi
                    j = cols[p]
                    dest[j] = _locate_hybrid_col(
                        pi, X, j;
                        cache=cache,
                        verify_safe=verify_safe,
                        use_multiproj=use_multiproj,
                        tol=tol,
                        boundary_tol=boundary_tol,
                    )
                end
            else
                @inbounds for p in lo:hi
                    j = cols[p]
                    dest[j] = _locate_hybrid_col_prefetched(
                        pi, X, j, cands;
                        verify_safe=verify_safe,
                        use_multiproj=use_multiproj,
                        tol=tol,
                        boundary_tol=boundary_tol,
                    )
                end
            end
        end
    else
        @inbounds for g in 1:ng
            lo = offsets[g]
            hi = offsets[g + 1] - 1
            lo <= hi || continue
            cands = if g == 1
                empty_cands
            elseif use_cache_buckets
                cachep.bucket_regions[g - 1]
            else
                pi.prefilter.spatial.buckets[g - 1]
            end
            if g == 1 || isempty(cands)
                for p in lo:hi
                    j = cols[p]
                    dest[j] = _locate_hybrid_col(
                        pi, X, j;
                        cache=cache,
                        verify_safe=verify_safe,
                        use_multiproj=use_multiproj,
                        tol=tol,
                        boundary_tol=boundary_tol,
                    )
                end
            else
                for p in lo:hi
                    j = cols[p]
                    dest[j] = _locate_hybrid_col_prefetched(
                        pi, X, j, cands;
                        verify_safe=verify_safe,
                        use_multiproj=use_multiproj,
                        tol=tol,
                        boundary_tol=boundary_tol,
                    )
                end
            end
        end
    end
    return true
end

function PLEncodingMap(n::Int,
                       sig_y::Vector{BitVector},
                       sig_z::Vector{BitVector},
                       regions::Vector{HPoly},
                       witnesses::AbstractVector{<:Tuple})
    w = Vector{Tuple}(undef, length(witnesses))
    @inbounds for i in eachindex(witnesses)
        w[i] = witnesses[i]
    end
    nr = length(regions)
    Af = Vector{Matrix{Float64}}(undef, nr)
    bf_strict = Vector{Vector{Float64}}(undef, nr)
    bf_relaxed = Vector{Vector{Float64}}(undef, nr)
    @inbounds for t in 1:nr
        hp = regions[t]
        Af[t] = Float64.(hp.A)
        bf_strict[t] = Float64.(hp.b)
        bf_relaxed[t] = Float64.(_relaxed_b(hp))
    end
    pf = _build_locate_prefilter(n, regions, w)
    return PLEncodingMap(n, sig_y, sig_z, regions, w, Af, bf_strict, bf_relaxed, pf, EncodingCache())
end

dimension(pi::PLEncodingMap) = pi.n
representatives(pi::PLEncodingMap) = pi.witnesses

@inline function _hpoly_float_state(Af::Matrix{Float64},
                                    bf::Vector{Float64},
                                    x::Vector{Float64};
                                    tol::Float64=LOCATE_FLOAT_TOL,
                                    boundary_tol::Float64=LOCATE_BOUNDARY_TOL)::Int8
    m = size(Af, 1)
    n = size(Af, 2)
    length(x) == n || error("_hpoly_float_state: dimension mismatch")
    near_boundary = false
    fast_outside = false
    @inbounds for i in 1:m
        s = 0.0
        abs_ax = 0.0
        for j in 1:n
            aij = Af[i, j]
            xj = x[j]
            ax = aij * xj
            s += ax
            abs_ax += abs(ax)
        end
        bfi = bf[i]
        slack = bfi - s
        if slack < -tol
            fast_outside = true
        end
        guard = LOCATE_GUARD_ABS + LOCATE_GUARD_REL * (abs(bfi) + abs_ax + abs(s))
        if slack < -tol - guard
            return Int8(-1)
        end
        if slack <= boundary_tol + guard
            near_boundary = true
        end
    end
    return (near_boundary || fast_outside) ? Int8(0) : Int8(1)
end

@inline function _hpoly_float_state(Af::Matrix{Float64},
                                    bf::Vector{Float64},
                                    x::NTuple{N,Float64};
                                    tol::Float64=LOCATE_FLOAT_TOL,
                                    boundary_tol::Float64=LOCATE_BOUNDARY_TOL)::Int8 where {N}
    m = size(Af, 1)
    n = size(Af, 2)
    N == n || error("_hpoly_float_state: dimension mismatch")
    near_boundary = false
    fast_outside = false
    @inbounds for i in 1:m
        s = 0.0
        abs_ax = 0.0
        for j in 1:n
            aij = Af[i, j]
            xj = x[j]
            ax = aij * xj
            s += ax
            abs_ax += abs(ax)
        end
        bfi = bf[i]
        slack = bfi - s
        if slack < -tol
            fast_outside = true
        end
        guard = LOCATE_GUARD_ABS + LOCATE_GUARD_REL * (abs(bfi) + abs_ax + abs(s))
        if slack < -tol - guard
            return Int8(-1)
        end
        if slack <= boundary_tol + guard
            near_boundary = true
        end
    end
    return (near_boundary || fast_outside) ? Int8(0) : Int8(1)
end

@inline function _hpoly_float_state(Af::AbstractMatrix{<:Real},
                                    bf::AbstractVector{<:Real},
                                    x::AbstractVector{<:Real};
                                    tol::Float64=LOCATE_FLOAT_TOL,
                                    boundary_tol::Float64=LOCATE_BOUNDARY_TOL)::Int8
    m = size(Af, 1)
    n = size(Af, 2)
    length(x) == n || error("_hpoly_float_state: dimension mismatch")
    near_boundary = false
    fast_outside = false
    @inbounds for i in 1:m
        s = 0.0
        abs_ax = 0.0
        for j in 1:n
            ax = float(Af[i, j]) * float(x[j])
            s += ax
            abs_ax += abs(ax)
        end
        slack = float(bf[i]) - s
        # Stage 1: very cheap float prefilter.
        if slack < -tol
            fast_outside = true
        end
        # Stage 2: guard-band check; only strongly outside rows are rejected.
        guard = LOCATE_GUARD_ABS + LOCATE_GUARD_REL * (abs(float(bf[i])) + abs_ax + abs(s))
        if slack < -tol - guard
            return Int8(-1)
        end
        if slack <= boundary_tol + guard
            near_boundary = true
        end
    end
    return (near_boundary || fast_outside) ? Int8(0) : Int8(1)
end

@inline function _hpoly_float_state(Af::AbstractMatrix{<:Real},
                                    bf::AbstractVector{<:Real},
                                    x::NTuple{N,<:Real};
                                    tol::Float64=LOCATE_FLOAT_TOL,
                                    boundary_tol::Float64=LOCATE_BOUNDARY_TOL) where {N}
    m = size(Af, 1)
    n = size(Af, 2)
    N == n || error("_hpoly_float_state: dimension mismatch")
    near_boundary = false
    fast_outside = false
    @inbounds for i in 1:m
        s = 0.0
        abs_ax = 0.0
        for j in 1:n
            ax = float(Af[i, j]) * float(x[j])
            s += ax
            abs_ax += abs(ax)
        end
        slack = float(bf[i]) - s
        if slack < -tol
            fast_outside = true
        end
        guard = LOCATE_GUARD_ABS + LOCATE_GUARD_REL * (abs(float(bf[i])) + abs_ax + abs(s))
        if slack < -tol - guard
            return Int8(-1)
        end
        if slack <= boundary_tol + guard
            near_boundary = true
        end
    end
    return (near_boundary || fast_outside) ? Int8(0) : Int8(1)
end

@inline function _hpoly_float_state_col(Af::Matrix{Float64},
                                        bf::Vector{Float64},
                                        X::Matrix{Float64}, col::Int;
                                        tol::Float64=LOCATE_FLOAT_TOL,
                                        boundary_tol::Float64=LOCATE_BOUNDARY_TOL)::Int8
    m = size(Af, 1)
    n = size(Af, 2)
    size(X, 1) == n || error("_hpoly_float_state_col: dimension mismatch")
    near_boundary = false
    fast_outside = false
    @inbounds for i in 1:m
        s = 0.0
        abs_ax = 0.0
        for j in 1:n
            aij = Af[i, j]
            xj = X[j, col]
            ax = aij * xj
            s += ax
            abs_ax += abs(ax)
        end
        bfi = bf[i]
        slack = bfi - s
        if slack < -tol
            fast_outside = true
        end
        guard = LOCATE_GUARD_ABS + LOCATE_GUARD_REL * (abs(bfi) + abs_ax + abs(s))
        if slack < -tol - guard
            return Int8(-1)
        end
        if slack <= boundary_tol + guard
            near_boundary = true
        end
    end
    return (near_boundary || fast_outside) ? Int8(0) : Int8(1)
end

@inline function _hpoly_float_state_col(Af::AbstractMatrix{<:Real},
                                        bf::AbstractVector{<:Real},
                                        X::AbstractMatrix{<:Real}, col::Int;
                                        tol::Float64=LOCATE_FLOAT_TOL,
                                        boundary_tol::Float64=LOCATE_BOUNDARY_TOL)::Int8
    m = size(Af, 1)
    n = size(Af, 2)
    size(X, 1) == n || error("_hpoly_float_state_col: dimension mismatch")
    near_boundary = false
    fast_outside = false
    @inbounds for i in 1:m
        s = 0.0
        abs_ax = 0.0
        for j in 1:n
            ax = float(Af[i, j]) * float(X[j, col])
            s += ax
            abs_ax += abs(ax)
        end
        slack = float(bf[i]) - s
        if slack < -tol
            fast_outside = true
        end
        guard = LOCATE_GUARD_ABS + LOCATE_GUARD_REL * (abs(float(bf[i])) + abs_ax + abs(s))
        if slack < -tol - guard
            return Int8(-1)
        end
        if slack <= boundary_tol + guard
            near_boundary = true
        end
    end
    return (near_boundary || fast_outside) ? Int8(0) : Int8(1)
end

@inline function _in_hpoly_col(h::HPoly, X::AbstractMatrix{<:Real}, col::Int)
    size(X, 1) == h.n || error("_in_hpoly_col: dimension mismatch")
    m = size(h.A, 1)
    @inbounds for i in 1:m
        s = zero(QQ)
        for j in 1:h.n
            s += h.A[i, j] * _toQQ(X[j, col])
        end
        if s > h.b[i]
            return false
        end
    end
    return true
end

@inline function _fill_qcol!(qcol::Vector{QQ}, X::AbstractMatrix{<:Real}, col::Int)
    @inbounds for j in 1:size(X, 1)
        qcol[j] = _toQQ(X[j, col])
    end
    return qcol
end

@inline function _row_coeff_hash(A::AbstractMatrix{QQ}, i::Int)
    h = hash(size(A, 2))
    @inbounds for j in 1:size(A, 2)
        h = hash(A[i, j], h)
    end
    return UInt64(h)
end

@inline function _row_coeffs_equal(row::Vector{QQ}, A::AbstractMatrix{QQ}, i::Int)
    length(row) == size(A, 2) || return false
    @inbounds for j in 1:size(A, 2)
        row[j] == A[i, j] || return false
    end
    return true
end

@inline function _row_dot_qcol(A::AbstractMatrix{QQ}, i::Int, qcol::AbstractVector{QQ})
    s = zero(QQ)
    @inbounds for j in 1:size(A, 2)
        s += A[i, j] * qcol[j]
    end
    return s
end

function _row_dot_qcol_cached!(scratch::_LocateQQColScratch, A::AbstractMatrix{QQ}, i::Int)
    if !_LOCATE_ROW_DOT_CACHE[]
        return _row_dot_qcol(A, i, scratch.qcol)
    end
    key = _row_coeff_hash(A, i)
    bucket = get!(scratch.dot_cache, key, Vector{_LocateQQDotEntry}())
    @inbounds for entry in bucket
        if _row_coeffs_equal(entry.row, A, i)
            return entry.value
        end
    end
    n = size(A, 2)
    row = Vector{QQ}(undef, n)
    s = zero(QQ)
    @inbounds for j in 1:n
        aij = A[i, j]
        row[j] = aij
        s += aij * scratch.qcol[j]
    end
    push!(bucket, _LocateQQDotEntry(row, s))
    return s
end

@inline function _in_hpoly_qcol(h::HPoly, qcol::AbstractVector{QQ})
    length(qcol) == h.n || error("_in_hpoly_qcol: dimension mismatch")
    m = size(h.A, 1)
    @inbounds for i in 1:m
        s = zero(QQ)
        for j in 1:h.n
            s += h.A[i, j] * qcol[j]
        end
        if s > h.b[i]
            return false
        end
    end
    return true
end

@inline function _in_hpoly_qcol_cached(h::HPoly, scratch::_LocateQQColScratch)
    length(scratch.qcol) == h.n || error("_in_hpoly_qcol_cached: dimension mismatch")
    m = size(h.A, 1)
    @inbounds for i in 1:m
        if _row_dot_qcol_cached!(scratch, h.A, i) > h.b[i]
            return false
        end
    end
    return true
end

@inline function _in_hpoly_col_cached(h::HPoly,
                                      X::AbstractMatrix{<:Real},
                                      col::Int,
                                      scratch,
                                      qready::Bool)
    if !_LOCATE_COL_QQ_CACHE[]
        return _in_hpoly_col(h, X, col), qready
    end
    if !qready
        _fill_qcol!(scratch.qcol, X, col)
        empty!(scratch.dot_cache)
        qready = true
    end
    return _in_hpoly_qcol_cached(h, scratch), qready
end

@inline function _locate_hybrid(pi::PLEncodingMap, x;
                                cache=nothing,
                                verify_safe::Bool=false,
                                use_multiproj::Bool=false,
                                tol::Float64=LOCATE_FLOAT_TOL,
                                boundary_tol::Float64=LOCATE_BOUNDARY_TOL)
    nregions = length(pi.regions)
    safe_idx = 0
    cands = cache isa PolyInBoxCache ? _bucket_candidates(cache, x) : nothing
    if cands !== nothing && !isempty(cands)
        @inbounds for idx in cands
            st = _hpoly_float_state(pi.Af[idx], pi.bf_strict[idx], x; tol=tol, boundary_tol=boundary_tol)
            if st == 1
                if safe_idx == 0
                    safe_idx = idx
                else
                    if _in_hpoly(pi.regions[safe_idx], x)
                        return safe_idx
                    end
                    if _in_hpoly(pi.regions[idx], x)
                        return idx
                    end
                    safe_idx = 0
                end
            elseif st == 0
                if _in_hpoly(pi.regions[idx], x)
                    return idx
                end
            end
        end
        if safe_idx != 0
            return verify_safe ? (_in_hpoly(pi.regions[safe_idx], x) ? safe_idx : 0) : safe_idx
        end
    end

    cands = _spatial_prefilter_candidates(pi.prefilter, x)
    if cands !== nothing && !isempty(cands)
        @inbounds for idx in cands
            st = _hpoly_float_state(pi.Af[idx], pi.bf_strict[idx], x; tol=tol, boundary_tol=boundary_tol)
            if st == 1
                if safe_idx == 0
                    safe_idx = idx
                else
                    if _in_hpoly(pi.regions[safe_idx], x)
                        return safe_idx
                    end
                    if _in_hpoly(pi.regions[idx], x)
                        return idx
                    end
                    safe_idx = 0
                end
            elseif st == 0
                if _in_hpoly(pi.regions[idx], x)
                    return idx
                end
            end
        end
        if safe_idx != 0
            return verify_safe ? (_in_hpoly(pi.regions[safe_idx], x) ? safe_idx : 0) : safe_idx
        end
    end

    mp = pi.prefilter.multiproj
    if use_multiproj && mp.enabled
        rngs = _multiproj_ranges(mp, x)
        if rngs !== nothing
            lo1, hi1, lo2, hi2, lo3, hi3, base = rngs
            pbase = mp.perm[base]
            lo = (base == 1 ? lo1 : (base == 2 ? lo2 : lo3))
            hi = (base == 1 ? hi1 : (base == 2 ? hi2 : hi3))
            base_len = hi - lo + 1
            prefer_len = (pi.prefilter.enabled ? pi.prefilter.window : 64)
            if base_len <= prefer_len
                @inbounds for k in lo:hi
                    idx = pbase[k]
                    _multiproj_accept(mp, idx, base, lo1, hi1, lo2, hi2, lo3, hi3) || continue
                    st = _hpoly_float_state(pi.Af[idx], pi.bf_strict[idx], x; tol=tol, boundary_tol=boundary_tol)
                    if st == 1
                        if safe_idx == 0
                            safe_idx = idx
                        else
                            if _in_hpoly(pi.regions[safe_idx], x)
                                return safe_idx
                            end
                            if _in_hpoly(pi.regions[idx], x)
                                return idx
                            end
                            safe_idx = 0
                        end
                    elseif st == 0
                        if _in_hpoly(pi.regions[idx], x)
                            return idx
                        end
                    end
                end
                if safe_idx != 0
                    return verify_safe ? (_in_hpoly(pi.regions[safe_idx], x) ? safe_idx : 0) : safe_idx
                end
            end
        end
    end

    if pi.prefilter.enabled
        x1 = float(x[1])
        c = searchsortedlast(pi.prefilter.x1_sorted, x1)
        lo = max(1, c - pi.prefilter.window)
        hi = min(nregions, c + pi.prefilter.window)
        @inbounds for k in lo:hi
            idx = pi.prefilter.perm[k]
            st = _hpoly_float_state(pi.Af[idx], pi.bf_strict[idx], x; tol=tol, boundary_tol=boundary_tol)
            if st == 1
                if safe_idx == 0
                    safe_idx = idx
                else
                    if _in_hpoly(pi.regions[safe_idx], x)
                        return safe_idx
                    end
                    if _in_hpoly(pi.regions[idx], x)
                        return idx
                    end
                    safe_idx = 0
                end
            elseif st == 0
                if _in_hpoly(pi.regions[idx], x)
                    return idx
                end
            end
        end
        if safe_idx != 0
            return verify_safe ? (_in_hpoly(pi.regions[safe_idx], x) ? safe_idx : 0) : safe_idx
        end

        @inbounds for idx in 1:nregions
            p = pi.prefilter.pos[idx]
            if p >= lo && p <= hi
                continue
            end
            st = _hpoly_float_state(pi.Af[idx], pi.bf_strict[idx], x; tol=tol, boundary_tol=boundary_tol)
            if st == 1
                if safe_idx == 0
                    safe_idx = idx
                else
                    if _in_hpoly(pi.regions[safe_idx], x)
                        return safe_idx
                    end
                    if _in_hpoly(pi.regions[idx], x)
                        return idx
                    end
                    safe_idx = 0
                end
            elseif st == 0
                if _in_hpoly(pi.regions[idx], x)
                    return idx
                end
            end
        end
        if safe_idx == 0
            return 0
        end
        return verify_safe ? (_in_hpoly(pi.regions[safe_idx], x) ? safe_idx : 0) : safe_idx
    end

    @inbounds for idx in 1:nregions
        st = _hpoly_float_state(pi.Af[idx], pi.bf_strict[idx], x; tol=tol, boundary_tol=boundary_tol)
        if st == 1
            if safe_idx == 0
                safe_idx = idx
            else
                if _in_hpoly(pi.regions[safe_idx], x)
                    return safe_idx
                end
                if _in_hpoly(pi.regions[idx], x)
                    return idx
                end
                safe_idx = 0
            end
        elseif st == 0
            if _in_hpoly(pi.regions[idx], x)
                return idx
            end
        end
    end
    if safe_idx == 0
        return 0
    end
    return verify_safe ? (_in_hpoly(pi.regions[safe_idx], x) ? safe_idx : 0) : safe_idx
end

@inline function _locate_hybrid_col(pi::PLEncodingMap, X::AbstractMatrix{<:Real}, col::Int;
                                    cache=nothing,
                                    verify_safe::Bool=false,
                                    use_multiproj::Bool=false,
                                    tol::Float64=LOCATE_FLOAT_TOL,
                                    boundary_tol::Float64=LOCATE_BOUNDARY_TOL)
    nregions = length(pi.regions)
    safe_idx = 0
    qscratch = _LOCATE_COL_QQ_CACHE[] ? _locate_qcol_scratch!(size(X, 1)) : nothing
    qready = false
    cands = cache isa PolyInBoxCache ? _bucket_candidates_col(cache, X, col) : nothing
    if cands !== nothing && !isempty(cands)
        @inbounds for idx in cands
            st = _hpoly_float_state_col(pi.Af[idx], pi.bf_strict[idx], X, col; tol=tol, boundary_tol=boundary_tol)
            if st == 1
                if safe_idx == 0
                    safe_idx = idx
                else
                    ok, qready = _in_hpoly_col_cached(pi.regions[safe_idx], X, col, qscratch, qready)
                    if ok
                        return safe_idx
                    end
                    ok, qready = _in_hpoly_col_cached(pi.regions[idx], X, col, qscratch, qready)
                    if ok
                        return idx
                    end
                    safe_idx = 0
                end
            elseif st == 0
                ok, qready = _in_hpoly_col_cached(pi.regions[idx], X, col, qscratch, qready)
                if ok
                    return idx
                end
            end
        end
        if safe_idx != 0
            if verify_safe
                ok, qready = _in_hpoly_col_cached(pi.regions[safe_idx], X, col, qscratch, qready)
                return ok ? safe_idx : 0
            end
            return safe_idx
        end
    end

    cands = _spatial_prefilter_candidates_col(pi.prefilter, X, col)
    if cands !== nothing && !isempty(cands)
        @inbounds for idx in cands
            st = _hpoly_float_state_col(pi.Af[idx], pi.bf_strict[idx], X, col; tol=tol, boundary_tol=boundary_tol)
            if st == 1
                if safe_idx == 0
                    safe_idx = idx
                else
                    ok, qready = _in_hpoly_col_cached(pi.regions[safe_idx], X, col, qscratch, qready)
                    if ok
                        return safe_idx
                    end
                    ok, qready = _in_hpoly_col_cached(pi.regions[idx], X, col, qscratch, qready)
                    if ok
                        return idx
                    end
                    safe_idx = 0
                end
            elseif st == 0
                ok, qready = _in_hpoly_col_cached(pi.regions[idx], X, col, qscratch, qready)
                if ok
                    return idx
                end
            end
        end
        if safe_idx != 0
            if verify_safe
                ok, qready = _in_hpoly_col_cached(pi.regions[safe_idx], X, col, qscratch, qready)
                return ok ? safe_idx : 0
            end
            return safe_idx
        end
    end

    mp = pi.prefilter.multiproj
    if use_multiproj && mp.enabled
        rngs = _multiproj_ranges_col(mp, X, col)
        if rngs !== nothing
            lo1, hi1, lo2, hi2, lo3, hi3, base = rngs
            pbase = mp.perm[base]
            lo = (base == 1 ? lo1 : (base == 2 ? lo2 : lo3))
            hi = (base == 1 ? hi1 : (base == 2 ? hi2 : hi3))
            base_len = hi - lo + 1
            prefer_len = (pi.prefilter.enabled ? pi.prefilter.window : 64)
            if base_len <= prefer_len
                @inbounds for k in lo:hi
                    idx = pbase[k]
                    _multiproj_accept(mp, idx, base, lo1, hi1, lo2, hi2, lo3, hi3) || continue
                    st = _hpoly_float_state_col(pi.Af[idx], pi.bf_strict[idx], X, col; tol=tol, boundary_tol=boundary_tol)
                    if st == 1
                        if safe_idx == 0
                            safe_idx = idx
                        else
                            ok, qready = _in_hpoly_col_cached(pi.regions[safe_idx], X, col, qscratch, qready)
                            if ok
                                return safe_idx
                            end
                            ok, qready = _in_hpoly_col_cached(pi.regions[idx], X, col, qscratch, qready)
                            if ok
                                return idx
                            end
                            safe_idx = 0
                        end
                    elseif st == 0
                        ok, qready = _in_hpoly_col_cached(pi.regions[idx], X, col, qscratch, qready)
                        if ok
                            return idx
                        end
                    end
                end
                if safe_idx != 0
                    if verify_safe
                        ok, qready = _in_hpoly_col_cached(pi.regions[safe_idx], X, col, qscratch, qready)
                        return ok ? safe_idx : 0
                    end
                    return safe_idx
                end
            end
        end
    end

    if pi.prefilter.enabled
        x1 = float(X[1, col])
        c = searchsortedlast(pi.prefilter.x1_sorted, x1)
        lo = max(1, c - pi.prefilter.window)
        hi = min(nregions, c + pi.prefilter.window)
        @inbounds for k in lo:hi
            idx = pi.prefilter.perm[k]
            st = _hpoly_float_state_col(pi.Af[idx], pi.bf_strict[idx], X, col; tol=tol, boundary_tol=boundary_tol)
            if st == 1
                if safe_idx == 0
                    safe_idx = idx
                else
                    ok, qready = _in_hpoly_col_cached(pi.regions[safe_idx], X, col, qscratch, qready)
                    if ok
                        return safe_idx
                    end
                    ok, qready = _in_hpoly_col_cached(pi.regions[idx], X, col, qscratch, qready)
                    if ok
                        return idx
                    end
                    safe_idx = 0
                end
            elseif st == 0
                ok, qready = _in_hpoly_col_cached(pi.regions[idx], X, col, qscratch, qready)
                if ok
                    return idx
                end
            end
        end
        if safe_idx != 0
            if verify_safe
                ok, qready = _in_hpoly_col_cached(pi.regions[safe_idx], X, col, qscratch, qready)
                return ok ? safe_idx : 0
            end
            return safe_idx
        end

        @inbounds for idx in 1:nregions
            p = pi.prefilter.pos[idx]
            if p >= lo && p <= hi
                continue
            end
            st = _hpoly_float_state_col(pi.Af[idx], pi.bf_strict[idx], X, col; tol=tol, boundary_tol=boundary_tol)
            if st == 1
                if safe_idx == 0
                    safe_idx = idx
                else
                    ok, qready = _in_hpoly_col_cached(pi.regions[safe_idx], X, col, qscratch, qready)
                    if ok
                        return safe_idx
                    end
                    ok, qready = _in_hpoly_col_cached(pi.regions[idx], X, col, qscratch, qready)
                    if ok
                        return idx
                    end
                    safe_idx = 0
                end
            elseif st == 0
                ok, qready = _in_hpoly_col_cached(pi.regions[idx], X, col, qscratch, qready)
                if ok
                    return idx
                end
            end
        end
        if safe_idx == 0
            return 0
        end
        if verify_safe
            ok, qready = _in_hpoly_col_cached(pi.regions[safe_idx], X, col, qscratch, qready)
            return ok ? safe_idx : 0
        end
        return safe_idx
    end

    @inbounds for idx in 1:nregions
        st = _hpoly_float_state_col(pi.Af[idx], pi.bf_strict[idx], X, col; tol=tol, boundary_tol=boundary_tol)
        if st == 1
            if safe_idx == 0
                safe_idx = idx
            else
                ok, qready = _in_hpoly_col_cached(pi.regions[safe_idx], X, col, qscratch, qready)
                if ok
                    return safe_idx
                end
                ok, qready = _in_hpoly_col_cached(pi.regions[idx], X, col, qscratch, qready)
                if ok
                    return idx
                end
                safe_idx = 0
            end
        elseif st == 0
            ok, qready = _in_hpoly_col_cached(pi.regions[idx], X, col, qscratch, qready)
            if ok
                return idx
            end
        end
    end
    if safe_idx == 0
        return 0
    end
    if verify_safe
        ok, qready = _in_hpoly_col_cached(pi.regions[safe_idx], X, col, qscratch, qready)
        return ok ? safe_idx : 0
    end
    return safe_idx
end

"""
    locate(pi, x; mode=:fast, tol=LOCATE_FLOAT_TOL, boundary_tol=LOCATE_BOUNDARY_TOL) -> Int

Locate a point `x` in the finite region decomposition carried by a PL encoding
object.

Mathematical meaning:
- returns the region id `r > 0` whose polyhedral cell contains `x`;
- returns `0` when `x` lies outside the represented region set, or when a
  verified query rejects a candidate near the boundary.

Cheap/default path:
- `mode=:fast` is the default and uses the owner-specific prefilter machinery
  (including spatial lookup when available) before exact H-poly membership.
- `mode=:verified` performs an additional safe-region check and is the right
  choice when boundary sensitivity matters more than raw speed.

Cache behavior:
- passing a [`PolyInBoxCache`](@ref) reuses bucket and clipped-geometry data for
  repeated queries in a fixed window;
- compiled encodings forward to the same kernel and can reuse attached
  `EncodingCache` state in downstream geometry workflows.
"""
function locate(pi::PLEncodingMap, x::AbstractVector;
                mode::Symbol=:fast,
                tol::Float64=LOCATE_FLOAT_TOL,
                boundary_tol::Float64=LOCATE_BOUNDARY_TOL)
    length(x) == pi.n || error("locate: dimension mismatch")
    mode0 = validate_pl_mode(mode)
    return _locate_hybrid(pi, x; verify_safe=(mode0 === :verified), tol=tol, boundary_tol=boundary_tol)
end

function locate(pi::PLEncodingMap, x::NTuple{N,<:Real};
                mode::Symbol=:fast,
                tol::Float64=LOCATE_FLOAT_TOL,
                boundary_tol::Float64=LOCATE_BOUNDARY_TOL) where {N}
    N == pi.n || error("locate: dimension mismatch")
    mode0 = validate_pl_mode(mode)
    return _locate_hybrid(pi, x; verify_safe=(mode0 === :verified), tol=tol, boundary_tol=boundary_tol)
end

function locate(cache, x::AbstractVector;
                mode::Symbol=:fast,
                tol::Float64=LOCATE_FLOAT_TOL,
                boundary_tol::Float64=LOCATE_BOUNDARY_TOL)
    cache isa PolyInBoxCache || error("locate(cache, x): cache must be PolyInBoxCache")
    mode0 = validate_pl_mode(mode)
    return _locate_hybrid(cache.pi, x; cache=cache, verify_safe=(mode0 === :verified),
                          tol=tol, boundary_tol=boundary_tol)
end

function locate(cache, x::NTuple{N,<:Real};
                mode::Symbol=:fast,
                tol::Float64=LOCATE_FLOAT_TOL,
                boundary_tol::Float64=LOCATE_BOUNDARY_TOL) where {N}
    cache isa PolyInBoxCache || error("locate(cache, x): cache must be PolyInBoxCache")
    mode0 = validate_pl_mode(mode)
    return _locate_hybrid(cache.pi, x; cache=cache, verify_safe=(mode0 === :verified),
                          tol=tol, boundary_tol=boundary_tol)
end

"""
    locate_many!(dest, pi_or_cache, X; threaded=true, mode=:fast,
                 tol=LOCATE_FLOAT_TOL, boundary_tol=LOCATE_BOUNDARY_TOL)

Fill `dest` with region ids for the column-batch of query points in `X`.

Mathematical meaning:
- `X` has shape `(n, npoints)`, with one query point per column;
- `dest[j]` is the region id of column `j`, or `0` when that point lies
  outside the represented region set (or fails verification near the boundary).

Cheap/default path:
- pass a bare [`PLEncodingMap`](@ref) for one-off or light workloads;
- pass a [`PolyInBoxCache`](@ref) when many queries share one box and you want
  bucketed lookup and cached geometry state to be reused.

The implementation chooses between scalar, grouped, and threaded paths
internally, but the mathematical contract is unchanged across those routes.
"""
function _locate_many_pl!(dest::AbstractVector{<:Integer}, pi_or_cache, X::AbstractMatrix{<:Real};
                          threaded::Bool=true,
                          mode::Symbol=:fast,
                          tol::Float64=LOCATE_FLOAT_TOL,
                          boundary_tol::Float64=LOCATE_BOUNDARY_TOL)
    cache = pi_or_cache isa PolyInBoxCache ? pi_or_cache : nothing
    pi = pi_or_cache isa PLEncodingMap ? pi_or_cache :
         (cache === nothing ? (hasproperty(pi_or_cache, :pi) ? getproperty(pi_or_cache, :pi) : nothing) : cache.pi)
    pi isa PLEncodingMap || error("locate_many!: expected PLEncodingMap or PolyInBoxCache")
    size(X, 1) == pi.n || error("locate_many!: X must have size (n, npoints) with n=$(pi.n)")
    length(dest) == size(X, 2) || error("locate_many!: destination length mismatch")

    npts = size(X, 2)
    mode0 = validate_pl_mode(mode)
    verify_safe = (mode0 === :verified)
    do_thread = threaded && _should_thread_locate_many(pi, cache, npts; grouped=false)
    use_multiproj = _should_use_multiproj_prefilter(
        pi.prefilter,
        length(pi.regions),
        npts,
        do_thread,
    )
    grouped_ok = _should_use_grouped_locate(pi, cache, npts)
    if grouped_ok && _locate_many_grouped!(
        dest, pi, cache, X;
        npts=npts,
        threaded=do_thread,
        verify_safe=verify_safe,
        use_multiproj=use_multiproj,
        tol=tol,
        boundary_tol=boundary_tol,
    )
        return dest
    end
    if do_thread
        Threads.@threads :static for j in 1:npts
            dest[j] = _locate_hybrid_col(
                pi, X, j;
                cache=cache,
                verify_safe=verify_safe,
                use_multiproj=use_multiproj,
                tol=tol,
                boundary_tol=boundary_tol,
            )
        end
    else
        @inbounds for j in 1:npts
            dest[j] = _locate_hybrid_col(
                pi, X, j;
                cache=cache,
                verify_safe=verify_safe,
                use_multiproj=use_multiproj,
                tol=tol,
                boundary_tol=boundary_tol,
            )
        end
    end
    return dest
end

function locate_many!(dest::AbstractVector{<:Integer}, pi_or_cache, X::AbstractMatrix{<:Real};
                      threaded::Bool=true,
                      mode::Symbol=:fast,
                      tol::Float64=LOCATE_FLOAT_TOL,
                      boundary_tol::Float64=LOCATE_BOUNDARY_TOL)
    return _locate_many_pl!(dest, pi_or_cache, X;
                            threaded=threaded,
                            mode=mode,
                            tol=tol,
                            boundary_tol=boundary_tol)
end

function locate_many!(dest::AbstractVector{<:Integer},
                      pi::PLEncodingMap,
                      X::AbstractMatrix{<:AbstractFloat};
                      threaded::Bool=true,
                      mode::Symbol=:fast,
                      tol::Float64=LOCATE_FLOAT_TOL,
                      boundary_tol::Float64=LOCATE_BOUNDARY_TOL)
    return _locate_many_pl!(dest, pi, X;
                            threaded=threaded,
                            mode=mode,
                            tol=tol,
                            boundary_tol=boundary_tol)
end

function locate_many(pi_or_cache, X::AbstractMatrix{<:Real};
                     threaded::Bool=true,
                     mode::Symbol=:fast,
                     tol::Float64=LOCATE_FLOAT_TOL,
                     boundary_tol::Float64=LOCATE_BOUNDARY_TOL)
    dest = Vector{Int}(undef, size(X, 2))
    locate_many!(dest, pi_or_cache, X; threaded=threaded, mode=mode, tol=tol, boundary_tol=boundary_tol)
    return dest
end

@inline function _locate_many_prefix!(dest::Vector{Int}, pi_or_cache, X::Matrix{Float64}, npts::Int;
                                      threaded::Bool=true,
                                      mode::Symbol=:fast,
                                      tol::Float64=LOCATE_FLOAT_TOL,
                                      boundary_tol::Float64=LOCATE_BOUNDARY_TOL)
    npts < 0 && error("_locate_many_prefix!: npts must be nonnegative")
    npts <= size(X, 2) || error("_locate_many_prefix!: npts exceeds number of columns")
    length(dest) >= npts || error("_locate_many_prefix!: destination too short")
    cache = pi_or_cache isa PolyInBoxCache ? pi_or_cache : nothing
    pi = pi_or_cache isa PLEncodingMap ? pi_or_cache :
         (cache === nothing ? (hasproperty(pi_or_cache, :pi) ? getproperty(pi_or_cache, :pi) : nothing) : cache.pi)
    pi isa PLEncodingMap || error("_locate_many_prefix!: expected PLEncodingMap or PolyInBoxCache")
    size(X, 1) == pi.n || error("_locate_many_prefix!: X must have size (n, npoints) with n=$(pi.n)")
    mode0 = validate_pl_mode(mode)
    verify_safe = (mode0 === :verified)
    do_thread = threaded && _should_thread_locate_many(pi, cache, npts; grouped=false)
    use_multiproj = _should_use_multiproj_prefilter(
        pi.prefilter,
        length(pi.regions),
        npts,
        do_thread,
    )
    grouped_ok = _should_use_grouped_locate(pi, cache, npts)
    if grouped_ok && _locate_many_grouped!(
        dest, pi, cache, X;
        npts=npts,
        threaded=do_thread,
        verify_safe=verify_safe,
        use_multiproj=use_multiproj,
        tol=tol,
        boundary_tol=boundary_tol,
    )
        return dest
    end
    if do_thread
        Threads.@threads :static for j in 1:npts
            dest[j] = _locate_hybrid_col(
                pi, X, j;
                cache=cache,
                verify_safe=verify_safe,
                use_multiproj=use_multiproj,
                tol=tol,
                boundary_tol=boundary_tol,
            )
        end
    else
        @inbounds for j in 1:npts
            dest[j] = _locate_hybrid_col(
                pi, X, j;
                cache=cache,
                verify_safe=verify_safe,
                use_multiproj=use_multiproj,
                tol=tol,
                boundary_tol=boundary_tol,
            )
        end
    end
    return dest
end

# ------------------------- Region geometry / sizes ----------------------------

# Fast (floating) membership check used for Monte Carlo geometry.
# For sampling-based geometry, exact QQ membership is overkill and can be slow,
# because it must rationalize every sampled Float64 coordinate.
@inline function _in_hpoly_float(Af::Matrix{Float64}, bf::Vector{Float64},
                                 x::Vector{Float64}; tol::Float64=1e-12,
                                 closure::Bool=true)::Bool
    m = size(Af, 1)
    n = size(Af, 2)
    length(x) == n || error("_in_hpoly_float: dimension mismatch")
    @inbounds for i in 1:m
        s = 0.0
        for j in 1:n
            s = muladd(Af[i, j], x[j], s)
        end
        bfi = bf[i]
        if closure
            if s > bfi + tol
                return false
            end
        else
            if s >= bfi - tol
                return false
            end
        end
    end
    return true
end

function _in_hpoly_float(Af::AbstractMatrix{<:Real}, bf::AbstractVector{<:Real},
                         x::AbstractVector{<:Real}; tol::Float64=1e-12,
                         closure::Bool=true)::Bool
    m = size(Af, 1)
    n = size(Af, 2)
    length(x) == n || error("_in_hpoly_float: dimension mismatch")
    for i in 1:m
        s = 0.0
        for j in 1:n
            s += float(Af[i, j]) * float(x[j])
        end
        bfi = float(bf[i])
        if closure
            if s > bfi + tol
                return false
            end
        else
            if s >= bfi - tol
                return false
            end
        end
    end
    return true
end

# ---- Polyhedra-in-box caching ------------------------------------------------

"""
    PolyInBoxCache

Cache object produced by `poly_in_box_cache`.

This cache is intended to accelerate repeated Polyhedra-based geometry calls
that all use the same encoding map and the same window `box=(a,b)`.
It stores (lazily) the polyhedra:
    P_r = region(r) cap box
and (also lazily) their V- and H-representations.

Users should treat the fields as internal. Construct with `poly_in_box_cache`
and pass via the `cache=` keyword to geometry routines.
"""
struct CachedFacet
    measure::Float64
    normal::Vector{Float64}
    unit::Vector{Float64}
    point::Vector{Float64}
end

const BoundaryBreakdownEntry = NamedTuple{
    (:measure, :normal, :point, :kind, :neighbor),
    Tuple{Float64, Vector{Float64}, Vector{Float64}, Symbol, Union{Nothing, Int}}
}
const BoundaryBreakdownCacheKey = Tuple{Int, Bool, Symbol, Float64, Float64}
const AdjacencyCacheKey = Tuple{Bool, Symbol, Float64, Float64}

struct _HRepFloatCache
    A::Matrix{Float64}
    b::Vector{Float64}
end

mutable struct _FacetClassifyScratch
    X::Matrix{Float64}
    in_box::BitVector
    loc::Vector{Int}
end

struct _AdjEdgeAcc
    r::Int
    s::Int
    m::Float64
end

mutable struct PolyInBoxCache{PolyT,VRepT,HRepT}
    pi::PLEncodingMap
    box_q::Tuple{Vector{QQ},Vector{QQ}}
    box_f::Tuple{Vector{Float64},Vector{Float64}}
    closure::Bool
    level::Symbol
    lock::Base.ReentrantLock
    # Precompiled float membership data for repeated locate/geometry probes.
    Af::Vector{Matrix{Float64}}
    bf_strict::Vector{Vector{Float64}}
    bf_relaxed::Vector{Vector{Float64}}
    # Lazy activity/materialization state per region:
    #   0 => unknown, 1 => active in the query box, -1 => empty intersection.
    activity_state::Vector{Int8}
    activity_scanned::Bool
    # Per-region AABBs in the cached box.
    aabb_lo::Vector{Vector{Float64}}
    aabb_hi::Vector{Vector{Float64}}
    active_regions::Vector{Int}
    active_mask::BitVector
    active_index::Vector{Int}
    # 2D spatial bucket index over first two coordinates.
    bucket_enabled::Bool
    bucket_nx::Int
    bucket_ny::Int
    bucket_x0::Float64
    bucket_y0::Float64
    bucket_dx::Float64
    bucket_dy::Float64
    bucket_regions::_PackedBuckets
    bucket_avg_len::Float64
    # Lazily cached per-region facet metadata (for boundary/adjacency).
    facets::Vector{Union{Nothing,Vector{CachedFacet}}}
    points_f::Vector{Union{Nothing,Matrix{Float64}}}
    hrep_float::Vector{Union{Nothing,_HRepFloatCache}}
    boundary_breakdown::Dict{BoundaryBreakdownCacheKey,Vector{BoundaryBreakdownEntry}}
    boundary_measure::Dict{BoundaryBreakdownCacheKey,Float64}
    adjacency::Dict{AdjacencyCacheKey,Dict{Tuple{Int,Int},Float64}}
    facet_classify_scratch::Vector{_FacetClassifyScratch}
    # Exact volume cache for repeated region_weights(method=:exact) on the same box.
    exact_weight::Vector{Float64}
    exact_weight_ready::BitVector
    # Exact centroid cache for repeated region_centroid(method=:polyhedra).
    exact_centroid::Vector{Vector{Float64}}
    exact_centroid_ready::BitVector
    poly::Vector{Union{Nothing,PolyT}}
    vrep::Vector{Union{Nothing,VRepT}}
    hrep::Vector{Union{Nothing,HRepT}}
end

"""
    ambient_dim(x)

Return the ambient Euclidean dimension of a `PLPolyhedra` object.

Accepted inputs include single polyhedra, PL unions, PL birth/death sets,
fringe presentations, PL encoding maps, and geometry caches.

This is a cheap scalar accessor. It reads stored metadata only and does not
materialize Polyhedra objects or geometry caches.
"""
@inline ambient_dim(hp::HPoly) = hp.n
@inline ambient_dim(U::PolyUnion) = U.n
@inline ambient_dim(U::PLUpset) = ambient_dim(U.U)
@inline ambient_dim(D::PLDownset) = ambient_dim(D.D)
@inline ambient_dim(F::PLFringe) = F.n
@inline ambient_dim(pi::PLEncodingMap) = pi.n
@inline ambient_dim(cache::PolyInBoxCache) = ambient_dim(cache.pi)

"""
    polyhedra(U)
    npolyhedra(U)

Return the convex pieces of a PL union object, and the corresponding piece
count.

Use `polyhedra(...)` when you want the actual `HPoly` pieces, and
`npolyhedra(...)` when you only want the cheap piece count for summaries or
notebook inspection.

These accessors are storage-stable and do not invoke Polyhedra.jl.
"""
@inline polyhedra(hp::HPoly) = HPoly[hp]
@inline polyhedra(U::PolyUnion) = U.parts
@inline polyhedra(U::PLUpset) = polyhedra(U.U)
@inline polyhedra(D::PLDownset) = polyhedra(D.D)

@inline npolyhedra(::HPoly) = 1
@inline npolyhedra(U::PolyUnion) = length(U.parts)
@inline npolyhedra(U::PLUpset) = npolyhedra(U.U)
@inline npolyhedra(D::PLDownset) = npolyhedra(D.D)

"""
    birth_upsets(F::PLFringe)
    death_downsets(F::PLFringe)
    coefficient_matrix(F::PLFringe)

Return the birth upsets, death downsets, and presentation matrix of the PL
fringe presentation `F`.
"""
@inline birth_upsets(F::PLFringe) = F.Ups
@inline death_downsets(F::PLFringe) = F.Downs
@inline coefficient_matrix(F::PLFringe) = F.Phi

"""
    nupsets(F::PLFringe) -> Int
    ndownsets(F::PLFringe) -> Int

Return the number of birth upsets and death downsets in the PL fringe
presentation `F`.

These are cheap scalar helpers for notebook and summary workflows.
"""
@inline nupsets(F::PLFringe) = length(birth_upsets(F))
@inline ndownsets(F::PLFringe) = length(death_downsets(F))

"""
    field(F::PLFringe) -> QQField

Return the coefficient field of the PL fringe presentation `F`.

PL fringes currently use exact rational coefficients, so this always returns
`QQField()`.
"""
@inline field(::PLFringe) = QQField()

"""
    nregions(pi)

Return the number of polyhedral regions represented by a PL encoding object.
"""
@inline nregions(pi::PLEncodingMap) = length(pi.regions)
@inline nregions(cache::PolyInBoxCache) = nregions(cache.pi)
@inline nregions(enc::CompiledEncoding{<:PLEncodingMap}) = nregions(enc.pi)

@inline _pl_signature_shape(sig_y::Vector{BitVector}, sig_z::Vector{BitVector}) =
    (; upsets=(isempty(sig_y) ? 0 : length(sig_y[1])),
       downsets=(isempty(sig_z) ? 0 : length(sig_z[1])))

"""
    region_witnesses(pi)
    region_witness(pi, r)

Return the stored witness points for a PL encoding object, or the witness of a
single region `r`.

These witnesses come from the feasibility step that builds the finite region
decomposition. They are the cheapest human-readable representatives of PL
regions, so prefer them for inspection before asking for heavier region-geometry
data.
"""
@inline region_witnesses(pi::PLEncodingMap) = pi.witnesses
@inline region_witnesses(cache::PolyInBoxCache) = region_witnesses(cache.pi)
@inline region_witnesses(enc::CompiledEncoding{<:PLEncodingMap}) = region_witnesses(enc.pi)
@inline region_witness(pi::PLEncodingMap, r::Integer) = (@boundscheck checkbounds(pi.witnesses, r); pi.witnesses[r])
@inline region_witness(cache::PolyInBoxCache, r::Integer) = region_witness(cache.pi, r)
@inline region_witness(enc::CompiledEncoding{<:PLEncodingMap}, r::Integer) = region_witness(enc.pi, r)

"""
    region_signature(pi, r) -> NamedTuple

Return the `(y, z)` signature of PL region `r` as
`(; y=Vector{Bool}, z=Vector{Bool})`.

This is intentionally materialized for human-readable inspection. Keep
[`pl_encoding_summary`](@ref) or [`pl_query_summary`](@ref) as the cheap/default
inspection path, and call `region_signature(...)` only when you actually want
the signature bits of a specific region.
"""
@inline function region_signature(pi::PLEncodingMap, r::Integer)
    @boundscheck begin
        checkbounds(pi.sig_y, r)
        checkbounds(pi.sig_z, r)
    end
    return (; y=collect(Bool, pi.sig_y[r]), z=collect(Bool, pi.sig_z[r]))
end
@inline region_signature(cache::PolyInBoxCache, r::Integer) = region_signature(cache.pi, r)
@inline region_signature(enc::CompiledEncoding{<:PLEncodingMap}, r::Integer) = region_signature(enc.pi, r)

"""
    has_spatial_index(pi) -> Bool

Return whether a PL encoding object has a spatial or bucket prefilter available
for repeated `locate` queries.

This is a cheap capability accessor. It tells you whether the fast query path
can prefilter candidate regions before exact H-poly membership checks.
"""
@inline has_spatial_index(pi::PLEncodingMap) = pi.prefilter.spatial.enabled
@inline has_spatial_index(cache::PolyInBoxCache) = cache.bucket_enabled || has_spatial_index(cache.pi)
@inline has_spatial_index(enc::CompiledEncoding{<:PLEncodingMap}) = has_spatial_index(enc.pi)

"""
    cache_box(cache)
    cache_level(cache)
    cached_region_count(cache)

Return the cached query box, cache level, and currently known active-region
count of a `PolyInBoxCache`.

Use these to inspect a cache object without reading internal fields directly:
- `cache_box(cache)` returns the exact rational box attached to the cache;
- `cache_level(cache)` reports `:light`, `:geometry`, or `:full`;
- `cached_region_count(cache)` reports how many regions are currently known to
  intersect the cached box.
"""
@inline cache_box(cache::PolyInBoxCache) = cache.box_q
@inline cache_level(cache::PolyInBoxCache) = cache.level
@inline cached_region_count(cache::PolyInBoxCache) = count(==(Int8(1)), cache.activity_state)

@inline _plpoly_issue_report(kind::Symbol, valid::Bool; kwargs...) = (; kind, valid, kwargs...)

@inline function _throw_invalid_plpoly(fn::Symbol, issues::Vector{String})
    throw(ArgumentError(string(fn) * ": " * join(issues, " ")))
end

@inline _plpoly_point_length(x::AbstractVector) = length(x)
@inline _plpoly_point_length(x::Tuple) = length(x)
@inline _plpoly_point_length(::Any) = nothing

@inline _plpoly_point_kind(x::AbstractVector{<:Integer}) = :integer_vector
@inline _plpoly_point_kind(x::AbstractVector{<:Real}) = :real_vector
@inline _plpoly_point_kind(x::NTuple{N,<:Integer}) where {N} = :integer_tuple
@inline _plpoly_point_kind(x::NTuple{N,<:Real}) where {N} = :real_tuple
@inline _plpoly_point_kind(x::Tuple) = all(v -> v isa Real, x) ? :real_tuple : :invalid
@inline _plpoly_point_kind(::Any) = :invalid

@inline _plpoly_box_endpoint_kind(x::AbstractVector{<:Integer}) = :integer_vector
@inline _plpoly_box_endpoint_kind(x::AbstractVector{<:Real}) = :real_vector
@inline _plpoly_box_endpoint_kind(x::NTuple{N,<:Integer}) where {N} = :integer_tuple
@inline _plpoly_box_endpoint_kind(x::NTuple{N,<:Real}) where {N} = :real_tuple
@inline _plpoly_box_endpoint_kind(x::Tuple) = all(v -> v isa Real, x) ? :real_tuple : :invalid
@inline _plpoly_box_endpoint_kind(::Any) = :invalid

@inline _plpoly_box_endpoint_length(x::AbstractVector) = length(x)
@inline _plpoly_box_endpoint_length(x::Tuple) = length(x)
@inline _plpoly_box_endpoint_length(::Any) = nothing

@inline _plpoly_poset_kind(::SignaturePoset) = :signature
@inline _plpoly_poset_kind(::Any) = :dense

@inline _maybe_unwrap_pl_pi(pi::PLEncodingMap) = pi
@inline _maybe_unwrap_pl_pi(cache::PolyInBoxCache) = cache.pi
@inline _maybe_unwrap_pl_pi(enc::CompiledEncoding{<:PLEncodingMap}) = enc.pi
@inline _maybe_unwrap_pl_pi(::Any) = nothing

@inline function _unwrap_pl_pi(pi_or_cache)
    pi = _maybe_unwrap_pl_pi(pi_or_cache)
    pi === nothing && throw(ArgumentError("expected PLEncodingMap, PolyInBoxCache, or CompiledEncoding{<:PLEncodingMap}"))
    return pi
end

"""
    PLPolyhedraValidationSummary

Pretty printable wrapper for reports returned by the owner-local `PLPolyhedra`
validation helpers.

Use [`plpolyhedra_validation_summary`](@ref) to turn a raw report from
[`check_hpoly`](@ref), [`check_poly_union`](@ref), [`check_pl_upset`](@ref),
[`check_pl_downset`](@ref), [`check_pl_fringe`](@ref),
[`check_pl_encoding_map`](@ref), [`check_poly_in_box_cache`](@ref),
[`check_pl_point`](@ref), [`check_pl_points`](@ref), or [`check_pl_box`](@ref)
into a notebook/REPL-friendly summary object.
"""
struct PLPolyhedraValidationSummary{R}
    report::R
end

"""
    plpolyhedra_validation_summary(report) -> PLPolyhedraValidationSummary

Wrap a raw `PLPolyhedra` validation report in a display-oriented summary
object.
"""
@inline plpolyhedra_validation_summary(report::NamedTuple) = PLPolyhedraValidationSummary(report)

"""
    check_hpoly(hp; throw=false) -> NamedTuple

Validate the stored H-representation of a single polyhedron.
"""
function check_hpoly(hp::HPoly; throw::Bool=false)
    issues = String[]
    size(hp.A, 2) == ambient_dim(hp) || push!(issues, "A has $(size(hp.A, 2)) columns, expected ambient dimension $(ambient_dim(hp)).")
    length(hp.b) == size(hp.A, 1) || push!(issues, "b has length $(length(hp.b)), expected $(size(hp.A, 1)) rows of A.")
    length(hp.strict_mask) == size(hp.A, 1) || push!(issues, "strict_mask has length $(length(hp.strict_mask)), expected $(size(hp.A, 1)).")
    hp.strict_eps >= zero(QQ) || push!(issues, "strict_eps must be nonnegative.")
    valid = isempty(issues)
    throw && !valid && _throw_invalid_plpoly(:check_hpoly, issues)
    return _plpoly_issue_report(:hpoly, valid;
                                ambient_dim=ambient_dim(hp),
                                nconstraints=size(hp.A, 1),
                                strict_constraints=count(identity, hp.strict_mask),
                                has_polyhedron=(hp.poly !== nothing),
                                issues=issues)
end

"""
    check_poly_union(U; throw=false) -> NamedTuple

Validate a finite union of H-polytopes.
"""
function check_poly_union(U::PolyUnion; throw::Bool=false)
    issues = String[]
    @inbounds for (i, hp) in enumerate(U.parts)
        ambient_dim(hp) == ambient_dim(U) ||
            push!(issues, "part $i has ambient dimension $(ambient_dim(hp)), expected $(ambient_dim(U)).")
        rep = check_hpoly(hp; throw=false)
        rep.valid || append!(issues, ["part $i: " * issue for issue in rep.issues])
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_plpoly(:check_poly_union, issues)
    return _plpoly_issue_report(:poly_union, valid;
                                ambient_dim=ambient_dim(U),
                                npolyhedra=npolyhedra(U),
                                issues=issues)
end

"""
    check_pl_upset(U; throw=false) -> NamedTuple

Validate a PL upset represented as a finite union of convex pieces.
"""
function check_pl_upset(U::PLUpset; throw::Bool=false)
    rep = check_poly_union(U.U; throw=false)
    issues = copy(rep.issues)
    valid = isempty(issues)
    throw && !valid && _throw_invalid_plpoly(:check_pl_upset, issues)
    return _plpoly_issue_report(:pl_upset, valid;
                                ambient_dim=ambient_dim(U),
                                npolyhedra=npolyhedra(U),
                                issues=issues)
end

"""
    check_pl_downset(D; throw=false) -> NamedTuple

Validate a PL downset represented as a finite union of convex pieces.
"""
function check_pl_downset(D::PLDownset; throw::Bool=false)
    rep = check_poly_union(D.D; throw=false)
    issues = copy(rep.issues)
    valid = isempty(issues)
    throw && !valid && _throw_invalid_plpoly(:check_pl_downset, issues)
    return _plpoly_issue_report(:pl_downset, valid;
                                ambient_dim=ambient_dim(D),
                                npolyhedra=npolyhedra(D),
                                issues=issues)
end

"""
    check_pl_fringe(F; throw=false) -> NamedTuple

Validate a PL fringe presentation before encoding it to a finite poset.
"""
function check_pl_fringe(F::PLFringe; throw::Bool=false)
    issues = String[]
    for (i, U) in enumerate(birth_upsets(F))
        ambient_dim(U) == ambient_dim(F) ||
            push!(issues, "birth upset $i has ambient dimension $(ambient_dim(U)), expected $(ambient_dim(F)).")
        rep = check_pl_upset(U; throw=false)
        rep.valid || append!(issues, ["birth upset $i: " * issue for issue in rep.issues])
    end
    for (j, D) in enumerate(death_downsets(F))
        ambient_dim(D) == ambient_dim(F) ||
            push!(issues, "death downset $j has ambient dimension $(ambient_dim(D)), expected $(ambient_dim(F)).")
        rep = check_pl_downset(D; throw=false)
        rep.valid || append!(issues, ["death downset $j: " * issue for issue in rep.issues])
    end
    size(coefficient_matrix(F), 1) == length(death_downsets(F)) ||
        push!(issues, "coefficient matrix has $(size(coefficient_matrix(F), 1)) rows, expected $(length(death_downsets(F))).")
    size(coefficient_matrix(F), 2) == length(birth_upsets(F)) ||
        push!(issues, "coefficient matrix has $(size(coefficient_matrix(F), 2)) columns, expected $(length(birth_upsets(F))).")
    valid = isempty(issues)
    throw && !valid && _throw_invalid_plpoly(:check_pl_fringe, issues)
    return _plpoly_issue_report(:pl_fringe, valid;
                                ambient_dim=ambient_dim(F),
                                nupsets=length(birth_upsets(F)),
                                ndownsets=length(death_downsets(F)),
                                matrix_size=size(coefficient_matrix(F)),
                                issues=issues)
end

"""
    check_pl_encoding_map(pi; throw=false) -> NamedTuple

Validate the internal shape contract of a PL encoding map.

This is the owner-local validator for [`PLEncodingMap`](@ref). It checks region
count consistency, witness sizes, signature widths, and the precompiled float
membership arrays used by [`locate`](@ref) and [`locate_many!`](@ref).
"""
function check_pl_encoding_map(pi_or_cache; throw::Bool=false)
    pi = _maybe_unwrap_pl_pi(pi_or_cache)
    issues = String[]
    if pi === nothing
        push!(issues, "expected PLEncodingMap, PolyInBoxCache, or CompiledEncoding{<:PLEncodingMap}.")
        valid = false
        throw && _throw_invalid_plpoly(:check_pl_encoding_map, issues)
        return _plpoly_issue_report(:pl_encoding_map, valid; ambient_dim=nothing, nregions=nothing, issues=issues)
    end
    nr = nregions(pi)
    lengths = (
        sig_y=length(pi.sig_y),
        sig_z=length(pi.sig_z),
        regions=length(pi.regions),
        witnesses=length(pi.witnesses),
        Af=length(pi.Af),
        bf_strict=length(pi.bf_strict),
        bf_relaxed=length(pi.bf_relaxed),
    )
    all(v -> v == nr, values(lengths)) ||
        push!(issues, "region-aligned storage lengths must all equal nregions=$nr (got $(repr(lengths))).")
    sig_shape = _pl_signature_shape(pi.sig_y, pi.sig_z)
    @inbounds for r in 1:nr
        ambient_dim(pi.regions[r]) == ambient_dim(pi) ||
            push!(issues, "region $r has ambient dimension $(ambient_dim(pi.regions[r])), expected $(ambient_dim(pi)).")
        length(pi.witnesses[r]) == ambient_dim(pi) ||
            push!(issues, "witness $r has length $(length(pi.witnesses[r])), expected $(ambient_dim(pi)).")
        length(pi.sig_y[r]) == sig_shape.upsets ||
            push!(issues, "sig_y[$r] has width $(length(pi.sig_y[r])), expected $(sig_shape.upsets).")
        length(pi.sig_z[r]) == sig_shape.downsets ||
            push!(issues, "sig_z[$r] has width $(length(pi.sig_z[r])), expected $(sig_shape.downsets).")
        size(pi.Af[r], 2) == ambient_dim(pi) ||
            push!(issues, "Af[$r] has $(size(pi.Af[r], 2)) columns, expected ambient dimension $(ambient_dim(pi)).")
        size(pi.Af[r], 1) == length(pi.bf_strict[r]) ||
            push!(issues, "Af[$r] row count does not match bf_strict[$r].")
        size(pi.Af[r], 1) == length(pi.bf_relaxed[r]) ||
            push!(issues, "Af[$r] row count does not match bf_relaxed[$r].")
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_plpoly(:check_pl_encoding_map, issues)
    return _plpoly_issue_report(:pl_encoding_map, valid;
                                ambient_dim=ambient_dim(pi),
                                nregions=nr,
                                signature_shape=sig_shape,
                                witness_count=length(pi.witnesses),
                                spatial_index_enabled=has_spatial_index(pi),
                                issues=issues)
end

"""
    check_poly_in_box_cache(cache; throw=false) -> NamedTuple

Validate a `PolyInBoxCache` against the canonical cache contract.
"""
function check_poly_in_box_cache(cache::PolyInBoxCache; throw::Bool=false)
    issues = String[]
    a, b = cache_box(cache)
    length(a) == ambient_dim(cache) || push!(issues, "cache box lower endpoint has length $(length(a)), expected $(ambient_dim(cache)).")
    length(b) == ambient_dim(cache) || push!(issues, "cache box upper endpoint has length $(length(b)), expected $(ambient_dim(cache)).")
    all(isfinite, Float64[float(v) for v in a]) || push!(issues, "cache box lower endpoint must be finite.")
    all(isfinite, Float64[float(v) for v in b]) || push!(issues, "cache box upper endpoint must be finite.")
    @inbounds for i in eachindex(a, b)
        a[i] <= b[i] || push!(issues, "cache box endpoint mismatch on axis $i: expected a[$i] <= b[$i].")
    end
    nr = nregions(cache)
    length(cache.activity_state) == nr || push!(issues, "activity_state has length $(length(cache.activity_state)), expected $nr.")
    length(cache.active_mask) == nr || push!(issues, "active_mask has length $(length(cache.active_mask)), expected $nr.")
    length(cache.active_index) == nr || push!(issues, "active_index has length $(length(cache.active_index)), expected $nr.")
    cache_level(cache) in (:light, :geometry, :full) ||
        push!(issues, "cache level must be one of :light, :geometry, :full.")
    cache.bucket_enabled && (cache.bucket_nx < 1 || cache.bucket_ny < 1) &&
        push!(issues, "bucket-enabled caches must have positive bucket grid dimensions.")
    valid = isempty(issues)
    throw && !valid && _throw_invalid_plpoly(:check_poly_in_box_cache, issues)
    return _plpoly_issue_report(:poly_in_box_cache, valid;
                                ambient_dim=ambient_dim(cache),
                                nregions=nr,
                                cached_region_count=cached_region_count(cache),
                                cache_level=cache_level(cache),
                                box=cache_box(cache),
                                bucket_lookup_enabled=cache.bucket_enabled,
                                issues=issues)
end

"""
    check_pl_point(pi, x; throw=false) -> NamedTuple

Validate a single point query for [`locate`](@ref).

Accepted inputs are integer or real tuples/vectors of ambient dimension
`ambient_dim(pi)`. This is the preferred notebook/REPL validator when you want
to confirm the query shape before calling `locate`.
"""
function check_pl_point(pi_or_cache, x; throw::Bool=false)
    pi = _maybe_unwrap_pl_pi(pi_or_cache)
    issues = String[]
    query_kind = _plpoly_point_kind(x)
    point_length = _plpoly_point_length(x)
    ambient = pi === nothing ? nothing : ambient_dim(pi)
    pi === nothing && push!(issues, "expected PLEncodingMap, PolyInBoxCache, or CompiledEncoding{<:PLEncodingMap}.")
    query_kind === :invalid && push!(issues, "point must be given as a real vector or tuple.")
    if point_length !== nothing && ambient !== nothing && point_length != ambient
        push!(issues, "point has length $point_length, expected ambient dimension $ambient.")
    end
    if query_kind !== :invalid && !(x isa Tuple || x isa AbstractVector{<:Real} || x isa AbstractVector{<:Integer})
        push!(issues, "point entries must be real.")
    elseif query_kind !== :invalid && !all(isfinite, Float64[float(v) for v in x])
        push!(issues, "point coordinates must be finite.")
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_plpoly(:check_pl_point, issues)
    return _plpoly_issue_report(:pl_point, valid;
                                ambient_dim=ambient,
                                query_kind=query_kind,
                                point_length=point_length,
                                issues=issues)
end

"""
    check_pl_points(pi, X; throw=false) -> NamedTuple

Validate a column-batch point matrix for [`locate_many!`](@ref).

The accepted contract is a real matrix `X` of shape `(ambient_dim(pi), npoints)`
whose columns are query points.
"""
function check_pl_points(pi_or_cache, X; throw::Bool=false)
    pi = _maybe_unwrap_pl_pi(pi_or_cache)
    issues = String[]
    ambient = pi === nothing ? nothing : ambient_dim(pi)
    matrix_size = X isa AbstractMatrix ? size(X) : nothing
    pi === nothing && push!(issues, "expected PLEncodingMap, PolyInBoxCache, or CompiledEncoding{<:PLEncodingMap}.")
    !(X isa AbstractMatrix) && push!(issues, "point batch must be an AbstractMatrix with points stored by columns.")
    if X isa AbstractMatrix
        eltype(X) <: Real || push!(issues, "point batch entries must be real.")
        ambient !== nothing && size(X, 1) != ambient &&
            push!(issues, "point batch has $(size(X, 1)) rows, expected ambient dimension $ambient.")
        (eltype(X) <: Real) && !all(isfinite, X) && push!(issues, "point batch entries must be finite.")
    end
    valid = isempty(issues)
    throw && !valid && _throw_invalid_plpoly(:check_pl_points, issues)
    return _plpoly_issue_report(:pl_points, valid;
                                ambient_dim=ambient,
                                matrix_size=matrix_size,
                                query_kind=(X isa AbstractMatrix ? :column_batch_matrix : :invalid),
                                issues=issues)
end

"""
    check_pl_box(pi, box; throw=false) -> NamedTuple

Validate a finite axis-aligned query box `box=(a,b)` for cache construction and
region-geometry queries.

Accepted endpoints are integer or real tuples/vectors of ambient dimension
`ambient_dim(pi)` satisfying `a[i] <= b[i]` for every axis.
"""
function check_pl_box(pi_or_cache, box; throw::Bool=false)
    pi = _maybe_unwrap_pl_pi(pi_or_cache)
    issues = String[]
    ambient = pi === nothing ? nothing : ambient_dim(pi)
    endpoint_types = nothing
    endpoint_lengths = nothing
    pi === nothing && push!(issues, "expected PLEncodingMap, PolyInBoxCache, or CompiledEncoding{<:PLEncodingMap}.")
    if !(box isa Tuple && length(box) == 2)
        push!(issues, "box must be a pair (a, b) of real endpoints.")
    else
        a, b = box
        kind_a = _plpoly_box_endpoint_kind(a)
        kind_b = _plpoly_box_endpoint_kind(b)
        endpoint_types = (kind_a, kind_b)
        kind_a === :invalid && push!(issues, "box lower endpoint must be a real vector or tuple.")
        kind_b === :invalid && push!(issues, "box upper endpoint must be a real vector or tuple.")
        len_a = _plpoly_box_endpoint_length(a)
        len_b = _plpoly_box_endpoint_length(b)
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
    throw && !valid && _throw_invalid_plpoly(:check_pl_box, issues)
    return _plpoly_issue_report(:pl_box, valid;
                                ambient_dim=ambient,
                                endpoint_types=endpoint_types,
                                endpoint_lengths=endpoint_lengths,
                                issues=issues)
end

@inline function _pl_region_effective_box(pi_or_cache, box, cache)
    if cache isa PolyInBoxCache
        return cache_box(cache)
    elseif pi_or_cache isa PolyInBoxCache
        return cache_box(pi_or_cache)
    else
        return box
    end
end

@inline function _pl_region_bbox_if_available(pi_or_cache, r::Int, box, cache)
    box === nothing && return nothing
    target = pi_or_cache isa PolyInBoxCache ? pi_or_cache.pi : pi_or_cache
    try
        return cache === nothing ?
            region_bbox(target, r; box=box) :
            region_bbox(target, r; box=box, cache=cache)
    catch
        return nothing
    end
end

"""
    check_pl_region(pi, r; box=nothing, throw=false) -> NamedTuple

Validate a region id `r` against a PL encoding object.

This helper checks:
- whether `r` is an integer region index in `1:nregions(pi)`,
- whether the stored witness and region signature are available,
- whether a finite box is required for bounded geometry queries, and
- whether an optional `box=(a,b)` has the correct shape.

If a valid box is supplied, the report also includes a best-effort `bbox`
payload for region `r`.
"""
function check_pl_region(pi_or_cache, r; box=nothing, throw::Bool=false)
    pi = _maybe_unwrap_pl_pi(pi_or_cache)
    issues = String[]
    ambient = pi === nothing ? nothing : ambient_dim(pi)
    finite_box_required = !(pi_or_cache isa PolyInBoxCache)
    nreg = pi === nothing ? nothing : nregions(pi)
    region_in_range = false
    witness = nothing
    signature_support_sizes = nothing
    bbox = nothing
    box_report = nothing

    pi === nothing && push!(issues, "expected PLEncodingMap, PolyInBoxCache, or CompiledEncoding{<:PLEncodingMap}.")
    r isa Integer || push!(issues, "region id must be an integer.")
    if pi !== nothing && r isa Integer
        region_in_range = 1 <= Int(r) <= nreg
        region_in_range || push!(issues, "region index $(repr(r)) is out of range for nregions=$nreg.")
        if region_in_range
            rr = Int(r)
            witness = region_witness(pi, rr)
            sig = region_signature(pi, rr)
            signature_support_sizes = (; y=count(identity, sig.y), z=count(identity, sig.z))
        end
    end

    effective_box = _pl_region_effective_box(pi_or_cache, box, nothing)
    if effective_box !== nothing
        box_report = check_pl_box(pi_or_cache, effective_box; throw=false)
        box_report.valid || append!(issues, ["box: " * issue for issue in box_report.issues])
    end
    if pi !== nothing && region_in_range && effective_box !== nothing && (box_report === nothing || box_report.valid)
        bbox = _pl_region_bbox_if_available(pi_or_cache, Int(r), effective_box, nothing)
    end

    valid = isempty(issues)
    throw && !valid && _throw_invalid_plpoly(:check_pl_region, issues)
    return _plpoly_issue_report(:pl_region, valid;
                                ambient_dim=ambient,
                                nregions=nreg,
                                region=(r isa Integer ? Int(r) : r),
                                region_in_range=region_in_range,
                                witness=witness,
                                signature_support_sizes=signature_support_sizes,
                                finite_box_required=finite_box_required,
                                box_provided=(effective_box !== nothing),
                                bbox=bbox,
                                issues=issues)
end

function _plpoly_describe(hp::HPoly)
    return (;
        kind=:hpoly,
        ambient_dim=ambient_dim(hp),
        nconstraints=size(hp.A, 1),
        strict_constraints=count(identity, hp.strict_mask),
        has_polyhedron=(hp.poly !== nothing),
    )
end

function _plpoly_describe(U::PolyUnion)
    return (;
        kind=:poly_union,
        ambient_dim=ambient_dim(U),
        npolyhedra=npolyhedra(U),
    )
end

function _plpoly_describe(U::PLUpset)
    return (;
        kind=:pl_upset,
        ambient_dim=ambient_dim(U),
        npolyhedra=npolyhedra(U),
    )
end

function _plpoly_describe(D::PLDownset)
    return (;
        kind=:pl_downset,
        ambient_dim=ambient_dim(D),
        npolyhedra=npolyhedra(D),
    )
end

function _plpoly_describe(F::PLFringe)
    return (;
        kind=:pl_fringe,
        ambient_dim=ambient_dim(F),
        nupsets=nupsets(F),
        ndownsets=ndownsets(F),
        field=field(F),
        matrix_size=size(coefficient_matrix(F)),
    )
end

function _plpoly_describe(pi::PLEncodingMap)
    return (;
        kind=:pl_encoding_map,
        ambient_dim=ambient_dim(pi),
        nregions=nregions(pi),
        signature_shape=_pl_signature_shape(pi.sig_y, pi.sig_z),
        witness_count=length(pi.witnesses),
        spatial_index_enabled=has_spatial_index(pi),
    )
end

function _plpoly_describe(cache::PolyInBoxCache)
    return (;
        kind=:poly_in_box_cache,
        ambient_dim=ambient_dim(cache),
        nregions=nregions(cache),
        cached_region_count=cached_region_count(cache),
        cache_level=cache_level(cache),
        box=cache_box(cache),
        spatial_index_enabled=has_spatial_index(cache.pi),
        bucket_lookup_enabled=cache.bucket_enabled,
    )
end

function Base.show(io::IO, hp::HPoly)
    d = _plpoly_describe(hp)
    print(io, "HPoly(n=", d.ambient_dim,
          ", constraints=", d.nconstraints,
          ", strict=", d.strict_constraints, ")")
end

function Base.show(io::IO, ::MIME"text/plain", hp::HPoly)
    d = _plpoly_describe(hp)
    print(io, "HPoly",
          "\n  ambient_dim = ", d.ambient_dim,
          "\n  nconstraints = ", d.nconstraints,
          "\n  strict_constraints = ", d.strict_constraints,
          "\n  has_polyhedron = ", d.has_polyhedron)
end

function Base.show(io::IO, U::PolyUnion)
    d = _plpoly_describe(U)
    print(io, "PolyUnion(n=", d.ambient_dim, ", pieces=", d.npolyhedra, ")")
end

function Base.show(io::IO, ::MIME"text/plain", U::PolyUnion)
    d = _plpoly_describe(U)
    print(io, "PolyUnion",
          "\n  ambient_dim = ", d.ambient_dim,
          "\n  npolyhedra = ", d.npolyhedra)
end

function Base.show(io::IO, U::PLUpset)
    d = _plpoly_describe(U)
    print(io, "PLUpset(n=", d.ambient_dim, ", pieces=", d.npolyhedra, ")")
end

function Base.show(io::IO, ::MIME"text/plain", U::PLUpset)
    d = _plpoly_describe(U)
    print(io, "PLUpset",
          "\n  ambient_dim = ", d.ambient_dim,
          "\n  npolyhedra = ", d.npolyhedra)
end

function Base.show(io::IO, D::PLDownset)
    d = _plpoly_describe(D)
    print(io, "PLDownset(n=", d.ambient_dim, ", pieces=", d.npolyhedra, ")")
end

function Base.show(io::IO, ::MIME"text/plain", D::PLDownset)
    d = _plpoly_describe(D)
    print(io, "PLDownset",
          "\n  ambient_dim = ", d.ambient_dim,
          "\n  npolyhedra = ", d.npolyhedra)
end

function Base.show(io::IO, F::PLFringe)
    d = _plpoly_describe(F)
    print(io, "PLFringe(n=", d.ambient_dim,
          ", upsets=", d.nupsets,
          ", downsets=", d.ndownsets,
          ", matrix_size=", d.matrix_size, ")")
end

function Base.show(io::IO, ::MIME"text/plain", F::PLFringe)
    d = _plpoly_describe(F)
    print(io, "PLFringe",
          "\n  ambient_dim = ", d.ambient_dim,
          "\n  nupsets = ", d.nupsets,
          "\n  ndownsets = ", d.ndownsets,
          "\n  matrix_size = ", d.matrix_size)
end

function Base.show(io::IO, pi::PLEncodingMap)
    d = _plpoly_describe(pi)
    print(io, "PLEncodingMap(n=", d.ambient_dim,
          ", regions=", d.nregions,
          ", signature_shape=", d.signature_shape,
          ", spatial_index=", d.spatial_index_enabled, ")")
end

function Base.show(io::IO, ::MIME"text/plain", pi::PLEncodingMap)
    d = _plpoly_describe(pi)
    print(io, "PLEncodingMap",
          "\n  ambient_dim = ", d.ambient_dim,
          "\n  nregions = ", d.nregions,
          "\n  signature_shape = ", d.signature_shape,
          "\n  witness_count = ", d.witness_count,
          "\n  spatial_index_enabled = ", d.spatial_index_enabled)
end

function Base.show(io::IO, cache::PolyInBoxCache)
    d = _plpoly_describe(cache)
    print(io, "PolyInBoxCache(n=", d.ambient_dim,
          ", regions=", d.nregions,
          ", cached=", d.cached_region_count,
          ", level=", repr(d.cache_level),
          ", bucket_lookup=", d.bucket_lookup_enabled, ")")
end

function Base.show(io::IO, ::MIME"text/plain", cache::PolyInBoxCache)
    d = _plpoly_describe(cache)
    print(io, "PolyInBoxCache",
          "\n  ambient_dim = ", d.ambient_dim,
          "\n  nregions = ", d.nregions,
          "\n  cached_region_count = ", d.cached_region_count,
          "\n  cache_level = ", repr(d.cache_level),
          "\n  box = ", d.box,
          "\n  spatial_index_enabled = ", d.spatial_index_enabled,
          "\n  bucket_lookup_enabled = ", d.bucket_lookup_enabled)
end

function Base.show(io::IO, summary::PLPolyhedraValidationSummary)
    r = summary.report
    print(io, "PLPolyhedraValidationSummary(kind=", r.kind,
          ", valid=", r.valid,
          ", issues=", length(r.issues), ")")
end

function Base.show(io::IO, ::MIME"text/plain", summary::PLPolyhedraValidationSummary)
    r = summary.report
    println(io, "PLPolyhedraValidationSummary")
    println(io, "  kind: ", r.kind)
    println(io, "  valid: ", r.valid)
    for key in propertynames(r)
        key in (:kind, :valid, :issues) && continue
        println(io, "  ", key, ": ", repr(getproperty(r, key)))
    end
    if isempty(r.issues)
        print(io, "  issues: []")
    else
        println(io, "  issues:")
        for issue in r.issues
            println(io, "    - ", issue)
        end
    end
end

"""
    pl_fringe_summary(F) -> NamedTuple

Owner-local inspection surface for a PL fringe presentation.

This is the discoverable `PLPolyhedra` summary entrypoint parallel to the shared
[`describe(F)`](@ref) surface. It reports ambient dimension, coefficient field,
birth/death counts, and presentation matrix size without requiring users to
inspect raw fields.
"""
@inline pl_fringe_summary(F::PLFringe) = _plpoly_describe(F)

"""
    pl_encoding_summary(x) -> NamedTuple

Owner-local inspection surface for PL encoding objects.

Accepted inputs include bare [`PLEncodingMap`](@ref) values, compiled
encodings, and geometry caches.

Cheap/default path:
- call `pl_encoding_summary(pi)` first when you want region counts, signature
  widths, and spatial-index availability;
- only inspect region-level data with [`pl_region_summary`](@ref) or
  [`region_signature`](@ref) when you actually need a specific region.

# Examples

```julia
using TamerOp

PLP = TamerOp.PLPolyhedra
opts = TamerOp.EncodingOptions(backend=:pl, max_regions=100)

U = PLP.PLUpset(PLP.PolyUnion(1, [PLP.make_hpoly([-1.0], [0.0])]))
D = PLP.PLDownset(PLP.PolyUnion(1, [PLP.make_hpoly([1.0], [1.0])]))
F = PLP.PLFringe([U], [D], reshape([1//1], 1, 1))

P, H, pi = PLP.encode_from_PL_fringe(F, opts)
PLP.pl_encoding_summary(pi)

r = PLP.locate(pi, [0.5])
cache = PLP.poly_in_box_cache(pi; box=(Float64[0.0], Float64[1.0]), closure=true)
TamerOp.RegionGeometry.region_bbox(pi, r; box=(Float64[0.0], Float64[1.0]), cache=cache)
```
"""
@inline pl_encoding_summary(pi::PLEncodingMap) = _plpoly_describe(pi)
@inline pl_encoding_summary(cache::PolyInBoxCache) = _plpoly_describe(cache)
@inline function pl_encoding_summary(enc::CompiledEncoding{<:PLEncodingMap})
    return (; _plpoly_describe(enc.pi)...,
            compiled=true,
            poset_kind=_plpoly_poset_kind(enc.P),
            poset_size=nvertices(enc.P))
end

@inline _plpoly_query_point_key(x::Tuple) = x
@inline _plpoly_query_point_key(x::AbstractVector) = Tuple(x)
@inline _plpoly_query_point_key(x) = x

"""
    pl_query_summary(pi, x) -> NamedTuple

Return a compact semantic summary of a single PL-encoding query.

This helper is intended for notebook and REPL exploration. It reports the query
point, the located region id, the stored witness, the support sizes of the
region signature, whether a spatial/bucket prefilter is available, and whether
the query landed outside the represented region set (`0` from [`locate`](@ref)).

This is the cheap/default inspection path for a single query point. Prefer it
before asking for heavier region geometry such as exact bounding boxes or
boundary measurements.
"""
function pl_query_summary(pi_or_cache, x)
    report = check_pl_point(pi_or_cache, x; throw=true)
    pi = _unwrap_pl_pi(pi_or_cache)
    rid = locate(pi_or_cache, x)
    sig_counts = if rid == 0
        nothing
    else
        sig = region_signature(pi, rid)
        (; y=count(identity, sig.y), z=count(identity, sig.z))
    end
    return (;
        kind=:pl_query,
        point=_plpoly_query_point_key(x),
        query_kind=report.query_kind,
        region=rid,
        witness=(rid == 0 ? nothing : region_witness(pi, rid)),
        signature_support_sizes=sig_counts,
        spatial_index_enabled=has_spatial_index(pi_or_cache),
        cache_level=(pi_or_cache isa PolyInBoxCache ? cache_level(pi_or_cache) : nothing),
        outside=(rid == 0),
    )
end

"""
    pl_region_summary(pi, r; box=nothing, cache=nothing) -> NamedTuple

Return a compact semantic summary of region `r` in a PL encoding object.

The summary reports:
- region id validity,
- the stored witness,
- signature support sizes,
- whether a finite box is required for bounded geometry queries, and
- a region bounding box when an effective finite box is available.

Use this when you already know the region id and want the canonical region-level
inspection payload in one place instead of separately calling
[`region_witness`](@ref), [`region_signature`](@ref), and [`region_bbox`](@ref).
"""
function pl_region_summary(pi_or_cache, r; box=nothing, cache=nothing)
    report = check_pl_region(pi_or_cache, r; box=box, throw=true)
    pi = _unwrap_pl_pi(pi_or_cache)
    effective_box = _pl_region_effective_box(pi_or_cache, box, cache)
    cache0 = cache === nothing && pi_or_cache isa PolyInBoxCache ? pi_or_cache : cache
    bbox = _pl_region_bbox_if_available(pi_or_cache, Int(r), effective_box, cache0)
    return (;
        kind=:pl_region,
        ambient_dim=ambient_dim(pi),
        nregions=nregions(pi),
        region=Int(r),
        witness=report.witness,
        signature_support_sizes=report.signature_support_sizes,
        finite_box_required=report.finite_box_required,
        box_provided=(effective_box !== nothing),
        bbox=bbox,
        spatial_index_enabled=has_spatial_index(pi_or_cache),
        cache_level=(cache0 isa PolyInBoxCache ? cache_level(cache0) : nothing),
    )
end

"""
    poly_cache_summary(cache) -> NamedTuple

Owner-local inspection surface for `PolyInBoxCache`.

Use this as the discoverable cache-summary entrypoint in notebooks and the
REPL. It reports the cached box, cache level, region counts, and whether
bucket/spatial lookup is currently available.
"""
@inline poly_cache_summary(cache::PolyInBoxCache) = _plpoly_describe(cache)

# Avoid dispatching to ZnEncoding's float-only generic when locating through a
# PolyInBoxCache with Float64 query batches.
function locate_many!(dest::AbstractVector{<:Integer},
                      cache::PolyInBoxCache,
                      X::Matrix{Float64};
                      threaded::Bool=true,
                      mode::Symbol=:fast,
                      tol::Float64=LOCATE_FLOAT_TOL,
                      boundary_tol::Float64=LOCATE_BOUNDARY_TOL)
    npts = size(X, 2)
    _locate_many_prefix!(dest, cache, X, npts;
                         threaded=threaded,
                         mode=mode,
                         tol=tol,
                         boundary_tol=boundary_tol)
    return dest
end

function locate_many!(dest::AbstractVector{<:Integer},
                      cache::PolyInBoxCache,
                      X::AbstractMatrix{<:AbstractFloat};
                      threaded::Bool=true,
                      mode::Symbol=:fast,
                      tol::Float64=LOCATE_FLOAT_TOL,
                      boundary_tol::Float64=LOCATE_BOUNDARY_TOL)
    return _locate_many_pl!(dest, cache, X;
                            threaded=threaded,
                            mode=mode,
                            tol=tol,
                            boundary_tol=boundary_tol)
end

@inline function _cache_level_rank(level::Symbol)::Int
    if level === :light
        return 1
    elseif level === :geometry
        return 2
    elseif level === :full
        return 3
    end
    error("cache level must be one of :light, :geometry, :full (got $level)")
end

@inline function _normalize_cache_level(level::Symbol)::Symbol
    _cache_level_rank(level) # validates
    return level
end

function _poly_cache_types(n::Int)
    A = Matrix{QQ}(undef, 2 * n, n)
    b = Vector{QQ}(undef, 2 * n)
    row = 0
    @inbounds for j in 1:n
        row += 1
        for k in 1:n
            A[row, k] = (k == j) ? one(QQ) : zero(QQ)
        end
        b[row] = one(QQ)
        row += 1
        for k in 1:n
            A[row, k] = (k == j) ? -one(QQ) : zero(QQ)
        end
        b[row] = zero(QQ)
    end
    p = Polyhedra.polyhedron(Polyhedra.hrep(A, b), _CDD)
    v = Polyhedra.vrep(p)
    h = Polyhedra.hrep(p)
    return typeof(p), typeof(v), typeof(h)
end

"""
    poly_in_box_cache(pi::PLEncodingMap; box, closure=true, level=:light) -> PolyInBoxCache

Construct a cache for Polyhedra-based geometry queries inside a fixed window `box=(a,b)`.

Mathematical meaning:
- the cache is tied to the encoding map `pi` and the clipped region family
  `region(r) intersect box`;
- it stores reusable exact-geometry artifacts for the regions that meet the
  query window.

Pass the resulting object via the `cache=` keyword to geometry routines such as:
  - `region_boundary_measure`
  - `region_boundary_measure_breakdown`
  - `region_centroid` (method=:polyhedra)
  - `region_adjacency`
  - `region_weights(method=:exact)`

Why this exists:
Building Polyhedra objects can dominate runtime if you call multiple geometric
queries on the same encoding map and the same box. This cache lets you pay that
cost once.

Cache levels:
- `:light`: keep locate/prefilter artifacts only; promote lazily if later
  geometry calls need more;
- `:geometry`: keep the exact per-region clipped polyhedra lazy, materializing
  them on first touch;
- `:full`: additionally precompute active-region buckets and heavy exact
  geometry support for repeated region-geometry queries.

Notes:
  - Requires Polyhedra.jl + CDDLib.jl.
  - The cache is tied to a specific `(pi, box, closure)`; passing it with
    different parameters throws an error.
  - Per-region exact geometry is materialized lazily on first touch.
"""
function poly_in_box_cache(pi::PLEncodingMap; box, closure::Bool=true, level::Symbol=:light)
    HAVE_POLY || error("poly_in_box_cache: Polyhedra.jl + CDDLib.jl required")
    box === nothing && error("poly_in_box_cache: please provide box=(a,b)")
    a_in, b_in = box
    length(a_in) == pi.n || error("poly_in_box_cache: box lower corner has wrong dimension")
    length(b_in) == pi.n || error("poly_in_box_cache: box upper corner has wrong dimension")

    a_q = _toQQ_vec(a_in)
    b_q = _toQQ_vec(b_in)
    for i in 1:pi.n
        a_q[i] <= b_q[i] || error("poly_in_box_cache: expected a[i] <= b[i] for all i")
    end

    a_f = Float64[float(a_q[i]) for i in 1:pi.n]
    b_f = Float64[float(b_q[i]) for i in 1:pi.n]
    any(!isfinite, a_f) && error("poly_in_box_cache: box lower bounds must be finite")
    any(!isfinite, b_f) && error("poly_in_box_cache: box upper bounds must be finite")

    nregions = length(pi.regions)
    nregions > 0 || error("poly_in_box_cache: encoding has no regions")

    level = _normalize_cache_level(level)

    # Reuse precompiled membership matrices from the encoding map.
    Af = pi.Af
    bf_strict = pi.bf_strict
    bf_relaxed = pi.bf_relaxed

    PolyT, VRepT, HRepT = _poly_cache_types(pi.n)

    poly = Vector{Union{Nothing,PolyT}}(undef, nregions)
    vrep = Vector{Union{Nothing,VRepT}}(undef, nregions)
    hrep = Vector{Union{Nothing,HRepT}}(undef, nregions)
    fill!(poly, nothing)
    fill!(vrep, nothing)
    fill!(hrep, nothing)

    # Region AABBs are computed lazily on first touch.
    aabb_lo = [fill(Inf, pi.n) for _ in 1:nregions]
    aabb_hi = [fill(-Inf, pi.n) for _ in 1:nregions]
    activity_state = zeros(Int8, nregions)
    activity_scanned = false
    active_regions = Int[]
    active_mask = falses(nregions)
    active_index = zeros(Int, nregions)
    points_f = Vector{Union{Nothing,Matrix{Float64}}}(undef, nregions)
    fill!(points_f, nothing)

    nx = 1
    ny = 1
    x0 = a_f[1]
    y0 = (pi.n >= 2 ? a_f[2] : 0.0)
    dx = max(b_f[1] - a_f[1], 1.0)
    dy = (pi.n >= 2 ? max(b_f[2] - a_f[2], 1.0) : 1.0)
    bucket_enabled = false
    bucket_regions = _empty_packed_buckets()
    bucket_avg_len = 0.0

    facets = Vector{Union{Nothing,Vector{CachedFacet}}}(undef, nregions)
    fill!(facets, nothing)
    boundary_breakdown = Dict{BoundaryBreakdownCacheKey,Vector{BoundaryBreakdownEntry}}()
    boundary_measure = Dict{BoundaryBreakdownCacheKey,Float64}()
    adjacency = Dict{AdjacencyCacheKey,Dict{Tuple{Int,Int},Float64}}()
    hrep_float = Vector{Union{Nothing,_HRepFloatCache}}(undef, nregions)
    fill!(hrep_float, nothing)
    facet_classify_scratch = [_FacetClassifyScratch(
        Matrix{Float64}(undef, pi.n, 0),
        falses(0),
        Int[],
    ) for _ in 1:max(1, Threads.nthreads())]
    exact_weight = zeros(Float64, nregions)
    exact_weight_ready = falses(nregions)
    exact_centroid = [zeros(Float64, pi.n) for _ in 1:nregions]
    exact_centroid_ready = falses(nregions)

    return PolyInBoxCache{PolyT,VRepT,HRepT}(pi, (a_q, b_q), (a_f, b_f), closure,
                                              level, Base.ReentrantLock(),
                                              Af, bf_strict, bf_relaxed,
                                              activity_state, activity_scanned,
                                              aabb_lo, aabb_hi, active_regions, active_mask, active_index,
                                              bucket_enabled, nx, ny, x0, y0, dx, dy, bucket_regions, bucket_avg_len,
                                              facets, points_f, hrep_float,
                                              boundary_breakdown, boundary_measure, adjacency, facet_classify_scratch,
                                              exact_weight, exact_weight_ready,
                                              exact_centroid, exact_centroid_ready,
                                              poly, vrep, hrep)
end

"""
    compile_geometry_cache(pi::PLEncodingMap; box, closure=true,
                           precompute_exact=true,
                           precompute_facets=true,
                           precompute_centroids=true,
                           level=:full) -> PolyInBoxCache

Build a reusable cache for repeated region-membership and geometry queries on a
fixed `(pi, box)` pair.

This is the eager/heavy companion to [`poly_in_box_cache`](@ref):
- use `poly_in_box_cache(...)` when you want a cheap cache shell that promotes
  lazily by query intent;
- use `compile_geometry_cache(...)` when you know ahead of time that many exact
  geometry queries will hit the same `(pi, box)` window.

Cache levels:
- `:light`: locate/prefilter artifacts only.
- `:geometry`: exact per-region polyhedra built lazily on first touch.
- `:full`: precompute region activity buckets and exact geometry metadata.

When `precompute_exact=true` (default), exact-geometry artifacts are built once:
- clipped region polyhedra / V- / H-representations,
- per-region exact volumes,
- cached facets for boundary/adjacency queries,
- cached exact centroids (with bbox midpoint fallback).
"""
function compile_geometry_cache(pi::PLEncodingMap; box, closure::Bool=true,
                                precompute_exact::Bool=true,
                                precompute_facets::Bool=true,
                                precompute_centroids::Bool=true,
                                level::Symbol=(precompute_exact ? :full : :geometry))
    cache = poly_in_box_cache(pi; box=box, closure=closure, level=:light)
    _promote_cache_level!(
        cache,
        level;
        intent=:compile,
        precompute_exact=precompute_exact,
        precompute_facets=precompute_facets,
        precompute_centroids=precompute_centroids,
        precompute_weights=precompute_exact,
    )
    return cache
end

# Clear cached polyhedra / representations (useful in long-running sessions).
function Base.empty!(cache::PolyInBoxCache)
    Base.lock(cache.lock)
    try
        fill!(cache.activity_state, 0)
        cache.activity_scanned = false
        cache.level = :light
        empty!(cache.active_regions)
        fill!(cache.active_mask, false)
        fill!(cache.active_index, 0)
        @inbounds for r in eachindex(cache.aabb_lo)
            fill!(cache.aabb_lo[r], Inf)
            fill!(cache.aabb_hi[r], -Inf)
        end
        cache.bucket_enabled = false
        cache.bucket_nx = 1
        cache.bucket_ny = 1
        cache.bucket_x0 = cache.box_f[1][1]
        cache.bucket_y0 = (cache.pi.n >= 2 ? cache.box_f[1][2] : 0.0)
        cache.bucket_dx = max(cache.box_f[2][1] - cache.box_f[1][1], 1.0)
        cache.bucket_dy = (cache.pi.n >= 2 ? max(cache.box_f[2][2] - cache.box_f[1][2], 1.0) : 1.0)
        cache.bucket_avg_len = 0.0
        cache.bucket_regions = _empty_packed_buckets()
    finally
        Base.unlock(cache.lock)
    end
    fill!(cache.poly, nothing)
    fill!(cache.vrep, nothing)
    fill!(cache.hrep, nothing)
    fill!(cache.facets, nothing)
    fill!(cache.points_f, nothing)
    fill!(cache.hrep_float, nothing)
    empty!(cache.boundary_breakdown)
    empty!(cache.boundary_measure)
    empty!(cache.adjacency)
    fill!(cache.exact_weight, 0.0)
    fill!(cache.exact_weight_ready, false)
    @inbounds for c in cache.exact_centroid
        fill!(c, 0.0)
    end
    fill!(cache.exact_centroid_ready, false)
    return cache
end

@inline function _is_aabb_active(lo::AbstractVector{<:Real}, hi::AbstractVector{<:Real})::Bool
    all(isfinite, lo) || return false
    all(isfinite, hi) || return false
    @inbounds for j in eachindex(lo)
        if hi[j] < lo[j]
            return false
        end
    end
    return true
end

@inline function _set_region_activity!(cache::PolyInBoxCache, r::Int, active::Bool)
    cache.activity_state[r] = active ? Int8(1) : Int8(-1)
    cache.active_mask[r] = active
    if active
        if cache.active_index[r] == 0
            push!(cache.active_regions, r)
            cache.active_index[r] = length(cache.active_regions)
        end
    else
        pos = cache.active_index[r]
        if pos != 0
            last = cache.active_regions[end]
            cache.active_regions[pos] = last
            cache.active_index[last] = pos
            pop!(cache.active_regions)
            cache.active_index[r] = 0
        end
    end
    return active
end

function _rebuild_active_regions!(cache::PolyInBoxCache)
    empty!(cache.active_regions)
    fill!(cache.active_mask, false)
    fill!(cache.active_index, 0)
    @inbounds for r in eachindex(cache.activity_state)
        if cache.activity_state[r] == Int8(1)
            push!(cache.active_regions, r)
            cache.active_mask[r] = true
            cache.active_index[r] = length(cache.active_regions)
        end
    end
    return cache.active_regions
end

function _compute_region_geometry(cache::PolyInBoxCache, r::Int)
    Pr = _hpoly_in_box_polyhedron(cache.pi.regions[r], cache.box_q; closure=cache.closure)
    Vr = Polyhedra.vrep(Pr)
    Hr = Polyhedra.hrep(Pr)
    raw = collect(Polyhedra.points(Vr))
    n = cache.pi.n
    lo = fill(Inf, n)
    hi = fill(-Inf, n)
    pts = if isempty(raw)
        zeros(Float64, 0, n)
    else
        out = Matrix{Float64}(undef, length(raw), n)
        @inbounds for i in eachindex(raw)
            x = _point_to_floatvec(raw[i], n)
            for j in 1:n
                out[i, j] = x[j]
                if x[j] < lo[j]
                    lo[j] = x[j]
                end
                if x[j] > hi[j]
                    hi[j] = x[j]
                end
            end
        end
        out
    end
    return Pr, Vr, Hr, pts, lo, hi
end

function _materialize_region_geometry!(cache::PolyInBoxCache, r::Int)
    st = cache.activity_state[r]
    st == Int8(0) || return (st == Int8(1))

    Pr, Vr, Hr, pts, lo, hi = _compute_region_geometry(cache, r)
    active = _is_aabb_active(lo, hi)

    Base.lock(cache.lock)
    try
        st2 = cache.activity_state[r]
        if st2 == Int8(0)
            cache.poly[r] = Pr
            cache.vrep[r] = Vr
            cache.hrep[r] = Hr
            cache.points_f[r] = pts
            copyto!(cache.aabb_lo[r], lo)
            copyto!(cache.aabb_hi[r], hi)
            _set_region_activity!(cache, r, active)
            # Any newly materialized region invalidates bucket metadata until rebuilt.
            cache.bucket_enabled = false
            cache.bucket_avg_len = 0.0
            cache.bucket_regions = _empty_packed_buckets()
        end
        return cache.activity_state[r] == Int8(1)
    finally
        Base.unlock(cache.lock)
    end
end

@inline function _cache_region_active(cache::PolyInBoxCache, r::Integer)::Bool
    (1 <= r <= length(cache.activity_state)) || error("region index out of range")
    return _materialize_region_geometry!(cache, Int(r))
end

function _materialize_pending_regions!(cache::PolyInBoxCache, pending::Vector{Int}; threaded::Bool=true)
    isempty(pending) && return nothing
    do_thread = threaded && _should_thread_region_loop(length(pending), _GEOM_BUILD_EST_OPS[])
    if do_thread
        Threads.@threads for i in eachindex(pending)
            r = pending[i]
            Pr, Vr, Hr, pts, lo, hi = _compute_region_geometry(cache, r)
            cache.poly[r] = Pr
            cache.vrep[r] = Vr
            cache.hrep[r] = Hr
            cache.points_f[r] = pts
            copyto!(cache.aabb_lo[r], lo)
            copyto!(cache.aabb_hi[r], hi)
            cache.activity_state[r] = _is_aabb_active(lo, hi) ? Int8(1) : Int8(-1)
        end
    else
        @inbounds for r in pending
            Pr, Vr, Hr, pts, lo, hi = _compute_region_geometry(cache, r)
            cache.poly[r] = Pr
            cache.vrep[r] = Vr
            cache.hrep[r] = Hr
            cache.points_f[r] = pts
            copyto!(cache.aabb_lo[r], lo)
            copyto!(cache.aabb_hi[r], hi)
            cache.activity_state[r] = _is_aabb_active(lo, hi) ? Int8(1) : Int8(-1)
        end
    end
    return nothing
end

function _ensure_all_region_activity!(cache::PolyInBoxCache; threaded::Bool=true)
    cache.activity_scanned && return cache.active_regions

    pending = Int[]
    sizehint!(pending, length(cache.activity_state))
    @inbounds for r in eachindex(cache.activity_state)
        if cache.activity_state[r] == Int8(0)
            push!(pending, r)
        end
    end
    _materialize_pending_regions!(cache, pending; threaded=threaded)

    Base.lock(cache.lock)
    try
        _rebuild_active_regions!(cache)
        cache.activity_scanned = true
    finally
        Base.unlock(cache.lock)
    end
    return cache.active_regions
end

function _build_bucket_index!(cache::PolyInBoxCache)
    if cache.bucket_enabled && cache.activity_scanned
        return cache
    end
    _ensure_all_region_activity!(cache; threaded=true)
    nactive = length(cache.active_regions)
    if nactive == 0 || cache.pi.n < 1
        cache.bucket_enabled = false
        cache.bucket_regions = _empty_packed_buckets()
        cache.bucket_avg_len = 0.0
        return cache
    end

    a_f, b_f = cache.box_f
    nx = max(4, min(128, round(Int, sqrt(nactive))))
    ny = cache.pi.n >= 2 ? max(4, min(128, round(Int, sqrt(nactive)))) : 1
    dx = max((b_f[1] - a_f[1]) / nx, 1e-12)
    dy = cache.pi.n >= 2 ? max((b_f[2] - a_f[2]) / ny, 1e-12) : 1.0
    x0 = a_f[1]
    y0 = (cache.pi.n >= 2 ? a_f[2] : 0.0)
    nb = nx * ny
    counts = zeros(Int, nb)
    @inbounds for r in cache.active_regions
        lo = cache.aabb_lo[r]
        hi = cache.aabb_hi[r]
        ix0 = clamp(Int(floor((lo[1] - x0) / dx)) + 1, 1, nx)
        ix1 = clamp(Int(floor((hi[1] - x0) / dx)) + 1, 1, nx)
        iy0 = 1
        iy1 = 1
        if cache.pi.n >= 2
            iy0 = clamp(Int(floor((lo[2] - y0) / dy)) + 1, 1, ny)
            iy1 = clamp(Int(floor((hi[2] - y0) / dy)) + 1, 1, ny)
        end
        for iy in iy0:iy1, ix in ix0:ix1
            counts[(iy - 1) * nx + ix] += 1
        end
    end

    ptr = Vector{Int}(undef, nb + 1)
    ptr[1] = 1
    @inbounds for bid in 1:nb
        ptr[bid + 1] = ptr[bid] + counts[bid]
    end
    idx = Vector{Int}(undef, ptr[end] - 1)
    writepos = copy(ptr[1:end-1])

    @inbounds for r in cache.active_regions
        lo = cache.aabb_lo[r]
        hi = cache.aabb_hi[r]
        ix0 = clamp(Int(floor((lo[1] - x0) / dx)) + 1, 1, nx)
        ix1 = clamp(Int(floor((hi[1] - x0) / dx)) + 1, 1, nx)
        iy0 = 1
        iy1 = 1
        if cache.pi.n >= 2
            iy0 = clamp(Int(floor((lo[2] - y0) / dy)) + 1, 1, ny)
            iy1 = clamp(Int(floor((hi[2] - y0) / dy)) + 1, 1, ny)
        end
        for iy in iy0:iy1, ix in ix0:ix1
            bid = (iy - 1) * nx + ix
            pos = writepos[bid]
            idx[pos] = r
            writepos[bid] = pos + 1
        end
    end

    total_bucket = ptr[end] - 1
    cache.bucket_enabled = true
    cache.bucket_nx = nx
    cache.bucket_ny = ny
    cache.bucket_x0 = x0
    cache.bucket_y0 = y0
    cache.bucket_dx = dx
    cache.bucket_dy = dy
    cache.bucket_regions = _PackedBuckets(ptr, idx)
    cache.bucket_avg_len = nb == 0 ? 0.0 : (total_bucket / nb)
    return cache
end

@inline function _cache_level_requires_materialization(level::Symbol, intent::Symbol)::Bool
    _cache_level_rank(level) >= _cache_level_rank(:full) && return true
    if level === :geometry
        return intent in (:weights_exact, :adjacency, :compile, :compiled_exact)
    end
    return false
end

@inline function _cache_level_requires_full_precompute(level::Symbol, intent::Symbol)::Bool
    level === :full || intent === :adjacency
end

function _promote_cache_level!(cache::PolyInBoxCache, level::Symbol;
                               intent::Symbol=:generic,
                               precompute_exact::Bool=false,
                               precompute_facets::Bool=false,
                               precompute_centroids::Bool=false,
                               precompute_weights::Bool=false)
    target = _normalize_cache_level(level)
    cur_rank = _cache_level_rank(cache.level)
    target_rank = _cache_level_rank(target)
    promoted = target_rank > cur_rank
    need_materialization = _cache_level_requires_materialization(target, intent) &&
                           (!cache.activity_scanned || !cache.bucket_enabled)
    if need_materialization
        _ensure_all_region_activity!(cache; threaded=true)
        _build_bucket_index!(cache)
    end
    need_full_seed = promoted && _cache_level_requires_full_precompute(target, intent)
    if precompute_exact || need_full_seed
        _ensure_all_region_activity!(cache; threaded=true)
        _build_bucket_index!(cache)
        _precompute_exact_geometry!(
            cache;
            precompute_facets=precompute_facets || target === :full,
            precompute_centroids=precompute_centroids || target === :full,
            precompute_weights=precompute_weights || target === :full,
        )
    end
    if promoted
        cache.level = target
    end
    return cache
end

@inline function _should_skip_auto_cache(pi::PLEncodingMap, intent::Symbol)::Bool
    if intent === :single_region_exact
        return length(pi.regions) <= _AUTO_GEOM_NO_CACHE_MAX_REGIONS[] && pi.n <= _AUTO_GEOM_NO_CACHE_MAX_DIM[]
    end
    return false
end

@inline function _canonical_box_key(pi::PLEncodingMap, box, closure::Bool)
    a_in, b_in = box
    n = pi.n
    length(a_in) == n || error("cache mismatch: expected box lower corner length $n")
    length(b_in) == n || error("cache mismatch: expected box upper corner length $n")
    a_key = ntuple(i -> _toQQ(a_in[i]), n)
    b_key = ntuple(i -> _toQQ(b_in[i]), n)
    return (UInt(objectid(pi)), a_key, b_key, closure)
end

function _geometry_cache_from_store!(store::EncodingCache, pi::PLEncodingMap, box, closure::Bool;
                                     level::Symbol=:light,
                                     intent::Symbol=:generic,
                                     precompute_exact::Bool=false,
                                     precompute_facets::Bool=false,
                                     precompute_centroids::Bool=false)
    _should_skip_auto_cache(pi, intent) && return nothing
    level = _normalize_cache_level(level)
    key = _canonical_box_key(pi, box, closure)

    existing = nothing
    Base.lock(store.lock)
    try
        entry = get(store.geometry, key, nothing)
        existing = (entry === nothing ? nothing : entry.value)
    finally
        Base.unlock(store.lock)
    end
    if existing !== nothing
        cache = existing
        cache isa PolyInBoxCache && _promote_cache_level!(
            cache,
            level;
            intent=intent,
            precompute_exact=precompute_exact,
            precompute_facets=precompute_facets,
            precompute_centroids=precompute_centroids,
            precompute_weights=precompute_exact,
        )
        return cache
    end

    built = compile_geometry_cache(
        pi;
        box=box,
        closure=closure,
        level=:light,
        precompute_exact=false,
        precompute_facets=false,
        precompute_centroids=false,
    )
    _promote_cache_level!(
        built,
        level;
        intent=intent,
        precompute_exact=precompute_exact,
        precompute_facets=precompute_facets,
        precompute_centroids=precompute_centroids,
        precompute_weights=precompute_exact,
    )
    Base.lock(store.lock)
    try
        entry = get(store.geometry, key, nothing)
        if entry === nothing
            store.geometry[key] = GeometryCachePayload(built)
            return built
        end
        existing = entry.value
    finally
        Base.unlock(store.lock)
    end
    cache = existing
    cache isa PolyInBoxCache && _promote_cache_level!(
        cache,
        level;
        intent=intent,
        precompute_exact=precompute_exact,
        precompute_facets=precompute_facets,
        precompute_centroids=precompute_centroids,
        precompute_weights=precompute_exact,
    )
    return cache
end

@inline function _auto_geometry_cache(pi::PLEncodingMap, cache, box, closure::Bool, mode::Symbol=:fast;
                                      level::Symbol=:light, intent::Symbol=:generic)
    if cache isa PolyInBoxCache
        _promote_cache_level!(cache, level; intent=intent)
        return cache
    end
    cache !== nothing && return cache
    box === nothing && return cache
    validate_pl_mode(mode)
    _should_skip_auto_cache(pi, intent) && return nothing
    return _geometry_cache_from_store!(
        pi.cache,
        pi,
        box,
        closure;
        level=level,
        intent=intent,
        precompute_exact=false,
        precompute_facets=false,
        precompute_centroids=false,
    )
end

function _check_cache_compatible(cache::PolyInBoxCache, pi::PLEncodingMap,
                                 box, closure::Bool)
    cache.pi === pi || error("cache mismatch: PolyInBoxCache was built for a different PLEncodingMap")
    cache.closure == closure || error("cache mismatch: closure=$(cache.closure) in cache, requested closure=$closure")
    if box !== nothing
        a_in, b_in = box
        length(a_in) == pi.n || error("cache mismatch: box lower corner has wrong dimension")
        length(b_in) == pi.n || error("cache mismatch: box upper corner has wrong dimension")
        a_q = _toQQ_vec(a_in)
        b_q = _toQQ_vec(b_in)
        a_q == cache.box_q[1] || error("cache mismatch: box lower corner differs from cache")
        b_q == cache.box_q[2] || error("cache mismatch: box upper corner differs from cache")
    end
    return nothing
end

function _poly_in_box(cache::PolyInBoxCache{PolyT}, r::Integer) where {PolyT}
    (1 <= r <= length(cache.poly)) || error("_poly_in_box: region index out of range")
    _materialize_region_geometry!(cache, Int(r))
    if cache.poly[r] === nothing
        cache.poly[r] = _hpoly_in_box_polyhedron(cache.pi.regions[Int(r)], cache.box_q; closure=cache.closure)
    end
    return cache.poly[r]::PolyT
end

function _vrep_in_box(cache::PolyInBoxCache{PolyT,VRepT}, r::Integer) where {PolyT,VRepT}
    (1 <= r <= length(cache.vrep)) || error("_vrep_in_box: region index out of range")
    _materialize_region_geometry!(cache, Int(r))
    if cache.vrep[r] === nothing
        cache.vrep[r] = Polyhedra.vrep(_poly_in_box(cache, r))
    end
    return cache.vrep[r]::VRepT
end

function _hrep_in_box(cache::PolyInBoxCache{PolyT,VRepT,HRepT}, r::Integer) where {PolyT,VRepT,HRepT}
    (1 <= r <= length(cache.hrep)) || error("_hrep_in_box: region index out of range")
    _materialize_region_geometry!(cache, Int(r))
    if cache.hrep[r] === nothing
        cache.hrep[r] = Polyhedra.hrep(_poly_in_box(cache, r))
    end
    return cache.hrep[r]::HRepT
end

function _hrep_float_in_box(cache::PolyInBoxCache, r::Integer)
    (1 <= r <= length(cache.hrep_float)) || error("_hrep_float_in_box: region index out of range")
    hf = cache.hrep_float[r]
    hf === nothing || return hf

    hre = _hrep_in_box(cache, r)
    hs = collect(Polyhedra.halfspaces(hre))
    m = length(hs)
    n = cache.pi.n
    A = Matrix{Float64}(undef, m, n)
    b = Vector{Float64}(undef, m)
    @inbounds for i in 1:m
        h = hs[i]
        j = 0
        for c in h.a
            j += 1
            A[i, j] = float(c)
        end
        j == n || error("_hrep_float_in_box: halfspace dimension mismatch")
        b[i] = float(_halfspace_rhs(h))
    end
    out = _HRepFloatCache(A, b)
    cache.hrep_float[r] = out
    return out
end

function _points_float_in_box(cache::PolyInBoxCache, r::Integer)
    (1 <= r <= length(cache.points_f)) || error("_points_float_in_box: region index out of range")
    _materialize_region_geometry!(cache, Int(r))
    pts = cache.points_f[r]
    pts === nothing || return pts

    vre = _vrep_in_box(cache, r)
    raw = collect(Polyhedra.points(vre))
    if isempty(raw)
        cache.points_f[r] = zeros(Float64, 0, cache.pi.n)
        return cache.points_f[r]::Matrix{Float64}
    end

    out = Matrix{Float64}(undef, length(raw), cache.pi.n)
    @inbounds for i in eachindex(raw)
        x = _point_to_floatvec(raw[i], cache.pi.n)
        for j in 1:cache.pi.n
            out[i, j] = x[j]
        end
    end
    cache.points_f[r] = out
    return out
end

@inline function _exact_centroid_from_cache(cache::PolyInBoxCache, r::Int)
    if cache.exact_centroid_ready[r]
        return cache.exact_centroid[r]
    end
    active = _cache_region_active(cache, r)
    c = cache.exact_centroid[r]
    try
        cm = Polyhedra.center_of_mass(_poly_in_box(cache, r))
        i = 0
        for ci in cm
            i += 1
            c[i] = float(ci)
        end
    catch
        lo = cache.aabb_lo[r]
        hi = cache.aabb_hi[r]
        if !active
            fill!(c, 0.0)
        else
            @inbounds for i in 1:cache.pi.n
                c[i] = 0.5 * (lo[i] + hi[i])
            end
        end
    end
    cache.exact_centroid_ready[r] = true
    return c
end

function _precompute_exact_geometry!(cache::PolyInBoxCache;
                                     precompute_facets::Bool=true,
                                     precompute_centroids::Bool=true,
                                     precompute_weights::Bool=true,
                                     facet_tol::Float64=1e-10)
    active = _ensure_all_region_activity!(cache; threaded=true)
    isempty(active) && return cache

    if precompute_weights
        vals = Vector{Float64}(undef, length(active))
        if _should_thread_region_loop(length(active), 256)
            Threads.@threads for i in eachindex(active)
                vals[i] = Polyhedra.volume(_poly_in_box(cache, active[i]))
            end
        else
            @inbounds for i in eachindex(active)
                vals[i] = Polyhedra.volume(_poly_in_box(cache, active[i]))
            end
        end
        @inbounds for i in eachindex(active)
            r = active[i]
            cache.exact_weight[r] = vals[i]
            cache.exact_weight_ready[r] = true
        end
    end

    if precompute_centroids
        if _should_thread_region_loop(length(active), 192)
            Threads.@threads for i in eachindex(active)
                _exact_centroid_from_cache(cache, active[i])
            end
        else
            @inbounds for r in active
                _exact_centroid_from_cache(cache, r)
            end
        end
    end

    if precompute_facets
        if _should_thread_region_loop(length(active), 128)
            Threads.@threads for i in eachindex(active)
                _region_facets(cache, active[i]; tol=facet_tol)
            end
        else
            @inbounds for r in active
                _region_facets(cache, r; tol=facet_tol)
            end
        end
    end

    return cache
end

function _box_from_cache_or_arg(box, cache)
    if box === nothing && cache isa PolyInBoxCache
        return cache.box_q
    end
    return box
end

_cache_box(cache::PolyInBoxCache) = cache.box_q
_region_encoding_cache(pi::PLEncodingMap) = pi.cache

_region_weights_cached(pi::PLEncodingMap; box, closure::Bool=true, cache, kwargs...) =
    region_weights(pi; box=box, cache=cache, closure=closure, kwargs...)

_region_weights_closure(pi::PLEncodingMap; box, closure::Bool=true, kwargs...) =
    region_weights(pi; box=box, closure=closure, kwargs...)

_region_weights_cached(pi::CompiledEncoding{<:PLEncodingMap}; box, closure::Bool=true, cache, kwargs...) =
    _region_weights_cached(_unwrap_encoding(pi); box=box, closure=closure, cache=cache, kwargs...)

_region_weights_closure(pi::CompiledEncoding{<:PLEncodingMap}; box, closure::Bool=true, kwargs...) =
    _region_weights_closure(_unwrap_encoding(pi); box=box, closure=closure, kwargs...)

function _region_bbox_fast(pi::PLEncodingMap, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache=nothing)
    cache isa PolyInBoxCache || return nothing
    _check_cache_compatible(cache, pi, box, closure)
    _cache_region_active(cache, r) || return nothing
    lo0 = cache.aabb_lo[r]
    hi0 = cache.aabb_hi[r]
    lo = Vector{Float64}(undef, pi.n)
    hi = Vector{Float64}(undef, pi.n)
    @inbounds for i in 1:pi.n
        lo[i] = lo0[i]
        hi[i] = hi0[i]
    end
    return (lo, hi)
end

_region_bbox_cached(pi::PLEncodingMap, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache) =
    _region_bbox_fast(pi, r; box=box, strict=strict, closure=closure, cache=cache)

_region_bbox_closure(pi::PLEncodingMap, r::Integer;
    box, strict::Bool=true, closure::Bool=true) =
    region_bbox(pi, r; box=box, closure=closure, strict=strict)

_region_bbox_cached(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache) =
    _region_bbox_cached(_unwrap_encoding(pi), r; box=box, strict=strict, closure=closure, cache=cache)

_region_bbox_closure(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer;
    box, strict::Bool=true, closure::Bool=true) =
    _region_bbox_closure(_unwrap_encoding(pi), r; box=box, strict=strict, closure=closure)

_region_bbox_fast(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache=nothing) =
    _region_bbox_fast(_unwrap_encoding(pi), r; box=box, strict=strict, closure=closure, cache=cache)

function _region_centroid_fast(pi::PLEncodingMap, r::Integer;
    box, method::Symbol=:bbox, closure::Bool=true, cache=nothing)
    cache isa PolyInBoxCache || return nothing
    _check_cache_compatible(cache, pi, box, closure)
    _cache_region_active(cache, r) || return nothing
    if method === :bbox
        lo0 = cache.aabb_lo[r]
        hi0 = cache.aabb_hi[r]
        c = Vector{Float64}(undef, pi.n)
        @inbounds for i in 1:pi.n
            c[i] = (lo0[i] + hi0[i]) / 2.0
        end
        return c
    elseif method === :polyhedra
        c0 = _exact_centroid_from_cache(cache, Int(r))
        c = Vector{Float64}(undef, pi.n)
        @inbounds for i in 1:pi.n
            c[i] = c0[i]
        end
        return c
    end
    return nothing
end

_region_centroid_cached(pi::PLEncodingMap, r::Integer;
    box, method::Symbol=:bbox, closure::Bool=true, cache) =
    _region_centroid_fast(pi, r; box=box, method=method, closure=closure, cache=cache)

_region_centroid_closure(pi::PLEncodingMap, r::Integer;
    box, method::Symbol=:bbox, closure::Bool=true) =
    region_centroid(pi, r; box=box, method=method, closure=closure)

_region_centroid_cached(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer;
    box, method::Symbol=:bbox, closure::Bool=true, cache) =
    _region_centroid_cached(_unwrap_encoding(pi), r; box=box, method=method, closure=closure, cache=cache)

_region_centroid_closure(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer;
    box, method::Symbol=:bbox, closure::Bool=true) =
    _region_centroid_closure(_unwrap_encoding(pi), r; box=box, method=method, closure=closure)

_region_centroid_fast(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer;
    box, method::Symbol=:bbox, closure::Bool=true, cache=nothing) =
    _region_centroid_fast(_unwrap_encoding(pi), r; box=box, method=method, closure=closure, cache=cache)

@inline function _boundary_measure_cache_key(cache::PolyInBoxCache, r::Integer;
    strict::Bool=true, mode::Symbol=:fast, tol::Float64=1e-10)
    a_f, b_f = cache.box_f
    width = maximum(b_f[i] - a_f[i] for i in 1:cache.pi.n)
    dstep = max(1e-8 * width, 1e-10)
    return (Int(r), strict, mode, dstep, tol)
end

@inline function _region_boundary_measure_cached_value(cache::PolyInBoxCache, r::Integer;
    strict::Bool=true, mode::Symbol=:fast, tol::Float64=1e-10)
    key = _boundary_measure_cache_key(cache, r; strict=strict, mode=mode, tol=tol)
    cached = get(cache.boundary_measure, key, nothing)
    cached === nothing || return cached
    breakdown = get(cache.boundary_breakdown, key, nothing)
    breakdown === nothing && return nothing
    s = 0.0
    @inbounds for e in breakdown
        s += e.measure
    end
    cache.boundary_measure[key] = s
    return s
end

function _region_volume_fast(pi::PLEncodingMap, r::Integer; box, closure::Bool=true, cache=nothing)
    cache isa PolyInBoxCache || return nothing
    _check_cache_compatible(cache, pi, box, closure)
    _cache_region_active(cache, r) || return 0.0
    if !cache.exact_weight_ready[r]
        cache.exact_weight[r] = Polyhedra.volume(_poly_in_box(cache, r))
        cache.exact_weight_ready[r] = true
    end
    return cache.exact_weight[r]
end

_region_volume_fast(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer; box, closure::Bool=true, cache=nothing) =
    _region_volume_fast(_unwrap_encoding(pi), r; box=box, closure=closure, cache=cache)

function _region_boundary_measure_fast(pi::PLEncodingMap, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache=nothing)
    cache isa PolyInBoxCache || return nothing
    _check_cache_compatible(cache, pi, box, closure)
    _cache_region_active(cache, r) || return 0.0
    cached = _region_boundary_measure_cached_value(cache, r; strict=strict)
    cached === nothing || return cached
    return region_boundary_measure(pi, r; box=box, strict=strict, closure=closure, cache=cache)
end

_region_boundary_measure_cached(pi::PLEncodingMap, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache) =
    _region_boundary_measure_fast(pi, r; box=box, strict=strict, closure=closure, cache=cache)

_region_boundary_measure_closure(pi::PLEncodingMap, r::Integer;
    box, strict::Bool=true, closure::Bool=true) =
    region_boundary_measure(pi, r; box=box, closure=closure, strict=strict)

_region_boundary_measure_cached(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache) =
    _region_boundary_measure_cached(_unwrap_encoding(pi), r; box=box, strict=strict, closure=closure, cache=cache)

_region_boundary_measure_closure(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer;
    box, strict::Bool=true, closure::Bool=true) =
    _region_boundary_measure_closure(_unwrap_encoding(pi), r; box=box, strict=strict, closure=closure)

function _region_boundary_measure_fast(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache=nothing)
    return _region_boundary_measure_fast(_unwrap_encoding(pi), r;
        box=box, strict=strict, closure=closure, cache=cache)
end

function _region_circumradius_fast(pi::PLEncodingMap, r::Integer;
    box, center=:bbox, metric::Symbol=:L2, method::Symbol=:bbox,
    strict::Bool=true, closure::Bool=true, cache=nothing)
    cache isa PolyInBoxCache || return nothing
    method === :bbox || return nothing
    _check_cache_compatible(cache, pi, box, closure)
    _cache_region_active(cache, r) || return 0.0
    c = if center === :bbox
        _region_centroid_fast(pi, r; box=box, method=:bbox, closure=closure, cache=cache)
    elseif center === :centroid
        _region_centroid_fast(pi, r; box=box, method=:polyhedra, closure=closure, cache=cache)
    else
        return nothing
    end
    bb = _region_bbox_fast(pi, r; box=box, strict=strict, closure=closure, cache=cache)
    bb === nothing && return 0.0
    lo, hi = bb
    return _bbox_circumradius_local(lo, hi, c, metric)
end

_region_circumradius_fast(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer;
    box, center=:bbox, metric::Symbol=:L2, method::Symbol=:bbox,
    strict::Bool=true, closure::Bool=true, cache=nothing) =
    _region_circumradius_fast(_unwrap_encoding(pi), r;
        box=box, center=center, metric=metric, method=method,
        strict=strict, closure=closure, cache=cache)

function _region_minkowski_functionals_fast(pi::PLEncodingMap, r::Integer;
    box, volume=nothing, boundary=nothing, mean_width_method::Symbol=:auto,
    mean_width_ndirs::Integer=256, mean_width_rng=Random.default_rng(),
    mean_width_directions=nothing, strict::Bool=true, closure::Bool=true,
    cache=nothing)
    cache isa PolyInBoxCache || return nothing
    _check_cache_compatible(cache, pi, box, closure)
    V = volume === nothing ? float(_region_volume_fast(pi, r; box=box, closure=closure, cache=cache)) : float(volume)
    S = boundary === nothing ? float(region_boundary_measure(pi, r;
        box=box, strict=strict, closure=closure, cache=cache)) : float(boundary)
    mw = if (mean_width_method === :auto || mean_width_method === :cauchy) && pi.n == 2
        S / Base.MathConstants.pi
    else
        region_mean_width(pi, r; box=box, method=mean_width_method,
            ndirs=mean_width_ndirs, rng=mean_width_rng,
            directions=mean_width_directions, strict=strict,
            closure=closure, cache=cache)
    end
    return (volume=V, boundary_measure=S, mean_width=float(mw))
end

_region_minkowski_functionals_fast(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer;
    box, volume=nothing, boundary=nothing, mean_width_method::Symbol=:auto,
    mean_width_ndirs::Integer=256, mean_width_rng=Random.default_rng(),
    mean_width_directions=nothing, strict::Bool=true, closure::Bool=true,
    cache=nothing) =
    _region_minkowski_functionals_fast(_unwrap_encoding(pi), r;
        box=box, volume=volume, boundary=boundary,
        mean_width_method=mean_width_method, mean_width_ndirs=mean_width_ndirs,
        mean_width_rng=mean_width_rng, mean_width_directions=mean_width_directions,
        strict=strict, closure=closure, cache=cache)

function _region_geometry_summary_fast(pi::PLEncodingMap, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache=nothing,
    mean_width_method::Symbol=:auto, mean_width_ndirs::Integer=256,
    mean_width_rng=Random.default_rng(), mean_width_directions=nothing,
    need_mean_width::Bool=false)
    cache0 = cache isa PolyInBoxCache ? cache : _auto_geometry_cache(
        pi, cache, box, closure, :fast;
        level=:geometry,
        intent=:single_region_exact,
    )
    box0 = _box_from_cache_or_arg(box, cache0)
    box0 === nothing && return nothing
    cache0 isa PolyInBoxCache && _check_cache_compatible(cache0, pi, box0, closure)
    V = _region_volume_fast(pi, r; box=box0, closure=closure, cache=cache0)
    V === nothing && return nothing
    S = _region_boundary_measure_fast(pi, r; box=box0, strict=strict, closure=closure, cache=cache0)
    S === nothing && return nothing
    mw = if !need_mean_width
        NaN
    elseif (mean_width_method === :auto || mean_width_method === :cauchy) && pi.n == 2
        float(S) / Base.MathConstants.pi
    else
        region_mean_width(pi, r; box=box0, method=mean_width_method,
            ndirs=mean_width_ndirs, rng=mean_width_rng,
            directions=mean_width_directions, strict=strict,
            closure=closure, cache=cache0)
    end
    # Boundary-to-volume and Minkowski callers only consume these scalars.
    # Avoid eagerly computing exact cached-poly centroid/circumradius payloads.
    return (volume=float(V), boundary_measure=float(S), mean_width=float(mw))
end

_region_geometry_summary_fast(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer;
    box, strict::Bool=true, closure::Bool=true, cache=nothing,
    mean_width_method::Symbol=:auto, mean_width_ndirs::Integer=256,
    mean_width_rng=Random.default_rng(), mean_width_directions=nothing,
    need_mean_width::Bool=false) =
    _region_geometry_summary_fast(_unwrap_encoding(pi), r;
        box=box, strict=strict, closure=closure, cache=cache,
        mean_width_method=mean_width_method, mean_width_ndirs=mean_width_ndirs,
        mean_width_rng=mean_width_rng, mean_width_directions=mean_width_directions,
        need_mean_width=need_mean_width)

@inline function _bucket_candidates(cache::PolyInBoxCache, x::AbstractVector{<:Real})
    if !cache.bucket_enabled || length(x) < 1
        return nothing
    end
    x1 = float(x[1])
    x1 < cache.box_f[1][1] && return nothing
    x1 > cache.box_f[2][1] && return nothing
    ix = clamp(Int(floor((x1 - cache.bucket_x0) / cache.bucket_dx)) + 1, 1, cache.bucket_nx)
    iy = 1
    if cache.pi.n >= 2
        x2 = float(x[2])
        x2 < cache.box_f[1][2] && return nothing
        x2 > cache.box_f[2][2] && return nothing
        iy = clamp(Int(floor((x2 - cache.bucket_y0) / cache.bucket_dy)) + 1, 1, cache.bucket_ny)
    end
    return cache.bucket_regions[(iy - 1) * cache.bucket_nx + ix]
end

@inline function _bucket_candidates_col(cache::PolyInBoxCache, X::AbstractMatrix{<:Real}, col::Int)
    if !cache.bucket_enabled || size(X, 1) < 1
        return nothing
    end
    x1 = float(X[1, col])
    x1 < cache.box_f[1][1] && return nothing
    x1 > cache.box_f[2][1] && return nothing
    ix = clamp(Int(floor((x1 - cache.bucket_x0) / cache.bucket_dx)) + 1, 1, cache.bucket_nx)
    iy = 1
    if cache.pi.n >= 2
        x2 = float(X[2, col])
        x2 < cache.box_f[1][2] && return nothing
        x2 > cache.box_f[2][2] && return nothing
        iy = clamp(Int(floor((x2 - cache.bucket_y0) / cache.bucket_dy)) + 1, 1, cache.bucket_ny)
    end
    return cache.bucket_regions[(iy - 1) * cache.bucket_nx + ix]
end

function _region_facets(cache::PolyInBoxCache, r::Integer; tol::Float64=1e-12)
    (1 <= r <= length(cache.facets)) || error("_region_facets: region index out of range")
    fr = cache.facets[r]
    fr === nothing || return fr

    n = cache.pi.n
    vre = _vrep_in_box(cache, r)
    hre = _hrep_in_box(cache, r)
    pts = collect(Polyhedra.points(vre))
    pts_f = _points_float_in_box(cache, r)
    hf = _hrep_float_in_box(cache, r)
    hs = collect(Polyhedra.halfspaces(hre))
    seen = Set{Tuple{Vararg{Int}}}()
    out = CachedFacet[]
    for (hidx, h) in enumerate(hs)
        idxs = _incident_vertex_indices(vre, pts, h;
                                        tol=tol,
                                        a_float=@view(hf.A[hidx, :]),
                                        b_float=hf.b[hidx],
                                        pts_float=pts_f)
        length(idxs) < n && continue
        sig = Tuple(sort(idxs))
        sig in seen && continue
        push!(seen, sig)

        face = Vector{Vector{Float64}}(undef, length(idxs))
        for (k, i) in enumerate(idxs)
            face[k] = _point_to_floatvec(pts[i], n)
        end

        x0 = zeros(Float64, n)
        for p in face
            @inbounds for i in 1:n
                x0[i] += p[i]
            end
        end
        x0 ./= length(face)

        normal = Vector{Float64}(undef, n)
        @inbounds for i in 1:n
            normal[i] = hf.A[hidx, i]
        end
        nn = sqrt(sum(normal[i]^2 for i in 1:n))
        nn == 0.0 && continue
        unit = normal ./ nn
        m = _facet_measure(face, normal)
        push!(out, CachedFacet(float(m), normal, unit, x0))
    end
    cache.facets[r] = out
    return out
end

@inline function _membership_mats(pi::PLEncodingMap, cache, use_relaxed_b::Bool)
    if cache isa PolyInBoxCache
        return cache.Af, (use_relaxed_b ? cache.bf_relaxed : cache.bf_strict)
    end
    return pi.Af, (use_relaxed_b ? pi.bf_relaxed : pi.bf_strict)
end

@inline function _facet_classify_scratch!(cache::PolyInBoxCache, npts::Int)
    tid = Threads.threadid()
    scratch0 = cache.facet_classify_scratch[tid]
    X = scratch0.X
    n = cache.pi.n
    if size(X, 1) != n || size(X, 2) < npts
        X = Matrix{Float64}(undef, n, npts)
        scratch0.X = X
    end
    in_box = scratch0.in_box
    length(in_box) >= npts || resize!(in_box, npts)
    loc = scratch0.loc
    length(loc) >= npts || resize!(loc, npts)
    return scratch0
end

@inline function _locate_region_float(pi::PLEncodingMap,
                                      Af::Vector{Matrix{Float64}},
                                      bf::Vector{Vector{Float64}},
                                      x::AbstractVector{<:Real};
                                      use_multiproj::Bool=false,
                                      tol::Float64=1e-12)
    nregions = length(Af)
    cands = _spatial_prefilter_candidates(pi.prefilter, x)
    if cands !== nothing && !isempty(cands)
        @inbounds for t in cands
            if _in_hpoly_float(Af[t], bf[t], x; tol=tol)
                return t
            end
        end
    end
    mp = pi.prefilter.multiproj
    if use_multiproj && mp.enabled
        rngs = _multiproj_ranges(mp, x)
        if rngs !== nothing
            lo1, hi1, lo2, hi2, lo3, hi3, base = rngs
            pbase = mp.perm[base]
            lo = (base == 1 ? lo1 : (base == 2 ? lo2 : lo3))
            hi = (base == 1 ? hi1 : (base == 2 ? hi2 : hi3))
            base_len = hi - lo + 1
            prefer_len = (pi.prefilter.enabled ? pi.prefilter.window : 64)
            if base_len <= prefer_len
                @inbounds for k in lo:hi
                    t = pbase[k]
                    _multiproj_accept(mp, t, base, lo1, hi1, lo2, hi2, lo3, hi3) || continue
                    if _in_hpoly_float(Af[t], bf[t], x; tol=tol)
                        return t
                    end
                end
            end
        end
    end
    if pi.prefilter.enabled
        x1 = float(x[1])
        c = searchsortedlast(pi.prefilter.x1_sorted, x1)
        lo = max(1, c - pi.prefilter.window)
        hi = min(nregions, c + pi.prefilter.window)
        @inbounds for k in lo:hi
            t = pi.prefilter.perm[k]
            if _in_hpoly_float(Af[t], bf[t], x; tol=tol)
                return t
            end
        end
        @inbounds for t in 1:nregions
            p = pi.prefilter.pos[t]
            if p >= lo && p <= hi
                continue
            end
            if _in_hpoly_float(Af[t], bf[t], x; tol=tol)
                return t
            end
        end
        return 0
    end
    @inbounds for t in 1:nregions
        if _in_hpoly_float(Af[t], bf[t], x; tol=tol)
            return t
        end
    end
    return 0
end

@inline function _locate_probe_batch_pi!(
    loc::Vector{Int},
    in_box::BitVector,
    pi::PLEncodingMap,
    X::Matrix{Float64},
    npts::Int,
    mode0::Symbol;
    strict::Bool,
    tol::Float64,
)
    do_thread = _should_thread_locate_many(pi, nothing, npts; grouped=false)
    if mode0 === :fast && !strict
        Af, bf = _membership_mats(pi, nothing, true)
        use_multiproj = _should_use_multiproj_prefilter(
            pi.prefilter,
            length(pi.regions),
            npts,
            do_thread,
        )
        if do_thread
            Threads.@threads :static for c in 1:npts
                if in_box[c]
                    loc[c] = _locate_region_float(
                        pi, Af, bf, @view(X[:, c]);
                        use_multiproj=use_multiproj,
                        tol=tol,
                    )
                else
                    loc[c] = 0
                end
            end
        else
            @inbounds for c in 1:npts
                if in_box[c]
                    loc[c] = _locate_region_float(
                        pi, Af, bf, @view(X[:, c]);
                        use_multiproj=use_multiproj,
                        tol=tol,
                    )
                else
                    loc[c] = 0
                end
            end
        end
        return loc
    end

    _locate_many_prefix!(loc, pi, X, npts;
                         threaded = do_thread,
                         mode = mode0,
                         tol = LOCATE_FLOAT_TOL,
                         boundary_tol = max(tol, LOCATE_BOUNDARY_TOL))
    @inbounds for c in 1:npts
        in_box[c] || (loc[c] = 0)
    end
    return loc
end

# Internal helpers for region_weights ------------------------------------------------

# Exact region weights via Polyhedra volume computations (if available).
function _region_weights_exact(pi::PLEncodingMap, box; closure::Bool=true, cache=nothing)
    HAVE_POLY || error("method=:exact requires Polyhedra/CDDLib. Use method=:mc instead.")
    nregions = length(pi.regions)
    w = zeros(Float64, nregions)
    if cache isa PolyInBoxCache
        _check_cache_compatible(cache, pi, box, closure)
        active = _ensure_all_region_activity!(cache; threaded=true)
        if !isempty(active)
            missing = Int[]
            sizehint!(missing, length(active))
            @inbounds for r in active
                cache.exact_weight_ready[r] || push!(missing, r)
            end
            if !isempty(missing)
                vals = Vector{Float64}(undef, length(missing))
                if _should_thread_region_loop(length(missing), 256)
                    Threads.@threads for i in eachindex(missing)
                        vals[i] = Polyhedra.volume(_poly_in_box(cache, missing[i]))
                    end
                else
                    @inbounds for i in eachindex(missing)
                        vals[i] = Polyhedra.volume(_poly_in_box(cache, missing[i]))
                    end
                end
                @inbounds for i in eachindex(missing)
                    r = missing[i]
                    cache.exact_weight[r] = vals[i]
                    cache.exact_weight_ready[r] = true
                end
            end
            @inbounds for r in active
                w[r] = cache.exact_weight[r]
            end
        end
        return w
    end

    (ell, u) = box
    n = length(ell)
    ellq = QQ.(ell)
    uq = QQ.(u)
    Aupper = Matrix{QQ}(I, n, n)
    Alower = -Matrix{QQ}(I, n, n)
    bupper = uq
    blower = -ellq

    @inbounds for r in 1:nregions
        A = pi.regions[r].A
        b = closure ? _relaxed_b(pi.regions[r]) : pi.regions[r].b
        Aall = vcat(A, Aupper, Alower)
        ball = vcat(b, bupper, blower)
        p = Polyhedra.polyhedron(Polyhedra.hrep(Aall, ball), CDDLib.Library(:exact))
        w[r] = Polyhedra.volume(p)
    end
    return w
end

# Monte Carlo region weights inside a finite box.
# Returns (weights, stderr, counts).
function _region_weights_mc(pi::PLEncodingMap, box;
    nsamples::Int=100_000,
    rng::AbstractRNG=Random.default_rng(),
    strict::Bool=true,
    closure::Bool=true,
    tol::Real=0.0,
    cache=nothing,
    mode::Symbol=:fast,
)
    (ell, u) = box
    n = length(ell)
    ellf = Float64.(ell)
    uf = Float64.(u)
    nregions = length(pi.regions)

    total_vol = 1.0
    @inbounds for i in 1:n
        total_vol *= (uf[i] - ellf[i])
    end

    mode0 = validate_pl_mode(mode)
    # Fast mode keeps float membership in the hot loop.
    # Verified mode delegates point classification to verified locate.
    Af, bf = _membership_mats(pi, cache, false)
    loc_target = cache isa PolyInBoxCache ? cache : pi

    counts = zeros(Int, nregions)
    x = Vector{Float64}(undef, n)

    for _ in 1:nsamples
        @inbounds for i in 1:n
            x[i] = rand(rng) * (uf[i] - ellf[i]) + ellf[i]
        end
        assigned = false
        if mode0 == :fast
            r = _locate_region_float(pi, Af, bf, x; tol=float(tol))
            if r != 0 && _in_hpoly_float(Af[r], bf[r], x; closure=closure, tol=tol)
                counts[r] += 1
                assigned = true
            end
        else
            r = locate(loc_target, x; mode=:verified, tol=LOCATE_FLOAT_TOL, boundary_tol=LOCATE_BOUNDARY_TOL)
            if r != 0
                inside = closure ? _in_hpoly(pi.regions[r], x) : _in_hpoly_open(pi.regions[r], x)
                if inside
                    counts[r] += 1
                    assigned = true
                end
            end
        end
        if !assigned && strict
            error("region_weights(...): sample fell outside all regions; try strict=false")
        end
    end

    w = total_vol .* (counts ./ nsamples)

    stderr = Vector{Float64}(undef, nregions)
    invn = 1.0 / nsamples
    @inbounds for r in 1:nregions
        p = counts[r] * invn
        stderr[r] = total_vol * sqrt(p * (1.0 - p) * invn)
    end

    return (w, stderr, counts)
end



"""
    region_weights(pi::PLEncodingMap; box=nothing, method=:exact, nsamples=100_000,
        rng=Random.default_rng(), strict=true, return_info=false, alpha=0.05)

Region weights (volumes) for PL polyhedral encodings.

For a finite query `box = (ell, u)`:
- `method=:exact` computes exact polytope volumes (requires optional Polyhedra/CDDLib).
- `method=:mc` estimates volumes by Monte Carlo sampling (always available).

If `return_info=false`, returns only the weight vector.

If `return_info=true`, returns:
- `weights`, `stderr`, `ci`, `alpha`, `method`, `total_volume`, `nsamples`, `counts`.

CI uses Wilson interval on the binomial proportion scaled by `total_volume`.
"""
function region_weights(pi::PLEncodingMap;
    box=nothing,
    cache=nothing,
    method::Symbol=:exact,
    nsamples::Int=100_000,
    rng::AbstractRNG=Random.default_rng(),
    strict::Bool=true,
    closure::Bool=true,
    mode::Symbol=:fast,
    return_info::Bool=false,
    alpha::Real=0.05,
)
    mode0 = validate_pl_mode(mode)
    cache0 = (method == :exact) ? _auto_geometry_cache(
        pi, cache, box, closure, mode0;
        level=:geometry,
        intent=:weights_exact,
    ) : cache
    box = _box_from_cache_or_arg(box, cache0)
    cache0 isa PolyInBoxCache && _check_cache_compatible(cache0, pi, box, closure)
    nregions = length(pi.regions)
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
            mode=mode0,
            total_volume=NaN,
            nsamples=0,
            counts=nothing)
    end

    (ell, u) = box
    n = length(ell)
    total_vol = 1.0
    for i in 1:n
        total_vol *= float(u[i] - ell[i])
    end

    if method == :exact
        w = _region_weights_exact(pi, box; closure=closure, cache=cache0)
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
            mode=mode0,
            total_volume=total_vol,
            nsamples=0,
            counts=nothing)
    elseif method == :mc
        w, stderr, counts = _region_weights_mc(pi, box; nsamples=nsamples, rng=rng,
            strict=strict, closure=closure, cache=cache0, mode=mode0)
        if !return_info
            return w
        end
        ci = Vector{Tuple{Float64, Float64}}(undef, length(w))
        a = float(alpha)
        for i in eachindex(w)
            (plo, phi) = _wilson_interval(counts[i], nsamples; alpha=a)
            ci[i] = (total_vol * plo, total_vol * phi)
        end
        return (weights=w,
            stderr=stderr,
            ci=ci,
            alpha=a,
            method=:mc,
            mode=mode0,
            total_volume=total_vol,
            nsamples=nsamples,
            counts=counts)
    else
        error("Unknown method: $method. Expected :exact or :mc.")
    end
end

function region_volume(pi::PLEncodingMap, r::Integer;
    box=nothing,
    cache=nothing,
    method::Symbol=:exact,
    nsamples::Int=100_000,
    rng::AbstractRNG=Random.default_rng(),
    strict::Bool=true,
    closure::Bool=true,
    mode::Symbol=:fast)
    (1 <= r <= length(pi.regions)) || error("region_volume: region index out of range")
    if box === nothing
        return 1.0
    end
    if method == :exact
        cache0 = _auto_geometry_cache(
            pi, cache, box, closure, validate_pl_mode(mode);
            level=:geometry,
            intent=:single_region_exact,
        )
        box0 = _box_from_cache_or_arg(box, cache0)
        box0 === nothing && error("region_volume: box=(a,b) is required")
        cache0 isa PolyInBoxCache && _check_cache_compatible(cache0, pi, box0, closure)
        fast = _region_volume_fast(pi, r; box=box0, closure=closure, cache=cache0)
        fast === nothing || return float(fast)
        HAVE_POLY || error("region_volume(method=:exact): Polyhedra.jl + CDDLib.jl required")
        return float(Polyhedra.volume(_hpoly_in_box_polyhedron(pi.regions[r], box0; closure=closure)))
    elseif method == :mc
        return float(region_weights(pi; box=box, cache=cache, method=:mc,
            nsamples=nsamples, rng=rng, strict=strict, closure=closure, mode=mode)[Int(r)])
    else
        error("region_volume: unknown method=$method (use :exact or :mc)")
    end
end


# ---- boundary measures, centroids, principal directions, adjacency ------------

# Build (hp intersect box) as an exact Polyhedra.jl polyhedron.
# If closure=true, strict rows are relaxed to closed halfspaces.
function _hpoly_in_box_polyhedron(hp::HPoly, box; closure::Bool=true)
    HAVE_POLY || error("_hpoly_in_box_polyhedron: Polyhedra.jl + CDDLib.jl required")

    a_in, b_in = box
    n = hp.n
    length(a_in) == n || error("_hpoly_in_box_polyhedron: box lower corner has wrong dimension")
    length(b_in) == n || error("_hpoly_in_box_polyhedron: box upper corner has wrong dimension")

    a = Float64[a_in[i] for i in 1:n]
    b = Float64[b_in[i] for i in 1:n]
    any(!isfinite, a) && error("_hpoly_in_box_polyhedron: box lower bounds must be finite")
    any(!isfinite, b) && error("_hpoly_in_box_polyhedron: box upper bounds must be finite")
    any(a .> b) && error("_hpoly_in_box_polyhedron: expected a[i] <= b[i]")

    bvec = closure ? _relaxed_b(hp) : hp.b

    m = size(hp.A, 1)
    Aall = Matrix{QQ}(undef, m + 2 * n, n)
    ball = Vector{QQ}(undef, m + 2 * n)

    for i in 1:m
        for j in 1:n
            Aall[i, j] = hp.A[i, j]
        end
        ball[i] = bvec[i]
    end

    row = m
    for j in 1:n
        row += 1
        for k in 1:n
            Aall[row, k] = (k == j) ? one(QQ) : zero(QQ)
        end
        ball[row] = _toQQ(b_in[j])

        row += 1
        for k in 1:n
            Aall[row, k] = (k == j) ? -one(QQ) : zero(QQ)
        end
        ball[row] = -_toQQ(a_in[j])
    end

    hre = Polyhedra.hrep(Aall, ball)
    return Polyhedra.polyhedron(hre, _CDD)
end

"""
    region_boundary_measure(pi::PLEncodingMap, r; box, closure=true, strict=true, cache=nothing) -> Float64

Compute the (n-1)-dimensional boundary measure of region `r` intersected with `box=(a,b)`.

For n==2 this is a perimeter; for n==3 this is a surface area. Uses Polyhedra.jl
exact `surface` (CDDLib backend).
"""
function region_boundary_measure(pi::PLEncodingMap, r::Integer; box=nothing,
                                 closure::Bool=true, strict::Bool=true,
                                 cache=nothing,
                                 mode::Symbol=:fast)::Float64
    HAVE_POLY || error("region_boundary_measure: Polyhedra.jl + CDDLib.jl required")
    mode0 = validate_pl_mode(mode)
    cache0 = _auto_geometry_cache(
        pi, cache, box, closure, mode0;
        level=:geometry,
        intent=:single_region_exact,
    )
    box = _box_from_cache_or_arg(box, cache0)
    box === nothing && error("region_boundary_measure: please provide box=(a,b)")
    (1 <= r <= length(pi.regions)) || error("region_boundary_measure: region index out of range")
    cache0 isa PolyInBoxCache && _check_cache_compatible(cache0, pi, box, closure)

    tol = 1e-10
    a_f, b_f = if cache0 isa PolyInBoxCache
        cache0.box_f
    else
        a_in, b_in = box
        n = pi.n
        Float64[a_in[i] for i in 1:n], Float64[b_in[i] for i in 1:n]
    end
    width = maximum(b_f[i] - a_f[i] for i in 1:pi.n)
    dstep = max(1e-8 * width, 1e-10)
    key = (Int(r), strict, mode0, dstep, tol)
    if cache0 isa PolyInBoxCache
        cached = get(cache0.boundary_measure, key, nothing)
        cached === nothing || return cached
    end

    # Polyhedra.jl does not implement surface(::CDDLib.Polyhedron).
    # Use our facet-based computation, which already supports caching.
    bd = region_boundary_measure_breakdown(pi, r;
        box=box, cache=cache0, closure=closure, strict=strict, mode=mode0, delta=dstep, tol=tol)

    s = 0.0
    for f in bd
        s += f.measure
    end
    cache0 isa PolyInBoxCache && (cache0.boundary_measure[key] = s)
    return s
end

"""
    region_boundary_measure_breakdown(pi::PLEncodingMap, r;
                                     box=nothing, cache=nothing, closure=true, strict=true,
                                     delta=nothing, tol=1e-10)

Per-facet decomposition of the boundary measure of region `r` intersected with `box=(a,b)`.

Return value: a vector of entries (one per facet). Each entry is a NamedTuple with fields:
  - measure  :: Float64
  - normal   :: Vector{Float64}   (supporting halfspace normal; not necessarily unit)
  - point    :: Vector{Float64}   (barycenter of the facet vertices)
  - kind     :: Symbol            (:internal, :box, :unknown)
  - neighbor :: Union{Nothing,Int}  (neighboring region across the facet, if detectable)

Facet measures are computed by projecting facet vertices to an (n-1)-dimensional coordinate system.
In high dimension this is a floating-point diagnostic.

If `cache` is a `PolyInBoxCache`, the `box` keyword may be omitted.
"""
function region_boundary_measure_breakdown(pi::PLEncodingMap, r::Integer;
                                           box=nothing,
                                           cache=nothing,
                                           closure::Bool=true,
                                           strict::Bool=true,
                                           mode::Symbol=:fast,
                                           delta::Union{Nothing,Real}=nothing,
                                           tol::Float64=1e-10)
    HAVE_POLY || error("region_boundary_measure_breakdown: Polyhedra.jl + CDDLib.jl required")
    mode0 = validate_pl_mode(mode)
    cache0 = _auto_geometry_cache(
        pi, cache, box, closure, mode0;
        level=:geometry,
        intent=:single_region_exact,
    )
    box = _box_from_cache_or_arg(box, cache0)
    box === nothing && error("region_boundary_measure_breakdown: please provide box=(a,b)")
    (1 <= r <= length(pi.regions)) || error("region_boundary_measure_breakdown: region index out of range")
    cache0 isa PolyInBoxCache && _check_cache_compatible(cache0, pi, box, closure)

    n = pi.n
    n >= 2 || error("region_boundary_measure_breakdown: only defined for ambient dimension n >= 2")

    a_f, b_f = if cache0 isa PolyInBoxCache
        cache0.box_f
    else
        a_in, b_in = box
        Float64[a_in[i] for i in 1:n], Float64[b_in[i] for i in 1:n]
    end

    width = maximum(b_f[i] - a_f[i] for i in 1:n)
    dstep = delta === nothing ? max(1e-8 * width, 1e-10) : Float64(delta)

    if cache0 isa PolyInBoxCache
        key = (Int(r), strict, mode0, dstep, tol)
        cached = get(cache0.boundary_breakdown, key, nothing)
        if cached !== nothing
            if !haskey(cache0.boundary_measure, key)
                s_cached = 0.0
                @inbounds for e in cached
                    s_cached += e.measure
                end
                cache0.boundary_measure[key] = s_cached
            end
            return cached
        end
    end

    out = BoundaryBreakdownEntry[]
    if cache0 isa PolyInBoxCache
        key = (Int(r), strict, mode0, dstep, tol)
        facets = _region_facets(cache0, r; tol=tol)
        if isempty(facets)
            cache0.boundary_breakdown[key] = out
            cache0.boundary_measure[key] = 0.0
            return out
        end
        kinds, neigh = _classify_cached_facets(
            pi, cache0, facets, r, a_f, b_f, dstep, mode0;
            strict = strict,
            tol = tol,
            strict_errmsg = "region_boundary_measure_breakdown: uncovered point near facet; try strict=false or closure=true",
        )
        sizehint!(out, length(facets))
        @inbounds for j in eachindex(facets)
            cf = facets[j]
            kind = if kinds[j] == UInt8(1)
                :box
            elseif kinds[j] == UInt8(2)
                :internal
            else
                :unknown
            end
            neighbor = (kinds[j] == UInt8(2)) ? neigh[j] : nothing
            push!(out, (measure=cf.measure, normal=cf.normal, point=cf.point, kind=kind, neighbor=neighbor))
        end
        cache0.boundary_breakdown[key] = out
        s_out = 0.0
        @inbounds for e in out
            s_out += e.measure
        end
        cache0.boundary_measure[key] = s_out
        return out
    else
        P = _hpoly_in_box_polyhedron(pi.regions[r], box; closure=closure)
        vre = Polyhedra.vrep(P)
        hre = Polyhedra.hrep(P)
        pts = collect(Polyhedra.points(vre))
        hs = collect(Polyhedra.halfspaces(hre))
        isempty(pts) && return out
        isempty(hs) && return out

        seen = Set{Tuple{Vararg{Int}}}()
        use_batch = _should_batch_facet_probes(:boundary, length(hs), n)
        if !use_batch
            Af, bf = _membership_mats(pi, cache0, !strict)
            @inline locate_fast(x::Vector{Float64}) = _locate_region_float(pi, Af, bf, x; tol=tol)
            @inline locate_verified(x::Vector{Float64}) = locate(pi, x; mode=:verified, tol=LOCATE_FLOAT_TOL, boundary_tol=max(tol, LOCATE_BOUNDARY_TOL))
            @inline locate_region(x::Vector{Float64}) = mode0 == :fast ? locate_fast(x) : locate_verified(x)

            for h in hs
                # Polyhedra.jl's incidence query is defined for the polyhedron, not for the
                # backend-specific V-representation (e.g. a CDD generator matrix).
                idxs = _incident_vertex_indices(vre, pts, h; tol=tol)
                length(idxs) < n && continue
                sig = Tuple(sort(idxs))
                sig in seen && continue
                push!(seen, sig)

                face = Vector{Vector{Float64}}(undef, length(idxs))
                for (k, i) in enumerate(idxs)
                    face[k] = _point_to_floatvec(pts[i], n)
                end

                x0 = zeros(Float64, n)
                for p in face
                    @inbounds for i in 1:n
                        x0[i] += p[i]
                    end
                end
                x0 ./= length(face)

                normal = Float64[float(c) for c in h.a]
                nn = sqrt(sum(normal[i]^2 for i in 1:n))
                nn == 0.0 && continue
                unit = normal ./ nn

                x_minus = x0 .- dstep .* unit
                x_plus  = x0 .+ dstep .* unit

                in_minus = all(a_f[i] - tol <= x_minus[i] <= b_f[i] + tol for i in 1:n)
                in_plus  = all(a_f[i] - tol <= x_plus[i]  <= b_f[i] + tol for i in 1:n)

                kind = :unknown
                neighbor = nothing

                if !(in_minus && in_plus)
                    kind = :box
                else
                    r1 = locate_region(x_minus)
                    r2 = locate_region(x_plus)
                    if r1 == 0 || r2 == 0
                        strict && error("region_boundary_measure_breakdown: uncovered point near facet; try strict=false or closure=true")
                        kind = :unknown
                        neighbor = nothing
                    else
                        if r1 == r && r2 != r
                            kind = :internal
                            neighbor = r2
                        elseif r2 == r && r1 != r
                            kind = :internal
                            neighbor = r1
                        else
                            kind = :unknown
                            neighbor = nothing
                        end
                    end
                end

                m = _facet_measure(face, normal)
                push!(out, (measure=float(m), normal=normal, point=x0, kind=kind, neighbor=neighbor))
            end
        else
            nf_cap = length(hs)
            sizehint!(out, nf_cap)
            measures = Float64[]
            normals = Vector{Vector{Float64}}()
            points = Vector{Vector{Float64}}()
            sizehint!(measures, nf_cap)
            sizehint!(normals, nf_cap)
            sizehint!(points, nf_cap)

            probes = Matrix{Float64}(undef, n, max(0, 2 * nf_cap))
            in_box = falses(max(0, 2 * nf_cap))
            nf = 0
            @inbounds for h in hs
                idxs = _incident_vertex_indices(vre, pts, h; tol=tol)
                length(idxs) < n && continue
                sig = Tuple(sort(idxs))
                sig in seen && continue
                push!(seen, sig)

                face = Vector{Vector{Float64}}(undef, length(idxs))
                for (k, i) in enumerate(idxs)
                    face[k] = _point_to_floatvec(pts[i], n)
                end

                x0 = zeros(Float64, n)
                for p in face
                    for i in 1:n
                        x0[i] += p[i]
                    end
                end
                x0 ./= length(face)

                normal = Float64[float(c) for c in h.a]
                nn = sqrt(sum(normal[i]^2 for i in 1:n))
                nn == 0.0 && continue
                unit = normal ./ nn

                nf += 1
                c1 = 2nf - 1
                c2 = c1 + 1
                for i in 1:n
                    xi = x0[i]
                    di = dstep * unit[i]
                    probes[i, c1] = xi - di
                    probes[i, c2] = xi + di
                end
                in_box[c1] = _in_box_col(probes, c1, a_f, b_f; tol=tol)
                in_box[c2] = _in_box_col(probes, c2, a_f, b_f; tol=tol)

                push!(measures, float(_facet_measure(face, normal)))
                push!(normals, normal)
                push!(points, x0)
            end

            nf == 0 && return out
            npts = 2 * nf
            loc = Vector{Int}(undef, npts)
            _locate_probe_batch_pi!(loc, in_box, pi, probes, npts, mode0; strict=strict, tol=tol)

            @inbounds for j in 1:nf
                c1 = 2j - 1
                c2 = c1 + 1
                kind = :unknown
                neighbor = nothing
                if !(in_box[c1] && in_box[c2])
                    kind = :box
                else
                    r1 = loc[c1]
                    r2 = loc[c2]
                    if r1 == 0 || r2 == 0
                        strict && error("region_boundary_measure_breakdown: uncovered point near facet; try strict=false or closure=true")
                    elseif r1 == r && r2 != r
                        kind = :internal
                        neighbor = r2
                    elseif r2 == r && r1 != r
                        kind = :internal
                        neighbor = r1
                    end
                end
                push!(out, (measure=measures[j], normal=normals[j], point=points[j], kind=kind, neighbor=neighbor))
            end
        end
    end

    return out
end


"""
    region_centroid(pi::PLEncodingMap, r; box, method=:polyhedra, closure=true, cache=nothing) -> Vector{Float64}

Centroid (center of mass) of region `r` inside `box=(a,b)`.

* method=:polyhedra (default): exact centroid via Polyhedra.center_of_mass when full-dimensional
* method=:bbox: fallback to bounding-box midpoint
"""
function region_centroid(pi::PLEncodingMap, r::Integer; box=nothing,
                         method::Symbol=:polyhedra, closure::Bool=true,
                         cache=nothing)
    cache0 = method == :polyhedra ? _auto_geometry_cache(
        pi, cache, box, closure, :fast;
        level=:geometry,
        intent=:single_region_exact,
    ) : cache
    box = _box_from_cache_or_arg(box, cache0)
    box === nothing && error("region_centroid: please provide box=(a,b)")
    (1 <= r <= length(pi.regions)) || error("region_centroid: region index out of range")

    if method == :bbox
        bb = region_bbox(pi, r; box=box, cache=cache0, closure=closure)
        bb === nothing && return zeros(Float64, pi.n)
        lo, hi = bb
        return 0.5 .* (lo .+ hi)
    elseif method != :polyhedra
        error("region_centroid: unsupported method=$method")
    end

    HAVE_POLY || error("region_centroid(method=:polyhedra): Polyhedra.jl + CDDLib.jl required")
    cache0 isa PolyInBoxCache && _check_cache_compatible(cache0, pi, box, closure)
    if cache0 isa PolyInBoxCache
        if !_cache_region_active(cache0, r)
            return zeros(Float64, pi.n)
        end
        c = _exact_centroid_from_cache(cache0, Int(r))
        out = Vector{Float64}(undef, pi.n)
        @inbounds for i in 1:pi.n
            out[i] = c[i]
        end
        return out
    end

    P = _hpoly_in_box_polyhedron(pi.regions[r], box; closure=closure)

    try
        c = Polyhedra.center_of_mass(P)
        v = Vector{Float64}(undef, pi.n)
        i = 0
        for ci in c
            i += 1
            v[i] = float(ci)
        end
        return v
    catch
        bb = region_bbox(pi, r; box=box)
        bb === nothing && return zeros(Float64, pi.n)
        lo, hi = bb
        return 0.5 .* (lo .+ hi)
    end
end

"""
    region_principal_directions(pi::PLEncodingMap, r; box, nsamples=20_000, rng=Random.default_rng(),
        strict=true, closure=true, max_proposals=10*nsamples,
        return_info=false, nbatches=0, cache=nothing)

Estimate the mean, covariance matrix, and principal directions for region `r`
inside a *finite* window `box = (ell, u)` by rejection sampling.

This PLPolyhedra specialization tests membership ONLY in region `r` using fast
Float64 inequality evaluation (no Polyhedra.jl required).

Return value (always):
- `mean`  : estimated mean of a uniform point in the region (within the box)
- `cov`   : estimated covariance matrix
- `evals` : eigenvalues of `cov`, sorted in descending order
- `evecs` : corresponding eigenvectors (columns), matching `evals`
- `n_accepted`, `n_proposed` : acceptance diagnostics

If `return_info=true`, additional uncertainty estimates are returned:
- `mean_stderr` : coordinatewise standard error for the mean estimate
- `evals_stderr`: standard error for the eigenvalue estimates, based on batching
- `batch_evals` : per-batch eigenvalue vectors (each sorted, descending)
- `batch_n_accepted` : accepted samples per batch
- `nbatches`    : number of batches actually used

Keyword notes:
- `closure` is accepted for API uniformity. For Monte Carlo sampling, the boundary
  has measure zero, so `closure` does not affect the estimate in practice.
- `strict=true` uses the exact H-rep inequalities for membership; `strict=false`
  uses a small relaxation `_relaxed_b(hp)` to reduce numerical false negatives near
  the boundary.
- If `cache isa PolyInBoxCache` and `box === nothing`, we reuse `cache.box_q` for
  convenience (this routine still uses the fast float membership test; it does NOT
  require intersecting polyhedra).

Speed notes:
- Avoids per-sample allocations by preallocating work vectors.
- Uses in-place Welford updates for covariance and optional per-batch covariance.
"""
function region_principal_directions(pi::PLEncodingMap, r::Integer;
    box=nothing,
    nsamples::Int=20_000,
    rng::AbstractRNG=Random.default_rng(),
    strict::Bool=true,
    closure::Bool=true,
    mode::Symbol=:fast,
    max_proposals::Int=10*nsamples,
    return_info::Bool=false,
    nbatches::Int=0,
    cache=nothing
)
    # Allow omitting `box` if the caller passes a PolyInBoxCache (mathematician-friendly).
    box = _box_from_cache_or_arg(box, cache)
    box === nothing && error("region_principal_directions: box=(a,b) is required")

    (1 <= r <= length(pi.regions)) || error("region_principal_directions: region index out of range")

    a_in, b_in = box
    n = pi.n
    length(a_in) == n || error("region_principal_directions: box lower corner has wrong dimension")
    length(b_in) == n || error("region_principal_directions: box upper corner has wrong dimension")

    # Convert box bounds to Float64 once (and validate finiteness).
    a = Vector{Float64}(undef, n)
    b = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        a[i] = float(a_in[i])
        b[i] = float(b_in[i])
    end
    any(!isfinite, a) && error("region_principal_directions: box lower bounds must be finite")
    any(!isfinite, b) && error("region_principal_directions: box upper bounds must be finite")
    any(a .> b) && error("region_principal_directions: expected a[i] <= b[i]")
    mode0 = validate_pl_mode(mode)

    # Region H-polytope data; use Float64 inequalities for the hot membership loop.
    hp = pi.regions[r]
    Af = Float64.(hp.A)
    bf = strict ? Float64.(hp.b) : Float64.(_relaxed_b(hp))
    tol = 1e-12

    # Welford accumulators for mean/cov.
    mu = zeros(Float64, n)
    C = zeros(Float64, n, n)

    # Preallocate work arrays to avoid per-sample allocations.
    x = Vector{Float64}(undef, n)
    delta = Vector{Float64}(undef, n)
    delta2 = Vector{Float64}(undef, n)

    nacc = 0
    nprop = 0

    # Optional batching for uncertainty estimates (standard error of eigenvalues).
    want_batches = return_info ? (nbatches > 0 ? nbatches : 10) : 0
    batch_evals = Vector{Vector{Float64}}()
    batch_n = Int[]
    if want_batches > 0
        sizehint!(batch_evals, want_batches)
        sizehint!(batch_n, want_batches)
    end

    mu_b = want_batches > 0 ? zeros(Float64, n) : Float64[]
    C_b = want_batches > 0 ? zeros(Float64, n, n) : Matrix{Float64}(undef, 0, 0)
    delta_b = want_batches > 0 ? zeros(Float64, n) : Float64[]
    delta2_b = want_batches > 0 ? zeros(Float64, n) : Float64[]
    nacc_b = 0
    batch_target = want_batches > 0 ? max(2, Int(floor(nsamples / want_batches))) : 0

    # Note: `closure` is accepted for API uniformity; we do not branch on it here.
    while (nacc < nsamples) && (nprop < max_proposals)
        nprop += 1
        @inbounds for i in 1:n
            x[i] = a[i] + rand(rng) * (b[i] - a[i])
        end

        if mode0 == :fast
            _in_hpoly_float(Af, bf, x; tol=tol) || continue
        else
            rr = cache isa PolyInBoxCache ? locate(cache, x; mode=:verified) : locate(pi, x; mode=:verified)
            rr == r || continue
        end

        nacc += 1

        # Global Welford update.
        if nacc == 1
            copyto!(mu, x)
        else
            @inbounds for i in 1:n
                delta[i] = x[i] - mu[i]
            end
            @inbounds for i in 1:n
                mu[i] += delta[i] / nacc
                delta2[i] = x[i] - mu[i]
            end
            @inbounds for i in 1:n, j in 1:n
                C[i, j] += delta[i] * delta2[j]
            end
        end

        # Per-batch Welford update (only on accepted samples).
        if want_batches > 0
            nacc_b += 1
            if nacc_b == 1
                copyto!(mu_b, x)
            else
                @inbounds for i in 1:n
                    delta_b[i] = x[i] - mu_b[i]
                end
                @inbounds for i in 1:n
                    mu_b[i] += delta_b[i] / nacc_b
                    delta2_b[i] = x[i] - mu_b[i]
                end
                @inbounds for i in 1:n, j in 1:n
                    C_b[i, j] += delta_b[i] * delta2_b[j]
                end
            end

            if nacc_b >= batch_target
                if nacc_b > 1
                    cov_b = C_b / (nacc_b - 1)
                    E_b = eigen(Symmetric(cov_b))
                    p_b = sortperm(E_b.values, rev=true)
                    push!(batch_evals, E_b.values[p_b])
                    push!(batch_n, nacc_b)
                end
                fill!(mu_b, 0.0)
                fill!(C_b, 0.0)
                nacc_b = 0
            end
        end
    end

    # Final partial batch.
    if want_batches > 0 && nacc_b > 1
        cov_b = C_b / (nacc_b - 1)
        E_b = eigen(Symmetric(cov_b))
        p_b = sortperm(E_b.values, rev=true)
        push!(batch_evals, E_b.values[p_b])
        push!(batch_n, nacc_b)
    end

    if nacc <= 1
        cov = zeros(Float64, n, n)
        evals = zeros(Float64, n)
        evecs = Matrix{Float64}(I, n, n)
        mean_stderr = fill(NaN, n)
        evals_stderr = fill(NaN, n)
    else
        cov = C / (nacc - 1)
        E = eigen(Symmetric(cov))
        p = sortperm(E.values, rev=true)
        evals = E.values[p]
        evecs = E.vectors[:, p]

        mean_stderr = sqrt.(diag(cov) ./ nacc)

        if length(batch_evals) >= 2
            k = length(batch_evals)
            evals_stderr = Vector{Float64}(undef, n)
            for i in 1:n
                s = 0.0
                ss = 0.0
                for bvals in batch_evals
                    v = bvals[i]
                    s += v
                    ss += v * v
                end
                m = s / k
                var = (ss - k * m * m) / (k - 1)
                var = var < 0.0 ? 0.0 : var
                evals_stderr[i] = sqrt(var) / sqrt(k)
            end
        else
            evals_stderr = fill(NaN, n)
        end
    end

    if !return_info
        return (mean=mu, cov=cov, evals=evals, evecs=evecs,
            n_accepted=nacc, n_proposed=nprop)
    end

    return (mean=mu, cov=cov, evals=evals, evecs=evecs,
        mean_stderr=mean_stderr, evals_stderr=evals_stderr,
        batch_evals=batch_evals, batch_n_accepted=batch_n, nbatches=length(batch_evals),
        n_accepted=nacc, n_proposed=nprop)
end



# ------------------------------------------------------------------------------
# Extra region geometry for polyhedral backends:
# - Chebyshev (largest inscribed) ball in L2/Linf/L1 norms (exact when possible)
# - Circumradius about a chosen center (exact via vertices for convex polytopes)
# - Mean width via support function evaluated on vertices
# ------------------------------------------------------------------------------

# Internal: dual norm coefficient for a constraint a cdot x <= b.
# For an r-ball in norm ||.||, the distance to the supporting hyperplane is
# (b - a cdot c) / ||a||_*, where ||.||_* is the dual norm.
@inline function _dual_norm_coeff(metric::Symbol, arow)
    if metric === :L2
        s2 = 0.0
        @inbounds for j in 1:length(arow)
            aj = float(arow[j])
            s2 += aj * aj
        end
        return sqrt(s2)
    elseif metric === :Linf
        # dual of Linf is L1
        s = 0.0
        @inbounds for j in 1:length(arow)
            s += abs(float(arow[j]))
        end
        return s
    elseif metric === :L1
        # dual of L1 is Linf
        m = 0.0
        @inbounds for j in 1:length(arow)
            m = max(m, abs(float(arow[j])))
        end
        return m
    else
        error("dual_norm_coeff: unknown metric=$metric (use :L2, :L1, :Linf)")
    end
end

# Internal: bbox-based circumradius around c.
@inline function _bbox_circumradius_local(lo::AbstractVector, hi::AbstractVector,
    c::AbstractVector, metric::Symbol)
    n = length(lo)
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
        error("bbox_circumradius: unknown metric=$metric (use :L2, :L1, :Linf)")
    end
end

"""
    region_chebyshev_ball(pi::PLEncodingMap, r; box, metric=:L2,
                          method=:auto, strict=true, closure=true,
                          cache=nothing, rng=Random.default_rng(),
                          max_proposals=20000) -> NamedTuple

Compute a Chebyshev (largest inscribed) ball for the convex polytope
`pi.regions[r]` intersected with `box=(a,b)`.

Methods:
- `method=:polyhedra`  exact LP in variables (center, radius) when Polyhedra is available.
                       For `metric=:L2`, LP uses Float64 (norms involve sqrt).
                       For `metric=:Linf` or `:L1`, LP is exact over rationals (QQ).
- `method=:rep`        fast fallback: pick an interior point and take minimum distance to
                       constraints (guaranteed inscribed, not necessarily optimal).
- `method=:auto`       uses `:polyhedra` when available, otherwise `:rep`.

Returns `(center, radius)`.
"""
function region_chebyshev_ball(pi::PLEncodingMap, r::Integer; box=nothing,
    metric::Symbol=:L2, method::Symbol=:auto,
    strict::Bool=true, closure::Bool=true, cache=nothing,
    rng=Random.default_rng(), max_proposals::Integer=20000)

    box === nothing && error("region_chebyshev_ball: box=(a,b) is required")
    a_box, b_box = box
    n = pi.n

    # Choose a default method.
    if method === :auto
        method = HAVE_POLY ? :polyhedra : :rep
    end

    if method === :polyhedra
        HAVE_POLY || error("region_chebyshev_ball(method=:polyhedra) requires Polyhedra")

        hp = pi.regions[r]
        bvec = closure ? _relaxed_b(hp) : hp.b

        # We add box constraints and r>=0.
        m = size(hp.A, 1)
        tot = m + 2*n + 1

        if metric === :L2
            # Float64 LP: A_aug * [c; r] <= b_aug
            Aaug = zeros(Float64, tot, n + 1)
            baug = zeros(Float64, tot)

            row = 0

            # Original region constraints.
            for i in 1:m
                row += 1
                s2 = 0.0
                for j in 1:n
                    aij = float(hp.A[i, j])
                    Aaug[row, j] = aij
                    s2 += aij * aij
                end
                Aaug[row, n + 1] = sqrt(s2)
                baug[row] = float(bvec[i])
            end

            # Box constraints: x_j <= b_j and -x_j <= -a_j.
            for j in 1:n
                row += 1
                Aaug[row, j] = 1.0
                Aaug[row, n + 1] = 1.0
                baug[row] = float(b_box[j])

                row += 1
                Aaug[row, j] = -1.0
                Aaug[row, n + 1] = 1.0
                baug[row] = -float(a_box[j])
            end

            # r >= 0  <=>  -r <= 0
            row += 1
            Aaug[row, n + 1] = -1.0
            baug[row] = 0.0

            poly = Polyhedra.polyhedron(Polyhedra.hrep(Aaug, baug), _CDD_FLOAT)
            vre = Polyhedra.vrep(poly)
            pts = collect(Polyhedra.points(vre))
            isempty(pts) && return (center=copy(pi.reps[r]), radius=0.0)

            best = pts[1]
            best_r = float(best[n + 1])
            for p in pts
                rr = float(p[n + 1])
                if rr > best_r
                    best_r = rr
                    best = p
                end
            end
            c = [float(best[j]) for j in 1:n]
            return (center=c, radius=max(best_r, 0.0))

        elseif metric === :Linf || metric === :L1
            # Exact rational LP for Linf/L1 norms.
            Aaug = Matrix{QQ}(undef, tot, n + 1)
            baug = Vector{QQ}(undef, tot)

            row = 0

            for i in 1:m
                row += 1
                for j in 1:n
                    Aaug[row, j] = hp.A[i, j]
                end

                if metric === :Linf
                    # dual is L1: sum |a_j|
                    s = zero(QQ)
                    for j in 1:n
                        s += abs(hp.A[i, j])
                    end
                    Aaug[row, n + 1] = s
                else
                    # metric == :L1, dual is Linf: max |a_j|
                    mm = zero(QQ)
                    for j in 1:n
                        mm = max(mm, abs(hp.A[i, j]))
                    end
                    Aaug[row, n + 1] = mm
                end

                baug[row] = bvec[i]
            end

            for j in 1:n
                row += 1
                for k in 1:n
                    Aaug[row, k] = zero(QQ)
                end
                Aaug[row, j] = one(QQ)
                Aaug[row, n + 1] = one(QQ)
                baug[row] = QQ(b_box[j])

                row += 1
                for k in 1:n
                    Aaug[row, k] = zero(QQ)
                end
                Aaug[row, j] = -one(QQ)
                Aaug[row, n + 1] = one(QQ)
                baug[row] = -QQ(a_box[j])
            end

            row += 1
            for k in 1:n
                Aaug[row, k] = zero(QQ)
            end
            Aaug[row, n + 1] = -one(QQ)
            baug[row] = zero(QQ)

            poly = Polyhedra.polyhedron(Polyhedra.hrep(Aaug, baug), _CDD)
            vre = Polyhedra.vrep(poly)
            pts = collect(Polyhedra.points(vre))
            isempty(pts) && return (center=copy(pi.reps[r]), radius=0.0)

            best = pts[1]
            best_r = best[n + 1]
            for p in pts
                rr = p[n + 1]
                if rr > best_r
                    best_r = rr
                    best = p
                end
            end
            c = [float(best[j]) for j in 1:n]
            return (center=c, radius=max(float(best_r), 0.0))
        else
            error("region_chebyshev_ball: unknown metric=$metric")
        end
    end

    # Fallback method=:rep (guaranteed inscribed but not necessarily maximal).
    if method !== :rep
        error("region_chebyshev_ball: unknown method=$method (use :auto, :polyhedra, :rep)")
    end

    hp = pi.regions[r]
    bvec = closure ? _relaxed_b(hp) : hp.b

    # Try to find a point in region intersect box. Start from rep clamped to box.
    c = copy(pi.reps[r])
    @inbounds for j in 1:n
        cj = c[j]
        aj = float(a_box[j])
        bj = float(b_box[j])
        c[j] = clamp(float(cj), aj, bj)
    end

    inside = _in_hpoly_float(hp, c; strict=strict, closure=closure)
    if !inside
        # Randomly search for an interior point.
        x = Vector{Float64}(undef, n)
        found = false
        for t in 1:max_proposals
            for j in 1:n
                x[j] = float(a_box[j]) + rand(rng) * (float(b_box[j]) - float(a_box[j]))
            end
            if _in_hpoly_float(hp, x; strict=strict, closure=closure)
                c .= x
                found = true
                break
            end
        end
        if !found
            return (center=c, radius=0.0)
        end
    end

    # Compute min distance to all halfspaces and box faces in the chosen norm.
    rad = Inf
    m = size(hp.A, 1)
    for i in 1:m
        slack = float(bvec[i])
        for j in 1:n
            slack -= float(hp.A[i, j]) * c[j]
        end
        denom = _dual_norm_coeff(metric, view(hp.A, i, :))
        if denom > 0
            rad = min(rad, slack / denom)
        end
    end

    for j in 1:n
        rad = min(rad, c[j] - float(a_box[j]))
        rad = min(rad, float(b_box[j]) - c[j])
    end

    return (center=c, radius=max(rad, 0.0))
end

"""
    region_circumradius(pi::PLEncodingMap, r; box, center=:bbox,
                        metric=:L2, method=:vertices,
                        strict=true, closure=true, cache=nothing,
                        max_combinations=200000, max_vertices=1000000) -> Float64

Circumradius of the convex polytope `pi.regions[r]` intersected with `box=(a,b)`,
about a chosen `center` (bbox center / centroid / chebyshev / explicit vector).

`method=:vertices` is exact for convex polytopes (farthest point is a vertex).
`method=:bbox` is a fast upper bound.
"""
function region_circumradius(pi::PLEncodingMap, r::Integer; box=nothing, center=:bbox,
    metric::Symbol=:L2, method::Symbol=:vertices,
    strict::Bool=true, closure::Bool=true, cache=nothing,
    max_combinations::Integer=200000, max_vertices::Integer=1000000)

    box === nothing && error("region_circumradius: box=(a,b) is required")
    metric = Symbol(metric)
    method = Symbol(method)

    # Choose center.
    c = nothing
    if center === :bbox
        lo, hi = region_bbox(pi, r; box=box, strict=strict, closure=closure, cache=cache,
            max_combinations=max_combinations, max_vertices=max_vertices)
        c = (lo .+ hi) ./ 2
    elseif center === :centroid
        c = region_centroid(pi, r; box=box, strict=strict, closure=closure, cache=cache,
            max_combinations=max_combinations, max_vertices=max_vertices)
    elseif center === :chebyshev
        c = region_chebyshev_ball(pi, r; box=box, metric=metric, strict=strict, closure=closure, cache=cache).center
    else
        c = center
    end

    if method === :bbox
        lo, hi = region_bbox(pi, r; box=box, strict=strict, closure=closure, cache=cache,
            max_combinations=max_combinations, max_vertices=max_vertices)
        return _bbox_circumradius_local(lo, hi, c, metric)
    elseif method !== :vertices
        error("region_circumradius: unknown method=$method (use :vertices or :bbox)")
    end

    HAVE_POLY || error("region_circumradius(method=:vertices) requires Polyhedra")

    # Prefer cached vrep if available.
    pts = nothing
    use_matrix = false
    if cache isa PolyInBoxCache
        _check_cache_compatible(cache, pi, box, closure)
        _cache_region_active(cache, r) || return 0.0
        pts = _points_float_in_box(cache, Int(r))
        use_matrix = true
    else
        hp = pi.regions[r]
        verts = _vertices_of_hpoly_in_box(hp, box; strict=strict, closure=closure,
            max_combinations=max_combinations, max_vertices=max_vertices)
        pts = verts === nothing ? QQ[] : verts
    end

    if use_matrix
        size(pts, 1) == 0 && return 0.0
    else
        isempty(pts) && return 0.0
    end

    rad = 0.0
    n = pi.n
    if use_matrix
        for i in 1:size(pts, 1)
            if metric === :L2
                s2 = 0.0
                for j in 1:n
                    dj = pts[i, j] - c[j]
                    s2 += dj * dj
                end
                rad = max(rad, sqrt(s2))
            elseif metric === :Linf
                dmax = 0.0
                for j in 1:n
                    dmax = max(dmax, abs(pts[i, j] - c[j]))
                end
                rad = max(rad, dmax)
            elseif metric === :L1
                s = 0.0
                for j in 1:n
                    s += abs(pts[i, j] - c[j])
                end
                rad = max(rad, s)
            else
                error("region_circumradius: unknown metric=$metric")
            end
        end
    else
        for p in pts
            if metric === :L2
                s2 = 0.0
                for j in 1:n
                    dj = float(p[j]) - c[j]
                    s2 += dj * dj
                end
                rad = max(rad, sqrt(s2))
            elseif metric === :Linf
                dmax = 0.0
                for j in 1:n
                    dmax = max(dmax, abs(float(p[j]) - c[j]))
                end
                rad = max(rad, dmax)
            elseif metric === :L1
                s = 0.0
                for j in 1:n
                    s += abs(float(p[j]) - c[j])
                end
                rad = max(rad, s)
            else
                error("region_circumradius: unknown metric=$metric")
            end
        end
    end
    return rad
end

# Random unit directions in R^n (columns are unit vectors).
function _random_unit_directions_pl(n::Integer, ndirs::Integer; rng=Random.default_rng())
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
    region_mean_width(pi::PLEncodingMap, r; box, method=:auto, ndirs=256,
                      rng=Random.default_rng(), directions=nothing,
                      strict=true, closure=true, cache=nothing,
                      max_combinations=200000, max_vertices=1000000) -> Float64

Mean width for a convex polytope intersected with a box.

- For n==2 and `method=:auto`, returns the exact convex planar formula:
  mean_width = perimeter / pi.
- For n>=3 (or `method=:vertices`), estimates mean width by sampling directions and
  evaluating support functions exactly on the vertex set.
"""
function region_mean_width(pi::PLEncodingMap, r::Integer; box=nothing,
    method::Symbol=:auto, ndirs::Integer=256,
    nsamples::Integer=0, max_proposals::Integer=0,
    rng=Random.default_rng(), directions=nothing,
    strict::Bool=true, closure::Bool=true, cache=nothing,
    max_combinations::Integer=200000, max_vertices::Integer=1000000)

    box === nothing && error("region_mean_width: box=(a,b) is required")
    n = pi.n
    method = Symbol(method)
    if method === :auto
        method = (n == 2) ? :cauchy : :vertices
    end

    if method === :cauchy
        n == 2 || error("region_mean_width(method=:cauchy) requires n==2, got n=$n")
        return region_perimeter(pi, r; box=box, strict=strict, closure=closure, cache=cache) /
            Base.MathConstants.pi
    elseif method !== :vertices
        error("region_mean_width: unknown method=$method (use :auto, :cauchy, :vertices)")
    end

    HAVE_POLY || error("region_mean_width(method=:vertices) requires Polyhedra")
    cache0 = _auto_geometry_cache(
        pi, cache, box, closure, :fast;
        level=:geometry,
        intent=:single_region_exact,
    )

    # Get vertices of region intersect box.
    pts = nothing
    use_matrix = false
    if cache0 isa PolyInBoxCache
        _check_cache_compatible(cache0, pi, box, closure)
        _cache_region_active(cache0, r) || return 0.0
        pts = _points_float_in_box(cache0, Int(r))
        use_matrix = true
    else
        hp = pi.regions[r]
        verts = _vertices_of_hpoly_in_box(hp, box; strict=strict, closure=closure,
            max_combinations=max_combinations, max_vertices=max_vertices)
        pts = verts === nothing ? QQ[] : verts
    end

    if use_matrix
        size(pts, 1) == 0 && return 0.0
    else
        isempty(pts) && return 0.0
    end

    U = directions === nothing ? _random_unit_directions_pl(n, ndirs; rng=rng) : directions
    size(U, 1) == n || error("region_mean_width: directions must have size (n,ndirs)")

    wsum = 0.0
    for j in 1:size(U, 2)
        u = view(U, :, j)
        maxv = -Inf
        minv = Inf
        if use_matrix
            for irow in 1:size(pts, 1)
                s = 0.0
                for i in 1:n
                    s += u[i] * pts[irow, i]
                end
                maxv = max(maxv, s)
                minv = min(minv, s)
            end
        else
            for p in pts
                s = 0.0
                for i in 1:n
                    s += u[i] * float(p[i])
                end
                maxv = max(maxv, s)
                minv = min(minv, s)
            end
        end
        wsum += (maxv - minv)
    end
    return wsum / float(size(U, 2))
end


# ---- adjacency graph helpers ----

function _convex_hull_2d(pts::Vector{NTuple{2,Float64}})
    sort!(pts)
    pts = unique(pts)
    if length(pts) <= 1
        return pts
    end
    function cross(o, a, b)
        return (a[1] - o[1]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[1] - o[1])
    end
    lower = NTuple{2,Float64}[]
    for p in pts
        while length(lower) >= 2 && cross(lower[end - 1], lower[end], p) <= 0.0
            pop!(lower)
        end
        push!(lower, p)
    end
    upper = NTuple{2,Float64}[]
    for p in reverse(pts)
        while length(upper) >= 2 && cross(upper[end - 1], upper[end], p) <= 0.0
            pop!(upper)
        end
        push!(upper, p)
    end
    return vcat(lower[1:end - 1], upper[1:end - 1])
end

function _polygon_area_2d(poly::Vector{NTuple{2,Float64}})
    m = length(poly)
    m < 3 && return 0.0
    s = 0.0
    for i in 1:m
        x1, y1 = poly[i]
        x2, y2 = poly[(i % m) + 1]
        s += x1 * y2 - x2 * y1
    end
    return abs(s) / 2.0
end

function _facet_measure(face_pts::Vector{Vector{Float64}}, normal::Vector{Float64})::Float64
    n = length(normal)
    an = norm(normal)
    an == 0.0 && return 0.0
    aunit = normal ./ an
    d = n - 1
    d <= 0 && return 0.0

    if n == 2
        tx = -aunit[2]
        ty = aunit[1]
        minp = Inf
        maxp = -Inf
        @inbounds for p in face_pts
            s = tx * p[1] + ty * p[2]
            s < minp && (minp = s)
            s > maxp && (maxp = s)
        end
        return (isfinite(minp) && isfinite(maxp)) ? (maxp - minp) : 0.0
    elseif n == 3
        nx, ny, nz = aunit
        rx, ry, rz = if abs(nx) < 0.9
            (1.0, 0.0, 0.0)
        else
            (0.0, 1.0, 0.0)
        end
        ux = ny * rz - nz * ry
        uy = nz * rx - nx * rz
        uz = nx * ry - ny * rx
        un = sqrt(ux^2 + uy^2 + uz^2)
        un == 0.0 && return 0.0
        inv_un = 1.0 / un
        ux *= inv_un
        uy *= inv_un
        uz *= inv_un
        vx = ny * uz - nz * uy
        vy = nz * ux - nx * uz
        vz = nx * uy - ny * ux
        pts2 = Vector{NTuple{2,Float64}}(undef, length(face_pts))
        @inbounds for i in eachindex(face_pts)
            p = face_pts[i]
            pts2[i] = (
                ux * p[1] + uy * p[2] + uz * p[3],
                vx * p[1] + vy * p[2] + vz * p[3],
            )
        end
        hull = _convex_hull_2d(pts2)
        return _polygon_area_2d(hull)
    end

    # IMPORTANT:
    # svd(1 x n) is thin by default in Julia. Thin SVD returns Vt of size 1 x n,
    # so V has only 1 column, and V[:, 2:end] becomes empty. That makes the
    # projected coordinates empty vectors and causes BoundsError at p[1] below.
    #
    # We need a full orthonormal basis of R^n where the first column is parallel
    # to aunit and the remaining n-1 columns span the orthogonal complement.
    F = svd(reshape(aunit, 1, n); full=true)

    V = Matrix(transpose(F.Vt))
    B = V[:, 2:end]  # n x d

    proj = Vector{Vector{Float64}}(undef, length(face_pts))
    for i in eachindex(face_pts)
        proj[i] = vec(transpose(B) * face_pts[i])
    end

    if d == 1
        xs = [p[1] for p in proj]
        return maximum(xs) - minimum(xs)
    elseif d == 2
        pts2 = [(p[1], p[2]) for p in proj]
        hull = _convex_hull_2d(pts2)
        return _polygon_area_2d(hull)
    else
        vre = Polyhedra.vrep(proj)
        P = Polyhedra.polyhedron(vre, _CDD_FLOAT)
        return float(Polyhedra.volume(P))
    end
end

function _in_box(x::Vector{Float64}, a::Vector{Float64}, b::Vector{Float64}; tol::Float64=1e-12)::Bool
    for i in 1:length(x)
        if x[i] < a[i] - tol || x[i] > b[i] + tol
            return false
        end
    end
    return true
end

@inline function _in_box_col(X::AbstractMatrix{Float64}, col::Int,
                             a::Vector{Float64}, b::Vector{Float64};
                             tol::Float64=1e-12)::Bool
    @inbounds for i in 1:size(X, 1)
        xi = X[i, col]
        if xi < a[i] - tol || xi > b[i] + tol
            return false
        end
    end
    return true
end

"""
Classify cached facets for a fixed region `r` by probing on both sides of each
facet normal in a single batched locate pass.

Returns `(kinds, neighbors)` where:
- `kinds[j] == UInt8(1)` means `:box`
- `kinds[j] == UInt8(2)` means `:internal`
- `kinds[j] == UInt8(3)` means `:unknown`
and `neighbors[j]` is the neighboring region id when internal, otherwise 0.
"""
function _classify_cached_facets(
    pi::PLEncodingMap,
    cache::PolyInBoxCache,
    facets::Vector{CachedFacet},
    r::Int,
    a_f::Vector{Float64},
    b_f::Vector{Float64},
    dstep::Float64,
    mode0::Symbol;
    strict::Bool=true,
    tol::Float64=1e-10,
    strict_errmsg::AbstractString="geometry: uncovered point near facet",
)
    nf = length(facets)
    kinds = Vector{UInt8}(undef, nf)
    neighbors = zeros(Int, nf)
    nf == 0 && return kinds, neighbors

    n = pi.n
    if !_should_batch_facet_probes(:cached, nf, n)
        x_minus = Vector{Float64}(undef, n)
        x_plus = Vector{Float64}(undef, n)
        @inbounds for j in 1:nf
            cf = facets[j]
            p = cf.point
            u = cf.unit
            for i in 1:n
                xi = p[i]
                di = dstep * u[i]
                x_minus[i] = xi - di
                x_plus[i] = xi + di
            end
            in_minus = _in_box(x_minus, a_f, b_f; tol=tol)
            in_plus = _in_box(x_plus, a_f, b_f; tol=tol)
            if !(in_minus && in_plus)
                kinds[j] = UInt8(1) # :box
                continue
            end
            r1 = locate(cache, x_minus; mode=mode0, tol=LOCATE_FLOAT_TOL, boundary_tol=max(tol, LOCATE_BOUNDARY_TOL))
            r2 = locate(cache, x_plus; mode=mode0, tol=LOCATE_FLOAT_TOL, boundary_tol=max(tol, LOCATE_BOUNDARY_TOL))
            if r1 == 0 || r2 == 0
                if strict
                    error(strict_errmsg)
                end
                kinds[j] = UInt8(3) # :unknown
                continue
            end
            if r1 == r && r2 != r
                kinds[j] = UInt8(2) # :internal
                neighbors[j] = r2
            elseif r2 == r && r1 != r
                kinds[j] = UInt8(2) # :internal
                neighbors[j] = r1
            else
                kinds[j] = UInt8(3) # :unknown
            end
        end
        return kinds, neighbors
    end

    npts = 2 * nf
    scratch = _facet_classify_scratch!(cache, npts)
    X = scratch.X
    in_box = scratch.in_box
    loc = scratch.loc

    @inbounds for j in 1:nf
        cf = facets[j]
        p = cf.point
        u = cf.unit
        c1 = 2j - 1
        c2 = c1 + 1
        for i in 1:n
            xi = p[i]
            di = dstep * u[i]
            X[i, c1] = xi - di
            X[i, c2] = xi + di
        end
        in_box[c1] = _in_box_col(X, c1, a_f, b_f; tol=tol)
        in_box[c2] = _in_box_col(X, c2, a_f, b_f; tol=tol)
    end

    _locate_many_prefix!(loc, cache, X, npts;
                         threaded = _should_thread_locate_many(pi, cache, npts; grouped=false),
                         mode = mode0,
                         tol = LOCATE_FLOAT_TOL,
                         boundary_tol = max(tol, LOCATE_BOUNDARY_TOL))
    @inbounds for c in 1:npts
        in_box[c] || (loc[c] = 0)
    end

    @inbounds for j in 1:nf
        c1 = 2j - 1
        c2 = c1 + 1
        if !(in_box[c1] && in_box[c2])
            kinds[j] = UInt8(1) # :box
            continue
        end

        r1 = loc[c1]
        r2 = loc[c2]
        if r1 == 0 || r2 == 0
            if strict
                error(strict_errmsg)
            end
            kinds[j] = UInt8(3) # :unknown
            continue
        end
        if r1 == r && r2 != r
            kinds[j] = UInt8(2) # :internal
            neighbors[j] = r2
        elseif r2 == r && r1 != r
            kinds[j] = UInt8(2) # :internal
            neighbors[j] = r1
        else
            kinds[j] = UInt8(3) # :unknown
        end
    end

    return kinds, neighbors
end

const _HALFSPACE_BETA_SYM = Symbol("\u03B2")

"""
    _halfspace_rhs(h) -> rhs

Return the right-hand-side constant term of a Polyhedra.HalfSpace inequality.

Polyhedra.jl has changed the HalfSpace field name for the constant term across versions
(from `b` to a Greek beta field). This helper hides that difference so the rest of the
geometry layer can be version-stable.

ASCII-only note:
We access the beta field via `Symbol("\\u03B2")` to avoid embedding Unicode in source.
"""
@inline function _halfspace_rhs(h)
    if hasproperty(h, :b)
        return getproperty(h, :b)
    elseif hasproperty(h, _HALFSPACE_BETA_SYM)
        return getproperty(h, _HALFSPACE_BETA_SYM)
    else
        # If Polyhedra changes again, this message makes diagnosis immediate.
        error("_halfspace_rhs: unsupported HalfSpace layout. propertynames(h) = $(propertynames(h))")
    end
end

"""
    _incident_vertex_indices(vre, pts, h; tol=1e-12) -> Vector{Int}

Return the indices (1-based, into `pts`) of vertices that lie on the supporting
hyperplane of the halfspace `h`.

This is a Polyhedra-compatibility helper.

Why this exists:
- Polyhedra.jl's `incidentpointindices` API is backend- and version-dependent.
  In particular, some combinations expect a Polyhedra "index object" into the
  H-representation instead of a `HalfSpace` object, and some V-rep objects do not
  carry the incidence interface in the way older Polyhedra versions did.

What we do:
1. Try `Polyhedra.incidentpointindices(vre, h)` if it works.
2. Otherwise compute incidence directly by checking the equality a'x == rhs.
   - If the data are exact (e.g. Rational{BigInt}), this is exact and stable.
   - Otherwise fall back to a floating dot product with tolerance.

Inputs:
- `vre` is the V-representation object (typically `Polyhedra.vrep(P)`).
- `pts` should be `collect(Polyhedra.points(vre))`.
- `h` should be a halfspace (typically from `Polyhedra.halfspaces(hre)`).

This helper is intentionally simple and "math first": it encodes the definition
of a facet as "the set of vertices where the defining inequality is tight".
"""
function _incident_vertex_indices(vre, pts, h;
                                  tol::Float64=1e-12,
                                  a_float::Union{Nothing,AbstractVector{Float64}}=nothing,
                                  b_float::Union{Nothing,Float64}=nothing,
                                  pts_float::Union{Nothing,Matrix{Float64}}=nothing)::Vector{Int}
    # 1) Use Polyhedra's incidence iterator when available.
    try
        return collect(Polyhedra.incidentpointindices(vre, h))
    catch
        # Fall through to direct computation.
    end

    # Pull out the RHS constant in a version-stable way (b vs beta).
    rhs = _halfspace_rhs(h)

    # 2) Direct computation, exact if possible.
    idxs = Int[]
    try
        for (i, p) in enumerate(pts)
            lhs = zero(rhs)
            for (ai, xi) in zip(h.a, p)
                lhs += ai * xi
            end
            if lhs == rhs
                push!(idxs, i)
            end
        end
        return idxs
    catch
        # Fall through to float+tolerance.
    end

    # 3) Float fallback with tolerance.
    a = a_float === nothing ? Float64[float(c) for c in h.a] : a_float
    b = b_float === nothing ? float(rhs) : b_float
    scale = 1.0 + abs(b)
    n = length(a)

    if pts_float !== nothing && size(pts_float, 1) == length(pts) && size(pts_float, 2) == n
        @inbounds for i in 1:size(pts_float, 1)
            s = 0.0
            for j in 1:n
                s = muladd(a[j], pts_float[i, j], s)
            end
            if abs(s - b) <= tol * scale
                push!(idxs, i)
            end
        end
        return idxs
    end

    for (i, p) in enumerate(pts)
        s = 0.0
        j = 0
        for xi in p
            j += 1
            j > n && break
            s += a[j] * float(xi)
        end
        j == n || continue

        if abs(s - b) <= tol * scale
            push!(idxs, i)
        end
    end

    return idxs
end


function _point_to_floatvec(p, n::Int)::Vector{Float64}
    v = Vector{Float64}(undef, n)
    i = 0
    for c in p
        i += 1
        v[i] = float(c)
    end
    i == n || error("_point_to_floatvec: dimension mismatch")
    return v
end

"""
    region_adjacency(pi::PLEncodingMap; box, strict=true)

Robust adjacency graph for polyhedral PL encodings inside `box=(a,b)`.

Edges are keyed by (r,s) with r < s and weighted by the (n-1)-dimensional measure
of the shared interface between regions r and s inside box.

Uses Polyhedra.jl + CDDLib exact facet enumeration and incidence queries.
"""
function region_adjacency(pi::PLEncodingMap; box=nothing, strict::Bool=true,
                          closure::Bool=true, cache=nothing, mode::Symbol=:fast)
    HAVE_POLY || error("region_adjacency: Polyhedra.jl + CDDLib.jl required")
    mode0 = validate_pl_mode(mode)
    cache0 = _auto_geometry_cache(
        pi, cache, box, closure, mode0;
        level=:full,
        intent=:adjacency,
    )
    box = _box_from_cache_or_arg(box, cache0)
    box === nothing && error("region_adjacency: please provide box=(a,b)")
    cache0 isa PolyInBoxCache && _check_cache_compatible(cache0, pi, box, closure)

    a_in, b_in = box
    n = pi.n
    a = Float64[a_in[i] for i in 1:n]
    b = Float64[b_in[i] for i in 1:n]

    nregions = length(pi.regions)
    tol = 1e-12

    width = maximum(b .- a)
    delta = max(1e-8 * width, 1e-10)

    edges = Dict{Tuple{Int,Int},Float64}()

    if cache0 isa PolyInBoxCache
        key = (strict, mode0, delta, tol)
        cached = get(cache0.adjacency, key, nothing)
        cached === nothing || return cached

        active = _ensure_all_region_activity!(cache0; threaded=true)
        # Phase 1: sequentially populate per-region boundary breakdown cache.
        for r in active
            bd_key = (Int(r), strict, mode0, delta, tol)
            if !haskey(cache0.boundary_breakdown, bd_key)
                region_boundary_measure_breakdown(
                    pi, r;
                    box = box,
                    cache = cache0,
                    closure = closure,
                    strict = strict,
                    mode = mode0,
                    delta = delta,
                    tol = tol,
                )
            end
        end

        # Phase 2: read-only extraction into deterministic per-region shards.
        locals = Vector{Vector{_AdjEdgeAcc}}(undef, length(active))
        if _should_thread_region_loop(length(active), 96)
            Threads.@threads for idx in eachindex(active)
                r = active[idx]
                bd = get(cache0.boundary_breakdown, (Int(r), strict, mode0, delta, tol), BoundaryBreakdownEntry[])
                shard = _AdjEdgeAcc[]
                @inbounds for f in bd
                    if f.kind == :internal
                        s = f.neighbor
                        if s !== nothing && r < s
                            push!(shard, _AdjEdgeAcc(r, s, f.measure))
                        end
                    end
                end
                locals[idx] = shard
            end
        else
            @inbounds for idx in eachindex(active)
                r = active[idx]
                bd = get(cache0.boundary_breakdown, (Int(r), strict, mode0, delta, tol), BoundaryBreakdownEntry[])
                shard = _AdjEdgeAcc[]
                @inbounds for f in bd
                    if f.kind == :internal
                        s = f.neighbor
                        if s !== nothing && r < s
                            push!(shard, _AdjEdgeAcc(r, s, f.measure))
                        end
                    end
                end
                locals[idx] = shard
            end
        end

        total = 0
        @inbounds for shard in locals
            total += length(shard)
        end
        if total > 0
            flat = Vector{_AdjEdgeAcc}(undef, total)
            pos = 1
            @inbounds for shard in locals
                for e in shard
                    flat[pos] = e
                    pos += 1
                end
            end
            sort!(flat; by=e -> (e.r, e.s))
            i = 1
            while i <= total
                r = flat[i].r
                s = flat[i].s
                m = 0.0
                while i <= total && flat[i].r == r && flat[i].s == s
                    m += flat[i].m
                    i += 1
                end
                edges[(r, s)] = m
            end
        end
        cache0.adjacency[key] = edges
        return edges
    end

    for r in 1:nregions
        P = _hpoly_in_box_polyhedron(pi.regions[r], box; closure=closure)
        vre = Polyhedra.vrep(P)
        hre = Polyhedra.hrep(P)
        pts = collect(Polyhedra.points(vre))
        hs = collect(Polyhedra.halfspaces(hre))

        use_batch = _should_batch_facet_probes(:adjacency, length(hs), n)
        if !use_batch
            Af, bf = _membership_mats(pi, cache0, !strict)
            @inline locate_fast(x::Vector{Float64}) = _locate_region_float(pi, Af, bf, x; tol=tol)
            @inline locate_verified(x::Vector{Float64}) = locate(pi, x; mode=:verified, tol=LOCATE_FLOAT_TOL, boundary_tol=LOCATE_BOUNDARY_TOL)
            @inline locate_region(x::Vector{Float64}) = mode0 == :fast ? locate_fast(x) : locate_verified(x)
            seen = Set{Tuple{Vararg{Int}}}()
            for h in hs
                # Polyhedra.jl's incidence query is defined for the polyhedron, not for the
                # backend-specific V-representation (e.g. a CDD generator matrix).
                idxs = _incident_vertex_indices(vre, pts, h; tol=tol)
                length(idxs) < n && continue
                sig = Tuple(sort(idxs))
                sig in seen && continue
                push!(seen, sig)

                face = Vector{Vector{Float64}}(undef, length(idxs))
                for (k, id) in enumerate(idxs)
                    face[k] = _point_to_floatvec(pts[id], n)
                end

                x0 = zeros(Float64, n)
                for p in face
                    x0 .+= p
                end
                x0 ./= length(face)

                normal = Vector{Float64}(undef, n)
                i = 0
                for c in h.a
                    i += 1
                    normal[i] = float(c)
                end

                an = norm(normal)
                an == 0.0 && continue
                aunit = normal ./ an
                x_minus = x0 .- delta .* aunit
                x_plus = x0 .+ delta .* aunit

                _in_box(x_minus, a, b; tol=1e-10) || continue
                _in_box(x_plus, a, b; tol=1e-10) || continue

                r_minus = locate_region(x_minus)
                r_plus = locate_region(x_plus)

                if r_minus == r && r_plus != r && r_plus != 0
                    s = r_plus
                elseif r_plus == r && r_minus != r && r_minus != 0
                    s = r_minus
                else
                    if strict && (r_minus == 0 || r_plus == 0)
                        error("region_adjacency: uncovered point near facet; set strict=false to ignore")
                    end
                    continue
                end

                if r < s
                    m = _facet_measure(face, normal)
                    edges[(r, s)] = get(edges, (r, s), 0.0) + m
                end
            end
        else
            seen = Set{Tuple{Vararg{Int}}}()
            nf_cap = length(hs)
            measures = Float64[]
            sizehint!(measures, nf_cap)
            probes = Matrix{Float64}(undef, n, max(0, 2 * nf_cap))
            in_box = falses(max(0, 2 * nf_cap))
            nf = 0

            @inbounds for h in hs
                idxs = _incident_vertex_indices(vre, pts, h; tol=tol)
                length(idxs) < n && continue
                sig = Tuple(sort(idxs))
                sig in seen && continue
                push!(seen, sig)

                face = Vector{Vector{Float64}}(undef, length(idxs))
                for (k, id) in enumerate(idxs)
                    face[k] = _point_to_floatvec(pts[id], n)
                end

                x0 = zeros(Float64, n)
                for p in face
                    for i in 1:n
                        x0[i] += p[i]
                    end
                end
                x0 ./= length(face)

                normal = Vector{Float64}(undef, n)
                i = 0
                for c in h.a
                    i += 1
                    normal[i] = float(c)
                end
                an = norm(normal)
                an == 0.0 && continue
                aunit = normal ./ an

                nf += 1
                c1 = 2nf - 1
                c2 = c1 + 1
                for i in 1:n
                    xi = x0[i]
                    di = delta * aunit[i]
                    probes[i, c1] = xi - di
                    probes[i, c2] = xi + di
                end
                in_box[c1] = _in_box_col(probes, c1, a, b; tol=1e-10)
                in_box[c2] = _in_box_col(probes, c2, a, b; tol=1e-10)
                push!(measures, _facet_measure(face, normal))
            end

            nf == 0 && continue
            npts = 2 * nf
            loc = Vector{Int}(undef, npts)
            _locate_probe_batch_pi!(loc, in_box, pi, probes, npts, mode0; strict=strict, tol=tol)

            @inbounds for j in 1:nf
                c1 = 2j - 1
                c2 = c1 + 1
                in_box[c1] && in_box[c2] || continue
                r_minus = loc[c1]
                r_plus = loc[c2]
                if r_minus == r && r_plus != r && r_plus != 0
                    s = r_plus
                elseif r_plus == r && r_minus != r && r_minus != 0
                    s = r_minus
                else
                    if strict && (r_minus == 0 || r_plus == 0)
                        error("region_adjacency: uncovered point near facet; set strict=false to ignore")
                    end
                    continue
                end
                if r < s
                    m = measures[j]
                    edges[(r, s)] = get(edges, (r, s), 0.0) + m
                end
            end
        end
    end

    return edges
end


# ---- bbox + diameter helpers for convex polyhedral regions -------------------

# Safe binomial count in BigInt (used to decide whether brute-force enumeration is feasible).
function _binomial_big(n::Int, k::Int)::BigInt
    if k < 0 || k > n
        return BigInt(0)
    end
    k = min(k, n - k)
    num = BigInt(1)
    den = BigInt(1)
    for i in 1:k
        num *= BigInt(n - k + i)
        den *= BigInt(i)
    end
    return div(num, den)
end

# Enumerate vertices of (hp intersect box) by brute force:
# choose n constraints, solve Aeq*x = beq, then check all inequalities.
#
# This is intended for small n (typical in multiparameter persistence: n=2 or 3).
# If the number of combinations is too large, return nothing.
function _vertices_of_hpoly_in_box(
    hp::HPoly,
    a::Vector{Float64},
    b::Vector{Float64};
    max_combinations::Int=200_000,
    max_vertices::Int=50_000
)
    n = hp.n
    length(a) == n || error("_vertices_of_hpoly_in_box: dimension mismatch")
    length(b) == n || error("_vertices_of_hpoly_in_box: dimension mismatch")

    m = size(hp.A, 1)
    M = m + 2 * n

    # Guard: too many combinations => do not attempt brute force.
    if _binomial_big(M, n) > BigInt(max_combinations)
        return nothing
    end

    # Build the full inequality system Aall * x <= ball (QQ exact).
    Aall = Matrix{QQ}(undef, M, n)
    ball = Vector{QQ}(undef, M)

    if m > 0
        Aall[1:m, :] .= hp.A
        ball[1:m]    .= hp.b
    end

    row = m + 1
    # Upper bounds: x_i <= b_i
    for i in 1:n
        for j in 1:n
            Aall[row, j] = (j == i) ? one(QQ) : zero(QQ)
        end
        ball[row] = _toQQ(b[i])
        row += 1
    end
    # Lower bounds: -x_i <= -a_i  (i.e. x_i >= a_i)
    for i in 1:n
        for j in 1:n
            Aall[row, j] = (j == i) ? -one(QQ) : zero(QQ)
        end
        ball[row] = -_toQQ(a[i])
        row += 1
    end

    verts = Set{Tuple{Vararg{QQ}}}()
    comb = Vector{Int}(undef, n)

    # Recursive combinations iterator that supports early termination.
    function rec(start::Int, depth::Int)::Bool
        if depth > n
            Aeq = Aall[comb, :]
            beq = ball[comb]
            x = nothing
            try
                x = Aeq \ beq
            catch
                return true
            end

            # Feasibility check (exact).
            for ii in 1:M
                s = zero(QQ)
                for jj in 1:n
                    s += Aall[ii, jj] * x[jj]
                end
                if s > ball[ii]
                    return true
                end
            end

            key = Tuple(x)
            if !(key in verts)
                push!(verts, key)
                if length(verts) > max_vertices
                    return false
                end
            end
            return true
        end

        last = M - (n - depth)
        for i in start:last
            comb[depth] = i
            ok = rec(i + 1, depth + 1)
            ok || return false
        end
        return true
    end

    ok = rec(1, 1)
    ok || return nothing
    return verts
end

"""
    region_bbox(pi::PLEncodingMap, r; box, max_combinations=200000, max_vertices=50000,
                nsamples=20000, rng=Random.default_rng(), tol=1e-12)
        -> Union{Nothing,Tuple{Vector{Float64},Vector{Float64}}}

Bounding box for region `r` inside a user-supplied ambient box `box=(a,b)`.

Strategy:
1) Try exact vertex enumeration of (region intersect box) when feasible (small n).
2) If that is too expensive, fall back to sampling `nsamples` points in the box
   and taking coordinatewise extrema among those that lie in the region.

Return `nothing` if the intersection is empty (or no samples hit the region in fallback).
"""
function region_bbox(
    pi::PLEncodingMap,
    r::Integer;
    box::Union{Nothing,Tuple{AbstractVector{<:Real},AbstractVector{<:Real}}}=nothing,
    cache=nothing,
    closure::Bool=true,
    strict::Bool=true,
    max_combinations::Int=200_000,
    max_vertices::Int=50_000,
    nsamples::Int=20_000,
    rng::AbstractRNG=Random.default_rng(),
    tol::Float64=1e-12
)
    nregions = length(pi.regions)
    (1 <= r <= nregions) || error("region_bbox: region index out of range")
    cache0 = _auto_geometry_cache(
        pi, cache, box, closure, :fast;
        level=:geometry,
        intent=:single_region_exact,
    )
    box = _box_from_cache_or_arg(box, cache0)
    box === nothing && error("region_bbox: box=(a,b) is required for PLEncodingMap (regions may be unbounded)")
    cache0 isa PolyInBoxCache && _check_cache_compatible(cache0, pi, box, closure)

    a_in, b_in = box
    length(a_in) == pi.n || error("region_bbox: box lower corner has wrong dimension")
    length(b_in) == pi.n || error("region_bbox: box upper corner has wrong dimension")

    a = [float(x) for x in a_in]
    b = [float(x) for x in b_in]
    for i in 1:pi.n
        a[i] <= b[i] || error("region_bbox: box must satisfy a[i] <= b[i] for all i")
    end

    if cache0 isa PolyInBoxCache
        _cache_region_active(cache0, r) || return nothing
        lo0 = cache0.aabb_lo[r]
        hi0 = cache0.aabb_hi[r]
        lo = Vector{Float64}(undef, pi.n)
        hi = Vector{Float64}(undef, pi.n)
        @inbounds for i in 1:pi.n
            lo[i] = lo0[i]
            hi[i] = hi0[i]
        end
        return (lo, hi)
    end

    hp = pi.regions[r]

    verts = _vertices_of_hpoly_in_box(hp, a, b;
                                      max_combinations=max_combinations,
                                      max_vertices=max_vertices)
    if verts !== nothing
        isempty(verts) && return nothing

        lo = fill(Inf, pi.n)
        hi = fill(-Inf, pi.n)
        for v in verts
            for i in 1:pi.n
                xi = float(v[i])
                lo[i] = min(lo[i], xi)
                hi[i] = max(hi[i], xi)
            end
        end
        return (lo, hi)
    end

    # Fallback: Monte Carlo bbox estimate inside the ambient box.
    Af = Float64.(hp.A)
    bf = Float64.(hp.b)

    lo = fill(Inf, pi.n)
    hi = fill(-Inf, pi.n)
    x = Vector{Float64}(undef, pi.n)
    hit = false

    nsamples > 0 || return nothing
    for s in 1:nsamples
        for i in 1:pi.n
            x[i] = a[i] + rand(rng) * (b[i] - a[i])
        end
        if _in_hpoly_float(Af, bf, x; tol=tol)
            for i in 1:pi.n
                lo[i] = min(lo[i], x[i])
                hi[i] = max(hi[i], x[i])
            end
            hit = true
        end
    end

    hit || return nothing
    return (lo, hi)
end

"""
    region_diameter(pi::PLEncodingMap, r; metric=:L2, box, method=:bbox, kwargs...) -> Real

Diameter estimates for convex polyhedral regions.

- `method=:bbox` (default): diagonal length of `region_bbox(pi, r; box=box, kwargs...)`.
- `method=:vertices`: exact diameter of (region intersect box), computed as the
  maximum pairwise distance among vertices (only feasible for small vertex counts).
"""
function region_diameter(
    pi::PLEncodingMap,
    r::Integer;
    metric::Symbol=:L2,
    box=nothing,
    cache=nothing,
    closure::Bool=true,
    strict::Bool=true,
    method::Symbol=:bbox,
    max_combinations::Int=200_000,
    max_vertices::Int=50_000,
    max_vertices_for_diameter::Int=2000
)
    if method == :bbox
        bb = region_bbox(pi, r; box=box, cache=cache, closure=closure, strict=strict,
                         max_combinations=max_combinations, max_vertices=max_vertices)
        bb === nothing && return 0.0
        a, b = bb
        if metric == :Linf
            d = 0.0
            for i in 1:length(a)
                li = float(b[i]) - float(a[i])
                isfinite(li) || return Inf
                d = max(d, abs(li))
            end
            return d
        elseif metric == :L2
            acc = 0.0
            for i in 1:length(a)
                li = float(b[i]) - float(a[i])
                isfinite(li) || return Inf
                acc += li * li
            end
            return sqrt(acc)
        elseif metric == :L1
            s = 0.0
            for i in 1:length(a)
                li = float(b[i]) - float(a[i])
                isfinite(li) || return Inf
                s += abs(li)
            end
            return s
        else
            error("region_diameter: metric must be :L1, :L2, or :Linf")
        end
    elseif method == :vertices
        box === nothing && error("region_diameter(method=:vertices): box=(a,b) is required")
        n = pi.n
        X = nothing
        if cache isa PolyInBoxCache
            _check_cache_compatible(cache, pi, box, closure)
            _cache_region_active(cache, r) || return 0.0
            X = _points_float_in_box(cache, Int(r))
            size(X, 1) > max_vertices_for_diameter && return region_diameter(pi, r; metric=metric, box=box, cache=cache,
                                                                              closure=closure, strict=strict, method=:bbox,
                                                                              max_combinations=max_combinations,
                                                                              max_vertices=max_vertices)
        else
            a_in, b_in = box
            a = [float(x) for x in a_in]
            b = [float(x) for x in b_in]

            hp = pi.regions[r]
            verts = _vertices_of_hpoly_in_box(hp, a, b;
                                              max_combinations=max_combinations,
                                              max_vertices=max_vertices)
            # If vertex enumeration is infeasible, fall back to bbox diameter.
            verts === nothing && return region_diameter(pi, r; metric=metric, box=box, cache=cache,
                                                       closure=closure, strict=strict, method=:bbox,
                                                       max_combinations=max_combinations,
                                                       max_vertices=max_vertices)
            isempty(verts) && return 0.0

            vt = collect(verts)
            nv = length(vt)
            nv > max_vertices_for_diameter && return region_diameter(pi, r; metric=metric, box=box, cache=cache,
                                                                     closure=closure, strict=strict, method=:bbox,
                                                                     max_combinations=max_combinations,
                                                                     max_vertices=max_vertices)

            Xloc = Matrix{Float64}(undef, nv, n)
            for i in 1:nv
                v = vt[i]
                for j in 1:n
                    Xloc[i, j] = float(v[j])
                end
            end
            X = Xloc
        end

        dmax = 0.0
        nv = size(X, 1)
        if metric == :L2
            for i in 1:(nv-1)
                for j in (i+1):nv
                    acc = 0.0
                    for k in 1:n
                        dk = X[i, k] - X[j, k]
                        acc += dk * dk
                    end
                    dmax = max(dmax, sqrt(acc))
                end
            end
        elseif metric == :Linf
            for i in 1:(nv-1)
                for j in (i+1):nv
                    dm = 0.0
                    for k in 1:n
                        dm = max(dm, abs(X[i, k] - X[j, k]))
                    end
                    dmax = max(dmax, dm)
                end
            end
        elseif metric == :L1
            for i in 1:(nv-1)
                for j in (i+1):nv
                    s = 0.0
                    for k in 1:n
                        s += abs(X[i, k] - X[j, k])
                    end
                    dmax = max(dmax, s)
                end
            end
        else
            error("region_diameter: metric must be :L1, :L2, or :Linf")
        end

        return dmax
    else
        error("region_diameter: method must be :bbox or :vertices")
    end
end

"""
    region_facet_count(pi::PLEncodingMap, r::Integer) -> Int

Return the number of inequalities in the stored H-representation of region `r`.

This is a quick complexity proxy (it may overcount facets if some inequalities
are redundant).
"""
function region_facet_count(pi::PLEncodingMap, r::Integer)
    (1 <= r <= length(pi.regions)) || error("region_facet_count: invalid region index")
    hp = pi.regions[Int(r)]
    return size(hp.A, 1)
end

"""
    region_vertex_count(pi::PLEncodingMap, r::Integer; box, max_combinations=200000, max_vertices=50000)
        -> Union{Nothing,Int}

Return the number of vertices of the polytope `region(r) intersection box`.

The vertex enumeration is done by checking combinations of active constraints.
If too many combinations are possible, this returns `nothing` (rather than
attempting an intractable enumeration).
"""
function region_vertex_count(
    pi::PLEncodingMap,
    r::Integer;
    box::Tuple{AbstractVector{<:Real},AbstractVector{<:Real}},
    max_combinations::Int=200_000,
    max_vertices::Int=50_000
)
    (1 <= r <= length(pi.regions)) || error("region_vertex_count: invalid region index")
    a_in, b_in = box
    length(a_in) == pi.n || error("region_vertex_count: box lower corner has wrong dimension")
    length(b_in) == pi.n || error("region_vertex_count: box upper corner has wrong dimension")

    a = [float(a_in[i]) for i in 1:pi.n]
    b = [float(b_in[i]) for i in 1:pi.n]

    verts = _vertices_of_hpoly_in_box(
        pi.regions[Int(r)],
        a,
        b;
        max_combinations=max_combinations,
        max_vertices=max_vertices
    )
    verts === nothing && return nothing
    return length(verts)
end



function _uptight_from_signatures(sig_y::Vector{BitVector}, sig_z::Vector{BitVector})
    rN = length(sig_y)
    leq = falses(rN, rN)
    for i in 1:rN
        leq[i,i] = true
    end
    for i in 1:rN, j in 1:rN
        yi, zi = sig_y[i], sig_z[i]
        yj, zj = sig_y[j], sig_z[j]
        leq[i,j] = all(yi .<= yj) && all(zi .<= zj)
    end
    for k in 1:rN, i in 1:rN, j in 1:rN
        leq[i,j] = leq[i,j] || (leq[i,k] && leq[k,j])
    end
    return FiniteFringe.FinitePoset(leq; check=false)
end

function _images_on_P(P::AbstractPoset,
                      sig_y::Vector{BitVector},
                      sig_z::Vector{BitVector},
                      ups_idx::AbstractVector{<:Integer},
                      downs_idx::AbstractVector{<:Integer})
    m = length(ups_idx)
    r = length(downs_idx)

    Uhat = Vector{FiniteFringe.Upset}(undef, m)
    Dhat = Vector{FiniteFringe.Downset}(undef, r)

    for (i, gi) in enumerate(ups_idx)
        mask = BitVector([sig_y[t][gi] == 1 for t in 1:nvertices(P)])
        Uhat[i] = FiniteFringe.upset_closure(P, mask)
    end
    for (j, gj) in enumerate(downs_idx)
        mask = BitVector([sig_z[t][gj] == 0 for t in 1:nvertices(P)])
        Dhat[j] = FiniteFringe.downset_closure(P, mask)
    end

    return Uhat, Dhat
end

function _monomialize_phi(phi::AbstractMatrix{QQ}, Uhat, Dhat)
    m = length(Dhat)
    n = length(Uhat)
    Phi = copy(phi)
    for j in 1:m, i in 1:n
        if !FiniteFringe.intersects(Uhat[i], Dhat[j])
            Phi[j,i] = zero(QQ)
        end
    end
    return Phi
end

"""
    encode_from_PL_fringe(Ups, Downs, Phi_in, opts::EncodingOptions; poset_kind=:signature) -> (P, H, pi)

Encode a single PL fringe presentation specified by explicit upsets, downsets,
and coefficient matrix.

This low-level overload is used by `encode_from_PL_fringe(::PLFringe, ::EncodingOptions)`.

`opts` is required.
- `opts.backend` must be `:auto` or `:pl`.
- `opts.max_regions` caps region enumeration (default: 10_000).
- `opts.strict_eps` controls strict inequality handling in feasibility checks (default: STRICT_EPS_QQ).
"""
function encode_from_PL_fringe(Ups::Vector{PLUpset},
                               Downs::Vector{PLDownset},
                               Phi_in::AbstractMatrix,
                               opts::EncodingOptions;
                               poset_kind::Symbol = :signature)
    if opts.backend != :auto && opts.backend != :pl
        error("encode_from_PL_fringe: EncodingOptions.backend must be :auto or :pl")
    end
    max_regions = opts.max_regions === nothing ? 10_000 : Int(opts.max_regions)
    strict_eps = opts.strict_eps === nothing ? STRICT_EPS_QQ : _toQQ(opts.strict_eps)

    _assert_PL_inputs(Ups, Downs)
    HAVE_POLY || error("Polyhedra/CDDLib not available; install Polyhedra.jl and CDDLib.jl.")

    feasible = enumerate_feasible_regions(Ups, Downs; max_regions=max_regions, strict_eps=strict_eps)

    if isempty(feasible)
        P = FiniteFringe.FinitePoset(reshape(Bool[true], 1, 1))
        Uhat = FiniteFringe.Upset[FiniteFringe.upset_closure(P, BitVector([false])) for _ in 1:length(Ups)]
        Dhat = FiniteFringe.Downset[FiniteFringe.downset_closure(P, BitVector([false])) for _ in 1:length(Downs)]
        Phi0 = zeros(QQ, length(Downs), length(Ups))
        H = FiniteFringe.FringeModule{QQ}(P, Uhat, Dhat, Phi0; field=QQField())
        n0 = (length(Ups) > 0 ? Ups[1].U.n : (length(Downs) > 0 ? Downs[1].D.n : 0))
        pi = PLEncodingMap(n0, BitVector[], BitVector[], HPoly[], Tuple[])
        return P, H, pi
    end

    sigy = Vector{BitVector}(undef, length(feasible))
    sigz = Vector{BitVector}(undef, length(feasible))
    regs = Vector{HPoly}(undef, length(feasible))
    n = feasible[1][3].n
    wits = Vector{Tuple}(undef, length(feasible))
    for (k, rec) in enumerate(feasible)
        sigy[k] = rec[1]
        sigz[k] = rec[2]
        regs[k] = rec[3]
        wits[k] = rec[4]
    end

    if poset_kind == :signature
        P = SignaturePoset(sigy, sigz)
    elseif poset_kind == :dense
        P = _uptight_from_signatures(sigy, sigz)
    else
        error("encode_from_PL_fringe: poset_kind must be :signature or :dense")
    end
    m = length(Ups)
    r = length(Downs)
    Uhat, Dhat = _images_on_P(P, sigy, sigz, 1:m, 1:r)

    Phi = _monomialize_phi(_toQQ_mat(Phi_in), Uhat, Dhat)
    H = FiniteFringe.FringeModule{QQ}(P, Uhat, Dhat, Phi; field=QQField())

    pi = PLEncodingMap(n, sigy, sigz, regs, wits)
    return P, H, pi
end

encode_from_PL_fringe(Ups::Vector{PLUpset},
                      Downs::Vector{PLDownset},
                      Phi_in::AbstractMatrix;
                      opts::EncodingOptions=EncodingOptions(),
                      poset_kind::Symbol = :signature) =
    encode_from_PL_fringe(Ups, Downs, Phi_in, opts;
                          poset_kind = poset_kind)

function encode_from_PL_fringe_with_tag(Ups, Downs, Phi_in, opts::EncodingOptions; poset_kind::Symbol = :signature)
    P, H, pi = encode_from_PL_fringe(Ups, Downs, Phi_in, opts; poset_kind = poset_kind)
    return P, H, pi, :PL
end

encode_from_PL_fringe_with_tag(Ups, Downs, Phi_in;
                               opts::EncodingOptions=EncodingOptions(),
                               poset_kind::Symbol = :signature) =
    encode_from_PL_fringe_with_tag(Ups, Downs, Phi_in, opts;
                                   poset_kind = poset_kind)

"""
    encode_from_PL_fringes(Fs, opts::EncodingOptions; poset_kind=:signature) -> (P, Hs, pi)

Build a single finite encoding poset `P` that simultaneously encodes all PL
birth/death shapes appearing in the input `PLFringe` presentations.

This is the PL/R^n analog of `ZnEncoding.encode_from_flanges` for Z^n:
- form the union of generator shapes across all inputs,
- enumerate feasible Y-signatures for that union,
- build the uptight region poset `P`,
- push each input down to a finite-poset `FringeModule` on `P`.

`opts` is required.
- `opts.backend` must be `:auto` or `:pl`.
- `opts.max_regions` caps region enumeration (default: 10_000).
- `opts.strict_eps` controls strict inequality handling (default: STRICT_EPS_QQ).
- `poset_kind`: `:signature` (structured, default) or `:dense` (materialized `FinitePoset`).

Return values:
- `P`  : the common encoding poset (a finite poset of regions)
- `Hs` : a vector of `FiniteFringe.FringeModule{QQ}`, one per input `PLFringe`
- `pi` : a classifier `pi : R^n -> P` (as `PLEncodingMap`)

Notes:
- The returned `P` is a common refinement. Encoding a single PL fringe alone
  may yield a smaller poset. Use `encode_from_PL_fringe` for that case.
"""
function encode_from_PL_fringes(Fs::AbstractVector{<:PLFringe}, opts::EncodingOptions;
                                poset_kind::Symbol = :signature)
    if opts.backend != :auto && opts.backend != :pl
        error("encode_from_PL_fringes: EncodingOptions.backend must be :auto or :pl")
    end
    max_regions = opts.max_regions === nothing ? 10_000 : Int(opts.max_regions)
    strict_eps = opts.strict_eps === nothing ? STRICT_EPS_QQ : _toQQ(opts.strict_eps)

    length(Fs) > 0 || error("encode_from_PL_fringes: need at least one PLFringe")
    HAVE_POLY || error("Polyhedra/CDDLib not available; install Polyhedra.jl and CDDLib.jl.")

    n = Fs[1].n
    for (k, F) in enumerate(Fs)
        F.n == n || error("encode_from_PL_fringes: fringe $k has ambient dimension $(F.n) but expected $n")
    end

    Ups_all = PLUpset[]
    Downs_all = PLDownset[]

    up_ranges = Vector{UnitRange{Int}}(undef, length(Fs))
    dn_ranges = Vector{UnitRange{Int}}(undef, length(Fs))

    up_off = 0
    dn_off = 0
    for (k, F) in enumerate(Fs)
        m = length(F.Ups)
        r = length(F.Downs)

        append!(Ups_all, F.Ups)
        append!(Downs_all, F.Downs)

        up_ranges[k] = (up_off + 1):(up_off + m)
        dn_ranges[k] = (dn_off + 1):(dn_off + r)

        up_off += m
        dn_off += r
    end

    feasible = enumerate_feasible_regions(Ups_all, Downs_all;
                                          max_regions=max_regions,
                                          strict_eps=strict_eps)

    if isempty(feasible)
        P = FiniteFringe.FinitePoset(reshape(Bool[true], 1, 1))

        Hs = Vector{FiniteFringe.FringeModule{QQ}}(undef, length(Fs))
        for (k, F) in enumerate(Fs)
            Uhat = FiniteFringe.Upset[
                FiniteFringe.upset_closure(P, BitVector([false])) for _ in 1:length(F.Ups)
            ]
            Dhat = FiniteFringe.Downset[
                FiniteFringe.downset_closure(P, BitVector([false])) for _ in 1:length(F.Downs)
            ]
            Phi0 = zeros(QQ, length(F.Downs), length(F.Ups))
            Hs[k] = FiniteFringe.FringeModule{QQ}(P, Uhat, Dhat, Phi0; field=QQField())
        end

        pi = PLEncodingMap(n, BitVector[], BitVector[], HPoly[], Tuple[])
        return P, Hs, pi
    end

    sigy = Vector{BitVector}(undef, length(feasible))
    sigz = Vector{BitVector}(undef, length(feasible))
    regs = Vector{HPoly}(undef, length(feasible))
    nfeas = feasible[1][3].n
    wits = Vector{Tuple}(undef, length(feasible))
    for (k, rec) in enumerate(feasible)
        sigy[k] = rec[1]
        sigz[k] = rec[2]
        regs[k] = rec[3]
        wits[k] = rec[4]
    end

    if poset_kind == :signature
        P = SignaturePoset(sigy, sigz)
    elseif poset_kind == :dense
        P = _uptight_from_signatures(sigy, sigz)
    else
        error("encode_from_PL_fringes: poset_kind must be :signature or :dense")
    end
    pi = PLEncodingMap(nfeas, sigy, sigz, regs, wits)

    Hs = Vector{FiniteFringe.FringeModule{QQ}}(undef, length(Fs))
    for (k, F) in enumerate(Fs)
        Uhat, Dhat = _images_on_P(P, sigy, sigz, up_ranges[k], dn_ranges[k])
        Phi = _monomialize_phi(_toQQ_mat(F.Phi), Uhat, Dhat)
        Hs[k] = FiniteFringe.FringeModule{QQ}(P, Uhat, Dhat, Phi; field=QQField())
    end

    return P, Hs, pi
end

encode_from_PL_fringes(Fs::AbstractVector{<:PLFringe};
                       opts::EncodingOptions=EncodingOptions(),
                       poset_kind::Symbol = :signature) =
    encode_from_PL_fringes(Fs, opts; poset_kind = poset_kind)

# Tuple-friendly overload.
function encode_from_PL_fringes(Fs::Tuple{Vararg{PLFringe}}, opts::EncodingOptions;
                                poset_kind::Symbol = :signature)
    return encode_from_PL_fringes(collect(Fs), opts; poset_kind = poset_kind)
end

encode_from_PL_fringes(Fs::Tuple{Vararg{PLFringe}};
                       opts::EncodingOptions=EncodingOptions(),
                       poset_kind::Symbol = :signature) =
    encode_from_PL_fringes(collect(Fs), opts; poset_kind = poset_kind)

# Small-arity overloads (avoid varargs-then-opts signatures).
function encode_from_PL_fringes(F::PLFringe, opts::EncodingOptions;
                                poset_kind::Symbol = :signature)
    return encode_from_PL_fringes(PLFringe[F], opts; poset_kind = poset_kind)
end

encode_from_PL_fringes(F::PLFringe;
                       opts::EncodingOptions=EncodingOptions(),
                       poset_kind::Symbol = :signature) =
    encode_from_PL_fringes(PLFringe[F], opts; poset_kind = poset_kind)

function encode_from_PL_fringes(F1::PLFringe, F2::PLFringe, opts::EncodingOptions;
                                poset_kind::Symbol = :signature)
    return encode_from_PL_fringes(PLFringe[F1, F2], opts; poset_kind = poset_kind)
end

encode_from_PL_fringes(F1::PLFringe, F2::PLFringe;
                       opts::EncodingOptions=EncodingOptions(),
                       poset_kind::Symbol = :signature) =
    encode_from_PL_fringes(PLFringe[F1, F2], opts; poset_kind = poset_kind)

function encode_from_PL_fringes(F1::PLFringe, F2::PLFringe, F3::PLFringe, opts::EncodingOptions;
                                poset_kind::Symbol = :signature)
    return encode_from_PL_fringes(PLFringe[F1, F2, F3], opts; poset_kind = poset_kind)
end

encode_from_PL_fringes(F1::PLFringe, F2::PLFringe, F3::PLFringe;
                       opts::EncodingOptions=EncodingOptions(),
                       poset_kind::Symbol = :signature) =
    encode_from_PL_fringes(PLFringe[F1, F2, F3], opts; poset_kind = poset_kind)

# -----------------------------------------------------------------------------
# CompiledEncoding forwarding (treat compiled encodings as primary)
# -----------------------------------------------------------------------------

@inline _unwrap_encoding(pi::CompiledEncoding{<:PLEncodingMap}) = pi.pi

@inline function _encoding_cache_from_compiled(pi::CompiledEncoding{<:PLEncodingMap})
    meta = pi.meta
    meta isa EncodingCache && return meta
    if meta isa NamedTuple && hasproperty(meta, :encoding_cache)
        ec = getproperty(meta, :encoding_cache)
        return ec isa EncodingCache ? ec : nothing
    end
    if meta isa AbstractDict && haskey(meta, :encoding_cache)
        ec = meta[:encoding_cache]
        return ec isa EncodingCache ? ec : nothing
    end
    return nothing
end

@inline function _mode_from_kwargs(kwargs)
    return validate_pl_mode(get(kwargs, :mode, :fast))
end

function _compiled_geometry_cache(pi::CompiledEncoding{<:PLEncodingMap}, box, closure::Bool, mode::Symbol;
                                  level::Symbol=:light, intent::Symbol=:generic)
    box === nothing && return nothing
    validate_pl_mode(mode)
    _should_skip_auto_cache(pi.pi, intent) && return nothing
    ec = _encoding_cache_from_compiled(pi)
    return _geometry_cache_from_store!(
        ec === nothing ? pi.pi.cache : ec,
        pi.pi,
        box,
        closure;
        level=level,
        intent=intent,
        precompute_exact=false,
        precompute_facets=false,
        precompute_centroids=false,
    )
end

@inline function _forward_with_auto_cache(f, pi::CompiledEncoding{<:PLEncodingMap};
                                          box=nothing, cache=nothing, closure::Bool=true,
                                          cache_level::Symbol=:light,
                                          cache_intent::Symbol=:generic,
                                          kwargs...)
    if cache === nothing
        cache = _compiled_geometry_cache(
            pi, box, closure, _mode_from_kwargs(kwargs);
            level=cache_level,
            intent=cache_intent,
        )
    end
    return cache === nothing ?
        f(pi.pi; box=box, closure=closure, kwargs...) :
        f(pi.pi; box=box, cache=cache, closure=closure, kwargs...)
end

region_weights(pi::CompiledEncoding{<:PLEncodingMap}; box=nothing, cache=nothing, closure::Bool=true, kwargs...) =
    _forward_with_auto_cache(region_weights, pi; box=box, cache=cache, closure=closure, kwargs...)

region_volume(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer; box=nothing, cache=nothing, closure::Bool=true, kwargs...) =
    _forward_with_auto_cache((p; kws...) -> region_volume(p, r; kws...), pi;
                             box=box, cache=cache, closure=closure,
                             cache_level=:geometry, cache_intent=:single_region_exact, kwargs...)

region_bbox(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer; box=nothing, cache=nothing, closure::Bool=true, kwargs...) =
    _forward_with_auto_cache((p; kws...) -> region_bbox(p, r; kws...), pi;
                             box=box, cache=cache, closure=closure,
                             cache_level=:geometry, cache_intent=:single_region_exact, kwargs...)

region_diameter(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer; box=nothing, cache=nothing, closure::Bool=true, kwargs...) =
    _forward_with_auto_cache((p; kws...) -> region_diameter(p, r; kws...), pi;
                             box=box, cache=cache, closure=closure,
                             cache_level=:geometry, cache_intent=:single_region_exact, kwargs...)

region_boundary_measure(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer; box=nothing, cache=nothing, closure::Bool=true, kwargs...) =
    _forward_with_auto_cache((p; kws...) -> region_boundary_measure(p, r; kws...), pi;
                             box=box, cache=cache, closure=closure,
                             cache_level=:geometry, cache_intent=:single_region_exact, kwargs...)

region_boundary_measure_breakdown(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer; box=nothing, cache=nothing, closure::Bool=true, kwargs...) =
    _forward_with_auto_cache((p; kws...) -> region_boundary_measure_breakdown(p, r; kws...), pi;
                             box=box, cache=cache, closure=closure,
                             cache_level=:geometry, cache_intent=:single_region_exact, kwargs...)

region_centroid(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer; box=nothing, cache=nothing, closure::Bool=true, kwargs...) =
    _forward_with_auto_cache((p; kws...) -> region_centroid(p, r; kws...), pi;
                             box=box, cache=cache, closure=closure,
                             cache_level=:geometry, cache_intent=:single_region_exact, kwargs...)

region_principal_directions(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer; box=nothing, cache=nothing, closure::Bool=true, kwargs...) =
    _forward_with_auto_cache((p; kws...) -> region_principal_directions(p, r; kws...), pi;
                             box=box, cache=cache, closure=closure, kwargs...)

region_chebyshev_ball(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer; box=nothing, cache=nothing, closure::Bool=true, kwargs...) =
    _forward_with_auto_cache((p; kws...) -> region_chebyshev_ball(p, r; kws...), pi;
                             box=box, cache=cache, closure=closure, kwargs...)

region_circumradius(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer; box=nothing, cache=nothing, closure::Bool=true, kwargs...) =
    _forward_with_auto_cache((p; kws...) -> region_circumradius(p, r; kws...), pi;
                             box=box, cache=cache, closure=closure, kwargs...)

region_mean_width(pi::CompiledEncoding{<:PLEncodingMap}, r::Integer; box=nothing, cache=nothing, closure::Bool=true, kwargs...) =
    _forward_with_auto_cache((p; kws...) -> region_mean_width(p, r; kws...), pi;
                             box=box, cache=cache, closure=closure, kwargs...)

region_adjacency(pi::CompiledEncoding{<:PLEncodingMap}; box=nothing, cache=nothing, closure::Bool=true, kwargs...) =
    _forward_with_auto_cache(region_adjacency, pi; box=box, cache=cache, closure=closure,
                             cache_level=:full, cache_intent=:adjacency, kwargs...)


end # module
