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
using ..CoreModules: QQ, AbstractPLikeEncodingMap, _wilson_interval
using Random

import ..CoreModules: locate, region_weights, region_bbox, region_diameter,
                      region_facet_count, region_vertex_count, region_adjacency,
                      region_boundary_measure, region_boundary_measure_breakdown, 
                      region_centroid, region_principal_directions,
                      region_chebyshev_ball, region_circumradius, region_mean_width

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

## NOTE:
## We intentionally keep a *single* canonical constructor shape for HPoly:
##
##     HPoly(n, A, b, poly, strict_mask, strict_eps)
##
## Older internal code used a 4-argument convenience constructor
## `HPoly(n, A, b, poly)` that silently filled in `strict_mask` and `strict_eps`.
## That constructor has been removed to avoid hidden behavior.

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

"""
    encode_from_PL_fringe(F::PLFringe; kwargs...) -> (P, H, pi)

Convenience overload: encode a `PLFringe` directly.
"""
function encode_from_PL_fringe(F::PLFringe; kwargs...)
    return encode_from_PL_fringe(F.Ups, F.Downs, F.Phi; kwargs...)
end


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

    # Canonical constructor: always specify strictness fields explicitly.
    # (The default in this codebase is that all inequalities are weak, i.e. <=.)
    strict_mask = falses(m)
    return HPoly(n, Aqq, bqq, poly, strict_mask, STRICT_EPS_QQ)
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
        if length(pts) > 0
            witness = Vector{Float64}(pts[1])
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

    results = Vector{Tuple{BitVector,BitVector,HPoly,Vector{Float64}}}()

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
                    push!(results, (BitVector(y), BitVector(z), hp, wit === nothing ? Float64[] : wit))
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
    collapsed = Vector{Tuple{BitVector,BitVector,HPoly,Vector{Float64}}}()
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

struct PLEncodingMap <: AbstractPLikeEncodingMap
    n::Int
    sig_y::Vector{BitVector}
    sig_z::Vector{BitVector}
    regions::Vector{HPoly}
    witnesses::Vector{Vector{Float64}}
end

# Locate by scanning stored region reps (works as long as each signature has a nonempty rep).
function locate(pi::PLEncodingMap, x::AbstractVector)
    length(x) == pi.n || error("locate: dimension mismatch")
    for (idx, hp) in enumerate(pi.regions)
        if _in_hpoly(hp, x)
            return idx
        end
    end
    return 0
end

# ------------------------- Region geometry / sizes ----------------------------

# Fast (floating) membership check used for Monte Carlo geometry.
# For sampling-based geometry, exact QQ membership is overkill and can be slow,
# because it must rationalize every sampled Float64 coordinate.
function _in_hpoly_float(Af::AbstractMatrix{<:Real}, bf::AbstractVector{<:Real},
                         x::AbstractVector{<:Real}; tol::Float64=1e-12)::Bool
    m = size(Af, 1)
    n = size(Af, 2)
    length(x) == n || error("_in_hpoly_float: dimension mismatch")
    for i in 1:m
        s = 0.0
        for j in 1:n
            s += float(Af[i, j]) * float(x[j])
        end
        if s > float(bf[i]) + tol
            return false
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
mutable struct PolyInBoxCache
    pi::PLEncodingMap
    box_q::Tuple{Vector{QQ},Vector{QQ}}
    box_f::Tuple{Vector{Float64},Vector{Float64}}
    closure::Bool
    poly::Vector{Any}
    vrep::Vector{Any}
    hrep::Vector{Any}
end

"""
    poly_in_box_cache(pi::PLEncodingMap; box, closure=true) -> PolyInBoxCache

Construct a cache for Polyhedra-based geometry queries inside a fixed window `box=(a,b)`.

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

Notes:
  - Requires Polyhedra.jl + CDDLib.jl.
  - The cache is tied to a specific `(pi, box, closure)`; passing it with
    different parameters throws an error.
"""
function poly_in_box_cache(pi::PLEncodingMap; box, closure::Bool=true)
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
    return PolyInBoxCache(pi, (a_q, b_q), (a_f, b_f), closure,
                          fill(nothing, nregions),
                          fill(nothing, nregions),
                          fill(nothing, nregions))
end

# Clear cached polyhedra / representations (useful in long-running sessions).
function Base.empty!(cache::PolyInBoxCache)
    fill!(cache.poly, nothing)
    fill!(cache.vrep, nothing)
    fill!(cache.hrep, nothing)
    return cache
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

function _poly_in_box(cache::PolyInBoxCache, r::Integer)
    (1 <= r <= length(cache.poly)) || error("_poly_in_box: region index out of range")
    if cache.poly[r] === nothing
        cache.poly[r] = _hpoly_in_box_polyhedron(cache.pi.regions[Int(r)], cache.box_q; closure=cache.closure)
    end
    return cache.poly[r]
end

function _vrep_in_box(cache::PolyInBoxCache, r::Integer)
    (1 <= r <= length(cache.vrep)) || error("_vrep_in_box: region index out of range")
    if cache.vrep[r] === nothing
        cache.vrep[r] = Polyhedra.vrep(_poly_in_box(cache, r))
    end
    return cache.vrep[r]
end

function _hrep_in_box(cache::PolyInBoxCache, r::Integer)
    (1 <= r <= length(cache.hrep)) || error("_hrep_in_box: region index out of range")
    if cache.hrep[r] === nothing
        cache.hrep[r] = Polyhedra.hrep(_poly_in_box(cache, r))
    end
    return cache.hrep[r]
end

function _box_from_cache_or_arg(box, cache)
    if box === nothing && cache isa PolyInBoxCache
        return cache.box_q
    end
    return box
end

# Internal helpers for region_weights ------------------------------------------------

# Exact region weights via Polyhedra volume computations (if available).
function _region_weights_exact(pi::PLEncodingMap, box)
    HAVE_POLYHEDRA_CDDLIB || error("method=:exact requires Polyhedra/CDDLib. Use method=:mc instead.")
    (ell, u) = box
    n = length(ell)

    w = zeros(Float64, pi.nregions)

    ellq = QQ.(ell)
    uq = QQ.(u)
    Aupper = Matrix{QQ}(I, n, n)
    Alower = -Matrix{QQ}(I, n, n)
    bupper = uq
    blower = -ellq

    for r in 1:pi.nregions
        A = pi.regions[r].A
        b = pi.regions[r].b
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
    tol::Real=0.0
)
    (ell, u) = box
    n = length(ell)
    ellf = Float64.(ell)
    uf = Float64.(u)

    total_vol = 1.0
    @inbounds for i in 1:n
        total_vol *= (uf[i] - ellf[i])
    end

    Af = [Float64.(pi.regions[r].A) for r in 1:pi.nregions]
    bf = [Float64.(pi.regions[r].b) for r in 1:pi.nregions]

    counts = zeros(Int, pi.nregions)
    x = Vector{Float64}(undef, n)

    for _ in 1:nsamples
        @inbounds for i in 1:n
            x[i] = rand(rng) * (uf[i] - ellf[i]) + ellf[i]
        end
        assigned = false
        for r in 1:pi.nregions
            if _in_hpoly_float(Af[r], bf[r], x; closure=closure, tol=tol)
                counts[r] += 1
                assigned = true
                break
            end
        end
        if !assigned && strict
            error("region_weights(...): sample fell outside all regions; try strict=false")
        end
    end

    w = total_vol .* (counts ./ nsamples)

    stderr = Vector{Float64}(undef, pi.nregions)
    invn = 1.0 / nsamples
    @inbounds for r in 1:pi.nregions
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
    method::Symbol=:exact,
    nsamples::Int=100_000,
    rng::AbstractRNG=Random.default_rng(),
    strict::Bool=true,
    return_info::Bool=false,
    alpha::Real=0.05,
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
    total_vol = 1.0
    for i in 1:n
        total_vol *= float(u[i] - ell[i])
    end

    if method == :exact
        w = _region_weights_exact(pi, box)
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
    elseif method == :mc
        w, stderr, counts = _region_weights_mc(pi, box; nsamples=nsamples, rng=rng, strict=strict)
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
            total_volume=total_vol,
            nsamples=nsamples,
            counts=counts)
    else
        error("Unknown method: $method. Expected :exact or :mc.")
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
                                 cache=nothing)::Float64
    HAVE_POLY || error("region_boundary_measure: Polyhedra.jl + CDDLib.jl required")
    box = _box_from_cache_or_arg(box, cache)
    box === nothing && error("region_boundary_measure: please provide box=(a,b)")
    (1 <= r <= length(pi.regions)) || error("region_boundary_measure: region index out of range")

    # Polyhedra.jl does not implement surface(::CDDLib.Polyhedron).
    # Use our facet-based computation, which already supports caching.
    bd = region_boundary_measure_breakdown(pi, r;
        box=box, cache=cache, closure=closure, strict=strict)

    s = 0.0
    for f in bd
        s += f.measure
    end
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
                                           delta::Union{Nothing,Real}=nothing,
                                           tol::Float64=1e-10)
    HAVE_POLY || error("region_boundary_measure_breakdown: Polyhedra.jl + CDDLib.jl required")
    box = _box_from_cache_or_arg(box, cache)
    box === nothing && error("region_boundary_measure_breakdown: please provide box=(a,b)")
    (1 <= r <= length(pi.regions)) || error("region_boundary_measure_breakdown: region index out of range")
    cache isa PolyInBoxCache && _check_cache_compatible(cache, pi, box, closure)

    n = pi.n
    n >= 2 || error("region_boundary_measure_breakdown: only defined for ambient dimension n >= 2")

    a_f, b_f = if cache isa PolyInBoxCache
        cache.box_f
    else
        a_in, b_in = box
        Float64[a_in[i] for i in 1:n], Float64[b_in[i] for i in 1:n]
    end

    nregions = length(pi.regions)
    Af = Vector{Matrix{Float64}}(undef, nregions)
    bf = Vector{Vector{Float64}}(undef, nregions)
    for t in 1:nregions
        hp = pi.regions[t]
        Af[t] = Float64.(hp.A)
        bf[t] = strict ? Float64.(hp.b) : Float64.(_relaxed_b(hp))
    end

    function locate_float(x::Vector{Float64})::Int
        for t in 1:nregions
            if _in_hpoly_float(Af[t], bf[t], x; tol=tol)
                return t
            end
        end
        return 0
    end

    P = cache isa PolyInBoxCache ? _poly_in_box(cache, r) :
        _hpoly_in_box_polyhedron(pi.regions[r], box; closure=closure)
    vre = cache isa PolyInBoxCache ? _vrep_in_box(cache, r) : Polyhedra.vrep(P)
    hre = cache isa PolyInBoxCache ? _hrep_in_box(cache, r) : Polyhedra.hrep(P)
    pts = collect(Polyhedra.points(vre))
    hs = collect(Polyhedra.halfspaces(hre))

    Tentry = NamedTuple{(:measure, :normal, :point, :kind, :neighbor),
                        Tuple{Float64, Vector{Float64}, Vector{Float64}, Symbol, Union{Nothing, Int}}}
    isempty(pts) && return Tentry[]
    isempty(hs) && return Tentry[]

    width = maximum(b_f[i] - a_f[i] for i in 1:n)
    dstep = delta === nothing ? max(1e-8 * width, 1e-10) : Float64(delta)

    seen = Set{Tuple{Vararg{Int}}}()
    out = Tentry[]

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
            r1 = locate_float(x_minus)
            r2 = locate_float(x_plus)
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
    box = _box_from_cache_or_arg(box, cache)
    box === nothing && error("region_centroid: please provide box=(a,b)")
    (1 <= r <= length(pi.regions)) || error("region_centroid: region index out of range")

    if method == :bbox
        bb = region_bbox(pi, r; box=box)
        bb === nothing && return zeros(Float64, pi.n)
        lo, hi = bb
        return 0.5 .* (lo .+ hi)
    elseif method != :polyhedra
        error("region_centroid: unsupported method=$method")
    end

    HAVE_POLY || error("region_centroid(method=:polyhedra): Polyhedra.jl + CDDLib.jl required")
    cache isa PolyInBoxCache && _check_cache_compatible(cache, pi, box, closure)
    P = cache isa PolyInBoxCache ? _poly_in_box(cache, r) : _hpoly_in_box_polyhedron(pi.regions[r], box; closure=closure)

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

        _in_hpoly_float(Af, bf, x; tol=tol) || continue

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
    if cache !== nothing
        vrep = _vrep_in_box(cache, r)
        pts = collect(Polyhedra.points(vrep))
    else
        hp = pi.regions[r]
        verts = _vertices_of_hpoly_in_box(hp, box; strict=strict, closure=closure,
            max_combinations=max_combinations, max_vertices=max_vertices)
        pts = verts === nothing ? QQ[] : verts
    end

    isempty(pts) && return 0.0

    rad = 0.0
    n = pi.n
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
        return region_perimeter(pi, r; box=box, strict=strict, closure=closure, cache=cache,
            max_combinations=max_combinations, max_vertices=max_vertices) / pi
    elseif method !== :vertices
        error("region_mean_width: unknown method=$method (use :auto, :cauchy, :vertices)")
    end

    HAVE_POLY || error("region_mean_width(method=:vertices) requires Polyhedra")

    # Get vertices of region intersect box.
    pts = nothing
    if cache !== nothing
        vrep = _vrep_in_box(cache, r)
        pts = collect(Polyhedra.points(vrep))
    else
        hp = pi.regions[r]
        verts = _vertices_of_hpoly_in_box(hp, box; strict=strict, closure=closure,
            max_combinations=max_combinations, max_vertices=max_vertices)
        pts = verts === nothing ? QQ[] : verts
    end

    isempty(pts) && return 0.0

    U = directions === nothing ? _random_unit_directions_pl(n, ndirs; rng=rng) : directions
    size(U, 1) == n || error("region_mean_width: directions must have size (n,ndirs)")

    wsum = 0.0
    for j in 1:size(U, 2)
        u = view(U, :, j)
        maxv = -Inf
        minv = Inf
        for p in pts
            s = 0.0
            for i in 1:n
                s += u[i] * float(p[i])
            end
            maxv = max(maxv, s)
            minv = min(minv, s)
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
function _incident_vertex_indices(vre, pts, h; tol::Float64=1e-12)::Vector{Int}
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
    a = Float64[float(c) for c in h.a]
    b = float(rhs)
    scale = 1.0 + abs(b)

    for (i, p) in enumerate(pts)
        s = 0.0
        j = 0
        for xi in p
            j += 1
            j > length(a) && break
            s += a[j] * float(xi)
        end
        j == length(a) || continue

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
                          closure::Bool=true, cache=nothing)
    HAVE_POLY || error("region_adjacency: Polyhedra.jl + CDDLib.jl required")
    box = _box_from_cache_or_arg(box, cache)
    box === nothing && error("region_adjacency: please provide box=(a,b)")
    cache isa PolyInBoxCache && _check_cache_compatible(cache, pi, box, closure)

    a_in, b_in = box
    n = pi.n
    a = Float64[a_in[i] for i in 1:n]
    b = Float64[b_in[i] for i in 1:n]

    nregions = length(pi.regions)
    Af = Vector{Matrix{Float64}}(undef, nregions)
    bf = Vector{Vector{Float64}}(undef, nregions)
    for t in 1:nregions
        hp = pi.regions[t]
        Af[t] = Float64.(hp.A)
        bf[t] = strict ? Float64.(hp.b) : Float64.(_relaxed_b(hp))
    end

    tol = 1e-12
    function locate_float(x::Vector{Float64})::Int
        for t in 1:nregions
            if _in_hpoly_float(Af[t], bf[t], x; tol=tol)
                return t
            end
        end
        return 0
    end

    width = maximum(b .- a)
    delta = max(1e-8 * width, 1e-10)

    edges = Dict{Tuple{Int,Int},Float64}()
    Pbox = Vector{Any}(undef, nregions)
    for r in 1:nregions
        Pbox[r] = cache isa PolyInBoxCache ? _poly_in_box(cache, r) :
            _hpoly_in_box_polyhedron(pi.regions[r], box; closure=closure)
    end

    for r in 1:nregions
        P = Pbox[r]
        vre = cache isa PolyInBoxCache ? _vrep_in_box(cache, r) : Polyhedra.vrep(P)
        hre = cache isa PolyInBoxCache ? _hrep_in_box(cache, r) : Polyhedra.hrep(P)
        pts = collect(Polyhedra.points(vre))
        hs = collect(Polyhedra.halfspaces(hre))

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

            r_minus = locate_float(x_minus)
            r_plus = locate_float(x_plus)

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
    max_combinations::Int=200_000,
    max_vertices::Int=50_000,
    nsamples::Int=20_000,
    rng::AbstractRNG=Random.default_rng(),
    tol::Float64=1e-12
)
    nregions = length(pi.regions)
    (1 <= r <= nregions) || error("region_bbox: region index out of range")
    box === nothing && error("region_bbox: box=(a,b) is required for PLEncodingMap (regions may be unbounded)")

    a_in, b_in = box
    length(a_in) == pi.n || error("region_bbox: box lower corner has wrong dimension")
    length(b_in) == pi.n || error("region_bbox: box upper corner has wrong dimension")

    a = [float(x) for x in a_in]
    b = [float(x) for x in b_in]
    for i in 1:pi.n
        a[i] <= b[i] || error("region_bbox: box must satisfy a[i] <= b[i] for all i")
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
    method::Symbol=:bbox,
    max_combinations::Int=200_000,
    max_vertices::Int=50_000,
    max_vertices_for_diameter::Int=2000
)
    if method == :bbox
        bb = region_bbox(pi, r; box=box, max_combinations=max_combinations, max_vertices=max_vertices)
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
        a_in, b_in = box
        a = [float(x) for x in a_in]
        b = [float(x) for x in b_in]

        hp = pi.regions[r]
        verts = _vertices_of_hpoly_in_box(hp, a, b;
                                          max_combinations=max_combinations,
                                          max_vertices=max_vertices)
        # If vertex enumeration is infeasible, fall back to bbox diameter.
        verts === nothing && return region_diameter(pi, r; metric=metric, box=box, method=:bbox,
                                                   max_combinations=max_combinations,
                                                   max_vertices=max_vertices)
        isempty(verts) && return 0.0

        vt = collect(verts)
        nv = length(vt)
        nv > max_vertices_for_diameter && return region_diameter(pi, r; metric=metric, box=box, method=:bbox,
                                                                 max_combinations=max_combinations,
                                                                 max_vertices=max_vertices)

        # Convert to a dense Float64 matrix for distance computations.
        n = pi.n
        X = Matrix{Float64}(undef, nv, n)
        for i in 1:nv
            v = vt[i]
            for j in 1:n
                X[i, j] = float(v[j])
            end
        end

        dmax = 0.0
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
    return FiniteFringe.FinitePoset(leq)
end

function _images_on_P(P::FiniteFringe.FinitePoset,
                      sig_y::Vector{BitVector},
                      sig_z::Vector{BitVector},
                      ups_idx::AbstractVector{<:Integer},
                      downs_idx::AbstractVector{<:Integer})
    m = length(ups_idx)
    r = length(downs_idx)

    Uhat = Vector{FiniteFringe.Upset}(undef, m)
    Dhat = Vector{FiniteFringe.Downset}(undef, r)

    for (i, gi) in enumerate(ups_idx)
        mask = BitVector([sig_y[t][gi] == 1 for t in 1:P.n])
        Uhat[i] = FiniteFringe.upset_closure(P, mask)
    end
    for (j, gj) in enumerate(downs_idx)
        mask = BitVector([sig_z[t][gj] == 0 for t in 1:P.n])
        Dhat[j] = FiniteFringe.downset_closure(P, mask)
    end

    return Uhat, Dhat
end

## NOTE:
## We intentionally do NOT provide the older convenience overload
## `_images_on_P(P, sig_y, sig_z, m, r)`.
## Callers must pass explicit index lists (e.g. 1:m) to avoid hidden semantics.


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

function encode_from_PL_fringe(Ups::Vector{PLUpset},
                               Downs::Vector{PLDownset},
                               Phi_in::AbstractMatrix{QQ};
                               max_regions::Int=10_000,
                               strict_eps::QQ=STRICT_EPS_QQ)
    _assert_PL_inputs(Ups, Downs)
    HAVE_POLY || error("Polyhedra/CDDLib not available; install Polyhedra.jl and CDDLib.jl.")

    feasible = enumerate_feasible_regions(Ups, Downs; max_regions=max_regions, strict_eps=strict_eps)

    if isempty(feasible)
        P = FiniteFringe.FinitePoset(reshape(Bool[true], 1, 1))
        Uhat = FiniteFringe.Upset[FiniteFringe.upset_closure(P, BitVector([false])) for _ in 1:length(Ups)]
        Dhat = FiniteFringe.Downset[FiniteFringe.downset_closure(P, BitVector([false])) for _ in 1:length(Downs)]
        Phi0 = zeros(QQ, length(Downs), length(Ups))
        H = FiniteFringe.FringeModule{QQ}(P, Uhat, Dhat, Phi0)
        pi = PLEncodingMap( (length(Ups)>0 ? Ups[1].U.n : (length(Downs)>0 ? Downs[1].D.n : 0)),
                            BitVector[], BitVector[], HPoly[], Vector{Vector{Float64}}() )
        return P, H, pi
    end

    sigy = Vector{BitVector}(undef, length(feasible))
    sigz = Vector{BitVector}(undef, length(feasible))
    regs = Vector{HPoly}(undef, length(feasible))
    wits = Vector{Vector{Float64}}(undef, length(feasible))
    for (k, rec) in enumerate(feasible)
        sigy[k] = rec[1]
        sigz[k] = rec[2]
        regs[k] = rec[3]
        wits[k] = rec[4]
    end

    P = _uptight_from_signatures(sigy, sigz)
    m = length(Ups)
    r = length(Downs)
    # Explicit index lists (no compatibility overload).
    Uhat, Dhat = _images_on_P(P, sigy, sigz, 1:m, 1:r)

    Phi = _monomialize_phi(Matrix{QQ}(Phi_in), Uhat, Dhat)
    H = FiniteFringe.FringeModule{QQ}(P, Uhat, Dhat, Phi)

    pi = PLEncodingMap(regs[1].n, sigy, sigz, regs, wits)
    return P, H, pi
end

function encode_from_PL_fringe_with_tag(Ups, Downs, Phi_in; kwargs...)
    P, H, pi = encode_from_PL_fringe(Ups, Downs, Phi_in; kwargs...)
    return P, H, pi, :PL
end

"""
    encode_from_PL_fringes(F1, F2, ...; max_regions=10_000, strict_eps=STRICT_EPS_QQ)
        -> (P, Hs, pi)

Build a single finite encoding poset `P` that simultaneously encodes *all* PL
birth/death shapes appearing in the input `PLFringe` presentations.

This is the PL/R^n analog of `ZnEncoding.encode_from_flanges` for Z^n:
- form the union of generator shapes across all inputs,
- enumerate feasible Y-signatures for that union,
- build the uptight region poset `P`,
- push each input down to a finite-poset `FringeModule` on `P`.

Return values:
- `P`  : the common encoding poset (a finite poset of regions)
- `Hs` : a vector of `FiniteFringe.FringeModule{QQ}`, one per input `PLFringe`
- `pi` : a classifier `pi : R^n -> P` (as `PLEncodingMap`)

Notes:
- The returned `P` is a common refinement. Encoding a single PL fringe alone
  may yield a smaller poset. Use `encode_from_PL_fringe` for that case.
"""
function encode_from_PL_fringes(Fs::Vararg{PLFringe};
                                max_regions::Int=10_000,
                                strict_eps::QQ=STRICT_EPS_QQ)

    length(Fs) > 0 || error("encode_from_PL_fringes: need at least one PLFringe")
    HAVE_POLY || error("Polyhedra/CDDLib not available; install Polyhedra.jl and CDDLib.jl.")

    # All inputs must live in the same ambient R^n.
    n = Fs[1].n
    for (k, F) in enumerate(Fs)
        F.n == n || error("encode_from_PL_fringes: fringe $k has ambient dimension $(F.n) but expected $n")
    end

    # Concatenate shapes, recording index ranges so each fringe can be pushed down.
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
        # Degenerate fallback: return a 1-vertex poset and all-zero fringe modules.
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
            Hs[k] = FiniteFringe.FringeModule{QQ}(P, Uhat, Dhat, Phi0)
        end

        pi = PLEncodingMap(n, BitVector[], BitVector[], HPoly[], Vector{Vector{Float64}}())
        return P, Hs, pi
    end

    sigy = Vector{BitVector}(undef, length(feasible))
    sigz = Vector{BitVector}(undef, length(feasible))
    regs = Vector{HPoly}(undef, length(feasible))
    wits = Vector{Vector{Float64}}(undef, length(feasible))
    for (k, rec) in enumerate(feasible)
        sigy[k] = rec[1]
        sigz[k] = rec[2]
        regs[k] = rec[3]
        wits[k] = rec[4]
    end

    P  = _uptight_from_signatures(sigy, sigz)
    pi = PLEncodingMap(regs[1].n, sigy, sigz, regs, wits)

    # Push each PLFringe down to P by slicing the global signatures.
    Hs = Vector{FiniteFringe.FringeModule{QQ}}(undef, length(Fs))
    for (k, F) in enumerate(Fs)
        Uhat, Dhat = _images_on_P(P, sigy, sigz, up_ranges[k], dn_ranges[k])
        Phi = _monomialize_phi(Matrix{QQ}(F.Phi), Uhat, Dhat)
        Hs[k] = FiniteFringe.FringeModule{QQ}(P, Uhat, Dhat, Phi)
    end

    return P, Hs, pi
end


export HPoly, PolyUnion, PLUpset, PLDownset, PLFringe, PLEncodingMap, make_hpoly,
       encode_from_PL_fringe, encode_from_PL_fringes,
       PolyInBoxCache, poly_in_box_cache


end # module
