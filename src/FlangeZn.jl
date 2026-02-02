module FlangeZn

using LinearAlgebra
using ..CoreModules: QQ

import ..ExactQQ: rankQQ

export Face, face,
       IndFlat, IndInj,
       Flange,
       canonical_matrix,
       active_flats, active_injectives,
       degree_matrix, dim_at,
       bounding_box,
       cross_validate, flange_to_axis

# ---------------------------------------------------------------------------
# Face
# ---------------------------------------------------------------------------

"""
    Face(n, coords)
    Face(n, idxs)

A face of the coordinate orthant in Z^n.

Internally we store the face as a `BitVector` mask of length `n` in the field
`coords`, where `coords[i] == true` means coordinate `i` is part of the face.

Constructor forms:

* `Face(n, coords::BitVector)` or `Face(n, coords::Vector{Bool})`:
  treat `coords` as a boolean mask of length `n`.

* `Face(n, idxs::Vector{Int})`:
  treat `idxs` as a list of indices in `1:n` that are `true` in the mask.

For user-facing code, prefer `face(n, coords)` which is more permissive about
the input container type.
"""
struct Face
    n::Int
    coords::BitVector
end

# Convert a list of indices in 1:n to a BitVector mask.
function _indices_to_bitvector(n::Int, idxs::Vector{Int})::BitVector
    mask = falses(n)
    for i in idxs
        if i < 1 || i > n
            error("Face: index $(i) out of range 1:$(n)")
        end
        mask[i] = true
    end
    return mask
end

# idx list constructor
Face(n::Int, idxs::Vector{Int}) = Face(n, _indices_to_bitvector(n, idxs))

# boolean mask constructor (Vector{Bool})
function Face(n::Int, coords::Vector{Bool})
    if length(coords) != n
        error("Face: expected Bool mask of length $(n), got $(length(coords))")
    end
    return Face(n, BitVector(coords))
end

"""
    face(n, coords)

Convenience constructor for `Face`:

* `coords` may be a `BitVector`, a `Vector{Bool}`, a `Vector{Int}` (index list),
  or any iterable producing `Bool` or `Int`.

We also accept `coords = []` (empty) to mean the zero-dimensional face
(no fixed coordinates).
"""
function face(n::Integer, coords)
    nn = Int(n)

    if coords isa BitVector
        if length(coords) != nn
            error("Face: expected BitVector of length $(nn), got $(length(coords))")
        end
        return Face(nn, coords)
    elseif coords isa Vector{Bool}
        return Face(nn, coords)
    elseif coords isa Vector{Int}
        # Interpret as an index list in 1:nn.
        if length(coords) == 0
            return Face(nn, falses(nn))
        end
        return Face(nn, coords)
    else
        # Try to interpret as iterable of Bool.
        try
            if length(coords) == 0
                return Face(nn, falses(nn))
            end
            return Face(nn, Bool[x for x in coords])
        catch
        end
        # Try to interpret as iterable of Int (index list).
        try
            if length(coords) == 0
                return Face(nn, falses(nn))
            end
            return Face(nn, Int[x for x in coords])
        catch
        end
        error(
            "face(n, coords): coords must be BitVector, Vector{Bool}, Vector{Int}, " *
            "or an iterable producing Bool/Int.",
        )
    end
end

# ---------------------------------------------------------------------------
# IndFlat / IndInj
# ---------------------------------------------------------------------------

"""
    IndFlat(tau, b; id=:F)

An indexed flat in Z^n, specified by:

* `tau::Face`  : which coordinates are fixed (the mask in `tau.coords`)
* `b`          : translation vector in Z^n
* `id::Symbol` : an arbitrary label (useful when multiple flats share the same
  underlying set)

Canonical constructor shape: `IndFlat(tau, b; id=...)`.
"""
struct IndFlat
    b::Vector{Int}
    tau::Face
    id::Symbol

    # Only allow canonical positional order (tau, b, id).
    # Defining any inner constructor suppresses Julia's default (b, tau, id) constructor.
    function IndFlat(tau::Face, b::Vector{Int}, id::Symbol)
        return new(b, tau, id)
    end
end

function IndFlat(tau::Face, b::AbstractVector{<:Integer}; id::Symbol = :F)
    return IndFlat(tau, Vector{Int}(b), id)
end

"""
    IndInj(tau, b; id=:E)

An indexed injective in Z^n (the down-set analogue of `IndFlat`).

Canonical constructor shape: `IndInj(tau, b; id=...)`.
"""
struct IndInj
    b::Vector{Int}
    tau::Face
    id::Symbol

    # Only allow canonical positional order (tau, b, id).
    function IndInj(tau::Face, b::Vector{Int}, id::Symbol)
        return new(b, tau, id)
    end
end

function IndInj(tau::Face, b::AbstractVector{<:Integer}; id::Symbol = :E)
    return IndInj(tau, Vector{Int}(b), id)
end


# Internal predicate: do b and g agree on the fixed coordinates of tau?
@inline function _matches_on_face(b::Vector{Int}, g::AbstractVector{<:Integer}, tau::Face)::Bool
    @inbounds for i in 1:tau.n
        if tau.coords[i] && b[i] != g[i]
            return false
        end
    end
    return true
end

"""
    active_flats(flats, g)

Return the indices of flats whose underlying sets contain the lattice point `g`.

This is used to form the local (fiberwise) matrix at `g` by selecting columns.
"""
function active_flats(flats::Vector{IndFlat}, g::Vector{Int})
    active = Int[]
    for (j, F) in enumerate(flats)
        if in_flat(F, g)
            push!(active, j)
        end
    end
    return active
end

"""
    active_injectives(injectives, g)

Return the indices of injectives whose underlying sets contain the lattice point `g`.

This is used to form the local (fiberwise) matrix at `g` by selecting rows.
"""
function active_injectives(injectives::Vector{IndInj}, g::Vector{Int})
    active = Int[]
    for (i, E) in enumerate(injectives)
        if in_inj(E, g)
            push!(active, i)
        end
    end
    return active
end


# ---------------------------------------------------------------------------
# Membership predicates
# ---------------------------------------------------------------------------

"""
    in_flat(F, g)

Return `true` if the lattice point `g` lies in the underlying set of the flat `F`.

Convention (consistent with ZnEncoding):
* if `F.tau.coords[i] == true` then coordinate `i` is FREE (Z-direction): no constraint
* otherwise coordinate `i` is inequality-constrained: `g[i] >= F.b[i]`
"""
function in_flat(F::IndFlat, g::AbstractVector{<:Integer})::Bool
    n = F.tau.n
    if length(g) != n
        error("in_flat: point dimension does not match face dimension")
    end

    @inbounds for i in 1:n
        if !F.tau.coords[i] && g[i] < F.b[i]
            return false
        end
    end
    return true
end

"""
    in_inj(E, g)

Return `true` if the lattice point `g` lies in the underlying set of the injective `E`.

Convention (consistent with ZnEncoding):
* if `E.tau.coords[i] == true` then coordinate `i` is FREE (Z-direction): no constraint
* otherwise coordinate `i` is inequality-constrained: `g[i] <= E.b[i]`
"""
function in_inj(E::IndInj, g::AbstractVector{<:Integer})::Bool
    n = E.tau.n
    if length(g) != n
        error("in_inj: point dimension does not match face dimension")
    end

    @inbounds for i in 1:n
        if !E.tau.coords[i] && g[i] > E.b[i]
            return false
        end
    end
    return true
end

# Tuple overloads: ZnEncoding often represents lattice points as NTuples
# for performance (no allocations). Support that here.

@inline function in_flat(F::IndFlat, g::NTuple{N,T})::Bool where {N,T<:Integer}
    n = F.tau.n
    if N != n
        error("in_flat: point dimension does not match face dimension")
    end

    @inbounds for i in 1:n
        if !F.tau.coords[i] && g[i] < F.b[i]
            return false
        end
    end
    return true
end

@inline function in_inj(E::IndInj, g::NTuple{N,T})::Bool where {N,T<:Integer}
    n = E.tau.n
    if N != n
        error("in_inj: point dimension does not match face dimension")
    end

    @inbounds for i in 1:n
        if !E.tau.coords[i] && g[i] > E.b[i]
            return false
        end
    end
    return true
end

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

"""
    intersects(F, E)

Return `true` if the underlying sets of an `IndFlat` and an `IndInj` can intersect.

With the "free coordinates" convention above, the only obstruction is when a coordinate
is constrained for both generators and the bounds are incompatible:
    F.b[i] > E.b[i]  (for some i with !F.tau.coords[i] && !E.tau.coords[i])
"""
function intersects(F::IndFlat, E::IndInj)::Bool
    n = F.tau.n
    if E.tau.n != n
        error("intersects: face dimensions do not match")
    end

    @inbounds for i in 1:n
        if !F.tau.coords[i] && !E.tau.coords[i] && (F.b[i] > E.b[i])
            return false
        end
    end
    return true
end

# ---------------------------------------------------------------------------
# Flange
# ---------------------------------------------------------------------------

"""
    Flange(n, flats, injectives, phi)

A flange over Z^n.

Fields (canonical, no aliases):

* `n::Int`
* `flats::Vector{IndFlat}`
* `injectives::Vector{IndInj}`
* `phi::Matrix{K}`

The matrix `phi` is interpreted fiberwise: at a point `g in Z^n`, the active
injectives/active flats determine a submatrix, and the fiberwise "dimension" is
the rank of that submatrix.
"""
struct Flange{K}
    n::Int
    flats::Vector{IndFlat}
    injectives::Vector{IndInj}
    phi::Matrix{K}

    function Flange{K}(
        n::Int,
        flats::Vector{IndFlat},
        injectives::Vector{IndInj},
        phi::AbstractMatrix{K},
    ) where {K}
        Phi = Matrix{K}(phi)
        if size(Phi, 1) != length(injectives) || size(Phi, 2) != length(flats)
            error(
                "Flange: phi must have size (#injectives, #flats) = " *
                "($(length(injectives)), $(length(flats))), got $(size(Phi))",
            )
        end

        # Enforce the monomial condition: if a flat and injective cannot intersect,
        # the corresponding matrix entry must be zero.
        @inbounds for i in 1:length(injectives)
            E = injectives[i]
            for j in 1:length(flats)
                F = flats[j]
                if !intersects(F, E)
                    Phi[i, j] = zero(K)
                end
            end
        end

        return new{K}(n, flats, injectives, Phi)
    end
end

# Outer constructors for convenience.
function Flange{K}(n::Int, flats::Vector{IndFlat}, injectives::Vector{IndInj}, phi::AbstractMatrix) where {K}
    return Flange{K}(n, flats, injectives, Matrix{K}(phi))
end

function Flange{K}(n::Int, flats::Vector{IndFlat}, injectives::Vector{IndInj}, phi::AbstractVector) where {K}
    Phi = reshape(collect(phi), length(injectives), length(flats))
    return Flange{K}(n, flats, injectives, Phi)
end

function Flange(n::Int, flats::Vector{IndFlat}, injectives::Vector{IndInj}, phi::AbstractMatrix{K}) where {K}
    return Flange{K}(n, flats, injectives, phi)
end


"""Convert a flange over an arbitrary coefficient type to a flange over QQ."""
function toQQ(FG::Flange{K}) where {K}
    return Flange{QQ}(FG.n, FG.flats, FG.injectives, QQ.(FG.phi))
end

# Convenience wrappers accepting a flange directly.
active_flats(FG::Flange, g::Vector{Int}) = active_flats(FG.flats, g)
active_injectives(FG::Flange, g::Vector{Int}) = active_injectives(FG.injectives, g)

"""
    degree_matrix(FG, g)

Return `(Phi_g, rows, cols)` where:

* `rows` are the active injective indices at `g`
* `cols` are the active flat indices at `g`
* `Phi_g = FG.phi[rows, cols]`
"""
function degree_matrix(FG::Flange{K}, g::Vector{Int}) where {K}
    rows = active_injectives(FG, g)
    cols = active_flats(FG, g)

    # Convention: if either side is empty at g, treat the local matrix as empty.
    if isempty(rows) || isempty(cols)
        return (zeros(K, 0, 0), Int[], Int[])
    end

    return (FG.phi[rows, cols], rows, cols)
end

"""
    dim_at(FG, g; rankfun=rank)

Fiberwise dimension of a flange at `g`.

By convention this is the rank of the local submatrix `Phi_g`.
"""
function dim_at(FG::Flange{K}, g::Vector{Int}; rankfun = rank) where {K}
    Phi_g, _, _ = degree_matrix(FG, g)
    if isempty(Phi_g)
        return 0
    end
    return rankfun(Phi_g)
end

# ---------------------------------------------------------------------------
# Unified API: dim_at on finite-poset modules (IndicatorResolutions.PModule)
#
# The umbrella module `PosetModules` exports `dim_at` from the flange layer
# (this file). Downstream code/tests also want to write `dim_at(M, q)` when `M`
# is a finite-poset module (a `PModule`).
#
# To avoid creating a second, unrelated `dim_at` function, we extend the *same*
# generic function here, where it is defined.
# ---------------------------------------------------------------------------

import ..IndicatorResolutions

"""
    dim_at(M::IndicatorResolutions.PModule, q::Integer) -> Int

Return the stalk (fiber) dimension of the finite-poset module `M` at vertex `q`.

This is the finite-encoding analogue of `dim_at(FG::Flange, g)`.
"""
@inline function dim_at(M::IndicatorResolutions.PModule, q::Integer)::Int
    qi = Int(q)
    1 <= qi <= length(M.dims) || error("dim_at: vertex index q=$(qi) out of range")
    @inbounds return M.dims[qi]
end


"""
    bounding_box(FG; margin=0)

Return an axis-aligned bounding box `(a, b)` for the region where the flange
can be nontrivial, based on the inequality coordinates of flats/injectives.

This is primarily used as a finite search window for encoding and testing.

For a coordinate `i`:

* flats contribute lower bounds `a[i] >= F.b[i] - margin` when `F.tau.coords[i] == false`
* injectives contribute upper bounds `b[i] <= E.b[i] + margin` when `E.tau.coords[i] == false`

Coordinates fixed by a face (`tau.coords[i] == true`) do not constrain the box.
"""
function bounding_box(FG::Flange; margin::Int = 0)
    n = FG.n
    a = fill(typemin(Int), n)
    b = fill(typemax(Int), n)

    for F in FG.flats
        @inbounds for i in 1:n
            if F.tau.coords[i]
                continue
            end
            a[i] = max(a[i], F.b[i] - margin)
        end
    end

    for E in FG.injectives
        @inbounds for i in 1:n
            if E.tau.coords[i]
                continue
            end
            b[i] = min(b[i], E.b[i] + margin)
        end
    end

    return (a, b)
end

# ---------------------------------------------------------------------------
# Minimization
# ---------------------------------------------------------------------------

# Helper: a hashable key for grouping flats/injectives by their underlying set.
@inline _underlying_key(F::IndFlat) = (Tuple(F.tau.coords), Tuple(F.b))
@inline _underlying_key(E::IndInj) = (Tuple(E.tau.coords), Tuple(E.b))

# Helper: test whether a vector is the zero vector.
@inline function _is_zero_vec(v)
    for x in v
        if !iszero(x)
            return false
        end
    end
    return true
end

# Helper: check whether w is a scalar multiple of v (including the all-zero case).
#
# This is used only inside `minimize`, where coefficients are expected to live in
# a field (e.g. QQ). If you work over a non-field coefficient type, you may want
# to disable minimization or replace this proportionality test.
function _is_proportional(v, w)::Bool
    if _is_zero_vec(v)
        return _is_zero_vec(w)
    end
    if _is_zero_vec(w)
        return true
    end

    # Find the first nonzero entry of v.
    idx = nothing
    for i in eachindex(v)
        if !iszero(v[i])
            idx = i
            break
        end
    end
    if idx === nothing
        # Should not happen (handled above), but keep it robust.
        return _is_zero_vec(w)
    end

    # Ratio c = w[idx] / v[idx]
    if iszero(w[idx])
        # w is not the zero vector here, so it cannot be proportional.
        return false
    end
    c = w[idx] / v[idx]

    @inbounds for i in eachindex(v)
        if w[i] != c * v[i]
            return false
        end
    end
    return true
end

"""
    minimize(FG; rankfun=rank)

Return a reduced flange obtained by removing proportional duplicate rows/columns
that correspond to *identical underlying* injectives/flats.

Rationale:

* If two flats have the same underlying set (same `tau` mask and same `b`) then
  they are either simultaneously active or inactive at every point `g`.
* If their columns in `phi` are proportional, dropping duplicates does not change
  the fiberwise rank function `dim_at(FG, g)`.

The same logic applies to injectives (rows).

This is primarily a performance optimization (smaller encodings).
"""
function minimize(FG::Flange{K}; rankfun = rank) where {K}
    flats = FG.flats
    injectives = FG.injectives
    Phi = FG.phi

    # --- Reduce columns (flats) ---
    col_groups = Dict{Tuple{Any, Any}, Vector{Int}}()
    for (j, F) in enumerate(flats)
        key = _underlying_key(F)
        push!(get!(col_groups, key, Int[]), j)
    end

    keep_cols = Int[]
    for idxs in values(col_groups)
        # Preserve original order by processing indices as-is.
        j0 = idxs[1]
        push!(keep_cols, j0)

        for j in idxs[2:end]
            if _is_proportional(view(Phi, :, j0), view(Phi, :, j))
                # Drop proportional duplicates.
                continue
            end
            push!(keep_cols, j)
        end
    end
    sort!(keep_cols)

    flats2 = flats[keep_cols]
    Phi2 = Phi[:, keep_cols]

    # --- Reduce rows (injectives) ---
    row_groups = Dict{Tuple{Any, Any}, Vector{Int}}()
    for (i, E) in enumerate(injectives)
        key = _underlying_key(E)
        push!(get!(row_groups, key, Int[]), i)
    end

    keep_rows = Int[]
    for idxs in values(row_groups)
        i0 = idxs[1]
        push!(keep_rows, i0)

        for i in idxs[2:end]
            if _is_proportional(view(Phi2, i0, :), view(Phi2, i, :))
                continue
            end
            push!(keep_rows, i)
        end
    end
    sort!(keep_rows)

    injectives2 = injectives[keep_rows]
    Phi3 = Phi2[keep_rows, :]

    return Flange{K}(FG.n, flats2, injectives2, Phi3)
end

# ---------------------------------------------------------------------------
# Canonical matrices
# ---------------------------------------------------------------------------

"""
    canonical_matrix(flats, injectives)

Build the (0,1) matrix with entry 1 exactly when the underlying flat and injective
can intersect.

This is useful for constructing small example flanges.
"""
function canonical_matrix(flats::Vector{IndFlat}, injectives::Vector{IndInj})
    m = length(injectives)
    n = length(flats)
    A = zeros(QQ, m, n)

    for (i, E) in enumerate(injectives)
        for (j, F) in enumerate(flats)
            if intersects(F, E)
                A[i, j] = QQ(1)
            end
        end
    end

    return A
end

"""A simple "identity" matrix between injectives and flats with identical underlying sets."""
function canonical_matrix(FG::Flange{K}) where {K}
    m = length(FG.injectives)
    n = length(FG.flats)
    A = zeros(K, m, n)

    for (i, E) in enumerate(FG.injectives)
        for (j, F) in enumerate(FG.flats)
            if E.tau.coords == F.tau.coords && E.b == F.b
                A[i, j] = one(K)
            end
        end
    end

    return A
end

#---------------------------------------------------------------------------
# Validation and cross checkers
#---------------------------------------------------------------------------

# Cross-validate a Z^n flange against an axis-aligned PL evaluation on the same
# lattice points in a convex-projection box [a,b].  No external Axis* module.

# Minimal axis-aligned shapes for cross-check (not exported elsewhere).
# IMPORTANT: we do not emulate +/- infinity with sentinels.  Instead we
# carry an explicit "free" bitmask that marks coordinates whose bound is
# ignored.  This keeps membership tests exact and avoids magic numbers.
struct AxisUpset{T}
    # a[i] is only used when free[i] == false; otherwise it is a dummy.
    a::Vector{T}      # lower thresholds
    free::BitVector   # coordinates that are free in Z^tau
end
struct AxisDownset{T}
    # b[i] is only used when free[i] == false; otherwise it is a dummy.
    b::Vector{T}      # upper thresholds
    free::BitVector   # coordinates that are free in Z^tau
end


struct AxisFringe{TU,TD,TF}
     n::Int
     births::Vector{AxisUpset{TU}}
     deaths::Vector{AxisDownset{TD}}
     Phi::Matrix{TF}  # same scalar "monomial" matrix
 end

"Translate Zn indecomposables to axis-aligned PL up/down sets for cross-checking."
function flange_to_axis(fr::Flange{K}) where {K}
    # Keep everything over QQ to avoid floating-point artifacts.
    n = fr.n
    births = AxisUpset{QQ}[]
    for F in fr.flats
        a = Vector{QQ}(undef, n)
        for i in 1:n
            # When F.tau.coords[i] is true (free coordinate), a[i] is never read.
            # We still store a QQ value to keep the vector concrete.
            a[i] = QQ(F.b[i])
        end
        push!(births, AxisUpset{QQ}(a, BitVector(F.tau.coords)))
    end
    deaths = AxisDownset{QQ}[]
    for E in fr.injectives
        b = Vector{QQ}(undef, n)
        for i in 1:n
            # Same comment as above: b[i] is ignored when the coordinate is free.
            b[i] = QQ(E.b[i])
        end
        push!(deaths, AxisDownset{QQ}(b, BitVector(E.tau.coords)))
    end
    # DO NOT coerce Phi to Float64; keep it exact over QQ.
    Phi = Matrix{QQ}(fr.phi)
    return AxisFringe{QQ,QQ,QQ}(n, births, deaths, Phi)
end

"Test membership of a lattice point x in an axis-aligned upset or downset (generic and exact)."
_contains(U::AxisUpset{T}, x::Vector{T}) where {T} =
    all(U.free[i] || (x[i] >= U.a[i]) for i in 1:length(x))
_contains(D::AxisDownset{T}, x::Vector{T}) where {T} =
    all(D.free[i] || (x[i] <= D.b[i]) for i in 1:length(x))

"""
    cross_validate(fr::Flange; margin=1, rankfun=rankQQ)

1. Build the convex-projection box [a,b] (heuristic 'bounding_box').
2. Evaluate 'dim M_g' for all integer 'g in [a,b]' via the flange.
3. Build an axis-aligned proxy and evaluate the same lattice points.
4. Compare; return '(all_equal?, report::Dict)'.
"""
function cross_validate(fr::Flange; margin=1, rankfun=rankQQ)
    a, b = bounding_box(fr; margin)
    ranges = (a[i]:b[i] for i in 1:fr.n)
    pts = collect(Iterators.product(ranges...))

    # Flange evaluation (exact over QQ if rankfun = rankQQ).
    dims_Z = Dict{Tuple{Vararg{Int}}, Int}()
    for t in pts
        g = collect(Int.(t))
        dims_Z[Tuple(g...)] = dim_at(fr, g; rankfun=rankfun)
    end

    # Axis-aligned proxy (stay exact; compare apples-to-apples with the flange)
    afr = flange_to_axis(fr)
    dims_PL = Dict{Tuple{Vararg{Int}}, Int}()
    for t in pts
        # Use rationals for coordinates to make membership tests exact.
        x = QQ.(collect(Int.(t)))
        rows = [ _contains(d, x) for d in afr.deaths ]
        cols = [ _contains(u, x) for u in afr.births ]
        idxr = findall(identity, rows)
        idxc = findall(identity, cols)
        if isempty(idxr) || isempty(idxc)
            dims_PL[Tuple(Int.(t)...)] = 0
        else
            # Rank exactly with the same provided rank function.
            Phi_x = afr.Phi[idxr, idxc]
            dims_PL[Tuple(Int.(t)...)] = rankfun(Phi_x)
        end
    end

    # Compare
    mism = Dict{Tuple{Vararg{Int}}, Tuple{Int,Int}}()
    for (k,v) in dims_Z
        if v != dims_PL[k]; mism[k] = (v, dims_PL[k]); end
    end
    ok = isempty(mism)
    report = Dict("box" => (a,b), "mismatches" => mism,
                  "tested" => length(pts), "agree" => length(pts) - length(mism))
    return ok, report
end

end # module
