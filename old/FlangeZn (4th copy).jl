module FlangeZn
# ------------------------------------------------------------------------------
# Flange presentations over Z^n (Miller section 5).
#
# A flange Phi : (\oplus flats) \to (\oplus injectives) is stored as a scalar matrix Phi with
# rows = indecomposable injectives, cols = indecomposable flats.  At a degree g,
# take the submatrix with the active rows/cols; dim M_g = rank(Phi_g).
# (Remark 5.5, 5.11; Def. 5.14; Prop. 5.17.)  See also section 1.4 for the narrative.
# ------------------------------------------------------------------------------

using LinearAlgebra
using ..CoreModules: QQ

# ---------------- faces, flats, injectives (ASCII) --------------------------------
"Face tau of N^n: a bitmask of 'free' coordinates (Z-directions)."
struct Face
    n::Int
    coords::BitVector
    function Face(n::Int, coords::AbstractVector{Bool})
        length(coords) == n || error("Face: coords must have length n")
        new(n, BitVector(coords))
    end
end

# Convenience constructors for Face.
#
# Two supported integer forms:
# 1) If length(coords) == n, treat coords as a 0/1 (or general integer) mask.
#    Nonzero entries mean "free coordinate".
# 2) Otherwise, treat coords as a list of 1-based coordinate indices that are free.
function Face(n::Int, coords::AbstractVector{<:Integer})
    if length(coords) == n
        # Integer mask form.
        return Face(n, map(!iszero, coords))
    end

    # Index-list form.
    mask = falses(n)
    for i in coords
        (1 <= i <= n) || error("Face: coordinate index out of bounds: $i (expected 1:$n)")
        mask[i] = true
    end
    return Face(n, mask)
end


"Indecomposable flat F = k[b + N^n + Z^tau]."
struct IndFlat
    b::Vector{Int}
    tau::Face
    id::Symbol

    # Canonical constructor (non-parametric):
    #   IndFlat(tau, b; id=:F)
    #
    # The coefficient ring does not affect the combinatorics of flats/injectives, so
    # IndFlat is intentionally non-parametric to keep a single, stable API surface.
    function IndFlat(tau::Face, b::AbstractVector{<:Integer}; id::Symbol=:F)
        length(b) == tau.n || error("IndFlat: length(b) must equal ambient dimension n = $(tau.n)")
        bb = b isa Vector{Int} ? b : Vector{Int}(b)
        return new(bb, tau, id)
    end
end

struct IndInj
    b::Vector{Int}
    tau::Face
    id::Symbol

    # Canonical constructor (non-parametric):
    #   IndInj(tau, b; id=:E)
    function IndInj(tau::Face, b::AbstractVector{<:Integer}; id::Symbol=:E)
        length(b) == tau.n || error("IndInj: length(b) must equal ambient dimension n = $(tau.n)")
        bb = b isa Vector{Int} ? b : Vector{Int}(b)
        return new(bb, tau, id)
    end
end

# Membership tests at degree g in Z^n
@inline function in_flat(F::IndFlat, g::AbstractVector{<:Integer})
    @inbounds for i in 1:length(g)
        F.tau.coords[i] && continue
        if g[i] < F.b[i]; return false; end
    end
    true
end

@inline function in_inj(E::IndInj, g::AbstractVector{<:Integer})
    @inbounds for i in 1:length(g)
        E.tau.coords[i] && continue
        if g[i] > E.b[i]; return false; end
    end
    true
end

# Tuple specializations avoid allocating temporary vectors when working with lattice
# points represented as `NTuple{N,Int}`. These are used heavily by memoized rank queries
# and rectangle signed barcodes.

@inline function in_flat(F::IndFlat, g::NTuple{N,<:Integer}) where {N}
    @assert N == length(F.b)
    @inbounds for i in 1:N
        if !F.tau.coords[i] && g[i] < F.b[i]
            return false
        end
    end
    return true
end

@inline function in_inj(E::IndInj, g::NTuple{N,<:Integer}) where {N}
    @assert N == length(E.b)
    @inbounds for i in 1:N
        if !E.tau.coords[i] && g[i] > E.b[i]
            return false
        end
    end
    return true
end


"Non-emptiness of the intersection F cap E (Prop. 5.17)."
function intersects(F::IndFlat, E::IndInj)
    n = length(F.b); n == length(E.b) || error("dimension mismatch")
    @inbounds for i in 1:n
        if !F.tau.coords[i] && !E.tau.coords[i] && (F.b[i] > E.b[i])
            return false
        end
    end
    true
end

# ---------------- flange container -------------------------------------------------
"Flange Phi : (oplus flats) to (oplus injectives) with scalar matrix Phi."
struct Flange{K}
    n::Int
    flats::Vector{IndFlat}
    injectives::Vector{IndInj}
    Phi::Matrix{K}
    function Flange{K}(n::Int, flats::Vector{IndFlat},
                       injectives::Vector{IndInj}, Phi::AbstractMatrix{K}) where {K}
        
        m_expected = length(injectives)
        n_expected = length(flats)
        if size(Phi,1) != m_expected || size(Phi,2) != n_expected
            error("Phi must have size (#injectives x #flats) = ($(m_expected) x $(n_expected)); got $(size(Phi)). " *
                "Convention: rows index injectives, cols index flats.")
        end
        
        size(Phi,1) == m_expected || error("Phi rows must equal #injectives")
        size(Phi,2) == n_expected      || error("Phi cols must equal #flats")
        M = Matrix{K}(Phi)
        for q in 1:m_expected, p in 1:n_expected
            if !intersects(flats[p], injectives[q]); M[q,p] = zero(K); end
        end
        new{K}(n, flats, injectives, M)
    end
end



"Canonical scalar matrix (1 when F_p cap E_q neq emptyset, else 0)."
function canonical_matrix(flats::Vector{IndFlat}, injectives::Vector{IndInj}, ::Type{K}=QQ) where {K}
    Phi = zeros(K, length(injectives), length(flats))
    for q in 1:length(injectives), p in 1:length(flats)
        if intersects(flats[p], injectives[q]); Phi[q,p] = one(K); end
    end
    Phi
end

# ---------------- degree-wise evaluation ------------------------------------------
"Active column indices (flats) at degree g."
active_flats(FG::Flange, g::AbstractVector{<:Integer}) =
    [p for p in 1:length(FG.flats) if in_flat(FG.flats[p], g)]

"Active row indices (injectives) at degree g."
active_injectives(FG::Flange, g::AbstractVector{<:Integer}) =
    [q for q in 1:length(FG.injectives) if in_inj(FG.injectives[q], g)]

"Return Phi_g = Phi[rows,cols] together with the row/col indices."
function degree_matrix(FG::Flange{K}, g::AbstractVector{<:Integer}) where {K}
    cols = active_flats(FG, g)
    rows = active_injectives(FG, g)

    # If either side has no active summands, the degree piece is zero.
    # The tests expect BOTH index lists empty and the matrix size (0,0).
    if isempty(cols) || isempty(rows)
        return zeros(K, 0, 0), Int[], Int[]
    end

    return FG.Phi[rows, cols], rows, cols
end


"Dimension dim M_g = rank(Phi_g)."
function dim_at(FG::Flange, g::AbstractVector{<:Integer}; rankfun=rank)
    A, _, _ = degree_matrix(FG, g)
    isempty(A) ? 0 : rankfun(Matrix{eltype(A)}(A))
end

# ---------------------------------------------------------------------------
# Unified API: dim_at on finite-poset modules (IndicatorResolutions.PModule)
#
# The umbrella module PosetModules exports `dim_at` from the Zn/Rn flange layer
# (this file). Downstream code/tests also want to write `dim_at(M, q)` when
# `M` is a finite-encoding module (a PModule).
#
# Rather than defining glue code in src/PosetModules.jl (your "API surface"
# file), we extend the *existing* generic `dim_at` right here, where it is
# defined. This keeps PosetModules.jl as a re-export-only file, but still gives
# a mathematician-friendly uniform query function.
# ---------------------------------------------------------------------------

# Import the sibling module so we can refer to its types without changing
# the PosetModules.jl API/re-export file.
import ..IndicatorResolutions

"""
    dim_at(M::IndicatorResolutions.PModule, q::Integer) -> Int

Return the stalk (fiber) dimension of the finite-poset module `M` at vertex `q`.

Mathematically: `dim_at(M, q) = dim_k M(q)`.

This method is defined on the same generic function as the flange query

    dim_at(FG::Flange, g::AbstractVector)

so users (and downstream code) can write `dim_at(M, ...)` uniformly, regardless
of whether `M` lives in the Zn/Rn flange layer or the finite-encoding layer.
"""
@inline function dim_at(M::IndicatorResolutions.PModule, q::Integer)::Int
    qi = Int(q)
    n = length(M.dims)
    if qi < 1 || qi > n
        error("dim_at: vertex index q=$(qi) out of range 1:$(n)")
    end
    @inbounds return M.dims[qi]
end


# ---------------- conservative bounding box for finite encoding --------------------
"Crude bounding box [a,b] that contains all non-trivial behaviour."
function bounding_box(FG::Flange; margin::Int=1)
    n = FG.n
    a = fill(div(typemin(Int),4), n)
    b = fill(div(typemax(Int),4), n)
    # Coordinates constrained by flats set lower bounds; by injectives set upper bounds
    for F in FG.flats, i in 1:n
        if !F.tau.coords[i]; a[i] = max(a[i], F.b[i] - margin); end
    end
    for E in FG.injectives, i in 1:n
        if !E.tau.coords[i]; b[i] = min(b[i], E.b[i] + margin); end
    end
    a, b
end


# --------------------- flange minimization ----------------------------------------
"""
    minimize(FG::Flange{K}) -> Flange{K}

Return a flange with:
  * zero columns (flat summands that never contribute) removed,
  * zero rows   (injective summands unused) removed,
  * duplicate flat columns that are proportional merged to one,
  * duplicate injective rows that are proportional merged to one.

The represented image submodule does not change; only the presentation shrinks.
"""
function minimize(FG::Flange{K}) where {K}
    Phi = FG.Phi
    m, n = size(Phi)

    # 1) drop zero columns and rows
    keep_cols = [any(x -> !iszero(x), Phi[:, j]) for j in 1:n]
    keep_rows = [any(x -> !iszero(x), Phi[i, :]) for i in 1:m]
    flats1 = [FG.flats[j] for j in 1:n if keep_cols[j]]
    inject1 = [FG.injectives[i] for i in 1:m if keep_rows[i]]
    Phi1 = Phi[keep_rows, keep_cols]

    # helper: detect proportional duplicates, but only merge if the underlying
    # indicator objects are identical (same b and tau).  This prevents incorrect merges
    # when two different flats/injectives happen to have proportional columns/rows.
    function proportional_groups_cols(A::AbstractMatrix{K}, same_label::Function) where {K}
        n1 = size(A, 2)
        groups = Dict{Int, Vector{Int}}()
        used = falses(n1)
        for j in 1:n1
            if used[j]; continue; end
            v = A[:, j]
            if all(iszero, v)
                groups[j] = [j]
                used[j] = true
                continue
            end
            groups[j] = [j]
            used[j] = true
            for k in (j+1):n1
                if used[k]; continue; end
                same_label(j, k) || continue
                w = A[:, k]
                if all(iszero, w); continue; end
                # test proportionality v ~ w
                t = 0
                for ii in 1:length(v)
                    if !iszero(v[ii])
                        t = ii
                        break
                    end
                end
                if t == 0; continue; end
                alpha = w[t] / v[t]
                ok = true
                for ii in 1:length(v)
                    if w[ii] != alpha * v[ii]
                        ok = false
                        break
                    end
                end
                if ok
                    push!(groups[j], k)
                    used[k] = true
                end
            end
        end
        return groups
    end

    proportional_groups_rows(A::AbstractMatrix{K}, same_label::Function) where {K} =
    proportional_groups_cols(transpose(A), same_label)

    # 2) merge proportional duplicate columns (keep one per group)
    same_flat = (j,k) -> (flats1[j].b == flats1[k].b && flats1[j].tau.coords == flats1[k].tau.coords)
    groupsC = proportional_groups_cols(Phi1, same_flat)
    keepC = falses(size(Phi1,2)); @inbounds for j in keys(groupsC); keepC[j] = true; end
    flats2 = [flats1[j] for j in 1:length(flats1) if keepC[j]]
    Phi2 = Phi1[:, keepC]

    # 3) merge proportional duplicate rows (keep one per group)
    same_inj = (j,k) -> (inject1[j].b == inject1[k].b && inject1[j].tau.coords == inject1[k].tau.coords)
    groupsR = proportional_groups_rows(Phi2, same_inj)
    keepR = falses(size(Phi2,1)); @inbounds for i in keys(groupsR); keepR[i] = true; end
    inject2 = [inject1[i] for i in 1:length(inject1) if keepR[i]]
    Phi3 = Phi2[keepR, :]

    # Rebuild flange (constructor will re-zero forbidden entries)
    return Flange{K}(FG.n, flats2, inject2, Phi3)
end



"""
    face(n, coords) -> Face

Convenience constructor for `Face` (a subset tau of {1,...,n}).

Why this exists:
- The low-level constructors `Face(n, ::AbstractVector{Bool})` and
  `Face(n, ::AbstractVector{<:Integer})` are intentionally type-strict.
- In particular, the literal `[]` has type `Vector{Any}`, so it will *not*
  dispatch to those methods.
- This helper accepts common "mathematician inputs" and coerces them.

Arguments
---------
- `n::Integer`: ambient dimension.
- `coords`: one of:
  * a boolean mask of length `n` (true means the coordinate is free, i.e. in tau),
  * an integer mask of length `n` (nonzero means free),
  * a list of integer indices (1-based) specifying which coordinates are free.

Examples
--------
    tau0  = face(2, [])                 # tau = empty set
    tau2  = face(2, [2])                # tau = {2}
    tau12 = face(2, [1, 2])             # tau = {1,2}
    tau_m = face(2, [false, true])      # same as tau = {2}
    tau_t = face(3, (1, 3))             # tau = {1,3}
"""
function face(n::Integer, coords)
    nn = Int(n)

    # Special case: `[]` is Vector{Any}; interpret as "no free coordinates".
    if coords isa AbstractVector && isempty(coords)
        return Face(nn, falses(nn))
    end

    if coords isa AbstractVector
        # Accept Bool masks even when eltype is Any (e.g. Any[true,false]).
        if all(x -> x isa Bool, coords)
            return Face(nn, Bool[x for x in coords])
        end

        # Accept integer masks or lists of indices even when eltype is Any.
        if all(x -> x isa Integer, coords)
            return Face(nn, Int[x for x in coords])
        end

        error("face: expected coords to be a vector of Bool or Integer; got eltype $(eltype(coords))")
    end

    # Nice ergonomic support for tuples like (1,2) or (false,true).
    if coords isa Tuple
        return face(nn, collect(coords))
    end

    error("face: expected coords to be an AbstractVector or Tuple; got $(typeof(coords))")
end


export Face, face, IndFlat, IndInj, Flange, canonical_matrix,
       active_flats, active_injectives, degree_matrix, dim_at, bounding_box
end # module
