module FiniteFringe

using SparseArrays, LinearAlgebra
using ..CoreModules: QQ
import ..ExactQQ: rankQQ

# =========================================
# Finite poset and indicator sets
# =========================================

"""
    FinitePoset(leq; check=true)

A finite poset stored by its order matrix.

The input `leq` is an `n x n` Boolean matrix with the intended meaning

    leq[i,j] == true    iff    i <= j.

By default (`check=true`) we validate that `leq` is a partial order, i.e. that it is

  * reflexive,
  * antisymmetric,
  * transitive.

The transitivity check is (in the worst case) cubic in `n`. For *known-good*
programmatically constructed posets (grid posets, boolean lattices, products of
chains, ...), you can set `check=false` to skip validation.

This is a performance feature; if you pass `check=false` and `leq` is not a
partial order, downstream computations may give nonsense.
"""
struct FinitePoset
    n::Int
    leq::BitMatrix           # leq[i,j] = true  iff i <= j
    function FinitePoset(leq::AbstractMatrix{Bool}; check::Bool=true)
        n1, n2 = size(leq)
        @assert n1 == n2 "leq must be square"

        # Normalize storage.
        L = leq isa BitMatrix ? leq : BitMatrix(leq)

        if check
            _validate_partial_order_matrix!(L)
        end

        new(n1, L)
    end
end

# Convenience constructor when you already know n.
# (This is also useful as a search/replace target in performance hot paths.)
FinitePoset(n::Int, leq::AbstractMatrix{Bool}; check::Bool=true) = begin
    @assert size(leq, 1) == n && size(leq, 2) == n "leq must be an n x n Boolean matrix"
    FinitePoset(leq; check=check)
end

leq(P::FinitePoset, i::Int, j::Int) = P.leq[i,j]


# ---- internal helpers ---------------------------------------------------------

# Return the smallest index j for which sub[j] is true and sup[j] is false.
# If sub is a subset of sup, return 0.
@inline function _subset_violation_index(sub::BitVector, sup::BitVector, n::Int)::Int
    @assert length(sub) == length(sup) == n
    sc = sub.chunks
    uc = sup.chunks
    nchunks = length(sc)

    # Mask off unused bits in the final word, to be robust to any garbage bits.
    r = n & 63
    lastmask = (r == 0) ? typemax(UInt64) : (UInt64(1) << r) - 1

    @inbounds for w in 1:nchunks
        diff = sc[w] & ~uc[w]
        if w == nchunks
            diff &= lastmask
        end
        if diff != 0
            tz = trailing_zeros(diff)
            j = (w - 1) * 64 + tz + 1
            return (j <= n) ? j : 0
        end
    end
    return 0
end

function _validate_partial_order_matrix!(L::BitMatrix)
    n1, n2 = size(L)
    @assert n1 == n2 "leq must be square"
    n = n1

    # Reflexive.
    @inbounds for i in 1:n
        if !L[i, i]
            error("FinitePoset: leq must be reflexive; missing leq[$i,$i] == true")
        end
    end

    # Antisymmetric: i<=j and j<=i implies i=j.
    @inbounds for i in 1:n
        for j in (i + 1):n
            if L[i, j] && L[j, i]
                error("FinitePoset: leq violates antisymmetry: leq[$i,$j] and leq[$j,$i] are both true")
            end
        end
    end

    # Transitive: i<=k and k<=j implies i<=j.
    #
    # Bitset formulation: whenever i<=k, the principal upset of k must be a subset
    # of the principal upset of i. We check subset failures via chunk-wise
    # operations to avoid scalar triple loops.
    rows = Vector{BitVector}(undef, n)
    @inbounds for i in 1:n
        rows[i] = L[i, :]
    end

    @inbounds for i in 1:n
        ri = rows[i]
        # NOTE: for BitVectors, the supported API is `findnext(bitvec, start)`.
        # The 3-argument form `findnext(bitvec, true, start)` is not defined for
        # BitArrays in Julia 1.12.
        k = findnext(ri, 1)
        while k !== nothing
            j = _subset_violation_index(rows[k], ri, n)
            if j != 0
                error("FinitePoset: leq violates transitivity at (i,k,j)=($i,$k,$j)")
            end
            k = findnext(ri, k + 1)
        end
    end

    return nothing
end


# Hasse cover relation (needed for resolutions):
# edge i -> j iff i<j and no k with i<k<j
#
# For user convenience, we return a lightweight wrapper that supports:
#   * adjacency queries via C[u,v] (Bool), and
#   * iteration over cover edges as (u,v) pairs (so Set(cover_edges(P)) works).
struct CoverEdges
    mat::BitMatrix
    edges::Vector{Tuple{Int,Int}}
end

Base.size(C::CoverEdges) = size(C.mat)
Base.getindex(C::CoverEdges, i::Int, j::Int) = C.mat[i,j]
Base.length(C::CoverEdges) = length(C.edges)
Base.eltype(::Type{CoverEdges}) = Tuple{Int,Int}
Base.IteratorSize(::Type{CoverEdges}) = Base.HasLength()
Base.iterate(C::CoverEdges, state::Int=1) =
    state > length(C.edges) ? nothing : (C.edges[state], state + 1)

# Allow BitMatrix(C) / Matrix(C) to recover the adjacency matrix when needed.
Base.convert(::Type{BitMatrix}, C::CoverEdges) = C.mat
Base.convert(::Type{Matrix{Bool}}, C::CoverEdges) = Matrix(C.mat)

# ---------------------------------------------------------------------------
# IMPORTANT constructor specializations:
#
# `CoverEdges` is iterable (over (i,j) edges). Julia's built-in `BitMatrix(x)` and
# `Matrix(x)` have generic iterator constructors, so without these methods
# `BitMatrix(C::CoverEdges)` attempts to interpret the edge iterator as a stream
# of Bool entries, which fails with a DimensionMismatch.
#
# Even though we provide `convert(BitMatrix, C)`, the `BitMatrix(C)` call does not
# necessarily go through `convert` because the iterator constructor is more
# specific than `convert` for iterable inputs.
#
# These specializations make the docstring guarantee "BitMatrix(C) works" true,
# and they are O(1) (no allocations).
# ---------------------------------------------------------------------------

# NOTE: In Julia, `BitMatrix` is a type alias for `BitArray{2}`. In practice,
# `BitMatrix(C)` may dispatch directly to `BitArray{2}(C)` (the underlying type
# constructor) instead of going through `convert`.
#
# To make `BitMatrix(C)` and `Matrix(C)` reliable (and O(1)), we provide
# constructors for the underlying types *and* for the alias names.

# Underlying type constructor (this is what `BitMatrix(C)` actually calls).
Base.BitArray{2}(C::CoverEdges) = C.mat

# A dense Bool matrix can be requested explicitly as `Array{Bool,2}(C)`.
Base.Array{Bool,2}(C::CoverEdges) = Matrix(C.mat)

# Alias names (readability / discoverability).
Base.BitMatrix(C::CoverEdges) = C.mat
Base.Matrix(C::CoverEdges) = Matrix(C.mat)


# Convenience: findall(C) returns the list of cover edges.
Base.findall(C::CoverEdges) = C.edges

# Cache cover edges by the identity of the underlying order matrix `P.leq`.
#
# IMPORTANT:
# - `FinitePoset` is an (immutable) struct, and `WeakKeyDict` requires *mutable*
#   keys because it attaches a finalizer to the key object.
# - `P.leq` is a `BitMatrix` (mutable) with stable identity and the same lifetime
#   as the poset data, so it is a safe weak-key "anchor".
#
# This preserves the intended behavior: compute cover edges once per poset,
# reuse them on repeated queries, and do not leak memory across many temporary
# posets.
const _COVER_EDGES_CACHE = WeakKeyDict{BitMatrix, CoverEdges}()


# BitVector helpers (chunk-level) for allocation-free set operations.
@inline function _tailmask(nbits::Int)::UInt64
    r = nbits & 63
    return (r == 0) ? typemax(UInt64) : (UInt64(1) << r) - 1
end

@inline function _or_chunks!(dest::BitVector, src::BitVector)
    dc = dest.chunks
    sc = src.chunks
    nchunks = length(dc)
    @inbounds for w in 1:nchunks
        dc[w] |= sc[w]
    end
    # Ensure unused tail bits are always zeroed.
    dc[end] &= _tailmask(length(dest))
    return dest
end

@inline function _andnot_chunks!(dest::BitVector, mask::BitVector)
    dc = dest.chunks
    mc = mask.chunks
    nchunks = length(dc)
    @inbounds for w in 1:nchunks
        dc[w] &= ~mc[w]
    end
    # Ensure unused tail bits are always zeroed.
    dc[end] &= _tailmask(length(dest))
    return dest
end

function _compute_cover_edges_bitset(P::FinitePoset)::CoverEdges
    n = P.n

    # Copy the rows of the order matrix as BitVectors so we can do chunk-wise
    # OR/AND operations. This is the key speedup over scalar triple loops.
    rows = Vector{BitVector}(undef, n)
    @inbounds for i in 1:n
        rows[i] = P.leq[i, :]
    end

    mat = falses(n, n)
    edges = Tuple{Int,Int}[]

    # Workspace reused across i.
    upper = falses(n)      # strict upper set of i, then the set of covers
    redundant = falses(n)  # elements proved non-minimal in upper

    @inbounds for i in 1:n
        copyto!(upper, rows[i])
        upper[i] = false
        fill!(redundant, false)

        # Mark all non-minimal elements of upper(i).
        #
        # For each k in upper(i), any element j > k is not minimal in upper(i),
        # hence i is not covered by j.
        k = findnext(upper, 1)
        while k !== nothing
            # We want strict_upper(k), i.e. exclude k itself. Since rows[k] is
            # reflexive, we OR it in and then restore redundant[k].
            prev = redundant[k]
            _or_chunks!(redundant, rows[k])
            redundant[k] = prev
            k = findnext(upper, k + 1)
        end

        # covers(i) = upper(i) \ redundant
        _andnot_chunks!(upper, redundant)

        j = findnext(upper, 1)
        while j !== nothing
            mat[i, j] = true
            push!(edges, (i, j))
            j = findnext(upper, j + 1)
        end
    end

    return CoverEdges(mat, edges)
end

"""
    cover_edges(P; cached=true)

Return the cover relation (Hasse diagram) of the finite poset `P`.

The return value is a `CoverEdges` object `C` supporting:

  * `C[i,j]`          (adjacency query),
  * iteration         (`for (i,j) in C`),
  * `findall(C)`      (edge list),
  * `BitMatrix(C)`    (adjacency matrix).

For performance, the result is cached per `FinitePoset` instance by default.
Pass `cached=false` to force recomputation.
"""
function cover_edges(P::FinitePoset; cached::Bool=true)
    # Use the order-matrix object identity as the cache key.
    L = P.leq
    if cached && haskey(_COVER_EDGES_CACHE, L)
        return _COVER_EDGES_CACHE[L]
    end
    C = _compute_cover_edges_bitset(P)
    if cached
        _COVER_EDGES_CACHE[L] = C
    end
    return C
end


struct Upset
    P::FinitePoset
    mask::BitVector
end
struct Downset
    P::FinitePoset
    mask::BitVector
end

struct _WPairData
    rows_by_ucomp::Vector{Vector{Int}}   # for each component of U_M[i], rows supported in that component
    rows_by_dcomp::Vector{Vector{Int}}   # for each component of D_N[t], rows supported in that component
end


Base.length(U::Upset) = length(U.mask)
Base.length(D::Downset) = length(D.mask)

# Allocation-free bitset predicates (used heavily in matching / slicing code).
@inline function is_subset(a::BitVector, b::BitVector)
    @assert length(a) == length(b)
    ac = a.chunks
    bc = b.chunks
    nchunks = length(ac)
    lastmask = _tailmask(length(a))
    @inbounds for w in 1:nchunks
        diff = ac[w] & ~bc[w]
        if w == nchunks
            diff &= lastmask
        end
        if diff != 0
            return false
        end
    end
    return true
end

@inline function intersects(a::BitVector, b::BitVector)
    @assert length(a) == length(b)
    ac = a.chunks
    bc = b.chunks
    nchunks = length(ac)
    lastmask = _tailmask(length(a))
    @inbounds for w in 1:nchunks
        v = ac[w] & bc[w]
        if w == nchunks
            v &= lastmask
        end
        if v != 0
            return true
        end
    end
    return false
end

is_subset(U1::Upset, U2::Upset) = is_subset(U1.mask, U2.mask)
is_subset(D1::Downset, D2::Downset) = is_subset(D1.mask, D2.mask)
intersects(U::Upset, D::Downset) = intersects(U.mask, D.mask)

# Upset/downset closures and principal sets (used in resolutions)
function upset_closure(P::FinitePoset, S::BitVector)
    U = copy(S)
    for i in 1:P.n, j in 1:P.n
        if U[i] && leq(P,i,j); U[j] = true; end
    end
    Upset(P, U)
end
function downset_closure(P::FinitePoset, S::BitVector)
    D = copy(S)
    for j in 1:P.n, i in 1:P.n
        if D[j] && leq(P,i,j); D[i] = true; end
    end
    Downset(P, D)
end

upset_from_generators(P::FinitePoset, gens::Vector{Int}) =
    upset_closure(P, BitVector([i in gens for i in 1:P.n]))
downset_from_generators(P::FinitePoset, gens::Vector{Int}) =
    downset_closure(P, BitVector([i in gens for i in 1:P.n]))

# Principal upset \uparrow p and principal downset \downarrow p (representables/corepresentables).
principal_upset(P::FinitePoset, p::Int) = Upset(P, BitVector([leq(P, p, q) for q in 1:P.n]))
principal_downset(P::FinitePoset, p::Int) = Downset(P, BitVector([leq(P, q, p) for q in 1:P.n]))

#############################
# Structural equality + hashing
#############################

import Base: ==, isequal, hash

# ---- FinitePoset ----
# Adjust field names (:n, :leq) to match your struct definition.
# From your printout it looks like FinitePoset(3, Bool[...]) so probably:
#   n::Int
#   leq::AbstractMatrix{Bool}   (or BitMatrix / Matrix{Bool})
==(P::FinitePoset, Q::FinitePoset) =
    (P.n == Q.n) && (P.leq == Q.leq)

isequal(P::FinitePoset, Q::FinitePoset) = (P == Q)

hash(P::FinitePoset, h::UInt) = hash(P.n, hash(P.leq, h))


# ---- Upset ----
# Adjust field names to match your Upset struct.
# From your printout: Upset(FinitePoset(...), Bool[...]) so likely:
#   P::FinitePoset
#   mem::AbstractVector{Bool}   (or BitVector)
==(U::Upset, V::Upset) =
    (U.P == V.P) && (U.mask == V.mask)

isequal(U::Upset, V::Upset) = (U == V)

hash(U::Upset, h::UInt) = hash(U.P, hash(U.mask, h))


# ---- Downset ----
# Same idea as Upset. Adjust field names.
==(D::Downset, E::Downset) =
    (D.P == E.P) && (D.mask == E.mask)

isequal(D::Downset, E::Downset) = (D == E)

hash(D::Downset, h::UInt) = hash(D.P, hash(D.mask, h))


# =========================================
# Fringe presentations (Defs. 3.16 - 3.17)
# =========================================

struct FringeModule{K, MAT<:AbstractMatrix{K}}
    P::FinitePoset
    U::Vector{Upset}                  # birth upsets (columns)
    D::Vector{Downset}                # death downsets (rows)
    phi::MAT                          # size |D| x |U|

    function FringeModule{K,MAT}(P::FinitePoset,
                                 U::Vector{Upset},
                                 D::Vector{Downset},
                                 phi::MAT) where {K, MAT<:AbstractMatrix{K}}
        @assert size(phi,1) == length(D) && size(phi,2) == length(U)

        M = new{K,MAT}(P, U, D, phi)
        _check_monomial_condition(M)
        return M
    end
end

# Allow calls like FringeModule{K}(P, U, D, phi) by inferring MAT from phi.
function FringeModule{K}(P::FinitePoset,
                         U::Vector{Upset},
                         D::Vector{Downset},
                         phi::AbstractMatrix{K}) where {K}
    return FringeModule{K, typeof(phi)}(P, U, D, phi)
end

# Convenience constructor (default dense; set store_sparse=true to store CSC).
function FringeModule(P::FinitePoset,
                      U::Vector{Upset},
                      D::Vector{Downset},
                      phi::AbstractMatrix{K}; store_sparse::Bool=false) where {K}
    phimat = store_sparse ? SparseArrays.sparse(phi) : Matrix{K}(phi)
    return FringeModule{K, typeof(phimat)}(P, U, D, phimat)
end

"""
    one_by_one_fringe(P, U, D[, scalar]) -> FringeModule

Convenience constructor for a 1x1 fringe presentation:

- one upset generator `U`,
- one downset cogenerator `D`,
- and a 1x1 structure matrix `phi = [scalar]`.

This is handy for tests, examples, and quickly building interval-like modules
on a finite poset without writing out the matrix by hand.

Notes:
- The 3-argument form defaults to `scalar = QQ(1)`.
- The 4-argument form is fully generic in the scalar type `K`.
- A keyword form `; scalar=...` is also provided for ergonomics.
- In addition to `U::Upset` / `D::Downset`, you may also pass membership masks
  `U_mask::AbstractVector{Bool}` and `D_mask::AbstractVector{Bool}`.
"""
function one_by_one_fringe(P::FinitePoset, U::Upset, D::Downset, scalar::K) where {K}
    phi = spzeros(K, 1, 1)
    phi[1, 1] = scalar
    return FringeModule{K}(P, [U], [D], phi)
end

# Default scalar (the project's base field QQ).
one_by_one_fringe(P::FinitePoset, U::Upset, D::Downset) =
    one_by_one_fringe(P, U, D, QQ(1))

# Keyword-friendly wrapper.
one_by_one_fringe(P::FinitePoset, U::Upset, D::Downset; scalar=QQ(1)) =
    one_by_one_fringe(P, U, D, scalar)

# ------------------ mask-based convenience overloads ------------------

function _coerce_bool_mask(P::FinitePoset, mask::AbstractVector{Bool}, name::AbstractString)::BitVector
    if length(mask) != P.n
        error(name * " mask must have length P.n=" * string(P.n) *
              "; got length " * string(length(mask)) * ".")
    end
    return (mask isa BitVector) ? mask : BitVector(mask)
end

function one_by_one_fringe(P::FinitePoset,
                           U_mask::AbstractVector{Bool},
                           D_mask::AbstractVector{Bool},
                           scalar::K) where {K}
    Um = _coerce_bool_mask(P, U_mask, "Upset")
    Dm = _coerce_bool_mask(P, D_mask, "Downset")
    return one_by_one_fringe(P, Upset(P, Um), Downset(P, Dm), scalar)
end

one_by_one_fringe(P::FinitePoset,
                  U_mask::AbstractVector{Bool},
                  D_mask::AbstractVector{Bool}) =
    one_by_one_fringe(P, U_mask, D_mask, QQ(1))

one_by_one_fringe(P::FinitePoset,
                  U_mask::AbstractVector{Bool},
                  D_mask::AbstractVector{Bool}; scalar=QQ(1)) =
    one_by_one_fringe(P, U_mask, D_mask, scalar)


# ------------------ mask-based convenience overloads ------------------

# Coerce a Bool mask to a BitVector and sanity-check its length.
function _coerce_bool_mask(P::FinitePoset, mask::AbstractVector{Bool}, name::AbstractString)::BitVector
    if length(mask) != P.n
        error(name * " mask must have length P.n=" * string(P.n) *
              "; got length " * string(length(mask)) * ".")
    end
    return (mask isa BitVector) ? mask : BitVector(mask)
end

# Allow U/D to be passed as membership masks (BitVector / Vector{Bool}).
function one_by_one_fringe(P::FinitePoset,
                           U_mask::AbstractVector{Bool},
                           D_mask::AbstractVector{Bool},
                           scalar::K) where {K}
    Um = _coerce_bool_mask(P, U_mask, "Upset")
    Dm = _coerce_bool_mask(P, D_mask, "Downset")
    return one_by_one_fringe(P, Upset(P, Um), Downset(P, Dm), scalar)
end

# Default scalar also for mask-based call.
one_by_one_fringe(P::FinitePoset,
                  U_mask::AbstractVector{Bool},
                  D_mask::AbstractVector{Bool}) =
    one_by_one_fringe(P, U_mask, D_mask, QQ(1))

# Keyword form for parity with the Upset/Downset signature.
one_by_one_fringe(P::FinitePoset,
                  U_mask::AbstractVector{Bool},
                  D_mask::AbstractVector{Bool}; scalar=QQ(1)) =
    one_by_one_fringe(P, U_mask, D_mask, scalar)



# Prop. 3.18: Nonzero entry only if U_i \cap D_j \neq \emptyset.
function _check_monomial_condition(M::FringeModule{K}) where {K}
    m, n = size(M.phi)
    @assert m == length(M.D) && n == length(M.U) "Dimension mismatch"

    phi = M.phi
    if phi isa SparseMatrixCSC
        I, J, V = findnz(phi)
        for t in eachindex(V)
            v = V[t]
            if v != zero(K)
                j = I[t]; i = J[t]
                @assert intersects(M.U[i], M.D[j]) "Nonzero phi[j,i] requires U[i] cap D[j] neq emptyset (Prop. 3.18)"
            end
        end
    else
        for j in 1:m, i in 1:n
            v = phi[j,i]
            if v != zero(K)
                @assert intersects(M.U[i], M.D[j]) "Nonzero phi[j,i] requires U[i] cap D[j] neq emptyset (Prop. 3.18)"
            end
        end
    end
end


# ------------------ evaluation (degreewise image; after Def. 3.17) ------------------


"""
    fiber_dimension(M::FringeModule{K}, q::Int) -> Int

Compute dim_k M_q as rank of phi_q : F_q \to E_q (degreewise image).
"""
function fiber_dimension(M::FringeModule{K}, q::Int) where {K}
    cols = findall(U -> U.mask[q], M.U)
    rows = findall(D -> D.mask[q], M.D)
    if isempty(cols) || isempty(rows); return 0; end

    phi_q = Matrix{QQ}(M.phi[rows, cols])
    return rankQQ(phi_q)
end

# ------------------ Hom for fringe modules via commuting squares ------------------
#
# We build the linear system for commuting squares in the indicator-module category
# using the full Hom descriptions from Prop. 3.10 (componentwise bases).
#
# V1 = Hom(F_M, F_N)  basis = components of U_M[i] contained in U_N[j]
# V2 = Hom(E_M, E_N)  basis = components of D_N[t] contained in D_M[s]
# W  = Hom(F_M, E_N)  basis = components of U_M[i] cap D_N[t]
#
# Then Hom(M,N) = ker(d0) / (ker(T) + ker(S)) with d0 = [T  -S].

"Undirected adjacency of the Hasse cover graph."
function _cover_undirected_adjacency(P::FinitePoset)
    C = cover_edges(P)
    adj = [Int[] for _ in 1:P.n]
    for (i, j) in C
        push!(adj[i], j)
        push!(adj[j], i)
    end
    return adj
end

"Connected components of a subset mask in the undirected Hasse cover graph."
function _component_data(adj::Vector{Vector{Int}}, mask::BitVector)
    n = length(mask)
    comp = fill(0, n)
    reps = Int[]
    cid = 0
    for v in 1:n
        if mask[v] && comp[v] == 0
            cid += 1
            push!(reps, v)
            queue = [v]
            comp[v] = cid
            head = 1
            while head <= length(queue)
                x = queue[head]; head += 1
                for y in adj[x]
                    if mask[y] && comp[y] == 0
                        comp[y] = cid
                        push!(queue, y)
                    end
                end
            end
        end
    end

    comp_masks = [falses(n) for _ in 1:cid]
    for v in 1:n
        c = comp[v]
        if c != 0
            comp_masks[c][v] = true
        end
    end
    return comp, cid, comp_masks, reps
end

"Dimension of Hom(M,N) over a field K using the commuting-square presentation."
function hom_dimension(M::FringeModule{K}, N::FringeModule{K}) where {K}
    @assert M.P === N.P "Posets must match"
    P = M.P

    # Precompute undirected cover adjacency once.
    adj = _cover_undirected_adjacency(P)

    nUM = length(M.U); nDM = length(M.D)
    nUN = length(N.U); nDN = length(N.D)

    # Component decompositions for all upsets in M and downsets in N.
    Ucomp_id_M    = Vector{Vector{Int}}(undef, nUM)
    Ucomp_masks_M = Vector{Vector{BitVector}}(undef, nUM)
    Ucomp_n_M     = Vector{Int}(undef, nUM)
    for i in 1:nUM
        comp_id, ncomp, comp_masks, _ = _component_data(adj, M.U[i].mask)
        Ucomp_id_M[i] = comp_id
        Ucomp_masks_M[i] = comp_masks
        Ucomp_n_M[i] = ncomp
    end

    Dcomp_id_N    = Vector{Vector{Int}}(undef, nDN)
    Dcomp_masks_N = Vector{Vector{BitVector}}(undef, nDN)
    Dcomp_n_N     = Vector{Int}(undef, nDN)
    for t in 1:nDN
        comp_id, ncomp, comp_masks, _ = _component_data(adj, N.D[t].mask)
        Dcomp_id_N[t] = comp_id
        Dcomp_masks_N[t] = comp_masks
        Dcomp_n_N[t] = ncomp
    end

    # Build W = oplus_{i,t} Hom(k[U_M[i]], k[D_N[t]]) with basis indexed by components of U_i cap D_t.
    w_index = Dict{Tuple{Int,Int},Int}()   # (iM,tN) -> index into w_data
    w_data  = _WPairData[]
    W_dim = 0

    for iM in 1:nUM
        for tN in 1:nDN
            mask_int = M.U[iM].mask .& N.D[tN].mask
            if any(mask_int)
                _, ncomp_int, _, reps_int = _component_data(adj, mask_int)
                base = W_dim
                W_dim += ncomp_int

                rows_by_u = [Int[] for _ in 1:Ucomp_n_M[iM]]
                rows_by_d = [Int[] for _ in 1:Dcomp_n_N[tN]]
                for c in 1:ncomp_int
                    row = base + c
                    v = reps_int[c]
                    cu = Ucomp_id_M[iM][v]
                    cd = Dcomp_id_N[tN][v]
                    push!(rows_by_u[cu], row)
                    push!(rows_by_d[cd], row)
                end

                push!(w_data, _WPairData(rows_by_u, rows_by_d))
                w_index[(iM,tN)] = length(w_data)
            end
        end
    end

    # V1 basis: components of U_M[i] contained in U_N[j].
    V1 = Tuple{Int,Int,Int}[]  # (iM, jN, compU)
    for iM in 1:nUM
        for jN in 1:nUN
            for cU in 1:Ucomp_n_M[iM]
                if is_subset(Ucomp_masks_M[iM][cU], N.U[jN].mask)
                    push!(V1, (iM, jN, cU))
                end
            end
        end
    end
    V1_dim = length(V1)

    # V2 basis: components of D_N[t] contained in D_M[s].
    V2 = Tuple{Int,Int,Int}[]  # (sM, tN, compD)
    for sM in 1:nDM
        for tN in 1:nDN
            for cD in 1:Dcomp_n_N[tN]
                if is_subset(Dcomp_masks_N[tN][cD], M.D[sM].mask)
                    push!(V2, (sM, tN, cD))
                end
            end
        end
    end
    V2_dim = length(V2)

    # Build T and S as dense matrices (exact arithmetic, small sizes expected).
    T = zeros(K, W_dim, V1_dim)
    for (col, (iM, jN, cU)) in enumerate(V1)
        for tN in 1:nDN
            val = N.phi[tN, jN]
            if val != zero(K)
                pid = get(w_index, (iM,tN), 0)
                if pid != 0
                    rows = w_data[pid].rows_by_ucomp[cU]
                    for r in rows
                        T[r, col] += val
                    end
                end
            end
        end
    end

    S = zeros(K, W_dim, V2_dim)
    for (col, (sM, tN, cD)) in enumerate(V2)
        for iM in 1:nUM
            val = M.phi[sM, iM]
            if val != zero(K)
                pid = get(w_index, (iM,tN), 0)
                if pid != 0
                    rows = w_data[pid].rows_by_dcomp[cD]
                    for r in rows
                        S[r, col] += val
                    end
                end
            end
        end
    end

    Tdense = Matrix{QQ}(T)
    Sdense = Matrix{QQ}(S)
    big = hcat(Tdense, -Sdense)

    rT   = rankQQ(Tdense)
    rS   = rankQQ(Sdense)
    rBig = rankQQ(big)

    dimKer_big = (V1_dim + V2_dim) - rBig
    dimKer_T   = V1_dim - rT
    dimKer_S   = V2_dim - rS
    return dimKer_big - (dimKer_T + dimKer_S)
end


# ---------- utility: dense\tosparse over K ----------

"""
    dense_to_sparse_K(A)

Convert a dense matrix `A` to a sparse matrix with the same element type.

For an explicit target coefficient type `K`, call `dense_to_sparse_K(A, K)`.
"""
function dense_to_sparse_K(A::AbstractMatrix{T}, ::Type{K}) where {T,K}
    m,n = size(A)
    S = spzeros(K, m, n)
    for j in 1:n, i in 1:m
        v = K(A[i,j])
        if v != zero(K); S[i,j] = v; end
    end
    S
end

dense_to_sparse_K(A::AbstractMatrix{T}) where {T} = dense_to_sparse_K(A, T)

export FinitePoset, CoverEdges, Upset, Downset, principal_upset, principal_downset,
       upset_from_generators, downset_from_generators, cover_edges,
       FringeModule, one_by_one_fringe, fiber_dimension, hom_dimension, dense_to_sparse_K

end # module


# -----------------------------------------------------------------------------
# IndicatorTypes (merged from the old src/IndicatorTypes.jl)
#
# This module contains small shared record types for one-step indicator
# presentations/copresentations (Def. 6.4 in Miller).
#
# We keep the module name `PosetModules.IndicatorTypes` and the type names
# unchanged for API stability. The only change is *file placement*:
# these types depend only on FiniteFringe + SparseArrays, so defining them here
# keeps the include-order simple (they are available early).
# -----------------------------------------------------------------------------

module IndicatorTypes
# Small shared types for indicator presentations/copresentations.
# These mirror the data in Def. 6.4 (Miller) and are used by Hom/Ext assembly.

using SparseArrays
using ..FiniteFringe: FinitePoset, Upset, Downset  # P, labeling sets

"""
    UpsetPresentation{K}

A one-step **upset presentation** (Def. 6.4): F1 --delta--> F0 -> M with labels by upsets.
Fields:
- `P`: underlying finite poset
- `U0`: column labels for F0 (birth upsets)
- `U1`: row labels for F1
- `delta`: block monomial matrix delta: (#U1) x (#U0), entries in `K`
"""
struct UpsetPresentation{K}
    P::FinitePoset
    U0::Vector{Upset}
    U1::Vector{Upset}
    delta::SparseMatrixCSC{K,Int}
end

"""
    DownsetCopresentation{K}

A one-step **downset copresentation** (Def. 6.4 dual): M -> E^0 --rho--> E^1 with labels by downsets.
Fields:
- `P`: underlying finite poset
- `D0`: row labels for E^0 (death downsets)
- `D1`: column labels for E^1
- `rho`: block monomial matrix rho: (#D1) x (#D0), entries in `K`
"""
struct DownsetCopresentation{K}
    P::FinitePoset
    D0::Vector{Downset}
    D1::Vector{Downset}
    rho::SparseMatrixCSC{K,Int}
end

export UpsetPresentation, DownsetCopresentation
end # module IndicatorTypes


module Encoding
# =============================================================================
# Finite encodings ("uptight posets") from a finite family of constant upsets.
#
# References: Miller section 4 (Defs. 4.12 - 4.18 and Thm. 4.19 - 4.22).
# =============================================================================

using SparseArrays
using ..FiniteFringe

# This submodule defines methods on CoreModules.AbstractPLikeEncodingMap and
# calls CoreModules.locate/dimension/etc., so we must import the sibling module
# binding into scope (not just individual names).
import ..CoreModules

# ----------------------------- Data structures -------------------------------

"""
    EncodingMap

A finite encoding `pi : Q to P`, where `Q` and `P` are finite posets.

Fields
- `Q` : source poset
- `P` : target poset (the uptight poset)
- `pi_of_q` : a vector of length `Q.n` with `pi_of_q[q] in 1:P.n`
"""
struct EncodingMap
    Q::FiniteFringe.FinitePoset
    P::FiniteFringe.FinitePoset
    pi_of_q::Vector{Int}
end

"""
    UptightEncoding

Bundle that stores `pi` together with the family `Y` of constant upsets used
to construct the uptight poset. This is handy for inspection and debugging.
"""
struct UptightEncoding
    pi::EncodingMap
    Y::Vector{FiniteFringe.Upset}
end

# ------------------ Postcomposition with a finite encoding map ---------------

"""
    PostcomposedEncodingMap(pi0, pi)

Postcompose an *ambient* encoding map `pi0` (e.g. Z^n- or R^n-encoding) with a *finite*
encoding map `pi : Q -> P` (an `EncodingMap`). Conceptually:

    x mapsto q = locate(pi0, x) mapsto pi(q)

This is used by `Workflow.coarsen` to keep user-facing `encode(...)` semantics stable
while compressing the *finite* encoding poset.
"""
struct PostcomposedEncodingMap{PI<:CoreModules.AbstractPLikeEncodingMap} <: CoreModules.AbstractPLikeEncodingMap
    pi0::PI
    pi_of_q::Vector{Int}              # pi : Q -> P encoded as forward table
    Pn::Int                           # number of regions in P
    reps_cache::Base.RefValue{Any}    # lazy cache for representatives(::...)
end

@inline PostcomposedEncodingMap(pi0::CoreModules.AbstractPLikeEncodingMap, pi::EncodingMap) =
    PostcomposedEncodingMap(pi0, pi.pi_of_q, pi.P.n, Ref{Any}(nothing))

@inline function CoreModules.locate(pi::PostcomposedEncodingMap, x::AbstractVector)
    q = CoreModules.locate(pi.pi0, x)
    q == 0 && return 0
    @inbounds return pi.pi_of_q[q]
end

# Disambiguation:
# There is a generic fallback
#   locate(::AbstractPLikeEncodingMap, ::NTuple{N,<:Real})
# in CoreModules. Since PostcomposedEncodingMap <: AbstractPLikeEncodingMap,
# we must ensure our method is MORE specific on the x-type too, otherwise
# calls like locate(pi, (q,)) become ambiguous.
@inline function CoreModules.locate(
    pi::PostcomposedEncodingMap,
    x::NTuple{N,T},
) where {N, T<:Real}
    q = CoreModules.locate(pi.pi0, x)
    q == 0 && return 0
    @inbounds return pi.pi_of_q[q]
end


@inline CoreModules.dimension(pi::PostcomposedEncodingMap) = CoreModules.dimension(pi.pi0)
@inline CoreModules.axes_from_encoding(pi::PostcomposedEncodingMap) = CoreModules.axes_from_encoding(pi.pi0)

function CoreModules.representatives(pi::PostcomposedEncodingMap)
    cached = pi.reps_cache[]
    cached !== nothing && return cached

    reps0 = CoreModules.representatives(pi.pi0)
    repsP = Vector{eltype(reps0)}(undef, pi.Pn)
    filled = falses(pi.Pn)

    @inbounds for q in eachindex(pi.pi_of_q)
        p = pi.pi_of_q[q]
        if !filled[p]
            repsP[p] = reps0[q]
            filled[p] = true
        end
    end

    all(filled) || error("PostcomposedEncodingMap: could not build representatives for all regions.")

    pi.reps_cache[] = repsP
    return repsP
end

# ----------------------- Uptight regions from a family Y ---------------------

# Partition Q into uptight regions: a ~ b iff they lie in exactly the same members of Y.
function _uptight_regions(Q::FiniteFringe.FinitePoset, Y::Vector{FiniteFringe.Upset})
    # Use tuples of Bool so Dict compares and hashes by contents.
    sigs = Dict{Tuple{Vararg{Bool}}, Vector{Int}}()
    for q in 1:Q.n
        key = ntuple(i -> Y[i].mask[q], length(Y))  # immutable, content-hashable
        push!(get!(sigs, key, Int[]), q)
    end
    return collect(values(sigs))
end

# Build the partial order on regions: A <= B if exists a \in A, b \in B with a <= b in Q (Prop. 4.15),
# then take transitive closure (Def. 4.17).
function _uptight_poset(Q::FiniteFringe.FinitePoset, regions::Vector{Vector{Int}})
    r = length(regions)
    rel = falses(r, r)
    for A in 1:r, B in 1:r
        if A == B
            rel[A,B] = true
            continue
        end
        found = false
        for a in regions[A], b in regions[B]
            if FiniteFringe.leq(Q, a, b); found = true; break; end
        end
        rel[A,B] = found
    end
    # Transitive closure (Floyd-Warshall boolean)
    for k in 1:r, i in 1:r, j in 1:r
        rel[i,j] = rel[i,j] || (rel[i,k] && rel[k,j])
    end
    FiniteFringe.FinitePoset(rel; check=false)
end

# Build the encoding map pi : Q \to P_Y
function _encoding_map(Q::FiniteFringe.FinitePoset,
                       P::FiniteFringe.FinitePoset,
                       regions::Vector{Vector{Int}})
    pi_of_q = zeros(Int, Q.n)
    for (idx, R) in enumerate(regions)
        for q in R
            pi_of_q[q] = idx
        end
    end
    EncodingMap(Q, P, pi_of_q)
end

# -------------------------- Image / preimage helpers -------------------------

"Image of a Q-upset under `pi` as a P-upset (Def. 4.12 / Remark 4.13)."
function image_upset(pi::EncodingMap, U::FiniteFringe.Upset)
    maskP = falses(pi.P.n)
    for q in 1:pi.Q.n
        if U.mask[q]; maskP[pi.pi_of_q[q]] = true; end
    end
    FiniteFringe.upset_closure(pi.P, maskP)
end

"Image of a Q-downset under `pi` as a P-downset."
function image_downset(pi::EncodingMap, D::FiniteFringe.Downset)
    maskP = falses(pi.P.n)
    for q in 1:pi.Q.n
        if D.mask[q]; maskP[pi.pi_of_q[q]] = true; end
    end
    FiniteFringe.downset_closure(pi.P, maskP)
end

"Preimage of a P-upset under `pi` as a Q-upset."
function preimage_upset(pi::EncodingMap, Uhat::FiniteFringe.Upset)
    maskQ = falses(pi.Q.n)
    for q in 1:pi.Q.n
        if Uhat.mask[pi.pi_of_q[q]]; maskQ[q] = true; end
    end
    FiniteFringe.upset_closure(pi.Q, maskQ)
end

"Preimage of a P-downset under `pi` as a Q-downset."
function preimage_downset(pi::EncodingMap, Dhat::FiniteFringe.Downset)
    maskQ = falses(pi.Q.n)
    for q in 1:pi.Q.n
        if Dhat.mask[pi.pi_of_q[q]]; maskQ[q] = true; end
    end
    FiniteFringe.downset_closure(pi.Q, maskQ)
end

# ---------------------------- Public constructors ----------------------------

"""
    build_uptight_encoding_from_fringe(M::FringeModule) -> UptightEncoding

Given a fringe presentation on `Q` with upsets `U_i` (births) and downsets `D_j` (deaths),
form the finite family `Y = { U_i } cup { complement(D_j) }` of constant upsets (Def. 4.18),
build the uptight regions (Defs. 4.12 - 4.17), and return the finite encoding `pi: Q -> P_Y`.
"""
function build_uptight_encoding_from_fringe(M::FiniteFringe.FringeModule)
    Q = M.P
    Y = FiniteFringe.Upset[]
    append!(Y, M.U)
    for Dj in M.D
        comp = BitVector(.!Dj.mask)                 # complement is also a Q-upset
        push!(Y, FiniteFringe.upset_closure(Q, comp))
    end
    regions = _uptight_regions(Q, Y)
    P = _uptight_poset(Q, regions)
    pi = _encoding_map(Q, P, regions)
    UptightEncoding(pi, Y)
end

"""
    pullback_fringe_along_encoding(H_hat::FringeModule_on_P, pi::EncodingMap) -> FringeModule_on_Q

Prop. 4.11 (used in the proof of Thm. 6.12): pull back a monomial matrix for a module on `P`
by replacing row labels `D_hat_j` with `pi^{-1}(D_hat_j)` and column labels `U_hat_i` with `pi^{-1}(U_hat_i)`.
The scalar matrix is unchanged.
"""
function pullback_fringe_along_encoding(Hhat::FiniteFringe.FringeModule, pi::EncodingMap)
    UQ = [preimage_upset(pi, Uhat) for Uhat in Hhat.U]
    DQ = [preimage_downset(pi, Dhat) for Dhat in Hhat.D]
    FiniteFringe.FringeModule{eltype(Hhat.phi)}(pi.Q, UQ, DQ, Hhat.phi)
end

"""
    pushforward_fringe_along_encoding(H::FringeModule_on_Q, pi::EncodingMap) -> FringeModule_on_P

Push a fringe presentation forward along a finite encoding map `pi : Q -> P`.

This sends each upset generator `U_i` of `H` to `image_upset(pi, U_i)` and each
downset generator `D_j` to `image_downset(pi, D_j)`, while keeping the scalar matrix
`phi` unchanged.
"""
function pushforward_fringe_along_encoding(H::FiniteFringe.FringeModule, pi::EncodingMap)
    Uhat = [image_upset(pi, U) for U in H.U]
    Dhat = [image_downset(pi, D) for D in H.D]
    FiniteFringe.FringeModule{eltype(H.phi)}(pi.P, Uhat, Dhat, H.phi)
end


export EncodingMap, UptightEncoding,
       build_uptight_encoding_from_fringe, pullback_fringe_along_encoding,
       image_upset, image_downset, preimage_upset, preimage_downset,
       pullback_fringe_along_encoding, pushforward_fringe_along_encoding       

end # module
