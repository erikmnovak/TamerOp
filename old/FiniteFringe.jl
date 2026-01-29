module FiniteFringe

using SparseArrays, LinearAlgebra
using ..CoreModules: QQ
import ..ExactQQ: rankQQ

# =========================================
# Finite poset and indicator sets
# =========================================

struct FinitePoset
    n::Int
    leq::BitMatrix           # leq[i,j] = true  iff i <= j
    function FinitePoset(leq::AbstractMatrix{Bool})
        n1, n2 = size(leq)
        @assert n1 == n2 "leq must be square"

        # Normalize storage.
        L = leq isa BitMatrix ? copy(leq) : BitMatrix(leq)

        # Reflexive.
        for i in 1:n1
            if !L[i,i]
                error("FinitePoset: leq must be reflexive; missing leq[$i,$i] == true")
            end
        end

        # Antisymmetric: i<=j and j<=i implies i=j.
        for i in 1:n1
            for j in i+1:n1
                if L[i,j] && L[j,i]
                    error("FinitePoset: leq violates antisymmetry: leq[$i,$j] and leq[$j,$i] are both true")
                end
            end
        end

        # Transitive: i<=k and k<=j implies i<=j.
        for i in 1:n1
            for k in 1:n1
                if L[i,k]
                    for j in 1:n1
                        if L[k,j] && !L[i,j]
                            error("FinitePoset: leq violates transitivity at (i,k,j)=($i,$k,$j)")
                        end
                    end
                end
            end
        end

        new(n1, L)
    end
end

leq(P::FinitePoset, i::Int, j::Int) = P.leq[i,j]


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

# Convenience: findall(C) returns the list of cover edges.
Base.findall(C::CoverEdges) = C.edges

function cover_edges(P::FinitePoset)
    mat = falses(P.n, P.n)
    edges = Tuple{Int,Int}[]
    for i in 1:P.n, j in 1:P.n
        if i != j && leq(P, i, j) && !leq(P, j, i) # i<j
            # check minimality
            is_cover = true
            for k in 1:P.n
                if k != i && k != j &&
                   leq(P, i, k) && leq(P, k, j) &&
                   (leq(P, k, i) == false) && (leq(P, j, k) == false)
                    is_cover = false
                    break
                end
            end
            if is_cover
                mat[i, j] = true
                push!(edges, (i, j))
            end
        end
    end
    return CoverEdges(BitMatrix(mat), edges)
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

is_subset(U1::Upset, U2::Upset) = all(.!U1.mask .| U2.mask)
is_subset(D1::Downset, D2::Downset) = all(.!D1.mask .| D2.mask)
is_subset(a::BitVector, b::BitVector) = all(.!a .| b)
intersects(U::Upset, D::Downset) = any(U.mask .& D.mask)

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

# =========================================
# Fringe presentations (Defs. 3.16 - 3.17)
# =========================================

struct FringeModule{K, MAT<:AbstractMatrix{K}}
    P::FinitePoset
    U::Vector{Upset}                  # births (columns)
    D::Vector{Downset}                # deaths (rows)
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
    for i in 1:P.n, j in 1:P.n
        if C[i,j]
            push!(adj[i], j)
            push!(adj[j], i)
        end
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
