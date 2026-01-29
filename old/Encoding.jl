module Encoding
# =============================================================================
# Finite encodings ("uptight posets") from a finite family of constant upsets.
#
# References: Miller section 4 (Defs. 4.12 - 4.18 and Thm. 4.19 - 4.22).
# =============================================================================

using SparseArrays
using ..FiniteFringe

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
    FiniteFringe.FinitePoset(rel)
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

export EncodingMap, UptightEncoding,
       build_uptight_encoding_from_fringe, pullback_fringe_along_encoding,
       image_upset, image_downset, preimage_upset, preimage_downset

end # module
