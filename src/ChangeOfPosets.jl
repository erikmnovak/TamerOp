module ChangeOfPosets

# =============================================================================
# Change-of-poset functors for modules:
#   - pullback (restriction) along a monotone map pi: Q -> P
#   - left Kan extension (left pushforward) along pi
#   - right Kan extension (right pushforward) along pi
#   - derived functors of the above (object-level)
#
# This is the "next layer" beyond Encoding.jl's image/preimage of up/down-sets.
# =============================================================================

using LinearAlgebra
using SparseArrays

import ..FiniteFringe
using ..FiniteFringe: AbstractPoset, FinitePoset, ProductPoset, cover_edges, leq, leq_matrix, nvertices, poset_equal,
                      downset_indices, upset_indices
using ..Encoding: EncodingMap
using ..CoreModules: QQ, ResolutionOptions, DerivedFunctorOptions

@inline _resolve_df_opts(opts::Union{DerivedFunctorOptions,Nothing}) =
    opts === nothing ? DerivedFunctorOptions() : opts
using ..ExactQQ: nullspaceQQ, solve_fullcolumnQQ

import ..IndicatorResolutions
using ..Modules: PModule, PMorphism, map_leq
import ..AbelianCategories: pullback

using ..DerivedFunctors: projective_resolution, injective_resolution,
                         lift_injective_chainmap

# lift_chainmap lives in DerivedFunctors.Functoriality.
using ..DerivedFunctors.Functoriality: lift_chainmap

import ..ModuleComplexes
using ..ModuleComplexes: ModuleCochainComplex, ModuleCochainMap,
                         cohomology_module, induced_map_on_cohomology_modules

export restriction,
       pushforward_left, pushforward_right,
       left_kan_extension, right_kan_extension,
       Lpushforward_left, Rpushforward_right,
       derived_pushforward_left, derived_pushforward_right,
       pushforward_left_complex, pushforward_right_complex,
       product_poset,
       encode_pmodules_to_common_poset

# -----------------------------------------------------------------------------
# Utilities: compatibility, monotonicity, terminal/initial detection
# -----------------------------------------------------------------------------

# Compare posets structurally.
@inline _same_poset(A::AbstractPoset, B::AbstractPoset)::Bool = poset_equal(A, B)

"""
    _check_monotone(pi)

Throw an error if `pi::EncodingMap` is not order-preserving.
It is enough to check cover edges, since they generate the order.
"""
function _check_monotone(pi::EncodingMap)
    Q = pi.Q
    P = pi.P
    f = pi.pi_of_q

    if length(f) != nvertices(Q)
        error("EncodingMap.pi_of_q must have length nvertices(Q)")
    end
    for q in 1:nvertices(Q)
        if f[q] < 1 || f[q] > nvertices(P)
            error("EncodingMap.pi_of_q values must lie in 1..nvertices(P)")
        end
    end

    for (u,v) in cover_edges(Q)
        if !leq(P, f[u], f[v])
            error("EncodingMap is not monotone: $u <= $v in Q but pi($u) !<= pi($v) in P")
        end
    end
    return nothing
end

"""
    _maximum_element(Q, S)

Return the maximum element of `S` (viewed as a subset of poset `Q`)
if it exists, otherwise return `nothing`.
"""
function _maximum_element(Q::AbstractPoset, S::Vector{Int})::Union{Int,Nothing}
    isempty(S) && return nothing
    cand = S[1]
    @inbounds for k in 2:length(S)
        s = S[k]
        if leq(Q, cand, s) && !leq(Q, s, cand)
            cand = s
        end
    end
    @inbounds for s in S
        if !leq(Q, s, cand)
            return nothing
        end
    end
    return cand
end

"""
    _minimum_element(Q, S)

Return the minimum element of `S` if it exists, else `nothing`.
"""
function _minimum_element(Q::AbstractPoset, S::Vector{Int})::Union{Int,Nothing}
    isempty(S) && return nothing
    cand = S[1]
    @inbounds for k in 2:length(S)
        s = S[k]
        if leq(Q, s, cand) && !leq(Q, cand, s)
            cand = s
        end
    end
    @inbounds for s in S
        if !leq(Q, cand, s)
            return nothing
        end
    end
    return cand
end

# -----------------------------------------------------------------------------
# Product posets and "common refinement" encoding for modules on different posets
# -----------------------------------------------------------------------------
#
# Motivation (Case C):
# If M1 is a PModule on P1 and M2 is a PModule on P2, the library's Hom(M1,M2)
# requires the SAME poset object. Mathematically, a standard way to compare them
# is to choose a common refinement poset and pull both modules back.
#
# The standard refinement is the cartesian product poset P = P1 x P2 with:
#   (i1,j1) <= (i2,j2)  iff  i1 <= i2 in P1  AND  j1 <= j2 in P2.
#
# Then we pull back M1 along pr1 : P -> P1 and M2 along pr2 : P -> P2, and compute
# Hom on the resulting common-poset modules.
#
# Performance notes:
#   * Constructing the product leq matrix is inherently O((n1*n2)^2) bits.
#   * Computing cover edges from scratch on the product is very expensive; we
#     pre-populate the cover cache using the known cover-edge structure of a
#     cartesian product poset:
#         covers are exactly "change one coordinate by a cover edge, keep the other fixed".
#   * Pullback along projections is implemented without calling map_leq on every edge
#     (which would allocate tons of identity matrices). We reuse identity matrices
#     and reuse the original cover-edge maps by reference.
#
# This is meant as a convenience layer (similar UX to encode_pmodules_from_flanges / PL_fringes):
#     P, Ms, pi1, pi2 = encode_pmodules_to_common_poset(M1, M2)
#     H = Hom(Ms[1], Ms[2])

# Cache product posets by identity of their leq matrices for dense posets.
# For structured posets, use object identity directly.
const _PRODUCT_POSET_CACHE = IdDict{BitMatrix, IdDict{BitMatrix, NamedTuple}}()
const _PRODUCT_POSET_OBJ_CACHE = IdDict{AbstractPoset, IdDict{AbstractPoset, NamedTuple}}()

# Pre-populate the FiniteFringe cover-edges cache for the cartesian product poset.
# This avoids the expensive generic cover-edge computation on P1 x P2.
function _cache_product_cover_edges!(Pprod::FinitePoset, P1::FinitePoset, P2::FinitePoset)
    n1, n2 = P1.n, P2.n
    n = n1 * n2

    C1 = cover_edges(P1)
    C2 = cover_edges(P2)

    # cover adjacency matrix + edge list
    mat = falses(n, n)
    edges = Vector{Tuple{Int,Int}}()
    sizehint!(edges, length(C1) * n2 + n1 * length(C2))

    @inbounds begin
        # Horizontal covers: (i1,j) <. (i2,j) for each cover i1<.i2 in P1, for each j.
        for (i1, i2) in C1
            base1 = (i1 - 1) * n2
            base2 = (i2 - 1) * n2
            for j in 1:n2
                u = base1 + j
                v = base2 + j
                mat[u, v] = true
                push!(edges, (u, v))
            end
        end

        # Vertical covers: (i,j1) <. (i,j2) for each cover j1<.j2 in P2, for each i.
        for i in 1:n1
            base = (i - 1) * n2
            for (j1, j2) in C2
                u = base + j1
                v = base + j2
                mat[u, v] = true
                push!(edges, (u, v))
            end
        end
    end

    # FiniteFringe caches cover data keyed by the leq BitMatrix identity.
    FiniteFringe._COVER_EDGES_CACHE[leq_matrix(Pprod)] = FiniteFringe.CoverEdges(mat, edges)
    return nothing
end

"""
    product_poset(P1::FinitePoset, P2::FinitePoset; check=false, cache_cover_edges=true, use_cache=true)

Construct the cartesian product poset P = P1 x P2 (vertex set size P1.n * P2.n) and return:

- `P`    : the product poset (a `FinitePoset` storing the full leq BitMatrix),
- `pi1`  : projection `EncodingMap(P -> P1)`,
- `pi2`  : projection `EncodingMap(P -> P2)`.

Index convention:
  Vertex `(i,j)` with `i in 1:P1.n`, `j in 1:P2.n` is stored at linear index
  `k = (i-1)*P2.n + j`.

Performance:
  - If `use_cache=true` and `check=false`, repeated calls with the same `P1` and `P2`
    objects reuse the previously constructed product poset.
  - If `cache_cover_edges=true`, we also pre-fill the cover-edge cache for the product,
    which makes downstream homological algebra much faster.
"""
function product_poset(
    P1::FinitePoset,
    P2::FinitePoset;
    check::Bool = false,
    cache_cover_edges::Bool = true,
    use_cache::Bool = true,
)
    # Fast cache path (only for the common "production" settings).
    if use_cache && !check && cache_cover_edges
        inner = get!(_PRODUCT_POSET_CACHE, leq_matrix(P1)) do
            IdDict{BitMatrix, NamedTuple}()
        end
        if haskey(inner, leq_matrix(P2))
            return inner[leq_matrix(P2)]
        end
    end

    n1, n2 = P1.n, P2.n
    n = n1 * n2

    # Build leq matrix as a block matrix:
    #   block(i1,i2) = leq_matrix(P2) if leq(P1,i1,i2) else 0.
    L = falses(n, n)

    @views @inbounds for i1 in 1:n1
        rr = ((i1 - 1) * n2 + 1):(i1 * n2)
        row1 = leq_matrix(P1)[i1, :]
        i2 = findnext(row1, 1)
        while i2 !== nothing
            cc = ((i2 - 1) * n2 + 1):(i2 * n2)
            copyto!(L[rr, cc], leq_matrix(P2))
            i2 = findnext(row1, i2 + 1)
        end
    end

    P = FinitePoset(L; check = check)

    # Projections P -> P1 and P -> P2 as EncodingMap objects.
    pi1_of_q = Vector{Int}(undef, n)
    pi2_of_q = Vector{Int}(undef, n)
    @inbounds for k in 1:n
        pi1_of_q[k] = div((k - 1), n2) + 1
        pi2_of_q[k] = (k - 1) % n2 + 1
    end

    pi1 = EncodingMap(P, P1, pi1_of_q)
    pi2 = EncodingMap(P, P2, pi2_of_q)

    if cache_cover_edges
        _cache_product_cover_edges!(P, P1, P2)
    end

    out = (P = P, pi1 = pi1, pi2 = pi2)

    if use_cache && !check && cache_cover_edges
        inner = get!(_PRODUCT_POSET_CACHE, leq_matrix(P1)) do
            IdDict{BitMatrix, NamedTuple}()
        end
        inner[leq_matrix(P2)] = out
    end

    return out
end

function product_poset(
    P1::AbstractPoset,
    P2::AbstractPoset;
    check::Bool = false,
    cache_cover_edges::Bool = true,
    use_cache::Bool = true,
)
    if use_cache
        inner = get!(_PRODUCT_POSET_OBJ_CACHE, P1) do
            IdDict{AbstractPoset, NamedTuple}()
        end
        if haskey(inner, P2)
            return inner[P2]
        end
    end

    n1, n2 = nvertices(P1), nvertices(P2)
    n = n1 * n2

    # Structured fallback: avoid materializing leq unless requested elsewhere.
    P = ProductPoset(P1, P2)

    # Projections P -> P1 and P -> P2 as EncodingMap objects.
    pi1_of_q = Vector{Int}(undef, n)
    pi2_of_q = Vector{Int}(undef, n)
    @inbounds for k in 1:n
        pi1_of_q[k] = ((k - 1) % n1) + 1
        pi2_of_q[k] = div((k - 1), n1) + 1
    end

    pi1 = EncodingMap(P, P1, pi1_of_q)
    pi2 = EncodingMap(P, P2, pi2_of_q)

    out = (P = P, pi1 = pi1, pi2 = pi2)
    if use_cache
        inner = get!(_PRODUCT_POSET_OBJ_CACHE, P1) do
            IdDict{AbstractPoset, NamedTuple}()
        end
        inner[P2] = out
    end

    return out
end

# Fast pullback of a module M on P1 to the product poset P1 x P2 via the projection to P1.
# The resulting module on Pprod has:
#   dims(i,j) = dims_P1(i)
#   horizontal cover maps (change i): copied from M
#   vertical cover maps (change j): identity (since projection is constant in i)
function _pullback_to_product_pr1(
    M::PModule{K},
    Pprod::FinitePoset,
    P1::FinitePoset,
    P2::FinitePoset,
    C1,
    C2,
) where {K}
    n1, n2 = P1.n, P2.n
    n = n1 * n2

    dims_out = Vector{Int}(undef, n)
    @inbounds for i in 1:n1
        di = M.dims[i]
        base = (i - 1) * n2
        for j in 1:n2
            dims_out[base + j] = di
        end
    end

    # Expected number of cover edges in the product.
    nedges = length(C1) * n2 + n1 * length(C2)
    edge_maps = Dict{Tuple{Int,Int}, Matrix{K}}()
    sizehint!(edge_maps, nedges)

    # Cache identity matrices by dimension to avoid repeated allocations.
    id_cache = Dict{Int, Matrix{K}}()
    get_id(d::Int) = get!(id_cache, d) do
        Matrix{K}(I, d, d)
    end

    @inbounds begin
        # Horizontal edges: replicate P1 cover maps across all j.
        for (i1, i2) in C1
            A = M.edge_maps[i1, i2]  # reuse by reference
            base1 = (i1 - 1) * n2
            base2 = (i2 - 1) * n2
            for j in 1:n2
                u = base1 + j
                v = base2 + j
                edge_maps[(u, v)] = A
            end
        end

        # Vertical edges: identity on each "row i".
        for i in 1:n1
            Ii = get_id(M.dims[i])
            base = (i - 1) * n2
            for (j1, j2) in C2
                u = base + j1
                v = base + j2
                edge_maps[(u, v)] = Ii
            end
        end
    end

    return PModule{K}(Pprod, dims_out, edge_maps)
end

# Fast pullback of a module M on P2 to the product poset P1 x P2 via the projection to P2.
# Symmetric to _pullback_to_product_pr1.
function _pullback_to_product_pr2(
    M::PModule{K},
    Pprod::FinitePoset,
    P1::FinitePoset,
    P2::FinitePoset,
    C1,
    C2,
) where {K}
    n1, n2 = P1.n, P2.n
    n = n1 * n2

    dims_out = Vector{Int}(undef, n)
    @inbounds for i in 1:n1
        base = (i - 1) * n2
        for j in 1:n2
            dims_out[base + j] = M.dims[j]
        end
    end

    nedges = length(C1) * n2 + n1 * length(C2)
    edge_maps = Dict{Tuple{Int,Int}, Matrix{K}}()
    sizehint!(edge_maps, nedges)

    id_cache = Dict{Int, Matrix{K}}()
    get_id(d::Int) = get!(id_cache, d) do
        Matrix{K}(I, d, d)
    end

    @inbounds begin
        # Vertical edges: replicate P2 cover maps across all i.
        for i in 1:n1
            base = (i - 1) * n2
            for (j1, j2) in C2
                A = M.edge_maps[j1, j2]  # reuse by reference
                u = base + j1
                v = base + j2
                edge_maps[(u, v)] = A
            end
        end

        # Horizontal edges: identity on each "column j".
        for j in 1:n2
            Ij = get_id(M.dims[j])
            for (i1, i2) in C1
                u = (i1 - 1) * n2 + j
                v = (i2 - 1) * n2 + j
                edge_maps[(u, v)] = Ij
            end
        end
    end

    return PModule{K}(Pprod, dims_out, edge_maps)
end

"""
    encode_pmodules_to_common_poset(M1::PModule, M2::PModule;
        method=:product, check_poset=false, cache_cover_edges=true, use_cache=true)

Turn "Case C" into the same UX pattern as Cases A/B.

Given:
  - `M1` a poset module on poset `P1 = M1.Q`,
  - `M2` a poset module on poset `P2 = M2.Q`,

return a named tuple `(P, Ms, pi1, pi2)` where:
  - `P` is a common refinement poset,
  - `Ms = [M1_on_P, M2_on_P]` are the pullbacks of M1 and M2 to P,
  - `pi1 : P -> P1` and `pi2 : P -> P2` are the refinement maps as `EncodingMap`s.

Default refinement (`method=:product`):
  P = P1 x P2 (cartesian product poset).

Special case (performance + usability):
  If P1 and P2 have identical leq matrices (even if they are different objects),
  this function rebases M2 onto P1 so that Hom works without blowing up to a product.

Typical usage:
    P, Ms, pi1, pi2 = encode_pmodules_to_common_poset(M1, M2)
    H = Hom(Ms[1], Ms[2])
    d = dim(H)
"""
function encode_pmodules_to_common_poset(
    M1::PModule{K},
    M2::PModule{K};
    method::Symbol = :product,
    check_poset::Bool = false,
    cache_cover_edges::Bool = true,
    use_cache::Bool = true,
) where {K}
    P1 = M1.Q
    P2 = M2.Q

    # Already on the same poset object: nothing to do.
    if P1 === P2
        n = nvertices(P1)
        id = collect(1:n)
        pi1 = EncodingMap(P1, P1, id)
        pi2 = EncodingMap(P1, P2, id)
        return (P = P1, Ms = [M1, M2], pi1 = pi1, pi2 = pi2)
    end

    # Structural equality: same leq, different objects. Avoid P1 x P2 blowup.
    if nvertices(P1) == nvertices(P2) && poset_equal(P1, P2)
        n = nvertices(P1)
        id = collect(1:n)
        pi1 = EncodingMap(P1, P1, id)
        pi2 = EncodingMap(P1, P2, id)

        # Rebase M2 onto P1 (indices match because leq matrices match).
        M2b = PModule{K}(P1, M2.dims, M2.edge_maps)
        return (P = P1, Ms = [M1, M2b], pi1 = pi1, pi2 = pi2)
    end

    if method != :product
        error("encode_pmodules_to_common_poset: only method=:product is implemented.")
    end

    prod = product_poset(P1, P2; check = check_poset, cache_cover_edges = cache_cover_edges, use_cache = use_cache)
    P = prod.P
    pi1 = prod.pi1
    pi2 = prod.pi2

    if P1 isa FinitePoset && P2 isa FinitePoset && P isa FinitePoset
        # Fast projection pullbacks (avoid map_leq allocations on the huge product).
        C1 = cover_edges(P1)
        C2 = cover_edges(P2)
        M1p = _pullback_to_product_pr1(M1, P, P1, P2, C1, C2)
        M2p = _pullback_to_product_pr2(M2, P, P1, P2, C1, C2)
        return (P = P, Ms = [M1p, M2p], pi1 = pi1, pi2 = pi2)
    end

    M1p = pullback(pi1, M1; check = check_poset)
    M2p = pullback(pi2, M2; check = check_poset)

    return (P = P, Ms = [M1p, M2p], pi1 = pi1, pi2 = pi2)
end


# -----------------------------------------------------------------------------
# Pullback (restriction)
# -----------------------------------------------------------------------------

# Pullback / restriction of modules along a monotone map

# Internal helper: compute pullback module along pi without re-checking monotonicity.
@inline function _pullback_module_no_check(
    pi::EncodingMap,
    M::PModule{QQ,MatT},
    C,
) where {QQ<:Rational, MatT<:AbstractMatrix{QQ}}
    Q = pi.Q
    P = pi.P
    @assert M.Q === P

    dims_out = Vector{Int}(undef, nvertices(Q))
    @inbounds for q in 1:nvertices(Q)
        dims_out[q] = M.dims[pi.pi_of_q[q]]
    end

    edge_maps_out = Dict{Tuple{Int,Int}, AbstractMatrix{QQ}}()
    sizehint!(edge_maps_out, length(C))

    @inbounds for (u, v) in C
        iu = pi.pi_of_q[u]
        iv = pi.pi_of_q[v]
        # If pi is monotone, iu <= iv in P and map_leq is defined.
        edge_maps_out[(u, v)] = map_leq(M, iu, iv)
    end

    return PModule{QQ}(Q, dims_out, edge_maps_out)
end

function pullback(
    pi::EncodingMap,
    M::PModule{QQ,MatT};
    check::Bool = true,
) where {QQ<:Rational, MatT<:AbstractMatrix{QQ}}
    check && _check_monotone(pi)
    C = cover_edges(pi.Q)
    return _pullback_module_no_check(pi, M, C)
end

function pullback(
    pi::EncodingMap,
    f::PMorphism{QQ};
    check::Bool = true,
) where {QQ<:Rational}
    check && _check_monotone(pi)
    C = cover_edges(pi.Q)

    # Pull back domain and codomain modules.
    dom_pb = _pullback_module_no_check(pi, f.dom, C)
    cod_pb = _pullback_module_no_check(pi, f.cod, C)

    # Pull back components pointwise along pi: (f_pb)_q = f_{pi(q)}.
    comps_pb = Vector{Matrix{QQ}}(undef, nvertices(pi.Q))
    @inbounds for q in 1:nvertices(pi.Q)
        comps_pb[q] = f.comps[pi.pi_of_q[q]]
    end

    return PMorphism(dom_pb, cod_pb, comps_pb)
end


"""
    restriction(pi, N; check=true)

Alias for `pullback(pi, N)`.
"""
@inline restriction(pi::EncodingMap, N::PModule; check::Bool=true) = pullback(pi, N; check=check)
@inline restriction(pi::EncodingMap, f::PMorphism; check::Bool=true) = pullback(pi, f; check=check)

# -----------------------------------------------------------------------------
# Left Kan extension (left pushforward)
# -----------------------------------------------------------------------------

# LeftKanData stores enough to compute maps functorially (including morphisms).
struct LeftKanData
    idxs::Vector{Vector{Int}}             # I_p subsets
    off::Vector{Dict{Int,Int}}            # offsets into direct sum
    dimS::Vector{Int}                     # ambient direct sum dimension
    W::Vector{Matrix{QQ}}                 # section V_p -> S_p
    L::Vector{Matrix{QQ}}                 # quotient map S_p -> V_p (L*W = I)
    dimV::Vector{Int}                     # dim V_p
end

# Compute offsets for direct sum \oplus_{q in idx} M(q)
function _offset_map(idxs::Vector{Int}, d::Vector{Int})
    off = Dict{Int,Int}()
    s = 0
    for q in idxs
        off[q] = s
        s += d[q]
    end
    return off, s
end

# Left inverse for a full-column-rank matrix using exact QQ linear algebra.
function _left_inverse_full_column(A::Matrix{QQ})
    m,n = size(A)
    if n == 0
        return zeros(QQ, 0, m)
    end
    G = transpose(A) * A
    return solve_fullcolumnQQ(G, transpose(A))
end

# Fiber downset index sets: I_p = { q | pi(q) <= p }
function _index_sets_left(pi::EncodingMap)
    Q = pi.Q
    P = pi.P
    f = pi.pi_of_q

    by_base = [Int[] for _ in 1:nvertices(P)]
    for q in 1:nvertices(Q)
        push!(by_base[f[q]], q)
    end

    idxs = Vector{Vector{Int}}(undef, nvertices(P))
    for p in 1:nvertices(P)
        lst = Int[]
        for v in downset_indices(P, p)
            append!(lst, by_base[v])
        end
        idxs[p] = lst
    end
    return idxs
end

function _left_kan_data(pi::EncodingMap, M::PModule{QQ}; check::Bool=true)
    if check
        _check_monotone(pi)
        if !_same_poset(pi.Q, M.Q)
            error("pushforward_left(pi, M): M must be a module on the domain poset pi.Q")
        end
    end

    Q = pi.Q
    P = pi.P
    d = M.dims

    # Use store-aligned cover-edge traversal to avoid keyed edge lookups in hot loops.
    store = M.edge_maps
    succs = store.succs
    maps_to_succ = store.maps_to_succ

    # Build once; reused many times in the terminal-object fast path.
    cacheQ = IndicatorResolutions._cover_cache(Q)

    idxs = _index_sets_left(pi)

    off = Vector{Dict{Int,Int}}(undef, nvertices(P))
    dimS = Vector{Int}(undef, nvertices(P))
    W = Vector{Matrix{QQ}}(undef, nvertices(P))
    L = Vector{Matrix{QQ}}(undef, nvertices(P))
    dimV = Vector{Int}(undef, nvertices(P))

    # Build each colimit space V_p as quotient of direct sum by relations.
    for p in 1:nvertices(P)
        ip = idxs[p]
        offp, Sp = _offset_map(ip, d)
        off[p] = offp
        dimS[p] = Sp

        if isempty(ip)
            W[p] = zeros(QQ, 0, 0)
            L[p] = zeros(QQ, 0, 0)
            dimV[p] = 0
            continue
        end

        # Fast path: if fiber has a terminal object qmax, colim = M(qmax).
        qmax = _maximum_element(Q, ip)
        if qmax !== nothing
            Vp = d[qmax]
            Wp = zeros(QQ, Sp, Vp)
            Lp = zeros(QQ, Vp, Sp)

            # W: include the qmax summand (identity block)
            oq = offp[qmax]
            for j in 1:Vp
                Wp[oq + j, j] = QQ(1)
            end

            # L: send each summand M(q) to M(qmax) via map_leq(q -> qmax)
            for q in ip
                dq = d[q]
                dq == 0 && continue
                oq = offp[q]
                if q == qmax
                    for j in 1:Vp
                        Lp[j, oq + j] = QQ(1)
                    end
                else
                    A = map_leq(M, q, qmax; cache=cacheQ)
                    @inbounds for i in 1:Vp
                        for j in 1:dq
                            v = A[i,j]
                            if v != 0
                                Lp[i, oq + j] = v
                            end
                        end
                    end
                end
            end

            W[p] = Wp
            L[p] = Lp
            dimV[p] = Vp
            continue
        end

        # General case: quotient by relations x_u - M(u<=v)(x_u) in v for cover edges u<v.
        inS = falses(nvertices(Q))
        for q in ip
            inS[q] = true
        end

        # Count relation columns: each cover edge (u,v) in the fiber contributes dim(M(u)) columns.
        ncols = 0
        @inbounds for u in 1:nvertices(Q)
            inS[u] || continue
            du = d[u]
            du == 0 && continue
            su = succs[u]
            for v in su
                if inS[v]
                    ncols += du
                end
            end
        end

        Irows = Int[]
        Jcols = Int[]
        Vvals = QQ[]
        col = 0

        @inbounds for u in 1:nvertices(Q)
            inS[u] || continue
            du = d[u]
            du == 0 && continue

            ou = offp[u]
            su = succs[u]
            maps_u = maps_to_succ[u]

            for jedge in eachindex(su)
                v = su[jedge]
                inS[v] || continue

                ov = offp[v]
                A = maps_u[jedge]  # may be sparse or dense

                for j in 1:du
                    col += 1

                    # +x_u,j
                    push!(Irows, ou + j)
                    push!(Jcols, col)
                    push!(Vvals, QQ(1))

                    # -(A * x_u)_j in the v-block
                    if A isa SparseMatrixCSC
                        @inbounds for ptr in A.colptr[j]:(A.colptr[j+1]-1)
                            r = A.rowval[ptr]
                            val = A.nzval[ptr]
                            push!(Irows, ov + r)
                            push!(Jcols, col)
                            push!(Vvals, -val)
                        end
                    else
                        @inbounds for r in 1:size(A, 1)
                            val = A[r, j]
                            iszero(val) && continue
                            push!(Irows, ov + r)
                            push!(Jcols, col)
                            push!(Vvals, -val)
                        end
                    end
                end
            end
        end


        Rel = sparse(Irows, Jcols, Vvals, Sp, ncols)
        Wp = nullspaceQQ(transpose(Rel))  # basis of quotient representatives
        Vp = size(Wp, 2)
        Lp = _left_inverse_full_column(Wp)

        W[p] = Wp
        L[p] = Lp
        dimV[p] = Vp
    end

    # Build structure maps along cover edges in P using inclusions of index sets.
    edge_maps = Dict{Tuple{Int,Int}, SparseMatrixCSC{QQ,Int}}()
    for (u,v) in cover_edges(P).edges
        Vu = dimV[u]
        Vv = dimV[v]
        if Vu == 0 || Vv == 0
            edge_maps[(u,v)] = spzeros(QQ, Vv, Vu)
            continue
        end

        # Map is induced by inclusion I_u subset I_v.
        # We avoid building the big inclusion matrix explicitly:
        #   F = L[v] * J * W[u]
        # where J embeds the u-direct-sum into the v-direct-sum.
        Fuv = zeros(QQ, Vv, Vu)
        offu = off[u]
        offv = off[v]
        Wu = W[u]
        Lv = L[v]

        for q in idxs[u]
            dq = d[q]
            dq == 0 && continue
            ru = (offu[q]+1):(offu[q]+dq)
            cv = (offv[q]+1):(offv[q]+dq)
            @views Fuv .+= Lv[:, cv] * Wu[ru, :]
        end

        edge_maps[(u,v)] = sparse(Fuv)
    end

    Mout = PModule{QQ}(P, dimV, edge_maps)
    data = LeftKanData(idxs, off, dimS, W, L, dimV)
    return Mout, data
end

"""
    pushforward_left(pi, M; check=true)

Left Kan extension (left pushforward) of a module along `pi: Q -> P`.

At `p in P` this is the colimit over the fiber downset:

    I_p = { q in Q | pi(q) <= p }.

This is left adjoint to `pullback(pi, -)`.

Fast path: if `I_p` has a maximum element, the colimit equals `M(max)`.
"""
function pushforward_left(pi::EncodingMap, M::PModule{QQ}; check::Bool=true)::PModule{QQ}
    Mout, _ = _left_kan_data(pi, M; check=check)
    return Mout
end

"""
    pushforward_left(pi, f; check=true)

Push forward a morphism of modules along `pi` using left Kan extension.
"""
function pushforward_left(pi::EncodingMap, f::PMorphism{QQ}; check::Bool=true)::PMorphism{QQ}
    if check
        _check_monotone(pi)
        if !_same_poset(pi.Q, f.dom.Q) || !_same_poset(pi.Q, f.cod.Q)
            error("pushforward_left(pi, f): f must be a morphism on the domain poset pi.Q")
        end
    end

    dom_out, data_dom = _left_kan_data(pi, f.dom; check=false)
    cod_out, data_cod = _left_kan_data(pi, f.cod; check=false)

    P = pi.P
    comps = Vector{Matrix{QQ}}(undef, nvertices(P))

    for p in 1:nvertices(P)
        Vd = data_dom.dimV[p]
        Vc = data_cod.dimV[p]
        if Vd == 0 || Vc == 0
            comps[p] = zeros(QQ, Vc, Vd)
            continue
        end

        # Induced map:
        #   Vp(dom) --W--> S(dom) --(oplus f_q)--> S(cod) --L--> Vp(cod)
        Fp = zeros(QQ, Vc, Vd)
        for q in data_dom.idxs[p]
            dd = f.dom.dims[q]
            dc = f.cod.dims[q]
            (dd == 0 || dc == 0) && continue

            rd = (data_dom.off[p][q]+1):(data_dom.off[p][q]+dd)
            cc = (data_cod.off[p][q]+1):(data_cod.off[p][q]+dc)

            @views Fp .+= data_cod.L[p][:, cc] * (f.comps[q] * data_dom.W[p][rd, :])
        end
        comps[p] = Fp
    end

    return PMorphism(dom_out, cod_out, comps)
end

left_kan_extension(pi::EncodingMap, M::PModule{QQ}; check::Bool=true) =
    pushforward_left(pi, M; check=check)

# -----------------------------------------------------------------------------
# Right Kan extension (right pushforward)
# -----------------------------------------------------------------------------

struct RightKanData
    idxs::Vector{Vector{Int}}             # J_p subsets
    off::Vector{Dict{Int,Int}}            # offsets into direct sum/product
    dimS::Vector{Int}                     # ambient dimension
    K::Vector{Matrix{QQ}}                 # section V_p -> S_p  (basis of limit)
    L::Vector{Matrix{QQ}}                 # coordinate map S_p -> V_p (L*K = I)
    dimV::Vector{Int}
end

# Fiber upset index sets: J_p = { q | p <= pi(q) }
function _index_sets_right(pi::EncodingMap)
    Q = pi.Q
    P = pi.P
    f = pi.pi_of_q

    by_base = [Int[] for _ in 1:nvertices(P)]
    for q in 1:nvertices(Q)
        push!(by_base[f[q]], q)
    end

    idxs = Vector{Vector{Int}}(undef, nvertices(P))
    for p in 1:nvertices(P)
        lst = Int[]
        for v in upset_indices(P, p)
            append!(lst, by_base[v])
        end
        idxs[p] = lst
    end
    return idxs
end

function _right_kan_data(pi::EncodingMap, M::PModule{QQ}; check::Bool=true)
    if check
        _check_monotone(pi)
        if !_same_poset(pi.Q, M.Q)
            error("pushforward_right(pi, M): M must be a module on the domain poset pi.Q")
        end
    end

    Q = pi.Q
    P = pi.P
    d = M.dims

    # Use store-aligned cover-edge traversal to avoid keyed edge lookups in hot loops.
    store = M.edge_maps
    succs = store.succs
    maps_to_succ = store.maps_to_succ

    # Build once; reused many times in the terminal-object fast path.
    cacheQ = IndicatorResolutions._cover_cache(Q)

    idxs = _index_sets_right(pi)

    off = Vector{Dict{Int,Int}}(undef, nvertices(P))
    dimS = Vector{Int}(undef, nvertices(P))
    K = Vector{Matrix{QQ}}(undef, nvertices(P))
    L = Vector{Matrix{QQ}}(undef, nvertices(P))
    dimV = Vector{Int}(undef, nvertices(P))

    for p in 1:nvertices(P)
        jp = idxs[p]
        offp, Sp = _offset_map(jp, d)
        off[p] = offp
        dimS[p] = Sp

        if isempty(jp)
            K[p] = zeros(QQ, 0, 0)
            L[p] = zeros(QQ, 0, 0)
            dimV[p] = 0
            continue
        end

        # Fast path: if fiber has an initial object qmin, limit = M(qmin).
        qmin = _minimum_element(Q, jp)
        if qmin !== nothing
            Vp = d[qmin]
            Kp = zeros(QQ, Sp, Vp)
            Lp = zeros(QQ, Vp, Sp)

            # K: cone embedding M(qmin) -> oplus_{q in jp} M(q), q component is map(qmin->q).
            for q in jp
                dq = d[q]
                dq == 0 && continue
                oq = offp[q]
                if q == qmin
                    for j in 1:Vp
                        Kp[oq + j, j] = QQ(1)
                    end
                else
                    A = map_leq(M, qmin, q; cache=cacheQ)
                    @inbounds for i in 1:dq
                        for j in 1:Vp
                            v = A[i,j]
                            if v != 0
                                Kp[oq + i, j] = v
                            end
                        end
                    end
                end
            end

            # L: projection to qmin component
            oq = offp[qmin]
            for j in 1:Vp
                Lp[j, oq + j] = QQ(1)
            end

            K[p] = Kp
            L[p] = Lp
            dimV[p] = Vp
            continue
        end

        # General case: limit as kernel of compatibility constraints x_v = M(u<=v) x_u.
        inS = falses(nvertices(Q))
        for q in jp
            inS[q] = true
        end

        # Each cover edge (u,v) contributes dim(M(v)) equations.
        nrows = 0
        @inbounds for u in 1:nvertices(Q)
            inS[u] || continue
            su = succs[u]
            for v in su
                if inS[v]
                    nrows += d[v]
                end
            end
        end

        Irows = Int[]
        Jcols = Int[]
        Vvals = QQ[]
        row0 = 0

        @inbounds for u in 1:nvertices(Q)
            inS[u] || continue

            ou = offp[u]
            du = d[u]

            su = succs[u]
            maps_u = maps_to_succ[u]

            for jedge in eachindex(su)
                v = su[jedge]
                inS[v] || continue

                dv = d[v]
                dv == 0 && continue

                ov = offp[v]
                A = maps_u[jedge]  # may be sparse or dense

                # Equations: x_v - A x_u = 0 in each coordinate of v.
                for k in 1:dv
                    push!(Irows, row0 + k)
                    push!(Jcols, ov + k)
                    push!(Vvals, QQ(1))
                end

                # Add -A into rows for v-coordinates.
                for j in 1:du
                    if A isa SparseMatrixCSC
                        @inbounds for ptr in A.colptr[j]:(A.colptr[j+1]-1)
                            k = A.rowval[ptr]
                            val = A.nzval[ptr]
                            push!(Irows, row0 + k)
                            push!(Jcols, ou + j)
                            push!(Vvals, -val)
                        end
                    else
                        @inbounds for k in 1:size(A, 1)
                            val = A[k, j]
                            iszero(val) && continue
                            push!(Irows, row0 + k)
                            push!(Jcols, ou + j)
                            push!(Vvals, -val)
                        end
                    end
                end

                row0 += dv
            end
        end

        C = sparse(Irows, Jcols, Vvals, nrows, Sp)
        Kp = nullspaceQQ(C)     # kernel basis
        Vp = size(Kp, 2)
        Lp = _left_inverse_full_column(Kp)

        K[p] = Kp
        L[p] = Lp
        dimV[p] = Vp
    end

    # Structure maps: restriction along J_v subset J_u for u<=v in P.
    edge_maps = Dict{Tuple{Int,Int}, SparseMatrixCSC{QQ,Int}}()
    for (u,v) in cover_edges(P).edges
        Vu = dimV[u]
        Vv = dimV[v]
        if Vu == 0 || Vv == 0
            edge_maps[(u,v)] = spzeros(QQ, Vv, Vu)
            continue
        end

        Fuv = zeros(QQ, Vv, Vu)
        offu = off[u]
        offv = off[v]
        Ku = K[u]
        Lv = L[v]

        for q in idxs[v]
            dq = d[q]
            dq == 0 && continue
            ru = (offu[q]+1):(offu[q]+dq)  # rows in S_u
            cv = (offv[q]+1):(offv[q]+dq)  # cols in S_v
            @views Fuv .+= Lv[:, cv] * Ku[ru, :]
        end

        edge_maps[(u,v)] = sparse(Fuv)
    end

    Mout = PModule{QQ}(P, dimV, edge_maps)
    data = RightKanData(idxs, off, dimS, K, L, dimV)
    return Mout, data
end

"""
    pushforward_right(pi, M; check=true)

Right Kan extension (right pushforward) along `pi: Q -> P`.

At `p in P` this is the limit over the fiber upset:

    J_p = { q in Q | p <= pi(q) }.

This is right adjoint to `pullback(pi, -)`.

Fast path: if `J_p` has a minimum element, the limit equals `M(min)`.
"""
function pushforward_right(pi::EncodingMap, M::PModule{QQ}; check::Bool=true)::PModule{QQ}
    Mout, _ = _right_kan_data(pi, M; check=check)
    return Mout
end

"""
    pushforward_right(pi, f; check=true)

Push forward a morphism of modules along `pi` using right Kan extension.
"""
function pushforward_right(pi::EncodingMap, f::PMorphism{QQ}; check::Bool=true)::PMorphism{QQ}
    if check
        _check_monotone(pi)
        if !_same_poset(pi.Q, f.dom.Q) || !_same_poset(pi.Q, f.cod.Q)
            error("pushforward_right(pi, f): f must be a morphism on the domain poset pi.Q")
        end
    end

    dom_out, data_dom = _right_kan_data(pi, f.dom; check=false)
    cod_out, data_cod = _right_kan_data(pi, f.cod; check=false)

    P = pi.P
    comps = Vector{Matrix{QQ}}(undef, nvertices(P))

    for p in 1:nvertices(P)
        Vd = data_dom.dimV[p]
        Vc = data_cod.dimV[p]
        if Vd == 0 || Vc == 0
            comps[p] = zeros(QQ, Vc, Vd)
            continue
        end

        Fp = zeros(QQ, Vc, Vd)
        for q in data_dom.idxs[p]
            dd = f.dom.dims[q]
            dc = f.cod.dims[q]
            (dd == 0 || dc == 0) && continue

            rd = (data_dom.off[p][q]+1):(data_dom.off[p][q]+dd)
            cc = (data_cod.off[p][q]+1):(data_cod.off[p][q]+dc)

            @views Fp .+= data_cod.L[p][:, cc] * (f.comps[q] * data_dom.K[p][rd, :])
        end
        comps[p] = Fp
    end

    return PMorphism(dom_out, cod_out, comps)
end

right_kan_extension(pi::EncodingMap, M::PModule{QQ}; check::Bool=true) =
    pushforward_right(pi, M; check=check)

# -----------------------------------------------------------------------------
# Derived functors (object-level)
# -----------------------------------------------------------------------------

"""
    pushforward_left_complex(pi, M, df; check=true, res=nothing)

Compute a cochain complex whose cohomology in degree -i is
`L_i pushforward_left(pi, M)` for i = 0..df.maxdeg.

Derived degree is controlled by `df.maxdeg`.

Speed: pass a precomputed `ProjectiveResolution` via `res` to avoid recomputation.
"""
function pushforward_left_complex(pi::EncodingMap, M::PModule{QQ}, df::DerivedFunctorOptions;
                                  check::Bool=true,
                                  res=nothing)
    if check
        _check_monotone(pi)
        if !_same_poset(pi.Q, M.Q)
            error("pushforward_left_complex: M must be on pi.Q")
        end
    end

    maxlen = df.maxdeg + 1
    if res === nothing
        res = projective_resolution(M, ResolutionOptions(maxlen=maxlen))
    else
        @assert res.M === M
        @assert length(res.Pmods) >= maxlen + 1
        @assert length(res.d_mor) >= maxlen
    end

    terms = Vector{PModule{QQ}}(undef, maxlen + 1)
    diffs = Vector{PMorphism{QQ}}(undef, maxlen)

    for k in 0:maxlen
        terms[maxlen - k + 1] = pushforward_left(pi, res.Pmods[k + 1]; check=false)
    end
    for k in 1:maxlen
        diffs[maxlen - k + 1] = pushforward_left(pi, res.d_mor[k]; check=false)
    end

    return ModuleCochainComplex(terms, diffs; tmin=-maxlen, check=check)
end

pushforward_left_complex(pi::EncodingMap, M::PModule{QQ};
                         opts::Union{DerivedFunctorOptions,Nothing}=nothing,
                         check::Bool=true,
                         res=nothing) =
    pushforward_left_complex(pi, M, _resolve_df_opts(opts); check=check, res=res)

"""
    Lpushforward_left(pi, M, df; check=true)

Return `[L_0, L_1, ..., L_df.maxdeg]` where `L_i = L_i pushforward_left(pi, M)`.

Derived degree is controlled by `df.maxdeg`.
"""
function Lpushforward_left(pi::EncodingMap, M::PModule{QQ}, df::DerivedFunctorOptions;
                           check::Bool=true)
    maxdeg = df.maxdeg
    C = pushforward_left_complex(pi, M, df; check=check)
    out = Vector{PModule{QQ}}(undef, maxdeg + 1)
    for i in 0:maxdeg
        out[i + 1] = cohomology_module(C, -i)
    end
    return out
end

Lpushforward_left(pi::EncodingMap, M::PModule{QQ};
                  opts::Union{DerivedFunctorOptions,Nothing}=nothing,
                  check::Bool=true) =
    Lpushforward_left(pi, M, _resolve_df_opts(opts); check=check)

derived_pushforward_left(pi::EncodingMap, M::PModule{QQ}, df::DerivedFunctorOptions;
                         check::Bool=true) =
    Lpushforward_left(pi, M, df; check=check)

derived_pushforward_left(pi::EncodingMap, M::PModule{QQ};
                         opts::Union{DerivedFunctorOptions,Nothing}=nothing,
                         check::Bool=true) =
    derived_pushforward_left(pi, M, _resolve_df_opts(opts); check=check)

"""
    pushforward_left_complex(pi, f, df; check=true, res_dom=nothing, res_cod=nothing)

Return a cochain map between the complexes `pushforward_left_complex(pi, f.dom, df)` and
`pushforward_left_complex(pi, f.cod, df)`.

The induced map on cohomology in degree `-i` is the map on left-derived functors
`L_i pushforward_left(pi, f)` for i = 0..df.maxdeg.

Implementation:
1. take projective resolutions of dom and cod
2. lift `f` canonically to a chain map between projective resolutions (coefficient form)
3. apply `pushforward_left` termwise
4. package as a `ModuleCochainMap`
"""
function pushforward_left_complex(pi::EncodingMap, f::PMorphism{QQ}, df::DerivedFunctorOptions;
                                  check::Bool=true,
                                  res_dom=nothing,
                                  res_cod=nothing)
    if check
        _check_monotone(pi)
        @assert _same_poset(pi.Q, f.dom.Q)
        @assert _same_poset(pi.Q, f.cod.Q)
    end

    maxlen = df.maxdeg + 1

    if res_dom === nothing
        res_dom = projective_resolution(f.dom, ResolutionOptions(maxlen=maxlen))
    end
    if res_cod === nothing
        res_cod = projective_resolution(f.cod, ResolutionOptions(maxlen=maxlen))
    end

    Cdom = pushforward_left_complex(pi, f.dom, df; check=false, res=res_dom)
    Ccod = pushforward_left_complex(pi, f.cod, df; check=false, res=res_cod)

    # Canonical chain-map lifting in coefficient form.
    H = lift_chainmap(res_dom, res_cod, f; maxlen=maxlen)

    # Convert coefficient matrices to honest PMorphisms on Q.
    phi = Vector{PMorphism{QQ}}(undef, maxlen + 1)
    for k in 0:maxlen
        phi[k + 1] = _pmorphism_from_upset_coeff(res_dom.Pmods[k + 1], res_cod.Pmods[k + 1],
                                                 res_dom.gens[k + 1], res_cod.gens[k + 1],
                                                 H[k + 1])
    end

    # Reverse order to match cochain degrees -(maxlen)..0
    comps = Vector{PMorphism{QQ}}(undef, maxlen + 1)
    for k in 0:maxlen
        idx = maxlen - k + 1
        comps[idx] = pushforward_left(pi, phi[k + 1]; check=false)
    end

    return ModuleCochainMap(Cdom, Ccod, comps; check=check)
end

pushforward_left_complex(pi::EncodingMap, f::PMorphism{QQ};
                         opts::Union{DerivedFunctorOptions,Nothing}=nothing,
                         check::Bool=true,
                         res_dom=nothing,
                         res_cod=nothing) =
    pushforward_left_complex(pi, f, _resolve_df_opts(opts);
                             check=check, res_dom=res_dom, res_cod=res_cod)

"""
    Lpushforward_left(pi, f, df; check=true, res_dom=nothing, res_cod=nothing)

Return the induced maps on left-derived pushforward:

    out[i+1] : L_i pushforward_left(pi, f.dom) -> L_i pushforward_left(pi, f.cod)

for i = 0..df.maxdeg.

Derived degree is controlled by `df.maxdeg`.
"""
function Lpushforward_left(pi::EncodingMap, f::PMorphism{QQ}, df::DerivedFunctorOptions;
                           check::Bool=true,
                           res_dom=nothing,
                           res_cod=nothing)
    maxdeg = df.maxdeg
    F = pushforward_left_complex(pi, f, df; check=check, res_dom=res_dom, res_cod=res_cod)
    out = Vector{PMorphism{QQ}}(undef, maxdeg + 1)
    for i in 0:maxdeg
        out[i + 1] = induced_map_on_cohomology_modules(F, -i)
    end
    return out
end

Lpushforward_left(pi::EncodingMap, f::PMorphism{QQ};
                  opts::Union{DerivedFunctorOptions,Nothing}=nothing,
                  check::Bool=true,
                  res_dom=nothing,
                  res_cod=nothing) =
    Lpushforward_left(pi, f, _resolve_df_opts(opts);
                      check=check, res_dom=res_dom, res_cod=res_cod)

derived_pushforward_left(pi::EncodingMap, f::PMorphism{QQ}, df::DerivedFunctorOptions;
                         check::Bool=true) =
    Lpushforward_left(pi, f, df; check=check)

derived_pushforward_left(pi::EncodingMap, f::PMorphism{QQ};
                         opts::Union{DerivedFunctorOptions,Nothing}=nothing,
                         check::Bool=true) =
    derived_pushforward_left(pi, f, _resolve_df_opts(opts); check=check)


"""
        pushforward_right_complex(pi, M, df; check=true, res=nothing)

Compute a cochain complex whose cohomology in degree i is
`R^i pushforward_right(pi, M)` for i = 0..df.maxdeg.

Derived degree is controlled by `df.maxdeg`

Speed: pass a precomputed `InjectiveResolution` via `res` to avoid recomputation.
"""
function pushforward_right_complex(pi::EncodingMap, M::PModule{QQ}, df::DerivedFunctorOptions;
                                   check::Bool=true,
                                   res=nothing)
    if check
        _check_monotone(pi)
        if !_same_poset(pi.Q, M.Q)
            error("pushforward_right_complex: M must be on pi.Q")
        end
    end

    maxlen = df.maxdeg + 1
    if res === nothing
        res = injective_resolution(M, ResolutionOptions(maxlen=maxlen))
    else
        @assert res.N === M
        @assert length(res.Emods) >= maxlen + 1
        @assert length(res.d_mor) >= maxlen
    end

    terms = Vector{PModule{QQ}}(undef, maxlen + 1)
    diffs = Vector{PMorphism{QQ}}(undef, maxlen)

    for k in 0:maxlen
        terms[k + 1] = pushforward_right(pi, res.Emods[k + 1]; check=false)
    end
    for k in 1:maxlen
        diffs[k] = pushforward_right(pi, res.d_mor[k]; check=false)
    end

    return ModuleCochainComplex(terms, diffs; tmin=0, check=check)
end

pushforward_right_complex(pi::EncodingMap, M::PModule{QQ};
                          opts::Union{DerivedFunctorOptions,Nothing}=nothing,
                          check::Bool=true,
                          res=nothing) =
    pushforward_right_complex(pi, M, _resolve_df_opts(opts); check=check, res=res)

"""
    Rpushforward_right(pi, M, df; check=true)

Return `[R^0, R^1, ..., R^df.maxdeg]` where `R^i = R^i pushforward_right(pi, M)`.

Derived degree is controlled by `df.maxdeg`.
"""
function Rpushforward_right(pi::EncodingMap, M::PModule{QQ}, df::DerivedFunctorOptions;
                            check::Bool=true)
    maxdeg = df.maxdeg
    C = pushforward_right_complex(pi, M, df; check=check)
    out = Vector{PModule{QQ}}(undef, maxdeg + 1)
    for i in 0:maxdeg
        out[i + 1] = cohomology_module(C, i)
    end
    return out
end

Rpushforward_right(pi::EncodingMap, M::PModule{QQ};
                   opts::Union{DerivedFunctorOptions,Nothing}=nothing,
                   check::Bool=true) =
    Rpushforward_right(pi, M, _resolve_df_opts(opts); check=check)

derived_pushforward_right(pi::EncodingMap, M::PModule{QQ}, df::DerivedFunctorOptions;
                          check::Bool=true) =
    Rpushforward_right(pi, M, df; check=check)

derived_pushforward_right(pi::EncodingMap, M::PModule{QQ};
                          opts::Union{DerivedFunctorOptions,Nothing}=nothing,
                          check::Bool=true) =
    derived_pushforward_right(pi, M, _resolve_df_opts(opts); check=check)
    
"""
    pushforward_right_complex(pi, f, df; check=true, res_dom=nothing, res_cod=nothing)

Return the induced cochain map between `pushforward_right_complex(pi, f.dom, df)` and
`pushforward_right_complex(pi, f.cod, df)`.

Cohomology in degree i gives the induced map:

    R^i pushforward_right(pi, f)
    
for i = 0..df.maxdeg.

Implementation:
1. take injective resolutions of dom and cod
2. canonically lift `f` to a chain map between injective resolutions
3. apply `pushforward_right` termwise
4. package as a `ModuleCochainMap`
"""
function pushforward_right_complex(pi::EncodingMap, f::PMorphism{QQ}, df::DerivedFunctorOptions;
                                   check::Bool=true,
                                   res_dom=nothing,
                                   res_cod=nothing)
    if check
        _check_monotone(pi)
        @assert _same_poset(pi.Q, f.dom.Q)
        @assert _same_poset(pi.Q, f.cod.Q)
    end

    maxlen = df.maxdeg + 1

    if res_dom === nothing
        res_dom = injective_resolution(f.dom, ResolutionOptions(maxlen=maxlen))
    end
    if res_cod === nothing
        res_cod = injective_resolution(f.cod, ResolutionOptions(maxlen=maxlen))
    end

    Cdom = pushforward_right_complex(pi, f.dom, df; check=false, res=res_dom)
    Ccod = pushforward_right_complex(pi, f.cod, df; check=false, res=res_cod)

    # Canonical injective chain-map lifting (now compatible with current gens format).
    phi = lift_injective_chainmap(f, res_dom, res_cod; upto=maxlen, check=check)

    comps = Vector{PMorphism{QQ}}(undef, maxlen + 1)
    for k in 0:maxlen
        comps[k + 1] = pushforward_right(pi, phi[k + 1]; check=false)
    end

    return ModuleCochainMap(Cdom, Ccod, comps; check=check)
end

pushforward_right_complex(pi::EncodingMap, f::PMorphism{QQ};
                          opts::Union{DerivedFunctorOptions,Nothing}=nothing,
                          check::Bool=true,
                          res_dom=nothing,
                          res_cod=nothing) =
    pushforward_right_complex(pi, f, _resolve_df_opts(opts);
                              check=check, res_dom=res_dom, res_cod=res_cod)


"""
    Rpushforward_right(pi, f, df; check=true, res_dom=nothing, res_cod=nothing)

Return the induced maps on right-derived pushforward:

    out[i+1] : R^i pushforward_right(pi, f.dom) -> R^i pushforward_right(pi, f.cod)

for i = 0..df.maxdeg.

Derived degree is controlled by `df.maxdeg`.
"""
function Rpushforward_right(pi::EncodingMap, f::PMorphism{QQ}, df::DerivedFunctorOptions;
                            check::Bool=true,
                            res_dom=nothing,
                            res_cod=nothing)
    maxdeg = df.maxdeg
    F = pushforward_right_complex(pi, f, df; check=check, res_dom=res_dom, res_cod=res_cod)
    out = Vector{PMorphism{QQ}}(undef, maxdeg + 1)
    for i in 0:maxdeg
        out[i + 1] = induced_map_on_cohomology_modules(F, i)
    end
    return out
end

Rpushforward_right(pi::EncodingMap, f::PMorphism{QQ};
                   opts::Union{DerivedFunctorOptions,Nothing}=nothing,
                   check::Bool=true,
                   res_dom=nothing,
                   res_cod=nothing) =
    Rpushforward_right(pi, f, _resolve_df_opts(opts);
                       check=check, res_dom=res_dom, res_cod=res_cod)

derived_pushforward_right(pi::EncodingMap, f::PMorphism{QQ}, df::DerivedFunctorOptions;
                          check::Bool=true) =
    Rpushforward_right(pi, f, df; check=check)

derived_pushforward_right(pi::EncodingMap, f::PMorphism{QQ};
                          opts::Union{DerivedFunctorOptions,Nothing}=nothing,
                          check::Bool=true) =
    derived_pushforward_right(pi, f, _resolve_df_opts(opts); check=check)

# -----------------------------------------------------------------------------
# Helpers: morphisms between direct sums of principal upsets (projectives)
# -----------------------------------------------------------------------------

# Active summand indices at each vertex u, respecting the canonical ordering used in projective covers.
function _active_upset_indices_from_bases(Q::AbstractPoset, bases::Vector{Int})
    n = nvertices(Q)
    by_base = [Int[] for _ in 1:n]
    for (j, b) in enumerate(bases)
        push!(by_base[b], j)
    end

    out = Vector{Vector{Int}}(undef, n)
    for u in 1:n
        idxs = Int[]
        for b in downset_indices(Q, u)
            append!(idxs, by_base[b])
        end
        out[u] = idxs
    end
    return out
end


"""
    _pmorphism_from_upset_coeff(dom, cod, dom_bases, cod_bases, C)

Internal: build a PMorphism between direct sums of principal upsets from a global coefficient matrix.

`C` has size (#cod_summands) x (#dom_summands).  At vertex `u`, the component is the restriction
to the summands active at `u`, in canonical order.
"""
function _pmorphism_from_upset_coeff(dom::PModule{QQ}, cod::PModule{QQ},
                                    dom_bases::Vector{Int}, cod_bases::Vector{Int},
                                    C::AbstractMatrix{QQ})::PMorphism{QQ}
    Q = dom.Q
    @assert Q === cod.Q

    act_dom = _active_upset_indices_from_bases(Q, dom_bases)
    act_cod = _active_upset_indices_from_bases(Q, cod_bases)

    comps = Vector{Matrix{QQ}}(undef, nvertices(Q))
    for u in 1:nvertices(Q)
        rows = act_cod[u]
        cols = act_dom[u]
        if isempty(rows) || isempty(cols)
            comps[u] = zeros(QQ, length(rows), length(cols))
        else
            comps[u] = Matrix(C[rows, cols])
        end
    end
    return PMorphism(dom, cod, comps)
end


end # module ChangeOfPosets
